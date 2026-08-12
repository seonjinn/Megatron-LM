# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from inspect import signature
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    merge_moe_packed_seq_params_from_cuda_graph_kwargs,
    pad_sequence_for_thd,
    split_mamba_packed_seq_params_for_cuda_graph,
    split_moe_packed_seq_params_for_cuda_graph,
    split_packed_seq_params_for_cuda_graph,
)


def test_packed_seq_params_preserves_legacy_positional_slots() -> None:
    assert list(signature(PackedSeqParams).parameters) == [
        "qkv_format",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen_q",
        "max_seqlen_kv",
        "local_cp_size",
        "cp_group",
        "total_tokens",
        "seq_idx",
        "tokens_per_sample",
        "pad_between_seqs",
        "seq_aux_loss_sample_ids",
        "seq_aux_loss_num_samples",
        "seq_aux_loss_max_samples",
    ]

    params = PackedSeqParams(
        "thd",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        True,
    )

    assert params.pad_between_seqs is True
    assert params.seq_aux_loss_sample_ids is None
    assert params.seq_aux_loss_num_samples is None
    assert params.seq_aux_loss_max_samples is None


def test_moe_packed_seq_params_cuda_graph_has_independent_namespace() -> None:
    source = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 8], dtype=torch.int32),
        total_tokens=12,
        seq_idx=torch.arange(12, dtype=torch.int32).unsqueeze(0),
        seq_aux_loss_sample_ids=torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int64
        ),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
        tokens_per_sample=4,
    )

    tensor_kwargs, static = split_moe_packed_seq_params_for_cuda_graph(source)

    assert set(tensor_kwargs) == {
        "_moe_packed_seq_params_seq_aux_loss_sample_ids",
        "_moe_packed_seq_params_seq_aux_loss_num_samples",
    }
    assert static == {"seq_aux_loss_max_samples": 3, "tokens_per_sample": 4}

    generic_tensor_kwargs, generic_static = split_packed_seq_params_for_cuda_graph(source)
    assert all(
        not key.startswith("_moe_packed_seq_params_")
        for key in (*generic_tensor_kwargs, *generic_static)
    )
    mamba_tensor_kwargs, mamba_static = split_mamba_packed_seq_params_for_cuda_graph(source)
    assert all(
        not key.startswith("_moe_packed_seq_params_")
        for key in (*mamba_tensor_kwargs, *mamba_static)
    )

    rebuilt = merge_moe_packed_seq_params_from_cuda_graph_kwargs(
        source, dict(tensor_kwargs), static
    )
    assert rebuilt is not None
    assert rebuilt is not source
    assert rebuilt.seq_aux_loss_sample_ids is source.seq_aux_loss_sample_ids
    assert rebuilt.seq_aux_loss_num_samples is source.seq_aux_loss_num_samples
    assert rebuilt.seq_aux_loss_max_samples == 3
    assert rebuilt.tokens_per_sample == 4

    source_sample_ids = source.seq_aux_loss_sample_ids
    source_num_samples = source.seq_aux_loss_num_samples
    rebuilt.seq_aux_loss_sample_ids = torch.tensor([2], dtype=torch.int64)
    rebuilt.seq_aux_loss_num_samples = torch.tensor(1, dtype=torch.int64)
    rebuilt.seq_aux_loss_max_samples = 1
    assert source.seq_aux_loss_sample_ids is source_sample_ids
    assert source.seq_aux_loss_num_samples is source_num_samples
    assert source.seq_aux_loss_max_samples == 3

    retained_kwargs = dict(tensor_kwargs)
    rebuilt = merge_moe_packed_seq_params_from_cuda_graph_kwargs(
        source, retained_kwargs, static, remove_from_kwargs=False
    )
    assert rebuilt is not source
    assert retained_kwargs == tensor_kwargs


@pytest.mark.parametrize(
    ("kwargs", "static_metadata", "field_name"),
    (
        (
            {"_moe_packed_seq_params_seq_aux_loss_sample_ids": 0},
            None,
            "seq_aux_loss_sample_ids",
        ),
        ({}, {"seq_aux_loss_max_samples": torch.tensor(3)}, "seq_aux_loss_max_samples"),
        ({}, {"tokens_per_sample": torch.tensor(4)}, "tokens_per_sample"),
    ),
)
def test_merge_moe_packed_seq_params_rejects_invalid_field_types(
    kwargs: dict[str, object], static_metadata: dict[str, object] | None, field_name: str
) -> None:
    with pytest.raises(TypeError, match=field_name):
        merge_moe_packed_seq_params_from_cuda_graph_kwargs(None, kwargs, static_metadata)


def test_pad_sequence_for_thd_preserves_seq_aux_loss_sample_ownership() -> None:
    sample_ids = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int64
    )
    num_samples = torch.tensor(2, dtype=torch.int64)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 8], dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_kv=5,
        seq_aux_loss_sample_ids=sample_ids,
        seq_aux_loss_num_samples=num_samples,
        seq_aux_loss_max_samples=3,
    )

    padded_params = pad_sequence_for_thd(
        tokens=torch.ones(1, 8, dtype=torch.long),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        target_len=16,
        max_num_seqs=4,
    )[4]

    assert padded_params.seq_aux_loss_sample_ids is sample_ids
    assert padded_params.seq_aux_loss_num_samples is num_samples
    assert padded_params.seq_aux_loss_max_samples == 3


def test_get_thd_padding_kwargs_preserves_configured_capacity() -> None:
    config = SimpleNamespace(
        pad_packed_seq_alignment=8,
        pad_packed_seq_to=128,
        thd_max_packed_sequences=7,
        cuda_graph_impl="local",
    )

    assert get_thd_padding_kwargs(config) == (8, 128, 7)


def test_get_thd_padding_kwargs_requires_te_sequence_capacity() -> None:
    config = SimpleNamespace(
        pad_packed_seq_alignment=None,
        pad_packed_seq_to=128,
        thd_max_packed_sequences=None,
        cuda_graph_impl="transformer_engine",
    )

    with pytest.raises(AssertionError, match="thd_max_packed_sequences"):
        get_thd_padding_kwargs(config)


def test_thd_graph_metadata_has_fixed_sequence_capacity() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7, 12], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor([0, 3, 7, 12], dtype=torch.int32, device="cuda"),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv_padded=torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda"),
        max_seqlen_q=5,
        max_seqlen_kv=5,
        pad_between_seqs=True,
    )

    padded = pad_sequence_for_thd(
        tokens=torch.ones(12, 1, dtype=torch.long, device="cuda"),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        target_len=16,
        max_num_seqs=7,
    )

    graph_params = padded[4]
    assert graph_params.cu_seqlens_q.numel() == 8
    assert graph_params.cu_seqlens_q_padded.numel() == 8
    assert graph_params.cu_seqlens_q[-1] <= graph_params.cu_seqlens_q_padded[-1]


def test_thd_cp_padding_mask_uses_local_tensor_occupancy() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 5, 12], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5, 12], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 5, 12], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 5, 12], dtype=torch.int32),
        max_seqlen_q=7,
        max_seqlen_kv=7,
        local_cp_size=2,
    )

    padded = pad_sequence_for_thd(
        tokens=torch.ones(1, 6, dtype=torch.long),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        target_len=8,
        max_num_seqs=7,
    )

    assert padded[0].shape[-1] == 8
    assert torch.equal(
        padded[5], torch.tensor([[False, False, False, False, False, False, True, True]])
    )


def test_thd_zigzag_padding_mask_uses_nonuniform_local_occupancy() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 9, 12], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 2, 9, 12], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 10, 12], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 3, 10, 12], dtype=torch.int32),
        max_seqlen_q=7,
        max_seqlen_kv=7,
        local_cp_size=2,
        pad_between_seqs=True,
    )

    padded = pad_sequence_for_thd(
        tokens=torch.ones(1, 5, dtype=torch.long),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        target_len=8,
        max_num_seqs=7,
    )

    assert torch.equal(
        padded[5], torch.tensor([[False, False, False, False, False, True, True, True]])
    )


def test_thd_cp_metadata_only_padding_fails_closed() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 5, 12], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5, 12], dtype=torch.int32),
        max_seqlen_q=7,
        max_seqlen_kv=7,
        local_cp_size=2,
    )

    with pytest.raises(AssertionError, match="local token-like tensor"):
        pad_sequence_for_thd(
            tokens=None,
            labels=None,
            loss_mask=None,
            position_ids=None,
            packed_seq_params=params,
            target_len=8,
            max_num_seqs=7,
        )


def test_thd_dummy_endpoints_preserve_asymmetric_q_kv_offsets() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 3, 6], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=3,
        pad_between_seqs=True,
    )

    padded = pad_sequence_for_thd(
        tokens=torch.ones(1, 8, dtype=torch.long),
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=params,
        target_len=16,
        max_num_seqs=7,
    )[4]

    assert padded.cu_seqlens_q.tolist() == [0, 3, 7, 15, 15, 15, 15, 15]
    assert padded.cu_seqlens_q_padded.tolist() == [0, 4, 8, 16, 16, 16, 16, 16]
    assert padded.cu_seqlens_kv.tolist() == [0, 2, 5, 15, 15, 15, 15, 15]
    assert padded.cu_seqlens_kv_padded.tolist() == [0, 3, 6, 16, 16, 16, 16, 16]
