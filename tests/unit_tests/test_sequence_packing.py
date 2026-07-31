# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
)


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
