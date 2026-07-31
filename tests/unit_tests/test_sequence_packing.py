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
