# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams


def test_packed_seq_params_declares_dynamic_cp_fields():
    field_names = {field.name for field in fields(PackedSeqParams)}

    assert {"local_cp_size", "cp_group", "total_tokens", "seq_idx"} <= field_names


def test_packed_seq_params_seq_idx_clamps_to_total_tokens():
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 4, 10], dtype=torch.int32),
        total_tokens=6,
    )

    assert packed_seq_params.seq_idx.tolist() == [[0, 0, 0, 0, 1, 1]]


def test_rope_cp_size_one_does_not_fallback_to_global_cp_group(monkeypatch):
    monkeypatch.setattr(
        parallel_state,
        "get_context_parallel_group",
        lambda *args, **kwargs: pytest.fail("unexpected global CP group fallback"),
    )

    config = SimpleNamespace(
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )
    t = torch.ones(4, 1, 2)
    freqs = torch.ones(4, 1, 1, 2)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    out = apply_rotary_pos_emb(
        t,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=None,
        cp_size=1,
        cp_rank=0,
    )

    assert out.shape == t.shape
