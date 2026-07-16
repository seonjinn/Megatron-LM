# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.extensions.transformer_engine import build_packed_seq_kwargs
from megatron.core.packed_seq_params import PackedSeqParams


def test_build_packed_seq_kwargs_excludes_internal_max_seqlen_tensors():
    """TE attention receives scalar max-seqlen metadata, never graph-buffer tensors."""
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    max_seqlen_tensor = torch.tensor([4], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        max_seqlen_q_tensor=max_seqlen_tensor,
        max_seqlen_kv_tensor=max_seqlen_tensor,
    )

    kwargs = build_packed_seq_kwargs(
        packed_seq_params,
        {
            "qkv_format",
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "max_seqlen_q",
            "max_seqlen_kv",
            "max_seqlen_q_tensor",
            "max_seqlen_kv_tensor",
        },
    )

    assert kwargs["qkv_format"] == "thd"
    assert kwargs["cu_seqlens_q"] is cu_seqlens
    assert kwargs["max_seqlen_q"] == 4
    assert "max_seqlen_q_tensor" not in kwargs
    assert "max_seqlen_kv_tensor" not in kwargs
