# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def test_seq_idx_clamps_padded_boundaries_to_total_tokens():
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 4, 10], dtype=torch.int32),
        total_tokens=6,
    )

    assert packed_seq_params.seq_idx.tolist() == [[0, 0, 0, 0, 1, 1]]
