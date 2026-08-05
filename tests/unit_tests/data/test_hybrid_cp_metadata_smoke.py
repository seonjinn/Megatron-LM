"""Dependency-light checks for multimodal HybridCP packed metadata."""

import unittest

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.multimodal.llava_model import (
    update_multimodal_packed_seq_params,
)
from megatron.core.ssm.mamba_mixer import _slice_packed_seq_idx_for_sequence_parallel
from megatron.core.utils import set_hybrid_cp_metadata


class _Group:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class HybridCPMetadataTest(unittest.TestCase):
    def test_group_metadata(self):
        group = _Group(2)
        params = PackedSeqParams(qkv_format="thd")
        self.assertIs(set_hybrid_cp_metadata(params, 2, group), params)
        self.assertEqual(params.local_cp_size, 2)
        self.assertIs(params.cp_group, group)

    def test_single_rank_metadata(self):
        params = PackedSeqParams(qkv_format="thd")
        set_hybrid_cp_metadata(params, 1)
        self.assertEqual(params.local_cp_size, 1)
        self.assertIsNone(params.cp_group)

    def test_multimodal_expansion_rebuilds_mamba_seq_idx(self):
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_kv_padded=torch.tensor([0, 4], dtype=torch.int32),
            total_tokens=4,
        )
        update_multimodal_packed_seq_params(params, torch.tensor([8], dtype=torch.int32))
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 8])
        self.assertEqual(params.cu_seqlens_q_padded.tolist(), [0, 8])
        self.assertEqual(params.total_tokens, 8)
        self.assertEqual(params.seq_idx.shape, (1, 8))

    def test_hybrid_cp_mamba_accepts_single_sample_media_expansion(self):
        seq_idx = torch.zeros((1, 4), dtype=torch.int32)
        result = _slice_packed_seq_idx_for_sequence_parallel(
            seq_idx, local_tokens=8, tp_rank=0, tp_size=8, allow_short_metadata=True
        )
        self.assertEqual(result.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
