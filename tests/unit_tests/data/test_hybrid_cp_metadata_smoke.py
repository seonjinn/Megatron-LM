"""Dependency-light checks for multimodal HybridCP packed metadata."""

import unittest

from megatron.core.packed_seq_params import PackedSeqParams
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


if __name__ == "__main__":
    unittest.main()
