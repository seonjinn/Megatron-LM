# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'checkpoint')
)

from saver_hf import HFCheckpointSaver


def _make_interleaved_qkv(num_heads, num_groups, head_dim):
    heads_per_group = num_heads // num_groups
    groups = []
    expected_q = []
    expected_k = []
    expected_v = []

    for group_idx in range(num_groups):
        q_heads = []
        for head_idx in range(heads_per_group):
            values = torch.arange(head_dim) + 1000 * group_idx + 100 * head_idx
            q_heads.append(values)
            expected_q.append(values)

        k = torch.arange(head_dim) + 10000 + 1000 * group_idx
        v = torch.arange(head_dim) + 20000 + 1000 * group_idx
        groups.append(torch.cat([*q_heads, k, v]))
        expected_k.append(k)
        expected_v.append(v)

    return (
        torch.cat(groups),
        torch.cat(expected_q),
        torch.cat(expected_k),
        torch.cat(expected_v),
    )


@pytest.mark.parametrize(
    ("num_heads", "num_groups", "tensor_parallel_size", "source_head_dim", "target_head_dim"),
    [
        # Super 3.5 configuration used by the TP8 conversion.
        (32, 2, 8, 128, 128),
        # Regression: target dimensions must be truncated inside every Q head,
        # not once across the complete per-group Q block.
        (4, 1, 2, 4, 2),
    ],
)
def test_recover_qkv_when_query_groups_are_fewer_than_tp(
    num_heads, num_groups, tensor_parallel_size, source_head_dim, target_head_dim
):
    saver = HFCheckpointSaver(args=None, queue=None)
    saver.md = SimpleNamespace(
        previous_tensor_parallel_size=tensor_parallel_size,
        num_attention_heads=num_heads,
        num_query_groups=num_groups,
        hidden_size=num_heads * target_head_dim,
    )

    qkv, expected_q, expected_k, expected_v = _make_interleaved_qkv(
        num_heads, num_groups, source_head_dim
    )
    qkv_weight = torch.stack([qkv, qkv + 1], dim=1)

    q, k, v = saver.recover_lm_qkv_weight(qkv_weight, target_head_dim)
    qb, kb, vb = saver.recover_lm_qkv_bias(qkv, target_head_dim)

    expected_q = expected_q.reshape(num_heads, source_head_dim)[:, :target_head_dim].reshape(-1)
    expected_k = expected_k.reshape(num_groups, source_head_dim)[:, :target_head_dim].reshape(-1)
    expected_v = expected_v.reshape(num_groups, source_head_dim)[:, :target_head_dim].reshape(-1)

    torch.testing.assert_close(q, torch.stack([expected_q, expected_q + 1], dim=1))
    torch.testing.assert_close(k, torch.stack([expected_k, expected_k + 1], dim=1))
    torch.testing.assert_close(v, torch.stack([expected_v, expected_v + 1], dim=1))
    torch.testing.assert_close(qb, expected_q)
    torch.testing.assert_close(kb, expected_k)
    torch.testing.assert_close(vb, expected_v)


def test_recover_qkv_rejects_non_divisible_query_groups():
    saver = HFCheckpointSaver(args=None, queue=None)
    saver.md = SimpleNamespace(
        previous_tensor_parallel_size=4,
        num_attention_heads=3,
        num_query_groups=2,
        hidden_size=12,
    )
    qkv = torch.zeros((3 + 2 * 2) * 4, 1)

    with pytest.raises(AssertionError, match="must be divisible"):
        saver.recover_lm_qkv_weight(qkv)
