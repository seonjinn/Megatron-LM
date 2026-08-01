# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the fixed-shape packed-THD CUDA Graph input contract."""

from argparse import ArgumentParser

import pytest
import torch

import megatron.core.packed_seq_params as packed_seq
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


def _make_packed_seq_params() -> packed_seq.PackedSeqParams:
    valid = torch.tensor([0, 3, 5], dtype=torch.int32)
    padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    return packed_seq.PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=valid,
        cu_seqlens_kv=valid.clone(),
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded.clone(),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        pad_between_seqs=True,
        cp_partition_mode="contiguous",
        tokens_per_sample=8,
    )


def _make_transformer_config(**overrides: object) -> TransformerConfig:
    values = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "cuda_graph_impl": "transformer_engine",
        "max_seqlen_per_dp_cp_rank": 128,
        "pad_packed_seq_alignment": "max",
        "thd_max_packed_sequences": 100,
        "thd_tail_padding_policy": "extend_last",
    }
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.internal
def test_extend_last_preserves_real_boundaries_and_fixes_all_input_shapes():
    """Static padding must not replace compact valid-token boundaries."""
    original = _make_packed_seq_params()
    original_valid = original.cu_seqlens_q.clone()
    original_padded = original.cu_seqlens_q_padded.clone()
    tokens = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    padding_mask = torch.tensor(
        [[False, False, False, True, False, False, True, True]], dtype=torch.bool
    )

    padded_tokens, _, _, _, padded, padded_mask = packed_seq.pad_sequence_for_thd(
        tokens,
        None,
        None,
        None,
        original,
        target_len=10,
        max_num_seqs=4,
        tail_padding_policy="extend_last",
        padding_mask=padding_mask,
        cp_size=1,
        cp_rank=0,
    )

    assert padded_tokens.shape == (1, 10)
    assert padded_mask.shape == (1, 10)
    assert padded_tokens[0, 8:].tolist() == [0, 0]
    assert padded_mask.tolist() == [
        [False, False, False, True, False, False, True, True, True, True]
    ]
    assert padded.cu_seqlens_q.tolist() == [0, 3, 5, 5, 5]
    assert padded.cu_seqlens_kv.tolist() == [0, 3, 5, 5, 5]
    assert padded.cu_seqlens_q_padded.tolist() == [0, 4, 10, 10, 10]
    assert padded.cu_seqlens_kv_padded.tolist() == [0, 4, 10, 10, 10]
    assert padded.pad_between_seqs is True
    assert padded.cp_partition_mode == "contiguous"
    assert padded.tokens_per_sample == 8
    assert torch.equal(original.cu_seqlens_q, original_valid)
    assert torch.equal(original.cu_seqlens_q_padded, original_padded)


@pytest.mark.internal
def test_static_padding_rejects_observed_token_count_above_configured_bound():
    """A token overflow must fail instead of truncating the packed batch."""
    params = _make_packed_seq_params()
    tokens = torch.ones((1, 9), dtype=torch.int64)

    with pytest.raises(ValueError, match=r"observed token count 9.*configured bound 8"):
        packed_seq.pad_sequence_for_thd(
            tokens,
            None,
            None,
            None,
            params,
            target_len=8,
            max_num_seqs=4,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )


@pytest.mark.internal
def test_static_padding_rejects_observed_sequence_count_above_configured_bound():
    """A sequence-count overflow must fail instead of truncating cu_seqlens."""
    cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    params = packed_seq.PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens.clone(),
        cu_seqlens_q_padded=cu_seqlens.clone(),
        cu_seqlens_kv_padded=cu_seqlens.clone(),
        max_seqlen_q=2,
        max_seqlen_kv=2,
    )

    with pytest.raises(ValueError, match=r"observed sequence count 3.*configured bound 2"):
        packed_seq.pad_sequence_for_thd(
            torch.ones((1, 6), dtype=torch.int64),
            None,
            None,
            None,
            params,
            target_len=8,
            max_num_seqs=2,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )


@pytest.mark.internal
def test_static_thd_arguments_parse_exact_profile_values():
    """The reusable profile's four THD graph options must be typed CLI arguments."""
    parser = ArgumentParser()
    ArgumentGroupFactory(TransformerConfig).build_group(parser, title="Transformer")

    args = parser.parse_args(
        [
            "--pad-packed-seq-alignment",
            "max",
            "--thd-max-packed-sequences",
            "100",
            "--thd-tail-padding-policy",
            "extend_last",
            "--cuda-graph-memory-report",
        ]
    )

    assert args.pad_packed_seq_alignment == "max"
    assert args.thd_max_packed_sequences == 100
    assert args.thd_tail_padding_policy == "extend_last"
    assert args.cuda_graph_memory_report is True


@pytest.mark.internal
def test_complete_static_thd_graph_configuration_is_accepted():
    """Explicit fixed token and sequence capacities form a valid graph contract."""
    config = _make_transformer_config(cuda_graph_memory_report=True)

    assert config.max_seqlen_per_dp_cp_rank == 128
    assert config.pad_packed_seq_alignment == "max"
    assert config.thd_max_packed_sequences == 100
    assert config.thd_tail_padding_policy == "extend_last"
    assert config.cuda_graph_memory_report is True


@pytest.mark.internal
@pytest.mark.parametrize(
    "missing,overrides",
    [
        ("--max-seqlen-per-dp-cp-rank", {"max_seqlen_per_dp_cp_rank": None}),
        ("--pad-packed-seq-alignment", {"pad_packed_seq_alignment": None}),
        ("--thd-max-packed-sequences", {"thd_max_packed_sequences": None}),
    ],
)
def test_static_thd_graph_configuration_rejects_missing_fixed_bound(
    missing: str, overrides: dict[str, object]
):
    """A partial static contract would permit replay with changing input shapes."""
    with pytest.raises(ValueError, match=missing):
        _make_transformer_config(**overrides)


@pytest.mark.internal
def test_non_thd_graph_configuration_does_not_require_static_thd_bounds():
    """Existing graph users without explicit THD fields remain unaffected."""
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=4, cuda_graph_impl="transformer_engine"
    )

    assert config.cuda_graph_impl == "transformer_engine"
