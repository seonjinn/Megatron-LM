# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical parity for compact and fixed-capacity Transformer Engine THD."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 1024
SIMILARITY_THRESHOLD = 0.999


def _round_up(value: int, divisor: int) -> int:
    return value if divisor <= 1 else (value + divisor - 1) // divisor * divisor


def _padded_lengths(seqlens: list[int], cp_size: int, tp_size: int) -> list[int]:
    padded = [_round_up(length, 2 * cp_size) for length in seqlens]
    remainder = sum(padded) % tp_size
    if remainder:
        padded[-1] += tp_size - remainder
    return padded


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    endpoints = [0]
    for length in lengths:
        endpoints.append(endpoints[-1] + length)
    return torch.tensor(endpoints, dtype=torch.int32, device="cuda")


def _compact_params(seqlens: list[int], cp_size: int, tp_size: int) -> PackedSeqParams:
    padded = _padded_lengths(seqlens, cp_size, tp_size)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=_cu_seqlens(seqlens),
        cu_seqlens_kv=_cu_seqlens(seqlens),
        cu_seqlens_q_padded=_cu_seqlens(padded),
        cu_seqlens_kv_padded=_cu_seqlens(padded),
        max_seqlen_q=max(padded),
        max_seqlen_kv=max(padded),
    )


def _zigzag_split(tensor: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    chunk_size = tensor.shape[0] // (2 * cp_size)
    first = tensor.narrow(0, cp_rank * chunk_size, chunk_size)
    second_rank = 2 * cp_size - cp_rank - 1
    second = tensor.narrow(0, second_rank * chunk_size, chunk_size)
    return torch.cat((first, second), dim=0)


def _zigzag_merge(chunks: list[torch.Tensor], cp_size: int) -> torch.Tensor:
    half = chunks[0].shape[0] // 2
    parts: list[torch.Tensor | None] = [None] * (2 * cp_size)
    for rank, chunk in enumerate(chunks):
        parts[rank] = chunk[:half]
        parts[2 * cp_size - rank - 1] = chunk[half:]
    assert all(part is not None for part in parts)
    return torch.cat([part for part in parts if part is not None], dim=0)


def _compact_input(
    sequence_data: list[torch.Tensor],
    seqlens: list[int],
    *,
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    padded = _padded_lengths(seqlens, cp_size, tp_size)
    cp_parts = []
    for data, seq_len, padded_len in zip(sequence_data, seqlens, padded):
        if padded_len > seq_len:
            data = torch.cat((data, data.new_zeros((padded_len - seq_len, HIDDEN_SIZE))))
        cp_parts.append(_zigzag_split(data, cp_rank, cp_size))
    cp_local = torch.cat(cp_parts, dim=0)
    tp_tokens = cp_local.shape[0] // tp_size
    return cp_local.narrow(0, tp_rank * tp_tokens, tp_tokens).unsqueeze(1).contiguous()


def _fixed_input_and_params(
    sequence_data: list[torch.Tensor],
    seqlens: list[int],
    *,
    capacity_tokens: int,
    max_num_seqs: int,
    cp_rank: int,
    cp_size: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, PackedSeqParams]:
    compact_cp = _compact_input(
        sequence_data, seqlens, cp_rank=cp_rank, cp_size=cp_size, tp_rank=0, tp_size=1
    ).squeeze(1)
    cp_target = capacity_tokens // cp_size
    if compact_cp.shape[0] > cp_target:
        raise ValueError("fixed THD capacity is smaller than natural occupancy")
    fixed_cp = torch.cat(
        (compact_cp, compact_cp.new_zeros((cp_target - compact_cp.shape[0], HIDDEN_SIZE)))
    )
    tp_tokens = cp_target // tp_size
    fixed_input = fixed_cp.narrow(0, tp_rank * tp_tokens, tp_tokens).unsqueeze(1).contiguous()
    compact_params = _compact_params(seqlens, cp_size, tp_size)
    padded = _padded_lengths(seqlens, cp_size, tp_size)
    token_probe = torch.zeros((1, sum(padded) // cp_size), dtype=torch.int64, device="cuda")
    _, _, _, _, fixed_params, _ = pad_sequence_for_thd(
        tokens=token_probe,
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=compact_params,
        target_len=cp_target,
        max_num_seqs=max_num_seqs,
        context_parallel_size=cp_size,
    )
    return fixed_input, fixed_params


def _gather_valid(
    local: torch.Tensor, seqlens: list[int], *, cp_size: int, tp_size: int
) -> torch.Tensor:
    tp_chunks = [torch.empty_like(local) for _ in range(tp_size)]
    dist.all_gather(
        tp_chunks, local.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
    )
    cp_local = torch.cat(tp_chunks, dim=0)
    padded = _padded_lengths(seqlens, cp_size, tp_size)
    offset = 0
    valid_sequences = []
    cp_group = parallel_state.get_context_parallel_group()
    for seq_len, padded_len in zip(seqlens, padded):
        local_len = padded_len // cp_size
        local_sequence = cp_local[offset : offset + local_len]
        cp_chunks = [torch.empty_like(local_sequence) for _ in range(cp_size)]
        dist.all_gather(cp_chunks, local_sequence.contiguous(), group=cp_group)
        valid_sequences.append(_zigzag_merge(cp_chunks, cp_size)[:seq_len])
        offset += local_len
    return torch.cat(valid_sequences, dim=0)


def _similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double()
    second = second.double()
    denominator = (first.square() + second.square()).sum()
    if not denominator:
        return 1.0
    return float((2.0 * (first * second).sum() / denominator).item())


def _assert_similar(name: str, first: torch.Tensor, second: torch.Tensor) -> None:
    similarity = _similarity(first, second)
    assert similarity > SIMILARITY_THRESHOLD, (
        f"{name}: tensor similarity {similarity:.6f} is below " f"{SIMILARITY_THRESHOLD}"
    )


def _build_layer(tp_size: int, cp_size: int) -> TransformerLayer:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        num_query_groups=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        sequence_parallel=True,
        cp_comm_type="p2p",
    )
    layer = TransformerLayer(config, get_gpt_layer_with_transformer_engine_spec().submodules)
    return layer.cuda()


@pytest.mark.internal
def test_fixed_capacity_thd_matches_compact_thd() -> None:
    """Compare eager output and gradients at the exact Nano TP2/CP2/SP topology."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 16:
        pytest.skip("Nano fixed-capacity THD parity requires exactly 16 ranks")
    if not torch.cuda.is_available():
        pytest.fail("Nano fixed-capacity THD parity requires CUDA", pytrace=False)

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    Utils.world_size = 16
    Utils.rank = global_rank
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=2, context_parallel_size=2
    )
    try:
        model_parallel_cuda_manual_seed(42)
        layer = _build_layer(tp_size=2, cp_size=2)
        cp_rank = parallel_state.get_context_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        seqlens = [17, 31, 11]

        torch.manual_seed(42 + dp_rank)
        sequence_data = [
            torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
            for seq_len in seqlens
        ]
        torch.manual_seed(142 + dp_rank)
        gradient_data = [
            torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
            for seq_len in seqlens
        ]

        compact_input = (
            _compact_input(
                sequence_data, seqlens, cp_rank=cp_rank, cp_size=2, tp_rank=tp_rank, tp_size=2
            )
            .detach()
            .requires_grad_(True)
        )
        compact_gradient = _compact_input(
            gradient_data, seqlens, cp_rank=cp_rank, cp_size=2, tp_rank=tp_rank, tp_size=2
        )
        compact_output, _ = layer(
            hidden_states=compact_input, packed_seq_params=_compact_params(seqlens, 2, 2)
        )
        compact_valid_output = _gather_valid(compact_output.detach(), seqlens, cp_size=2, tp_size=2)
        compact_output.backward(compact_gradient)
        compact_valid_input_grad = _gather_valid(
            compact_input.grad.detach(), seqlens, cp_size=2, tp_size=2
        )
        compact_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
            if parameter.grad is not None
        }
        layer.zero_grad(set_to_none=True)

        fixed_input, fixed_params = _fixed_input_and_params(
            sequence_data,
            seqlens,
            capacity_tokens=128,
            max_num_seqs=8,
            cp_rank=cp_rank,
            cp_size=2,
            tp_rank=tp_rank,
            tp_size=2,
        )
        fixed_input = fixed_input.detach().requires_grad_(True)
        fixed_gradient, _ = _fixed_input_and_params(
            gradient_data,
            seqlens,
            capacity_tokens=128,
            max_num_seqs=8,
            cp_rank=cp_rank,
            cp_size=2,
            tp_rank=tp_rank,
            tp_size=2,
        )
        fixed_output, _ = layer(hidden_states=fixed_input, packed_seq_params=fixed_params)
        fixed_valid_output = _gather_valid(fixed_output.detach(), seqlens, cp_size=2, tp_size=2)
        fixed_output.backward(fixed_gradient)
        fixed_valid_input_grad = _gather_valid(
            fixed_input.grad.detach(), seqlens, cp_size=2, tp_size=2
        )
        fixed_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
            if parameter.grad is not None
        }

        assert compact_parameter_grads.keys() == fixed_parameter_grads.keys()
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        for name, parameter in layer.named_parameters():
            if name not in compact_parameter_grads:
                continue
            dist.all_reduce(compact_parameter_grads[name], group=dp_cp_group)
            dist.all_reduce(fixed_parameter_grads[name], group=dp_cp_group)
            if getattr(parameter, "sequence_parallel", False):
                dist.all_reduce(compact_parameter_grads[name], group=tp_group)
                dist.all_reduce(fixed_parameter_grads[name], group=tp_group)

        assert fixed_params.cu_seqlens_q.numel() == 9
        assert fixed_params.cu_seqlens_q_padded.numel() == 9
        assert fixed_params.max_seqlen_q == 128
        assert fixed_input.shape[0] == 32
        _assert_similar("fixed-capacity valid output", compact_valid_output, fixed_valid_output)
        _assert_similar(
            "fixed-capacity valid input gradient", compact_valid_input_grad, fixed_valid_input_grad
        )
        for name in compact_parameter_grads:
            _assert_similar(
                f"fixed-capacity grad[{name}]",
                compact_parameter_grads[name],
                fixed_parameter_grads[name],
            )
    finally:
        Utils.destroy_model_parallel()
