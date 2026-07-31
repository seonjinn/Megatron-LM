# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Immutable non-Tensor replay state for partial MoE CUDA graphs."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TensorReplaySignature:
    """Exact Tensor metadata required by a CUDA graph continuation."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: tuple[int, ...]


@dataclass(frozen=True)
class AlltoAllCudaGraphState:
    """Structural AlltoAll metadata produced by dispatch preprocessing."""

    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size
    capacity: int | None
    num_out_tokens: int | None


@dataclass(frozen=True)
class HybridEPCudaGraphState:
    """Structural fixed-capacity HybridEP metadata produced before dispatch."""

    original_num_tokens: int
    padded_num_tokens: int
    capacity: int | None
    num_permuted_tokens: int
    tokens_per_expert: tuple[int, ...] | None


@dataclass(frozen=True)
class MoECudaGraphReplayState:
    """Dispatcher-owned replay state paired with one captured graph index."""

    dispatcher_kind: Literal["alltoall", "flex-hybridep"]
    input_signature: TensorReplaySignature
    flattened_input_shape: torch.Size
    topology_fingerprint: tuple[tuple[str, object], ...]
    backend_state: AlltoAllCudaGraphState | HybridEPCudaGraphState


def get_tensor_replay_signature(tensor: torch.Tensor) -> TensorReplaySignature:
    """Return the exact replay-relevant metadata for ``tensor``."""

    stride = tuple(tensor.stride()) if tensor.layout == torch.strided else ()
    return TensorReplaySignature(
        shape=tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
        stride=stride,
    )


def get_flattened_input_shape(tensor: torch.Tensor) -> torch.Size:
    """Return the explicit ``[physical_tokens, hidden]`` input contract."""

    if tensor.ndim == 0 or tensor.shape[-1] == 0:
        raise RuntimeError(
            f"MoE CUDA graph input must have a non-empty hidden dimension, got {tensor.shape}."
        )
    return torch.Size((tensor.numel() // tensor.shape[-1], tensor.shape[-1]))


def validate_tensor_replay_signature(
    tensor: torch.Tensor, expected: TensorReplaySignature, *, boundary: str
) -> None:
    """Fail when ``tensor`` does not exactly match ``expected`` at a graph boundary."""

    actual = get_tensor_replay_signature(tensor)
    if actual != expected:
        raise RuntimeError(
            f"MoE CUDA graph {boundary} signature mismatch: expected {expected}, got {actual}. "
            "Recapture the graph for this physical capacity."
        )


def to_optional_int(value: object, *, field_name: str) -> int | None:
    """Convert scalar dispatcher metadata to an immutable Python integer."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if torch.is_tensor(value) and value.numel() == 1:
        return int(value.item())
    raise RuntimeError(f"MoE CUDA graph state requires scalar {field_name}, got {value!r}.")
