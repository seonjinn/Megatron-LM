# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch
import torch.distributed as dist
from torch import Tensor

CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX = "_packed_seq_params_"

PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
)

PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS = (
    "qkv_format",
    "max_seqlen_q",
    "max_seqlen_kv",
    "local_cp_size",
    "cp_group",
)

MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX = "_mamba_packed_seq_params_"

MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS = ("seq_idx",)

MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_CP_TENSOR_FIELDS = (
    "cu_seqlens_q",
    "cu_seqlens_q_padded",
)


@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    total_tokens: int = None
    seq_idx: Tensor = None
    tokens_per_sample: int = None
    pad_between_seqs: bool = None

    def __post_init__(self):
        """Pre-compute seq_idx for Mamba mixer CUDA graph compatibility.

        If total_tokens is 16 (for example), this method takes packed_seq_params.cu_seqlens_q_padded
        (or cu_seqlens_q) which is of the form [0, 5, 7, 11] and returns a tensor of the form
        [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        which is [0]*(5-0) + [1]*(7-5) + [2]*(11-7) + [3]*(16-11)
        In the above example, there are three sequences in the pack.
        In general, the output has an additional sequence index (e.g. 0, 1, 2, 3) so that any tokens
        beyond the last padded input sequence are accounted for as an extra sequence. However, If
        cu_seqlens_q_padded[-1] == max_seqlen then this additional sequence index will not be
        included.
        """
        cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if isinstance(cu_seqlens, Tensor) and self.total_tokens is not None:
            total_tokens_tensor = torch.tensor(
                [self.total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
            # Example: [0, 5, 7, 11] -> [0, 5, 7, 11, 16]
            cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
            # Example: [0, 5, 7, 11, 16] -> [5, 2, 4, 5]
            seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
            # Clamp to non-negative: cu_seqlens_q_padded may not be strictly
            # monotonic when context parallelism slices sequences across ranks,
            # or when padded cumulative lengths exceed total_tokens (e.g. the
            # appended total_tokens sentinel is smaller than cu_seqlens[-1]
            # due to padding). In either case the diff can go negative, which
            # causes torch.repeat_interleave to fail.
            seq_lengths = seq_lengths.clamp(min=0)
            # Example: [5, 2, 4, 5] -> [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
            self.seq_idx = (
                torch.repeat_interleave(
                    torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths
                )
                .to(torch.int32)
                .unsqueeze(0)  # Add a batch dimension
            )


def _cuda_graph_packed_seq_params_key(field_name: str, prefix: str) -> str:
    return f"{prefix}{field_name}"


def split_packed_seq_params_for_cuda_graph(
    packed_seq_params: PackedSeqParams | None, prefix: str = CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
) -> tuple[dict[str, Tensor | None], dict[str, object]]:
    """Split ``PackedSeqParams`` into graph Tensor inputs and static metadata.

    Transformer Engine CUDA graph inputs must be tensors or ``None``. ``PackedSeqParams`` mixes
    dynamic Tensor fields, such as cumulative sequence lengths, with static metadata, such as THD
    format and max sequence lengths. This helper keeps only the fields TE attention consumes;
    Mamba-only fields such as ``total_tokens`` and ``seq_idx`` stay outside this graph boundary.
    """
    if packed_seq_params is None:
        return {}, {}

    tensor_kwargs = {}
    for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS:
        value = getattr(packed_seq_params, field_name)
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} must be a Tensor or None for CUDA graphs, "
                f"got {type(value).__name__}."
            )
        if value is not None:
            tensor_kwargs[_cuda_graph_packed_seq_params_key(field_name, prefix)] = value

    static_metadata = {}
    for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS:
        value = getattr(packed_seq_params, field_name)
        if isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} is static CUDA graph metadata and must not be "
                "a Tensor."
            )
        static_metadata[field_name] = value

    return tensor_kwargs, static_metadata


def has_packed_seq_params_cuda_graph_kwargs(
    kwargs: Mapping[str, object], prefix: str = CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
) -> bool:
    """Return whether ``kwargs`` contains flattened ``PackedSeqParams`` Tensor fields."""
    return any(
        _cuda_graph_packed_seq_params_key(field_name, prefix) in kwargs
        for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS
    )


def build_packed_seq_params_from_cuda_graph_kwargs(
    kwargs: MutableMapping[str, object],
    static_metadata: Mapping[str, object] | None,
    prefix: str = CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    remove_from_kwargs: bool = True,
) -> PackedSeqParams | None:
    """Rebuild ``PackedSeqParams`` from flattened CUDA graph kwargs.

    Args:
        kwargs: Graph kwargs that may contain flattened packed-sequence Tensor fields.
        static_metadata: Non-Tensor metadata produced by
            :func:`split_packed_seq_params_for_cuda_graph`.
        prefix: Prefix used for flattened Tensor fields.
        remove_from_kwargs: Whether to pop consumed flattened fields from ``kwargs``.
    """
    packed_seq_params_kwargs = dict(static_metadata or {})
    found_tensor_field = False
    for field_name in PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS:
        key = _cuda_graph_packed_seq_params_key(field_name, prefix)
        if key not in kwargs:
            continue
        found_tensor_field = True
        value = kwargs.pop(key) if remove_from_kwargs else kwargs[key]
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"Flattened PackedSeqParams field {key} must be a Tensor or None, "
                f"got {type(value).__name__}."
            )
        packed_seq_params_kwargs[field_name] = value

    if not packed_seq_params_kwargs and not found_tensor_field:
        return None

    return PackedSeqParams(**packed_seq_params_kwargs)


def split_mamba_packed_seq_params_for_cuda_graph(
    packed_seq_params: PackedSeqParams | None, include_cp_fields: bool
) -> tuple[dict[str, Tensor | None], dict[str, object]]:
    """Split Mamba-consumed packed sequence inputs for a TE CUDA graph."""
    static_metadata: dict[str, object] = {
        "packed_seq_params_present": packed_seq_params is not None,
        "include_cp_fields": include_cp_fields,
    }
    if packed_seq_params is None:
        static_metadata["tensor_field_names"] = ()
        return {}, static_metadata

    if not isinstance(packed_seq_params.seq_idx, Tensor):
        raise TypeError(
            "PackedSeqParams.seq_idx must be a Tensor for Mamba CUDA graphs, "
            f"got {type(packed_seq_params.seq_idx).__name__}."
        )

    consumed_fields = MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS
    if include_cp_fields:
        consumed_fields += MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_CP_TENSOR_FIELDS

    tensor_kwargs: dict[str, Tensor | None] = {}
    for field_name in consumed_fields:
        value = getattr(packed_seq_params, field_name)
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"PackedSeqParams.{field_name} must be a Tensor or None for Mamba CUDA graphs, "
                f"got {type(value).__name__}."
            )
        if value is not None:
            tensor_kwargs[
                _cuda_graph_packed_seq_params_key(
                    field_name, MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
                )
            ] = value

    static_metadata["tensor_field_names"] = tuple(sorted(tensor_kwargs))
    return tensor_kwargs, static_metadata


def build_mamba_packed_seq_params_from_cuda_graph_kwargs(
    kwargs: Mapping[str, object], static_metadata: Mapping[str, object]
) -> PackedSeqParams | None:
    """Rebuild Mamba ``PackedSeqParams`` without regenerating ``seq_idx``."""
    metadata = dict(static_metadata)
    packed_seq_params_present = metadata.pop("packed_seq_params_present")
    include_cp_fields = metadata.pop("include_cp_fields")
    tensor_field_names = metadata.pop("tensor_field_names")
    assert not metadata, (
        "Unexpected Mamba PackedSeqParams CUDA graph static metadata: "
        f"{', '.join(sorted(metadata))}."
    )
    assert isinstance(packed_seq_params_present, bool)
    assert isinstance(include_cp_fields, bool)
    assert isinstance(tensor_field_names, tuple)
    if not packed_seq_params_present:
        return None

    seq_idx_key = _cuda_graph_packed_seq_params_key(
        "seq_idx", MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
    )
    consumed_fields = MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS
    if include_cp_fields:
        consumed_fields += MAMBA_PACKED_SEQ_PARAMS_CUDA_GRAPH_CP_TENSOR_FIELDS
    expected_keys = {
        _cuda_graph_packed_seq_params_key(
            field_name, MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
        ): field_name
        for field_name in consumed_fields
    }
    assert set(tensor_field_names) <= set(expected_keys), (
        "Mamba PackedSeqParams CUDA graph metadata contains unconsumed Tensor fields."
    )

    remaining_kwargs = dict(kwargs)
    packed_seq_params_kwargs: dict[str, Tensor | None] = {}
    required_tensor_field_names = set(tensor_field_names)
    required_tensor_field_names.add(seq_idx_key)
    for key in sorted(required_tensor_field_names):
        assert key in remaining_kwargs, (
            f"Flattened Mamba PackedSeqParams field {key} is required by the captured graph."
        )
        value = remaining_kwargs.pop(key)
        if value is not None and not isinstance(value, Tensor):
            raise TypeError(
                f"Flattened Mamba PackedSeqParams field {key} must be a Tensor or None, "
                f"got {type(value).__name__}."
            )
        packed_seq_params_kwargs[expected_keys[key]] = value
    assert isinstance(packed_seq_params_kwargs["seq_idx"], Tensor), (
        f"Flattened Mamba PackedSeqParams field {seq_idx_key} must be a Tensor."
    )

    return PackedSeqParams(**packed_seq_params_kwargs)
