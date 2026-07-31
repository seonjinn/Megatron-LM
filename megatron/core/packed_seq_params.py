# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, MutableMapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig

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
    "pad_between_seqs",
    "tokens_per_sample",
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


def get_thd_padding_kwargs(
    config: ModelParallelConfig,
) -> tuple[int | None, int | None, int | None]:
    """Return THD token alignment, fixed token length, and sequence capacity."""
    alignment = config.pad_packed_seq_alignment
    target_len = config.pad_packed_seq_to
    max_sequences = config.thd_max_packed_sequences
    if config.cuda_graph_impl == "transformer_engine":
        assert max_sequences is not None, (
            "THD Transformer Engine CUDA graphs require " "thd_max_packed_sequences."
        )
    return alignment, target_len, max_sequences


def _pad_sequence_tensor(tensor: Tensor | None, target_len: int) -> Tensor | None:
    """Pad a token-like tensor along its final dimension."""
    if tensor is None:
        return None
    actual_len = tensor.shape[-1]
    assert (
        actual_len <= target_len
    ), f"Packed THD tensor length ({actual_len}) exceeds padding target ({target_len})."
    return F.pad(tensor, (0, target_len - actual_len)) if actual_len < target_len else tensor


def _pad_cu_seqlens(cu_seqlens: Tensor, target_entries: int) -> Tensor:
    """Pad cumulative sequence lengths to a fixed entry capacity."""
    actual_entries = cu_seqlens.numel()
    assert actual_entries <= target_entries, (
        f"Actual THD sequence count ({actual_entries - 1}) exceeds configured capacity "
        f"({target_entries - 1})."
    )
    if actual_entries == target_entries:
        return cu_seqlens
    padded = torch.full(
        (target_entries,), cu_seqlens[-1].item(), dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    padded[:actual_entries] = cu_seqlens
    return padded


def _append_cu_seqlens_endpoint(cu_seqlens: Tensor, endpoint: int) -> Tensor:
    """Append one dummy-sequence endpoint."""
    tail = torch.full((1,), endpoint, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.cat((cu_seqlens, tail))


def _resolve_thd_context_parallel_size(
    packed_seq_params: PackedSeqParams, context_parallel_size: int | None
) -> int:
    """Resolve CP size without reading a global process group."""
    if context_parallel_size is not None:
        return int(context_parallel_size)
    if packed_seq_params.cp_group is not None:
        return int(dist.get_world_size(group=packed_seq_params.cp_group))
    if packed_seq_params.local_cp_size is not None:
        return int(packed_seq_params.local_cp_size)
    return 1


def pad_sequence_for_thd(
    tokens: Tensor | None,
    labels: Tensor | None,
    loss_mask: Tensor | None,
    position_ids: Tensor | None,
    packed_seq_params: PackedSeqParams,
    alignment: int | None = None,
    target_len: int | None = None,
    max_num_seqs: int | None = None,
    context_parallel_size: int | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, PackedSeqParams, Tensor]:
    """Pad packed THD inputs and cumulative metadata for CUDA graph replay.

    The token tail is represented as one appended dummy sequence. Logical
    cumulative lengths preserve compact valid-token coordinates, while padded
    cumulative lengths preserve physical storage coordinates.
    """
    assert (alignment is None) != (
        target_len is None
    ), "Exactly one of alignment or target_len must be provided for THD padding."

    physical_q = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    assert physical_q is not None, "THD padding requires cu_seqlens_q metadata."
    physical_kv = (
        packed_seq_params.cu_seqlens_kv_padded
        if packed_seq_params.cu_seqlens_kv_padded is not None
        else packed_seq_params.cu_seqlens_kv
    )
    if physical_kv is None:
        physical_kv = physical_q
    global_q_actual_len = int(physical_q[-1].item())
    global_kv_actual_len = int(physical_kv[-1].item())
    cp_size = _resolve_thd_context_parallel_size(packed_seq_params, context_parallel_size)

    local_tensor_lengths = [
        int(tensor.shape[-1])
        for tensor in (tokens, labels, loss_mask, position_ids)
        if tensor is not None
    ]
    if local_tensor_lengths:
        assert all(length == local_tensor_lengths[0] for length in local_tensor_lengths), (
            "THD token-like tensors must have the same pre-padding local length, "
            f"got {local_tensor_lengths}."
        )
        local_actual_len = local_tensor_lengths[0]
    else:
        assert cp_size == 1, (
            "THD CP metadata-only padding requires a pre-padding local token-like tensor "
            "to determine exact local occupancy."
        )
        local_actual_len = global_q_actual_len

    if target_len is None:
        assert alignment is not None and alignment > 0
        target_len = ((local_actual_len + alignment - 1) // alignment) * alignment
    else:
        target_len = int(target_len)

    global_target_len = target_len * cp_size
    assert global_q_actual_len <= global_target_len, (
        f"Packed THD Q length ({global_q_actual_len}) exceeds padding target "
        f"({global_target_len})."
    )
    assert global_kv_actual_len <= global_target_len, (
        f"Packed THD KV length ({global_kv_actual_len}) exceeds padding target "
        f"({global_target_len})."
    )

    tokens = _pad_sequence_tensor(tokens, target_len)
    labels = _pad_sequence_tensor(labels, target_len)
    loss_mask = _pad_sequence_tensor(loss_mask, target_len)
    position_ids = _pad_sequence_tensor(position_ids, target_len)

    cu_seqlens_q = packed_seq_params.cu_seqlens_q
    assert cu_seqlens_q is not None, "THD padding requires cu_seqlens_q metadata."
    cu_seqlens_kv = (
        packed_seq_params.cu_seqlens_kv
        if packed_seq_params.cu_seqlens_kv is not None
        else cu_seqlens_q
    )
    cu_seqlens_q_padded = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else cu_seqlens_q
    )
    cu_seqlens_kv_padded = (
        packed_seq_params.cu_seqlens_kv_padded
        if packed_seq_params.cu_seqlens_kv_padded is not None
        else cu_seqlens_kv
    )

    q_dummy_len = global_target_len - int(cu_seqlens_q_padded[-1].item())
    kv_dummy_len = global_target_len - int(cu_seqlens_kv_padded[-1].item())
    if q_dummy_len or kv_dummy_len:
        if cp_size > 1:
            for name, dummy_len in (("Q", q_dummy_len), ("KV", kv_dummy_len)):
                assert dummy_len % (2 * cp_size) == 0, (
                    f"THD {name} dummy padding length ({dummy_len}) must be divisible by "
                    f"2 * context_parallel_size ({2 * cp_size}) for zigzag partitioning."
                )

        q_has_inter_sequence_padding = (
            packed_seq_params.pad_between_seqs is True
            or not torch.equal(cu_seqlens_q, cu_seqlens_q_padded)
        )
        kv_has_inter_sequence_padding = (
            packed_seq_params.pad_between_seqs is True
            or not torch.equal(cu_seqlens_kv, cu_seqlens_kv_padded)
        )
        q_dummy_end = (
            int(cu_seqlens_q[-1].item()) + q_dummy_len
            if q_has_inter_sequence_padding
            else global_target_len
        )
        kv_dummy_end = (
            int(cu_seqlens_kv[-1].item()) + kv_dummy_len
            if kv_has_inter_sequence_padding
            else global_target_len
        )
        cu_seqlens_q = _append_cu_seqlens_endpoint(cu_seqlens_q, q_dummy_end)
        cu_seqlens_kv = _append_cu_seqlens_endpoint(cu_seqlens_kv, kv_dummy_end)
        cu_seqlens_q_padded = _append_cu_seqlens_endpoint(cu_seqlens_q_padded, global_target_len)
        cu_seqlens_kv_padded = _append_cu_seqlens_endpoint(cu_seqlens_kv_padded, global_target_len)

    if max_num_seqs is not None:
        target_entries = int(max_num_seqs) + 1
        cu_seqlens_q = _pad_cu_seqlens(cu_seqlens_q, target_entries)
        cu_seqlens_kv = _pad_cu_seqlens(cu_seqlens_kv, target_entries)
        cu_seqlens_q_padded = _pad_cu_seqlens(cu_seqlens_q_padded, target_entries)
        cu_seqlens_kv_padded = _pad_cu_seqlens(cu_seqlens_kv_padded, target_entries)

    padded_params = PackedSeqParams(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        max_seqlen_q=(
            global_target_len
            if max_num_seqs is not None
            else max(packed_seq_params.max_seqlen_q or 0, q_dummy_len)
        ),
        max_seqlen_kv=(
            global_target_len
            if max_num_seqs is not None
            else max(packed_seq_params.max_seqlen_kv or 0, kv_dummy_len)
        ),
        local_cp_size=packed_seq_params.local_cp_size,
        cp_group=packed_seq_params.cp_group,
        total_tokens=None if max_num_seqs is not None else target_len,
        tokens_per_sample=packed_seq_params.tokens_per_sample,
        pad_between_seqs=True if max_num_seqs is not None else packed_seq_params.pad_between_seqs,
    )
    padding_mask = (
        torch.arange(target_len, device=physical_q.device).unsqueeze(0) >= local_actual_len
    )
    return tokens, labels, loss_mask, position_ids, padded_params, padding_mask


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
