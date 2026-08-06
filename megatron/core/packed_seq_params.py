# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, replace
from typing import Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


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
    pad_between_seqs: Optional[bool] = None
    cp_partition_mode: Literal["zigzag", "contiguous"] = "zigzag"
    tokens_per_sample: int = None
    # ``False`` marks an overflow batch that must bypass a fixed-shape graph.
    # ``None`` preserves the normal eligibility inference for eager callers.
    cuda_graph_eligible: Optional[bool] = None

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


def _pad_seq_tensor(tensor: Optional[Tensor], target_len: int) -> Optional[Tensor]:
    """Pad a token-like tensor to a fixed final-dimension length."""
    if tensor is None:
        return None

    observed_tokens = int(tensor.shape[-1])
    if observed_tokens > target_len:
        raise ValueError(
            f"Packed THD observed token count {observed_tokens} exceeds configured bound "
            f"{target_len}; refusing to truncate."
        )
    if observed_tokens == target_len:
        return tensor
    return F.pad(tensor, (0, target_len - observed_tokens), value=0)


def _pad_padding_mask(mask: Tensor, target_len: int) -> Tensor:
    """Pad a bool mask with ``True`` for every appended token slot."""
    observed_tokens = int(mask.shape[-1])
    if observed_tokens > target_len:
        raise ValueError(
            f"Packed THD observed token count {observed_tokens} exceeds configured bound "
            f"{target_len}; refusing to truncate."
        )
    if observed_tokens == target_len:
        return mask

    tail_shape = list(mask.shape)
    tail_shape[-1] = target_len - observed_tokens
    tail = torch.ones(tail_shape, dtype=mask.dtype, device=mask.device)
    return torch.cat((mask, tail), dim=-1)


def _pad_cu_seqlens(cu_seqlens: Optional[Tensor], target_entries: int) -> Optional[Tensor]:
    """Pad cumulative sequence metadata by repeating its final boundary."""
    if cu_seqlens is None:
        return None

    observed_sequences = int(cu_seqlens.shape[0]) - 1
    configured_sequences = target_entries - 1
    if observed_sequences > configured_sequences:
        raise ValueError(
            f"Packed THD observed sequence count {observed_sequences} exceeds configured bound "
            f"{configured_sequences}; refusing to truncate."
        )
    if observed_sequences == configured_sequences:
        return cu_seqlens

    padded = torch.full(
        (target_entries,),
        int(cu_seqlens[-1].item()),
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    padded[: cu_seqlens.shape[0]] = cu_seqlens
    return padded


def _append_boundary(cu_seqlens: Optional[Tensor], boundary: int) -> Optional[Tensor]:
    """Append one cumulative sequence boundary without mutating the input."""
    if cu_seqlens is None:
        return None
    tail = torch.full((1,), boundary, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.cat((cu_seqlens, tail), dim=0)


def _extend_last_boundary(cu_seqlens: Optional[Tensor], boundary: int) -> Optional[Tensor]:
    """Move only the final physical boundary to the fixed token capacity."""
    if cu_seqlens is None:
        return None
    if cu_seqlens.numel() < 2:
        raise ValueError("THD extend_last padding requires at least one packed sequence.")

    observed_tokens = int(cu_seqlens[-1].item())
    if observed_tokens > boundary:
        raise ValueError(
            f"Packed THD observed token count {observed_tokens} exceeds configured bound "
            f"{boundary}; refusing to truncate."
        )
    extended = cu_seqlens.clone()
    extended[-1] = boundary
    return extended


def _resolve_thd_cp_size(
    packed_seq_params: PackedSeqParams,
    cp_group: Optional[dist.ProcessGroup],
    cp_size: Optional[int],
) -> int:
    """Resolve the explicit, packed, or initialized context-parallel size."""
    if cp_group is not None:
        return int(dist.get_world_size(group=cp_group))
    if cp_size is not None:
        return int(cp_size)
    if packed_seq_params.cp_group is not None:
        return int(dist.get_world_size(group=packed_seq_params.cp_group))
    if packed_seq_params.local_cp_size is not None:
        return int(packed_seq_params.local_cp_size)

    from megatron.core import parallel_state

    return int(parallel_state.get_context_parallel_world_size()) or 1


def _updated_max_seqlen(
    cu_seqlens_padded: Optional[Tensor], current_max: Optional[int]
) -> Optional[int]:
    """Return a maximum that covers every resulting physical sequence."""
    if cu_seqlens_padded is None or cu_seqlens_padded.numel() < 2:
        return current_max
    physical_lengths = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    resulting_max = int(physical_lengths.max().item())
    return max(current_max or 0, resulting_max)


def packed_thd_exceeds_capacity(
    packed_seq_params: PackedSeqParams,
    token_tensors: Tuple[Optional[Tensor], ...],
    target_len: Optional[int],
    max_num_seqs: Optional[int],
) -> bool:
    """Return whether a packed batch exceeds a declared static THD surface."""
    if target_len is not None:
        observed_tokens = max(
            [int(tensor.shape[-1]) for tensor in token_tensors if tensor is not None]
            + [
                int(cu_seqlens[-1].item())
                for cu_seqlens in (
                    packed_seq_params.cu_seqlens_q,
                    packed_seq_params.cu_seqlens_kv,
                    packed_seq_params.cu_seqlens_q_padded,
                    packed_seq_params.cu_seqlens_kv_padded,
                )
                if cu_seqlens is not None
            ]
        )
        if observed_tokens > int(target_len):
            return True

    if max_num_seqs is not None:
        observed_sequences = max(
            [
                int(cu_seqlens.shape[0]) - 1
                for cu_seqlens in (
                    packed_seq_params.cu_seqlens_q,
                    packed_seq_params.cu_seqlens_kv,
                )
                if cu_seqlens is not None
            ]
            or [0]
        )
        if observed_sequences > int(max_num_seqs):
            return True
    return False


def packed_thd_matches_static_bounds(
    packed_seq_params: PackedSeqParams,
    max_seqlen_per_dp_cp_rank: Optional[int],
    thd_max_packed_sequences: Optional[int],
    tail_padding_policy: Literal["append_dummy_seq", "extend_last"],
    cp_size: int = 1,
) -> bool:
    """Check that runtime THD metadata matches a TE graph's fixed input shape."""
    if (
        max_seqlen_per_dp_cp_rank is None
        or thd_max_packed_sequences is None
        or cp_size <= 0
    ):
        return False

    expected_tokens = int(max_seqlen_per_dp_cp_rank) * int(cp_size)
    reserve_dummy_slot = tail_padding_policy != "extend_last"
    expected_entries = int(thd_max_packed_sequences) + 1 + int(reserve_dummy_slot)
    metadata = (
        packed_seq_params.cu_seqlens_q,
        packed_seq_params.cu_seqlens_kv,
        packed_seq_params.cu_seqlens_q_padded,
        packed_seq_params.cu_seqlens_kv_padded,
    )
    if any(value is None for value in metadata):
        return False
    if any(int(value.shape[0]) != expected_entries for value in metadata):
        return False
    if any(
        int(value[-1].item()) != expected_tokens
        for value in (
            packed_seq_params.cu_seqlens_q_padded,
            packed_seq_params.cu_seqlens_kv_padded,
        )
    ):
        return False
    if packed_seq_params.total_tokens is not None and int(packed_seq_params.total_tokens) != expected_tokens:
        return False
    return True


def _round_up_to_alignment(value: int, alignment: int) -> int:
    """Round ``value`` up to a positive alignment."""
    if alignment <= 0:
        raise ValueError(f"Packed THD alignment must be positive, got {alignment}.")
    return ((value + alignment - 1) // alignment) * alignment


def get_thd_padding_kwargs(
    pad_packed_seq_alignment: Union[int, Literal["max"]],
    max_seqlen_per_dp_cp_rank: Optional[int],
    thd_max_packed_sequences: Optional[int],
    cuda_graph_static: bool,
    cp_size: int = 1,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Resolve the typed arguments for :func:`pad_sequence_for_thd`.

    ``max_seqlen_per_dp_cp_rank`` is a per-rank bound. The external
    multimodal loader calls this helper before context-parallel partitioning,
    so its input tensors still use global token coordinates. Resolve the
    global capacity here instead of making every caller duplicate the
    ``local_capacity * cp_size`` rule.
    """
    if cp_size <= 0:
        raise ValueError(f"Context-parallel size must be positive, got {cp_size}.")
    if max_seqlen_per_dp_cp_rank is not None:
        token_capacity = int(max_seqlen_per_dp_cp_rank) * int(cp_size)
    else:
        token_capacity = None

    if cuda_graph_static:
        if token_capacity is None:
            raise ValueError(
                "--max-seqlen-per-dp-cp-rank is required for static THD CUDA Graph padding."
            )
        if pad_packed_seq_alignment != "max":
            # The command-line option is documented and validated as a
            # *per-rank* value.  The external multimodal loader invokes this
            # helper before CP slicing, however, so its tensors use global
            # coordinates.  Treat either spelling as the same fixed graph
            # surface instead of rejecting the valid ``--pad-packed-seq-
            # alignment=<local capacity>`` form for CP > 1.
            numeric_alignment = int(pad_packed_seq_alignment)
            if numeric_alignment not in (int(max_seqlen_per_dp_cp_rank), token_capacity):
                raise ValueError(
                    "Static THD CUDA Graph padding requires a fixed target equal to "
                    "the per-rank --max-seqlen-per-dp-cp-rank or its global CP capacity "
                    "(local capacity * CP size); use --pad-packed-seq-alignment=max or "
                    "one of those numeric values."
                )
        return None, token_capacity, thd_max_packed_sequences

    if pad_packed_seq_alignment == "max":
        if token_capacity is None:
            raise ValueError(
                "--max-seqlen-per-dp-cp-rank is required when " "--pad-packed-seq-alignment=max."
            )
        return None, token_capacity, thd_max_packed_sequences

    return int(pad_packed_seq_alignment), None, None


def resolve_thd_tail_padding_policy(config: object) -> Literal["append_dummy_seq", "extend_last"]:
    """Return the configured THD tail policy with the eager-compatible default."""
    policy = getattr(config, "thd_tail_padding_policy", None) or "append_dummy_seq"
    if policy not in ("append_dummy_seq", "extend_last"):
        raise ValueError(f"Unsupported thd_tail_padding_policy: {policy!r}.")
    return policy


def pad_sequence_for_thd(
    tokens: Optional[Tensor],
    labels: Optional[Tensor],
    loss_mask: Optional[Tensor],
    position_ids: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    alignment: Optional[int] = None,
    target_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    tail_padding_policy: Literal["append_dummy_seq", "extend_last"] = "append_dummy_seq",
    padding_mask: Optional[Tensor] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
) -> Tuple[
    Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], PackedSeqParams, Tensor
]:
    """Pad packed THD tensors and metadata to declared static capacities.

    Compact ``cu_seqlens`` remain valid-token coordinates. With ``extend_last``, only
    the final padded boundary grows to cover the fixed token capacity. Exceeding either
    capacity raises with the observed and configured values; inputs are never truncated.
    """
    if (alignment is None) == (target_len is None):
        raise ValueError("Exactly one of alignment or target_len must be provided for THD padding.")
    if tail_padding_policy not in ("append_dummy_seq", "extend_last"):
        raise ValueError(f"Unsupported THD tail padding policy: {tail_padding_policy!r}.")

    resolved_cp_size = _resolve_thd_cp_size(packed_seq_params, cp_group, cp_size)
    if resolved_cp_size != 1:
        raise ValueError(
            "Static THD padding with context parallelism must be applied before CP slicing."
        )

    token_like_tensors = (tokens, labels, loss_mask, position_ids)
    observed_lengths = [
        int(tensor.shape[-1]) for tensor in token_like_tensors if tensor is not None
    ]
    for cu_seqlens in (
        packed_seq_params.cu_seqlens_q_padded,
        packed_seq_params.cu_seqlens_kv_padded,
        packed_seq_params.cu_seqlens_q,
        packed_seq_params.cu_seqlens_kv,
    ):
        if cu_seqlens is not None:
            observed_lengths.append(int(cu_seqlens[-1].item()))
    if not observed_lengths:
        raise ValueError("Packed THD padding requires a token-like tensor or cu_seqlens metadata.")

    observed_tokens = max(observed_lengths)
    configured_tokens = (
        int(target_len)
        if target_len is not None
        else _round_up_to_alignment(observed_tokens, int(alignment))
    )
    if observed_tokens > configured_tokens:
        raise ValueError(
            f"Packed THD observed token count {observed_tokens} exceeds configured bound "
            f"{configured_tokens}; refusing to truncate."
        )

    cu_seqlens_q = packed_seq_params.cu_seqlens_q
    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
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
    physical_end = max(
        int(cu[-1].item()) for cu in (cu_seqlens_q_padded, cu_seqlens_kv_padded) if cu is not None
    )
    has_tail = physical_end < configured_tokens
    physical_tail = configured_tokens - physical_end

    if max_num_seqs is not None:
        observed_real_sequences = max(
            int(cu.shape[0]) - 1 for cu in (cu_seqlens_q, cu_seqlens_kv) if cu is not None
        )
        if observed_real_sequences > max_num_seqs:
            raise ValueError(
                f"Packed THD observed sequence count {observed_real_sequences} exceeds configured "
                f"bound {max_num_seqs}; refusing to truncate."
            )

    if tail_padding_policy == "extend_last":
        cu_seqlens_q_padded = _extend_last_boundary(cu_seqlens_q_padded, configured_tokens)
        cu_seqlens_kv_padded = _extend_last_boundary(cu_seqlens_kv_padded, configured_tokens)
    elif has_tail:
        if cu_seqlens_q is not None:
            cu_seqlens_q = _append_boundary(
                cu_seqlens_q, int(cu_seqlens_q[-1].item()) + physical_tail
            )
        if cu_seqlens_kv is not None:
            cu_seqlens_kv = _append_boundary(
                cu_seqlens_kv, int(cu_seqlens_kv[-1].item()) + physical_tail
            )
        cu_seqlens_q_padded = _append_boundary(cu_seqlens_q_padded, configured_tokens)
        cu_seqlens_kv_padded = _append_boundary(cu_seqlens_kv_padded, configured_tokens)

    if max_num_seqs is not None:
        target_entries = max_num_seqs + 1 + (1 if tail_padding_policy == "append_dummy_seq" else 0)
        cu_seqlens_q = _pad_cu_seqlens(cu_seqlens_q, target_entries)
        cu_seqlens_kv = _pad_cu_seqlens(cu_seqlens_kv, target_entries)
        cu_seqlens_q_padded = _pad_cu_seqlens(cu_seqlens_q_padded, target_entries)
        cu_seqlens_kv_padded = _pad_cu_seqlens(cu_seqlens_kv_padded, target_entries)

    padded_tokens = _pad_seq_tensor(tokens, configured_tokens)
    padded_labels = _pad_seq_tensor(labels, configured_tokens)
    padded_loss_mask = _pad_seq_tensor(loss_mask, configured_tokens)
    padded_position_ids = _pad_seq_tensor(position_ids, configured_tokens)

    if padding_mask is None:
        mask_device = next(
            (
                tensor.device
                for tensor in (*token_like_tensors, cu_seqlens_q, cu_seqlens_q_padded)
                if tensor is not None
            ),
            torch.device("cpu"),
        )
        mask_actual = int(tokens.shape[-1]) if tokens is not None else physical_end
        padding_mask = torch.zeros((1, mask_actual), dtype=torch.bool, device=mask_device)
    padded_padding_mask = _pad_padding_mask(padding_mask, configured_tokens)
    actual_tokens = int(tokens.shape[-1]) if tokens is not None else physical_end
    tail_mask_shape = [1] * (padded_padding_mask.ndim - 1) + [configured_tokens]
    explicit_tail_mask = (
        torch.arange(configured_tokens, device=padded_padding_mask.device) >= actual_tokens
    ).reshape(tail_mask_shape)
    padded_padding_mask = padded_padding_mask | explicit_tail_mask

    max_seqlen_q = (
        configured_tokens
        if max_num_seqs is not None
        else _updated_max_seqlen(cu_seqlens_q_padded, packed_seq_params.max_seqlen_q)
    )
    max_seqlen_kv = (
        configured_tokens
        if max_num_seqs is not None
        else _updated_max_seqlen(cu_seqlens_kv_padded, packed_seq_params.max_seqlen_kv)
    )

    padded_params = replace(
        packed_seq_params,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        total_tokens=configured_tokens,
        seq_idx=None,
        pad_between_seqs=(
            True
            if max_num_seqs is not None or tail_padding_policy == "extend_last"
            else packed_seq_params.pad_between_seqs
        ),
    )
    return (
        padded_tokens,
        padded_labels,
        padded_loss_mask,
        padded_position_ids,
        padded_params,
        padded_padding_mask,
    )
