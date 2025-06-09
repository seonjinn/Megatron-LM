# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Multimodal Sequence Parallel (SP) and Context Parallel (CP) functionality."""

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
)


def get_padding(
    seq_len, cp_size, tp_size, has_sp, decoder_tp_comm_overlap=False, decoder_seq_len=None, fp8_enabled=False
):
    """Calculate padding needed for SP, CP, TP comm overlap, and FP8.

    Args:
        seq_len (int): Model sequence length.
        cp_size (int): Context parallel size.
        tp_size (int): Tensor parallel size.
        has_sp (bool): Model uses sequence parallelism.
        decoder_tp_comm_overlap (bool): Decoder (LLM) uses tensor parallel communication overlap.
        decoder_seq_len (int): Decoder (LLM) maximum sequence length.
        fp8_enabled (bool): FP8 is enabled.

    Returns:
        padding (int): Padding needed given model configuration.
    """

    padding = 0
    # TP Comm overlap is performed with combined text+image embeddings.
    if has_sp and decoder_tp_comm_overlap:
        # If TP Comm Overlap is enabled for combined text+image embedding in LM backbone,
        # user needs to provide decoder_seq_len with any potential padding needed for SP+CP
        assert (
            decoder_seq_len is not None
        ), "Please provide decoder seq length when using TP comm overlap for LM backbone"
        padding = decoder_seq_len - seq_len
        return padding

    padding_factor = 1
    if has_sp and cp_size > 1:
        # Padding to multiple of tp_size * cp_size * 2 when using CP + SP.
        padding_factor = max(tp_size * cp_size, cp_size * 2)
    elif cp_size > 1:
        padding_factor = cp_size * 2
    elif has_sp:
        padding_factor = tp_size

    if fp8_enabled:
        # FP8 must be padded to multiple of 16.
        fp8_factor = 16 * padding_factor
        padding_factor = (padding_factor + fp8_factor - 1) // fp8_factor * fp8_factor

    padding = int((seq_len + padding_factor - 1) // padding_factor * padding_factor) - seq_len

    return padding


def get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, use_packed_sequence=False):
    """Get PackedSeqParams for CP.

    Args:
        tokens (torch.Tensor): [batch, seq_len] input tokens.
        img_seq_len (int): Image sequence length.
        padding_needed (int): Padding to add.
        cp_size (int): Context parallel size.
        use_packed_sequence (bool): Uses sequence packing.

    Returns:
        packed_seq_params (PackedSeqParams): Parameters to be sent to Transformer Engine.
    """
    batch_size = tokens.shape[0]
    # Calculate the valid token seq len that LM backbone should compute on
    combined_valid_seqlen = tokens.shape[1] + img_seq_len - padding_needed
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * (combined_valid_seqlen),
        step=(combined_valid_seqlen),
        dtype=torch.int32,
        device=tokens.device,
    )
    # Calculate the total padded token seq len
    combined_padded_seqlen = tokens.shape[1] + img_seq_len
    cu_seqlens_padded = None
    qkv_format = 'sbhd'
    if cp_size > 1 and (padding_needed > 0 or use_packed_sequence):
        # Provide cu_seqlens_<q/kv>_padded for CP support
        cu_seqlens_padded = torch.arange(
            0,
            (batch_size + 1) * (combined_padded_seqlen),
            step=(combined_padded_seqlen),
            dtype=torch.int32,
            device=tokens.device,
        )
        # CP with padding mask type requires THD format
        qkv_format = 'thd'

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=combined_padded_seqlen,
        max_seqlen_kv=combined_padded_seqlen,
        qkv_format=qkv_format,
    )

    return packed_seq_params


def split_to_context_parallel_ranks(global_t):
    """Split the tensor global_t into context parallel world size parts.

    Args:
        global_t: [batch, ...]

    Returns:
        local_t: [samples_per_rank, ...]. samples_per_rank is the # of samples per CP rank.
        global_pad: Total padding to have equal samples_per_rank across context parallel ranks.
    """
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()

    # t: [batch, ...]
    # Number of samples per context parallel rank, rounded up.
    samples_per_rank = (global_t.shape[0] + cp_size - 1) // cp_size

    # Total padding to have equal samples_per_rank across context parallel ranks.
    global_pad = samples_per_rank * cp_size - global_t.shape[0]

    # Get the local slice
    local_t = global_t[cp_rank * samples_per_rank : (cp_rank + 1) * samples_per_rank]
    # Pad the local slice to equal size if needed.
    if local_t.shape[0] < samples_per_rank:
        local_pad = samples_per_rank - local_t.shape[0]
        zeros = torch.zeros(
            local_pad, *local_t.shape[1:], device=local_t.device, dtype=local_t.dtype
        )
        local_t = torch.cat([local_t, zeros], dim=0)

    return local_t, global_pad


def _gather_along_second_dim(local_t):
    group = get_context_parallel_group()
    cp_size = get_context_parallel_world_size()
    # Bypass the function if we are using only 1 context parallel rank.
    if cp_size == 1:
        return local_t

    tensor_list = [
        torch.empty(
            local_t.shape,
            device=local_t.device,
            dtype=local_t.dtype,
        )
        for _ in range(cp_size)
    ]
    torch.distributed.all_gather(tensor_list, local_t, group=group)
    global_t = torch.cat(tensor_list, dim=1)

    return global_t


def _reduce_scatter_along_second_dim(global_t):
    cp_size = get_context_parallel_world_size()
    # Bypass the function if we are using only 1 CP rank.
    if cp_size == 1:
        return global_t

    assert global_t.shape[1] % cp_size == 0
    samples_per_rank = global_t.shape[1] // cp_size

    tensor_list = [global_t[:, cp_rank * samples_per_rank : (cp_rank + 1) * samples_per_rank] for cp_rank in range(cp_size)]

    local_t = torch.zeros(global_t.shape[0], samples_per_rank, *global_t.shape[2:], device=global_t.device, dtype=global_t.dtype)

    torch.distributed.reduce_scatter(local_t, tensor_list, group=get_context_parallel_group())

    return local_t


class GatherFromContextParallelRanks(torch.autograd.Function):
    """Gather the input from context parallel ranks."""
    @staticmethod
    def symbolic(
        graph,
        input_,
    ):
        """Symbolic function for tracing."""
        return _gather_along_second_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _gather_along_second_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce_scatter_along_second_dim(grad_output)


def gather_from_context_parallel_ranks(local_t, global_pad):
    global_t = GatherFromContextParallelRanks.apply(local_t)

    if global_pad > 0:
        global_t = global_t[:, :-global_pad]

    return global_t
