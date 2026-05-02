# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


CUDA_GRAPH_MAX_PACKED_SEQS: int = 2048
_CG_SHARED_BUFFERS: dict = {}


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
    max_seqlen_q_tensor: Tensor = None
    max_seqlen_kv_tensor: Tensor = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    total_tokens: int = None
    seq_idx: Tensor = None

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

    @staticmethod
    def pad_cu_seqlens(cu_seqlens: Tensor, target_len: int) -> Tensor:
        """Pad cu_seqlens to a fixed length for CUDA graph replay."""
        actual_len = cu_seqlens.shape[0]
        if actual_len >= target_len:
            return cu_seqlens[:target_len]
        padded = cu_seqlens.new_empty(target_len)
        padded[:actual_len] = cu_seqlens
        padded[actual_len:] = cu_seqlens[-1]
        return padded

    def ensure_cg_padded(self, target_len: int) -> None:
        """Cache padded cu_seqlens tensors for graph replay."""
        if getattr(self, "_cg_pad_target", None) == target_len:
            return
        self._cg_pad_target = target_len
        self._cg_padded_q = PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_q, target_len)
        self._cg_padded_kv = PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_kv, target_len)
        self._cg_padded_qp = (
            PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_q_padded, target_len)
            if self.cu_seqlens_q_padded is not None
            else None
        )
        self._cg_padded_kvp = (
            PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_kv_padded, target_len)
            if self.cu_seqlens_kv_padded is not None
            else None
        )

    @classmethod
    def get_or_create_shared_cg_buffers(
        cls,
        seq_length: int,
        max_seqs: int,
        device: torch.device,
        *,
        tag: str = "attn",
    ) -> Dict[str, Tensor]:
        """Return shared buffer tensors used by all layers in the same graph bucket."""
        key = (tag, seq_length, max_seqs, int(device.index or 0))
        if key not in _CG_SHARED_BUFFERS:
            _, buffers = cls.create_dummy_for_cuda_graph(seq_length, max_seqs=max_seqs)
            buffers["_last_updated_psp"] = None
            _CG_SHARED_BUFFERS[key] = buffers
        return _CG_SHARED_BUFFERS[key]

    @classmethod
    def create_dummy_for_cuda_graph(
        cls, seq_length: int, max_seqs: int = CUDA_GRAPH_MAX_PACKED_SEQS
    ) -> Tuple["PackedSeqParams", Dict[str, Tensor]]:
        """Create a dummy PackedSeqParams and shared tensor buffers for graph capture."""
        cu_seqlens_len = max_seqs + 1
        device = torch.cuda.current_device()
        dtype = torch.int32

        cu_seqlens_q = torch.zeros(cu_seqlens_len, dtype=dtype, device=device)
        cu_seqlens_q[1:] = seq_length
        cu_seqlens_kv = cu_seqlens_q.clone()
        cu_seqlens_q_padded = cu_seqlens_q.clone()
        cu_seqlens_kv_padded = cu_seqlens_q.clone()
        max_seqlen_q_tensor = torch.tensor([seq_length], dtype=dtype, device=device)
        max_seqlen_kv_tensor = torch.tensor([seq_length], dtype=dtype, device=device)

        psp = cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=seq_length,
            max_seqlen_kv=seq_length,
            max_seqlen_q_tensor=max_seqlen_q_tensor,
            max_seqlen_kv_tensor=max_seqlen_kv_tensor,
        )
        buffers = {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "cu_seqlens_q_padded": cu_seqlens_q_padded,
            "cu_seqlens_kv_padded": cu_seqlens_kv_padded,
            "max_seqlen_q_tensor": max_seqlen_q_tensor,
            "max_seqlen_kv_tensor": max_seqlen_kv_tensor,
        }
        return psp, buffers
