# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_layer import TransformerLayer


def _make_attention_packed_seq_params():
    cu_seqlens = torch.IntTensor([0, 4, 9, 16])
    cu_seqlens_padded = torch.IntTensor([0, 8, 12, 16])
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        local_cp_size=1,
    )


def test_te_cuda_graph_partial_attn_only_flow():
    from megatron.core.packed_seq_params import (
        CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
        split_packed_seq_params_for_cuda_graph,
    )
    from megatron.core.transformer.enums import CudaGraphModule

    class _ConfigStub:
        def __init__(self, cuda_graph_modules):
            self.cuda_graph_modules = cuda_graph_modules
            self.delay_offload_until_cuda_graph = False

    class _TestLayer:
        _set_te_cuda_graph_packed_seq_params_static_metadata = (
            TransformerLayer._set_te_cuda_graph_packed_seq_params_static_metadata
        )
        _flatten_te_cuda_graph_packed_seq_params = (
            TransformerLayer._flatten_te_cuda_graph_packed_seq_params
        )
        _te_cuda_graph_replay = TransformerLayer._te_cuda_graph_replay

        def __init__(self, cuda_graph_modules):
            self.config = _ConfigStub(cuda_graph_modules)
            self.attn_called = False
            self.replay_impl_called = False
            self.replay_impl_args = None
            self.replay_impl_kwargs = None
            self.replay_impl_context = None

        def _forward_attention(self, *args, **kwargs):
            self.attn_called = True
            return torch.ones(2, 1, 4) * 2.0, "attn_context"

        def _te_cuda_graph_replay_impl(self, args, kwargs, context):
            self.replay_impl_called = True
            self.replay_impl_args = args
            self.replay_impl_kwargs = kwargs
            self.replay_impl_context = context
            return torch.ones(2, 1, 4) * 3.0

    packed_seq_params = _make_attention_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)

    layer_attn = _TestLayer([CudaGraphModule.attn])
    layer_attn._set_te_cuda_graph_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs
    )
    layer_attn._te_cuda_graph_replay(
        packed_seq_params=packed_seq_params, hidden_states=torch.ones(2, 1, 4)
    )

    assert not layer_attn.attn_called
    assert layer_attn.replay_impl_called
    assert layer_attn.replay_impl_context is None
    assert "packed_seq_params" not in layer_attn.replay_impl_kwargs
    assert (
        f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q"
        in layer_attn.replay_impl_kwargs
    )

    layer_mlp = _TestLayer([CudaGraphModule.mlp])
    layer_mlp._te_cuda_graph_replay(
        packed_seq_params=packed_seq_params, hidden_states=torch.ones(2, 1, 4)
    )

    assert layer_mlp.attn_called
    assert layer_mlp.replay_impl_called
    assert layer_mlp.replay_impl_context == "attn_context"
    assert len(layer_mlp.replay_impl_args) == 1
    assert torch.equal(layer_mlp.replay_impl_args[0], torch.ones(2, 1, 4) * 2.0)
    assert layer_mlp.replay_impl_kwargs == {}


def _make_mamba_packed_seq_params(cu_q, cu_kv, total_tokens, seq_idx):
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor(cu_q),
        cu_seqlens_kv=torch.IntTensor(cu_kv),
        cu_seqlens_q_padded=torch.IntTensor(cu_q),
        cu_seqlens_kv_padded=torch.IntTensor(cu_kv),
        max_seqlen_q=8,
        max_seqlen_kv=8,
        local_cp_size=1,
        total_tokens=total_tokens,
    )
    params.seq_idx = torch.IntTensor([seq_idx])
    return params


@pytest.mark.parametrize(
    (
        "capture_cu_q",
        "capture_cu_kv",
        "capture_total_tokens",
        "capture_seq_idx",
        "replay_cu_q",
        "replay_cu_kv",
        "replay_total_tokens",
        "replay_seq_idx",
    ),
    [
        (
            [0, 2, 5],
            [0, 3, 5],
            5,
            [0, 0, 1, 1, 1, 2],
            [0, 1, 4, 6],
            [0, 2, 4, 6],
            6,
            [0, 1, 1, 2, 2, 2],
        ),
        (
            [0, 1, 4],
            [0, 2, 4],
            4,
            [0, 1, 1, 1, 2, 2],
            [0, 3, 6],
            [0, 1, 6],
            6,
            [0, 0, 0, 1, 1, 1],
        ),
    ],
)
def test_mamba_replay_updates_make_graphed_callables_packed_inputs_without_static_changes(
    capture_cu_q,
    capture_cu_kv,
    capture_total_tokens,
    capture_seq_idx,
    replay_cu_q,
    replay_cu_kv,
    replay_total_tokens,
    replay_seq_idx,
):
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.cuda_graphs import (
        _add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs,
    )

    class _MambaLayerCudaGraphStub:
        _set_te_cuda_graph_mamba_packed_seq_params_static_metadata = (
            MambaLayer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata
        )
        _flatten_te_cuda_graph_mamba_packed_seq_params = (
            MambaLayer._flatten_te_cuda_graph_mamba_packed_seq_params
        )

    capture_params = _make_mamba_packed_seq_params(
        capture_cu_q, capture_cu_kv, capture_total_tokens, capture_seq_idx
    )
    replay_params = _make_mamba_packed_seq_params(
        replay_cu_q, replay_cu_kv, replay_total_tokens, replay_seq_idx
    )
    assert not torch.equal(capture_params.cu_seqlens_q, replay_params.cu_seqlens_q)
    assert not torch.equal(capture_params.cu_seqlens_kv, replay_params.cu_seqlens_kv)
    assert capture_params.total_tokens != replay_params.total_tokens
    assert not torch.equal(capture_params.seq_idx, replay_params.seq_idx)

    layer = _MambaLayerCudaGraphStub()
    make_graphed_callables_sample_kwargs = {}
    _add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs(
        layer, make_graphed_callables_sample_kwargs, capture_params
    )
    captured_static_metadata = dict(
        layer._te_cuda_graph_mamba_packed_seq_params_static_metadata
    )

    replay_kwargs = {"packed_seq_params": replay_params}
    layer._flatten_te_cuda_graph_mamba_packed_seq_params(replay_kwargs)

    seq_idx_key = "_mamba_packed_seq_params_seq_idx"
    assert make_graphed_callables_sample_kwargs[seq_idx_key] is capture_params.seq_idx
    assert replay_kwargs == {seq_idx_key: replay_params.seq_idx}
    assert (
        layer._te_cuda_graph_mamba_packed_seq_params_static_metadata
        == captured_static_metadata
    )
    assert "total_tokens" not in captured_static_metadata
