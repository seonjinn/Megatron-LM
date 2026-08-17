# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.router_replay import (
    ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG,
    RouterReplay,
    RouterReplayAction,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


@pytest.fixture
def layer() -> TransformerLayer:
    result = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(result)
    result.config = SimpleNamespace(
        context_parallel_size=1,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe_router],
        delay_offload_until_cuda_graph=False,
        hidden_size=8,
        moe_enable_routing_replay=True,
        moe_router_fusion=False,
        moe_router_topk=2,
        num_moe_experts=4,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        thd_max_packed_sequences=None,
    )
    result.is_moe_layer = True
    result.self_attention = IdentityOp()
    result.mlp = SimpleNamespace(router=SimpleNamespace(router_replay=RouterReplay()))
    result.cuda_graph_manual_hooks = []
    result.cuda_graphs = []
    result.offload_module_in_cuda_graph = False
    return result


def test_router_replay_graph_rejects_missing_input_before_manual_hooks(
    layer: TransformerLayer, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"hook": 0, "graph": 0}

    def replay(_self, *args, **kwargs):
        for hook, hook_args in layer.cuda_graph_manual_hooks:
            hook(*hook_args)
        return layer.cuda_graphs[0](*args, **kwargs)

    layer.cuda_graph_manual_hooks = [
        (lambda: calls.__setitem__("hook", calls["hook"] + 1), ())
    ]
    layer.cuda_graphs = [
        lambda *args, **kwargs: calls.__setitem__("graph", calls["graph"] + 1)
    ]
    monkeypatch.setattr(GraphableMegatronModule, "_te_cuda_graph_replay", replay)

    with pytest.raises(RuntimeError, match="router replay CUDA graph input"):
        layer._te_cuda_graph_replay(hidden_states=torch.randn(4, 1, 8))

    assert calls == {"hook": 0, "graph": 0}


def test_router_replay_graph_values_change_without_signature_change(
    layer: TransformerLayer,
) -> None:
    first = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    second = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)

    first_signature = layer._validate_te_cuda_graph_router_replay_input(first)
    second_signature = layer._validate_te_cuda_graph_router_replay_input(second)

    assert first_signature == second_signature


def test_router_replay_graph_inserts_current_layer_route_kwarg(
    layer: TransformerLayer,
) -> None:
    hidden_states = torch.randn(2, 1, 8)
    route_input = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    observed = {}
    layer.mlp.router.router_replay.target_topk_idx = route_input
    layer._forward_attention = lambda *args, **kwargs: (args[0], None)

    def replay_impl(args, kwargs, context, *, eager_packed_seq_params=None):
        observed.update(kwargs)
        return args[0]

    layer._te_cuda_graph_replay_impl = replay_impl

    layer._te_cuda_graph_replay(hidden_states)

    assert observed[ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG] is route_input


def test_router_replay_static_input_uses_canonical_fixed_capacity_rows(
    layer: TransformerLayer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    static_inputs = layer.get_layer_static_inputs(seq_length=3, micro_batch_size=2)

    assert torch.equal(
        static_inputs[ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG],
        torch.tensor([[0, 1]] * 6, dtype=torch.long),
    )
    assert static_inputs[ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG].is_contiguous()


def test_router_replay_capture_uses_only_graph_kwarg_for_router(
    layer: TransformerLayer,
) -> None:
    route_input = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    seen = {}

    def forward_mlp(hidden_states, *, padding_mask, packed_seq_params):
        replay = layer.mlp.router.router_replay
        seen["target"] = replay.target_topk_idx
        seen["action"] = replay.router_replay_action
        return hidden_states

    layer._forward_mlp = forward_mlp

    output = layer._te_cuda_graph_capture(
        torch.randn(2, 1, 8), **{ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG: route_input}
    )

    assert output[0].shape == (2, 1, 8)
    assert seen == {"target": route_input, "action": RouterReplayAction.REPLAY_FORWARD}
    assert layer.mlp.router.router_replay.target_topk_idx is None
    assert layer.mlp.router.router_replay.router_replay_action is None


@pytest.mark.parametrize(
    "cuda_graph_modules",
    [[], [CudaGraphModule.moe], [CudaGraphModule.moe_preprocess]],
)
def test_router_replay_cuda_graph_rejects_unsupported_router_scope(
    layer: TransformerLayer, cuda_graph_modules: list[CudaGraphModule]
) -> None:
    layer.config.cuda_graph_modules = cuda_graph_modules

    with pytest.raises(ValueError, match="router replay.*CUDA graph scope"):
        layer._validate_te_cuda_graph_router_replay_scope()


def test_router_replay_cuda_graph_rejects_fused_router(layer: TransformerLayer) -> None:
    layer.config.moe_router_fusion = True

    with pytest.raises(ValueError, match="moe_router_fusion"):
        layer._validate_te_cuda_graph_router_replay_scope()
