# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import config
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import (
    MoEModelTestContainer,
    permute_fusion_params,
)


def test_placeholder():
    """This is here because otherwise there's no other test in this module (all disabled)
    and pytest would fail."""
    pass


def _make_replay_state_dispatcher() -> MoEAlltoAllTokenDispatcher:
    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_expert_capacity_factor=1.25,
        moe_expert_rank_capacity_factor=None,
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    dispatcher.tp_size = 2
    dispatcher.ep_size = 4
    dispatcher.router_topk = 2
    dispatcher.num_experts = 8
    dispatcher.num_local_experts = 2
    dispatcher.drop_and_pad = True
    dispatcher.hidden_shape = torch.Size((2, 3, 4))
    dispatcher.hidden_shape_before_permute = torch.Size((6, 4))
    dispatcher.capacity = 5
    dispatcher.num_out_tokens = torch.tensor(40)
    return dispatcher


def test_alltoall_cudagraph_replay_state_restores_structural_geometry() -> None:
    dispatcher = _make_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.float32)
    preprocessed = torch.empty((40, 4), dtype=torch.float32)

    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)
    dispatcher.hidden_shape = torch.Size((3, 2, 4))
    dispatcher.hidden_shape_before_permute = torch.Size((4, 6))
    dispatcher.capacity = 99
    dispatcher.num_out_tokens = 3

    dispatcher.restore_cudagraph_replay_state(state, graph_input, preprocessed)

    assert dispatcher.hidden_shape == torch.Size((2, 3, 4))
    assert dispatcher.hidden_shape_before_permute == torch.Size((6, 4))
    assert dispatcher.capacity == 5
    assert dispatcher.num_out_tokens == 40


@pytest.mark.parametrize(
    "changed_input",
    [
        torch.empty((3, 2, 4), dtype=torch.float32),
        torch.empty((2, 3, 4), dtype=torch.float64),
        torch.empty_strided((2, 3, 4), (1, 2, 6), dtype=torch.float32),
        torch.empty((2, 3, 4), device="meta", dtype=torch.float32),
        torch.empty((2, 3, 4), dtype=torch.float32).to_sparse(),
    ],
    ids=["shape", "dtype", "stride", "device", "layout"],
)
def test_alltoall_cudagraph_replay_state_rejects_exact_input_signature_changes(
    changed_input: torch.Tensor,
) -> None:
    dispatcher = _make_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.float32)
    preprocessed = torch.empty((40, 4), dtype=torch.float32)
    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)

    with pytest.raises(RuntimeError, match="input signature"):
        dispatcher.restore_cudagraph_replay_state(state, changed_input, preprocessed)


def test_alltoall_cudagraph_replay_state_rejects_flattened_shape_change() -> None:
    dispatcher = _make_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.float32)
    dispatcher.hidden_shape_before_permute = torch.Size((3, 8))

    with pytest.raises(RuntimeError, match="flattened input shape"):
        dispatcher.snapshot_cudagraph_replay_state(
            graph_input, torch.empty((40, 4), dtype=torch.float32)
        )


def test_alltoall_cudagraph_continuation_rejects_same_numel_wrong_shape() -> None:
    dispatcher = _make_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.float32)
    state = dispatcher.snapshot_cudagraph_replay_state(
        graph_input, torch.empty((40, 4), dtype=torch.float32)
    )

    with pytest.raises(RuntimeError, match="continuation signature"):
        dispatcher.validate_cudagraph_continuation(
            state, torch.empty((3, 2, 4), dtype=torch.float32)
        )


def test_alltoall_cudagraph_replay_state_rejects_topology_change() -> None:
    dispatcher = _make_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.float32)
    preprocessed = torch.empty((40, 4), dtype=torch.float32)
    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)
    dispatcher.router_topk = 4

    with pytest.raises(RuntimeError, match="topology fingerprint"):
        dispatcher.restore_cudagraph_replay_state(state, graph_input, preprocessed)


class TestAlltoAllDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("deterministic", [False, True])
    def test_forward_backward(self, tp_size, ep_size, permute_fusion, deterministic, monkeypatch):
        if deterministic:
            # We only need to exercise the deterministic branches in moe_utils.
            # Enabling global determinism (torch.use_deterministic_algorithms(True))
            # would require CUBLAS_WORKSPACE_CONFIG and can slow other tests.
            # Monkeypatching here is per-test scoped and avoids global side effects.
            monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
            # Deterministic branch is exercised on the unfused path
            if permute_fusion:
                pytest.skip("Deterministic path tested only for unfused (permute_fusion=False)")
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_capacity_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_capacity_padding_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.6,
            moe_pad_expert_input_to_capacity=True,
            moe_permute_fusion=permute_fusion,
        )
        container.dispatcher_drop_and_pad_test()

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("experimental_fusion", [True, False])
    def test_router_padding_for_fp8_forward_backward(
        self, tp_size, ep_size, permute_fusion, experimental_fusion
    ):
        if experimental_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=4,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False
