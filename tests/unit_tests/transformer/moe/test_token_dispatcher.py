# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config, parallel_state
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.capacity_tracker import (
    destroy_moe_capacity_tracker,
    get_moe_capacity_tracker,
)
from megatron.core.transformer.moe.fused_a2a import HYBRIDEP_TOKEN_ALIGNMENT, reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_capacity
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEFlexTokenDispatcher,
    _HybridEPManager,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def token_permutation(token_dispatcher, hidden_states, probs, indices):
    hidden_states, probs = token_dispatcher.dispatch_preprocess(hidden_states, indices, probs)
    hidden_states, probs = token_dispatcher.token_dispatch(hidden_states, probs)
    hidden_states, tokens_per_expert, permuted_probs = token_dispatcher.dispatch_postprocess(
        hidden_states, probs
    )
    return hidden_states, tokens_per_expert, permuted_probs


def token_unpermutation(token_dispatcher, hidden_states):
    hidden_states = token_dispatcher.combine_preprocess(hidden_states)
    hidden_states = token_dispatcher.token_combine(hidden_states)
    hidden_states = token_dispatcher.combine_postprocess(hidden_states)
    return hidden_states, None


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        test_dtype=torch.float32,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        self.test_dtype = test_dtype
        if moe_tp_size is None:
            moe_tp_size = tp_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=1,
            moe_router_dtype="fp32",
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_flex_dispatcher_backend=kwargs.get("moe_flex_dispatcher_backend", None),
            moe_expert_rank_capacity_factor=kwargs.get("moe_expert_rank_capacity_factor", None),
            moe_ncclep_zero_copy=kwargs.get("moe_ncclep_zero_copy", False),
            use_transformer_engine_op_fuser=kwargs.get("use_transformer_engine_op_fuser", False),
            gated_linear_unit=kwargs.get("gated_linear_unit", False),
            activation_func=kwargs.get("activation_func", F.gelu),
            fp8=kwargs.get("fp8", None),
            fp8_recipe=kwargs.get("fp8_recipe", "delayed"),
            calculate_per_token_loss=kwargs.get("calculate_per_token_loss", False),
        )

        # init moe layer
        self.moe_layer = self.new_moe_layer()

    def new_moe_layer(self, **kargs):
        new_config = dataclasses.replace(self.config, **kargs)
        if new_config.use_transformer_engine_op_fuser:
            # op-fuser needs the TE grouped-MLP experts (they accept output_buffer/grad_input_buffer
            # for the ncclEP zero-copy path); the local spec yields SequentialMLP, which does not.
            mlp_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=new_config.num_moe_experts, moe_grouped_gemm=new_config.moe_grouped_gemm
            ).submodules.mlp
        else:
            mlp_spec = get_gpt_layer_local_submodules(
                num_experts=self.config.num_moe_experts,
                moe_grouped_gemm=self.config.moe_grouped_gemm,
            ).mlp
        submodules = get_submodules(mlp_spec)
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(new_config, submodules).cuda().to(dtype=self.test_dtype)
        moe_layer.set_layer_number(0)
        return moe_layer

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_dropless_test(self):
        moe_layer = self.moe_layer
        bs = 32
        seql = 8
        # TODO: Find why setting manual seed can cause the test to fail
        # Manual seed to differentiate input data for each rank
        # rank = torch.distributed.get_rank()
        # torch.manual_seed(1000 + rank)
        hidden_states = torch.randn((bs, seql, moe_layer.config.hidden_size), dtype=self.test_dtype)
        hidden_states = hidden_states.cuda()
        # Permute and then unpermute data are supposed to restore original data
        ans = hidden_states
        hidden_states.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(hidden_states)
        probs = torch.ones_like(probs) / moe_layer.router.topk

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs, indices
        )

        permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)
        permuted_local_hidden_states = permuted_local_hidden_states.to(dtype=self.test_dtype)

        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_local_hidden_states
        )

        # reduce across TP rank equals to multiply data by a scale of ETP
        scale = moe_layer.config.expert_tensor_parallel_size
        restored_hidden_states = restored_hidden_states / scale

        torch.testing.assert_close(
            restored_hidden_states, ans
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        torch.testing.assert_close(
            hidden_states.grad, ans
        ), "Restored hidden states do not match original hidden states"

    @pytest.mark.internal
    def moe_layer_zero_copy_parity_test(self):
        """Full MoE-layer fwd+bwd with ncclEP zero-copy OFF then ON (identical weights), asserting
        parity. Runs the real op-fuser experts so fc2-out/fc1-dgrad are written straight into the
        symm combine/dispatch buffers (verified via is_symm_backed) -- the pure permute/unpermute
        harness cannot exercise this path."""
        from transformer_engine.pytorch.ep import is_symm_backed

        from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize
        from megatron.core.transformer.moe.token_dispatcher import _NCCLEPManager

        torch.manual_seed(42)
        x = torch.randn((32, 8, self.config.hidden_size), dtype=self.test_dtype).cuda()

        def run(layer):
            inp = x.clone().detach().requires_grad_(True)
            out, _ = layer(inp)  # full fwd: dispatch -> op-fuser experts -> combine
            out.sum().backward()  # bwd: dispatch-bwd reads the symm grad buffer
            return out.detach(), inp.grad.detach()

        def reset_ep():
            # zero_copy mode is fixed at ep_bootstrap (process-global); finalize + drop the shared
            # symm classvars so the next layer re-bootstraps in the other mode.
            nccl_ep_finalize()
            _NCCLEPManager._zc_fwd_token_buf = None
            _NCCLEPManager._zc_bwd_token_buf = None
            _NCCLEPManager._zc_recv_topk_weights_buf = None

        ref_layer = self.new_moe_layer(moe_ncclep_zero_copy=False)
        out_ref, grad_ref = run(ref_layer)

        reset_ep()
        zc_layer = self.new_moe_layer(moe_ncclep_zero_copy=True)
        zc_layer.load_state_dict(ref_layer.state_dict())  # identical weights
        out_zc, grad_zc = run(zc_layer)

        # the combine forward buffer must be an allocated, registered symm window (zero-copy engaged)
        fwd_buf = _NCCLEPManager._zc_fwd_token_buf
        assert fwd_buf is not None, "zero-copy forward symm buffer was not allocated"
        assert is_symm_backed(fwd_buf), "zero-copy forward buffer is not symm-mem-backed"
        reset_ep()

        assert not torch.isnan(out_zc).any() and not torch.isnan(grad_zc).any()
        torch.testing.assert_close(out_zc, out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(grad_zc, grad_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.internal
    def dispatcher_capacity_test(self):
        moe_layer = self.moe_layer
        num_tokens = 16
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        )
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(hidden_states)

        # Create the answer.
        prob_mask = probs != 0
        probs = torch.ones_like(probs) * prob_mask / moe_layer.router.topk
        local_probss = probs
        restored_hidden_states_answer = hidden_states * local_probss.sum(dim=1).unsqueeze(1)
        restored_hidden_states_answer = restored_hidden_states_answer.to(dtype=self.test_dtype)

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs, indices
        )

        # Check tokens per expert not exceed the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            <= capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert exceed the capacity"

        permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)

        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size
        permuted_local_hidden_states = permuted_local_hidden_states.to(dtype=self.test_dtype)

        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_local_hidden_states
        )
        torch.testing.assert_close(
            restored_hidden_states, restored_hidden_states_answer
        ), "Restored hidden states does not match"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        torch.testing.assert_close(
            hidden_states.grad, restored_hidden_states_answer
        ), "Gradient of hidden states should be same as hidden states"

    @pytest.mark.internal
    def dispatcher_drop_and_pad_test(self):
        """Test if the tokens are dropped and padded correctly.

        Since the probs of padded tokens are 0, the combined results for
        dispatching with or without padding should be the same.
        """
        moe_layer = self.new_moe_layer(moe_pad_expert_input_to_capacity=False)

        num_tokens = 16
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        ).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = apply_module(moe_layer.router)(hidden_states)
        permuted_input_1, tokens_per_expert, permuted_probs_1 = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs_1, indices_1
        )
        permuted_input_1 = permuted_input_1 * permuted_probs_1.unsqueeze(-1)
        permuted_input_1 = permuted_input_1.to(dtype=self.test_dtype)
        forward_answer, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_input_1
        )
        torch.autograd.backward(forward_answer, forward_answer)
        backward_answer = hidden_states.grad.clone()
        hidden_states.grad = None
        torch.cuda.synchronize()
        # End

        moe_layer_2 = self.new_moe_layer(moe_pad_expert_input_to_capacity=True)
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = apply_module(moe_layer_2.router)(hidden_states)
        permuted_input_2, tokens_per_expert, permuted_probs_2 = token_permutation(
            moe_layer_2.token_dispatcher, hidden_states, probs_2, indices_2
        )
        permuted_input_2 = permuted_input_2 * permuted_probs_2.unsqueeze(-1)
        permuted_input_2 = permuted_input_2.to(dtype=self.test_dtype)
        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer_2.token_dispatcher, permuted_input_2
        )

        # # Check tokens per expert equals to the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            == capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert should be the same as the capacity"
        torch.testing.assert_close(
            restored_hidden_states, forward_answer
        ), "Restored hidden states does not match"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        torch.testing.assert_close(
            hidden_states.grad, backward_answer
        ), "Gradient of hidden states should be same as hidden states"

    @pytest.mark.internal
    def dispatcher_router_padding_for_fp8_test(self):
        """Test if the routing map is padded correctly for FP8 training.

        The test runs the forward flow twice:
        1. First with moe_router_padding_for_quantization=False
        2. Then with moe_router_padding_for_quantization=True

        We verify that:
        1. The results are the same in both cases
        2. The number of tokens received by each expert is padded to a multiple of 16
        """
        # First run with moe_router_padding_for_quantization = False
        moe_layer = self.new_moe_layer(moe_router_padding_for_quantization=False)

        num_tokens = 32
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        ).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = apply_module(moe_layer.router)(hidden_states)
        permuted_input_1, tokens_per_expert_1, permuted_probs_1 = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs_1, indices_1
        )
        permuted_input_1 = permuted_input_1 * permuted_probs_1.unsqueeze(-1)
        permuted_input_1 = permuted_input_1.to(dtype=self.test_dtype)
        restored_hidden_states_1, _ = token_unpermutation(
            moe_layer.token_dispatcher, permuted_input_1
        )
        torch.autograd.backward(restored_hidden_states_1, restored_hidden_states_1)
        grad_1 = hidden_states.grad.clone()
        hidden_states.grad = None

        # Run with moe_router_padding_for_quantization = True
        moe_layer_2 = self.new_moe_layer(moe_router_padding_for_quantization=True, fp8="hybrid")
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = apply_module(moe_layer_2.router)(hidden_states)
        permuted_input_2, tokens_per_expert_2, permuted_probs_2 = token_permutation(
            moe_layer_2.token_dispatcher, hidden_states, probs_2, indices_2
        )
        assert (
            sum(tokens_per_expert_2) == permuted_input_2.shape[0]
        ), f"number of tokens is not the same, {sum(tokens_per_expert_2)} != {permuted_input_2.shape[0]}"
        # when there is only one expert, the tokens is not enough for router padding
        if moe_layer_2.num_local_experts > 1:
            assert torch.all(
                tokens_per_expert_2 % 16 == 0
            ), "number of tokens for expert is not a multiple of 16"

        permuted_input_2 = permuted_input_2 * permuted_probs_2.unsqueeze(-1)
        permuted_input_2 = permuted_input_2.to(dtype=self.test_dtype)
        restored_hidden_states_2, _ = token_unpermutation(
            moe_layer_2.token_dispatcher, permuted_input_2
        )

        # Check that the results are the same
        torch.testing.assert_close(
            restored_hidden_states_1, restored_hidden_states_2
        ), "Restored hidden states do not match between padded and non-padded versions"

        # Check gradients
        torch.autograd.backward(restored_hidden_states_2, restored_hidden_states_2)
        torch.testing.assert_close(
            grad_1, hidden_states.grad
        ), "Gradients do not match between padded and non-padded versions"

    def set_params(self):
        # TODO: Set consistent parameters for various parallelisms.
        raise NotImplementedError

    def destroy(self):
        Utils.destroy_model_parallel()


permute_fusion_params = [False]
if is_te_min_version("2.1.0"):
    permute_fusion_params.append(True)


class TestAllgatherDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(8, 1), (1, 8), (2, 4), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_permute_fusion=permute_fusion,
        )

        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize(
        "tp_size,ep_size,moe_tp_size", [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 2, 4)]
    )
    def test_moe_tp_forward_backward(self, tp_size, ep_size, moe_tp_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            moe_tp_size=moe_tp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            sequence_parallel=True,
            moe_permute_fusion=permute_fusion,
            use_cpu_initialization=False,
        )

        container.dispatcher_dropless_test()


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def is_nccl_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    return HAVE_TE_EP


def is_nccl_ep_zero_copy_available():
    """Zero-copy needs the newer TE symm-mem APIs (symm_mem_alloc/is_symm_backed), which a plain
    NCCL-EP build lacks -- gate zero-copy tests on these separately from is_nccl_ep_available()."""
    if not is_nccl_ep_available():
        return False
    try:
        from transformer_engine.pytorch.ep import is_symm_backed, symm_mem_alloc  # noqa: F401
    except ImportError:
        return False
    return True


def is_op_fuser_available():
    """The static-shape/zero-copy path runs the TE op-fuser grouped GEMM (needs TE>=2.14 ops)."""
    try:
        from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
    except ImportError:
        return False
    return is_te_min_version("2.14.0")


@pytest.fixture
def capacity_tracker():
    destroy_moe_capacity_tracker()
    tracker = get_moe_capacity_tracker()
    tracker.initialize(torch.device("cpu"))
    yield tracker
    destroy_moe_capacity_tracker()


def _make_hybridep_capacity_manager(rank_capacity_factor: float | None) -> _HybridEPManager:
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.group = SimpleNamespace(size=lambda: 1)
    manager.num_local_experts = 2
    manager.num_experts = 2
    manager.config = SimpleNamespace(
        fp8=False,
        fp4=False,
        moe_flex_dispatcher_num_sms=8,
        moe_hybridep_num_blocks_permute=4,
        moe_hybridep_num_blocks_unpermute=4,
        moe_permute_fusion_into_hybridep=False,
        moe_hybridep_num_sms_preprocessing=4,
    )
    manager.moe_expert_rank_capacity_factor = rank_capacity_factor
    manager.drop_and_pad = False
    manager.num_permuted_tokens = 4 if rank_capacity_factor is not None else None
    manager.token_probs = torch.ones((2, 2), dtype=torch.float32)
    manager.routing_map = torch.ones((2, 2), dtype=torch.bool)
    manager.handle = None
    manager.pad_multiple = None
    manager._padded_num_tokens = 2
    manager.over_budget = torch.zeros(1, dtype=torch.bool)
    return manager


def test_hybridep_capacity_tracker_records_static_rank_overflow(
    monkeypatch, capacity_tracker
) -> None:
    manager = _make_hybridep_capacity_manager(rank_capacity_factor=1.0)
    overflow = torch.tensor(1, dtype=torch.int64)

    def fake_hybrid_ep_dispatch(**kwargs):
        return kwargs["x"], kwargs["probs"].flatten(), None, torch.tensor([2, 2]), [overflow]

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_dispatch",
        fake_hybrid_ep_dispatch,
    )

    manager.dispatch(torch.ones((2, 4)))

    snapshot = capacity_tracker.snapshot()
    assert snapshot.selected_assignments == 0
    assert snapshot.dropped_assignments == 0
    assert snapshot.valid_token_drops == 0
    assert snapshot.rank_overflow_events == 1


def test_hybridep_dropless_capacity_tracker_preserves_dynamic_allocation(
    monkeypatch, capacity_tracker
) -> None:
    manager = _make_hybridep_capacity_manager(rank_capacity_factor=None)
    opaque_handle = object()
    observed_bound = None

    def fake_hybrid_ep_dispatch(**kwargs):
        nonlocal observed_bound
        observed_bound = kwargs["num_permuted_tokens"]
        return kwargs["x"], kwargs["probs"].flatten(), None, torch.tensor([1, 3]), opaque_handle

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_dispatch",
        fake_hybrid_ep_dispatch,
    )

    manager.dispatch(torch.ones((2, 4)))

    assert observed_bound is None
    assert manager.handle is opaque_handle
    assert manager.num_permuted_tokens == 4
    snapshot = capacity_tracker.snapshot()
    assert snapshot.selected_assignments == 0
    assert snapshot.dropped_assignments == 0
    assert snapshot.valid_token_drops == 0
    assert snapshot.rank_overflow_events == 0


def test_hybridep_manager_releases_completed_forward_autograd_references(
    monkeypatch, capacity_tracker
) -> None:
    manager = _make_hybridep_capacity_manager(rank_capacity_factor=None)
    manager._original_num_tokens = 2
    source_probs = torch.arange(4, dtype=torch.float32, requires_grad=True)
    manager.token_probs = (source_probs + 1).reshape(2, 2)
    expected_token_probs = manager.token_probs.detach().clone()
    hidden_states = torch.ones((2, 4), dtype=torch.float32, requires_grad=True)

    def fake_hybrid_ep_dispatch(**kwargs):
        dispatched_hidden = kwargs["x"] * 2
        dispatched_probs = kwargs["probs"].flatten() * 3
        return dispatched_hidden, dispatched_probs, None, torch.tensor([1, 3]), object()

    def fake_hybrid_ep_combine(**kwargs):
        return kwargs["x"] * 5

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_dispatch",
        fake_hybrid_ep_dispatch,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine",
        fake_hybrid_ep_combine,
    )

    dispatched_hidden = manager.dispatch(hidden_states)
    downstream_probs = manager.dispatched_probs

    torch.testing.assert_close(manager.token_probs, expected_token_probs)
    assert not manager.token_probs.requires_grad
    assert manager.token_probs.grad_fn is None
    assert downstream_probs.requires_grad

    combined_hidden = manager.combine(dispatched_hidden)

    torch.testing.assert_close(manager.dispatched_probs, downstream_probs.detach())
    assert not manager.dispatched_probs.requires_grad
    assert manager.dispatched_probs.grad_fn is None

    (combined_hidden.sum() + downstream_probs.sum()).backward()
    torch.testing.assert_close(hidden_states.grad, torch.full_like(hidden_states, 10))
    torch.testing.assert_close(source_probs.grad, torch.full_like(source_probs, 3))


def test_hybridep_pad_uneven_dispatch_inputs_metadata(monkeypatch):
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.group = object()
    manager.num_local_experts = 2
    manager.num_experts = 4
    manager.config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_hybridep_pad_uneven_dispatch_inputs=True,
    )
    manager.moe_expert_rank_capacity_factor = None
    manager.drop_and_pad = False

    local_num_tokens = 17
    max_num_tokens_across_ep = 70
    padded_num_tokens = (
        max_num_tokens_across_ep + -max_num_tokens_across_ep % HYBRIDEP_TOKEN_ALIGNMENT
    )
    routing_map = torch.ones((local_num_tokens, manager.num_experts), dtype=torch.bool)
    probs = torch.ones((local_num_tokens, manager.num_experts), dtype=torch.float32)

    def fake_all_reduce(tensor, op=None, group=None):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is manager.group
        tensor.fill_(max_num_tokens_across_ep)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    manager.setup_metadata(routing_map, probs)

    assert manager._original_num_tokens == local_num_tokens
    assert manager._padded_num_tokens == padded_num_tokens
    assert manager.routing_map.shape == (padded_num_tokens, manager.num_experts)
    assert manager.token_probs.shape == (padded_num_tokens, manager.num_experts)
    torch.testing.assert_close(manager.routing_map[:local_num_tokens], routing_map)
    torch.testing.assert_close(manager.token_probs[:local_num_tokens], probs)
    assert not manager.routing_map[local_num_tokens:].any()
    assert not manager.token_probs[local_num_tokens:].any()


def _make_hybridep_replay_state_dispatcher(
    *, drop_and_pad: bool = True, rank_capacity: float | None = None
) -> MoEFlexTokenDispatcher:
    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_capacity_factor=1.0 if drop_and_pad else None,
        moe_expert_rank_capacity_factor=rank_capacity,
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    dispatcher.tp_size = 2
    dispatcher.ep_size = 4
    dispatcher.num_local_experts = 2
    dispatcher.hidden_shape = torch.Size((2, 3, 4))

    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.drop_and_pad = drop_and_pad
    manager._original_num_tokens = 6
    manager._padded_num_tokens = 6
    manager.capacity = 5 if drop_and_pad else None
    manager.num_permuted_tokens = 40
    manager.tokens_per_expert = torch.tensor([20, 20], dtype=torch.long)
    manager.handle = object()
    manager.pad_multiple = 16
    dispatcher._comm_manager = manager
    return dispatcher


def test_hybridep_cudagraph_replay_state_restores_fixed_capacity_metadata() -> None:
    dispatcher = _make_hybridep_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    preprocessed = torch.empty((6, 4), dtype=torch.bfloat16)

    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)
    manager = dispatcher._comm_manager
    manager._original_num_tokens = 3
    manager._padded_num_tokens = 4
    manager.capacity = 99
    manager.num_permuted_tokens = 7
    manager.tokens_per_expert = torch.tensor([3, 4])

    dispatcher.restore_cudagraph_replay_state(state, graph_input, preprocessed)

    assert manager._original_num_tokens == 6
    assert manager._padded_num_tokens == 6
    assert manager.capacity == 5
    assert manager.num_permuted_tokens == 40
    assert tuple(manager.tokens_per_expert.tolist()) == (20, 20)
    assert manager.tokens_per_expert.device.type == "cpu"
    assert state.backend_state.tokens_per_expert == (20, 20)
    assert manager.handle is not None
    assert manager.pad_multiple == 16


def test_hybridep_cudagraph_replay_state_restores_fixed_rank_capacity_metadata() -> None:
    dispatcher = _make_hybridep_replay_state_dispatcher(drop_and_pad=False, rank_capacity=1.0)
    graph_input = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    preprocessed = torch.empty((6, 4), dtype=torch.bfloat16)

    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)
    dispatcher._comm_manager.num_permuted_tokens = 7

    dispatcher.restore_cudagraph_replay_state(state, graph_input, preprocessed)

    assert dispatcher._comm_manager.num_permuted_tokens == 40
    assert state.backend_state.tokens_per_expert is None


def test_hybridep_cudagraph_dropless_replay_preserves_eager_dispatch_ownership(
    monkeypatch,
) -> None:
    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_capacity_factor=None,
        moe_expert_rank_capacity_factor=None,
        moe_hybridep_pad_uneven_dispatch_inputs=False,
        moe_router_topk=2,
    )
    dispatcher.tp_size = 1
    dispatcher.ep_size = 1
    dispatcher.num_local_experts = 2
    dispatcher.num_experts = 2
    dispatcher.shared_experts = None

    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.group = SimpleNamespace(size=lambda: 1)
    manager.num_local_experts = 2
    manager.num_experts = 2
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False,
        moe_router_topk=2,
        fp8=False,
        fp4=False,
        moe_flex_dispatcher_num_sms=20,
        moe_hybridep_num_blocks_permute=4,
        moe_hybridep_num_blocks_unpermute=4,
        moe_permute_fusion_into_hybridep=False,
        moe_hybridep_num_sms_preprocessing=4,
    )
    manager.moe_expert_rank_capacity_factor = None
    manager.drop_and_pad = False
    manager.capacity = None
    manager.num_permuted_tokens = None
    manager.handle = None
    manager.pad_multiple = None
    dispatcher._comm_manager = manager

    graph_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    capture_routing_map = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, True],
            [True, False],
            [False, True],
            [True, True],
        ]
    )
    replay_cases = [
        (
            torch.tensor(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, True],
                ]
            ),
            ((0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)),
            (3, 3),
        ),
        (
            capture_routing_map,
            (
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 1),
                (3, 0),
                (4, 1),
                (5, 0),
                (5, 1),
            ),
            (4, 4),
        ),
    ]
    observed_dispatches = []
    observed_combines = []

    def fake_hybrid_ep_dispatch(**kwargs):
        selected_route_tensor = kwargs["routing_map"].nonzero(as_tuple=False)
        selected_routes = tuple(tuple(int(value) for value in row) for row in selected_route_tensor)
        token_indices = selected_route_tensor[:, 0]
        expert_indices = selected_route_tensor[:, 1]
        tokens_per_expert = torch.bincount(expert_indices, minlength=manager.num_local_experts)
        handle = SimpleNamespace(
            replay_index=len(observed_dispatches),
            token_indices=token_indices,
            num_permuted_tokens=len(selected_routes),
        )
        observed_dispatches.append(
            {
                "bound": kwargs["num_permuted_tokens"],
                "handle": handle,
                "routes": selected_routes,
                "tokens_per_expert": tuple(int(value) for value in tokens_per_expert),
            }
        )
        dispatched_hidden = kwargs["x"].index_select(0, token_indices)
        dispatched_probs = kwargs["probs"][
            selected_route_tensor[:, 0],
            selected_route_tensor[:, 1],
        ]
        return dispatched_hidden, dispatched_probs, None, tokens_per_expert, handle

    def fake_hybrid_ep_combine(**kwargs):
        num_permuted_tokens = int(kwargs["num_permuted_tokens"].item())
        observed_combines.append(
            {
                "handle": kwargs["handle"],
                "num_permuted_tokens": num_permuted_tokens,
                "input_rows": kwargs["x"].shape[0],
            }
        )
        combined = kwargs["x"].new_zeros(
            (graph_input.numel() // graph_input.shape[-1], graph_input.shape[-1])
        )
        combined.index_add_(0, kwargs["handle"].token_indices, kwargs["x"])
        return combined

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_dispatch",
        fake_hybrid_ep_dispatch,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine",
        fake_hybrid_ep_combine,
    )

    capture_preprocessed, _ = dispatcher.dispatch_preprocess(
        graph_input, capture_routing_map, capture_routing_map.to(dtype=torch.float32)
    )
    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, capture_preprocessed)
    assert state.backend_state.num_permuted_tokens is None

    for replay_index, (routing_map, expected_routes, expected_tokens_per_expert) in enumerate(
        replay_cases
    ):
        if replay_index > 0:
            assert manager._original_num_tokens is None
            assert manager._padded_num_tokens is None
            assert manager.num_permuted_tokens is None
            assert manager.handle is None

        probs = routing_map.to(dtype=torch.float32) / routing_map.sum(dim=-1, keepdim=True)
        preprocessed, probs = dispatcher.dispatch_preprocess(graph_input, routing_map, probs)
        manager.num_permuted_tokens = 999
        dispatcher.restore_cudagraph_replay_state(state, graph_input, preprocessed)

        assert manager._original_num_tokens == 6
        assert manager._padded_num_tokens == 6
        assert manager.num_permuted_tokens is None
        assert manager.handle is None

        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(preprocessed, probs)
        expert_input, tokens_per_expert, permuted_probs = dispatcher.dispatch_postprocess(
            dispatched_hidden, dispatched_probs
        )
        assert expert_input is dispatched_hidden
        assert tuple(int(value) for value in tokens_per_expert) == expected_tokens_per_expert
        assert observed_dispatches[replay_index]["routes"] == expected_routes
        assert observed_dispatches[replay_index]["tokens_per_expert"] == expected_tokens_per_expert
        assert observed_dispatches[replay_index]["bound"] is None
        assert manager.num_permuted_tokens.item() == len(expected_routes)
        assert manager.handle is observed_dispatches[replay_index]["handle"]

        expert_output = expert_input * permuted_probs.unsqueeze(-1)
        combine_input = dispatcher.combine_preprocess(expert_output)
        combined = dispatcher.token_combine(combine_input)
        output = dispatcher.combine_postprocess(combined)

        assert observed_combines[replay_index]["handle"] is observed_dispatches[replay_index][
            "handle"
        ]
        assert observed_combines[replay_index]["num_permuted_tokens"] == len(expected_routes)
        assert observed_combines[replay_index]["input_rows"] == len(expected_routes)
        assert manager._original_num_tokens is None
        assert manager._padded_num_tokens is None
        assert manager.num_permuted_tokens is None
        assert manager.handle is None
        torch.testing.assert_close(output, graph_input)
        dispatcher.validate_cudagraph_continuation(state, output)

    assert [dispatch["bound"] for dispatch in observed_dispatches] == [None, None]
    assert observed_dispatches[0]["handle"] is not observed_dispatches[1]["handle"]
    assert [combine["num_permuted_tokens"] for combine in observed_combines] == [6, 8]


def test_hybridep_cudagraph_continuation_requires_restored_physical_rows() -> None:
    dispatcher = _make_hybridep_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    preprocessed = torch.empty((6, 4), dtype=torch.bfloat16)
    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)

    with pytest.raises(RuntimeError, match="original_num_tokens"):
        dispatcher.validate_cudagraph_continuation(
            dataclasses.replace(
                state, backend_state=dataclasses.replace(state.backend_state, original_num_tokens=5)
            ),
            graph_input,
        )


@pytest.mark.parametrize("backend", ["deepep", "ncclep"])
@pytest.mark.parametrize("static_shape", [False, True])
def test_flex_cudagraph_replay_state_rejects_unsupported_backends_before_capture(
    backend: str, static_shape: bool
) -> None:
    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_flex_dispatcher_backend=backend, moe_ncclep_static_shape=static_shape
    )

    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        dispatcher.validate_cudagraph_replay_capability()


def test_hybridep_cudagraph_replay_state_rejects_backend_mutation() -> None:
    dispatcher = _make_hybridep_replay_state_dispatcher()
    graph_input = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    preprocessed = torch.empty((6, 4), dtype=torch.bfloat16)
    state = dispatcher.snapshot_cudagraph_replay_state(graph_input, preprocessed)
    dispatcher.config.moe_flex_dispatcher_backend = "deepep"

    with pytest.raises(RuntimeError, match="topology fingerprint"):
        dispatcher.restore_cudagraph_replay_state(state, graph_input, preprocessed)


def test_allgather_cudagraph_replay_state_rejects_packed_sparse_routes() -> None:
    dispatcher = MoEAllGatherTokenDispatcher.__new__(MoEAllGatherTokenDispatcher)
    dispatcher.config = SimpleNamespace(thd_max_packed_sequences=8)

    with pytest.raises(RuntimeError, match="AllGather"):
        dispatcher.validate_cudagraph_replay_capability()


@pytest.mark.skipif(
    not is_deep_ep_available() and not is_hybrid_ep_available(),
    reason="Deep EP and Hybrid EP are not available",
)
class TestFlexDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize(
        "moe_flex_dispatcher_backend",
        [
            "deepep",
            "hybridep",
            # NCCL EP aborts in dev CI with a pybind11 GIL dec_ref failure.
            pytest.param("ncclep", marks=pytest.mark.flaky_in_dev),
        ],
    )
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if moe_flex_dispatcher_backend == "ncclep" and not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_test()
        # reset experimental flag to False
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not is_nccl_ep_zero_copy_available(), reason="NCCL EP zero-copy TE API is not available"
    )
    @pytest.mark.skipif(
        not is_op_fuser_available(), reason="op-fuser (static-shape/zero-copy) needs TE>=2.14"
    )
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8)])
    def test_forward_backward_zero_copy(self, tp_size, ep_size):
        # zero-copy requires a capacity factor, which requires BOTH op-fuser and grouped_gemm; bf16
        # so no
        # fp8/Blackwell dependency. The op-fuser needs tp=1 and a SwiGLU activation. Parity: the
        # zero-copy IO path must match the staged (no-zc) path.
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            # ncclep sizes a per-rank recv buffer from this and overflow HARD-TRAPS; size generously.
            moe_expert_rank_capacity_factor=8.0,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.moe_layer_zero_copy_parity_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_capacity_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_capacity_test()
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", [True])
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_router_padding_for_fp8_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=4,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False
