# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import pytest
import torch

from megatron.core.models.common import fine_grained_callables as common_callables
from megatron.core.models.common.fine_grained_callables import build_layer_callables
from megatron.core.models.gpt import fine_grained_callables as gpt_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    DummyNode,
    DummyState,
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    get_valid_token_dispatcher_types,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


def run_model_ref_with_capture(model, input_tensors, iterations):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    output_tensors = []
    for i in range(iterations):
        output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_model_submodules_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()

    output_tensors = []
    # get callables
    callables, dw = build_layer_callables(model)
    attn, dispatch, moe, combine, post_process = callables
    assert post_process is None
    dummy_model = DummyState()
    dummy_model.decoder = DummyState()
    dummy_model.decoder.final_layernorm = None
    for i in range(microbatches):
        # build mock func/state
        node = DummyNode()
        node.is_mtp = False
        node.chunk_state.model = dummy_model

        # attn fwd
        local_tokens, probs = attn(node, input_tensors[i])

        # dispatch fwd
        dispatched_tokens = dispatch(node, local_tokens, probs)

        # moe fwd
        expert_output = moe(node, dispatched_tokens)

        # combine fwd
        hidden_states = combine(node, expert_output)

        # loss
        output_tensors.append(hidden_states)
        hidden_states.backward(torch.ones_like(hidden_states))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def test_mtp_pre_dispatch_applies_hybrid_empty_decoder_final_norm(monkeypatch):
    """Covers the HybridModel empty-decoder MTP pre-dispatch final_norm path."""

    from megatron.core.models.hybrid.hybrid_model import HybridModel

    def inner_pre_dispatch(_node, hidden_states):
        return hidden_states

    def unused_forward(*_args, **_kwargs):
        raise AssertionError("only MTP pre-dispatch should run in this test")

    def fake_build_layer_callables(_layer):
        return (
            [inner_pre_dispatch, unused_forward, unused_forward, unused_forward, None],
            {"pre_dispatch_computation": object()},
        )

    class FakeMTPConfig:
        sequence_parallel = False
        cuda_graph_modules = []

    class FakeMTPLayer:
        config = FakeMTPConfig()
        eh_proj = object()
        mtp_model_layer = _make_task7_fine_grained_transformer_leaf(moe=True)

        def _get_embeddings(
            self, input_ids, position_ids, embedding, hidden_states, packed_seq_params, padding_mask
        ):
            return input_ids, position_ids, padding_mask, None, hidden_states

        def _concat_embeddings(self, hidden_states, decoder_input):
            return hidden_states

        def _postprocess(self, hidden_states):
            return hidden_states

    monkeypatch.setattr(common_callables, "build_layer_callables", fake_build_layer_callables)
    monkeypatch.setattr(common_callables, "get_layer_moe_metadata", lambda _layer: (True, 1))
    monkeypatch.setattr(common_callables, "get_mtp_layer_offset", lambda _config, _vp_stage: 0)

    model = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.decoder = DummyState()
    model.decoder.layers = []
    model.decoder.final_norm = lambda hidden_states: hidden_states + 4.0
    model.embedding = object()
    model.vp_stage = None

    node = DummyNode()
    node.chunk_state = DummyState()
    node.chunk_state.model = model
    node.chunk_state.context = None
    node.chunk_state.packed_seq_params = None
    node.is_first_layer = True

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2).requires_grad_()
    expected = hidden_states + 4.0
    forward_funcs, _ = common_callables.build_mtp_layer_callables(FakeMTPLayer())

    output = forward_funcs[0](node, hidden_states)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(node.chunk_state.mtp_hidden_states[0], expected)


def test_pre_dispatch_forwards_padding_mask_to_moe_route():
    """The split MoE route receives the structural padding mask by identity."""

    routing_map = torch.tensor([[True, False], [False, True], [True, False]])

    class RecordingMoELayer(MoELayer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.use_shared_expert = False
            self.shared_expert_overlap = False
            self.route_padding_mask = None
            self.preprocess_routing_map = None

        def shared_experts_compute(self, hidden_states):
            return None

        def route(self, hidden_states, padding_mask=None):
            self.route_padding_mask = padding_mask
            probs = torch.ones_like(routing_map, dtype=hidden_states.dtype)
            return probs, routing_map

        def preprocess(self, hidden_states, probs, received_routing_map):
            self.preprocess_routing_map = received_routing_map
            return hidden_states, probs

    mlp = RecordingMoELayer()
    layer = DummyState()
    layer.config = DummyState()
    layer.config.moe_token_dispatcher_type = "alltoall"
    layer.config.moe_flex_dispatcher_backend = None
    layer.mlp = mlp
    layer.offload_mlp_norm = False
    layer.recompute_pre_mlp_layernorm = False
    layer.pre_mlp_layernorm = torch.nn.Identity()
    layer._forward_attention = lambda **kwargs: (kwargs["hidden_states"], None)
    layer._forward_mlp = lambda *_args, **_kwargs: None

    def init_backward_dw_wrapper():
        layer.backward_dw_wrapper = object()

    layer.init_backward_dw_wrapper = init_backward_dw_wrapper

    padding_mask = torch.tensor([[False], [True], [False]])
    node = DummyNode()
    node.layer_state = DummyState()
    node.chunk_state = DummyState()
    node.chunk_state.padding_mask = padding_mask
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)

    forward_funcs, _ = gpt_callables.build_transformer_layer_callables(layer)
    forward_funcs[0](node, hidden_states)

    assert mlp.route_padding_mask is padding_mask
    assert mlp.preprocess_routing_map is routing_map


def _make_task7_fine_grained_transformer_leaf(*, moe):
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP

    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attention = IdentityOp()
    layer.cross_attention = IdentityOp()
    if moe:
        layer.mlp = MoELayer.__new__(MoELayer)
        torch.nn.Module.__init__(layer.mlp)
        layer.mlp.num_local_experts = 2
    else:
        layer.mlp = MLP.__new__(MLP)
        torch.nn.Module.__init__(layer.mlp)
    return layer


def _make_task7_fine_grained_mamba_leaf():
    from megatron.core.ssm.mamba_layer import MambaLayer

    layer = MambaLayer.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    return layer


def _make_task7_hybrid_stack(layers):
    from megatron.core.models.hybrid.hybrid_block import HybridStack

    stack = HybridStack.__new__(HybridStack)
    torch.nn.Module.__init__(stack)
    stack.layers = torch.nn.ModuleList(layers)
    return stack


def _make_task7_mtp_wrapper(inner_layer):
    from types import SimpleNamespace

    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(sequence_parallel=False, cuda_graph_modules=[])
    layer.mtp_model_layer = inner_layer
    layer.eh_proj = torch.nn.Identity()
    return layer


def _task7_fake_transformer_callables(layer, calls):
    calls.append(layer)

    def passthrough(_node, hidden_states, *_args, **_kwargs):
        return hidden_states

    return (
        [passthrough, passthrough, passthrough, passthrough, None],
        {"pre_dispatch_computation": object()},
    )


def test_one_moe_leaf_hybrid_mtp_preserves_legacy_five_callable_contract(monkeypatch):
    moe = _make_task7_fine_grained_transformer_leaf(moe=True)
    mtp = _make_task7_mtp_wrapper(_make_task7_hybrid_stack([moe]))
    calls = []
    monkeypatch.setattr(
        common_callables,
        "build_transformer_layer_callables",
        lambda layer: _task7_fake_transformer_callables(layer, calls),
    )

    forward_funcs, backward_dw = build_layer_callables(mtp)

    assert calls == [moe]
    assert len(forward_funcs) == 5
    assert all(callable(func) for func in forward_funcs)
    assert common_callables.get_layer_moe_metadata(mtp) == (True, 2)
    assert backward_dw["pre_dispatch_computation"][1] is mtp.eh_proj


@pytest.mark.parametrize("case", ["zero", "dense", "mamba", "mixed", "multi"])
def test_unsupported_hybrid_mtp_fails_before_callable_schedule(monkeypatch, case):
    if case == "zero":
        layers = []
    elif case == "dense":
        layers = [_make_task7_fine_grained_transformer_leaf(moe=False)]
    elif case == "mamba":
        layers = [_make_task7_fine_grained_mamba_leaf()]
    elif case == "mixed":
        layers = [
            _make_task7_fine_grained_transformer_leaf(moe=True),
            _make_task7_fine_grained_mamba_leaf(),
            _make_task7_fine_grained_transformer_leaf(moe=False),
        ]
    else:
        layers = [
            _make_task7_fine_grained_transformer_leaf(moe=True),
            _make_task7_fine_grained_transformer_leaf(moe=True),
        ]
    mtp = _make_task7_mtp_wrapper(_make_task7_hybrid_stack(layers))
    calls = []
    monkeypatch.setattr(
        common_callables,
        "build_transformer_layer_callables",
        lambda layer: _task7_fake_transformer_callables(layer, calls),
    )

    with pytest.raises(ValueError, match="fine-grained MTP.*exactly one.*MoE Transformer"):
        build_layer_callables(mtp)

    assert calls == []


def test_one_moe_leaf_fine_grained_mtp_keeps_packed_execution_unsupported(monkeypatch):
    """The supported five-callable layout still fails before executing a packed MTP node."""

    from types import MethodType, SimpleNamespace

    from megatron.core.packed_seq_params import PackedSeqParams

    moe = _make_task7_fine_grained_transformer_leaf(moe=True)
    mtp = _make_task7_mtp_wrapper(_make_task7_hybrid_stack([moe]))
    build_calls = []
    execution_calls = []

    def fake_transformer_callables(layer):
        build_calls.append(layer)

        def reject_execution(_node, hidden_states, *_args, **_kwargs):
            execution_calls.append(layer)
            return hidden_states

        return (
            [reject_execution, reject_execution, reject_execution, reject_execution, None],
            {"pre_dispatch_computation": object()},
        )

    def fake_get_embeddings(
        self, input_ids, position_ids, embedding, hidden_states, packed_seq_params, padding_mask
    ):
        return input_ids, position_ids, padding_mask, None, hidden_states

    monkeypatch.setattr(
        common_callables, "build_transformer_layer_callables", fake_transformer_callables
    )
    monkeypatch.setattr(mtp, "_get_embeddings", MethodType(fake_get_embeddings, mtp))
    forward_funcs, _ = build_layer_callables(mtp)
    node = DummyNode()
    node.is_first_layer = False
    node.chunk_state = DummyState()
    node.chunk_state.input_ids = torch.ones((1, 4), dtype=torch.int64)
    node.chunk_state.position_ids = torch.arange(4, dtype=torch.int64).unsqueeze(0)
    node.chunk_state.padding_mask = torch.zeros((1, 4), dtype=torch.bool)
    node.chunk_state.context = None
    node.chunk_state.model = SimpleNamespace(embedding=object())
    node.chunk_state.packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        seq_aux_loss_sample_ids=torch.zeros(4, dtype=torch.int64),
        seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.int64),
        seq_aux_loss_max_samples=2,
    )

    with pytest.raises(AssertionError, match="sequence packing.*not yet supported"):
        forward_funcs[0](node, torch.ones((4, 1, 8)))

    assert build_calls == [moe]
    assert execution_calls == []


def test_single_transformer_gpt_mtp_and_non_mtp_callable_dispatch_regressions(monkeypatch):
    moe = _make_task7_fine_grained_transformer_leaf(moe=True)
    mtp = _make_task7_mtp_wrapper(moe)
    calls = []
    monkeypatch.setattr(
        common_callables,
        "build_transformer_layer_callables",
        lambda layer: _task7_fake_transformer_callables(layer, calls),
    )

    mtp_forward_funcs, _ = build_layer_callables(mtp)
    direct_forward_funcs, _ = build_layer_callables(moe)

    assert len(mtp_forward_funcs) == 5
    assert direct_forward_funcs[-1] is None
    assert calls == [moe, moe]
    assert common_callables.get_layer_moe_metadata(mtp) == (True, 2)
    assert common_callables.get_layer_moe_metadata(moe) == (True, 2)


@pytest.mark.parametrize("case", ["dense", "mamba", "opaque", "unscoped_moe"])
def test_unsupported_direct_mtp_fails_before_callable_schedule(monkeypatch, case):
    if case == "dense":
        inner_layer = _make_task7_fine_grained_transformer_leaf(moe=False)
    elif case == "mamba":
        inner_layer = _make_task7_fine_grained_mamba_leaf()
    elif case == "unscoped_moe":
        inner_layer = _make_task7_fine_grained_transformer_leaf(moe=True)
    else:
        inner_layer = torch.nn.Identity()
    mtp = _make_task7_mtp_wrapper(inner_layer)
    if case == "unscoped_moe":
        from megatron.core.transformer.enums import CudaGraphModule

        mtp.config.cuda_graph_modules = [CudaGraphModule.mamba]
    calls = []
    monkeypatch.setattr(
        common_callables,
        "build_transformer_layer_callables",
        lambda layer: _task7_fake_transformer_callables(layer, calls),
    )

    with pytest.raises(ValueError, match="fine-grained MTP.*exactly one.*MoE Transformer"):
        build_layer_callables(mtp)

    assert calls == []


class TestTransformerLayerSubmoduleCallables:
    """
    Test class for transformer layer submodule callables.

    This class contains tests to verify that the transformer layer submodule callables
    provide the same results as the reference implementation.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("permute_fusion", [True, False])
    def test_1f1b_overlap(self, dispatcher_type, grouped_gemm, permute_fusion):
        """
        Tests the 1-forward-1-backward overlap optimization.

        This test verifies that the all-to-all overlap optimization produces
        the same results as the reference implementation.
        """

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        qk_layernorm = True
        extra_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
            "moe_permute_fusion": permute_fusion,
            "qk_layernorm": qk_layernorm,
        }
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        config = get_test_config(extra_kwargs=extra_kwargs, moe_grouped_gemm=grouped_gemm)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=8,
                moe_grouped_gemm=grouped_gemm,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=True,
            )
            model = TransformerLayer(config, transformer_layer_submodules)

            params = reset_model(model)
            input_tensors = [build_data() for _ in range(microbatches)]

            capture_ref = run_model_ref_with_capture(model, input_tensors, microbatches)
            reset_model(model, params)
            capture_callables = run_model_submodules_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_callables, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
            Utils.destroy_model_parallel()
