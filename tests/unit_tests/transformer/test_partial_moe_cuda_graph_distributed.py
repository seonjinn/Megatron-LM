# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.moe.capacity_tracker import (
    destroy_moe_capacity_tracker,
    get_moe_capacity_tracker,
)
from megatron.core.transformer.moe.cuda_graph_replay import (
    AlltoAllCudaGraphState,
    HybridEPCudaGraphState,
    MoECudaGraphReplayState,
)
from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP, reset_hybrid_ep_buffer
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

CAPTURE_WARMUPS = 3
CHANGED_ROUTE_REPLAYS = 20
HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 16
MICRO_BATCH_SIZE = 1


@dataclass(frozen=True)
class _TopologyCase:
    row_id: str
    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: int
    expert_parallel_size: int
    num_experts: int
    router_topk: int
    dispatcher: str
    layer_type: type[TransformerLayer]
    has_shared_expert: bool
    moe_latent_size: int | None = None


CASES = (
    _TopologyCase(
        row_id="dropless_hybridep_nano16",
        world_size=16,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        context_parallel_size=2,
        expert_parallel_size=8,
        num_experts=8,
        router_topk=6,
        dispatcher="hybridep",
        layer_type=MoETransformerLayer,
        has_shared_expert=True,
    ),
    _TopologyCase(
        row_id="dropless_alltoall_qwen30_16",
        world_size=16,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=16,
        num_experts=16,
        router_topk=8,
        dispatcher="alltoall",
        layer_type=TransformerLayer,
        has_shared_expert=False,
    ),
    _TopologyCase(
        row_id="dropless_alltoall_super32",
        world_size=32,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=16,
        num_experts=32,
        router_topk=22,
        dispatcher="alltoall",
        layer_type=MoETransformerLayer,
        has_shared_expert=True,
        moe_latent_size=16,
    ),
    _TopologyCase(
        row_id="dropless_hybridep_qwen235_64",
        world_size=64,
        tensor_parallel_size=2,
        pipeline_parallel_size=4,
        context_parallel_size=2,
        expert_parallel_size=16,
        num_experts=16,
        router_topk=8,
        dispatcher="hybridep",
        layer_type=TransformerLayer,
        has_shared_expert=False,
    ),
)


@dataclass(frozen=True)
class _RouteSnapshot:
    probabilities: torch.Tensor
    expert_ids: torch.Tensor
    expert_counts: torch.Tensor


class _Decoder(torch.nn.Module):
    def __init__(self, layer: TransformerLayer) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList((layer,))


class _PartialMoEModel(torch.nn.Module):
    def __init__(self, config: TransformerConfig, layer: TransformerLayer) -> None:
        super().__init__()
        self.config = config
        self.decoder = _Decoder(layer)

    @property
    def layer(self) -> TransformerLayer:
        return self.decoder.layers[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        output, context = self.layer(
            hidden_states,
            attention_mask=None,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )
        assert context is None
        return output

    def zero_grad_buffer(self) -> None:
        self.zero_grad(set_to_none=True)


def _make_config(case: _TopologyCase, *, graph: bool) -> TransformerConfig:
    dispatcher_kwargs: dict[str, object]
    if case.dispatcher == "hybridep":
        dispatcher_kwargs = {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "hybridep",
            "moe_hybridep_pad_uneven_dispatch_inputs": False,
        }
    else:
        dispatcher_kwargs = {"moe_token_dispatcher_type": "alltoall"}
    return TransformerConfig(
        num_layers=case.pipeline_parallel_size,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=64,
        tensor_model_parallel_size=case.tensor_parallel_size,
        pipeline_model_parallel_size=case.pipeline_parallel_size,
        context_parallel_size=case.context_parallel_size,
        expert_model_parallel_size=case.expert_parallel_size,
        expert_tensor_parallel_size=1,
        sequence_parallel=case.tensor_parallel_size > 1,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=case.num_experts,
        moe_ffn_hidden_size=64,
        moe_grouped_gemm=False,
        moe_router_topk=case.router_topk,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=0.0,
        moe_expert_capacity_factor=None,
        moe_expert_rank_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_shared_expert_intermediate_size=32 if case.has_shared_expert else None,
        moe_shared_expert_overlap=False,
        moe_latent_size=case.moe_latent_size,
        cuda_graph_impl="transformer_engine" if graph else "none",
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        cuda_graph_warmup_steps=CAPTURE_WARMUPS,
        thd_max_packed_sequences=4,
        **dispatcher_kwargs,
    )


def _make_model(case: _TopologyCase, config: TransformerConfig) -> _PartialMoEModel:
    full_submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=case.num_experts,
        moe_grouped_gemm=False,
    )
    submodules = TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=IdentityOp,
        self_attn_bda=IdentityFuncOp,
        pre_cross_attn_layernorm=IdentityOp,
        cross_attention=IdentityOp,
        cross_attn_bda=IdentityFuncOp,
        pre_mlp_layernorm=full_submodules.pre_mlp_layernorm,
        mlp=full_submodules.mlp,
        mlp_bda=full_submodules.mlp_bda,
    )
    layer = case.layer_type(config, submodules, layer_number=1, add_layer_offset=False)
    model = _PartialMoEModel(config, layer).cuda()
    assert type(model.layer) is case.layer_type
    if case.layer_type is TransformerLayer:
        assert "_record_te_cuda_graph_dispatcher_replay_state" in TransformerLayer.__dict__
        assert "_restore_te_cuda_graph_dispatcher_replay_state" in TransformerLayer.__dict__
    assert model.layer.mlp.use_shared_expert is case.has_shared_expert
    assert (model.layer.mlp.shared_experts is not None) is case.has_shared_expert
    if case.dispatcher == "hybridep":
        assert isinstance(model.layer.mlp.token_dispatcher, MoEFlexTokenDispatcher)
        assert model.layer.mlp.token_dispatcher.config.moe_flex_dispatcher_backend == "hybridep"
    else:
        assert isinstance(model.layer.mlp.token_dispatcher, MoEAlltoAllTokenDispatcher)
    return model


def _validate_process_groups(case: _TopologyCase) -> ProcessGroupCollection:
    groups = ProcessGroupCollection.use_mpu_process_groups()
    assert groups.tp.size() == case.tensor_parallel_size
    assert groups.pp.size() == case.pipeline_parallel_size
    assert groups.cp.size() == case.context_parallel_size
    assert groups.ep.size() == case.expert_parallel_size
    assert groups.expt_tp.size() == 1
    assert torch.distributed.get_world_size() == case.world_size
    return groups


def _make_packed_inputs(
    case: _TopologyCase, groups: ProcessGroupCollection
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, PackedSeqParams]:
    physical_tokens = (
        SEQUENCE_LENGTH // case.context_parallel_size // case.tensor_parallel_size
    )
    route_inputs = []
    for route_offset in (0, case.num_experts - case.router_topk):
        hidden_states = torch.full(
            (physical_tokens, MICRO_BATCH_SIZE, HIDDEN_SIZE),
            -0.25,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        for token in range(physical_tokens):
            for choice in range(case.router_topk):
                expert = (route_offset + choice + token) % case.num_experts
                hidden_states[token, 0, expert] = 2.0 + choice / 8.0
        route_inputs.append(hidden_states)
    padding_mask = torch.zeros(
        (MICRO_BATCH_SIZE, physical_tokens),
        dtype=torch.bool,
        device=torch.cuda.current_device(),
    )
    padding_mask[:, -1] = True
    valid_tokens = physical_tokens - 1
    cu_seqlens = torch.tensor(
        (0, valid_tokens), dtype=torch.int32, device=torch.cuda.current_device()
    )
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=valid_tokens,
        max_seqlen_kv=valid_tokens,
        local_cp_size=case.context_parallel_size,
        cp_group=groups.cp,
        total_tokens=physical_tokens,
        tokens_per_sample=physical_tokens,
        seq_aux_loss_sample_ids=torch.zeros(
            physical_tokens, dtype=torch.int64, device=torch.cuda.current_device()
        ),
        seq_aux_loss_num_samples=torch.ones(
            (), dtype=torch.int64, device=torch.cuda.current_device()
        ),
        seq_aux_loss_max_samples=1,
    )
    return (route_inputs[0], route_inputs[1]), padding_mask, packed_seq_params


def _set_deterministic_router(model: _PartialMoEModel, case: _TopologyCase) -> None:
    with torch.no_grad():
        weight = model.layer.mlp.router.weight
        weight.zero_()
        for expert in range(case.num_experts):
            weight[expert, expert] = 4.0


def _route_snapshot(model: _PartialMoEModel) -> _RouteSnapshot:
    dispatcher = model.layer.mlp.token_dispatcher
    if isinstance(dispatcher, MoEAlltoAllTokenDispatcher):
        routing_map = dispatcher.routing_map
        probabilities = dispatcher.probs
    else:
        assert isinstance(dispatcher, MoEFlexTokenDispatcher)
        routing_map = dispatcher._comm_manager.routing_map
        probabilities = dispatcher._comm_manager.token_probs
    assert torch.is_tensor(routing_map) and torch.is_tensor(probabilities)
    routing_map = routing_map.detach().clone()
    expert_axis = torch.arange(
        routing_map.shape[-1], dtype=torch.int64, device=routing_map.device
    )
    expert_ids = torch.where(routing_map, expert_axis, -1)
    expert_counts = routing_map.sum(dim=tuple(range(routing_map.ndim - 1)))
    return _RouteSnapshot(
        probabilities=probabilities.detach().clone(),
        expert_ids=expert_ids,
        expert_counts=expert_counts,
    )


def _assert_capacity_is_zero() -> None:
    snapshot = get_moe_capacity_tracker().snapshot()
    for value in (
        snapshot.selected_assignments,
        snapshot.dropped_assignments,
        snapshot.valid_token_drops,
        snapshot.rank_overflow_events,
    ):
        assert value.item() == 0


def _count_method(
    owner: object, name: str, counters: dict[str, int], key: str
) -> None:
    original = getattr(owner, name)

    def counted(*args: object, **kwargs: object) -> object:
        counters[key] += 1
        return original(*args, **kwargs)

    setattr(owner, name, counted)


def _assert_replay_geometry(
    case: _TopologyCase,
    state: MoECudaGraphReplayState,
    graph_input: torch.Tensor,
) -> None:
    physical_shape = torch.Size((graph_input.shape[0], HIDDEN_SIZE))
    assert state.input_signature.shape == graph_input.shape
    assert state.flattened_input_shape == physical_shape
    if case.dispatcher == "hybridep":
        assert state.dispatcher_kind == "flex-hybridep"
        assert isinstance(state.backend_state, HybridEPCudaGraphState)
        assert state.backend_state.original_num_tokens == graph_input.shape[0]
        assert state.backend_state.padded_num_tokens == graph_input.shape[0]
        assert state.backend_state.capacity is None
        assert state.backend_state.num_permuted_tokens is None
        assert state.backend_state.tokens_per_expert is None
    else:
        assert state.dispatcher_kind == "alltoall"
        assert isinstance(state.backend_state, AlltoAllCudaGraphState)
        hidden_size = case.moe_latent_size or HIDDEN_SIZE
        assert state.backend_state.hidden_shape == torch.Size(
            (*graph_input.shape[:-1], hidden_size)
        )
        assert state.backend_state.hidden_shape_before_permute == torch.Size(
            (graph_input.shape[0], hidden_size)
        )
        assert state.backend_state.capacity is None
        assert state.backend_state.num_out_tokens is not None
        assert state.backend_state.preprocessed_signature is not None
        assert state.backend_state.preprocessed_signature.shape[-1] == hidden_size


def _assert_tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.internal
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.row_id)
def test_dropless_partial_moe_cuda_graph_distributed(case: _TopologyCase) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != case.world_size:
        pytest.skip(f"{case.row_id} requires exactly {case.world_size} ranks")
    if not torch.cuda.is_available():
        pytest.fail(f"{case.row_id} requires CUDA", pytrace=False)
    if not is_te_min_version("2.10.0"):
        pytest.fail(
            f"{case.row_id} requires TransformerEngine 2.10 or newer", pytrace=False
        )
    if case.dispatcher == "hybridep" and not HAVE_HYBRIDEP:
        pytest.fail(f"{case.row_id} requires HybridEP", pytrace=False)

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    Utils.world_size = case.world_size
    Utils.rank = global_rank
    helper: TECudaGraphHelper | None = None
    counters = {
        "router": 0,
        "preprocess": 0,
        "dispatch": 0,
        "expert": 0,
        "combine": 0,
        "postprocess": 0,
        "token_dispatch": 0,
        "token_combine": 0,
    }

    try:
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=case.tensor_parallel_size,
            pipeline_model_parallel_size=case.pipeline_parallel_size,
            context_parallel_size=case.context_parallel_size,
            expert_tensor_parallel_size=1,
            expert_model_parallel_size=case.expert_parallel_size,
        )
        groups = _validate_process_groups(case)
        data_parallel_size = groups.dp.size()
        init_num_microbatches_calculator(
            rank=global_rank,
            global_batch_size=data_parallel_size,
            micro_batch_size=MICRO_BATCH_SIZE,
            data_parallel_size=data_parallel_size,
        )
        model_parallel_cuda_manual_seed(1234)
        graph_config = _make_config(case, graph=True)
        eager_config = replace(
            graph_config,
            cuda_graph_impl="none",
            cuda_graph_modules=[],
        )
        eager_model = _make_model(case, eager_config)
        graph_model = _make_model(case, graph_config)
        graph_model.load_state_dict(eager_model.state_dict())
        _set_deterministic_router(eager_model, case)
        _set_deterministic_router(graph_model, case)
        route_inputs, padding_mask, packed_seq_params = _make_packed_inputs(case, groups)

        capacity_tracker = get_moe_capacity_tracker()
        capacity_tracker.initialize(torch.device("cuda", local_rank))
        for warmup in range(CAPTURE_WARMUPS):
            graph_model.zero_grad(set_to_none=True)
            capacity_tracker.reset()
            output = graph_model(
                route_inputs[warmup % 2],
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
            output.float().square().mean().backward()
            _assert_capacity_is_zero()

        helper = TECudaGraphHelper(
            model=[graph_model],
            config=graph_config,
            seq_length=SEQUENCE_LENGTH,
            micro_batch_size=MICRO_BATCH_SIZE,
            optimizers=[],
            pg_collection=groups,
            sample_packed_seq_params=packed_seq_params,
        )
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert helper._compatibility_bank_manager is not None
        manager = helper._compatibility_bank_manager
        counter_start = manager.snapshot_execution_counters()
        states = graph_model.layer._te_cuda_graph_dispatcher_replay_states
        assert len(states) == 1
        _assert_replay_geometry(case, states[0], route_inputs[0])

        mlp = graph_model.layer.mlp
        dispatcher = mlp.token_dispatcher
        _count_method(mlp.router, "forward", counters, "router")
        _count_method(dispatcher, "dispatch_preprocess", counters, "preprocess")
        _count_method(mlp, "dispatch", counters, "dispatch")
        _count_method(mlp, "routed_experts_compute", counters, "expert")
        _count_method(mlp, "combine", counters, "combine")
        _count_method(mlp, "postprocess", counters, "postprocess")
        _count_method(dispatcher, "token_dispatch", counters, "token_dispatch")
        _count_method(dispatcher, "token_combine", counters, "token_combine")

        observed_routes: dict[int, _RouteSnapshot] = {}
        valid_tokens = ~padding_mask.reshape(-1)
        for replay in range(CHANGED_ROUTE_REPLAYS):
            route_index = replay % 2
            hidden_states = route_inputs[route_index]
            eager_model.zero_grad(set_to_none=True)
            graph_model.zero_grad(set_to_none=True)

            capacity_tracker.reset()
            eager_output = eager_model(
                hidden_states,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
            eager_routes = _route_snapshot(eager_model)
            eager_valid_output = eager_output.reshape(-1, HIDDEN_SIZE)[valid_tokens]
            eager_loss = eager_valid_output.float().square().mean()
            eager_loss.backward()
            _assert_capacity_is_zero()

            capacity_tracker.reset()
            graph_output = graph_model(
                hidden_states,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
            graph_routes = _route_snapshot(graph_model)
            graph_valid_output = graph_output.reshape(-1, HIDDEN_SIZE)[valid_tokens]
            graph_loss = graph_valid_output.float().square().mean()
            graph_loss.backward()
            _assert_capacity_is_zero()

            _assert_tensor_parity(graph_valid_output, eager_valid_output)
            _assert_tensor_parity(graph_loss, eager_loss)
            _assert_tensor_parity(graph_routes.probabilities, eager_routes.probabilities)
            assert torch.equal(graph_routes.expert_ids, eager_routes.expert_ids)
            assert torch.equal(graph_routes.expert_counts, eager_routes.expert_counts)
            graph_model.layer.mlp.token_dispatcher.validate_cudagraph_continuation(
                states[0], graph_output
            )
            observed_routes.setdefault(route_index, graph_routes)

            eager_parameters = dict(eager_model.named_parameters())
            graph_parameters = dict(graph_model.named_parameters())
            assert eager_parameters.keys() == graph_parameters.keys()
            with torch.no_grad():
                for name in eager_parameters:
                    eager_parameter = eager_parameters[name]
                    graph_parameter = graph_parameters[name]
                    assert eager_parameter.grad is not None, name
                    assert graph_parameter.grad is not None, name
                    _assert_tensor_parity(graph_parameter.grad, eager_parameter.grad)
                    eager_delta = -0.01 * eager_parameter.grad.float()
                    graph_delta = -0.01 * graph_parameter.grad.float()
                    _assert_tensor_parity(graph_delta, eager_delta)
                    eager_parameter.add_(eager_delta.to(eager_parameter.dtype))
                    graph_parameter.add_(graph_delta.to(graph_parameter.dtype))

        assert not torch.equal(
            observed_routes[0].expert_ids, observed_routes[1].expert_ids
        )
        counter_delta = manager.execution_counter_delta(counter_start)
        assert counter_delta.eligible_calls == CHANGED_ROUTE_REPLAYS
        assert counter_delta.graph_calls == CHANGED_ROUTE_REPLAYS
        assert counter_delta.graph_calls > 0
        assert counters == {
            "router": 0,
            "preprocess": 0,
            "dispatch": CHANGED_ROUTE_REPLAYS,
            "expert": CHANGED_ROUTE_REPLAYS,
            "combine": CHANGED_ROUTE_REPLAYS,
            "postprocess": CHANGED_ROUTE_REPLAYS,
            "token_dispatch": CHANGED_ROUTE_REPLAYS,
            "token_combine": CHANGED_ROUTE_REPLAYS,
        }
    finally:
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()
        destroy_moe_capacity_tracker()
        destroy_num_microbatches_calculator()
        if case.dispatcher == "hybridep":
            reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()
