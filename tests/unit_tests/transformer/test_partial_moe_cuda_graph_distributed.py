# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import gc
import os
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType

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
from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
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
from megatron.core.transformer.moe import moe_utils, router as router_module
from megatron.core.transformer.moe.fused_a2a import (
    HAVE_HYBRIDEP,
    reset_hybrid_ep_buffer,
)
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
DISABLE_NANO_SHARED_EXPERT_ENV = "MCORE_TEST_DISABLE_NANO_SHARED_EXPERT"
IDENTITY_NANO_PRE_MLP_LAYERNORM_ENV = (
    "MCORE_TEST_IDENTITY_NANO_PRE_MLP_LAYERNORM"
)
NANO_TP1_ENV = "MCORE_TEST_NANO_TP1"
DISABLE_ROUTER_TE_GENERAL_GEMM_ENV = "MCORE_TEST_DISABLE_ROUTER_TE_GENERAL_GEMM"
USE_AUTOGRAD_ROUTER_LINEAR_ENV = "MCORE_TEST_USE_AUTOGRAD_ROUTER_LINEAR"
NANO_CG_SUBMODULE_ENV = "MCORE_TEST_NANO_CG_SUBMODULE"
CAPTURE_ONLY_ENV = "MCORE_TEST_CAPTURE_ONLY"
ZERO_GRAD_BEFORE_CAPTURE_ENV = "MCORE_TEST_ZERO_GRAD_BEFORE_CAPTURE"
SKIP_MODEL_WARMUP_ENV = "MCORE_TEST_SKIP_MODEL_WARMUP"
RELEASE_WARMUP_GRAPH_ENV = "MCORE_TEST_RELEASE_WARMUP_GRAPH"
RESET_HYBRIDEP_BEFORE_CAPTURE_ENV = "MCORE_TEST_RESET_HYBRIDEP_BEFORE_CAPTURE"
FORWARD_ONLY_MODEL_WARMUP_ENV = "MCORE_TEST_FORWARD_ONLY_MODEL_WARMUP"
LINEAR_MODEL_WARMUP_ENV = "MCORE_TEST_LINEAR_MODEL_WARMUP"


def _autograd_router_linear(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    router_dtype: torch.dtype,
) -> torch.Tensor:
    return F.linear(
        inp.to(router_dtype),
        weight.to(router_dtype),
        None if bias is None else bias.to(router_dtype),
    )


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
    cuda_graph_modules: tuple[CudaGraphModule, ...]
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
        cuda_graph_modules=(CudaGraphModule.moe_router,),
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
        cuda_graph_modules=(
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        ),
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
        cuda_graph_modules=(
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        ),
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
        cuda_graph_modules=(CudaGraphModule.moe_router,),
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
        pipeline_dtype=torch.bfloat16,
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
        cuda_graph_modules=list(case.cuda_graph_modules),
        cuda_graph_warmup_steps=CAPTURE_WARMUPS,
        thd_max_packed_sequences=4,
        **dispatcher_kwargs,
    )


def _make_model(case: _TopologyCase, config: TransformerConfig) -> _PartialMoEModel:
    full_submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=case.num_experts,
        moe_grouped_gemm=False,
    )
    pre_mlp_layernorm = full_submodules.pre_mlp_layernorm
    if os.environ.get(IDENTITY_NANO_PRE_MLP_LAYERNORM_ENV) == "1":
        if case.row_id != "dropless_hybridep_nano16":
            raise ValueError(
                f"{IDENTITY_NANO_PRE_MLP_LAYERNORM_ENV}=1 only supports "
                "dropless_hybridep_nano16"
            )
        pre_mlp_layernorm = IdentityOp
    submodules = TransformerLayerSubmodules(
        input_layernorm=IdentityOp,
        self_attention=IdentityOp,
        self_attn_bda=IdentityFuncOp,
        pre_cross_attn_layernorm=IdentityOp,
        cross_attention=IdentityOp,
        cross_attn_bda=IdentityFuncOp,
        pre_mlp_layernorm=pre_mlp_layernorm,
        mlp=full_submodules.mlp,
        mlp_bda=full_submodules.mlp_bda,
    )
    layer = case.layer_type(config, submodules, layer_number=1, add_layer_offset=False)
    model = _PartialMoEModel(config, layer).cuda()
    assert type(model.layer) is case.layer_type
    if case.layer_type is TransformerLayer:
        assert (
            "_record_te_cuda_graph_dispatcher_replay_state" in TransformerLayer.__dict__
        )
        assert (
            "_restore_te_cuda_graph_dispatcher_replay_state"
            in TransformerLayer.__dict__
        )
    assert model.layer.mlp.use_shared_expert is case.has_shared_expert
    assert (model.layer.mlp.shared_experts is not None) is case.has_shared_expert
    if case.dispatcher == "hybridep":
        assert isinstance(model.layer.mlp.token_dispatcher, MoEFlexTokenDispatcher)
        assert (
            model.layer.mlp.token_dispatcher.config.moe_flex_dispatcher_backend
            == "hybridep"
        )
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
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    PackedSeqParams,
    PackedSeqParams,
]:
    logical_tokens = SEQUENCE_LENGTH - 2 * case.context_parallel_size
    assert logical_tokens % (2 * case.context_parallel_size) == 0
    assert logical_tokens % case.tensor_parallel_size == 0
    global_route_inputs = []
    for route_offset in (0, case.num_experts - case.router_topk):
        hidden_states = torch.full(
            (logical_tokens, MICRO_BATCH_SIZE, HIDDEN_SIZE),
            -0.25,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        for token in range(logical_tokens):
            for choice in range(case.router_topk):
                expert = (route_offset + choice + token) % case.num_experts
                hidden_states[token, 0, expert] = 2.0 + choice / 8.0
        global_route_inputs.append(hidden_states)

    cp_rank = torch.distributed.get_rank(group=groups.cp)
    tp_rank = torch.distributed.get_rank(group=groups.tp)
    compact_cp_inputs = tuple(
        _zigzag_split(route_input, cp_rank, case.context_parallel_size)
        for route_input in global_route_inputs
    )
    compact_cp_tokens = compact_cp_inputs[0].shape[0]
    fixed_cp_tokens = SEQUENCE_LENGTH // case.context_parallel_size
    assert compact_cp_tokens <= fixed_cp_tokens
    compact_tp_tokens = compact_cp_tokens // case.tensor_parallel_size
    fixed_tp_tokens = fixed_cp_tokens // case.tensor_parallel_size
    compact_route_inputs = tuple(
        route_input.narrow(
            0, tp_rank * compact_tp_tokens, compact_tp_tokens
        ).contiguous()
        for route_input in compact_cp_inputs
    )
    fixed_cp_inputs = tuple(
        torch.cat(
            (
                route_input,
                route_input.new_zeros(
                    fixed_cp_tokens - compact_cp_tokens,
                    MICRO_BATCH_SIZE,
                    HIDDEN_SIZE,
                ),
            ),
            dim=0,
        )
        for route_input in compact_cp_inputs
    )
    route_inputs = tuple(
        route_input.narrow(0, tp_rank * fixed_tp_tokens, fixed_tp_tokens).contiguous()
        for route_input in fixed_cp_inputs
    )
    compact_padding_mask = torch.zeros(
        (MICRO_BATCH_SIZE, compact_tp_tokens),
        dtype=torch.bool,
        device=torch.cuda.current_device(),
    )
    cu_seqlens = torch.tensor(
        (0, logical_tokens), dtype=torch.int32, device=torch.cuda.current_device()
    )
    compact_packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=logical_tokens,
        max_seqlen_kv=logical_tokens,
        local_cp_size=case.context_parallel_size,
        cp_group=groups.cp,
        total_tokens=compact_tp_tokens,
        tokens_per_sample=compact_tp_tokens,
        seq_aux_loss_sample_ids=torch.zeros(
            compact_tp_tokens,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        ),
        seq_aux_loss_num_samples=torch.ones(
            (), dtype=torch.int64, device=torch.cuda.current_device()
        ),
        seq_aux_loss_max_samples=3,
    )
    cp_dummy_tokens = torch.zeros(
        (MICRO_BATCH_SIZE, compact_cp_tokens),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    _, _, _, _, packed_seq_params, cp_padding_mask = pad_sequence_for_thd(
        tokens=cp_dummy_tokens,
        labels=None,
        loss_mask=None,
        position_ids=None,
        packed_seq_params=compact_packed_seq_params,
        target_len=fixed_cp_tokens,
        max_num_seqs=4,
        context_parallel_size=case.context_parallel_size,
    )
    packed_seq_params.total_tokens = fixed_tp_tokens
    packed_seq_params.tokens_per_sample = fixed_tp_tokens
    packed_seq_params.seq_aux_loss_sample_ids = torch.zeros(
        fixed_tp_tokens, dtype=torch.int64, device=torch.cuda.current_device()
    )
    packed_seq_params.seq_aux_loss_num_samples = torch.ones(
        (), dtype=torch.int64, device=torch.cuda.current_device()
    )
    packed_seq_params.seq_aux_loss_max_samples = 3
    padding_mask = cp_padding_mask.narrow(
        1, tp_rank * fixed_tp_tokens, fixed_tp_tokens
    ).contiguous()
    return (
        (route_inputs[0], route_inputs[1]),
        (compact_route_inputs[0], compact_route_inputs[1]),
        padding_mask,
        compact_padding_mask,
        packed_seq_params,
        compact_packed_seq_params,
    )


def _zigzag_split(tensor: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    if cp_size <= 1:
        return tensor
    chunk_size = tensor.shape[0] // (2 * cp_size)
    first = tensor.narrow(0, cp_rank * chunk_size, chunk_size)
    second_rank = 2 * cp_size - cp_rank - 1
    second = tensor.narrow(0, second_rank * chunk_size, chunk_size)
    return torch.cat((first, second), dim=0)


def _zigzag_merge(chunks: list[torch.Tensor], cp_size: int) -> torch.Tensor:
    if cp_size <= 1:
        return chunks[0]
    half = chunks[0].shape[0] // 2
    parts: list[torch.Tensor | None] = [None] * (2 * cp_size)
    for rank, chunk in enumerate(chunks):
        parts[rank] = chunk[:half]
        parts[2 * cp_size - rank - 1] = chunk[half:]
    assert all(part is not None for part in parts)
    return torch.cat([part for part in parts if part is not None], dim=0)


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


def _abort_all_process_groups() -> None:
    abort_process_group = getattr(
        torch.distributed.distributed_c10d, "_abort_process_group", None
    )
    assert abort_process_group is not None
    abort_process_group()


def _write_failure_trace(global_rank: int, error: BaseException) -> None:
    trace_root = Path(os.environ.get("RUN_LOG_ROOT", "/tmp"))
    trace_dir = trace_root / "mcore-distributed-failure-traces" / os.environ.get(
        "SLURM_JOB_ID", "local"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"rank-{global_rank}.log"
    trace_path.write_text(
        f"rank={global_rank} failed before distributed teardown\n"
        + "".join(traceback.format_exception(error))
    )


def _count_method(owner: object, name: str, counters: dict[str, int], key: str) -> None:
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


def _named_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def _reduced_named_gradients(
    model: torch.nn.Module,
    gradients: dict[str, torch.Tensor],
    groups: ProcessGroupCollection,
) -> dict[str, torch.Tensor]:
    reduced: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        gradient = gradients[name].clone()
        data_group = (
            groups.dp_cp if getattr(parameter, "allreduce", True) else groups.expt_dp
        )
        if data_group.size() > 1:
            torch.distributed.all_reduce(gradient, group=data_group)
        if getattr(parameter, "sequence_parallel", False) and groups.tp.size() > 1:
            torch.distributed.all_reduce(gradient, group=groups.tp)
        reduced[name] = gradient
    return reduced


def _gather_logical_tokens(
    local: torch.Tensor,
    case: _TopologyCase,
    groups: ProcessGroupCollection,
    logical_tokens: int,
) -> torch.Tensor:
    tp_chunks = [torch.empty_like(local) for _ in range(case.tensor_parallel_size)]
    torch.distributed.all_gather(tp_chunks, local.contiguous(), group=groups.tp)
    cp_local = torch.cat(tp_chunks, dim=0)
    logical_cp_tokens = logical_tokens // case.context_parallel_size
    cp_local = cp_local[:logical_cp_tokens]
    cp_chunks = [torch.empty_like(cp_local) for _ in range(case.context_parallel_size)]
    torch.distributed.all_gather(cp_chunks, cp_local.contiguous(), group=groups.cp)
    return _zigzag_merge(cp_chunks, case.context_parallel_size)


def _losses(
    local_output: torch.Tensor,
    case: _TopologyCase,
    groups: ProcessGroupCollection,
    logical_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = logical_tokens * HIDDEN_SIZE
    local_loss = local_output.float().square().sum() / denominator
    global_loss = local_loss.detach().clone()
    torch.distributed.all_reduce(global_loss, group=groups.tp_cp)
    return local_loss, global_loss


def _assert_global_route_parity(
    actual: _RouteSnapshot,
    expected: _RouteSnapshot,
    case: _TopologyCase,
    groups: ProcessGroupCollection,
    logical_tokens: int,
) -> torch.Tensor:
    actual_probabilities = _gather_logical_tokens(
        actual.probabilities, case, groups, logical_tokens
    )
    expected_probabilities = _gather_logical_tokens(
        expected.probabilities, case, groups, logical_tokens
    )
    actual_expert_ids = _gather_logical_tokens(
        actual.expert_ids, case, groups, logical_tokens
    )
    expected_expert_ids = _gather_logical_tokens(
        expected.expert_ids, case, groups, logical_tokens
    )
    _assert_tensor_parity(actual_probabilities, expected_probabilities)
    assert torch.equal(actual_expert_ids, expected_expert_ids)
    assert torch.equal(
        actual_expert_ids.ge(0).sum(dim=0), expected_expert_ids.ge(0).sum(dim=0)
    )
    return actual_expert_ids


def _assert_structural_padding_is_zero(
    snapshot: _RouteSnapshot, valid_tokens: torch.Tensor
) -> None:
    padding_tokens = ~valid_tokens
    assert not snapshot.probabilities[padding_tokens].any()
    assert snapshot.expert_ids[padding_tokens].eq(-1).all()


@pytest.mark.internal
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.row_id)
def test_dropless_partial_moe_cuda_graph_distributed(case: _TopologyCase) -> None:
    if os.environ.get(DISABLE_NANO_SHARED_EXPERT_ENV) == "1":
        if case.row_id != "dropless_hybridep_nano16":
            pytest.fail(
                f"{DISABLE_NANO_SHARED_EXPERT_ENV}=1 only supports "
                "dropless_hybridep_nano16",
                pytrace=False,
            )
        case = replace(case, has_shared_expert=False)
    if os.environ.get(NANO_TP1_ENV) == "1":
        if case.row_id != "dropless_hybridep_nano16":
            pytest.fail(
                f"{NANO_TP1_ENV}=1 only supports dropless_hybridep_nano16",
                pytrace=False,
            )
        case = replace(
            case,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

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
    previous_async_error_handling = os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING")
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    helper: TECudaGraphHelper | None = None
    groups: ProcessGroupCollection | None = None
    original_te_general_gemm = moe_utils.te_general_gemm
    original_router_gating_linear = router_module.router_gating_linear
    if os.environ.get(DISABLE_ROUTER_TE_GENERAL_GEMM_ENV) == "1":
        moe_utils.te_general_gemm = None
    if os.environ.get(USE_AUTOGRAD_ROUTER_LINEAR_ENV) == "1":
        router_module.router_gating_linear = _autograd_router_linear
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
        probe_submodule: torch.nn.Module | None = None
        selected_submodule = os.environ.get(NANO_CG_SUBMODULE_ENV)
        if selected_submodule is not None:
            if case.row_id != "dropless_hybridep_nano16":
                pytest.fail(
                    f"{NANO_CG_SUBMODULE_ENV} only supports "
                    "dropless_hybridep_nano16",
                    pytrace=False,
                )
            if selected_submodule == "router":
                submodules = [graph_model.layer.mlp.router]
            elif selected_submodule == "layernorm":
                submodules = [graph_model.layer.pre_mlp_layernorm]
            elif selected_submodule == "linear":
                submodules = [
                    torch.nn.Linear(
                        HIDDEN_SIZE,
                        HIDDEN_SIZE,
                        bias=False,
                        device=torch.device("cuda", torch.cuda.current_device()),
                        dtype=torch.bfloat16,
                    )
                ]
                probe_submodule = submodules[0]
            else:
                pytest.fail(
                    f"{NANO_CG_SUBMODULE_ENV} must be router, layernorm, or linear",
                    pytrace=False,
                )
            graph_model.layer._get_submodules_under_cudagraphs = MethodType(
                lambda _: submodules,
                graph_model.layer,
            )
        graph_model.load_state_dict(eager_model.state_dict())
        _set_deterministic_router(eager_model, case)
        _set_deterministic_router(graph_model, case)
        (
            route_inputs,
            compact_route_inputs,
            padding_mask,
            compact_padding_mask,
            packed_seq_params,
            compact_packed_seq_params,
        ) = _make_packed_inputs(case, groups)

        capacity_tracker = get_moe_capacity_tracker()
        capacity_tracker.initialize(torch.device("cuda", local_rank))
        model_warmups = (
            0 if os.environ.get(SKIP_MODEL_WARMUP_ENV) == "1" else CAPTURE_WARMUPS
        )
        for warmup in range(model_warmups):
            if os.environ.get(LINEAR_MODEL_WARMUP_ENV) == "1":
                assert probe_submodule is not None
                probe_submodule.zero_grad(set_to_none=True)
                probe_input = torch.randn(
                    SEQUENCE_LENGTH // case.context_parallel_size,
                    MICRO_BATCH_SIZE,
                    HIDDEN_SIZE,
                    dtype=torch.bfloat16,
                    device=torch.device("cuda", torch.cuda.current_device()),
                    requires_grad=True,
                )
                output = probe_submodule(probe_input)
            else:
                graph_model.zero_grad(set_to_none=True)
                capacity_tracker.reset()
                output = graph_model(
                    route_inputs[warmup % 2],
                    padding_mask=padding_mask,
                    packed_seq_params=packed_seq_params,
                )
            if os.environ.get(FORWARD_ONLY_MODEL_WARMUP_ENV) != "1":
                output.float().square().mean().backward()
            if os.environ.get(LINEAR_MODEL_WARMUP_ENV) != "1":
                _assert_capacity_is_zero()

        if model_warmups and os.environ.get(RELEASE_WARMUP_GRAPH_ENV) == "1":
            del output
            if os.environ.get(LINEAR_MODEL_WARMUP_ENV) == "1":
                del probe_input
            graph_model.zero_grad(set_to_none=True)
            if probe_submodule is not None:
                probe_submodule.zero_grad(set_to_none=True)
            gc.collect()
        if os.environ.get(RESET_HYBRIDEP_BEFORE_CAPTURE_ENV) == "1":
            torch.cuda.synchronize()
            reset_hybrid_ep_buffer()
            gc.collect()

        if os.environ.get(ZERO_GRAD_BEFORE_CAPTURE_ENV) == "1":
            graph_model.zero_grad(set_to_none=True)

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
        if os.environ.get(CAPTURE_ONLY_ENV) == "1":
            return
        assert helper._compatibility_bank_manager is not None
        manager = helper._compatibility_bank_manager
        counter_start = manager.snapshot_execution_counters()
        states = graph_model.layer._te_cuda_graph_dispatcher_replay_states
        if CudaGraphModule.moe_preprocess in case.cuda_graph_modules:
            assert len(states) == 1
            _assert_replay_geometry(case, states[0], route_inputs[0])
        else:
            assert not states

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

        observed_routes: dict[int, torch.Tensor] = {}
        route_history: list[torch.Tensor] = []
        valid_tokens = ~padding_mask.reshape(-1)
        logical_tokens = SEQUENCE_LENGTH - 2 * case.context_parallel_size
        last_fixed_reduced_gradients: dict[str, torch.Tensor] | None = None
        last_graph_reduced_gradients: dict[str, torch.Tensor] | None = None
        for replay in range(CHANGED_ROUTE_REPLAYS):
            route_index = replay % 2
            compact_hidden_states = (
                compact_route_inputs[route_index].detach().clone().requires_grad_(True)
            )
            fixed_hidden_states = (
                route_inputs[route_index].detach().clone().requires_grad_(True)
            )
            graph_hidden_states = (
                route_inputs[route_index].detach().clone().requires_grad_(True)
            )

            eager_model.zero_grad(set_to_none=True)
            capacity_tracker.reset()
            compact_output = eager_model(
                compact_hidden_states,
                padding_mask=compact_padding_mask,
                packed_seq_params=compact_packed_seq_params,
            )
            compact_routes = _route_snapshot(eager_model)
            compact_flat_output = compact_output.reshape(-1, HIDDEN_SIZE)
            compact_local_loss, compact_global_loss = _losses(
                compact_flat_output, case, groups, logical_tokens
            )
            compact_local_loss.backward()
            _assert_capacity_is_zero()
            compact_gradients = _named_gradients(eager_model)
            assert compact_hidden_states.grad is not None
            compact_input_gradient = compact_hidden_states.grad.detach().clone()

            eager_model.zero_grad(set_to_none=True)
            capacity_tracker.reset()
            fixed_output = eager_model(
                fixed_hidden_states,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
            fixed_routes = _route_snapshot(eager_model)
            fixed_flat_output = fixed_output.reshape(-1, HIDDEN_SIZE)
            fixed_valid_output = fixed_flat_output[valid_tokens]
            fixed_local_loss, fixed_global_loss = _losses(
                fixed_valid_output, case, groups, logical_tokens
            )
            fixed_local_loss.backward()
            _assert_capacity_is_zero()
            fixed_gradients = _named_gradients(eager_model)
            assert fixed_hidden_states.grad is not None
            fixed_input_gradient = fixed_hidden_states.grad.detach().clone()

            graph_model.zero_grad(set_to_none=True)
            capacity_tracker.reset()
            graph_output = graph_model(
                graph_hidden_states,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
            graph_routes = _route_snapshot(graph_model)
            graph_flat_output = graph_output.reshape(-1, HIDDEN_SIZE)
            graph_valid_output = graph_flat_output[valid_tokens]
            graph_local_loss, graph_global_loss = _losses(
                graph_valid_output, case, groups, logical_tokens
            )
            graph_local_loss.backward()
            _assert_capacity_is_zero()
            graph_gradients = _named_gradients(graph_model)
            assert graph_hidden_states.grad is not None
            graph_input_gradient = graph_hidden_states.grad.detach().clone()

            compact_global_output = _gather_logical_tokens(
                compact_flat_output.detach(), case, groups, logical_tokens
            )
            fixed_global_output = _gather_logical_tokens(
                fixed_flat_output.detach(), case, groups, logical_tokens
            )
            graph_global_output = _gather_logical_tokens(
                graph_flat_output.detach(), case, groups, logical_tokens
            )
            _assert_tensor_parity(fixed_global_output, compact_global_output)
            _assert_tensor_parity(fixed_global_loss, compact_global_loss)
            _assert_global_route_parity(
                fixed_routes, compact_routes, case, groups, logical_tokens
            )
            if case.dispatcher == "hybridep":
                _assert_structural_padding_is_zero(fixed_routes, valid_tokens)
            assert not fixed_flat_output[~valid_tokens].any()
            compact_global_input_gradient = _gather_logical_tokens(
                compact_input_gradient.reshape(-1, HIDDEN_SIZE),
                case,
                groups,
                logical_tokens,
            )
            fixed_global_input_gradient = _gather_logical_tokens(
                fixed_input_gradient.reshape(-1, HIDDEN_SIZE),
                case,
                groups,
                logical_tokens,
            )
            _assert_tensor_parity(
                fixed_global_input_gradient, compact_global_input_gradient
            )
            assert not fixed_input_gradient.reshape(-1, HIDDEN_SIZE)[
                ~valid_tokens
            ].any()

            _assert_tensor_parity(graph_global_output, fixed_global_output)
            _assert_tensor_parity(graph_global_loss, fixed_global_loss)
            graph_global_routes = _assert_global_route_parity(
                graph_routes, fixed_routes, case, groups, logical_tokens
            )
            assert graph_routes.probabilities.dtype == torch.float32
            if case.dispatcher == "hybridep":
                _assert_structural_padding_is_zero(graph_routes, valid_tokens)
            assert not graph_flat_output[~valid_tokens].any()
            graph_global_input_gradient = _gather_logical_tokens(
                graph_input_gradient.reshape(-1, HIDDEN_SIZE),
                case,
                groups,
                logical_tokens,
            )
            _assert_tensor_parity(
                graph_global_input_gradient, fixed_global_input_gradient
            )
            assert not graph_input_gradient.reshape(-1, HIDDEN_SIZE)[
                ~valid_tokens
            ].any()
            if states:
                graph_model.layer.mlp.token_dispatcher.validate_cudagraph_continuation(
                    states[0], graph_output
                )
            observed_routes.setdefault(route_index, graph_global_routes)
            route_history.append(graph_global_routes)

            for name in fixed_gradients:
                _assert_tensor_parity(graph_gradients[name], fixed_gradients[name])
            compact_reduced_gradients = _reduced_named_gradients(
                eager_model, compact_gradients, groups
            )
            fixed_reduced_gradients = _reduced_named_gradients(
                eager_model, fixed_gradients, groups
            )
            graph_reduced_gradients = _reduced_named_gradients(
                graph_model, graph_gradients, groups
            )
            for name in fixed_gradients:
                _assert_tensor_parity(
                    fixed_reduced_gradients[name], compact_reduced_gradients[name]
                )
                _assert_tensor_parity(
                    graph_reduced_gradients[name], fixed_reduced_gradients[name]
                )
            last_fixed_reduced_gradients = fixed_reduced_gradients
            last_graph_reduced_gradients = graph_reduced_gradients

        assert not torch.equal(observed_routes[0], observed_routes[1])
        assert torch.equal(route_history[0], route_history[2])
        assert last_fixed_reduced_gradients is not None
        assert last_graph_reduced_gradients is not None
        eager_parameters = dict(eager_model.named_parameters())
        graph_parameters = dict(graph_model.named_parameters())
        assert eager_parameters.keys() == graph_parameters.keys()
        with torch.no_grad():
            for name in eager_parameters:
                eager_parameter = eager_parameters[name]
                graph_parameter = graph_parameters[name]
                eager_delta = -0.01 * last_fixed_reduced_gradients[name].float()
                graph_delta = -0.01 * last_graph_reduced_gradients[name].float()
                _assert_tensor_parity(graph_delta, eager_delta)
                eager_parameter.add_(eager_delta.to(eager_parameter.dtype))
                graph_parameter.add_(graph_delta.to(graph_parameter.dtype))
                _assert_tensor_parity(graph_parameter, eager_parameter)
        counter_delta = manager.execution_counter_delta(counter_start)
        assert counter_delta.eligible_calls == CHANGED_ROUTE_REPLAYS
        assert counter_delta.graph_calls == CHANGED_ROUTE_REPLAYS
        assert counter_delta.graph_calls > 0
        assert counters == {
            "router": 0,
            "preprocess": (
                0
                if CudaGraphModule.moe_preprocess in case.cuda_graph_modules
                else CHANGED_ROUTE_REPLAYS
            ),
            "dispatch": CHANGED_ROUTE_REPLAYS,
            "expert": CHANGED_ROUTE_REPLAYS,
            "combine": CHANGED_ROUTE_REPLAYS,
            "postprocess": CHANGED_ROUTE_REPLAYS,
            "token_dispatch": CHANGED_ROUTE_REPLAYS,
            "token_combine": CHANGED_ROUTE_REPLAYS,
        }
    except BaseException as error:
        _write_failure_trace(global_rank, error)
        print(
            f"rank={global_rank} failed before distributed teardown",
            flush=True,
        )
        traceback.print_exception(error)
        raise
    finally:
        moe_utils.te_general_gemm = original_te_general_gemm
        router_module.router_gating_linear = original_router_gating_linear
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()
        destroy_moe_capacity_tracker()
        destroy_num_microbatches_calculator()
        if case.dispatcher == "hybridep":
            reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()
        if torch.distributed.is_initialized():
            _abort_all_process_groups()
        if previous_async_error_handling is None:
            os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)
        else:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                previous_async_error_handling
            )
