# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the fixed-shape packed-THD CUDA Graph input contract."""

from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch

import megatron.core.packed_seq_params as packed_seq
from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.transformer.cuda_graphs import (
    HAVE_TE_GRAPHS,
    TECudaGraphHelper,
    _set_capture_end,
    is_graph_capturing,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP, reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import _add_network_size_args
from tests.unit_tests.test_utilities import Utils


def _make_packed_seq_params() -> packed_seq.PackedSeqParams:
    valid = torch.tensor([0, 3, 5], dtype=torch.int32)
    padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    return packed_seq.PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=valid,
        cu_seqlens_kv=valid.clone(),
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded.clone(),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        pad_between_seqs=True,
        cp_partition_mode="contiguous",
        tokens_per_sample=8,
    )


def _make_transformer_config(**overrides: object) -> TransformerConfig:
    values = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "cuda_graph_impl": "transformer_engine",
        "max_seqlen_per_dp_cp_rank": 128,
        "pad_packed_seq_alignment": "max",
        "thd_max_packed_sequences": 100,
        "thd_tail_padding_policy": "extend_last",
    }
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.internal
def test_moe_layer_aligns_sequence_parallel_padding_mask(monkeypatch):
    """A batch-first mask follows the same sequence shard as its hidden states."""
    layer = object.__new__(MoELayer)
    layer.config = SimpleNamespace(sequence_parallel=True, tensor_model_parallel_size=2)
    padding_mask = torch.tensor([[False, True, False, True, False, True, False, True], [True] * 8])
    hidden_states = torch.empty(4, 2, 16)

    def scatter(mask: torch.Tensor) -> torch.Tensor:
        assert torch.equal(mask, padding_mask.transpose(0, 1))
        return mask[:4]

    monkeypatch.setattr(tensor_parallel, "scatter_to_sequence_parallel_region", scatter)

    assert torch.equal(layer._align_padding_mask(padding_mask, hidden_states), padding_mask[:, :4])


@pytest.mark.internal
def test_moe_layer_rejects_unalignable_padding_mask():
    """A mask that is neither local nor a sequence-parallel global shape is rejected."""
    layer = object.__new__(MoELayer)
    layer.config = SimpleNamespace(sequence_parallel=True, tensor_model_parallel_size=2)

    with pytest.raises(AssertionError, match="cannot be aligned"):
        layer._align_padding_mask(torch.zeros(2, 7, dtype=torch.bool), torch.empty(4, 2, 16))


@pytest.mark.internal
def test_packed_seq_params_round_trip_through_tensor_graph_kwargs():
    """Replay must preserve compact and physical THD metadata without passing a dataclass."""
    original = _make_packed_seq_params()
    tensor_fields = {"cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"}
    kwargs = {"packed_seq_params": original}

    TransformerLayer._decompose_packed_seq_params_to_kwargs(kwargs)

    assert set(kwargs) == tensor_fields
    assert all(value is None or isinstance(value, torch.Tensor) for value in kwargs.values())

    config = _make_transformer_config()
    config.cp_partition_mode = "contiguous"
    layer = object.__new__(TransformerLayer)
    layer.config = config
    layer._reconstruct_packed_seq_params_from_kwargs(kwargs)

    reconstructed = kwargs["packed_seq_params"]
    assert reconstructed.qkv_format == "thd"
    assert reconstructed.pad_between_seqs is True
    assert reconstructed.cp_partition_mode == "contiguous"
    assert reconstructed.max_seqlen_q == 128
    assert reconstructed.max_seqlen_kv == 128
    for field in tensor_fields:
        assert torch.equal(getattr(reconstructed, field), getattr(original, field))


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_static_thd_attention_inputs_use_explicit_bounds_without_scheduler():
    """Explicit THD graph bounds must fix every attention input shape without a scheduler."""
    config = _make_transformer_config(cuda_graph_modules=["attn"])
    assert getattr(config, "sequence_packing_scheduler", None) is None
    layer = object.__new__(TransformerLayer)
    layer.config = config
    layer.self_attention = SimpleNamespace()

    static_inputs = layer.get_layer_static_inputs(seq_length=17, micro_batch_size=3)

    assert static_inputs["hidden_states"].shape == (128, 1, 16)
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        cu_seqlens = static_inputs[name]
        assert cu_seqlens.shape == (101,)
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0] == 0
        assert torch.all(cu_seqlens[1:] == 128)
    assert static_inputs["padding_mask"].shape == (1, 128)
    assert static_inputs["padding_mask"].dtype == torch.bool
    assert not static_inputs["padding_mask"].any()


@pytest.mark.internal
def test_extend_last_preserves_real_boundaries_and_fixes_all_input_shapes():
    """Static padding must not replace compact valid-token boundaries."""
    original = _make_packed_seq_params()
    original_valid = original.cu_seqlens_q.clone()
    original_padded = original.cu_seqlens_q_padded.clone()
    tokens = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    padding_mask = torch.tensor(
        [[False, False, False, True, False, False, True, True]], dtype=torch.bool
    )

    padded_tokens, _, _, _, padded, padded_mask = packed_seq.pad_sequence_for_thd(
        tokens,
        None,
        None,
        None,
        original,
        target_len=10,
        max_num_seqs=4,
        tail_padding_policy="extend_last",
        padding_mask=padding_mask,
        cp_size=1,
        cp_rank=0,
    )

    assert padded_tokens.shape == (1, 10)
    assert padded_mask.shape == (1, 10)
    assert padded_tokens[0, 8:].tolist() == [0, 0]
    assert padded_mask.tolist() == [
        [False, False, False, True, False, False, True, True, True, True]
    ]
    assert padded.cu_seqlens_q.tolist() == [0, 3, 5, 5, 5]
    assert padded.cu_seqlens_kv.tolist() == [0, 3, 5, 5, 5]
    assert padded.cu_seqlens_q_padded.tolist() == [0, 4, 10, 10, 10]
    assert padded.cu_seqlens_kv_padded.tolist() == [0, 4, 10, 10, 10]
    assert padded.pad_between_seqs is True
    assert padded.cp_partition_mode == "contiguous"
    assert padded.tokens_per_sample == 8
    assert torch.equal(original.cu_seqlens_q, original_valid)
    assert torch.equal(original.cu_seqlens_q_padded, original_padded)


@pytest.mark.internal
def test_append_dummy_preserves_physical_gaps_at_real_sequence_capacity():
    """The sequence bound counts real sequences and reserves a dummy-tail slot."""
    original = _make_packed_seq_params()

    _, _, _, _, padded, _ = packed_seq.pad_sequence_for_thd(
        torch.ones((1, 8), dtype=torch.int64),
        None,
        None,
        None,
        original,
        target_len=12,
        max_num_seqs=2,
        tail_padding_policy="append_dummy_seq",
        cp_size=1,
    )

    assert padded.cu_seqlens_q.tolist() == [0, 3, 5, 9]
    assert padded.cu_seqlens_kv.tolist() == [0, 3, 5, 9]
    assert padded.cu_seqlens_q_padded.tolist() == [0, 4, 8, 12]
    assert padded.cu_seqlens_kv_padded.tolist() == [0, 4, 8, 12]


@pytest.mark.internal
@pytest.mark.parametrize(
    "tail_padding_policy,target_len,expected_valid,expected_padded,expected_max",
    [
        ("append_dummy_seq", 14, [0, 3, 5, 11], [0, 4, 8, 14], 6),
        ("extend_last", 10, [0, 3, 5], [0, 4, 10], 6),
    ],
)
def test_tail_padding_recomputes_maxima_without_sequence_bound(
    tail_padding_policy: Literal["append_dummy_seq", "extend_last"],
    target_len: int,
    expected_valid: list[int],
    expected_padded: list[int],
    expected_max: int,
):
    """THD kernel maxima must cover every resulting physical sequence."""
    _, _, _, _, padded, _ = packed_seq.pad_sequence_for_thd(
        torch.ones((1, 8), dtype=torch.int64),
        None,
        None,
        None,
        _make_packed_seq_params(),
        target_len=target_len,
        tail_padding_policy=tail_padding_policy,
        cp_size=1,
    )

    assert padded.cu_seqlens_q.tolist() == expected_valid
    assert padded.cu_seqlens_kv.tolist() == expected_valid
    assert padded.cu_seqlens_q_padded.tolist() == expected_padded
    assert padded.cu_seqlens_kv_padded.tolist() == expected_padded
    assert padded.max_seqlen_q == expected_max
    assert padded.max_seqlen_kv == expected_max


@pytest.mark.internal
def test_process_group_only_cp_context_rejects_padding(monkeypatch: pytest.MonkeyPatch):
    """A CP process group must not silently fall back to a one-rank context."""
    fake_group = cast(torch.distributed.ProcessGroup, object())
    params = _make_packed_seq_params()
    params.cp_group = fake_group

    def fake_world_size(group: object) -> int:
        assert group is fake_group
        return 2

    monkeypatch.setattr(packed_seq.dist, "get_world_size", fake_world_size)

    with pytest.raises(ValueError, match="before CP slicing"):
        packed_seq.pad_sequence_for_thd(
            torch.ones((1, 8), dtype=torch.int64),
            None,
            None,
            None,
            params,
            target_len=10,
            tail_padding_policy="extend_last",
        )


@pytest.mark.internal
def test_target_sized_padding_mask_still_masks_appended_token_tail():
    """An existing fixed-shape mask must be merged with the explicit token tail."""
    padding_mask = torch.zeros((2, 12), dtype=torch.bool)
    padding_mask[1, 2] = True

    _, _, _, _, _, padded_mask = packed_seq.pad_sequence_for_thd(
        torch.ones((1, 8), dtype=torch.int64),
        None,
        None,
        None,
        _make_packed_seq_params(),
        target_len=12,
        tail_padding_policy="extend_last",
        padding_mask=padding_mask,
        cp_size=1,
    )

    assert padded_mask.shape == (2, 12)
    assert padded_mask.dtype == torch.bool
    expected_mask = torch.zeros((2, 12), dtype=torch.bool)
    expected_mask[1, 2] = True
    expected_mask[:, 8:] = True
    assert torch.equal(padded_mask, expected_mask)


@pytest.mark.internal
def test_static_padding_rejects_observed_token_count_above_configured_bound():
    """A token overflow must fail instead of truncating the packed batch."""
    params = _make_packed_seq_params()
    tokens = torch.ones((1, 9), dtype=torch.int64)

    with pytest.raises(ValueError, match=r"observed token count 9.*configured bound 8"):
        packed_seq.pad_sequence_for_thd(
            tokens,
            None,
            None,
            None,
            params,
            target_len=8,
            max_num_seqs=4,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )


@pytest.mark.internal
def test_static_padding_rejects_observed_sequence_count_above_configured_bound():
    """A sequence-count overflow must fail instead of truncating cu_seqlens."""
    cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    params = packed_seq.PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens.clone(),
        cu_seqlens_q_padded=cu_seqlens.clone(),
        cu_seqlens_kv_padded=cu_seqlens.clone(),
        max_seqlen_q=2,
        max_seqlen_kv=2,
    )

    with pytest.raises(ValueError, match=r"observed sequence count 3.*configured bound 2"):
        packed_seq.pad_sequence_for_thd(
            torch.ones((1, 6), dtype=torch.int64),
            None,
            None,
            None,
            params,
            target_len=8,
            max_num_seqs=2,
            tail_padding_policy="extend_last",
            cp_size=1,
            cp_rank=0,
        )


@pytest.mark.internal
def test_static_thd_arguments_parse_exact_profile_values():
    """The reusable profile's four THD graph options must be typed CLI arguments."""
    parser = ArgumentParser()
    _add_network_size_args(parser)

    args = parser.parse_args(
        [
            "--pad-packed-seq-alignment",
            "max",
            "--thd-max-packed-sequences",
            "100",
            "--thd-tail-padding-policy",
            "extend_last",
            "--cuda-graph-memory-report",
        ]
    )

    assert args.pad_packed_seq_alignment == "max"
    assert args.thd_max_packed_sequences == 100
    assert args.thd_tail_padding_policy == "extend_last"
    assert args.cuda_graph_memory_report is True


@pytest.mark.internal
def test_complete_static_thd_graph_configuration_is_accepted():
    """Explicit fixed token and sequence capacities form a valid graph contract."""
    config = _make_transformer_config(cuda_graph_memory_report=True)

    assert config.max_seqlen_per_dp_cp_rank == 128
    assert config.pad_packed_seq_alignment == "max"
    assert config.thd_max_packed_sequences == 100
    assert config.thd_tail_padding_policy == "extend_last"
    assert config.cuda_graph_memory_report is True


@pytest.mark.internal
@pytest.mark.parametrize(
    "missing,overrides",
    [
        ("--max-seqlen-per-dp-cp-rank", {"max_seqlen_per_dp_cp_rank": None}),
        ("--pad-packed-seq-alignment", {"pad_packed_seq_alignment": None}),
        ("--thd-max-packed-sequences", {"thd_max_packed_sequences": None}),
    ],
)
def test_static_thd_graph_configuration_rejects_missing_fixed_bound(
    missing: str, overrides: dict[str, object]
):
    """A partial static contract would permit replay with changing input shapes."""
    with pytest.raises(ValueError, match=missing):
        _make_transformer_config(**overrides)


@pytest.mark.internal
def test_non_thd_graph_configuration_does_not_require_static_thd_bounds():
    """Existing graph users without explicit THD fields remain unaffected."""
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=4, cuda_graph_impl="transformer_engine"
    )

    assert config.cuda_graph_impl == "transformer_engine"


@pytest.mark.internal
def test_moe_preprocess_graph_scope_requires_moe_router() -> None:
    """A preprocess-only graph cannot resume the ungraphed router boundary."""
    with pytest.raises(AssertionError, match="only supported with moe_router"):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=8,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.moe_preprocess],
        )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_HYBRIDEP, reason="HybridEP is not available")
@pytest.mark.skipif(
    not (HAVE_TE_GRAPHS and is_te_min_version("2.10.0")),
    reason="HybridEP partial CUDA graph coverage requires TransformerEngine >= 2.10.0",
)
@pytest.mark.parametrize(
    "cuda_graph_modules",
    [[CudaGraphModule.moe_router], [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess]],
    ids=["moe_router", "moe_router_and_preprocess"],
)
@pytest.mark.parametrize("has_padding", [False, True], ids=["unpadded_control", "padded_routes"])
def test_dropless_hybridep_router_graph_boundary_preserves_padded_routes_and_gradients(
    cuda_graph_modules: list[CudaGraphModule], has_padding: bool
) -> None:
    """TE partial capture must preserve dropless HybridEP routing across padded THD rows.

    The break this catches is a router boundary that leaves padded rows selected when a
    captured HybridEP dispatch consumes its sparse routing metadata.
    """
    cuda_graph_helper = None
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1
    )
    init_num_microbatches_calculator(
        rank=Utils.rank,
        global_batch_size=Utils.world_size,
        micro_batch_size=1,
        data_parallel_size=Utils.world_size,
        decrease_batch_size_if_needed=False,
    )
    try:
        num_experts = 4
        config = TransformerConfig(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.1,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_hybridep_pad_uneven_dispatch_inputs=True,
            use_cpu_initialization=True,
            bf16=True,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=cuda_graph_modules,
            max_seqlen_per_dp_cp_rank=8,
            pad_packed_seq_alignment="max",
            thd_max_packed_sequences=1,
            thd_tail_padding_policy="extend_last",
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(
                num_experts=num_experts, moe_grouped_gemm=False
            ),
            vocab_size=16,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            position_embedding_type="none",
        )
        model = model.cuda().to(dtype=torch.bfloat16)
        model.train()

        def zero_grad_buffer() -> None:
            for parameter in model.parameters():
                parameter.grad = None

        setattr(model, "zero_grad_buffer", zero_grad_buffer)
        layer = model.decoder.layers[0]
        moe_layer = layer.mlp

        torch.manual_seed(1234 + Utils.rank)
        base_hidden = torch.randn(8, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        padding_mask = torch.tensor(
            (
                [[False], [False], [True], [False], [True], [False], [True], [True]]
                if has_padding
                else [[False]] * 8
            ),
            device="cuda",
        ).transpose(0, 1)
        padded_rows = padding_mask.transpose(0, 1).reshape(-1)

        static_hidden = base_hidden.detach().clone().requires_grad_()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        metrics_tracker = get_moe_metrics_tracker()
        metrics_tracker.clear()
        cuda_graph_helper = TECudaGraphHelper(
            model=[model], config=config, seq_length=8, micro_batch_size=1, optimizers=[]
        )
        cuda_graph_helper.create_cudagraphs()
        assert cuda_graph_helper.graphs_created()
        captured_aux_loss = metrics_tracker.metrics["load_balancing_loss"].values.detach().clone()
        metrics_tracker.clear()

        captured_outputs = layer.cuda_graphs[0](static_hidden, padding_mask=padding_mask)
        captured_probs = captured_outputs[1]
        if CudaGraphModule.moe_preprocess in cuda_graph_modules:
            captured_attrs = dict(
                zip(moe_layer.token_dispatcher.valid_cudagraph_attrs, captured_outputs[2:-1])
            )
            captured_routing_map = captured_attrs["_comm_manager.routing_map"]
        else:
            captured_routing_map = captured_outputs[2]
        captured_probs.sum().backward()
        torch.cuda.synchronize()

        captured_probs = captured_probs.detach().clone()
        captured_routing_map = captured_routing_map.detach().clone()
        captured_input_grad = static_hidden.grad.detach().clone()
        captured_router_grad = moe_layer.router.weight.grad.detach().clone()

        model.zero_grad(set_to_none=True)
        eager_hidden = base_hidden.detach().clone().requires_grad_()
        eager_pre_mlp_hidden = layer._forward_pre_mlp_layernorm(eager_hidden)
        eager_probs, eager_routing_map = moe_layer.route(
            eager_pre_mlp_hidden, padding_mask.transpose(0, 1)
        )
        if CudaGraphModule.moe_preprocess in cuda_graph_modules:
            _, eager_probs = moe_layer.preprocess(
                eager_pre_mlp_hidden, eager_probs, eager_routing_map
            )
            eager_routing_map = moe_layer.token_dispatcher._comm_manager.routing_map
        eager_input_grad, eager_router_grad = torch.autograd.grad(
            eager_probs.sum(), (eager_hidden, moe_layer.router.weight)
        )
        eager_aux_loss = metrics_tracker.metrics["load_balancing_loss"].values.detach().clone()

        torch.testing.assert_close(captured_probs, eager_probs)
        assert torch.equal(captured_routing_map, eager_routing_map)
        torch.testing.assert_close(captured_aux_loss, eager_aux_loss)
        original_tokens = padded_rows.numel()
        assert torch.count_nonzero(captured_probs[:original_tokens][padded_rows]) == 0
        assert not captured_routing_map[:original_tokens][padded_rows].any()
        assert torch.count_nonzero(captured_probs[original_tokens:]) == 0
        assert not captured_routing_map[original_tokens:].any()
        torch.testing.assert_close(captured_input_grad, eager_input_grad)
        torch.testing.assert_close(captured_router_grad, eager_router_grad)
    finally:
        if cuda_graph_helper is not None and cuda_graph_helper.graphs_created():
            cuda_graph_helper.delete_cuda_graphs()
        reset_hybrid_ep_buffer()
        get_moe_metrics_tracker().clear()
        Utils.destroy_model_parallel()
        destroy_num_microbatches_calculator()
        _set_capture_end()
        assert not is_graph_capturing()
