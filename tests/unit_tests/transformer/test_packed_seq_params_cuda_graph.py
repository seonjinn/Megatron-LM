# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import MethodType, SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import (
    CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS,
    PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS,
    PackedSeqParams,
    build_packed_seq_params_from_cuda_graph_kwargs,
    has_packed_seq_params_cuda_graph_kwargs,
    split_moe_packed_seq_params_for_cuda_graph,
    split_packed_seq_params_for_cuda_graph,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import (
    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs,
)
from megatron.core.transformer.enums import AttnMaskType, CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


class _TransformerLayerCudaGraphStub:
    _set_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._set_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._get_te_cuda_graph_packed_seq_params_static_metadata
    )
    _validate_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._validate_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_tensor_kwarg_names = (
        TransformerLayer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names
    )
    _validate_te_cuda_graph_packed_seq_params_tensor_kwargs = (
        TransformerLayer._validate_te_cuda_graph_packed_seq_params_tensor_kwargs
    )
    _rebuild_te_cuda_graph_packed_seq_params = (
        TransformerLayer._rebuild_te_cuda_graph_packed_seq_params
    )
    _flatten_te_cuda_graph_packed_seq_params = (
        TransformerLayer._flatten_te_cuda_graph_packed_seq_params
    )
    _set_te_cuda_graph_moe_packed_seq_params_static_metadata = (
        TransformerLayer._set_te_cuda_graph_moe_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_moe_packed_seq_params_static_metadata = (
        TransformerLayer._get_te_cuda_graph_moe_packed_seq_params_static_metadata
    )
    _validate_te_cuda_graph_moe_packed_seq_params_kwargs = (
        TransformerLayer._validate_te_cuda_graph_moe_packed_seq_params_kwargs
    )
    _rebuild_te_cuda_graph_moe_packed_seq_params = (
        TransformerLayer._rebuild_te_cuda_graph_moe_packed_seq_params
    )
    _flatten_te_cuda_graph_moe_packed_seq_params = (
        TransformerLayer._flatten_te_cuda_graph_moe_packed_seq_params
    )
    _te_cuda_graph_owns_moe_packed_seq_params = (
        TransformerLayer._te_cuda_graph_owns_moe_packed_seq_params
    )
    _te_cuda_graph_owns_packed_seq_params = (
        TransformerLayer._te_cuda_graph_owns_packed_seq_params
    )
    _te_cuda_graph_capture = TransformerLayer._te_cuda_graph_capture
    _te_cuda_graph_replay = TransformerLayer._te_cuda_graph_replay


def _make_seq_aux_loss_packed_seq_params(
    *,
    num_tokens: int = 8,
    num_samples: int = 2,
    max_samples: int = 3,
    device: torch.device | str = "cpu",
) -> PackedSeqParams:
    sample_ids = torch.arange(num_tokens, device=device, dtype=torch.int64).remainder(num_samples)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        cu_seqlens_kv=torch.tensor([0, 4, 8], dtype=torch.int32, device=device),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        pad_between_seqs=True,
        seq_aux_loss_sample_ids=sample_ids,
        seq_aux_loss_num_samples=torch.tensor(num_samples, dtype=torch.int64, device=device),
        seq_aux_loss_max_samples=max_samples,
    )


def _install_moe_graph_contract(
    layer: _TransformerLayerCudaGraphStub, packed_seq_params: PackedSeqParams
) -> dict[str, torch.Tensor]:
    tensor_kwargs, static_metadata = split_moe_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    layer._set_te_cuda_graph_moe_packed_seq_params_static_metadata(
        static_metadata, tensor_kwargs
    )
    return tensor_kwargs


def _make_mlp_propagation_layer(mlp: torch.nn.Module, *, is_moe: bool) -> TransformerLayer:
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        fp32_residual_connection=False,
        mlp_chunks_for_prefill=1,
        mlp_chunks_for_training=4,
        transformer_impl="transformer_engine",
        inference_fuse_tp_communication=False,
        cuda_graph_impl="none",
    )
    layer.is_moe_layer = is_moe
    layer.mlp = mlp
    layer.recompute_mlp = False
    layer._forward_pre_mlp_layernorm = MethodType(lambda self, hidden: hidden, layer)
    layer._forward_post_mlp = MethodType(lambda self, output, _residual: output[0], layer)
    layer.train()
    return layer


def test_moe_sample_ownership_disables_training_chunks_and_reaches_eager_mlp(
    monkeypatch,
) -> None:
    calls = []

    class _MoEObserver(torch.nn.Module):
        def forward(self, hidden_states, padding_mask=None, packed_seq_params=None):
            calls.append((hidden_states, padding_mask, packed_seq_params))
            return hidden_states, None

    layer = _make_mlp_propagation_layer(_MoEObserver(), is_moe=True)
    packed = _make_seq_aux_loss_packed_seq_params()
    hidden_states = torch.zeros((8, 1, 4))
    padding_mask = torch.ones((1, 8), dtype=torch.bool)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.nvtx_range_push", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.nvtx_range_pop", lambda **_kwargs: None
    )

    layer._forward_mlp(
        hidden_states,
        padding_mask=padding_mask,
        packed_seq_params=packed,
    )

    assert len(calls) == 1
    assert calls[0][0] is hidden_states
    assert calls[0][1] is padding_mask
    assert calls[0][2] is packed


def test_recomputed_moe_mlp_receives_original_sample_ownership(monkeypatch) -> None:
    observed = {}

    class _MoEObserver(torch.nn.Module):
        def forward(self, hidden_states, padding_mask=None, packed_seq_params=None):
            observed["packed_seq_params"] = packed_seq_params
            observed["padding_mask"] = padding_mask
            return hidden_states, None

    layer = _make_mlp_propagation_layer(_MoEObserver(), is_moe=True)
    layer.recompute_mlp = True
    layer.config.fp8 = None
    layer.config.fp4 = None
    packed = _make_seq_aux_loss_packed_seq_params()
    padding_mask = torch.ones((1, 8), dtype=torch.bool)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.nvtx_range_push", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.nvtx_range_pop", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.tensor_parallel.checkpoint",
        lambda function, _distribute_saved_activations, *args: function(*args),
    )

    layer._forward_mlp(
        torch.zeros((8, 1, 4)),
        padding_mask=padding_mask,
        packed_seq_params=packed,
    )

    assert observed["packed_seq_params"] is packed
    assert observed["padding_mask"] is padding_mask


def test_dense_mlp_still_chunks_without_receiving_moe_packed_seq_params(monkeypatch) -> None:
    calls = []

    class _DenseObserver(torch.nn.Module):
        def forward(self, hidden_states, padding_mask=None):
            calls.append((hidden_states, padding_mask))
            return hidden_states, None

    layer = _make_mlp_propagation_layer(_DenseObserver(), is_moe=False)
    hidden_states = torch.zeros((8, 1, 4))
    padding_mask = torch.ones((1, 8), dtype=torch.bool)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.nvtx_range_push", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_layer.nvtx_range_pop", lambda **_kwargs: None
    )

    layer._forward_mlp(
        hidden_states,
        padding_mask=padding_mask,
        packed_seq_params=_make_seq_aux_loss_packed_seq_params(),
    )

    assert len(calls) == 4
    assert all(call_hidden.shape == (2, 1, 4) for call_hidden, _ in calls)
    assert all(call_padding_mask is None for _, call_padding_mask in calls)


def test_te_attention_does_not_forward_seq_aux_loss_packed_fields(monkeypatch) -> None:
    forwarded_kwargs: dict[str, object] = {}
    te_attention_base = TEDotProductAttention.__mro__[1]

    monkeypatch.setenv("NVTE_APPLY_QK_LAYER_SCALING", "0")
    monkeypatch.setattr(te_attention_base, "__init__", lambda self, **kwargs: None)

    def capture_forward(self, query, key, value, attention_mask, **kwargs):
        forwarded_kwargs.update(kwargs)
        return query

    monkeypatch.setattr(te_attention_base, "forward", capture_forward)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
    )
    attention = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=ProcessGroupCollection(tp=None, cp=None),
    )
    query = torch.ones(8, 1, 2, 4)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 8], dtype=torch.int32),
        seq_aux_loss_sample_ids=torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int64
        ),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )

    result = attention.forward(
        query,
        query,
        query,
        attention_mask=None,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=packed_seq_params,
    )

    assert result is query
    assert "seq_aux_loss_sample_ids" not in forwarded_kwargs
    assert "seq_aux_loss_num_samples" not in forwarded_kwargs
    assert "seq_aux_loss_max_samples" not in forwarded_kwargs


def test_transformer_layer_thd_static_inputs_include_local_padding_mask(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        context_parallel_size=2,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
        hidden_size=8,
        cuda_graph_modules=[],
        thd_max_packed_sequences=7,
    )
    layer.self_attention = IdentityOp()

    static_inputs = layer.get_layer_static_inputs(seq_length=16, micro_batch_size=3)

    padding_mask = static_inputs["padding_mask"]
    assert padding_mask.shape == (3, 4)
    assert padding_mask.dtype == torch.bool
    assert not padding_mask.any()


def test_attention_only_replay_passes_packed_metadata_and_padding_mask_to_mlp(monkeypatch) -> None:
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.attn],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
    )
    layer.is_moe_layer = True
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 4, 8]),
        cu_seqlens_kv=torch.IntTensor([0, 4, 8]),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        tokens_per_sample=4,
        pad_between_seqs=True,
        seq_aux_loss_sample_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    padding_mask = torch.tensor([[False, False, False, True, False, False, True, True]])
    hidden_states = torch.ones(8, 1, 4)
    observed = {}
    graph_kwargs = {}

    def forward_mlp(self, hidden_states, padding_mask=None, packed_seq_params=None):
        observed["hidden_states"] = hidden_states
        observed["padding_mask"] = padding_mask
        observed["packed_seq_params"] = packed_seq_params
        return hidden_states

    layer._forward_mlp = MethodType(forward_mlp, layer)
    layer._te_cuda_graph_replay_index = MethodType(lambda self, _microbatch: 0, layer)

    def replay_graph(self, *args, **kwargs):
        graph_kwargs.update(kwargs)
        return (hidden_states,)

    monkeypatch.setattr(
        GraphableMegatronModule,
        "_te_cuda_graph_replay",
        replay_graph,
    )

    layer._te_cuda_graph_replay(
        hidden_states,
        padding_mask=padding_mask,
        packed_seq_params=packed_seq_params,
    )

    assert observed["hidden_states"] is hidden_states
    assert observed["padding_mask"] is padding_mask
    assert observed["packed_seq_params"] is packed_seq_params
    assert observed["packed_seq_params"].seq_aux_loss_sample_ids is (
        packed_seq_params.seq_aux_loss_sample_ids
    )
    assert "packed_seq_params" not in graph_kwargs
    assert any(key.startswith(CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX) for key in graph_kwargs)
    assert not any(key.startswith(MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX) for key in graph_kwargs)


def test_non_attention_moe_replay_preserves_padding_and_graph_owned_sample_inputs() -> None:
    captured = _make_seq_aux_loss_packed_seq_params()
    captured.tokens_per_sample = 4

    class _RouterReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=[CudaGraphModule.moe_router],
                delay_offload_until_cuda_graph=False,
            )
            self.self_attention = IdentityOp()
            self.is_moe_layer = True
            self.attention_inputs = None
            self.replay_inputs = None
            self.graph_calls = 0
            self.fallback_calls = 0

        def _forward_attention(self, hidden_states, **kwargs):
            self.attention_inputs = kwargs
            return hidden_states + 1, "context"

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            eager_mlp_kwargs = dict(kwargs)
            self._rebuild_te_cuda_graph_moe_packed_seq_params(eager_mlp_kwargs)
            self.replay_inputs = (args, kwargs, context, eager_packed_seq_params)
            self.rebuilt_packed_seq_params = eager_mlp_kwargs["packed_seq_params"]
            return args[0]

    layer = _RouterReplayLayer()
    _install_moe_graph_contract(layer, captured)
    replay = _make_seq_aux_loss_packed_seq_params(num_samples=1)
    replay.tokens_per_sample = 4
    replay.seq_aux_loss_sample_ids.zero_()
    padding_mask = torch.ones((1, 8), dtype=torch.bool)
    hidden_states = torch.zeros((8, 1, 4))

    layer._te_cuda_graph_replay(
        hidden_states, packed_seq_params=replay, padding_mask=padding_mask
    )

    assert layer.attention_inputs["packed_seq_params"] is replay
    args, graph_kwargs, context, eager_packed_seq_params = layer.replay_inputs
    assert torch.equal(args[0], hidden_states + 1)
    assert context == "context"
    assert eager_packed_seq_params is None
    assert graph_kwargs["padding_mask"] is padding_mask
    assert graph_kwargs[
        f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_sample_ids"
    ] is replay.seq_aux_loss_sample_ids
    assert graph_kwargs[
        f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_num_samples"
    ] is replay.seq_aux_loss_num_samples
    assert replay.seq_aux_loss_num_samples.item() == 1
    assert layer._get_te_cuda_graph_moe_packed_seq_params_static_metadata()[
        "tokens_per_sample"
    ] == 4
    assert layer.rebuilt_packed_seq_params.tokens_per_sample == 4
    assert layer.rebuilt_packed_seq_params.seq_aux_loss_sample_ids is (
        replay.seq_aux_loss_sample_ids
    )
    assert not replay.seq_aux_loss_sample_ids.any()
    assert graph_kwargs["padding_mask"].all()
    assert "packed_seq_params" not in graph_kwargs
    assert layer.graph_calls == 1
    assert layer.fallback_calls == 0


def test_identity_attention_moe_rebuild_preserves_captured_tokens_per_sample() -> None:
    captured = _make_seq_aux_loss_packed_seq_params()
    captured.tokens_per_sample = 4
    replay = _make_seq_aux_loss_packed_seq_params(num_samples=1)
    replay.tokens_per_sample = 4
    replay.seq_aux_loss_sample_ids.zero_()

    class _CaptureLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(cuda_graph_modules=[CudaGraphModule.moe_router])
            self.offload_module_in_cuda_graph = False
            self.self_attention = IdentityOp()
            self.is_moe_layer = True
            self.mlp_kwargs = None

        def _forward_mlp(self, hidden_states, padding_mask=None, packed_seq_params=None):
            self.mlp_kwargs = packed_seq_params
            return hidden_states

    layer = _CaptureLayer()
    _install_moe_graph_contract(layer, captured)
    moe_tensor_kwargs, _ = split_moe_packed_seq_params_for_cuda_graph(replay)

    layer._te_cuda_graph_capture(torch.ones(8, 1, 4), **moe_tensor_kwargs)

    assert layer.mlp_kwargs.tokens_per_sample == 4
    assert layer.mlp_kwargs.seq_aux_loss_sample_ids is replay.seq_aux_loss_sample_ids


@pytest.mark.parametrize(
    "cuda_graph_modules",
    (
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ),
)
def test_identity_attention_moe_replay_rejects_uncaptured_tokens_per_sample(
    cuda_graph_modules: list[CudaGraphModule],
) -> None:
    class _RejectingReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=cuda_graph_modules,
                delay_offload_until_cuda_graph=False,
            )
            self.self_attention = IdentityOp()
            self.is_moe_layer = True
            self.attention_calls = 0
            self.graph_calls = 0

        def _forward_attention(self, hidden_states, **_kwargs):
            self.attention_calls += 1
            return hidden_states, None

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            return args[0]

    layer = _RejectingReplayLayer()

    with pytest.raises(ValueError, match="captured without MoE.*ownership"):
        layer._te_cuda_graph_replay(
            torch.zeros((8, 1, 4)),
            packed_seq_params=PackedSeqParams(tokens_per_sample=4),
        )

    assert layer.attention_calls == 0
    assert layer.graph_calls == 0


@pytest.mark.parametrize(
    "ownership",
    ["complete", "sample_ids_only", "num_samples_only", "capacity_only", "tokens_per_sample_only"],
)
def test_moe_owning_replay_rejects_ownership_without_captured_contract_before_execution(
    ownership: str,
) -> None:
    class _RejectingReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=[CudaGraphModule.moe_router],
                delay_offload_until_cuda_graph=False,
            )
            self.is_moe_layer = True
            self.attention_calls = 0
            self.graph_calls = 0

        def _forward_attention(self, hidden_states, **_kwargs):
            self.attention_calls += 1
            return hidden_states, None

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            return args[0]

    layer = _RejectingReplayLayer()
    if ownership == "complete":
        replay = _make_seq_aux_loss_packed_seq_params()
    elif ownership == "sample_ids_only":
        replay = PackedSeqParams(seq_aux_loss_sample_ids=torch.zeros(8, dtype=torch.int64))
    elif ownership == "num_samples_only":
        replay = PackedSeqParams(seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.int64))
    elif ownership == "capacity_only":
        replay = PackedSeqParams(seq_aux_loss_max_samples=4)
    else:
        replay = PackedSeqParams(tokens_per_sample=4)

    with pytest.raises(ValueError, match="captured without MoE.*ownership"):
        layer._te_cuda_graph_replay(
            torch.zeros((8, 1, 4)),
            packed_seq_params=replay,
            padding_mask=torch.ones((1, 8), dtype=torch.bool),
        )

    assert layer.attention_calls == 0
    assert layer.graph_calls == 0


def test_moe_owning_replay_without_contract_accepts_absent_ownership() -> None:
    class _OrdinaryReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=[CudaGraphModule.moe_router],
                delay_offload_until_cuda_graph=False,
            )
            self.is_moe_layer = True
            self.attention_calls = 0
            self.graph_calls = 0

        def _forward_attention(self, hidden_states, **_kwargs):
            self.attention_calls += 1
            return hidden_states, None

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            return args[0]

    layer = _OrdinaryReplayLayer()
    hidden_states = torch.zeros((8, 1, 4))

    output = layer._te_cuda_graph_replay(
        hidden_states,
        packed_seq_params=PackedSeqParams(),
    )

    assert output is hidden_states
    assert layer.attention_calls == 1
    assert layer.graph_calls == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("device", "device"),
        ("layout", "layout"),
        ("stride", "stride"),
        ("type", "must be a Tensor"),
        ("presence", "missing"),
        ("capacity", "seq_aux_loss_max_samples"),
    ],
)
def test_moe_replay_rejects_signature_drift_before_eager_attention(
    mutation: str, match: str
) -> None:
    captured = _make_seq_aux_loss_packed_seq_params()

    class _RejectingReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=[CudaGraphModule.moe_router],
                delay_offload_until_cuda_graph=False,
            )
            self.is_moe_layer = True
            self.attention_calls = 0
            self.graph_calls = 0

        def _forward_attention(self, hidden_states, **_kwargs):
            self.attention_calls += 1
            return hidden_states, None

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            return args[0]

    layer = _RejectingReplayLayer()
    _install_moe_graph_contract(layer, captured)
    replay = _make_seq_aux_loss_packed_seq_params()
    if mutation == "shape":
        replay.seq_aux_loss_sample_ids = torch.zeros(7, dtype=torch.int64)
    elif mutation == "dtype":
        replay.seq_aux_loss_sample_ids = replay.seq_aux_loss_sample_ids.to(torch.int32)
    elif mutation == "device":
        replay.seq_aux_loss_sample_ids = torch.empty(8, dtype=torch.int64, device="meta")
    elif mutation == "layout":
        replay.seq_aux_loss_sample_ids = torch.sparse_coo_tensor(
            torch.tensor([[0, 3]]), torch.tensor([0, 1]), (8,), dtype=torch.int64
        )
    elif mutation == "stride":
        replay.seq_aux_loss_sample_ids = torch.empty_strided((8,), (2,), dtype=torch.int64)
    elif mutation == "type":
        replay.seq_aux_loss_sample_ids = object()
    elif mutation == "presence":
        replay.seq_aux_loss_num_samples = None
    else:
        replay.seq_aux_loss_max_samples = 4

    with pytest.raises((AssertionError, ValueError), match=match):
        layer._te_cuda_graph_replay(
            torch.zeros((8, 1, 4)),
            packed_seq_params=replay,
            padding_mask=torch.zeros((1, 8), dtype=torch.bool),
        )

    assert layer.attention_calls == 0
    assert layer.graph_calls == 0


def test_moe_replay_rejects_unknown_prefixed_key_before_eager_attention() -> None:
    captured = _make_seq_aux_loss_packed_seq_params()

    class _RejectingReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=[CudaGraphModule.moe_router],
                delay_offload_until_cuda_graph=False,
            )
            self.is_moe_layer = True
            self.attention_calls = 0

        def _forward_attention(self, hidden_states, **_kwargs):
            self.attention_calls += 1
            return hidden_states, None

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            raise AssertionError("graph callable must not be reached")

    layer = _RejectingReplayLayer()
    _install_moe_graph_contract(layer, captured)

    with pytest.raises(ValueError, match="unexpected"):
        layer._te_cuda_graph_replay(
            torch.zeros((8, 1, 4)),
            packed_seq_params=captured,
            _moe_packed_seq_params_unexpected=torch.zeros((), dtype=torch.int64),
        )

    assert layer.attention_calls == 0


def test_attention_only_replay_overlap_routes_local_packed_batch_and_restores_shape(
    monkeypatch,
) -> None:
    observed = {}

    class _RouterObserver(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            padding_mask=None,
            *,
            seq_aux_loss_sample_ids=None,
            seq_aux_loss_num_samples=None,
            seq_aux_loss_max_samples=None,
        ):
            observed["route_hidden_states"] = hidden_states.clone()
            observed["route_padding_mask"] = padding_mask.clone()
            observed["seq_aux_loss_sample_ids"] = seq_aux_loss_sample_ids
            observed["seq_aux_loss_num_samples"] = seq_aux_loss_num_samples
            observed["seq_aux_loss_max_samples"] = seq_aux_loss_max_samples
            token_ids = hidden_states.reshape(-1)
            probs = torch.stack((token_ids, token_ids + 100), dim=-1)
            routing_map = torch.stack(
                (token_ids.remainder(2) == 0, token_ids.remainder(2) == 1), dim=-1
            )
            return probs, routing_map

    class _TokenDispatcher:
        def __init__(self):
            self.hidden_shape = None

        def combine_postprocess(self, output):
            return output.view(self.hidden_shape)

    class _OverlapMlp(torch.nn.Module):
        route = MoELayer.route
        postprocess = MoELayer.postprocess

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(cuda_graph_impl="none", moe_latent_size=None)
            self.router = _RouterObserver()
            self.token_dispatcher = _TokenDispatcher()

        def shared_experts_compute(self, hidden_states):
            observed["shared_expert_shape"] = hidden_states.shape
            return hidden_states

        def preprocess(self, hidden_states, probs, routing_map):
            observed["preprocess_hidden_states"] = hidden_states.clone()
            observed["preprocess_probs"] = probs.clone()
            observed["preprocess_routing_map"] = routing_map.clone()
            self.token_dispatcher.hidden_shape = hidden_states.shape
            return hidden_states.reshape(-1, hidden_states.shape[-1]), probs

    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.attn],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=True,
    )
    layer.is_moe_layer = True
    layer.pre_mlp_layernorm = IdentityOp()
    layer.mlp = _OverlapMlp()
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 4, 8]),
        cu_seqlens_kv=torch.IntTensor([0, 4, 8]),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        tokens_per_sample=4,
        pad_between_seqs=True,
        seq_aux_loss_sample_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
    monkeypatch.setattr(
        GraphableMegatronModule,
        "_te_cuda_graph_replay",
        lambda self, *args, **kwargs: (hidden_states,),
    )

    residual, local_tokens, probs, shared_expert_output = layer._te_cuda_graph_replay_impl(
        (),
        {"padding_mask": padding_mask, **tensor_kwargs},
        None,
        eager_packed_seq_params=packed_seq_params,
    )

    expected_route_hidden_states = torch.tensor(
        [[[0.0], [4.0]], [[1.0], [5.0]], [[2.0], [6.0]], [[3.0], [7.0]]]
    )
    assert torch.equal(observed["route_hidden_states"], expected_route_hidden_states)
    assert torch.equal(observed["route_padding_mask"], padding_mask.transpose(0, 1))
    assert observed["seq_aux_loss_sample_ids"] is packed_seq_params.seq_aux_loss_sample_ids
    assert observed["seq_aux_loss_num_samples"] is packed_seq_params.seq_aux_loss_num_samples
    assert observed["seq_aux_loss_max_samples"] == 3
    assert torch.equal(observed["preprocess_hidden_states"], hidden_states)
    assert torch.equal(observed["preprocess_probs"][:, 0], torch.arange(8, dtype=torch.float32))
    assert torch.equal(observed["preprocess_routing_map"][:, 0], torch.arange(8).remainder(2) == 0)
    assert observed["shared_expert_shape"] == hidden_states.shape
    assert residual.shape == hidden_states.shape
    assert shared_expert_output.shape == hidden_states.shape
    assert local_tokens.shape == (8, 1)
    assert probs.shape == (8, 2)
    assert layer.mlp.token_dispatcher.hidden_shape == hidden_states.shape

    combined = layer.mlp.postprocess(local_tokens, shared_expert_output)
    output = combined + residual
    assert torch.equal(output, hidden_states * 3)


def test_packed_graph_static_metadata_keeps_pad_between_seqs() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        tokens_per_sample=8,
        pad_between_seqs=True,
    )

    tensor_kwargs, static = split_packed_seq_params_for_cuda_graph(params)
    rebuilt = build_packed_seq_params_from_cuda_graph_kwargs(dict(tensor_kwargs), static)

    assert rebuilt.pad_between_seqs is True
    assert rebuilt.tokens_per_sample == 8


def test_packed_graph_rejects_changed_static_pad_between_seqs() -> None:
    layer = _TransformerLayerCudaGraphStub()
    captured = PackedSeqParams(qkv_format="thd", pad_between_seqs=True, tokens_per_sample=8)
    tensor_kwargs, static = split_packed_seq_params_for_cuda_graph(captured)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static, tensor_kwargs)
    replay = PackedSeqParams(qkv_format="thd", pad_between_seqs=False, tokens_per_sample=8)

    with pytest.raises(ValueError, match="pad_between_seqs"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": replay})


def test_packed_graph_rejects_changed_static_tokens_per_sample() -> None:
    layer = _TransformerLayerCudaGraphStub()
    captured = PackedSeqParams(qkv_format="thd", pad_between_seqs=True, tokens_per_sample=8)
    tensor_kwargs, static = split_packed_seq_params_for_cuda_graph(captured)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static, tensor_kwargs)
    replay = PackedSeqParams(qkv_format="thd", pad_between_seqs=True, tokens_per_sample=16)

    with pytest.raises(ValueError, match="tokens_per_sample"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": replay})


def _make_packed_seq_params():
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


def test_split_packed_seq_params_for_cuda_graph_separates_tensors_from_metadata():
    packed_seq_params = _make_packed_seq_params()

    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)

    assert set(static_metadata) == set(PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS)
    assert static_metadata == {
        "qkv_format": "thd",
        "max_seqlen_q": 8,
        "max_seqlen_kv": 8,
        "local_cp_size": 1,
        "cp_group": None,
        "pad_between_seqs": None,
        "tokens_per_sample": None,
    }
    assert all(not isinstance(value, torch.Tensor) for value in static_metadata.values())

    expected_tensor_fields = {
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
    }
    assert set(tensor_kwargs) == {
        f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}{field}" for field in expected_tensor_fields
    }
    assert set(PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS) >= expected_tensor_fields
    for value in tensor_kwargs.values():
        assert isinstance(value, torch.Tensor)


def test_has_packed_seq_params_cuda_graph_kwargs_detects_flattened_fields():
    tensor_kwargs, _ = split_packed_seq_params_for_cuda_graph(_make_packed_seq_params())

    assert has_packed_seq_params_cuda_graph_kwargs(tensor_kwargs)
    assert not has_packed_seq_params_cuda_graph_kwargs({"hidden_states": torch.ones(2, 1, 4)})
    assert build_packed_seq_params_from_cuda_graph_kwargs({}, None) is None


def test_build_packed_seq_params_from_cuda_graph_kwargs_pops_flattened_fields():
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    kwargs = {"hidden_states": torch.ones(2, 1, 4), **tensor_kwargs}

    rebuilt = build_packed_seq_params_from_cuda_graph_kwargs(kwargs, static_metadata)

    assert set(kwargs) == {"hidden_states"}
    assert rebuilt.qkv_format == "thd"
    assert rebuilt.max_seqlen_q == 8
    assert rebuilt.max_seqlen_kv == 8
    assert rebuilt.local_cp_size == 1
    assert rebuilt.cp_group is None
    assert rebuilt.total_tokens is None
    assert rebuilt.seq_idx is None
    assert torch.equal(rebuilt.cu_seqlens_q, packed_seq_params.cu_seqlens_q)
    assert torch.equal(rebuilt.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv)
    assert torch.equal(rebuilt.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q_padded)
    assert torch.equal(rebuilt.cu_seqlens_kv_padded, packed_seq_params.cu_seqlens_kv_padded)


def test_build_packed_seq_params_from_cuda_graph_kwargs_can_keep_kwargs_intact():
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(
        _make_packed_seq_params()
    )
    kwargs = dict(tensor_kwargs)

    build_packed_seq_params_from_cuda_graph_kwargs(
        kwargs, static_metadata, remove_from_kwargs=False
    )

    assert kwargs == tensor_kwargs


def test_split_packed_seq_params_for_cuda_graph_rejects_static_tensor_metadata():
    packed_seq_params = _make_packed_seq_params()
    packed_seq_params.max_seqlen_q = torch.IntTensor([8])

    with pytest.raises(TypeError, match="max_seqlen_q"):
        split_packed_seq_params_for_cuda_graph(packed_seq_params)


def test_split_packed_seq_params_for_cuda_graph_ignores_mamba_only_fields():
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 2, 5]),
        max_seqlen_q=3,
        max_seqlen_kv=3,
        total_tokens=5,
    )
    assert packed_seq_params.seq_idx is not None

    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)

    assert f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx" not in tensor_kwargs
    assert "total_tokens" not in static_metadata


def test_transformer_layer_rebuilds_flattened_cuda_graph_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    kwargs = {"hidden_states": torch.ones(2, 1, 4), **tensor_kwargs}

    layer._rebuild_te_cuda_graph_packed_seq_params(kwargs)

    assert set(kwargs) == {"hidden_states", "packed_seq_params"}
    rebuilt = kwargs["packed_seq_params"]
    assert rebuilt.qkv_format == "thd"
    assert rebuilt.max_seqlen_q == 8
    assert rebuilt.max_seqlen_kv == 8
    assert torch.equal(rebuilt.cu_seqlens_q, packed_seq_params.cu_seqlens_q)
    assert torch.equal(rebuilt.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv)


def test_whole_layer_capture_passes_packed_metadata_and_padding_mask_to_mlp():
    class _ConfigStub:
        cuda_graph_modules = []

    class _CaptureLayer(_TransformerLayerCudaGraphStub):
        def __init__(self):
            self.config = _ConfigStub()
            self.offload_module_in_cuda_graph = False
            self.is_moe_layer = True
            self.mlp_kwargs = None

        def _forward_attention(self, *args, **kwargs):
            return kwargs["hidden_states"], None

        def _forward_mlp(self, hidden_states, padding_mask=None, packed_seq_params=None):
            self.mlp_kwargs = {"padding_mask": padding_mask, "packed_seq_params": packed_seq_params}
            return hidden_states

    layer = _CaptureLayer()
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 4, 8]),
        cu_seqlens_kv=torch.IntTensor([0, 4, 8]),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        tokens_per_sample=4,
        pad_between_seqs=True,
        seq_aux_loss_sample_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    moe_tensor_kwargs = _install_moe_graph_contract(layer, packed_seq_params)
    hidden_states = torch.ones(8, 1, 4)
    padding_mask = torch.tensor([[False, False, False, True, False, False, True, True]])

    layer._te_cuda_graph_capture(
        hidden_states=hidden_states,
        padding_mask=padding_mask,
        **tensor_kwargs,
        **moe_tensor_kwargs,
    )

    assert layer.mlp_kwargs["padding_mask"] is padding_mask
    rebuilt = layer.mlp_kwargs["packed_seq_params"]
    assert rebuilt is not packed_seq_params
    assert rebuilt.tokens_per_sample == 4
    assert rebuilt.pad_between_seqs is True
    assert rebuilt.seq_aux_loss_sample_ids is packed_seq_params.seq_aux_loss_sample_ids
    assert rebuilt.seq_aux_loss_num_samples is packed_seq_params.seq_aux_loss_num_samples
    assert rebuilt.seq_aux_loss_max_samples == 3


def test_transformer_layer_flattens_replay_time_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    attention_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    kwargs = {"attention_mask": attention_mask, "packed_seq_params": packed_seq_params}

    layer._flatten_te_cuda_graph_packed_seq_params(kwargs)

    assert kwargs["attention_mask"] is attention_mask
    assert "packed_seq_params" not in kwargs
    assert set(tensor_kwargs).issubset(kwargs)
    for key, value in tensor_kwargs.items():
        assert kwargs[key] is value


def test_transformer_layer_rejects_replay_without_captured_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    _, static_metadata = split_packed_seq_params_for_cuda_graph(_make_packed_seq_params())
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata)

    with pytest.raises(ValueError, match="captured with packed_seq_params"):
        layer._flatten_te_cuda_graph_packed_seq_params({"hidden_states": torch.ones(2, 1, 4)})


def test_transformer_layer_rejects_changed_packed_seq_params_static_metadata():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    _, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata)
    packed_seq_params.max_seqlen_q = 4

    with pytest.raises(ValueError, match="max_seqlen_q"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": packed_seq_params})


def test_transformer_layer_rejects_changed_packed_seq_params_tensor_fields():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    packed_seq_params.cu_seqlens_q_padded = None

    with pytest.raises(ValueError, match="Tensor fields"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": packed_seq_params})


def test_transformer_layer_rejects_replay_with_overlapping_flattened_kwargs():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    existing_key = f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q"

    with pytest.raises(ValueError, match="overlap"):
        layer._flatten_te_cuda_graph_packed_seq_params(
            {existing_key: torch.IntTensor([0]), "packed_seq_params": packed_seq_params}
        )


def test_transformer_layer_rebuild_rejects_flattened_and_object_payload() -> None:
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)

    with pytest.raises(ValueError, match="either as flattened"):
        layer._rebuild_te_cuda_graph_packed_seq_params(
            {**tensor_kwargs, "packed_seq_params": packed_seq_params}
        )


def test_transformer_layer_rebuild_rejects_flattened_payload_without_static_contract() -> None:
    layer = _TransformerLayerCudaGraphStub()
    tensor_kwargs, _ = split_packed_seq_params_for_cuda_graph(_make_packed_seq_params())

    with pytest.raises(ValueError, match="require static metadata"):
        layer._rebuild_te_cuda_graph_packed_seq_params(tensor_kwargs)


def test_te_cuda_graph_sample_kwargs_include_flattened_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    expected_tensor_kwargs, expected_static_metadata = split_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    attention_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    sample_kwargs = {"attention_mask": attention_mask}

    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(layer, sample_kwargs, packed_seq_params)

    assert sample_kwargs["attention_mask"] is attention_mask
    assert set(expected_tensor_kwargs).issubset(sample_kwargs)
    for key, value in expected_tensor_kwargs.items():
        assert sample_kwargs[key] is value
    assert layer._get_te_cuda_graph_packed_seq_params_static_metadata() == expected_static_metadata
    assert layer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names() == tuple(
        sorted(expected_tensor_kwargs)
    )


def test_te_cuda_graph_sample_kwargs_noop_without_packed_seq_params():
    layer = _TransformerLayerCudaGraphStub()
    attention_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    sample_kwargs = {"attention_mask": attention_mask}

    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(layer, sample_kwargs, None)

    assert sample_kwargs == {"attention_mask": attention_mask}
    assert layer._get_te_cuda_graph_packed_seq_params_static_metadata() is None


def test_te_cuda_graph_sample_kwargs_reject_overlapping_flattened_keys():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    sample_kwargs = {f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q": torch.IntTensor([0])}

    with pytest.raises(AssertionError, match="overlap"):
        _add_packed_seq_params_to_te_cuda_graph_sample_kwargs(
            layer, sample_kwargs, packed_seq_params
        )


_GENERIC_PACKED_FIELD_VALUES = {
    "cu_seqlens_q": torch.IntTensor([0, 1]),
    "cu_seqlens_kv": torch.IntTensor([0, 1]),
    "cu_seqlens_q_padded": torch.IntTensor([0, 1]),
    "cu_seqlens_kv_padded": torch.IntTensor([0, 1]),
    "qkv_format": "thd",
    "max_seqlen_q": 1,
    "max_seqlen_kv": 1,
    "local_cp_size": 1,
    "cp_group": object(),
    "pad_between_seqs": True,
    "tokens_per_sample": 1,
}


@pytest.mark.parametrize("cuda_graph_modules", ([], [CudaGraphModule.attn]))
@pytest.mark.parametrize(
    "field_name",
    PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS + PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS,
)
def test_attention_replay_rejects_uncaptured_generic_packed_payload_before_execution(
    cuda_graph_modules: list[CudaGraphModule], field_name: str
) -> None:
    class _RejectingReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=cuda_graph_modules,
                delay_offload_until_cuda_graph=False,
            )
            self.self_attention = torch.nn.Identity()
            self.is_moe_layer = False
            self.graph_calls = 0

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            return args[0]

    layer = _RejectingReplayLayer()
    packed_seq_params = PackedSeqParams(
        **{field_name: _GENERIC_PACKED_FIELD_VALUES[field_name]}
    )

    with pytest.raises(ValueError, match="attention graph was captured without"):
        layer._te_cuda_graph_replay(
            torch.ones(1, 1, 4), packed_seq_params=packed_seq_params
        )

    assert layer.graph_calls == 0


@pytest.mark.parametrize(
    "self_attention,packed_seq_params",
    (
        (
            IdentityOp(),
            PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=torch.IntTensor([0, 1]),
                cu_seqlens_kv=torch.IntTensor([0, 1]),
                max_seqlen_q=1,
                max_seqlen_kv=1,
            ),
        ),
        (
            torch.nn.Identity(),
            PackedSeqParams(total_tokens=1, seq_idx=torch.IntTensor([[0]])),
        ),
    ),
    ids=("identity-attention", "mamba-only"),
)
def test_attention_replay_without_generic_contract_accepts_unowned_packed_payload(
    self_attention: torch.nn.Module, packed_seq_params: PackedSeqParams
) -> None:
    class _AcceptingReplayLayer(_TransformerLayerCudaGraphStub):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                cuda_graph_modules=[CudaGraphModule.attn],
                delay_offload_until_cuda_graph=False,
            )
            self.self_attention = self_attention
            self.is_moe_layer = False
            self.graph_calls = 0

        def _te_cuda_graph_replay_impl(
            self, args, kwargs, context, *, eager_packed_seq_params=None
        ):
            self.graph_calls += 1
            assert "packed_seq_params" not in kwargs
            return args[0]

    layer = _AcceptingReplayLayer()
    hidden_states = torch.ones(1, 1, 4)

    output = layer._te_cuda_graph_replay(
        hidden_states, packed_seq_params=packed_seq_params
    )

    assert output is hidden_states
    assert layer.graph_calls == 1


def test_te_cuda_graph_partial_attn_only_flow():
    class _ConfigStub:
        def __init__(self, cuda_graph_modules):
            self.cuda_graph_modules = cuda_graph_modules
            self.delay_offload_until_cuda_graph = False

    class _TestLayer(_TransformerLayerCudaGraphStub):
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

    # Case 1: When CudaGraphModule.attn is captured
    layer_attn = _TestLayer([CudaGraphModule.attn])
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer_attn._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)

    kwargs = {"packed_seq_params": packed_seq_params, "hidden_states": torch.ones(2, 1, 4)}
    layer_attn._te_cuda_graph_replay(**kwargs)

    assert not layer_attn.attn_called
    assert layer_attn.replay_impl_called
    assert layer_attn.replay_impl_context is None
    assert "packed_seq_params" not in layer_attn.replay_impl_kwargs
    assert f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q" in layer_attn.replay_impl_kwargs

    # Case 2: When CudaGraphModule.attn is NOT captured (e.g. only mlp is captured)
    layer_mlp = _TestLayer([CudaGraphModule.mlp])

    kwargs = {"packed_seq_params": packed_seq_params, "hidden_states": torch.ones(2, 1, 4)}
    layer_mlp._te_cuda_graph_replay(**kwargs)

    assert layer_mlp.attn_called
    assert layer_mlp.replay_impl_called
    assert layer_mlp.replay_impl_context == "attn_context"
    assert len(layer_mlp.replay_impl_args) == 1
    assert torch.equal(layer_mlp.replay_impl_args[0], torch.ones(2, 1, 4) * 2.0)
    assert layer_mlp.replay_impl_kwargs == {}


def test_seq_idx_determinism_across_replays():
    cu_seqlens = torch.IntTensor([0, 3, 7, 10])
    cu_seqlens_padded = torch.IntTensor([0, 4, 8, 12])

    params1 = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=10,
    )

    params2 = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=10,
    )

    assert params1.seq_idx is not None
    assert params2.seq_idx is not None
    assert torch.equal(params1.seq_idx, params2.seq_idx)
    assert params1.seq_idx.shape == params2.seq_idx.shape
    assert params1.seq_idx.dtype == torch.int32
