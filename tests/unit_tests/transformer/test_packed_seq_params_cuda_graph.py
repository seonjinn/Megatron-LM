# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Callable
from types import MethodType, SimpleNamespace

import pytest
import torch

import megatron.core.packed_seq_params as packed_seq_module
import megatron.core.transformer.cuda_graphs as cuda_graphs_module
from megatron.core.packed_seq_params import (
    CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PACKED_SEQ_PARAMS_CUDA_GRAPH_STATIC_FIELDS,
    PACKED_SEQ_PARAMS_CUDA_GRAPH_TENSOR_FIELDS,
    PackedSeqParams,
    build_packed_seq_params_from_cuda_graph_kwargs,
    has_packed_seq_params_cuda_graph_kwargs,
    split_packed_seq_params_for_cuda_graph,
)
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.cuda_graphs import (
    TECudaGraphHelper,
    _add_packed_seq_params_to_te_cuda_graph_sample_kwargs,
    validate_packed_partial_moe_cuda_graph,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.moe_utils import MoECudaGraphTensorStore
from megatron.core.transformer.transformer_layer import TransformerLayer


class _TransformerLayerCudaGraphStub:
    _te_cuda_graph_captures_attention = (
        TransformerLayer._te_cuda_graph_captures_attention
    )
    _reconcile_packed_partial_cudagraph_tensor = (
        TransformerLayer._reconcile_packed_partial_cudagraph_tensor
    )
    _reconcile_packed_partial_cudagraph_residual = (
        TransformerLayer._reconcile_packed_partial_cudagraph_residual
    )
    _reconcile_packed_partial_cudagraph_post_mlp_inputs = (
        TransformerLayer._reconcile_packed_partial_cudagraph_post_mlp_inputs
    )
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


def _make_mamba_packed_seq_params() -> PackedSeqParams:
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.IntTensor([0, 2, 5]),
        cu_seqlens_kv=torch.IntTensor([0, 3, 5]),
        cu_seqlens_q_padded=torch.IntTensor([0, 4, 8]),
        cu_seqlens_kv_padded=torch.IntTensor([0, 6, 8]),
        max_seqlen_q=4,
        max_seqlen_kv=6,
        local_cp_size=2,
        total_tokens=8,
    )
    packed_seq_params.seq_idx = torch.IntTensor([[0, 0, 1, 1, 1, 2, 2, 2]])
    return packed_seq_params


def _get_mamba_graph_helpers() -> tuple[Callable[..., object], Callable[..., object]]:
    split = getattr(
        packed_seq_module,
        "split_mamba_packed_seq_params_for_cuda_graph",
        None,
    )
    build = getattr(
        packed_seq_module,
        "build_mamba_packed_seq_params_from_cuda_graph_kwargs",
        None,
    )
    assert callable(split)
    assert callable(build)
    return split, build


def test_mamba_graph_schema_uses_only_consumed_tensor_fields() -> None:
    split, _ = _get_mamba_graph_helpers()
    packed = _make_mamba_packed_seq_params()
    tensor_kwargs, static = split(packed, include_cp_fields=True)
    assert set(tensor_kwargs) == {
        "_mamba_packed_seq_params_seq_idx",
        "_mamba_packed_seq_params_cu_seqlens_q",
        "_mamba_packed_seq_params_cu_seqlens_q_padded",
    }
    assert "_mamba_packed_seq_params_cu_seqlens_kv" not in tensor_kwargs
    assert "total_tokens" not in static


def test_mamba_graph_schema_without_cp_uses_only_seq_idx() -> None:
    split, _ = _get_mamba_graph_helpers()
    tensor_kwargs, _ = split(_make_mamba_packed_seq_params(), include_cp_fields=False)
    assert set(tensor_kwargs) == {"_mamba_packed_seq_params_seq_idx"}


def test_mamba_graph_schema_requires_seq_idx_tensor() -> None:
    split, _ = _get_mamba_graph_helpers()
    packed = _make_mamba_packed_seq_params()
    packed.seq_idx = None

    with pytest.raises(TypeError, match="seq_idx must be a Tensor"):
        split(packed, include_cp_fields=False)


def test_mamba_rebuild_requires_prefixed_seq_idx_key() -> None:
    _, build = _get_mamba_graph_helpers()
    static_metadata = {
        "packed_seq_params_present": True,
        "include_cp_fields": False,
        "tensor_field_names": (),
    }

    with pytest.raises(AssertionError, match="_mamba_packed_seq_params_seq_idx"):
        build({}, static_metadata)


def test_mamba_rebuild_preserves_supplied_seq_idx_identity() -> None:
    split, build = _get_mamba_graph_helpers()
    packed = _make_mamba_packed_seq_params()
    tensor_kwargs, static = split(packed, include_cp_fields=True)
    rebuilt = build(dict(tensor_kwargs), static)
    assert rebuilt.seq_idx is packed.seq_idx
    assert rebuilt.total_tokens is None


def _get_mamba_layer_graph_methods() -> tuple[Callable[..., object], Callable[..., object]]:
    flatten = getattr(
        MambaLayer,
        "_flatten_te_cuda_graph_mamba_packed_seq_params",
        None,
    )
    replay = getattr(MambaLayer, "_te_cuda_graph_replay", None)
    assert callable(flatten)
    assert callable(replay)
    return flatten, replay


def _get_add_mamba_sample_helper() -> Callable[..., object]:
    helper = getattr(
        cuda_graphs_module,
        "_add_mamba_packed_seq_params_to_te_cuda_graph_sample_kwargs",
        None,
    )
    assert callable(helper)
    return helper


def _make_mamba_layer_for_graph_test(
    monkeypatch: pytest.MonkeyPatch,
    packed_seq_params: PackedSeqParams,
    *,
    context_parallel_size: int = 1,
) -> tuple[MambaLayer, dict[str, bool]]:
    layer = MambaLayer.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(context_parallel_size=context_parallel_size)
    _get_add_mamba_sample_helper()(layer, {}, packed_seq_params)
    te_called = {"value": False}

    def _fake_te_replay(
        self: GraphableMegatronModule, *args: object, **kwargs: object
    ) -> object:
        te_called["value"] = True
        return object()

    monkeypatch.setattr(GraphableMegatronModule, "_te_cuda_graph_replay", _fake_te_replay)
    return layer, te_called


def test_mamba_layer_replay_rejects_changed_seq_idx_shape_before_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, replay = _get_mamba_layer_graph_methods()
    captured = _make_mamba_packed_seq_params()
    layer, te_called = _make_mamba_layer_for_graph_test(monkeypatch, captured)
    replayed = _make_mamba_packed_seq_params()
    replayed.seq_idx = torch.zeros((1, captured.seq_idx.shape[1] + 1), dtype=torch.int32)

    with pytest.raises(AssertionError, match="shape"):
        replay(layer, packed_seq_params=replayed)

    assert not te_called["value"]


def test_mamba_layer_replay_rejects_changed_seq_idx_dtype_before_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, replay = _get_mamba_layer_graph_methods()
    captured = _make_mamba_packed_seq_params()
    layer, te_called = _make_mamba_layer_for_graph_test(monkeypatch, captured)
    replayed = _make_mamba_packed_seq_params()
    replayed.seq_idx = replayed.seq_idx.to(torch.int64)

    with pytest.raises(AssertionError, match="dtype"):
        replay(layer, packed_seq_params=replayed)

    assert not te_called["value"]


def test_mamba_layer_replay_rejects_changed_static_metadata_before_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, replay = _get_mamba_layer_graph_methods()
    captured = _make_mamba_packed_seq_params()
    layer, te_called = _make_mamba_layer_for_graph_test(monkeypatch, captured)
    layer._te_cuda_graph_mamba_packed_seq_params_static_metadata[
        "include_cp_fields"
    ] = True

    with pytest.raises(AssertionError, match="static metadata"):
        replay(layer, packed_seq_params=_make_mamba_packed_seq_params())

    assert not te_called["value"]


def test_mamba_layer_replay_rejects_changed_dynamic_field_set_before_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, replay = _get_mamba_layer_graph_methods()
    captured = _make_mamba_packed_seq_params()
    layer, te_called = _make_mamba_layer_for_graph_test(monkeypatch, captured)
    replayed = _make_mamba_packed_seq_params()
    replayed.seq_idx = None

    with pytest.raises(TypeError, match="seq_idx must be a Tensor"):
        replay(layer, packed_seq_params=replayed)

    assert not te_called["value"]


def test_mamba_layer_sample_stores_exact_tensor_signatures() -> None:
    layer = MambaLayer.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(context_parallel_size=1)
    packed_seq_params = _make_mamba_packed_seq_params()

    _get_add_mamba_sample_helper()(layer, {}, packed_seq_params)

    signatures = layer._te_cuda_graph_mamba_packed_seq_params_tensor_signatures
    assert signatures == {
        "_mamba_packed_seq_params_seq_idx": (
            packed_seq_params.seq_idx.shape,
            packed_seq_params.seq_idx.dtype,
            packed_seq_params.seq_idx.device,
            packed_seq_params.seq_idx.layout,
            packed_seq_params.seq_idx.stride(),
        )
    }


def _get_mamba_graph_callable_sample_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    packed_seq_params: PackedSeqParams,
    *,
    cuda_graph_modules: list[CudaGraphModule],
    context_parallel_size: int,
) -> dict[str, object]:
    monkeypatch.setattr(cuda_graphs_module, "is_te_min_version", lambda version: True)
    layer = MambaLayer.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        context_parallel_size=context_parallel_size,
        cuda_graph_modules=cuda_graph_modules,
    )
    layer.get_layer_static_inputs = lambda seq_length, micro_batch_size: {
        "hidden_states": torch.ones(seq_length, micro_batch_size, 4)
    }
    chunk = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.config = layer.config
    helper.seq_length = 8
    helper.micro_batch_size = 1
    helper.sample_packed_seq_params = packed_seq_params
    helper.num_model_chunks = 1
    helper.num_microbatches = 1
    helper.num_layers_per_chunk = [1]
    helper.callables_per_chunk = [[layer]]
    helper.flattened_callables = [layer]
    helper.chunks_with_decoder = [chunk]

    _, sample_kwargs = helper._get_sample_arguments([1, -1])
    return sample_kwargs[0]


def test_helper_adds_mamba_sample_for_explicit_scope_without_attention_kv_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_seq_params = _make_mamba_packed_seq_params()
    sample_kwargs = _get_mamba_graph_callable_sample_kwargs(
        monkeypatch,
        packed_seq_params,
        cuda_graph_modules=[CudaGraphModule.mamba],
        context_parallel_size=1,
    )

    assert set(sample_kwargs) == {"_mamba_packed_seq_params_seq_idx"}
    assert sample_kwargs["_mamba_packed_seq_params_seq_idx"] is packed_seq_params.seq_idx
    assert not any("kv" in key for key in sample_kwargs)


def test_helper_adds_mamba_sample_for_whole_layer_scope_with_cp_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_seq_params = _make_mamba_packed_seq_params()
    sample_kwargs = _get_mamba_graph_callable_sample_kwargs(
        monkeypatch,
        packed_seq_params,
        cuda_graph_modules=[],
        context_parallel_size=2,
    )

    assert set(sample_kwargs) == {
        "_mamba_packed_seq_params_seq_idx",
        "_mamba_packed_seq_params_cu_seqlens_q",
        "_mamba_packed_seq_params_cu_seqlens_q_padded",
    }
    assert not any("kv" in key for key in sample_kwargs)


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

    with pytest.raises(AssertionError, match="captured with packed_seq_params"):
        layer._flatten_te_cuda_graph_packed_seq_params({"hidden_states": torch.ones(2, 1, 4)})


def test_transformer_layer_rejects_changed_packed_seq_params_static_metadata():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    _, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata)
    packed_seq_params.max_seqlen_q = 4

    with pytest.raises(AssertionError, match="max_seqlen_q"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": packed_seq_params})


def test_transformer_layer_rejects_changed_packed_seq_params_tensor_fields():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    packed_seq_params.cu_seqlens_q_padded = None

    with pytest.raises(AssertionError, match="Tensor fields"):
        layer._flatten_te_cuda_graph_packed_seq_params({"packed_seq_params": packed_seq_params})


def test_transformer_layer_rejects_replay_with_overlapping_flattened_kwargs():
    layer = _TransformerLayerCudaGraphStub()
    packed_seq_params = _make_packed_seq_params()
    tensor_kwargs, static_metadata = split_packed_seq_params_for_cuda_graph(packed_seq_params)
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(static_metadata, tensor_kwargs)
    existing_key = f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}cu_seqlens_q"

    with pytest.raises(AssertionError, match="overlap"):
        layer._flatten_te_cuda_graph_packed_seq_params(
            {existing_key: torch.IntTensor([0]), "packed_seq_params": packed_seq_params}
        )


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


def test_te_cuda_graph_partial_replay_flow():

    class _ConfigStub:
        def __init__(self, cuda_graph_modules):
            self.cuda_graph_modules = cuda_graph_modules
            self.delay_offload_until_cuda_graph = False
            self.moe_shared_expert_intermediate_size = 64
            self.moe_shared_expert_overlap = False
            self.overlap_moe_expert_parallel_comm = False
            self.cuda_graph_impl = "transformer_engine"

    class _TestLayer(_TransformerLayerCudaGraphStub):
        _te_cuda_graph_replay = TransformerLayer._te_cuda_graph_replay

        def __init__(
            self,
            cuda_graph_modules,
            *,
            is_moe_layer=False,
            has_self_attention=True,
        ):
            self.config = _ConfigStub(cuda_graph_modules)
            self.is_moe_layer = is_moe_layer
            self.self_attention = object() if has_self_attention else IdentityOp()
            self.mlp = SimpleNamespace(cudagraph_tensor_store=MoECudaGraphTensorStore())
            self.attn_called = False
            self.replay_impl_called = False
            self.replay_impl_args = None
            self.replay_impl_kwargs = None
            self.replay_impl_context = None
            self.replay_impl_packed_seq = None

        def _forward_attention(self, *args, **kwargs):
            self.attn_called = True
            return torch.ones(2, 1, 4) * 2.0, "attn_context"

        def _te_cuda_graph_replay_impl(self, args, kwargs, context):
            self.replay_impl_called = True
            self.replay_impl_args = args
            self.replay_impl_kwargs = kwargs
            self.replay_impl_context = context
            self.replay_impl_packed_seq = (
                self.mlp.cudagraph_tensor_store.is_packed_seq_replay
            )
            if self.is_moe_layer and self.config.overlap_moe_expert_parallel_comm:
                self.mlp.cudagraph_tensor_store.hidden_states = torch.ones(2, 1, 4)
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
    assert layer_attn.replay_impl_packed_seq is False

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
    assert layer_mlp.replay_impl_packed_seq is False

    # Case 3: A hybrid MoE layer without self-attention runs its identity attention path
    # eagerly even when the model-level graph scope also includes attention.
    layer_hybrid_moe = _TestLayer(
        [
            CudaGraphModule.attn,
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        ],
        is_moe_layer=True,
        has_self_attention=False,
    )
    layer_hybrid_moe._te_cuda_graph_replay(**kwargs)

    assert layer_hybrid_moe.attn_called
    assert layer_hybrid_moe.replay_impl_called
    assert layer_hybrid_moe.replay_impl_kwargs == {}
    assert layer_hybrid_moe.replay_impl_packed_seq is True

    # Case 4: Every supported packed partial MoE replay exposes scoped state without
    # changing the impl contract, independently of shared-expert configuration.
    for shared_expert_intermediate_size, shared_expert_overlap in (
        (64, False),
        (None, False),
        (64, True),
    ):
        layer_moe = _TestLayer(
            [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            is_moe_layer=True,
        )
        layer_moe.config.moe_shared_expert_intermediate_size = shared_expert_intermediate_size
        layer_moe.config.moe_shared_expert_overlap = shared_expert_overlap
        layer_moe._te_cuda_graph_replay(**kwargs)

        assert layer_moe.replay_impl_packed_seq is True
        assert layer_moe.mlp.cudagraph_tensor_store.is_packed_seq_replay is False

        layer_moe._te_cuda_graph_replay(hidden_states=torch.ones(2, 1, 4))

        assert layer_moe.replay_impl_packed_seq is False
        assert layer_moe.mlp.cudagraph_tensor_store.is_packed_seq_replay is False

    # Case 5: Packed partial MoE fails closed with EP overlap and retains no global flag.
    layer_overlap = _TestLayer(
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        is_moe_layer=True,
    )
    layer_overlap.config.overlap_moe_expert_parallel_comm = True
    for _ in range(2):
        with pytest.raises(ValueError, match="EP communication overlap"):
            layer_overlap._te_cuda_graph_replay(**kwargs)
        assert layer_overlap.mlp.cudagraph_tensor_store.is_packed_seq_replay is False

    for _ in range(2):
        layer_overlap._te_cuda_graph_replay(hidden_states=torch.ones(2, 1, 4))
        assert layer_overlap.replay_impl_packed_seq is False
        assert layer_overlap.mlp.cudagraph_tensor_store.is_packed_seq_replay is False


def test_validate_packed_partial_moe_cuda_graph_fails_closed_with_ep_overlap():
    config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        overlap_moe_expert_parallel_comm=True,
    )

    with pytest.raises(ValueError, match="EP communication overlap"):
        validate_packed_partial_moe_cuda_graph(config, has_packed_seq_params=True)

    validate_packed_partial_moe_cuda_graph(config, has_packed_seq_params=False)
    config.overlap_moe_expert_parallel_comm = False
    validate_packed_partial_moe_cuda_graph(config, has_packed_seq_params=True)


@pytest.mark.parametrize(
    ("shared_expert_intermediate_size", "shared_expert_overlap"),
    ((64, False), (None, False), (64, True)),
    ids=("shared-expert", "no-shared-expert", "shared-expert-overlap"),
)
def test_packed_partial_moe_post_mlp_inputs_use_eager_mlp_extent(
    shared_expert_intermediate_size, shared_expert_overlap
):
    layer = _TransformerLayerCudaGraphStub()
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        moe_shared_expert_intermediate_size=shared_expert_intermediate_size,
        moe_shared_expert_overlap=shared_expert_overlap,
        overlap_moe_expert_parallel_comm=False,
    )
    layer.mlp = SimpleNamespace(
        cudagraph_tensor_store=SimpleNamespace(is_packed_seq_replay=True)
    )
    captured_residual = torch.empty(16, 1, 8)
    eager_mlp_output = torch.empty(12, 1, 8)
    mlp_bias = torch.empty(8)
    replay_hidden_states = torch.empty(16, 1, 8)

    mlp_output_with_bias, residual = (
        layer._reconcile_packed_partial_cudagraph_post_mlp_inputs(
            (eager_mlp_output, mlp_bias),
            captured_residual,
            replay_hidden_states,
        )
    )

    assert residual.shape == (12, 1, 8)
    assert residual.data_ptr() == captured_residual.data_ptr()
    assert mlp_output_with_bias[0].shape == (12, 1, 8)
    assert mlp_output_with_bias[0] is eager_mlp_output
    assert mlp_output_with_bias[1] is mlp_bias


def test_packed_partial_moe_rejects_mlp_output_larger_than_replay_input():
    layer = _TransformerLayerCudaGraphStub()
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        overlap_moe_expert_parallel_comm=False,
    )
    layer.mlp = SimpleNamespace(
        cudagraph_tensor_store=SimpleNamespace(is_packed_seq_replay=True)
    )

    with pytest.raises(RuntimeError, match="replay hidden states replay exceeds captured capacity"):
        layer._reconcile_packed_partial_cudagraph_post_mlp_inputs(
            (torch.empty(16, 1, 8), None),
            torch.empty(16, 1, 8),
            torch.empty(12, 1, 8),
        )


def test_packed_partial_moe_rejects_mlp_output_larger_than_residual_capacity():
    layer = _TransformerLayerCudaGraphStub()
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        overlap_moe_expert_parallel_comm=False,
    )
    layer.mlp = SimpleNamespace(
        cudagraph_tensor_store=SimpleNamespace(is_packed_seq_replay=True)
    )

    with pytest.raises(RuntimeError, match="residual replay exceeds captured capacity"):
        layer._reconcile_packed_partial_cudagraph_post_mlp_inputs(
            (torch.empty(16, 1, 8), None),
            torch.empty(12, 1, 8),
            torch.empty(16, 1, 8),
        )


def test_te_cuda_graph_replay_exits_offload_when_packed_state_setup_fails():
    class FailingStore(MoECudaGraphTensorStore):
        def set(self, **kwargs):
            raise RuntimeError("packed state setup failed")

    events = []
    layer = _TransformerLayerCudaGraphStub()
    layer._te_cuda_graph_replay = MethodType(TransformerLayer._te_cuda_graph_replay, layer)
    layer._forward_attention = MethodType(
        lambda bound_layer, *args, **kwargs: (torch.ones(2, 1, 4), None), layer
    )
    layer._te_cuda_graph_replay_impl = MethodType(
        lambda bound_layer, args, kwargs, context: torch.ones(2, 1, 4), layer
    )
    layer.config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        delay_offload_until_cuda_graph=True,
        moe_shared_expert_intermediate_size=64,
        moe_shared_expert_overlap=False,
        overlap_moe_expert_parallel_comm=False,
    )
    layer.is_moe_layer = True
    layer.mlp = SimpleNamespace(cudagraph_tensor_store=FailingStore())
    layer.off_interface = SimpleNamespace(
        enter_replay=lambda: events.append("enter"),
        exit_replay=lambda: events.append("exit"),
    )

    with pytest.raises(RuntimeError, match="packed state setup failed"):
        layer._te_cuda_graph_replay(
            hidden_states=torch.ones(2, 1, 4),
            packed_seq_params=_make_packed_seq_params(),
        )

    assert events == ["enter", "exit"]


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
