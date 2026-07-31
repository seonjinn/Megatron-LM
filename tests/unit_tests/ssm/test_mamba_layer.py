# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import (
    CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PackedSeqParams,
    split_mamba_packed_seq_params_for_cuda_graph,
    split_packed_seq_params_for_cuda_graph,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from tests.unit_tests.test_utilities import Utils


class _MambaLayerCudaGraphStub:
    _set_te_cuda_graph_mamba_packed_seq_params_static_metadata = (
        MambaLayer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_mamba_packed_seq_params_static_metadata = (
        MambaLayer._get_te_cuda_graph_mamba_packed_seq_params_static_metadata
    )
    _validate_te_cuda_graph_mamba_static_metadata = staticmethod(
        MambaLayer._validate_te_cuda_graph_mamba_static_metadata
    )
    _validate_te_cuda_graph_mamba_tensor_kwargs = (
        MambaLayer._validate_te_cuda_graph_mamba_tensor_kwargs
    )
    _set_te_cuda_graph_packed_seq_params_static_metadata = (
        MambaLayer._set_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_static_metadata = (
        MambaLayer._get_te_cuda_graph_packed_seq_params_static_metadata
    )
    _validate_te_cuda_graph_packed_seq_params_static_metadata = (
        MambaLayer._validate_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_tensor_kwarg_names = (
        MambaLayer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names
    )
    _validate_te_cuda_graph_packed_seq_params_tensor_kwargs = (
        MambaLayer._validate_te_cuda_graph_packed_seq_params_tensor_kwargs
    )
    _rebuild_te_cuda_graph_packed_seq_params = MambaLayer._rebuild_te_cuda_graph_packed_seq_params
    _flatten_te_cuda_graph_packed_seq_params = MambaLayer._flatten_te_cuda_graph_packed_seq_params
    _rebuild_te_cuda_graph_mamba_packed_seq_params = (
        MambaLayer._rebuild_te_cuda_graph_mamba_packed_seq_params
    )
    _flatten_te_cuda_graph_mamba_packed_seq_params = (
        MambaLayer._flatten_te_cuda_graph_mamba_packed_seq_params
    )
    _te_cuda_graph_capture = MambaLayer._te_cuda_graph_capture

    def forward(self, *args, **kwargs):
        self.forward_args = args
        self.forward_kwargs = kwargs
        return kwargs.get("hidden_states", args[0] if args else None)


def _make_mamba_packed_seq_params(total_tokens: int = 16) -> PackedSeqParams:
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        local_cp_size=1,
        total_tokens=total_tokens,
        seq_aux_loss_sample_ids=torch.cat(
            (
                torch.zeros(total_tokens // 2, dtype=torch.int64),
                torch.ones(total_tokens - total_tokens // 2, dtype=torch.int64),
            )
        ),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )


def _capture_mamba_packed_seq_params(
    layer: _MambaLayerCudaGraphStub, packed_seq_params: PackedSeqParams
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generic_tensor_kwargs, generic_static = split_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    mamba_tensor_kwargs, mamba_static = split_mamba_packed_seq_params_for_cuda_graph(
        packed_seq_params
    )
    layer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        mamba_static, mamba_tensor_kwargs
    )
    layer._set_te_cuda_graph_packed_seq_params_static_metadata(
        generic_static, generic_tensor_kwargs
    )
    assert not any(key.startswith("_moe_packed_seq_params_") for key in generic_tensor_kwargs)
    assert not any(key.startswith("_moe_packed_seq_params_") for key in mamba_tensor_kwargs)
    assert "seq_aux_loss_max_samples" not in generic_static
    assert "seq_aux_loss_max_samples" not in mamba_static
    return generic_tensor_kwargs, mamba_tensor_kwargs


def test_mamba_graph_uses_capacity_sized_seq_idx() -> None:
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        total_tokens=16,
    )

    tensor_kwargs, static = split_mamba_packed_seq_params_for_cuda_graph(params)

    assert tensor_kwargs["_mamba_packed_seq_params_seq_idx"].shape == (1, 16)
    assert "cu_seqlens_kv" not in tensor_kwargs
    assert static["qkv_format"] == "thd"
    assert static["total_tokens"] == 16


def test_mamba_replay_rejects_seq_idx_shape_change() -> None:
    layer = _MambaLayerCudaGraphStub()
    captured = _make_mamba_packed_seq_params(total_tokens=16)
    tensor_kwargs, static = split_mamba_packed_seq_params_for_cuda_graph(captured)
    layer._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(static, tensor_kwargs)
    replay = _make_mamba_packed_seq_params(total_tokens=12)

    with pytest.raises(AssertionError, match="total_tokens"):
        layer._flatten_te_cuda_graph_mamba_packed_seq_params({"packed_seq_params": replay})


def test_mamba_capture_rebuilds_packed_seq_params_without_recomputing_seq_idx() -> None:
    layer = _MambaLayerCudaGraphStub()
    packed_seq_params = _make_mamba_packed_seq_params()
    generic_tensor_kwargs, mamba_tensor_kwargs = _capture_mamba_packed_seq_params(
        layer, packed_seq_params
    )
    hidden_states = torch.ones(16, 1, 4)
    graph_kwargs = {"hidden_states": hidden_states, **generic_tensor_kwargs, **mamba_tensor_kwargs}

    assert not any(key.startswith("_moe_packed_seq_params_") for key in graph_kwargs)

    layer._te_cuda_graph_capture(**graph_kwargs)

    assert set(layer.forward_kwargs) == {"hidden_states", "packed_seq_params"}
    rebuilt = layer.forward_kwargs["packed_seq_params"]
    assert rebuilt.total_tokens == 16
    assert rebuilt.seq_idx is packed_seq_params.seq_idx
    assert rebuilt.cu_seqlens_q is packed_seq_params.cu_seqlens_q
    assert rebuilt.cu_seqlens_kv is packed_seq_params.cu_seqlens_kv
    assert rebuilt.seq_aux_loss_sample_ids is None
    assert rebuilt.seq_aux_loss_num_samples is None
    assert rebuilt.seq_aux_loss_max_samples is None
    assert packed_seq_params.seq_aux_loss_sample_ids is not None
    assert packed_seq_params.seq_aux_loss_num_samples is not None
    assert packed_seq_params.seq_aux_loss_max_samples == 3


def test_mamba_replay_flattens_generic_and_mamba_tensor_inputs() -> None:
    layer = _MambaLayerCudaGraphStub()
    packed_seq_params = _make_mamba_packed_seq_params()
    generic_tensor_kwargs, mamba_tensor_kwargs = _capture_mamba_packed_seq_params(
        layer, packed_seq_params
    )
    kwargs = {"packed_seq_params": packed_seq_params}

    layer._flatten_te_cuda_graph_mamba_packed_seq_params(kwargs)

    assert "packed_seq_params" not in kwargs
    assert set(kwargs) == set(generic_tensor_kwargs) | set(mamba_tensor_kwargs)
    assert not any(key.startswith("_moe_packed_seq_params_") for key in kwargs)
    assert kwargs[f"{MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx"] is (
        packed_seq_params.seq_idx
    )
    assert f"{CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx" not in kwargs


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda tensor: None, "Tensor fields"),
        (lambda tensor: tensor[:, :-1], "shape"),
        (lambda tensor: tensor.to(torch.int64), "dtype"),
        (lambda tensor: torch.empty_like(tensor, device="meta"), "device"),
        (lambda tensor: tensor.to_sparse(), "layout"),
        (
            lambda tensor: torch.empty((tensor.shape[0], tensor.shape[1] * 2), dtype=tensor.dtype)[
                :, ::2
            ],
            "stride",
        ),
    ],
)
def test_mamba_replay_rejects_changed_seq_idx_tensor_contract(mutation, message) -> None:
    layer = _MambaLayerCudaGraphStub()
    packed_seq_params = _make_mamba_packed_seq_params()
    _capture_mamba_packed_seq_params(layer, packed_seq_params)
    replay = _make_mamba_packed_seq_params()
    replay.seq_idx = mutation(replay.seq_idx)

    with pytest.raises(AssertionError, match=message):
        layer._flatten_te_cuda_graph_mamba_packed_seq_params({"packed_seq_params": replay})


def test_mamba_replay_rejects_packed_seq_params_presence_change() -> None:
    captured_with_packed = _MambaLayerCudaGraphStub()
    _capture_mamba_packed_seq_params(captured_with_packed, _make_mamba_packed_seq_params())
    with pytest.raises(AssertionError, match="captured with packed_seq_params"):
        captured_with_packed._flatten_te_cuda_graph_mamba_packed_seq_params({})

    captured_without_packed = _MambaLayerCudaGraphStub()
    captured_without_packed._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(None)
    with pytest.raises(AssertionError, match="captured without packed_seq_params"):
        captured_without_packed._flatten_te_cuda_graph_mamba_packed_seq_params(
            {"packed_seq_params": _make_mamba_packed_seq_params()}
        )


def test_mamba_replay_rejects_overlapping_flattened_input() -> None:
    layer = _MambaLayerCudaGraphStub()
    packed_seq_params = _make_mamba_packed_seq_params()
    _capture_mamba_packed_seq_params(layer, packed_seq_params)
    seq_idx_key = f"{MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx"

    with pytest.raises(AssertionError, match="overlap"):
        layer._flatten_te_cuda_graph_mamba_packed_seq_params(
            {seq_idx_key: packed_seq_params.seq_idx, "packed_seq_params": packed_seq_params}
        )


@pytest.mark.internal
class TestMambaLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=1,
            num_attention_heads=1,
            layernorm_epsilon=1e-6,
            use_cpu_initialization=True,
        )
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        # Use an explicit norm so the test can verify the configured epsilon.
        mamba_submodules = replace(
            hybrid_stack_spec.submodules.mamba_layer.submodules, norm=WrappedTorchNorm
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.layer = MambaLayer(transformer_config, mamba_submodules, pg_collection=pg_collection)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_configured_layernorm_epsilon(self):
        assert self.layer.norm.eps == self.layer.config.layernorm_epsilon

    def test_gpu_forward(self):
        layer = self.layer
        layer.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == layer.config.hidden_size
        assert output.dtype == torch.float32
