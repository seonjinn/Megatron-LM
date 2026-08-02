# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from tests.unit_tests.test_utilities import Utils


def _make_static_thd_mamba_layer(
    *, sequence_parallel: bool = False, tensor_model_parallel_size: int = 1
) -> MambaLayer:
    layer = object.__new__(MambaLayer)
    layer.config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.mamba],
        max_seqlen_per_dp_cp_rank=8,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=2,
        thd_tail_padding_policy="extend_last",
    )
    return layer


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "sequence_parallel,tensor_model_parallel_size,expected_hidden_tokens",
    [(False, 1, 8), (True, 2, 4)],
    ids=["unsharded", "sequence_parallel"],
)
def test_mamba_static_thd_inputs_include_local_packed_seq_idx(
    sequence_parallel: bool, tensor_model_parallel_size: int, expected_hidden_tokens: int
) -> None:
    """Static TE capture keeps the packed map at the pre-sequence-parallel token bound."""
    layer = _make_static_thd_mamba_layer(
        sequence_parallel=sequence_parallel, tensor_model_parallel_size=tensor_model_parallel_size
    )

    static_inputs = layer.get_layer_static_inputs(seq_length=23, micro_batch_size=3)

    assert static_inputs["hidden_states"].shape == (expected_hidden_tokens, 1, 16)
    assert static_inputs["packed_seq_idx"].shape == (1, 8)
    assert static_inputs["packed_seq_idx"].dtype == torch.int32
    assert not static_inputs["packed_seq_idx"].any()
    assert all(isinstance(value, torch.Tensor) for value in static_inputs.values())


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mamba_local_graph_static_inputs_do_not_add_te_packed_seq_idx() -> None:
    """The packed tensor adapter must not alter the existing local CUDA Graph interface."""
    layer = _make_static_thd_mamba_layer()
    layer.config.cuda_graph_impl = "local"

    static_inputs = layer.get_layer_static_inputs(seq_length=8, micro_batch_size=1)

    assert set(static_inputs) == {"hidden_states"}


@pytest.mark.internal
@pytest.mark.parametrize("cuda_graph_impl", ["none", "local"], ids=["eager", "local"])
def test_mamba_non_te_forward_preserves_public_packed_seq_params(cuda_graph_impl: str) -> None:
    """Eager and local configurations must pass the caller's metadata object through unchanged."""
    layer = object.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    layer.config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4)
    layer.config.cuda_graph_impl = cuda_graph_impl
    layer.norm = IdentityOp()
    observed: dict[str, object] = {}

    class Mixer(torch.nn.Module):

        def forward(
            self,
            hidden_states: torch.Tensor,
            *,
            inference_context: object | None,
            packed_seq_params: PackedSeqParams | None,
        ) -> tuple[torch.Tensor, None]:
            observed["packed_seq_params"] = packed_seq_params
            return hidden_states, None

    layer.mixer = Mixer()
    layer.mamba_bda = get_bias_dropout_add
    layer.hidden_dropout = 0.0
    layer.bias_dropout_add_exec_handler = torch.enable_grad
    packed_seq_params = PackedSeqParams(
        qkv_format="thd", seq_idx=torch.zeros((1, 8), dtype=torch.int32), total_tokens=8
    )
    hidden_states = torch.ones((8, 1, 16))

    output = layer(hidden_states, packed_seq_params=packed_seq_params)

    assert observed["packed_seq_params"] is packed_seq_params
    assert torch.equal(output, hidden_states * 2)


@pytest.mark.internal
def test_mamba_te_capture_reconstructs_minimal_packed_seq_params() -> None:
    """The production mixer must receive Python metadata rebuilt inside the graph boundary."""
    layer = _make_static_thd_mamba_layer()
    hidden_states = torch.ones((8, 1, 16))
    packed_seq_idx = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32)
    observed: dict[str, object] = {}

    def forward(hidden_states: torch.Tensor, *, packed_seq_params: PackedSeqParams) -> torch.Tensor:
        observed["packed_seq_params"] = packed_seq_params
        return hidden_states

    layer.forward = forward

    output = layer._te_cuda_graph_capture(hidden_states, packed_seq_idx=packed_seq_idx)

    assert output is hidden_states
    reconstructed = observed["packed_seq_params"]
    assert isinstance(reconstructed, PackedSeqParams)
    assert reconstructed.qkv_format == "thd"
    assert reconstructed.seq_idx is packed_seq_idx
    assert reconstructed.total_tokens == 8
    assert reconstructed.cu_seqlens_q is None
    assert reconstructed.cu_seqlens_kv is None


@pytest.mark.internal
def test_mamba_te_replay_forwards_only_tensorized_packed_seq_idx() -> None:
    """Replay must remove the runtime dataclass before entering the TE graph callable."""
    layer = _make_static_thd_mamba_layer()
    layer.cuda_graph_manual_hooks = []
    packed_seq_idx = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32)
    hidden_states = torch.ones((8, 1, 16))
    observed: dict[str, object] = {}

    def graph(
        hidden_states: torch.Tensor, *, packed_seq_idx: torch.Tensor, is_first_microbatch: bool
    ) -> torch.Tensor:
        observed["packed_seq_idx"] = packed_seq_idx
        observed["is_first_microbatch"] = is_first_microbatch
        return hidden_states + packed_seq_idx.transpose(0, 1).unsqueeze(-1)

    layer.cuda_graphs = [graph]
    packed_seq_params = PackedSeqParams(qkv_format="thd", seq_idx=packed_seq_idx, total_tokens=8)

    output = layer._te_cuda_graph_replay(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=packed_seq_params,
    )

    assert observed["packed_seq_idx"] is packed_seq_idx
    assert observed["is_first_microbatch"] is True
    assert torch.equal(output[:, 0, 0], torch.tensor([1, 1, 2, 2, 2, 3, 3, 3]))


@pytest.mark.internal
@pytest.mark.parametrize(
    "packed_seq_params,diagnostic",
    [
        (None, "requires packed_seq_params"),
        (
            PackedSeqParams(qkv_format="thd", seq_idx=[0] * 8, total_tokens=8),
            "seq_idx must be a torch.Tensor",
        ),
        (
            PackedSeqParams(
                qkv_format="thd", seq_idx=torch.zeros((1, 8), dtype=torch.int64), total_tokens=8
            ),
            "seq_idx must have dtype torch.int32",
        ),
        (
            PackedSeqParams(
                qkv_format="thd", seq_idx=torch.zeros((8,), dtype=torch.int32), total_tokens=8
            ),
            r"seq_idx must have shape \[1, 8\]",
        ),
        (
            PackedSeqParams(
                qkv_format="thd", seq_idx=torch.zeros((1, 9), dtype=torch.int32), total_tokens=8
            ),
            r"seq_idx must have shape \[1, 8\]",
        ),
    ],
    ids=["missing", "non_tensor", "wrong_dtype", "missing_batch_dim", "wrong_token_bound"],
)
def test_mamba_te_replay_rejects_invalid_packed_seq_idx(
    packed_seq_params: PackedSeqParams | None, diagnostic: str
) -> None:
    """Malformed runtime sequence metadata must fail before a stale graph can be replayed."""
    layer = _make_static_thd_mamba_layer()
    layer.cuda_graph_manual_hooks = []
    layer.cuda_graphs = [lambda hidden_states, **kwargs: hidden_states]
    kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}

    with pytest.raises((TypeError, ValueError), match=diagnostic):
        layer._te_cuda_graph_replay(torch.ones((8, 1, 16)), **kwargs)


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
            use_cpu_initialization=True,
        )
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.layer = MambaLayer(
            transformer_config,
            hybrid_stack_spec.submodules.mamba_layer.submodules,
            pg_collection=pg_collection,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

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
