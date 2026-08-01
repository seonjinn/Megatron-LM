# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False

from tests.unit_tests.test_utilities import Utils


class _FakeCPGroup:
    """Minimal context-parallel group interface for THD RoPE tests."""

    def __init__(self, size: int = 1, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _PackedTHDRoPECaptureLayer(TransformerLayer):
    """Minimal attention-scope layer that exercises packed metadata reconstruction."""

    def __init__(self, config: TransformerConfig) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self.is_moe_layer = False

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        assert rotary_pos_emb is not None
        assert packed_seq_params is not None
        assert padding_mask is not None
        output = apply_rotary_pos_emb(
            hidden_states,
            rotary_pos_emb,
            self.config,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
            cp_group=_FakeCPGroup(),
            max_seqlen=packed_seq_params.max_seqlen_q,
        )
        metadata_bias = (
            packed_seq_params.cu_seqlens_kv[-1]
            + packed_seq_params.cu_seqlens_q_padded[1]
            + packed_seq_params.cu_seqlens_kv_padded[1]
        ).to(output.dtype)
        output = output + metadata_bias.reshape(1, 1, 1) * 0.01
        return output.masked_fill(padding_mask.T.unsqueeze(-1), 0.0), None


def _thd_rope_reference(
    tensor: torch.Tensor, freqs: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    """Tensor-only reference with literal packed positions derived by the test."""
    packed_freqs = freqs.index_select(0, position_ids).squeeze(1)
    rotary_dim = packed_freqs.shape[-1]
    rotary, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    first_half, second_half = torch.chunk(rotary, 2, dim=-1)
    rotated = torch.cat((-second_half, first_half), dim=-1)
    output = rotary * torch.cos(packed_freqs) + rotated * torch.sin(packed_freqs)
    return torch.cat((output, passthrough), dim=-1)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "mrope_section,freqs_are_packed,position_values",
    [
        pytest.param(None, True, [0, 1, 2, 3, 4, 5, 6, 7], id="explicit-packed"),
        pytest.param(None, False, [0, 1, 2, 3, 0, 1, 2, 3], id="explicit-reset"),
        pytest.param([1, 1, 2], None, [0, 1, 2, 3, 4, 5, 6, 7], id="mrope-default"),
        pytest.param([1, 1, 2], False, [0, 1, 2, 3, 0, 1, 2, 3], id="mrope-explicit-reset"),
    ],
)
def test_thd_rope_equal_length_freqs_use_declared_layout(
    mrope_section: list[int] | None, freqs_are_packed: bool | None, position_values: list[int]
) -> None:
    """An exact-length frequency table must not make packed and reset layouts ambiguous."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        apply_rope_fusion=False,
        mrope_section=mrope_section,
    )
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    freqs = torch.linspace(-0.7, 0.9, 8 * 8, device=device).reshape(8, 1, 1, 8)
    tensor = torch.linspace(-1.0, 1.0, 8 * 2 * 8, device=device).reshape(8, 2, 8)
    expected_positions = torch.tensor(position_values, dtype=torch.long, device=device)

    layout_kwargs = {} if freqs_are_packed is None else {"freqs_are_packed": freqs_are_packed}
    output = apply_rotary_pos_emb(
        tensor,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=_FakeCPGroup(),
        max_seqlen=8,
        **layout_kwargs,
    )

    assert torch.allclose(output, _thd_rope_reference(tensor, freqs, expected_positions))


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "freqs_are_packed,position_values",
    [
        pytest.param(True, [0, 1, 2, 3, 4, 5, 6, 7], id="explicit-packed"),
        pytest.param(False, [0, 1, 2, 3, 0, 1, 2, 3], id="explicit-reset"),
    ],
)
def test_fused_thd_rope_equal_length_freqs_use_declared_layout(
    freqs_are_packed: bool, position_values: list[int]
) -> None:
    """The public fused configuration must honor an explicitly declared THD layout."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=2, apply_rope_fusion=True
    )
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    freqs = torch.linspace(-0.7, 0.9, 8 * 8, device=device).reshape(8, 1, 1, 8)
    tensor = torch.linspace(-1.0, 1.0, 8 * 2 * 8, device=device).reshape(8, 2, 8)
    expected_positions = torch.tensor(position_values, dtype=torch.long, device=device)

    output = apply_rotary_pos_emb(
        tensor,
        freqs,
        config,
        cu_seqlens=cu_seqlens,
        cp_group=_FakeCPGroup(),
        max_seqlen=8,
        freqs_are_packed=freqs_are_packed,
    )

    assert torch.allclose(output, _thd_rope_reference(tensor, freqs, expected_positions))


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_thd_rope_cuda_graph_replays_equal_length_packed_freqs() -> None:
    """An explicit packed layout remains static while a captured RoPE graph replays."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=2, apply_rope_fusion=False
    )
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
    freqs = torch.linspace(-0.7, 0.9, 8 * 8, device=device).reshape(8, 1, 1, 8)
    capture_input = torch.zeros(8, 2, 8, device=device)
    replay_input = torch.linspace(-1.0, 1.0, capture_input.numel(), device=device).reshape_as(
        capture_input
    )
    packed_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=device)
    reset_positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device)

    def apply_thd_rope() -> torch.Tensor:
        return apply_rotary_pos_emb(
            capture_input,
            freqs,
            config,
            cu_seqlens=cu_seqlens,
            cp_group=_FakeCPGroup(),
            max_seqlen=8,
            freqs_are_packed=True,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        apply_thd_rope()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = apply_thd_rope()

    capture_input.copy_(replay_input)
    graph.replay()

    assert torch.allclose(
        captured_output, _thd_rope_reference(replay_input, freqs, packed_positions)
    )
    assert not torch.allclose(
        captured_output, _thd_rope_reference(replay_input, freqs, reset_positions)
    )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "cu_values,position_values,alternate_position_values",
    [
        ([0, 4, 8], [0, 1, 2, 3, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3, 4, 5, 6]),
        ([0, 3, 8], [0, 1, 2, 0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]),
    ],
)
def test_thd_rope_cuda_graph_accepts_max_seqlen_with_forward_and_gradient_parity(
    cu_values: list[int], position_values: list[int], alternate_position_values: list[int]
) -> None:
    """The public THD RoPE path must remain tensor-only during capture, including a padded tail."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=2, apply_rope_fusion=False
    )
    cp_group = _FakeCPGroup()
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)
    position_ids = torch.tensor(position_values, dtype=torch.long, device=device)
    alternate_position_ids = torch.tensor(
        alternate_position_values, dtype=torch.long, device=device
    )
    freqs = torch.linspace(-0.7, 0.9, 7 * 8, device=device).reshape(7, 1, 1, 8)
    base = torch.linspace(-1.0, 1.0, 10 * 2 * 8, device=device).reshape(10, 2, 8)

    def apply_thd_rope(tensor: torch.Tensor) -> torch.Tensor:
        return apply_rotary_pos_emb(
            tensor, freqs, config, cu_seqlens=cu_seqlens, cp_group=cp_group, max_seqlen=7
        )

    capture_input = base.clone()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        apply_thd_rope(capture_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = apply_thd_rope(capture_input)
    graph.replay()

    expected_output = _thd_rope_reference(base, freqs, position_ids)
    assert torch.allclose(captured_output, expected_output)
    assert not torch.allclose(
        captured_output, _thd_rope_reference(base, freqs, alternate_position_ids)
    )

    actual_input = base.clone().requires_grad_()
    reference_input = base.clone().requires_grad_()
    output_gradient = torch.linspace(-0.5, 0.5, base.numel(), device=device).reshape_as(base)
    actual_gradient = torch.autograd.grad(
        apply_thd_rope(actual_input), actual_input, grad_outputs=output_gradient
    )[0]
    reference_gradient = torch.autograd.grad(
        _thd_rope_reference(reference_input, freqs, position_ids),
        reference_input,
        grad_outputs=output_gradient,
    )[0]
    assert torch.allclose(actual_gradient, reference_gradient)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transformer_layer_thd_graph_replays_two_packs_with_forward_and_gradient_parity() -> None:
    """One attention graph must replay changed packed metadata in forward and backward."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        apply_rope_fusion=False,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn],
        max_seqlen_per_dp_cp_rank=7,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=2,
        thd_tail_padding_policy="extend_last",
    )
    layer = _PackedTHDRoPECaptureLayer(config)
    base_hidden = torch.linspace(-1.0, 1.0, 10 * 2 * 8, device=device).reshape(10, 2, 8)
    base_freqs = torch.linspace(-0.7, 0.9, 7 * 8, device=device).reshape(7, 1, 1, 8)
    output_gradient = torch.linspace(-0.5, 0.5, base_hidden.numel(), device=device).reshape_as(
        base_hidden
    )
    packs: list[tuple[list[int], list[int], list[bool], list[int], float]] = [
        ([0, 4, 8], [0, 4, 10], [False] * 8 + [True, True], [0, 1, 2, 3, 0, 1, 2, 3, 4, 5], 0.16),
        (
            [0, 3, 8],
            [0, 5, 10],
            [False, False, False, True, False, False, False, False, False, True],
            [0, 1, 2, 0, 1, 2, 3, 4, 5, 6],
            0.18,
        ),
    ]

    first_compact, first_padded, first_mask, _, _ = packs[0]
    static_hidden = base_hidden.clone().requires_grad_()
    static_freqs = base_freqs.clone()
    static_cu_q = torch.tensor(first_compact, dtype=torch.int32, device=device)
    static_cu_kv = static_cu_q.clone()
    static_cu_q_padded = torch.tensor(first_padded, dtype=torch.int32, device=device)
    static_cu_kv_padded = static_cu_q_padded.clone()
    static_padding_mask = torch.tensor(first_mask, dtype=torch.bool, device=device).reshape(1, -1)

    def run_static_step() -> torch.Tensor:
        return layer._te_cuda_graph_capture(
            static_hidden,
            rotary_pos_emb=static_freqs,
            cu_seqlens_q=static_cu_q,
            cu_seqlens_kv=static_cu_kv,
            cu_seqlens_q_padded=static_cu_q_padded,
            cu_seqlens_kv_padded=static_cu_kv_padded,
            padding_mask=static_padding_mask,
        )[0]

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            static_hidden.grad = None
            run_static_step().backward(output_gradient)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    static_hidden.grad = torch.zeros_like(static_hidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_hidden.grad.zero_()
        captured_output = run_static_step()
        captured_output.backward(output_gradient)
    captured_gradient = static_hidden.grad

    replay_outputs: list[torch.Tensor] = []
    for compact_values, padded_values, mask_values, position_values, metadata_bias in packs:
        compact = torch.tensor(compact_values, dtype=torch.int32, device=device)
        padded = torch.tensor(padded_values, dtype=torch.int32, device=device)
        padding_mask = torch.tensor(mask_values, dtype=torch.bool, device=device).reshape(1, -1)
        with torch.no_grad():
            static_hidden.copy_(base_hidden)
            static_freqs.copy_(base_freqs)
            static_cu_q.copy_(compact)
            static_cu_kv.copy_(compact)
            static_cu_q_padded.copy_(padded)
            static_cu_kv_padded.copy_(padded)
            static_padding_mask.copy_(padding_mask)
        graph.replay()

        actual_output = captured_output.clone()
        actual_gradient = captured_gradient.clone()
        reference_input = base_hidden.clone().requires_grad_()
        positions = torch.tensor(position_values, dtype=torch.long, device=device)
        reference_output = _thd_rope_reference(reference_input, base_freqs, positions)
        reference_output = reference_output + metadata_bias
        reference_output = reference_output.masked_fill(padding_mask.T.unsqueeze(-1), 0.0)
        reference_gradient = torch.autograd.grad(
            reference_output, reference_input, grad_outputs=output_gradient
        )[0]

        assert torch.allclose(actual_output, reference_output)
        assert torch.allclose(actual_gradient, reference_gradient)
        replay_outputs.append(actual_output)

    assert not torch.allclose(replay_outputs[0], replay_outputs[1])


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_thd_rope_odd_local_cp_segment_uses_ceil_floor_partition() -> None:
    """An odd CP-local sequence keeps the extra token in the forward segment."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=2, apply_rope_fusion=False
    )
    cp_group = _FakeCPGroup(size=2, rank=0)
    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32, device=device)
    expected_positions = torch.tensor([0, 1, 5], dtype=torch.long, device=device)
    floor_only_positions = torch.tensor([0, 5, 5], dtype=torch.long, device=device)
    freqs = torch.linspace(-0.7, 0.9, 6 * 8, device=device).reshape(6, 1, 1, 8)
    tensor = torch.linspace(-1.0, 1.0, 3 * 2 * 8, device=device).reshape(3, 2, 8)

    output = apply_rotary_pos_emb(
        tensor, freqs, config, cu_seqlens=cu_seqlens, cp_group=cp_group, max_seqlen=6
    )

    assert torch.allclose(output, _thd_rope_reference(tensor, freqs, expected_positions))
    assert not torch.allclose(output, _thd_rope_reference(tensor, freqs, floor_only_positions))


class TestMultimodalRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = MultimodalRotaryEmbedding(self.kv_channels, self.rotary_percent)

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, MultimodalRotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(torch.Tensor(3, 1, 64), mrope_section=[16, 24, 24])
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope_cpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=True
        )
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        del self.rope_cpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_cpu_init, RotaryEmbedding)
        assert self.rope_cpu_init.inv_freq.device.type == 'cpu'
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        output = self.rope_gpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_forward(self):
        output = self.rope_cpu_init(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestQKVRotaryEmbedding:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.seq_len = 64
        self.num_heads = 1
        self.kv_channels = 128
        self.rotary_percent = 1.0
        self.rope_gpu_init = RotaryEmbedding(
            self.kv_channels, self.rotary_percent, use_cpu_initialization=False
        )
        self.transformer_config = TransformerConfig(
            num_attention_heads=self.num_heads, num_layers=1, apply_rope_fusion=True
        )

    def teardown_method(self, method):
        del self.rope_gpu_init
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_constructor(self):
        assert isinstance(self.rope_gpu_init, RotaryEmbedding)
        assert self.rope_gpu_init.inv_freq.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAVE_FUSED_QKV_ROPE, reason="Fused QKV RoPE not available.")
    def test_gpu_forward(self):
        pos_embed = self.rope_gpu_init(self.seq_len)
        assert pos_embed.shape[0] == self.seq_len
        assert pos_embed.shape[1] == 1
        assert pos_embed.shape[2] == 1
        assert pos_embed.shape[3] == self.kv_channels
        assert pos_embed.dtype == torch.float32
        assert pos_embed.device.type == 'cuda'

        qkv_split_arg_list = [self.kv_channels * 4, self.kv_channels, self.kv_channels]
        # Create input tensors
        qkv = torch.randn(self.seq_len, 1, self.num_heads, self.kv_channels * 6, device="cuda")
        (query_in, key_in, value_in) = torch.split(qkv, qkv_split_arg_list, dim=3)

        query_in = query_in.reshape(query_in.shape[0], query_in.shape[1], -1, self.kv_channels)
        q_out_ref = apply_rotary_pos_emb(query_in, pos_embed, self.transformer_config)
        k_out_ref = apply_rotary_pos_emb(key_in, pos_embed, self.transformer_config)
        q_out, k_out, _ = apply_fused_qkv_rotary_pos_emb(
            qkv, pos_embed, pos_embed, qkv_split_arg_list
        )

        assert (
            q_out_ref.numel() == q_out.numel()
        ), f"Output sizes do not match for Q: {q_out.shape} != {q_out_ref.shape}"
        assert (
            k_out_ref.numel() == k_out.numel()
        ), f"Output sizes do not match for K: {k_out.shape} != {k_out_ref.shape}"
        assert torch.allclose(q_out_ref, q_out), f"Outputs do not match for Q"
        assert torch.allclose(k_out_ref, k_out), f"Outputs do not match for K"
