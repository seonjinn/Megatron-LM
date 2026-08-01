# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

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
