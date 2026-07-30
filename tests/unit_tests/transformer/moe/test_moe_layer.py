# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _packed_partial_cudagraph_moe_layer(
    *,
    cuda_graph_modules: list,
    shared_expert_overlap: bool = False,
) -> MoELayer:
    """Build the minimum MoE object needed to test replay output reconciliation."""
    moe_layer = MoELayer.__new__(MoELayer)
    moe_layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=cuda_graph_modules,
        moe_shared_expert_overlap=shared_expert_overlap,
        overlap_moe_expert_parallel_comm=False,
    )
    moe_layer.cudagraph_tensor_store = SimpleNamespace(is_packed_seq_replay=True)
    return moe_layer


def test_packed_partial_cudagraph_shared_expert_output_uses_logical_token_extent() -> None:
    """Packed router+preprocess replay must not add the captured padded tail."""
    from megatron.core.transformer.enums import CudaGraphModule

    moe_layer = _packed_partial_cudagraph_moe_layer(
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess]
    )
    routed_expert_output = torch.randn(12, 8, requires_grad=True)
    captured_shared_expert_output = torch.randn(16, 8, requires_grad=True)

    shared_expert_output = moe_layer._reconcile_packed_partial_cudagraph_shared_expert_output(
        routed_expert_output, captured_shared_expert_output
    )

    assert shared_expert_output.shape == routed_expert_output.shape
    assert shared_expert_output.data_ptr() == captured_shared_expert_output.data_ptr()
    (routed_expert_output + shared_expert_output).sum().backward()
    assert torch.equal(routed_expert_output.grad, torch.ones_like(routed_expert_output))
    assert torch.equal(
        captured_shared_expert_output.grad,
        torch.cat(
            [torch.ones_like(routed_expert_output), torch.zeros(4, 8)],
            dim=0,
        ),
    )


def test_packed_partial_cudagraph_shared_expert_output_rejects_invalid_replay() -> None:
    """Replay must reject incompatible hidden dimensions and insufficient capture capacity."""
    from megatron.core.transformer.enums import CudaGraphModule

    moe_layer = _packed_partial_cudagraph_moe_layer(
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess]
    )

    with pytest.raises(RuntimeError, match="trailing dimensions"):
        moe_layer._reconcile_packed_partial_cudagraph_shared_expert_output(
            torch.empty(12, 8), torch.empty(16, 7)
        )
    with pytest.raises(RuntimeError, match="captured capacity"):
        moe_layer._reconcile_packed_partial_cudagraph_shared_expert_output(
            torch.empty(17, 8), torch.empty(16, 8)
        )


def test_router_only_cudagraph_keeps_shared_expert_output_unchanged() -> None:
    """The packed-output reconciliation is intentionally limited to router+preprocess graphs."""
    from megatron.core.transformer.enums import CudaGraphModule

    moe_layer = _packed_partial_cudagraph_moe_layer(
        cuda_graph_modules=[CudaGraphModule.moe_router]
    )
    captured_shared_expert_output = torch.empty(16, 8)

    assert (
        moe_layer._reconcile_packed_partial_cudagraph_shared_expert_output(
            torch.empty(12, 8), captured_shared_expert_output
        )
        is captured_shared_expert_output
    )


class TestMoELayerInit:
    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_te_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_legacy_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.skip(
        "Late init of parallel_state was broken after parallel states refactor MR2988."
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["alltoall", "allgather"])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_moe_with_late_initialize(
        self, moe_token_dispatcher_type, grouped_gemm, tp_size, ep_size
    ):
        num_moe_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        moe_layer = MoELayer(transformer_config, submodules).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = torch.randn(
            16, 4, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:

    @pytest.mark.parametrize("moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")])
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            num_moe_experts=2,
            add_bias_linear=False,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_decoder_block_spec(self.transformer_config, False)
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerFP16:
    """Test MoE layer with FP16 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2), (4, 2)])
    def test_moe_layer_fp16_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, tp_size, ep_size
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=True,
            params_dtype=torch.float16,
        )

        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_moe_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        moe_layer = MoELayer(transformer_config, submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
            requires_grad=True,
        )

        # Forward pass
        output, _ = moe_layer(hidden_states)

        assert output.dtype == torch.float16, f"Expected fp16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.float16
        ), f"Expected fp16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerRecompute:
    """Test MoE layer with recompute enabled (activation checkpointing).

    Tests both code paths:
    - fp8=False: uses tensor_parallel.checkpoint
    - fp8=True: uses te_checkpoint (requires TE >= 1.7.0)
    """

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("with_padding_mask", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (4, 2)])
    @pytest.mark.parametrize("fp8", [False, True])
    def test_moe_layer_recompute_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, with_padding_mask, tp_size, ep_size, fp8
    ):
        """Test MoE layer forward and backward pass with recompute enabled.

        When fp8=False, uses tensor_parallel.checkpoint.
        When fp8=True, uses te_checkpoint (requires TE >= 1.7.0).
        """
        # Skip fp8 tests if TE version is not sufficient
        if fp8 and not is_te_min_version("1.7.0.dev0"):
            pytest.skip("FP8 MoE recompute requires TE 1.7.0 and later.")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            # Enable recompute for MoE layer
            recompute_granularity="selective",
            recompute_modules=["moe"],
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp8=fp8,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Use TE spec for fp8, local spec otherwise
        if fp8:
            submodules = get_submodules(
                get_gpt_layer_with_transformer_engine_submodules(
                    num_experts=num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
        else:
            submodules = get_submodules(
                get_gpt_layer_local_submodules(
                    num_experts=num_moe_experts, moe_grouped_gemm=False
                ).mlp
            )
        assert isinstance(submodules, MoESubmodules)

        moe_layer = MoELayer(transformer_config, submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Create padding mask if needed: shape [batch_size, sequence_length]
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.ones(
                micro_batch_size,
                sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.bool,
            )
            # Mark last 4 tokens as padding for each batch
            padding_mask[:, -4:] = False

        output, _ = moe_layer(hidden_states, padding_mask=padding_mask)

        assert output.dtype == torch.bfloat16, f"Expected bf16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass - this is where recompute/checkpoint is actually used
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.bfloat16
        ), f"Expected bf16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
