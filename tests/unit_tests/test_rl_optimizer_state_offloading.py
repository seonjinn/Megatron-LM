# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for cyclic RL optimizer-state offload and reload."""

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.cpu_offloading.optimizer_state_offloader import OptimizerStateOffloader
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: F401

    TE_FUSED_ADAM_AVAILABLE = True
except ImportError:
    TE_FUSED_ADAM_AVAILABLE = False


class _TinyModel(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _create_model_and_optimizer():
    model = _TinyModel().bfloat16().cuda()
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        model,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            bf16=True,
            lr=1.0e-3,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
        ),
        [model],
    )
    return model, optimizer


def _train_step(model, optimizer) -> None:
    inputs = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_cyclic_rl_offload_preserves_state_tensor_and_parameter_identity():
    Utils.initialize_model_parallel()
    try:
        model, optimizer = _create_model_and_optimizer()
        _train_step(model, optimizer)
        assert isinstance(optimizer, ChainedOptimizer)
        distributed_optimizer = optimizer.chained_optimizers[0]
        offloader = OptimizerStateOffloader(distributed_optimizer)
        offloader.mark_optimizer_states_initialized()

        parameter_ids = [
            id(parameter)
            for group in distributed_optimizer.optimizer.param_groups
            for parameter in group["params"]
        ]
        state_tensor_ids = {
            (parameter, key): id(value)
            for parameter, state in distributed_optimizer.optimizer.state.items()
            for key, value in state.items()
            if key in (*offloader.OPTIMIZER_STATE_KEYS, offloader.MASTER_WEIGHT_KEY)
            and isinstance(value, torch.Tensor)
        }
        state_keys = {key for _, key in state_tensor_ids}
        assert parameter_ids
        assert {"exp_avg", "exp_avg_sq"} <= state_keys
        if offloader.optimizer_contains_master_weights:
            assert offloader.MASTER_WEIGHT_KEY in state_keys

        for _ in range(5):
            expected = {
                slot: distributed_optimizer.optimizer.state[slot[0]][slot[1]].clone()
                for slot in state_tensor_ids
            }
            offloader.offload()
            torch.cuda.synchronize()
            offloader.release_gpu_memory()

            assert offloader.is_offloaded
            for slot in state_tensor_ids:
                offloaded = distributed_optimizer.optimizer.state[slot[0]][slot[1]]
                assert offloaded.untyped_storage().size() == 0

            assert all(
                buffer.is_pinned()
                for buffers in offloader._opt_state_cpu_buffers.values()
                for buffer in buffers.values()
            )

            offloader.reload()
            offloader.sync_before_step()
            assert not offloader.is_offloaded
            for slot, tensor_id in state_tensor_ids.items():
                restored = distributed_optimizer.optimizer.state[slot[0]][slot[1]]
                assert id(restored) == tensor_id
                torch.testing.assert_close(restored, expected[slot])

        assert parameter_ids == [
            id(parameter)
            for group in distributed_optimizer.optimizer.param_groups
            for parameter in group["params"]
        ]
        _train_step(model, optimizer)
    finally:
        Utils.destroy_model_parallel()
