# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
import types
from collections.abc import Callable

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
    _mtp_logits_are_vocab_sharded,
    process_mtp_loss,
    roll_tensor,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank, is_te_min_version, unwrap_model
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
else:
    TEColumnParallelGroupedLinear = None

_SEED = 42


class TestMultiTokenPredictionLayer:
    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    def _create_config_and_mtp_block_spec(self, tp, cp, use_te=False, mtp_num_layers=2):
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        config = TransformerConfig(
            mtp_num_layers=mtp_num_layers,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=True if tp > 1 else False,
            context_parallel_size=cp,  # Enable CP for MTP testing
        )
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        else:
            transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=use_te
        )
        return config, mtp_block_spec

    def test_mtp_detach_heads_config(self):
        """Test that mtp_detach_heads config defaults to False."""
        config = TransformerConfig(
            num_layers=4, hidden_size=64, num_attention_heads=8, use_cpu_initialization=True
        )
        assert config.mtp_detach_heads is False

    def test_constructor_with_detach_heads(self):
        """Test construction of MTP module with mtp_detach_heads=True."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            mtp_detach_heads=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=False
        )
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert mtp.config.mtp_detach_heads is True

        # Verify all parameters are tagged for separate MTP grad-norm handling.
        for name, param in mtp.named_parameters():
            assert (
                getattr(param, 'grad_norm_group', None) == 'mtp'
            ), f"Parameter {name} missing grad_norm_group attribute"

    @pytest.mark.parametrize(('tp'), [(1), (2), (4)])
    def test_constructor_local(self, tp):
        """Test basic construction of MTP module."""

        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.mtp_num_layers
        for i in range(config.mtp_num_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size / tp
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size * 2
            assert mtp.layers[i].mtp_model_layer is not None
        num_weights = sum([p.numel() for p in mtp.parameters()])
        if tp == 1:
            assert num_weights == 58560 * config.mtp_num_layers
        elif tp == 2:
            assert num_weights == 29664 * config.mtp_num_layers
        elif tp == 4:
            assert num_weights == 15216 * config.mtp_num_layers

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(('tp', 'cp'), [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_constructor_ues_te(self, tp, cp):
        """Test basic construction of MTP module."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp, cp, use_te=True)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.mtp_num_layers
        for i in range(config.mtp_num_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size / tp
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size * 2
            assert mtp.layers[i].mtp_model_layer is not None
        num_weights = sum([p.numel() for p in mtp.parameters()])
        if tp == 1:
            assert num_weights == 58560 * config.mtp_num_layers
        elif tp == 2:
            assert num_weights == 29664 * config.mtp_num_layers
        elif tp == 4:
            assert num_weights == 15216 * config.mtp_num_layers

    def test_get_embeddings_rolls_padding_mask(self):
        """Test that _get_embeddings rolls padding_mask alongside input ids."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 6
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 0, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        padding_mask = torch.tensor(
            [[False, False, False, False, True, True], [False, False, False, True, True, True]]
        )
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)

        def fake_embedding(input_ids, position_ids):
            return torch.zeros(seq_len, batch_size, config.hidden_size, dtype=hidden_states.dtype)

        rolled_input_ids, rolled_position_ids, rolled_padding_mask, _, _ = (
            mtp_layer._get_embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                padding_mask=padding_mask,
                embedding=fake_embedding,
                hidden_states=hidden_states,
                packed_seq_params=None,
            )
        )

        expected_input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
        expected_position_ids, _ = roll_tensor(position_ids, shifts=-1, dims=-1)
        expected_padding_mask = torch.tensor(
            [[False, False, False, True, True, True], [False, False, True, True, True, True]]
        )

        assert torch.equal(rolled_input_ids, expected_input_ids)
        assert torch.equal(rolled_position_ids, expected_position_ids)
        assert torch.equal(rolled_padding_mask, expected_padding_mask)
        assert rolled_padding_mask[:, -1].all()
        assert rolled_padding_mask[0, 3:].all()
        assert rolled_padding_mask[1, 2:].all()

    def test_forward_propagates_rolled_padding_mask(self, monkeypatch):
        """Test forward passes rolled padding_mask to transformer path."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)
        seen = {}

        def fake_embedding(input_ids, position_ids):
            return torch.zeros(seq_len, batch_size, config.hidden_size, dtype=hidden_states.dtype)

        def fake_proj_and_transformer_layer(
            self,
            hidden_states,
            decoder_input,
            attention_mask=None,
            padding_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            inference_params=None,
            packed_seq_params=None,
            sequence_len_offset=None,
        ):
            seen["padding_mask"] = padding_mask
            return hidden_states

        monkeypatch.setattr(
            mtp_layer,
            "_proj_and_transformer_layer",
            types.MethodType(fake_proj_and_transformer_layer, mtp_layer),
        )

        _, _, _, returned_padding_mask = mtp_layer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            embedding=fake_embedding,
        )

        expected_padding_mask = torch.tensor(
            [[False, False, True, True], [False, True, True, True]]
        )
        assert torch.equal(seen["padding_mask"], expected_padding_mask)
        assert torch.equal(returned_padding_mask, expected_padding_mask)
        assert seen["padding_mask"][:, -1].all()

    def test_get_embeddings_keeps_packed_segment_and_capacity_tails_padded(self) -> None:
        """Packed MTP shifts cannot revive per-segment or fixed-capacity padding."""

        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        mtp_layer = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec).layers[0]
        sample_ids = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            dtype=torch.int64,
        )
        num_samples = torch.tensor(2, dtype=torch.int64)
        original_sample_ids = sample_ids.clone()
        original_num_samples = num_samples.clone()
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 4, 12, 16], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 4, 12, 16], dtype=torch.int32),
            seq_aux_loss_sample_ids=sample_ids,
            seq_aux_loss_num_samples=num_samples,
            seq_aux_loss_max_samples=3,
        )
        padding_mask = torch.tensor(
            [
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ]
        )
        hidden_states = torch.randn(16, 1, config.hidden_size)

        def fake_embedding(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            return torch.zeros(16, 1, config.hidden_size, dtype=hidden_states.dtype)

        _, _, rolled_padding_mask, _, _ = mtp_layer._get_embeddings(
            input_ids=torch.arange(16, dtype=torch.int64).unsqueeze(0),
            position_ids=torch.arange(16, dtype=torch.int64).unsqueeze(0),
            padding_mask=padding_mask,
            embedding=fake_embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        expected_padding_mask = torch.tensor(
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ]
        )
        assert torch.equal(rolled_padding_mask, expected_padding_mask)
        assert rolled_padding_mask[0, 2:4].all()
        assert rolled_padding_mask[0, 8:12].all()
        assert rolled_padding_mask[0, 12:].all()
        assert packed_seq_params.seq_aux_loss_sample_ids is sample_ids
        assert packed_seq_params.seq_aux_loss_num_samples is num_samples
        assert packed_seq_params.seq_aux_loss_max_samples == 3
        assert torch.equal(sample_ids, original_sample_ids)
        assert torch.equal(num_samples, original_num_samples)

    @pytest.mark.parametrize("mtp_depth", (1, 2))
    def test_normal_mtp_rolls_token_inputs_without_rolling_packed_ownership(
        self, monkeypatch, mtp_depth
    ):
        """Every normal MTP depth rolls token inputs but preserves ownership identity."""

        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(
            tp=1, cp=1, mtp_num_layers=mtp_depth
        )
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        sample_ids = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            dtype=torch.int64,
        )
        num_samples = torch.tensor(2, dtype=torch.int64)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 4, 12, 16], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 4, 12, 16], dtype=torch.int32),
            seq_aux_loss_sample_ids=sample_ids,
            seq_aux_loss_num_samples=num_samples,
            seq_aux_loss_max_samples=3,
        )
        original_sample_ids = sample_ids.clone()
        original_num_samples = num_samples.clone()
        input_ids = torch.tensor(
            [[10, 11, 12, 0, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int64,
        )
        position_ids = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]],
            dtype=torch.int64,
        )
        padding_mask = torch.tensor(
            [
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ]
        )
        hidden_states = torch.randn(16, 1, config.hidden_size)
        embedding_calls = []
        packed_calls = []

        def recording_embedding(input_ids, position_ids):
            embedding_calls.append((input_ids.clone(), position_ids.clone()))
            return torch.zeros(16, 1, config.hidden_size, dtype=hidden_states.dtype)

        def record_transformer_inputs(
            self,
            hidden_states,
            decoder_input,
            attention_mask=None,
            padding_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            inference_params=None,
            packed_seq_params=None,
            sequence_len_offset=None,
        ):
            packed_calls.append((packed_seq_params, padding_mask.clone()))
            return hidden_states

        for layer in mtp.layers:
            monkeypatch.setattr(
                layer,
                "_proj_and_transformer_layer",
                types.MethodType(record_transformer_inputs, layer),
            )

        output = mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=torch.ones((1, 1, 16, 16), dtype=torch.bool),
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
            embedding=recording_embedding,
        )

        expected_input_ids = (
            torch.tensor(
                [[11, 12, 0, 0, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int64,
            ),
            torch.tensor(
                [[12, 0, 0, 0, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int64,
            ),
        )
        expected_position_ids = (
            torch.tensor(
                [[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0]],
                dtype=torch.int64,
            ),
            torch.tensor(
                [[2, 3, 0, 0, 2, 3, 4, 5, 6, 7, 0, 0, 2, 3, 0, 0]],
                dtype=torch.int64,
            ),
        )
        expected_padding_masks = (
            torch.tensor(
                [
                    [
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ]
            ),
        )
        assert len(embedding_calls) == len(packed_calls) == mtp_depth
        for depth in range(mtp_depth):
            torch.testing.assert_close(embedding_calls[depth][0], expected_input_ids[depth])
            torch.testing.assert_close(embedding_calls[depth][1], expected_position_ids[depth])
            assert packed_calls[depth][0] is packed_seq_params
            torch.testing.assert_close(packed_calls[depth][1], expected_padding_masks[depth])
            assert packed_calls[depth][0].seq_aux_loss_sample_ids is sample_ids
            assert packed_calls[depth][0].seq_aux_loss_num_samples is num_samples
            assert packed_calls[depth][0].seq_aux_loss_max_samples == 3
            assert packed_calls[depth][1][0, 12:].all()
        assert torch.equal(sample_ids, original_sample_ids)
        assert torch.equal(num_samples, original_num_samples)
        assert packed_seq_params.seq_aux_loss_sample_ids is sample_ids
        assert output.shape == (16 * (1 + mtp_depth), 1, config.hidden_size)

    @pytest.mark.parametrize("tp_rank", (0, 1))
    def test_sequence_parallel_get_embeddings_gathers_padding_mask_before_packed_roll(
        self, monkeypatch: pytest.MonkeyPatch, tp_rank: int
    ) -> None:
        """MTP rolls a reconstructed TP sequence, then restores the local SP shard."""

        tp_group = object()
        cp_group = types.SimpleNamespace(size=lambda: 1)
        mtp_layer = types.SimpleNamespace(
            sequence_parallel=True,
            tp_group=tp_group,
            cp_group=cp_group,
            config=types.SimpleNamespace(mtp_detach_heads=False),
        )
        input_ids = torch.tensor(
            [[10, 11, 12, 0, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int64,
        )
        position_ids = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]],
            dtype=torch.int64,
        )
        full_padding_masks = (
            torch.tensor(
                [
                    [
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ]
            ),
        )
        expected_input_ids = (
            torch.tensor(
                [[11, 12, 0, 0, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int64,
            ),
            torch.tensor(
                [[12, 0, 0, 0, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int64,
            ),
        )
        expected_position_ids = (
            torch.tensor(
                [[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0]],
                dtype=torch.int64,
            ),
            torch.tensor(
                [[2, 3, 0, 0, 2, 3, 4, 5, 6, 7, 0, 0, 2, 3, 0, 0]],
                dtype=torch.int64,
            ),
        )
        shard_start = tp_rank * 8
        padding_mask = full_padding_masks[0].narrow(1, shard_start, 8).contiguous()
        sample_ids = (
            torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                dtype=torch.int64,
            )
            .narrow(0, shard_start, 8)
            .contiguous()
        )
        num_samples = torch.tensor(2, dtype=torch.int64)
        original_sample_ids = sample_ids.clone()
        original_num_samples = num_samples.clone()
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 4, 12, 16], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 4, 12, 16], dtype=torch.int32),
            seq_aux_loss_sample_ids=sample_ids,
            seq_aux_loss_num_samples=num_samples,
            seq_aux_loss_max_samples=3,
        )
        hidden_states = torch.zeros(8, 1, 4)
        collective_order = []

        def fake_gather(
            local_mask_sequence_first: torch.Tensor,
            tensor_parallel_output_grad: bool = True,
            group: object | None = None,
        ) -> torch.Tensor:
            depth = collective_order.count("tp_gather")
            assert tensor_parallel_output_grad is False
            assert group is tp_group
            torch.testing.assert_close(
                local_mask_sequence_first,
                full_padding_masks[depth]
                .narrow(1, shard_start, 8)
                .transpose(0, 1)
                .contiguous(),
            )
            collective_order.append("tp_gather")
            return full_padding_masks[depth].transpose(0, 1).contiguous()

        def fake_scatter(
            full_mask_sequence_first: torch.Tensor, group: object | None = None
        ) -> torch.Tensor:
            depth = collective_order.count("tp_scatter")
            assert group is tp_group
            torch.testing.assert_close(
                full_mask_sequence_first,
                full_padding_masks[depth + 1].transpose(0, 1).contiguous(),
            )
            collective_order.append("tp_scatter")
            return full_mask_sequence_first.narrow(0, shard_start, 8).contiguous()

        mtp_module = sys.modules[MultiTokenPredictionLayer.__module__]
        monkeypatch.setattr(
            mtp_module, "gather_from_sequence_parallel_region", fake_gather, raising=False
        )
        monkeypatch.setattr(mtp_module, "scatter_to_sequence_parallel_region", fake_scatter)

        def fake_embedding(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            return torch.zeros_like(hidden_states)

        for depth in range(2):
            input_ids, position_ids, padding_mask, _, hidden_states = (
                MultiTokenPredictionLayer._get_embeddings(
                    mtp_layer,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    padding_mask=padding_mask,
                    embedding=fake_embedding,
                    hidden_states=hidden_states,
                    packed_seq_params=packed_seq_params,
                )
            )
            torch.testing.assert_close(input_ids, expected_input_ids[depth])
            torch.testing.assert_close(position_ids, expected_position_ids[depth])
            torch.testing.assert_close(
                padding_mask,
                full_padding_masks[depth + 1].narrow(1, shard_start, 8),
            )

        assert collective_order == ["tp_gather", "tp_scatter"] * 2
        assert full_padding_masks[1][0, [3, 11, 15]].all()
        assert full_padding_masks[2][0, [3, 11, 15]].all()
        assert full_padding_masks[1][0, 12:].all()
        assert full_padding_masks[2][0, 12:].all()
        assert packed_seq_params.seq_aux_loss_sample_ids is sample_ids
        assert packed_seq_params.seq_aux_loss_num_samples is num_samples
        assert packed_seq_params.seq_aux_loss_max_samples == 3
        assert torch.equal(sample_ids, original_sample_ids)
        assert torch.equal(num_samples, original_num_samples)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "cp_size",
        (
            pytest.param(1, id="tp2_cp1_sp"),
            pytest.param(2, id="tp2_cp2_sp"),
        ),
    )
    def test_sequence_parallel_get_embeddings_padding_mask_tp_cp_cuda(
        self, cp_size: int
    ) -> None:
        """Real TP/CP collectives preserve packed MTP boundaries for two depths."""

        required_world_size = 2 * cp_size
        world_size = Utils.world_size
        if world_size < required_world_size or world_size % required_world_size:
            pytest.skip(
                f"requires a world size divisible by TP*CP={required_world_size}, got {world_size}"
            )

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2,
            context_parallel_size=cp_size,
        )
        world_size = torch.distributed.get_world_size()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_rank = torch.distributed.get_rank(group=pg_collection.tp)
        cp_rank = torch.distributed.get_rank(group=pg_collection.cp)
        device = torch.device("cuda", torch.cuda.current_device())
        mtp_layer = types.SimpleNamespace(
            sequence_parallel=True,
            tp_group=pg_collection.tp,
            cp_group=pg_collection.cp,
            config=types.SimpleNamespace(mtp_detach_heads=False),
        )

        global_input_ids = torch.tensor(
            [[10, 11, 12, 0, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int64,
            device=device,
        )
        global_position_ids = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]],
            dtype=torch.int64,
            device=device,
        )
        global_padding_masks = (
            torch.tensor(
                [
                    [
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ],
                device=device,
            ),
            torch.tensor(
                [
                    [
                        False,
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ],
                device=device,
            ),
            torch.tensor(
                [
                    [
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                ],
                device=device,
            ),
        )
        expected_global_input_ids = (
            torch.tensor(
                [[11, 12, 0, 0, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [[12, 0, 0, 0, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.int64,
                device=device,
            ),
        )
        expected_global_position_ids = (
            torch.tensor(
                [[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0]],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [[2, 3, 0, 0, 2, 3, 4, 5, 6, 7, 0, 0, 2, 3, 0, 0]],
                dtype=torch.int64,
                device=device,
            ),
        )

        cp_indices = []
        for start, end in ((0, 4), (4, 12), (12, 16)):
            shard_width = (end - start) // (2 * cp_size)
            first_start = start + cp_rank * shard_width
            mirrored_rank = 2 * cp_size - cp_rank - 1
            second_start = start + mirrored_rank * shard_width
            cp_indices.extend(range(first_start, first_start + shard_width))
            cp_indices.extend(range(second_start, second_start + shard_width))
        cp_indices_tensor = torch.tensor(cp_indices, dtype=torch.int64, device=device)
        cp_sequence_length = len(cp_indices)
        sp_sequence_length = cp_sequence_length // 2
        sp_start = tp_rank * sp_sequence_length

        input_ids = global_input_ids.index_select(1, cp_indices_tensor)
        position_ids = global_position_ids.index_select(1, cp_indices_tensor)
        padding_mask = (
            global_padding_masks[0]
            .index_select(1, cp_indices_tensor)
            .narrow(1, sp_start, sp_sequence_length)
            .contiguous()
        )
        global_sample_ids = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            dtype=torch.int64,
            device=device,
        )
        sample_ids = (
            global_sample_ids.index_select(0, cp_indices_tensor)
            .narrow(0, sp_start, sp_sequence_length)
            .contiguous()
        )
        num_samples = torch.tensor(2, dtype=torch.int64, device=device)
        original_sample_ids = sample_ids.clone()
        original_num_samples = num_samples.clone()
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 4, 12, 16], dtype=torch.int32, device=device),
            cu_seqlens_kv=torch.tensor([0, 4, 12, 16], dtype=torch.int32, device=device),
            seq_aux_loss_sample_ids=sample_ids,
            seq_aux_loss_num_samples=num_samples,
            seq_aux_loss_max_samples=3,
        )
        hidden_states = torch.zeros(sp_sequence_length, 1, 4, device=device)

        def fake_embedding(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            return torch.zeros_like(hidden_states)

        actual_depth_outputs: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        expected_depth_outputs: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for depth in range(2):
            input_ids, position_ids, padding_mask, _, hidden_states = (
                MultiTokenPredictionLayer._get_embeddings(
                    mtp_layer,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    padding_mask=padding_mask,
                    embedding=fake_embedding,
                    hidden_states=hidden_states,
                    packed_seq_params=packed_seq_params,
                )
            )
            expected_input_ids = expected_global_input_ids[depth].index_select(
                1, cp_indices_tensor
            )
            expected_position_ids = expected_global_position_ids[depth].index_select(
                1, cp_indices_tensor
            )
            expected_padding_mask = (
                global_padding_masks[depth + 1]
                .index_select(1, cp_indices_tensor)
                .narrow(1, sp_start, sp_sequence_length)
            )
            actual_depth_outputs.append((input_ids, position_ids, padding_mask))
            expected_depth_outputs.append(
                (expected_input_ids, expected_position_ids, expected_padding_mask)
            )

        local_failures: list[str] = []

        def record_close(
            name: str, actual: torch.Tensor, expected: torch.Tensor
        ) -> None:
            try:
                torch.testing.assert_close(actual, expected)
            except Exception as error:
                local_failures.append(f"{name}: {error}")

        def record_predicate(
            name: str, predicate: Callable[[], bool | torch.Tensor]
        ) -> None:
            try:
                result = predicate()
                passed = bool(result.item()) if isinstance(result, torch.Tensor) else result
                if not passed:
                    local_failures.append(name)
            except Exception as error:
                local_failures.append(f"{name}: {error}")

        local_global_indices = cp_indices_tensor.narrow(
            0, sp_start, sp_sequence_length
        )
        segment_end_indices = torch.tensor([3, 11, 15], dtype=torch.int64, device=device)
        segment_end_mask = torch.any(
            local_global_indices.unsqueeze(-1) == segment_end_indices, dim=-1
        )
        capacity_tail_mask = local_global_indices >= 12
        for depth, (actual_output, expected_output) in enumerate(
            zip(actual_depth_outputs, expected_depth_outputs), start=1
        ):
            actual_input_ids, actual_position_ids, actual_padding_mask = actual_output
            expected_input_ids, expected_position_ids, expected_padding_mask = expected_output
            record_close(f"depth {depth} input ids", actual_input_ids, expected_input_ids)
            record_close(
                f"depth {depth} position ids", actual_position_ids, expected_position_ids
            )
            record_close(
                f"depth {depth} padding mask", actual_padding_mask, expected_padding_mask
            )
            record_predicate(
                f"depth {depth} physical segment end is not padded",
                lambda mask=actual_padding_mask: mask[0, segment_end_mask].all(),
            )
            record_predicate(
                f"depth {depth} capacity tail is not padded",
                lambda mask=actual_padding_mask: mask[0, capacity_tail_mask].all(),
            )

        for depth, global_padding_mask in enumerate(global_padding_masks[1:], start=1):
            record_predicate(
                f"depth {depth} literal oracle segment end is not padded",
                lambda mask=global_padding_mask: mask[0, [3, 11, 15]].all(),
            )
            record_predicate(
                f"depth {depth} literal oracle capacity tail is not padded",
                lambda mask=global_padding_mask: mask[0, 12:].all(),
            )
        record_predicate(
            "packed sample ids object identity changed",
            lambda: packed_seq_params.seq_aux_loss_sample_ids is sample_ids,
        )
        record_predicate(
            "packed num samples object identity changed",
            lambda: packed_seq_params.seq_aux_loss_num_samples is num_samples,
        )
        record_predicate(
            "packed max samples changed",
            lambda: packed_seq_params.seq_aux_loss_max_samples == 3,
        )
        record_predicate(
            "packed sample ids values changed",
            lambda: torch.equal(sample_ids, original_sample_ids),
        )
        record_predicate(
            "packed num samples value changed",
            lambda: torch.equal(num_samples, original_num_samples),
        )

        failure_flag = torch.tensor(
            1 if local_failures else 0, dtype=torch.int32, device=device
        )
        torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.MAX)
        if failure_flag.item():
            rank = torch.distributed.get_rank()
            details = (
                "; ".join(local_failures)
                if local_failures
                else "local invariants passed; another rank reported a mismatch"
            )
            pytest.fail(
                f"distributed MTP mask parity failed on rank {rank}/{world_size}: {details}",
                pytrace=False,
            )

    def test_get_embeddings_detaches_decoder_input(self):
        """With mtp_detach_heads=True, _get_embeddings detaches decoder_input (severing
        gradient flow to the shared embedding) while still returning a hidden_states
        tensor that requires grad so MTP layer params and activation checkpointing work."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        config.mtp_detach_heads = True
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        # hidden_states arrives without requires_grad (it is detached upstream by the block).
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        emb_weight = torch.nn.Parameter(torch.randn(seq_len, batch_size, config.hidden_size))

        def fake_embedding(input_ids, position_ids):
            return emb_weight.clone()

        _, _, _, decoder_input, returned_hidden_states = mtp_layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=fake_embedding,
            hidden_states=hidden_states,
            packed_seq_params=None,
        )

        # decoder_input is detached from the embedding graph.
        assert decoder_input.requires_grad is False
        assert decoder_input.grad_fn is None
        # hidden_states is still marked requires_grad so checkpointing and the MTP
        # layer parameters keep a differentiable path.
        assert returned_hidden_states.requires_grad is True

    @pytest.mark.parametrize(
        ("embedding_scatters", "expect_scatter"), [(False, True), (True, False), (None, False)]
    )
    def test_get_embeddings_scatters_when_embedding_does_not(
        self, monkeypatch, embedding_scatters, expect_scatter
    ):
        """Multimodal language models build LanguageModelEmbedding with
        scatter_to_sequence_parallel=False so media features can be inserted into a
        full-length embedding, and their outer forward performs the sequence-parallel
        scatter afterwards. _get_embeddings calls the embedding directly, so it has to
        apply the same scatter; otherwise decoder_input stays full length while the
        backbone hidden_states arrive sequence-parallel sharded and _concat_embeddings
        fails to concatenate them.

        embedding_scatters=None covers a plain callable with no such attribute, which
        must keep the previous behaviour of not scattering.
        """
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        config.sequence_parallel = True
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)
        mtp_layer = mtp.layers[0]

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1)
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        embeddings = torch.randn(seq_len, batch_size, config.hidden_size)

        def embedding(input_ids, position_ids):
            return embeddings.clone()

        if embedding_scatters is not None:
            embedding.scatter_to_sequence_parallel = embedding_scatters

        scattered = []

        def fake_scatter(tensor, group=None):
            scattered.append(tensor)
            return tensor

        monkeypatch.setattr(
            "megatron.core.transformer.multi_token_prediction."
            "scatter_to_sequence_parallel_region",
            fake_scatter,
        )

        mtp_layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=None,
        )

        assert (len(scattered) == 1) is expect_scatter

    @pytest.mark.parametrize("detach_heads", [False, True])
    def test_forward_detach_heads_gradient_flow(self, monkeypatch, detach_heads):
        """Block-level check of mtp_detach_heads: with the flag on, MTP gradients must
        not reach the main-model hidden_states or the shared embedding, while the MTP
        layer parameters still receive gradients."""
        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp=1, cp=1)
        config.mtp_detach_heads = detach_heads
        # Runs on GPU because _concat_embeddings exercises the (fused) norm and
        # projection kernels; the rest of the MTP transformer layer is stubbed out.
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec).cuda()

        # Replace each MTP transformer layer with an identity so the test isolates
        # gradient flow to the detach logic (not the attention kernels). Must be an
        # nn.Module since it is assigned as a child module of the layer.
        class _IdentityMTPLayer(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states, None

        for layer in mtp.layers:
            monkeypatch.setattr(layer, "mtp_model_layer", _IdentityMTPLayer())

        seq_len = 4
        batch_size = 2
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64).cuda()
        position_ids = torch.arange(seq_len, dtype=torch.int64).repeat(batch_size, 1).cuda()
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool).cuda()
        hidden_states = torch.randn(
            seq_len, batch_size, config.hidden_size, device="cuda", requires_grad=True
        )
        emb_weight = torch.nn.Parameter(
            torch.randn(seq_len, batch_size, config.hidden_size, device="cuda")
        )

        def fake_embedding(input_ids, position_ids):
            return emb_weight.clone()

        output = mtp.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            embedding=fake_embedding,
        )

        # forward concatenates [main_hidden_states, mtp_out_0, mtp_out_1] along dim 0;
        # back-propagate only from the MTP outputs to mimic the MTP loss path.
        mtp_outputs = output[seq_len:]
        mtp_outputs.sum().backward()

        # MTP layer parameters always receive gradients.
        for layer in mtp.layers:
            assert layer.enorm.weight.grad is not None
            assert layer.hnorm.weight.grad is not None
            assert layer.eh_proj.weight.grad is not None

        if detach_heads:
            # Gradients must not reach the main model or the shared embedding.
            # The returned block output still includes the original hidden-state
            # chunk, so autograd may allocate a zero grad for it through cat().
            if hidden_states.grad is not None:
                torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
            assert emb_weight.grad is None
        else:
            assert hidden_states.grad is not None
            assert emb_weight.grad is not None

    @pytest.mark.parametrize("detach_heads", [False, True])
    def test_process_mtp_loss_detaches_output_weight(self, detach_heads):
        """process_mtp_loss must detach the output-head weight when mtp_detach_heads=True
        so the MTP loss does not update the (shared) output projection weight."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            mtp_detach_heads=detach_heads,
        )

        seq_len = 4
        batch_size = 2
        vocab_size = 16
        # hidden_states is the concatenation [main, mtp_0, mtp_1] along the sequence dim;
        # requires_grad so the returned tensor stays in the autograd graph for backward.
        hidden_states = torch.randn(
            (1 + config.mtp_num_layers) * seq_len,
            batch_size,
            config.hidden_size,
            requires_grad=True,
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        output_weight = torch.nn.Parameter(torch.randn(vocab_size, config.hidden_size))

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            # hidden: [s, b, h] -> logits: [s, b, vocab]
            return torch.matmul(hidden, weight.t()), None

        def compute_language_model_loss(labels, logits):
            # per-token loss of shape [b, s] that depends on logits (hence output_weight).
            return logits.sum(dim=-1).transpose(0, 1)

        result = process_mtp_loss(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=output_weight,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
        )
        result.sum().backward()

        if detach_heads:
            assert output_weight.grad is None
        else:
            assert output_weight.grad is not None

    def test_process_mtp_loss_all_masked_positions_have_zero_gradient(self):
        """An all-zero mask must make MTP a finite no-op, including after mask rolling."""
        config = TransformerConfig(
            mtp_num_layers=1,
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            use_cpu_initialization=True,
        )
        seq_len = 4
        hidden_states = torch.randn(
            (1 + config.mtp_num_layers) * seq_len, 1, config.hidden_size, requires_grad=True
        )
        output_weight = torch.nn.Parameter(torch.randn(16, config.hidden_size))

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return torch.matmul(hidden, weight.t()), None

        def compute_language_model_loss(labels, logits):
            return logits.square().sum(dim=-1).transpose(0, 1)

        result = process_mtp_loss(
            hidden_states=hidden_states,
            labels=torch.arange(seq_len).unsqueeze(0),
            loss_mask=torch.zeros(1, seq_len),
            output_layer=output_layer,
            output_weight=output_weight,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
        )
        result.sum().backward()

        assert torch.isfinite(result).all()
        assert torch.count_nonzero(hidden_states.grad[seq_len:]) == 0
        assert output_weight.grad is not None
        assert torch.count_nonzero(output_weight.grad) == 0


class TestMultiTokenPrediction:
    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=True
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_block_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

        return model

    def create_test_args(
        self, tp, cp, sequence_length, micro_batch_size, fp8=None, full_recompute=False
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_multi_token_predictioin.py']
        args = parse_args()
        args.num_layers = 2
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.padded_vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.context_parallel_size = cp
        args.position_embedding_type = 'rope'
        args.num_experts = 8
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.moe_router_topk = 2
        args.moe_router_pre_softmax = False
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        if HAVE_TE:
            # only use grouped gemm if there is TE
            args.moe_grouped_gemm = True
        else:
            args.moe_grouped_gemm = False
        args.bf16 = True
        if fp8 is not None:
            args.fp8 = 'e4m3'
        if full_recompute:
            args.recompute_granularity = 'full'
            args.recompute_method = 'uniform'
            args.recompute_num_layers = 1
        else:
            args.recompute_granularity = None
        args.add_bias_linear = False
        args.swiglu = True

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        return batch

    def get_packed_batch(self, seq_lengths, micro_batch_size):
        """
        Create a packed sequence batch with multiple sequences of varying lengths.

        Args:
            seq_lengths: List of sequence lengths (e.g., [10, 15, 8] for 3 sequences)
            micro_batch_size: Batch size (typically 1 for packed sequences)

        Returns:
            batch: Dictionary containing packed sequences and PackedSeqParams
        """
        total_seq_length = sum(seq_lengths)

        # Create packed input_ids, labels, and position_ids
        input_ids_list = []
        labels_list = []
        position_ids_list = []

        for seq_len in seq_lengths:
            data = list(range(seq_len))
            input_ids_list.extend(data)
            labels_list.extend([x + 1 for x in data])
            position_ids_list.extend(data)

        # Convert to tensors with shape [batch, total_seq_length]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).unsqueeze(0).cuda()
        labels = torch.tensor(labels_list, dtype=torch.int64).unsqueeze(0).cuda()
        position_ids = torch.tensor(position_ids_list, dtype=torch.int64).unsqueeze(0).cuda()

        # Create attention mask for packed sequences (all ones for simplicity)
        attention_mask = torch.ones(
            (micro_batch_size, 1, total_seq_length, total_seq_length), dtype=bool
        ).cuda()

        # Create loss mask with shape [batch, total_seq_length]
        loss_mask = torch.ones(micro_batch_size, total_seq_length).cuda()

        # Create cumulative sequence lengths for PackedSeqParams
        cu_seqlens = torch.tensor(
            [0] + [sum(seq_lengths[: i + 1]) for i in range(len(seq_lengths))], dtype=torch.int32
        ).cuda()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(seq_lengths),
            max_seqlen_kv=max(seq_lengths),
            qkv_format='thd',
        )

        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'packed_seq_params': packed_seq_params,
        }
        return batch

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_sharded_state_dict(self, tp, cp):
        """Test MTP with different tensor parallel sizes."""
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        model_parallel_cuda_manual_seed(_SEED)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_cfg = gpt_config_from_args(args)
        builder_cls = model_cfg.get_builder_cls()
        builder = builder_cls(model_cfg)
        gpt_model = builder.build_distributed_models(
            pg_collection=pg_collection, wrap_with_ddp=False
        )
        sharded_state_dict = gpt_model[0].sharded_state_dict()
        for i in range(args.mtp_num_layers):
            assert f"mtp.layers.{i}.enorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.hnorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.eh_proj.weight" in sharded_state_dict.keys()

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize("full_recompute", [False, True])
    @pytest.mark.parametrize(
        ("tp", "cp"), [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2)]
    )
    def test_forward_backward(self, tmp_path_dist_ckpt, tp, cp, full_recompute):
        """Test MTP forward and backward with gptmodel."""
        tp_ref = 1
        cp_ref = 1
        args = self.create_test_args(tp_ref, cp_ref, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_ref, context_parallel_size=cp_ref
        )
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
        gpt_model_ref, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )
        output_ref = gpt_model_ref[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        tracker = MTPLossLoggingHelper.tracker
        mtp_loss_ref = None
        assert "loss_values" in tracker
        mtp_loss_ref = tracker['loss_values'].clone()
        MTPLossLoggingHelper.clean_metrics_in_tracker()

        iteration = 123
        num_floating_point_operations_so_far = 456

        def set_ckpt_path(ckpt_path):
            args.save = ckpt_path
            args.load = ckpt_path

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_mtp_model_reconfiguration_model_A'
        ) as ckpt_dir_A:
            set_ckpt_path(ckpt_dir_A)
            save_checkpoint(
                iteration,
                gpt_model_ref,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

            expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
            assert os.path.exists(expected_ckpt_path)

            # Test with different TP/CP configuration
            Utils.destroy_model_parallel()
            args = self.create_test_args(
                tp, cp, self.seq_length, self.micro_batch_size, full_recompute=full_recompute
            )
            set_args(args)
            set_ckpt_path(ckpt_dir_A)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
            gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                ModelType.encoder_or_decoder, self.model_provider
            )
            load_checkpoint(gpt_model, optimizer, opt_param_scheduler, strict=False)
            batch["output_ref"] = output_ref
            # Get batch for current CP rank (handles CP tensor splitting)
            batch = get_batch_on_this_cp_rank(
                batch, is_hybrid_cp=False, cp_group=get_context_parallel_group()
            )
            tokens, labels, loss_mask, attention_mask, position_ids, output_ref = batch.values()
            output = gpt_model[0].forward(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            tracker = MTPLossLoggingHelper.tracker
            assert "loss_values" in tracker
            mtp_loss = tracker['loss_values'].clone()
            # Average MTP loss across CP ranks for comparison with reference
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp'])
            torch.distributed.all_reduce(
                mtp_loss, group=pg_collection.cp, op=torch.distributed.ReduceOp.AVG
            )
            MTPLossLoggingHelper.clean_metrics_in_tracker()
            assert torch.allclose(output_ref, output, rtol=1e-03, atol=1e-03)
            assert torch.allclose(mtp_loss, mtp_loss_ref, rtol=1e-02, atol=1e-02)

            # Check output shapes - sequence length is divided by CP size
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length / cp

            # Verify gradients
            loss = output.mean()
            loss.backward()
            # for param in gpt_model[0].parameters():
            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("1.7.0"),
        reason="Only transformer-engine>=1.7.0 supports MoE FP8 training",
    )
    @pytest.mark.parametrize("full_recompute", [False, True])
    def test_fp8_support(self, full_recompute):
        """Test MTP with FP8 training enabled."""
        tp = 1
        cp = 1
        fp8 = 'e4m3'
        args = self.create_test_args(
            tp, cp, self.seq_length, self.micro_batch_size, fp8, full_recompute=full_recompute
        )
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
        gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )

        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

        assert output.dtype == torch.float32  # Output should be converted back to float32

        loss = output.mean()
        loss.backward()

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (2, 2)])
    def test_packed_sequences(self, tp, cp):
        """Test MTP with packed sequences."""
        # Create args with packed sequences support
        seq_lengths = [16, 24, 12]  # Three sequences of different lengths
        total_seq_length = sum(seq_lengths)

        args = self.create_test_args(tp, cp, total_seq_length, micro_batch_size=1)
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        # Get packed batch
        batch = self.get_packed_batch(seq_lengths, micro_batch_size=1)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        packed_seq_params = batch['packed_seq_params']

        # Create model
        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        # Forward pass with packed sequences
        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

        # Verify output shape
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == total_seq_length

        # Verify MTP loss was computed
        tracker = MTPLossLoggingHelper.tracker
        assert "loss_values" in tracker
        mtp_loss = tracker['loss_values'].clone()
        assert mtp_loss.shape[0] == args.mtp_num_layers
        MTPLossLoggingHelper.clean_metrics_in_tracker()

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Verify gradients exist
        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"Gradient missing for {name}"

    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    def test_packed_sequences_with_full_recompute(self):
        """MTP + packed sequences + full activation recomputation.

        Regression: MTP._checkpointed_forward used to forward
        ``packed_seq_params`` (a non-tensor PackedSeqParams object) directly
        to ``tensor_parallel.checkpoint``. CheckpointFunction.save_for_backward
        only accepts tensors and ``None``, so this raised
        ``TypeError: save_for_backward can only save variables, but argument
        N is of type PackedSeqParams``. Non-tensor kwargs must be captured
        by closure, not forwarded as args.
        """
        seq_lengths = [16, 24, 12]
        total_seq_length = sum(seq_lengths)

        args = self.create_test_args(
            tp=1, cp=1, sequence_length=total_seq_length, micro_batch_size=1, full_recompute=True
        )
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)

        batch = self.get_packed_batch(seq_lengths, micro_batch_size=1)

        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        gpt_model, _, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        output = gpt_model[0].forward(
            input_ids=batch['tokens'],
            position_ids=batch['position_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            loss_mask=batch['loss_mask'],
            packed_seq_params=batch['packed_seq_params'],
        )

        # Backward must run end-to-end through the recomputed MTP layer.
        loss = output.mean()
        loss.backward()

        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"Gradient missing for {name}"

    def test_roll_tensor_none_input(self):
        """Test that roll_tensor returns (None, None) when given None input."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        result, sum_val = roll_tensor(None, shifts=-1, dims=-1)
        assert result is None
        assert sum_val is None
        Utils.destroy_model_parallel()

    def test_roll_tensor_shifts_left_and_zeroes_last(self):
        """Test that roll_tensor(-1) shifts left and zeroes the last position.

        This is the primitive used to derive MTP labels from input_ids when labels
        are not provided (RL training): label[i] = input_id[i+1], last position zeroed.
        The end-to-end derivation is covered by process_mtp_loss (see input_ids path).
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        # Simulate input_ids [batch=2, seq=5]
        input_ids = torch.tensor(
            [[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]], dtype=torch.int64
        ).cuda()
        rolled, _ = roll_tensor(input_ids, shifts=-1, dims=-1)

        # Expected: each row shifted left by 1, last element zeroed.
        expected = torch.tensor(
            [[20, 30, 40, 50, 0], [70, 80, 90, 100, 0]], dtype=torch.int64
        ).cuda()
        assert torch.equal(rolled, expected)
        Utils.destroy_model_parallel()

    def test_process_mtp_loss_skips_when_no_labels_and_no_input_ids(self):
        """When labels and input_ids are both None, MTP loss is skipped (early return)."""
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        hidden_states = torch.ones(2, 1, 4)
        called = {'value': False}

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return hidden.clone(), None

        def compute_language_model_loss(mtp_labels, mtp_logits):
            called['value'] = True
            return torch.ones_like(mtp_labels, dtype=mtp_logits.dtype)

        out = process_mtp_loss(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            cp_group=None,
            packed_seq_params=None,
            input_ids=None,
        )

        # First chunk is returned unchanged and the loss is never computed.
        assert not called['value']
        assert torch.equal(out, torch.chunk(hidden_states, 2, dim=0)[0])

    def test_process_mtp_loss_derives_labels_from_input_ids(self):
        """When labels is None (RL), labels are derived from input_ids by rolling left.

        process_mtp_loss rolls once to build the SFT-format labels (label[i] =
        input_id[i+1]) and once more per MTP layer, so MTP head 0 targets input_id[i+2].
        The loss_mask is rolled in lockstep so the fabricated trailing label is masked.
        """
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        # hidden_states is chunked into (1 + mtp_num_layers) along dim 0.
        hidden_states = torch.ones(2, 1, 5)
        input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        seen = {'labels': None, 'masked_loss': None}

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return hidden.clone(), None

        def compute_language_model_loss(mtp_labels, mtp_logits):
            seen['labels'] = mtp_labels.clone()
            # Per-position loss of 1.0 so loss_mask * loss exposes the active mask.
            return torch.ones_like(mtp_labels, dtype=torch.float32)

        process_mtp_loss(
            hidden_states=hidden_states,
            labels=None,
            loss_mask=None,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            cp_group=None,
            packed_seq_params=None,
            input_ids=input_ids,
        )

        # input_ids rolled twice (once to SFT format, once in the MTP layer loop):
        # [10,20,30,40,50] -> [20,30,40,50,0] -> [30,40,50,0,0].
        assert seen['labels'] is not None, "loss should be computed in RL mode"
        assert torch.equal(seen['labels'], torch.tensor([[30, 40, 50, 0, 0]], dtype=torch.long))

    @pytest.mark.parametrize("cp", [1, 2])
    def test_roll_tensor_with_packed_sequences(self, cp):
        """Test roll_tensor function with packed sequences, with and without CP.

        For CP=1: Tests standard packed sequence rolling with verified expected values
        For CP=2: Tests CP-enabled rolling executes without errors
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
        cp_group = get_context_parallel_group() if cp > 1 else None
        cp_rank = torch.distributed.get_rank(group=cp_group) if cp_group is not None else 0

        if cp == 1:
            # Test case: Simple packed sequences (CP disabled)
            tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
            cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_kv=3,
                qkv_format='thd',
            )

            # Roll by -1 (shift left)
            rolled, sum_val = roll_tensor(
                tensor, shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )

            # Expected: [2, 3, 0, 5, 0] - boundaries at indices 2 and 4 are zeroed
            expected = torch.tensor([2, 3, 0, 5, 0], dtype=torch.float32).cuda()
            assert torch.equal(rolled, expected), f"Expected {expected}, got {rolled}"
        else:
            # Test case: Packed sequences with CP=2
            # Two sequences:
            #   seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
            #   seq2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

            if cp_rank == 0:
                # CP Rank 0: first half of each sequence
                tensor = torch.tensor(
                    [1, 2, 7, 8, 11, 12, 13, 20, 21, 22], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [2, 3, 8, 0, 12, 13, 14, 21, 22, 0], dtype=torch.float32
                ).cuda()
            else:
                # CP Rank 1: second half of each sequence
                tensor = torch.tensor(
                    [3, 4, 5, 6, 14, 15, 16, 17, 18, 19], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [4, 5, 6, 7, 15, 16, 17, 18, 19, 20], dtype=torch.float32
                ).cuda()

            cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=6,  # max(4, 6) - max local seq length per sequence
                max_seqlen_kv=6,
                qkv_format='thd',
            )

            # Roll by -1 (shift left) with CP communication
            rolled, sum_val = roll_tensor(
                tensor, shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )

            # Verify the rolled tensor matches expected values
            assert (
                rolled.shape == expected.shape
            ), f"Shape mismatch: expected {expected.shape}, got {rolled.shape}"
            assert torch.equal(
                rolled, expected
            ), f"CP Rank {cp_rank}: Expected\n{expected}\nbut got\n{rolled}\nDiff:\n{rolled - expected}"

            # Verify sum is correct
            assert sum_val.numel() == 1, "Sum should be a scalar"

        Utils.destroy_model_parallel()


class TestMTPLossLoggingHelper:
    def setup_method(self, method):
        self.num_layers = 4
        # Reset the tracker before each test
        MTPLossLoggingHelper.tracker = {}

    def teardown_method(self, method):
        # Clean up the tracker after each test
        MTPLossLoggingHelper.tracker = {}

    def test_save_metrics_to_tracker(self):
        """Test saving metrics to tracker."""
        loss = torch.tensor(1.3)
        correct = torch.tensor(5.0)
        total = torch.tensor(10.0)
        layer_number = 2
        num_layers = self.num_layers

        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=loss,
            correct=correct,
            total=total,
            layer_number=layer_number,
            num_layers=num_layers,
        )

        tracker = MTPLossLoggingHelper.tracker
        assert "loss_values" in tracker
        assert tracker["loss_values"].shape == (num_layers,)
        assert tracker["loss_values"][layer_number] == loss
        assert tracker["correct_values"][layer_number] == correct
        assert tracker["total_values"][layer_number] == total
        assert tracker["reduce_group"] is None
        assert tracker["avg_group"] is None

    def test_mtp_logits_are_vocab_sharded(self):
        """Test detection for vocab-sharded versus gathered MTP logits."""

        class DummyOutputLayer:
            def __init__(self, gather_output):
                self.gather_output = gather_output

        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=True), None) is False
        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=False), None) is True
        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=True), True) is False
        assert _mtp_logits_are_vocab_sharded(DummyOutputLayer(gather_output=True), False) is True

    def test_track_mtp_metrics(self):
        """Test tracking MTP metrics including acceptance rate."""
        num_layers = self.num_layers
        loss = torch.tensor(2.3)
        correct = torch.tensor(7.0)
        total = torch.tensor(10.0)

        for i in range(num_layers):
            MTPLossLoggingHelper.save_metrics_to_tracker(
                loss=loss, correct=correct, total=total, layer_number=i, num_layers=num_layers
            )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        class DummyWandBWriter:
            def log(self, metrics, iteration):
                pass

        loss_scale = 1.5
        iteration = 2
        writer = DummyWriter()
        wandb_writer = DummyWandBWriter()
        total_loss_dict = {}

        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

        # Verify loss uses the legacy normalized MTP loss scaled by loss_scale.
        expected_loss = loss * loss_scale
        for i in range(num_layers):
            assert f"mtp_{i+1} loss" in writer.scalars
            assert torch.isclose(torch.as_tensor(writer.scalars[f"mtp_{i+1} loss"]), expected_loss)
            assert torch.isclose(total_loss_dict[f"mtp_{i+1} loss"], expected_loss)

        # Verify acceptance rate is computed as (correct / total) * 100
        expected_rate = (correct / total) * 100.0
        for i in range(num_layers):
            assert f"mtp_{i+1}_acceptance_rate" in writer.scalars
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i+1}_acceptance_rate"]), expected_rate
            )
            assert f"mtp_{i+1}_cumulative_acceptance_rate" in writer.scalars
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i+1}_cumulative_acceptance_rate"]),
                expected_rate,
            )

        raw_counter_suffixes = ("_sum", "_tokens", "_correct", "_total")
        assert not any(key.endswith(raw_counter_suffixes) for key in total_loss_dict)

        second_correct = torch.tensor(3.0)
        second_total = torch.tensor(10.0)
        for i in range(num_layers):
            MTPLossLoggingHelper.save_metrics_to_tracker(
                loss=loss,
                correct=second_correct,
                total=second_total,
                layer_number=i,
                num_layers=num_layers,
            )

        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale,
            iteration=iteration + 1,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

        expected_second_rate = (second_correct / second_total) * 100.0
        expected_cumulative_rate = ((correct + second_correct) / (total + second_total)) * 100.0
        for i in range(num_layers):
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i+1}_acceptance_rate"]), expected_second_rate
            )
            assert torch.isclose(
                torch.as_tensor(writer.scalars[f"mtp_{i+1}_cumulative_acceptance_rate"]),
                expected_cumulative_rate,
            )
            assert torch.isclose(total_loss_dict[f"mtp_{i+1} loss"], expected_loss * 2)

        # Verify tracker is cleaned
        assert torch.all(MTPLossLoggingHelper.tracker["loss_values"] == 0)
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None

    def test_track_mtp_loss_preserves_legacy_normalized_loss_semantics(self):
        """MTP loss logging should not become token-weighted when acceptance counters are added."""
        first_loss = torch.tensor(10.0)
        second_loss = torch.tensor(2.0)
        correct = torch.tensor(0.0)
        total = torch.tensor(1.0)
        loss_scale = torch.tensor(0.5)
        layer_number = 0

        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=first_loss, correct=correct, total=total, layer_number=layer_number, num_layers=1
        )
        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=second_loss, correct=correct, total=total, layer_number=layer_number, num_layers=1
        )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        writer = DummyWriter()
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale, iteration=1, writer=writer, total_loss_dict={}
        )

        logged_loss = torch.as_tensor(writer.scalars["mtp_1 loss"])
        expected_legacy_loss = (first_loss + second_loss) * loss_scale
        token_weighted_loss = torch.tensor(40.0 / 12.0)
        assert torch.isclose(logged_loss, expected_legacy_loss)
        assert not torch.isclose(logged_loss, token_weighted_loss)


class TestMultiTokenPredictionHybrid:
    """Test Multi-Token Prediction with Mamba hybrid models."""

    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    def model_provider(self, pre_process=True, post_process=True, **config_kwargs):
        """Model provider for Mamba hybrid models with MTP.

        Uses the unified pattern syntax where MTP is configured via hybrid_layer_pattern:
        Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."
        Example: "M*M*/M*/M*" = main decoder "M*M*", MTP pattern "M*" with 2 depths
        """
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)

        # MTP is configured via unified pattern in hybrid_layer_pattern
        # HybridModel creates the MTP block internally based on the parsed pattern
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            hybrid_layer_pattern=args.hybrid_layer_pattern,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
        return model

    def create_test_args(
        self, tp, cp, sequence_length, micro_batch_size, fp8=None, full_recompute=False
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_multi_token_prediction_hybrid.py']
        args = parse_args()
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.padded_vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.num_query_groups = 8
        args.mamba_num_groups = 4
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.context_parallel_size = cp
        args.position_embedding_type = 'rope'
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        args.bf16 = True
        # Unified pattern: "main/mtp/mtp" - main decoder "M*M*", MTP pattern "M*" with 2 depths
        args.hybrid_layer_pattern = "M*M*/M*/M*"

        if fp8 is not None:
            args.fp8 = 'e4m3'
        if full_recompute:
            args.recompute_granularity = 'full'
            args.recompute_method = 'uniform'
            args.recompute_num_layers = 1
        else:
            args.recompute_granularity = None
        args.add_bias_linear = False
        args.swiglu = True

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        return batch

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1)])
    def test_sharded_state_dict_mamba(self, tp, cp):
        """Test MTP with Mamba hybrid model - sharded state dict."""
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        model_parallel_cuda_manual_seed(_SEED)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_cfg = hybrid_config_from_args(args)
        builder_cls = model_cfg.get_builder_cls()
        builder = builder_cls(model_cfg)
        mamba_model = builder.build_distributed_models(
            pg_collection=pg_collection, wrap_with_ddp=False
        )
        sharded_state_dict = mamba_model[0].sharded_state_dict()

        # Verify MTP layers are in the state dict
        for i in range(args.mtp_num_layers):
            assert f"mtp.layers.{i}.enorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.hnorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.eh_proj.weight" in sharded_state_dict.keys()

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1)])
    def test_forward_backward_mamba(self, tmp_path_dist_ckpt, tp, cp):
        """Test MTP forward and backward with Mamba hybrid model."""
        tp_ref = 1
        cp_ref = 1
        args = self.create_test_args(tp_ref, cp_ref, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_ref, context_parallel_size=cp_ref
        )
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()

        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = Utils.pretrain_config_from_global_args(args, "hybrid")
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        mamba_model_ref, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=pg_collection,
        )

        output_ref = mamba_model_ref[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        tracker = MTPLossLoggingHelper.tracker
        mtp_loss_ref = None
        assert "loss_values" in tracker
        mtp_loss_ref = tracker['loss_values'].clone()
        MTPLossLoggingHelper.clean_metrics_in_tracker()

        iteration = 123
        num_floating_point_operations_so_far = 456

        def set_ckpt_path(ckpt_path):
            args.save = ckpt_path
            args.load = ckpt_path

        with TempNamedDir(tmp_path_dist_ckpt / 'test_mtp_mamba_model_reconfiguration') as ckpt_dir:
            set_ckpt_path(ckpt_dir)
            save_checkpoint(
                iteration,
                mamba_model_ref,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

            expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
            assert os.path.exists(expected_ckpt_path)

            Utils.destroy_model_parallel()
            args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
            set_args(args)
            set_ckpt_path(ckpt_dir)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

            model_parallel_cuda_manual_seed(_SEED)
            cfg_container = Utils.pretrain_config_from_global_args(args, "hybrid")
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            mamba_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                ModelType.encoder_or_decoder,
                self.model_provider,
                cfg_container=cfg_container,
                pg_collection=pg_collection,
            )
            load_checkpoint(mamba_model, optimizer, opt_param_scheduler, strict=False)

            batch["output_ref"] = output_ref
            batch = get_batch_on_this_cp_rank(
                batch, is_hybrid_cp=False, cp_group=get_context_parallel_group()
            )
            tokens, labels, loss_mask, attention_mask, position_ids, output_ref = batch.values()
            output = mamba_model[0].forward(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            tracker = MTPLossLoggingHelper.tracker
            assert "loss_values" in tracker
            mtp_loss = tracker['loss_values'].clone()
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp'])
            torch.distributed.all_reduce(
                mtp_loss, group=pg_collection.cp, op=torch.distributed.ReduceOp.AVG
            )
            MTPLossLoggingHelper.clean_metrics_in_tracker()
            assert torch.allclose(output_ref, output, rtol=1e-03, atol=1e-03)
            assert torch.allclose(mtp_loss, mtp_loss_ref, rtol=1e-02, atol=1e-02)

            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length / cp

            loss = output.mean()
            loss.backward()
            for name, param in mamba_model[0].named_parameters():
                assert param.main_grad is not None

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    def test_attention_mask_validation_mamba(self):
        """Test that attention mask type validation works for Mamba hybrid models."""
        tp = 1
        cp = 1
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_cfg = hybrid_config_from_args(args)
        builder_cls = model_cfg.get_builder_cls()
        builder = builder_cls(model_cfg)
        try:
            model_parallel_cuda_manual_seed(_SEED)
            mamba_model = builder.build_distributed_models(
                pg_collection=pg_collection, wrap_with_ddp=False
            )
            mamba_model = unwrap_model(mamba_model)
            assert isinstance(mamba_model[0], HybridModel)
            assert mamba_model[0].mtp is not None
        except AssertionError as e:
            if "Multi-Token Prediction (MTP) is not yet supported" in str(e):
                pytest.fail(f"Attention mask validation failed for Mamba hybrid model: {e}")
            else:
                raise
