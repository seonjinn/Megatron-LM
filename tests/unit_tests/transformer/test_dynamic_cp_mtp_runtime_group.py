# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import types
import unittest
from unittest import mock

import torch

import megatron.core.packed_seq_params as packed_seq_module
import megatron.core.transformer.multi_token_prediction as mtp_module
from megatron.core.packed_seq_params import PackedSeqParams


class DynamicCPMTPRuntimeGroupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.static_cp_group = object()
        self.dynamic_cp_group = object()
        self.packed_seq_params = PackedSeqParams(cp_group=self.dynamic_cp_group)

    def test_resolver_prefers_the_runtime_group(self) -> None:
        resolve_cp_group = getattr(packed_seq_module, "resolve_cp_group", None)

        self.assertIsNotNone(resolve_cp_group)
        self.assertIs(
            resolve_cp_group(self.static_cp_group, self.packed_seq_params),
            self.dynamic_cp_group,
        )
        self.assertIs(resolve_cp_group(self.static_cp_group, None), self.static_cp_group)

    def test_resolver_preserves_a_dynamic_cp_singleton(self) -> None:
        resolve_cp_group = getattr(packed_seq_module, "resolve_cp_group", None)
        singleton_params = PackedSeqParams(local_cp_size=1, cp_group=None)

        self.assertIsNotNone(resolve_cp_group)
        self.assertIsNone(resolve_cp_group(self.static_cp_group, singleton_params))

    def test_mtp_embedding_roll_uses_the_runtime_group(self) -> None:
        observed_groups = []

        def fake_roll_tensor(tensor, *, cp_group, **_kwargs):
            observed_groups.append(cp_group)
            return tensor, tensor.sum()

        layer = types.SimpleNamespace(cp_group=self.static_cp_group)
        input_ids = torch.tensor([[1, 2]], dtype=torch.int64)
        position_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        hidden_states = torch.zeros((2, 1, 4), requires_grad=True)

        with mock.patch.object(mtp_module, "roll_tensor", side_effect=fake_roll_tensor):
            mtp_module.MultiTokenPredictionLayer._get_embeddings(
                layer,
                input_ids=input_ids,
                position_ids=position_ids,
                embedding=lambda input_ids, position_ids: torch.zeros((2, 1, 4)),
                hidden_states=hidden_states,
                packed_seq_params=self.packed_seq_params,
            )

        self.assertEqual(observed_groups, [self.dynamic_cp_group, self.dynamic_cp_group])

    def test_mtp_loss_roll_uses_the_runtime_group(self) -> None:
        observed_groups = []

        def fake_roll_tensor(tensor, *, cp_group, **_kwargs):
            observed_groups.append(cp_group)
            return tensor, tensor.sum()

        config = types.SimpleNamespace(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.1,
            calculate_per_token_loss=True,
        )
        hidden_states = torch.zeros((4, 1, 2), requires_grad=True)
        labels = torch.ones((1, 2), dtype=torch.int64)
        loss_mask = torch.ones((1, 2), dtype=torch.float32)

        with mock.patch.object(mtp_module, "roll_tensor", side_effect=fake_roll_tensor):
            mtp_module.process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=lambda hidden_states, **_kwargs: (loss_mask.clone(), None),
                output_weight=None,
                runtime_gather_output=False,
                is_training=False,
                compute_language_model_loss=lambda labels, logits: torch.ones_like(loss_mask),
                config=config,
                cp_group=self.static_cp_group,
                packed_seq_params=self.packed_seq_params,
            )

        self.assertEqual(observed_groups, [self.dynamic_cp_group, self.dynamic_cp_group])


if __name__ == "__main__":
    unittest.main()
