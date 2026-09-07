# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_gpt_model_propagates_decoder_root_name() -> None:
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=128,
            max_sequence_length=16,
            position_embedding_type="none",
        )
    finally:
        Utils.destroy_model_parallel()

    assert model.decoder.name == "decoder"
