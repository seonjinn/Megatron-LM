# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
import pytest
import torch


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


@pytest.mark.parametrize("fp8_param", [False, True])
def test_gpt_mixed_scope_storage_matches_final_recipe(fp8_param: bool) -> None:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.quantization.quant_config import GlobMatcher, RecipeConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    recipe = RecipeConfig(
        matchers=[
            GlobMatcher("decoder.layers.1.self_attention.linear_qkv", "mxfp8"),
            GlobMatcher("decoder.layers.1.self_attention.linear_proj", "mxfp8"),
            GlobMatcher("*", "bf16"),
        ],
        config_dict={
            "bf16": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"override_quantized_autocast": True},
            },
            "mxfp8": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {
                    "fp8_quantization_recipe": "mxfp8",
                    "fp8_param": fp8_param,
                    "override_quantized_autocast": True,
                },
            },
        },
    )
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=3, hidden_size=128, num_attention_heads=4,
            params_dtype=torch.bfloat16, bf16=True,
            fp8="e4m3", fp8_recipe="mxfp8", fp8_param=fp8_param,
            quant_recipe=recipe,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128, max_sequence_length=16, position_embedding_type="none",
        )
        for index, layer in enumerate(model.decoder.layers):
            for projection in (layer.self_attention.linear_qkv, layer.self_attention.linear_proj):
                assert isinstance(projection.weight, MXFP8Tensor) == (fp8_param and index == 1)
            for projection in (layer.mlp.linear_fc1, layer.mlp.linear_fc2):
                assert not isinstance(projection.weight, MXFP8Tensor)
                assert projection.weight.dtype == torch.bfloat16
        del model
    finally:
        Utils.destroy_model_parallel()
