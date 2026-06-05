# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Core model schemas."""

import typing as T

from schema_base import ModelSchema


def get_core_transformer_block_key(model_key):
    return {
        "GPT" : "decoder",
        "BERT" : "encoder",
        "hybrid" : "decoder",
    }[model_key]


class CoreSchema(ModelSchema):

    def __init__(self, model_type, layer_schema, prefix):
        block_key = get_core_transformer_block_key(model_type)
        super().__init__({
            "embeddings" : {
                "pos" : f"{prefix}embedding.position_embeddings.weight",
                "word" : f"{prefix}embedding.word_embeddings.weight",
            },
            "layer_prefix" : f"{prefix}{block_key}.layers",
            "layer" : layer_schema,
            "final_norm" : {
                "weight" : f"{prefix}{block_key}.final_layernorm.weight",
                "bias" : f"{prefix}{block_key}.final_layernorm.bias",
            },
            "output_layer" : {
                "weight" : f"{prefix}output_layer.weight",
            },
            "pooler" : {
                "weight" : f"{prefix}pooler.dense.weight",
                "bias" : f"{prefix}pooler.dense.bias",
            },
            "lm_head" : {
                "dense_weight" : f"{prefix}lm_head.dense.weight",
                "dense_bias" : f"{prefix}lm_head.dense.bias",
                "norm_weight" : f"{prefix}lm_head.layer_norm.weight",
                "norm_bias" : f"{prefix}lm_head.layer_norm.bias",
            },
            "binary_head" : {
                "weight" : f"{prefix}binary_head.weight",
                "bias" : f"{prefix}binary_head.bias",
            },
        })


class CoreLocalSchema(CoreSchema):

    def __init__(self, model_type, prefix, extra_layer_schema):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "input_layernorm.weight",
            "self_attn_norm_bias" : "input_layernorm.bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",
            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",

            # Replace with linear (nemotron-nas)
            "linear_attn_norm_weight" : "input_layernorm.weight",
            "linear_attn_norm_bias" : "input_layernorm.bias",
            "linear_attn_weight" : "self_attention.weight",
            "linear_attn_bias" : "self_attention.bias",
            "linear_mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "linear_mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "linear_mlp_weight" : "mlp.weight",
            "linear_mlp_bias" : "mlp.bias",

        } | extra_layer_schema, prefix=prefix)


class CoreTESchema(CoreSchema):

    def __init__(self, model_type, prefix, extra_layer_schema):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "mlp.linear_fc1.layer_norm_weight",
            "mlp_norm_bias" : "mlp.linear_fc1.layer_norm_bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",

            # Replace with linear (nemotron-nas)
            "linear_attn_norm_weight" : "self_attention.layer_norm_weight",
            "linear_attn_norm_bias" : "self_attention.layer_norm_bias",
            "linear_attn_weight" : "self_attention.weight",
            "linear_attn_bias" : "self_attention.bias",
            "linear_mlp_norm_weight" : "mlp.layer_norm_weight",
            "linear_mlp_norm_bias" : "mlp.layer_norm_bias",
            "linear_mlp_weight" : "mlp.weight",
            "linear_mlp_bias" : "mlp.bias",

        } | extra_layer_schema, prefix=prefix)


class CoreMoETESchema(CoreSchema):

    def __init__(self, model_type, num_experts, expert_model_parallel_size, prefix, extra_layer_schema):
        num_local_experts = num_experts // expert_model_parallel_size
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias",

            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",

            "router_weight" : "mlp.router.weight",

            **{f"mlp_fc1_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight" for expert_idx in range(num_local_experts) },
            **{f"mlp_fc2_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight" for expert_idx in range(num_local_experts) },

            # Shared experts (not EP-split): treat like normal linear layers
            "mlp_shared_fc1_weight" : "mlp.shared_experts.linear_fc1.weight",
            "mlp_shared_fc2_weight" : "mlp.shared_experts.linear_fc2.weight",

            # MoE latent projections (duplicated across TP, not sharded)
            "fc1_latent_proj_weight" : "mlp.fc1_latent_proj.weight",
            "fc1_latent_proj_bias" : "mlp.fc1_latent_proj.bias",
            "fc2_latent_proj_weight" : "mlp.fc2_latent_proj.weight",
            "fc2_latent_proj_bias" : "mlp.fc2_latent_proj.bias",

        } | extra_layer_schema, prefix=prefix)


class CoreHybridBaseSchema(ModelSchema):

    def __init__(self, model_type, layer_schema, prefix):
        block_key = get_core_transformer_block_key(model_type)
        super().__init__({
            "embeddings" : {
                "pos" : f"{prefix}embedding.position_embeddings.weight",
                "word" : f"{prefix}embedding.word_embeddings.weight",
            },
            "layer_prefix" : f"{prefix}{block_key}.layers",
            "layer" : layer_schema,
            "final_norm" : {
                "weight" : f"{prefix}{block_key}.final_norm.weight",
                "bias" : f"{prefix}{block_key}.final_norm.bias",
            },
            "output_layer" : {
                "weight" : f"{prefix}output_layer.weight",
            },
            "pooler" : {
                "weight" : f"{prefix}pooler.dense.weight",
                "bias" : f"{prefix}pooler.dense.bias",
            },
            "lm_head" : {
                "dense_weight" : f"{prefix}lm_head.dense.weight",
                "dense_bias" : f"{prefix}lm_head.dense.bias",
                "norm_weight" : f"{prefix}lm_head.layer_norm.weight",
                "norm_bias" : f"{prefix}lm_head.layer_norm.bias",
            },
            "binary_head" : {
                "weight" : f"{prefix}binary_head.weight",
                "bias" : f"{prefix}binary_head.bias",
            },
        })


class CoreHybridTESchema(CoreHybridBaseSchema):

    def __init__(self, model_type, prefix, extra_layer_schema):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",

            # MLP.
            "mlp_norm_weight" : "mlp.linear_fc1.layer_norm_weight",
            "mlp_norm_bias" : "mlp.linear_fc1.layer_norm_bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",

            # Mixer.
            "mixer_dt_bias" : "mixer.dt_bias",
            "mixer_D" : "mixer.D",
            "mixer_A_log" : "mixer.A_log",
            "mixer_in_proj_layer_norm_weight" : "mixer.in_proj.layer_norm_weight",
            "mixer_in_proj_weight" : "mixer.in_proj.weight",
            "mixer_conv1d_weight" : "mixer.conv1d.weight",
            "mixer_conv1d_bias" : "mixer.conv1d.bias",
            "mixer_norm_weight" : "mixer.norm.weight",
            "mixer_out_proj_weight" : "mixer.out_proj.weight",

        } | extra_layer_schema, prefix=prefix)


class CoreHybridMoETESchema(CoreHybridTESchema):
    def __init__(self, model_type, num_experts, expert_model_parallel_size, prefix, extra_layer_schema):
        num_local_experts = num_experts // expert_model_parallel_size
        super().__init__(model_type, extra_layer_schema={
            "pre_mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "pre_mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "router_weight" : "mlp.router.weight",
            "router_bias" : "mlp.router.expert_bias",
            **{f"mlp_fc1_weight.{expert_idx}" : f"mlp.experts.linear_fc1.weight{expert_idx}" for expert_idx in range(num_local_experts) },
            **{f"mlp_fc2_weight.{expert_idx}" : f"mlp.experts.linear_fc2.weight{expert_idx}" for expert_idx in range(num_local_experts) },

            # Shared experts (not EP-split): treat like normal linear layers
            "mlp_shared_fc1_weight" : "mlp.shared_experts.linear_fc1.weight",
            "mlp_shared_fc2_weight" : "mlp.shared_experts.linear_fc2.weight",

            # MoE latent projections (duplicated across TP, not sharded)
            "fc1_latent_proj_weight" : "mlp.fc1_latent_proj.weight",
            "fc1_latent_proj_bias" : "mlp.fc1_latent_proj.bias",
            "fc2_latent_proj_weight" : "mlp.fc2_latent_proj.weight",
            "fc2_latent_proj_bias" : "mlp.fc2_latent_proj.bias",
        } | extra_layer_schema, prefix=prefix)

def get_model_schema(
    model_type: T.Literal["GPT", "BERT", "hybrid"],
    transformer_impl: T.Literal["transformer_engine", "local"],
    num_experts: T.Optional[int] = None,
    expert_model_parallel_size: T.Optional[int] = None,
    prefix: T.Optional[str] = "",
    extra_layer_schema: T.Optional[dict] = {},
) -> CoreSchema:
    if num_experts is not None and num_experts > 0:
        # Only support TE setter for MOE
        assert transformer_impl == "transformer_engine"
        assert isinstance(expert_model_parallel_size, int)
        if model_type == "hybrid":
            return CoreHybridMoETESchema(model_type, num_experts, expert_model_parallel_size, prefix, extra_layer_schema)
        return CoreMoETESchema(model_type, num_experts, expert_model_parallel_size, prefix, extra_layer_schema)
    if model_type == "hybrid":
        return CoreHybridTESchema(model_type, prefix, extra_layer_schema)
    return {
        "local" : CoreLocalSchema,
        "transformer_engine" : CoreTESchema,
    }[transformer_impl](model_type, prefix, extra_layer_schema)


# ============================================================================
# MTP (Multi-Token Prediction) Schema Classes
# ============================================================================

class MTPSchema:
    """Schema for Multi-Token Prediction (MTP) block parameters.

    MTP block structure:
    - mtp.layers[i].enorm - embedding normalization
    - mtp.layers[i].hnorm - hidden states normalization
    - mtp.layers[i].eh_proj - column-parallel linear projection (2*hidden -> hidden)
    - mtp.layers[i].mtp_model_layer - transformer layer (or hybrid layer)
    - mtp.layers[i].final_layernorm - final layer normalization
    """

    def __init__(self, transformer_impl: str, prefix: str = ""):
        self.prefix = prefix
        self.transformer_impl = transformer_impl

        # MTP-specific parameter paths (within each MTP layer)
        if transformer_impl == "transformer_engine":
            self._mtp_layer_schema = {
                # Norms use TE layer norm (weight directly on module)
                "enorm_weight": "enorm.weight",
                "enorm_bias": "enorm.bias",
                "hnorm_weight": "hnorm.weight",
                "hnorm_bias": "hnorm.bias",
                # eh_proj is column-parallel linear
                "eh_proj_weight": "eh_proj.weight",
                # final layernorm
                "final_layernorm_weight": "final_layernorm.weight",
                "final_layernorm_bias": "final_layernorm.bias",
            }
            # Transformer layer schema within mtp_model_layer
            self._mtp_transformer_layer_schema = {
                # Self attention (TE style - norm fused into linear)
                "self_attn_norm_weight": "mtp_model_layer.self_attention.linear_qkv.layer_norm_weight",
                "self_attn_norm_bias": "mtp_model_layer.self_attention.linear_qkv.layer_norm_bias",
                "self_attn_qkv_weight": "mtp_model_layer.self_attention.linear_qkv.weight",
                "self_attn_qkv_bias": "mtp_model_layer.self_attention.linear_qkv.bias",
                "self_attn_proj_weight": "mtp_model_layer.self_attention.linear_proj.weight",
                "self_attn_proj_bias": "mtp_model_layer.self_attention.linear_proj.bias",
                # MLP (TE style - norm fused into linear)
                "mlp_norm_weight": "mtp_model_layer.mlp.linear_fc1.layer_norm_weight",
                "mlp_norm_bias": "mtp_model_layer.mlp.linear_fc1.layer_norm_bias",
                "mlp_fc1_weight": "mtp_model_layer.mlp.linear_fc1.weight",
                "mlp_fc1_bias": "mtp_model_layer.mlp.linear_fc1.bias",
                "mlp_fc2_weight": "mtp_model_layer.mlp.linear_fc2.weight",
                "mlp_fc2_bias": "mtp_model_layer.mlp.linear_fc2.bias",
            }
        else:
            # Local implementation
            self._mtp_layer_schema = {
                "enorm_weight": "enorm.weight",
                "enorm_bias": "enorm.bias",
                "hnorm_weight": "hnorm.weight",
                "hnorm_bias": "hnorm.bias",
                "eh_proj_weight": "eh_proj.weight",
                "final_layernorm_weight": "final_layernorm.weight",
                "final_layernorm_bias": "final_layernorm.bias",
            }
            # Transformer layer schema within mtp_model_layer (local style)
            self._mtp_transformer_layer_schema = {
                "self_attn_norm_weight": "mtp_model_layer.input_layernorm.weight",
                "self_attn_norm_bias": "mtp_model_layer.input_layernorm.bias",
                "self_attn_qkv_weight": "mtp_model_layer.self_attention.linear_qkv.weight",
                "self_attn_qkv_bias": "mtp_model_layer.self_attention.linear_qkv.bias",
                "self_attn_proj_weight": "mtp_model_layer.self_attention.linear_proj.weight",
                "self_attn_proj_bias": "mtp_model_layer.self_attention.linear_proj.bias",
                "mlp_norm_weight": "mtp_model_layer.pre_mlp_layernorm.weight",
                "mlp_norm_bias": "mtp_model_layer.pre_mlp_layernorm.bias",
                "mlp_fc1_weight": "mtp_model_layer.mlp.linear_fc1.weight",
                "mlp_fc1_bias": "mtp_model_layer.mlp.linear_fc1.bias",
                "mlp_fc2_weight": "mtp_model_layer.mlp.linear_fc2.weight",
                "mlp_fc2_bias": "mtp_model_layer.mlp.linear_fc2.bias",
            }

    def _get_mtp_layers(self, model):
        """Get the MTP layers list from the model."""
        mtp_path = f"{self.prefix}mtp" if self.prefix else "mtp"
        # Use _get_deep_attr to handle dotted prefixes like "language_model.mtp"
        mtp_block = self._get_deep_attr(model, mtp_path)
        if mtp_block is None:
            return None
        return mtp_block.layers

    def get_num_mtp_layers(self, model):
        """Get the number of MTP layers in the model."""
        layers = self._get_mtp_layers(model)
        return len(layers) if layers is not None else 0

    def _get_deep_attr(self, obj, path):
        """Get a nested attribute from an object."""
        import torch
        if path is None:
            return None
        path_parts = path.split(".")
        for key in path_parts:
            try:
                obj = getattr(obj, key)
            except AttributeError:
                return None
        if isinstance(obj, torch.Tensor):
            obj = obj.data
        return obj

    def _set_deep_tensor(self, obj, path, src):
        """Set a nested tensor attribute."""
        import torch
        if src is None:
            return
        dst = self._get_deep_attr(obj, path)
        if dst is None:
            return
        assert isinstance(src, torch.Tensor), f"src is <{type(src).__name__}> at path '{path}'."
        assert isinstance(dst, torch.Tensor), f"dst is <{type(dst).__name__}> at path '{path}'."
        dst.copy_(src)

    def get_mtp_layer(self, model, layer_idx):
        """Get MTP layer parameters for a specific layer index."""
        layers = self._get_mtp_layers(model)
        if layers is None or layer_idx >= len(layers):
            return None

        mtp_layer = layers[layer_idx]
        params = {}

        # Get MTP-specific params
        for key, path in self._mtp_layer_schema.items():
            params[key] = self._get_deep_attr(mtp_layer, path)

        # Get transformer layer params within mtp_model_layer
        for key, path in self._mtp_transformer_layer_schema.items():
            params[key] = self._get_deep_attr(mtp_layer, path)

        return params

    def set_mtp_layer(self, model, layer_idx, params):
        """Set MTP layer parameters for a specific layer index."""
        layers = self._get_mtp_layers(model)
        if layers is None or layer_idx >= len(layers):
            return

        mtp_layer = layers[layer_idx]

        # Set MTP-specific params
        for key, path in self._mtp_layer_schema.items():
            if key in params:
                self._set_deep_tensor(mtp_layer, path, params[key])

        # Set transformer layer params within mtp_model_layer
        for key, path in self._mtp_transformer_layer_schema.items():
            if key in params:
                self._set_deep_tensor(mtp_layer, path, params[key])


class MTPHybridSchema(MTPSchema):
    """Schema for MTP layers with hybrid (Mamba) model layer.

    When mtp_hybrid_override_pattern is set, the mtp_model_layer is a MambaHybridStack
    instead of a TransformerLayer.
    """

    def __init__(self, transformer_impl: str, prefix: str = ""):
        super().__init__(transformer_impl, prefix)

        # Override transformer layer schema for hybrid model
        # The mtp_model_layer.decoder contains hybrid layers
        self._mtp_transformer_layer_schema = {
            # For hybrid, the mtp_model_layer is a MambaHybridStack with decoder.layers
            # We handle this differently - return the layers list for external handling
        }

        # Mamba mixer schema (for Mamba layers in hybrid)
        self._mamba_layer_schema = {
            "mixer_dt_bias": "mixer.dt_bias",
            "mixer_D": "mixer.D",
            "mixer_A_log": "mixer.A_log",
            "mixer_in_proj_layer_norm_weight": "mixer.in_proj.layer_norm_weight",
            "mixer_in_proj_weight": "mixer.in_proj.weight",
            "mixer_conv1d_weight": "mixer.conv1d.weight",
            "mixer_conv1d_bias": "mixer.conv1d.bias",
            "mixer_norm_weight": "mixer.norm.weight",
            "mixer_out_proj_weight": "mixer.out_proj.weight",
        }

    def get_mtp_model_layers(self, model, mtp_layer_idx):
        """Get the decoder layers from the mtp_model_layer (for hybrid models)."""
        layers = self._get_mtp_layers(model)
        if layers is None or mtp_layer_idx >= len(layers):
            return None

        mtp_layer = layers[mtp_layer_idx]
        mtp_model_layer = getattr(mtp_layer, "mtp_model_layer", None)
        if mtp_model_layer is None:
            return None

        # For hybrid, mtp_model_layer is a MambaHybridStack with decoder attribute
        decoder = getattr(mtp_model_layer, "decoder", None)
        if decoder is not None:
            return decoder.layers

        # Fallback: check if it has layers directly
        return getattr(mtp_model_layer, "layers", None)


def get_mtp_schema(
    transformer_impl: T.Literal["transformer_engine", "local"],
    is_hybrid: bool = False,
    prefix: str = "",
) -> MTPSchema:
    """Factory function to get the appropriate MTP schema.

    Args:
        transformer_impl: Which transformer implementation is used.
        is_hybrid: Whether the model uses hybrid (Mamba) MTP layers.
        prefix: Prefix path to the model (e.g., "language_model." for LLaVA).

    Returns:
        MTPSchema instance for the given configuration.
    """
    if is_hybrid:
        return MTPHybridSchema(transformer_impl, prefix)
    return MTPSchema(transformer_impl, prefix)
