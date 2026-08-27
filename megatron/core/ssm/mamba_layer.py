# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import (
    CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PackedSeqParams,
    split_mamba_packed_seq_params_for_cuda_graph,
    split_packed_seq_params_for_cuda_graph,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params


def _cuda_graph_tensor_signature(tensor: Tensor) -> tuple:
    """Return the Tensor properties that must stay stable across graph replay."""
    stride = tuple(tensor.stride()) if tensor.layout == torch.strided else None
    return tuple(tensor.shape), tensor.dtype, tensor.device, tensor.layout, stride


@dataclass
class MambaLayerSubmodules:
    """
    Configuration class for specifying the submodules of a Mamba layer.

    This class defines the structure and default implementations for various
    components of a Mamba layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        mamba_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
    """

    norm: LayerNormBuilder = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaLayer(GraphableMegatronModule):
    """
    A single Mamba layer.

    Mamba layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
        name: str | None = None,
    ):
        """Initialize Mamba Layer.

        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for MambaLayer"
        self.tp_group = pg_collection.tp

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=(name + f".mixer") if name is not None else None,
        )
        self.norm = submodules.norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def create_mcore_cudagraph_manager(self, config):
        """Register the mamba layer for cudagraphs."""
        assert self.config.cuda_graph_impl == "local"

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if (
            not self.config.cuda_graph_modules
            and self.config.inference_cuda_graph_scope != InferenceCudaGraphScope.block
        ) or CudaGraphModule.mamba in self.config.cuda_graph_modules:
            self.cudagraph_manager = CudaGraphManager(config)

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and ssm states shapes per request."""
        return self.mixer.mamba_state_shapes_per_request()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Perform a forward pass through the Mamba layer.

        This method implements the core computation of a Mamba layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Whole-layer + mixer lens spans, mirroring transformer_layer.py so the hybrid
        # model's Mamba layers aren't a blind spot in the per-layer breakdown (they were
        # ~34s of uninstrumented first-iteration warmup). No-op unless the 'layer' span
        # group is enabled, so zero cost on normal runs.
        with _otel_managed_span(
            'layer', 'megatron.layer.forward', **{'megatron.layer_number': self.layer_number}
        ):
            residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()

            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            hidden_states = apply_module(self.norm)(hidden_states)

            # Mamba mixer: conv + selective SSM/SSD -- the compute block, analog of the
            # transformer layer's self_attention/mlp (this is where the SSD kernel autotune
            # lands on the first pass).
            with _otel_managed_span('layer', 'megatron.layer.mamba'):
                mixer_out_with_bias = self.mixer(
                    hidden_states,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                )

            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mamba_bda(
                    training=self.training, fused=self.config.bias_dropout_fusion
                )(mixer_out_with_bias, residual, self.hidden_dropout)

            return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the mamba layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the mamba layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def get_layer_static_inputs(
        self,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> dict[str, Tensor]:
        """Get static Mamba inputs, including packed-sequence graph Tensor fields."""
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if packed_seq_params is None:
            self._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(None)
            return static_inputs

        generic_tensor_kwargs, generic_static_metadata = split_packed_seq_params_for_cuda_graph(
            packed_seq_params
        )
        mamba_tensor_kwargs, mamba_static_metadata = split_mamba_packed_seq_params_for_cuda_graph(
            packed_seq_params
        )
        self._set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
            mamba_static_metadata, mamba_tensor_kwargs
        )
        self._set_te_cuda_graph_packed_seq_params_static_metadata(
            generic_static_metadata, generic_tensor_kwargs
        )

        packed_tensor_kwargs = {**generic_tensor_kwargs, **mamba_tensor_kwargs}
        duplicate_keys = set(static_inputs) & set(packed_tensor_kwargs)
        assert not duplicate_keys, (
            "Mamba PackedSeqParams CUDA graph Tensor kwargs overlap with existing static inputs: "
            f"{', '.join(sorted(duplicate_keys))}."
        )
        static_inputs.update(packed_tensor_kwargs)
        return static_inputs

    def _set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        self, static_metadata, tensor_kwargs=None
    ):
        """Store the Mamba-only packed-sequence capture contract."""
        if static_metadata is None:
            assert tensor_kwargs is None
            self._te_cuda_graph_mamba_packed_seq_params_static_metadata = None
            self._te_cuda_graph_mamba_packed_seq_params_tensor_signatures = None
            return

        tensor_kwargs = dict(tensor_kwargs or {})
        self._te_cuda_graph_mamba_packed_seq_params_static_metadata = dict(static_metadata)
        self._te_cuda_graph_mamba_packed_seq_params_tensor_signatures = {
            key: _cuda_graph_tensor_signature(value) for key, value in tensor_kwargs.items()
        }

    def _get_te_cuda_graph_mamba_packed_seq_params_static_metadata(self):
        """Return Mamba-only packed-sequence metadata used during capture."""
        return getattr(self, '_te_cuda_graph_mamba_packed_seq_params_static_metadata', None)

    @staticmethod
    def _validate_te_cuda_graph_mamba_static_metadata(
        expected_static_metadata, static_metadata, description
    ):
        """Validate static packed-sequence metadata against the capture contract."""
        mismatched_fields = []
        for field_name in sorted(set(expected_static_metadata) | set(static_metadata)):
            expected_value = expected_static_metadata.get(field_name)
            actual_value = static_metadata.get(field_name)
            if expected_value is actual_value:
                continue
            if expected_value != actual_value:
                mismatched_fields.append(field_name)

        assert not mismatched_fields, (
            f"TE CUDA graph replay received {description} static metadata that differs from "
            "capture. Recapture the graph for changed fields: "
            f"{', '.join(mismatched_fields)}."
        )

    def _validate_te_cuda_graph_mamba_tensor_kwargs(self, tensor_kwargs):
        """Validate Mamba-only Tensor fields and signatures against capture."""
        expected_signatures = getattr(
            self, '_te_cuda_graph_mamba_packed_seq_params_tensor_signatures', None
        )
        assert expected_signatures is not None

        expected_names = set(expected_signatures)
        actual_names = set(tensor_kwargs)
        missing_names = sorted(expected_names - actual_names)
        extra_names = sorted(actual_names - expected_names)
        assert not missing_names and not extra_names, (
            "TE CUDA graph replay received Mamba PackedSeqParams Tensor fields that differ "
            "from capture. Recapture the graph for missing fields "
            f"{missing_names} and extra fields {extra_names}."
        )

        signature_fields = ("shape", "dtype", "device", "layout", "stride")
        for key in sorted(expected_names):
            expected_signature = expected_signatures[key]
            actual_signature = _cuda_graph_tensor_signature(tensor_kwargs[key])
            mismatched_signature_fields = [
                field_name
                for field_name, expected_value, actual_value in zip(
                    signature_fields, expected_signature, actual_signature
                )
                if expected_value != actual_value
            ]
            assert not mismatched_signature_fields, (
                f"TE CUDA graph replay received Mamba PackedSeqParams Tensor input {key} "
                "with a signature that differs from capture: "
                f"{', '.join(mismatched_signature_fields)}. Recapture the graph."
            )

    def _rebuild_te_cuda_graph_mamba_packed_seq_params(self, kwargs):
        """Rebuild packed inputs for Mamba capture without recomputing ``seq_idx``."""
        generic_tensor_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key.startswith(CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX)
        }
        mamba_tensor_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key.startswith(MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX)
        }
        if not generic_tensor_kwargs and not mamba_tensor_kwargs:
            return

        assert kwargs.get('packed_seq_params') is None, (
            "PackedSeqParams must be passed either as flattened Mamba CUDA graph kwargs or as "
            "packed_seq_params, but not both."
        )
        static_metadata = self._get_te_cuda_graph_mamba_packed_seq_params_static_metadata()
        assert (
            static_metadata is not None
        ), "Flattened Mamba PackedSeqParams Tensor fields require captured static metadata."
        self._validate_te_cuda_graph_mamba_tensor_kwargs(mamba_tensor_kwargs)

        if generic_tensor_kwargs:
            self._rebuild_te_cuda_graph_packed_seq_params(kwargs)
            packed_seq_params = kwargs.pop('packed_seq_params')
        else:
            packed_seq_params = PackedSeqParams()

        for field_name in ("qkv_format", "local_cp_size", "cp_group"):
            setattr(packed_seq_params, field_name, static_metadata[field_name])
        packed_seq_params.total_tokens = static_metadata["total_tokens"]
        seq_idx_key = f"{MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx"
        packed_seq_params.seq_idx = kwargs.pop(seq_idx_key, None)
        kwargs['packed_seq_params'] = packed_seq_params

    def _flatten_te_cuda_graph_mamba_packed_seq_params(self, kwargs):
        """Flatten replay-time packed inputs for Mamba TE CUDA graphs."""
        packed_seq_params = kwargs.get('packed_seq_params')
        expected_static_metadata = self._get_te_cuda_graph_mamba_packed_seq_params_static_metadata()
        if packed_seq_params is None:
            assert expected_static_metadata is None, (
                "TE CUDA graph was captured with packed_seq_params, so replay must also pass "
                "packed_seq_params with matching Mamba metadata."
            )
            return

        assert expected_static_metadata is not None, (
            "TE CUDA graph replay received packed_seq_params, but the graph was captured without "
            "packed_seq_params. Recapture the Mamba graph with matching packed inputs."
        )
        mamba_tensor_kwargs, mamba_static_metadata = split_mamba_packed_seq_params_for_cuda_graph(
            packed_seq_params
        )
        self._validate_te_cuda_graph_mamba_static_metadata(
            expected_static_metadata, mamba_static_metadata, "Mamba PackedSeqParams"
        )
        self._validate_te_cuda_graph_mamba_tensor_kwargs(mamba_tensor_kwargs)

        generic_capture_metadata = self._get_te_cuda_graph_packed_seq_params_static_metadata()
        if generic_capture_metadata is None:
            kwargs.pop('packed_seq_params')
        else:
            self._flatten_te_cuda_graph_packed_seq_params(kwargs)

        duplicate_keys = set(kwargs) & set(mamba_tensor_kwargs)
        assert not duplicate_keys, (
            "Mamba PackedSeqParams CUDA graph Tensor kwargs overlap with existing replay kwargs: "
            f"{', '.join(sorted(duplicate_keys))}."
        )
        kwargs.update(mamba_tensor_kwargs)

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Capture Mamba with a reconstructed ``PackedSeqParams`` input."""
        self._rebuild_te_cuda_graph_mamba_packed_seq_params(kwargs)
        return self.forward(*args, **kwargs)

    _set_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._set_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._get_te_cuda_graph_packed_seq_params_static_metadata
    )
    _validate_te_cuda_graph_packed_seq_params_static_metadata = (
        TransformerLayer._validate_te_cuda_graph_packed_seq_params_static_metadata
    )
    _get_te_cuda_graph_packed_seq_params_tensor_kwarg_names = (
        TransformerLayer._get_te_cuda_graph_packed_seq_params_tensor_kwarg_names
    )
    _validate_te_cuda_graph_packed_seq_params_tensor_kwargs = (
        TransformerLayer._validate_te_cuda_graph_packed_seq_params_tensor_kwargs
    )
    _rebuild_te_cuda_graph_packed_seq_params = (
        TransformerLayer._rebuild_te_cuda_graph_packed_seq_params
    )
    _flatten_te_cuda_graph_packed_seq_params = (
        TransformerLayer._flatten_te_cuda_graph_packed_seq_params
    )

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, `inference_context` is excluded from input list.
        """
        assert kwargs.get('inference_context') is None, (
            "CUDA graph accepts only Tensor inputs. inference_context is excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        self._flatten_te_cuda_graph_mamba_packed_seq_params(kwargs)
        return super()._te_cuda_graph_replay(*args, **kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs.
        if (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('inference_context') is None
            and not torch.is_inference_mode_enabled()  # for inference eager dummy_forward
        ):
            return True
        elif InferenceMode.is_active() and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs.get('inference_context') is not None
            and not self.config.cuda_graph_modules  # empty-list = per-layer CUDA graphs
        ):
            context = kwargs['inference_context']
            using_cuda_graph = (context.is_static_batching() and context.is_decode_only()) or (
                not context.is_static_batching() and context.using_cuda_graph_this_step()
            )
            return using_cuda_graph
        return False
