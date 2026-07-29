# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import (
    MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    PackedSeqParams,
    build_mamba_packed_seq_params_from_cuda_graph_kwargs,
    split_mamba_packed_seq_params_for_cuda_graph,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params


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

        residual = hidden_states
        if self.config.fp32_residual_connection:
            residual = residual.float()

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = apply_module(self.norm)(hidden_states)

        mixer_out_with_bias = self.mixer(
            hidden_states, inference_context=inference_context, packed_seq_params=packed_seq_params
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

    @staticmethod
    def _te_cuda_graph_tensor_signature(tensor: Tensor) -> tuple[object, ...]:
        """Return the Tensor contract that must remain fixed across graph replays."""
        return (
            tensor.shape,
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.stride(),
        )

    def _set_te_cuda_graph_mamba_packed_seq_params_static_metadata(
        self,
        static_metadata: Mapping[str, object],
        tensor_kwargs: Mapping[str, Tensor | None],
    ) -> None:
        """Store the packed Mamba contract used for TE graph capture."""
        self._te_cuda_graph_mamba_packed_seq_params_static_metadata = dict(static_metadata)
        self._te_cuda_graph_mamba_packed_seq_params_tensor_signatures = {
            name: self._te_cuda_graph_tensor_signature(value)
            for name, value in tensor_kwargs.items()
            if isinstance(value, Tensor)
        }

    def _validate_te_cuda_graph_mamba_packed_seq_params(
        self,
        static_metadata: Mapping[str, object],
        tensor_kwargs: Mapping[str, Tensor | None],
    ) -> None:
        """Validate packed Mamba replay inputs before invoking Transformer Engine."""
        expected_static_metadata = getattr(
            self, "_te_cuda_graph_mamba_packed_seq_params_static_metadata", None
        )
        assert expected_static_metadata is not None, (
            "TE CUDA graph replay received Mamba packed sequence inputs without a captured "
            "static metadata contract."
        )

        expected_signatures = getattr(
            self, "_te_cuda_graph_mamba_packed_seq_params_tensor_signatures", None
        )
        assert expected_signatures is not None
        actual_signatures = {
            name: self._te_cuda_graph_tensor_signature(value)
            for name, value in tensor_kwargs.items()
            if isinstance(value, Tensor)
        }
        missing_names = sorted(set(expected_signatures) - set(actual_signatures))
        extra_names = sorted(set(actual_signatures) - set(expected_signatures))
        assert not missing_names and not extra_names, (
            "TE CUDA graph replay received Mamba PackedSeqParams with Tensor fields that differ "
            f"from capture: missing {missing_names}, extra {extra_names}."
        )

        for name, expected_signature in expected_signatures.items():
            actual_signature = actual_signatures[name]
            signature_fields = ("shape", "dtype", "device", "layout", "stride")
            mismatches = [
                field_name
                for field_name, expected_value, actual_value in zip(
                    signature_fields, expected_signature, actual_signature
                )
                if expected_value != actual_value
            ]
            assert not mismatches, (
                f"TE CUDA graph replay Tensor {name} changed "
                f"{', '.join(mismatches)} from capture."
            )

        assert expected_static_metadata == dict(static_metadata), (
            "TE CUDA graph replay received Mamba PackedSeqParams with static metadata that "
            "differs from capture."
        )

    def _rebuild_te_cuda_graph_mamba_packed_seq_params(
        self, kwargs: dict[str, object]
    ) -> None:
        """Rebuild Mamba ``PackedSeqParams`` from flattened TE graph capture kwargs."""
        tensor_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key.startswith(MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX)
        }
        if not tensor_kwargs:
            return

        assert kwargs.get("packed_seq_params") is None, (
            "Mamba PackedSeqParams must be passed either as flattened TE CUDA graph kwargs or as "
            "packed_seq_params, but not both."
        )
        static_metadata = getattr(
            self, "_te_cuda_graph_mamba_packed_seq_params_static_metadata", None
        )
        assert static_metadata is not None, (
            "Flattened Mamba PackedSeqParams Tensor fields require static metadata captured on "
            "the MambaLayer."
        )
        self._validate_te_cuda_graph_mamba_packed_seq_params(
            static_metadata, tensor_kwargs
        )
        packed_seq_params = build_mamba_packed_seq_params_from_cuda_graph_kwargs(
            tensor_kwargs, static_metadata
        )
        for key in tensor_kwargs:
            kwargs.pop(key)
        kwargs["packed_seq_params"] = packed_seq_params

    def _flatten_te_cuda_graph_mamba_packed_seq_params(
        self, kwargs: dict[str, object]
    ) -> None:
        """Flatten replay-time Mamba ``PackedSeqParams`` into TE graph Tensor kwargs."""
        packed_seq_params = kwargs.get("packed_seq_params")
        expected_static_metadata = getattr(
            self, "_te_cuda_graph_mamba_packed_seq_params_static_metadata", None
        )
        if packed_seq_params is None and expected_static_metadata is None:
            return

        include_cp_fields = self.config.context_parallel_size > 1
        tensor_kwargs, static_metadata = split_mamba_packed_seq_params_for_cuda_graph(
            packed_seq_params, include_cp_fields
        )
        self._validate_te_cuda_graph_mamba_packed_seq_params(
            static_metadata, tensor_kwargs
        )

        duplicate_keys = set(kwargs) & set(tensor_kwargs)
        assert not duplicate_keys, (
            "Mamba PackedSeqParams CUDA graph Tensor kwargs overlap with existing replay kwargs: "
            f"{', '.join(sorted(duplicate_keys))}."
        )
        kwargs.pop("packed_seq_params", None)
        kwargs.update(tensor_kwargs)

    def _te_cuda_graph_capture(self, *args: object, **kwargs: object) -> Tensor:
        self._rebuild_te_cuda_graph_mamba_packed_seq_params(kwargs)
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, *args: object, **kwargs: object) -> object:
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
