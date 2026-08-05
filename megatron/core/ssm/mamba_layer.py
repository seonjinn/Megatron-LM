# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams, resolve_thd_tail_padding_policy
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormInterface
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params


class LayerNormBuilder(Protocol):
    """A protocol showing how MambaLayer expects to construct its LayerNorm."""

    def __call__(self, config: TransformerConfig, hidden_size: int, /) -> LayerNormInterface: ...


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
        self.norm = submodules.norm(self.config, self.config.hidden_size)
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

    def _is_static_thd_te_cuda_graph(self) -> bool:
        """Return whether this layer uses the fixed packed-THD TE graph interface."""
        return self.config.cuda_graph_impl == "transformer_engine" and self._is_thd_cuda_graph()

    def get_layer_static_inputs(
        self, seq_length: int, micro_batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Get static Mamba inputs for the fixed-shape THD TE graph interface."""
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if self._is_static_thd_te_cuda_graph():
            fixed_tokens = self.config.max_seqlen_per_dp_cp_rank
            max_real_sequences = self.config.thd_max_packed_sequences
            assert fixed_tokens is not None
            assert max_real_sequences is not None
            global_token_capacity = fixed_tokens * self.config.context_parallel_size
            reserve_dummy_slot = resolve_thd_tail_padding_policy(self.config) != "extend_last"
            cu_entries = max_real_sequences + 1 + int(reserve_dummy_slot)
            cu_seqlens = torch.full(
                (cu_entries,),
                global_token_capacity,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            cu_seqlens[0] = 0
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
            ):
                static_inputs[name] = cu_seqlens.clone()
            static_inputs["packed_seq_idx"] = torch.zeros(
                (1, fixed_tokens), dtype=torch.int32, device=torch.cuda.current_device()
            )
        return static_inputs

    def _validate_static_thd_cu_seqlens(
        self, cu_seqlens: dict[str, object], device: torch.device
    ) -> None:
        """Validate fixed-shape cumulative sequence tensors used by TE graphs."""
        max_real_sequences = self.config.thd_max_packed_sequences
        assert max_real_sequences is not None
        reserve_dummy_slot = resolve_thd_tail_padding_policy(self.config) != "extend_last"
        expected_shape = (max_real_sequences + 1 + int(reserve_dummy_slot),)
        for name, value in cu_seqlens.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Packed THD Mamba CUDA Graph {name} must be a torch.Tensor, "
                    f"got {type(value).__name__}."
                )
            if value.dtype != torch.int32:
                raise TypeError(
                    f"Packed THD Mamba CUDA Graph {name} must have dtype torch.int32, "
                    f"got {value.dtype}."
                )
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Packed THD Mamba CUDA Graph {name} must have shape "
                    f"{list(expected_shape)}, got {list(value.shape)}."
                )
            if value.device != device:
                raise ValueError(
                    f"Packed THD Mamba CUDA Graph {name} must be on {device}, "
                    f"got {value.device}."
                )

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Reconstruct packed Python metadata inside the TE graph capture boundary."""
        if self._is_static_thd_te_cuda_graph():
            fixed_tokens = self.config.max_seqlen_per_dp_cp_rank
            assert fixed_tokens is not None
            packed_seq_idx = kwargs.pop("packed_seq_idx", None)
            cu_seqlens = {
                name: kwargs.pop(name, None)
                for name in (
                    "cu_seqlens_q",
                    "cu_seqlens_kv",
                    "cu_seqlens_q_padded",
                    "cu_seqlens_kv_padded",
                )
            }
            device = args[0].device if args else kwargs["hidden_states"].device
            self._validate_static_thd_cu_seqlens(cu_seqlens, device=device)
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                seq_idx=packed_seq_idx,
                cu_seqlens_q=cu_seqlens["cu_seqlens_q"],
                cu_seqlens_kv=cu_seqlens["cu_seqlens_kv"],
                cu_seqlens_q_padded=cu_seqlens["cu_seqlens_q_padded"],
                cu_seqlens_kv_padded=cu_seqlens["cu_seqlens_kv_padded"],
                max_seqlen_q=fixed_tokens * self.config.context_parallel_size,
                max_seqlen_kv=fixed_tokens * self.config.context_parallel_size,
                total_tokens=fixed_tokens * self.config.context_parallel_size,
                pad_between_seqs=True,
                cp_partition_mode=getattr(self.config, "cp_partition_mode", "zigzag"),
            )
            # PackedSeqParams derives seq_idx from cu_seqlens when present. Keep
            # the explicit static map as a graph input for Mamba's existing
            # sequence-parallel metadata contract.
            packed_seq_params.seq_idx = packed_seq_idx
            kwargs["packed_seq_params"] = packed_seq_params
        return super()._te_cuda_graph_capture(*args, **kwargs)

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
        if self._is_static_thd_te_cuda_graph():
            packed_seq_params = kwargs.pop("packed_seq_params", None)
            if packed_seq_params is None:
                raise ValueError(
                    "Packed THD Mamba CUDA Graph replay requires packed_seq_params metadata."
                )

            packed_seq_idx = packed_seq_params.seq_idx
            if not isinstance(packed_seq_idx, torch.Tensor):
                raise TypeError(
                    "Packed THD Mamba CUDA Graph seq_idx must be a torch.Tensor, "
                    f"got {type(packed_seq_idx).__name__}."
                )
            if packed_seq_idx.dtype != torch.int32:
                raise TypeError(
                    "Packed THD Mamba CUDA Graph seq_idx must have dtype torch.int32, "
                    f"got {packed_seq_idx.dtype}."
                )

            fixed_tokens = self.config.max_seqlen_per_dp_cp_rank
            assert fixed_tokens is not None
            expected_shape = (1, fixed_tokens)
            if tuple(packed_seq_idx.shape) != expected_shape:
                raise ValueError(
                    "Packed THD Mamba CUDA Graph seq_idx must have shape "
                    f"[1, {fixed_tokens}], got {list(packed_seq_idx.shape)}."
                )

            cu_seqlens = {
                name: getattr(packed_seq_params, name, None)
                for name in (
                    "cu_seqlens_q",
                    "cu_seqlens_kv",
                    "cu_seqlens_q_padded",
                    "cu_seqlens_kv_padded",
                )
            }
            self._validate_static_thd_cu_seqlens(cu_seqlens, device=packed_seq_idx.device)

            kwargs.pop("attention_mask", None)
            kwargs.pop("inference_context", None)
            kwargs["packed_seq_idx"] = packed_seq_idx
            kwargs.update(cu_seqlens)
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
