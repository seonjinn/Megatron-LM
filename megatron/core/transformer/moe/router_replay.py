# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, List, Optional, Tuple

import torch

ROUTER_REPLAY_CUDA_GRAPH_INPUT_CAPABILITY = "r3_router_cuda_graph_input_v1"
ROUTER_REPLAY_CUDA_GRAPH_INPUT_KWARG = "router_replay_indices"


@dataclass(frozen=True)
class RouterReplayCudaGraphInputSignature:
    """Describes a validated RouterReplay CUDA graph input tensor."""

    shape: tuple[int, int]
    dtype: torch.dtype
    device_type: str
    topk: int
    num_experts: int


def validate_router_replay_cuda_graph_input(
    indices: torch.Tensor,
    *,
    structural_padding_mask: torch.Tensor,
    expected_tokens: int,
    topk: int,
    num_experts: int,
) -> RouterReplayCudaGraphInputSignature:
    """Validates fixed-capacity router indices supplied to a CUDA graph.

    Structural padding must occupy a contiguous tail and use the canonical
    ``arange(topk)`` route. Missing routes are invalid in graph mode.
    """
    if not isinstance(indices, torch.Tensor):
        raise TypeError("Router replay CUDA graph indices must be a torch.Tensor.")
    if indices.dtype != torch.long:
        raise TypeError("Router replay CUDA graph indices must use torch.long dtype.")
    if not indices.is_contiguous():
        raise ValueError("Router replay CUDA graph indices must be contiguous.")
    if indices.ndim != 2:
        raise ValueError("Router replay CUDA graph indices must be two-dimensional.")
    if expected_tokens < 0:
        raise ValueError("Router replay CUDA graph expected_tokens must be non-negative.")
    if topk <= 0:
        raise ValueError("Router replay CUDA graph topk must be positive.")
    if num_experts <= 0:
        raise ValueError("Router replay CUDA graph num_experts must be positive.")
    if indices.shape != (expected_tokens, topk):
        raise ValueError(
            "Router replay CUDA graph indices shape must equal "
            f"({expected_tokens}, {topk}), got {tuple(indices.shape)}."
        )
    if not isinstance(structural_padding_mask, torch.Tensor):
        raise TypeError("Router replay structural padding mask must be a torch.Tensor.")
    if structural_padding_mask.dtype != torch.bool:
        raise TypeError("Router replay structural padding mask must use torch.bool dtype.")
    if structural_padding_mask.shape != (expected_tokens,):
        raise ValueError(
            "Router replay structural padding mask shape must equal "
            f"({expected_tokens},), got {tuple(structural_padding_mask.shape)}."
        )
    if structural_padding_mask.device != indices.device:
        raise ValueError("Router replay structural padding mask must be on the indices device.")
    if (indices < 0).any():
        raise ValueError("Router replay CUDA graph indices contain a missing-route sentinel.")
    if (indices >= num_experts).any():
        raise ValueError("Router replay CUDA graph indices are outside the expert range.")
    sorted_indices, _ = torch.sort(indices, dim=1)
    if (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any():
        raise ValueError("Router replay CUDA graph indices contain duplicate experts.")

    if structural_padding_mask.any():
        first_structural_token = int(structural_padding_mask.nonzero()[0].item())
        if structural_padding_mask[:first_structural_token].any() or not structural_padding_mask[
            first_structural_token:
        ].all():
            raise ValueError("Router replay structural padding must form a fixed-capacity tail.")
        structural_indices = indices[structural_padding_mask]
        structural_dummy = torch.arange(topk, device=indices.device, dtype=indices.dtype)
        if not torch.equal(structural_indices, structural_dummy.expand_as(structural_indices)):
            raise ValueError("Router replay structural dummy routes must equal arange(topk).")

    return RouterReplayCudaGraphInputSignature(
        shape=(expected_tokens, topk),
        dtype=indices.dtype,
        device_type=indices.device.type,
        topk=topk,
        num_experts=num_experts,
    )


class RouterReplayAction(Enum):
    """
    A Enum to define the actions for router replay.
    """

    RECORD = "record"  # Record the topk indices for replay
    REPLAY_FORWARD = "replay_forward"  # Replay the recorded topk indices for forward pass
    REPLAY_BACKWARD = "replay_backward"  # Replay topk indices for re-compute during backward pass


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    global_router_replay_instances: List['RouterReplay'] = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: List[torch.Tensor]):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.global_router_replay_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match instances ({len(RouterReplay.global_router_replay_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.global_router_replay_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> List[torch.Tensor]:
        """
        Collects the recorded topk indices from all RouterReplay instances.
        :return: A list of tensors, each containing the recorded topk indices for a layer.
        """
        return [
            router.get_recorded_indices() for router in RouterReplay.global_router_replay_instances
        ]

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_indices()

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        """Sets the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_router_replay_action()

    @staticmethod
    def clear_global_router_replay_instances():
        """Clear the global list of router replay instances to prevent memory leaks."""
        RouterReplay.global_router_replay_instances.clear()

    @staticmethod
    def set_global_static_buffers(static_buffer: torch.Tensor):
        """Sets static buffers for all router instances from a combined buffer.

        Args:
            static_buffer: Tensor of shape [max_tokens, num_layers, topk].
                          Each layer's RouterReplay gets a slice [:, layer_idx, :].
        """
        num_layers = len(RouterReplay.global_router_replay_instances)
        assert static_buffer.shape[1] == num_layers, (
            f"Buffer has {static_buffer.shape[1]} layers but there are "
            f"{num_layers} RouterReplay instances."
        )
        for layer_idx, router_instance in enumerate(RouterReplay.global_router_replay_instances):
            # Each layer gets a view of shape [max_tokens, topk]
            router_instance.set_static_buffer(static_buffer[:, layer_idx, :])

    @staticmethod
    def clear_global_static_buffers():
        """Clears static buffers from all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_static_buffer()

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.target_topk_idx: Optional[torch.Tensor] = None  # Target topk indices for replay
        self.recorded_topk_idx: Optional[torch.Tensor] = None  # Recorded topk indices for replay
        self.router_replay_action: Optional[RouterReplayAction] = (
            None  # Router replay action for this layer
        )
        self.replay_backward_list: List[torch.Tensor] = (
            []
        )  # List of tensors for backward pass replay
        self.static_buffer: Optional[torch.Tensor] = None  # Static buffer for CUDA graph
        self.layer_number: Optional[int] = None
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    @contextmanager
    def use_cuda_graph_input(self, indices: torch.Tensor) -> Iterator[None]:
        """Temporarily replays graph-owned router indices during capture."""
        previous_target = self.target_topk_idx
        previous_action = self.router_replay_action
        self.target_topk_idx = indices
        self.router_replay_action = RouterReplayAction.REPLAY_FORWARD
        try:
            yield
        finally:
            self.target_topk_idx = previous_target
            self.router_replay_action = previous_action

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Callable[
            [torch.Tensor, int, Optional[int], Optional[int]], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A wrapper for top-k computation that handles different replay actions.

        Args:
            scores (torch.Tensor): The scores to compute top-k on.
            topk (int): The number of top elements to select.
            num_groups (Optional[int]): Number of expert groups for group-limited routing.
            group_topk (Optional[int]): Number of groups to select for each token.
            default_compute_topk (Callable): The default top-k computation function, which
                                             should return a tuple of (values, indices).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the top-k values and indices.
        """
        if self.router_replay_action == RouterReplayAction.RECORD:
            probs, top_indices = default_compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            self.record_indices(top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            top_indices = self.target_topk_idx
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return default_compute_topk(scores, topk, num_groups, group_topk)

    def set_static_buffer(self, buffer: torch.Tensor):
        """Sets a static buffer for CUDA graph compatible recording.

        Args:
            buffer: Tensor of shape [max_tokens, topk] to copy routing indices into.
        """
        self.static_buffer = buffer

    def clear_static_buffer(self):
        """Clears the static buffer."""
        self.static_buffer = None

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices.

        If a static buffer is set (for CUDA graph compatibility), copies into it.
        Otherwise, just stores the tensor reference.
        """
        if self.static_buffer is not None:
            # Copy into static buffer for CUDA graph compatibility.
            num_tokens = topk_indices.shape[0]
            self.static_buffer[:num_tokens].copy_(topk_indices)
            self.recorded_topk_idx = self.static_buffer[:num_tokens]
        else:
            self.recorded_topk_idx = topk_indices
