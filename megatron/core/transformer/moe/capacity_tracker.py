# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Device-resident counters for MoE capacity safety events."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoECapacitySnapshot:
    """A point-in-time copy of process-local MoE capacity counters."""

    selected_assignments: torch.Tensor
    dropped_assignments: torch.Tensor
    valid_token_drops: torch.Tensor
    rank_overflow_events: torch.Tensor


class MoECapacityTracker:
    """Accumulate process-local MoE capacity counters on a fixed device buffer."""

    def __init__(self) -> None:
        self._counters: torch.Tensor | None = None

    @property
    def initialized(self) -> bool:
        """Whether the fixed counter buffer has been allocated."""
        return self._counters is not None

    def initialize(self, device: torch.device) -> None:
        """Allocate the counter buffer once on ``device``."""
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._counters is None:
            self._counters = torch.zeros(4, dtype=torch.int64, device=device)
        elif self._counters.device != device:
            raise RuntimeError(f"MoE capacity tracker is on {self._counters.device}, not {device}.")

    def reset(self) -> None:
        """Clear all counters without replacing their storage."""
        self._require_counters().zero_()

    def snapshot(self) -> MoECapacitySnapshot:
        """Return a non-aliasing copy of all counters."""
        values = self._require_counters().clone()
        return MoECapacitySnapshot(values[0], values[1], values[2], values[3])

    def record_assignments(
        self,
        selected_assignments: torch.Tensor,
        dropped_assignments: torch.Tensor,
        valid_token_drops: torch.Tensor,
    ) -> None:
        """Accumulate router assignment and drop counts."""
        counters = self._require_counters()
        selected_assignments = self._validate_count(selected_assignments, counters.device)
        dropped_assignments = self._validate_count(dropped_assignments, counters.device)
        valid_token_drops = self._validate_count(valid_token_drops, counters.device)
        counters[0].add_(selected_assignments)
        counters[1].add_(dropped_assignments)
        counters[2].add_(valid_token_drops)

    def record_rank_overflow(self, rank_overflow_events: torch.Tensor) -> None:
        """Accumulate HybridEP static rank-capacity overflow events."""
        counters = self._require_counters()
        rank_overflow_events = self._validate_count(rank_overflow_events, counters.device)
        counters[3].add_(rank_overflow_events)

    def _require_counters(self) -> torch.Tensor:
        if self._counters is None:
            raise RuntimeError("MoE capacity tracker is not initialized.")
        return self._counters

    @staticmethod
    def _validate_count(count: torch.Tensor, device: torch.device) -> torch.Tensor:
        if count.device != device:
            raise RuntimeError(f"MoE capacity tracker is on {device}, not {count.device}.")
        if count.numel() != 1:
            raise RuntimeError("MoE capacity tracker records must be scalar tensors.")
        torch._assert_async(count >= 0, "MoE capacity tracker records must be nonnegative.")
        return count.to(dtype=torch.int64).reshape(())


_MOE_CAPACITY_TRACKER: MoECapacityTracker | None = None


def get_moe_capacity_tracker() -> MoECapacityTracker:
    """Return the process-global MoE capacity tracker, creating it if needed."""
    global _MOE_CAPACITY_TRACKER
    if _MOE_CAPACITY_TRACKER is None:
        _MOE_CAPACITY_TRACKER = MoECapacityTracker()
    return _MOE_CAPACITY_TRACKER


def destroy_moe_capacity_tracker() -> None:
    """Destroy the process-global MoE capacity tracker."""
    global _MOE_CAPACITY_TRACKER
    _MOE_CAPACITY_TRACKER = None
