# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.capacity_tracker import (
    MoECapacityTracker,
    destroy_moe_capacity_tracker,
    get_moe_capacity_tracker,
)


@pytest.fixture(autouse=True)
def clean_global_tracker():
    destroy_moe_capacity_tracker()
    yield
    destroy_moe_capacity_tracker()


def test_records_require_initialization() -> None:
    tracker = MoECapacityTracker()

    with pytest.raises(RuntimeError, match="not initialized"):
        tracker.record_assignments(torch.tensor(1), torch.tensor(0), torch.tensor(0))
    with pytest.raises(RuntimeError, match="not initialized"):
        tracker.record_rank_overflow(torch.tensor(0))


def test_initialize_and_reset_preserve_counter_storage() -> None:
    tracker = MoECapacityTracker()
    tracker.initialize(torch.device("cpu"))
    counters = tracker._counters
    assert counters is not None
    pointer = counters.data_ptr()

    tracker.initialize(torch.device("cpu"))
    tracker.record_assignments(torch.tensor(4), torch.tensor(2), torch.tensor(1))
    tracker.record_rank_overflow(torch.tensor(1))
    tracker.reset()

    assert tracker._counters is counters
    assert tracker._counters.data_ptr() == pointer
    torch.testing.assert_close(tracker._counters, torch.zeros(4, dtype=torch.int64))


def test_snapshot_does_not_alias_mutable_counters() -> None:
    tracker = MoECapacityTracker()
    tracker.initialize(torch.device("cpu"))
    tracker.record_assignments(torch.tensor(5), torch.tensor(2), torch.tensor(1))
    tracker.record_rank_overflow(torch.tensor(1))

    snapshot = tracker.snapshot()
    tracker.reset()

    assert snapshot.selected_assignments == 5
    assert snapshot.dropped_assignments == 2
    assert snapshot.valid_token_drops == 1
    assert snapshot.rank_overflow_events == 1
    assert snapshot.selected_assignments.data_ptr() != tracker._counters.data_ptr()


@pytest.mark.parametrize(
    "record",
    [
        lambda tracker: tracker.record_assignments(
            torch.tensor(-1), torch.tensor(0), torch.tensor(0)
        ),
        lambda tracker: tracker.record_assignments(
            torch.tensor(1), torch.tensor(-1), torch.tensor(0)
        ),
        lambda tracker: tracker.record_assignments(
            torch.tensor(1), torch.tensor(0), torch.tensor(-1)
        ),
        lambda tracker: tracker.record_rank_overflow(torch.tensor(-1)),
    ],
)
def test_negative_records_are_rejected(record) -> None:
    tracker = MoECapacityTracker()
    tracker.initialize(torch.device("cpu"))

    with pytest.raises(RuntimeError, match="nonnegative"):
        record(tracker)


def test_device_change_and_record_device_mismatch_are_rejected() -> None:
    tracker = MoECapacityTracker()
    tracker.initialize(torch.device("cpu"))

    with pytest.raises(RuntimeError, match="not meta"):
        tracker.initialize(torch.device("meta"))
    with pytest.raises(RuntimeError, match="not meta"):
        tracker.record_rank_overflow(torch.empty((), device="meta", dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unindexed_cuda_device_is_idempotent_for_current_device() -> None:
    tracker = MoECapacityTracker()
    tracker.initialize(torch.device("cuda"))
    counters = tracker._counters
    assert counters is not None

    tracker.initialize(torch.device("cuda", torch.cuda.current_device()))

    assert tracker._counters is counters


def test_records_accumulate_and_cast_scalar_tensors_to_int64() -> None:
    tracker = MoECapacityTracker()
    tracker.initialize(torch.device("cpu"))

    tracker.record_assignments(
        torch.tensor(3, dtype=torch.int32),
        torch.tensor(1, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
    )
    tracker.record_assignments(
        torch.tensor(5, dtype=torch.int32),
        torch.tensor(2, dtype=torch.int32),
        torch.tensor(2, dtype=torch.int32),
    )
    tracker.record_rank_overflow(torch.tensor(True))

    snapshot = tracker.snapshot()
    assert snapshot.selected_assignments.dtype == torch.int64
    assert snapshot.selected_assignments == 8
    assert snapshot.dropped_assignments == 3
    assert snapshot.valid_token_drops == 2
    assert snapshot.rank_overflow_events == 1


def test_global_tracker_owner_can_be_destroyed() -> None:
    tracker = get_moe_capacity_tracker()
    assert get_moe_capacity_tracker() is tracker

    destroy_moe_capacity_tracker()

    assert get_moe_capacity_tracker() is not tracker


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_records_accumulate_during_cuda_graph_capture() -> None:
    tracker = MoECapacityTracker()
    device = torch.device("cuda", torch.cuda.current_device())
    tracker.initialize(device)
    assignments = torch.tensor(3, device=device)
    drops = torch.tensor(1, device=device)
    valid_drops = torch.tensor(1, device=device)
    overflow = torch.tensor(True, device=device)
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        tracker.record_assignments(assignments, drops, valid_drops)
        tracker.record_rank_overflow(overflow)

    graph.replay()
    snapshot = tracker.snapshot()
    assert snapshot.selected_assignments == 3
    assert snapshot.dropped_assignments == 1
    assert snapshot.valid_token_drops == 1
    assert snapshot.rank_overflow_events == 1

    graph.replay()
    snapshot = tracker.snapshot()
    assert snapshot.selected_assignments == 6
    assert snapshot.dropped_assignments == 2
    assert snapshot.valid_token_drops == 2
    assert snapshot.rank_overflow_events == 2
