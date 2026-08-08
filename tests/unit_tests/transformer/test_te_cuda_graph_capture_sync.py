# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from megatron.core.transformer import cuda_graphs as cuda_graphs_module
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper


def test_te_capture_synchronizes_ranks_after_te_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object | None]] = []
    captured_kwargs: dict[str, Any] = {}
    tp_cp_group = object()
    layer = SimpleNamespace()
    cuda_graph_helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    cuda_graph_helper.flattened_callables = [layer]
    cuda_graph_helper.callables_per_chunk = [[layer]]
    cuda_graph_helper.num_microbatches = 1
    cuda_graph_helper.config = SimpleNamespace(
        sequence_parallel=False, overlap_moe_expert_parallel_comm=False
    )
    cuda_graph_helper.pg_collection = SimpleNamespace(tp_cp=tp_cp_group)
    cuda_graph_helper._graphs_created = False
    cuda_graph_helper._graph_count = 0
    cuda_graph_helper._start_capturing = lambda: 0.0
    cuda_graph_helper._get_cuda_graph_input_data = lambda: ((object(),), {})
    cuda_graph_helper._finish_capturing = lambda start_time: None

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: events.append(("cuda_synchronize", None)),
    )

    def record_barrier(*args: Any, **kwargs: Any) -> None:
        group = kwargs.get("group", args[0] if args else None)
        events.append(("distributed_barrier", group))

    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        record_barrier,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def capture(
        callables: tuple[Any, ...], sample_args: tuple[Any, ...], **kwargs: Any
    ) -> tuple[object, ...]:
        captured_kwargs.update(kwargs)
        return (object(),)

    monkeypatch.setattr(cuda_graphs_module, "make_graphed_callables", capture, raising=False)

    cuda_graph_helper.create_cudagraphs()

    post_warmup_hook = captured_kwargs.get("post_warmup_hook")
    events_before_hook = tuple(events)
    if callable(post_warmup_hook):
        post_warmup_hook()

    assert (callable(post_warmup_hook), events_before_hook, events) == (
        True,
        (),
        [
            ("cuda_synchronize", None),
            ("distributed_barrier", tp_cp_group),
            ("cuda_synchronize", None),
        ],
    )
