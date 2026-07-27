# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import megatron.training.initialize as initialize


def test_jit_warmup_precedes_optional_tp_dp_cp_communicator_warmup(monkeypatch):
    events = []

    monkeypatch.setattr(initialize, "is_torch_min_version", lambda _: True)
    monkeypatch.setattr(
        initialize,
        "_warmup_jit_function",
        lambda tp_size=None: events.append(("jit", tp_size)),
    )
    monkeypatch.setattr(
        initialize.mpu,
        "_warmup_tensor_and_data_parallel_group_with_cp_if_requested",
        lambda: events.append(("communicator", None)),
    )

    initialize.set_jit_fusion_options(tp_size=8)

    assert events == [("jit", 8), ("communicator", None)]
