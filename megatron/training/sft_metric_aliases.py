# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import TypeVar

LossScalar = TypeVar("LossScalar")


def build_sft_metric_aliases(
    *,
    enabled: bool,
    e2e_step_time_s: float,
    main_lm_loss: LossScalar | None,
) -> dict[str, float | LossScalar]:
    if not enabled:
        return {}

    metrics: dict[str, float | LossScalar] = {
        "performance/e2e_step_time_s": e2e_step_time_s,
    }
    if main_lm_loss is not None:
        metrics["accuracy/main_lm_loss"] = main_lm_loss
    return metrics
