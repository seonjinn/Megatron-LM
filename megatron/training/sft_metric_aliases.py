# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Mapping
from typing import Protocol, TypeVar

LossScalar = TypeVar("LossScalar")


class MetricWriter(Protocol):
    def log(self, payload: dict[str, object], step: int) -> None:
        """Write one metric payload at a training step."""


def build_sft_metric_aliases(
    *,
    enabled: bool,
    e2e_step_time_s: float,
    main_lm_loss: LossScalar,
) -> dict[str, float | LossScalar]:
    """Build aliases from scalars already produced by the native training loop.

    Args:
        enabled: Whether alias logging is enabled.
        e2e_step_time_s: Native iteration time for the strict no-validation run.
        main_lm_loss: Native SFT language-model loss object.

    Returns:
        The two alias metrics when enabled, otherwise an empty mapping.
    """
    if not enabled:
        return {}

    metrics: dict[str, float | LossScalar] = {
        "performance/e2e_step_time_s": e2e_step_time_s,
        "accuracy/main_lm_loss": main_lm_loss,
    }
    return metrics


def log_sft_metric_aliases(
    *,
    writer: MetricWriter | None,
    enabled: bool,
    is_sft: bool,
    iteration: int,
    e2e_step_time_s: float,
    loss_dict: Mapping[str, object],
) -> bool:
    """Emit strict SFT aliases without adding measurements or synchronization.

    Args:
        writer: Existing W&B-compatible writer, if configured.
        enabled: Whether alias logging was requested.
        is_sft: Whether the active workload is SFT.
        iteration: Native one-based training iteration.
        e2e_step_time_s: Native iteration time for the strict no-validation run.
        loss_dict: Native per-iteration loss mapping.

    Returns:
        True when one payload was emitted, otherwise False.

    Raises:
        KeyError: If enabled SFT logging does not contain the required `lm loss`.
    """
    if writer is None or not enabled or not is_sft:
        return False

    payload = build_sft_metric_aliases(
        enabled=True,
        e2e_step_time_s=e2e_step_time_s,
        main_lm_loss=loss_dict["lm loss"],
    )
    writer.log(payload, iteration)
    return True
