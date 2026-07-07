# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common comparison metrics emitted by Megatron-LM SFT training."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SFTComparisonObservation:
    """Native scalars for one SFT training event and optional validation.

    Attributes:
        step: One-based Megatron-LM training iteration.
        train_step_time_s: Native elapsed iteration time in seconds, excluding validation.
        validation_time_s: Rank-last wall time around the in-loop evaluation call.
        main_lm_loss: Native interval-averaged language-model training loss.
        validation_loss: Native language-model validation loss.
        grad_norm: Native training gradient norm.
        learning_rate: Native training learning rate.
    """

    step: int
    train_step_time_s: float
    validation_time_s: float | None = None
    main_lm_loss: float | None = None
    validation_loss: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None


def _add_optional_metric(
    metrics: dict[str, float | int],
    metric_name: str,
    field_name: str,
    value: float | None,
) -> None:
    if value is None:
        return
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    metrics[metric_name] = value


def _build_training_metrics(
    observation: SFTComparisonObservation,
) -> dict[str, float | int]:
    if not math.isfinite(observation.train_step_time_s):
        raise ValueError(
            f"train_step_time_s must be finite, got {observation.train_step_time_s!r}"
        )

    metrics: dict[str, float | int] = {
        "comparison/step": observation.step,
        "performance/train_step_time_s": observation.train_step_time_s,
    }
    optional_metrics = (
        ("accuracy/main_lm_loss", "main_lm_loss", observation.main_lm_loss),
        ("accuracy/grad_norm", "grad_norm", observation.grad_norm),
        ("accuracy/learning_rate", "learning_rate", observation.learning_rate),
    )
    for metric_name, field_name, value in optional_metrics:
        _add_optional_metric(metrics, metric_name, field_name, value)
    return metrics


def build_training_comparison_metrics(
    observation: SFTComparisonObservation,
) -> dict[str, float | int]:
    """Build one comparison payload for a non-validation SFT training event.

    Args:
        observation: Native scalars captured at a Megatron-LM log event.

    Returns:
        Shared comparison metrics. Unavailable optional values are omitted.

    Raises:
        ValueError: If an emitted scalar is NaN or infinite.
    """

    metrics = _build_training_metrics(observation)
    metrics["performance/e2e_step_time_s"] = observation.train_step_time_s
    metrics["context/is_validation_step"] = 0
    return metrics


def build_validation_comparison_metrics(
    observation: SFTComparisonObservation,
) -> dict[str, float | int]:
    """Build one comparison payload for an SFT training event with validation.

    Args:
        observation: Native training scalars plus the in-loop validation observation.

    Returns:
        Shared comparison metrics with validation-inclusive E2E time. Unavailable
        optional values are omitted.

    Raises:
        ValueError: If validation wall time is absent or an emitted scalar is
            NaN or infinite.
    """

    if observation.validation_time_s is None:
        raise ValueError("validation_time_s is required for validation metrics")
    if not math.isfinite(observation.validation_time_s):
        raise ValueError(
            f"validation_time_s must be finite, got {observation.validation_time_s!r}"
        )

    metrics = _build_training_metrics(observation)
    e2e_step_time_s = observation.train_step_time_s + observation.validation_time_s
    if not math.isfinite(e2e_step_time_s):
        raise ValueError(f"e2e_step_time_s must be finite, got {e2e_step_time_s!r}")
    metrics["performance/e2e_step_time_s"] = e2e_step_time_s
    metrics["performance/validation_time_s"] = observation.validation_time_s
    _add_optional_metric(
        metrics,
        "accuracy/validation_loss",
        "validation_loss",
        observation.validation_loss,
    )
    metrics["context/is_validation_step"] = 1
    return metrics
