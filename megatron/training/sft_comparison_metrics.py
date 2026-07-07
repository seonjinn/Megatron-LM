# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common comparison metrics emitted by Megatron-LM SFT training."""

import math
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast


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
    train_step_time_s: int | float
    validation_time_s: int | float | None = None
    main_lm_loss: int | float | None = None
    validation_loss: int | float | None = None
    grad_norm: int | float | None = None
    learning_rate: int | float | None = None


@dataclass(frozen=True)
class SFTValidationResult:
    """Outcome of validation associated with one SFT training event.

    Attributes:
        attempted: Whether in-loop validation started for the event.
        completed: Whether every requested validation iteration completed.
        validation_time_s: Last-rank wall duration observed around evaluation.
        validation_loss: Native language-model validation loss, when available.
    """

    attempted: bool
    completed: bool
    validation_time_s: int | float | None = None
    validation_loss: int | float | None = None


class ComparisonMetricWriter(Protocol):
    """Minimal W&B-compatible writer interface used by comparison logging."""

    def log(
        self,
        data: dict[str, float | int],
        *,
        step: int,
        commit: bool,
    ) -> None:
        """Log one comparison payload."""


def validate_sft_comparison_configuration(*, enabled: bool, log_interval: int) -> None:
    """Validate configuration required for exact per-step comparison metrics.

    Args:
        enabled: Whether common comparison metric logging was requested.
        log_interval: Native Megatron-LM logging interval.

    Raises:
        ValueError: If comparison logging is enabled without per-step native logging.
    """

    if enabled and log_interval != 1:
        raise ValueError(
            "--log-comparison-metrics requires --log-interval 1 "
            "so comparison metrics retain exact per-step semantics"
        )


def _normalize_step(value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"step must be a Python int, got {type(value).__name__}")
    return int(value)


def _normalize_float(field_name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(
            f"{field_name} must be a Python int or float, got {type(value).__name__}"
        )
    normalized = float(cast(int | float, value))
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return normalized


def _add_optional_metric(
    metrics: dict[str, float | int],
    metric_name: str,
    field_name: str,
    value: int | float | None,
) -> None:
    if value is None:
        return
    metrics[metric_name] = _normalize_float(field_name, value)


def _build_training_metrics(
    observation: SFTComparisonObservation,
) -> tuple[dict[str, float | int], float]:
    step = _normalize_step(observation.step)
    train_step_time_s = _normalize_float(
        "train_step_time_s", observation.train_step_time_s
    )

    metrics: dict[str, float | int] = {
        "comparison/step": step,
        "performance/train_step_time_s": train_step_time_s,
    }
    optional_metrics = (
        ("accuracy/main_lm_loss", "main_lm_loss", observation.main_lm_loss),
        ("accuracy/grad_norm", "grad_norm", observation.grad_norm),
        ("accuracy/learning_rate", "learning_rate", observation.learning_rate),
    )
    for metric_name, field_name, value in optional_metrics:
        _add_optional_metric(metrics, metric_name, field_name, value)
    return metrics, train_step_time_s


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

    metrics, train_step_time_s = _build_training_metrics(observation)
    metrics["performance/e2e_step_time_s"] = train_step_time_s
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
    validation_time_s = _normalize_float(
        "validation_time_s", observation.validation_time_s
    )

    metrics, train_step_time_s = _build_training_metrics(observation)
    e2e_step_time_s = train_step_time_s + validation_time_s
    if not math.isfinite(e2e_step_time_s):
        raise ValueError(f"e2e_step_time_s must be finite, got {e2e_step_time_s!r}")
    metrics["performance/e2e_step_time_s"] = e2e_step_time_s
    metrics["performance/validation_time_s"] = validation_time_s
    _add_optional_metric(
        metrics,
        "accuracy/validation_loss",
        "validation_loss",
        observation.validation_loss,
    )
    metrics["context/is_validation_step"] = 1
    return metrics


def log_sft_comparison_event(
    *,
    writer: ComparisonMetricWriter,
    observation: SFTComparisonObservation,
    validation_result: SFTValidationResult,
    event_scope: Literal["training", "final_validation"],
) -> bool:
    """Emit one complete common row for an eligible SFT training event.

    Args:
        writer: W&B-compatible metric writer.
        observation: Native training scalars for the current event.
        validation_result: Explicit validation attempt and completion state.
        event_scope: Training-loop events are eligible; final validation is not.

    Returns:
        Whether a comparison payload was emitted.

    Raises:
        ValueError: If the event scope or validation state is inconsistent.
        TypeError: If any emitted observation uses a non-Python scalar.
    """

    if event_scope == "final_validation":
        return False
    if event_scope != "training":
        raise ValueError(f"unsupported comparison event scope: {event_scope!r}")
    if validation_result.attempted and not validation_result.completed:
        return False
    if validation_result.completed and not validation_result.attempted:
        raise ValueError("completed validation must have attempted=True")

    if validation_result.completed:
        payload = build_validation_comparison_metrics(
            replace(
                observation,
                validation_time_s=validation_result.validation_time_s,
                validation_loss=validation_result.validation_loss,
            )
        )
    else:
        payload = build_training_comparison_metrics(observation)

    step = payload["comparison/step"]
    if type(step) is not int:
        raise AssertionError("normalized comparison step must be an int")
    writer.log(payload, step=step, commit=False)
    return True
