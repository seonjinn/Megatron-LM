# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import ast
import importlib.util
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER_PATH = _REPO_ROOT / "megatron/training/sft_comparison_metrics.py"
_ARGUMENTS_PATH = _REPO_ROOT / "megatron/training/arguments.py"
_TRAINING_PATH = _REPO_ROOT / "megatron/training/training.py"


class _FakeWandbWriter:
    def __init__(self) -> None:
        self.log_calls: list[tuple[dict[str, float | int], int, bool]] = []

    def log(
        self,
        data: dict[str, float | int],
        *,
        step: int,
        commit: bool,
    ) -> None:
        self.log_calls.append((dict(data), step, commit))


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sft_comparison_metrics", _ADAPTER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_node(path: Path, function_name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found in {path}")


def _load_isolated_function(
    path: Path, function_name: str
) -> Callable[[argparse.ArgumentParser], argparse.ArgumentParser]:
    function = _function_node(path, function_name)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"argparse": argparse}
    exec(compile(module, path, "exec"), namespace)
    return namespace[function_name]


def test_builds_training_comparison_metrics() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=19,
        train_step_time_s=55.28,
        main_lm_loss=2.5176,
        grad_norm=42.0,
        learning_rate=4.2e-7,
    )

    assert adapter.build_training_comparison_metrics(observation) == {
        "comparison/step": 19,
        "performance/train_step_time_s": 55.28,
        "performance/e2e_step_time_s": 55.28,
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/is_validation_step": 0,
    }


def test_builds_one_coherent_validation_step_payload() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        validation_time_s=58.645,
        main_lm_loss=2.5176,
        validation_loss=2.5803,
        grad_norm=42.0,
        learning_rate=4.2e-7,
    )

    assert adapter.build_validation_comparison_metrics(observation) == {
        "comparison/step": 20,
        "performance/train_step_time_s": 55.28,
        "performance/e2e_step_time_s": pytest.approx(113.925),
        "performance/validation_time_s": 58.645,
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/validation_loss": 2.5803,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/is_validation_step": 1,
    }


def test_omits_unavailable_training_metrics() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=3,
        train_step_time_s=1.25,
    )

    assert adapter.build_training_comparison_metrics(observation) == {
        "comparison/step": 3,
        "performance/train_step_time_s": 1.25,
        "performance/e2e_step_time_s": 1.25,
        "context/is_validation_step": 0,
    }


def test_normalizes_accepted_python_numbers_to_schema_types() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55,
        validation_time_s=58,
        main_lm_loss=2,
        validation_loss=3,
        grad_norm=42,
        learning_rate=0,
    )

    metrics = adapter.build_validation_comparison_metrics(observation)

    assert metrics == {
        "comparison/step": 20,
        "performance/train_step_time_s": 55.0,
        "performance/e2e_step_time_s": 113.0,
        "performance/validation_time_s": 58.0,
        "accuracy/main_lm_loss": 2.0,
        "accuracy/validation_loss": 3.0,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 0.0,
        "context/is_validation_step": 1,
    }
    assert type(metrics["comparison/step"]) is int
    assert type(metrics["context/is_validation_step"]) is int
    assert all(
        type(value) is float
        for name, value in metrics.items()
        if name not in {"comparison/step", "context/is_validation_step"}
    )


@pytest.mark.parametrize(
    ("field_name", "value", "builder_name"),
    [
        ("step", True, "build_training_comparison_metrics"),
        ("step", 20.0, "build_training_comparison_metrics"),
        ("step", Decimal("20"), "build_training_comparison_metrics"),
        ("train_step_time_s", False, "build_training_comparison_metrics"),
        ("train_step_time_s", Decimal("55.28"), "build_training_comparison_metrics"),
        ("main_lm_loss", True, "build_training_comparison_metrics"),
        ("main_lm_loss", Decimal("2.5"), "build_training_comparison_metrics"),
        ("grad_norm", False, "build_training_comparison_metrics"),
        ("grad_norm", Decimal("42"), "build_training_comparison_metrics"),
        ("learning_rate", True, "build_training_comparison_metrics"),
        ("learning_rate", Decimal("0.1"), "build_training_comparison_metrics"),
        ("validation_time_s", False, "build_validation_comparison_metrics"),
        ("validation_time_s", Decimal("58.6"), "build_validation_comparison_metrics"),
        ("validation_loss", True, "build_validation_comparison_metrics"),
        ("validation_loss", Decimal("2.6"), "build_validation_comparison_metrics"),
    ],
)
def test_rejects_non_python_schema_scalars(
    field_name: str,
    value: object,
    builder_name: str,
) -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        validation_time_s=58.645,
    )

    with pytest.raises(TypeError, match=field_name):
        getattr(adapter, builder_name)(replace(observation, **{field_name: value}))


@pytest.mark.parametrize(
    ("field_name", "value", "builder_name"),
    [
        ("train_step_time_s", math.nan, "build_training_comparison_metrics"),
        ("main_lm_loss", math.inf, "build_training_comparison_metrics"),
        ("grad_norm", -math.inf, "build_training_comparison_metrics"),
        ("learning_rate", math.nan, "build_training_comparison_metrics"),
        ("validation_time_s", math.inf, "build_validation_comparison_metrics"),
        ("validation_loss", math.nan, "build_validation_comparison_metrics"),
    ],
)
def test_rejects_non_finite_metrics(field_name: str, value: float, builder_name: str) -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        validation_time_s=58.645,
    )

    with pytest.raises(ValueError, match=field_name):
        getattr(adapter, builder_name)(replace(observation, **{field_name: value}))


def test_validation_metrics_require_validation_wall_time() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        validation_loss=2.5803,
    )

    with pytest.raises(ValueError, match="validation_time_s"):
        adapter.build_validation_comparison_metrics(observation)


def test_rejects_non_finite_combined_e2e_time() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=1e308,
        validation_time_s=1e308,
    )

    with pytest.raises(ValueError, match="e2e_step_time_s"):
        adapter.build_validation_comparison_metrics(observation)


def test_comparison_metric_argument_is_opt_in() -> None:
    add_sft_args = _load_isolated_function(_ARGUMENTS_PATH, "_add_sft_args")
    parser = argparse.ArgumentParser()
    add_sft_args(parser)

    assert parser.parse_args([]).log_comparison_metrics is False
    assert parser.parse_args(["--log-comparison-metrics"]).log_comparison_metrics is True


@pytest.mark.parametrize("log_interval", [1, 2, 100])
def test_comparison_configuration_is_unrestricted_when_disabled(log_interval: int) -> None:
    adapter = _load_adapter()

    adapter.validate_sft_comparison_configuration(
        enabled=False,
        log_interval=log_interval,
    )


def test_comparison_configuration_accepts_per_step_logging() -> None:
    adapter = _load_adapter()

    adapter.validate_sft_comparison_configuration(enabled=True, log_interval=1)


@pytest.mark.parametrize("log_interval", [2, 20, 100])
def test_comparison_configuration_rejects_aggregated_logging(log_interval: int) -> None:
    adapter = _load_adapter()

    with pytest.raises(
        ValueError,
        match="--log-comparison-metrics requires --log-interval 1",
    ):
        adapter.validate_sft_comparison_configuration(
            enabled=True,
            log_interval=log_interval,
        )


def test_pretrain_validates_comparison_configuration_before_setup() -> None:
    pretrain = _function_node(_TRAINING_PATH, "pretrain")
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(pretrain)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "validate_sft_comparison_configuration",
            "set_jit_fusion_options",
        }
    }

    assert calls["validate_sft_comparison_configuration"] < calls["set_jit_fusion_options"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (argparse.Namespace(), False),
        (argparse.Namespace(sft=True), False),
        (argparse.Namespace(sft=False, log_comparison_metrics=True), False),
        (argparse.Namespace(sft=True, log_comparison_metrics=True), True),
        (
            argparse.Namespace(
                sft=True,
                log_comparison_metrics=True,
                perform_rl_step=True,
            ),
            False,
        ),
    ],
)
def test_comparison_metrics_are_sft_only(args: argparse.Namespace, expected: bool) -> None:
    is_enabled = _load_isolated_function(_TRAINING_PATH, "_sft_comparison_metrics_enabled")

    assert is_enabled(args) is expected


def test_logs_one_training_event_with_native_commit_semantics() -> None:
    adapter = _load_adapter()
    writer = _FakeWandbWriter()
    observation = adapter.SFTComparisonObservation(
        step=19,
        train_step_time_s=55.28,
        main_lm_loss=2.5176,
        grad_norm=42.0,
        learning_rate=4.2e-7,
    )
    validation_result = adapter.SFTValidationResult(
        attempted=False,
        completed=False,
    )

    emitted = adapter.log_sft_comparison_event(
        writer=writer,
        observation=observation,
        validation_result=validation_result,
        event_scope="training",
    )

    assert emitted is True
    assert writer.log_calls == [
        (
            {
                "comparison/step": 19,
                "performance/train_step_time_s": 55.28,
                "performance/e2e_step_time_s": 55.28,
                "accuracy/main_lm_loss": 2.5176,
                "accuracy/grad_norm": 42.0,
                "accuracy/learning_rate": 4.2e-7,
                "context/is_validation_step": 0,
            },
            19,
            False,
        )
    ]


def test_logs_one_validation_event_with_combined_e2e() -> None:
    adapter = _load_adapter()
    writer = _FakeWandbWriter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        main_lm_loss=2.5176,
        grad_norm=42.0,
        learning_rate=4.2e-7,
    )
    validation_result = adapter.SFTValidationResult(
        attempted=True,
        completed=True,
        validation_time_s=58.645,
        validation_loss=2.5803,
    )

    emitted = adapter.log_sft_comparison_event(
        writer=writer,
        observation=observation,
        validation_result=validation_result,
        event_scope="training",
    )

    assert emitted is True
    assert len(writer.log_calls) == 1
    payload, step, commit = writer.log_calls[0]
    assert payload == {
        "comparison/step": 20,
        "performance/train_step_time_s": 55.28,
        "performance/e2e_step_time_s": pytest.approx(113.925),
        "performance/validation_time_s": 58.645,
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/validation_loss": 2.5803,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/is_validation_step": 1,
    }
    assert step == 20
    assert commit is False


def test_suppresses_attempted_incomplete_validation_event() -> None:
    adapter = _load_adapter()
    writer = _FakeWandbWriter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
    )
    validation_result = adapter.SFTValidationResult(
        attempted=True,
        completed=False,
        validation_time_s=10.0,
    )

    emitted = adapter.log_sft_comparison_event(
        writer=writer,
        observation=observation,
        validation_result=validation_result,
        event_scope="training",
    )

    assert emitted is False
    assert writer.log_calls == []


def test_omits_final_validation_from_common_metrics() -> None:
    adapter = _load_adapter()
    writer = _FakeWandbWriter()
    observation = adapter.SFTComparisonObservation(
        step=200,
        train_step_time_s=55.28,
    )
    validation_result = adapter.SFTValidationResult(
        attempted=True,
        completed=True,
        validation_time_s=58.645,
        validation_loss=2.5803,
    )

    emitted = adapter.log_sft_comparison_event(
        writer=writer,
        observation=observation,
        validation_result=validation_result,
        event_scope="final_validation",
    )

    assert emitted is False
    assert writer.log_calls == []


def test_wandb_comparison_axis_matches_nemo_rl() -> None:
    pretrain = _function_node(_TRAINING_PATH, "pretrain")
    definitions = []
    for node in ast.walk(pretrain):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "define_metric":
            continue
        definitions.append(ast.unparse(node))

    assert definitions == [
        "wandb_writer.define_metric('comparison/step')",
        "wandb_writer.define_metric('performance/*', step_metric='comparison/step')",
        "wandb_writer.define_metric('accuracy/*', step_metric='comparison/step')",
        "wandb_writer.define_metric('context/*', step_metric='comparison/step')",
    ]


def test_training_loop_delegates_common_event_logging_once() -> None:
    train = _function_node(_TRAINING_PATH, "train")
    evaluate_and_print_results = _function_node(_TRAINING_PATH, "evaluate_and_print_results")

    comparison_log_calls = []
    for node in ast.walk(train):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "log_sft_comparison_event":
            comparison_log_calls.append(node)

    assert len(comparison_log_calls) == 1
    scope_keyword = next(
        keyword
        for keyword in comparison_log_calls[0].keywords
        if keyword.arg == "event_scope"
    )
    assert ast.literal_eval(scope_keyword.value) == "training"
    assert not any(
        isinstance(node, ast.Name) and node.id == "comparison_payload"
        for node in ast.walk(evaluate_and_print_results)
    )


def test_timelimit_returns_explicit_incomplete_validation_result() -> None:
    evaluate_and_print_results = _function_node(_TRAINING_PATH, "evaluate_and_print_results")
    timelimit_branch = next(
        node
        for node in ast.walk(evaluate_and_print_results)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "timelimit"
    )
    result_calls = [
        node.value
        for node in ast.walk(timelimit_branch)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    ]

    assert len(result_calls) == 1
    result_call = result_calls[0]
    assert isinstance(result_call.func, ast.Name)
    assert result_call.func.id == "SFTValidationResult"
    fields = {keyword.arg: ast.literal_eval(keyword.value) for keyword in result_call.keywords}
    assert fields["attempted"] is True
    assert fields["completed"] is False


def test_post_training_validation_explicitly_omits_common_metrics() -> None:
    pretrain = _function_node(_TRAINING_PATH, "pretrain")
    final_evaluation_calls = []
    for node in ast.walk(pretrain):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "evaluate_and_print_results":
            final_evaluation_calls.append(node)

    assert len(final_evaluation_calls) == 2
    for call in final_evaluation_calls:
        collect_keyword = next(
            (keyword for keyword in call.keywords if keyword.arg == "collect_comparison_metrics"),
            None,
        )
        assert collect_keyword is not None
        assert ast.literal_eval(collect_keyword.value) is False
