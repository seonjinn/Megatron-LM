# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import ast
import importlib.util
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER_PATH = _REPO_ROOT / "megatron/training/sft_comparison_metrics.py"
_ARGUMENTS_PATH = _REPO_ROOT / "megatron/training/arguments.py"
_TRAINING_PATH = _REPO_ROOT / "megatron/training/training.py"


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


def test_common_payload_is_logged_once_in_loop_not_by_final_validation() -> None:
    train = _function_node(_TRAINING_PATH, "train")
    evaluate_and_print_results = _function_node(_TRAINING_PATH, "evaluate_and_print_results")

    comparison_log_calls = []
    for node in ast.walk(train):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "log" or not node.args:
            continue
        if isinstance(node.args[0], ast.Name) and node.args[0].id == "comparison_payload":
            comparison_log_calls.append(node)

    assert len(comparison_log_calls) == 1
    assert len(comparison_log_calls[0].args) == 1
    assert [keyword.arg for keyword in comparison_log_calls[0].keywords] == ["step", "commit"]
    assert isinstance(comparison_log_calls[0].keywords[0].value, ast.Name)
    assert comparison_log_calls[0].keywords[0].value.id == "iteration"
    assert ast.literal_eval(comparison_log_calls[0].keywords[1].value) is False
    assert not any(
        isinstance(node, ast.Name) and node.id == "comparison_payload"
        for node in ast.walk(evaluate_and_print_results)
    )


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
