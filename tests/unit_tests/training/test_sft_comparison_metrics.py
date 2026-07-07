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
_TRAINING_UTILS_PATH = _REPO_ROOT / "megatron/training/utils/common_utils.py"


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


class _TensorLikeScalar:
    def __init__(self, value: float) -> None:
        self.value = value
        self.item_calls = 0

    def item(self) -> float:
        self.item_calls += 1
        return self.value


class _FakeCudaScalar:
    def __init__(
        self,
        value: float,
        operations: list[str],
        *,
        is_copy: bool = False,
    ) -> None:
        self.value = value
        self.operations = operations
        self.is_copy = is_copy
        self.item_calls = 0

    def numel(self) -> int:
        self.operations.append("numel")
        return 1

    def detach(self) -> "_FakeCudaScalar":
        self.operations.append("detach")
        return self

    def reshape(self, size: int) -> "_FakeCudaScalar":
        self.operations.append(f"reshape:{size}")
        return self

    def clone(self) -> "_FakeCudaScalar":
        self.operations.append("clone")
        return _FakeCudaScalar(self.value, self.operations, is_copy=True)

    def to(self, *, dtype: object) -> "_FakeCudaScalar":
        self.operations.append(f"to:{dtype}")
        return self

    def item(self) -> float:
        self.operations.append("item")
        self.item_calls += 1
        return self.value


class _FakeDistributed:
    class ReduceOp:
        MAX = "max"

    def __init__(self, operations: list[str]) -> None:
        self.operations = operations
        self.reduced_tensor: _FakeCudaScalar | None = None

    def all_reduce(self, tensor: _FakeCudaScalar, *, op: object, group: object) -> None:
        assert tensor.is_copy
        assert op == self.ReduceOp.MAX
        self.operations.append("all_reduce")
        self.reduced_tensor = tensor


class _FakeCuda:
    @staticmethod
    def current_device() -> int:
        return 0


class _FakeTorch:
    Tensor = _FakeCudaScalar
    float32 = "float32"

    def __init__(self, operations: list[str]) -> None:
        self.operations = operations
        self.distributed = _FakeDistributed(operations)
        self.cuda = _FakeCuda()
        self.tensor_calls = 0
        self.tensor_inputs: list[object] = []

    def tensor(self, value: object, *, dtype: object, device: object) -> _FakeCudaScalar:
        self.tensor_calls += 1
        self.tensor_inputs.append(value)
        self.operations.append("torch.tensor")
        assert isinstance(value, list)
        assert len(value) == 1
        assert dtype == self.float32
        assert device == 0
        return _FakeCudaScalar(float(value[0]), self.operations, is_copy=True)


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


def _load_reducer(fake_torch: _FakeTorch) -> Callable[..., float | None]:
    function = _function_node(
        _TRAINING_UTILS_PATH,
        "reduce_max_stat_across_model_parallel_group",
    )
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future_annotations, function], type_ignores=[])
    )
    namespace = {"torch": fake_torch, "mpu": object()}
    exec(compile(module, _TRAINING_UTILS_PATH, "exec"), namespace)
    return namespace[function.name]


def test_builds_training_comparison_metrics() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=19,
        train_step_time_s=55.28,
        processed_tokens=16_631_382,
        num_gpus=512,
        main_lm_loss=2.5176,
        grad_norm=42.0,
        learning_rate=4.2e-7,
    )

    assert adapter.build_training_comparison_metrics(observation) == {
        "comparison/step": 19,
        "performance/train_step_time_s": 55.28,
        "performance/e2e_step_time_s": 55.28,
        "throughput/processed_tokens_per_second": pytest.approx(16_631_382 / 55.28),
        "throughput/processed_tokens_per_second_per_gpu": pytest.approx(
            16_631_382 / 55.28 / 512
        ),
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/processed_tokens": 16_631_382,
        "context/num_gpus": 512,
        "context/is_validation_step": 0,
    }


def test_builds_one_coherent_validation_step_payload() -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=55.28,
        processed_tokens=16_631_382,
        num_gpus=512,
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
        "throughput/processed_tokens_per_second": pytest.approx(16_631_382 / 55.28),
        "throughput/processed_tokens_per_second_per_gpu": pytest.approx(
            16_631_382 / 55.28 / 512
        ),
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/validation_loss": 2.5803,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/processed_tokens": 16_631_382,
        "context/num_gpus": 512,
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


def test_materializes_tensor_like_grad_norm_once_for_native_and_comparison() -> None:
    adapter = _load_adapter()
    producer_value = _TensorLikeScalar(42.25)

    grad_norm = adapter.normalize_sft_metric_producer_scalar(
        "grad_norm",
        producer_value,
    )
    native_wandb_payload = {"grad-norm": grad_norm}
    observation = adapter.capture_sft_comparison_step(
        state=adapter.SFTComparisonStepState(),
        step=1,
        train_active_time_s=4.5,
        advanced=True,
        main_lm_loss=2.5,
        grad_norm=grad_norm,
        learning_rate=1e-5,
        processed_tokens=64,
        num_gpus=8,
    )
    comparison_payload = adapter.build_training_comparison_metrics(observation)

    assert producer_value.item_calls == 1
    assert type(grad_norm) is float
    assert native_wandb_payload["grad-norm"] == 42.25
    assert comparison_payload["accuracy/grad_norm"] == 42.25


def test_reduced_logging_scalar_is_materialized_once_at_producer() -> None:
    reducer = _function_node(
        _TRAINING_UTILS_PATH,
        "reduce_max_stat_across_model_parallel_group",
    )
    item_calls = [
        node
        for node in ast.walk(reducer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]

    assert len(item_calls) == 1


def test_cuda_logging_scalar_stays_tensor_until_final_materialization() -> None:
    operations: list[str] = []
    fake_torch = _FakeTorch(operations)
    reducer = _load_reducer(fake_torch)
    producer_tensor = _FakeCudaScalar(42.25, operations)

    result = reducer(producer_tensor, group=object())

    assert result == 42.25
    assert fake_torch.tensor_calls == 0
    assert producer_tensor.item_calls == 0
    assert fake_torch.distributed.reduced_tensor is not producer_tensor
    assert fake_torch.distributed.reduced_tensor is not None
    assert fake_torch.distributed.reduced_tensor.item_calls == 1
    assert operations == [
        "numel",
        "detach",
        "reshape:1",
        "clone",
        "to:float32",
        "all_reduce",
        "item",
    ]


def test_reducer_type_accepts_tensor_float_or_none() -> None:
    reducer = _function_node(
        _TRAINING_UTILS_PATH,
        "reduce_max_stat_across_model_parallel_group",
    )

    assert ast.unparse(reducer.args.args[0].annotation) == "torch.Tensor | float | None"


@pytest.mark.parametrize(
    ("producer_value", "expected_result", "expected_tensor_input"),
    [
        (3.5, 3.5, [3.5]),
        (None, None, [-1.0]),
    ],
)
def test_reducer_preserves_python_scalar_and_none_behavior(
    producer_value: float | None,
    expected_result: float | None,
    expected_tensor_input: list[float],
) -> None:
    operations: list[str] = []
    fake_torch = _FakeTorch(operations)
    reducer = _load_reducer(fake_torch)

    result = reducer(producer_value, group=object())

    assert result == expected_result
    assert fake_torch.tensor_calls == 1
    assert fake_torch.tensor_inputs == [expected_tensor_input]
    assert operations == ["torch.tensor", "all_reduce", "item"]


def test_captures_independent_first_and_second_step_time_and_loss() -> None:
    adapter = _load_adapter()
    state = adapter.SFTComparisonStepState()

    first = adapter.capture_sft_comparison_step(
        state=state,
        step=1,
        train_active_time_s=10.0,
        advanced=True,
        main_lm_loss=2.75,
        grad_norm=4.0,
        learning_rate=1e-5,
        processed_tokens=64,
        num_gpus=8,
    )
    second = adapter.capture_sft_comparison_step(
        state=state,
        step=2,
        train_active_time_s=16.5,
        advanced=True,
        main_lm_loss=1.25,
        grad_norm=3.0,
        learning_rate=2e-5,
        processed_tokens=64,
        num_gpus=8,
    )

    assert first.train_step_time_s == 10.0
    assert first.main_lm_loss == 2.75
    assert second.train_step_time_s == 6.5
    assert second.main_lm_loss == 1.25


def test_capture_normalizes_exact_real_token_count_to_int() -> None:
    adapter = _load_adapter()

    observation = adapter.capture_sft_comparison_step(
        state=adapter.SFTComparisonStepState(),
        step=1,
        train_active_time_s=10.0,
        advanced=True,
        main_lm_loss=2.75,
        grad_norm=4.0,
        learning_rate=1e-5,
        processed_tokens=64.0,
        num_gpus=8,
    )

    assert observation is not None
    assert observation.processed_tokens == 64
    assert type(observation.processed_tokens) is int


def test_skipped_step_omits_loss_without_losing_timer_progress() -> None:
    adapter = _load_adapter()
    state = adapter.SFTComparisonStepState(train_active_time_s=16.5)

    skipped = adapter.capture_sft_comparison_step(
        state=state,
        step=3,
        train_active_time_s=21.0,
        advanced=False,
        main_lm_loss=0.0,
        grad_norm=None,
        learning_rate=2e-5,
        processed_tokens=64,
        num_gpus=8,
    )
    payload = adapter.build_training_comparison_metrics(skipped)

    assert skipped.train_step_time_s == 4.5
    assert skipped.main_lm_loss is None
    assert state.train_active_time_s == 21.0
    assert "accuracy/main_lm_loss" not in payload


def test_dummy_skip_suppresses_and_rebaselines_next_comparison_step() -> None:
    adapter = _load_adapter()
    state = adapter.SFTComparisonStepState()
    first = adapter.capture_sft_comparison_step(
        state=state,
        step=1,
        train_active_time_s=10.0,
        advanced=True,
        main_lm_loss=2.75,
        grad_norm=4.0,
        learning_rate=1e-5,
        processed_tokens=64,
        num_gpus=8,
    )

    adapter.invalidate_sft_comparison_step_timer(state)
    after_skip = adapter.capture_sft_comparison_step(
        state=state,
        step=3,
        train_active_time_s=24.0,
        advanced=True,
        main_lm_loss=2.25,
        grad_norm=3.5,
        learning_rate=1e-5,
        processed_tokens=64,
        num_gpus=8,
    )
    resumed = adapter.capture_sft_comparison_step(
        state=state,
        step=4,
        train_active_time_s=30.5,
        advanced=True,
        main_lm_loss=2.0,
        grad_norm=3.0,
        learning_rate=1e-5,
        processed_tokens=64,
        num_gpus=8,
    )

    assert first is not None
    assert first.train_step_time_s == 10.0
    assert after_skip is None
    assert resumed is not None
    assert resumed.step == 4
    assert resumed.train_step_time_s == 6.5
    assert resumed.main_lm_loss == 2.0
    assert state.train_active_time_s == 30.5


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


@pytest.mark.parametrize(
    ("processed_tokens", "num_gpus", "train_step_time_s", "field_name"),
    [
        (None, 512, 55.28, "processed_tokens"),
        (16_631_382, None, 55.28, "num_gpus"),
        (-1, 512, 55.28, "processed_tokens"),
        (16_631_382, 0, 55.28, "num_gpus"),
        (16_631_382, 512, 0.0, "train_step_time_s"),
    ],
)
def test_rejects_invalid_token_throughput_boundaries(
    processed_tokens: int | None,
    num_gpus: int | None,
    train_step_time_s: float,
    field_name: str,
) -> None:
    adapter = _load_adapter()
    observation = adapter.SFTComparisonObservation(
        step=20,
        train_step_time_s=train_step_time_s,
        processed_tokens=processed_tokens,
        num_gpus=num_gpus,
    )

    with pytest.raises(ValueError, match=field_name):
        adapter.build_training_comparison_metrics(observation)


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
        processed_tokens=16_631_382,
        num_gpus=512,
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
        "throughput/processed_tokens_per_second": pytest.approx(16_631_382 / 55.28),
        "throughput/processed_tokens_per_second_per_gpu": pytest.approx(
            16_631_382 / 55.28 / 512
        ),
        "accuracy/main_lm_loss": 2.5176,
        "accuracy/validation_loss": 2.5803,
        "accuracy/grad_norm": 42.0,
        "accuracy/learning_rate": 4.2e-7,
        "context/processed_tokens": 16_631_382,
        "context/num_gpus": 512,
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
        "wandb_writer.define_metric('throughput/*', step_metric='comparison/step')",
        "wandb_writer.define_metric('accuracy/*', step_metric='comparison/step')",
        "wandb_writer.define_metric('context/*', step_metric='comparison/step')",
    ]


def test_wandb_config_describes_exact_per_event_train_time() -> None:
    pretrain = _function_node(_TRAINING_PATH, "pretrain")
    config_updates = [
        ast.literal_eval(update)
        for node in ast.walk(pretrain)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and node.args
        and isinstance((update := node.args[0]), ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "comparison_metric_scopes"
            for key in update.keys
        )
    ]
    comparison_config = next(
        update["comparison_metric_scopes"]
        for update in config_updates
        if "comparison_metric_scopes" in update
    )

    assert (
        comparison_config["train_step_time_s"]
        == "native interval-time per-event active-time delta excluding validation"
    )


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


def test_training_log_uses_exact_current_step_producers() -> None:
    training_log = _function_node(_TRAINING_PATH, "training_log")
    calls = [
        ast.unparse(node)
        for node in ast.walk(training_log)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id
            in {
                "capture_sft_comparison_step",
                "normalize_sft_metric_producer_scalar",
            }
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "active_time"
        )
    ]

    assert "normalize_sft_metric_producer_scalar('grad_norm', grad_norm)" in calls
    assert "normalize_sft_metric_producer_scalar('main_lm_loss', loss_dict[key])" in calls
    assert "timers('interval-time').active_time()" in calls
    capture_call = next(call for call in calls if call.startswith("capture_sft_comparison_step("))
    assert "train_active_time_s=comparison_train_active_time_s" in capture_call
    assert "advanced=not bool(skipped_iter)" in capture_call
    assert "main_lm_loss=comparison_main_lm_loss" in capture_call
    assert "grad_norm=grad_norm" in capture_call


def test_training_log_uses_real_global_tokens_for_throughput() -> None:
    training_log = _function_node(_TRAINING_PATH, "training_log")
    capture_call = next(
        node
        for node in ast.walk(training_log)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "capture_sft_comparison_step"
    )
    capture_keywords = {keyword.arg: keyword.value for keyword in capture_call.keywords}

    assert ast.unparse(capture_keywords["processed_tokens"]) == (
        "total_real_tokens_in_batch if total_real_tokens_in_batch is not None "
        "else batch_size * args.seq_length"
    )
    assert ast.unparse(capture_keywords["num_gpus"]) == "args.world_size"


def test_dummy_train_step_invalidates_comparison_timer_before_continue() -> None:
    train = _function_node(_TRAINING_PATH, "train")
    skip_branch = next(
        node
        for node in ast.walk(train)
        if isinstance(node, ast.If) and "iterations_to_skip" in ast.unparse(node.test)
    )
    invalidation_calls = [
        node
        for node in ast.walk(skip_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "invalidate_sft_comparison_step_timer"
    ]
    continue_node = next(
        node for node in ast.walk(skip_branch) if isinstance(node, ast.Continue)
    )

    assert len(invalidation_calls) == 1
    assert ast.unparse(invalidation_calls[0]) == (
        "invalidate_sft_comparison_step_timer(comparison_state.step_state)"
    )
    assert invalidation_calls[0].lineno < continue_node.lineno


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
