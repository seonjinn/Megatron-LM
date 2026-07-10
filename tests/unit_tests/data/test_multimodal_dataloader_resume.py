# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_DATALOADER_RESUME_PATH = (
    Path(__file__).parents[3] / "examples" / "multimodal" / "dataloader_resume.py"
)
_DATALOADER_RESUME_SPEC = importlib.util.spec_from_file_location(
    "dataloader_resume", _DATALOADER_RESUME_PATH
)
assert _DATALOADER_RESUME_SPEC is not None
assert _DATALOADER_RESUME_SPEC.loader is not None

dataloader_resume = importlib.util.module_from_spec(_DATALOADER_RESUME_SPEC)
_DATALOADER_RESUME_SPEC.loader.exec_module(dataloader_resume)

is_initial_checkpoint_load_without_dataloader_state = (
    dataloader_resume.is_initial_checkpoint_load_without_dataloader_state
)
should_strictly_load_dataloader_state = dataloader_resume.should_strictly_load_dataloader_state


def _args(**overrides) -> SimpleNamespace:
    values = {
        "iteration": 0,
        "finetune": False,
        "pretrained_checkpoint": None,
        "strict_dataloader_state_load": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_initial_finetune_load_tolerates_missing_dataloader_state():
    assert is_initial_checkpoint_load_without_dataloader_state(_args(finetune=True))


def test_initial_pretrained_load_tolerates_missing_dataloader_state():
    assert is_initial_checkpoint_load_without_dataloader_state(
        _args(pretrained_checkpoint="/checkpoints/initial")
    )


def test_nonzero_resume_requires_dataloader_state():
    assert not is_initial_checkpoint_load_without_dataloader_state(
        _args(iteration=250, finetune=True, pretrained_checkpoint="/checkpoints/initial")
    )


def test_iteration_zero_resume_without_initial_load_requires_dataloader_state():
    assert not is_initial_checkpoint_load_without_dataloader_state(_args())


def test_dataloader_state_load_is_tolerant_by_default():
    assert not should_strictly_load_dataloader_state(_args())


def test_dataloader_state_load_can_be_strict():
    assert should_strictly_load_dataloader_state(_args(strict_dataloader_state_load=True))
