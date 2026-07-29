# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for supported DeepEP import layouts."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest


class FakeBuffer:
    """Stand-in for the DeepEP buffer type during import."""


class FakeEventHandle:
    """Stand-in for the DeepEP event handle type during import."""


class FakeEventOverlap:
    """Stand-in for the DeepEP event overlap type during import."""


def load_fused_a2a_under_unique_name(layout: str) -> ModuleType:
    """Load fused_a2a without reusing a module imported by another test."""
    fused_a2a_path = (
        Path(__file__).resolve().parents[4] / "megatron/core/transformer/moe/fused_a2a.py"
    )
    module_name = f"test_fused_a2a_deepep_imports_{layout}"
    spec = spec_from_file_location(module_name, fused_a2a_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ("dd758_main", "f725_hybrid_ep"))
def test_fused_a2a_accepts_supported_deepep_exports(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """Enable DeepEP when event overlap is exported from the package root."""
    deep_ep = ModuleType("deep_ep")
    deep_ep.Buffer = FakeBuffer
    deep_ep.EventOverlap = FakeEventOverlap
    utils = ModuleType("deep_ep.utils")
    utils.EventHandle = FakeEventHandle
    if layout == "f725_hybrid_ep":
        utils.EventOverlap = FakeEventOverlap
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", utils)

    module = load_fused_a2a_under_unique_name(layout)

    assert module.HAVE_DEEP_EP is True
