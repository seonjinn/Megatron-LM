"""Tests for the multimodal DynamicCP compatibility boundary."""

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).parents[2]
    / "megatron"
    / "training"
    / "dynamic_context_parallel.py"
)
_SPEC = importlib.util.spec_from_file_location("dynamic_context_parallel", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_dynamic_context_parallel_alias_enables_external_hybrid_scheduler():
    args = Namespace(dynamic_context_parallel=True, hybrid_context_parallel=False)

    _MODULE.normalize_dynamic_context_parallel_args(args)

    assert args.dynamic_context_parallel is True
    assert args.hybrid_context_parallel is True


def test_dynamic_and_legacy_context_parallel_flags_cannot_be_combined():
    args = Namespace(dynamic_context_parallel=True, hybrid_context_parallel=True)

    with pytest.raises(ValueError, match="Cannot set both"):
        _MODULE.normalize_dynamic_context_parallel_args(args)


@pytest.mark.parametrize("minimum_size", [0, 3])
def test_invalid_dynamic_context_parallel_minimum_size_is_rejected(minimum_size):
    args = Namespace(
        dynamic_context_parallel=True,
        hybrid_context_parallel=False,
        dynamic_context_parallel_min_size=minimum_size,
    )

    with pytest.raises(ValueError, match="power of two"):
        _MODULE.normalize_dynamic_context_parallel_args(args)
