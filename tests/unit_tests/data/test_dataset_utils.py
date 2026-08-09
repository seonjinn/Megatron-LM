# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import ModuleType

from megatron.core.datasets import utils as dataset_utils


def test_compile_helpers_uses_prebuilt_extension_without_makefile(monkeypatch, tmp_path):
    extension_name = "megatron.core.datasets.helpers_cpp"
    monkeypatch.setitem(sys.modules, extension_name, ModuleType(extension_name))
    monkeypatch.setattr(dataset_utils, "__file__", str(tmp_path / "utils.py"))

    dataset_utils.compile_helpers()
