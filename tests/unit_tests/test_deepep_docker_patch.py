from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEEP_EP_PATCH = REPOSITORY_ROOT / "docker" / "patches" / "deepep.patch"
DEPENDENCY_VERIFIER = REPOSITORY_ROOT / "docker" / "verify_cuda_driver_dependencies.py"
CUDA_DRIVER_STUB_DIR = "/usr/local/cuda/targets/sbsa-linux/lib/stubs"
PINNED_SETUP_SOURCE = """\
def get_extension_hybrid_ep_cpp():
    sources = [
        "csrc/hybrid_ep/hybrid_ep.cu",
    ]
    include_dirs = [
        "csrc/hybrid_ep/",
        "csrc/hybrid_ep/backend/",
    ]
    library_dirs = []
    libraries = ["cuda", "nvtx3interop"]
    extra_objects = []


def get_extension_deep_ep_cpp():
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']
"""


def _assigned_list(source: str, function_name: str, variable_name: str) -> list[str]:
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        )
    )
    value = ast.literal_eval(assignment.value)
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return value


def test_deepep_patch_links_extensions_against_cuda_driver_stub(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "DeepEP"
    checkout.mkdir()
    setup_path = checkout / "setup.py"
    setup_path.write_text(PINNED_SETUP_SOURCE, encoding="utf-8")

    result = subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(DEEP_EP_PATCH)],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    patched_source = setup_path.read_text(encoding="utf-8")
    for function_name in (
        "get_extension_hybrid_ep_cpp",
        "get_extension_deep_ep_cpp",
    ):
        assert _assigned_list(patched_source, function_name, "library_dirs") == [
            CUDA_DRIVER_STUB_DIR
        ]


def _write_readelf(path: Path, missing_module: str | None = None) -> None:
    condition = (
        f'if [[ "$2" == *"{missing_module}"* ]]; then'
        if missing_module is not None
        else "if false; then"
    )
    path.write_text(
        f"""#!/usr/bin/env bash
set -eu
{condition}
    printf '%s\n' ' 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]'
else
    printf '%s\n' ' 0x0000000000000001 (NEEDED) Shared library: [libcuda.so.1]'
fi
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_dependency_verifier(
    tmp_path: Path, missing_module: str | None = None
) -> subprocess.CompletedProcess[str]:
    modules = tmp_path / "modules"
    modules.mkdir()
    for module_name in ("deep_ep_cpp", "hybrid_ep_cpp"):
        (modules / f"{module_name}.so").touch()
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    _write_readelf(binary_dir / "readelf", missing_module)
    environment = os.environ | {
        "PATH": f"{binary_dir}:{os.environ['PATH']}",
        "PYTHONPATH": str(modules),
    }
    return subprocess.run(
        ["python3", str(DEPENDENCY_VERIFIER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dependency_verifier_accepts_both_cuda_driver_dependencies(
    tmp_path: Path,
) -> None:
    result = _run_dependency_verifier(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "cuda_driver_dependency=deep_ep_cpp:libcuda.so.1",
        "cuda_driver_dependency=hybrid_ep_cpp:libcuda.so.1",
    ]


def test_dependency_verifier_rejects_missing_cuda_driver_dependency(
    tmp_path: Path,
) -> None:
    result = _run_dependency_verifier(tmp_path, missing_module="hybrid_ep_cpp")

    assert result.returncode != 0
    assert "hybrid_ep_cpp" in result.stderr
    assert "libcuda.so.1" in result.stderr
