from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path


CUDA_DRIVER_LIBRARY = "libcuda.so.1"
EXTENSION_MODULES = ("deep_ep_cpp", "hybrid_ep_cpp")
NEEDED_PATTERN = re.compile(r"\(NEEDED\).*Shared library: \[([^]]+)]")


def extension_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"extension module is unavailable: {module_name}")
    path = Path(spec.origin)
    if not path.is_file():
        raise RuntimeError(
            f"extension module has no shared object: {module_name}: {path}"
        )
    return path


def needed_libraries(path: Path) -> set[str]:
    result = subprocess.run(
        ["readelf", "-d", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"readelf failed for {path}: {detail}")
    return set(NEEDED_PATTERN.findall(result.stdout))


def main() -> int:
    try:
        for module_name in EXTENSION_MODULES:
            dependencies = needed_libraries(extension_path(module_name))
            if CUDA_DRIVER_LIBRARY not in dependencies:
                raise RuntimeError(
                    f"{module_name} does not require {CUDA_DRIVER_LIBRARY}: "
                    f"found {sorted(dependencies)}"
                )
            print(f"cuda_driver_dependency={module_name}:{CUDA_DRIVER_LIBRARY}")
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
