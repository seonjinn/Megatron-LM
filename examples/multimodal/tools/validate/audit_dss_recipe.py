# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Statically validate DSS and nested YAML paths in an Energon recipe."""

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

DSS_URI_RE = re.compile(
    r"^(?P<scheme>filesystem\+dss|dss)://"
    r"(?P<dataset>[^/@]+)@(?P<version>[^/]+)"
    r"(?:/(?P<subpath>.*))?$"
)


@dataclass(frozen=True)
class Reference:
    """A path reference found in a recipe YAML."""

    source: Path
    kind: str
    value: str


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _path_values(value: Any) -> Iterable[str]:
    """Yield strings stored below keys named ``path``."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "path":
                yield from _strings(child)
            else:
                yield from _path_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _path_values(child)


def _aux_dss_values(value: Any) -> Iterable[str]:
    """Yield DSS strings stored below ``aux`` mappings."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "aux":
                for string in _strings(child):
                    if DSS_URI_RE.fullmatch(string):
                        yield string
            else:
                yield from _aux_dss_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _aux_dss_values(child)


def resolve_dss_uri(uri: str, cache_dir: Path) -> Path:
    """Resolve a DSS URI exactly as ``NVDATASET_CACHE_DIR`` does."""
    match = DSS_URI_RE.fullmatch(uri)
    if match is None:
        raise ValueError(f"Not a DSS URI: {uri}")
    path = cache_dir / match["dataset"] / match["version"]
    if match["subpath"]:
        path /= match["subpath"]
    return path


def collect_references(root: Path) -> tuple[list[Path], list[Reference], list[str]]:
    """Recursively parse local YAML includes and collect DSS references."""
    pending: list[tuple[Path, tuple[Path, ...]]] = [(root.resolve(), ())]
    documents: dict[Path, Any] = {}
    seen_yaml_files: set[Path] = set()
    yaml_files: list[Path] = []
    references: list[Reference] = []
    errors: list[str] = []

    while pending:
        yaml_path, ancestors = pending.pop()
        if yaml_path in ancestors:
            cycle = " -> ".join(str(path) for path in (*ancestors, yaml_path))
            errors.append(f"Recipe YAML include cycle: {cycle}")
            continue
        if yaml_path not in seen_yaml_files:
            seen_yaml_files.add(yaml_path)
            yaml_files.append(yaml_path)

        if not yaml_path.is_file():
            errors.append(f"Missing recipe YAML: {yaml_path}")
            continue
        if yaml_path in documents:
            document = documents[yaml_path]
        else:
            try:
                with yaml_path.open(encoding="utf-8") as stream:
                    document = yaml.safe_load(stream)
            except (OSError, yaml.YAMLError) as error:
                errors.append(f"Cannot parse recipe YAML {yaml_path}: {error}")
                continue
            documents[yaml_path] = document

        for path_value in _path_values(document):
            if DSS_URI_RE.fullmatch(path_value):
                references.append(Reference(yaml_path, "primary", path_value))
            elif path_value.endswith((".yaml", ".yml")):
                include = Path(path_value)
                if not include.is_absolute():
                    include = yaml_path.parent / include
                pending.append((include.resolve(), (*ancestors, yaml_path)))
            elif "://" not in path_value:
                local_path = Path(path_value)
                if not local_path.is_absolute():
                    local_path = yaml_path.parent / local_path
                if not local_path.exists():
                    errors.append(f"Missing local dataset path from {yaml_path}: {local_path}")

        for aux_value in _aux_dss_values(document):
            references.append(Reference(yaml_path, "aux", aux_value))

    return yaml_files, references, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively validate local YAML and DSS paths in an Energon recipe."
    )
    parser.add_argument("recipe", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("NVDATASET_CACHE_DIR", "/home/svc-dss/cache/nemotron")),
    )
    args = parser.parse_args()

    yaml_files, references, errors = collect_references(args.recipe)
    missing_dss: list[tuple[Reference, Path]] = []
    for reference in references:
        resolved = resolve_dss_uri(reference.value, args.cache_dir)
        if not resolved.exists():
            missing_dss.append((reference, resolved))
            continue

        match = DSS_URI_RE.fullmatch(reference.value)
        assert match is not None
        if (
            reference.kind == "primary"
            and match["scheme"] == "dss"
            and not match["subpath"]
            and resolved.is_dir()
            and not (resolved / ".nv-meta").exists()
        ):
            jsonl_files = sorted(resolved.glob("*.jsonl"))
            if jsonl_files:
                jsonl_names = ", ".join(path.name for path in jsonl_files)
                errors.append(
                    f"Unindexed JSONL directory used as a primary dataset from "
                    f"{reference.source}: {reference.value}. Reference an exact "
                    f"JSONL subpath instead; found: {jsonl_names}"
                )

    print(f"Recipe YAML files parsed: {len(yaml_files)}")
    print(f"DSS primary references: {sum(reference.kind == 'primary' for reference in references)}")
    print(f"DSS auxiliary references: {sum(reference.kind == 'aux' for reference in references)}")
    print(f"Unique DSS URIs: {len({reference.value for reference in references})}")

    repeated_primary = Counter(
        reference.value for reference in references if reference.kind == "primary"
    )
    repeated_primary = Counter({uri: count for uri, count in repeated_primary.items() if count > 1})
    if repeated_primary:
        print(f"Repeated primary DSS URIs: {len(repeated_primary)}")
        for uri, count in repeated_primary.most_common():
            print(f"  {count}x {uri}")

    for reference, resolved in missing_dss:
        errors.append(
            f"Missing {reference.kind} DSS path from {reference.source}: "
            f"{reference.value} -> {resolved}"
        )

    if errors:
        print(f"ERRORS: {len(errors)}")
        for error in errors:
            print(f"  {error}")
        return 1

    print("All referenced YAML, local dataset, and DSS paths exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
