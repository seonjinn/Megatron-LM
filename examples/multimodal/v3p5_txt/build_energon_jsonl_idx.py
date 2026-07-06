#!/usr/bin/env python3
"""Build Energon 7.x sidecar indexes for JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.jsonl.ijsonl import (
    IJSONL_SUFFIX,
    IJsonlIndexReader,
    IJsonlIndexWriter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL file(s) or directories containing JSONL files to index",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing index")
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob used when an input path is a directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index at most this many JSONL files after path expansion",
    )
    parser.add_argument(
        "--link-root",
        type=Path,
        default=None,
        help=(
            "Optional writable directory where symlinks to the source JSONLs are created. "
            "Indexes are written next to those symlinks instead of next to the source files."
        ),
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Optional MetadatasetV2 YAML to write for the indexed JSONLs",
    )
    parser.add_argument(
        "--cook",
        default="openai_messages_jsonl",
        help="subflavors.cook value to write when --output-yaml is used",
    )
    parser.add_argument(
        "--category",
        default="openai_messages_jsonl",
        help="subflavors.category value to write when --output-yaml is used",
    )
    parser.add_argument(
        "--repetitions",
        type=float,
        default=1.0,
        help="blend_epochized repetitions value to write when --output-yaml is used",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Write subflavors.skip_chat_template: true in the YAML",
    )
    parser.add_argument(
        "--absolute-paths-in-yaml",
        action="store_true",
        help="Write absolute JSONL paths in YAML instead of paths relative to the YAML",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of JSONL index build workers",
    )
    return parser.parse_args()


def iter_jsonl_paths(paths: list[str], pattern: str) -> list[Path]:
    jsonl_paths: list[Path] = []
    for path_arg in paths:
        path = Path(path_arg)
        if path.is_dir():
            jsonl_paths.extend(sorted(p for p in path.glob(pattern) if p.is_file()))
        elif path.is_file():
            jsonl_paths.append(path)
        else:
            raise FileNotFoundError(path)
    return jsonl_paths


def prepare_index_target(source_path: Path, link_root: Path | None) -> Path:
    if link_root is None:
        return source_path

    link_root.mkdir(parents=True, exist_ok=True)
    link_path = link_root / source_path.name
    if link_path.is_symlink():
        existing_target = Path(os.readlink(link_path))
        if not existing_target.is_absolute():
            existing_target = (link_path.parent / existing_target).resolve()
        if existing_target.resolve() != source_path.resolve():
            raise FileExistsError(
                f"{link_path} points to {existing_target}, not {source_path}"
            )
    elif link_path.exists():
        raise FileExistsError(f"{link_path} already exists and is not a symlink")
    else:
        link_path.symlink_to(source_path.resolve())
    return link_path


def build_index(jsonl_path: Path, force: bool = False) -> int:
    index_path = EPath(str(jsonl_path)).with_suffix(IJSONL_SUFFIX)
    if index_path.is_file() and not force:
        print(f"index_exists={index_path}")
        print(f"count_samples={IJsonlIndexReader.count_samples(str(jsonl_path))}")
        return IJsonlIndexReader.count_samples(str(jsonl_path))

    count = 0
    offset = 0
    with IJsonlIndexWriter(EPath(str(jsonl_path))) as writer:
        writer.append(offset)
        with jsonl_path.open("rb") as handle:
            for line in handle:
                offset += len(line)
                writer.append(offset)
                count += 1

    print(f"jsonl_path={jsonl_path}")
    print(f"index_path={index_path}")
    print(f"count_lines={count}")
    print(f"count_samples={IJsonlIndexReader.count_samples(str(jsonl_path))}")
    print(f"final_offset={offset}")
    print(f"index_suffix={IJSONL_SUFFIX}")
    return count


def yaml_scalar(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def absolute_path_without_resolving_symlinks(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def yaml_dataset_path(jsonl_path: Path, output_yaml: Path, absolute_paths: bool) -> str:
    jsonl_path = absolute_path_without_resolving_symlinks(jsonl_path)
    if absolute_paths:
        return str(jsonl_path)
    output_yaml_parent = absolute_path_without_resolving_symlinks(output_yaml).parent
    return os.path.relpath(jsonl_path, output_yaml_parent)


def write_metadataset_yaml(
    output_yaml: Path,
    entries: list[tuple[Path, int]],
    *,
    cook: str,
    category: str,
    repetitions: float,
    skip_chat_template: bool,
    absolute_paths: bool,
) -> None:
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "__module__: megatron.energon",
        "__class__: MetadatasetV2",
        "splits:",
        "  train:",
        "    blend_epochized:",
    ]
    for jsonl_path, count in entries:
        dataset_path = yaml_dataset_path(jsonl_path, output_yaml, absolute_paths)
        lines.extend(
            [
                f"    - repetitions: {repetitions}",
                f"      path: {yaml_scalar(dataset_path)}",
                "      subflavors:",
            ]
        )
        if skip_chat_template:
            lines.append("        skip_chat_template: true")
        lines.extend(
            [
                f"        name: {yaml_scalar(jsonl_path.name)}",
                f"        cook: {yaml_scalar(cook)}",
                f"        length: {count}",
                f"        orig_length: {count}",
                f"        category: {yaml_scalar(category)}",
            ]
        )
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output_yaml={output_yaml}")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    jsonl_paths = iter_jsonl_paths(args.paths, args.glob)
    if args.limit is not None:
        jsonl_paths = jsonl_paths[: args.limit]
    if not jsonl_paths:
        raise ValueError("No JSONL files found")

    index_targets = [
        prepare_index_target(source_path, args.link_root) for source_path in jsonl_paths
    ]
    build_one = partial(build_index, force=args.force)
    if args.workers == 1:
        counts = [build_one(jsonl_path) for jsonl_path in index_targets]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            counts = list(executor.map(build_one, index_targets))

    total = sum(counts)
    yaml_entries = list(zip(index_targets, counts))

    if args.output_yaml is not None:
        write_metadataset_yaml(
            args.output_yaml,
            yaml_entries,
            cook=args.cook,
            category=args.category,
            repetitions=args.repetitions,
            skip_chat_template=args.skip_chat_template,
            absolute_paths=args.absolute_paths_in_yaml,
        )

    print(f"indexed_files={len(jsonl_paths)}")
    print(f"total_samples={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
