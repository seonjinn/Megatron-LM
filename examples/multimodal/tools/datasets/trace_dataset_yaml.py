#!/usr/bin/env python3
"""Trace a MetadatasetV2 YAML and find all datasets not under a given path.

Recursively resolves nested YAML references.  Relative paths in each YAML
are resolved relative to the parent directory of that YAML file.

Usage
-----
    python trace_dataset_yaml.py <yaml_path> <filter_path> [options]

Examples
--------
    # Find datasets whose media_source is not under a given tree
    python trace_dataset_yaml.py recipe.yaml /data/image_data --check media

    # Check both dataset path and media_source (default)
    python trace_dataset_yaml.py recipe.yaml /data/image_data

    # Only look at the train split
    python trace_dataset_yaml.py recipe.yaml /data/image_data --split train

    # Write results to a file
    python trace_dataset_yaml.py recipe.yaml /data/image_data -o out.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


import yaml


# ── data model ───────────────────────────────────────────────────────────────


@dataclass
class DatasetEntry:
    """A leaf dataset entry extracted from the YAML tree."""

    name: str
    raw_path: str
    resolved_path: str
    media_source: str
    media_source_path: str  # media_source with ``filesystem://`` stripped
    is_relative: bool
    source_yaml: str  # which YAML file this entry was defined in
    split: str = ""  # which split (train, val, …) this entry belongs to
    repetitions: Optional[float] = None
    weight: Optional[float] = None
    subflavors: dict = field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────────────────


def _resolve_path(raw_path: str, yaml_dir: str) -> str:
    """Resolve *raw_path* relative to *yaml_dir* if it is not absolute."""
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.normpath(os.path.join(yaml_dir, raw_path))


def _strip_media_source(media_source: str) -> str:
    """Strip the ``filesystem://`` prefix from a media_source value."""
    if media_source.startswith("filesystem://"):
        return media_source[len("filesystem://"):]
    return media_source


def _is_yaml_path(path: str) -> bool:
    """Heuristic: does this path reference another YAML file?"""
    return path.endswith(".yaml") or path.endswith(".yml")


def _is_under(path: str, parent: str) -> bool:
    """Return True if *path* is equal to or a child of *parent*."""
    p = os.path.normpath(path)
    par = os.path.normpath(parent)
    return p == par or p.startswith(par + os.sep)


# ── recursive parser ─────────────────────────────────────────────────────────


def _parse_entries(
    blend_list: list,
    yaml_dir: str,
    source_yaml: str,
    split: str,
    visited: Set[str],
) -> List[DatasetEntry]:
    """Parse a list of dataset / nested-YAML entries."""
    results: List[DatasetEntry] = []
    for entry in blend_list:
        if not isinstance(entry, dict):
            continue

        raw_path = entry.get("path", "")
        if not raw_path:
            continue

        resolved = _resolve_path(raw_path, yaml_dir)

        # Nested YAML → recurse.
        if _is_yaml_path(raw_path):
            nested = _load_yaml(resolved, visited, split_filter=None)
            # Propagate the current split name to nested entries.
            for ds in nested:
                if not ds.split:
                    ds.split = split
            results.extend(nested)
            continue

        # Leaf dataset.
        subflavors = entry.get("subflavors") or {}
        aux = entry.get("aux") or {}
        media_source = aux.get("media_source", "")
        name = subflavors.get("name", "") or os.path.basename(raw_path)

        results.append(
            DatasetEntry(
                name=name,
                raw_path=raw_path,
                resolved_path=resolved,
                media_source=media_source,
                media_source_path=_strip_media_source(media_source),
                is_relative=not os.path.isabs(raw_path),
                source_yaml=source_yaml,
                split=split,
                repetitions=entry.get("repetitions"),
                weight=entry.get("weight"),
                subflavors=subflavors,
            )
        )
    return results


def _parse_split(
    split_name: str,
    split_cfg: dict,
    yaml_dir: str,
    source_yaml: str,
    visited: Set[str],
) -> List[DatasetEntry]:
    """Extract dataset entries from a single split configuration.

    Supports ``blend_epochized``, ``blend``, ``datasets`` (legacy), and a
    direct ``path`` (single-dataset) structure.
    """
    results: List[DatasetEntry] = []

    for key in ("blend_epochized", "blend", "datasets"):
        blend_list = split_cfg.get(key)
        if blend_list and isinstance(blend_list, list):
            results.extend(
                _parse_entries(blend_list, yaml_dir, source_yaml, split_name, visited)
            )

    # Direct single-dataset path.
    if "path" in split_cfg and not any(
        k in split_cfg for k in ("blend_epochized", "blend", "datasets")
    ):
        raw_path = split_cfg["path"]
        resolved = _resolve_path(raw_path, yaml_dir)
        if _is_yaml_path(raw_path):
            nested = _load_yaml(resolved, visited, split_filter=None)
            for ds in nested:
                if not ds.split:
                    ds.split = split_name
            results.extend(nested)
        else:
            subflavors = split_cfg.get("subflavors") or {}
            aux = split_cfg.get("aux") or {}
            media_source = aux.get("media_source", "")
            name = subflavors.get("name", "") or os.path.basename(raw_path)
            results.append(
                DatasetEntry(
                    name=name,
                    raw_path=raw_path,
                    resolved_path=resolved,
                    media_source=media_source,
                    media_source_path=_strip_media_source(media_source),
                    is_relative=not os.path.isabs(raw_path),
                    source_yaml=source_yaml,
                    split=split_name,
                    subflavors=subflavors,
                )
            )

    return results


def _load_yaml(
    yaml_path: str,
    visited: Set[str] | None = None,
    split_filter: str | None = None,
) -> List[DatasetEntry]:
    """Recursively load a MetadatasetV2 / Metadataset YAML.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file.
    visited:
        Set of already-visited real-paths (cycle detection).
    split_filter:
        If given, only parse this split (e.g. ``"train"``).  ``None`` means
        all splits.
    """
    if visited is None:
        visited = set()

    real = os.path.realpath(yaml_path)
    if real in visited:
        print(
            f"WARNING: circular reference detected, skipping: {yaml_path}",
            file=sys.stderr,
        )
        return []
    visited.add(real)

    if not os.path.isfile(yaml_path):
        print(f"WARNING: YAML file not found: {yaml_path}", file=sys.stderr)
        return []

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print(f"WARNING: unexpected YAML structure in {yaml_path}", file=sys.stderr)
        return []

    splits: Dict = data.get("splits", {})
    if not isinstance(splits, dict):
        return []

    results: List[DatasetEntry] = []
    for split_name, split_cfg in splits.items():
        if split_filter and split_name != split_filter:
            continue
        if isinstance(split_cfg, dict):
            results.extend(
                _parse_split(split_name, split_cfg, yaml_dir, yaml_path, visited)
            )

    return results


# ── public API ───────────────────────────────────────────────────────────────


def load_datasets(
    yaml_path: str,
    split: str | None = None,
) -> List[DatasetEntry]:
    """Load all leaf dataset entries from *yaml_path* (recursively).

    Parameters
    ----------
    yaml_path:
        Root YAML config to parse.
    split:
        Restrict to a specific split (e.g. ``"train"``).  ``None`` parses all.
    """
    return _load_yaml(yaml_path, split_filter=split)


def filter_datasets(
    datasets: List[DatasetEntry],
    filter_path: str,
    *,
    check: str = "both",
) -> List[DatasetEntry]:
    """Return datasets that are NOT fully under *filter_path*.

    Parameters
    ----------
    check:
        What to check against *filter_path*:

        * ``"media"`` – only the media_source path.
        * ``"path"``  – only the resolved dataset path.
        * ``"both"``  – either one (a dataset is included if *any* of its
          paths is outside *filter_path*).
    """
    result: List[DatasetEntry] = []
    for ds in datasets:
        ms_under = (
            _is_under(ds.media_source_path, filter_path)
            if ds.media_source_path
            else False
        )
        path_under = _is_under(ds.resolved_path, filter_path)

        if check == "media":
            outside = not ms_under
        elif check == "path":
            outside = not path_under
        else:  # "both"
            outside = not ms_under or not path_under

        if outside:
            result.append(ds)
    return result


# ── report writer ────────────────────────────────────────────────────────────


def _write_report(
    datasets: List[DatasetEntry],
    total: int,
    filter_path: str,
    yaml_path: str,
    check_mode: str,
    out,
):
    """Write a human-readable report to *out* (file-like)."""
    # Split into two buckets: media_source outside, vs. only path outside.
    media_outside: List[DatasetEntry] = []
    path_only_outside: List[DatasetEntry] = []

    for ds in datasets:
        ms_under = (
            _is_under(ds.media_source_path, filter_path)
            if ds.media_source_path
            else False
        )
        if not ms_under:
            media_outside.append(ds)
        else:
            path_only_outside.append(ds)

    w = out.write
    sep = "=" * 100

    w(f"Datasets in {yaml_path} NOT under {filter_path}\n")
    w(f"Check mode: {check_mode}\n")
    w(f"Total datasets in YAML: {total}\n")
    w(f"Total non-matching: {len(datasets)}\n")
    w(f"{sep}\n\n")

    # ── Section 1: media_source outside ──────────────────────────────────
    w(f"SECTION 1: Datasets with media_source NOT under filter path [{len(media_outside)}]\n")
    w(f"{'-' * 100}\n\n")
    for i, ds in enumerate(media_outside, 1):
        _write_entry(w, i, ds, yaml_path)

    # ── Section 2: path outside but media_source inside ──────────────────
    if path_only_outside:
        w(f"\n{sep}\n")
        w(
            f"SECTION 2: Datasets whose path is outside filter path "
            f"(media_source is under it) [{len(path_only_outside)}]\n"
        )
        w(f"{'-' * 100}\n\n")
        for i, ds in enumerate(path_only_outside, 1):
            _write_entry(w, i, ds, yaml_path)

    # ── Unique media sources ─────────────────────────────────────────────
    unique_outside = sorted({ds.media_source for ds in media_outside if ds.media_source})
    if unique_outside:
        w(f"\n{sep}\n")
        w(f"Unique media sources NOT under filter path ({len(unique_outside)}):\n")
        w(f"{sep}\n\n")
        for ms in unique_outside:
            w(f"  {ms}\n")

    unique_inside = sorted({ds.media_source for ds in path_only_outside if ds.media_source})
    if unique_inside:
        w(f"\n{sep}\n")
        w(
            f"Unique media sources from path-only-outside datasets "
            f"(under filter path) ({len(unique_inside)}):\n"
        )
        w(f"{sep}\n\n")
        for ms in unique_inside:
            w(f"  {ms}\n")


def _write_entry(w, idx: int, ds: DatasetEntry, root_yaml: str):
    """Write one numbered dataset entry."""
    w(f"{idx}. {ds.name}\n")
    if ds.is_relative:
        w(f"   relative path: {ds.raw_path}\n")
        w(f"   resolved path: {ds.resolved_path}\n")
    else:
        w(f"   path: {ds.raw_path}\n")
    w(f"   media_source: {ds.media_source}\n")
    if ds.source_yaml != os.path.abspath(root_yaml):
        w(f"   source yaml: {ds.source_yaml}\n")
    w("\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find datasets in a MetadatasetV2 YAML that are not under a "
            "given path.  Recursively resolves nested YAML references."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("yaml_path", help="Path to the dataset YAML file.")
    parser.add_argument(
        "filter_path",
        help="Base path to filter against.  Datasets not under this path are reported.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write report to this file instead of stdout.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help='Only process this split (e.g. "train").  Default: all splits.',
    )
    parser.add_argument(
        "--check",
        choices=["media", "path", "both"],
        default="both",
        help=(
            "What to check against the filter path.  "
            '"media" = media_source only, '
            '"path" = dataset path only, '
            '"both" = either (default).'
        ),
    )
    args = parser.parse_args()

    all_datasets = load_datasets(args.yaml_path, split=args.split)
    outside = filter_datasets(all_datasets, args.filter_path, check=args.check)

    split_str = f" (split={args.split})" if args.split else ""
    print(f"Total datasets{split_str}: {len(all_datasets)}", file=sys.stderr)
    print(f"Outside filter path (check={args.check}): {len(outside)}", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            _write_report(
                outside, len(all_datasets), args.filter_path, args.yaml_path,
                args.check, f,
            )
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        _write_report(
            outside, len(all_datasets), args.filter_path, args.yaml_path,
            args.check, sys.stdout,
        )


if __name__ == "__main__":
    main()
