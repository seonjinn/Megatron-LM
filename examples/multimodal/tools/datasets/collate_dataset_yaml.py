#!/usr/bin/env python3
"""Collate external datasets from a MetadatasetV2 YAML under a single directory.

Given a dataset YAML and a target directory ("filter path"), this script:

1. Traces all datasets recursively (including nested YAML references).
2. For datasets / media sources whose paths are outside the target directory,
   maps them to new destinations under it and generates copy commands.
3. Generates rewritten YAML file(s) pointing to the new locations, preserving
   the recursive YAML structure when one exists.

Relative paths in each YAML are resolved relative to the parent directory of
that YAML file.

Usage
-----
    python collate_dataset_yaml.py <yaml> <filter_path> <output_dir> [options]

Examples
--------
    # Basic collation
    python collate_dataset_yaml.py recipe.yaml /data/image_data ./collated

    # Generate per-owner copy scripts
    python collate_dataset_yaml.py recipe.yaml /data/image_data ./collated --by-owner

Output structure
----------------
    <output_dir>/
        copy_all.sh               # All copy commands in one script
        copy_<owner>.sh           # Per-owner scripts (with --by-owner)
        yamls/
            <root>.yaml           # Rewritten root YAML
            <nested>.yaml         # Rewritten nested YAMLs (if any)
        summary.txt               # Human-readable summary of all mappings
"""

from __future__ import annotations

import argparse
import os
import pwd
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import yaml


# ── utilities ────────────────────────────────────────────────────────────────
# Shared with trace_dataset_yaml.py; duplicated here so the script is
# self-contained and can be run without any package setup.


def _resolve_path(raw_path: str, yaml_dir: str) -> str:
    """Resolve *raw_path* relative to *yaml_dir* when it is not absolute."""
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.normpath(os.path.join(yaml_dir, raw_path))


def _strip_media_source(media_source: str) -> str:
    """Strip the ``filesystem://`` prefix from a media_source value."""
    if media_source.startswith("filesystem://"):
        return media_source[len("filesystem://"):]
    return media_source


def _is_yaml_path(path: str) -> bool:
    """Heuristic: does *path* reference another YAML file?"""
    return path.endswith(".yaml") or path.endswith(".yml")


def _is_under(path: str, parent: str) -> bool:
    """Return True if *path* is equal to or a child of *parent*."""
    p = os.path.normpath(path)
    par = os.path.normpath(parent)
    return p == par or p.startswith(par + os.sep)


def _extract_owner(path: str) -> str:
    """Best-effort extraction of a dataset owner from a filesystem path.

    Resolution order:

    1. ``/users/<name>/`` anywhere in the path  → *name*.
    2. ``/projects/<project>/...`` without a ``/users/`` segment → try
       ``os.stat`` on the first few ancestor directories that exist to get
       the real filesystem owner.  This resolves project-level directories
       like ``llmservice_fm_vision/camio_data/`` to their actual creator.
    3. Fall back to the project name (e.g. ``llmservice_fm_vision``).
    4. ``"unknown"`` if nothing matched.
    """
    parts = path.replace("\\", "/").split("/")

    # 1. Explicit /users/<name>/ wins.
    for i, part in enumerate(parts):
        if part == "users" and i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]

    # 2/3. /projects/<project>/... — try stat, fall back to project name.
    for i, part in enumerate(parts):
        if part == "projects" and i + 1 < len(parts) and parts[i + 1]:
            project = parts[i + 1]
            # Try to stat progressively deeper paths to find a real owner.
            # Start from the first child after the project and work outward
            # so we get the most specific owner.
            for depth in range(i + 3, min(i + 6, len(parts) + 1)):
                candidate = "/".join(parts[:depth])
                if not candidate or candidate == "/":
                    continue
                try:
                    st = os.stat(candidate)
                    owner = pwd.getpwuid(st.st_uid).pw_name
                    # Skip generic/service accounts that aren't informative.
                    if owner not in ("root", "nobody", ""):
                        return owner
                except (OSError, KeyError):
                    continue
            return project

    return "unknown"


# ── path mapper ──────────────────────────────────────────────────────────────


class PathMapper:
    """Deterministically map external paths to destinations under *filter_path*.

    Destinations live under ``<filter_path>/external_datasets/`` (for dataset
    dirs) and ``<filter_path>/external_media/`` (for media sources), organised
    by owner.  Numeric suffixes are appended when names collide.
    """

    def __init__(self, filter_path: str):
        self.filter_path = os.path.normpath(filter_path)

        # source → destination  (absolute paths)
        self.dataset_map: Dict[str, str] = {}
        self.media_map: Dict[str, str] = {}

        # Track used relative names per kind to detect collisions.
        self._used: Dict[str, Set[str]] = defaultdict(set)

    # ── public ───────────────────────────────────────────────────────────

    def map_dataset(self, resolved_path: str) -> str:
        """Return the destination path for a dataset directory."""
        norm = os.path.normpath(resolved_path)
        if _is_under(norm, self.filter_path):
            return norm
        if norm in self.dataset_map:
            return self.dataset_map[norm]

        basename = os.path.basename(norm)
        unique_name = self._unique("dataset", basename)
        dest = os.path.join(self.filter_path, "external_datasets", unique_name)
        self.dataset_map[norm] = dest
        return dest

    def map_media(self, media_source_path: str) -> str:
        """Return the destination path for a media source directory."""
        ms = os.path.normpath(media_source_path)
        if not ms or ms in ("/", "."):
            return media_source_path  # unmappable
        if _is_under(ms, self.filter_path):
            return ms
        if ms in self.media_map:
            return self.media_map[ms]

        basename = os.path.basename(ms.rstrip("/")) or "root"
        unique_name = self._unique("media", basename)
        dest = os.path.join(self.filter_path, "external_media", unique_name)
        self.media_map[ms] = dest
        return dest

    # ── internal ─────────────────────────────────────────────────────────

    def _unique(self, kind: str, basename: str) -> str:
        """Return *basename*, appending ``_N`` on collision."""
        if basename not in self._used[kind]:
            self._used[kind].add(basename)
            return basename
        i = 2
        while f"{basename}_{i}" in self._used[kind]:
            i += 1
        result = f"{basename}_{i}"
        self._used[kind].add(result)
        return result


# ── YAML collator ────────────────────────────────────────────────────────────


class YamlCollator:
    """Walk a MetadatasetV2 YAML tree, rewrite paths, and emit copy scripts."""

    def __init__(self, filter_path: str, output_dir: str, *,
                 copy_links: bool = False):
        self.filter_path = os.path.normpath(filter_path)
        self.output_dir = os.path.abspath(output_dir)
        self.yaml_dir = os.path.join(self.output_dir, "yamls")
        self._rsync_flags = "-aL" if copy_links else "-a"

        self.mapper = PathMapper(filter_path)

        # realpath(src_yaml) → output yaml path
        self._yaml_map: Dict[str, str] = {}
        self._yaml_names_used: Set[str] = set()
        self._visited: Set[str] = set()

        # Deduplicated copy commands: (src, dst, owner, kind)
        self._copy_keys: Set[Tuple[str, str]] = set()
        self.copies: List[Tuple[str, str, str, str]] = []

        # Datasets skipped due to unmappable media_source (filesystem:///).
        # Each entry: (dataset_name, resolved_src, dest_path, media_source, owner)
        self.skipped: List[Tuple[str, str, str, str, str]] = []

        self.warnings: List[str] = []

    # ── public interface ─────────────────────────────────────────────────

    def process(self, root_yaml: str) -> str:
        """Process *root_yaml* recursively and return the new root YAML path."""
        return self._process_yaml(root_yaml)

    def write_copy_scripts(self, *, by_owner: bool = False) -> List[str]:
        """Write shell script(s) with rsync commands.  Return paths written."""
        os.makedirs(self.output_dir, exist_ok=True)
        written: List[str] = []

        # ── combined script ──────────────────────────────────────────────
        all_path = os.path.join(self.output_dir, "copy_all.sh")
        sorted_copies = sorted(self.copies, key=lambda c: (c[3], c[2], c[0]))
        with open(all_path, "w") as f:
            self._write_script_header(f)
            cur_kind = None
            for src, dst, owner, kind in sorted_copies:
                if kind != cur_kind:
                    f.write(f"# {'─' * 70}\n")
                    f.write(f"# {kind.upper()} COPIES\n")
                    f.write(f"# {'─' * 70}\n\n")
                    cur_kind = kind
                f.write(f"# owner: {owner}\n")
                f.write(f'mkdir -p "{dst}"\n')
                f.write(f'rsync {self._rsync_flags} "{src}/" "{dst}/"\n\n')
        written.append(all_path)

        # ── per-owner scripts ────────────────────────────────────────────
        if by_owner:
            by_own: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
            for src, dst, owner, kind in self.copies:
                by_own[owner].append((src, dst, kind))

            for owner, entries in sorted(by_own.items()):
                path = os.path.join(self.output_dir, f"copy_{owner}.sh")
                with open(path, "w") as f:
                    self._write_script_header(f, owner=owner)
                    for src, dst, kind in sorted(entries, key=lambda e: (e[2], e[0])):
                        f.write(f"# [{kind}]\n")
                        f.write(f'mkdir -p "{dst}"\n')
                        f.write(f'rsync {self._rsync_flags} "{src}/" "{dst}/"\n\n')
                written.append(path)

        # ── scripts for skipped datasets (unmappable media_source) ───────
        if self.skipped:
            written.extend(
                self._write_manual_copy_scripts(by_owner=by_owner)
            )

        return written

    def _write_manual_copy_scripts(
        self, *, by_owner: bool = False,
    ) -> List[str]:
        """Write copy scripts for datasets with unmappable media_source.

        These datasets use ``filesystem:///`` so their images are referenced
        by absolute paths embedded in the shards.  The generated scripts copy
        the dataset directories to their mapped destinations; the user must
        handle the media separately.
        """
        written: List[str] = []

        # ── combined manual script ───────────────────────────────────────
        all_path = os.path.join(self.output_dir, "manual_copy_all.sh")
        sorted_skipped = sorted(self.skipped, key=lambda s: (s[4], s[0]))
        with open(all_path, "w") as f:
            self._write_script_header(f)
            f.write(
                "# These datasets have media_source='filesystem:///' meaning images\n"
                "# are referenced by absolute path in the shards.  The dataset\n"
                "# directories are copied below; you must handle the media separately.\n\n"
            )
            for ds_name, src, dst, media_source, owner in sorted_skipped:
                f.write(f"# dataset: {ds_name}\n")
                f.write(f"# media_source: {media_source}\n")
                f.write(f"# owner: {owner}\n")
                f.write(f'mkdir -p "{dst}"\n')
                f.write(f'rsync {self._rsync_flags} "{src}/" "{dst}/"\n\n')
        written.append(all_path)

        # ── per-owner manual scripts ─────────────────────────────────────
        if by_owner:
            by_own: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
            for ds_name, src, dst, media_source, owner in self.skipped:
                by_own[owner].append((ds_name, src, dst, media_source))

            for owner, entries in sorted(by_own.items()):
                path = os.path.join(self.output_dir, f"manual_copy_{owner}.sh")
                with open(path, "w") as f:
                    self._write_script_header(f, owner=owner)
                    f.write(
                        "# These datasets have media_source='filesystem:///' meaning images\n"
                        "# are referenced by absolute path in the shards.  The dataset\n"
                        "# directories are copied below; you must handle the media separately.\n\n"
                    )
                    for ds_name, src, dst, media_source in sorted(entries):
                        f.write(f"# dataset: {ds_name}\n")
                        f.write(f"# media_source: {media_source}\n")
                        f.write(f'mkdir -p "{dst}"\n')
                        f.write(f'rsync {self._rsync_flags} "{src}/" "{dst}/"\n\n')
                written.append(path)

        return written

    def write_summary(self) -> str:
        """Write a human-readable summary.  Return the path written."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "summary.txt")

        dataset_copies = [(s, d, o) for s, d, o, k in self.copies if k == "dataset"]
        media_copies = [(s, d, o) for s, d, o, k in self.copies if k == "media"]

        with open(path, "w") as f:
            w = f.write
            sep = "=" * 90

            w(f"Collation Summary\n{sep}\n")
            w(f"Filter path : {self.filter_path}\n")
            w(f"Output dir  : {self.output_dir}\n\n")
            w(f"Dataset copies : {len(dataset_copies)}\n")
            w(f"Media copies   : {len(media_copies)}\n")
            w(f"Skipped        : {len(self.skipped)} (unmappable media_source)\n")
            w(f"YAML files     : {len(self._yaml_map)}\n\n")

            if dataset_copies:
                w(f"Dataset Copies\n{'-' * 90}\n")
                for src, dst, owner in sorted(dataset_copies):
                    w(f"  {src}\n")
                    w(f"    -> {dst}\n")
                    w(f"    owner: {owner}\n\n")

            if media_copies:
                w(f"Media Copies\n{'-' * 90}\n")
                for src, dst, owner in sorted(media_copies):
                    w(f"  {src}\n")
                    w(f"    -> {dst}\n")
                    w(f"    owner: {owner}\n\n")

            if self.skipped:
                w(f"Skipped Datasets (unmappable media_source)\n{'-' * 90}\n")
                w(f"These datasets use media_source='filesystem:///' so images are\n")
                w(f"referenced by absolute path in the shards.  The dataset directories\n")
                w(f"are copied via manual_copy_*.sh but the media must be handled\n")
                w(f"separately.\n\n")
                for ds_name, src, dst, media_source, owner in sorted(self.skipped):
                    w(f"  {ds_name}\n")
                    w(f"    src:          {src}\n")
                    w(f"    dest:         {dst}\n")
                    w(f"    media_source: {media_source}\n")
                    w(f"    owner:        {owner}\n\n")

            if self._yaml_map:
                w(f"YAML Files\n{'-' * 90}\n")
                for src_real, dst in sorted(self._yaml_map.items()):
                    w(f"  {src_real}\n    -> {dst}\n\n")

            if self.warnings:
                w(f"\nWarnings\n{'-' * 90}\n")
                for warning in self.warnings:
                    w(f"  - {warning}\n")

        return path

    # ── recursive YAML processor ─────────────────────────────────────────

    def _process_yaml(self, yaml_path: str) -> str:
        """Load, rewrite, and save a single YAML.  Recurse for nested refs."""
        real = os.path.realpath(yaml_path)

        # Already processed → return the cached output path.
        if real in self._yaml_map:
            return self._yaml_map[real]

        # Cycle guard.
        if real in self._visited:
            self.warnings.append(f"Circular YAML reference: {yaml_path}")
            return yaml_path
        self._visited.add(real)

        if not os.path.isfile(yaml_path):
            self.warnings.append(f"YAML not found: {yaml_path}")
            return yaml_path

        # Reserve the output path *before* recursing so that any nested YAML
        # that references *this* YAML can resolve it.
        dest = self._yaml_output_path(yaml_path)
        self._yaml_map[real] = dest

        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            self.warnings.append(f"Unexpected YAML structure: {yaml_path}")
            return yaml_path

        splits = data.get("splits", {})
        if isinstance(splits, dict):
            for split_cfg in splits.values():
                if isinstance(split_cfg, dict):
                    self._process_split(split_cfg, yaml_dir)

        # Write the rewritten YAML.
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            yaml.dump(
                data, f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=4096,
            )

        return dest

    def _process_split(self, split_cfg: dict, yaml_dir: str):
        """Process all entries in a single split (in-place mutation)."""
        for key in ("blend_epochized", "blend", "datasets"):
            entries = split_cfg.get(key)
            if entries and isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        self._process_entry(entry, yaml_dir)

        # Direct single-dataset path.
        if "path" in split_cfg and not any(
            k in split_cfg for k in ("blend_epochized", "blend", "datasets")
        ):
            self._process_entry(split_cfg, yaml_dir)

    def _process_entry(self, entry: dict, yaml_dir: str):
        """Rewrite one entry's ``path`` and ``media_source`` if external."""
        raw_path = entry.get("path", "")
        if not raw_path:
            return

        resolved = _resolve_path(raw_path, yaml_dir)

        # ── nested YAML ──────────────────────────────────────────────────
        if _is_yaml_path(raw_path):
            new_yaml = self._process_yaml(resolved)
            entry["path"] = new_yaml
            return

        # ── check for unmappable media_source first ──────────────────────
        # When media_source is ``filesystem:///`` the images are referenced
        # by absolute path inside the dataset shards.  We cannot relocate
        # the media, so we also skip copying the dataset itself and record
        # it in ``self.skipped`` for separate handling.
        aux = entry.get("aux")
        media_source = ""
        if isinstance(aux, dict):
            media_source = aux.get("media_source", "")

        ms_path = _strip_media_source(media_source) if media_source else ""
        if media_source and (not ms_path or ms_path in ("/", ".")):
            ds_name = entry.get("subflavors", {}).get("name", raw_path)
            owner = _extract_owner(resolved)
            # Map the dataset path to a destination even though we can't map
            # the media.  The dataset copy goes into manual_copy_*.sh.
            if not _is_under(resolved, self.filter_path):
                dest_path = self.mapper.map_dataset(resolved)
                entry["path"] = dest_path
            else:
                dest_path = resolved
            self.skipped.append((ds_name, resolved, dest_path, media_source, owner))
            self.warnings.append(
                f"Unmappable media_source '{media_source}' for dataset "
                f"'{ds_name}' (owner: {owner})"
            )
            # Inject a FIXME key so the generated YAML is visibly broken
            # until the user resolves the media_source for this dataset.
            entry["FIXME"] = (
                f"media_source '{media_source}' references absolute paths "
                f"embedded in shards — copy media manually and update "
                f"media_source (owner: {owner})"
            )
            return

        # ── leaf dataset path ────────────────────────────────────────────
        if not _is_under(resolved, self.filter_path):
            new_path = self.mapper.map_dataset(resolved)
            self._add_copy(resolved, new_path, _extract_owner(resolved), "dataset")
            entry["path"] = new_path

        # ── media source ─────────────────────────────────────────────────
        if not media_source or not isinstance(aux, dict):
            return

        if not _is_under(ms_path, self.filter_path):
            new_ms = self.mapper.map_media(ms_path)
            self._add_copy(ms_path, new_ms, _extract_owner(ms_path), "media")
            aux["media_source"] = "filesystem://" + new_ms

    # ── helpers ──────────────────────────────────────────────────────────

    def _add_copy(self, src: str, dst: str, owner: str, kind: str):
        key = (src, dst)
        if key in self._copy_keys:
            return
        self._copy_keys.add(key)
        self.copies.append((src, dst, owner, kind))

    def _yaml_output_path(self, yaml_path: str) -> str:
        """Choose a unique filename inside ``self.yaml_dir``."""
        basename = os.path.basename(yaml_path)
        name, ext = os.path.splitext(basename)

        if basename not in self._yaml_names_used:
            self._yaml_names_used.add(basename)
            return os.path.join(self.yaml_dir, basename)

        # Incorporate parent directory name to disambiguate.
        parent = os.path.basename(os.path.dirname(os.path.abspath(yaml_path)))
        candidate = f"{parent}_{name}{ext}"
        if candidate not in self._yaml_names_used:
            self._yaml_names_used.add(candidate)
            return os.path.join(self.yaml_dir, candidate)

        # Fall back to numeric suffix.
        i = 2
        while f"{name}_{i}{ext}" in self._yaml_names_used:
            i += 1
        final = f"{name}_{i}{ext}"
        self._yaml_names_used.add(final)
        return os.path.join(self.yaml_dir, final)

    def _write_script_header(self, f, *, owner: str | None = None):
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated by collate_dataset_yaml.py — review before running!\n")
        f.write(f"# Filter path: {self.filter_path}\n")
        if owner:
            f.write(f"# Owner: {owner}\n")
        f.write("\nset -euo pipefail\n\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collate external datasets from a MetadatasetV2 YAML under a "
            "single target directory.  Generates copy scripts and rewritten "
            "YAMLs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "yaml_path",
        help="Path to the root dataset YAML file.",
    )
    parser.add_argument(
        "filter_path",
        help=(
            "Target directory.  Datasets / media already under this path are "
            "left in place; everything else is mapped to destinations under it."
        ),
    )
    parser.add_argument(
        "output_dir",
        help="Directory for generated scripts, YAMLs, and summary.",
    )
    parser.add_argument(
        "--by-owner",
        action="store_true",
        help="Also generate per-owner copy scripts.",
    )
    parser.add_argument(
        "--copy-links",
        action="store_true",
        help="Dereference symlinks during copy (rsync -L).  Without this flag "
             "symlinks are preserved as-is.",
    )
    args = parser.parse_args()

    collator = YamlCollator(args.filter_path, args.output_dir,
                            copy_links=args.copy_links)
    new_yaml = collator.process(args.yaml_path)
    scripts = collator.write_copy_scripts(by_owner=args.by_owner)
    summary = collator.write_summary()

    # ── stderr recap ─────────────────────────────────────────────────────
    n_ds = sum(1 for *_, k in collator.copies if k == "dataset")
    n_ms = sum(1 for *_, k in collator.copies if k == "media")
    print(f"Dataset copies : {n_ds}", file=sys.stderr)
    print(f"Media copies   : {n_ms}", file=sys.stderr)
    print(
        f"Skipped        : {len(collator.skipped)}"
        f" (unmappable media — datasets copied via manual_copy_*.sh)",
        file=sys.stderr,
    )
    print(f"YAML files     : {len(collator._yaml_map)}", file=sys.stderr)
    print(f"New root YAML  : {new_yaml}", file=sys.stderr)
    for s in scripts:
        print(f"Copy script    : {s}", file=sys.stderr)
    print(f"Summary        : {summary}", file=sys.stderr)

    if collator.warnings:
        print(f"\n{len(collator.warnings)} warning(s):", file=sys.stderr)
        for w in collator.warnings:
            print(f"  - {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
