#!/usr/bin/env python3
"""General-purpose tool to copy MetadatasetV2 datasets to S3 and generate
rewritten YAMLs for a new cluster.

Two-step workflow:

  1. ``analyze`` — Parse input YAMLs (recursively following nested YAML
     references), resolve symlinks, generate dataset/media mapping files,
     dm-copy shell scripts, and a warnings report.

  2. ``rewrite`` — Using the mappings from step 1, rewrite all input +
     nested YAMLs so that path:/media_source: values point to paths on the
     new cluster.

Usage
-----
    # Step 1: Analyze YAMLs, generate staging dir + copy script
    python copy_to_s3.py analyze recipe.yaml [more.yaml ...] \\
        --s3-dest "team-foo:bucket/prefix" \\
        --output-dir ./my_copy/ \\
        --dm-args "--slurm-nodes 1"

    # Step 2: (user runs the staged copy)
    ./my_copy/dm_copy_staged.sh --dry-run

    # Step 3: Generate rewritten YAMLs for a new cluster
    python copy_to_s3.py rewrite recipe.yaml [more.yaml ...] \\
        --output-dir ./my_copy/ \\
        --new-root "/scratch/fsw/portfolios/.../my_project"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────


def _resolve(path: str) -> str:
    """Resolve a path, following symlinks at every level."""
    try:
        return os.path.realpath(path)
    except Exception:
        return path


def _dir_size_bytes(path: str) -> Optional[int]:
    """Return total size in bytes of *path* (file or directory tree).

    Follows symlinks so the result matches what ``dm job copy
    --follow-symlinks`` would actually transfer.  Returns ``None`` if
    the path is inaccessible.
    """
    if not os.path.exists(path):
        return None
    if os.path.isfile(path):
        try:
            return os.stat(path, follow_symlinks=True).st_size
        except OSError:
            return None
    total = 0
    try:
        for dirpath, _dirs, filenames in os.walk(path, followlinks=True):
            for fname in filenames:
                try:
                    total += os.stat(
                        os.path.join(dirpath, fname), follow_symlinks=True,
                    ).st_size
                except OSError:
                    pass
    except OSError:
        return None
    return total


def _fmt_size(nbytes: int) -> str:
    """Format *nbytes* as a human-readable string (e.g. ``4.2 GiB``)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            if unit == "B":
                return f"{nbytes} {unit}"
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"  # unreachable


# Names that are not descriptive on their own — the algorithm walks up
# past these to anchor on a meaningful parent component.
_GENERIC_NAMES: Set[str] = {
    "images", "image", "train", "val", "test", "data", "video",
    "videos", "audio", "manifests", "splits", "shards", "train2014",
    "train2017", "val2014", "val2017", "development", "22khz",
    "100k",
}


def _compute_mini_paths(paths: List[str]) -> Dict[str, str]:
    """Compute shortest unique suffix for each resolved path.

    Two-phase approach:

    1. **Meaningful anchor** — for each path, start from the basename and
       walk up past any generic components (*images*, *train*, *data*, …)
       to find a descriptive anchor.  This ensures the mini-path always
       contains a useful name, not just ``images`` or ``train``.

    2. **Uniqueness** — if any mini-paths still collide, extend all
       conflicting paths by one parent component.  Repeat until unique.

    Parameters
    ----------
    paths : list of str
        Absolute resolved paths (should be deduplicated by the caller).

    Returns
    -------
    dict
        Mapping of ``resolved_path -> mini_path``.
    """
    if not paths:
        return {}

    path_parts: Dict[str, List[str]] = {
        p: [c for c in p.split("/") if c] for p in paths
    }

    # Phase 1: initialise depth by skipping generic trailing components.
    depth: Dict[str, int] = {}
    for p, parts in path_parts.items():
        d = 1
        while d < len(parts) and parts[-d].lower() in _GENERIC_NAMES:
            d += 1
        depth[p] = d

    # Phase 2: resolve collisions by extending one component at a time.
    # Also ensure the leading component of each mini-path is not generic
    # (e.g. prefer ``grounding_data/images/coco`` over ``images/coco``).
    result: Dict[str, str] = {}
    remaining = set(paths)

    while remaining:
        suffix_map: Dict[str, List[str]] = {}
        for p in remaining:
            parts = path_parts[p]
            d = min(depth[p], len(parts))
            suffix = "/".join(parts[-d:])
            suffix_map.setdefault(suffix, []).append(p)

        newly_resolved: List[str] = []
        for suffix, group in suffix_map.items():
            if len(group) == 1:
                p = group[0]
                parts = path_parts[p]
                first_component = suffix.split("/", 1)[0]
                # If the leading component is generic and we can extend, do so.
                if first_component.lower() in _GENERIC_NAMES and depth[p] < len(parts):
                    depth[p] += 1
                else:
                    result[p] = suffix
                    newly_resolved.append(p)
            else:
                for p in group:
                    if depth[p] >= len(path_parts[p]):
                        result[p] = suffix  # exhausted — use full path
                        newly_resolved.append(p)
                    else:
                        depth[p] += 1

        for p in newly_resolved:
            remaining.discard(p)

        if not newly_resolved:  # safety valve — avoid infinite loop
            for p in remaining:
                result[p] = "/".join(path_parts[p])
            break

    return result


def _is_yaml(path: str) -> bool:
    return path.endswith(".yaml") or path.endswith(".yml")


def _extract_media_path(raw_media: str) -> Tuple[Optional[str], bool, bool]:
    """Extract the filesystem path from a media_source value.

    Returns ``(abs_path, has_prefix, is_root)``.

    * *abs_path* is ``None`` for root/empty filesystem URIs.
    * *has_prefix* is ``True`` if the value had a ``filesystem://`` prefix.
    * *is_root* is ``True`` for ``filesystem:///`` (root) entries — these
      mean the dataset metadata contains absolute paths and the media
      source is ``/``.
    """
    # Root/empty filesystem URI — all slash-stripped variants that can
    # result from an original ``filesystem:///`` after rstrip("/").
    if raw_media in ("filesystem:///", "filesystem://", "filesystem:/", "filesystem:"):
        return None, True, True
    if raw_media.startswith("filesystem://"):
        path = raw_media[len("filesystem://"):]
        path = path.rstrip("/")
        if not path or path == "/":
            return None, True, True
        return path, True, False
    # Bare path (no filesystem:// prefix, e.g. granary-style).
    return raw_media.rstrip("/"), False, False


# ── data structures ──────────────────────────────────────────────────────────


@dataclass
class Warning:
    kind: str          # "missing_dataset", "missing_media", "root_media_source"
    path: str
    source_yaml: str
    media_source: str = ""


@dataclass
class AnalysisResult:
    dataset_dirs: Dict[str, str]   # resolved_path -> mini_path
    media_dirs: Dict[str, str]     # resolved_path -> mini_path
    yaml_files: List[str]          # all discovered YAML files (absolute)
    yaml_refs: Dict[str, str]      # resolved_yaml_path -> output basename
    warnings: List[Warning] = field(default_factory=list)


# ── YAML value extraction ────────────────────────────────────────────────────


def _extract_values(
    yaml_path: str,
    splits: Optional[Set[str]] = None,
) -> List[Tuple[str, str]]:
    """Extract uncommented ``path:`` and ``media_source:`` values.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file.
    splits:
        If provided, only return values from these splits (e.g.
        ``{"train"}``).  ``None`` means all splits.

    Returns a list of ``(type, raw_value)`` where *type* is ``"path"`` or
    ``"media"``.  Path values have trailing slashes stripped; media values
    are left as-is (to preserve ``filesystem:///``).
    """
    with open(yaml_path) as f:
        lines = f.readlines()

    results: List[Tuple[str, str]] = []
    in_splits = False
    current_split: Optional[str] = None

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        # Track split context for filtering.
        # "splits:" at the top level (no leading whitespace or minimal)
        if re.match(r"^splits:\s*$", line):
            in_splits = True
            current_split = None
            continue

        if in_splits:
            # Split name: key at 2-space indent (e.g. "  train:")
            m_split = re.match(r"^  (\w+):\s*$", line)
            if m_split:
                current_split = m_split.group(1)
                continue
            # Left the splits block (non-indented, non-empty line)
            if line and line[0] != " " and line.strip():
                in_splits = False
                current_split = None

        # If split filtering is active, skip entries not in desired splits
        if splits is not None and current_split is not None and current_split not in splits:
            continue

        # path: lines (both "- path:" list items and "path:" mappings)
        m = re.match(r"^(\s+(?:-\s+)?path:\s+)(.+)$", line)
        if m:
            raw = m.group(2).strip().rstrip("/")
            results.append(("path", raw))
            continue

        # media_source: lines — do NOT rstrip("/") to preserve filesystem:///
        m = re.match(r"^(\s+media_source:\s+)(.+)$", line)
        if m:
            raw = m.group(2).strip()
            results.append(("media", raw))
            continue

    return results


# ── YAML discovery (recursive) ───────────────────────────────────────────────


def _discover_yamls(
    yaml_paths: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Discover all YAML files — input plus recursively nested references.

    Returns:
        all_yamls: list of absolute paths (input first, then nested).
        yaml_refs: map of ``resolved_path -> output_basename`` for every YAML.
    """
    visited: Set[str] = set()
    all_yamls: List[str] = []
    yaml_refs: Dict[str, str] = {}  # resolved -> output basename
    basenames_used: Dict[str, int] = {}  # collision counter

    def _assign_basename(real_path: str, preferred: str) -> str:
        """Assign a unique output basename, adding a suffix on collision."""
        if preferred in basenames_used:
            basenames_used[preferred] += 1
            name, ext = os.path.splitext(preferred)
            out = f"{name}_{basenames_used[preferred]}{ext}"
        else:
            basenames_used[preferred] = 1
            out = preferred
        yaml_refs[real_path] = out
        return out

    def _visit(yaml_path: str):
        real = _resolve(yaml_path)
        if real in visited:
            return
        visited.add(real)

        abs_path = os.path.abspath(yaml_path)
        all_yamls.append(abs_path)

        if not os.path.isfile(yaml_path):
            print(f"WARNING: YAML file not found: {yaml_path}", file=sys.stderr)
            return

        yaml_dir = os.path.dirname(abs_path)
        values = _extract_values(yaml_path)

        for typ, raw in values:
            if typ == "path" and _is_yaml(raw):
                if os.path.isabs(raw):
                    nested = raw
                else:
                    nested = os.path.normpath(os.path.join(yaml_dir, raw))
                nested_real = _resolve(nested)
                if nested_real not in visited:
                    _assign_basename(nested_real, os.path.basename(nested))
                    _visit(nested)

    for yp in yaml_paths:
        real = _resolve(yp)
        if real not in visited:
            _assign_basename(real, os.path.basename(os.path.abspath(yp)))
            _visit(yp)

    return all_yamls, yaml_refs


# ── core analysis ────────────────────────────────────────────────────────────


def analyze_yamls(
    yaml_paths: List[str],
    splits: Optional[Set[str]] = None,
) -> AnalysisResult:
    """Analyze input YAMLs recursively, building dataset and media mappings.

    Mini-paths are computed in batch using shortest-unique-suffix so that
    every mini-path is as short as possible while remaining unique.

    Parameters
    ----------
    splits:
        If provided, only include entries from these splits (e.g.
        ``{"train", "val"}``).  ``None`` means all splits.
    """
    all_yamls, yaml_refs = _discover_yamls(yaml_paths)

    # ── Pass 1: collect resolved paths and warnings ──────────────────
    dataset_resolved: List[str] = []
    media_resolved: List[str] = []
    dataset_seen: Set[str] = set()
    media_seen: Set[str] = set()
    warnings: List[Warning] = []

    for yaml_path in all_yamls:
        if not os.path.isfile(yaml_path):
            continue

        yaml_dir = os.path.dirname(yaml_path)
        values = _extract_values(yaml_path, splits=splits)

        for typ, raw in values:
            if typ == "path":
                if _is_yaml(raw):
                    continue  # nested YAML — handled by discovery

                # Resolve
                if os.path.isabs(raw):
                    resolved = _resolve(raw)
                else:
                    resolved = _resolve(os.path.normpath(os.path.join(yaml_dir, raw)))

                # Copy unit: .jsonl → parent dir, otherwise dir itself
                if raw.endswith(".jsonl"):
                    copy_dir = os.path.dirname(resolved)
                else:
                    copy_dir = resolved

                if copy_dir not in dataset_seen:
                    dataset_seen.add(copy_dir)
                    dataset_resolved.append(copy_dir)

                if not os.path.exists(copy_dir):
                    warnings.append(Warning(
                        kind="missing_dataset",
                        path=copy_dir,
                        source_yaml=yaml_path,
                    ))

            elif typ == "media":
                abs_path, has_prefix, is_root = _extract_media_path(raw)

                if is_root:
                    warnings.append(Warning(
                        kind="root_media_source",
                        path="",
                        source_yaml=yaml_path,
                        media_source=raw,
                    ))
                    continue

                if abs_path is None:
                    continue

                resolved = _resolve(abs_path)
                if resolved not in media_seen:
                    media_seen.add(resolved)
                    media_resolved.append(resolved)

                if not os.path.exists(resolved):
                    warnings.append(Warning(
                        kind="missing_media",
                        path=resolved,
                        source_yaml=yaml_path,
                    ))

    # ── Pass 2: batch-compute mini-paths ─────────────────────────────
    ds_minis = _compute_mini_paths(dataset_resolved)
    ms_minis = _compute_mini_paths(media_resolved)

    dataset_dirs = {p: ds_minis[p] for p in dataset_resolved}
    media_dirs = {p: ms_minis[p] for p in media_resolved}

    return AnalysisResult(
        dataset_dirs=dataset_dirs,
        media_dirs=media_dirs,
        yaml_files=all_yamls,
        yaml_refs=yaml_refs,
        warnings=warnings,
    )


# ── output: mapping files ───────────────────────────────────────────────────


def write_mappings(
    result: AnalysisResult,
    output_dir: str,
    s3_dest: str,
    dataset_suffix: str,
    media_suffix: str,
):
    """Write ``datasets_mapping.txt`` and ``media_mapping.txt``."""
    os.makedirs(output_dir, exist_ok=True)

    ds_file = os.path.join(output_dir, "datasets_mapping.txt")
    with open(ds_file, "w") as f:
        f.write("# MINI_PATH | REAL_PATH (symlink-resolved)\n")
        f.write(f"# Copy to: {s3_dest}{dataset_suffix}/<MINI_PATH>\n")
        for real in sorted(result.dataset_dirs):
            f.write(f"{result.dataset_dirs[real]} | {real}\n")

    ms_file = os.path.join(output_dir, "media_mapping.txt")
    with open(ms_file, "w") as f:
        f.write("# MINI_PATH | REAL_PATH (symlink-resolved)\n")
        f.write(f"# Copy to: {s3_dest}{media_suffix}/<MINI_PATH>\n")
        for real in sorted(result.media_dirs):
            f.write(f"{result.media_dirs[real]} | {real}\n")

    print(f"Wrote {len(result.dataset_dirs)} dataset mappings to {ds_file}", file=sys.stderr)
    print(f"Wrote {len(result.media_dirs)} media mappings to {ms_file}", file=sys.stderr)


# ── output: staging directory + copy script ─────────────────────────────────


def _create_staging_links(
    dirs: Dict[str, str],
    staging_root: str,
    suffix: str,
    label: str,
) -> int:
    """Create symlinks under *staging_root*/*suffix*/ for each entry in *dirs*.

    Handles nested mini-paths (where one mini-path is a prefix of another).
    If the child's real path is under the parent's real path, the child is
    skipped (it's already reachable through the parent symlink).  If they
    point to unrelated locations, the parent symlink is replaced with a real
    directory and children are symlinked individually.

    Returns the number of symlinks created.
    """
    base = os.path.join(staging_root, suffix.lstrip("/"))
    created = 0
    skipped = 0

    # Sort by mini-path so parents come before children.
    sorted_entries = sorted(dirs.items(), key=lambda kv: kv[1])

    # Track active parent symlinks: list of (mini_path, resolved_path).
    active_parents: List[Tuple[str, str]] = []

    for resolved, mini in sorted_entries:
        mini = mini.rstrip("/")
        link_path = os.path.join(base, mini)

        # Check if any active parent mini-path is a prefix of this one.
        ancestor = None
        for parent_mini, parent_resolved in active_parents:
            if mini.startswith(parent_mini + "/"):
                ancestor = (parent_mini, parent_resolved)
                break

        if ancestor is not None:
            parent_mini, parent_resolved = ancestor
            # Is the child's real path actually under the parent's real path?
            if resolved.startswith(parent_resolved.rstrip("/") + "/"):
                # Redundant — already reachable via parent symlink.
                skipped += 1
                continue
            else:
                # Independent paths with nested mini-paths.
                # Replace the parent symlink with a real directory.
                parent_link = os.path.join(base, parent_mini)
                if os.path.islink(parent_link):
                    old_target = os.readlink(parent_link)
                    os.unlink(parent_link)
                    os.makedirs(parent_link, exist_ok=True)
                    # Re-create the parent's content as individual symlinks
                    # for each item in the original target directory.
                    if os.path.isdir(old_target):
                        for entry in os.listdir(old_target):
                            child_link = os.path.join(parent_link, entry)
                            child_target = os.path.join(old_target, entry)
                            if not os.path.exists(child_link):
                                os.symlink(child_target, child_link)
                    # Remove this from active parents since it's now a real dir.
                    active_parents = [
                        (m, r) for m, r in active_parents
                        if m != parent_mini
                    ]

        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(resolved, link_path)
        created += 1
        active_parents.append((mini, resolved))

    if skipped:
        print(
            f"    {label}: {created} symlinks created, "
            f"{skipped} nested entries skipped (reachable via parent)",
            file=sys.stderr,
        )
    else:
        print(f"    {label}: {created} symlinks created", file=sys.stderr)

    return created


def build_staging_dir(
    result: AnalysisResult,
    output_dir: str,
    dataset_suffix: str,
    media_suffix: str,
) -> str:
    """Build a staging directory with symlinks mirroring the S3 layout.

    The staging tree is the exact local replica of the S3 destination
    structure.  Each leaf is a symlink pointing to the real data on the
    cluster.  A single ``dm job copy --follow-symlinks`` of this tree
    copies everything to S3.

    Returns the path to the staging root directory.
    """
    staging_root = os.path.join(output_dir, "staging")

    # Clean up any previous staging dir for idempotency.
    if os.path.exists(staging_root):
        shutil.rmtree(staging_root)

    os.makedirs(staging_root, exist_ok=True)

    print(f"  Building staging directory: {staging_root}/", file=sys.stderr)
    _create_staging_links(
        result.dataset_dirs, staging_root, dataset_suffix, "datasets",
    )
    _create_staging_links(
        result.media_dirs, staging_root, media_suffix, "media",
    )

    return staging_root


_STAGED_COPY_SCRIPT_TEMPLATE = r'''#!/bin/bash
#
# DM Copy - Staged copy to S3
# Copies the entire staging directory (which contains symlinks to real data)
# to the S3 destination using --follow-symlinks.
#
# Usage: ./dm_copy_staged.sh [--dry-run]
#

STAGING_DIR="$(dirname "$0")/staging"
S3_DEST="{s3_dest}"
DM_ARGS="{dm_args}"

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "DM Copy - Staged Copy to S3"
echo "=============================================="
echo "Staging dir: $STAGING_DIR"
echo "S3 dest:     $S3_DEST"
echo "DM args:     $DM_ARGS"
echo "Dry run:     $DRY_RUN"
echo ""

if [ ! -d "$STAGING_DIR" ]; then
    echo "ERROR: Staging directory not found: $STAGING_DIR"
    exit 1
fi

cmd="dm job copy -y $DM_ARGS --follow-symlinks --slurm-srun-extra-args \"--container-mounts=\\\"/lustre:/lustre\\\"\" \"$STAGING_DIR\" \"$S3_DEST\""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] $cmd"
    echo ""
    echo "This was a dry run. No jobs were actually submitted."
    echo "Remove --dry-run to execute the command."
else
    echo "[SUBMIT] $cmd"
    eval "$cmd"
fi
'''


def write_staged_copy_script(
    output_dir: str,
    s3_dest: str,
    dm_args: str,
) -> str:
    """Generate ``dm_copy_staged.sh`` for copying the staging tree to S3."""
    os.makedirs(output_dir, exist_ok=True)

    script_path = os.path.join(output_dir, "dm_copy_staged.sh")
    with open(script_path, "w") as f:
        f.write(_STAGED_COPY_SCRIPT_TEMPLATE.format(
            s3_dest=s3_dest,
            dm_args=dm_args,
        ))
    os.chmod(
        script_path,
        os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP,
    )

    print(f"  Wrote copy script: {script_path}", file=sys.stderr)
    return script_path


# ── output: parse helpers for review-mappings ────────────────────────────────


def _parse_mapping_header(filepath: str) -> Tuple[str, str]:
    """Parse the ``# Copy to:`` header → ``(s3_dest_with_suffix,)``.

    The header format is: ``# Copy to: <s3_dest><suffix>/<MINI_PATH>``
    Returns ``(s3_dest, suffix)`` where *suffix* includes the leading ``/``
    (e.g. ``/datasets``).
    """
    with open(filepath) as f:
        for line in f:
            if line.startswith("# Copy to: "):
                # "# Copy to: team:bucket/datasets/<MINI_PATH>"
                value = line.strip().removeprefix("# Copy to: ")
                # Strip trailing "/<MINI_PATH>"
                value = value.removesuffix("/<MINI_PATH>")
                # value is now "team:bucket/datasets"
                # The suffix is the last path-like segment starting with /
                # s3_dest can contain : and / (e.g. "team:bucket/prefix")
                # suffix is always /word (e.g. /datasets, /media_sources)
                # Split from the right on the last /
                last_slash = value.rfind("/")
                if last_slash >= 0:
                    s3_dest = value[:last_slash]
                    suffix = value[last_slash:]  # e.g. "/datasets"
                    return s3_dest, suffix
                return value, ""
    raise ValueError(f"No '# Copy to:' header found in {filepath}")


def _parse_dm_args_from_script(output_dir: str) -> str:
    """Extract ``DM_ARGS`` value from existing ``dm_copy_staged.sh``.

    Returns the value string, or ``""`` if the script doesn't exist.
    """
    script = os.path.join(output_dir, "dm_copy_staged.sh")
    if not os.path.isfile(script):
        return ""
    with open(script) as f:
        for line in f:
            # Match: DM_ARGS="..."
            if line.startswith("DM_ARGS="):
                # Strip DM_ARGS=" and trailing "
                val = line.strip().removeprefix("DM_ARGS=")
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                return val
    return ""


# ── review-mappings ──────────────────────────────────────────────────────────


def run_review_mappings(args) -> int:
    """Re-review mini-paths on existing mapping files, then regenerate outputs."""
    output_dir = os.path.abspath(args.output_dir)

    ds_file = os.path.join(output_dir, "datasets_mapping.txt")
    ms_file = os.path.join(output_dir, "media_mapping.txt")

    if not os.path.isfile(ds_file):
        print(f"ERROR: {ds_file} not found. Run 'interactive' or 'analyze' first.", file=sys.stderr)
        return 1
    if not os.path.isfile(ms_file):
        print(f"ERROR: {ms_file} not found. Run 'interactive' or 'analyze' first.", file=sys.stderr)
        return 1

    # Load existing mappings.
    dataset_dirs = _load_mapping(ds_file)
    media_dirs = _load_mapping(ms_file)

    # Extract s3_dest and suffixes from the header comments.
    s3_dest_ds, dataset_suffix = _parse_mapping_header(ds_file)
    s3_dest_ms, media_suffix = _parse_mapping_header(ms_file)
    s3_dest = s3_dest_ds  # Both should agree.
    if s3_dest_ds != s3_dest_ms:
        print(
            f"WARNING: S3 dest mismatch between mapping files: "
            f"{s3_dest_ds!r} vs {s3_dest_ms!r}. Using datasets value.",
            file=sys.stderr,
        )

    # Extract dm_args from existing copy script (or override).
    dm_args = args.dm_args if args.dm_args is not None else _parse_dm_args_from_script(output_dir)

    # Decide which namespaces to review.
    review_datasets = args.datasets or (not args.datasets and not args.media)
    review_media = args.media or (not args.datasets and not args.media)

    llm_only = args.llm_only
    editor_only = args.editor_only

    print(f"  Datasets: {len(dataset_dirs)} entries", file=sys.stderr)
    print(f"  Media:    {len(media_dirs)} entries", file=sys.stderr)
    print(f"  S3 dest:  {s3_dest}", file=sys.stderr)
    print(f"  DM args:  {dm_args!r}", file=sys.stderr)
    print(file=sys.stderr)

    ds_changed = False
    ms_changed = False

    for do_review, dirs, label in [
        (review_datasets, dataset_dirs, "dataset"),
        (review_media, media_dirs, "media"),
    ]:
        if not do_review:
            continue

        if editor_only:
            changed = _edit_mini_paths(dirs, label)
        elif llm_only:
            changed = _llm_review_mini_paths(dirs, label)
        else:
            _llm_review_mini_paths(dirs, label)
            changed = _edit_mini_paths(dirs, label)

        if label == "dataset":
            ds_changed = changed
        else:
            ms_changed = changed

    if not ds_changed and not ms_changed:
        print("No changes made.", file=sys.stderr)
        return 0

    # Rebuild outputs.
    result = AnalysisResult(
        dataset_dirs=dataset_dirs,
        media_dirs=media_dirs,
        yaml_files=[],
        yaml_refs={},
    )

    write_mappings(result, output_dir, s3_dest, dataset_suffix, media_suffix)
    build_staging_dir(result, output_dir, dataset_suffix, media_suffix)
    write_staged_copy_script(output_dir, s3_dest, dm_args)

    print(file=sys.stderr)
    if ds_changed:
        print(f"  Updated dataset mini-paths ({len(dataset_dirs)} entries)", file=sys.stderr)
    if ms_changed:
        print(f"  Updated media mini-paths ({len(media_dirs)} entries)", file=sys.stderr)
    print(f"  Regenerated staging/ and dm_copy_staged.sh", file=sys.stderr)
    return 0


# ── size ─────────────────────────────────────────────────────────────────────


def _print_size_section(
    label: str,
    dirs: Dict[str, str],
) -> Tuple[int, int]:
    """Print per-entry sizes for one namespace.  Returns (total_bytes, inaccessible_count)."""
    entries: List[Tuple[int, str, str]] = []  # (size, mini, resolved)
    inaccessible = 0

    total = len(dirs)
    for i, (resolved, mini) in enumerate(sorted(dirs.items(), key=lambda kv: kv[1]), 1):
        print(f"\r  Scanning {label} {i}/{total}...", end="", file=sys.stderr, flush=True)
        size = _dir_size_bytes(resolved)
        if size is None:
            inaccessible += 1
        else:
            entries.append((size, mini, resolved))
    print(f"\r{' ' * 60}\r", end="", file=sys.stderr, flush=True)

    # Sort by size descending.
    entries.sort(key=lambda e: e[0], reverse=True)

    total_bytes = sum(e[0] for e in entries)

    print(f"{label} ({len(dirs)} entries):", file=sys.stderr)
    for size, mini, resolved in entries:
        # Shorten resolved path for display.
        short = resolved
        if len(short) > 60:
            short = "..." + short[-57:]
        print(
            f"  {_fmt_size(size):>10s}  {mini:<50s}  ({short})",
            file=sys.stderr,
        )
    if inaccessible:
        print(
            f"  {'???':>10s}  ({inaccessible} inaccessible path(s))",
            file=sys.stderr,
        )
    print(f"  {'──────────':>10s}", file=sys.stderr)
    note = f"{len(dirs)} entries"
    if inaccessible:
        note += f", {inaccessible} inaccessible"
    print(f"  {_fmt_size(total_bytes):>10s}  total ({note})", file=sys.stderr)
    print(file=sys.stderr)
    return total_bytes, inaccessible


def run_size(args) -> int:
    """Report disk sizes of all datasets and media in the input YAMLs."""
    splits = set(args.splits) if args.splits else None
    result = analyze_yamls(args.yamls, splits=splits)

    show_datasets = args.datasets or (not args.datasets and not args.media)
    show_media = args.media or (not args.datasets and not args.media)

    print(file=sys.stderr)
    ds_bytes = ds_inacc = ms_bytes = ms_inacc = 0
    if show_datasets:
        ds_bytes, ds_inacc = _print_size_section("Datasets", result.dataset_dirs)
    if show_media:
        ms_bytes, ms_inacc = _print_size_section("Media sources", result.media_dirs)

    grand = ds_bytes + ms_bytes
    if show_datasets and show_media:
        print(f"Grand total: {_fmt_size(grand)}", file=sys.stderr)
    if ds_inacc + ms_inacc:
        print(
            f"  ({ds_inacc + ms_inacc} inaccessible path(s) not included)",
            file=sys.stderr,
        )
    return 0


# ── output: warnings ────────────────────────────────────────────────────────


def write_warnings(result: AnalysisResult, output_dir: str):
    """Write ``warnings.txt`` with missing paths and root-media entries."""
    os.makedirs(output_dir, exist_ok=True)

    missing_ds = [w for w in result.warnings if w.kind == "missing_dataset"]
    missing_ms = [w for w in result.warnings if w.kind == "missing_media"]
    root_ms = [w for w in result.warnings if w.kind == "root_media_source"]

    warn_file = os.path.join(output_dir, "warnings.txt")
    with open(warn_file, "w") as f:
        f.write("# Warnings from copy_to_s3.py analyze\n")
        f.write(f"# Total warnings: {len(result.warnings)}\n\n")

        # Section 1: Missing paths
        f.write(f"SECTION 1: Missing paths ({len(missing_ds) + len(missing_ms)})\n")
        f.write(f"{'-' * 80}\n\n")

        for w in missing_ds:
            f.write(f"MISSING DATASET: {w.path}\n")
            f.write(f"  source_yaml: {w.source_yaml}\n\n")

        for w in missing_ms:
            f.write(f"MISSING MEDIA: {w.path}\n")
            f.write(f"  source_yaml: {w.source_yaml}\n\n")

        # Section 2: Root filesystem:/// entries
        f.write(f"\nSECTION 2: Root filesystem:/// media sources ({len(root_ms)})\n")
        f.write(f"{'-' * 80}\n\n")

        # Deduplicate by source_yaml for readability
        seen_yamls: set = set()
        for w in root_ms:
            key = w.source_yaml
            if key in seen_yamls:
                continue
            seen_yamls.add(key)
            count = sum(1 for w2 in root_ms if w2.source_yaml == key)
            f.write(f"ROOT_MEDIA_SOURCE: ({count} entries)\n")
            f.write(f"  media_source: filesystem:///\n")
            f.write(f"  source_yaml: {w.source_yaml}\n")
            f.write(
                f"  NOTE: Data references absolute paths in metadata. On the new cluster,\n"
                f"        the path will be / + <absolute_path_in_metadata> which won't\n"
                f"        resolve unless the media exists at the same absolute path.\n\n"
            )

    print(
        f"Warnings: {len(missing_ds)} missing datasets, "
        f"{len(missing_ms)} missing media, "
        f"{len(root_ms)} root filesystem:/// entries",
        file=sys.stderr,
    )
    print(f"Wrote warnings to {warn_file}", file=sys.stderr)


# ── YAML rewriter ────────────────────────────────────────────────────────────


def _load_mapping(filepath: str) -> Dict[str, str]:
    """Load a ``MINI_PATH | REAL_PATH`` mapping file into ``{real: mini}``."""
    mapping: Dict[str, str] = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" | ")
            if len(parts) == 2:
                mini = parts[0].strip().rstrip("/")
                real = parts[1].strip()
                mapping[real] = mini
    return mapping


def rewrite_yamls(
    yaml_paths: List[str],
    output_dir: str,
    new_root: str,
    dataset_suffix: str,
    media_suffix: str,
    strip_prefix: str,
    fix_media_overrides_file: Optional[str] = None,
) -> int:
    """Rewrite all input + nested YAMLs with new-root paths.

    Returns 0 on success, 1 if any YAML had unknown entries.
    """
    all_yamls, yaml_refs = _discover_yamls(yaml_paths)

    ds_mapping_file = os.path.join(output_dir, "datasets_mapping.txt")
    ms_mapping_file = os.path.join(output_dir, "media_mapping.txt")

    if not os.path.isfile(ds_mapping_file):
        print(f"ERROR: {ds_mapping_file} not found. Run 'analyze' first.", file=sys.stderr)
        return 1
    if not os.path.isfile(ms_mapping_file):
        print(f"ERROR: {ms_mapping_file} not found. Run 'analyze' first.", file=sys.stderr)
        return 1

    dataset_mapping = _load_mapping(ds_mapping_file)
    media_mapping = _load_mapping(ms_mapping_file)

    # Load fix-media overrides if provided
    fix_media_overrides: Optional[Dict[str, Tuple[str, str]]] = None
    if fix_media_overrides_file:
        fix_media_overrides = _load_fix_media_overrides(fix_media_overrides_file)
        print(f"Loaded {len(fix_media_overrides)} fix-media overrides", file=sys.stderr)

        # Also load the fix-media mapping to include those media dirs
        fix_mapping_file = os.path.join(output_dir, "fix_media_mapping.txt")
        if os.path.isfile(fix_mapping_file):
            fix_media_mapping = _load_mapping(fix_mapping_file)
            media_mapping.update(fix_media_mapping)
            print(f"Merged {len(fix_media_mapping)} fix-media media mappings", file=sys.stderr)

        # Add the fixed dataset dirs to the dataset mapping.
        # Use "fixed_datasets/<name>" as the mini path since the fixed
        # datasets are in a temp dir that doesn't follow the normal path
        # convention.
        for orig, (new_path, _) in fix_media_overrides.items():
            resolved_new = _resolve(new_path)
            ds_name = os.path.basename(resolved_new)
            mini = f"fixed_datasets/{ds_name}"
            dataset_mapping[resolved_new] = mini

    print(f"Loaded {len(dataset_mapping)} dataset mappings", file=sys.stderr)
    print(f"Loaded {len(media_mapping)} media mappings", file=sys.stderr)

    yamls_dir = os.path.join(output_dir, "yamls")
    os.makedirs(yamls_dir, exist_ok=True)

    any_errors = False

    for yaml_path in all_yamls:
        real = _resolve(yaml_path)
        out_basename = yaml_refs.get(real, os.path.basename(yaml_path))
        out_path = os.path.join(yamls_dir, out_basename)

        if not os.path.isfile(yaml_path):
            print(f"WARNING: skipping missing YAML: {yaml_path}", file=sys.stderr)
            continue

        errors = _rewrite_single_yaml(
            yaml_path, out_path, out_basename,
            dataset_mapping, media_mapping, yaml_refs,
            new_root, dataset_suffix, media_suffix,
            fix_media_overrides=fix_media_overrides,
        )
        any_errors |= errors

    if any_errors:
        print("\n*** SOME FILES HAD UNHANDLED ENTRIES — REVIEW ABOVE ***", file=sys.stderr)
    else:
        print("\nALL FILES GENERATED SUCCESSFULLY — NO UNHANDLED ENTRIES", file=sys.stderr)

    return 1 if any_errors else 0


def _rewrite_single_yaml(
    input_path: str,
    output_path: str,
    label: str,
    dataset_mapping: Dict[str, str],
    media_mapping: Dict[str, str],
    yaml_refs: Dict[str, str],
    new_root: str,
    dataset_suffix: str,
    media_suffix: str,
    fix_media_overrides: Optional[Dict[str, Tuple[str, str]]] = None,
) -> bool:
    """Rewrite a single YAML file.  Returns ``True`` if there were unknowns."""
    with open(input_path) as f:
        lines = f.readlines()

    yaml_dir = os.path.dirname(os.path.abspath(input_path))
    new_dataset_root = f"{new_root}{dataset_suffix}"
    new_media_root = f"{new_root}{media_suffix}"

    stats: Dict[str, object] = {
        "path_yaml": 0, "path_dir": 0, "path_jsonl": 0,
        "path_unknown": 0, "path_unknown_list": [],
        "media_root": 0, "media_prefixed": 0, "media_bare": 0,
        "media_unknown": 0, "media_unknown_list": [],
        "media_fixed": 0,
    }

    # Build a reverse lookup: resolved original path -> (new_path, new_media)
    override_lookup: Dict[str, Tuple[str, str]] = {}
    if fix_media_overrides:
        for orig, (new_path, new_media) in fix_media_overrides.items():
            override_lookup[_resolve(orig)] = (new_path, new_media)

    output_lines: List[str] = []
    # Track the last path: line's resolved value so we can match the
    # subsequent media_source: line to an override.
    last_resolved_path: Optional[str] = None

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith("#"):
            output_lines.append(line)
            continue

        # path: lines
        m = re.match(r"^(\s+(?:-\s+)?path:\s+)(.+)$", line)
        if m:
            prefix = m.group(1)
            path_val = m.group(2).strip()

            # Check if this path matches a fix-media override
            raw = path_val.rstrip("/")
            if os.path.isabs(raw):
                resolved = _resolve(raw)
            else:
                resolved = _resolve(os.path.normpath(os.path.join(yaml_dir, raw)))
            # For .jsonl paths, the override key is the parent dir
            if raw.endswith(".jsonl"):
                override_key = os.path.dirname(resolved)
            else:
                override_key = resolved

            if override_key in override_lookup:
                last_resolved_path = override_key
                new_ds_path, _ = override_lookup[override_key]
                # Transform the NEW dataset path through the dataset mapping
                new_val = _transform_path(
                    new_ds_path, yaml_dir, dataset_mapping, yaml_refs,
                    new_dataset_root, stats,
                )
            else:
                last_resolved_path = None
                new_val = _transform_path(
                    path_val, yaml_dir, dataset_mapping, yaml_refs,
                    new_dataset_root, stats,
                )
            output_lines.append(f"{prefix}{new_val}\n")
            continue

        # media_source: lines
        m = re.match(r"^(\s+media_source:\s+)(.+)$", line)
        if m:
            prefix = m.group(1)
            ms_val = m.group(2).strip()

            # If the preceding path: was overridden, use the override's media_source
            if last_resolved_path is not None and last_resolved_path in override_lookup:
                _, new_media = override_lookup[last_resolved_path]
                # Transform the new media source through the media mapping
                new_val = _transform_media(
                    new_media, media_mapping, new_media_root, stats,
                )
                stats["media_fixed"] += 1
                # Don't count this as "root" since we fixed it
                last_resolved_path = None
            else:
                new_val = _transform_media(
                    ms_val, media_mapping, new_media_root, stats,
                )
            output_lines.append(f"{prefix}{new_val}\n")
            continue

        output_lines.append(line)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(output_lines)

    # Print summary
    total_p = stats["path_yaml"] + stats["path_dir"] + stats["path_jsonl"] + stats["path_unknown"]
    total_m = stats["media_root"] + stats["media_prefixed"] + stats["media_bare"] + stats["media_unknown"] + stats["media_fixed"]

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {label}", file=sys.stderr)
    print(f"  {input_path}", file=sys.stderr)
    print(f"  -> {output_path}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Paths ({total_p} total):", file=sys.stderr)
    for key, lbl in [("path_yaml", "YAML refs"), ("path_dir", "Directory"),
                      ("path_jsonl", "JSONL")]:
        if stats[key]:
            print(f"    {lbl:18s} {stats[key]}", file=sys.stderr)
    if stats["path_unknown"]:
        print(f"    {'*** UNKNOWN':18s} {stats['path_unknown']}", file=sys.stderr)
        for p in stats["path_unknown_list"]:
            print(f"      {p}", file=sys.stderr)

    print(f"  Media ({total_m} total):", file=sys.stderr)
    for key, lbl in [("media_fixed", "Fixed (override)"), ("media_root", "Root (as-is)"),
                      ("media_prefixed", "Prefixed"), ("media_bare", "Bare")]:
        if stats[key]:
            print(f"    {lbl:18s} {stats[key]}", file=sys.stderr)
    if stats["media_unknown"]:
        print(f"    {'*** UNKNOWN':18s} {stats['media_unknown']}", file=sys.stderr)
        for ms in stats["media_unknown_list"]:
            print(f"      {ms}", file=sys.stderr)

    has_errors = stats["path_unknown"] > 0 or stats["media_unknown"] > 0
    if has_errors:
        print("  *** HAS UNHANDLED ENTRIES ***", file=sys.stderr)
    else:
        print("  All entries handled.", file=sys.stderr)

    return has_errors


def _transform_path(
    path_val: str,
    yaml_dir: str,
    dataset_mapping: Dict[str, str],
    yaml_refs: Dict[str, str],
    new_root: str,
    stats: dict,
) -> str:
    """Transform a ``path:`` value for the new cluster."""
    raw = path_val.rstrip("/")

    # Resolve
    if os.path.isabs(raw):
        resolved = _resolve(raw)
    else:
        resolved = _resolve(os.path.normpath(os.path.join(yaml_dir, raw)))

    # Nested YAML reference → output basename
    if _is_yaml(raw):
        if resolved in yaml_refs:
            stats["path_yaml"] += 1
            return yaml_refs[resolved]
        stats["path_unknown"] += 1
        stats["path_unknown_list"].append(raw)
        return raw

    # .jsonl → look up parent dir
    if raw.endswith(".jsonl"):
        parent = os.path.dirname(resolved)
        filename = os.path.basename(resolved)
        if parent in dataset_mapping:
            mini = dataset_mapping[parent]
            stats["path_jsonl"] += 1
            return f"{new_root}/{mini}/{filename}"
        stats["path_unknown"] += 1
        stats["path_unknown_list"].append(raw)
        return raw

    # Directory → look up directly
    if resolved in dataset_mapping:
        mini = dataset_mapping[resolved]
        stats["path_dir"] += 1
        return f"{new_root}/{mini}"

    stats["path_unknown"] += 1
    stats["path_unknown_list"].append(raw)
    return raw


def _transform_media(
    ms_val: str,
    media_mapping: Dict[str, str],
    new_media_root: str,
    stats: dict,
) -> str:
    """Transform a ``media_source:`` value for the new cluster."""
    raw = ms_val.strip()

    abs_path, has_prefix, is_root = _extract_media_path(raw)

    if is_root:
        stats["media_root"] += 1
        return raw  # pass through unchanged

    if abs_path is None:
        return raw

    resolved = _resolve(abs_path)

    if has_prefix:
        if resolved in media_mapping:
            mini = media_mapping[resolved]
            stats["media_prefixed"] += 1
            return f"filesystem://{new_media_root}/{mini}"
        stats["media_unknown"] += 1
        stats["media_unknown_list"].append(raw)
        return raw

    # Bare path
    if resolved in media_mapping:
        mini = media_mapping[resolved]
        stats["media_bare"] += 1
        return f"{new_media_root}/{mini}"

    stats["media_unknown"] += 1
    stats["media_unknown_list"].append(raw)
    return raw


# ── fix-media: data structures ──────────────────────────────────────────────


@dataclass
class DatasetMediaInfo:
    dataset_path: str        # original dataset dir
    common_prefix: str       # e.g. "lustre/fs1/portfolios/nvr/.../GroundCUA/images"
    media_source_path: str   # "/" + common_prefix (the absolute media root)
    image_count: int         # total images found
    shard_count: int         # number of tar shards


# ── fix-media: helpers ──────────────────────────────────────────────────────


def _find_image_values(obj) -> List[str]:
    """Recursively find all ``{"t": "image", "value": "..."}`` fragments."""
    results: List[str] = []
    if isinstance(obj, dict):
        if obj.get("t") == "image" and "value" in obj:
            results.append(obj["value"])
        for v in obj.values():
            results.extend(_find_image_values(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_find_image_values(item))
    return results


def _common_dir_prefix(paths: List[str]) -> str:
    """Compute the longest common *directory* prefix of a list of paths.

    For example, given ``["a/b/c/img1.png", "a/b/c/img2.png"]`` returns
    ``"a/b/c"``.  Given ``["a/b/c/d/img.png", "a/b/e/img.png"]`` returns
    ``"a/b"``.
    """
    if not paths:
        return ""
    dirs = [os.path.dirname(p) for p in paths]
    prefix = os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0]
    return prefix


def _strip_prefix_from_value(value: str, prefix: str) -> str:
    """Strip *prefix* from *value*, returning the relative remainder.

    *prefix* should NOT have a trailing slash.  The result will NOT have a
    leading slash.
    """
    if not prefix:
        return value
    expect = prefix + "/"
    if value.startswith(expect):
        return value[len(expect):]
    # Edge case: value is exactly the prefix (a directory itself)
    if value == prefix:
        return ""
    return value


def _rewrite_image_values(obj, prefix: str):
    """Recursively strip *prefix* from all image value fields **in place**."""
    if isinstance(obj, dict):
        if obj.get("t") == "image" and "value" in obj:
            obj["value"] = _strip_prefix_from_value(obj["value"], prefix)
        for v in obj.values():
            _rewrite_image_values(v, prefix)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_image_values(item, prefix)


# ── fix-media: scan ─────────────────────────────────────────────────────────


def _scan_dataset_media(dataset_path: str) -> DatasetMediaInfo:
    """Scan tar shards in *dataset_path* to find image paths and their common prefix."""
    import glob as _glob

    tar_files = sorted(_glob.glob(os.path.join(dataset_path, "shard-*.tar")))
    if not tar_files:
        raise ValueError(f"No shard-*.tar files found in {dataset_path}")

    all_image_paths: List[str] = []

    for tf_path in tar_files:
        with tarfile.open(tf_path, "r:") as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".json"):
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data = json.loads(fobj.read().decode("utf-8"))
                all_image_paths.extend(_find_image_values(data))

    if not all_image_paths:
        raise ValueError(f"No image references found in tar shards of {dataset_path}")

    prefix = _common_dir_prefix(all_image_paths)
    media_source = "/" + prefix if prefix else "/"

    return DatasetMediaInfo(
        dataset_path=dataset_path,
        common_prefix=prefix,
        media_source_path=media_source,
        image_count=len(all_image_paths),
        shard_count=len(tar_files),
    )


# ── fix-media: rewrite tars ─────────────────────────────────────────────────


def _rewrite_dataset_tars(
    info: DatasetMediaInfo,
    output_dir: str,
) -> str:
    """Rewrite tar shards, stripping the common prefix from image paths.

    Returns the path to the output directory containing the rewritten tars.
    """
    import glob as _glob

    dataset_name = os.path.basename(info.dataset_path)
    out_dataset_dir = os.path.join(output_dir, "fixed_datasets", dataset_name)
    os.makedirs(out_dataset_dir, exist_ok=True)

    tar_files = sorted(_glob.glob(os.path.join(info.dataset_path, "shard-*.tar")))

    for tf_path in tar_files:
        out_tf_path = os.path.join(out_dataset_dir, os.path.basename(tf_path))
        with tarfile.open(tf_path, "r:") as tf_in, \
             tarfile.open(out_tf_path, "w:") as tf_out:
            for member in tf_in.getmembers():
                fobj = tf_in.extractfile(member)
                if fobj is None:
                    # Directory entry — add as-is
                    tf_out.addfile(member)
                    continue

                raw_bytes = fobj.read()

                if member.name.endswith(".json"):
                    data = json.loads(raw_bytes.decode("utf-8"))
                    _rewrite_image_values(data, info.common_prefix)
                    new_bytes = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                else:
                    new_bytes = raw_bytes

                # Create new tarinfo with updated size
                new_info = tarfile.TarInfo(name=member.name)
                new_info.size = len(new_bytes)
                new_info.mtime = member.mtime
                new_info.mode = member.mode
                new_info.uid = member.uid
                new_info.gid = member.gid
                new_info.uname = member.uname
                new_info.gname = member.gname

                tf_out.addfile(new_info, io.BytesIO(new_bytes))

    # Copy .nv-meta config files from original dataset so energon prepare
    # can run non-interactively (--tar-index-only).  We copy split.yaml,
    # dataset.yaml, and .info.json but NOT the sqlite index (byte offsets
    # differ in the rewritten tars).
    src_meta = os.path.join(info.dataset_path, ".nv-meta")
    if os.path.isdir(src_meta):
        dst_meta = os.path.join(out_dataset_dir, ".nv-meta")
        os.makedirs(dst_meta, exist_ok=True)
        for meta_file in ("split.yaml", "dataset.yaml", ".info.json", ".info.yaml"):
            src = os.path.join(src_meta, meta_file)
            if os.path.isfile(src):
                import shutil
                shutil.copy2(src, os.path.join(dst_meta, meta_file))

    return out_dataset_dir


# ── fix-media: discover from YAML ───────────────────────────────────────────


def _discover_root_media_datasets(
    yaml_paths: List[str],
) -> List[str]:
    """Find dataset dirs whose media_source is filesystem:/// (root).

    Walks all YAMLs recursively, identifies entries with root media_source,
    and returns the resolved dataset directory paths.
    """
    all_yamls, _ = _discover_yamls(yaml_paths)
    dataset_paths: List[str] = []

    for yaml_path in all_yamls:
        if not os.path.isfile(yaml_path):
            continue

        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        values = _extract_values(yaml_path)

        # We need to pair path: entries with their subsequent media_source:
        # Walk sequentially, tracking the last path seen.
        last_path: Optional[str] = None
        for typ, raw in values:
            if typ == "path":
                if _is_yaml(raw):
                    last_path = None
                    continue
                if os.path.isabs(raw):
                    resolved = _resolve(raw)
                else:
                    resolved = _resolve(os.path.normpath(os.path.join(yaml_dir, raw)))
                # Copy unit: .jsonl → parent dir
                if raw.endswith(".jsonl"):
                    last_path = os.path.dirname(resolved)
                else:
                    last_path = resolved
            elif typ == "media":
                _, _, is_root = _extract_media_path(raw)
                if is_root and last_path is not None:
                    if last_path not in dataset_paths:
                        dataset_paths.append(last_path)
                last_path = None

    return dataset_paths


# ── fix-media: output writers ───────────────────────────────────────────────


def _write_fix_media_report(
    infos: List[DatasetMediaInfo],
    output_dirs: List[str],
    output_dir: str,
):
    """Write ``fix_media_report.txt``."""
    report_path = os.path.join(output_dir, "fix_media_report.txt")
    with open(report_path, "w") as f:
        f.write("# fix-media report\n")
        f.write(f"# Datasets processed: {len(infos)}\n\n")
        for info, out_ds_dir in zip(infos, output_dirs):
            f.write(f"Dataset: {info.dataset_path}\n")
            f.write(f"  Shards: {info.shard_count}\n")
            f.write(f"  Images: {info.image_count:,}\n")
            f.write(f"  Common prefix (stripped): {info.common_prefix}/\n")
            f.write(f"  New media_source: filesystem://{info.media_source_path}\n")
            f.write(f"  Output: {out_ds_dir}/\n\n")
    print(f"Wrote report to {report_path}", file=sys.stderr)


def _write_fix_media_mapping(
    infos: List[DatasetMediaInfo],
    output_dir: str,
):
    """Write ``fix_media_mapping.txt`` — media dirs discovered from tars."""
    mapping_path = os.path.join(output_dir, "fix_media_mapping.txt")
    # Deduplicate: multiple datasets may share a media source
    unique: List[str] = []
    seen: Set[str] = set()
    for info in infos:
        resolved = _resolve(info.media_source_path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)

    minis = _compute_mini_paths(unique)

    with open(mapping_path, "w") as f:
        f.write("# MINI_PATH | REAL_PATH (media directories for fixed datasets)\n")
        f.write("# Use with: dm_copy_media.sh --mapping fix_media_mapping.txt\n")
        for real in sorted(minis):
            f.write(f"{minis[real]} | {real}\n")
    print(f"Wrote {len(minis)} media mappings to {mapping_path}", file=sys.stderr)


def _write_energon_prepare_script(
    output_dirs: List[str],
    output_dir: str,
):
    """Write ``energon_prepare.sh``.

    Since .nv-meta config files (split.yaml, dataset.yaml, .info.json) are
    copied from the original datasets during tar rewriting, we only need to
    regenerate the tar index (--tar-index-only).  This makes the script fully
    non-interactive.
    """
    script_path = os.path.join(output_dir, "energon_prepare.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Regenerate tar indices for fixed datasets.\n")
        f.write("# split.yaml, dataset.yaml, and .info.json are already copied from\n")
        f.write("# the original datasets, so only the sqlite index needs rebuilding.\n")
        f.write("# Usage: ./energon_prepare.sh\n")
        f.write("#\n")
        f.write("# NOTE: energon must be installed (pip install megatron-energon).\n\n")
        f.write("set -e\n\n")
        for out_ds_dir in sorted(output_dirs):
            f.write(f'energon prepare --tar-index-only "{out_ds_dir}"\n')
        f.write("\necho 'All datasets prepared successfully.'\n")
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"Wrote energon prepare script to {script_path}", file=sys.stderr)


def _write_yaml_overrides(
    infos: List[DatasetMediaInfo],
    output_dirs: List[str],
    output_dir: str,
):
    """Write ``fix_media_yaml_overrides.txt`` for integration with rewrite."""
    overrides_path = os.path.join(output_dir, "fix_media_yaml_overrides.txt")
    with open(overrides_path, "w") as f:
        f.write("# ORIGINAL_DATASET_PATH | NEW_DATASET_PATH | NEW_MEDIA_SOURCE\n")
        f.write("# Use with: copy_to_s3.py rewrite --fix-media-overrides <this file>\n")
        for info, out_ds_dir in zip(infos, output_dirs):
            media_src = f"filesystem://{info.media_source_path}"
            f.write(f"{info.dataset_path} | {os.path.abspath(out_ds_dir)} | {media_src}\n")
    print(f"Wrote YAML overrides to {overrides_path}", file=sys.stderr)


# ── fix-media: main entry point ─────────────────────────────────────────────


def run_fix_media(
    dataset_paths: List[str],
    output_dir: str,
) -> int:
    """Execute the fix-media workflow: scan, rewrite, generate outputs."""
    os.makedirs(output_dir, exist_ok=True)

    infos: List[DatasetMediaInfo] = []
    output_dirs: List[str] = []

    # Phase 1: Scan
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Phase 1: Scanning {len(dataset_paths)} dataset(s)", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    for ds_path in dataset_paths:
        name = os.path.basename(ds_path)
        print(f"\n  Scanning {name} ...", file=sys.stderr)
        try:
            info = _scan_dataset_media(ds_path)
        except ValueError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        infos.append(info)
        print(
            f"    {info.shard_count} shards, "
            f"{info.image_count:,} images, "
            f"prefix: {info.common_prefix}",
            file=sys.stderr,
        )

    if not infos:
        print("ERROR: No datasets could be scanned.", file=sys.stderr)
        return 1

    # Phase 2: Rewrite tars
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Phase 2: Rewriting tar shards", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    for info in infos:
        name = os.path.basename(info.dataset_path)
        print(f"\n  Rewriting {name} ({info.shard_count} shards) ...", file=sys.stderr)
        out_ds_dir = _rewrite_dataset_tars(info, output_dir)
        output_dirs.append(out_ds_dir)
        print(f"    -> {out_ds_dir}", file=sys.stderr)

    # Phase 3: Write outputs
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Phase 3: Writing output files", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    _write_fix_media_report(infos, output_dirs, output_dir)
    _write_fix_media_mapping(infos, output_dir)
    _write_energon_prepare_script(output_dirs, output_dir)
    _write_yaml_overrides(infos, output_dirs, output_dir)

    # Summary
    total_images = sum(i.image_count for i in infos)
    total_shards = sum(i.shard_count for i in infos)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  fix-media complete", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Datasets: {len(infos)}", file=sys.stderr)
    print(f"  Shards rewritten: {total_shards}", file=sys.stderr)
    print(f"  Image refs processed: {total_images:,}", file=sys.stderr)
    print(f"\n  Output directory: {output_dir}/", file=sys.stderr)
    print(f"  Next steps:", file=sys.stderr)
    print(f"    1. Review fix_media_report.txt", file=sys.stderr)
    print(f"    2. Copy fixed datasets + media to S3", file=sys.stderr)
    print(f"    3. Run energon_prepare.sh on target cluster", file=sys.stderr)
    print(f"    4. Run rewrite with --fix-media-overrides fix_media_yaml_overrides.txt", file=sys.stderr)

    return 0


# ── fix-media: rewrite integration ──────────────────────────────────────────


def _load_fix_media_overrides(filepath: str) -> Dict[str, Tuple[str, str]]:
    """Load overrides file: ``{original_path: (new_path, new_media_source)}``."""
    overrides: Dict[str, Tuple[str, str]] = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" | ")
            if len(parts) == 3:
                orig = parts[0].strip()
                new_path = parts[1].strip()
                new_media = parts[2].strip()
                overrides[orig] = (new_path, new_media)
    return overrides


# ── interactive: helpers ─────────────────────────────────────────────────────


def _prompt(message: str, default: str = "", yes: bool = False) -> str:
    """Prompt the user for input, returning *default* if ``--yes`` or empty."""
    suffix = f" [{default}]" if default else ""
    if yes:
        print(f"{message}{suffix}: {default}", file=sys.stderr)
        return default
    try:
        val = input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("", file=sys.stderr)
        return default
    return val if val else default


def _confirm(message: str, default: bool = True, yes: bool = False) -> bool:
    """Ask a yes/no question, returning *default* if ``--yes`` or empty."""
    hint = "Y/n" if default else "y/N"
    if yes:
        answer = "y" if default else "n"
        print(f"{message} [{hint}]: {answer}", file=sys.stderr)
        return default
    try:
        val = input(f"{message} [{hint}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("", file=sys.stderr)
        return default
    if not val:
        return default
    return val in ("y", "yes")


def _phase_header(title: str):
    """Print a phase header to stderr."""
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)


# ── interactive: merge fix-media into result ─────────────────────────────────


def _merge_fix_media_into_result(
    result: AnalysisResult,
    infos: List[DatasetMediaInfo],
    output_dirs: List[str],
) -> None:
    """Merge fix-media results into *result* in place.

    Adds fixed dataset output dirs and discovered media source dirs into the
    main ``AnalysisResult`` so that unified mapping files and copy scripts
    cover everything.  Removes ``root_media_source`` warnings since they are
    now handled.

    Media mini-paths are **recomputed** for the entire combined set so that
    newly added media paths participate in the uniqueness disambiguation.
    """
    needs_media_recompute = False
    for info, out_ds_dir in zip(infos, output_dirs):
        # Fixed datasets use a synthetic prefix — not part of the suffix algo.
        resolved_new = _resolve(out_ds_dir)
        ds_name = os.path.basename(resolved_new)
        result.dataset_dirs[resolved_new] = f"fixed_datasets/{ds_name}"

        # Add new media paths (will be assigned mini-paths below).
        resolved_media = _resolve(info.media_source_path)
        if resolved_media not in result.media_dirs:
            result.media_dirs[resolved_media] = ""  # placeholder
            needs_media_recompute = True

    # Recompute all media mini-paths so the new paths participate in
    # the uniqueness disambiguation alongside the originals.
    if needs_media_recompute:
        all_media = list(result.media_dirs.keys())
        minis = _compute_mini_paths(all_media)
        for p, mini in minis.items():
            result.media_dirs[p] = mini

    result.warnings = [
        w for w in result.warnings if w.kind != "root_media_source"
    ]


# ── interactive: rewrite script writer ───────────────────────────────────────


def _write_rewrite_script(
    yaml_paths: List[str],
    output_dir: str,
    strip_prefix: str,
    dataset_suffix: str,
    media_suffix: str,
    fix_media_overrides_file: Optional[str],
):
    """Write ``rewrite_yamls.sh`` — a wrapper to re-run rewrite with a new root."""
    script_path = os.path.join(output_dir, "rewrite_yamls.sh")
    # Use the absolute path to copy_to_s3.py
    tool_path = os.path.abspath(__file__)

    yaml_args = " ".join(f'"{os.path.abspath(y)}"' for y in yaml_paths)

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#\n")
        f.write("# Rewrite YAMLs for a new cluster.\n")
        f.write("# Usage: ./rewrite_yamls.sh <new-root>\n")
        f.write("#\n")
        f.write("# This script calls copy_to_s3.py rewrite with the correct\n")
        f.write("# arguments. You only need to provide the new dataset root.\n")
        f.write("#\n\n")
        f.write('NEW_ROOT="${1:?Usage: $0 <new-root>}"\n\n')
        f.write(f'python3 "{tool_path}" rewrite \\\n')
        f.write(f"    {yaml_args} \\\n")
        f.write(f'    -o "{os.path.abspath(output_dir)}" \\\n')
        f.write(f'    --new-root "$NEW_ROOT" \\\n')
        f.write(f'    --strip-prefix "{strip_prefix}" \\\n')
        f.write(f'    --dataset-suffix "{dataset_suffix}" \\\n')
        f.write(f'    --media-suffix "{media_suffix}"')
        if fix_media_overrides_file:
            f.write(f' \\\n    --fix-media-overrides "{os.path.abspath(fix_media_overrides_file)}"')
        f.write("\n")
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"Wrote rewrite script to {script_path}", file=sys.stderr)


# ── interactive: mini-path editor ─────────────────────────────────────────────


def _edit_mini_paths(dirs: Dict[str, str], label: str) -> bool:
    """Open ``$EDITOR`` to let the user review/rename mini-paths.

    *dirs* maps ``resolved_path -> mini_path``.  The user can edit the
    mini-path (left side) in the temp file.  Returns True if any changes
    were made.
    """
    # Sort for stable display order.
    sorted_entries = sorted(dirs.items(), key=lambda kv: kv[1])

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix=f"mini_paths_{label}_", delete=False,
    )
    try:
        tmp.write(f"# {label} mini-paths — edit the LEFT side only, keep the | separator\n")
        tmp.write(f"# WARNING: mini-paths must be unique (no duplicates on the left)\n")
        tmp.write(f"# Lines starting with # are ignored.  Blank lines are ignored.\n")
        tmp.write(f"# {len(sorted_entries)} entries\n")
        tmp.write("#\n")
        for resolved, mini in sorted_entries:
            tmp.write(f"{mini} | {resolved}\n")
        tmp.close()

        editor = os.environ.get("EDITOR", "vi")

        while True:
            subprocess.call([editor, tmp.name])

            # Parse the edited file.
            new_dirs: Dict[str, str] = {}
            seen_minis: Dict[str, str] = {}  # mini -> resolved (for dup detection)
            errors: List[str] = []

            with open(tmp.name) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "|" not in line:
                        errors.append(f"  Line {lineno}: missing '|' separator")
                        continue
                    mini, _, resolved = line.partition("|")
                    mini = mini.strip()
                    resolved = resolved.strip()
                    if not mini:
                        errors.append(f"  Line {lineno}: empty mini-path")
                        continue
                    if not resolved:
                        errors.append(f"  Line {lineno}: empty resolved path")
                        continue
                    if mini in seen_minis:
                        errors.append(
                            f"  Line {lineno}: duplicate mini-path '{mini}' "
                            f"(first used for {seen_minis[mini]})"
                        )
                        continue
                    seen_minis[mini] = resolved
                    new_dirs[resolved] = mini

            if errors:
                print(f"\n  Validation errors:", file=sys.stderr)
                for e in errors:
                    print(e, file=sys.stderr)
                retry = _confirm("  Re-open editor to fix?", default=True, yes=False)
                if not retry:
                    print("  Keeping original mini-paths.", file=sys.stderr)
                    return False
                continue

            # Check for missing entries (resolved paths in original but not in edit).
            missing = set(dirs.keys()) - set(new_dirs.keys())
            if missing:
                print(f"\n  WARNING: {len(missing)} resolved path(s) were removed:", file=sys.stderr)
                for m in sorted(missing)[:5]:
                    print(f"    {m}", file=sys.stderr)
                if len(missing) > 5:
                    print(f"    ... and {len(missing) - 5} more", file=sys.stderr)
                retry = _confirm("  Re-open editor to fix?", default=True, yes=False)
                if not retry:
                    print("  Keeping original mini-paths.", file=sys.stderr)
                    return False
                continue

            # Apply changes.
            changed = False
            for resolved, mini in new_dirs.items():
                if resolved in dirs and dirs[resolved] != mini:
                    changed = True
                dirs[resolved] = mini

            if changed:
                print(f"  Mini-paths updated.", file=sys.stderr)
            else:
                print(f"  No changes made.", file=sys.stderr)
            return changed

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _make_sibling_batches(
    sorted_entries: List[Tuple[str, str]],
    max_batch_size: int = 100,
) -> List[List[Tuple[str, str]]]:
    """Group entries by shared path prefix, then pack groups into batches.

    Sibling entries (those sharing a common parent directory) are kept in the
    same batch so the LLM can discover and apply groupings.  Groups are then
    packed into batches of up to *max_batch_size* entries.

    Large groups are automatically subdivided at a deeper path level so that
    no single group vastly exceeds *max_batch_size*.

    Parameters
    ----------
    sorted_entries : list of (resolved_path, mini_path)
        Must already be sorted by resolved_path.
    max_batch_size : int
        Target maximum entries per batch.
    """
    if not sorted_entries:
        return []

    def _meaningful_parts(resolved: str) -> List[str]:
        """Return path components after the project directory."""
        parts = resolved.split("/")
        try:
            idx = parts.index("projects")
            return parts[idx + 2:]
        except ValueError:
            return parts[-4:]

    def _group_key(resolved: str, depth: int) -> str:
        return "/".join(_meaningful_parts(resolved)[:depth])

    max_depth = 12  # safety limit to avoid infinite recursion

    def _split_group(
        entries: List[Tuple[str, str]], depth: int,
    ) -> List[List[Tuple[str, str]]]:
        """Recursively subdivide a group that exceeds *max_batch_size*."""
        if len(entries) <= max_batch_size or depth >= max_depth:
            return [entries]

        # Try splitting at the next depth level.
        sub_groups: List[List[Tuple[str, str]]] = []
        current_key: Optional[str] = None
        current_group: List[Tuple[str, str]] = []

        for entry in entries:
            key = _group_key(entry[0], depth + 1)
            if key != current_key:
                if current_group:
                    sub_groups.append(current_group)
                current_group = [entry]
                current_key = key
            else:
                current_group.append(entry)
        if current_group:
            sub_groups.append(current_group)

        # If this depth didn't split, try the next depth (skip through
        # single-component levels like .../playground/...).
        if len(sub_groups) <= 1:
            return _split_group(entries, depth + 1)

        # Recursively split sub-groups that are still too large.
        result: List[List[Tuple[str, str]]] = []
        for sg in sub_groups:
            result.extend(_split_group(sg, depth + 1))
        return result

    # Initial grouping at depth 2.
    groups: List[List[Tuple[str, str]]] = []
    current_key: Optional[str] = None
    current_group: List[Tuple[str, str]] = []

    for entry in sorted_entries:
        key = _group_key(entry[0], 2)
        if key != current_key:
            if current_group:
                groups.append(current_group)
            current_group = [entry]
            current_key = key
        else:
            current_group.append(entry)
    if current_group:
        groups.append(current_group)

    # Subdivide any groups that exceed max_batch_size.
    final_groups: List[List[Tuple[str, str]]] = []
    for g in groups:
        final_groups.extend(_split_group(g, 2))

    # Pack groups into batches.
    batches: List[List[Tuple[str, str]]] = []
    current_batch: List[Tuple[str, str]] = []

    for group in final_groups:
        if current_batch and len(current_batch) + len(group) > max_batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(group)
    if current_batch:
        batches.append(current_batch)

    return batches


def _build_review_prompt(
    label: str,
    batch: List[Tuple[str, str]],
    prior_prefixes: List[str],
) -> str:
    """Build the LLM review prompt for a single batch of entries."""
    data_lines = "".join(
        f"{mini} | {resolved}\n" for resolved, mini in batch
    )

    context_section = ""
    if prior_prefixes:
        prefix_list = "\n".join(f"  {p}" for p in sorted(prior_prefixes))
        context_section = (
            f"## Context from previous batches\n"
            f"\n"
            f"The following group prefixes have already been assigned in earlier\n"
            f"batches.  You may reuse them if entries belong to the same group,\n"
            f"but do NOT create new prefixes that conflict with these:\n"
            f"\n"
            f"{prefix_list}\n"
            f"\n"
        )

    return (
        f"You are reorganizing {label} source paths into a clean directory structure.\n"
        f"\n"
        f"You will receive a list of entries in the format: CURRENT_MINI_PATH | FULL_RESOLVED_PATH\n"
        f"\n"
        f"Your job is to replace each CURRENT_MINI_PATH with a better one that creates a "
        f"well-organized directory tree.\n"
        f"\n"
        f"## How to read the resolved paths\n"
        f"\n"
        f"The resolved paths follow patterns like:\n"
        f"  /lustre/.../projects/<project>/datasets/<collection>/<subcollection>/<name>\n"
        f"  /lustre/.../projects/<project>/users/<username>/datasets/<collection>/<name>\n"
        f"\n"
        f"Ignore everything up to and including the project name (e.g. llmservice_fm_vision, "
        f"llmservice_fm_audio) — that is cluster infrastructure, not meaningful.\n"
        f"\n"
        f"## What to do\n"
        f"\n"
        f"1. LOOK FOR SIBLINGS — entries whose resolved paths share a common parent directory. "
        f"These MUST share a common mini-path prefix. For example, if 5 entries all live under "
        f".../avlm/ALM_SFT/AVLM_SFT_AUDIO/, they should all start with a common prefix like "
        f"avlm_sft_audio/.\n"
        f"\n"
        f"2. CHOOSE MEANINGFUL GROUP NAMES — use the collection or project name from the "
        f"resolved path, not generic names. For .../avlm/video_audio_raw/... use something like "
        f"avlm_video_raw/, for .../grounding_data/images/... use grounding/.\n"
        f"\n"
        f"3. KEEP LEAF NAMES — the actual dataset name (the last meaningful component) should "
        f"usually be preserved as-is.\n"
        f"\n"
        f"4. DON'T OVER-NEST — if a dataset is the only entry in its collection, it can stay "
        f"flat. Grouping matters most when there are 2+ siblings.\n"
        f"\n"
        f"## Example\n"
        f"\n"
        f"Input:\n"
        f"action2sound | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/Ego/action2sound\n"
        f"ego_10 | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/Ego/ego_10\n"
        f"MMTrail | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/MMTrail\n"
        f"MiraData | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/MiraData\n"
        f"\n"
        f"Good output:\n"
        f"avlm_sft_audio/action2sound | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/Ego/action2sound\n"
        f"avlm_sft_audio/ego_10 | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/Ego/ego_10\n"
        f"avlm_sft_audio/MMTrail | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/MMTrail\n"
        f"avlm_sft_audio/MiraData | /lustre/.../datasets/avlm/ALM_SFT/AVLM_SFT_AUDIO/MiraData\n"
        f"\n"
        f"{context_section}"
        f"## Rules\n"
        f"- Output ONLY lines in format: MINI_PATH | RESOLVED_PATH\n"
        f"- Every input entry must appear in output. Do not skip entries.\n"
        f"- Mini-paths must be unique.\n"
        f"- No spaces in mini-paths.\n"
        f"- Preserve meaningful casing (VGGSound, TACOS, etc).\n"
        f"- No comments or explanation in output.\n"
        f"\n"
        f"## Input\n"
        f"\n"
        f"Here are {len(batch)} {label} mini-paths to reorganize:\n"
        f"\n"
        f"{data_lines}"
    )


def _parse_llm_response(response: str) -> Dict[str, str]:
    """Parse ``MINI_PATH | RESOLVED_PATH`` lines from an LLM response.

    Strips code fences, comments, and preamble text.  Returns a dict
    mapping *resolved_path* → *new_mini_path*.
    """
    suggestions: Dict[str, str] = {}
    for line in response.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if "|" not in line:
            continue
        mini, _, resolved = line.partition("|")
        mini = mini.strip().rstrip("/")
        resolved = resolved.strip()
        if mini and resolved:
            suggestions[resolved] = mini
    return suggestions


def _collect_prefixes(dirs: Dict[str, str]) -> List[str]:
    """Return the set of first-component prefixes used in *dirs* values."""
    prefixes: Set[str] = set()
    for mini in dirs.values():
        first = mini.split("/", 1)[0]
        if first != mini:  # has a prefix (not a flat name)
            prefixes.add(first + "/")
    return sorted(prefixes)


def _llm_review_mini_paths(
    dirs: Dict[str, str],
    label: str,
    max_batch_size: int = 100,
) -> bool:
    """Use the ``claude`` CLI to review and suggest improved mini-paths.

    Entries are batched by shared path prefix so siblings stay together.
    Each batch is sent to ``claude -p`` separately, and group prefixes
    from earlier batches are fed as context to later ones for consistency.

    Returns True always (the editor opens after for final review).
    """
    # Check that the claude CLI is available.
    if not _which("claude"):
        print(
            "  ERROR: 'claude' CLI not found on PATH.\n"
            "  Install from: https://docs.anthropic.com/en/docs/claude-code",
            file=sys.stderr,
        )
        return False

    sorted_entries = sorted(dirs.items(), key=lambda kv: kv[0])  # sort by resolved path
    batches = _make_sibling_batches(sorted_entries, max_batch_size)

    n_batches = len(batches)
    print(
        f"  Reviewing {len(sorted_entries)} {label} mini-paths "
        f"in {n_batches} batch(es)...",
        file=sys.stderr,
    )

    total_changes = 0
    prior_prefixes: List[str] = []

    for i, batch in enumerate(batches, 1):
        print(
            f"    Batch {i}/{n_batches} ({len(batch)} entries)...",
            file=sys.stderr,
            end=" ",
        )

        prompt = _build_review_prompt(label, batch, prior_prefixes)

        try:
            proc = subprocess.run(
                ["claude", "-p"],
                input=prompt, capture_output=True, text=True,
            )
        except FileNotFoundError:
            print("ERROR: 'claude' CLI not found.", file=sys.stderr)
            return False

        if proc.returncode != 0:
            err_msg = (proc.stdout or proc.stderr or "(no output)").strip()[:300]
            print(f"ERROR (exit {proc.returncode}): {err_msg}", file=sys.stderr)
            continue  # skip this batch, try the next

        # claude -p writes its response to stderr.
        response = proc.stderr or proc.stdout or ""
        suggestions = _parse_llm_response(response)

        # Apply suggestions.
        batch_changes = 0
        for resolved, new_mini in suggestions.items():
            if resolved in dirs and dirs[resolved] != new_mini:
                dirs[resolved] = new_mini
                batch_changes += 1

        total_changes += batch_changes
        print(f"{batch_changes} change(s).", file=sys.stderr)

        # Collect prefixes used so far to inform subsequent batches.
        prior_prefixes = _collect_prefixes(dirs)

    if total_changes:
        print(f"  LLM suggested {total_changes} total change(s).", file=sys.stderr)
    else:
        print(f"  No changes suggested.", file=sys.stderr)

    # Check for duplicate mini-paths introduced across batches.
    mini_to_resolved: Dict[str, List[str]] = {}
    for resolved, mini in dirs.items():
        mini_to_resolved.setdefault(mini, []).append(resolved)

    dupes = {m: ps for m, ps in mini_to_resolved.items() if len(ps) > 1}
    if dupes:
        print(
            f"\n  WARNING: {len(dupes)} duplicate mini-path(s) detected "
            f"(will need fixing in editor):",
            file=sys.stderr,
        )
        for mini, paths in sorted(dupes.items())[:10]:
            print(f"    '{mini}' used by {len(paths)} entries", file=sys.stderr)
        if len(dupes) > 10:
            print(f"    ... and {len(dupes) - 10} more", file=sys.stderr)

    return True


def _which(cmd: str) -> Optional[str]:
    """Return the path to *cmd* if it's on PATH, else None."""
    return shutil.which(cmd)


# ── interactive: orchestration ───────────────────────────────────────────────


def run_interactive(args) -> int:
    """Run the full interactive pipeline."""
    yes = args.yes

    # ── Phase 1: Configuration ───────────────────────────────────────────
    _phase_header("Phase 1: Configuration")

    s3_dest = args.s3_dest
    if not s3_dest:
        if yes:
            print("ERROR: --s3-dest is required with --yes", file=sys.stderr)
            return 1
        while not s3_dest:
            s3_dest = _prompt("S3 destination (e.g. team-foo:bucket/prefix)", "", yes)
            if not s3_dest:
                print("  S3 destination cannot be empty.", file=sys.stderr)

    dm_args = args.dm_args or ""
    dataset_suffix = args.s3_dataset_suffix
    media_suffix = args.s3_media_suffix
    strip_prefix = args.strip_prefix
    splits = set(args.splits) if args.splits else None
    output_dir = args.output_dir

    print(f"\n  Settings:", file=sys.stderr)
    print(f"    S3 dest:         {s3_dest}", file=sys.stderr)
    print(f"    Dataset subdir:  {s3_dest}{dataset_suffix}/", file=sys.stderr)
    print(f"    Media subdir:    {s3_dest}{media_suffix}/", file=sys.stderr)
    if dm_args:
        print(f"    DM args:         {dm_args}", file=sys.stderr)
    print(f"    Splits:          {splits or 'all'}", file=sys.stderr)
    print(f"    Output dir:      {output_dir}", file=sys.stderr)

    # ── Phase 2: Analyze ─────────────────────────────────────────────────
    _phase_header("Phase 2: Analyzing YAMLs")

    print(f"  Analyzing {len(args.yamls)} YAML file(s)...\n", file=sys.stderr)
    result = analyze_yamls(args.yamls, splits=splits)

    missing_ds = [w for w in result.warnings if w.kind == "missing_dataset"]
    missing_ms = [w for w in result.warnings if w.kind == "missing_media"]
    root_ms = [w for w in result.warnings if w.kind == "root_media_source"]

    print(f"  Analysis complete:", file=sys.stderr)
    print(f"    YAML files discovered: {len(result.yaml_files)}", file=sys.stderr)
    print(f"    Unique dataset dirs:   {len(result.dataset_dirs)}", file=sys.stderr)
    print(f"    Unique media dirs:     {len(result.media_dirs)}", file=sys.stderr)
    print(f"    Warnings:              {len(result.warnings)}", file=sys.stderr)
    if missing_ds:
        print(f"      - {len(missing_ds)} missing dataset(s)", file=sys.stderr)
    if missing_ms:
        print(f"      - {len(missing_ms)} missing media source(s)", file=sys.stderr)
    if root_ms:
        print(f"      - {len(root_ms)} root filesystem:/// entry/entries", file=sys.stderr)

    # ── Phase 3: Fix Broken Media ────────────────────────────────────────
    fix_media_ran = False
    fix_media_infos: List[DatasetMediaInfo] = []
    fix_media_output_dirs: List[str] = []
    fix_media_overrides_file: Optional[str] = None

    if root_ms and not args.no_fix_media:
        _phase_header("Phase 3: Fix Broken Media")

        root_ds_paths = _discover_root_media_datasets(args.yamls)

        print(
            f"  Found {len(root_ms)} root filesystem:/// warning(s) "
            f"affecting {len(root_ds_paths)} dataset(s).",
            file=sys.stderr,
        )
        print(
            "  These datasets have absolute image paths baked into their tar\n"
            "  metadata. They need to be rewritten with relative paths to work\n"
            "  on a new cluster.\n",
            file=sys.stderr,
        )
        print(f"  Affected datasets:", file=sys.stderr)
        for i, ds in enumerate(root_ds_paths, 1):
            print(f"    {i}. {ds}", file=sys.stderr)

        do_fix = _confirm("\n  Fix these datasets now?", default=True, yes=yes)

        if do_fix:
            print(f"\n  Scanning {len(root_ds_paths)} dataset(s)...", file=sys.stderr)
            for ds_path in root_ds_paths:
                name = os.path.basename(ds_path)
                print(f"    Scanning {name} ...", file=sys.stderr, end=" ")
                try:
                    info = _scan_dataset_media(ds_path)
                except ValueError as e:
                    print(f"ERROR: {e}", file=sys.stderr)
                    continue
                fix_media_infos.append(info)
                print(
                    f"{info.shard_count} shards, "
                    f"{info.image_count:,} images, "
                    f"prefix: {info.common_prefix}",
                    file=sys.stderr,
                )

            if fix_media_infos:
                print(f"\n  Rewriting tar shards...", file=sys.stderr)
                for info in fix_media_infos:
                    name = os.path.basename(info.dataset_path)
                    print(f"    Rewriting {name} ({info.shard_count} shards) ...", file=sys.stderr)
                    out_ds_dir = _rewrite_dataset_tars(info, output_dir)
                    fix_media_output_dirs.append(out_ds_dir)
                    print(f"      -> {out_ds_dir}", file=sys.stderr)

                # Write fix-media specific outputs
                _write_fix_media_report(fix_media_infos, fix_media_output_dirs, output_dir)
                _write_fix_media_mapping(fix_media_infos, output_dir)
                _write_energon_prepare_script(fix_media_output_dirs, output_dir)
                _write_yaml_overrides(fix_media_infos, fix_media_output_dirs, output_dir)
                fix_media_overrides_file = os.path.join(output_dir, "fix_media_yaml_overrides.txt")

                # Merge into main result
                _merge_fix_media_into_result(
                    result, fix_media_infos, fix_media_output_dirs,
                )
                fix_media_ran = True

                total_images = sum(i.image_count for i in fix_media_infos)
                total_shards = sum(i.shard_count for i in fix_media_infos)
                print(
                    f"\n  Fix-media complete: {len(fix_media_infos)} dataset(s), "
                    f"{total_shards} shards, {total_images:,} images",
                    file=sys.stderr,
                )
                print("  Results merged into main mappings.", file=sys.stderr)
        else:
            print("  Skipping fix-media.", file=sys.stderr)
    elif root_ms and args.no_fix_media:
        print(
            f"\n  NOTE: {len(root_ms)} root filesystem:/// warning(s) found "
            "but --no-fix-media was set. Skipping.",
            file=sys.stderr,
        )

    # ── Phase 3.5: Review Mini-Paths ────────────────────────────────────
    if not yes:
        _phase_header("Phase 3.5: Review Mini-Paths")

        print(
            "  Mini-paths determine the S3 object structure and rewritten\n"
            "  YAML paths. You can review and rename them manually in\n"
            "  $EDITOR, or have an LLM suggest more descriptive names.\n",
            file=sys.stderr,
        )
        print(
            f"    Dataset mini-paths: {len(result.dataset_dirs)}",
            file=sys.stderr,
        )
        print(
            f"    Media mini-paths:   {len(result.media_dirs)}",
            file=sys.stderr,
        )

        for dir_label, dir_dict in [("dataset", result.dataset_dirs), ("media", result.media_dirs)]:
            print(f"\n  How would you like to review {dir_label} mini-paths?", file=sys.stderr)
            print(f"    1. Skip (keep as-is)", file=sys.stderr)
            print(f"    2. Edit in $EDITOR", file=sys.stderr)
            print(f"    3. LLM review (uses claude CLI)", file=sys.stderr)
            choice = _prompt("  Choice [1/2/3]", "1", yes)
            if choice == "2":
                _edit_mini_paths(dir_dict, dir_label)
            elif choice == "3":
                _llm_review_mini_paths(dir_dict, dir_label)
                print("  Opening $EDITOR for final review...", file=sys.stderr)
                _edit_mini_paths(dir_dict, dir_label)

    # ── Phase 4: Write Unified Outputs ───────────────────────────────────
    _phase_header("Phase 4: Writing Output Files")

    write_mappings(result, output_dir, s3_dest, dataset_suffix, media_suffix)
    build_staging_dir(result, output_dir, dataset_suffix, media_suffix)
    write_staged_copy_script(output_dir, s3_dest, dm_args)
    write_warnings(result, output_dir)

    # ── Phase 5: Rewrite YAMLs ───────────────────────────────────────────
    did_rewrite = False

    if not args.no_rewrite:
        _phase_header("Phase 5: Rewrite YAMLs")

        do_rewrite = _confirm("Generate rewritten YAMLs now?", default=True, yes=yes)

        if do_rewrite:
            new_root = args.new_root
            if not new_root:
                if yes:
                    print(
                        "ERROR: --new-root is required with --yes when rewrite is enabled",
                        file=sys.stderr,
                    )
                    return 1
                while not new_root:
                    new_root = _prompt("New root path on target cluster", "", yes)
                    if not new_root:
                        print("  New root cannot be empty.", file=sys.stderr)

            rc = rewrite_yamls(
                args.yamls,
                output_dir=output_dir,
                new_root=new_root,
                dataset_suffix=dataset_suffix,
                media_suffix=media_suffix,
                strip_prefix=strip_prefix,
                fix_media_overrides_file=fix_media_overrides_file,
            )
            did_rewrite = True
            if rc != 0:
                print("\n  WARNING: Some YAMLs had unhandled entries (see above).", file=sys.stderr)
        else:
            print("  Skipping YAML rewrite.", file=sys.stderr)

    # ── Phase 5b: Rewrite script ─────────────────────────────────────────
    wrote_rewrite_script = False

    if not args.no_rewrite:
        _write_rewrite_script(
            args.yamls, output_dir, strip_prefix,
            dataset_suffix, media_suffix,
            fix_media_overrides_file,
        )
        wrote_rewrite_script = True

    # ── Phase 6: Summary ─────────────────────────────────────────────────
    _phase_header("Summary")

    print(f"  Output directory: {output_dir}/\n", file=sys.stderr)
    print(f"  Generated files:", file=sys.stderr)
    print(f"    datasets_mapping.txt      {len(result.dataset_dirs)} dataset mappings", file=sys.stderr)
    print(f"    media_mapping.txt         {len(result.media_dirs)} media mappings", file=sys.stderr)
    print(f"    staging/                  Symlink tree mirroring S3 layout", file=sys.stderr)
    print(f"    dm_copy_staged.sh         Single copy script (--follow-symlinks)", file=sys.stderr)
    print(f"    warnings.txt              {len(result.warnings)} warning(s)", file=sys.stderr)
    if fix_media_ran:
        print(f"    fix_media_report.txt      {len(fix_media_infos)} fixed dataset(s)", file=sys.stderr)
        print(f"    fix_media_mapping.txt     fix-media media dirs", file=sys.stderr)
        print(f"    fix_media_yaml_overrides.txt", file=sys.stderr)
        print(f"    energon_prepare.sh        {len(fix_media_output_dirs)} dataset(s) to prepare", file=sys.stderr)
        print(f"    fixed_datasets/           Rewritten tar shards", file=sys.stderr)
    if did_rewrite:
        yamls_dir = os.path.join(output_dir, "yamls")
        if os.path.isdir(yamls_dir):
            n = len(os.listdir(yamls_dir))
            print(f"    yamls/                    {n} rewritten YAML(s)", file=sys.stderr)
    if wrote_rewrite_script:
        print(f"    rewrite_yamls.sh          Rewrite YAMLs with a new root", file=sys.stderr)

    out = output_dir.rstrip("/")
    print(f"\n  Next steps:", file=sys.stderr)
    step = 1
    print(f"    {step}. Review warnings.txt", file=sys.stderr); step += 1
    if fix_media_ran:
        print(f"    {step}. Run: {out}/energon_prepare.sh", file=sys.stderr); step += 1
    print(f"    {step}. Run: {out}/dm_copy_staged.sh --dry-run", file=sys.stderr); step += 1
    print(f"    {step}. Remove --dry-run to copy data to S3", file=sys.stderr); step += 1
    if did_rewrite:
        print(f"    {step}. Deploy rewritten YAMLs from {out}/yamls/ dir", file=sys.stderr); step += 1
    if wrote_rewrite_script:
        print(f"    {step}. To rewrite YAMLs later: {out}/rewrite_yamls.sh <new-root>", file=sys.stderr); step += 1

    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy MetadatasetV2 datasets to S3 and generate rewritten YAMLs "
            "for a new cluster."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── analyze ──────────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze YAMLs and generate mappings + copy scripts.",
    )
    p_analyze.add_argument("yamls", nargs="+", help="Input YAML file(s).")
    p_analyze.add_argument(
        "--s3-dest", required=True,
        help="S3 destination for datasets (e.g. team-foo:bucket/prefix).",
    )
    p_analyze.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for mappings and scripts.",
    )
    p_analyze.add_argument(
        "--dm-args", default="--slurm-nodes 1",
        help="Extra args for dm job copy (datasets). Default: --slurm-nodes 1",
    )
    p_analyze.add_argument(
        "--s3-dataset-suffix", default="/datasets",
        help="Subdir appended to --s3-dest for datasets. Default: /datasets",
    )
    p_analyze.add_argument(
        "--s3-media-suffix", default="/media_sources",
        help="Subdir appended to --s3-dest for media. Default: /media_sources",
    )
    p_analyze.add_argument(
        "--splits", nargs="*", default=None,
        help=(
            "Only include entries from these splits (e.g. --splits train). "
            "Default: all splits. Use --splits train val to include both."
        ),
    )

    # ── rewrite ──────────────────────────────────────────────────────────
    p_rewrite = subparsers.add_parser(
        "rewrite",
        help="Generate rewritten YAMLs for a new cluster.",
    )
    p_rewrite.add_argument(
        "yamls", nargs="+",
        help="Input YAML file(s) (same as used with analyze).",
    )
    p_rewrite.add_argument(
        "--output-dir", "-o", required=True,
        help="Output dir (must contain mappings from analyze).",
    )
    p_rewrite.add_argument(
        "--new-root", required=True,
        help="New root path on the target cluster for datasets.",
    )
    p_rewrite.add_argument(
        "--strip-prefix", default="/portfolios/",
        help="Strip prefix for mapping-file lookups during rewrite. Default: /portfolios/",
    )
    p_rewrite.add_argument(
        "--dataset-suffix", default="/datasets",
        help="Subdir appended to --new-root for datasets. Default: /datasets",
    )
    p_rewrite.add_argument(
        "--media-suffix", default="/media_sources",
        help="Subdir appended to --new-root for media. Default: /media_sources",
    )
    p_rewrite.add_argument(
        "--fix-media-overrides",
        help=(
            "Path to fix_media_yaml_overrides.txt from fix-media step. "
            "Overrides dataset paths and media_source for fixed datasets."
        ),
    )

    # ── fix-media ─────────────────────────────────────────────────────────
    p_fix = subparsers.add_parser(
        "fix-media",
        help="Fix datasets with filesystem:/// by rewriting tar paths to relative.",
    )
    p_fix.add_argument(
        "--datasets", nargs="*",
        help="Dataset directory paths to fix.",
    )
    p_fix.add_argument(
        "--from-yaml", nargs="*",
        help="Auto-discover filesystem:/// datasets from YAML file(s).",
    )
    p_fix.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for fixed datasets and helper files.",
    )

    # ── interactive ───────────────────────────────────────────────────────
    p_interactive = subparsers.add_parser(
        "interactive",
        help="Guided workflow: analyze, fix broken media, and rewrite YAMLs.",
    )
    p_interactive.add_argument("yamls", nargs="+", help="Input YAML file(s).")
    p_interactive.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for all generated files.",
    )
    p_interactive.add_argument(
        "--s3-dest", default=None,
        help="S3 destination (e.g. team-foo:bucket/prefix). Prompted if not given.",
    )
    p_interactive.add_argument(
        "--new-root", default=None,
        help="New root path on target cluster. Prompted if not given.",
    )
    p_interactive.add_argument(
        "--strip-prefix", default="/portfolios/",
        help="Strip prefix for rewrite lookups and generated scripts. Default: /portfolios/",
    )
    p_interactive.add_argument(
        "--s3-dataset-suffix", default="/datasets",
        help="Subdir appended to --s3-dest for datasets. Default: /datasets",
    )
    p_interactive.add_argument(
        "--s3-media-suffix", default="/media_sources",
        help="Subdir appended to --s3-dest for media. Default: /media_sources",
    )
    p_interactive.add_argument(
        "--dm-args", default=None,
        help="Extra args for dm job copy (datasets). Default: none.",
    )
    p_interactive.add_argument(
        "--splits", nargs="*", default=None,
        help="Only include entries from these splits. Default: all.",
    )
    p_interactive.add_argument(
        "--yes", "-y", action="store_true", default=False,
        help="Accept all defaults without prompting (non-interactive mode).",
    )
    p_interactive.add_argument(
        "--no-fix-media", action="store_true", default=False,
        help="Skip fix-media step entirely.",
    )
    p_interactive.add_argument(
        "--no-rewrite", action="store_true", default=False,
        help="Skip YAML rewrite step entirely.",
    )

    # ── review-mappings ──────────────────────────────────────────────────
    p_review = subparsers.add_parser(
        "review-mappings",
        help="Re-review mini-paths in existing mapping files.",
    )
    p_review.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory containing mapping files from a prior run.",
    )
    p_review.add_argument(
        "--datasets", action="store_true", default=False,
        help="Review dataset mini-paths.",
    )
    p_review.add_argument(
        "--media", action="store_true", default=False,
        help="Review media mini-paths.",
    )
    p_review.add_argument(
        "--llm-only", action="store_true", default=False,
        help="Only run LLM review (skip editor).",
    )
    p_review.add_argument(
        "--editor-only", action="store_true", default=False,
        help="Only open editor (skip LLM review).",
    )
    p_review.add_argument(
        "--dm-args", default=None,
        help="Override DM args for copy script. Default: read from existing script.",
    )

    # ── size ─────────────────────────────────────────────────────────────
    p_size = subparsers.add_parser(
        "size",
        help="Report disk sizes of all datasets and media in the input YAMLs.",
    )
    p_size.add_argument("yamls", nargs="+", help="Input YAML file(s).")
    p_size.add_argument(
        "--splits", nargs="*", default=None,
        help="Only include entries from these splits. Default: all.",
    )
    p_size.add_argument(
        "--datasets", action="store_true", default=False,
        help="Only report dataset sizes.",
    )
    p_size.add_argument(
        "--media", action="store_true", default=False,
        help="Only report media sizes.",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "analyze":
        splits = set(args.splits) if args.splits else None

        result = analyze_yamls(args.yamls, splits=splits)

        print(f"\nAnalysis complete:", file=sys.stderr)
        print(f"  YAML files discovered: {len(result.yaml_files)}", file=sys.stderr)
        print(f"  Unique dataset dirs: {len(result.dataset_dirs)}", file=sys.stderr)
        print(f"  Unique media dirs: {len(result.media_dirs)}", file=sys.stderr)
        print(f"  Warnings: {len(result.warnings)}", file=sys.stderr)

        write_mappings(result, args.output_dir, args.s3_dest,
                       args.s3_dataset_suffix, args.s3_media_suffix)
        build_staging_dir(result, args.output_dir,
                          args.s3_dataset_suffix, args.s3_media_suffix)
        write_staged_copy_script(args.output_dir, args.s3_dest, args.dm_args)
        write_warnings(result, args.output_dir)

        print(f"\nDone. Output in {args.output_dir}/", file=sys.stderr)
        print(f"  Next steps:", file=sys.stderr)
        print(f"    1. Review warnings.txt", file=sys.stderr)
        print(f"    2. Run dm_copy_staged.sh --dry-run", file=sys.stderr)
        print(f"    3. Remove --dry-run to copy data", file=sys.stderr)
        print(f"    4. Run: copy_to_s3.py rewrite ... --new-root <path>", file=sys.stderr)
        return 0

    elif args.command == "rewrite":
        return rewrite_yamls(
            args.yamls,
            output_dir=args.output_dir,
            new_root=args.new_root,
            dataset_suffix=args.dataset_suffix,
            media_suffix=args.media_suffix,
            strip_prefix=args.strip_prefix,
            fix_media_overrides_file=args.fix_media_overrides,
        )

    elif args.command == "fix-media":
        dataset_paths: List[str] = []

        if args.from_yaml:
            print("Auto-discovering filesystem:/// datasets from YAML(s)...", file=sys.stderr)
            discovered = _discover_root_media_datasets(args.from_yaml)
            print(f"  Found {len(discovered)} dataset(s) with root media_source", file=sys.stderr)
            dataset_paths.extend(discovered)

        if args.datasets:
            dataset_paths.extend(args.datasets)

        if not dataset_paths:
            print(
                "ERROR: No datasets specified. Use --datasets or --from-yaml.",
                file=sys.stderr,
            )
            return 1

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for p in dataset_paths:
            rp = _resolve(p)
            if rp not in seen:
                seen.add(rp)
                unique.append(rp)

        return run_fix_media(unique, args.output_dir)

    elif args.command == "interactive":
        return run_interactive(args)

    elif args.command == "review-mappings":
        return run_review_mappings(args)

    elif args.command == "size":
        return run_size(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
