#!/usr/bin/env python3
"""
Media Source Access Checker Script
===================================
This script parses a YAML configuration file to extract:
1. Dataset paths (containing .tar.idx files)
2. aux/media_source paths — on-disk roots for images or webdataset shards referenced by
   JSONL / conversation-style blends

And checks if folders and files have read access for 'all' (others) or 'group'.


Features:
- Extracts both 'path' and 'media_source' from YAML configuration
- Specifically checks .tar.idx files in dataset paths
- Checks folder and file read permissions
- Optional verification: check that actual files exist under aux dirs (not just empty directory trees from a failed copy)
- Tracks progress with a progress tracker
- Generates detailed reports (console + file)

Usage:
    python check_media_access.py <yaml_file> [--output <report_file>] [--max-files <N>] [--verify-aux-media nonempty] [--verify-shard-paths]
Example:
    # Very quick
    python check_media_access.py config.yaml --output report.txt
    python check_media_access.py config.yaml --output report.txt --verify-aux-media nonempty --verify-shard-paths
"""

import os
import sys
import stat
import tarfile
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import time


@dataclass
class FileAccessInfo:
    """Stores access information for a single file/folder."""
    path: str
    exists: bool
    is_directory: bool
    has_group_read: bool = False
    has_other_read: bool = False
    has_any_read: bool = False  # True if group OR other has read
    permissions: str = ""
    error: str = ""


@dataclass
class TarIdxReport:
    """Report for .tar.idx files in a dataset path."""
    total_tar_idx_files: int = 0
    tar_idx_with_read: int = 0
    tar_idx_without_read: int = 0
    sample_tar_idx_without_read: List[str] = field(default_factory=list)


@dataclass
class MediaSourceReport:
    """Report for a single media source path."""
    media_source_path: str
    path_type: str = "media_source"  # "media_source" or "dataset_path"
    folder_access: Optional[FileAccessInfo] = None
    files_checked: int = 0
    files_with_group_read: int = 0
    files_with_other_read: int = 0
    files_with_any_read: int = 0
    files_without_read: int = 0
    sample_files_without_read: List[str] = field(default_factory=list)
    tar_idx_report: Optional[TarIdxReport] = None
    errors: List[str] = field(default_factory=list)
    # aux.media_source verification (path_type == "media_source" only)
    aux_media_verify_mode: str = "off"  # off | nonempty | content
    media_dir_nonempty: Optional[bool] = None


@dataclass
class ProgressTracker:
    """Tracks progress of the access check."""
    total_sources: int = 0
    processed_sources: int = 0
    total_files_checked: int = 0
    sources_accessible: int = 0
    sources_inaccessible: int = 0
    sources_with_errors: int = 0
    # Dataset path specific tracking
    total_dataset_paths: int = 0
    dataset_paths_accessible: int = 0
    total_tar_idx_files: int = 0
    tar_idx_accessible: int = 0
    tar_idx_inaccessible: int = 0
    # Media source specific tracking
    total_media_sources: int = 0
    media_sources_accessible: int = 0
    start_time: float = field(default_factory=time.time)
    def update(self, report: MediaSourceReport):
        """Update tracker with results from a media source check."""
        self.processed_sources += 1
        self.total_files_checked += report.files_checked

        is_accessible = report.folder_access and report.folder_access.exists and report.folder_access.has_any_read

        if is_accessible:
            self.sources_accessible += 1
        elif report.errors:
            self.sources_with_errors += 1
        else:
            self.sources_inaccessible += 1
        # Track by path type
        if report.path_type == "dataset_path":
            if is_accessible:
                self.dataset_paths_accessible += 1
            # Track tar.idx files
            if report.tar_idx_report:
                self.total_tar_idx_files += report.tar_idx_report.total_tar_idx_files
                self.tar_idx_accessible += report.tar_idx_report.tar_idx_with_read
                self.tar_idx_inaccessible += report.tar_idx_report.tar_idx_without_read
        elif report.path_type == "media_source":
            if is_accessible:
                self.media_sources_accessible += 1
    def get_progress_string(self) -> str:
        """Get a formatted progress string."""
        elapsed = time.time() - self.start_time
        percent = (self.processed_sources / self.total_sources * 100) if self.total_sources > 0 else 0
        tar_idx_str = f" | tar.idx: {self.tar_idx_accessible}/{self.total_tar_idx_files}" if self.total_tar_idx_files > 0 else ""
        return (
            f"[{self.processed_sources}/{self.total_sources}] ({percent:.1f}%) | "
            f"Accessible: {self.sources_accessible} | "
            f"Inaccessible: {self.sources_inaccessible} | "
            f"Errors: {self.sources_with_errors}{tar_idx_str} | "
            f"Elapsed: {elapsed:.1f}s"
        )


def check_permissions(path: str) -> FileAccessInfo:
    """
    Check read permissions for a file or directory.
    Returns FileAccessInfo with permission details.
    """
    info = FileAccessInfo(
        path=path,
        exists=False,
        is_directory=False
    )
    try:
        if not os.path.exists(path):
            info.error = "Path does not exist"
            return info

        info.exists = True
        info.is_directory = os.path.isdir(path)

        # Get file stats
        file_stat = os.stat(path)
        mode = file_stat.st_mode

        # Check group read permission (bit 5, or S_IRGRP)
        info.has_group_read = bool(mode & stat.S_IRGRP)

        # Check other/all read permission (bit 2, or S_IROTH)
        info.has_other_read = bool(mode & stat.S_IROTH)

        # Has any read access (group OR other)
        info.has_any_read = info.has_group_read or info.has_other_read

        # Get human-readable permissions string
        info.permissions = stat.filemode(mode)

    except PermissionError as e:
        info.error = f"Permission denied: {e}"
    except OSError as e:
        info.error = f"OS error: {e}"
    except Exception as e:
        info.error = f"Unexpected error: {e}"

    return info


def _strip_filesystem_media_prefix(value: str) -> str:
    """Normalize filesystem:// / filesystem:/// prefixes on aux media paths."""
    clean_path = value
    if clean_path.startswith("filesystem:///"):
        clean_path = clean_path[len("filesystem://"):]
    elif clean_path.startswith("filesystem://"):
        clean_path = clean_path[len("filesystem://"):]
    return clean_path


def jsonl_has_media_references(jsonl_path: str, max_lines: int = 32) -> bool:
    """Sample a JSONL file and return True if any lines contain media fragment references.
    Checks both conversation-style fragment tags and top-level media keys."""
    MEDIA_FRAGMENT_TAGS = ('"t": "image"', '"t": "video"', '"t": "audio"', '"t": "video_frame"',
                           '<image>', '<video>', '<sound>', '<video-sound>')
    MEDIA_TOP_KEYS = {"images", "image", "video", "videos", "audio", "audios", "sound", "speech"}
    try:
        lines_read = 0
        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if lines_read >= max_lines:
                    break
                lines_read += 1
                for tag in MEDIA_FRAGMENT_TAGS:
                    if tag in line:
                        return True
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                for key in MEDIA_TOP_KEYS:
                    if data.get(key):
                        return True
    except OSError:
        pass
    return False


def _resolve_media_path(key: str, media_root: str) -> str:
    """Resolve a media key against a media_source root (mirrors Energon FileStore semantics)."""
    key = key.strip()
    if not key:
        return ""
    if os.path.isabs(key):
        return os.path.normpath(key)
    return os.path.normpath(os.path.join(media_root, key))


def verify_shard_image_paths(
    shard_dir: str,
    media_root: str,
    max_tars: int = 3,
    max_records_per_tar: int = 16,
    missing_sample_cap: int = 10,
) -> Tuple[List[str], int, int]:
    """
    Sample tar shards from a WebDataset directory and verify that image paths
    referenced in the JSON records actually exist on disk under media_root.

    Returns:
        (warning_messages, records_checked, total_missing)
    """
    warnings: List[str] = []
    records_checked = 0
    total_missing = 0

    try:
        tar_files = sorted(
            e.path for e in os.scandir(shard_dir)
            if e.name.endswith('.tar') and e.is_file()
        )
    except OSError as e:
        return [f"[shard] Cannot scan {shard_dir}: {e}"], 0, 0

    if not tar_files:
        return [f"[shard] No .tar files in {shard_dir}"], 0, 0

    # Sample evenly across available tars
    step = max(1, len(tar_files) // max_tars)
    sampled = tar_files[::step][:max_tars]

    for tar_path in sampled:
        try:
            with tarfile.open(tar_path) as tf:
                members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith('.json')]
                for member in members[:max_records_per_tar]:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        data = json.loads(f.read())
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    records_checked += 1
                    for msg in data.get("conversation", []) or []:
                        for frag in msg.get("fragments", []) or []:
                            if not isinstance(frag, dict):
                                continue
                            if frag.get("t") not in ("image", "video", "video_frame", "audio"):
                                continue
                            v = frag.get("value")
                            if not isinstance(v, str):
                                continue
                            full = _resolve_media_path(v, media_root)
                            if not full:
                                continue
                            if not os.path.isfile(full):
                                total_missing += 1
                                if len(warnings) < missing_sample_cap:
                                    warnings.append(
                                        f"[shard] Missing: {full} "
                                        f"(key={v!r}, shard={os.path.basename(tar_path)})"
                                    )
        except Exception as e:
            warnings.append(f"[shard] Error reading {tar_path}: {e}")

    return warnings, records_checked, total_missing


def directory_has_any_file(path: str, max_depth: int = 4) -> bool:
    """True if directory contains at least one regular file (not just subdirectories).
    Walks up to max_depth levels deep; stops as soon as one file is found."""
    try:
        for root, dirs, files in os.walk(path):
            if files:
                return True
            depth = root[len(path):].count(os.sep)
            if depth >= max_depth:
                dirs.clear()
    except OSError:
        pass
    return False


def extract_paths_from_yaml(
    yaml_path: str, base_path: str = None, visited: set = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract both dataset paths and media_source paths from a YAML configuration file.
    Recursively follows references to other YAML files.

    Returns:
        Tuple of (dataset_paths, media_source_paths, warnings, shard_pairs)
        where shard_pairs is a list of (dataset_path, media_source) for shard verification.
    """
    dataset_paths = []
    media_sources = []
    warnings = []
    shard_pairs = []  # (dataset_path, media_source) for --verify-shard-paths

    if visited is None:
        visited = set()

    # Determine base path for relative paths
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(yaml_path))

    # Prevent infinite recursion
    abs_yaml_path = os.path.abspath(yaml_path)
    if abs_yaml_path in visited:
        return dataset_paths, media_sources, warnings, shard_pairs
    visited.add(abs_yaml_path)

    try:
        with open(yaml_path, 'r') as f:
            content = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        # Fallback: try to extract using text parsing
        dp, ms, w = extract_paths_fallback(yaml_path, base_path)
        return dp, ms, w, []

    def find_paths(obj: Any, parent_key: str = ""):
        """Recursively find all path and media_source values in nested structure."""
        if isinstance(obj, dict):
            # Pre-check: if this entry's path points to a YAML, recurse into it
            if 'path' in obj and isinstance(obj['path'], str):
                path = obj['path']
                if not os.path.isabs(path):
                    path = os.path.join(base_path, path)

                if path.endswith(('.yaml', '.yml')):
                    if os.path.abspath(path) not in visited:
                        if os.path.exists(path):
                            sub_dp, sub_ms, sub_warn, sub_pairs = extract_paths_from_yaml(
                                path, visited=visited
                            )
                            dataset_paths.extend(sub_dp)
                            media_sources.extend(sub_ms)
                            warnings.extend(sub_warn)
                            shard_pairs.extend(sub_pairs)
                        else:
                            warnings.append(
                                f"Referenced YAML not found: {path} (from {yaml_path})"
                            )
                    return  # Don't process children of YAML reference entries

                # Collect (dataset_path, media_source) pair for shard verification
                aux = obj.get('aux', {})
                if isinstance(aux, dict) and isinstance(aux.get('media_source'), str):
                    ms = _strip_filesystem_media_prefix(aux['media_source'])
                    shard_pairs.append((path, ms))

                # Check for missing aux/media_source on entries with cook subflavor
                subflavors = obj.get('subflavors', {})
                if isinstance(subflavors, dict) and 'cook' in subflavors:
                    if not (isinstance(aux, dict) and 'media_source' in aux):
                        cook_type = subflavors['cook']
                        warnings.append(
                            f"Dataset with cook='{cook_type}' missing aux/media_source: "
                            f"{obj['path']} (in {yaml_path})"
                        )

            for key, value in obj.items():
                if key == "media_source" and isinstance(value, str):
                    cleaned = _strip_filesystem_media_prefix(value)
                    if cleaned == "/":
                        # Only process filesystem:/// at the entry level (where 'path' is also
                        # present). When find_paths recurses into the aux sub-dict, obj has no
                        # 'path' key — skip to avoid spurious <unknown> warnings; shard_pairs
                        # collection at the entry level already captured this pair.
                        if "path" not in obj:
                            pass
                        else:
                            dataset_path = obj.get("path", "")
                            if not os.path.isabs(dataset_path):
                                dataset_path = os.path.join(base_path, dataset_path)
                            if os.path.isfile(dataset_path):
                                # JSONL: verify it genuinely has no media fragment references
                                if jsonl_has_media_references(dataset_path):
                                    warnings.append(
                                        f"media_source is 'filesystem:///' but JSONL contains media "
                                        f"fragment references — external media path is missing: "
                                        f"{dataset_path} (in {yaml_path})"
                                    )
                                # else: text-only convention is valid — skip silently
                            elif os.path.isdir(dataset_path):
                                # Shard dir with filesystem:/// — covered by --verify-shard-paths
                                pass
                            else:
                                warnings.append(
                                    f"media_source is 'filesystem:///' but dataset path not found: "
                                    f"{dataset_path} (in {yaml_path})"
                                )
                    else:
                        media_sources.append(cleaned)
                elif key == "path" and isinstance(value, str):
                    # Handle dataset paths (may be relative or absolute)
                    path = value
                    if not os.path.isabs(path):
                        # Try to resolve relative path
                        path = os.path.join(base_path, path)
                    dataset_paths.append(path)
                else:
                    find_paths(value, key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_paths(item, parent_key)

    find_paths(content)
    return dataset_paths, media_sources, warnings, shard_pairs


def extract_paths_fallback(yaml_path: str, base_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Fallback method to extract paths using text parsing.
    Used when YAML parsing fails (e.g., for very large files).
    Note: Fallback does not support recursive YAML parsing or missing-aux detection.
    """
    dataset_paths = []
    media_sources = []
    warnings = []

    try:
        with open(yaml_path, 'r') as f:
            for line in f:
                if 'media_source:' in line:
                    parts = line.split('media_source:', 1)
                    if len(parts) > 1:
                        path = parts[1].strip()
                        if path.startswith("filesystem:///"):
                            path = "/" + path[len("filesystem:///"):]
                        elif path.startswith("filesystem://"):
                            path = path[len("filesystem://"):]
                        path = path.strip('"\'')
                        if path:
                            media_sources.append(path)
                # Extract dataset paths (lines with 'path:' but not 'media_source')
                elif '      path:' in line or line.strip().startswith('path:'):
                    parts = line.split('path:', 1)
                    if len(parts) > 1:
                        path = parts[1].strip()
                        path = path.strip('"\'')
                        if path and not path.startswith('filesystem'):
                            if not os.path.isabs(path):
                                path = os.path.join(base_path, path)
                            dataset_paths.append(path)
    except Exception as e:
        print(f"Error reading file: {e}")

    warnings.append(f"Used text fallback for {yaml_path} (recursive parsing unavailable)")
    return dataset_paths, media_sources, warnings


# Keep backward compatibility
def extract_media_sources_from_yaml(yaml_path: str) -> List[str]:
    """
    Extract all aux/media_source paths from a YAML configuration file.
    Backward compatible function.
    """
    _, media_sources, _ = extract_paths_from_yaml(yaml_path)
    return media_sources


def check_media_source(
    media_source_path: str,
    max_files: int = 100,
    sample_size: int = 5,
    path_type: str = "media_source",
    check_tar_idx: bool = False,
    verify_aux_media: str = "off",
) -> MediaSourceReport:
    """
    Check access permissions for a media source folder and its contents.
    Args:
        media_source_path: Path to the media source folder
        max_files: Maximum number of files to check within the folder
        sample_size: Number of problematic files to include in the report
        path_type: Type of path ("media_source" or "dataset_path")
        check_tar_idx: If True, specifically check .tar.idx files
        verify_aux_media: "off" | "nonempty" — for path_type media_source only;
            nonempty walks the directory tree to verify actual files exist (not just empty dirs).
    Returns:
        MediaSourceReport with detailed access information
    """
    report = MediaSourceReport(media_source_path=media_source_path, path_type=path_type)
    report.aux_media_verify_mode = verify_aux_media

    # Check the folder itself
    report.folder_access = check_permissions(media_source_path)

    if not report.folder_access.exists:
        report.errors.append(f"Folder does not exist: {media_source_path}")
        return report

    if not report.folder_access.is_directory:
        # JSONL/JSON dataset files are valid path: entries — just verify read access on the file itself
        if path_type == "dataset_path" and os.path.isfile(media_source_path):
            return report
        report.errors.append(f"Path is not a directory: {media_source_path}")
        return report

    # aux.media_source: check that actual files exist (not just empty dirs)
    if path_type == "media_source" and verify_aux_media == "nonempty":
        report.media_dir_nonempty = directory_has_any_file(media_source_path)
        if not report.media_dir_nonempty:
            report.errors.append(
                "No files found under aux media directory (directories exist but no files — possible failed copy)"
            )

    # Initialize tar.idx report if checking dataset paths
    if check_tar_idx:
        report.tar_idx_report = TarIdxReport()

    # Check files within the folder
    try:
        files_checked = 0
        tar_idx_files = []
        for entry in os.scandir(media_source_path):
            # Always check .tar.idx files if requested
            if check_tar_idx and entry.name.endswith('.tar.idx'):
                tar_idx_files.append(entry)

            if files_checked >= max_files:
                continue  # Still collect tar.idx files

            try:
                file_info = check_permissions(entry.path)
                files_checked += 1
                report.files_checked += 1
                if file_info.has_group_read:
                    report.files_with_group_read += 1
                if file_info.has_other_read:
                    report.files_with_other_read += 1
                if file_info.has_any_read:
                    report.files_with_any_read += 1
                else:
                    report.files_without_read += 1
                    if len(report.sample_files_without_read) < sample_size:
                        report.sample_files_without_read.append(entry.path)

            except Exception as e:
                report.errors.append(f"Error checking {entry.path}: {e}")

        # Check all .tar.idx files specifically
        if check_tar_idx and report.tar_idx_report:
            report.tar_idx_report.total_tar_idx_files = len(tar_idx_files)

            for entry in tar_idx_files:
                try:
                    file_info = check_permissions(entry.path)
                    if file_info.has_any_read:
                        report.tar_idx_report.tar_idx_with_read += 1
                    else:
                        report.tar_idx_report.tar_idx_without_read += 1
                        if len(report.tar_idx_report.sample_tar_idx_without_read) < sample_size:
                            report.tar_idx_report.sample_tar_idx_without_read.append(entry.path)
                except Exception as e:
                    report.errors.append(f"Error checking tar.idx {entry.path}: {e}")
    except PermissionError as e:
        report.errors.append(f"Cannot read directory contents: {e}")
    except Exception as e:
        report.errors.append(f"Error scanning directory: {e}")
    return report


def format_report(
    reports: List[MediaSourceReport],
    tracker: ProgressTracker,
    yaml_path: str,
    config_warnings: List[str] = None,
    verify_aux_media: str = "off",
    shard_warnings: List[str] = None,
    shard_stats: Optional[Dict] = None,
) -> str:
    """Format the complete report as a string."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("MEDIA SOURCE & DATASET PATH ACCESS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"YAML Config: {yaml_path}")
    if verify_aux_media == "nonempty":
        lines.append(
            "AUX media verification: nonempty (checks that actual files exist under aux.media_source)"
        )
    if shard_stats:
        lines.append(
            f"Shard image path verification: {shard_stats.get('shard_dirs_checked', 0)} dirs, "
            f"{shard_stats.get('records_checked', 0)} records sampled, "
            f"{shard_stats.get('missing_total', 0)} missing paths"
        )
    lines.append("")

    # Configuration warnings (missing aux, broken YAML refs, etc.)
    if config_warnings:
        lines.append("-" * 80)
        lines.append(f"CONFIGURATION WARNINGS ({len(config_warnings)})")
        lines.append("-" * 80)
        for w in config_warnings:
            lines.append(f"  {w}")
        lines.append("")

    if shard_warnings:
        lines.append("-" * 80)
        lines.append(f"SHARD IMAGE PATH WARNINGS ({len(shard_warnings)})")
        lines.append("-" * 80)
        for w in shard_warnings:
            lines.append(f"  {w}")
        lines.append("")

    # Summary
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total paths checked: {tracker.total_sources}")
    lines.append(f"  - Dataset paths: {tracker.total_dataset_paths}")
    lines.append(f"  - Media sources: {tracker.total_media_sources}")
    lines.append(f"Paths with read access (group/other): {tracker.sources_accessible}")
    lines.append(f"Paths without read access: {tracker.sources_inaccessible}")
    lines.append(f"Paths with errors: {tracker.sources_with_errors}")
    lines.append(f"Total files checked: {tracker.total_files_checked}")
    lines.append("")
    # .tar.idx Summary
    if tracker.total_tar_idx_files > 0:
        lines.append("-" * 80)
        lines.append(".tar.idx FILES SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total .tar.idx files found: {tracker.total_tar_idx_files}")
        lines.append(f"  - With read access: {tracker.tar_idx_accessible}")
        lines.append(f"  - Without read access: {tracker.tar_idx_inaccessible}")
        if tracker.tar_idx_inaccessible > 0:
            lines.append("  *** WARNING: Some .tar.idx files are NOT readable! ***")
        lines.append("")

    lines.append(f"Time elapsed: {time.time() - tracker.start_time:.2f} seconds")
    lines.append("")

    # Separate reports by type
    dataset_reports = [r for r in reports if r.path_type == "dataset_path"]
    media_reports = [r for r in reports if r.path_type == "media_source"]

    # Consolidated issues summary
    issue_lines = []
    for report in dataset_reports:
        tag = "[DATASET]"
        if report.errors:
            for e in report.errors:
                issue_lines.append(f"  {tag} {report.media_source_path}\n           ERROR: {e}")
        elif report.folder_access and not report.folder_access.has_any_read:
            issue_lines.append(f"  {tag} {report.media_source_path}\n           No group/other read ({report.folder_access.permissions})")
        if report.tar_idx_report and report.tar_idx_report.tar_idx_without_read > 0:
            tr = report.tar_idx_report
            samples = ", ".join(os.path.basename(f) for f in tr.sample_tar_idx_without_read[:3])
            issue_lines.append(f"  {tag} {report.media_source_path}\n           {tr.tar_idx_without_read}/{tr.total_tar_idx_files} .tar.idx inaccessible: {samples}")
    for report in media_reports:
        tag = "[MEDIA]  "
        if report.errors:
            for e in report.errors:
                issue_lines.append(f"  {tag} {report.media_source_path}\n           ERROR: {e}")
        elif report.folder_access and not report.folder_access.has_any_read:
            issue_lines.append(f"  {tag} {report.media_source_path}\n           No group/other read ({report.folder_access.permissions})")
        if report.files_without_read > 0:
            extra = report.files_without_read - len(report.sample_files_without_read)
            suffix = f" ({extra} more not shown)" if extra > 0 else ""
            paths_str = "\n             ".join(report.sample_files_without_read)
            issue_lines.append(
                f"  {tag} {report.media_source_path}\n"
                f"           {report.files_without_read} file(s) without read access{suffix}:\n"
                f"             {paths_str}"
            )
    if shard_warnings:
        for w in shard_warnings:
            issue_lines.append(f"  [SHARD]   {w}")
    if issue_lines:
        lines.append("-" * 80)
        lines.append(f"ISSUES SUMMARY ({len(issue_lines)} issue(s))")
        lines.append("-" * 80)
        for il in issue_lines:
            lines.append(il)
        lines.append("")
    else:
        lines.append("All paths accessible — no issues found.")
        lines.append("")

    # Categorize reports
    def categorize(report_list):
        accessible = []
        inaccessible = []
        errors = []
        tar_idx_issues = []
        for report in report_list:
            if report.errors:
                errors.append(report)
            elif report.folder_access and report.folder_access.exists and report.folder_access.has_any_read:
                accessible.append(report)
                # Check for tar.idx issues
                if report.tar_idx_report and report.tar_idx_report.tar_idx_without_read > 0:
                    tar_idx_issues.append(report)
            else:
                inaccessible.append(report)

        return accessible, inaccessible, errors, tar_idx_issues

    # Dataset paths section
    if dataset_reports:
        lines.append("=" * 80)
        lines.append("DATASET PATHS (containing .tar.idx files)")
        lines.append("=" * 80)

        accessible, inaccessible, errors, tar_idx_issues = categorize(dataset_reports)

        # Show tar.idx issues first (critical)
        if tar_idx_issues:
            lines.append("")
            lines.append("-" * 80)
            lines.append("!!! DATASET PATHS WITH INACCESSIBLE .tar.idx FILES !!!")
            lines.append("-" * 80)
            for report in tar_idx_issues:
                lines.append(f"\n  Path: {report.media_source_path}")
                lines.append(f"  Folder permissions: {report.folder_access.permissions if report.folder_access else 'N/A'}")
                tr = report.tar_idx_report
                lines.append(f"  .tar.idx files: {tr.total_tar_idx_files} total, {tr.tar_idx_without_read} WITHOUT read access")
                for f in tr.sample_tar_idx_without_read[:5]:
                    lines.append(f"    - {os.path.basename(f)}")
        # Inaccessible dataset paths
        if inaccessible:
            lines.append("")
            lines.append("-" * 80)
            lines.append("INACCESSIBLE DATASET PATHS")
            lines.append("-" * 80)
            for report in inaccessible:
                lines.append(f"\n  Path: {report.media_source_path}")
                if report.folder_access:
                    lines.append(f"  Permissions: {report.folder_access.permissions}")
                    lines.append(f"  Group read: {report.folder_access.has_group_read}")
                    lines.append(f"  Other read: {report.folder_access.has_other_read}")
        # Errors
        if errors:
            lines.append("")
            lines.append("-" * 80)
            lines.append("DATASET PATHS WITH ERRORS")
            lines.append("-" * 80)
            for report in errors:
                lines.append(f"\n  Path: {report.media_source_path}")
                for error in report.errors:
                    lines.append(f"    ERROR: {error}")
        # Accessible dataset paths
        if accessible:
            lines.append("")
            lines.append("-" * 80)
            lines.append("ACCESSIBLE DATASET PATHS")
            lines.append("-" * 80)
            for report in accessible:
                perm_str = report.folder_access.permissions if report.folder_access else "N/A"
                tr = report.tar_idx_report
                tar_idx_str = ""
                if tr and tr.total_tar_idx_files > 0:
                    tar_idx_str = f" | .tar.idx: {tr.tar_idx_with_read}/{tr.total_tar_idx_files}"
                lines.append(f"  [OK] {report.media_source_path}")
                lines.append(f"       Permissions: {perm_str}{tar_idx_str}")

        lines.append("")

    # Media sources section
    if media_reports:
        lines.append("=" * 80)
        lines.append("MEDIA SOURCES (image/video directories)")
        lines.append("=" * 80)

        accessible, inaccessible, errors, _ = categorize(media_reports)

        # Inaccessible media sources
        if inaccessible:
            lines.append("")
            lines.append("-" * 80)
            lines.append("INACCESSIBLE MEDIA SOURCES")
            lines.append("-" * 80)
            for report in inaccessible:
                lines.append(f"\n  Path: {report.media_source_path}")
                if report.folder_access:
                    lines.append(f"  Permissions: {report.folder_access.permissions}")
        # Errors
        if errors:
            lines.append("")
            lines.append("-" * 80)
            lines.append("MEDIA SOURCES WITH ERRORS")
            lines.append("-" * 80)
            for report in errors:
                lines.append(f"\n  Path: {report.media_source_path}")
                for error in report.errors:
                    lines.append(f"    ERROR: {error}")
        # Accessible media sources
        if accessible:
            lines.append("")
            lines.append("-" * 80)
            lines.append("ACCESSIBLE MEDIA SOURCES")
            lines.append("-" * 80)
            for report in accessible:
                perm_str = report.folder_access.permissions if report.folder_access else "N/A"
                files_ok = f"{report.files_with_any_read}/{report.files_checked}"
                lines.append(f"  [OK] {report.media_source_path}")
                lines.append(f"       Permissions: {perm_str} | Files with read: {files_ok}")
                if report.files_without_read > 0:
                    extra = report.files_without_read - len(report.sample_files_without_read)
                    suffix = f" ({extra} more not shown)" if extra > 0 else ""
                    lines.append(f"       WARNING: {report.files_without_read} files without read access{suffix}:")
                    for p in report.sample_files_without_read:
                        lines.append(f"         {p}")
                if report.aux_media_verify_mode != "off":
                    bits = []
                    if report.media_dir_nonempty is not None:
                        bits.append(f"nonempty={report.media_dir_nonempty}")
                    if bits:
                        lines.append(
                            f"       aux verify ({report.aux_media_verify_mode}): "
                            + ", ".join(bits)
                        )

        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_json_report(
    reports: List[MediaSourceReport],
    tracker: ProgressTracker,
    yaml_path: str,
    output_path: str,
    verify_aux_media: str = "off",
):
    """Save the report in JSON format for programmatic access."""
    json_output_path = output_path.rsplit('.', 1)[0] + '.json'
    data = {
        "generated_at": datetime.now().isoformat(),
        "yaml_config": yaml_path,
        "verify_aux_media": verify_aux_media,
        "summary": {
            "total_sources": tracker.total_sources,
            "total_dataset_paths": tracker.total_dataset_paths,
            "total_media_sources": tracker.total_media_sources,
            "accessible": tracker.sources_accessible,
            "inaccessible": tracker.sources_inaccessible,
            "errors": tracker.sources_with_errors,
            "total_files_checked": tracker.total_files_checked,
            "tar_idx": {
                "total": tracker.total_tar_idx_files,
                "accessible": tracker.tar_idx_accessible,
                "inaccessible": tracker.tar_idx_inaccessible
            },
            "elapsed_seconds": time.time() - tracker.start_time
        },
        "dataset_paths": [],
        "media_sources": []
    }
    for report in reports:
        report_dict = {
            "path": report.media_source_path,
            "folder_exists": report.folder_access.exists if report.folder_access else False,
            "folder_permissions": report.folder_access.permissions if report.folder_access else None,
            "folder_group_read": report.folder_access.has_group_read if report.folder_access else False,
            "folder_other_read": report.folder_access.has_other_read if report.folder_access else False,
            "files_checked": report.files_checked,
            "files_with_group_read": report.files_with_group_read,
            "files_with_other_read": report.files_with_other_read,
            "files_without_read": report.files_without_read,
            "errors": report.errors
        }
        # Add tar.idx info for dataset paths
        if report.tar_idx_report:
            report_dict["tar_idx"] = {
                "total": report.tar_idx_report.total_tar_idx_files,
                "with_read": report.tar_idx_report.tar_idx_with_read,
                "without_read": report.tar_idx_report.tar_idx_without_read,
                "sample_without_read": report.tar_idx_report.sample_tar_idx_without_read
            }
        if report.path_type == "media_source":
            report_dict["aux_media_verify_mode"] = report.aux_media_verify_mode
            report_dict["media_dir_nonempty"] = report.media_dir_nonempty
        if report.path_type == "dataset_path":
            data["dataset_paths"].append(report_dict)
        else:
            data["media_sources"].append(report_dict)

    with open(json_output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"JSON report saved to: {json_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check read access permissions for dataset paths and media_source paths in a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_media_access.py config.yaml
  python check_media_access.py config.yaml --output report.txt
  python check_media_access.py config.yaml --max-files 50 --verbose
  python check_media_access.py config.yaml --dataset-only   # Only check dataset paths
  python check_media_access.py config.yaml --media-only     # Only check media sources
  python check_media_access.py config.yaml --verify-aux-media nonempty
  python check_media_access.py config.yaml --verify-shard-paths
  python check_media_access.py config.yaml --verify-aux-media nonempty --verify-shard-paths
        """
    )
    parser.add_argument(
        "yaml_file",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for the report (default: media_access_report_<timestamp>.txt)"
    )
    parser.add_argument(
        "--max-files", "-m",
        type=int,
        default=100,
        help="Maximum number of files to check per source (default: 100)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose progress output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also save report in JSON format"
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only check dataset paths (containing .tar.idx files)"
    )
    parser.add_argument(
        "--media-only",
        action="store_true",
        help="Only check media source paths"
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Base path for resolving relative dataset paths (default: YAML file directory)"
    )
    parser.add_argument(
        "--verify-aux-media",
        choices=["off", "nonempty"],
        default="off",
        help=(
            "For aux.media_source: 'nonempty' walks up to 4 directory levels "
            "deep and fails if no actual files are found (catches failed copies that leave only "
            "empty directory trees). Default: off."
        ),
    )
    parser.add_argument(
        "--verify-shard-paths",
        action="store_true",
        help=(
            "For WebDataset shard directories: sample tar files and verify that image paths "
            "referenced in the JSON records exist on disk under aux.media_source. "
            "Slower but confirms training will actually find the images."
        ),
    )
    parser.add_argument(
        "--shard-sample-tars",
        type=int,
        default=3,
        help="Number of tar files to sample per shard directory (default: 3).",
    )
    parser.add_argument(
        "--shard-sample-records",
        type=int,
        default=16,
        help="Number of JSON records to read per tar file (default: 16).",
    )
    parser.add_argument(
        "--skip-permissions",
        action="store_true",
        help=(
            "Skip all permission checks (dataset paths and media sources). "
            "Use with --verify-shard-paths to run only shard content verification."
        ),
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.yaml_file):
        print(f"Error: YAML file not found: {args.yaml_file}")
        sys.exit(1)
    # Set default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"media_access_report_{timestamp}.txt"

    print(f"Reading YAML configuration: {args.yaml_file}")

    # Extract both dataset paths and media sources (recursively follows sub-YAMLs)
    dataset_paths, media_sources, config_warnings, shard_pairs = extract_paths_from_yaml(
        args.yaml_file, args.base_path
    )

    # Remove duplicates while preserving order
    def deduplicate(paths):
        seen = set()
        unique = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    dataset_paths = deduplicate(dataset_paths)
    media_sources = deduplicate(media_sources)

    # Apply filters
    if args.dataset_only:
        media_sources = []
    if args.media_only:
        dataset_paths = []

    print(f"Found {len(dataset_paths)} unique dataset paths (with .tar.idx files)")
    print(f"Found {len(media_sources)} unique media source paths")
    if args.verify_aux_media == "nonempty":
        print("Aux media verification: nonempty (checking for actual files, not just directories)")

    # Show configuration warnings (missing aux, broken YAML refs, etc.)
    if config_warnings:
        print(f"\n{'=' * 60}")
        print(f"CONFIGURATION WARNINGS ({len(config_warnings)})")
        print(f"{'=' * 60}")
        for w in config_warnings:
            print(f"  {w}")
        print()

    total_paths = len(dataset_paths) + len(media_sources)
    if total_paths == 0 and not args.skip_permissions:
        print("No paths found in the YAML file.")
        sys.exit(0)

    # Initialize tracker
    tracker = ProgressTracker(
        total_sources=0 if args.skip_permissions else total_paths,
        total_dataset_paths=0 if args.skip_permissions else len(dataset_paths),
        total_media_sources=0 if args.skip_permissions else len(media_sources),
    )

    # Process all paths
    reports = []

    if args.skip_permissions:
        print("\nSkipping permission checks (--skip-permissions).")
    else:
        print("\nChecking access permissions...")
        print("-" * 60)

        # Process dataset paths (check .tar.idx files)
        for i, path in enumerate(dataset_paths, 1):
            if args.verbose:
                print(f"\n[Dataset {i}/{len(dataset_paths)}] Checking: {path}")

            report = check_media_source(
                path,
                max_files=args.max_files,
                path_type="dataset_path",
                check_tar_idx=True
            )
            reports.append(report)
            tracker.update(report)
            # Print progress
            if args.verbose:
                status = "OK" if (report.folder_access and report.folder_access.has_any_read) else "FAIL"
                print(f"  Status: {status}")
                if report.tar_idx_report:
                    tr = report.tar_idx_report
                    print(f"  .tar.idx files: {tr.total_tar_idx_files} total, {tr.tar_idx_with_read} readable")
                if report.errors:
                    for err in report.errors:
                        print(f"  Error: {err}")
            else:
                print(f"\r{tracker.get_progress_string()}", end="", flush=True)

        # Process media sources
        for i, source in enumerate(media_sources, 1):
            if args.verbose:
                print(f"\n[Media {i}/{len(media_sources)}] Checking: {source}")

            report = check_media_source(
                source,
                max_files=args.max_files,
                path_type="media_source",
                check_tar_idx=False,
                verify_aux_media=args.verify_aux_media,
            )
            reports.append(report)
            tracker.update(report)
            # Print progress
            if args.verbose:
                status = "OK" if (report.folder_access and report.folder_access.has_any_read) else "FAIL"
                print(f"  Status: {status}")
                if report.errors:
                    for err in report.errors:
                        print(f"  Error: {err}")
            else:
                print(f"\r{tracker.get_progress_string()}", end="", flush=True)

        print("\n")

    # Shard image path verification
    shard_warnings: List[str] = []
    shard_stats: Dict = {}
    if args.verify_shard_paths:
        # Pre-count unique shard dirs for progress display
        unique_shard_dirs = list({
            (p, m) for p, m in shard_pairs if os.path.isdir(p)
        })
        print(f"Verifying shard image paths ({len(unique_shard_dirs)} unique shard dirs)...")
        seen_pairs: set = set()
        total_records = 0
        total_missing = 0
        dirs_checked = 0
        shard_start = time.time()
        for ds_path, media_root in shard_pairs:
            if not os.path.isdir(ds_path):
                continue  # only shard directories
            pair = (ds_path, media_root)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if args.verbose:
                print(f"  [{len(seen_pairs)}/{len(unique_shard_dirs)}] {ds_path}")
            else:
                elapsed = time.time() - shard_start
                print(
                    f"\r  [{len(seen_pairs)}/{len(unique_shard_dirs)}] "
                    f"records={total_records} missing={total_missing} "
                    f"elapsed={elapsed:.0f}s",
                    end="", flush=True,
                )
            w, recs, miss = verify_shard_image_paths(
                ds_path, media_root,
                max_tars=args.shard_sample_tars,
                max_records_per_tar=args.shard_sample_records,
            )
            shard_warnings.extend(w)
            total_records += recs
            total_missing += miss
            if recs > 0:
                dirs_checked += 1
        print()
        shard_stats = {
            "shard_dirs_checked": dirs_checked,
            "records_checked": total_records,
            "missing_total": total_missing,
        }
        if shard_warnings:
            print(f"  Shard verification: {total_missing} missing image path(s) across {dirs_checked} dirs")
            for w in shard_warnings[:10]:
                print(f"    {w}")
            if len(shard_warnings) > 10:
                print(f"    ... and {len(shard_warnings) - 10} more (see report file)")
        else:
            print(f"  Shard verification: all sampled image paths exist ({dirs_checked} dirs, {total_records} records)")

    # Generate and save report
    report_text = format_report(
        reports,
        tracker,
        args.yaml_file,
        config_warnings,
        verify_aux_media=args.verify_aux_media,
        shard_warnings=shard_warnings,
        shard_stats=shard_stats if args.verify_shard_paths else None,
    )

    with open(args.output, 'w') as f:
        f.write(report_text)

    print(f"Report saved to: {args.output}")

    # Save JSON report if requested
    if args.json:
        save_json_report(
            reports,
            tracker,
            args.yaml_file,
            args.output,
            verify_aux_media=args.verify_aux_media,
        )

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total paths checked: {tracker.total_sources}")
    print(f"  - Dataset paths: {tracker.total_dataset_paths}")
    print(f"  - Media sources: {tracker.total_media_sources}")
    print(f"Accessible (group/other read): {tracker.sources_accessible}")
    print(f"Inaccessible: {tracker.sources_inaccessible}")
    print(f"Errors: {tracker.sources_with_errors}")
    print(f"Total files checked: {tracker.total_files_checked}")
    if tracker.total_tar_idx_files > 0:
        print(f"\n.tar.idx Files:")
        print(f"  Total found: {tracker.total_tar_idx_files}")
        print(f"  Readable: {tracker.tar_idx_accessible}")
        print(f"  NOT readable: {tracker.tar_idx_inaccessible}")
    if args.verify_aux_media == "nonempty":
        media_reports = [r for r in reports if r.path_type == "media_source"]
        aux_fail = sum(1 for r in media_reports if r.errors)
        print(f"\nAux nonempty verification: {aux_fail} media path(s) with issues")
    # Print paths with inaccessible tar.idx files
    if tracker.tar_idx_inaccessible > 0:
        print("\n" + "=" * 60)
        print("PATHS WITH INACCESSIBLE .tar.idx FILES:")
        print("=" * 60)
        for report in reports:
            if report.tar_idx_report and report.tar_idx_report.tar_idx_without_read > 0:
                print(report.media_source_path)

    # Show config warnings in summary
    if config_warnings:
        print(f"\nConfiguration warnings: {len(config_warnings)}")
        for w in config_warnings:
            print(f"  {w}")

    # Exit with non-zero if there are issues
    has_issues = (
        tracker.sources_inaccessible > 0 or
        tracker.sources_with_errors > 0 or
        tracker.tar_idx_inaccessible > 0 or
        bool(shard_warnings)
    )
    has_config_warnings = bool(config_warnings)

    if has_issues or has_config_warnings:
        if has_issues:
            print("\n*** WARNING: Some paths or .tar.idx files are not accessible! ***")
        if has_config_warnings:
            print("\n*** WARNING: Configuration issues detected (missing aux/media_source, broken YAML refs)! ***")
        sys.exit(1)

    if args.verify_aux_media != "off":
        print("\nAll checks passed (permissions and aux media verification).")
    else:
        print("\nAll paths and .tar.idx files are accessible!")
    sys.exit(0)


if __name__ == "__main__":
    main()
