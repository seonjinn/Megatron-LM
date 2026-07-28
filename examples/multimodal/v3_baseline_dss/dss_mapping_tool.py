#!/usr/bin/env python3
"""Inspect legacy-to-DSS mappings and audit legacy recipe include graphs.

This tool is deliberately read-only with respect to recipe YAMLs. Mapping a
filesystem path can be ambiguous, so it reports candidates instead of
rewriting a recipe automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO
from urllib.parse import urlparse

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING = SCRIPT_DIR / "legacy_to_dss_mapping.csv"
DEFAULT_NORMALIZED_MAPPING = SCRIPT_DIR / "legacy_to_dss_mapping_normalized.csv"
DEFAULT_ALIASES = SCRIPT_DIR / "path_prefix_aliases.csv"

RAW_FIELDS = ["recipe", "kind", "action", "source_path", "dss_reference", "upload_root"]
ALIAS_FIELDS = ["source_prefix", "canonical_prefix", "scope", "confidence", "notes"]
NORMALIZED_FIELDS = [
    "source_row",
    "recipe",
    "role",
    "action",
    "legacy_path",
    "canonical_legacy_path",
    "mapping_status",
    "reference_label",
    "dss_reference",
    "dataset_name",
    "snapshot_name",
    "dss_subpath",
    "upload_root",
]
AUDIT_FIELDS = [
    "source_yaml",
    "field",
    "role",
    "legacy_path",
    "match_status",
    "candidate_count",
    "candidate_dss_references",
    "candidate_actions",
    "candidate_source_rows",
    "notes",
]

DSS_REFERENCE_RE = re.compile(r"(?:(?P<label>[A-Za-z0-9_.-]+)=)?(?P<reference>dss://[^;\s]+)")
DSS_SCHEMES = ("dss://", "filesystem+dss://")
YAML_SUFFIXES = {".yaml", ".yml"}


@dataclass(frozen=True)
class MappingEntry:
    """One normalized legacy-path-to-DSS candidate."""

    source_row: int
    recipe: str
    role: str
    action: str
    legacy_path: str
    canonical_legacy_path: str
    mapping_status: str
    reference_label: str
    dss_reference: str
    dataset_name: str
    snapshot_name: str
    dss_subpath: str
    upload_root: str


@dataclass(frozen=True)
class LookupResult:
    """Candidate mappings and the strength of their path match."""

    match_type: str
    query_path: str
    canonical_query_path: str
    candidates: tuple[MappingEntry, ...]


@dataclass(frozen=True)
class AuditRow:
    """One path or include discovered while traversing a recipe."""

    source_yaml: str
    field: str
    role: str
    legacy_path: str
    match_status: str
    candidate_count: int
    candidate_dss_references: str
    candidate_actions: str
    candidate_source_rows: str
    notes: str


def parse_dss_reference(reference: str) -> tuple[str, str, str]:
    """Split a DSS URI into dataset name, snapshot name, and subpath."""

    if not reference.startswith("dss://"):
        raise ValueError(f"not a DSS reference: {reference}")
    dataset_and_version, separator, subpath = reference.removeprefix("dss://").partition("/")
    if "@" not in dataset_and_version:
        raise ValueError(f"DSS reference has no snapshot name: {reference}")
    dataset_name, snapshot_name = dataset_and_version.rsplit("@", 1)
    if not dataset_name or not snapshot_name:
        raise ValueError(f"invalid DSS reference: {reference}")
    return dataset_name, snapshot_name, subpath if separator else ""


def normalize_legacy_path(value: str) -> str:
    """Normalize a filesystem URI or path without guessing Lustre aliases."""

    value = value.strip()
    if value.startswith("filesystem://"):
        value = urlparse(value).path
    if value != "/":
        value = value.rstrip("/")
    return value


def load_aliases(path: Path) -> list[tuple[str, str]]:
    """Load declared path-prefix aliases in longest-prefix-first order."""

    aliases: list[tuple[str, str]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ALIAS_FIELDS:
            raise ValueError(
                f"{path}: expected CSV fields {ALIAS_FIELDS}, found {reader.fieldnames}"
            )
        for source_row, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"{path}:{source_row}: extra CSV columns")
            source_prefix = normalize_legacy_path(raw["source_prefix"])
            canonical_prefix = normalize_legacy_path(raw["canonical_prefix"])
            if not source_prefix.startswith("/") or not canonical_prefix.startswith("/"):
                raise ValueError(f"{path}:{source_row}: aliases must be absolute paths")
            aliases.append((source_prefix, canonical_prefix))
    return sorted(aliases, key=lambda alias: len(alias[0]), reverse=True)


def canonicalize_legacy_path(value: str, aliases: Sequence[tuple[str, str]]) -> str:
    """Apply the first declared path-prefix alias to a normalized path."""

    value = normalize_legacy_path(value)
    for source_prefix, canonical_prefix in aliases:
        if value == source_prefix:
            return canonical_prefix
        if value.startswith(f"{source_prefix}/"):
            return f"{canonical_prefix}{value[len(source_prefix):]}"
    return value


def load_mapping(path: Path, aliases: Sequence[tuple[str, str]]) -> list[MappingEntry]:
    """Load and normalize the historical mapping registry."""

    entries: list[MappingEntry] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RAW_FIELDS:
            raise ValueError(f"{path}: expected CSV fields {RAW_FIELDS}, found {reader.fieldnames}")
        for source_row, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"{path}:{source_row}: extra CSV columns")
            legacy_path = normalize_legacy_path(raw["source_path"])
            canonical_legacy_path = canonicalize_legacy_path(legacy_path, aliases)
            matches = list(DSS_REFERENCE_RE.finditer(raw["dss_reference"]))
            if not matches:
                entries.append(
                    MappingEntry(
                        source_row=source_row,
                        recipe=raw["recipe"],
                        role=raw["kind"],
                        action=raw["action"],
                        legacy_path=legacy_path,
                        canonical_legacy_path=canonical_legacy_path,
                        mapping_status="removed",
                        reference_label="",
                        dss_reference="",
                        dataset_name="",
                        snapshot_name="",
                        dss_subpath="",
                        upload_root=raw["upload_root"],
                    )
                )
                continue
            for match in matches:
                reference = match.group("reference")
                dataset_name, snapshot_name, subpath = parse_dss_reference(reference)
                entries.append(
                    MappingEntry(
                        source_row=source_row,
                        recipe=raw["recipe"],
                        role=raw["kind"],
                        action=raw["action"],
                        legacy_path=legacy_path,
                        canonical_legacy_path=canonical_legacy_path,
                        mapping_status="mapped",
                        reference_label=match.group("label") or "",
                        dss_reference=reference,
                        dataset_name=dataset_name,
                        snapshot_name=snapshot_name,
                        dss_subpath=subpath,
                        upload_root=raw["upload_root"],
                    )
                )
    return entries


def filter_entries(
    entries: Iterable[MappingEntry], recipe_key: str | None, role: str | None
) -> list[MappingEntry]:
    """Filter candidates by historical recipe key and role."""

    return [
        entry
        for entry in entries
        if entry.mapping_status == "mapped"
        and (recipe_key is None or entry.recipe == recipe_key)
        and (role is None or entry.role == role)
    ]


def deduplicate_entries(entries: Iterable[MappingEntry]) -> tuple[MappingEntry, ...]:
    """Deduplicate identical candidates while preserving CSV order."""

    result: list[MappingEntry] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for entry in entries:
        key = (
            entry.recipe,
            entry.role,
            entry.canonical_legacy_path,
            entry.dss_reference,
            entry.action,
        )
        if key not in seen:
            result.append(entry)
            seen.add(key)
    return tuple(result)


def lookup_mapping(
    entries: Sequence[MappingEntry],
    legacy_path: str,
    aliases: Sequence[tuple[str, str]],
    recipe_key: str | None = None,
    role: str | None = None,
) -> LookupResult:
    """Find exact mappings, or report longest-prefix ancestor candidates."""

    normalized_path = normalize_legacy_path(legacy_path)
    canonical_path = canonicalize_legacy_path(normalized_path, aliases)
    filtered = filter_entries(entries, recipe_key, role)
    exact = [entry for entry in filtered if entry.canonical_legacy_path == canonical_path]
    if exact:
        return LookupResult("exact", normalized_path, canonical_path, deduplicate_entries(exact))

    ancestor_candidates = [
        entry
        for entry in filtered
        if entry.canonical_legacy_path not in {"", "/"}
        and canonical_path.startswith(f"{entry.canonical_legacy_path}/")
    ]
    if not ancestor_candidates:
        return LookupResult("none", normalized_path, canonical_path, ())
    longest_prefix = max(len(entry.canonical_legacy_path) for entry in ancestor_candidates)
    longest = [
        entry for entry in ancestor_candidates if len(entry.canonical_legacy_path) == longest_prefix
    ]
    return LookupResult("ancestor", normalized_path, canonical_path, deduplicate_entries(longest))


def write_rows(rows: Iterable[dict[str, Any]], fields: Sequence[str], stream: TextIO) -> None:
    """Write dictionaries as CSV with a fixed schema."""

    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def command_normalize(args: argparse.Namespace) -> int:
    """Generate the one-candidate-per-row normalized mapping artifact."""

    entries = load_mapping(args.mapping, load_aliases(args.aliases))
    output_rows = [asdict(entry) for entry in entries]
    if args.check:
        if not args.output.exists():
            print(f"missing normalized mapping: {args.output}", file=sys.stderr)
            return 2
        with args.output.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
        if existing != [
            {field: str(row[field]) for field in NORMALIZED_FIELDS} for row in output_rows
        ]:
            print(
                f"{args.output} is stale; regenerate it with the normalize command", file=sys.stderr
            )
            return 2
        print(f"{args.output}: up to date ({len(entries)} normalized rows)")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        write_rows(output_rows, NORMALIZED_FIELDS, stream)
    print(f"wrote {len(entries)} normalized rows to {args.output}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Validate mapping syntax and summarize ambiguity."""

    entries = load_mapping(args.mapping, load_aliases(args.aliases))
    mapped = [entry for entry in entries if entry.mapping_status == "mapped"]
    removed = [entry for entry in entries if entry.mapping_status == "removed"]
    by_path: dict[tuple[str, str, str], set[str]] = {}
    for entry in mapped:
        key = (entry.recipe, entry.role, entry.canonical_legacy_path)
        by_path.setdefault(key, set()).add(entry.dss_reference)
    ambiguous = {key: refs for key, refs in by_path.items() if len(refs) > 1}
    summary = {
        "normalized_rows": len(entries),
        "mapped_rows": len(mapped),
        "removed_rows": len(removed),
        "recipe_keys": sorted({entry.recipe for entry in entries}),
        "ambiguous_recipe_role_path_keys": len(ambiguous),
    }
    print(json.dumps(summary, indent=2))
    return 0


def mapping_entry_dict(entry: MappingEntry) -> dict[str, Any]:
    """Return concise lookup output for one mapping candidate."""

    return {
        "match_type": "",
        "source_row": entry.source_row,
        "recipe": entry.recipe,
        "role": entry.role,
        "action": entry.action,
        "legacy_path": entry.legacy_path,
        "canonical_legacy_path": entry.canonical_legacy_path,
        "dss_reference": entry.dss_reference,
        "reference_label": entry.reference_label,
    }


def command_lookup(args: argparse.Namespace) -> int:
    """Look up one legacy path and print all candidates."""

    aliases = load_aliases(args.aliases)
    result = lookup_mapping(
        load_mapping(args.mapping, aliases), args.legacy_path, aliases, args.recipe_key, args.role
    )
    rows = [mapping_entry_dict(entry) for entry in result.candidates]
    for row in rows:
        row["match_type"] = result.match_type
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        fields = [
            "match_type",
            "source_row",
            "recipe",
            "role",
            "action",
            "legacy_path",
            "canonical_legacy_path",
            "dss_reference",
            "reference_label",
        ]
        write_rows(rows, fields, sys.stdout)
    return 0 if rows else 2


class RecipeAuditor:
    """Recursively inventory a YAML recipe and report mapping candidates."""

    def __init__(
        self,
        entries: Sequence[MappingEntry],
        aliases: Sequence[tuple[str, str]],
        recipe_key: str,
        include_search_roots: Sequence[Path],
    ) -> None:
        """Initialize a recipe auditor."""

        self.entries = entries
        self.aliases = aliases
        self.recipe_key = recipe_key
        self.include_search_roots = [root.resolve() for root in include_search_roots]
        self.rows: list[AuditRow] = []
        self.visited: set[Path] = set()
        self.active_stack: set[Path] = set()

    def audit(self, recipe: Path) -> list[AuditRow]:
        """Audit a recipe and all resolvable YAML includes."""

        self._audit_yaml(recipe.resolve())
        return self.rows

    def _audit_yaml(self, path: Path) -> None:
        if path in self.active_stack:
            self.rows.append(
                AuditRow(
                    source_yaml=str(path),
                    field="path",
                    role="include",
                    legacy_path=str(path),
                    match_status="include_cycle",
                    candidate_count=0,
                    candidate_dss_references="",
                    candidate_actions="",
                    candidate_source_rows="",
                    notes="include cycle detected",
                )
            )
            return
        if path in self.visited:
            return
        self.visited.add(path)
        self.active_stack.add(path)
        try:
            with path.open() as stream:
                document = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            self.rows.append(
                AuditRow(
                    source_yaml=str(path),
                    field="path",
                    role="include",
                    legacy_path=str(path),
                    match_status="unresolved_include",
                    candidate_count=0,
                    candidate_dss_references="",
                    candidate_actions="",
                    candidate_source_rows="",
                    notes=str(error),
                )
            )
            self.active_stack.remove(path)
            return
        self._walk(document, path)
        self.active_stack.remove(path)

    def _walk(self, node: Any, source_yaml: Path) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "path" and isinstance(value, str):
                    self._handle_path(value, source_yaml)
                elif key == "aux" and isinstance(value, dict):
                    self._handle_aux(value, source_yaml, "aux")
                else:
                    self._walk(value, source_yaml)
        elif isinstance(node, list):
            for value in node:
                self._walk(value, source_yaml)

    def _handle_aux(self, node: dict[str, Any], source_yaml: Path, prefix: str) -> None:
        for key, value in node.items():
            field = f"{prefix}.{key}"
            if isinstance(value, str) and self._looks_like_path(value):
                self._record_dataset(value, source_yaml, field, "auxiliary")
            elif isinstance(value, dict):
                self._handle_aux(value, source_yaml, field)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, str) and self._looks_like_path(item):
                        self._record_dataset(item, source_yaml, f"{field}[{index}]", "auxiliary")

    def _handle_path(self, value: str, source_yaml: Path) -> None:
        if self._is_yaml_path(value):
            include_path, note = self._resolve_include(value, source_yaml.parent)
            status = "include_resolved" if include_path else "unresolved_include"
            self.rows.append(
                AuditRow(
                    source_yaml=str(source_yaml),
                    field="path",
                    role="include",
                    legacy_path=value,
                    match_status=status,
                    candidate_count=0,
                    candidate_dss_references="",
                    candidate_actions="",
                    candidate_source_rows="",
                    notes=note,
                )
            )
            if include_path:
                self._audit_yaml(include_path)
            return
        self._record_dataset(value, source_yaml, "path", "primary")

    def _record_dataset(self, value: str, source_yaml: Path, field: str, role: str) -> None:
        if value.startswith(DSS_SCHEMES):
            self.rows.append(
                AuditRow(
                    source_yaml=str(source_yaml),
                    field=field,
                    role=role,
                    legacy_path=value,
                    match_status="already_dss",
                    candidate_count=1,
                    candidate_dss_references=value,
                    candidate_actions="",
                    candidate_source_rows="",
                    notes="",
                )
            )
            return
        legacy_path = normalize_legacy_path(value)
        if legacy_path and not legacy_path.startswith("/"):
            legacy_path = str((source_yaml.parent / legacy_path).resolve())
        result = lookup_mapping(
            self.entries, legacy_path, self.aliases, recipe_key=self.recipe_key, role=role
        )
        unique_references = sorted({candidate.dss_reference for candidate in result.candidates})
        if result.match_type == "none":
            status = "unmapped"
        elif result.match_type == "ancestor":
            status = "ancestor_candidate"
        elif len(unique_references) == 1:
            status = "mapped_exact"
        else:
            status = "ambiguous_exact"
        self.rows.append(
            AuditRow(
                source_yaml=str(source_yaml),
                field=field,
                role=role,
                legacy_path=legacy_path,
                match_status=status,
                candidate_count=len(unique_references),
                candidate_dss_references=";".join(unique_references),
                candidate_actions=";".join(
                    sorted({candidate.action for candidate in result.candidates})
                ),
                candidate_source_rows=";".join(
                    str(candidate.source_row) for candidate in result.candidates
                ),
                notes=self._mapping_notes(result, status),
            )
        )

    @staticmethod
    def _mapping_notes(result: LookupResult, status: str) -> str:
        notes: list[str] = []
        if result.query_path != result.canonical_query_path:
            notes.append(f"path alias applied: {result.canonical_query_path}")
        if status == "ancestor_candidate":
            notes.append("review ancestor mapping; relative DSS subpath is not inferred")
        return "; ".join(notes)

    def _resolve_include(self, value: str, source_directory: Path) -> tuple[Path | None, str]:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = source_directory / candidate
        if candidate.is_file():
            return candidate.resolve(), "resolved directly"
        matches: list[Path] = []
        for root in self.include_search_roots:
            matches.extend(path.resolve() for path in root.rglob(Path(value).name))
        unique_matches = sorted(set(matches))
        if len(unique_matches) == 1:
            return unique_matches[0], f"resolved by basename under search root: {unique_matches[0]}"
        if not unique_matches:
            return None, "include does not exist; copy it locally or add --include-search-root"
        return None, f"include basename is ambiguous across {len(unique_matches)} files"

    @staticmethod
    def _is_yaml_path(value: str) -> bool:
        return Path(value).suffix.lower() in YAML_SUFFIXES

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return value.startswith(("/", "filesystem://", *DSS_SCHEMES))


def command_audit(args: argparse.Namespace) -> int:
    """Audit a YAML graph and write a reviewable CSV report."""

    aliases = load_aliases(args.aliases)
    auditor = RecipeAuditor(
        load_mapping(args.mapping, aliases), aliases, args.recipe_key, args.include_search_root
    )
    rows = auditor.audit(args.recipe)
    output_rows = [asdict(row) for row in rows]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            write_rows(output_rows, AUDIT_FIELDS, stream)
        print(f"wrote {len(rows)} audit rows to {args.output}")
    else:
        write_rows(output_rows, AUDIT_FIELDS, sys.stdout)
    blockers = {
        "unmapped",
        "ambiguous_exact",
        "ancestor_candidate",
        "unresolved_include",
        "include_cycle",
    }
    blocker_count = sum(row.match_status in blockers for row in rows)
    if blocker_count:
        print(f"audit requires review: {blocker_count} blocking rows", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help=f"raw mapping CSV (default: {DEFAULT_MAPPING})",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=DEFAULT_ALIASES,
        help=f"declared path-prefix aliases (default: {DEFAULT_ALIASES})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate and summarize the raw mapping CSV"
    )
    validate_parser.set_defaults(func=command_validate)

    normalize_parser = subparsers.add_parser(
        "normalize", help="generate the one-candidate-per-row mapping CSV"
    )
    normalize_parser.add_argument("--output", type=Path, default=DEFAULT_NORMALIZED_MAPPING)
    normalize_parser.add_argument(
        "--check", action="store_true", help="fail if the output is missing or stale"
    )
    normalize_parser.set_defaults(func=command_normalize)

    lookup_parser = subparsers.add_parser(
        "lookup", help="look up one exact legacy path without choosing ambiguities"
    )
    lookup_parser.add_argument("legacy_path")
    lookup_parser.add_argument("--recipe-key")
    lookup_parser.add_argument("--role", choices=("primary", "auxiliary"))
    lookup_parser.add_argument("--format", choices=("csv", "json"), default="csv")
    lookup_parser.set_defaults(func=command_lookup)

    audit_parser = subparsers.add_parser(
        "audit", help="recursively inventory and map a legacy YAML graph"
    )
    audit_parser.add_argument("recipe", type=Path)
    audit_parser.add_argument(
        "--recipe-key",
        required=True,
        help="mapping recipe key, such as v14_vlm_16k, v14_vlm_49k, or omni_262k",
    )
    audit_parser.add_argument(
        "--include-search-root",
        action="append",
        type=Path,
        default=[],
        help="fallback root for an unavailable include; may be repeated",
    )
    audit_parser.add_argument("--output", type=Path)
    audit_parser.set_defaults(func=command_audit)
    return parser


def main() -> int:
    """Run the requested mapping operation."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
