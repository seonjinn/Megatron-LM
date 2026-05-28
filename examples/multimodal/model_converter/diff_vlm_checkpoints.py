#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Utility to diff two Megatron-LM VLM checkpoints for a **single iteration**
# produced with the directory structure:
#   <iter_dir>/
#     mp_rank_00/
#       model_optim_rng.pt
#     mp_rank_01/
#       model_optim_rng.pt
#     ...
#
# Example:
# python examples/multimodal/model_converter/diff_vlm_checkpoints.py \
#     --dir-a /lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/nemotron_5p5_9b_v2/vlm/iter_0000001 \
#     --dir-b /lustre/fsw/portfolios/llmservice/users/cmccarthy/nemotron_5p5_9b_v2_c_radio_v2_vlm_tp4/iter_0000001 \
#     --model-only
#
# By default, the script will:
#   * Compare all common mp_rank_*/model_optim_rng.pt files in the two iter dirs
#   * Report missing mp_ranks / keys, shape mismatches, and tensor value deltas.

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _find_mp_rank_dirs(iter_dir: str) -> List[str]:
    """Return a sorted list of mp_rank_XX directory names present in iter_dir."""
    mp_re = re.compile(r"^mp_rank_(\d+)")
    ranks: List[Tuple[int, str]] = []
    for name in os.listdir(iter_dir):
        full = os.path.join(iter_dir, name)
        if os.path.isdir(full):
            m = mp_re.match(name)
            if m:
                ranks.append((int(m.group(1)), name))
    ranks.sort()
    return [name for _, name in ranks]


def _load_checkpoint(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    # weights_only=False since Megatron checkpoints contain optimizer & RNG.
    return torch.load(path, map_location="cpu", weights_only=False)


def _rank_key(name: str) -> str:
    """Extract the full rank suffix from an mp_rank directory name."""
    m = re.match(r"^mp_rank_(\d+(?:_\d+)*)$", name)
    if not m:
        raise ValueError(f"Unexpected mp_rank dir name: {name}")
    return m.group(1)


def _tensor_stats(delta: torch.Tensor) -> Dict[str, float]:
    """Return simple statistics for a tensor difference."""
    with torch.no_grad():
        # Ensure we are working in floating point to support integer / byte tensors.
        if not (torch.is_floating_point(delta) or delta.is_complex()):
            delta = delta.to(torch.float32)
        abs_delta = delta.abs()
        return {
            "max": float(abs_delta.max().item()) if abs_delta.numel() > 0 else 0.0,
            "mean": float(abs_delta.mean().item()) if abs_delta.numel() > 0 else 0.0,
        }


def _diff_state_dicts(
    state_a: Dict,
    state_b: Dict,
    dict_name: str,
    atol: float,
    rtol: float,
    max_report: int,
    key_prefixes: Optional[Iterable[str]] = None,
) -> Tuple[int, int, int]:
    """Diff two (sub-)state dicts, such as the 'model' key.

    Prints a human-readable summary to stdout.
    """
    # Optionally ignore bookkeeping / placeholder keys that we do not care about,
    # and restrict to a set of user-provided prefixes if requested.
    prefixes: Optional[Tuple[str, ...]] = None
    if key_prefixes is not None:
        prefixes = tuple(p for p in key_prefixes if p)

    def _is_ignored_key(k: str) -> bool:
        # FP8 placeholder / auxiliary state we want to skip.
        if prefixes is not None and not any(k.startswith(p) for p in prefixes):
            return True
        return False

    keys_a = {k for k in state_a.keys() if not _is_ignored_key(k)}
    keys_b = {k for k in state_b.keys() if not _is_ignored_key(k)}

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    only_a_final = [k for k in only_a if k.startswith("_extra_state")]
    only_b_final = [k for k in only_b if k.startswith("_extra_state")]

    print(f"  [{dict_name}] keys only in A: {len(only_a_final)} (ignored {len(only_a) - len(only_a_final)} extra state keys)")
    for k in only_a_final[:max_report]:
        print(f"    A-only: {k}")
    if len(only_a_final) > max_report:
        print(f"    ... {len(only_a_final) - max_report} more")

    print(f"  [{dict_name}] keys only in B: {len(only_b_final)} (ignored {len(only_b) - len(only_b_final)} extra state keys)")
    for k in only_b_final[:max_report]:
        print(f"    B-only: {k}")
    if len(only_b_final) > max_report:
        print(f"    ... {len(only_b_final) - max_report} more")

    shape_mismatches: List[str] = []
    value_diffs: List[Tuple[str, Dict[str, float]]] = []
    skipped_non_tensors: int = 0

    total_common = len(common)
    if total_common > 0:
        if prefixes is not None and len(prefixes) > 0:
            print(f"  [{dict_name}] comparing {total_common} common keys with prefixes: {prefixes}")
        else:
            print(f"  [{dict_name}] comparing {total_common} common keys...")
    # Emit lightweight progress updates for large dicts so the user sees activity.
    progress_every = max(1, total_common // 50) if total_common > 0 else 1

    for idx, k in enumerate(common, 1):
        va = state_a[k]
        vb = state_b[k]

        # For non-tensors, just report inequality.
        if not isinstance(va, torch.Tensor) or not isinstance(vb, torch.Tensor):
            if va != vb:
                skipped_non_tensors += 1
            continue

        if va.shape != vb.shape:
            shape_mismatches.append(k)
            continue

        # Promote to floating point for numeric comparison when needed.
        if not (torch.is_floating_point(va) or va.is_complex()):
            va_cmp = va.to(torch.float32)
            vb_cmp = vb.to(torch.float32)
        else:
            va_cmp = va
            vb_cmp = vb

        # Compare values with tolerances.
        diff = (va_cmp - vb_cmp).detach()
        stats = _tensor_stats(diff)
        # Use a simple torch.allclose-style check on the comparison tensors.
        if not torch.allclose(va_cmp, vb_cmp, atol=atol, rtol=rtol):
            value_diffs.append((k, stats))

        # Print a simple inline progress indicator for large numbers of keys.
        if total_common > 1000 and idx % progress_every == 0:
            pct = int(idx * 100 / total_common)
            sys.stdout.write(
                f"\r  [{dict_name}] progress: {idx}/{total_common} ({pct}%)"
            )
            sys.stdout.flush()

    if total_common > 1000:
        # Ensure we end the progress line cleanly.
        print()

    print(f"  [{dict_name}] shape mismatches: {len(shape_mismatches)}")
    for k in shape_mismatches[:max_report]:
        print(f"    shape mismatch: {k} | A: {tuple(state_a[k].shape)} "
              f"B: {tuple(state_b[k].shape)}")
    if len(shape_mismatches) > max_report:
        print(f"    ... {len(shape_mismatches) - max_report} more")

    print(f"  [{dict_name}] tensor value diffs (beyond tol): {len(value_diffs)}")
    for k, stats in value_diffs[:max_report]:
        print(
            f"    value diff: {k} | max_abs={stats['max']:.6g} "
            f"mean_abs={stats['mean']:.6g}"
        )
    if len(value_diffs) > max_report:
        print(f"    ... {len(value_diffs) - max_report} more")

    if skipped_non_tensors > 0:
        print(f"  [{dict_name}] non-tensor keys with unequal values: {skipped_non_tensors}")

    return len(shape_mismatches), len(value_diffs), skipped_non_tensors


def diff_rank_files(
    file_a: str,
    file_b: str,
    atol: float,
    rtol: float,
    max_report: int,
    compare_model_only: bool,
    key_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Diff two rank checkpoint files."""
    print(f"Comparing rank files:\n  A: {file_a}\n  B: {file_b}")
    ckpt_a = _load_checkpoint(file_a)
    ckpt_b = _load_checkpoint(file_b)

    top_keys_a = set(ckpt_a.keys())
    top_keys_b = set(ckpt_b.keys())
    print(f"Top-level keys A: {sorted(top_keys_a)}")
    print(f"Top-level keys B: {sorted(top_keys_b)}")

    shape_mismatches_total = 0
    value_diffs_total = 0

    if "model" in ckpt_a and "model" in ckpt_b:
        sm, vd, _ = _diff_state_dicts(
            ckpt_a["model"],
            ckpt_b["model"],
            "model",
            atol,
            rtol,
            max_report,
            key_prefixes=key_prefixes,
        )
        shape_mismatches_total += sm
        value_diffs_total += vd
    elif compare_model_only:
        print("WARNING: 'model' key missing in one or both checkpoints; skipping.")

    if not compare_model_only:
        # Compare everything at the top level (excluding 'model' which is handled above).
        ckpt_a_top = {k: v for k, v in ckpt_a.items() if k != "model"}
        ckpt_b_top = {k: v for k, v in ckpt_b.items() if k != "model"}
        sm, vd, _ = _diff_state_dicts(
            ckpt_a_top,
            ckpt_b_top,
            "top_level_excluding_model",
            atol,
            rtol,
            max_report,
            key_prefixes=key_prefixes,
        )
        shape_mismatches_total += sm
        value_diffs_total += vd

    return {
        "shape_mismatches": shape_mismatches_total,
        "value_diffs": value_diffs_total,
    }

def _print_rank_keys(checkpoint_path: str, label: str) -> None:
    """Load a single checkpoint file and print its keys."""
    print(f"Loading checkpoint for {label}: {checkpoint_path}")
    ckpt = _load_checkpoint(checkpoint_path)

    top_keys = sorted(ckpt.keys())
    print(f"=== {label} ===")
    print(f"Top-level keys ({len(top_keys)}):")
    for k in top_keys:
        print(f"  {k}")

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        model_keys = sorted(ckpt["model"].keys())
        print(f"\nModel parameter keys ({len(model_keys)}):")
        for k in model_keys:
            print(f"  {k}")
    else:
        print("\nNo 'model' key found in this checkpoint.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diff two Megatron-LM VLM checkpoints for a single iteration with a tp/mp structure.\n"
            "Provide two iteration directories (each containing mp_rank_* subdirs) as "
            "shown in the example in this file's header."
        )
    )
    parser.add_argument(
        "--dir-a",
        type=str,
        required=True,
        help="Path to first iteration directory (iter_XXXXXXX folder containing mp_rank_* subdirs).",
    )
    parser.add_argument(
        "--dir-b",
        type=str,
        required=True,
        help="Path to second iteration directory (iter_XXXXXXX folder containing mp_rank_* subdirs).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for tensor comparison (passed to torch.allclose).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for tensor comparison (passed to torch.allclose).",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Maximum number of entries to print per category (keys-only, mismatches, etc.).",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="If set, only compare the 'model' state dict and ignore other top-level keys.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of mp_rank identifiers to diff "
            "(e.g., '0,1,2' or '00_000,00_002'). Default: diff all ranks "
            "present in both dirs."
        ),
    )
    parser.add_argument(
        "--key-prefixes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of key prefixes to compare. "
            "If set, only parameters whose names start with one of these prefixes "
            "will be considered (after ignoring '_extra_state*' keys). "
            "Example: --key-prefixes vision_model,language_model"
        ),
    )
    parser.add_argument(
        "--print-keys-rank",
        type=str,
        default=None,
        help=(
            "If set, load this mp_rank index (from run A and/or run B), print its "
            "top-level and 'model' keys, then exit without running the diff."
        ),
    )
    return parser


def parse_key_prefixes(prefixes_arg: Optional[str]) -> Optional[List[str]]:
    if prefixes_arg is None:
        return None
    prefixes: List[str] = []
    for part in prefixes_arg.split(","):
        part = part.strip()
        if not part:
            continue
        prefixes.append(part)
    return prefixes


def parse_rank_filter(ranks_arg: Optional[str]) -> Optional[Iterable[str]]:
    if ranks_arg is None:
        return None
    ranks: List[str] = []
    for part in ranks_arg.split(","):
        part = part.strip()
        if not part:
            continue
        ranks.append(part)
    return ranks


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    key_prefixes = parse_key_prefixes(getattr(args, "key_prefixes", None))

    iter_dir_a = os.path.abspath(args.dir_a)
    iter_dir_b = os.path.abspath(args.dir_b)

    print(f"Iteration A directory: {iter_dir_a}")
    print(f"Iteration B directory: {iter_dir_b}")

    mp_dirs_a = _find_mp_rank_dirs(iter_dir_a)
    mp_dirs_b = _find_mp_rank_dirs(iter_dir_b)

    print(f"Found {len(mp_dirs_a)} mp_rank dirs in A: {mp_dirs_a}")
    print(f"Found {len(mp_dirs_b)} mp_rank dirs in B: {mp_dirs_b}")

    # Keep the full rank suffix so mp_rank_00_000 and mp_rank_00_002 stay distinct.
    ranks_a = {_rank_key(n): n for n in mp_dirs_a}
    ranks_b = {_rank_key(n): n for n in mp_dirs_b}

    all_common_ranks = sorted(set(ranks_a.keys()) & set(ranks_b.keys()))
    if not all_common_ranks:
        raise RuntimeError("No common mp_rank indices found between A and B.")

    # Optional: print keys for a single rank and exit early.
    if args.print_keys_rank is not None:
        rank_key = args.print_keys_rank
        available_in_a = rank_key in ranks_a
        available_in_b = rank_key in ranks_b

        if not available_in_a and not available_in_b:
            raise RuntimeError(
                f"Requested mp_rank {rank_key} not found in either run A or run B."
            )

        if available_in_a:
            dir_name_a = ranks_a[rank_key]
            file_a = os.path.join(iter_dir_a, dir_name_a, "model_optim_rng.pt")
            _print_rank_keys(file_a, f"Run A, mp_rank_{rank_key}")

        if available_in_b:
            dir_name_b = ranks_b[rank_key]
            file_b = os.path.join(iter_dir_b, dir_name_b, "model_optim_rng.pt")
            _print_rank_keys(file_b, f"Run B, mp_rank_{rank_key}")

        return

    ranks_filter = parse_rank_filter(args.ranks)
    if ranks_filter is not None:
        ranks_to_compare = sorted(set(all_common_ranks) & set(ranks_filter))
        if not ranks_to_compare:
            raise RuntimeError(
                "No overlapping mp_rank indices between provided --ranks and checkpoint dirs."
            )
    else:
        ranks_to_compare = all_common_ranks

    missing_in_a = sorted(set(ranks_b.keys()) - set(ranks_a.keys()))
    missing_in_b = sorted(set(ranks_a.keys()) - set(ranks_b.keys()))
    if missing_in_a:
        print(f"WARNING: mp_ranks present only in B: {missing_in_a}")
    if missing_in_b:
        print(f"WARNING: mp_ranks present only in A: {missing_in_b}")

    print(f"Ranks to compare: {ranks_to_compare}")

    failed_ranks: List[str] = []

    for rank_key in ranks_to_compare:
        dir_name_a = ranks_a[rank_key]
        dir_name_b = ranks_b[rank_key]
        file_a = os.path.join(iter_dir_a, dir_name_a, "model_optim_rng.pt")
        file_b = os.path.join(iter_dir_b, dir_name_b, "model_optim_rng.pt")

        print("=" * 80)
        print(f"mp_rank_{rank_key}:")
        try:
            result = diff_rank_files(
                file_a=file_a,
                file_b=file_b,
                atol=args.atol,
                rtol=args.rtol,
                max_report=args.max_report,
                compare_model_only=args.model_only,
                key_prefixes=key_prefixes,
            )
            if result["shape_mismatches"] > 0 or result["value_diffs"] > 0:
                failed_ranks.append(rank_key)
                print(
                    f"ERROR: mp_rank_{rank_key} has "
                    f"{result['shape_mismatches']} shape mismatches and "
                    f"{result['value_diffs']} tensor value diffs."
                )
        except Exception as exc:  # noqa: BLE001 - surface any failure per rank
            failed_ranks.append(rank_key)
            print(f"ERROR: diff failed for mp_rank_{rank_key}: {exc}")

    if failed_ranks:
        print(f"Completed with failures for mp_ranks: {failed_ranks}")
        sys.exit(1)
    else:
        print("All requested mp_rank diffs completed successfully.")


if __name__ == "__main__":
    sys.exit(main())
