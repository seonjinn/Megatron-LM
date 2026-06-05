#!/usr/bin/env python3
"""
Compare two CONVERT_SUPER mcore checkpoints.

Produces three separate reports:
  1. Model weight tensors  — shape/value comparison per key
  2. Args namespace        — flattened key-by-key diff, with path-specific
                             fields (save, wandb_save_dir, …) shown separately
  3. MTP weight keys       — which checkpoint (if either) contains mtp.* tensors

Usage (on an interactive node):
    python examples/multimodal/super/compare_converted_checkpoints.py \\
        /path/to/ckpt_a/tp_1 \\
        /path/to/ckpt_b/tp_1 \\
        [--iter iter_0053520] \\
        [--label-a "ours"] [--label-b "theirs"]
"""

import argparse
import sys
from pathlib import Path

import torch

# Args fields that are expected to differ between two different runs
# (checkpoint paths, WandB settings, training-specific counters).
# These are shown in a separate "path/run-specific diffs" section, not
# flagged as unexpected.
PATH_OR_RUN_FIELDS = {
    "save", "load", "pretrained_checkpoint",
    "wandb_save_dir", "wandb_exp_name", "wandb_resume_id",
    "rank", "local_rank", "consumed_train_samples", "consumed_valid_samples",
    "skipped_train_samples", "iteration",
}


def find_iter_dir(ckpt_dir: Path, iter_name: str | None) -> Path:
    if iter_name:
        d = ckpt_dir / iter_name
        if not d.is_dir():
            raise ValueError(f"Iteration dir not found: {d}")
        return d
    iter_dirs = sorted(ckpt_dir.glob("iter_*"))
    if not iter_dirs:
        raise ValueError(f"No iter_* directories found in {ckpt_dir}")
    return iter_dirs[-1]


def load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def namespace_to_dict(obj) -> dict:
    """Recursively convert argparse.Namespace (and nested ones) to plain dicts."""
    import argparse
    if isinstance(obj, argparse.Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(namespace_to_dict(v) for v in obj)
    return obj


def flatten(obj, prefix=""):
    """Flatten a nested dict/list/tuple into {dotted.key: leaf_value} pairs."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            items.update(flatten(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = obj
    return items


def compare_args(d1: dict, d2: dict, label_a: str, label_b: str) -> bool:
    """
    Compare two args dicts (already flattened).
    Prints a structured diff. Returns True if no *unexpected* differences.
    """
    keys1, keys2 = set(d1), set(d2)
    only1 = keys1 - keys2
    only2 = keys2 - keys1
    common = keys1 & keys2

    path_diffs = {}   # expected path/run-specific differences
    real_diffs = {}   # unexpected semantic differences

    for k in sorted(common):
        v1, v2 = d1[k], d2[k]
        try:
            equal = v1 == v2
        except Exception:
            equal = False
        if not equal:
            # Split leaf key (last segment) to check against PATH_OR_RUN_FIELDS
            leaf = k.split(".")[-1].split("[")[0]
            if leaf in PATH_OR_RUN_FIELDS:
                path_diffs[k] = (v1, v2)
            else:
                real_diffs[k] = (v1, v2)

    only1_path = {k for k in only1 if k.split(".")[-1].split("[")[0] in PATH_OR_RUN_FIELDS}
    only1_real = only1 - only1_path
    only2_path = {k for k in only2 if k.split(".")[-1].split("[")[0] in PATH_OR_RUN_FIELDS}
    only2_real = only2 - only2_path

    if path_diffs or only1_path or only2_path:
        print(f"  Path/run-specific diffs (expected):")
        for k, (v1, v2) in sorted(path_diffs.items()):
            print(f"    {k}:")
            print(f"      {label_a}: {v1!r}")
            print(f"      {label_b}: {v2!r}")
        for k in sorted(only1_path):
            print(f"    {k}: only in {label_a} = {d1[k]!r}")
        for k in sorted(only2_path):
            print(f"    {k}: only in {label_b} = {d2[k]!r}")

    if real_diffs or only1_real or only2_real:
        print(f"  Semantic diffs (may be significant):")
        for k, (v1, v2) in sorted(real_diffs.items()):
            print(f"    {k}:")
            print(f"      {label_a}: {v1!r}")
            print(f"      {label_b}: {v2!r}")
        for k in sorted(only1_real):
            print(f"    {k}: only in {label_a} = {d1[k]!r}")
        for k in sorted(only2_real):
            print(f"    {k}: only in {label_b} = {d2[k]!r}")
        return False
    else:
        matching = len(common) - len(real_diffs)
        print(f"  {matching} args identical, {len(path_diffs)} path/run-specific diffs (expected)")
        return True


def compare_model_weights(flat1: dict, flat2: dict,
                          file_label: str, label_a: str, label_b: str,
                          results: dict, check_values: bool = True):
    """
    Compare flattened model weight dicts.
    Shape/dtype mismatches always go to weight_shape_mismatched (architecture problem).
    Value mismatches go to weight_value_mismatched (expected for different training runs).
    MTP-only keys are routed to their own buckets.
    """
    keys1, keys2 = set(flat1), set(flat2)
    only1 = keys1 - keys2
    only2 = keys2 - keys1

    for k in only1:
        bucket = "mtp_only_in_a" if "mtp" in k.lower() else "unexpected_only_in_a"
        results[bucket].add(f"{file_label}::{k}")

    for k in only2:
        bucket = "mtp_only_in_b" if "mtp" in k.lower() else "unexpected_only_in_b"
        results[bucket].add(f"{file_label}::{k}")

    for k in keys1 & keys2:
        v1, v2 = flat1[k], flat2[k]
        # _extra_state holds TransformerEngine FP8 scaling buffers, not trainable
        # weights. Size differences are expected (empty after conversion vs. live
        # FP8 state). Track separately so they don't pollute the shape-mismatch count.
        leaf = k.split(".")[-1].split("[")[0]
        if leaf == "_extra_state":
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if v1.shape != v2.shape:
                    results["extra_state_diffs"] += 1
            continue
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            if v1.shape != v2.shape or v1.dtype != v2.dtype:
                results["weight_shape_mismatched"].append(
                    f"{file_label}::{k}  "
                    f"shapes=({v1.shape} vs {v2.shape})  "
                    f"dtypes=({v1.dtype} vs {v2.dtype})"
                )
            elif check_values and not torch.equal(v1, v2):
                results["weight_value_mismatched"] += 1
            else:
                results["weight_identical"] += 1
        elif v1 != v2:
            # Non-tensor scalar in model dict (rare)
            results["weight_shape_mismatched"].append(
                f"{file_label}::{k}  scalar values=({v1!r} vs {v2!r})"
            )
        else:
            results["weight_identical"] += 1


def main():
    parser = argparse.ArgumentParser(description="Compare two CONVERT_SUPER mcore checkpoints")
    parser.add_argument("ckpt_a", help="First checkpoint tp_1 directory")
    parser.add_argument("ckpt_b", help="Second checkpoint tp_1 directory")
    parser.add_argument("--iter", help="Iteration folder (e.g. iter_0053520); auto-detected if omitted")
    parser.add_argument("--label-a", default="ckpt_a", help="Label for first checkpoint")
    parser.add_argument("--label-b", default="ckpt_b", help="Label for second checkpoint")
    parser.add_argument("--check-values", action="store_true", default=False,
                        help="Also verify weight values are bit-identical (use for same-run comparison)")
    cli = parser.parse_args()

    dir_a = Path(cli.ckpt_a)
    dir_b = Path(cli.ckpt_b)
    label_a, label_b = cli.label_a, cli.label_b

    base_a = find_iter_dir(dir_a, cli.iter)
    base_b = find_iter_dir(dir_b, cli.iter)
    print(f"{label_a:12s}: {base_a}")
    print(f"{label_b:12s}: {base_b}")

    files_a = {f.relative_to(base_a): f for f in sorted(base_a.rglob("*.pt"))}
    files_b = {f.relative_to(base_b): f for f in sorted(base_b.rglob("*.pt"))}

    only_a = set(files_a) - set(files_b)
    only_b = set(files_b) - set(files_a)
    common = set(files_a) & set(files_b)

    if only_a:
        print(f"\nFiles only in {label_a}: {sorted(only_a)}")
    if only_b:
        print(f"\nFiles only in {label_b}: {sorted(only_b)}")
    print(f"\nComparing {len(common)} common rank file(s)...\n")

    weight_results = {
        "mtp_only_in_a": set(),
        "mtp_only_in_b": set(),
        "unexpected_only_in_a": set(),
        "unexpected_only_in_b": set(),
        "weight_shape_mismatched": [],   # always a problem
        "weight_value_mismatched": 0,    # expected for different training runs
        "weight_identical": 0,
        "extra_state_diffs": 0,          # TE FP8 buffers — not trainable weights
    }
    args_ok = True

    for rel in sorted(common):
        print(f"── {rel}")
        sd_a = load_pt(files_a[rel])
        sd_b = load_pt(files_b[rel])

        # ── Args ──────────────────────────────────────────────────────
        sec_a = sd_a.get("args")
        sec_b = sd_b.get("args")
        print(f"  [args]")
        if sec_a is None and sec_b is None:
            print("  (no args section in either)")
        elif sec_a is None or sec_b is None:
            print(f"  args present in one checkpoint only!")
            args_ok = False
        else:
            flat_a = flatten(namespace_to_dict(sec_a))
            flat_b = flatten(namespace_to_dict(sec_b))
            if not compare_args(flat_a, flat_b, label_a, label_b):
                args_ok = False

        # ── Model weights ─────────────────────────────────────────────
        print(f"  [model weights]")
        m_a = sd_a.get("model")
        m_b = sd_b.get("model")
        if m_a is None and m_b is None:
            print("  (no model section)")
        elif m_a is None or m_b is None:
            print("  model section present in one checkpoint only!")
            weight_results["weight_mismatched"].append(f"{rel}::model (absent in one)")
        else:
            before_identical = weight_results["weight_identical"]
            before_shape_mm = len(weight_results["weight_shape_mismatched"])
            before_value_mm = weight_results["weight_value_mismatched"]
            compare_model_weights(flatten(m_a), flatten(m_b),
                                  str(rel), label_a, label_b, weight_results,
                                  check_values=cli.check_values)
            n_identical = weight_results["weight_identical"] - before_identical
            n_shape_mm = len(weight_results["weight_shape_mismatched"]) - before_shape_mm
            n_value_mm = weight_results["weight_value_mismatched"] - before_value_mm
            summary = f"  {n_identical} tensors: shapes match"
            if n_shape_mm:
                summary += f", {n_shape_mm} SHAPE MISMATCHES"
            if cli.check_values:
                summary += f", {n_value_mm} value diffs" if n_value_mm else ", values identical"
            print(summary)
        print()

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print(f"\nModel weights identical : {weight_results['weight_identical']}")

    mtp_a = weight_results["mtp_only_in_a"]
    mtp_b = weight_results["mtp_only_in_b"]
    if mtp_a:
        print(f"\nMTP tensors only in {label_a} : {len(mtp_a)}")
        for k in sorted(mtp_a)[:20]:
            print(f"  + {k}")
        if len(mtp_a) > 20:
            print(f"  ... ({len(mtp_a)} total)")
    if mtp_b:
        print(f"\nMTP tensors only in {label_b} : {len(mtp_b)}")
        for k in sorted(mtp_b)[:20]:
            print(f"  + {k}")
        if len(mtp_b) > 20:
            print(f"  ... ({len(mtp_b)} total)")
    if not mtp_a and not mtp_b:
        print(f"\nNo MTP tensors in either checkpoint.")

    u_a = weight_results["unexpected_only_in_a"]
    if u_a:
        print(f"\nUnexpected keys only in {label_a} : {len(u_a)}")
        for k in sorted(u_a):
            print(f"  ! {k}")

    u_b = weight_results["unexpected_only_in_b"]
    if u_b:
        print(f"\nUnexpected keys only in {label_b} : {len(u_b)}")
        for k in sorted(u_b):
            print(f"  ! {k}")

    extra_diffs = weight_results["extra_state_diffs"]
    if extra_diffs:
        print(f"\n_extra_state size diffs (TE/FP8 buffers, not weights) : {extra_diffs}  [ignored]")

    shape_mm = weight_results["weight_shape_mismatched"]
    if shape_mm:
        print(f"\nShape/dtype mismatches (architecture problem) : {len(shape_mm)}")
        for item in shape_mm[:20]:
            print(f"  !! {item}")
        if len(shape_mm) > 20:
            print(f"  ... ({len(shape_mm)} total)")
    else:
        print(f"\nNo shape/dtype mismatches — architectures match.")

    if cli.check_values:
        val_mm = weight_results["weight_value_mismatched"]
        if val_mm:
            print(f"\nValue-different tensors (same shape/dtype) : {val_mm}")
        else:
            print(f"\nAll shared weight values are bit-identical.")

    weights_ok = not shape_mm and not u_a and not u_b

    print()
    if weights_ok and args_ok:
        print("PASS: checkpoints are semantically equivalent.")
    elif weights_ok and not args_ok:
        print("PASS (weights): weight tensors are identical; see args diffs above.")
    else:
        print("FAIL: weight differences detected (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
