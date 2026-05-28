#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from collections import defaultdict

from tqdm import tqdm

def parse_iter_num(name):
    if name.startswith("iter_"):
        name = name.split("_", 1)[1]
    try:
        return int(name)
    except ValueError:
        return None


def find_latest_iter(iter_dir):
    """Find the latest iteration. Checks for latest_checkpointed_iteration.txt
    in both iter_dir and its parent (covers both standard and tp_* layouts)."""
    latest_iter = None

    for search_dir in [iter_dir, os.path.dirname(iter_dir)]:
        latest_file = os.path.join(search_dir, "latest_checkpointed_iteration.txt")
        if os.path.isfile(latest_file):
            with open(latest_file, "r") as f:
                latest_iter = f.read().strip()
            if latest_iter:
                break

    if not latest_iter:
        iters = []
        for name in os.listdir(iter_dir):
            if name.startswith("iter_"):
                iter_num = parse_iter_num(name)
                if iter_num is not None:
                    iters.append(iter_num)
        if iters:
            latest_iter = str(max(iters))

    return latest_iter

def main():
    """
    Example usage:
        # Dry-run, print paths to delete and exit
        python examples/multimodal/tools/delete_checkpoints.py --filter _01

        # Dry-run, include tp_1 and tp_1_hf directories
        python examples/multimodal/tools/delete_checkpoints.py --filter _01 --check-tp-dirs

        # Delete, but still prompt for confirmation first
        python examples/multimodal/tools/delete_checkpoints.py --filter _01 --delete
    """
    parser = argparse.ArgumentParser(description="Delete non-latest checkpoint dirs.")
    parser.add_argument(
        "--root",
        default=os.path.join(os.environ.get("SHARE_OUTPUT", ""), "workspace", "output"),
        help="Root output dir (default: $SHARE_OUTPUT/workspace/output)",
    )
    parser.add_argument(
        "--filter",
        dest="name_filter",
        required=True,
        help="Regex filter for run directory names",
    )
    parser.add_argument(
        "--ignore-filter",
        dest="ignore_filter",
        default=None,
        help="Regex to exclude run directory names after --filter",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete; otherwise default is dry-run",
    )
    parser.add_argument(
        "--delete-interval",
        type=int,
        default=None,
        help="Only delete checkpoints at this iteration interval (always keep latest)",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=1,
        help="Keep the last N checkpoint directories (default: keep latest only)",
    )
    parser.add_argument(
        "--check-tp-dirs",
        action="store_true",
        help="Also search checkpoints/tp_1/ and checkpoints/tp_1_hf/ for iter_* dirs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all matched directories and skip reasons",
    )
    args = parser.parse_args()

    root_dir = os.path.expandvars(os.path.expanduser(args.root))

    if not os.path.isdir(root_dir):
        print(f"Missing `--root` and default dir does not exist: {root_dir}")
        return

    if args.delete_interval is not None and args.delete_interval <= 0:
        parser.error("--delete-interval must be a positive integer")

    name_re = re.compile(args.name_filter)
    ignore_re = re.compile(args.ignore_filter) if args.ignore_filter else None

    def dir_size_bytes(path):
        total = 0
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    total += os.path.getsize(file_path)
                except OSError:
                    pass
        return total

    delete_paths = []
    run_total_checkpoints = {}  # run_name -> total checkpoint count
    run_names = [name for name in os.listdir(root_dir)]
    for run_name in tqdm(run_names, desc="Runs", unit="run"):
        run_dir = os.path.join(root_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        if not name_re.search(run_name):
            continue
        if ignore_re and ignore_re.search(run_name):
            continue

        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            if args.verbose:
                tqdm.write(f"Matched: {run_name}")
                tqdm.write(f"  -> no checkpoints directory")
            continue

        # Build list of directories to scan for iter_* entries
        iter_dirs = [checkpoints_dir]
        if args.check_tp_dirs:
            for subdir in ["tp_1", "tp_1_hf"]:
                p = os.path.join(checkpoints_dir, subdir)
                if os.path.isdir(p):
                    iter_dirs.append(p)

        if args.verbose:
            tqdm.write(f"Matched: {run_name}")

        # Collect iter_* from all iter_dirs; track (iter_num, dir, name)
        all_iter_entries = []
        seen_iters = set()
        for d in iter_dirs:
            for name in os.listdir(d):
                if not name.startswith("iter_"):
                    continue
                iter_num = parse_iter_num(name)
                if iter_num is not None:
                    all_iter_entries.append((iter_num, d, name))
                    seen_iters.add(iter_num)

        if not seen_iters:
            if args.verbose:
                tqdm.write(f"  -> 0 checkpoint(s), no iter_* found")
            continue

        # Determine which iterations to keep (by unique iter number)
        sorted_iters = sorted(seen_iters, reverse=True)
        keep_count = args.keep_last_n if args.keep_last_n is not None else 1
        keep_iters = set(sorted_iters[:keep_count])
        run_total_checkpoints[run_name] = len(seen_iters)

        run_delete_count = 0
        for iter_num, d, name in all_iter_entries:
            if iter_num in keep_iters:
                continue
            if args.delete_interval and (iter_num % args.delete_interval != 0):
                continue
            delete_paths.append((run_name, os.path.join(d, name)))
            run_delete_count += 1

        if args.verbose:
            if run_delete_count == 0:
                tqdm.write(f"  -> {len(seen_iters)} unique iter(s) across {len(iter_dirs)} dir(s), all kept")
            else:
                tqdm.write(f"  -> {run_delete_count} checkpoint dir(s) to delete ({len(seen_iters)} unique iters across {len(iter_dirs)} dir(s))")

    if not delete_paths:
        print("No checkpoint directories to delete.")
        return

    total_bytes = sum(dir_size_bytes(path) for _, path in delete_paths)
    total_gb = total_bytes / (1024 ** 3)
    total_tb = total_bytes / (1024 ** 4)

    # Group delete paths by run name
    by_run = defaultdict(list)
    for run_name, path in delete_paths:
        iter_name = os.path.basename(path)
        parent = os.path.basename(os.path.dirname(path))
        label = f"{parent}/{iter_name}" if parent != "checkpoints" else iter_name
        by_run[run_name].append(label)

    print("Will delete the following checkpoint directories:")
    for run_name in sorted(by_run.keys()):
        entries = sorted(by_run[run_name])
        total = run_total_checkpoints.get(run_name, 0)
        unique_iters_to_delete = len(set(
            os.path.basename(p) for rn, p in delete_paths if rn == run_name
        ))
        remaining = total - unique_iters_to_delete
        print(f"  {run_name}: {', '.join(entries)} (remaining: {remaining})")
    print(f"Total: {len(delete_paths)} checkpoint dir(s) across {len(by_run)} run(s)")
    print(f"Total size: {total_gb:.2f} GiB ({total_tb:.2f} TiB)")

    if not args.delete:
        print("Dry-run mode (default). Re-run with --delete to remove these.")
        return

    answer = input("Proceed with deletion? [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    for run_name, path in tqdm(delete_paths, desc="Deleting", unit="dir"):
        short_path = os.path.relpath(path, root_dir)
        if not os.path.exists(path):
            tqdm.write(f"Skipping (already deleted): {short_path}")
            continue
        tqdm.write(f"Deleting {short_path}")
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            tqdm.write(f"  -> already deleted (race condition)")
        except OSError as e:
            tqdm.write(f"  -> failed: {e}")

if __name__ == "__main__":
    main()
