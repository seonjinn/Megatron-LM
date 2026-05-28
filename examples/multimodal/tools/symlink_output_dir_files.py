#!/usr/bin/env python3
"""
Script to create symbolic links for all files from a source directory to a destination directory.
Creates actual directories (not symlinks) so you can write to them, while symlinking files for reading.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional


def symlink_files(
    src: str,
    dest: str,
    ignore_dirs: Optional[list[str]] = None,
    ignore_files: Optional[list[str]] = None,
    symlink_dirs: Optional[list[str]] = None,
    symlink_children_of: Optional[list[str]] = None,
    except_children: Optional[list[str]] = None,
    dry_run: bool = False,
    verbose: bool = False,
):
    """
    Create symbolic links for all files from src to dest.

    Args:
        src: Source directory path
        dest: Destination directory path
        ignore_dirs: List of directory names to ignore (default: ['code_snapshots', 'logs'])
        ignore_files: List of filenames to skip (e.g. ['config.yaml'])
        symlink_dirs: List of directory names to symlink entirely (default: ['tensorboard'])
        symlink_children_of: List of directory names whose immediate subdirectories should be
            symlinked entirely (default: ['checkpoints']). Useful for checkpoint directories
            where each child (e.g. iter_0001000/) should be a single directory symlink.
        except_children: List of subdirectory names to exclude from symlink_children_of
            (default: ['tp_1']). These will be walked into normally instead of being symlinked.
        dry_run: If True, only print what would be done without actually doing it
        verbose: If True, print each operation
    """
    if ignore_dirs is None:
        ignore_dirs = ['code_snapshots', 'logs']
    if ignore_files is None:
        ignore_files = []
    ignore_files_set = set(ignore_files)
    if symlink_dirs is None:
        symlink_dirs = ['tensorboard']
    if symlink_children_of is None:
        symlink_children_of = ['checkpoints']
    if except_children is None:
        except_children = ['tp_1', 'tp_1_hf']

    src_path = Path(src).resolve()
    dest_path = Path(dest).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")

    if not src_path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src_path}")

    # Create destination root if it doesn't exist
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)

    files_linked = 0
    dirs_created = 0
    dirs_linked = 0
    files_skipped = 0
    dirs_skipped = 0

    for root, dirs, files in os.walk(src_path):
        # Filter out ignored directories (modifies dirs in-place to prevent descending)
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # Calculate relative path from source root
        rel_path = Path(root).relative_to(src_path)
        dest_subdir = dest_path / rel_path

        # Create the directory in destination (not a symlink)
        if not dest_subdir.exists():
            if not dry_run:
                dest_subdir.mkdir(parents=True, exist_ok=True)
            dirs_created += 1

        # Handle directories that should be symlinked entirely
        dirs_to_symlink = [d for d in dirs if d in symlink_dirs]
        for dir_name in dirs_to_symlink:
            src_dir = Path(root) / dir_name
            dest_dir = dest_subdir / dir_name

            if dest_dir.exists() or dest_dir.is_symlink():
                if verbose or dry_run:
                    print(f"Skipping dir symlink (exists): {rel_path / dir_name}")
                dirs_skipped += 1
            else:
                if verbose or dry_run:
                    print(f"Symlinking dir: {rel_path / dir_name}")
                if not dry_run:
                    dest_dir.symlink_to(src_dir)
                dirs_linked += 1

        # Remove symlinked dirs from the walk list so we don't descend into them
        dirs[:] = [d for d in dirs if d not in symlink_dirs]

        # Handle symlink_children_of: if current dir matches, symlink all its
        # immediate child directories (except those in except_children)
        current_dir_name = Path(root).name
        if symlink_children_of and current_dir_name in symlink_children_of:
            children_to_symlink = [d for d in dirs if d not in except_children]
            for dir_name in children_to_symlink:
                src_dir = Path(root) / dir_name
                dest_dir = dest_subdir / dir_name

                if dest_dir.exists() or dest_dir.is_symlink():
                    if verbose or dry_run:
                        print(f"Skipping child dir symlink (exists): {rel_path / dir_name}")
                    dirs_skipped += 1
                else:
                    if verbose or dry_run:
                        print(f"Symlinking child dir: {rel_path / dir_name}")
                    if not dry_run:
                        dest_dir.symlink_to(src_dir)
                    dirs_linked += 1

            # Remove symlinked children from walk list so we don't descend into them
            dirs[:] = [d for d in dirs if d not in children_to_symlink]

            # Copy (not symlink) latest_checkpointed_iteration.txt so it doesn't
            # change when the source keeps training. Then verify the iter dir exists.
            latest_txt = "latest_checkpointed_iteration.txt"
            if latest_txt in files:
                files = [f for f in files if f != latest_txt]
                src_latest = Path(root) / latest_txt
                dest_latest = dest_subdir / latest_txt
                if not (dest_latest.exists() or dest_latest.is_symlink()):
                    if not dry_run:
                        shutil.copy2(str(src_latest), str(dest_latest))
                    files_linked += 1
                    latest_iter = int(src_latest.read_text().strip())
                    iter_dir = dest_subdir / f"iter_{latest_iter:07d}"
                    if not iter_dir.exists():
                        print(f"Warning: {latest_txt} references iter {latest_iter} "
                              f"but {iter_dir} does not exist in destination")

        # Create symlinks for all files
        dir_files_linked = 0
        dir_files_skipped = 0
        dir_files_ignored = 0
        for file in files:
            if file in ignore_files_set:
                files_skipped += 1
                dir_files_skipped += 1
                continue

            src_file = Path(root) / file
            dest_file = dest_subdir / file

            if dest_file.exists() or dest_file.is_symlink():
                files_skipped += 1
                dir_files_skipped += 1
                continue

            if not dry_run:
                dest_file.symlink_to(src_file)
            files_linked += 1
            dir_files_linked += 1

        # Print per-directory summary
        if verbose or dry_run:
            if dir_files_linked > 0 or dir_files_skipped > 0 or dir_files_ignored > 0:
                parts = []
                if dir_files_linked > 0:
                    parts.append(f"{dir_files_linked} files linked")
                if dir_files_skipped > 0:
                    parts.append(f"{dir_files_skipped} skipped")
                if dir_files_ignored > 0:
                    parts.append(f"{dir_files_ignored} ignored")
                print(f"{rel_path or '.'}: {', '.join(parts)}")

    print("\nSummary:")
    print(f"  Directories created: {dirs_created}")
    print(f"  Directories symlinked: {dirs_linked}")
    print(f"  Files linked: {files_linked}")
    print(f"  Directories skipped (already exist): {dirs_skipped}")
    print(f"  Files skipped (already exist): {files_skipped}")
    print(f"  Ignored directories: {ignore_dirs}")
    if ignore_files_set:
        print(f"  Ignored files: {ignore_files_set}")
    print(f"  Symlinked directories: {symlink_dirs}")
    print(f"  Symlink children of: {symlink_children_of} (except {except_children})")

    if dry_run:
        print("\n(Dry run - no changes were made)")


def parse_iter_number(iter_name: str) -> int:
    """Parse iteration number from either iter_125 or 125."""
    match = re.fullmatch(r"(?:iter_)?(\d+)", iter_name)
    if not match:
        raise ValueError(
            f"Invalid iteration name: {iter_name}. Expected 125 or iter_125."
        )
    return int(match.group(1))


def format_iter_name(iter_num: int) -> str:
    return f"iter_{iter_num:07d}"


def discover_hf_checkpoint_sources(
    src_path: Path,
    input_src: str,
    requested_iters: Optional[list[str]] = None,
) -> list[tuple[str, Path]]:
    """
    Discover source directories for eval-compatible HF checkpoint layouts.

    Supported source types:
      - omni-rl: an RL-style directory containing one or more iter_* subdirectories,
        or a single iter_* directory.
      - hf: a flat HF checkpoint directory with model files at the source root.
    """
    requested_iter_nums = None
    if requested_iters:
        requested_iter_nums = [parse_iter_number(iter_name) for iter_name in requested_iters]

    if input_src == "hf":
        if requested_iter_nums is not None and len(requested_iter_nums) > 1:
            raise ValueError(
                "Flat HF checkpoint sources accept at most one value for --iters."
            )

        flat_iter_num = 1 if requested_iter_nums is None else requested_iter_nums[0]
        return [(format_iter_name(flat_iter_num), src_path.resolve())]

    if src_path.name.startswith("iter_"):
        candidates = [src_path.resolve()]
    else:
        candidates = sorted(
            [
                path.resolve()
                for path in src_path.iterdir()
                if path.is_dir() and re.fullmatch(r"iter_\d+", path.name)
            ],
            key=lambda path: parse_iter_number(path.name),
        )

    if not candidates:
        raise FileNotFoundError(
            f"No iter_* directories found in source: {src_path}"
        )

    requested_iter_set = None
    if requested_iter_nums is not None:
        requested_iter_set = set(requested_iter_nums)

    selected: list[tuple[str, Path]] = []
    found_iter_nums = set()
    for candidate in candidates:
        iter_num = parse_iter_number(candidate.name)
        if requested_iter_set is not None and iter_num not in requested_iter_set:
            continue
        found_iter_nums.add(iter_num)
        selected.append((format_iter_name(iter_num), candidate))

    if requested_iter_set is not None:
        missing = sorted(requested_iter_set - found_iter_nums)
        if missing:
            missing_str = ", ".join(str(iter_num) for iter_num in missing)
            raise FileNotFoundError(
                f"Requested iterations not found under {src_path}: {missing_str}"
            )

    return selected


def symlink_hf_checkpoint_layout(
    src: str,
    dest: str,
    input_src: str,
    iterations: Optional[list[str]] = None,
    tp_folder: str = "tp_1_hf",
    dry_run: bool = False,
    verbose: bool = False,
):
    """
    Create a lightweight eval-compatible layout for HF checkpoints.

    This adapts either an omni-rl source with iter_* directories or a flat HF
    checkpoint directory into the layout expected by run_all_benchmark_vllm_auto.sh:

      <dest>/checkpoints/tp_1_hf/iter_0000125/mcore_to_hf/

    For input_src="hf", the destination iteration defaults to iter_0000001
    unless overridden with a single value via --iters. For input_src="omni-rl",
    the source must already contain iter_* directories.
    """
    src_path = Path(src).resolve()
    dest_path = Path(dest).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")
    if not src_path.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src_path}")
    if input_src == "hf" and src_path == dest_path:
        raise ValueError(
            "For --input-src hf, --src and --dest must be different directories. "
            "Use a separate destination such as a _dup dir."
        )

    iter_dirs = discover_hf_checkpoint_sources(src_path, input_src, iterations)
    checkpoints_dir = dest_path / "checkpoints"
    tp_dir = checkpoints_dir / tp_folder

    if not dry_run:
        tp_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    latest_iter_num = None

    for iter_name, src_iter_dir in iter_dirs:
        iter_num = parse_iter_number(iter_name)
        latest_iter_num = iter_num if latest_iter_num is None else max(latest_iter_num, iter_num)
        dest_iter_dir = tp_dir / iter_name
        model_dir = dest_iter_dir / "mcore_to_hf"

        if verbose or dry_run:
            print(f"Preparing {iter_name} -> {src_iter_dir}")

        if model_dir.is_symlink():
            raise FileExistsError(
                f"Destination exists as a symlink; expected a real directory: {model_dir}"
            )

        if model_dir.exists():
            if any(model_dir.iterdir()):
                if verbose or dry_run:
                    print(f"Skipping existing populated directory: {model_dir}")
                skipped += 1
                continue
        elif not dry_run:
            model_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            symlink_files(
                src=str(src_iter_dir),
                dest=str(model_dir),
                ignore_dirs=[],
                ignore_files=[],
                symlink_dirs=[],
                symlink_children_of=[],
                except_children=[],
                dry_run=True,
                verbose=verbose,
            )
        else:
            dest_iter_dir.mkdir(parents=True, exist_ok=True)
            symlink_files(
                src=str(src_iter_dir),
                dest=str(model_dir),
                ignore_dirs=[],
                ignore_files=[],
                symlink_dirs=[],
                symlink_children_of=[],
                except_children=[],
                dry_run=False,
                verbose=verbose,
            )
        linked += 1

    if latest_iter_num is not None:
        latest_path = tp_dir / "latest_checkpointed_iteration.txt"
        latest_contents = f"{latest_iter_num}\n"
        if verbose or dry_run:
            print(f"Writing latest checkpoint marker: {latest_path} -> {latest_iter_num}")
        if not dry_run:
            latest_path.write_text(latest_contents)

    print("\nSummary:")
    print(f"  Destination root: {dest_path}")
    print(f"  TP folder: {tp_folder}")
    print(f"  Iterations linked: {linked}")
    print(f"  Iterations skipped (already linked): {skipped}")
    print(f"  Iteration names: {', '.join(iter_name for iter_name, _ in iter_dirs)}")

    if dry_run:
        print("\n(Dry run - no changes were made)")


def symlink_hf_eval_layout(
    src: str,
    dest: str,
    iterations: Optional[list[str]] = None,
    tp_folder: str = "tp_1_hf",
    dry_run: bool = False,
    verbose: bool = False,
):
    """Backward-compatible wrapper for older --layout hf_eval usage."""
    symlink_hf_checkpoint_layout(
        src=src,
        dest=dest,
        input_src="omni-rl",
        iterations=iterations,
        tp_folder=tp_folder,
        dry_run=dry_run,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create symbolic links for files from source to destination directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --src /path/to/source --dest /path/to/dest
  %(prog)s --src /path/to/source --dest /path/to/dest --dry-run --verbose
  %(prog)s --input-src omni-rl --src /path/to/rl_checkpoints --dest /path/to/output/model
  %(prog)s --input-src hf --src /path/to/downloaded_hf_checkpoint --dest /path/to/output/model
  %(prog)s --input-src hf --src /path/to/downloaded_hf_checkpoint --dest /path/to/output/model --iters 125
        """
    )

    parser.add_argument(
        "--input-src",
        choices=["megatron", "omni-rl", "hf"],
        default="megatron",
        help="Interpret the source as a generic Megatron output dir, an omni-rl iter_* tree, or a flat HF checkpoint dir"
    )
    parser.add_argument(
        "--layout",
        choices=["generic", "hf_eval"],
        default="generic",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--hf-checkpoint",
        action="store_true",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source directory to read files from"
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory to create symlinks in"
    )
    parser.add_argument(
        "--iters",
        nargs="+",
        default=None,
        help="Iteration numbers or iter_* names to include for omni-rl sources. For HF sources, this optionally sets the single destination iteration name; otherwise it defaults to 1."
    )
    parser.add_argument(
        "--tp-folder",
        default="tp_1_hf",
        help="Checkpoint subdirectory to create for omni-rl or HF eval layouts (default: tp_1_hf)"
    )
    parser.add_argument(
        "--ignore-folders",
        nargs="+",
        default=['logs', 'wandb', 'dataloader'],
        help="Additional directory names to ignore"
    )
    parser.add_argument(
        "--ignore-files",
        nargs="+",
        default=[],
        help="File names to ignore (e.g. config.yaml config.json)"
    )
    parser.add_argument(
        "--symlink-dirs", "-s",
        nargs="+",
        default=['tensorboard', 'code', 'code_snapshot'],
        help="Directory names to symlink entirely instead of recursing into (default: tensorboard, code, code_snapshot)"
    )
    parser.add_argument(
        "--symlink-children-of",
        nargs="+",
        default=['checkpoints'],
        help="Directory names whose immediate subdirectories should be symlinked entirely "
             "(default: checkpoints). E.g. checkpoint iteration dirs under checkpoints/ "
             "will be symlinked as whole directories."
    )
    parser.add_argument(
        "--except-children",
        nargs="+",
        default=['tp_1', 'tp_1_hf'],
        help="Subdirectory names to exclude from --symlink-children-of; these will be walked "
             "into normally (default: tp_1, tp_1_hf)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each operation"
    )

    args = parser.parse_args()

    input_src = args.input_src
    if args.layout == "hf_eval":
        input_src = "omni-rl"
    elif args.hf_checkpoint:
        input_src = "hf"

    if input_src in {"omni-rl", "hf"}:
        symlink_hf_checkpoint_layout(
            src=args.src,
            dest=args.dest,
            input_src=input_src,
            iterations=args.iters,
            tp_folder=args.tp_folder,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    else:
        symlink_files(
            src=args.src,
            dest=args.dest,
            ignore_dirs=args.ignore_folders,
            ignore_files=args.ignore_files,
            symlink_dirs=args.symlink_dirs,
            symlink_children_of=args.symlink_children_of,
            except_children=args.except_children,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
