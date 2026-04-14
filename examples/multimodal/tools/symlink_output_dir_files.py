#!/usr/bin/env python3
"""
Script to create symbolic links for all files from a source directory to a destination directory.
Creates actual directories (not symlinks) so you can write to them, while symlinking files for reading.
"""

import argparse
import os
from pathlib import Path
from typing import Optional


def symlink_files(
    src: str,
    dest: str,
    ignore_dirs: Optional[list[str]] = None,
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
    if symlink_dirs is None:
        symlink_dirs = ['tensorboard']
    if symlink_children_of is None:
        symlink_children_of = ['checkpoints']
    if except_children is None:
        except_children = ['tp_1']

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

        # Create symlinks for all files
        dir_files_linked = 0
        dir_files_skipped = 0
        for file in files:
            src_file = Path(root) / file
            dest_file = dest_subdir / file

            try:
                dest_already = dest_file.exists() or dest_file.is_symlink()
            except OSError as e:
                # e.g. PermissionError on stat() for paths we cannot traverse (Lustre/NFS, stale perms)
                if verbose or dry_run:
                    print(f"Skipping dest (cannot stat): {dest_file}: {e}")
                files_skipped += 1
                dir_files_skipped += 1
                continue
            if dest_already:
                files_skipped += 1
                dir_files_skipped += 1
                continue

            if not dry_run:
                dest_file.symlink_to(src_file)
            files_linked += 1
            dir_files_linked += 1

        # Print per-directory summary
        if verbose or dry_run:
            if dir_files_linked > 0 or dir_files_skipped > 0:
                parts = []
                if dir_files_linked > 0:
                    parts.append(f"{dir_files_linked} files linked")
                if dir_files_skipped > 0:
                    parts.append(f"{dir_files_skipped} skipped")
                print(f"{rel_path or '.'}: {', '.join(parts)}")

    print(f"\nSummary:")
    print(f"  Directories created: {dirs_created}")
    print(f"  Directories symlinked: {dirs_linked}")
    print(f"  Files linked: {files_linked}")
    print(f"  Directories skipped (already exist): {dirs_skipped}")
    print(f"  Files skipped (already exist): {files_skipped}")
    print(f"  Ignored directories: {ignore_dirs}")
    print(f"  Symlinked directories: {symlink_dirs}")
    print(f"  Symlink children of: {symlink_children_of} (except {except_children})")

    if dry_run:
        print("\n(Dry run - no changes were made)")


def main():
    parser = argparse.ArgumentParser(
        description="Create symbolic links for files from source to destination directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --src /path/to/source --dest /path/to/dest
  %(prog)s --src /path/to/source --dest /path/to/dest --dry-run --verbose
        """
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
        "--ignore", "-i",
        nargs="+",
        default=['logs', 'wandb', 'dataloader'],
        help="Directory names to ignore completely (default: logs, wandb)"
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
        default=['tp_1'],
        help="Subdirectory names to exclude from --symlink-children-of; these will be walked "
             "into normally (default: tp_1)"
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

    symlink_files(
        src=args.src,
        dest=args.dest,
        ignore_dirs=args.ignore,
        symlink_dirs=args.symlink_dirs,
        symlink_children_of=args.symlink_children_of,
        except_children=args.except_children,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
