#!/usr/bin/env python3
"""
Merge vision model components from one checkpoint with other components from another checkpoint.

This script combines:
- vision_model.* and vision_projection.* tensors from the source checkpoint
- All other tensors from the target checkpoint
"""

import os
import sys
import torch
import argparse
import shutil
from pathlib import Path
from typing import Dict, List


def get_rank_folders(checkpoint_path: str) -> List[str]:
    """
    Get all mp_rank folders in a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint iteration directory

    Returns:
        List of rank folder names (e.g., ['mp_rank_00_000', 'mp_rank_00_002', ...])
    """
    checkpoint_dir = Path(checkpoint_path)
    rank_folders = sorted([
        d.name for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith('mp_rank_')
    ])
    return rank_folders


def merge_checkpoint_rank(
    vision_ckpt_path: str,
    base_ckpt_path: str,
    output_ckpt_path: str,
    rank_folder: str,
    verbose: bool = False
) -> None:
    """
    Merge a single rank's checkpoint file.

    Args:
        vision_ckpt_path: Path to checkpoint with vision components to use
        base_ckpt_path: Path to checkpoint with base components to use
        output_ckpt_path: Path to save merged checkpoint
        rank_folder: Name of the rank folder for base and output (e.g., 'mp_rank_00_000')
        verbose: Print detailed information
    """
    vision_rank_path = Path(vision_ckpt_path) / rank_folder / "model_optim_rng.pt"
    base_rank_path = Path(base_ckpt_path) / rank_folder / "model_optim_rng.pt"
    output_rank_path = Path(output_ckpt_path) / rank_folder / "model_optim_rng.pt"

    # Create output directory
    output_rank_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nProcessing rank: {rank_folder}")
        print(f"  Vision source: {vision_rank_path}")
        print(f"  Base source: {base_rank_path}")
        print(f"  Output: {output_rank_path}")

    # Load checkpoints
    print(f"Loading vision checkpoint for {rank_folder}...")
    vision_ckpt = torch.load(vision_rank_path, map_location='cpu', weights_only=False)

    print(f"Loading base checkpoint for {rank_folder}...")
    base_ckpt = torch.load(base_rank_path, map_location='cpu', weights_only=False)

    # Start with base checkpoint
    merged_ckpt = base_ckpt

    # Extract vision model keys from vision checkpoint
    vision_model_keys = [k for k in vision_ckpt['model'].keys()
                         if k.startswith('vision_model.') or k.startswith('vision_projection.')]

    # Extract non-vision keys from base checkpoint
    non_vision_keys = [k for k in base_ckpt['model'].keys()
                       if not k.startswith('vision_model.') and not k.startswith('vision_projection.')]

    if verbose:
        print(f"  Vision keys to copy: {len(vision_model_keys)}")
        print(f"  Base keys to keep: {len(non_vision_keys)}")

    # Replace vision components in merged checkpoint
    vision_keys_copied = 0
    vision_keys_not_in_base = 0

    for key in vision_model_keys:
        if key in vision_ckpt['model']:
            merged_ckpt['model'][key] = vision_ckpt['model'][key]
            vision_keys_copied += 1
            if key not in base_ckpt['model']:
                vision_keys_not_in_base += 1

    if verbose:
        print(f"  Copied {vision_keys_copied} vision keys")
        if vision_keys_not_in_base > 0:
            print(f"  Added {vision_keys_not_in_base} new vision keys not in base")

    # Save merged checkpoint
    print(f"Saving merged checkpoint for {rank_folder}...")
    torch.save(merged_ckpt, output_rank_path)

    if verbose:
        print(f"  Successfully saved to {output_rank_path}")


def merge_vision_checkpoints(
    vision_ckpt_path: str,
    base_ckpt_path: str,
    output_ckpt_path: str,
    verbose: bool = False
) -> None:
    """
    Merge vision model from one checkpoint with base model from another.

    This function:
    1. Takes vision_model.* and vision_projection.* from vision_ckpt_path
    2. Takes all other components from base_ckpt_path
    3. Saves merged checkpoint to output_ckpt_path

    Args:
        vision_ckpt_path: Path to checkpoint iteration with vision components to use
        base_ckpt_path: Path to checkpoint iteration with base components to use
        output_ckpt_path: Path to save merged checkpoint iteration
        verbose: Print detailed information
    """
    print(f"\n{'='*80}")
    print(f"Merging Vision Checkpoints")
    print(f"{'='*80}")
    print(f"Vision source: {vision_ckpt_path}")
    print(f"Base source: {base_ckpt_path}")
    print(f"Output: {output_ckpt_path}")
    print(f"{'='*80}\n")

    # Get rank folders from base checkpoint (which should have all ranks)
    base_rank_folders = get_rank_folders(base_ckpt_path)
    print(f"Found {len(base_rank_folders)} rank folders in base checkpoint")

    # Get rank folders from vision checkpoint
    vision_rank_folders = get_rank_folders(vision_ckpt_path)
    print(f"Found {len(vision_rank_folders)} rank folders in vision checkpoint")

    # Process each rank
    for rank_folder in base_rank_folders:
        merge_checkpoint_rank(
            vision_ckpt_path=vision_ckpt_path,
            base_ckpt_path=base_ckpt_path,
            output_ckpt_path=output_ckpt_path,
            rank_folder=rank_folder,
            verbose=verbose
        )

    # Copy any additional files (like latest_checkpointed_iteration.txt)
    for item in Path(base_ckpt_path).parent.iterdir():
        if item.is_file():
            output_file = Path(output_ckpt_path).parent / item.name
            if not output_file.exists():
                shutil.copy2(item, output_file)
                if verbose:
                    print(f"Copied {item.name} to output directory")

    print(f"\n{'='*80}")
    print(f"Merge complete! Output saved to: {output_ckpt_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge vision model from one checkpoint with base model from another",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python merge_vision_checkpoints.py \\
    --vision-ckpt /path/to/vision/checkpoints/iter_0005253 \\
    --base-ckpt /path/to/base/checkpoints/iter_0016542 \\
    --output-ckpt /path/to/output/checkpoints/iter_0000001 \\
    --verbose

This will create a new checkpoint at the output path with:
  - vision_model.* and vision_projection.* from the vision checkpoint
  - All other components from the base checkpoint
        """
    )

    parser.add_argument(
        '--vision-ckpt',
        type=str,
        required=True,
        help='Path to checkpoint iteration with vision components to use'
    )

    parser.add_argument(
        '--base-ckpt',
        type=str,
        required=True,
        help='Path to checkpoint iteration with base components to use'
    )

    parser.add_argument(
        '--output-ckpt',
        type=str,
        required=True,
        help='Path to save merged checkpoint iteration'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during merge'
    )

    args = parser.parse_args()

    merge_vision_checkpoints(
        vision_ckpt_path=args.vision_ckpt,
        base_ckpt_path=args.base_ckpt,
        output_ckpt_path=args.output_ckpt,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
