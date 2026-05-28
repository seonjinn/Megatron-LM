#!/usr/bin/env python3
"""
Script to copy a model checkpoint and modify the vision_model.radio_model.model.patch_generator.cls_token.token
weight from shape (16, 1280) to (8, 1280) by taking the last 8 rows.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def modify_cls_token_in_safetensors(input_file: str, output_file: str, target_key: str):
    """
    Modify a specific tensor in a safetensors file.

    Args:
        input_file: Path to input safetensors file
        output_file: Path to output safetensors file
        target_key: Key of the tensor to modify
    """
    print(f"Processing {input_file}")

    # Load all tensors from the input file
    tensors = {}
    metadata = {}

    with safe_open(input_file, framework="pt", device="cpu") as f:
        # Copy metadata if available
        if hasattr(f, 'metadata'):
            metadata = f.metadata()

        # Load all tensors
        for key in f.keys():
            tensor = f.get_tensor(key)

            if key == target_key:
                print(f"Found target tensor '{key}' with shape {tensor.shape}")

                # Verify the tensor has the expected shape
                if tensor.shape[0] != 16 or tensor.shape[1] != 1280:
                    raise ValueError(f"Expected tensor shape (16, 1280), but got {tensor.shape}")

                # Take the last 8 rows
                modified_tensor = tensor[-8:]  # This takes the last 8 rows: tensor[8:16]
                print(f"Modified tensor shape: {modified_tensor.shape}")
                tensors[key] = modified_tensor
            else:
                tensors[key] = tensor

    # Save modified tensors to output file
    save_file(tensors, output_file, metadata=metadata)
    print(f"Saved modified file to {output_file}")


def copy_and_modify_checkpoint(source_dir: str, target_dir: str, target_weight: str = "vision_model.radio_model.model.patch_generator.cls_token.token"):
    """
    Copy a model checkpoint and modify the specified weight.

    Args:
        source_dir: Source checkpoint directory
        target_dir: Target checkpoint directory
        target_weight: Name of the weight to modify
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if target_path.exists():
        print(f"Warning: Target directory already exists: {target_dir}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(target_path)

    print(f"Creating target directory: {target_dir}")
    target_path.mkdir(parents=True, exist_ok=True)

    # Read the index file to determine which shard contains the target weight
    index_file = source_path / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    with open(index_file, 'r') as f:
        index_data = json.load(f)

    # Find which shard contains the target weight
    weight_map = index_data.get("weight_map", {})
    target_shard = weight_map.get(target_weight)

    if not target_shard:
        raise ValueError(f"Target weight '{target_weight}' not found in weight map")

    print(f"Target weight '{target_weight}' found in shard: {target_shard}")

    # Copy all files, modifying the target shard
    for item in source_path.iterdir():
        if item.is_file():
            target_file = target_path / item.name

            if item.name == target_shard:
                # This is the shard containing our target weight - modify it
                print(f"Modifying shard: {item.name}")
                modify_cls_token_in_safetensors(str(item), str(target_file), target_weight)
            else:
                # Regular copy for other files
                print(f"Copying: {item.name}")
                shutil.copy2(item, target_file)
        elif item.is_dir():
            # Copy directories recursively
            print(f"Copying directory: {item.name}")
            shutil.copytree(item, target_path / item.name)

    print(f"\nSuccessfully copied and modified checkpoint to: {target_dir}")
    print(f"Modified weight: {target_weight}")
    print(f"Shape changed from (16, 1280) to (8, 1280) by taking the last 8 rows")


def main():
    parser = argparse.ArgumentParser(
        description="Copy model checkpoint and modify cls_token tensor"
    )
    parser.add_argument(
        "source_dir",
        help="Source checkpoint directory"
    )
    parser.add_argument(
        "target_dir",
        help="Target checkpoint directory"
    )
    parser.add_argument(
        "--target-weight",
        default="vision_model.radio_model.model.patch_generator.cls_token.token",
        help="Name of the weight tensor to modify (default: vision_model.radio_model.model.patch_generator.cls_token.token)"
    )

    args = parser.parse_args()

    try:
        copy_and_modify_checkpoint(args.source_dir, args.target_dir, args.target_weight)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())