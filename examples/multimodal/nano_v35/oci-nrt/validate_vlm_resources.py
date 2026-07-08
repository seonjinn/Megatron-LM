#!/usr/bin/env python3
"""Validate copied Nano v3.5 VLM resources before submitting bootstrap jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file is missing or empty: {path}")


def validate_tokenizer(tokenizer: Path) -> int:
    required = (
        tokenizer / "tokenizer.json",
        tokenizer / "tokenizer_config.json",
        tokenizer / "special_tokens_map.json",
        tokenizer / "chat_template.jinja",
        tokenizer / "config.json",
    )
    for path in required:
        require_file(path)

    tokenizer_json = json.loads((tokenizer / "tokenizer.json").read_text(encoding="utf-8"))
    model_config = json.loads((tokenizer / "config.json").read_text(encoding="utf-8"))
    if "model" not in tokenizer_json:
        raise ValueError(f"Tokenizer model payload is missing from {tokenizer / 'tokenizer.json'}")
    if model_config.get("pad_token_id") is None:
        raise ValueError(f"pad_token_id is missing from {tokenizer / 'config.json'}")
    return int(model_config["pad_token_id"])


def validate_hf_checkpoint(checkpoint: Path) -> int:
    index_path = checkpoint / "model.safetensors.index.json"
    require_file(index_path)
    require_file(checkpoint / "config.json")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError(f"No checkpoint shards are listed in {index_path}")
    for shard in shards:
        require_file(checkpoint / shard)
    return len(shards)


def validate_radio_checkpoint(checkpoint: Path) -> int:
    latest_path = checkpoint / "latest_checkpointed_iteration.txt"
    require_file(latest_path)
    iteration = int(latest_path.read_text(encoding="utf-8").strip())
    iteration_dir = checkpoint / f"iter_{iteration:07d}"
    rank_files = sorted(iteration_dir.glob("mp_rank_*/model_optim_rng.pt"))
    if len(rank_files) != 2:
        raise ValueError(f"Expected two RADIO TP rank files in {iteration_dir}, found {len(rank_files)}")
    for rank_file in rank_files:
        require_file(rank_file)
    return iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--hf-checkpoint", required=True, type=Path)
    parser.add_argument("--radio-checkpoint", required=True, type=Path)
    parser.add_argument("--container", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_file(args.container)
    pad_token_id = validate_tokenizer(args.tokenizer)
    shard_count = validate_hf_checkpoint(args.hf_checkpoint)
    radio_iteration = validate_radio_checkpoint(args.radio_checkpoint)

    print(f"validated container: {args.container}")
    print(f"validated standalone tokenizer: {args.tokenizer} (pad_token_id={pad_token_id})")
    print(f"validated HF checkpoint: {args.hf_checkpoint} ({shard_count} shards)")
    print(f"validated RADIO checkpoint: {args.radio_checkpoint} (iteration {radio_iteration}, TP=2)")


if __name__ == "__main__":
    main()
