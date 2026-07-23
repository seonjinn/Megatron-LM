#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _rank_dirs(iter_dir: Path) -> list[Path]:
    ranks = sorted(p for p in iter_dir.iterdir() if p.is_dir() and p.name.startswith("mp_rank_"))
    if not ranks:
        raise FileNotFoundError(f"No mp_rank_* directories found in {iter_dir}")
    return ranks


def _tp_rank(rank_name: str) -> int:
    parts = rank_name.split("_")
    if len(parts) < 3 or parts[0] != "mp" or parts[1] != "rank":
        raise ValueError(f"Unexpected rank directory name: {rank_name}")
    return int(parts[2])


def _vision_rank_dir(vision_iter_dir: Path, lm_rank_name: str, tp_rank: int) -> Path:
    candidates = [
        vision_iter_dir / lm_rank_name,
        vision_iter_dir / f"mp_rank_{tp_rank:02d}",
        vision_iter_dir / f"mp_rank_{tp_rank:02d}_000",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No RADIO vision rank directory found for TP rank {tp_rank} under {vision_iter_dir}"
    )


def _language_model_state(state: dict) -> dict:
    model = state["model"]
    prefixed = {key: value for key, value in model.items() if key.startswith("language_model.")}
    if prefixed:
        return prefixed
    return {f"language_model.{key}": value for key, value in model.items()}


def _vision_model_state(state: dict) -> dict:
    prefixed = {
        key: value
        for key, value in state["model"].items()
        if key.startswith("vision_model.") or key.startswith("vision_projection.")
    }
    if prefixed:
        return prefixed
    return {f"vision_model.{key}": value for key, value in state["model"].items()}


def combine(
    lm_dir: Path,
    vision_dir: Path,
    output_dir: Path,
    lm_iteration: int,
    vision_iteration: int,
    output_iteration: int,
) -> None:
    lm_iter_dir = lm_dir / f"iter_{lm_iteration:07d}"
    vision_iter_dir = vision_dir / f"iter_{vision_iteration:07d}"
    output_iter_dir = output_dir / f"iter_{output_iteration:07d}"

    if not lm_iter_dir.is_dir():
        raise FileNotFoundError(f"LM iteration directory does not exist: {lm_iter_dir}")
    if not vision_iter_dir.is_dir():
        raise FileNotFoundError(f"Vision iteration directory does not exist: {vision_iter_dir}")

    for lm_rank_dir in _rank_dirs(lm_iter_dir):
        tp_rank = _tp_rank(lm_rank_dir.name)
        vision_rank_dir = _vision_rank_dir(vision_iter_dir, lm_rank_dir.name, tp_rank)

        lm_path = lm_rank_dir / "model_optim_rng.pt"
        vision_path = vision_rank_dir / "model_optim_rng.pt"
        output_path = output_iter_dir / lm_rank_dir.name / "model_optim_rng.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Combining {lm_rank_dir.name}:")
        print(f"  LM:     {lm_path}")
        print(f"  RADIO:  {vision_path}")
        print(f"  Output: {output_path}")

        lm_state = torch.load(lm_path, map_location="cpu", weights_only=False)
        vision_state = torch.load(vision_path, map_location="cpu", weights_only=False)

        combined = lm_state.copy()
        combined["model"] = {}
        combined["model"].update(_language_model_state(lm_state))
        combined["model"].update(_vision_model_state(vision_state))

        torch.save(combined, output_path)

    latest = output_dir / "latest_checkpointed_iteration.txt"
    latest.write_text(f"{output_iteration}\n", encoding="utf-8")
    print(f"Wrote {latest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine nano v3.5 LLM shards with RADIO TP shards")
    parser.add_argument("--lm-dir", required=True, type=Path, help="Top-level Megatron LM checkpoint dir")
    parser.add_argument(
        "--vision-dir",
        required=True,
        type=Path,
        help="Top-level RADIO or VLM checkpoint dir to source vision weights from",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Top-level output checkpoint dir")
    parser.add_argument("--iteration", default=None, type=int, help="Backward-compatible iteration for all inputs")
    parser.add_argument("--lm-iteration", default=None, type=int, help="LM checkpoint iteration")
    parser.add_argument("--vision-iteration", default=None, type=int, help="Vision checkpoint iteration")
    parser.add_argument("--output-iteration", default=None, type=int, help="Output checkpoint iteration")
    args = parser.parse_args()

    default_iteration = args.iteration or 1
    combine(
        args.lm_dir,
        args.vision_dir,
        args.output_dir,
        args.lm_iteration or default_iteration,
        args.vision_iteration or default_iteration,
        args.output_iteration or default_iteration,
    )


if __name__ == "__main__":
    main()
