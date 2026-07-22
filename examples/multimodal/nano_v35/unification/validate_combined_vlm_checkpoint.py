#!/usr/bin/env python3
"""Validate a Nano v3.5 TP2/EP32 checkpoint assembled with RADIO weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def language_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key if key.startswith("language_model.") else f"language_model.{key}": value
        for key, value in state["model"].items()
    }


def vision_state(state: dict[str, Any]) -> dict[str, Any]:
    prefixed = {
        key: value
        for key, value in state["model"].items()
        if key.startswith("vision_model.") or key.startswith("vision_projection.")
    }
    if prefixed:
        return prefixed
    return {f"vision_model.{key}": value for key, value in state["model"].items()}


def assert_value_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"{name}: expected tensor, found {type(actual).__name__}")
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                f"{name}: metadata mismatch: "
                f"actual={actual.shape}/{actual.dtype}, expected={expected.shape}/{expected.dtype}"
            )
        if not torch.equal(actual, expected):
            raise ValueError(f"{name}: tensor contents differ")
    elif actual != expected:
        raise ValueError(f"{name}: non-tensor value differs")


def validate_rank(lm_dir: Path, radio_dir: Path, combined_dir: Path, rank: str) -> None:
    tp_rank = int(rank.split("_")[2])
    lm = load_checkpoint(lm_dir / rank / "model_optim_rng.pt")
    radio = load_checkpoint(radio_dir / f"mp_rank_{tp_rank:02d}" / "model_optim_rng.pt")
    combined = load_checkpoint(combined_dir / rank / "model_optim_rng.pt")

    expected = language_state(lm)
    expected.update(vision_state(radio))
    actual = combined["model"]
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"{rank}: model-key mismatch; missing={missing[:10]}, extra={extra[:10]}")

    for key, value in expected.items():
        assert_value_equal(f"{rank}:{key}", actual[key], value)

    language_keys = sum(key.startswith("language_model.") for key in actual)
    vision_keys = sum(key.startswith("vision_model.") for key in actual)
    projection_keys = sum(key.startswith("vision_projection.") for key in actual)
    print(
        f"validated {rank}: language_keys={language_keys}, "
        f"vision_keys={vision_keys}, projection_keys={projection_keys}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm", required=True, type=Path)
    parser.add_argument("--radio", required=True, type=Path)
    parser.add_argument("--combined", required=True, type=Path)
    args = parser.parse_args()

    for root in (args.lm, args.radio, args.combined):
        latest = root / "latest_checkpointed_iteration.txt"
        if latest.read_text(encoding="utf-8").strip() != "1":
            raise ValueError(f"Expected iteration 1 in {latest}")

    lm_iter = args.lm / "iter_0000001"
    radio_iter = args.radio / "iter_0000001"
    combined_iter = args.combined / "iter_0000001"
    expected_ranks = {
        f"mp_rank_{tp:02d}_{ep:03d}" for tp in range(2) for ep in range(32)
    }
    actual_ranks = {
        path.name
        for path in combined_iter.iterdir()
        if path.is_dir() and (path / "model_optim_rng.pt").is_file()
    }
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"Rank layout mismatch; missing={sorted(expected_ranks - actual_ranks)}, "
            f"extra={sorted(actual_ranks - expected_ranks)}"
        )

    # Validate an endpoint from each tensor-parallel rank byte-for-byte at the
    # tensor level. Rank-layout checks above cover all 64 EP shards.
    for rank in ("mp_rank_00_000", "mp_rank_01_031"):
        validate_rank(lm_iter, radio_iter, combined_iter, rank)

    print(f"validated combined checkpoint: {args.combined} (TP=2, EP=32, ranks=64)")


if __name__ == "__main__":
    main()
