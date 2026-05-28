#!/usr/bin/env python3
"""Report per-sub-dataset sample counts, average tokens, and total tokens for a recipe YAML.

Two modes
---------
**Sample-count-only** (fast, no Megatron/GPU dependencies):

    python examples/multimodal/tools/datasets/dataset_stats.py recipe.yaml

**Token estimation** (needs tokenizer, CPU-only via torchrun):

    torchrun --nproc_per_node=1 examples/multimodal/tools/datasets/dataset_stats.py \\
        recipe.yaml --estimate-tokens --samples-per-dataset 50 \\
        --training-script examples/multimodal/v3_omni_staged_conv3d/sft_long_context_0318.sh

    The --training-script flag automatically extracts all Megatron args (tokenizer,
    vision config, sequence lengths, etc.) from the training shell script via its
    DRY_RUN mode.  You can still override individual args on the command line.

    Without --training-script you must pass the args manually:

    torchrun --nproc_per_node=1 examples/multimodal/tools/datasets/dataset_stats.py \\
        recipe.yaml --estimate-tokens --target-relative-precision 0.01 --confidence 0.99 \\
        --samples-per-dataset 100000 --num-workers 8 --cache-pool-workers 1 \\
        --dataset-parallelism 1 --training-script training_script.sh --progress-update-seconds 1

**Adaptive confidence-based token estimation**:

    torchrun --nproc_per_node=1 examples/multimodal/tools/datasets/dataset_stats.py \\
        recipe.yaml --estimate-tokens --samples-per-dataset 1000 \\
        --target-relative-precision 0.05 --confidence 0.95 \\
        --training-script examples/multimodal/v3_omni_staged_conv3d/sft_long_context_0318.sh

    This stops sampling each dataset early once the estimated total-token count
    reaches +/-5% relative precision at 95% confidence, or once
    --samples-per-dataset is reached.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import timedelta
from statistics import NormalDist
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import yaml


# ── Extract args from training script ────────────────────────────────────────


def _extract_args_from_script(script_path: str) -> List[str]:
    """Extract Megatron CLI args from a training shell script.

    Runs the script with ``DRY_RUN=1 INTERACTIVE=1`` which makes it print
    the full ``python train.py <OPTIONS>`` command and exit.  The printed
    args are then parsed and returned as a list of strings, with training-only
    args (checkpointing, parallelism, optimizer, etc.) stripped out.
    """
    import shlex
    import subprocess

    script_path = os.path.abspath(script_path)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")

    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["INTERACTIVE"] = "1"
    # Provide dummy values for env vars the scripts typically require.
    env.setdefault("WANDB_API_KEY", "dummy")
    env.setdefault("WANDB_PROJECT", "dummy")
    env.setdefault("WANDB_ENTITY", "dummy")
    env.setdefault("SLURM_GPUS_ON_NODE", "8")
    env.setdefault("SLURM_JOB_USER", env.get("USER", "user"))
    env.setdefault("SLURM_NNODES", "1")

    result = subprocess.run(
        ["bash", script_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(script_path) or ".",
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Training script exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    # DRY_RUN prints a line like:
    #   python -u /path/to/train.py --arg1 val1 --arg2 val2 ...
    raw_tokens = None
    for line in result.stdout.strip().splitlines():
        if "train.py" in line:
            idx = line.index("train.py") + len("train.py")
            args_str = line[idx:]
            try:
                raw_tokens = shlex.split(args_str)
            except ValueError:
                raw_tokens = args_str.split()
            break

    if raw_tokens is None:
        raise RuntimeError(
            f"Could not find 'train.py' command in DRY_RUN output.\n"
            f"stdout:\n{result.stdout[-2000:]}"
        )

    # Filter out args that are irrelevant or harmful for token estimation.
    # These are exact flags or prefixes; a flag matches if it equals a key
    # or starts with a prefix that ends with "-".
    _SKIP_EXACT = {
        "--use-checkpoint-args",
        "--load", "--save", "--pretrained-checkpoint", "--dataloader-save",
        "--data-path", "--prompt-path",
        "--transformer-impl", "--use-te", "--bf16", "--fp16",
        "--use-distributed-optimizer", "--ckpt-format",
        "--tensor-model-parallel-size", "--pipeline-model-parallel-size",
        "--expert-model-parallel-size", "--expert-tensor-parallel-size",
        "--context-parallel-size",
        "--global-batch-size", "--micro-batch-size",
        "--train-full-dataset",
        "--num-workers",
        "--tensorboard-dir",
        "--log-interval", "--eval-iters", "--eval-interval", "--save-interval",
        "--exit-duration-in-mins",
        "--distributed-timeout-minutes",
        "--dataloader-type",
        "--lr", "--min-lr", "--lr-decay-style", "--lr-warmup-fraction",
        "--weight-decay", "--clip-grad",
        "--adam-beta1", "--adam-beta2",
        "--use-loss-scaling",
        "--attention-dropout", "--hidden-dropout",
        "--num-experts",
        "--sequence-parallel",
        "--freeze-sound-model", "--freeze-sound-projection",
        "--freeze-LM", "--freeze-vision-model",
        "--enable-experimental",
    }
    _SKIP_PREFIXES = (
        "--wandb-",
        "--recompute-",
        "--fp8-",
        "--moe-",
        "--app-tag-",
        "--first-last-layers-",
        "--num-layers-at-",
    )

    def _should_skip(flag: str) -> bool:
        if flag in _SKIP_EXACT:
            return True
        return any(flag.startswith(p) for p in _SKIP_PREFIXES)

    filtered: List[str] = []
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if tok.startswith("--") and _should_skip(tok):
            # Skip flag and all its non-flag values (handles multi-value args
            # like --recompute-modules core_attn mlp layernorm moe_act moe).
            i += 1
            while i < len(raw_tokens) and not raw_tokens[i].startswith("--"):
                i += 1
            continue
        filtered.append(tok)
        i += 1

    return filtered


# ── YAML parsing (reused from trace_dataset_yaml) ───────────────────────────


def _resolve_path(raw_path: str, yaml_dir: str) -> str:
    """Resolve *raw_path* relative to *yaml_dir* if it is not absolute."""
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.normpath(os.path.join(yaml_dir, raw_path))


def _is_yaml_path(path: str) -> bool:
    """Heuristic: does this path reference another YAML file?"""
    return path.endswith(".yaml") or path.endswith(".yml")


@dataclass
class TopLevelEntry:
    """One entry from the top-level recipe's blend_epochized list."""

    path: str  # resolved absolute path
    raw_path: str  # as written in the YAML
    repetitions: float
    subflavors: dict = field(default_factory=dict)
    aux: dict = field(default_factory=dict)


@dataclass
class EntryStats:
    """Aggregated stats for one top-level entry."""

    entry: TopLevelEntry
    total_samples: int = 0
    avg_tokens: float = 0.0
    sampled_token_count: int = 0
    token_ci_half_width: float = 0.0
    token_relative_precision: float = float("inf")
    token_precision_target_met: bool = False
    from_cache: bool = False


@dataclass
class RunningStats:
    """Online mean/variance tracker for sampled token lengths."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    total: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)


@dataclass
class TokenEstimate:
    """Approximate total-token estimate for one dataset."""

    sampled_count: int
    avg_tokens: float
    est_total_tokens: float
    ci_half_width: float
    relative_precision: float
    confidence: float
    exact: bool = False


@dataclass
class EntryRunProgress:
    """Live progress for one currently running entry."""

    name: str
    start_time: float
    sampled_count: int = 0
    sample_budget: int | None = None
    total_samples: int | None = None
    relative_precision: float | None = None


_STATS_CACHE_VERSION = 1


# ── Parse top-level recipe ──────────────────────────────────────────────────


def parse_top_level_entries(
    yaml_path: str, split: str = "train"
) -> List[TopLevelEntry]:
    """Parse the top-level recipe YAML and return its blend entries.

    Only reads the top level — does NOT recurse into nested YAMLs.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML structure in {yaml_path}")

    splits = data.get("splits", {})
    split_cfg = splits.get(split)
    if not split_cfg or not isinstance(split_cfg, dict):
        raise ValueError(f"Split '{split}' not found in {yaml_path}")

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

    entries: List[TopLevelEntry] = []
    for key in ("blend_epochized", "blend", "datasets"):
        blend_list = split_cfg.get(key)
        if not blend_list or not isinstance(blend_list, list):
            continue
        for item in blend_list:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path", "")
            if not raw_path:
                continue
            resolved = _resolve_path(raw_path, yaml_dir)
            repetitions = float(item.get("repetitions", item.get("weight", 1.0)))
            subflavors = item.get("subflavors") or {}
            aux = item.get("aux") or {}
            entries.append(
                TopLevelEntry(
                    path=resolved,
                    raw_path=raw_path,
                    repetitions=repetitions,
                    subflavors=subflavors,
                    aux=aux,
                )
            )
        break  # only process the first matching key

    return entries


# ── Lightweight task encoder (cookers only, no media loading) ────────────────


def _make_counting_encoder():
    """Create a minimal task encoder that has the right cookers/decoder
    for energon to recognise the crude datasets, but does no heavy processing.

    This avoids needing full Megatron init just for sample counting.
    """
    from typing import Any

    from megatron.energon import DefaultTaskEncoder, stateless
    from data_loading.task_encoder import MultiModalTaskEncoder
    from data_loading.conversation_sample import ConversationSample

    class CountingTaskEncoder(
        DefaultTaskEncoder[ConversationSample, Any, Any, dict]
    ):
        """Encoder that inherits cookers/decoder but skips all processing."""

        decoder = MultiModalTaskEncoder.decoder
        cookers = MultiModalTaskEncoder.cookers

        @stateless(restore_seeds=True)
        def encode_sample(self, sample: ConversationSample):
            return {"__key__": sample.__key__}

        def batch(self, samples):
            assert len(samples) == 1
            return samples[0]

    return CountingTaskEncoder()


# ── Wrapper YAML for jsonl entries ──────────────────────────────────────────


def _is_jsonl_path(path: str) -> bool:
    """Check if a path points to a jsonl file."""
    return path.endswith(".jsonl")


def _make_wrapper_yaml(entry: TopLevelEntry, tmp_dir: str) -> str:
    """Create a temporary MetadatasetV2 YAML that wraps a single jsonl entry.

    Energon's get_train_dataset needs a YAML metadataset to properly handle
    jsonl entries with subflavors (cook) and aux (media_source) metadata.
    When we break a recipe into individual entries for per-dataset stats,
    jsonl entries lose this context.  This function recreates it.
    """
    blend_entry = {"path": entry.path, "repetitions": 1.0}
    if entry.subflavors:
        blend_entry["subflavors"] = entry.subflavors
    if entry.aux:
        blend_entry["aux"] = entry.aux

    wrapper = {
        "__class__": "MetadatasetV2",
        "__module__": "megatron.energon",
        "splits": {
            "train": {
                "blend_epochized": [blend_entry],
            }
        },
    }

    wrapper_path = os.path.join(tmp_dir, "wrapper.yaml")
    with open(wrapper_path, "w") as f:
        yaml.dump(wrapper, f, default_flow_style=False)
    return wrapper_path


# ── Sample counting via energon ─────────────────────────────────────────────


def _count_samples_energon(
    entry: TopLevelEntry,
    num_workers: int = 0,
    cache_pool_workers: int = 1,
) -> int:
    """Use energon's get_train_dataset to count total samples for a single entry.

    Uses get_train_dataset (not get_val_datasets) so that inner repetitions
    within nested YAMLs are correctly applied to the total count.
    For jsonl entries, creates a temporary wrapper YAML with the required
    subflavors and aux metadata.
    """
    from megatron.energon import (
        FileStoreCachePool,
        WorkerConfig,
        get_savable_loader,
        get_train_dataset,
    )

    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=num_workers)

    if _is_jsonl_path(entry.path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper_path = _make_wrapper_yaml(entry, tmp_dir)
            dataset = get_train_dataset(
                wrapper_path,
                batch_size=1,
                task_encoder=_make_counting_encoder(),
                worker_config=worker_config,
                shuffle_buffer_size=None,
                max_samples_per_sequence=None,
                packing_buffer_size=None,
                repeat=False,
            )

            loader = get_savable_loader(
                dataset,
                watchdog_timeout_seconds=60,
                cache_pool=FileStoreCachePool(
                    num_workers=cache_pool_workers,
                    max_cache_size_gbytes=2,
                    method="raw",
                ),
            )
            total = len(loader)
            del loader, dataset
    else:
        dataset = get_train_dataset(
            entry.path,
            batch_size=1,
            task_encoder=_make_counting_encoder(),
            worker_config=worker_config,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=None,
            repeat=False,
        )

        loader = get_savable_loader(
            dataset,
            watchdog_timeout_seconds=60,
            cache_pool=FileStoreCachePool(
                num_workers=cache_pool_workers,
                max_cache_size_gbytes=2,
                method="raw",
            ),
        )
        total = len(loader)
        del loader, dataset

    gc.collect()
    return total


# ── Token estimation ────────────────────────────────────────────────────────


def _make_token_counting_encoder():
    """Create a task encoder that computes token counts without loading media.

    Inherits cookers and preencode_sample from MultiModalTaskEncoder (which
    computes total_len purely from metadata — image dimensions, video duration,
    audio duration, text tokenization).  Overrides postencode_sample to skip
    the expensive _load_media() call and just passes total_len through.

    If --sound-model-type is not provided (transform_audio is None), installs
    a lightweight fallback that estimates audio token counts from duration
    metadata so that audio samples don't crash.
    """
    import math

    from megatron.energon import stateless
    from megatron.training import get_args

    from megatron.core.models.multimodal.llava_model import SOUND_TOKEN

    from data_loading.audio_processing import AudioParams, AudioPreprocessingStrategy
    from data_loading.conversation_sample import AudioMedia
    from data_loading.task_encoder import MultiModalTaskEncoder

    class _FallbackAudioEstimator(AudioPreprocessingStrategy):
        """Estimate audio token count from duration metadata only (no model needed)."""

        def __init__(self, target_freq: int, embedding_size: int, clip_duration: int):
            self._target_freq = target_freq
            self._embedding_size = embedding_size
            self._clip_duration = clip_duration

        def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:
            params_list = []
            for media in media_list:
                num_samples = int((media.audio_duration - 0.1) * self._target_freq)
                clip_samples = int(self._clip_duration * self._target_freq)
                num_clips = max(1, math.ceil(num_samples / clip_samples))
                params_list.append(AudioParams(
                    num_embeddings=num_clips * self._embedding_size,
                    audio_length=num_clips * clip_samples,
                    num_clips=num_clips,
                    timestamps=(0, num_clips * self._clip_duration),
                    media=media,
                ))
            return params_list

        def apply_params(self, params: AudioParams):
            raise RuntimeError("apply_params should not be called in token-counting mode")

    class TokenCountingTaskEncoder(MultiModalTaskEncoder):
        """Runs preencode_sample (token counting) but skips media loading."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # If no sound model was configured, install a metadata-only fallback
            # so that preencode_sample doesn't crash on AudioMedia fragments.
            if self.transform_audio is None:
                args = get_args()
                self.transform_audio = _FallbackAudioEstimator(
                    target_freq=getattr(args, "sound_target_rate", 16000),
                    embedding_size=getattr(args, "sound_embedding_size", 750),
                    clip_duration=getattr(args, "sound_clip_duration", 30),
                )
                self.sound_token_id = self.tokenizer.convert_tokens_to_ids(SOUND_TOKEN)

        @stateless(restore_seeds=True)
        def postencode_sample(self, sample):
            """Skip media loading — we only need total_len from preencode_sample."""
            return {"total_len": sample.total_len, "__key__": sample.__key__}

        def batch(self, samples):
            assert len(samples) == 1
            return samples[0]

        def encode_batch(self, batch):
            return batch

    return TokenCountingTaskEncoder(is_val=True)


def _estimate_tokens_energon(
    entry: TopLevelEntry,
    max_samples: int,
    confidence: float,
    target_relative_precision: float | None = None,
    min_samples_for_precision: int = 30,
    num_workers: int = 0,
    cache_pool_workers: int = 1,
    progress_callback: Callable[[int, int, int, float | None], None] | None = None,
) -> Tuple[int, TokenEstimate]:
    """Count total samples and estimate avg tokens per sample for a single entry.

    Uses get_train_dataset so inner repetitions are applied correctly.
    Iterates sampled entries to estimate average token length via preencode_sample
    (metadata-only, no media loading).
    For jsonl entries, creates a temporary wrapper YAML with the required
    subflavors and aux metadata.
    Returns (total_samples, token_estimate).
    """
    from megatron.energon import (
        FileStoreCachePool,
        WorkerConfig,
        get_savable_loader,
        get_train_dataset,
    )

    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=num_workers)

    task_encoder = _make_token_counting_encoder()

    def _create_and_iterate(dataset_path: str):
        dataset = get_train_dataset(
            dataset_path,
            batch_size=1,
            task_encoder=task_encoder,
            worker_config=worker_config,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            packing_buffer_size=None,
            repeat=False,
        )

        loader = get_savable_loader(
            dataset,
            watchdog_timeout_seconds=120,
            cache_pool=FileStoreCachePool(
                num_workers=cache_pool_workers,
                max_cache_size_gbytes=8,
                method="raw",
            ),
        )
        total = len(loader)

        sample_budget = min(max_samples, total)
        token_stats = RunningStats()
        precision_floor = min(min_samples_for_precision, sample_budget)
        if progress_callback is not None:
            progress_callback(0, sample_budget, total, None)

        for batch in loader:
            if token_stats.count >= sample_budget:
                break
            tl = batch.get("total_len")
            if tl is not None:
                import torch

                if isinstance(tl, torch.Tensor):
                    token_stats.add(float(tl.item()))
                else:
                    token_stats.add(float(tl))

            current_relative_precision = None
            if (
                target_relative_precision is not None
                and token_stats.count >= precision_floor
            ):
                current_estimate = _estimate_total_tokens_from_sample(
                    total, token_stats, confidence
                )
                current_relative_precision = current_estimate.relative_precision
                if progress_callback is not None:
                    progress_callback(
                        token_stats.count,
                        sample_budget,
                        total,
                        current_relative_precision,
                    )
                if current_estimate.relative_precision <= target_relative_precision:
                    break
            elif progress_callback is not None:
                progress_callback(
                    token_stats.count,
                    sample_budget,
                    total,
                    current_relative_precision,
                )

        if token_stats.count == 0:
            raise RuntimeError("Failed to collect any token lengths from the dataset")

        estimate = _estimate_total_tokens_from_sample(total, token_stats, confidence)
        if progress_callback is not None:
            progress_callback(
                token_stats.count,
                sample_budget,
                total,
                estimate.relative_precision,
            )

        del loader, dataset
        return total, estimate

    if _is_jsonl_path(entry.path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper_path = _make_wrapper_yaml(entry, tmp_dir)
            total, estimate = _create_and_iterate(wrapper_path)
    else:
        total, estimate = _create_and_iterate(entry.path)

    gc.collect()
    return total, estimate


# ── Formatting helpers ──────────────────────────────────────────────────────


def _fmt_count(n: int) -> str:
    """Format a count with thousands separators."""
    return f"{n:,}"


def _fmt_tokens(n: float) -> str:
    """Format a token count in human-readable form (e.g. 53.3B, 1.6B, 850M)."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.0f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.0f}K"
    else:
        return str(int(n))


def _short_name(path: str) -> str:
    """Return just the filename from a path."""
    return os.path.basename(path)


def _validate_confidence(confidence: float) -> None:
    """Validate a two-sided confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"--confidence must be between 0 and 1, got {confidence}")


def _confidence_z_value(confidence: float) -> float:
    """Return the z-score for a two-sided normal confidence interval."""
    _validate_confidence(confidence)
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def _estimate_total_tokens_from_sample(
    population_size: int,
    token_stats: RunningStats,
    confidence: float,
) -> TokenEstimate:
    """Estimate total tokens with a normal CI and finite-population correction."""
    if population_size < 1:
        raise ValueError(f"population_size must be >= 1, got {population_size}")
    if token_stats.count < 1:
        raise ValueError("At least one sampled token length is required")

    sampled_count = token_stats.count
    avg_tokens = token_stats.mean
    est_total_tokens = population_size * avg_tokens

    if sampled_count >= population_size:
        return TokenEstimate(
            sampled_count=sampled_count,
            avg_tokens=avg_tokens,
            est_total_tokens=token_stats.total,
            ci_half_width=0.0,
            relative_precision=0.0,
            confidence=confidence,
            exact=True,
        )

    if sampled_count < 2:
        return TokenEstimate(
            sampled_count=sampled_count,
            avg_tokens=avg_tokens,
            est_total_tokens=est_total_tokens,
            ci_half_width=float("inf"),
            relative_precision=float("inf"),
            confidence=confidence,
            exact=False,
        )

    z_value = _confidence_z_value(confidence)
    fpc = math.sqrt((population_size - sampled_count) / (population_size - 1))
    se_mean = math.sqrt(token_stats.sample_variance / sampled_count) * fpc
    ci_half_width = population_size * z_value * se_mean

    if est_total_tokens == 0.0:
        relative_precision = 0.0 if ci_half_width == 0.0 else float("inf")
    else:
        relative_precision = ci_half_width / abs(est_total_tokens)

    return TokenEstimate(
        sampled_count=sampled_count,
        avg_tokens=avg_tokens,
        est_total_tokens=est_total_tokens,
        ci_half_width=ci_half_width,
        relative_precision=relative_precision,
        confidence=confidence,
        exact=False,
    )


def _compute_entry_stats(
    entry: TopLevelEntry,
    estimate_tokens: bool,
    samples_per_dataset: int,
    confidence: float,
    target_relative_precision: float | None,
    num_workers: int,
    cache_pool_workers: int,
    progress_callback: Callable[[int, int, int, float | None], None] | None = None,
) -> EntryStats:
    """Compute stats for a single top-level entry."""
    if estimate_tokens:
        total, token_estimate = _estimate_tokens_energon(
            entry,
            max_samples=samples_per_dataset,
            confidence=confidence,
            target_relative_precision=target_relative_precision,
            num_workers=num_workers,
            cache_pool_workers=cache_pool_workers,
            progress_callback=progress_callback,
        )
        return EntryStats(
            entry=entry,
            total_samples=total,
            avg_tokens=token_estimate.avg_tokens,
            sampled_token_count=token_estimate.sampled_count,
            token_ci_half_width=token_estimate.ci_half_width,
            token_relative_precision=token_estimate.relative_precision,
            token_precision_target_met=(
                False
                if target_relative_precision is None
                else token_estimate.relative_precision <= target_relative_precision
            ),
        )

    total = _count_samples_energon(
        entry,
        num_workers=num_workers,
        cache_pool_workers=cache_pool_workers,
    )
    return EntryStats(entry=entry, total_samples=total, avg_tokens=0.0)


def _entry_progress_suffix(stat: EntryStats) -> str:
    """Return a short stderr progress summary for a completed entry."""
    if stat.sampled_token_count == 0:
        return f"{_fmt_count(stat.total_samples)} samples"

    if stat.sampled_token_count >= stat.total_samples:
        precision_text = "exact"
    else:
        precision_text = f"{100 * stat.token_relative_precision:.1f}%"
    return (
        f"{_fmt_count(stat.total_samples)} samples, used {stat.sampled_token_count}, "
        f"precision {precision_text}"
    )


def _distributed_rank_info() -> tuple[int, int]:
    """Return (rank, world_size) when torch.distributed is initialized, else (0, 1)."""
    try:
        import torch
    except ImportError:
        return 0, 1

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds in a compact human-readable form."""
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _format_rank_progress(
    completed: int,
    total: int,
    active_entries: list[EntryRunProgress],
) -> str:
    """Return a compact per-rank progress summary."""
    if total <= 0:
        return "idle"

    bar_width = 20
    filled = min(bar_width, int(bar_width * completed / total))
    bar = "#" * filled + "-" * (bar_width - filled)
    summary = f"[{bar}] {completed}/{total} done"
    if active_entries:
        running_text = ", ".join(
            _format_active_entry_progress(progress)
            for progress in active_entries
        )
        summary += f"; running: {running_text}"
    return summary


def _format_active_entry_progress(progress: EntryRunProgress) -> str:
    """Format the progress text for one active entry."""
    elapsed = _format_duration(time.monotonic() - progress.start_time)
    parts = [f"{progress.name} ({elapsed})"]
    if progress.sample_budget is not None:
        parts.append(f"samples {progress.sampled_count}/{progress.sample_budget}")
    if progress.relative_precision is not None and math.isfinite(progress.relative_precision):
        parts.append(f"prec {100 * progress.relative_precision:.1f}%")
    return ", ".join(parts)


# ── Persistent per-entry cache ─────────────────────────────────────────────


def _default_cache_path() -> str:
    """Return the default persistent cache file path."""
    return str(Path(__file__).with_name("dataset_stats_cache.json"))


def _path_signature(path: str) -> dict:
    """Return a lightweight fingerprint of the entry path for cache invalidation."""
    try:
        stat_result = os.stat(path)
    except OSError:
        return {"exists": False}

    return {
        "exists": True,
        "is_dir": os.path.isdir(path),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _normalize_cached_float(value: float) -> float | str:
    """Encode non-finite floats into JSON-safe strings."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def _restore_cached_float(value: float | str) -> float:
    """Decode cached JSON-safe float payloads back to Python floats."""
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan":
        return float("nan")
    return float(value)


def _make_cache_key(
    entry: TopLevelEntry,
    estimate_tokens: bool,
    samples_per_dataset: int,
    confidence: float,
    target_relative_precision: float | None,
    megatron_argv: list[str],
) -> str:
    """Return a stable cache key for one top-level entry under the current settings."""
    key_payload = {
        "version": _STATS_CACHE_VERSION,
        "entry": {
            "path": os.path.abspath(entry.path),
            "subflavors": entry.subflavors or {},
            "aux": entry.aux or {},
            "path_signature": _path_signature(entry.path),
        },
        "mode": {
            "estimate_tokens": estimate_tokens,
            "samples_per_dataset": samples_per_dataset,
            "confidence": confidence,
            "target_relative_precision": target_relative_precision,
            "megatron_argv": megatron_argv if estimate_tokens else [],
        },
    }
    key_json = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()


def _serialize_entry_stats_for_cache(stat: EntryStats) -> dict:
    """Convert computed stats to a cacheable JSON payload."""
    return {
        "total_samples": stat.total_samples,
        "avg_tokens": stat.avg_tokens,
        "sampled_token_count": stat.sampled_token_count,
        "token_ci_half_width": _normalize_cached_float(stat.token_ci_half_width),
        "token_relative_precision": _normalize_cached_float(stat.token_relative_precision),
        "token_precision_target_met": stat.token_precision_target_met,
    }


def _deserialize_entry_stats_from_cache(entry: TopLevelEntry, payload: dict) -> EntryStats:
    """Rebuild EntryStats for the current entry from cached JSON data."""
    return EntryStats(
        entry=entry,
        total_samples=int(payload["total_samples"]),
        avg_tokens=float(payload["avg_tokens"]),
        sampled_token_count=int(payload["sampled_token_count"]),
        token_ci_half_width=_restore_cached_float(payload["token_ci_half_width"]),
        token_relative_precision=_restore_cached_float(payload["token_relative_precision"]),
        token_precision_target_met=bool(payload["token_precision_target_met"]),
        from_cache=True,
    )


def _load_stats_cache(cache_path: str) -> dict[str, dict]:
    """Load the persistent entry-stats cache from disk."""
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path) as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or payload.get("cache_version") != _STATS_CACHE_VERSION:
        return {}

    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    return entries


def _save_stats_cache(cache_path: str, new_entries: dict[str, dict]) -> int:
    """Merge and atomically persist freshly computed cache entries."""
    import fcntl

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    lock_path = f"{cache_path}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        merged_entries = {}
        if os.path.exists(cache_path):
            try:
                merged_entries = _load_stats_cache(cache_path)
            except (OSError, json.JSONDecodeError, ValueError):
                merged_entries = {}

        merged_entries.update(new_entries)

        payload = {
            "cache_version": _STATS_CACHE_VERSION,
            "entries": merged_entries,
        }
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        os.replace(tmp_path, cache_path)

    return len(new_entries)


# ── Report printing ─────────────────────────────────────────────────────────


def print_report(
    yaml_path: str,
    split: str,
    stats: List[EntryStats],
    show_tokens: bool,
    confidence: float,
    target_relative_precision: float | None = None,
):
    """Print the formatted report to stdout."""
    print(f"\nRecipe: {os.path.basename(yaml_path)} ({split} split)")
    print(f"  Path: {yaml_path}")

    if show_tokens:
        ci_label = f"CI@{confidence * 100:.0f}%"
        header = (
            f" {'#':>3}  {'Entry (nested YAML)':<45} {'Samples':>10}  {'Rep':>5}"
            f"  {'Eff.Samples':>12}  {'Used':>6}  {'Avg Tok':>8}"
            f"  {'Est. Tokens':>12}  {ci_label:>12}  {'Prec.':>7}"
        )
    else:
        header = f" {'#':>3}  {'Entry (nested YAML)':<45} {'Samples':>10}  {'Rep':>5}  {'Eff.Samples':>12}"

    sep_line = "\u2550" * len(header)
    dash_line = "\u2500" * len(header)

    print(sep_line)
    print(header)
    print(dash_line)

    grand_samples = 0
    grand_eff_samples = 0
    grand_tokens = 0.0
    grand_ci_half_width_sq = 0.0

    for idx, s in enumerate(stats, 1):
        name = _short_name(s.entry.path)
        rep = s.entry.repetitions
        eff = int(s.total_samples * rep)
        grand_samples += s.total_samples
        grand_eff_samples += eff

        if show_tokens and s.avg_tokens > 0:
            est_tok = eff * s.avg_tokens
            ci_half_width = rep * s.token_ci_half_width
            grand_tokens += est_tok
            grand_ci_half_width_sq += ci_half_width * ci_half_width
            if s.sampled_token_count >= s.total_samples:
                precision_text = " exact"
            else:
                precision_text = f"{100 * s.token_relative_precision:5.1f}%"
                if (
                    target_relative_precision is not None
                    and not s.token_precision_target_met
                ):
                    precision_text += "!"
            print(
                f" {idx:3d}  {name:<45} {_fmt_count(s.total_samples):>10}  {rep:5.3g}"
                f"  {_fmt_count(eff):>12}  {s.sampled_token_count:>6}"
                f"  ~{int(s.avg_tokens):>6}  {_fmt_tokens(est_tok):>12}"
                f"  ±{_fmt_tokens(ci_half_width):>11}  {precision_text:>7}"
            )
        else:
            print(
                f" {idx:3d}  {name:<45} {_fmt_count(s.total_samples):>10}  {rep:5.3g}  {_fmt_count(eff):>12}"
                + (
                    ""
                    if not show_tokens
                    else "  "
                    + " " * 6
                    + "  "
                    + " " * 8
                    + "  "
                    + " " * 12
                    + "  "
                    + " " * 12
                    + "  "
                    + " " * 7
                )
            )

    print(dash_line)
    if show_tokens and grand_tokens > 0:
        grand_ci_half_width = math.sqrt(grand_ci_half_width_sq)
        print(
            f" {'':3}  {'TOTAL':<45} {_fmt_count(grand_samples):>10}  {'':5}"
            f"  {_fmt_count(grand_eff_samples):>12}  {'':6}  {'':8}"
            f"  {_fmt_tokens(grand_tokens):>12}  ±{_fmt_tokens(grand_ci_half_width):>11}"
        )
    else:
        print(
            f" {'':3}  {'TOTAL':<45} {_fmt_count(grand_samples):>10}  {'':5}  {_fmt_count(grand_eff_samples):>12}"
        )
    print(sep_line)
    if show_tokens:
        print(
            f"Approximate two-sided confidence intervals use a normal approximation with finite-population correction at {confidence * 100:.1f}% confidence."
        )
        if target_relative_precision is not None:
            print(
                f"A trailing '!' in Prec. means the target of {100 * target_relative_precision:.1f}% relative precision was not reached before the sample budget was exhausted."
            )
    print()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    # We parse only our own args first; remaining args are passed to Megatron
    # when --estimate-tokens is used.
    parser = argparse.ArgumentParser(
        description="Report dataset sample counts and token estimates for a recipe YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("recipe_yaml", help="Path to the top-level recipe YAML file.")
    parser.add_argument(
        "--split",
        default="train",
        help='Which split to analyze (default: "train").',
    )
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        default=False,
        help="Estimate avg tokens per sample (requires Megatron init + torchrun).",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=50,
        help="Maximum number of samples to iterate per dataset for token estimation (default: 50).",
    )
    parser.add_argument(
        "--target-relative-precision",
        type=float,
        default=None,
        help="Stop sampling a dataset early once the estimated total-token count reaches this relative precision at --confidence (for example 0.05 for +/-5%%). "
        "When omitted, exactly --samples-per-dataset samples are used.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level for token-estimate intervals (default: 0.95).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Energon dataloader worker processes per dataset (default: 0).",
    )
    parser.add_argument(
        "--cache-pool-workers",
        type=int,
        default=1,
        help="FileStoreCachePool worker count per dataset (default: 1).",
    )
    parser.add_argument(
        "--dataset-parallelism",
        type=int,
        default=1,
        help="Number of top-level datasets to process concurrently (default: 1).",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=720,
        help="Timeout for torch.distributed collectives in minutes when --estimate-tokens is used with torchrun (default: 720). "
        "Increase this when rank workloads are imbalanced and some ranks wait a long time at the final gather.",
    )
    parser.add_argument(
        "--progress-update-seconds",
        type=int,
        default=30,
        help="Emit a per-rank heartbeat every N seconds while datasets are running (default: 30). "
        "Set to 0 to disable periodic progress updates.",
    )
    parser.add_argument(
        "--cache-file",
        default=_default_cache_path(),
        help="Path to the persistent per-entry cache JSON file "
        f"(default: {_default_cache_path()}).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable cache reads and writes for this run.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        default=False,
        help="Ignore cached hits and recompute stats before updating the cache.",
    )
    parser.add_argument(
        "--training-script",
        default=None,
        help="Path to a training .sh script.  Megatron args (tokenizer, vision, "
        "sequence lengths, etc.) are extracted automatically via DRY_RUN.  "
        "Any extra args on the command line override the script's values.",
    )

    args, remaining_argv = parser.parse_known_args()

    if args.samples_per_dataset < 1:
        parser.error("--samples-per-dataset must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.cache_pool_workers < 1:
        parser.error("--cache-pool-workers must be >= 1")
    if args.dataset_parallelism < 1:
        parser.error("--dataset-parallelism must be >= 1")
    if args.distributed_timeout_minutes < 1:
        parser.error("--distributed-timeout-minutes must be >= 1")
    if args.progress_update_seconds < 0:
        parser.error("--progress-update-seconds must be >= 0")
    if args.target_relative_precision is not None and args.target_relative_precision <= 0:
        parser.error("--target-relative-precision must be > 0")
    try:
        _validate_confidence(args.confidence)
    except ValueError as exc:
        parser.error(str(exc))
    if args.target_relative_precision is not None and not args.estimate_tokens:
        parser.error("--target-relative-precision requires --estimate-tokens")
    if args.refresh_cache and args.no_cache:
        parser.error("--refresh-cache cannot be used with --no-cache")

    # ── Step 0: Extract args from training script (if provided) ────────
    if args.training_script is not None:
        try:
            script_args = _extract_args_from_script(args.training_script)
            print(
                f"Extracted {len(script_args)} args from {os.path.basename(args.training_script)}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"WARNING: Failed to extract args from training script: {e}", file=sys.stderr)
            script_args = []

        if script_args:
            # CLI args (remaining_argv) take precedence over script args.
            # Build a set of flag names the user explicitly passed on the CLI.
            cli_flags = {
                tok for tok in remaining_argv if tok.startswith("--")
            }
            # Walk script_args: skip any --flag (and its values) already in cli_flags.
            merged = []
            i = 0
            while i < len(script_args):
                tok = script_args[i]
                if tok.startswith("--") and tok in cli_flags:
                    # Skip this flag and all its values
                    i += 1
                    while i < len(script_args) and not script_args[i].startswith("--"):
                        i += 1
                    continue
                merged.append(tok)
                i += 1
            remaining_argv = merged + remaining_argv

    # ── Step 1: Parse top-level recipe ──────────────────────────────────
    recipe_path = os.path.abspath(args.recipe_yaml)
    entries = parse_top_level_entries(recipe_path, split=args.split)

    if not entries:
        print(f"No entries found in {recipe_path} for split '{args.split}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entries)} top-level entries in {os.path.basename(recipe_path)}", file=sys.stderr)

    # ── Step 2: Set up paths (needed for data_loading imports) ──────────
    megatron_root = str(Path(__file__).parent.parent.parent.parent.parent)
    multimodal_root = str(Path(__file__).parent.parent.parent)
    if megatron_root not in sys.path:
        sys.path.insert(0, megatron_root)
    if multimodal_root not in sys.path:
        sys.path.insert(0, multimodal_root)

    cache_enabled = not args.no_cache
    cache_path = os.path.abspath(args.cache_file)
    cache_entries: dict[str, dict] = {}
    if cache_enabled:
        try:
            cache_entries = _load_stats_cache(cache_path)
            print(
                f"Loaded {len(cache_entries)} cached entry stats from {cache_path}",
                file=sys.stderr,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(
                f"WARNING: Failed to load cache {cache_path}: {exc}",
                file=sys.stderr,
            )
            cache_entries = {}

    # ── Step 3: Initialize Megatron if needed ───────────────────────────
    if args.estimate_tokens:
        # Mock transformer_engine for CPU-only operation
        try:
            import transformer_engine
        except (ImportError, RuntimeError):
            import types
            import unittest.mock

            transformer_engine = types.ModuleType("transformer_engine")
            transformer_engine.pytorch = types.ModuleType("transformer_engine.pytorch")
            transformer_engine.pytorch.Linear = unittest.mock.Mock()
            transformer_engine.pytorch.LayerNormLinear = unittest.mock.Mock()
            transformer_engine.pytorch.GroupedLinear = unittest.mock.Mock()
            transformer_engine.pytorch.DotProductAttention = unittest.mock.Mock()
            transformer_engine.pytorch.Sequential = unittest.mock.Mock()
            transformer_engine.pytorch.DelayedScaling = unittest.mock.Mock()
            transformer_engine.pytorch.CudaRNGStatesTracker = unittest.mock.Mock()
            transformer_engine.pytorch.distributed = types.ModuleType(
                "transformer_engine.pytorch.distributed"
            )
            transformer_engine.pytorch.distributed.CudaRNGStatesTracker = (
                unittest.mock.Mock()
            )
            transformer_engine.pytorch.distributed.get_all_rng_states = (
                unittest.mock.Mock()
            )
            transformer_engine.pytorch.distributed.activation_recompute_forward = (
                unittest.mock.Mock()
            )
            transformer_engine.pytorch.distributed.checkpoint = types.ModuleType(
                "transformer_engine.pytorch.distributed.checkpoint"
            )
            transformer_engine.pytorch.tensor = types.ModuleType(
                "transformer_engine.pytorch.tensor"
            )
            transformer_engine.pytorch.tensor.QuantizedTensor = unittest.mock.Mock()
            transformer_engine.pytorch.ops = types.ModuleType(
                "transformer_engine.pytorch.ops"
            )
            transformer_engine.pytorch.ops.Sequential = unittest.mock.Mock()
            transformer_engine.pytorch.ops.Linear = unittest.mock.Mock()
            transformer_engine.pytorch.ops.LayerNorm = unittest.mock.Mock()
            transformer_engine.pytorch.ops.FusibleOperation = unittest.mock.Mock()
            transformer_engine.pytorch.ops.RMSNorm = unittest.mock.Mock()
            transformer_engine.pytorch.ops.GELU = unittest.mock.Mock()
            transformer_engine.pytorch.ops.GEGLU = unittest.mock.Mock()
            transformer_engine.pytorch.ops.SwiGLU = unittest.mock.Mock()
            transformer_engine.pytorch.ops.ReLU = unittest.mock.Mock()
            transformer_engine.pytorch.ops.ReGLU = unittest.mock.Mock()
            transformer_engine.pytorch.fp8 = types.ModuleType(
                "transformer_engine.pytorch.fp8"
            )
            transformer_engine.pytorch.fp8.fp8_model_init = unittest.mock.Mock()
            transformer_engine.pytorch.fp8.fp8_autocast = unittest.mock.Mock()
            transformer_engine.pytorch.fp8.check_fp8_support = unittest.mock.Mock()
            transformer_engine.pytorch.fp8.FP8GlobalStateManager = (
                unittest.mock.Mock()
            )
            transformer_engine.common = types.ModuleType("transformer_engine.common")
            transformer_engine.common.recipe = types.ModuleType(
                "transformer_engine.common.recipe"
            )
            transformer_engine.common.recipe.DelayedScaling = unittest.mock.Mock()

            sys.modules["transformer_engine"] = transformer_engine
            sys.modules["transformer_engine.pytorch"] = transformer_engine.pytorch
            sys.modules["transformer_engine.pytorch.tensor"] = (
                transformer_engine.pytorch.tensor
            )
            sys.modules["transformer_engine.pytorch.distributed"] = (
                transformer_engine.pytorch.distributed
            )
            sys.modules["transformer_engine.pytorch.fp8"] = (
                transformer_engine.pytorch.fp8
            )
            sys.modules["transformer_engine.common"] = transformer_engine.common
            sys.modules["transformer_engine.common.recipe"] = (
                transformer_engine.common.recipe
            )

        import torch

        torch.distributed.init_process_group(
            backend="gloo",
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

        from multimodal_args import add_multimodal_extra_args
        from megatron.training.initialize import initialize_megatron

        # Re-inject remaining args so Megatron can parse them.
        # Also inject --prompt-path default if not already provided.
        prompt_path_default = os.path.join(multimodal_root, "manual_prompts.json")
        if "--prompt-path" not in remaining_argv:
            remaining_argv += ["--prompt-path", prompt_path_default]
        sys.argv = [sys.argv[0]] + remaining_argv

        initialize_megatron(
            args_defaults={
                "tokenizer_type": "GPT2BPETokenizer",
                "micro_batch_size": 1,
                # Model arch defaults — required by validate_args but unused
                # (we never build a model). Can be overridden from CLI.
                "num_layers": 1,
                "hidden_size": 64,
                "num_attention_heads": 1,
                "seq_length": 256,
                "max_position_embeddings": 262144,
            },
            extra_args_provider=add_multimodal_extra_args,
            allow_no_cuda=True,
            skip_mpu_initialization=True,
        )

        print(
            "Megatron initialized (CPU-only, no model build). "
            f"torch.distributed timeout={args.distributed_timeout_minutes} min.",
            file=sys.stderr,
        )

    # ── Step 3: Collect stats for each top-level entry ──────────────────
    rank, world_size = _distributed_rank_info()
    rank_prefix = "" if world_size == 1 else f"[rank {rank}/{world_size}] "
    assigned = [(i, entry) for i, entry in enumerate(entries) if i % world_size == rank]
    all_stats: List[EntryStats] = [EntryStats(entry=entry) for entry in entries]
    dataset_parallelism = min(args.dataset_parallelism, max(1, len(assigned)))

    if world_size > 1 and rank == 0:
        print(
            f"Sharding {len(entries)} datasets across torchrun world_size={world_size}",
            file=sys.stderr,
        )
    if dataset_parallelism > 1:
        print(
            f"{rank_prefix}Processing {len(assigned)} assigned datasets with "
            f"dataset_parallelism={dataset_parallelism}, num_workers={args.num_workers}, "
            f"cache_pool_workers={args.cache_pool_workers}",
            file=sys.stderr,
        )

    for i, entry in assigned:
        entry_name = _short_name(entry.path)
        print(
            f"{rank_prefix}[{i + 1}/{len(entries)}] queued {entry_name} (rep={entry.repetitions})",
            file=sys.stderr,
        )

    def _entry_cache_key(entry: TopLevelEntry) -> str:
        return _make_cache_key(
            entry=entry,
            estimate_tokens=args.estimate_tokens,
            samples_per_dataset=args.samples_per_dataset,
            confidence=args.confidence,
            target_relative_precision=args.target_relative_precision,
            megatron_argv=remaining_argv,
        )

    def _update_entry_sample_progress(
        index: int,
        sampled_count: int,
        sample_budget: int,
        total_samples: int,
        relative_precision: float | None,
    ) -> None:
        with progress_lock:
            progress = active_entries.get(index)
            if progress is None:
                return
            progress.sampled_count = sampled_count
            progress.sample_budget = sample_budget
            progress.total_samples = total_samples
            progress.relative_precision = relative_precision

    def _run_entry(index: int, entry: TopLevelEntry):
        cache_key = _entry_cache_key(entry)
        if cache_enabled and not args.refresh_cache:
            cached_payload = cache_entries.get(cache_key)
            if cached_payload is not None:
                return index, _deserialize_entry_stats_from_cache(entry, cached_payload)
        stat = _compute_entry_stats(
            entry=entry,
            estimate_tokens=args.estimate_tokens,
            samples_per_dataset=args.samples_per_dataset,
            confidence=args.confidence,
            target_relative_precision=args.target_relative_precision,
            num_workers=args.num_workers,
            cache_pool_workers=args.cache_pool_workers,
            progress_callback=lambda sampled_count, sample_budget, total_samples, relative_precision: _update_entry_sample_progress(
                index,
                sampled_count,
                sample_budget,
                total_samples,
                relative_precision,
            ),
        )
        return index, stat

    def _persist_entry_if_needed(stat: EntryStats) -> None:
        if not cache_enabled or stat.from_cache:
            return
        cache_key = _entry_cache_key(stat.entry)
        cache_payload = _serialize_entry_stats_for_cache(stat)
        cache_entries[cache_key] = cache_payload
        try:
            _save_stats_cache(cache_path, {cache_key: cache_payload})
        except OSError as exc:
            print(
                f"WARNING: Failed to update cache {cache_path}: {exc}",
                file=sys.stderr,
            )

    local_results: list[tuple[int, EntryStats]] = []
    progress_lock = threading.Lock()
    active_entries: dict[int, EntryRunProgress] = {}
    completed_local = 0
    heartbeat_stop = threading.Event()

    def _progress_snapshot() -> str:
        with progress_lock:
            active_snapshot = sorted(active_entries.values(), key=lambda item: item.name)
            return _format_rank_progress(
                completed=completed_local,
                total=len(assigned),
                active_entries=active_snapshot,
            )

    def _mark_entry_started(index: int, entry_name: str) -> None:
        with progress_lock:
            active_entries[index] = EntryRunProgress(
                name=entry_name,
                start_time=time.monotonic(),
            )

    def _mark_entry_finished(index: int) -> None:
        nonlocal completed_local
        with progress_lock:
            active_entries.pop(index, None)
            completed_local += 1

    def _start_progress_heartbeat() -> threading.Thread | None:
        if args.progress_update_seconds <= 0 or not assigned:
            return None

        def _heartbeat() -> None:
            while not heartbeat_stop.wait(args.progress_update_seconds):
                print(
                    f"{rank_prefix}progress {_progress_snapshot()}",
                    file=sys.stderr,
                )

        thread = threading.Thread(target=_heartbeat, name=f"dataset-stats-progress-rank-{rank}")
        thread.daemon = True
        thread.start()
        return thread

    progress_thread = _start_progress_heartbeat()

    try:
        if dataset_parallelism == 1:
            for i, entry in assigned:
                entry_name = _short_name(entry.path)
                _mark_entry_started(i, entry_name)
                print(
                    f"{rank_prefix}[{i + 1}/{len(entries)}] running {entry_name} ...",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )
                try:
                    _, stat = _run_entry(i, entry)
                except Exception as e:
                    _mark_entry_finished(i)
                    print(f" ERROR: {e}", file=sys.stderr)
                    continue
                _mark_entry_finished(i)
                local_results.append((i, stat))
                _persist_entry_if_needed(stat)
                cache_prefix = "cache hit, " if stat.from_cache else ""
                print(f" {cache_prefix}{_entry_progress_suffix(stat)}", file=sys.stderr)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=dataset_parallelism) as executor:
                future_to_meta = {}
                for i, entry in assigned:
                    entry_name = _short_name(entry.path)
                    _mark_entry_started(i, entry_name)
                    future_to_meta[executor.submit(_run_entry, i, entry)] = (i, entry)
                completed = 0
                for future in concurrent.futures.as_completed(future_to_meta):
                    i, entry = future_to_meta[future]
                    entry_name = _short_name(entry.path)
                    completed += 1
                    try:
                        _, stat = future.result()
                    except Exception as e:
                        _mark_entry_finished(i)
                        print(
                            f"{rank_prefix}[{completed}/{len(assigned)}] {entry_name} ERROR: {e}",
                            file=sys.stderr,
                        )
                        continue
                    _mark_entry_finished(i)
                    local_results.append((i, stat))
                    _persist_entry_if_needed(stat)
                    cache_prefix = "cache hit, " if stat.from_cache else ""
                    print(
                        f"{rank_prefix}[{completed}/{len(assigned)}] {entry_name} {cache_prefix}{_entry_progress_suffix(stat)}",
                        file=sys.stderr,
                    )
    finally:
        heartbeat_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1.0)

    if assigned:
        print(f"{rank_prefix}progress {_progress_snapshot()}", file=sys.stderr)

    if world_size > 1:
        import torch

        gathered_results: list[list[tuple[int, EntryStats]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_results, local_results)
        if rank == 0:
            for rank_results in gathered_results:
                if rank_results is None:
                    continue
                for idx, stat in rank_results:
                    all_stats[idx] = stat
    else:
        for idx, stat in local_results:
            all_stats[idx] = stat

    # ── Step 4: Print report ────────────────────────────────────────────
    if rank == 0:
        print_report(
            yaml_path=recipe_path,
            split=args.split,
            stats=all_stats,
            show_tokens=args.estimate_tokens,
            confidence=args.confidence,
            target_relative_precision=args.target_relative_precision,
        )

    # ── Cleanup: destroy the process group so gloo threads don't hang ──
    if args.estimate_tokens:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
