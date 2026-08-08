# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import os
from collections.abc import Mapping, Sequence
from itertools import accumulate
from typing import Any, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.hybrid_cp_schedule import (
    BalancedCPScheduler,
    summarize_hybrid_cp_schedule,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator


def _hybrid_cp_debug(message: str) -> None:
    """Emit rank-local HybridCP scheduling diagnostics when explicitly enabled."""

    if os.environ.get("MEGATRON_HYBRID_CP_DEBUG") != "1":
        return
    rank = "?"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())
    print(f"[HYBRID_CP_DEBUG][rank={rank}] {message}", flush=True)


def collect_hybrid_cp_microbatches(data_iterator, num_microbatches: int) -> list[Any]:
    """Consume exactly one global batch from an underlying data iterator.

    The legacy multimodal HybridCP wrapper was called once per optimizer step
    but consumed only one item from the external iterator.  That silently
    dropped the remaining global-batch microbatches whenever
    ``global_batch_size > micro_batch_size * data_parallel_size``.  Keep the
    consumption contract explicit and testable so the scheduler can route the
    complete global batch before forming its CP groups.
    """

    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be positive, got {num_microbatches}")
    return [next(data_iterator) for _ in range(num_microbatches)]


def get_hybrid_cp_sample_lengths(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    """Return aligned real and padded lengths for scheduled logical samples."""

    real_lengths: list[int] = []
    padded_lengths: list[int] = []
    for sample in samples:
        cu_seqlens = _require_tensor(sample, "cu_seqlens")
        cu_seqlens_padded = sample.get("cu_seqlens_padded", cu_seqlens)
        if not isinstance(cu_seqlens_padded, torch.Tensor):
            raise ValueError("multimodal HybridCP cu_seqlens_padded must be a tensor")
        if cu_seqlens.shape != cu_seqlens_padded.shape:
            raise ValueError("HybridCP real and padded sequence boundaries must have equal shapes")
        for index in range(cu_seqlens.numel() - 1):
            real_length = int(cu_seqlens[index + 1].item() - cu_seqlens[index].item())
            if real_length == 0:
                continue
            padded_length = int(
                cu_seqlens_padded[index + 1].item() - cu_seqlens_padded[index].item()
            )
            if padded_length < real_length:
                raise ValueError(
                    f"HybridCP padded length {padded_length} is smaller than "
                    f"real length {real_length}"
                )
            real_lengths.append(real_length)
            padded_lengths.append(padded_length)
    return real_lengths, padded_lengths


def summarize_hybrid_cp_multimodal_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Summarize media work represented by routed multimodal samples."""

    stats = {
        "hybrid_cp/samples_with_vision": 0,
        "hybrid_cp/samples_with_video": 0,
        "hybrid_cp/samples_with_audio": 0,
        "hybrid_cp/vision_tiles": 0,
        "hybrid_cp/video_frames": 0,
        "hybrid_cp/audio_clips": 0,
    }
    for sample in samples:
        num_tiles = sample.get("num_tiles")
        tiles = (
            [int(value) for value in num_tiles.detach().cpu().reshape(-1).tolist()]
            if isinstance(num_tiles, torch.Tensor)
            else []
        )
        num_frames = sample.get("num_frames")
        frames = (
            [int(value) for value in num_frames.detach().cpu().reshape(-1).tolist()]
            if isinstance(num_frames, torch.Tensor)
            else []
        )
        num_sound_clips = sample.get("num_sound_clips")
        sound_clips = (
            [int(value) for value in num_sound_clips.detach().cpu().reshape(-1).tolist()]
            if isinstance(num_sound_clips, torch.Tensor)
            else []
        )

        if any(tile > 0 for tile in tiles):
            stats["hybrid_cp/samples_with_vision"] += 1
        if any(frame > 1 for frame in frames):
            stats["hybrid_cp/samples_with_video"] += 1
        if any(count > 0 for count in sound_clips):
            stats["hybrid_cp/samples_with_audio"] += 1
        stats["hybrid_cp/vision_tiles"] += sum(max(0, tile) for tile in tiles)
        stats["hybrid_cp/video_frames"] += sum(frame for frame in frames if frame > 1)
        stats["hybrid_cp/audio_clips"] += sum(max(0, count) for count in sound_clips)

    return {key: float(value) for key, value in stats.items()}


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"multimodal HybridCP batch key {key!r} must be a tensor")
    return value


def _nested_int_metadata(batch: Mapping[str, Any], key: str, batch_size: int) -> list[list[list[int]]]:
    value = batch.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"multimodal HybridCP batch key {key!r} must be nested metadata")
    if len(value) != batch_size:
        raise ValueError(
            f"multimodal HybridCP batch key {key!r} has {len(value)} rows, "
            f"expected {batch_size}"
        )
    result: list[list[list[int]]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"multimodal HybridCP batch key {key!r} has an invalid row")
        result.append(
            [
                [int(item) for item in sample]
                for sample in row
            ]
        )
    return result


def _sample_image_offsets(
    sample_image_counts: Sequence[int], sample_index: int
) -> tuple[int, int]:
    start = sum(int(count) for count in sample_image_counts[:sample_index])
    end = start + int(sample_image_counts[sample_index])
    return start, end


def _sample_media_offsets(
    sample_media: Sequence[Sequence[int]], sample_index: int
) -> tuple[int, int]:
    start = sum(len(media) for media in sample_media[:sample_index])
    end = start + len(sample_media[sample_index])
    return start, end


def _slice_optional_tensor(
    batch: Mapping[str, Any], key: str, batch_index: int, start: int, end: int
) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"multimodal HybridCP batch key {key!r} must be a tensor")
    if value.dim() == 0:
        return value.clone()
    if value.dim() >= 1 and value.shape[0] > batch_index and key in {"tokens", "labels", "position_ids", "loss_mask"}:
        return value[batch_index, start:end].contiguous()
    return value[start:end].contiguous()


def unpack_multimodal_batch(batch: Mapping[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Split one Energon packed dict into wire-format sub-samples.

    HybridCP schedules individual packed sub-samples, while the Omni external
    dataloader emits a dict containing a batch dimension and media tensors
    concatenated across all sub-samples.  The returned tensors deliberately use
    their variable dimension as dimension zero so the all-to-all route can send
    each key with its own split sizes.  ``restore_multimodal_hybrid_cp_sample``
    rebuilds the batch-shaped dict consumed by ``examples/multimodal/train.py``.
    """

    cu_lengths = _require_tensor(batch, "cu_lengths")
    cu_lengths_padded = _require_tensor(batch, "cu_lengths_padded")
    tokens = _require_tensor(batch, "tokens")
    labels = _require_tensor(batch, "labels")
    if cu_lengths.dim() != 2 or cu_lengths_padded.dim() != 2:
        raise ValueError("multimodal HybridCP cu_lengths tensors must be 2-D")
    if tokens.dim() != 2 or labels.dim() != 2:
        raise ValueError("multimodal HybridCP tokens and labels tensors must be 2-D")
    if (
        cu_lengths.shape != cu_lengths_padded.shape
        or cu_lengths.shape[0] != tokens.shape[0]
        or labels.shape[0] != tokens.shape[0]
    ):
        raise ValueError("multimodal HybridCP batch dimensions are inconsistent")
    if labels.shape[1] != tokens.shape[1] + 1:
        raise ValueError(
            "multimodal HybridCP labels must have one more token than tokens"
        )

    batch_size = int(cu_lengths.shape[0])
    if batch_size != 1:
        raise ValueError(
            "multimodal HybridCP currently requires micro-batch-size 1; "
            f"received {batch_size} packed rows"
        )

    sample_token_lengths_raw = batch.get("sample_token_lengths")
    if not isinstance(sample_token_lengths_raw, Sequence) or isinstance(
        sample_token_lengths_raw, (str, bytes, bytearray)
    ):
        raise ValueError(
            "multimodal HybridCP requires sample_token_lengths metadata from the task encoder"
        )
    if len(sample_token_lengths_raw) != batch_size:
        raise ValueError(
            "multimodal HybridCP sample_token_lengths has "
            f"{len(sample_token_lengths_raw)} rows, expected {batch_size}"
        )
    raw_length_row = sample_token_lengths_raw[0]
    if not isinstance(raw_length_row, Sequence) or isinstance(
        raw_length_row, (str, bytes, bytearray)
    ):
        raise ValueError("multimodal HybridCP sample_token_lengths has an invalid row")
    sample_token_lengths = [int(length) for length in raw_length_row]
    if any(length <= 0 for length in sample_token_lengths):
        raise ValueError("multimodal HybridCP sample_token_lengths must be positive")
    raw_token_total = sum(sample_token_lengths)
    if raw_token_total > tokens.shape[1]:
        raise ValueError(
            "multimodal HybridCP sample_token_lengths sum to "
            f"{raw_token_total}, but tokens has width {tokens.shape[1]}"
        )
    raw_token_offsets = [0, *accumulate(sample_token_lengths)]

    sample_image_counts_raw = batch.get("sample_image_counts")
    if not isinstance(sample_image_counts_raw, Sequence) or isinstance(
        sample_image_counts_raw, (str, bytes, bytearray)
    ):
        raise ValueError(
            "multimodal HybridCP requires sample_image_counts metadata from the task encoder"
        )
    sample_image_counts = [int(count) for count in sample_image_counts_raw[0]]
    sample_num_tiles = _nested_int_metadata(batch, "sample_num_tiles", batch_size)[0]
    sample_num_frames = _nested_int_metadata(batch, "sample_num_frames", batch_size)[0]
    if len(sample_image_counts) != cu_lengths.shape[1] - 1:
        raise ValueError("sample_image_counts does not match cu_lengths")
    if len(sample_token_lengths) != len(sample_image_counts):
        raise ValueError("sample_token_lengths does not match cu_lengths")
    if len(sample_num_tiles) != len(sample_image_counts) or len(sample_num_frames) != len(sample_image_counts):
        raise ValueError("per-sample media metadata does not match cu_lengths")
    total_image_count = sum(sample_image_counts)

    sample_num_sound_clips = batch.get("sample_num_sound_clips")
    if sample_num_sound_clips is None:
        sample_num_sound_clips_rows = [[] for _ in sample_image_counts]
    else:
        sample_num_sound_clips_rows = _nested_int_metadata(
            batch, "sample_num_sound_clips", batch_size
        )[0]
        if len(sample_num_sound_clips_rows) != len(sample_image_counts):
            raise ValueError("per-sample audio metadata does not match cu_lengths")

    vision_cu_lengths = batch.get("vision_cu_lengths")
    if vision_cu_lengths is not None and not isinstance(vision_cu_lengths, torch.Tensor):
        raise ValueError("multimodal HybridCP vision_cu_lengths must be a tensor")
    if isinstance(vision_cu_lengths, torch.Tensor):
        if vision_cu_lengths.dim() != 2 or vision_cu_lengths.shape[0] != 1:
            raise ValueError("multimodal HybridCP vision_cu_lengths must have shape [1, N]")
        vision_offsets = vision_cu_lengths[0]
    else:
        vision_offsets = None

    imgs = batch.get("imgs")
    if imgs is not None and not isinstance(imgs, torch.Tensor):
        raise ValueError("multimodal HybridCP imgs must be a tensor")
    imgs_sizes = batch.get("imgs_sizes")
    if imgs_sizes is not None and not isinstance(imgs_sizes, torch.Tensor):
        raise ValueError("multimodal HybridCP imgs_sizes must be a tensor")

    has_pad_img = batch.get("has_pad_img")
    if has_pad_img is None:
        # Older task encoders did not emit this optional FP8 image-padding flag.
        has_pad_img = torch.tensor(False, dtype=torch.bool)
    elif not isinstance(has_pad_img, torch.Tensor) or has_pad_img.numel() != 1:
        raise ValueError("multimodal HybridCP batch key 'has_pad_img' must be a scalar tensor")
    else:
        has_pad_img = has_pad_img.reshape(()).to(dtype=torch.bool)
    if bool(has_pad_img.item()):
        raise ValueError(
            "multimodal HybridCP does not support FP8 padded vision images"
        )

    if vision_offsets is not None:
        if vision_offsets.numel() == 0:
            raise ValueError("multimodal HybridCP vision_cu_lengths cannot be empty")
        if int(vision_offsets[0].item()) != 0:
            raise ValueError("multimodal HybridCP vision_cu_lengths must start at zero")
        if vision_offsets.numel() > 1:
            expected_offset_count = total_image_count + 1
            if vision_offsets.numel() != expected_offset_count:
                raise ValueError(
                    "multimodal HybridCP vision_cu_lengths has "
                    f"{vision_offsets.numel()} offsets for {total_image_count} images"
                )
            if bool(torch.any(vision_offsets[1:] <= vision_offsets[:-1]).item()):
                raise ValueError(
                    "multimodal HybridCP vision_cu_lengths must be strictly increasing"
                )

    sound_clips = batch.get("sound_clips")
    sound_length = batch.get("sound_length")
    sound_timestamps = batch.get("sound_timestamps")
    num_sound_clips = batch.get("num_sound_clips")
    tensor_sound_keys = (sound_clips, sound_length, sound_timestamps, num_sound_clips)
    if any(value is not None and not isinstance(value, torch.Tensor) for value in tensor_sound_keys):
        raise ValueError("multimodal HybridCP audio fields must be tensors")

    samples: list[dict[str, torch.Tensor]] = []
    for sample_index in range(len(sample_image_counts)):
        expanded_start = int(cu_lengths_padded[0, sample_index].item())
        expanded_end = int(cu_lengths[0, sample_index + 1].item())
        expanded_padded_end = int(cu_lengths_padded[0, sample_index + 1].item())
        if expanded_end < expanded_start or expanded_padded_end < expanded_end:
            raise ValueError("multimodal HybridCP cu_lengths are not monotonic")
        raw_start = raw_token_offsets[sample_index]
        raw_end = raw_token_offsets[sample_index + 1]

        sample: dict[str, torch.Tensor] = {
            "tokens": tokens[0, raw_start:raw_end].contiguous(),
            "labels": labels[0, raw_start : raw_end + 1].contiguous(),
            "cu_seqlens": torch.tensor(
                [0, expanded_end - expanded_start], dtype=cu_lengths.dtype
            ),
            "cu_seqlens_padded": torch.tensor(
                [0, expanded_padded_end - expanded_start],
                dtype=cu_lengths_padded.dtype,
            ),
            "max_seqlen": torch.tensor(
                expanded_padded_end - expanded_start, dtype=torch.int32
            ),
            "sample_lengths": torch.tensor(
                [expanded_end - expanded_start], dtype=torch.int32
            ),
            "samples_seen": torch.tensor(1, dtype=torch.int32),
            # Preserve the scalar expected by examples/multimodal/train.py.
            "has_pad_img": has_pad_img.clone(),
        }

        for key in ("position_ids", "loss_mask", "attention_mask"):
            value = _slice_optional_tensor(batch, key, 0, raw_start, raw_end)
            if value is not None:
                sample[key] = value

        image_start, image_end = _sample_image_offsets(sample_image_counts, sample_index)
        if vision_offsets is not None and vision_offsets.numel() > 1 and image_end > image_start:
            vision_start = int(vision_offsets[image_start].item())
            vision_end = int(vision_offsets[image_end].item())
            if imgs is not None:
                if imgs.dim() >= 3 and imgs.shape[0] == 1:
                    sample["imgs"] = imgs[0, vision_start:vision_end].contiguous()
                else:
                    sample["imgs"] = imgs[vision_start:vision_end].contiguous()
            if imgs_sizes is not None:
                sample["imgs_sizes"] = imgs_sizes[image_start:image_end].contiguous()
            sample["vision_cu_lengths"] = (
                vision_offsets[image_start : image_end + 1] - vision_offsets[image_start]
            ).contiguous()
            sample["vision_max_lengths"] = torch.tensor(
                int((vision_offsets[image_start + 1 : image_end + 1] - vision_offsets[image_start:image_end]).max().item()),
                dtype=torch.int32,
            )
        elif image_end > image_start:
            if imgs is None or image_end > imgs.shape[0]:
                available_images = 0 if imgs is None else int(imgs.shape[0])
                raise ValueError(
                    "multimodal HybridCP fixed-resolution batch requires "
                    f"{image_end} images, received {available_images}"
                )
            if imgs_sizes is None or image_end > imgs_sizes.shape[0]:
                available_sizes = 0 if imgs_sizes is None else int(imgs_sizes.shape[0])
                raise ValueError(
                    "multimodal HybridCP fixed-resolution batch requires "
                    f"{image_end} image sizes, received {available_sizes}"
                )
            sample["imgs"] = imgs[image_start:image_end].contiguous()
            sample["imgs_sizes"] = imgs_sizes[image_start:image_end].contiguous()
            sample["vision_cu_lengths"] = torch.tensor([0], dtype=torch.int32)
            sample["vision_max_lengths"] = torch.tensor([0], dtype=torch.int32)
        else:
            sample["imgs"] = torch.tensor([[0]], dtype=torch.float32)
            sample["imgs_sizes"] = torch.tensor([[0, 0]], dtype=torch.int32)
            sample["vision_cu_lengths"] = torch.tensor([0], dtype=torch.int32)
            sample["vision_max_lengths"] = torch.tensor([0], dtype=torch.int32)

        sample["num_tiles"] = torch.tensor(
            sample_num_tiles[sample_index] or [0], dtype=torch.int32
        )
        sample["num_frames"] = torch.tensor(
            sample_num_frames[sample_index] or [0], dtype=torch.int32
        )

        clip_start = sum(
            sum(int(value) for value in row) for row in sample_num_sound_clips_rows[:sample_index]
        )
        clip_end = clip_start + sum(
            int(value) for value in sample_num_sound_clips_rows[sample_index]
        )
        if sound_clips is not None and sample_num_sound_clips_rows[sample_index]:
            sample["sound_clips"] = sound_clips[clip_start:clip_end].contiguous()
            sample["sound_length"] = sound_length[clip_start:clip_end].contiguous()
            sample["sound_timestamps"] = sound_timestamps[clip_start:clip_end].contiguous()
            audio_start, audio_end = _sample_media_offsets(
                sample_num_sound_clips_rows, sample_index
            )
            sample["num_sound_clips"] = num_sound_clips[
                audio_start:audio_end
            ].contiguous()
        else:
            sample["sound_clips"] = torch.tensor([[0]], dtype=torch.float32)
            sample["sound_length"] = torch.tensor([[0]], dtype=torch.int64)
            sample["sound_timestamps"] = torch.tensor([[0]], dtype=torch.float32)
            sample["num_sound_clips"] = torch.tensor([[0]], dtype=torch.int64)

        samples.append(sample)
    return samples


def pack_multimodal_hybrid_cp_samples(
    samples: Sequence[Mapping[str, torch.Tensor]],
    local_cp_size: int,
    max_padded_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pack routed multimodal samples into one batch-shaped DynamicCP payload."""

    if not samples:
        raise ValueError("multimodal HybridCP payload must contain at least one sample")
    if local_cp_size < 1:
        raise ValueError(f"local_cp_size must be positive, got {local_cp_size}")

    real_lengths = [int(sample["cu_seqlens"][-1].item()) for sample in samples]
    padded_lengths = [int(sample["cu_seqlens_padded"][-1].item()) for sample in samples]
    total_padded_tokens = sum(padded_lengths)
    if max_padded_tokens is not None and total_padded_tokens > max_padded_tokens:
        raise ValueError(
            f"multimodal HybridCP payload has {total_padded_tokens} padded tokens, "
            f"capacity is {max_padded_tokens}"
        )

    restored = dict(samples[0])
    restored["tokens"] = torch.cat([sample["tokens"] for sample in samples]).unsqueeze(0)
    restored["labels"] = torch.cat(
        [samples[0]["labels"][:1], *[sample["labels"][1:] for sample in samples]]
    ).unsqueeze(0)
    padded_boundaries = [0, *accumulate(padded_lengths)]
    real_boundaries = [
        padded_boundaries[index] + real_length
        for index, real_length in enumerate(real_lengths)
    ]
    restored["cu_lengths"] = torch.tensor(
        [0, *real_boundaries],
        dtype=samples[0]["cu_seqlens"].dtype,
        device=samples[0]["cu_seqlens"].device,
    ).unsqueeze(0)
    restored["cu_lengths_padded"] = torch.tensor(
        padded_boundaries,
        dtype=samples[0]["cu_seqlens_padded"].dtype,
        device=samples[0]["cu_seqlens_padded"].device,
    ).unsqueeze(0)
    restored["max_lengths"] = torch.tensor(
        [max(real_lengths)], dtype=torch.int32, device=samples[0]["tokens"].device
    )
    restored["sample_lengths"] = torch.tensor(
        [real_lengths], dtype=torch.int32, device=samples[0]["tokens"].device
    )
    restored["samples_seen"] = torch.tensor(
        sum(int(sample["samples_seen"].item()) for sample in samples),
        dtype=torch.int32,
        device=samples[0]["tokens"].device,
    )
    restored["local_cp_size"] = torch.tensor(local_cp_size, dtype=torch.int32)

    for key in ("position_ids", "loss_mask", "attention_mask"):
        present = [key in sample for sample in samples]
        if any(present) and not all(present):
            raise ValueError(
                f"multimodal HybridCP payload has inconsistent optional key {key!r}"
            )
        if all(present):
            restored[key] = torch.cat([sample[key] for sample in samples], dim=0)

    vision_samples = [
        sample for sample in samples if bool(torch.any(sample["num_tiles"] > 0).item())
    ]
    if vision_samples:
        packed_imgs = torch.cat([sample["imgs"] for sample in vision_samples], dim=0)
        restored["imgs"] = packed_imgs.unsqueeze(0) if packed_imgs.dim() == 2 else packed_imgs
        restored["imgs_sizes"] = torch.cat(
            [sample["imgs_sizes"] for sample in vision_samples], dim=0
        )
        vision_offsets = [0]
        vision_total = 0
        for sample in vision_samples:
            boundaries = [
                int(value)
                for value in sample["vision_cu_lengths"].detach().cpu().tolist()[1:]
            ]
            vision_offsets.extend(vision_total + boundary for boundary in boundaries)
            vision_total = vision_offsets[-1]
        restored["vision_cu_lengths"] = torch.tensor(
            [vision_offsets],
            dtype=vision_samples[0]["vision_cu_lengths"].dtype,
            device=vision_samples[0]["vision_cu_lengths"].device,
        )
        restored["vision_max_lengths"] = torch.stack(
            [sample["vision_max_lengths"].reshape(()) for sample in vision_samples]
        ).max().reshape(1)
        restored["num_tiles"] = torch.cat(
            [sample["num_tiles"] for sample in vision_samples], dim=0
        )
        restored["num_frames"] = torch.cat(
            [sample["num_frames"] for sample in vision_samples], dim=0
        )
    else:
        for key in (
            "imgs",
            "imgs_sizes",
            "vision_cu_lengths",
            "vision_max_lengths",
            "num_tiles",
            "num_frames",
        ):
            restored[key] = samples[0][key].clone()
        restored["vision_cu_lengths"] = restored["vision_cu_lengths"].unsqueeze(0)
        restored["vision_max_lengths"] = restored["vision_max_lengths"].reshape(1)

    restored["has_pad_img"] = torch.stack(
        [sample["has_pad_img"].reshape(()) for sample in samples]
    ).any()

    audio_samples = [
        sample
        for sample in samples
        if bool(torch.any(sample["num_sound_clips"] > 0).item())
    ]
    audio_keys = ("sound_clips", "sound_length", "sound_timestamps", "num_sound_clips")
    if audio_samples:
        for key in audio_keys:
            restored[key] = torch.cat([sample[key] for sample in audio_samples], dim=0)
    else:
        for key in audio_keys:
            restored[key] = samples[0][key].clone()

    restored.pop("cu_seqlens", None)
    restored.pop("cu_seqlens_padded", None)
    restored.pop("max_seqlen", None)
    return restored


def restore_multimodal_hybrid_cp_sample(
    sample: Mapping[str, torch.Tensor], local_cp_size: int | None = None
) -> dict[str, torch.Tensor]:
    """Restore one wire-format sample to the multimodal ``get_batch`` schema."""

    return pack_multimodal_hybrid_cp_samples(
        [sample], local_cp_size=1 if local_cp_size is None else local_cp_size
    )


def prepare_hybrid_cp_payload_iterator(
    batch: Mapping[int, Mapping[str, torch.Tensor]],
    sample_ids: Sequence[int],
    local_cp_size: int,
    max_padded_tokens: int,
) -> RerunDataIterator:
    """Create one replayable iterator item for a rank's scheduled Omni payload."""

    if not sample_ids:
        raise ValueError("HybridCP rank payload must contain at least one sample ID")
    missing_sample_ids = [sample_id for sample_id in sample_ids if sample_id not in batch]
    if missing_sample_ids:
        raise ValueError(f"HybridCP rank payload is missing sample_ids={missing_sample_ids}")
    payload = pack_multimodal_hybrid_cp_samples(
        [batch[sample_id] for sample_id in sample_ids],
        local_cp_size=local_cp_size,
        max_padded_tokens=max_padded_tokens,
    )
    return RerunDataIterator(iter([payload]))


class HybridCPDataLoaderWrapper:
    """
    A wrapper class that wraps around an existing data_iterator.
    For every __next__ call,
    1. Each DP rank pulls a batch of packed samples.
    2. Extracts the sequence lengths of each sub-sample and all-gathers across the DP group.
    3. Schedules the sub-samples to the DPxCP ranks using the BalancedCPScheduler.
    4. Based on the schedule, reroutes the sub-samples to the correct rank using all-to-all.
    5. Returns the assigned sub-samples to this rank.

    Args:
        data_iterator: The original data_iterator to wrap around
        config: The config object containing the max_seqlen_per_dp_cp_rank
        dp_cp_group: Data parallel context parallel group.
    """

    def __init__(
        self, data_iterator, config, pg_collection: Optional[ProcessGroupCollection] = None
    ):
        self.data_iterator = data_iterator
        self.config = config
        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            self.dp_group = parallel_state.get_data_parallel_group()
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.dp_cp_group = pg_collection.dp_cp
            self.dp_group = pg_collection.dp
            self.tp_group = pg_collection.tp
        assert (
            self.dp_cp_group is not None and self.dp_group is not None and self.tp_group is not None
        ), "dp_cp_group, dp_group, tp_group must not be None when using hybrid context parallel"

        self.cp_balancing_scheduler = BalancedCPScheduler(
            max_seq_len_per_rank=self.config.max_seqlen_per_dp_cp_rank,
            dp_cp_group=self.dp_cp_group,
            min_cp_size=self.config.dynamic_context_parallel_min_size,
        )

        self.total_hdp_gpus = self.dp_cp_group.size()

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def get_global_seqlens(self, subsample_seqlens: torch.Tensor) -> List[int]:
        """
        Gathers the sequence lengths of all subsamples from all DP ranks.
        Each DP rank loads the same number of microbatches but each microbatch
        may have a different number of subsamples.

        We find the number of subsamples each rank holds and then gather the
        sequence lengths of all subsamples from all ranks.
        """
        # Collect the number of subsamples from all ranks
        local_len = torch.tensor([subsample_seqlens.shape[0]], dtype=torch.int32).cuda()
        dp_subsample_count = [torch.zeros_like(local_len) for _ in range(self.dp_group.size())]
        torch.distributed.all_gather(dp_subsample_count, local_len, group=self.dp_group)

        # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
        dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
        max_sub_samples = int(dp_subsample_counts.max().item())

        if local_len.item() < max_sub_samples:
            subsample_seqlens_padded = torch.cat(
                [
                    subsample_seqlens,
                    torch.zeros(max_sub_samples - local_len.item(), dtype=torch.int32).cuda(),
                ],
                dim=0,
            )
        else:
            subsample_seqlens_padded = subsample_seqlens

        # Gather the subsample_seqlens from all ranks
        seqlens_gathered = [
            torch.empty_like(subsample_seqlens_padded) for _ in range(self.dp_group.size())
        ]
        torch.distributed.all_gather(
            seqlens_gathered, subsample_seqlens_padded, group=self.dp_group
        )

        # Trim each seqlens_gathered to the length of the correct sample
        for dp_rank, seqlen in enumerate(seqlens_gathered):
            seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

        seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
        seqlens_gathered = seqlens_gathered.cpu().tolist()

        # Calculate the offsets to assign unique global ID to each subsample.
        csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum[:-1]], dim=0)

        return seqlens_gathered, offsets

    def get_global_id_seqlens(self, num_local_subsamples, offsets, seqlens_gathered):
        """
        Calculates the global ID for each subsample.

        We assign a unique global ID to each subsample.

        Returns:
        global_id_seqlens: list of (global_id, seqlen) tuples for scheduling.
        global_ids_this_rank: list of global IDs locally present on this rank.
        """
        dp_rank = self.dp_group.rank()
        global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()
        # Create a list of (global_id, seqlen) tuples for scheduling
        global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]
        # Get the global IDs locally present on this rank
        global_ids_this_rank = global_ids[
            offsets[dp_rank] : offsets[dp_rank] + num_local_subsamples
        ]

        return global_id_seqlens, global_ids_this_rank

    def _gid_to_src_rank(self, gid: int, offsets: List[int]) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        # Since the torch.distributed.get_process_group_ranks
        # provides the global rank, we need to consider TP
        hdp_rank = (
            torch.distributed.get_process_group_ranks(self.dp_group)[dp_src_rank]
            // self.tp_group.size()
        )
        return hdp_rank

    def reroute_samples_to_hdp_ranks(
        self,
        batch,
        global_ids_this_rank,
        global_id_seqlens,
        sample_id_groups,
        offsets,
        multimodal: bool = False,
    ):
        """
        Reroutes the sub-samples to the correct rank after scheduling.

        For each key in the batch dict, we perform an all-to-all communication
        to transfer the data to the correct ranks.
        Since all CP ranks within a DP group have the same data, we only need
        to transfer data between matching CP ranks.
        """
        if multimodal:
            return self._reroute_multimodal_samples(
                batch,
                global_ids_this_rank,
                global_id_seqlens,
                sample_id_groups,
                offsets,
            )

        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
        hdp_rank = self.dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(self.dp_group)
        # Here we actually want to get the DP group's rank within the HDP group,
        # we need to consider TP
        dp_ranks = [r // self.tp_group.size() for r in dp_ranks]

        data_keys = batch[0].keys()

        # Create the send plan
        combined_sample_id_groups: List[List[int]] = [[] for _ in range(self.total_hdp_gpus)]

        for d in range(self.total_hdp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[d].extend(sample_id_group[d])

        for dest_rank in range(self.total_hdp_gpus):
            combined_sample_id_groups[dest_rank].sort()

        # Filter out samples that are not present on this rank
        send_ids_sorted = [
            gid
            for d in dp_ranks
            for gid in combined_sample_id_groups[d]
            if gid in global_ids_this_rank
        ]
        # send_counts = [len(combined_sample_id_groups[d]) for d in range(self.total_hdp_gpus)]

        send_lens_split = [0] * self.total_hdp_gpus
        for dest_rank in range(self.total_hdp_gpus):
            if dest_rank in dp_ranks:
                send_lens_split[dest_rank] = sum(
                    [
                        global_id_seqlens[gid][1]
                        for gid in combined_sample_id_groups[dest_rank]
                        if gid in global_ids_this_rank
                    ]
                )
            else:
                # We only need to share local data with DP ranks that have different data.
                send_lens_split[dest_rank] = 0

        # Create the recv plan
        recv_sample_id_groups = [[] for _ in range(self.total_hdp_gpus)]
        for gid in combined_sample_id_groups[hdp_rank]:
            src_rank = self._gid_to_src_rank(gid, offsets)
            recv_sample_id_groups[src_rank].append(gid)

        recv_lens_split = [0] * self.total_hdp_gpus
        for src_rank in range(self.total_hdp_gpus):
            recv_lens_split[src_rank] = sum(
                [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
            )

        recv_ids_sorted = [
            gid for d in range(self.total_hdp_gpus) for gid in recv_sample_id_groups[d]
        ]
        recv_counts = [len(recv_sample_id_groups[d]) for d in range(self.total_hdp_gpus)]

        recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

        def _pack_sample_by_key(key: str) -> torch.Tensor:
            flattened_tensors = []
            for gid in send_ids_sorted:
                t = batch[gid2local_id[gid]][key].to(torch.cuda.current_device(), non_blocking=True)
                flattened_tensors.append(t)
            return (
                torch.cat(flattened_tensors, dim=0)
                if flattened_tensors
                else torch.empty(0, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
            )

        def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
            cursor = 0
            for i, gid in enumerate(recv_ids_sorted):
                sample_len = global_id_seqlens[gid][1]
                recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
                cursor += sample_len

        for key in data_keys:
            send_tensor = _pack_sample_by_key(key)
            recv_tensor = torch.empty(
                sum(recv_lens_split), device=torch.cuda.current_device(), dtype=send_tensor.dtype
            )
            torch.distributed.all_to_all_single(
                output=recv_tensor,
                input=send_tensor,
                output_split_sizes=recv_lens_split,
                input_split_sizes=send_lens_split,
                group=self.dp_cp_group,
            )
            _unpack_sample_by_key(key, recv_tensor)

        recv_sample_with_id = {
            recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)
        }
        return recv_sample_with_id

    def _gather_multimodal_shapes(self, batch, global_ids_this_rank):
        local_shapes = {
            int(global_id): {
                key: tuple(int(dim) for dim in value.shape)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            for global_id, sample in zip(global_ids_this_rank.tolist(), batch)
        }
        gathered_shapes = [None for _ in range(self.dp_group.size())]
        torch.distributed.all_gather_object(
            gathered_shapes, local_shapes, group=self.dp_group
        )
        global_shapes = {}
        for rank_shapes in gathered_shapes:
            if rank_shapes is None:
                continue
            global_shapes.update(rank_shapes)
        return global_shapes

    def _reroute_multimodal_samples(
        self, batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
    ):
        """Route variable-shaped multimodal tensors with per-key split sizes.

        The original HybridCP implementation assumes every field has the same
        first dimension as the text sequence.  Energon batches violate that
        assumption: image patch tokens, image metadata, and audio clips each
        have independent lengths.  Gather the small shape metadata once, then
        route every tensor flattened with its own all-to-all split sizes.
        """

        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank.tolist())}
        hdp_rank = self.dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(self.dp_group)
        dp_ranks = [r // self.tp_group.size() for r in dp_ranks]

        combined_sample_id_groups: List[List[int]] = [
            [] for _ in range(self.total_hdp_gpus)
        ]
        for dest_rank in range(self.total_hdp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[dest_rank].extend(sample_id_group[dest_rank])
            combined_sample_id_groups[dest_rank].sort()

        send_ids_sorted = [
            gid
            for dest_rank in dp_ranks
            for gid in combined_sample_id_groups[dest_rank]
            if gid in gid2local_id
        ]

        recv_sample_id_groups = [[] for _ in range(self.total_hdp_gpus)]
        for gid in combined_sample_id_groups[hdp_rank]:
            src_rank = self._gid_to_src_rank(gid, offsets)
            recv_sample_id_groups[src_rank].append(gid)
        recv_ids_sorted = [
            gid
            for source_rank in range(self.total_hdp_gpus)
            for gid in recv_sample_id_groups[source_rank]
        ]
        recv_samples = [{} for _ in recv_ids_sorted]

        global_shapes = self._gather_multimodal_shapes(batch, global_ids_this_rank)
        data_keys = tuple(batch[0].keys())

        def _numel(global_id: int, key: str) -> int:
            try:
                shape = global_shapes[global_id][key]
            except KeyError as error:
                raise ValueError(
                    f"missing multimodal shape metadata for global sample {global_id}, key {key!r}"
                ) from error
            result = 1
            for dimension in shape:
                result *= dimension
            return result

        for key in data_keys:
            if not all(isinstance(sample[key], torch.Tensor) for sample in batch):
                raise ValueError(f"multimodal HybridCP field {key!r} must be a tensor")

            send_lens_split = [0] * self.total_hdp_gpus
            for dest_rank in range(self.total_hdp_gpus):
                if dest_rank not in dp_ranks:
                    continue
                send_lens_split[dest_rank] = sum(
                    _numel(gid, key)
                    for gid in combined_sample_id_groups[dest_rank]
                    if gid in gid2local_id
                )

            recv_lens_split = [
                sum(_numel(gid, key) for gid in recv_sample_id_groups[source_rank])
                for source_rank in range(self.total_hdp_gpus)
            ]

            device = torch.cuda.current_device()
            if send_ids_sorted:
                send_tensor = torch.cat(
                    [
                        batch[gid2local_id[gid]][key]
                        .to(device, non_blocking=True)
                        .reshape(-1)
                        for gid in send_ids_sorted
                    ],
                    dim=0,
                )
            else:
                send_tensor = torch.empty(
                    0,
                    device=device,
                    dtype=batch[0][key].dtype,
                )
            recv_tensor = torch.empty(
                sum(recv_lens_split), device=device, dtype=send_tensor.dtype
            )
            torch.distributed.all_to_all_single(
                recv_tensor,
                send_tensor,
                output_split_sizes=recv_lens_split,
                input_split_sizes=send_lens_split,
                group=self.dp_cp_group,
            )

            cursor = 0
            for index, global_id in enumerate(recv_ids_sorted):
                shape = global_shapes[global_id][key]
                numel = _numel(global_id, key)
                recv_samples[index][key] = recv_tensor[cursor : cursor + numel].reshape(shape)
                cursor += numel

        return {
            recv_id: recv_samples[index]
            for index, recv_id in enumerate(recv_ids_sorted)
        }

    def unpack_batch(self, batch):
        """
        Unpacks the packed samples into a list of sub-samples.
        Since each sub-sample may be routed to different DPxCP ranks,
        we unpack the sample here to avoid unnecessarily transferring
        the entire packed sample.
        """
        batch_unpacked = []
        for sample in batch:
            for sub_sample in range(sample["cu_seqlens"].shape[0] - 1):
                sub_sample_dict = {}
                start_idx = sample["cu_seqlens"][sub_sample]
                end_idx = sample["cu_seqlens"][sub_sample + 1]
                if end_idx - start_idx == 0:
                    continue
                for key in sample.keys():
                    if key in ["cu_seqlens", "batch_idx", "max_seqlen"]:
                        continue
                    sub_sample_dict[key] = sample[key][start_idx:end_idx]
                batch_unpacked.append(sub_sample_dict)
        return batch_unpacked

    def __next__(self) -> Any:
        """
        Get one scheduled global batch from the dataset.

        The external loader yields one packed item per microbatch.  Consume the
        full global batch before scheduling so dynamic CP can distribute all
        microbatches participating in the optimizer step.  The returned third
        field is the number of unique routed samples for global ``samples_seen``
        accounting; the fourth contains scheduler and modality diagnostics.
        """
        if self.data_iterator is None:
            # TP0 reads from data_iterator, others receive via broadcast.
            return None, None, None, None

        from megatron.core.num_microbatches_calculator import get_num_microbatches

        num_microbatches = get_num_microbatches()
        if num_microbatches is None:
            num_microbatches = 1
        raw_batches = collect_hybrid_cp_microbatches(self.data_iterator, num_microbatches)
        _hybrid_cp_debug(
            f"loader raw_batches={len(raw_batches)} num_microbatches={num_microbatches}"
        )

        batch = []
        is_multimodal = False
        for raw_batch in raw_batches:
            raw_is_multimodal = isinstance(raw_batch, Mapping) and "cu_lengths" in raw_batch
            is_multimodal = is_multimodal or raw_is_multimodal
            if raw_is_multimodal:
                batch.extend(unpack_multimodal_batch(raw_batch))
            elif isinstance(raw_batch, Mapping):
                batch.append(raw_batch)
            else:
                batch.extend(raw_batch)

        local_real_lengths, local_padded_lengths = get_hybrid_cp_sample_lengths(batch)
        subsample_seqlens = torch.tensor(local_real_lengths, dtype=torch.int32).cuda()
        subsample_padded_seqlens = torch.tensor(
            local_padded_lengths, dtype=torch.int32
        ).cuda()
        _hybrid_cp_debug(
            f"loader local_samples={len(batch)} local_nonzero_samples={subsample_seqlens.numel()} "
            f"local_seqlens={subsample_seqlens.detach().cpu().tolist()} multimodal={is_multimodal}"
        )

        seqlens_gathered, offsets = self.get_global_seqlens(subsample_seqlens)
        padded_seqlens_gathered, padded_offsets = self.get_global_seqlens(
            subsample_padded_seqlens
        )
        if not torch.equal(offsets, padded_offsets):
            raise RuntimeError("HybridCP real and padded sequence gathers disagree on rank offsets")

        global_id_seqlens, global_ids_this_rank = self.get_global_id_seqlens(
            subsample_seqlens.shape[0], offsets, seqlens_gathered
        )
        global_id_padded_seqlens = {
            sample_id: int(padded_length)
            for sample_id, padded_length in enumerate(padded_seqlens_gathered)
        }

        groups, sample_id_groups = self.cp_balancing_scheduler.get_groups_and_subsamples(
            global_id_seqlens,
            self.config,
            padded_seqlens=global_id_padded_seqlens,
            pack_payloads=is_multimodal,
        )
        schedule_stats = summarize_hybrid_cp_schedule(
            global_id_seqlens,
            sample_id_groups,
            padded_seqlens=global_id_padded_seqlens,
        )
        multimodal_stats = summarize_hybrid_cp_multimodal_samples(batch if is_multimodal else [])
        multimodal_keys = tuple(multimodal_stats)
        multimodal_values = torch.tensor(
            [multimodal_stats[key] for key in multimodal_keys],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(multimodal_values, group=self.dp_group)
        schedule_stats.update(
            {
                key: float(multimodal_values[index].item())
                for index, key in enumerate(multimodal_keys)
            }
        )
        _hybrid_cp_debug(
            f"loader schedule hdp_rank={self.dp_cp_group.rank()} global_samples={len(global_id_seqlens)} "
            f"groups={len(groups)} per_group_counts="
            f"{[[len(ids) for ids in group] for group in sample_id_groups]}"
        )

        if not is_multimodal:
            batch = self.unpack_batch(batch)
        samples_this_rank_with_id = self.reroute_samples_to_hdp_ranks(
            batch,
            global_ids_this_rank,
            global_id_seqlens,
            sample_id_groups,
            offsets,
            multimodal=is_multimodal,
        )
        _hybrid_cp_debug(
            f"loader reroute received_ids={sorted(int(gid) for gid in samples_this_rank_with_id)}"
        )
        return (
            samples_this_rank_with_id,
            sample_id_groups,
            len(global_id_seqlens),
            schedule_stats,
        )
