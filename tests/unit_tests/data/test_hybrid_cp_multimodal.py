import pytest
import torch

from megatron.core.datasets.data_schedule import (
    collect_hybrid_cp_microbatches,
    get_hybrid_cp_sample_lengths,
    pack_multimodal_hybrid_cp_samples,
    prepare_hybrid_cp_payload_iterator,
    restore_multimodal_hybrid_cp_sample,
    summarize_hybrid_cp_multimodal_samples,
    unpack_multimodal_batch,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import set_hybrid_cp_metadata


def _packed_batch():
    return {
        "tokens": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64),
        "labels": torch.tensor([[99, 20, 21, 22, 23, 24, 25]], dtype=torch.int64),
        "cu_lengths": torch.tensor([[0, 2, 5]], dtype=torch.int32),
        "cu_lengths_padded": torch.tensor([[0, 2, 6]], dtype=torch.int32),
        "sample_token_lengths": [[2, 4]],
        "max_lengths": torch.tensor([4], dtype=torch.int32),
        "imgs": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        "imgs_sizes": torch.tensor([[16, 16], [32, 32]], dtype=torch.int32),
        "vision_cu_lengths": torch.tensor([[0, 2, 4]], dtype=torch.int32),
        "vision_max_lengths": torch.tensor([2], dtype=torch.int32),
        "num_tiles": torch.tensor([1, 1], dtype=torch.int32),
        "num_frames": torch.tensor([1, 1], dtype=torch.int32),
        "sample_image_counts": [[1, 1]],
        "sample_num_tiles": [[[1], [1]]],
        "sample_num_frames": [[[1], [1]]],
        "has_pad_img": torch.tensor(False),
    }


def _fixed_resolution_packed_batch():
    batch = _packed_batch()
    batch.update(
        {
            "imgs": torch.tensor(
                [[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]],
                dtype=torch.float32,
            ),
            "imgs_sizes": torch.tensor(
                [[16, 16], [16, 16], [32, 32], [32, 32]],
                dtype=torch.int32,
            ),
            "vision_cu_lengths": torch.tensor([[0]], dtype=torch.int32),
            "vision_max_lengths": torch.tensor([0], dtype=torch.int32),
            "num_tiles": torch.tensor([2, 2], dtype=torch.int32),
            "num_frames": torch.tensor([1, 1], dtype=torch.int32),
            "sample_image_counts": [[2, 2]],
            "sample_num_tiles": [[[2], [2]]],
            "sample_num_frames": [[[1], [1]]],
        }
    )
    return batch


def _media_expanded_packed_batch(*, vision_first: bool):
    image_token = 18
    vision_tokens = [image_token, 11, 12, 0]
    text_tokens = [21, 22, 23, 0]
    routed_tokens = (
        vision_tokens + text_tokens if vision_first else text_tokens + vision_tokens
    )
    ordered_tokens = [*routed_tokens, *([0] * 8)]
    image_counts = [1, 0] if vision_first else [0, 1]
    per_sample_tiles = [[1], []] if vision_first else [[], [1]]
    per_sample_frames = [[1], []] if vision_first else [[], [1]]
    return {
        "tokens": torch.tensor([ordered_tokens], dtype=torch.int64),
        "labels": torch.tensor([[99, *ordered_tokens]], dtype=torch.int64),
        # These are media-expanded LM boundaries, not raw token-array offsets.
        "cu_lengths": torch.tensor([[0, 7, 13]], dtype=torch.int32),
        "cu_lengths_padded": torch.tensor([[0, 8, 16]], dtype=torch.int32),
        "sample_token_lengths": [[4, 4]],
        "max_lengths": torch.tensor([8], dtype=torch.int32),
        "imgs": torch.tensor([[[[1.0]]]], dtype=torch.float32),
        "imgs_sizes": torch.tensor([[16, 16]], dtype=torch.int32),
        "vision_cu_lengths": torch.tensor([[0]], dtype=torch.int32),
        "vision_max_lengths": torch.tensor([0], dtype=torch.int32),
        "num_tiles": torch.tensor([1], dtype=torch.int32),
        "num_frames": torch.tensor([1], dtype=torch.int32),
        "sample_image_counts": [image_counts],
        "sample_num_tiles": [[*per_sample_tiles]],
        "sample_num_frames": [[*per_sample_frames]],
        "has_pad_img": torch.tensor(False),
    }


def test_unpack_multimodal_batch_preserves_text_and_vision_boundaries():
    samples = unpack_multimodal_batch(_packed_batch())

    assert len(samples) == 2
    assert samples[0]["cu_seqlens"].tolist() == [0, 2]
    assert samples[0]["cu_seqlens_padded"].tolist() == [0, 2]
    assert samples[0]["tokens"].tolist() == [10, 11]
    assert samples[0]["labels"].tolist() == [99, 20, 21]
    assert samples[0]["imgs"].squeeze(-1).tolist() == [1.0, 2.0]
    assert samples[0]["vision_cu_lengths"].tolist() == [0, 2]
    assert samples[1]["tokens"].tolist() == [12, 13, 14, 15]
    assert samples[1]["imgs"].squeeze(-1).tolist() == [3.0, 4.0]
    assert samples[1]["vision_cu_lengths"].tolist() == [0, 2]


def test_unpack_uses_raw_token_boundaries_when_vision_precedes_text():
    samples = unpack_multimodal_batch(
        _media_expanded_packed_batch(vision_first=True)
    )

    assert samples[0]["tokens"].tolist() == [18, 11, 12, 0]
    assert samples[1]["tokens"].tolist() == [21, 22, 23, 0]
    assert int((samples[0]["tokens"] == 18).sum().item()) == len(
        samples[0]["num_tiles"]
    )
    assert int((samples[1]["tokens"] == 18).sum().item()) == 0


def test_unpack_uses_raw_token_boundaries_when_text_precedes_vision():
    samples = unpack_multimodal_batch(
        _media_expanded_packed_batch(vision_first=False)
    )

    assert samples[0]["tokens"].tolist() == [21, 22, 23, 0]
    assert samples[1]["tokens"].tolist() == [18, 11, 12, 0]
    assert int((samples[0]["tokens"] == 18).sum().item()) == 0
    assert int((samples[1]["tokens"] == 18).sum().item()) == len(
        samples[1]["num_tiles"]
    )


def test_media_expanded_batch_round_trip_preserves_both_boundary_domains():
    batch = _media_expanded_packed_batch(vision_first=True)
    samples = unpack_multimodal_batch(batch)

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["tokens"].tolist() == [batch["tokens"][0, :8].tolist()]
    assert packed["labels"].tolist() == [batch["labels"][0, :9].tolist()]
    assert packed["cu_lengths"].tolist() == batch["cu_lengths"].tolist()
    assert packed["cu_lengths_padded"].tolist() == batch[
        "cu_lengths_padded"
    ].tolist()


def test_unpack_rejects_missing_raw_token_boundaries():
    batch = _packed_batch()
    del batch["sample_token_lengths"]

    with pytest.raises(ValueError, match="sample_token_lengths"):
        unpack_multimodal_batch(batch)


def test_unpack_rejects_raw_token_boundaries_exceeding_tensor_width():
    batch = _packed_batch()
    batch["sample_token_lengths"] = [[2, 5]]

    with pytest.raises(ValueError, match="sum to 7, but tokens has width 6"):
        unpack_multimodal_batch(batch)


def test_unpack_multimodal_batch_slices_fixed_resolution_images():
    samples = unpack_multimodal_batch(_fixed_resolution_packed_batch())

    assert samples[0]["imgs"].flatten().tolist() == [1.0, 2.0]
    assert samples[0]["imgs_sizes"].tolist() == [[16, 16], [16, 16]]
    assert samples[0]["vision_cu_lengths"].tolist() == [0]
    assert samples[0]["vision_max_lengths"].item() == 0
    assert samples[1]["imgs"].flatten().tolist() == [3.0, 4.0]
    assert samples[1]["imgs_sizes"].tolist() == [[32, 32], [32, 32]]


def test_fixed_resolution_vision_survives_unpack_pack_round_trip():
    samples = unpack_multimodal_batch(_fixed_resolution_packed_batch())

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["imgs"].flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert packed["imgs_sizes"].tolist() == [
        [16, 16],
        [16, 16],
        [32, 32],
        [32, 32],
    ]
    assert packed["num_tiles"].tolist() == [2, 2]
    assert packed["num_frames"].tolist() == [1, 1]
    assert packed["vision_cu_lengths"].tolist() == [[0]]
    assert packed["vision_max_lengths"].tolist() == [0]


def test_unpack_multimodal_batch_rejects_short_dynamic_vision_offsets():
    batch = _fixed_resolution_packed_batch()
    batch["vision_cu_lengths"] = torch.tensor([[0, 1]], dtype=torch.int32)

    with pytest.raises(
        ValueError,
        match="vision_cu_lengths has 2 offsets for 4 images",
    ):
        unpack_multimodal_batch(batch)


def test_unpack_multimodal_batch_rejects_surplus_dynamic_vision_offsets():
    batch = _packed_batch()
    batch["vision_cu_lengths"] = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    with pytest.raises(
        ValueError,
        match="vision_cu_lengths has 4 offsets for 2 images",
    ):
        unpack_multimodal_batch(batch)


def test_unpack_multimodal_batch_rejects_duplicate_dynamic_vision_offsets():
    batch = _packed_batch()
    batch["vision_cu_lengths"] = torch.tensor([[0, 2, 2]], dtype=torch.int32)

    with pytest.raises(
        ValueError,
        match="vision_cu_lengths must be strictly increasing",
    ):
        unpack_multimodal_batch(batch)


def test_unpack_multimodal_batch_rejects_fp8_padded_vision_image():
    batch = _packed_batch()
    batch["has_pad_img"] = torch.tensor(True)

    with pytest.raises(
        ValueError,
        match="does not support FP8 padded vision images",
    ):
        unpack_multimodal_batch(batch)


def test_unpack_audio_only_sample_preserves_clip_counts():
    batch = _packed_batch()
    batch.update(
        {
            "imgs": torch.tensor([[[0.0]]]),
            "imgs_sizes": torch.tensor([[0, 0]], dtype=torch.int32),
            "vision_cu_lengths": torch.tensor([[0]], dtype=torch.int32),
            "vision_max_lengths": torch.tensor([0], dtype=torch.int32),
            "num_tiles": torch.tensor([0], dtype=torch.int32),
            "num_frames": torch.tensor([0], dtype=torch.int32),
            "sample_image_counts": [[0, 0]],
            "sample_num_tiles": [[[], []]],
            "sample_num_frames": [[[], []]],
            "sound_clips": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "sound_length": torch.tensor([2], dtype=torch.int64),
            "sound_timestamps": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "num_sound_clips": torch.tensor([1], dtype=torch.int64),
            "sample_num_sound_clips": [[[1], []]],
        }
    )

    samples = unpack_multimodal_batch(batch)

    assert samples[0]["sound_clips"].tolist() == [[1.0, 2.0]]
    assert samples[0]["num_sound_clips"].tolist() == [1]
    assert samples[1]["num_sound_clips"].tolist() == [[0]]


def test_multimodal_summary_counts_media_without_dummy_placeholders():
    samples = unpack_multimodal_batch(_packed_batch())
    samples[0]["num_frames"] = torch.tensor([1, 8], dtype=torch.int32)
    samples[0]["num_tiles"] = torch.tensor([2, 3], dtype=torch.int32)
    samples[0]["num_sound_clips"] = torch.tensor([2], dtype=torch.int32)

    stats = summarize_hybrid_cp_multimodal_samples(samples)

    assert stats == {
        "hybrid_cp/samples_with_vision": 2,
        "hybrid_cp/samples_with_video": 1,
        "hybrid_cp/samples_with_audio": 1,
        "hybrid_cp/vision_tiles": 6,
        "hybrid_cp/video_frames": 8,
        "hybrid_cp/audio_clips": 2,
    }


def test_restore_multimodal_hybrid_cp_sample_rebuilds_get_batch_contract():
    sample = unpack_multimodal_batch(_packed_batch())[0]

    restored = restore_multimodal_hybrid_cp_sample(sample, local_cp_size=2)

    assert restored["tokens"].shape == (1, 2)
    assert restored["labels"].shape == (1, 3)
    assert restored["cu_lengths"].tolist() == [[0, 2]]
    assert restored["cu_lengths_padded"].tolist() == [[0, 2]]
    assert restored["max_lengths"].tolist() == [2]
    assert restored["imgs"].shape == (1, 2, 1)
    assert restored["vision_cu_lengths"].tolist() == [[0, 2]]
    assert restored["has_pad_img"].item() is False
    assert restored["local_cp_size"].item() == 2


def test_pack_multimodal_hybrid_cp_samples_preserves_text_boundaries():
    samples = unpack_multimodal_batch(_packed_batch())

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["tokens"].tolist() == [[10, 11, 12, 13, 14, 15]]
    assert packed["labels"].tolist() == [[99, 20, 21, 22, 23, 24, 25]]
    assert packed["cu_lengths"].tolist() == [[0, 2, 5]]
    assert packed["cu_lengths_padded"].tolist() == [[0, 2, 6]]
    assert packed["sample_lengths"].tolist() == [[2, 3]]
    assert packed["max_lengths"].tolist() == [3]
    assert packed["samples_seen"].item() == 2
    assert packed["local_cp_size"].item() == 2


def test_packed_payload_offsets_real_boundaries_after_internal_padding():
    batch = _packed_batch()
    batch["tokens"] = torch.tensor(
        [[10, 11, 0, 0, 12, 13, 14, 15]], dtype=torch.int64
    )
    batch["labels"] = torch.tensor(
        [[99, 20, -100, -100, 21, 22, 23, 24, 25]], dtype=torch.int64
    )
    batch["cu_lengths"] = torch.tensor([[0, 2, 7]], dtype=torch.int32)
    batch["cu_lengths_padded"] = torch.tensor([[0, 4, 8]], dtype=torch.int32)
    batch["sample_token_lengths"] = [[4, 4]]

    samples = unpack_multimodal_batch(batch)

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert samples[1]["tokens"].tolist() == [12, 13, 14, 15]
    assert samples[1]["cu_seqlens"].tolist() == [0, 3]
    assert samples[1]["cu_seqlens_padded"].tolist() == [0, 4]
    assert packed["tokens"].tolist() == [[10, 11, 0, 0, 12, 13, 14, 15]]
    assert packed["labels"].tolist() == [[99, 20, -100, -100, 21, 22, 23, 24, 25]]
    assert packed["cu_lengths"].tolist() == [[0, 2, 7]]
    assert packed["cu_lengths_padded"].tolist() == [[0, 4, 8]]


def test_packed_payload_rebuilds_vision_boundaries():
    samples = unpack_multimodal_batch(_packed_batch())

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["imgs"].shape == (1, 4, 1)
    assert packed["imgs"].flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert packed["imgs_sizes"].tolist() == [[16, 16], [32, 32]]
    assert packed["vision_cu_lengths"].tolist() == [[0, 2, 4]]
    assert packed["vision_max_lengths"].tolist() == [2]
    assert packed["num_tiles"].tolist() == [1, 1]
    assert packed["num_frames"].tolist() == [1, 1]


def test_packed_payload_drops_text_only_vision_dummy():
    samples = unpack_multimodal_batch(_packed_batch())
    samples[0].update(
        {
            "imgs": torch.tensor([[0.0]]),
            "imgs_sizes": torch.tensor([[0, 0]], dtype=torch.int32),
            "vision_cu_lengths": torch.tensor([0], dtype=torch.int32),
            "vision_max_lengths": torch.tensor([0], dtype=torch.int32),
            "num_tiles": torch.tensor([0], dtype=torch.int32),
            "num_frames": torch.tensor([0], dtype=torch.int32),
        }
    )

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["imgs"].shape == (1, 2, 1)
    assert packed["imgs"].flatten().tolist() == [3.0, 4.0]
    assert packed["imgs_sizes"].tolist() == [[32, 32]]
    assert packed["vision_cu_lengths"].tolist() == [[0, 2]]
    assert packed["num_tiles"].tolist() == [1]
    assert packed["num_frames"].tolist() == [1]


def test_packed_payload_restores_one_vision_dummy_for_all_text():
    samples = unpack_multimodal_batch(_packed_batch())
    for sample in samples:
        sample.update(
            {
                "imgs": torch.tensor([[0.0]]),
                "imgs_sizes": torch.tensor([[0, 0]], dtype=torch.int32),
                "vision_cu_lengths": torch.tensor([0], dtype=torch.int32),
                "vision_max_lengths": torch.tensor([0], dtype=torch.int32),
                "num_tiles": torch.tensor([0], dtype=torch.int32),
                "num_frames": torch.tensor([0], dtype=torch.int32),
            }
        )

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["imgs"].shape == (1, 1)
    assert packed["imgs_sizes"].tolist() == [[0, 0]]
    assert packed["vision_cu_lengths"].tolist() == [[0]]
    assert packed["vision_max_lengths"].tolist() == [0]
    assert packed["num_tiles"].tolist() == [0]
    assert packed["num_frames"].tolist() == [0]


def test_packed_payload_drops_text_only_audio_dummy():
    samples = unpack_multimodal_batch(_packed_batch())
    samples[1].update(
        {
            "sound_clips": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "sound_length": torch.tensor([2], dtype=torch.int64),
            "sound_timestamps": torch.tensor([[0.25]], dtype=torch.float32),
            "num_sound_clips": torch.tensor([1], dtype=torch.int64),
        }
    )

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["sound_clips"].tolist() == [[1.0, 2.0]]
    assert packed["sound_length"].tolist() == [2]
    assert packed["sound_timestamps"].tolist() == [[0.25]]
    assert packed["num_sound_clips"].tolist() == [1]


def test_packed_payload_ors_image_padding_flag():
    samples = unpack_multimodal_batch(_packed_batch())
    samples[1]["has_pad_img"] = torch.tensor(True)

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["has_pad_img"].item() is True


def test_packed_payload_concatenates_optional_token_fields():
    samples = unpack_multimodal_batch(_packed_batch())
    samples[0]["loss_mask"] = torch.tensor([1.0, 0.0])
    samples[1]["loss_mask"] = torch.tensor([1.0, 1.0, 0.0, 0.0])

    packed = pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)

    assert packed["loss_mask"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_packed_payload_rejects_inconsistent_optional_token_fields():
    samples = unpack_multimodal_batch(_packed_batch())
    samples[0]["loss_mask"] = torch.tensor([1.0, 0.0])

    with pytest.raises(ValueError, match="inconsistent optional key 'loss_mask'"):
        pack_multimodal_hybrid_cp_samples(samples, local_cp_size=2)


def test_packed_payload_rejects_padded_token_capacity_overflow():
    samples = unpack_multimodal_batch(_packed_batch())

    with pytest.raises(ValueError, match="6 padded tokens, capacity is 5"):
        pack_multimodal_hybrid_cp_samples(
            samples, local_cp_size=2, max_padded_tokens=5
        )


def test_prepare_hybrid_cp_payload_iterator_emits_one_packed_item():
    samples = unpack_multimodal_batch(_packed_batch())

    data_iterator = prepare_hybrid_cp_payload_iterator(
        {10: samples[0], 11: samples[1]},
        sample_ids=[10, 11],
        local_cp_size=2,
        max_padded_tokens=6,
    )

    packed = next(data_iterator)
    assert packed["tokens"].tolist() == [[10, 11, 12, 13, 14, 15]]
    assert packed["cu_lengths"].tolist() == [[0, 2, 5]]
    with pytest.raises(StopIteration):
        next(data_iterator)


def test_get_hybrid_cp_sample_lengths_keeps_real_and_padded_lengths_separate():
    samples = unpack_multimodal_batch(_packed_batch())

    real_lengths, padded_lengths = get_hybrid_cp_sample_lengths(samples)

    assert real_lengths == [2, 3]
    assert padded_lengths == [2, 4]


def test_unpack_multimodal_batch_requires_per_sample_media_metadata():
    batch = _packed_batch()
    del batch["sample_image_counts"]

    with pytest.raises(ValueError, match="sample_image_counts"):
        unpack_multimodal_batch(batch)


def test_hybrid_cp_metadata_is_attached_to_language_packed_params():
    """Dynamic CP must reach TE through the packed language metadata."""
    class Group:
        def size(self):
            return 2

    group = Group()
    params = PackedSeqParams(qkv_format="thd")

    result = set_hybrid_cp_metadata(params, local_cp_size=2, cp_group=group)

    assert result is params
    assert result.local_cp_size == 2
    assert result.cp_group is group


def test_hybrid_cp_metadata_disables_cp_for_single_rank_samples():
    params = PackedSeqParams(qkv_format="thd")

    set_hybrid_cp_metadata(params, local_cp_size=1)

    assert params.local_cp_size == 1
    assert params.cp_group is None


def test_collect_hybrid_cp_microbatches_consumes_the_global_batch():
    iterator = iter([{"id": 1}, {"id": 2}, {"id": 3}])

    result = collect_hybrid_cp_microbatches(iterator, num_microbatches=2)

    assert result == [{"id": 1}, {"id": 2}]
    assert next(iterator) == {"id": 3}


def test_collect_hybrid_cp_microbatches_rejects_empty_global_batches():
    with pytest.raises(ValueError, match="num_microbatches must be positive"):
        collect_hybrid_cp_microbatches(iter(()), num_microbatches=0)
