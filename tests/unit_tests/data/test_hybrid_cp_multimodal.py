import pytest
import torch

from megatron.core.datasets.data_schedule import (
    restore_multimodal_hybrid_cp_sample,
    unpack_multimodal_batch,
)


def _packed_batch():
    return {
        "tokens": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64),
        "labels": torch.tensor([[99, 20, 21, 22, 23, 24, 25]], dtype=torch.int64),
        "cu_lengths": torch.tensor([[0, 2, 5]], dtype=torch.int32),
        "cu_lengths_padded": torch.tensor([[0, 2, 6]], dtype=torch.int32),
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
    assert restored["local_cp_size"].item() == 2


def test_unpack_multimodal_batch_requires_per_sample_media_metadata():
    batch = _packed_batch()
    del batch["sample_image_counts"]

    with pytest.raises(ValueError, match="sample_image_counts"):
        unpack_multimodal_batch(batch)
