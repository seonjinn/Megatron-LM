# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MULTIMODAL_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_EXAMPLE_DIR))

import dataloader_provider  # noqa: E402


@pytest.mark.parametrize(
    ("num_workers", "configured_prefetch_factor", "expected_prefetch_factor"),
    [(0, 4, None), (1, 4, 4), (1, None, 2)],
)
def test_new_dataloader_prefetch_factor(
    monkeypatch, num_workers, configured_prefetch_factor, expected_prefetch_factor
):
    arg_values = dict(
        dataloader_seed=0,
        encoder_pipeline_model_parallel_size=0,
        load=None,
        num_workers=num_workers,
        packing_buffer_size=10000,
    )
    if configured_prefetch_factor is not None:
        arg_values["dataloader_prefetch_factor"] = configured_prefetch_factor
    args = SimpleNamespace(**arg_values)
    loader_kwargs = {}

    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)
    monkeypatch.setattr(dataloader_provider, "is_dataloader_rank", lambda _: True)
    monkeypatch.setattr(dataloader_provider.parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dataloader_provider.parallel_state, "get_data_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        dataloader_provider.parallel_state, "get_data_parallel_group", lambda: object()
    )
    monkeypatch.setattr(dataloader_provider, "WorkerConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        dataloader_provider,
        "datasets_provider",
        lambda task_encoder, worker_config: ("train-dataset", None, None),
    )
    monkeypatch.setattr(dataloader_provider, "use_new_dataloader_path", lambda: True)
    monkeypatch.setattr(dataloader_provider, "FileStoreCachePool", lambda **kwargs: kwargs)

    def fake_get_savable_loader(dataset, **kwargs):
        loader_kwargs.update(kwargs)
        return [dataset]

    monkeypatch.setattr(dataloader_provider, "get_savable_loader", fake_get_savable_loader)

    train_loader, valid_loader, test_loader = (
        dataloader_provider.train_valid_test_dataloaders_provider(
            train_val_test_num_samples=None, task_encoder=object()
        )
    )

    assert train_loader._dataloader == ["train-dataset"]
    assert valid_loader is None
    assert test_loader._dataloader is None
    if expected_prefetch_factor is None:
        assert "prefetch_factor" not in loader_kwargs
    else:
        assert loader_kwargs["prefetch_factor"] == expected_prefetch_factor


def test_new_dataloader_rejects_non_positive_prefetch_factor(monkeypatch):
    args = SimpleNamespace(dataloader_prefetch_factor=0)
    monkeypatch.setattr(dataloader_provider, "get_args", lambda: args)

    with pytest.raises(ValueError, match="must be positive"):
        dataloader_provider.train_valid_test_dataloaders_provider(
            train_val_test_num_samples=None, task_encoder=object()
        )
