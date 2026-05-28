# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import transformer_engine
except (ImportError, RuntimeError):
    import unittest.mock
    import types
    import sys
    transformer_engine = types.ModuleType("transformer_engine")
    transformer_engine.pytorch = types.ModuleType("transformer_engine.pytorch")
    transformer_engine.pytorch.Linear = unittest.mock.Mock()
    transformer_engine.pytorch.LayerNormLinear = unittest.mock.Mock()
    transformer_engine.pytorch.GroupedLinear = unittest.mock.Mock()
    transformer_engine.pytorch.DotProductAttention = unittest.mock.Mock()
    transformer_engine.pytorch.Sequential = unittest.mock.Mock()
    transformer_engine.pytorch.DelayedScaling = unittest.mock.Mock()
    transformer_engine.pytorch.CudaRNGStatesTracker = unittest.mock.Mock()
    transformer_engine.pytorch.distributed = types.ModuleType("transformer_engine.pytorch.distributed")
    transformer_engine.pytorch.distributed.CudaRNGStatesTracker = unittest.mock.Mock()
    transformer_engine.pytorch.distributed.get_all_rng_states = unittest.mock.Mock()
    transformer_engine.pytorch.distributed.activation_recompute_forward = unittest.mock.Mock()
    transformer_engine.pytorch.distributed.checkpoint = types.ModuleType("transformer_engine.pytorch.distributed.checkpoint")
    transformer_engine.pytorch.tensor = types.ModuleType("transformer_engine.pytorch.tensor")
    transformer_engine.pytorch.tensor.QuantizedTensor = unittest.mock.Mock()
    transformer_engine.pytorch.ops = types.ModuleType("transformer_engine.pytorch.ops")
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
    transformer_engine.pytorch.fp8 = types.ModuleType("transformer_engine.pytorch.fp8")
    transformer_engine.pytorch.fp8.fp8_model_init = unittest.mock.Mock()
    transformer_engine.pytorch.fp8.fp8_autocast = unittest.mock.Mock()
    transformer_engine.pytorch.fp8.check_fp8_support = unittest.mock.Mock()
    transformer_engine.pytorch.fp8.FP8GlobalStateManager = unittest.mock.Mock()
    transformer_engine.common = types.ModuleType("transformer_engine.common")
    transformer_engine.common.recipe = types.ModuleType("transformer_engine.common.recipe")
    transformer_engine.common.recipe.DelayedScaling = unittest.mock.Mock()

    sys.modules["transformer_engine"] = transformer_engine
    sys.modules["transformer_engine.pytorch"] = transformer_engine.pytorch
    sys.modules["transformer_engine.pytorch.tensor"] = transformer_engine.pytorch.tensor
    sys.modules["transformer_engine.pytorch.distributed"] = transformer_engine.pytorch.distributed
    sys.modules["transformer_engine.pytorch.fp8"] = transformer_engine.pytorch.fp8
    sys.modules["transformer_engine.common"] = transformer_engine.common
    sys.modules["transformer_engine.common.recipe"] = transformer_engine.common.recipe

import faulthandler
import gc
import os
import time
from typing import Any
import argparse

from PIL import Image
import torch

try:
    # To speed up the broken worker shutdown in torch dataloader.
    torch.utils.data._utils.worker.MP_STATUS_CHECK_INTERVAL = 0.1
    torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.1
except AttributeError:
    pass


from megatron.energon import (
    WorkerConfig,
    get_savable_loader,
    get_val_datasets,
    FileStoreCachePool,
    DefaultTaskEncoder,
    Lazy,
    stateless,
)
from megatron.energon.av import AVDecoder
from data_loading.task_encoder import MultiModalTaskEncoder
import tqdm


from data_loading.conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
)


class ValidatingTaskEncoder(
    DefaultTaskEncoder[
        ConversationSample,
        Any,
        Any,
        dict,
    ]
):
    """Stripped down task encoder for validating dataset."""

    decoder = MultiModalTaskEncoder.decoder
    cookers = MultiModalTaskEncoder.cookers

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: ConversationSample) -> ConversationSample:
        self._load_media(sample)
        # Just pass the metadata out.
        return {
            "__key__": sample.__key__,
            '__sources__': sample.__sources__,
            '__restore_key__': sample.__restore_key__,
            "__subflavors__": sample.__subflavors__,
        }

    def batch(self, samples: list[Any]) -> Any:
        assert len(samples) == 1
        return samples[0]

    def _load_media(self, sample: ConversationSample) -> None:
        """Loads all lazy media in the sample."""
        for msg in sample.conversation:
            for frag in msg.fragments:
                if isinstance(frag, str):
                    pass
                elif isinstance(frag, (ImageMedia, VideoMedia, AudioMedia, VideoFrameMedia)):
                    if isinstance(frag.value, Lazy):
                        frag.value = frag.value.get()
                    elif isinstance(frag.value, (AVDecoder, Image.Image)):
                        print("WARNING: Media should be lazy!")
                    else:
                        raise ValueError(f"Unexpected media type: {type(frag.value)}")
                else:
                    raise ValueError(f"Unexpected media type: {type(frag)}")


def main():
    parser = argparse.ArgumentParser(description="Validate multimodal datasets")
    parser.add_argument("data_path", type=str, help="Path(s) to the dataset(s)")
    parser.add_argument("--steps-per-dataset", type=int, default=1, help="Number of steps per dataset")
    args = parser.parse_args()

    faulthandler.enable()
    print(f"PID: {os.getpid()}")

    # torch.distributed.init_process_group(backend='gloo')

    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=1,
    )

    print(f"worker_config: {worker_config}")

    total_samples_iterated = 0
    total_samples = 0

    datasets = get_val_datasets(
        args.data_path,
        split_part="train",
        task_encoder=ValidatingTaskEncoder(),
        worker_config=worker_config,
        batch_size=1,
    )
    # To cope with torch's bug when instantiating multiple dataloaders and deallocating them.
    gc.collect()
    gc.freeze()
    for dataset, dataset_factory in tqdm.tqdm(datasets, desc="Datasets of recipe"):
        ds_name = str(dataset_factory.subflavors.get("name", getattr(dataset_factory, 'path', f'dataset_{total_samples_iterated}')))
        dataloader = get_savable_loader(
            dataset,
            gc_collect_every_n_steps=100000,
            cache_pool=FileStoreCachePool(
                num_workers=8,
                max_cache_size_gbytes=8,
                method="raw",
            ),
            watchdog_timeout_seconds=120,
        )

        step = -1
        start = time.time()
        total_samples += len(dataloader)
        with tqdm.tqdm(total=len(dataloader), position=1, desc=ds_name) as pbar:
            # Iterate over the train dataloader
            for step, _batch in enumerate(dataloader):
                total_samples_iterated += 1
                pbar.update(1)
                if step + 1 >= args.steps_per_dataset:
                    break
        assert step >= 0, f"Did not iterate any step of dataset {ds_name}"
        print(f"Dataset {ds_name} verified in {time.time() - start}sec with {step + 1} steps")

        # To cope with torch's bug when instantiating multiple dataloaders and deallocating them.
        del dataloader
        gc.collect()

    print("Verified all datasets")
    print(f"Total samples of datasets: {total_samples}")
    print(f"Total samples loaded: {total_samples_iterated}")


if __name__ == "__main__":
    main()
