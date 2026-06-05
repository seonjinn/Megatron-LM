# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""video processor class for Qwen2-VL."""

import math
from typing import Optional, Union

from transformers.image_processing_utils import (
    BatchFeature,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    SizeDict,
    get_image_size,
)
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)
from transformers.utils.import_utils import requires
from transformers.video_processing_utils import (
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    BaseVideoProcessor,
)
from transformers.video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
import torchvision.transforms as T

from .processing_utils import get_internvl_target_ratios, calculate_targets


if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


if is_torch_available():
    import torch


@requires(backends=("torchvision",))
class NemotronH_Nano_Omni_Reasoning_V3VideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, image_size=512, max_num_tiles=12, norm_mean=None, norm_std=None, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        video_metadata: Union[list[VideoMetadata], list[dict]],
        do_sample_frames: bool,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional["torch.Tensor"] = None,
        **kwargs,
    ):
        if do_sample_frames:
            # Sample video frames
            videos = [
                self.sample_frames(
                    video,
                    metadata=metadata,
                    num_frames=num_frames,
                    fps=fps,
                )
                for video, metadata in zip(videos, video_metadata)
            ]

        # We need to sample frames first before moving to device, if `do_sample_frames=True`. Otherwise
        # moving the whole video incurs high GPU mem usage for long videos
        if device is not None:
            videos = [video.to(device) for video in videos]

        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            batch_size, grid_t, channel = stacked_videos.shape[:3]

            target_ratios = get_internvl_target_ratios(1, self.max_num_tiles)
            blocks, resize_width, resize_height = calculate_targets(
                width,
                height,
                target_ratios,
                self.image_size
            )
            stacked_videos = self.resize(
                image=stacked_videos,
                size=SizeDict(height=resize_height, width=resize_width),
                interpolation=T.InterpolationMode.BICUBIC,
            )
            # stacked_videos = T.Resize((resize_width, resize_height), interpolation=T.InterpolationMode.BICUBIC)(stacked_videos)
            norm_mean = torch.as_tensor(self.norm_mean, dtype=stacked_videos.dtype, device=stacked_videos.device).view(1, 1, 3, 1, 1)
            norm_std  = torch.as_tensor(self.norm_std,  dtype=stacked_videos.dtype, device=stacked_videos.device).view(1, 1, 3, 1, 1)
            stacked_videos = (stacked_videos - norm_mean) / norm_std
            resized_videos_grouped[shape] = stacked_videos
            grid_h, grid_w = resize_height // self.image_size, resize_width // self.image_size
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(resized_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )

    def get_num_of_video_patches(self, num_frames: int, height: int, width: int):
        """
        A utility that returns number of video patches a given video size.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video.
            width (`int`):
                Width of the input video.
        Returns:
            `Tuple(int, int)`: Number of placeholder tokens required and number of patches per image.
        """
        target_ratios = get_internvl_target_ratios(1, self.max_num_tiles)
        blocks, _, _ = calculate_targets(
            width,
            height,
            target_ratios,
            self.image_size
        )
        return num_frames * blocks


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3VideoProcessor"]
