from typing import List, Optional, Union, Any, Dict

from PIL import Image
import torch
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast, divide_to_patches
from transformers.image_utils import (make_list_of_images, get_image_size,
                                      get_image_type, ImageInput, ImageType, ChannelDimension)
from transformers.utils import TensorType
import torchvision.transforms as T



class NemotronNanoVLV2ImageProcessor(BaseImageProcessorFast):
    model_input_names = ["pixel_values"]

    def __init__(self, image_size=512, max_num_tiles=12, use_thumbnail=True, norm_mean=None, norm_std=None, do_rescale=True, patch_size=16, downsample_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.do_rescale = do_rescale
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

    def _process_image(
        self,
        image: ImageInput,
        **kwargs,
    ) -> torch.Tensor:
        image_type = get_image_type(image)
        if image_type == ImageType.PIL:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = T.ToTensor()(image)
        return image

    def _preprocess(
        self,
        images: List[torch.Tensor],
        image_size: int = None,
        max_num_tiles: int = None,
        use_thumbnail: bool = None,
        do_rescale: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        image_size = image_size if image_size is not None else self.image_size
        max_num_tiles = max_num_tiles if max_num_tiles is not None else self.max_num_tiles
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale

        images = make_list_of_images(images)

        all_patches = []
        num_patches = []
        for image in images:
            patches = dynamic_preprocess(image, image_size, max_num_tiles, use_thumbnail)
            all_patches.extend(patches)
            num_patches.append(len(patches))

        pixel_values = torch.stack(all_patches, dim=0)
        norm_mean = torch.Tensor(self.norm_mean).view(1, 3, 1, 1)
        norm_std = torch.Tensor(self.norm_std).view(1, 3, 1, 1)
        pixel_values = (pixel_values - norm_mean) / norm_std
        return BatchFeature(data={"pixel_values": pixel_values, "num_patches": num_patches}, tensor_type=return_tensors)


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def calculate_targets(
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    return blocks, target_width, target_height


def dynamic_preprocess(image, image_size=512, max_num_tiles=12, use_thumbnail=True):
    orig_height, orig_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
    target_ratios = get_internvl_target_ratios(1, max_num_tiles)

    blocks, target_width, target_height = calculate_targets(
        orig_width,
        orig_height,
        target_ratios,
        image_size
    )
    # resize the image
    resized_img = T.Resize((target_height, target_width), interpolation=T.InterpolationMode.BICUBIC)(image)
    patches = divide_to_patches(resized_img, image_size)
    assert len(patches) == blocks
    if use_thumbnail and len(patches) != 1:
        thumbnail_img = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)(image)
        patches.append(thumbnail_img)

    return patches
