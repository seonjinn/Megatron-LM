from typing import List, Optional, Union, Any, Dict

from PIL import Image
import torch
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast, divide_to_patches
from transformers.image_utils import (make_list_of_images, get_image_size,
                                      get_image_type, ImageInput, ImageType, ChannelDimension)
from transformers.utils import TensorType
import torchvision.transforms as T


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_factor = float('-inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor_based_on_area_n_ratio = min(
            (ratio[0]*ratio[1]*image_size*image_size)/ area, 0.6
            )* min(
                target_aspect_ratio/aspect_ratio, aspect_ratio/target_aspect_ratio)
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
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
    resized_img = T.Resize((target_width, target_height), interpolation=T.InterpolationMode.BICUBIC)(image)
    patches = divide_to_patches(resized_img, image_size)
    assert len(patches) == blocks
    if use_thumbnail and len(patches) != 1:
        thumbnail_img = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)(image)
        patches.append(thumbnail_img)

    return patches
