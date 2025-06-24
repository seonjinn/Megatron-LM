# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
from typing import Optional

from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode

IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
SIGLIP_PIXEL_MEAN = [0.5, 0.5, 0.5]
SIGLIP_PIXEL_STD = [0.5, 0.5, 0.5]
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
RADIO_G_PIXEL_MEAN = [0.4850, 0.4560, 0.4060]
RADIO_G_PIXEL_STD = [0.2230, 0.2240, 0.2250]


pixel_statistics = {
    "clip": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "siglip": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
    "internvit": (IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD),
    "radio": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "radio-g": (RADIO_G_PIXEL_MEAN, RADIO_G_PIXEL_STD),
    "huggingface": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
    "radio_siglip_move": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "cradio-v1": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "cradio-g": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
}


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
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


def find_closest_area_weighted_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
):
    """
    Find the best number of tiles based on the aspect ratio and the area covered by the tiles.
    """
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor_based_on_area_n_ratio = min(
            (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
        ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio
    return best_ratio


class ImageTransform:
    """Image transformation."""

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276

    def __init__(
        self,
        vision_model_type: str,
        use_tiling: bool = False,
        tile_size: int = 224,
        use_thumbnail: bool = False,
        augment: bool = False,
        min_num_tiles: int = 1,
        max_num_tiles: int = 1,
        find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
    ):
        # print(f"Transformation params: {vision_model_type=}, {use_tiling=}, {tile_size=}, {use_thumbnail=}, {augment=}, {min_num_tiles=}, {max_num_tiles=}, {find_closest_aspect_ratio_fn=}")
        self._transform = _build_transform(tile_size, vision_model_type)
        self._vision_model_type = vision_model_type
        self._use_tiling = use_tiling
        self._tile_size = tile_size
        self._use_thumbnail = use_thumbnail
        self._augment = augment
        self._min_num_tiles = min_num_tiles
        self._max_num_tiles = max_num_tiles
        self._find_closest_aspect_ratio_fn = find_closest_aspect_ratio_fn

        # Calculate all possible aspect ratios for each max_num_tiles.
        self.target_ratios = {
            max_num_tiles: sorted(set(
                (x, y)
                for n in range(self._min_num_tiles, max_num_tiles + 1)
                for x in range(1, n + 1)
                for y in range(1, n + 1)
                if x * y <= max_num_tiles and x * y >= self._min_num_tiles
            ), key=lambda x: x[0] * x[1])
            for max_num_tiles in range(self._min_num_tiles, self._max_num_tiles + 1)
        }

        assert not augment, "Image augmentation not implemented."

    def transform(self, image: Image.Image, tiling: tuple[int, int]):
        # calculate the target width and height
        target_width = self._tile_size * tiling[0]
        target_height = self._tile_size * tiling[1]
        blocks = tiling[0] * tiling[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self._tile_size)) * self._tile_size,
                (i // (target_width // self._tile_size)) * self._tile_size,
                ((i % (target_width // self._tile_size)) + 1) * self._tile_size,
                ((i // (target_width // self._tile_size)) + 1) * self._tile_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if self._use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self._tile_size, self._tile_size))
            processed_images.append(thumbnail_img)

        return [self._transform(img) for img in processed_images]

    def compute_tiling(
        self, img_size: tuple[int, int], max_num_tiles: Optional[int] = None
    ) -> tuple[tuple[int, int], int]:
        if self._use_tiling:
            if max_num_tiles is None:
                max_num_tiles = self._max_num_tiles
            else:
                assert self._min_num_tiles <= max_num_tiles <= self._max_num_tiles, (
                    f"max_num_tiles={max_num_tiles} must be between {self._min_num_tiles} and {self._max_num_tiles}"
                )

            aspect_ratio = img_size[0] / img_size[1]

            # calculate the existing image aspect ratio
            target_ratios = self.target_ratios[max_num_tiles]

            # find the closest aspect ratio to the target
            tiling = self._find_closest_aspect_ratio_fn(
                aspect_ratio, target_ratios, img_size[0], img_size[1], self._tile_size
            )
            num_tiles = tiling[0] * tiling[1]
            if self._use_thumbnail and num_tiles != 1:
                num_tiles += 1

            return tiling, num_tiles
        else:
            return (1, 1), 1
    
    def __str__(self):
        return f"ImageTransform(vision_model_type={self._vision_model_type}, use_tiling={self._use_tiling}, tile_size={self._tile_size}, use_thumbnail={self._use_thumbnail}, augment={self._augment}, min_num_tiles={self._min_num_tiles}, max_num_tiles={self._max_num_tiles}, find_closest_aspect_ratio_fn={self._find_closest_aspect_ratio_fn})"


class TileDegradationMap:
    def __init__(
        self,
        tile_degradation_map: dict[int, int] = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1, 1: 1},
        max_num_tiles: int = 12,
    ):
        self._tile_degradation_map = tile_degradation_map
        self._max_num_tiles = max_num_tiles

    def compute_tilings(
        self,
        img_sizes: list[tuple[int, int]],
        img_transforms: "list[ImageTransform]",
        tiles_available: int,
    ) -> tuple[list[tuple[int, int]], list[int]]:
        max_num_tiles = self._max_num_tiles
        while True:
            tilings = []
            img_num_tiles = []
            for img_size, img_transform in zip(img_sizes, img_transforms):
                tiling, num_tiles = img_transform.compute_tiling(
                    img_size, max_num_tiles
                )
                img_num_tiles.append(num_tiles)
                tilings.append(tiling)
            if max_num_tiles == 1:
                break
            if sum(img_num_tiles) > tiles_available:
                if max_num_tiles in self._tile_degradation_map:
                    max_num_tiles = self._tile_degradation_map[max_num_tiles]
                else:
                    raise RuntimeError(
                        (
                            f"Tried to decrease the number of tiles {max_num_tiles} but it's not ",
                            f"defined in the degradation map {self._tile_degradation_map}",
                        )
                    )
            else:
                break
        return tilings, img_num_tiles
    
    def __str__(self):
        return f"TileDegradationMap(tile_degradation_map={self._tile_degradation_map}, max_num_tiles={self._max_num_tiles})"

class NoopTileDegradationMap:
    def __init__(self, max_num_tiles: int = 12):
        self._max_num_tiles = max_num_tiles

    def compute_tilings(
            self,
            img_sizes: list[tuple[int, int]],
            img_transforms: list[ImageTransform],
            tiles_available: int,
    ) -> tuple[list[tuple[int, int]], list[int]]:
        max_num_tiles = self._max_num_tiles
        tilings = []
        img_num_tiles = []
        for img_size, img_transform in zip(img_sizes, img_transforms):
            tiling, num_tiles = img_transform.compute_tiling(
                img_size, max_num_tiles
            )
            img_num_tiles.append(num_tiles)
            tilings.append(tiling)
        return tilings, img_num_tiles


# Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
# and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276
def _build_transform(input_size: int, vision_model_type: str):
    if vision_model_type in ("siglip", "internvit", "radio", "radio-g"):
        pixel_mean, pixel_std = pixel_statistics[vision_model_type]

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )
    # From the official CLIP repo.
    elif vision_model_type == "clip":
        pixel_mean, pixel_std = pixel_statistics[vision_model_type]

        transform = Compose(
            [
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )
    elif vision_model_type.startswith("hf://"):
        from megatron.core.models.huggingface.module import get_hf_model_type

        model_type = get_hf_model_type(vision_model_type)
        if "siglip" in model_type:
            from transformers.models.siglip.image_processing_siglip import (
                SiglipImageProcessor,
            )

            processor = SiglipImageProcessor(
                size={"height": input_size, "width": input_size}
            )

            def transform(x):
                x = x.convert("RGB") if x.mode != "RGB" else x
                x = processor(x, return_tensors="pt")
                return x["pixel_values"][0]
        else:
            raise NotImplementedError(
                f"image processing not defined for huggingface model {vision_model_type}"
            )
    else:
        raise NotImplementedError(
            f"image processing not defined for vision model {vision_model_type}"
        )

    return transform
