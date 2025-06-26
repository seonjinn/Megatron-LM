# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Callable, Optional

import einops
import torch
from torchvision import transforms as T
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode

from data_loading.conversation_sample import (
    ImageMedia,
    VideoFrameMedia,
)

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


def find_closest_area_weighted_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
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


@dataclass
class ImageTilingParams:
    media: ImageMedia | VideoFrameMedia
    num_tiles: int
    num_embeddings: int


class ImageTilingStrategy(ABC):
    """
    Base class for image tiling strategies.
    A tiling strategy is a function that takes a list of media and returns a list of image tiling parameters.
    These can then be used to apply the tiling to the media.

    Subclasses must implement the `compute_params` and `apply_params` methods.

    The `transform` method is a convenience method that computes the transformation parameters and applies the transformation to the media.

    """

    def transform(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Transform the media and compute the transformation parameters.
        """
        transform_media_list = self.compute_params(media_list, num_tokens_available)
        return [
            self.apply_params(transform_media)
            for transform_media in transform_media_list
        ]

    @abstractmethod
    def compute_params(
        self, media_list: list[ImageMedia | VideoFrameMedia], num_tokens_available: int
    ) -> list[ImageTilingParams]:
        """
        Compute the transformation parameters and the number of tokens to use for the media.

        Args:
            media_list: List of media to transform
            num_tokens_available: Number of tokens available for all media

        Returns:
            list of transformation parameters with the media
        """
        ...

    @abstractmethod
    def apply_params(self, transform_media: ImageTilingParams) -> list[torch.Tensor]:
        """
        Apply the transformation parameters to the media.

        Args:
            transform_media: The media to apply the transformation to

        Returns:
            list of transformed media tensors
        """
        ...

    @abstractmethod
    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int], list[int]]:
        """
        Stack the images into a single tensor.

        Args:
            media_list: List of images to stack

        Returns:
            tuple of (stacked media, image sizes, vision cu lengths, vision max lengths)
        """
        ...


class _FixedSizeStrategy(ImageTilingStrategy):
    """
    Base class for fixed size image tiling strategies.
    """

    def __init__(
        self,
        vision_model_type: str,
        target_width: int,
        target_height: int,
        embeddings_per_image: int,
    ):
        self._vision_model_type = vision_model_type
        self._target_width = target_width
        self._target_height = target_height
        self._embeddings_per_image = embeddings_per_image
        self._transform = self._build_transform(
            (target_width, target_height), vision_model_type
        )

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276
    @staticmethod
    def _build_transform(target_size: tuple[int, int], vision_model_type: str):
        """
        Build a transform for a given vision model type and target size.
        """
        if vision_model_type in ("siglip", "internvit", "radio", "radio-g"):
            pixel_mean, pixel_std = pixel_statistics[vision_model_type]

            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (target_size[1], target_size[0]),
                        interpolation=InterpolationMode.BICUBIC,
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
                        (target_size[1], target_size[0]),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
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
                    size={"height": target_size[1], "width": target_size[0]}
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

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int] | None, list[int] | None]:
        return (
            torch.stack(images),
            torch.tensor(
                [(img.shape[1], img.shape[2]) for img in images], dtype=torch.int32
            ),
            None,
            None,
        )


class NoTilingStrategy(_FixedSizeStrategy):
    """
    A simple image transformation that resizes the image to the target width and height.
    """

    def __init__(
        self,
        vision_model_type: str,
        target_width: int,
        target_height: int,
        embeddings_per_image: int,
        augment: bool = False,
    ):
        super().__init__(
            vision_model_type=vision_model_type,
            target_width=target_width,
            target_height=target_height,
            embeddings_per_image=embeddings_per_image,
        )

        assert not augment, "Image augmentation not implemented."

    def apply_params(self, transform_media: ImageTilingParams) -> list[torch.Tensor]:
        return [self._transform(transform_media.media.value)]

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: Optional[int] = None,
    ) -> list[ImageTilingParams]:
        return [
            ImageTilingParams(
                media=media, num_tiles=1, num_embeddings=self._embeddings_per_image
            )
            for media in media_list
        ]

    def __str__(self):
        return f"SimpleImageTransform(vision_model_type={self._vision_model_type}, num_tokens_per_image={self._embeddings_per_image})"


@dataclass
class ImageTilingParamsV1(ImageTilingParams):
    tiling: tuple[int, int]


class ImageTilingStrategyV1(_FixedSizeStrategy):
    """Tiling image transformation.

    This transformation splits the image into a grid of tiles and applies the transformation to each tile.
    """

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276

    def __init__(
        self,
        vision_model_type: str,
        tile_size: int,
        use_thumbnail: bool,
        augment: bool,
        min_num_tiles: int,
        max_num_tiles: int,
        embeddings_per_tile: int,
        find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
    ):
        super().__init__(
            vision_model_type=vision_model_type,
            target_width=tile_size,
            target_height=tile_size,
            embeddings_per_image=embeddings_per_tile,
        )

        # print(f"Transformation params: {vision_model_type=}, {use_tiling=}, {tile_size=}, {use_thumbnail=}, {augment=}, {min_num_tiles=}, {max_num_tiles=}, {find_closest_aspect_ratio_fn=}")
        self._tile_size = tile_size
        self._use_thumbnail = use_thumbnail
        self._min_num_tiles = min_num_tiles
        self._max_num_tiles = max_num_tiles
        self._find_closest_aspect_ratio_fn = find_closest_aspect_ratio_fn

        # Calculate all possible aspect ratios for each max_num_tiles.
        self.target_ratios = {
            max_num_tiles: sorted(
                set(
                    (x, y)
                    for n in range(self._min_num_tiles, max_num_tiles + 1)
                    for x in range(1, n + 1)
                    for y in range(1, n + 1)
                    if x * y <= max_num_tiles and x * y >= self._min_num_tiles
                ),
                key=lambda x: x[0] * x[1],
            )
            for max_num_tiles in range(self._min_num_tiles, self._max_num_tiles + 1)
        }

        assert not augment, "Image augmentation not implemented."

    def apply_params(self, transform_media: ImageTilingParams) -> list[torch.Tensor]:
        assert isinstance(transform_media, ImageTilingParamsV1)
        image = transform_media.media.value
        # calculate the target width and height
        target_width = self._tile_size * transform_media.tiling[0]
        target_height = self._tile_size * transform_media.tiling[1]
        blocks = transform_media.tiling[0] * transform_media.tiling[1]

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

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: Optional[int] = None,
    ) -> list[ImageTilingParamsV1]:
        max_num_tiles = min(
            num_tokens_available // self._embeddings_per_image, self._max_num_tiles
        )

        # calculate the existing image aspect ratio
        target_ratios = self.target_ratios[max_num_tiles]

        params = []
        for media in media_list:
            if isinstance(media, ImageMedia):
                img_size = (media.width, media.height)
            elif isinstance(media, VideoFrameMedia):
                img_size = (media.video_width, media.video_height)
            else:
                raise ValueError(f"Unsupported media type: {type(media)}")

            aspect_ratio = img_size[0] / img_size[1]

            # find the closest aspect ratio to the target
            tiling = self._find_closest_aspect_ratio_fn(
                aspect_ratio, target_ratios, img_size[0], img_size[1], self._tile_size
            )
            num_tiles = tiling[0] * tiling[1]
            if self._use_thumbnail and num_tiles != 1:
                num_tiles += 1

            params.append(
                ImageTilingParamsV1(
                    media=media,
                    num_tiles=num_tiles,
                    num_embeddings=num_tiles * self._embeddings_per_image,
                    tiling=tiling,
                )
            )

        return params

    def __str__(self):
        return f"TilingImageTransform(vision_model_type={self._vision_model_type}, tile_size={self._tile_size}, use_thumbnail={self._use_thumbnail}, min_num_tiles={self._min_num_tiles}, max_num_tiles={self._max_num_tiles}, embeddings_per_tile={self._embeddings_per_image}, find_closest_aspect_ratio_fn={self._find_closest_aspect_ratio_fn})"


class TileDegradationStrategy(ImageTilingStrategy):
    """Strategy for tiling images and video frames, each with their own tiling strategy, while trying to match the
    number of tokens left in the sample by reducing the number of tiles if needed.
    """

    # Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
    # and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276

    def __init__(
        self,
        image_strategy: ImageTilingStrategy,
        video_frame_strategy: ImageTilingStrategy,
        embeddings_per_tile: int,
        max_num_tiles: int,
        tile_degradation_map: dict[int, int] = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1},
    ):
        self._image_strategy = image_strategy
        self._video_frame_strategy = video_frame_strategy
        self._embeddings_per_tile = embeddings_per_tile
        self._max_num_tiles = max_num_tiles
        self._tile_degradation_map = tile_degradation_map

    def apply_params(self, transform_media: ImageTilingParams) -> list[torch.Tensor]:
        if isinstance(transform_media.media, ImageMedia):
            return self._image_strategy.apply_params(transform_media)
        elif isinstance(transform_media.media, VideoFrameMedia):
            return self._video_frame_strategy.apply_params(transform_media)
        else:
            raise ValueError(f"Unsupported media type: {type(transform_media.media)}")

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
    ) -> list[ImageTilingParams]:
        max_num_tiles = self._max_num_tiles
        while True:
            params = []
            img_num_tiles = []
            for media in media_list:
                if isinstance(media, ImageMedia):
                    media_params = self._image_strategy.compute_params(
                        [media], max_num_tiles * self._embeddings_per_tile
                    )[0]
                elif isinstance(media, VideoFrameMedia):
                    media_params = self._video_frame_strategy.compute_params(
                        [media], max_num_tiles * self._embeddings_per_tile
                    )[0]
                else:
                    raise ValueError(f"Unsupported media type: {type(media)}")
                img_num_tiles.append(media_params.num_tiles)
                params.append(media_params)
            if max_num_tiles == 1 or num_tokens_available is None:
                break
            if sum(img_num_tiles) > max_num_tiles:
                if max_num_tiles in self._tile_degradation_map:
                    max_num_tiles = self._tile_degradation_map[max_num_tiles]
                else:
                    # End of degradation
                    break
            else:
                break
        return params

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int] | None, list[int] | None]:
        return self._image_strategy.stack(images)

    def __str__(self):
        return f"TileDegradationImageTransform(max_num_tiles={self._max_num_tiles}, image_transform={self._image_strategy}, video_frame_transform={self._video_frame_strategy})"


@dataclass
class DynamicResolutionParams(ImageTilingParams):
    patch_size: tuple[int, int]


class DynamicResolutionImageTilingStrategy(ImageTilingStrategy):
    """Preprocess an image with dynamic resolution for vision transformers.

    This function resizes an image to optimize the number of patches while respecting
    constraints on minimum/maximum patches, minimum side length, and compatibility
    with pixel shuffle or convolution merging operations.

    The algorithm works by:
    1. Computing the initial patch grid size based on the image dimensions and res_step
    2. Scaling the patch grid to fit within the max_patches constraint
    3. Ensuring the result has at least min_patches
    4. Optionally enforcing a minimum side length constraint
    5. Rounding patch dimensions to even numbers for pixel_shuffle/conv_merging compatibility
    6. Resizing the image to the computed target dimensions

    Note:
        The function preserves aspect ratio as much as possible while satisfying all constraints.
        When constraints conflict (e.g., min_side vs max_patches), the function prioritizes
        staying within max_patches while maximizing the image size.

    Example:
        >>> from PIL import Image
        >>> img = Image.open("example.jpg")  # 800x600 image
        >>> strategy = DynamicResolutionImageTilingStrategy(vision_model_type="radio", min_patches=4, max_patches=64, res_step=14, get_num_embeddings=lambda x, y: x * y * 2)
        >>> params = strategy.compute_params([img])
        >>> img_tensor = strategy.apply_params(params[0])
        >>> # Returns image resized to maintain aspect ratio with 4-64 patches of size 14x14
    """

    def __init__(
        self,
        vision_model_type: str,
        min_num_patches: int,
        max_num_patches: int,
        patch_size: int,
        get_num_embeddings: Callable[[int, int], int],
        factor_max: float = 1.0,
        pixel_shuffle: bool = False,
        min_side: int | None = None,
        conv_merging: bool = False,
    ):
        """
        Args:
            vision_model_type: Vision model type.
            min_num_patches: Minimum number of patches required. Defaults to 1.
            max_num_patches: Maximum number of patches allowed. Defaults to 128.
            patch_size: Resolution step size (patch dimension). Defaults to 16.
            get_num_embeddings: Function to get the number of embeddings from the patch size (width, height).
            factor_max: Maximum scaling factor to apply. Defaults to 1.0.
            pixel_shuffle: Whether to ensure compatibility with pixel shuffle operations by rounding to even patch
                dimensions. Defaults to False.
            min_side: Minimum side length in pixels. If specified, ensures at least one side meets this constraint.
                Defaults to None.
            conv_merging: Whether to ensure compatibility with convolution merging by rounding to even patch dimensions.
                Defaults to False.
        """
        assert "radio" in vision_model_type, (
            "Dynamic resolution is only supported for radio models"
        )
        self._vision_model_type = vision_model_type
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches
        self._patch_size = patch_size
        self._get_num_embeddings = get_num_embeddings
        self._factor_max = factor_max
        self._pixel_shuffle = pixel_shuffle
        self._min_side = min_side
        self._conv_merging = conv_merging

        pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
        self._transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )

    def apply_params(self, params: DynamicResolutionParams) -> list[torch.Tensor]:
        # resize the image
        resized_img = params.media.value.resize(
            (
                params.patch_size[0] * self._patch_size,
                params.patch_size[1] * self._patch_size,
            )
        )
        return [self._transform(resized_img)]

    def compute_params(
        self,
        media_list: list[ImageMedia | VideoFrameMedia],
        num_tokens_available: int | None = None,
    ) -> list[ImageTilingParams]:
        params = []
        for media in media_list:
            if isinstance(media, ImageMedia):
                orig_width, orig_height = media.width, media.height
            elif isinstance(media, VideoFrameMedia):
                orig_width, orig_height = media.video_width, media.video_height
            else:
                raise ValueError(f"Unsupported media type: {type(media)}")

            closest_patch_height = round(orig_height / self._patch_size + 0.5)
            closest_patch_width = round(orig_width / self._patch_size + 0.5)
            patches = closest_patch_height * closest_patch_width

            factor = min(math.sqrt(self._max_num_patches / patches), self._factor_max)
            target_patch_height = math.floor(factor * closest_patch_height)
            target_patch_width = math.floor(factor * closest_patch_width)

            if target_patch_height * target_patch_width < self._min_num_patches:
                up_factor = math.sqrt(
                    self._min_num_patches / (target_patch_height * target_patch_width)
                )
                target_patch_height = math.ceil(up_factor * target_patch_height)
                target_patch_width = math.ceil(up_factor * target_patch_width)

            if (
                self._min_side is not None
                and min(target_patch_width, target_patch_height) * self._patch_size
                < self._min_side
            ):
                if target_patch_width <= target_patch_height:
                    up_factor = self._min_side / (target_patch_width * self._patch_size)
                    new_patch_height = math.ceil(up_factor * target_patch_height)
                    new_patch_width = math.ceil(up_factor * target_patch_width)

                    if new_patch_height * new_patch_width > self._max_num_patches:
                        # If only one side can be min_side, make as big as possible at native aspect ratio while staying below max_patches
                        if (
                            max(self._max_num_patches // new_patch_width, 1)
                            * self._patch_size
                            < self._min_side
                        ):
                            up_factor = math.sqrt(
                                self._max_num_patches
                                / (target_patch_height * target_patch_width)
                            )
                            target_patch_height = math.floor(
                                up_factor * target_patch_height
                            )
                            target_patch_width = math.floor(
                                up_factor * target_patch_width
                            )
                        target_patch_width = new_patch_width
                        target_patch_height = max(
                            self._max_num_patches // new_patch_width, 1
                        )
                    else:
                        target_patch_height = new_patch_height
                        target_patch_width = new_patch_width
                else:
                    up_factor = self._min_side / (
                        target_patch_height * self._patch_size
                    )
                    new_patch_height = math.ceil(up_factor * target_patch_height)
                    new_patch_width = math.ceil(up_factor * target_patch_width)

                    if new_patch_height * new_patch_width > self._max_num_patches:
                        # If only one side can be min_side, make as big as possible at native aspect ratio while staying below max_patches
                        if (
                            max(self._max_num_patches // new_patch_height, 1)
                            * self._patch_size
                            < self._min_side
                        ):
                            up_factor = math.sqrt(
                                self._max_num_patches
                                / (target_patch_height * target_patch_width)
                            )
                            target_patch_height = math.floor(
                                up_factor * target_patch_height
                            )
                            target_patch_width = math.floor(
                                up_factor * target_patch_width
                            )
                        else:
                            target_patch_height = new_patch_height
                            target_patch_width = max(
                                self._max_num_patches // new_patch_height, 1
                            )
                    else:
                        target_patch_height = new_patch_height
                        target_patch_width = new_patch_width

            # Rounds to nearest even number for pixel shuffle compatibility,
            if self._pixel_shuffle or self._conv_merging:
                if target_patch_height % 2 != 0:
                    if (
                        target_patch_height + 1
                    ) * target_patch_width <= self._max_num_patches:
                        target_patch_height += 1
                    else:
                        target_patch_height -= 1
                if target_patch_width % 2 != 0:
                    if (
                        target_patch_height * (target_patch_width + 1)
                        <= self._max_num_patches
                    ):
                        target_patch_width += 1
                    else:
                        target_patch_width -= 1
            assert target_patch_height * target_patch_width <= self._max_num_patches

            params.append(
                DynamicResolutionParams(
                    media=media,
                    num_tiles=1,
                    num_embeddings=self._get_num_embeddings(
                        target_patch_width * self._patch_size,
                        target_patch_height * self._patch_size,
                    ),
                    patch_size=(target_patch_width, target_patch_height),
                )
            )
        return params

    def stack(
        self, images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[int] | None, list[int] | None]:
        imgs_sizes = torch.tensor(
            [[img.shape[1], img.shape[2]] for img in images], dtype=torch.int32
        )

        def rearrange_img(x):
            py = x.shape[-2] // self._patch_size
            px = x.shape[-1] // self._patch_size
            x = einops.rearrange(
                x,
                "c (py yy) (px xx) -> (py px) (c yy xx)",
                py=py,
                yy=self._patch_size,
                px=px,
                xx=self._patch_size,
            )
            return x

        imgs = [rearrange_img(img) for img in images]

        current_length = 0
        max_length = 0
        vision_cu_lengths = [0]
        for img in imgs:
            if max_length < img.shape[0]:
                max_length = img.shape[0]
            current_length += img.shape[0]
            vision_cu_lengths.append(current_length)

        vision_cu_lengths = torch.tensor(vision_cu_lengths, dtype=torch.int32)
        vision_max_lengths = torch.tensor(max_length, dtype=torch.int32)

        return (
            torch.cat(imgs, dim=0).unsqueeze(0),
            imgs_sizes,
            vision_cu_lengths,
            vision_max_lengths,
        )

    def __str__(self):
        return f"DynamicResolutionImageTransform(vision_model_type={self._vision_model_type}, min_num_patches={self._min_num_patches}, max_num_patches={self._max_num_patches}, patch_size={self._patch_size}, pixel_shuffle={self._pixel_shuffle}, conv_merging={self._conv_merging})"
