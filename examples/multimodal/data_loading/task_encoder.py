# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import json
import os
import random
import sys
from collections import defaultdict
from typing import Hashable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.energon import (
    Batch,
    Cooker,
    DefaultTaskEncoder,
    Lazy,
    Sample,
    SampleDecoder,
    stateless,
)
from megatron.energon.edataclass import edataclass
from megatron.energon.av import AVDecoder
from megatron.training import get_args, get_tokenizer

from .conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    Media,
    VideoFrameMedia,
    VideoMedia,
)
from .cookers.conversation import cook_conversation
from .cookers.eagle import cook_eagle
from .image_processing import (
    ImageTilingStrategy,
    NoopTileDegradationMap,
    TileDegradationMap,
    find_closest_area_weighted_aspect_ratio,
    find_closest_aspect_ratio,
)
from .knapsacks import (
    balanced_greedy_knapsack,
    greedy_knapsack,
)

try:
    from megatron.core.models.multimodal.context_parallel import get_padding
except ImportError:
    get_padding = None


@edataclass
class ConversationTaskSample(Sample):
    # (c, h, w)
    imgs: List[torch.Tensor]
    num_tiles: torch.Tensor
    tokens: torch.Tensor
    total_len: int  # Total token count in the sample, including text and image tokens
    total_len_padded: int  # Total token count in the sample, including text and image tokens, padded to the context parallel size
    labels: torch.Tensor = None


@edataclass
class PreprocessedImageMedia:
    media: ImageMedia | VideoFrameMedia
    tiling: tuple[int, int]
    size: tuple[int, int]


@edataclass
class PreEncodedTaskSample(Sample):
    tokens: torch.Tensor
    labels: torch.Tensor
    images: list[PreprocessedImageMedia]
    num_tiles: list[int]
    total_len: int
    total_len_padded: int


@edataclass
class PackedTaskSample(Sample):
    """Dataclass to store a single packed sample (not a batch).

    P = Number of sub-samples in the packed sample
    seq_len = Total sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    __key__: list[str]  # Sample name
    tokens: torch.Tensor  # Input tokens packed into a single tensor (seq_len,)
    labels: torch.Tensor  # Target tokens packed into a single tensor (seq_len,)
    imgs: List[torch.Tensor]  # Input images
    num_tiles: list[int]  # Number of tiles for each image of each sample (num_imgs)
    max_length: int  # Maximum length across sub-samples.
    cu_lengths: List[
        int
    ]  # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths_padded: List[
        int
    ]  # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    samples_seen: int  # Number of samples in the packed sample


# Typing for the resulting batch data after encode_batch()
@edataclass
class BatchedPackedTaskSample(Batch):
    """Dataclass to store a batch of packed samples.

    N = Batch size
    P = Number of samples in the packed sample
    seq_len = Maximum sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    tokens: torch.Tensor  # Input tokens packed and padded (N, seq_len)
    labels: torch.Tensor  # Target tokens packed and padded (N, seq_len)
    imgs: (
        torch.Tensor
    )  # All image tiles stacked into a single tensor (num_tiles, C, H, W)
    num_tiles: List[List[int]]  # Number of tiles per image (N, num_imgs)
    max_lengths: List[int]  # Maximum length across sub-samples (N,)
    cu_lengths: List[
        List[int]
    ]  # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths_padded: List[
        List[int]
    ]  # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    samples_seen: int  # Number of samples in the packed batch


class LegacyConversation(TypedDict):
    """Typing for the conversation format used by the legacy tokenizer."""

    role: Literal["user", "assistant", "system"]
    content: str


class MultiModalTaskEncoder(
    DefaultTaskEncoder[
        ConversationSample,
        Union[ConversationTaskSample, PackedTaskSample],
        BatchedPackedTaskSample,
        dict,
    ]
):
    """A simple task encoder for VLMs (LlavaSample only)."""

    decoder = SampleDecoder(
        image_decode="pil", av_decode="AVDecoder", guess_content=True
    )

    cookers = [
        Cooker(cook_eagle, has_subflavors={"cook": "eagle"}),
        Cooker(cook_conversation, has_subflavors={"cook": "conversation"}),
    ]

    def __init__(self, is_val: bool = False):
        super().__init__()
        self.args = get_args()
        self.tokenizer = get_tokenizer()
        with open(self.args.prompt_path, "r") as f:
            self.manual_prompts = json.load(f)
        # TODO: There is four seq lengths now: seq_length, decoder_seq_length, dataloader_seq_length, packing_seq_length
        # Why do we need all of these? Can we simplify?
        # The docs are not clear.
        self.dataloader_seq_length = self.args.dataloader_seq_length
        # TODO: It's unclear why we need this separately. This should be the same as dataloader_seq_length which should be decoder_seq_length?
        self.packing_seq_length = self.args.packing_seq_length
        self.is_packing_enabled = (
            self.args.packing_buffer_size is not None
            and self.args.packing_buffer_size > 0
            and not is_val
        )
        if self.dataloader_seq_length and self.packing_seq_length:
            assert self.dataloader_seq_length >= self.packing_seq_length, (
                "dataloader sequence length must be greater than or equal to the packing sequence length"
            )

        if self.is_packing_enabled:
            assert self.packing_seq_length > 0, "packing sequence length must be set"

        self.get_num_image_embeddings = ImageEmbeddings(
            img_h=self.args.img_h,
            img_w=self.args.img_w,
            patch_dim=self.args.patch_dim,
            vision_model_type=self.args.vision_model_type,
            disable_vision_class_token=self.args.disable_vision_class_token,
            class_token_len=1,
            pixel_shuffle=self.args.pixel_shuffle,
            use_tile_tags=self.args.use_tile_tags,
            max_num_tiles=self.args.max_num_tiles,
            tokenizer_type=self.args.tokenizer_prompt_format,
            use_image_break_token=self.args.image_break_token is not None,
            conv_merging=self.args.conv_merging,
            dynamic=self.args.dynamic_resolution,
        )

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        # This map is used to reduce the number of tiles used per image if the number of tokens is
        # larger than the decoder_seq_length.
        self.num_tiles_degradation_map = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1, 1: 1}

        assert self.args.img_h == self.args.img_w, "img_h and img_w must be the same"
        self.transform_image = ImageTilingStrategy(
            vision_model_type=self.args.vision_model_type,
            use_tiling=self.args.use_tiling,
            tile_size=self.args.img_h,
            use_thumbnail=self.args.use_thumbnail,
            augment=False,
            min_num_tiles=1,
            max_num_tiles=self.args.max_num_tiles,
            find_closest_aspect_ratio_fn=(
                find_closest_area_weighted_aspect_ratio
                if self.args.use_area_weighted_aspect_ratio
                else find_closest_aspect_ratio
            ),
        )
        self.transform_video_frame = ImageTilingStrategy(
            vision_model_type=self.args.vision_model_type,
            use_tiling=False,
            use_thumbnail=True,
            tile_size=self.args.img_h,
            augment=False,
            min_num_tiles=1,
            max_num_tiles=1,
        )

        if self.args.dynamic_resolution:
            self.tile_degradation_map = NoopTileDegradationMap(
                max_num_tiles=self.args.max_num_tiles,
            )
        else:
            self.tile_degradation_map = TileDegradationMap(
                max_num_tiles=self.args.max_num_tiles,
            )

        if self.args.packing_knapsack_algorithm == "greedy_knapsack":
            self.packing_knapsack_algorithm = greedy_knapsack
        elif self.args.packing_knapsack_algorithm == "balanced_greedy_knapsack":
            self.packing_knapsack_algorithm = balanced_greedy_knapsack
        else:
            raise ValueError(
                f"Unknown knapsack algorithm: {self.args.packing_knapsack_algorithm}"
            )
        print(f"TaskEncoder params:\n  {self.packing_seq_length=}\n  {self.num_image_embeddings_per_tile=}\n  {self.transform_image=}\n  {self.transform_video_frame=}\n  {self.tile_degradation_map=}\n  {self.packing_knapsack_algorithm=}")

    @staticmethod
    def get_seq_frames_v2(
        total_num_frames: int,
        desired_num_frames: int = -1,
        stride: int = -1,
        temporal_jitter: bool = False,
    ) -> list[int]:
        """
        Calculate the indices of frames to extract from a video.

        Parameters:
            total_num_frames: Total number of frames in the video.
            desired_num_frames: Desired number of frames to extract.
            stride: Stride of the frames to extract.
            temporal_jitter: Whether to jitter the frames.

        Returns:
            List of indices of frames to extract.
        """

        assert (
            desired_num_frames > 0
            or stride > 0
            and not (desired_num_frames > 0 and stride > 0)
        )

        if stride > 0:
            desired_num_frames = len(list(range(0, total_num_frames, stride)))

        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(total_num_frames - 1) / desired_num_frames
        # print(f"seg_size: {seg_size}")

        # Calculate start and end indices for all segments at once
        i = torch.arange(desired_num_frames)
        starts = torch.round(seg_size * i).to(torch.int32)
        ends = torch.round(seg_size * (i + 1)).to(torch.int32)

        # Calculate middle indices for all segments
        seq = ((starts + ends) // 2).tolist()

        if temporal_jitter:
            shift_base = int(seg_size / 2)
            # Generate random shifts for all frames at once
            random_shifts = torch.randint(-shift_base, shift_base + 1, (len(seq),))
            seq = torch.tensor(seq) + random_shifts
            # Clip values to valid range
            seq = torch.clamp(seq, 0, total_num_frames - 1).tolist()

        return seq

    @staticmethod
    def get_seq_frames_v3(
        total_duration: float,
        desired_num_frames: int = 0,
        temporal_jitter: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the timestamps of frames to extract from a video.

        Parameters:
            total_duration: Total duration of the video.
            desired_num_frames: Desired number of frames to extract.
            temporal_jitter: Whether to jitter the frames.

        Returns:
            List of timestamps of frames to extract.
        """
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(total_duration - 1) / desired_num_frames
        # print(f"seg_size: {seg_size}")

        # Middle of each segment
        seq = seg_size * (
            torch.arange(desired_num_frames).to(dtype=torch.float32) + 0.5
        )

        if temporal_jitter:
            jitter_size = seg_size * 0.5
            # Generate random shifts for all frames at once
            seq += (
                torch.rand(len(seq), dtype=torch.float32) * (jitter_size * 2)
                - jitter_size
            )
            # Clip values to valid range
            seq = torch.clamp(seq, 0, total_duration)

        return seq

    def video_to_frames(self, video: VideoMedia) -> list[Media]:
        """Convert a video to a list of video frame and text according to the settings."""
        video_duration = video.metadata["video_duration"]
        video_num_frames = video.metadata["video_num_frames"]
        start_time = 0
        if video.start_time is not None:
            if video.end_time is not None:
                video_duration = video.end_time - video.start_time
            else:
                video_duration = video_duration - video.start_time
            start_time = video.start_time
        elif video.end_time is not None:
            video_duration = video.end_time

        if video_num_frames < self.args.video_min_num_frames:
            # some videos are too short or low-fps, just use the whole video, like sthv2, smit, llava-hound (2fps)
            sample_num_frames = video_num_frames
        else:
            default_sample_num_frames = int(
                self.args.video_default_fps * video_duration
            )
            sample_num_frames = min(
                max(default_sample_num_frames, self.args.video_min_num_frames),
                self.args.video_max_num_frames,
            )

        frame_timestamps = self.get_seq_frames_v3(
            video_duration, sample_num_frames, self.args.video_frame_temporal_jitter
        )
        frame_timestamps += start_time

        return ["This is a video:\n"] + [
            media
            for i, timestamp in enumerate(frame_timestamps)
            for media in (
                f"Frame {i + 1} sampled at {timestamp:.2f} seconds: ",
                # TODO: Orginal eagle repro
                # TextMedia(value=f"Frame {i + 1} sampled at {timestamp:.2f} seconds: <image-{i}>"),
                VideoFrameMedia(
                    value=video.value,
                    timestamp=float(timestamp),
                    metadata={
                        "video_width": video.video_width,
                        "video_height": video.video_height,
                    },
                ),
                "\n",
            )
        ]

    @stateless(restore_seeds=True)
    def preencode_sample(self, sample: ConversationSample) -> PreEncodedTaskSample:
        """Encode sample."""
        # In-place convert VideoMedia to VideoFrameMedia (and text)
        for message in sample.conversation:
            idx = 0
            while idx < len(message.fragments):
                if isinstance(message.fragments[idx], VideoMedia):
                    frames = self.video_to_frames(message.fragments[idx])
                    message.fragments[idx : idx + 1] = frames
                    idx += len(frames)
                else:
                    idx += 1

        legacy_conversation: list[LegacyConversation] = [
            {"role": "system", "content": "Answer the questions."},
            # TODO: Orginal eagle repro
            # {"role": "system", "content": "You are an AI assistant whose name is Eagle-Next."},
        ]

        image_media = []
        image_sizes = []
        image_transforms = []

        # Format the conversation as a list of "user" / "assistant" turns.
        for message in sample.conversation:
            assert message.sender in ["user", "assistant"], (
                f"unexpected sender {message.sender} in {sample.conversation}"
            )

            content = ""
            for fragment in message.fragments:
                if isinstance(fragment, str):
                    content += fragment
                elif isinstance(fragment, ImageMedia):
                    content += IMAGE_TOKEN
                    image_media.append(fragment)
                    image_sizes.append(
                        (fragment.width, fragment.height)
                    )
                    image_transforms.append(self.transform_image)
                elif isinstance(fragment, VideoFrameMedia):
                    content += IMAGE_TOKEN
                    image_media.append(fragment)
                    image_sizes.append(
                        (
                            fragment.video_width,
                            fragment.video_height,
                        )
                    )
                    image_transforms.append(self.transform_video_frame)
                elif isinstance(fragment, VideoMedia):
                    raise ValueError(
                        "VideoMedia should have been converted to VideoFrameMedia."
                    )
                elif isinstance(fragment, AudioMedia):
                    raise ValueError("Audio not supported yet.")

            legacy_conversation.append({"role": message.sender, "content": content})

        input_ids, target = self.tokenizer.tokenize_conversation(
            legacy_conversation, True, False
        )
        input_ids = torch.as_tensor(input_ids)
        target = torch.as_tensor(target)
        assert len(target) > 0 and (target != IGNORE_INDEX).any(), (
            f"target is empty: {target}, DETOKENIZED:\n\n{self.tokenizer.detokenize(input_ids)}\n\nCONVERSATION:\n\n"
            + "".join([f"{m.sender}: {m.fragments}\n" for m in sample.conversation])
        )

        max_image_token_allowed = self.args.decoder_seq_length - len(input_ids) - 4

        params = self.tile_degradation_map.compute_tilings(
            image_sizes, image_transforms, max_image_token_allowed
        )

        preprocessed_image_media = [
            PreprocessedImageMedia(media=media, params=params)
            for media, params in zip(image_media, image_params)
        ]

        input_ids, target = self._truncate_to_decoder_seq_len(input_ids, target, num_tiles)

        # We need to ensure that there are at least some trainable tokens in the sample.
        assert self._target_has_trainable_tokens(input_ids, num_tiles, target), (
            f"Sample has no trainable tokens: {self.tokenizer.detokenize(input_ids)}"
        )

        total_len, total_len_padded, input_ids, target = self._pad_for_context_parallel(
            input_ids, target, num_tiles
        )

        return PreEncodedTaskSample.derive_from(
            sample,
            tokens=input_ids,
            labels=target,
            images=preprocessed_image_media,
            num_tiles=num_tiles,
            total_len=total_len,
            total_len_padded=total_len_padded,
        )

    @stateless(restore_seeds=True)
    def postencode_sample(self, sample: PreEncodedTaskSample) -> PackedTaskSample:
        self._load_media(sample)

        # Transform the images
        image_tiles = []
        for media in sample.images:
            if isinstance(media.media, VideoFrameMedia):
                image_tiles.extend(
                    self.transform_video_frame.apply_params(
                        media.media.value, media.tiling
                    )
                )
            elif isinstance(media.media, ImageMedia):
                image_tiles.extend(
                    self.transform_image.apply_params(media.media.value, media.tiling)
                )
            else:
                raise ValueError(f"Unexpected media type: {type(media.media)}")

        # Make this a packed sample (if used without packing, it will be the same next code)
        return PackedTaskSample.derive_from(
            sample,
            __key__=[sample.__key__],
            tokens=sample.tokens,
            labels=sample.labels,
            imgs=image_tiles,
            num_tiles=sample.num_tiles,
            max_length=sample.total_len_padded,
            cu_lengths=torch.tensor([0, sample.total_len], dtype=torch.int32),
            cu_lengths_padded=torch.tensor(
                [0, sample.total_len_padded], dtype=torch.int32
            ),
            samples_seen=torch.tensor(1, dtype=torch.int32),
        )


    @stateless(restore_seeds=True)
    def select_samples_to_pack(
        self, samples: List[PreEncodedTaskSample]
    ) -> List[List[PreEncodedTaskSample]]:
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        lengths = [sample.total_len_padded for sample in samples]

        packed_samples = self.packing_knapsack_algorithm(
            lengths, samples, self.packing_seq_length
        )
        # Shuffle the packed samples
        random.shuffle(packed_samples)

        # TODO: Save iterated samples
        # with open(f"tmpdata/packed_samples_{os.getpid()}.json", "a") as f:
        #     f.write(json.dumps(
        #         {
        #             "lengths": [int(sample.total_len_padded) for sample in samples],
        #             "samples": [
        #                 {
        #                     "key": sample.__key__,
        #                     "images": [img.media.value.fname for img in sample.images],
        #                     "image_sizes": [(img.media.metadata.get("width", img.media.metadata.get("video_width")), img.media.metadata.get("height", img.media.metadata.get("video_height"))) for img in sample.images],
        #                     "length": int(sample.total_len_padded),
        #                     "tokens": [int(token) for token in sample.tokens],
        #                     "detokens": [self.tokenizer.detokenize([token]) for token in sample.tokens],
        #                     "tiling": [img.tiling for img in sample.images],
        #                     "num_tiles": sample.num_tiles,
        #                     "name": sample.__subflavors__["name"],
        #                 } for sample in samples
        #             ],
        #         }
        #     ) + "\n")

        print(
            f"[pid={os.getpid()}] Computed packing from {len(samples)} samples to {len(packed_samples)} packed samples (avg {len(samples) / len(packed_samples)} samples per packed sample)"
        )
        print(
            f"[pid={os.getpid()}] Computed packing from {sum(lengths)} tokens (avg {sum(lengths) / len(samples)} tokens per sample, avg {sum(lengths) / len(packed_samples)} tokens per packed sample)"
        )
        print(
            f"[pid={os.getpid()}] Packing efficiency: {sum(lengths)} / {len(packed_samples) * self.packing_seq_length} = {sum(lengths) / (len(packed_samples) * self.packing_seq_length) * 100:.2f}%"
        )
        assert sum(len(s) for s in packed_samples) == len(samples), (
            "knapsack discarded some samples"
        )

        return packed_samples

    @stateless
    def pack_selected_samples(
        self, samples: List[PackedTaskSample]
    ) -> List[PackedTaskSample]:
        """
        Function to pack a list of ImageTaskSamplePacked into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of ImageTaskSamplePacked instances to pack into one sample.

        Returns:
            ImageTaskSamplePacked instance.
        """
        cu_lengths = [0]
        cu_lengths_padded = [0]
        # Process each sample and build lists that we will concatenate to create the packed sample.
        for sample in samples:
            sample_len = sample.cu_lengths[-1]
            sample_len_padded = sample.cu_lengths_padded[-1]

            cu_lengths.append(cu_lengths_padded[-1] + sample_len)
            cu_lengths_padded.append(cu_lengths_padded[-1] + sample_len_padded)

        assert cu_lengths_padded[-1] <= self.packing_seq_length, (
            f"Packed sample exceeds the maximum sequence length of {self.packing_seq_length}: {samples}"
        )
        assert all(isinstance(sample.imgs, list) for sample in samples), (
            "All images must be tensors"
        )
        assert all(
            isinstance(img, torch.Tensor) for sample in samples for img in sample.imgs
        ), "All images must be tensors"

        return PackedTaskSample(
            __key__=[k for s in samples for k in s.__key__],
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavors__={
                idx: sample.__subflavors__ for idx, sample in enumerate(samples)
            },
            tokens=torch.cat([sample.tokens for sample in samples], dim=0),
            labels=torch.cat([sample.labels for sample in samples], dim=0),
            imgs=[img for sample in samples for img in sample.imgs],
            cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
            cu_lengths_padded=torch.tensor(cu_lengths_padded, dtype=torch.int32),
            max_length=max(sample.max_length for sample in samples),
            num_tiles=[n for s in samples for n in s.num_tiles],
            samples_seen=sum(s.samples_seen for s in samples),
        )

    def batch(self, samples: List[PackedTaskSample]) -> BatchedPackedTaskSample:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            assert self.args.vision_model_type in ("radio", "radio-g", "cradio-g"), "Dynamic resolution only works with radio right now"
            imgs = torch.stack(imgs)

        # Pad image packed seq length to be % 16 if using fp8 and dynamic resolution
        has_fp8 = self.args.fp8 is not None
        has_pad_img = torch.tensor(False)
        if has_fp8 and self.args.dynamic_resolution:
            img_seq_len = 0
            for img in imgs:
                img_seq_len += (img.shape[1] // self.args.patch_dim) * (img.shape[2] // self.args.patch_dim)
            padding_needed = get_padding(img_seq_len, self.args.context_parallel_size, self.args.tensor_model_parallel_size, self.args.sequence_parallel, fp8_enabled=has_fp8)
            if padding_needed > 0:
                pad_img = torch.zeros([3, self.args.patch_dim, padding_needed * self.args.patch_dim])
                imgs.append(pad_img)
                has_pad_img = torch.tensor(True)

        imgs, imgs_sizes, vision_cu_lengths, vision_max_lengths = process_images(
            imgs, self.args.patch_dim, self.args.dynamic_resolution, batch_mode=True
        )

        # Set default values if no vision metadata was returned (static resolution case)
        if vision_cu_lengths is None:
            vision_cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        if vision_max_lengths is None:
            vision_max_lengths = torch.tensor([[0]], dtype=torch.int32)

        # If the user hasn't defined a target dataloader sequence length, then use the max along the sample lengths.
        max_seq_len = self.dataloader_seq_length
        if not max_seq_len:
            max_seq_len = max(len(s.tokens) for s in samples)

        tokens = torch.full(
            (len(samples), max_seq_len), self.tokenizer.pad, dtype=torch.int64
        )
        # +1 to accommodate shift to left by one later.
        labels = torch.full(
            (len(samples), max_seq_len + 1), self.tokenizer.pad, dtype=torch.int64
        )

        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.tokens))
            target_len = min(max_seq_len + 1, len(s.labels))

            tokens[i, :text_len] = s.tokens[:text_len]
            labels[i, :target_len] = s.labels[:target_len]

        num_tiles = torch.tensor(
            [n for s in samples for n in s.num_tiles], dtype=torch.int32
        )
        if len(num_tiles) == 0:
            num_tiles = torch.tensor([[0]], dtype=torch.int32)

        cu_lengths = torch.stack([s.cu_lengths for s in samples])
        cu_lengths_padded = torch.stack([s.cu_lengths_padded for s in samples])
        max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)

        if self.dataloader_seq_length is not None:
            cu_lengths[:][-1] = self.dataloader_seq_length
            cu_lengths_padded[:][-1] = self.dataloader_seq_length
            new_max_length = cu_lengths_padded[:][-1] - cu_lengths[:][-2]
            max_lengths = torch.max(max_lengths, new_max_length)

        return BatchedPackedTaskSample(
            __key__=[s.__key__ for s in samples],
            __restore_key__=[s.__restore_key__ for s in samples],
            __subflavors__=samples[0].__subflavors__,
            tokens=tokens,
            labels=labels,
            imgs=imgs,
            num_tiles=num_tiles,
            cu_lengths=cu_lengths,
            cu_lengths_padded=cu_lengths_padded,
            max_lengths=max_lengths,
            imgs_sizes=imgs_sizes,
            vision_cu_lengths=vision_cu_lengths,
            vision_max_lengths=vision_max_lengths,
            has_pad_img=has_pad_img,
            samples_seen=sum(s.samples_seen for s in samples),
        )

    def encode_batch(self, batch: BatchedPackedTaskSample) -> dict:
        return dataclasses.asdict(batch)

    def _load_media(self, sample: PreEncodedTaskSample) -> None:
        """Loads all lazy media in the sample."""
        if len(sample.images) > 1:
            medias: dict[
                Lazy[AVDecoder] | Lazy[Image.Image], list[PreprocessedImageMedia]
            ] = defaultdict(list)
            # Group by video and frame index
            for media in sample.images:
                # video is a Lazy[AVDecoder] (it's hashable, all pointing to the same file are the same object)
                medias[media.media.value].append(media)

            for media, frames in medias.items():
                media_value = media.get()
                if isinstance(media_value, AVDecoder):
                    media_value.suppress_warnings = True
                    frame_clips = media_value.get_clips(
                        video_clip_ranges=[
                            (frame.media.timestamp, frame.media.timestamp)
                            for frame in frames
                        ],
                        video_unit="seconds",
                    )
                    # print(f"frame_clips {video.fname}: {len(frame_clips.video_clips)}: {[img.shape for img in frame_clips.video_clips]}")
                    images = [
                        tensor_to_pil(img[0][0]) for img in frame_clips.video_clips
                    ]

                    if len(images) < len(frames):
                        last_image = images[-1]
                        images.extend([last_image] * (len(frames) - len(images)))

                    for frame, image in zip(frames, images):
                        frame.media.value = image
                elif isinstance(media_value, Image.Image):
                    for frame in frames:
                        frame.media.value = media_value
                else:
                    raise ValueError(f"Unexpected media type: {type(media_value)}")
        else:
            for media in sample.images:
                media.media.value = media.media.value.get()

    def _target_has_trainable_tokens(
        self, input_ids: torch.Tensor, num_tiles: list[int], target: torch.Tensor
    ) -> bool:
        # Compute the loss mask based on extending the image tags with the proper
        # number of image tokens, extracting the first self.args.decoder_seq_length tokens, and
        # ensuring that some of these tokens have a loss mask > 0.
        # Note that this is a bit hacky because we reproduce here parts of the logics which are in
        # the model itself. Ideally, the data sampler would return the already processed inputs
        # and targets to avoid this duplication.
        expanded_target = target.clone()
        expanded_target[input_ids == self.img_token_id] = self.img_token_id
        expanded_target = self._replace_value_with_repetition(
            expanded_target,
            self.img_token_id,
            self.num_image_embeddings_per_tile * torch.tensor(num_tiles),
            IGNORE_INDEX,
        )
        loss_mask = torch.ones(expanded_target.size(), dtype=torch.float)
        loss_mask[expanded_target == self.tokenizer.pad] = 0.0  # mask paddings
        loss_mask[expanded_target == IGNORE_INDEX] = 0.0  # mask prompts
        loss_mask = torch.cat((loss_mask[1:], torch.zeros((1,))))
        loss_mask = loss_mask[: self.args.decoder_seq_length]
        return torch.sum(loss_mask) > 0

    def _replace_value_with_repetition(
        self,
        arr: torch.Tensor,
        token_to_replace: int,
        num_repetition: torch.Tensor,
        new_token: int,
    ) -> torch.Tensor:
        """
        Replace every occurrence of value V in the input array with R repetitions of W.

        Args:
            arr (Array): Input array to be modified
            token_to_replace: token to be replaced
            new_token: new token
            num_repetition (Array): number of repetition of new token.

        Returns:
            Array: New array with token_to_replace replaced by num_repetition repetitions of
             new_token
        """
        error_msg = (
            "The number of image tokens must match the length of the tile tensor."
        )
        assert torch.sum(arr == token_to_replace) == len(num_repetition), error_msg
        result = []
        idx = 0
        for item in arr:
            if item == token_to_replace:
                # If the current item matches token_to_replace, add R copies of W
                result.extend([new_token] * num_repetition[idx])
                idx += 1
            else:
                # Otherwise, keep the original item
                result.append(item)

        return torch.tensor(result)

    def _pad_for_context_parallel(
        self, input_ids: torch.Tensor, target: torch.Tensor, num_tiles: list[int]
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        total_len = self._get_total_seq_length(input_ids, num_tiles)
        total_len_padded = total_len
        if getattr(self.args, 'context_parallel_size', 1) > 1:
            padding_needed = get_padding(
                total_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
            )
            padding1 = torch.ones(padding_needed) * self.tokenizer.pad
            padding2 = torch.ones(padding_needed) * IGNORE_INDEX
            input_ids = torch.cat([input_ids, padding1])
            target = torch.cat([target, padding2])

            # A100 doesn't support cu_seqlens != cu_seqlens_padded. hack=True forces them to be same.
            # hack=False only supported on H100.
            hack = True
            if hack:
                # Pad everything
                total_len = total_len + padding_needed
                total_len_padded = total_len
            else:
                # Pad only padding.
                total_len_padded = total_len + padding_needed
        return total_len, total_len_padded, input_ids, target

    def _get_total_seq_length(self, input_ids: torch.Tensor, num_tiles):
        """Calculate expected sequence length given text tokens length and number of tiles."""
        self.get_num_image_embeddings(num_tiles)
        total_num_images = len(num_tiles)
        total_num_tiles = sum(num_tiles)
        total_len = (
            len(input_ids)
            + total_num_tiles * self.num_image_embeddings_per_tile
            - total_num_images
        )
        return total_len

    def _truncate_to_decoder_seq_len(
        self, input_ids: torch.Tensor, target: torch.Tensor, num_tiles: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate tokens and labels if they exceed sequence length."""
        total_num_images = len(num_tiles)
        total_num_tiles = sum(num_tiles)
        total_img_embeddings_len = total_num_tiles * self.num_image_embeddings_per_tile
        max_text_tokens = (
            self.packing_seq_length - 12 - total_img_embeddings_len + total_num_images
        )

        input_ids = input_ids[:max_text_tokens]
        target = target[:max_text_tokens]

        # If truncate causes all labels to be ignored, then skip the sample
        if len(target) == 0 or (target == IGNORE_INDEX).all():
            raise ValueError(
                f"all targets will be ignored after truncation: {input_ids} {target}"
            )

        return input_ids, target

tensor_to_pil = ToPILImage()
