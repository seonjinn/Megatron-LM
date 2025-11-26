# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Literal, TypedDict, Union, Tuple
import numpy as np

from .cookers.audio_conversation import cook_audio_conversation
from .cookers.granary import (
    cook_granary_english_webdataset,
    cook_granary_english_jsonl,
)
from .cookers.omcat_legacy_audio_conversation import cook_omcat_legacy_conversation_monolithic
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN, SOUND_TOKEN
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.energon import (
    Batch,
    Cooker,
    DefaultTaskEncoder,
    Lazy,
    Sample,
    SampleDecoder,
    stateless,
    __version__ as energon_version,
)
from megatron.energon.edataclass import edataclass
from megatron.energon.av import AVDecoder
from megatron.training import get_args, get_tokenizer

from .audio_processing import AudioParams, AudioTransformStrategy, AudioTransformParakeetStrategy
from .conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    Media,
    VideoFrameMedia,
    VideoMedia,
)
from .cookers.conversation import (
    cook_conversation,
    cook_general_conversations_webdataset,
    cook_general_conversations_jsonl,
)
from .cookers.audio_conversation import cook_audio_conversation
from .cookers.eagle import cook_eagle
from .image_processing import (
    ImageTilingParams,
    create_image_tiling_strategy,
)
from .knapsacks import (
    balanced_greedy_knapsack,
    bucketing_greedy_knapsack,
    greedy_knapsack,
)


AUDIO_MAX_DURATION_SECONDS = 900
AUDIO_MIN_DURATION_SECONDS = 0.1


def _clean_think(match: re.Match) -> str:
    """Helper to strip whitespace inside <think> tags during preprocessing."""
    clean_content = match.group(1).strip()
    if clean_content:
        clean_content = "\n" + clean_content + "\n"
    return f"<think>{clean_content}</think>"


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
class PreEncodedTaskSample(Sample):
    tokens: torch.Tensor
    labels: torch.Tensor
    images: list[ImageTilingParams]
    audio: list[AudioParams]
    total_len: int
    total_len_padded: int
    num_frames: list[int]


@edataclass
class PackedTaskSample(Sample):
    """Dataclass to store a single packed sample (not a batch).

    P = Number of sub-samples in the packed sample
    seq_len = Total sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    # Sample name
    __key__: list[str]
    # Input tokens packed into a single tensor (seq_len,)
    tokens: torch.Tensor
    # Target tokens packed into a single tensor (seq_len,)
    labels: torch.Tensor
    # Maximum length across sub-samples.
    max_length: int
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths: list[int]
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths_padded: list[int]

    # Input images
    imgs: list[torch.Tensor]
    # Number of tiles for each image of each sample (num_imgs)
    num_tiles: list[int]

    # Number of frames used per VideoMedia / ImageMedia (1 frame for ImageMedia)
    num_frames: list[int]

    # Number of samples in the packed sample
    samples_seen: int

    # Sound
    sound_clips: list[torch.Tensor]
    sound_length: list[int]
    sound_timestamps: list[tuple[int, int]]
    num_sound_clips: list[int]


# Typing for the resulting batch data after encode_batch()
@edataclass
class BatchedPackedTaskSample(Batch):
    """Dataclass to store a batch of packed samples.

    N = Batch size
    P = Number of samples in the packed sample
    seq_len = Maximum sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    # Input tokens packed and padded (N, seq_len)
    tokens: torch.Tensor
    # Target tokens packed and padded (N, seq_len)
    labels: torch.Tensor
    # Maximum length across sub-samples (N,)
    max_lengths: list[int]
    # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths: list[list[int]]
    # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths_padded: list[list[int]]

    # All image tiles stacked into a single tensor (num_tiles, C, H, W)
    imgs: torch.Tensor
    # Number of tiles per image (N, num_imgs)
    num_tiles: list[list[int]]
    # Size of each image tile
    imgs_sizes: list[tuple[int, int]]
    # Maximum length across sub-samples. (N,)
    vision_max_lengths: list[int]
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (N, num_imgs)
    vision_cu_lengths: list[list[int]]

    # Number of samples in the packed batch
    samples_seen: int

    # Whether the batch has a padded image
    has_pad_img: bool

    # "Batched" version of number of frames used per VideoMedia / ImageMedia (1 frame for ImageMedia)
    num_frames: list[list[int]]

    # Sound
    sound_clips: torch.Tensor
    sound_length: torch.Tensor
    sound_timestamps: torch.Tensor
    num_sound_clips: torch.Tensor


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
        Cooker(cook_audio_conversation, has_subflavors={"cook": "audio_conversation"}),
        Cooker(cook_granary_english_webdataset, has_subflavors={"cook": "granary_english_webdataset"}),
        Cooker(cook_granary_english_jsonl, has_subflavors={"cook": "granary_english_jsonl"}),
        Cooker(cook_omcat_legacy_conversation_monolithic, has_subflavors={"cook": "omcat_legacy_conversation_monolithic"}),
        Cooker(cook_general_conversations_webdataset, has_subflavors={"cook": "general_conversations_webdataset"}),
        Cooker(cook_general_conversations_jsonl, has_subflavors={"cook": "general_conversations_jsonl"}),
    ]

    def __init__(self, is_val: bool = False, tiling_augment_prob: float = 0.4):
        super().__init__()
        self.is_val = is_val
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

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        # This map is used to reduce the number of tiles used per image if the number of tokens is
        # larger than the decoder_seq_length.
        self.num_tiles_degradation_map = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1, 1: 1}

        self.tiling_augment_prob = tiling_augment_prob

        # Create the image tiling strategy using the refactored function
        self.image_tiling_strategy = create_image_tiling_strategy(self.args)

        if self.args.packing_knapsack_algorithm == "greedy_knapsack":
            self.packing_knapsack_algorithm = greedy_knapsack
        elif self.args.packing_knapsack_algorithm == "balanced_greedy_knapsack":
            self.packing_knapsack_algorithm = balanced_greedy_knapsack
        elif self.args.packing_knapsack_algorithm == "bucketing_greedy_knapsack":
            self.packing_knapsack_algorithm = bucketing_greedy_knapsack
        else:
            raise ValueError(
                f"Unknown knapsack algorithm: {self.args.packing_knapsack_algorithm}")

        if getattr(self.args, "sound_model_type", None) is not None:
            self.sound_token_id = self.tokenizer.convert_tokens_to_ids(SOUND_TOKEN)
            if 'parakeet' in self.args.sound_model_type.lower():
                self.transform_audio = AudioTransformParakeetStrategy(
                    sound_model_type=self.args.sound_model_type, 
                    target_freq=self.args.sound_target_rate, 
                    embedding_size=self.args.sound_embedding_size, 
                    clip_duration=self.args.sound_clip_duration, 
                    min_duration=self.args.sound_min_duration,
                    pad_to_clip_duration=self.args.sound_pad_to_clip_duration
                )
            else:
                self.transform_audio = AudioTransformStrategy(self.args.sound_model_type, self.args.sound_target_rate, self.args.sound_embedding_size, self.args.sound_clip_duration)
        else:
            self.sound_token_id = None
            self.transform_audio = None
        print(f"{type(self).__name__} initialized. Energon Version: {energon_version}")

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

    def video_to_frames(self, video: VideoMedia) -> Tuple[list[Media], int]:
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
                # f"Frame {i + 1} sampled at {timestamp:.2f} seconds: <image-{i}>",
                VideoFrameMedia(
                    value=video.value,
                    timestamp=float(timestamp),
                    metadata={
                        "video_width": video.video_width,
                        "video_height": video.video_height
                    },
                ),
                "\n",
            )
        ], len(frame_timestamps)

    @stateless(restore_seeds=True)
    def preencode_sample(self, sample: ConversationSample) -> PreEncodedTaskSample:
        """Encode sample."""

        # In-place convert VideoMedia to VideoFrameMedia (and text)
        # Some really large video files cause decoding to take a long time potentially leading to issues.
        allow_large_videos = getattr(self.args, "allow_large_videos", False)
        data_augment = sample.__subflavors__.get("data_augment", False) and not self.is_val
        tiling_augment_prob = sample.__subflavors__.get("tiling_augment_prob", self.tiling_augment_prob)
        aggregated_num_frames = []

        # We tentatively extract the first message if it's a system prompt and use this rather than
        # the default. After this, we expect no system prompt in the conversation.

        has_system_message = sample.conversation[0].sender == "system"
        system_prompt = ""
        if has_system_message:
            system_prompt = sample.conversation[0].fragments[0]
            sample.conversation = sample.conversation[1:]

        if system_prompt == "" and self.args.tokenizer_prompt_format == "nemotron6-moe":
            system_prompt = "Answer the questions."

        legacy_conversation: list[LegacyConversation] = [
            {"role": "system", "content": system_prompt}
        ]

        for message in sample.conversation:
            idx = 0
            while idx < len(message.fragments):
                fragment = message.fragments[idx]

                if isinstance(fragment, VideoMedia):
                    if not allow_large_videos and fragment.clip_duration > 60*10:
                        raise ValueError(f"Video is too large: {fragment.value}")

                    frames, num_frames = self.video_to_frames(fragment)
                    message.fragments[idx : idx + 1] = frames
                    idx += len(frames)
                    aggregated_num_frames.append(num_frames)
                elif isinstance(fragment, AudioMedia):
                    assert fragment.audio_duration >= AUDIO_MIN_DURATION_SECONDS, f"Audio duration is too short: {fragment.audio_duration}sec < {AUDIO_MIN_DURATION_SECONDS}s"
                    assert fragment.audio_duration <= AUDIO_MAX_DURATION_SECONDS, f"Audio duration is too long: {fragment.audio_duration}sec > {AUDIO_MAX_DURATION_SECONDS}s"
                    idx += 1
                else:
                    if isinstance(fragment, (ImageMedia, VideoFrameMedia)):
                        # Image or a single frame
                        aggregated_num_frames.append(1)
                    elif isinstance(fragment, str):
                        # Text fragment
                        pass
                    elif isinstance(fragment, bytes):
                        raise ValueError(f"Could not convert bytes to known media type: {fragment[:100]!r}")
                    else:
                        raise ValueError(f"Unexpected media type: {type(fragment)}. Fragment: {fragment}")
                    idx += 1

        image_media: list[ImageMedia | VideoFrameMedia] = []
        audio_media_params: list[AudioMedia] = []

        # Format the conversation as a list of "user" / "assistant" turns.
        for message in sample.conversation:
            if not self.args.relax_sender_check:
                assert message.sender in ["user", "assistant"], (
                    f"unexpected sender {message.sender} in {sample.conversation}"
                )

            content = ""
            for fragment in message.fragments:
                if isinstance(fragment, str):
                    content += fragment
                    assert IMAGE_TOKEN not in fragment, f"{IMAGE_TOKEN!r} in sample with key: {sample.__key__} and subflavors: {sample.__subflavors__}"
                elif isinstance(fragment, ImageMedia):
                    content += IMAGE_TOKEN
                    image_media.append(fragment)
                elif isinstance(fragment, VideoFrameMedia):
                    content += IMAGE_TOKEN
                    image_media.append(fragment)
                elif isinstance(fragment, VideoMedia):
                    raise ValueError(
                        "VideoMedia should have been converted to VideoFrameMedia."
                    )
                elif isinstance(fragment, AudioMedia):
                    audio_params = self.transform_audio.compute_params([fragment])

                    content += "<so_start>" + SOUND_TOKEN * audio_params[0].num_embeddings + "<so_end>"
                    audio_media_params.append(audio_params[0])
            
            if self.args.only_keep_samples_with_img and len(image_media) == 0:
                raise ValueError(f"Sample has no image: {sample.__key__}")

            prompt_format = self.args.tokenizer_prompt_format

            if prompt_format in ("nemotron-h-5p5-reasoning", "nemotron6-moe") and message.sender == "assistant" and not self.args.relax_thinking_trace_check:
                think_start_count = content.count("<think>")
                think_end_count = content.count("</think>")
                if think_start_count == 0 and think_end_count == 0:
                    # Add think tags in non-reasoning mode, if missing
                    think = "<think></think>" if prompt_format == "nemotron6-moe" else "<think></think>\n\n"
                    content = think + content.strip()
                else:
                    # There should be exactly one of each, otherwise it's invalid
                    assert think_start_count == 1 and think_end_count == 1, (
                        f"Found sample with {think_start_count} <think> tags and {think_end_count} </think> tags in sample with "
                        f"key: {sample.__key__} and subflavors: {sample.__subflavors__}")

                    # </think> should come after <think>, otherwise it's invalid
                    start_idx = content.find("<think>")
                    end_idx = content.find("</think>")
                    assert start_idx < end_idx, (
                        f"Found sample with </think> tags before </think> tags in sample with "
                        f"key: {sample.__key__} and subflavors: {sample.__subflavors__}")

                    # Clean up content inside <think> tags and strip surrounding whitespace
                    content = re.sub(r"<think>(.*?)</think>", _clean_think, content, re.DOTALL)

                    # Ensure </think> is always followed by N newlines and no other whitespace
                    replacement = "</think>\n" if prompt_format == "nemotron6-moe" else "</think>\n\n"
                    content = re.sub(r'</think>\s*', replacement, content)

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
        image_media_params = self.image_tiling_strategy.compute_params(
            image_media,
            max_image_token_allowed,
            data_augment=data_augment,
            tiling_augment_prob=tiling_augment_prob
        )

        # We need to compare the number of sound tokens before and after truncation
        # If the numbers are different, raise an error to skip this sample
        if self.sound_token_id is not None:
            num_sound_tokens_before_truncation = (input_ids == self.sound_token_id).sum()
        else:
            num_sound_tokens_before_truncation = 0

        input_ids, target = self._truncate_to_decoder_seq_len(
            input_ids, target, image_media_params, audio_media_params
        )

        if self.sound_token_id is not None:
            num_sound_tokens_after_truncation = (input_ids == self.sound_token_id).sum()
        else:
            num_sound_tokens_after_truncation = 0

        assert num_sound_tokens_before_truncation == num_sound_tokens_after_truncation, (
            f"Number of sound tokens changed after truncation: "
            f"{num_sound_tokens_before_truncation} -> {num_sound_tokens_after_truncation}"
        )

        # We need to ensure that there are at least some trainable tokens in the sample.
        has_trainable_tokens = self._target_has_trainable_tokens(
            input_ids, target, image_media_params, audio_media_params
        )

        assert has_trainable_tokens, f"Sample has no trainable tokens: {self.tokenizer.detokenize(input_ids)}"

        total_len, total_len_padded, input_ids, target = self._pad_for_context_parallel_and_fp8(
            input_ids, target, image_media_params, audio_media_params
        )

        return PreEncodedTaskSample.derive_from(
            sample,
            tokens=input_ids,
            labels=target,
            images=image_media_params,
            audio=audio_media_params,
            total_len=total_len,
            total_len_padded=total_len_padded,
            num_frames=aggregated_num_frames,
        )

    @stateless(restore_seeds=True)
    def postencode_sample(self, sample: PreEncodedTaskSample) -> PackedTaskSample:
        self._load_media(sample)

        data_augment = sample.__subflavors__.get("data_augment", False) and not self.is_val

        # Transform the images
        image_tiles = []
        for media_idx, media in enumerate(sample.images):
            # Debug: Save images if DEBUG environment variable is set to 1
            if os.environ.get("DEBUG_DATALOADER", "0") == "1":
                self._debug_save_image(media, media_idx, sample.__key__, data_augment)
            
            image_tiles.extend(self.image_tiling_strategy.apply_params(media, data_augment=data_augment))

        sound_clips = []
        sound_length = []
        sound_timestamp = []
        num_sound_clips = []
        for media in sample.audio:
            audio, audio_length = self.transform_audio.apply_params(media)
            sound_clips.append(audio)
            sound_length.append(audio_length)
            sound_timestamp.append(media.timestamps)
            num_sound_clips.append(media.num_clips)

        # Make this a packed sample (if used without packing, it will be the same next code)
        return PackedTaskSample.derive_from(
            sample,
            __key__=[sample.__key__],
            tokens=sample.tokens,
            labels=sample.labels,
            imgs=image_tiles,
            num_tiles=[media.num_tiles for media in sample.images],
            num_frames=sample.num_frames,
            max_length=sample.total_len_padded,
            cu_lengths=torch.tensor([0, sample.total_len], dtype=torch.int32),
            cu_lengths_padded=torch.tensor(
                [0, sample.total_len_padded], dtype=torch.int32
            ),
            sound_clips=sound_clips,
            sound_length=sound_length,
            sound_timestamps=sound_timestamp,
            num_sound_clips=num_sound_clips,
            samples_seen=torch.tensor(1, dtype=torch.int32),
        )

    def _debug_save_image(self, media, media_idx, sample_key, data_augment):
        """Save debug images with original and transformed sizes."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from datetime import datetime
        
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(os.getcwd(), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Get original image and size
        original_image = media.media.value
        
        if isinstance(media.media, ImageMedia):
            orig_width, orig_height = media.media.width, media.media.height
        elif isinstance(media.media, VideoFrameMedia):
            orig_width, orig_height = media.media.video_width, media.media.video_height
        else:
            return  # Skip if not a supported media type
        
        # Apply the transformation to get the processed tiles
        transformed_tiles = self.image_tiling_strategy.apply_params(media, data_augment=data_augment)
        
        # Get the normalization stats for denormalization
        from .image_processing import pixel_statistics
        pixel_mean, pixel_std = pixel_statistics.get(
            self.args.vision_model_type, 
            ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        )
        
        # Convert lists to tensors for denormalization
        mean = torch.tensor(pixel_mean).view(3, 1, 1)
        std = torch.tensor(pixel_std).view(3, 1, 1)
        
        # Create a figure with subplots
        num_tiles = len(transformed_tiles)
        fig, axes = plt.subplots(1, num_tiles + 1, figsize=(5 * (num_tiles + 1), 5))
        if num_tiles == 0:
            axes = [axes]
        
        # Plot original image
        ax = axes[0] if num_tiles > 0 else axes
        ax.imshow(original_image)
        ax.set_title(f"Original Image\nSize: {orig_width}x{orig_height}", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Plot transformed tiles
        for tile_idx, tile_tensor in enumerate(transformed_tiles):
            ax = axes[tile_idx + 1]
            
            # Denormalize the tensor: img = img * std + mean
            denormalized = tile_tensor * std + mean
            
            # Clamp to [0, 1] range
            denormalized = torch.clamp(denormalized, 0, 1)
            
            # Convert to numpy and transpose from CxHxW to HxWxC
            tile_image = denormalized.permute(1, 2, 0).cpu().numpy()
            
            # Get the new size
            new_height, new_width = tile_image.shape[:2]
            
            ax.imshow(tile_image)
            ax.set_title(
                f"Tile {tile_idx + 1}/{num_tiles}\n"
                f"Size: {tile_tensor.shape[2]}x{tile_tensor.shape[1]}\n"
                f"Original: {orig_width}x{orig_height}",
                fontsize=9,
                fontweight='bold'
            )
            ax.axis('off')
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_key = str(sample_key).replace("/", "_").replace("\\", "_")[:50]
        filename = f"{safe_key}_media{media_idx}_{timestamp}.png"
        filepath = os.path.join(debug_dir, filename)

        print(f"[DEBUG] Saved debug image to: {filepath}")
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

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
        # Print the 5% - 50% - 95% percentiles of the lengths
        print(f"[pid={os.getpid()}] 5% percentile of lengths: {np.percentile(lengths, 5)}")
        print(f"[pid={os.getpid()}] 50% percentile of lengths: {np.percentile(lengths, 50)}")
        print(f"[pid={os.getpid()}] 95% percentile of lengths: {np.percentile(lengths, 95)}")

        # trainable_tokens = [len([int(token) for token in sample.labels if token != IGNORE_INDEX]) for sample in samples]
        # print(f"[pid={os.getpid()}] 5% percentile of trainable tokens: {np.percentile(trainable_tokens, 5)}")
        # print(f"[pid={os.getpid()}] 50% percentile of trainable tokens: {np.percentile(trainable_tokens, 50)}")
        # print(f"[pid={os.getpid()}] 95% percentile of trainable tokens: {np.percentile(trainable_tokens, 95)}")

        return packed_samples

    @stateless
    def pack_selected_samples(
        self, samples: List[PackedTaskSample]
    ) -> PackedTaskSample:
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
        ), f"All images must be tensors: {[type(img) for sample in samples for img in sample.imgs]}"

        if self.args.allow_cross_sample_attention:
            cu_lengths = [0, cu_lengths[-1]]
            cu_lengths_padded = [0, cu_lengths_padded[-1]]
            max_length = sum(sample.max_length for sample in samples)
        else:
            max_length = max(sample.max_length for sample in samples)

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
            max_length=max_length,
            num_tiles=[n for s in samples for n in s.num_tiles],
            num_frames=[n for s in samples for n in s.num_frames],
            sound_clips=[sc for sample in samples for sc in sample.sound_clips],
            sound_length=[sl for sample in samples for sl in sample.sound_length],
            sound_timestamps=[st for sample in samples for st in sample.sound_timestamps],
            num_sound_clips=[ns for sample in samples for ns in sample.num_sound_clips],
            samples_seen=sum(s.samples_seen for s in samples),
        )

    def batch(self, samples: List[PackedTaskSample]) -> BatchedPackedTaskSample:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]

        # Pad image packed seq length to be % 16 if using fp8 and dynamic resolution
        has_fp8 = self.args.fp8 is not None
        has_pad_img = torch.tensor(False)
        if has_fp8 and self.args.dynamic_resolution:
            img_seq_len = 0
            for img in imgs:
                img_seq_len += (img.shape[1] // self.args.patch_dim) * (
                    img.shape[2] // self.args.patch_dim
                )
            padding_needed = get_padding(
                img_seq_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
                self.args.tp_comm_overlap,
                self.args.decoder_seq_length,
                fp8_enabled=has_fp8,
            )
            if padding_needed > 0:
                pad_img = torch.zeros(
                    [3, self.args.patch_dim, padding_needed * self.args.patch_dim]
                )
                imgs.append(pad_img)
                has_pad_img = torch.tensor(True)

        imgs, imgs_sizes, vision_cu_lengths, vision_max_lengths = (
            self.image_tiling_strategy.stack(imgs)
        )

        # For batch mode, wrap in additional dimension for consistency
        if imgs is None:
            imgs = torch.tensor([[0]], dtype=torch.float32)
        if imgs_sizes is None:
            imgs_sizes = torch.tensor([[0, 0]], dtype=torch.int32)
        # Set default values if no vision metadata was returned (static resolution case)
        if vision_cu_lengths is None:
            vision_cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        else:
            # Shape: (1, batch_size + 1)
            vision_cu_lengths = vision_cu_lengths.unsqueeze(0)
        if vision_max_lengths is None:
            vision_max_lengths = torch.tensor([[0]], dtype=torch.int32)
        else:
            # Shape: (1,)
            vision_max_lengths = vision_max_lengths.unsqueeze(0)

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

        num_frames = torch.tensor(
            [n for s in samples for n in s.num_frames], dtype=torch.int32
        )
        if len(num_frames) == 0:
            num_frames = torch.tensor([[0]], dtype=torch.int32)

        cu_lengths = torch.stack([s.cu_lengths for s in samples])
        cu_lengths_padded = torch.stack([s.cu_lengths_padded for s in samples])
        max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)

        if self.dataloader_seq_length is not None:
            cu_lengths[0][-1] = self.dataloader_seq_length
            cu_lengths_padded[0][-1] = self.dataloader_seq_length
            new_max_length = cu_lengths_padded[0][-1] - cu_lengths[0][-2]
            max_lengths = torch.max(max_lengths, new_max_length)

        # Pad entire sequence to be a multiple of 16 if using fp8.
        if has_fp8:
            total_seq_len = cu_lengths_padded[0][-1]
            padding_needed = get_padding(
                total_seq_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
                self.args.tp_comm_overlap,
                self.args.decoder_seq_length,
                fp8_enabled=has_fp8,
            )
            if padding_needed > 0:
                tokens = torch.cat([tokens, torch.full((tokens.shape[0], padding_needed), self.tokenizer.pad, dtype=torch.int64)], dim=1)
                labels = torch.cat([labels, torch.full((labels.shape[0], padding_needed), IGNORE_INDEX, dtype=torch.int64)], dim=1)
                cu_lengths[0][-1] += padding_needed
                cu_lengths_padded[0][-1] += padding_needed
                new_max_length = cu_lengths_padded[0][-1] - cu_lengths[0][-2]
                max_lengths = torch.max(max_lengths, new_max_length)


        sound_clips = torch.tensor([[0]], dtype=torch.float32)
        sound_length = torch.tensor([[0]], dtype=torch.int64)
        sound_timestamps = torch.tensor([[0]], dtype=torch.float32)
        num_sound_clips = torch.tensor([[0]], dtype=torch.int64)

        all_sound_clips = [sc for sample in samples for sc in sample.sound_clips]
        if all_sound_clips:
            # note(pzelasko): all_sound_clips is a list of tensors shaped (num_clips, sound_seq_len)
            # we need to flatten it and then pad it to the same length
            sound_clips = torch.nn.utils.rnn.pad_sequence([sc for clips in all_sound_clips for sc in clips], batch_first=True)
            sound_lengths = []
            for sample in samples:
                for sound_length in sample.sound_length:
                    for sl in sound_length:
                        sound_lengths.append(sl)
            sound_length = torch.tensor(sound_lengths, dtype=torch.int64)
            sound_timestamps = torch.tensor([st for sample in samples for st in sample.sound_timestamps], dtype=torch.float32)
            num_sound_clips = torch.tensor([ns for sample in samples for ns in sample.num_sound_clips], dtype=torch.int64)

        return BatchedPackedTaskSample(
            __key__=[s.__key__ for s in samples],
            __restore_key__=[s.__restore_key__ for s in samples],
            __subflavors__=samples[0].__subflavors__,
            tokens=tokens,
            labels=labels,
            imgs=imgs,
            num_tiles=num_tiles,
            num_frames=num_frames,
            cu_lengths=cu_lengths,
            cu_lengths_padded=cu_lengths_padded,
            max_lengths=max_lengths,
            imgs_sizes=imgs_sizes,
            vision_cu_lengths=vision_cu_lengths,
            vision_max_lengths=vision_max_lengths,
            has_pad_img=has_pad_img,
            sound_clips=sound_clips,
            sound_length=sound_length,
            sound_timestamps=sound_timestamps,
            num_sound_clips=num_sound_clips,
            samples_seen=sum(s.samples_seen for s in samples),
        )

    def encode_batch(self, batch: BatchedPackedTaskSample) -> dict:
        return dataclasses.asdict(batch)

    def _load_media(self, sample: PreEncodedTaskSample) -> None:
        """Loads all lazy media in the sample."""
        if len(sample.images) > 1:
            medias: dict[
                Lazy[AVDecoder] | Lazy[Image.Image], list[ImageTilingParams]
            ] = defaultdict(list)
            # Group by video and frame index
            for media in sample.images:
                # video is a Lazy[AVDecoder] (it's hashable, all pointing to the same file are the same object)
                medias[media.media.value].append(media)

            for media, frames in medias.items():
                media_value = media.get(sample)
                if isinstance(media_value, AVDecoder):
                    media_value.suppress_warnings = True
                    frame_clips = media_value.get_clips(
                        video_clip_ranges=[
                            (frame.media.timestamp, frame.media.timestamp)
                            for frame in frames
                        ],
                        video_unit="seconds",
                    )
                    images = [
                        tensor_to_pil(img[0]) for img in frame_clips.video_clips
                    ]

                    if len(images) < len(frames):
                        last_image = images[-1]
                        images.extend([last_image] * (len(frames) - len(images)))

                    for frame, image in zip(frames, images):
                        frame.media.value = image
                elif isinstance(media_value, Image.Image):
                    for frame in frames:
                        frame.media.value = media_value
                elif isinstance(media_value, str):
                    # Text fragment
                    pass
                elif isinstance(media_value, bytes):
                    raise ValueError(f"Unable to parse bytes as known media type: {media_value[:100]!r}")
                else:
                    raise ValueError(f"Unexpected media type: {type(media_value)}. Media: {media}")
        else:
            for media in sample.images:
                media.media.value = media.media.value.get(sample)
        for media in sample.audio:
            val = media.media.value.get(sample)
            if isinstance(val, tuple) or isinstance(val, list):
                val = val[0]
            media.media.value = val

    def _target_has_trainable_tokens(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
        audio_media_params: list[AudioParams],
    ) -> bool:
        """
        Check if the target has trainable tokens.

        Args:
            input_ids: Input tokens.
            target: Target tokens.
            image_tiling_params: Image tiling parameters.
            audio_media_params: Audio media parameters.

        Returns:
            True if the target has trainable tokens, False otherwise.
        """
        # Compute the loss mask based on extending the image tags with the proper
        # number of image tokens, extracting the first self.args.decoder_seq_length tokens, and
        # ensuring that some of these tokens have a loss mask > 0.
        # Note that this is a bit hacky because we reproduce here parts of the logics which are in
        # the model itself. Ideally, the data sampler would return the already processed inputs
        # and targets to avoid this duplication.
        expanded_target = target.clone()
        expanded_target[input_ids == self.img_token_id] = self.img_token_id
        if self.sound_token_id is not None:
            expanded_target[input_ids == self.sound_token_id] = self.sound_token_id
        expanded_target = self._replace_value_with_repetition(
            expanded_target,
            self.img_token_id,
            torch.tensor([media.num_embeddings for media in image_tiling_params]),
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
            arr: Input array to be modified
            token_to_replace: Token to be replaced
            num_repetition: Number of repetition of new token. Size must match the number of `token_to_replace` in the
                input array.
            new_token: New token to replace the `token_to_replace` with.

        Returns:
            Array: New array with token_to_replace replaced by num_repetition repetitions of new_token
        """
        assert torch.sum(arr == token_to_replace) == len(num_repetition), (
            "The number of tokens to replace must match the length of the tile tensor."
        )

        # Convert to list for easier manipulation
        arr_list = arr.tolist()
        result = []
        idx = 0
        for item in arr_list:
            if item == token_to_replace:
                # If the current item matches token_to_replace, add R copies of new_token
                result.extend([new_token] * num_repetition[idx].item())
                idx += 1
            else:
                # Otherwise, keep the original item
                result.append(item)

        return torch.tensor(result, dtype=arr.dtype, device=arr.device)

    def _pad_for_context_parallel_and_fp8(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
        audio_media_params: list[AudioParams],
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        total_len = self._get_total_seq_length(input_ids, image_tiling_params, audio_media_params)
        total_len_padded = total_len
        # With context parallel or sequence parallel, we need to pad individual sequences.
        # However, with FP8, we need to only pad the entire sequence to be a multiple of 16.
        if getattr(self.args, "context_parallel_size", 1) > 1 or self.args.sequence_parallel:
            padding_needed = get_padding(
                total_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
                self.args.tp_comm_overlap,
                self.args.decoder_seq_length,
                fp8_enabled=False,
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

    def _get_total_seq_length(
        self, input_ids: torch.Tensor, image_tiling_params: list[ImageTilingParams], audio_media_params: list[AudioParams]
    ):
        """Calculate expected sequence length given text tokens length and number of tiles."""
        total_num_images = len(image_tiling_params)
        total_num_image_embeddings = sum(media.num_embeddings for media in image_tiling_params)
        # Audio embeddings are already expanded.
        total_len = (
            len(input_ids)
            + total_num_image_embeddings
            - total_num_images
        )
        return total_len

    def _truncate_to_decoder_seq_len(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
        audio_media_params: list[AudioParams],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate tokens and labels if they exceed sequence length."""
        total_img_embeddings_len = sum(
            media.num_embeddings for media in image_tiling_params
        )
        total_num_images = len(image_tiling_params)
        total_num_audio_embeddings = sum(media.num_embeddings for media in audio_media_params)
        total_num_audio = len(audio_media_params)
        packing_seq_length = self.args.packing_seq_length if self.args.packing_seq_length > 0 else self.args.decoder_seq_length
        max_text_tokens = (
            packing_seq_length - total_img_embeddings_len - total_num_audio_embeddings + total_num_images + total_num_audio
        )

        truncated_input_ids = input_ids[:max_text_tokens]
        truncated_target = target[:max_text_tokens]

        # If truncate causes all labels to be ignored, then skip the sample
        if len(truncated_target) == 0 or (truncated_target == IGNORE_INDEX).all():
            raise ValueError(
                "All targets will be ignored after truncation: \n"
                f"original input: {input_ids} \n"
                f"original target: {target} \n"
                f"truncated input:  {truncated_input_ids} \n"
                f"truncated target: {truncated_target} \n"
                f"max_text_tokens: {max_text_tokens} \n"
                f"total_img_embeddings_len: {total_img_embeddings_len} \n"
                f"total_num_images: {total_num_images} \n"
                f"total_num_audio_embeddings: {total_num_audio_embeddings} \n"
                f"total_num_audio: {total_num_audio} \n"
            )

        return truncated_input_ids, truncated_target


tensor_to_pil = ToPILImage()
