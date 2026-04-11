# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
from typing import Literal, Optional, Union

import torch
from PIL import Image

from megatron.energon import Lazy, Sample
from megatron.energon.edataclass import edataclass
from megatron.energon.av import AVDecoder


@edataclass
class Media:
    """A media object in a conversation."""

    pass


@edataclass
class ImageMedia(Media):
    """An image media object in a conversation."""

    value: Union[torch.Tensor, Image.Image, Lazy[Image.Image], str]

    metadata: dict[str, Union[str, int, float, bool]] | None = None

    @property
    def width(self) -> int:
        return self.metadata["width"]

    @property
    def height(self) -> int:
        return self.metadata["height"]

    @width.setter
    def width(self, value):
        self.metadata["width"] = value

    @height.setter
    def height(self, value):
        self.metadata["height"] = value


@edataclass
class VideoMedia(Media):
    """A video media object in a conversation. May contain audio."""

    value: Union[AVDecoder, torch.Tensor, Lazy[AVDecoder], str]

    #: If set, the video needs to be trimmed to the given range in seconds.
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    metadata: dict[str, Union[str, int, float, bool]] | None = None

    @property
    def clip_duration(self) -> float:
        start_time = self.start_time
        end_time = self.end_time
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = self.metadata["video_duration"]
        return end_time - start_time

    @property
    def video_width(self) -> int:
        return self.metadata["video_width"]

    @property
    def video_height(self) -> int:
        return self.metadata["video_height"]

    @property
    def video_duration(self) -> float:
        return self.metadata["video_duration"]

    @property
    def video_num_frames(self) -> int:
        return self.metadata["video_num_frames"]

    @property
    def video_fps(self) -> float:
        return self.metadata["video_fps"]


@edataclass
class VideoFrameMedia(Media):
    """A video frame media object in a conversation."""

    value: Union[AVDecoder, torch.Tensor, Lazy[AVDecoder], str]

    timestamp: Optional[float] = None

    # Frame index: original frame index in source video; non-integer means we're interpolating
    # Sample index: consecutive index (0, 1, 2, ...) within the sampled frames for the video media
    frame_index: Optional[Union[int, float]] = None
    sample_index: Optional[int] = None

    metadata: dict[str, Union[str, int, float, bool]] | None = None

    @property
    def video_width(self) -> int:
        return self.metadata["video_width"]

    @property
    def video_height(self) -> int:
        return self.metadata["video_height"]


@edataclass
class AudioMedia(Media):
    """An audio media object in a conversation."""

    value: Union[AVDecoder, torch.Tensor, Lazy[AVDecoder], str]

    metadata: dict[str, Union[str, int, float, bool]] | None = None

    @property
    def audio_duration(self) -> float:
        return self.metadata["audio_duration"]

    @property
    def audio_samples_per_second(self) -> int:
        return self.metadata["audio_sample_rate"]

    @property
    def audio_channels(self) -> int:
        return self.metadata["audio_channels"]


@edataclass
class Message:
    """A message in a conversation between a user and an assistant."""

    #: The sender of the message
    sender: Literal["user", "assistant", "system", "tool"]

    #: The message content
    fragments: list[Media | str]


@edataclass
class ConversationSample(Sample):
    """Sample type for a conversation between a user and an assistant.

    Can include media of various types.
    """

    __MEDIA_TYPES__ = {
        "image": ImageMedia,
        "video": VideoMedia,
        "video_frame": VideoFrameMedia,
        "audio": AudioMedia,
    }
    __MEDIA_TYPES_REVERSE__ = {v: k for k, v in __MEDIA_TYPES__.items()}

    #: The messages in the conversation
    conversation: list[Message]

    @staticmethod
    def from_json(json_data: dict, **kwargs) -> "ConversationSample":
        return ConversationSample(
            conversation=[
                Message(
                    sender=msg["sender"],
                    fragments=[
                        (
                            frag
                            if isinstance(frag, str)
                            # TODO: This is a hack to support legacy formatted text media in the conversation
                            else (
                                frag["value"]
                                if frag["t"] == "text"
                                else ConversationSample.__MEDIA_TYPES__[frag.pop("t")](
                                    **frag
                                )
                            )
                        )
                        for frag in msg["fragments"]
                    ],
                )
                for msg in json_data["conversation"]
            ],
            **kwargs,
        )

    def to_json(self) -> dict:
        return dict(
            conversation=[
                dict(
                    sender=msg.sender,
                    fragments=[
                        frag
                        if isinstance(frag, str)
                        else dict(
                            t=ConversationSample.__MEDIA_TYPES_REVERSE__[type(frag)],
                            **dataclasses.asdict(frag),
                        )
                        for frag in msg.fragments
                    ],
                )
                for msg in self.conversation
            ],
        )
