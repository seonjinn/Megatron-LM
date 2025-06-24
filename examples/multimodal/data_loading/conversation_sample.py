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


@edataclass
class VideoMedia(Media):
    """A video media object in a conversation. May contain audio."""

    value: Union[AVDecoder, torch.Tensor, Lazy[AVDecoder], str]

    #: If set, the video needs to be trimmed to the given range in seconds.
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    metadata: dict[str, Union[str, int, float, bool]] | None = None


@edataclass
class VideoFrameMedia(Media):
    """A video frame media object in a conversation."""

    value: Union[AVDecoder, torch.Tensor, Lazy[AVDecoder], str]

    timestamp: Optional[float] = None
    frame_index: Optional[int] = None

    metadata: dict[str, Union[str, int, float, bool]] | None = None


@edataclass
class AudioMedia(Media):
    """An audio media object in a conversation."""

    value: Union[AVDecoder, torch.Tensor, Lazy[AVDecoder], str]

    metadata: dict[str, Union[str, int, float, bool]] | None = None


@edataclass
class TextMedia(Media):
    """A text media object in a conversation."""

    value: str


@edataclass
class Message:
    """A message in a conversation between a user and an assistant."""

    #: The sender of the message
    sender: Literal["user", "assistant", "system"]

    #: The message content
    fragments: list[Media]


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
        "text": TextMedia,
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
                        ConversationSample.__MEDIA_TYPES__[frag.pop("t")](**frag)
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
                        dict(
                            t=ConversationSample.__MEDIA_TYPES_REVERSE__[type(frag)],
                            **dataclasses.asdict(frag),
                        )
                        for frag in msg.fragments
                    ],
                )
                for msg in self.conversation
            ],
        )
