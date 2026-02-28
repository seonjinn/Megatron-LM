# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from collections import defaultdict
import dataclasses
from pathlib import Path
import re

from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless, SourceInfo
from megatron.energon.av import AVDecoder
from PIL import Image

from ..conversation_sample import (
    Message,
    Media,
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
)


warn_about_slow_media_loading = defaultdict(lambda: True)

allowed_tags = ["image", "video", "sound", "video-sound"]

# Build a pattern like: <image>|<video>|<sound>|<video-sound>
tag_pattern = re.compile(
    r"<(" + "|".join(re.escape(tag) for tag in allowed_tags) + ")>"
)

def convert_tag_to_objects(tag: str, raw: dict) -> list[Media]:
    if tag == "video-sound":
        # For video-sound, we insert both video and sound
        tags = ["vis_video", "vis_sound"]
    else:
        tags = [tag]

    result = []
    for ctag in tags:
        for key, value in raw.items():
            if key.startswith(ctag):
                if ctag == "image":
                    result.append(ImageMedia(value=value))
                elif ctag == "video" or ctag == "vis_video":
                    result.append(VideoMedia(value=value))
                elif ctag == "sound" or ctag == "vis_sound":
                    result.append(AudioMedia(value=value))
                else:
                    raise ValueError(f"Unknown tag: {ctag}")

    if len(result) == 0:
        raise ValueError(f"Tag {tag} not found in sample {raw['id']}: {raw}")

    return result


def convert_message(data: dict, msg: dict, tags_appeared: set) -> Message:

    fragments = []

    parts = re.split(tag_pattern, msg["value"])

    # Convert the parts to message fragments
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Odd indices are the captured tags (without angle brackets)
            if part in tags_appeared:
                raise ValueError(f"Tag {part} appeared twice in sample {data['id']}: {data}")
            tags_appeared.add(part)
            fragments += convert_tag_to_objects(part, data)
        else:
            # Even indices are plain text, but skip empty strings
            if part.strip():
                fragments.append(part)

    if msg["from"] == "human":
        msg["from"] = "user"
    elif msg["from"] == "gpt":
        msg["from"] = "assistant"
    else:
        raise ValueError(f"Unknown sender: {msg['from']}")

    return Message(
        sender=msg["from"],
        fragments=fragments,
    )


_re_clean_path = re.compile(r"(?:^\./|/\.(?=/))")


@stateless
@cooker(need_primary=True, need_cache=True)
def cook_audio_conversation(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    """Loads datasets that have the media in separate files (polylithic)."""
    global warn_about_slow_media_loading

    data = sample["json"]

    # from pprint import pprint
    # print("ConversationSample json:")
    # pprint(data)

    tags_appeared = set()

    # Build the conversation
    conversation = []
    for msg in data["conversations"]:
        conversation.append(convert_message(data, msg, tags_appeared))

    # Check that all data in the sample is covered by the tags
    for key, _ in data.items():
        if key not in allowed_tags:
            continue

        if key in tags_appeared:
            # All good. This is covered
            continue

        if key == "sound-video" and "video-sound" in tags_appeared:
            # All good. This is covered
            continue

        raise ValueError(f"Tag {key} not covered in sample {data['id']}: {data}")

    cs = ConversationSample(
        conversation=conversation,
        **basic_sample_keys(sample),
    )

    for msg in cs.conversation:
        for frag in msg.fragments:
            if isinstance(frag, (ImageMedia, VideoMedia, AudioMedia, VideoFrameMedia)):
                if media_source is not None:
                    # print(f"Cooking {frag.value!r} from {media_source.get_path()!r}")
                    if frag.metadata is None:
                        try:
                            frag.metadata = dataclasses.asdict(media_source.get_media_metadata(frag.value))
                        except Exception as e:
                            if warn_about_slow_media_loading[media_source.get_path()]:
                                print(f"WARNING: Dataset {media_source.get_path()} not prepared with media metadata, slow metadata for {frag.value}: {e!r}")
                                warn_about_slow_media_loading[media_source.get_path()] = False
                    if frag.metadata is None:
                        val = cache.get(media_source, frag.value, cs)
                    cs.__sources__ = (
                        *cs.__sources__,
                        SourceInfo(dataset_path=media_source.get_path(), index=frag.value, shard_name=None, file_names=(frag.value,)),
                    )
                    frag.value = cache.get_lazy(media_source, frag.value)
                    # frag.value = media_source.get(frag.value, cs)
                    # if isinstance(frag, ImageMedia):
                    #     assert isinstance(frag.value, Image.Image), f"ImageMedia must be an Image.Image, got {type(frag.value)}"
                    # elif isinstance(frag, VideoMedia):
                    #     assert isinstance(frag.value, AVDecoder), f"VideoMedia must be an AVDecoder, got {type(frag.value)}"
                    # elif isinstance(frag, AudioMedia):
                    #     assert isinstance(frag.value, AVDecoder), f"AudioMedia must be an AVDecoder, got {type(frag.value)}"
                    # elif isinstance(frag, VideoFrameMedia):
                    #     assert isinstance(frag.value, AVDecoder), f"VideoFrameMedia must be an AVDecoder, got {type(frag.value)}"
                else:
                    path = _re_clean_path.sub("", frag.value)
                    # We need to find the media source for the fragment
                    for prefix, aux_key in cs.__subflavors__["aux_data_prefixes"].items():
                        if path.startswith(prefix):
                            # print(f"Cooking {frag.value!r} from {media_sources[aux_key].get_path()!r}")
                            # Matching the prefix, so use that media source
                            if frag.metadata is None:
                                try:
                                    frag.metadata = dataclasses.asdict(media_sources[aux_key].get_media_metadata(path[len(prefix):]))
                                except Exception as e:
                                    if warn_about_slow_media_loading[media_sources[aux_key].get_path()]:
                                        print(f"WARNING: Dataset {media_sources[aux_key].get_path()} not prepared with media metadata, slow metadata for {path[len(prefix):]}: {e!r}")
                                        warn_about_slow_media_loading[media_sources[aux_key].get_path()] = False
                            if frag.metadata is None:
                                val = cache.get(media_sources[aux_key], path[len(prefix):], cs)
                            cs.__sources__ = (
                                *cs.__sources__,
                                SourceInfo(dataset_path=media_sources[aux_key].get_path(), index=path[len(prefix):], shard_name=None, file_names=(path[len(prefix):],)),
                            )
                            frag.value = cache.get_lazy(media_sources[aux_key], path[len(prefix):])
                            # frag.value = media_sources[aux_key].get(path[len(prefix):], cs)
                            break
                    else:
                        raise ValueError(f"No prefix for {path!r} in {cs.__subflavors__['aux_data_prefixes']} for {cs.__sources__}")

                if frag.metadata is None:
                    if isinstance(frag, ImageMedia):
                        frag.metadata = dict(
                            width=val.width, height=val.height, format=val.format, mode=val.mode
                        )
                    elif isinstance(frag, (VideoMedia, AudioMedia, VideoFrameMedia)):
                        frag.metadata = dataclasses.asdict(val.get_metadata())

                        # if isinstance(frag, ImageMedia):
                        #     assert isinstance(frag.value, Image.Image), f"ImageMedia must be an Image.Image, got {type(frag.value)}"
                        # elif isinstance(frag, VideoMedia):
                        #     assert isinstance(frag.value, AVDecoder), f"VideoMedia must be an AVDecoder, got {type(frag.value)}"
                        # elif isinstance(frag, AudioMedia):
                        #     assert isinstance(frag.value, AVDecoder), f"AudioMedia must be an AVDecoder, got {type(frag.value)}"
                        # elif isinstance(frag, VideoFrameMedia):
                        #     assert isinstance(frag.value, AVDecoder), f"VideoFrameMedia must be an AVDecoder, got {type(frag.value)}"

            elif isinstance(frag, str):
                # No source
                pass
            else:
                raise ValueError(f"Unknown fragment type: {type(frag)}")

    # print("ConversationSample after cooking:")
    # pprint(cs.conversation)

    return cs
