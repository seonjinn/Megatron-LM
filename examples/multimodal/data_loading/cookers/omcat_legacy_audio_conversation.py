# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from collections import defaultdict
import dataclasses
import re

from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless

from ..conversation_sample import (
    Message,
    Media,
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
)


# List of allowed tags
allowed_tags = ["image", "video", "sound", "video-sound"]

additional_tags = {
    'image': ['png', 'jpeg', 'jpg', 'img'],
    'video': ['mp4'],
    'sound': ['wav', 'flac', "mp3"],
}

# WARNING: values cannot be used as the keys in the same dict to avoid cyclic graph
tags_mapping_sample_to_allowed = {
    'speech': 'sound',
    'speeches': 'sound',
    'audio': 'sound',
    'audios': 'sound',
    'images': 'image',
    'videos': 'video',
}

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
        tag_map = additional_tags.get(ctag, ())
        for tag in tag_map:
            if tag in raw:
                key = tag
                break
        else:
            # For capitalized extensions.
            for key, _ in raw.items():
                if key.lower() in tag_map:
                    break
            else:
                raise ValueError(f"Tag {ctag} not found in {tag_map} in {raw.keys()}")

        if ctag == "image":
            result.append(ImageMedia(value=key))
        elif ctag == "video" or ctag == "vis_video":
            result.append(VideoMedia(value=key))
        elif ctag == "sound" or ctag == "vis_sound":
            result.append(AudioMedia(value=key))
        else:
            raise ValueError(f"Unknown tag: {ctag}")

    if len(result) == 0:
        raise ValueError(f"Tag {tag} not found in sample {raw['__key__']}: {raw.keys()}")

    return result


def convert_message(sample: dict, msg: dict, tags_appeared: set) -> Message:

    fragments = []

    for tag in tags_mapping_sample_to_allowed:
        tag_str = '<' + tag + '>'
        if tag_str in msg["value"]:
            tag_str_mapped = '<' + tags_mapping_sample_to_allowed[tag] + '>'
            msg["value"] = msg["value"].replace(tag_str, tag_str_mapped)

    parts = re.split(tag_pattern, msg["value"])

    # Convert the parts to message fragments
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Odd indices are the captured tags (without angle brackets)
            if part in tags_appeared:
                raise ValueError(f"Tag {part} appeared twice in sample {sample['__key__']}: {sample.keys()}")
            tags_appeared.add(part)
            fragments += convert_tag_to_objects(part, sample)
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


warn_about_slow_media_loading = defaultdict(lambda: True)

_re_clean_path = re.compile(r"(?:^\./|/\.(?=/))")


@stateless
@cooker(need_cache=True, need_primary=True)
def cook_omcat_legacy_conversation_monolithic(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
) -> ConversationSample:
    """Loads audio datasets that have the media in the same shards (monolithic) with legacy format."""
    global warn_about_slow_media_loading
    data = sample["json"]

    for tag in tags_mapping_sample_to_allowed:
        if tag in data and tag != tags_mapping_sample_to_allowed[tag]:
            data[tags_mapping_sample_to_allowed[tag]] = data[tag]
            del data[tag]

    # from pprint import pprint
    # print("ConversationSample json:")
    # pprint(data)

    tags_appeared = set()

    # Build the conversation
    conversation = []
    for msg in data["conversations"]:
        conversation.append(convert_message(sample, msg, tags_appeared))

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

        raise ValueError(f"Tag {key} not covered in sample {data['__key__']}: {data}")

    cs = ConversationSample(
        conversation=conversation,
        **basic_sample_keys(sample),
    )

    for msg in cs.conversation:
        for frag in msg.fragments:
            if isinstance(frag, (ImageMedia, VideoMedia, AudioMedia, VideoFrameMedia)):
                val = sample[frag.value]
                # frag.value = cache.get_lazy(primary, sample['__key__'].rsplit(".tar/", 1)[-1] + f".{frag.value}")
                frag.value = cache.to_cache(val, sample['__key__'] + f".{frag.value}")
                # frag.value = media_source.get(frag.value, cs)
                # if isinstance(frag, ImageMedia):
                #     assert isinstance(frag.value, Image.Image), f"ImageMedia must be an Image.Image, got {type(frag.value)}"
                # elif isinstance(frag, VideoMedia):
                #     assert isinstance(frag.value, AVDecoder), f"VideoMedia must be an AVDecoder, got {type(frag.value)}"
                # elif isinstance(frag, AudioMedia):
                #     assert isinstance(frag.value, AVDecoder), f"AudioMedia must be an AVDecoder, got {type(frag.value)}"
                # elif isinstance(frag, VideoFrameMedia):
                #     assert isinstance(frag.value, AVDecoder), f"VideoFrameMedia must be an AVDecoder, got {type(frag.value)}"
                if frag.metadata is None:
                    try:
                        frag.metadata = dataclasses.asdict(primary.get_media_metadata(f".{frag.value}"))
                    except Exception as e:
                        if warn_about_slow_media_loading[primary.get_path()]:
                            print(f"WARNING: Dataset {primary.get_path()} not prepared with media metadata, slow metadata for .{frag.value}: {e!r}")
                            warn_about_slow_media_loading[primary.get_path()] = False

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
