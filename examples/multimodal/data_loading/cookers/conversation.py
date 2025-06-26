# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless

from ..conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
)


@stateless
@cooker(need_cache=True)
def cook_conversation(
    sample: dict,
    cache: CachePool,
    media_source: FileStore,
) -> ConversationSample:
    data = sample["json"]
    cs = ConversationSample.from_json(data, **basic_sample_keys(sample))

    for msg in cs.conversation:
        for frag in msg.fragments:
            if isinstance(frag, (ImageMedia, VideoMedia, AudioMedia, VideoFrameMedia)):
                frag.value = cache.get_lazy(media_source, frag.value)
            elif isinstance(frag, str):
                # No source
                pass
            else:
                raise ValueError(f"Unknown fragment type: {type(frag)}")

    return cs
