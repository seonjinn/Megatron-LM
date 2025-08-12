# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from dataclasses import asdict

from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless

from ..conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
    Message,
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


@stateless
@cooker(need_cache=True)
def cook_audio(
    sample: dict,
    cache: CachePool,
) -> ConversationSample:

    cs = ConversationSample(conversation=[], **basic_sample_keys(sample))

    for turn in sample["json"]["conversations"]:
        role = None
        if turn["from"] == "human":
            role = "user"
        elif turn["from"] == "gpt":
            role = "assistant"
        else:
            raise ValueError(f"Unknown role: {turn['from']}")

        text = turn["value"]
        msg = Message(sender=role, fragments=[text])

        if "<video-sound>" in text or "<video>" in text or "<audio>" in text or "<image>" in text:
            if "<video-sound>" in text:
                val = sample["vis_video.mp4"]
                metadata = asdict(val.get_metadata())

                msg.fragments.append(VideoMedia(value=val, metadata=metadata))

                val = sample["vis_sound.wav"]
                metadata = asdict(val.get_metadata())
                msg.fragments.append(AudioMedia(value=val, metadata=metadata))
            elif "<video>" in text:
                msg.fragments.append(VideoMedia(value=turn["value"]))

                import os
                if int(os.environ.get("RANK", 0)) == 0:
                    breakpoint()
                else:
                    import time
                    time.sleep(1000)

            elif "<audio>" in text:
                msg.fragments.append(AudioMedia(value=turn["value"]))

                import os
                if int(os.environ.get("RANK", 0)) == 0:
                    breakpoint()
                else:
                    import time
                    time.sleep(1000)

            elif "<image>" in text:
                msg.fragments.append(ImageMedia(value=turn["value"]))

                import os
                if int(os.environ.get("RANK", 0)) == 0:
                    breakpoint()
                else:
                    import time
                    time.sleep(1000)


        cs.conversation.append(msg)

    return cs
