# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
from collections import defaultdict

from megatron.energon import CachePool, FileStore, SourceInfo, basic_sample_keys, cooker, stateless

from ..conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
    Message,
)

from ..conversation_base import (
    conversation_tags_mapping_sample_to_allowed,
    conversation_convert_message,
    conversation_post_processing,
)

warn_about_slow_media_loading = defaultdict(lambda: True)

@stateless
@cooker(need_cache=True)
def cook_conversation(
    sample: dict,
    cache: CachePool,
    media_source: FileStore | None = None,
) -> ConversationSample:
    global warn_about_slow_media_loading

    data = sample["json"]
    cs = ConversationSample.from_json(data, **basic_sample_keys(sample))

    for msg in cs.conversation:
        for frag in msg.fragments:
            if isinstance(frag, (ImageMedia, VideoMedia, AudioMedia, VideoFrameMedia)):
                if media_source is None:
                    raise ValueError("cook_conversation requires media_source for samples with media fragments")
                if frag.metadata is None:
                    try:
                        frag.metadata = dataclasses.asdict(media_source.get_media_metadata(frag.value))
                    except Exception as e:
                        if warn_about_slow_media_loading[media_source.get_path()]:
                            print(f"WARNING: Dataset {media_source.get_path()} not prepared with media metadata, slow metadata for {frag.value}: {e!r}")
                            warn_about_slow_media_loading[media_source.get_path()] = False
                cs.__sources__ = (
                    *cs.__sources__,
                    SourceInfo(dataset_path=media_source.get_path(), index=frag.value, shard_name=None, file_names=(frag.value,)),
                )
                frag.value = cache.get_lazy(media_source, frag.value)
            elif isinstance(frag, str):
                # No source
                pass
            else:
                raise ValueError(f"Unknown fragment type: {type(frag)}")

    return cs


@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_webdataset(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    """Loads general conversation-based datasets of webdataset format (monolithic or polylithic).
    Each sample can be single/multi-turn converstaions with multiple modalities.
    Each modality can have one or more number of media objects.
    There is no requiement of where the media tag (e.g. '<sound>') should appear in the conversations.

    The structure of the shard files could be like this:

    `tar -tvf shard_0.tar`:
    ```python
    sample_000001.2345ew.flac
    sample_000001.35tags.mp4
    sample_000001.as23ds.jpg
    sample_000001.gd1dtg.wav
    sample_000001.gds233.jpg
    sample_000002.asf234.wav
    ...
    ```

    ```json structure
    {
      "sound": ["sample_000001.2345ew.flac", "sample_000001.gd1dtg.wav"],
      "video": "sample_000001.35tags.mp4",
      "image": ["sample_000001.as23ds.jpg", "sample_000001.gds233.jpg"],
      "conversations": [
        {
          "from": "user",
          "value": "<sound>"
        },
        {
          "from": "assistant",
          "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition."
        },
        {
          "from": "user",
          "value": "Describe what is NeMo based on the tutorial video: <video> and the information in the two images: <image> <image>. Combine that information with sound <sound>. Answer: "
        },
        {
          "from": "assistant",
          "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability."
        }
      ]
    }
    ```
    """

    data = sample["json"].copy()
    for tag in conversation_tags_mapping_sample_to_allowed:
        if tag in data and tag != conversation_tags_mapping_sample_to_allowed[tag]:
            sample['json'][conversation_tags_mapping_sample_to_allowed[tag]] = data[tag]
            del sample['json'][tag]
    # update the data
    data = sample["json"].copy()

    media_index = defaultdict(int)
    tried_default_extensions = set()

    # Build the conversation
    conversation = []
    for msg in data["conversations"]:
        conversation.append(
            conversation_convert_message(
                data,
                msg,
                media_index,
                raw=sample,
                check_if_media_file_exist=False,
                tried_default_extensions=tried_default_extensions,
                tags_mapping_sample_to_allowed=conversation_tags_mapping_sample_to_allowed,
            )
        )

    # Check if all media files are retrieved
    for media in media_index:
        medias = data[media]
        if not isinstance(medias, list):
            medias = [medias]
        if media_index[media] != len(medias):
            raise ValueError(f"Retrieved {media_index[media]}/{len(medias)} {media} files from {sample}")

    return conversation_post_processing(
        conversation,
        sample,
        cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@stateless
@cooker(need_primary=True, need_cache=True)
def cook_general_conversations_jsonl(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    data = sample["json"].copy()

    """Loads general conversation datasets that have the json (manifest) files and media files in separate files (jsonl datasets).
    The json(l) file structure is the same as the cook_general_conversations_webdataset
    """
    for tag in conversation_tags_mapping_sample_to_allowed:
        if tag in data and tag != conversation_tags_mapping_sample_to_allowed[tag]:
            sample["json"][conversation_tags_mapping_sample_to_allowed[tag]] = data[tag]
            del sample["json"][tag]

    data = sample["json"].copy()

    media_index = defaultdict(int)
    tried_default_extensions = set()

    # Build the conversation
    conversation = []
    for msg in data["conversations"]:
        conversation.append(
            conversation_convert_message(
                data,
                msg,
                media_index,
                check_if_media_file_exist=False,
                tried_default_extensions=tried_default_extensions,
                tags_mapping_sample_to_allowed=conversation_tags_mapping_sample_to_allowed,
            )
        )

    # Check if all media files are retrieved
    for media in media_index:
        medias = data[media]
        if not isinstance(medias, list):
            medias = [medias]
        if media_index[media] != len(medias):
            raise ValueError(f"Retrieved {media_index[media]}/{len(medias)} {media} files from {sample}")

    return conversation_post_processing(
        conversation,
        sample,
        cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )
