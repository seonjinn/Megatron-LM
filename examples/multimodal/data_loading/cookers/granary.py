# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from collections import defaultdict

from megatron.energon import CachePool, FileStore, cooker, stateless

from ..conversation_sample import (
    ConversationSample,
)

from ..conversation_base import (
    conversation_tags_mapping_sample_to_allowed,
    conversation_convert_message,
    conversation_post_processing,
)

tags_mapping_sample_to_allowed = {
    **conversation_tags_mapping_sample_to_allowed,
    **{
        'audio_filepath': 'sound',
    }
}


granary_english_question = "<sound>. \nTranscribe the spoken content to written english text, with punctuations and capitalizations."


@stateless
@cooker(need_cache=True, need_primary=True)
def cook_granary_english_webdataset(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    """Loads Granary audio datasets of webdataset format (monolithic or polylithic)."""

    data = sample["json"].copy()
    for tag in tags_mapping_sample_to_allowed:
        if tag in data and tag != tags_mapping_sample_to_allowed[tag]:
            sample['json'][tags_mapping_sample_to_allowed[tag]] = data[tag]
            del sample['json'][tag]
    # update the data
    data = sample["json"].copy()

    media_index = defaultdict(int)
    tried_default_extensions = set()

    # Build the conversation
    conversation = []
    data["conversations"] = [
        {"from": "user", "value": granary_english_question},
        {"from": "assistant", "value": data["text"]},
    ]
    for msg in data["conversations"]:
        conversation.append(
            conversation_convert_message(
                data,
                msg,
                media_index,
                raw=sample,
                check_if_media_file_exist=False,
                tried_default_extensions=tried_default_extensions,
                tags_mapping_sample_to_allowed=tags_mapping_sample_to_allowed,
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
def cook_granary_english_jsonl(
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> ConversationSample:
    data = sample["json"].copy()

    """Loads Granary datasets that have the json (manifest) files and media files in separate files (jsonl datasets)."""
    for tag in tags_mapping_sample_to_allowed:
        if tag in data and tag != tags_mapping_sample_to_allowed[tag]:
            sample["json"][tags_mapping_sample_to_allowed[tag]] = data[tag]
            del sample["json"][tag]

    data = sample["json"].copy()

    media_index = defaultdict(int)
    tried_default_extensions = set()

    # Build the conversation
    conversation = []
    data["conversations"] = [
        {"from": "user", "value": granary_english_question},
        {"from": "assistant", "value": data["text"]},
    ]
    for msg in data["conversations"]:
        conversation.append(
            conversation_convert_message(
                data,
                msg,
                media_index,
                check_if_media_file_exist=False,
                tried_default_extensions=tried_default_extensions,
                tags_mapping_sample_to_allowed=tags_mapping_sample_to_allowed,
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
