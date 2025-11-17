# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import os
import re
import io
import copy
import dataclasses
from PIL import Image
from pathlib import Path
from collections import defaultdict

from megatron.energon.av import AVDecoder
from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless

from .misc import retrieve_media_source

from .conversation_sample import (
    Media,
    AudioMedia,
    ConversationSample,
    ImageMedia,
    VideoFrameMedia,
    VideoMedia,
    Message,
)


# List of allowed tags
conversation_allowed_tags = {
    'image': '<image>',
    'video': '<video>',
    'sound': '<sound>',
    'video-sound': '<video-sound>',
}

conversation_default_media_extensions = {
    'image': ['png','jpeg','jpg', 'img'],
    'video': ['mp4'],
    'video-sound': ['mp4'],
    'sound': ['wav', 'flac', "mp3"],
}

# WARNING: values cannot be used as the keys in the same dict to avoid cyclic graph
conversation_tags_mapping_sample_to_allowed = {
    'speech': 'sound',
    'speeches': 'sound',
    'audio': 'sound',
    'audios': 'sound',
    'images': 'image',
    'videos': 'video',
}


# map the senders from the sample to the allowed ones
# allowed senders can be found in conversation_sample.py
conversation_sender_mapping_sample_to_allowed = {
    'human': 'user',
    'gpt': 'assistant',
    'agent': 'assistant',
}


# Build a pattern like: <image>|<video>|<sound>|<video-sound>
conversation_tag_pattern = re.compile(
    r"(" + "|".join(re.escape(tag) for tag in conversation_allowed_tags.values()) + ")"
)


def conversation_open_medias(media_tag: str, media):
    if media_tag == "image":
        if isinstance(media, bytes):
            media = Image.open(io.BytesIO(media))
        elif not isinstance(media, Image.Image):
            media = Image.open(media)
    elif media_tag == "video" or \
        media_tag == "vis_video":
        if not isinstance(media, AVDecoder):
            if isinstance(media, str):
                media = Path(media).read_bytes()
            if isinstance(media, bytes):
                media = io.BytesIO(media)
            media = AVDecoder(media)
    elif media_tag == "audio" or \
        media_tag == "sound" or \
        media_tag == "vis_sound":
        if not isinstance(media, AVDecoder):
            if isinstance(media, str):
                media = Path(media).read_bytes()
            if isinstance(media, bytes):
                media = io.BytesIO(media)
            media = AVDecoder(media)
    else:
        raise ValueError(f"Unknown tag: {tag}")

    return media


def conversation_convert_tag_to_objects(tag: str, value) -> list[Media]:
    if tag == "video-sound":
        # For video-sound, we insert both video and sound
        tags = ["vis_video", "vis_sound"]
    else:
        tags = [tag]

    result = []
    for tag in tags:
        if tag == "image":
            result.append(
                ImageMedia(value=value)
            )
        elif tag == "video" or tag == "vis_video":
            result.append(
                VideoMedia(value=value)
            )
        elif tag == "sound" or tag == "vis_sound":
            result.append(
                AudioMedia(value=value)
            )
        else:
            raise ValueError(f"Unknown tag: {tag}")
    
    if len(result) == 0:
        raise ValueError(f"Tag {tag} not found in sample: {raw}")

    return result


def conversation_convert_message(
    meta: dict,
    msg: dict,
    media_index: dict,
    raw: dict = {},
    check_if_media_file_exist=True,
    tried_default_extensions: set = set(),
    tags_mapping_sample_to_allowed: dict = conversation_tags_mapping_sample_to_allowed,
) -> Message:
    """
    Args:
        raw: dictionary with all webdataset compliant keys of a sample. 
            Emtpy for jsonl dataset, non-empty otherwise
        meta: 
    """
    fragments = []
    for tag in tags_mapping_sample_to_allowed:
        tag_str = '<' + tag + '>'
        if tag_str in msg["value"]:
            tag_str_mapped = '<' + tags_mapping_sample_to_allowed[tag] + '>'
            msg["value"] = msg["value"].replace(tag_str, tag_str_mapped)
    parts = re.split(conversation_tag_pattern, msg["value"])
    
    # Convert the parts to message fragments
    empty_text = True
    for i, part in enumerate(parts):
        if part in conversation_allowed_tags.values():
            tag = part.strip('<>')
            if not isinstance(meta[tag], list):
                meta[tag] = [meta[tag]]
            # try to extract the media object from the shard
            ext = os.path.basename(meta[tag][media_index[tag]]).split('.', 1)[1]
            if raw and ext not in raw and \
                tag not in tried_default_extensions and \
                tag in conversation_default_media_extensions:
                # try the default extension
                for ext in conversation_default_media_extensions[tag]:
                    if ext in raw:
                        tried_default_extensions.add(ext)
                        break
            media_file = None
            if ext in raw:
                media_file = ext
            elif isinstance(meta[tag][media_index[tag]], str) and \
                os.path.isfile(meta[tag][media_index[tag]]):
                # if cannot get it from the shard files, try to find the local file
                media_file = meta[tag][media_index[tag]]
            elif check_if_media_file_exist:
                sample_to_print = raw if raw else meta
                raise ValueError(f"Cannot find the media file {meta[tag][media_index[tag]]} from {sample_to_print} or locally.")
            else:
                media_file = meta[tag][media_index[tag]]
            media_index[tag] += 1
            fragments += conversation_convert_tag_to_objects(tag, media_file)
        else:
            # Even indices are plain text, but skip empty strings
            if part.strip():
                fragments.append(part)
                empty_text = False
                
    if empty_text:
        fragments.append(' ')

    sender = msg["from"]
    if sender in conversation_sender_mapping_sample_to_allowed:
        sender = conversation_sender_mapping_sample_to_allowed[sender]
    return Message(
        sender=sender,
        fragments=fragments,
    ) 


warn_about_slow_media_loading = defaultdict(lambda: True)


def conversation_post_processing(
    conversation: list,
    sample: dict,
    cache: CachePool,
    primary: FileStore,
    media_source: FileStore | None = None,
    process_conversation_in_place: bool = True,
    **media_sources: FileStore,
) -> ConversationSample:
    global warn_about_slow_media_loading
    
    messages = conversation
    if not process_conversation_in_place:
        messages = copy.deepcopy(messages)

    for msg in messages:
        for frag in msg.fragments:
            if isinstance(frag, (ImageMedia, VideoMedia, AudioMedia, VideoFrameMedia)):
                if isinstance(frag.value, str):
                    if frag.value in sample:
                        val = conversation_open_medias(
                            ConversationSample.__MEDIA_TYPES_REVERSE__[type(frag)],
                            sample[frag.value]
                        )
                        if frag.metadata is None:
                            # Try to fetch the metadata directly from the primary dataset
                            try:
                                frag.metadata = dataclasses.asdict(primary.get_media_metadata(f".{frag.value}"))
                            except Exception as e:
                                if warn_about_slow_media_loading[primary.get_path()]:
                                    print(f"WARNING: Error getting media metadata for [{primary.get_path()}] {f'.{frag.value}'}: {e}")
                                    warn_about_slow_media_loading[primary.get_path()] = False
                        try:
                            frag.value = cache.to_cache(val, sample['__key__'] + f".{frag.value}")
                        except:
                            # let's try again on the raw bytes
                            if isinstance(sample[frag.value], bytes):
                                frag.value = cache.to_cache(sample[frag.value], sample['__key__'] + f".{frag.value}")
                            else:
                                raise ValueError(f"fragment's value: {val} cannot be cached.")
                    else:
                        # it is a media file outside the primary dataset
                        media_path = frag.value
                        media_dirname = os.path.dirname(media_path)
                        media_basename = os.path.basename(media_path)
                        media_extension = media_basename.rsplit(".", 1)[-1]

                        # get the local or remote media source
                        current_media_source = media_source
                        if current_media_source is None and \
                            media_sources and "aux_data_prefixes" in sample["__subflavors__"]:
                            current_media_source = retrieve_media_source(
                                media_path, media_sources, sample["__subflavors__"]["aux_data_prefixes"],
                            )

                        # check if the auxiliary media dataset is energon prepared
                        media_path_properly_defined_in_metadataset = current_media_source is not None

                        if not media_path_properly_defined_in_metadataset:
                            if os.path.isfile(media_path):
                                print(f"Warning: Media file {media_path} of {primary.get_path()} is an absolute path, loading is slow.")
                                val = media_path
                            else:
                                raise ValueError(f"Cannot find media file {media_path} in {sample}")

                        # process the media file
                        if media_path_properly_defined_in_metadataset:
                            m_path = media_path
                            if os.path.isabs(m_path) and not os.path.isfile(m_path):
                                # if the media path is absolute path and cannot be found locally,
                                # then use the basename and rely on the aux path in the metadataset
                                m_path = media_basename
                            if frag.metadata is None:
                                try:
                                    frag.metadata = dataclasses.asdict(current_media_source.get_media_metadata(m_path))
                                except Exception as e:
                                    if warn_about_slow_media_loading[current_media_source.get_path()]:
                                        print(f"WARNING: Error getting media metadata for [{current_media_source.get_path()}] {m_path}: {e}")
                                        warn_about_slow_media_loading[current_media_source.get_path()] = False
                            if frag.metadata is None:
                                val = cache.get(current_media_source, m_path)
                            frag.value = cache.get_lazy(current_media_source, m_path)
                        else:
                            # load and cache the media file on the fly
                            # warning: slow
                            val_opened = conversation_open_medias(
                                ConversationSample.__MEDIA_TYPES_REVERSE__[type(frag)],
                                val,
                            )
                            try:
                                frag.value = cache.to_cache(val_opened, sample['__key__'] + f".{media_extension}")
                                val = val_opened
                            except:
                                # let's try again on the raw bytes
                                if isinstance(val, str):
                                    val = Path(val).read_bytes()
                                if isinstance(val, bytes):
                                    frag.value = cache.to_cache(val, sample['__key__'] + f".{media_extension}")
                                else:
                                    raise ValueError(f"fragment's value: {val_opened} cannot be cached.")
                else:
                    raise NotImplementedError(f"Postprocessing on media type {type(frag.value)} hasn't been implemented yet.")

                if frag.metadata is None:
                    if isinstance(frag, ImageMedia):
                        frag.metadata = dict(
                            width=val.width, height=val.height, format=val.format, mode=val.mode
                        )
                    elif isinstance(frag, (VideoMedia, AudioMedia, VideoFrameMedia)):
                        frag_val = val
                        if isinstance(frag_val, bytes):
                            frag_val = AVDecoder(io.BytesIO(frag_val))
                        frag.metadata = dataclasses.asdict(frag_val.get_metadata())

            elif isinstance(frag, str):
                # No source
                pass
            else:
                raise ValueError(f"Unknown fragment type: {type(frag)}")
    
    return ConversationSample(
        conversation=messages,
        **basic_sample_keys(sample),
    )
