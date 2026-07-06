# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
from collections import defaultdict

from megatron.energon import CachePool, FileStore, SourceInfo, basic_sample_keys, cooker, stateless

from ..conversation_base import (
    conversation_convert_message,
    conversation_post_processing,
    conversation_tags_mapping_sample_to_allowed,
)
from ..conversation_sample import (
    AudioMedia,
    ConversationSample,
    ImageMedia,
    Message,
    VideoFrameMedia,
    VideoMedia,
)

warn_about_slow_media_loading = defaultdict(lambda: True)

NO_TOOL_SYSTEM_CONTENT = (
    "<|im_start|>system\n"
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools.<|im_end|>\n"
)
LEGACY_SYSTEM_CONTENT = (
    "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"
)
EMPTY_SYSTEM_CONTENT = "<|im_start|>system\n<|im_end|>\n"


def _openai_message_content_to_fragments(content) -> list[str]:
    """Convert OpenAI-style message content into text fragments."""
    if content is None:
        return [""]
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, str):
                fragments.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type") or part.get("t")
                if part_type in (None, "text"):
                    fragments.append(part.get("text") or part.get("content") or part.get("value") or "")
                else:
                    raise ValueError(
                        "openai_messages_jsonl only supports text content parts, "
                        f"got type={part_type!r}"
                    )
            else:
                raise ValueError(f"Unsupported OpenAI message content part: {type(part)}")
        return fragments
    raise ValueError(f"Unsupported OpenAI message content: {type(content)}")


def _openai_role_to_sender(role: str) -> str:
    if role in ("system", "user", "assistant", "tool"):
        return role
    if role == "function":
        return "tool"
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    raise ValueError(f"Unsupported OpenAI message role: {role!r}")


def _normalize_nano_sft_text_messages(messages: list[dict]) -> list[Message]:
    """Match the text cleanup used by the Nano 3.5 offline SFT packer."""
    conversation = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"OpenAI messages entries must be objects, got {type(msg)}")
        sender = _openai_role_to_sender(msg["role"])
        content = "".join(_openai_message_content_to_fragments(msg.get("content")))
        conversation.append(Message(sender=sender, fragments=[content]))

    if conversation[0].sender != "system":
        first_content = conversation[0].fragments[0]
        if first_content.startswith(EMPTY_SYSTEM_CONTENT):
            conversation[0].fragments[0] = first_content.replace(EMPTY_SYSTEM_CONTENT, "")
        conversation = [Message(sender="system", fragments=[EMPTY_SYSTEM_CONTENT])] + conversation
    elif conversation[0].fragments[0] in (NO_TOOL_SYSTEM_CONTENT, LEGACY_SYSTEM_CONTENT):
        conversation[0].fragments[0] = EMPTY_SYSTEM_CONTENT

    for message in conversation:
        if message.sender == "tool":
            message.sender = "user"

        content = message.fragments[0]
        if (
            message.sender == "user"
            and "<|im_end|>\n<|im_start|>assistant\n<think></think>\n" in content
        ):
            message.fragments[0] = content.replace(
                "<|im_end|>\n<|im_start|>assistant\n<think></think>\n",
                "<|im_end|>\n<|im_start|>assistant\n<think></think>",
            )
        elif message.sender == "assistant":
            message.fragments[0] = content.rstrip() + "\n"

    for idx, message in enumerate(conversation):
        content = message.fragments[0]
        if message.sender == "user" and idx < len(conversation) - 1:
            next_message = conversation[idx + 1]
            if (
                content.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n")
                and next_message.fragments[0].startswith("\n</think>")
            ):
                message.fragments[0] = content.replace(
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    "<|im_end|>\n<|im_start|>assistant\n<think></think>",
                )
                next_message.fragments[0] = next_message.fragments[0][len("\n</think>") :].lstrip()
        elif (
            message.sender == "assistant"
            and idx > 0
            and content.startswith("\n")
            and conversation[idx - 1].fragments[0].endswith("\n")
        ):
            message.fragments[0] = content.lstrip()

    return conversation


def _split_openai_messages_at_system(messages: list[dict]) -> list[list[dict]]:
    """Split an offline-packed Nano SFT row into original conversations."""
    conversations: list[list[dict]] = []
    current: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"OpenAI messages entries must be objects, got {type(message)}")
        if message.get("role") == "system" and current:
            conversations.append(current)
            current = []
        current.append(message)
    if current:
        conversations.append(current)
    return conversations


@stateless
@cooker(need_cache=True)
def cook_openai_messages_jsonl(
    sample: dict,
    cache: CachePool,
    media_source: FileStore | None = None,
) -> ConversationSample:
    """Load OpenAI-style JSONL rows with messages[*].role/content."""
    data = sample["json"]
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("openai_messages_jsonl requires a non-empty messages list")

    conversation = _normalize_nano_sft_text_messages(messages)
    for idx, msg in enumerate(conversation):
        sender = msg.sender
        if sender == "system" and idx > 0:
            raise ValueError(
                "openai_messages_jsonl only supports a leading system message. "
                "Split rows with repeated system prompts before using this cooker."
            )

    return ConversationSample(conversation=conversation, **basic_sample_keys(sample))


@stateless
@cooker(need_cache=True)
def cook_openai_messages_offline_packed_jsonl(
    sample: dict,
    cache: CachePool,
    media_source: FileStore | None = None,
) -> ConversationSample:
    """Load Nano-style offline-packed JSONL rows with merged ``messages``.

    Each row is one already-packed training item. The row may contain multiple
    conversations concatenated together, separated by repeated ``system`` turns.
    Unlike ``openai_messages_jsonl``, this cooker preserves those repeated
    systems so the task encoder can tokenize each original conversation
    separately and emit packed ``cu_lengths`` without running online packing.
    """
    data = sample["json"]
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("openai_messages_offline_packed_jsonl requires a non-empty messages list")

    conversation: list[Message] = []
    for split_messages in _split_openai_messages_at_system(messages):
        conversation.extend(_normalize_nano_sft_text_messages(split_messages))

    if not conversation:
        raise ValueError("openai_messages_offline_packed_jsonl produced an empty conversation")

    sample_keys = basic_sample_keys(sample)
    subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
    subflavors["offline_packed_messages"] = True
    sample_keys["__subflavors__"] = subflavors
    return ConversationSample(conversation=conversation, **sample_keys)


def _basic_sample_keys_with_json_dataset(sample: dict, data: dict) -> dict:
    """Preserve optional raw JSONL dataset metadata for task-encoder filters."""
    sample_keys = basic_sample_keys(sample)
    dataset_name = data.get("dataset")
    if dataset_name is not None:
        subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
        subflavors["dataset"] = dataset_name
        sample_keys["__subflavors__"] = subflavors
    return sample_keys


@stateless
@cooker(need_cache=True)
def cook_conversation(
    sample: dict,
    cache: CachePool,
    media_source: FileStore | None = None,
) -> ConversationSample:
    global warn_about_slow_media_loading

    data = sample["json"]
    cs = ConversationSample.from_json(data, **_basic_sample_keys_with_json_dataset(sample, data))

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
