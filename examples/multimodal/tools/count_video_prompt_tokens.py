#!/usr/bin/env python3
"""
Calculate the number of text tokens generated from video prompts.

This script generates video prompts (as defined in task_encoder.py) and counts
the number of tokens using the exact same tokenization as the training code.

Must be run within a megatron-supported environment.

Usage:
    python count_video_prompt_tokens.py --num-frames 32 --video-prompt-version 1
    python count_video_prompt_tokens.py --num-frames 64 --video-prompt-version 2 --temporal-patch-size 4
"""

import argparse
import sys
import numpy as np
from typing import Tuple, List

from megatron.training.tokenizer.tokenizer import build_tokenizer


# Default tokenizer path
DEFAULT_TOKENIZER_MODEL = (
    "/lustre/fsw/portfolios/llmservice/users/amalasanjayd/checkpoints/"
    "nano-v2-sft-lr5e-6-128k-nollama-thinkfix-ep2/checkpoints/"
    "nano-v2-sft-lr5e-6-128k-nollama-thinkfix-ep2/iter_0006000/"
)

# Default settings from sft_nemotron_5p5_hybrid_9b_radio_v4_so400m_rc3.sh
DEFAULT_PROMPT_FORMAT = "nemotron-h-5p5-reasoning"
DEFAULT_SPECIAL_TOKENS = [
    "<image>", "<img>", "</img>", "<quad>", "</quad>", "<ref>", "</ref>", "<box>", "</box>"
]
DEFAULT_IMAGE_TAG_TYPE = "internvl"


class TokenizerArgs:
    """
    Mock args object for build_tokenizer with the required attributes.
    Mimics the args structure expected by megatron.training.tokenizer.tokenizer.build_tokenizer.

    Default values are from sft_nemotron_5p5_hybrid_9b_radio_v4_so400m_rc3.sh
    """
    def __init__(
        self,
        tokenizer_model: str,
        tokenizer_prompt_format: str = DEFAULT_PROMPT_FORMAT,
        special_tokens: List[str] = None,
        image_tag_type: str = DEFAULT_IMAGE_TAG_TYPE,
        force_system_message: bool = False,
        make_vocab_size_divisible_by: int = 16512,  # From shell script line 322
        tensor_model_parallel_size: int = 4,  # TP=4 from shell script line 139
    ):
        # Required for build_tokenizer (MultimodalTokenizer case)
        self.tokenizer_type = "MultimodalTokenizer"
        self.tokenizer_model = tokenizer_model
        self.tokenizer_prompt_format = tokenizer_prompt_format
        self.special_tokens = special_tokens if special_tokens is not None else DEFAULT_SPECIAL_TOKENS
        self.image_tag_type = image_tag_type
        self.force_system_message = force_system_message

        # Required for _vocab_size_with_padding (from shell script)
        # --make-vocab-size-divisible-by 16512 (line 322)
        # --tensor-model-parallel-size ${TP} where TP=4 (lines 139, 337)
        self.padded_vocab_size = None
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.tensor_model_parallel_size = tensor_model_parallel_size

        # For print statement in build_tokenizer
        self.rank = 0


def generate_video_prompt_v1(
    frame_timestamps: np.ndarray,
    temporal_patch_size: int = 1,
) -> Tuple[str, int]:
    """
    Generate video prompt version 1: Each frame on its own line.

    Matches the logic in task_encoder.py video_to_frames() for video_prompt_version == 1.

    Returns:
        Tuple of (prompt_text, num_image_tokens)
    """
    lines = ["This is a video:\n"]
    num_image_tokens = 0

    for i, timestamp in enumerate(frame_timestamps):
        lines.append(f"Frame {i + 1} sampled at {timestamp:.2f} seconds: ")

        # IMAGE_TOKEN (<image>) is added when (sample_index + 1) % temporal_patch_size == 0
        # in preencode_sample(), but we add it here for counting purposes
        if (i + 1) % temporal_patch_size == 0:
            num_image_tokens += 1
            lines.append("<image>")

        lines.append("\n")

    return "".join(lines), num_image_tokens


def generate_video_prompt_v2(
    frame_timestamps: np.ndarray,
    temporal_patch_size: int,
) -> Tuple[str, int]:
    """
    Generate video prompt version 2: Group T frames with "and", one <image> per group.

    Matches the logic in task_encoder.py video_to_frames() for video_prompt_version == 2.

    Returns:
        Tuple of (prompt_text, num_image_tokens)
    """
    lines = ["This is a video:\n"]
    num_image_tokens = 0
    T = temporal_patch_size

    for group_start in range(0, len(frame_timestamps), T):
        group_text_parts = []
        for j in range(T):
            sample_idx = group_start + j
            if sample_idx < len(frame_timestamps):
                timestamp = frame_timestamps[sample_idx]
                frame_str = "Frame" if j == 0 else "frame"
                group_text_parts.append(f"{frame_str} {sample_idx + 1} sampled at {timestamp:.2f} seconds")

        if group_text_parts:
            lines.append(" and ".join(group_text_parts) + ": ")
            num_image_tokens += 1
            lines.append("<image>")
            lines.append("\n")

    return "".join(lines), num_image_tokens


def generate_video_prompt_v3(
    frame_timestamps: np.ndarray,
    temporal_patch_size: int,
) -> Tuple[str, int]:
    """
    Generate video prompt version 3: Compact format with just frame ranges.

    Placeholder for future compact prompt format.

    Returns:
        Tuple of (prompt_text, num_image_tokens)
    """
    lines = ["Video content:\n"]
    num_image_tokens = 0
    T = temporal_patch_size

    for group_start in range(0, len(frame_timestamps), T):
        group_end = min(group_start + T - 1, len(frame_timestamps) - 1)
        start_time = frame_timestamps[group_start]
        end_time = frame_timestamps[group_end]

        if group_start == group_end:
            lines.append(f"[{start_time:.2f}s]: ")
        else:
            lines.append(f"[{start_time:.2f}s-{end_time:.2f}s]: ")

        num_image_tokens += 1
        lines.append("<image>")
        lines.append("\n")

    return "".join(lines), num_image_tokens


def get_seq_frames(video_duration: float, num_frames: int, jitter: bool = False) -> np.ndarray:
    """
    Generate evenly spaced frame timestamps (simplified version of get_seq_frames_v3).

    Args:
        video_duration: Total duration of the video in seconds.
        num_frames: Number of frames to sample.
        jitter: Whether to add temporal jitter (not implemented here for simplicity).

    Returns:
        Array of frame timestamps in seconds.
    """
    if num_frames == 1:
        return np.array([video_duration / 2])

    # Evenly spaced timestamps
    timestamps = np.linspace(0, video_duration, num_frames, endpoint=False)
    # Offset to center of each segment
    segment_duration = video_duration / num_frames
    timestamps = timestamps + segment_duration / 2

    return timestamps


def build_conversation_for_tokenization(prompt_text: str) -> List[dict]:
    """
    Build a conversation in the format expected by tokenize_conversation.

    This mimics how task_encoder.py builds conversations for tokenization.
    """
    # Simple user/assistant conversation with the video prompt as user content
    conversation = [
        {"role": "user", "content": prompt_text + "Describe this video."},
        {"role": "assistant", "content": "This is a test response."},
    ]
    return conversation


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the number of text tokens generated from video prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test version 1 with 32 frames
    python count_video_prompt_tokens.py --num-frames 32 --video-prompt-version 1

    # Test version 2 with temporal compression
    python count_video_prompt_tokens.py --num-frames 64 --video-prompt-version 2 --temporal-patch-size 4

    # Compare all versions
    python count_video_prompt_tokens.py --num-frames 32 --compare-all

    # Test with custom tokenizer
    python count_video_prompt_tokens.py --num-frames 32 --tokenizer-model /path/to/tokenizer
        """
    )

    # Video parameters
    parser.add_argument(
        "--num-frames", type=int, default=32,
        help="Number of frames sampled from the video (default: 32)"
    )
    parser.add_argument(
        "--video-duration", type=float, default=10.0,
        help="Video duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--temporal-patch-size", type=int, default=1,
        help="Temporal patch size for tubelet compression (default: 1)"
    )

    # Prompt version
    parser.add_argument(
        "--video-prompt-version", type=int, default=1, choices=[1, 2, 3],
        help="Video prompt version to use (default: 1)"
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Compare all prompt versions"
    )

    # Tokenizer settings (defaults from sft_nemotron_5p5_hybrid_9b_radio_v4_so400m_rc3.sh)
    parser.add_argument(
        "--tokenizer-model", type=str, default=DEFAULT_TOKENIZER_MODEL,
        help=f"Path to tokenizer model (default: {DEFAULT_TOKENIZER_MODEL})"
    )
    parser.add_argument(
        "--tokenizer-prompt-format", type=str, default=DEFAULT_PROMPT_FORMAT,
        help=f"Tokenizer prompt format (default: {DEFAULT_PROMPT_FORMAT})"
    )
    parser.add_argument(
        "--image-tag-type", type=str, default=DEFAULT_IMAGE_TAG_TYPE,
        choices=["nvlm", "internvl", ""],
        help=f"Image tag type (default: {DEFAULT_IMAGE_TAG_TYPE})"
    )
    parser.add_argument(
        "--make-vocab-size-divisible-by", type=int, default=16512,
        help="Make vocab size divisible by this value (default: 16512 from shell script)"
    )
    parser.add_argument(
        "--tensor-model-parallel-size", type=int, default=4,
        help="Tensor model parallel size for vocab padding (default: 4 from shell script TP=4)"
    )

    # Output options
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the generated prompt text"
    )
    parser.add_argument(
        "--show-tokens", action="store_true",
        help="Print the individual tokens"
    )
    parser.add_argument(
        "--show-conversation", action="store_true",
        help="Print the full tokenized conversation"
    )

    args = parser.parse_args()

    # Validate temporal patch size
    if args.temporal_patch_size > 1:
        if args.num_frames % args.temporal_patch_size != 0:
            adjusted = (args.num_frames // args.temporal_patch_size) * args.temporal_patch_size
            print(f"Warning: num_frames ({args.num_frames}) is not divisible by "
                  f"temporal_patch_size ({args.temporal_patch_size}). "
                  f"Adjusting to {adjusted}.")
            args.num_frames = adjusted

    # Generate frame timestamps
    frame_timestamps = get_seq_frames(args.video_duration, args.num_frames)

    # Build tokenizer using the exact same function as megatron training
    print(f"Loading tokenizer from: {args.tokenizer_model}")
    print(f"  Prompt format: {args.tokenizer_prompt_format}")
    print(f"  Image tag type: {args.image_tag_type}")
    print(f"  Make vocab size divisible by: {args.make_vocab_size_divisible_by}")
    print(f"  Tensor model parallel size: {args.tensor_model_parallel_size}")

    tokenizer_args = TokenizerArgs(
        tokenizer_model=args.tokenizer_model,
        tokenizer_prompt_format=args.tokenizer_prompt_format,
        image_tag_type=args.image_tag_type,
        make_vocab_size_divisible_by=args.make_vocab_size_divisible_by,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
    )
    tokenizer = build_tokenizer(tokenizer_args)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"Padded vocab size: {tokenizer_args.padded_vocab_size}")
    print()

    # Prompt generators
    generators = {
        1: ("Version 1 (frame per line)", generate_video_prompt_v1),
        2: ("Version 2 (grouped frames)", generate_video_prompt_v2),
        3: ("Version 3 (compact time ranges)", generate_video_prompt_v3),
    }

    # Determine which versions to test
    versions_to_test = list(generators.keys()) if args.compare_all else [args.video_prompt_version]

    print("=" * 70)
    print(f"Video Parameters:")
    print(f"  Duration: {args.video_duration:.1f}s")
    print(f"  Num frames: {args.num_frames}")
    print(f"  Temporal patch size: {args.temporal_patch_size}")
    print(f"  Effective tubelets: {args.num_frames // args.temporal_patch_size}")
    print("=" * 70)
    print()

    results = []

    for version in versions_to_test:
        name, generator = generators[version]
        prompt_text, num_image_tokens = generator(frame_timestamps, args.temporal_patch_size)

        # Build conversation and tokenize using the same method as task_encoder.py
        conversation = build_conversation_for_tokenization(prompt_text)

        # Tokenize conversation (same as task_encoder.py line 648-650)
        input_ids, target = tokenizer.tokenize_conversation(
            conversation, True, False, train_only_on_last_assistant_turn=False
        )

        # Count tokens in just the video prompt portion
        # Tokenize just the prompt text for comparison
        prompt_only_conversation = [
            {"role": "user", "content": prompt_text},
        ]
        prompt_tokens, _ = tokenizer.tokenize_conversation(
            prompt_only_conversation, True, False
        )

        # The prompt tokens include the user header and formatting
        # Let's also count raw prompt tokens without conversation formatting
        raw_prompt_tokens = tokenizer._tokenizer.encode(prompt_text, add_special_tokens=False)

        results.append({
            "version": version,
            "name": name,
            "prompt_text": prompt_text,
            "raw_prompt_tokens": len(raw_prompt_tokens),
            "conversation_tokens": len(input_ids),
            "prompt_in_conversation_tokens": len(prompt_tokens),
            "image_tokens": num_image_tokens,
        })

        print(f"{'=' * 70}")
        print(f"Prompt {name}")
        print(f"{'=' * 70}")

        if args.show_prompt:
            print("\nGenerated prompt:")
            print("-" * 40)
            print(prompt_text)
            print("-" * 40)

        if args.show_tokens:
            print("\nRaw prompt tokens (without conversation formatting):")
            print("-" * 40)
            for i, tok_id in enumerate(raw_prompt_tokens[:100]):  # Limit to first 100
                tok_str = tokenizer.detokenize([tok_id])
                print(f"  {i:4d}: {tok_id:6d} -> {repr(tok_str)}")
            if len(raw_prompt_tokens) > 100:
                print(f"  ... ({len(raw_prompt_tokens) - 100} more tokens)")
            print("-" * 40)

        if args.show_conversation:
            print("\nFull conversation tokens:")
            print("-" * 40)
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Decoded: {tokenizer.detokenize(input_ids.tolist())}")
            print("-" * 40)

        print(f"\nResults:")
        print(f"  Raw prompt tokens (no formatting): {len(raw_prompt_tokens)}")
        print(f"  Prompt in conversation (with user header): {len(prompt_tokens)}")
        print(f"  Full conversation tokens: {len(input_ids)}")
        print(f"  Image token positions: {num_image_tokens}")
        print(f"  Characters in prompt: {len(prompt_text)}")
        print()

    # Summary comparison if multiple versions
    if len(results) > 1:
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Version':<35} {'Raw Tokens':>11} {'Conv Tokens':>12} {'Images':>7}")
        print("-" * 70)
        for r in results:
            print(f"{r['name']:<35} {r['raw_prompt_tokens']:>11} {r['prompt_in_conversation_tokens']:>12} {r['image_tokens']:>7}")
        print("-" * 70)

        # Show savings
        if len(results) >= 2:
            baseline = results[0]["raw_prompt_tokens"]
            print("\nToken savings vs Version 1 (raw prompt tokens):")
            for r in results[1:]:
                saved = baseline - r["raw_prompt_tokens"]
                pct = (saved / baseline * 100) if baseline > 0 else 0
                print(f"  {r['name']}: {saved:+d} tokens ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
