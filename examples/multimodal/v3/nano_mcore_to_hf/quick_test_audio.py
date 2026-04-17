# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick test script for audio inference with NVIDIA Nemotron Nano VL model.

This script demonstrates how to use the model for audio understanding tasks.

Usage:
    python quick_test_audio.py --model_path /path/to/model --audio_path /path/to/audio.wav

The audio file should be a WAV file sampled at 16kHz. If your audio is in a different
format or sample rate, it will be automatically resampled.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def load_model(model_path: str, device: str = "cuda:0"):
    """Load the VLM model and processor.

    Args:
        model_path: Path to the pretrained model
        device: Device to load the model on

    Returns:
        Tuple of (model, tokenizer, processor)
    """
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
        dtype=torch.bfloat16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully!")
    return model, tokenizer, processor


def test_audio_transcription(
    model,
    tokenizer,
    processor,
    audio_path: str,
    prompt_text: str = "Transcribe the audio.",
    device: str = "cuda:0",
    max_new_tokens: int = 1024,
    do_sample: bool = False,
):
    """Test model inference on an audio file.

    Args:
        model: The VLM model with audio support
        tokenizer: The tokenizer
        processor: The processor
        audio_path: Path to the audio file
        prompt_text: Text prompt for the model
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling for generation
    """
    print(f"\nProcessing audio: {audio_path}")

    # Prepare messages with audio token embedded directly in text.
    # The chat template does not expand {"type": "audio"} content blocks into
    # <so_embedding> tokens — it just stringifies the list. So we place the
    # audio placeholder token in the user message text and let the processor
    # expand it to the correct number of repeated tokens.
    audio_token = getattr(tokenizer, "audio_token", "<so_embedding>")
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": f"{audio_token}\n{prompt_text}"},
    ]

    # Generate prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs - processor handles audio loading and token expansion
    inputs = processor(
        text=[prompt],
        audio=[audio_path],
        return_tensors="pt",
    )

    # Get sound clips before moving to device (they're numpy arrays, not tensors)
    sound_clips = inputs.pop("sound_clips", None)

    # Move tensor inputs to device
    inputs = inputs.to(device)

    print(f"Input ids shape: {inputs.input_ids.shape}")

    if sound_clips is not None:
        if isinstance(sound_clips, list):
            print(f"Sound clips: {len(sound_clips)} clips, first clip length: {len(sound_clips[0])} samples")
        else:
            print(f"Sound clips: {type(sound_clips)}")

    # Generate output - model handles feature extraction from raw waveforms
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        sound_clips=sound_clips,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode output — trim to only generated tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"\n{'='*50}")
    print(f"Prompt: {prompt_text}")
    print(f"{'='*50}")
    print(f"Output: {output_text}\n")


def test_audio_understanding(
    model,
    tokenizer,
    processor,
    audio_path: str,
    device: str = "cuda:0",
    max_new_tokens: int = 1024,
):
    """Test various audio understanding prompts.

    Args:
        model: The VLM model with audio support
        tokenizer: The tokenizer
        processor: The processor
        audio_path: Path to the audio file
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
    """
    prompts = [
        "Transcribe the audio.",
        "What is being said in this audio?",
        "Describe the audio content.",
    ]

    for prompt in prompts:
        test_audio_transcription(
            model, tokenizer, processor, audio_path,
            prompt_text=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )


def main():
    parser = argparse.ArgumentParser(description="Test audio inference with VLM model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file (WAV format, preferably 16kHz)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for audio understanding (default: runs multiple prompts)"
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer, processor = load_model(args.model_path, args.device)

    print("=" * 50)
    print("Testing Audio Inference")
    print("=" * 50)

    if args.prompt:
        # Single custom prompt
        test_audio_transcription(
            model, tokenizer, processor,
            args.audio_path,
            prompt_text=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens
        )
    else:
        # Multiple prompts
        test_audio_understanding(
            model, tokenizer, processor,
            args.audio_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens
        )


if __name__ == "__main__":
    main()
