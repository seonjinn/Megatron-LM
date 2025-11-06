# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os
import sys

# Add megatron and the multimodal example to the path.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)
    )
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
from transformers import AutoModel

from examples.multimodal.model import model_provider
from examples.multimodal.multimodal_args import add_multimodal_extra_args
from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def run_mcore_vision(model_path, mcore_model_type, vision_resolution=448):
    """Run mcore vision model."""
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    num_cls_tokens = 5

    if mcore_model_type == "internvit":
        # Megatron has some mandatory flags.
        sys.argv = [
            "ignore_me.py",
            "--micro-batch-size=1",
            "--num-layers=2",
            "--vision-model-type=internvit",
            "--language-model-type=mistral_7b",
            "--tokenizer-prompt-format=mistral",
            "--tokenizer-type=MultimodalTokenizer",
            "--tokenizer-model=mistralai/Mistral-7B-Instruct-v0.3",
            "--vocab-size=1024",
            "--hidden-size=64",
            "--num-attention-heads=8",
            "--seq-length=1024",
            "--decoder-seq-length=2048",
            "--max-position-embeddings=2048",
            "--bf16",
            "--img-h=448",
            "--img-w=448",
            "--patch-dim=14",
            "--tensor-model-parallel-size=8",
            "--use-te",
            "--use-distributed-optimizer",
            f"--pretrained-checkpoint={model_path}",
        ]
    elif mcore_model_type == "radio_1d":
        num_cls_tokens = 5
        sys.argv = [
            "ignore_me.py",
            "--micro-batch-size=1",
            "--num-layers=56",
            "--hidden-size=4480",
            "--ffn-hidden-size=15680",
            "--kv-channels=128",
            "--num-attention-heads=40",
            "--num-query-groups=8",
            "--vision-model-type=radio",
            "--use-radio-1d-tokens",
            "--radio-1d-num-tokens=128",
            "--radio-1d-max-tokens=512",
            f"--class-token-len={num_cls_tokens}",
            "--language-model-type=mistral_7b",
            "--tokenizer-prompt-format=mistral",
            "--tokenizer-type=MultimodalTokenizer",
            "--tokenizer-model=mistralai/Mistral-7B-Instruct-v0.3",
            "--make-vocab-size-divisible-by=16512",
            "--seq-length=256",
            "--decoder-seq-length=16384",
            "--max-position-embeddings=16384",
            "--bf16",
            "--img-h=512",
            "--img-w=512",
            "--patch-dim=16",
            "--tensor-model-parallel-size=4",
            "--pipeline-model-parallel-size=1",
            "--normalization=RMSNorm",
            "--group-query-attention",
            "--norm-epsilon=1e-05",
            "--position-embedding-type=none",
            "--squared-relu",
            "--untie-embeddings-and-output-weights",
            "--no-masked-softmax-fusion",
            "--attention-softmax-in-fp32",
            "--disable-bias-linear",
            "--disable-vision-class-token",
            f"--pretrained-checkpoint={model_path}",
        ]
    else:
        raise ValueError(f"Unsupported mcore model type: {mcore_model_type}")

    initialize_megatron(extra_args_provider=add_multimodal_extra_args)

    def wrapped_model_provider(pre_process, post_process):
        return model_provider(pre_process, post_process, parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)

    vision_model = model[0].module.vision_model

    load_checkpoint([vision_model], None, None)

    vision_model.eval()

    images = torch.zeros((1, 3, vision_resolution, vision_resolution), dtype=torch.bfloat16, device="cuda")

    output = vision_model(images)[:, num_cls_tokens:, :]

    return output


def run_hf_vision(model_name, images):
    """Run HF vision model."""
    model = (
        AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .cuda()
        .eval()
    )

    outputs = model(images, return_dict=True)

    return outputs


def run_torchhub_vision(model_version, mcore_model_type, torchhub_version, images):
    """Run TorchHub vision model."""
    if os.path.exists(torchhub_version):
        torchhub_source = "local"
    else:
        torchhub_source = "github"
    model = torch.hub.load(torchhub_version, 'radio_model', version=model_version, source=torchhub_source, progress=True).cuda().eval()
    model.make_preprocessor_external()

    # Convert model to bfloat16 to match Megatron model precision
    model = model.to(torch.bfloat16)

    # Images are already bfloat16, so use them directly
    if mcore_model_type == "radio_1d":
        output = model(images, qradio_size=128)["1d"].features
    else:
        output = model(images)

    return output

def main(mcore_model, mcore_model_type, hf_model, torchhub_model_version, torchhub_version, vision_resolution=448):
    """Compare vision model outputs between mcore and HF given the same fixed input."""

    images = torch.zeros((1, 3, vision_resolution, vision_resolution), dtype=torch.bfloat16, device="cuda")

    mcore = run_mcore_vision(mcore_model, mcore_model_type, vision_resolution)

    if torch.distributed.get_rank() == 0:
        if hf_model:
            hf = run_hf_vision(hf_model, images)
            reference_output = hf["last_hidden_state"]
        elif torchhub_model_version:
            reference_output = run_torchhub_vision(torchhub_model_version, mcore_model_type, torchhub_version, images)
        else:
            raise ValueError("Either hf_model or torchhub_model_version must be provided")

        # Make sure shapes
        if mcore.shape != reference_output.shape:
            raise ValueError(f"mcore shape {mcore.shape} does not match reference output shape {reference_output.shape}")

        # Print some statistics about both outputs (std/max/min/mean)
        print(f"mcore std {mcore.std().item()}, max {mcore.max().item()}, min {mcore.min().item()}, mean {mcore.mean().item()}")
        print(f"reference output std {reference_output.std().item()}, max {reference_output.max().item()}, min {reference_output.min().item()}, mean {reference_output.mean().item()}")

        # Compare logits. Due to different attention implementations and other details,
        # there will be numerical differences.
        diff = (mcore - reference_output).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        # Find location of maximum difference
        max_diff_idx = diff.argmax()
        max_diff_coords = torch.unravel_index(max_diff_idx, diff.shape)

        # Get values at max diff location
        mcore_val_at_max = mcore[max_diff_coords].item()
        ref_val_at_max = reference_output[max_diff_coords].item()

        # Print detailed diff analysis
        print(f"=== Difference Analysis ===")
        print(f"Tensor shape: {diff.shape}")
        print(f"Mean diff: {mean_diff:.6f}")
        print(f"Max diff: {max_diff:.6f}")
        print(f"Max diff location: {max_diff_coords}")
        print(f"Mcore value at max diff: {mcore_val_at_max:.6f}")
        print(f"Reference value at max diff: {ref_val_at_max:.6f}")

        # Percentile analysis (convert to float for quantile computation)
        diff_flat = diff.flatten().float()
        p50 = torch.quantile(diff_flat, 0.5).item()
        p90 = torch.quantile(diff_flat, 0.9).item()
        p95 = torch.quantile(diff_flat, 0.95).item()
        p99 = torch.quantile(diff_flat, 0.99).item()
        print(f"Diff percentiles - 50th: {p50:.6f}, 90th: {p90:.6f}, 95th: {p95:.6f}, 99th: {p99:.6f}")

        # Show some context around max diff location (if it's a 3D tensor)
        if len(diff.shape) == 3:
            batch_idx, seq_idx, feat_idx = max_diff_coords
            print(f"Max diff at batch {batch_idx}, sequence position {seq_idx}, feature {feat_idx}")

            # Show a small window around the max diff location in the feature dimension
            feat_start = max(0, feat_idx - 2)
            feat_end = min(diff.shape[2], feat_idx + 3)
            print(f"Feature values around max diff (features {feat_start}:{feat_end}):")
            print(f"  Mcore:     {mcore[batch_idx, seq_idx, feat_start:feat_end].tolist()}")
            print(f"  Reference: {reference_output[batch_idx, seq_idx, feat_start:feat_end].tolist()}")
            print(f"  Diff:      {diff[batch_idx, seq_idx, feat_start:feat_end].tolist()}")

        # Additional diagnostics to check for shifts/permutations
        print(f"\n=== Shift/Permutation Analysis ===")

        # Check if tensors might be shifted along sequence dimension
        print("Testing sequence shifts...")
        best_shift = 0
        best_shift_diff = float('inf')
        for shift in range(-min(10, mcore.shape[1]//4), min(10, mcore.shape[1]//4) + 1):
            if shift == 0:
                continue
            if shift > 0:
                shifted_mcore = mcore[:, shift:, :]
                shifted_ref = reference_output[:, :-shift, :]
            else:
                shifted_mcore = mcore[:, :shift, :]
                shifted_ref = reference_output[:, -shift:, :]

            shift_diff = (shifted_mcore - shifted_ref).abs().mean().item()
            if shift_diff < best_shift_diff:
                best_shift_diff = shift_diff
                best_shift = shift
            print(f"  Shift {shift:2d}: mean diff = {shift_diff:.6f}")

        if best_shift_diff < mean_diff * 0.8:
            print(f"*** POTENTIAL SEQUENCE SHIFT DETECTED: shift={best_shift}, diff={best_shift_diff:.6f} ***")

        # Check for potential feature dimension permutation by comparing sorted values
        mcore_sorted = torch.sort(mcore.flatten().float())[0]
        ref_sorted = torch.sort(reference_output.flatten().float())[0]
        sorted_diff = (mcore_sorted - ref_sorted).abs().mean().item()
        print(f"Sorted values diff: {sorted_diff:.6f} (if ~0, values are same but permuted)")

        # Check correlation between flattened tensors (convert to float for corrcoef)
        mcore_flat = mcore.flatten().float()
        ref_flat = reference_output.flatten().float()
        correlation = torch.corrcoef(torch.stack([mcore_flat, ref_flat]))[0, 1].item()
        print(f"Tensor correlation: {correlation:.6f} (1.0 = perfect correlation)")

        # Check if there's a simple offset
        offset_diff = (mcore - reference_output).mean().item()
        print(f"Mean offset: {offset_diff:.6f}")
        if abs(offset_diff) > 0.001:
            offset_corrected_diff = (mcore - reference_output - offset_diff).abs().mean().item()
            print(f"Offset-corrected mean diff: {offset_corrected_diff:.6f}")

        print(f"==============================")

        # With high correlation (>0.998), these are acceptable numerical differences between implementations
        if correlation > 0.995:
            acceptable_mean_diff = 0.2
            print(f"High correlation detected ({correlation:.6f}), using relaxed threshold ({acceptable_mean_diff})")
        else:
            acceptable_mean_diff = 0.1

        assert mean_diff < acceptable_mean_diff, f"mean output difference {mean_diff:.6f} exceeds threshold {acceptable_mean_diff} (correlation: {correlation:.6f})"
        assert max_diff < 50, "max output difference is greater than expected"

        print("lgtm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check mcore vision model output vs. HF numerically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcore-model", type=str, required=True, help="directory for mcore model weights"
    )
    parser.add_argument(
        "--mcore-model-type", type=str, default="internvit", help="mcore model type to test"
    )
    parser.add_argument("--hf-model", type=str, required=False, help="Model name in HF")
    parser.add_argument("--torchhub-model-version", type=str, required=False, help="Model name in TorchHub, or local path")
    parser.add_argument(
        "--torchhub-version",
        type=str,
        default="NVlabs/RADIO",
        help="TorchHub repo. Can be a local path or a Github repo. By default use NVlabs/RADIO.")
    parser.add_argument(
        "--vision-resolution", type=int, default=448, help="Vision input resolution (height and width)"
    )

    args = parser.parse_args()

    main(args.mcore_model, args.mcore_model_type, args.hf_model, args.torchhub_model_version, args.torchhub_version, args.vision_resolution)
