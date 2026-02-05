import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _build_param_mapping(vlm_param_name: str) -> str | None:
    """Map VLM parameter names to standalone LLM names.

    Returns the mapped parameter name or ``None`` if the parameter does not
    belong to the language model.
    """
    if not vlm_param_name.startswith("language_model."):
        return None

    if vlm_param_name.startswith("language_model.backbone."):
        # language_model.backbone.*  -> backbone.*
        return vlm_param_name.replace("language_model.", "", 1)

    if vlm_param_name.startswith("language_model.lm_head."):
        # language_model.lm_head.*  -> lm_head.*
        return vlm_param_name.replace("language_model.", "", 1)

    if vlm_param_name == "language_model.lm_head.weight":
        # Special-case exact match
        return "lm_head.weight"

    # Fallback: strip the language_model prefix.
    return vlm_param_name.replace("language_model.", "", 1)


@torch.no_grad()
def extract_lm_weights(input_model_path: str | os.PathLike, output_model_path: str | os.PathLike) -> None:
    """Extract language-model weights from a vision-language model checkpoint.

    Parameters
    ----------
    input_model_path: str or Path
        Path to the *VLM* Hugging Face checkpoint directory.
    output_model_path: str or Path
        Directory where the extracted *LLM* checkpoint will be written. The
        directory is created if it does not exist.
    """

    input_model_path = Path(input_model_path).expanduser().resolve()
    output_model_path = Path(output_model_path).expanduser().resolve()
    output_model_path.mkdir(parents=True, exist_ok=True)

    index_file = input_model_path / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Cannot find index file: {index_file}")

    with open(index_file, "r") as f:
        index_data = json.load(f)

    vlm_weight_map: dict[str, str] = index_data["weight_map"]

    # Build mapping: new_param_name -> original_shard_file
    lm_param_to_shard: dict[str, str] = {}
    original_to_new: dict[str, str] = {}

    for full_name, shard_file in vlm_weight_map.items():
        new_name = _build_param_mapping(full_name)
        if new_name is None:
            # Non-language-model weight (e.g. vision encoder); skip.
            continue
        lm_param_to_shard[new_name] = shard_file
        original_to_new[full_name] = new_name

    if not lm_param_to_shard:
        raise RuntimeError("No language-model parameters were found in the provided VLM checkpoint.")

    print(f"Found {len(lm_param_to_shard)} language-model parameters to extract.")

    # Group parameters by shard to avoid opening the same file repeatedly.
    shard_to_params: dict[str, list[str]] = {}
    for new_name, shard_file in lm_param_to_shard.items():
        shard_to_params.setdefault(shard_file, []).append(new_name)

    new_weight_map: dict[str, str] = {}
    total_size_bytes = 0

    for shard_file, params in shard_to_params.items():
        shard_path = input_model_path / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file listed in index not found: {shard_path}")

        shard_tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for new_name in params:
                # Map back to original VLM tensor name
                original_name = next(k for k, v in original_to_new.items() if v == new_name)
                tensor = f.get_tensor(original_name)
                shard_tensors[new_name] = tensor

        if not shard_tensors:
            # No LM tensors in this shard; skip writing it.
            continue

        dest_shard_path = output_model_path / shard_file
        dest_shard_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {len(shard_tensors)} tensors to {dest_shard_path}")
        save_file(shard_tensors, str(dest_shard_path))

        # Update metadata and weight_map
        total_size_bytes += sum(t.element_size() * t.numel() for t in shard_tensors.values())
        for name in shard_tensors.keys():
            new_weight_map[name] = shard_file

    # Build new index file
    new_index = {
        "metadata": {
            "total_size": total_size_bytes,
        },
        "weight_map": new_weight_map,
    }

    with open(output_model_path / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Try to copy language-model config if it exists inside the VLM checkpoint.
    lm_config_candidate = input_model_path / "language_model" / "config.json"
    if lm_config_candidate.exists():
        shutil.copy2(lm_config_candidate, output_model_path / "config.json")
    else:
        # Fallback: copy root-level config if present.
        root_config = input_model_path / "config.json"
        if root_config.exists():
            shutil.copy2(root_config, output_model_path / "config.json")

    print(f"Extraction complete. LLM checkpoint saved to: {output_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract standalone language-model weights from a vision-language Hugging Face checkpoint",
    )
    parser.add_argument("input_model_path", help="Path to the VLM HF checkpoint directory")
    parser.add_argument("output_model_path", help="Destination directory for the extracted LLM checkpoint")
    args = parser.parse_args()

    extract_lm_weights(args.input_model_path, args.output_model_path)


if __name__ == "__main__":
    main()
