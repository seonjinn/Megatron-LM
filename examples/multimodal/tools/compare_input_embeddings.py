#!/usr/bin/env python3
# Compare HF input embeddings vs Megatron TP/CP checkpoint embeddings.

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM


MP_RANK_RE = re.compile(r"^mp_rank_(\d+)(?:_(\d+))?$")


def find_mp_rank_dirs(iter_dir: Path) -> Dict[int, List[Tuple[int, Path]]]:
    """Return mapping tp_rank -> list of (cp_rank, path)."""
    tp_map: Dict[int, List[Tuple[int, Path]]] = {}
    for name in os.listdir(iter_dir):
        match = MP_RANK_RE.match(name)
        if not match:
            continue
        tp_rank = int(match.group(1))
        cp_rank = int(match.group(2)) if match.group(2) is not None else 0
        tp_map.setdefault(tp_rank, []).append((cp_rank, iter_dir / name))
    return tp_map


def pick_cp_ranks(tp_map: Dict[int, List[Tuple[int, Path]]]) -> Dict[int, Path]:
    """Pick the smallest cp rank per tp rank."""
    selected = {}
    for tp_rank, entries in tp_map.items():
        cp_rank, path = sorted(entries, key=lambda x: x[0])[0]
        selected[tp_rank] = path
        print(f"Using tp_rank={tp_rank}, cp_rank={cp_rank}: {path}")
    return selected


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    ckpt_path = path / "model_optim_rng.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in obj:
        raise KeyError(f"'model' key missing in {ckpt_path}")
    return obj["model"]


def find_embedding_key(state: Dict[str, torch.Tensor], hidden_size: int) -> str:
    candidates = []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dim() != 2:
            continue
        if v.shape[1] != hidden_size:
            continue
        if k.endswith("word_embeddings.weight") or k.endswith("embedding.weight"):
            candidates.append(k)
        elif "word_embeddings" in k and k.endswith("weight"):
            candidates.append(k)
    if not candidates:
        raise KeyError("Could not find embedding weight in checkpoint state dict.")
    # Prefer language_model.* if present.
    for k in candidates:
        if k.startswith("language_model."):
            return k
    return candidates[0]


def concat_tp_embeddings(tp_paths: Dict[int, Path], hidden_size: int) -> torch.Tensor:
    shards = []
    embedding_key = None
    for tp_rank in sorted(tp_paths.keys()):
        state = load_state_dict(tp_paths[tp_rank])
        if embedding_key is None:
            embedding_key = find_embedding_key(state, hidden_size)
            print(f"Using embedding key: {embedding_key}")
        if embedding_key not in state:
            raise KeyError(f"Missing embedding key {embedding_key} in {tp_paths[tp_rank]}")
        shard = state[embedding_key]
        if shard.shape[1] != hidden_size:
            raise ValueError(
                f"Hidden size mismatch for tp_rank={tp_rank}: "
                f"{shard.shape[1]} vs {hidden_size}"
            )
        shards.append(shard)
    return torch.cat(shards, dim=0)


def compare_embeddings(hf_emb: torch.Tensor, mg_emb: torch.Tensor) -> None:
    hf_vocab, hidden = hf_emb.shape
    mg_vocab = mg_emb.shape[0]
    trim_vocab = min(hf_vocab, mg_vocab)
    if mg_vocab != hf_vocab:
        print(f"Vocab size differs: hf={hf_vocab} megatron={mg_vocab} (trim={trim_vocab})")

    hf_cmp = hf_emb[:trim_vocab].float()
    mg_cmp = mg_emb[:trim_vocab].float()

    diff = hf_cmp - mg_cmp
    max_abs = diff.abs().max().item() if diff.numel() else 0.0
    mean_abs = diff.abs().mean().item() if diff.numel() else 0.0
    rms = diff.pow(2).mean().sqrt().item() if diff.numel() else 0.0

    print(f"Hidden size: {hidden}")
    print(f"Compared vocab size: {trim_vocab}")
    print(f"Max abs diff: {max_abs:.6g}")
    print(f"Mean abs diff: {mean_abs:.6g}")
    print(f"RMS diff: {rms:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF input embeddings vs Megatron TP/CP checkpoint embeddings."
    )
    parser.add_argument(
        "--hf-model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--iter-dir",
        required=True,
        help="Path to Megatron iter_XXXX directory with mp_rank_* subdirs.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="dtype used to load HF model on CPU.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loader.",
    )
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"Loading HF model: {args.hf_model}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_emb = hf_model.get_input_embeddings().weight.detach().cpu()
    print(f"HF embeddings shape: {tuple(hf_emb.shape)}")

    iter_dir = Path(args.iter_dir)
    tp_map = find_mp_rank_dirs(iter_dir)
    if not tp_map:
        raise RuntimeError(f"No mp_rank_* directories found in {iter_dir}")
    tp_paths = pick_cp_ranks(tp_map)

    mg_emb = concat_tp_embeddings(tp_paths, hf_emb.shape[1])
    print(f"Megatron embeddings shape (concat TP): {tuple(mg_emb.shape)}")

    compare_embeddings(hf_emb, mg_emb)


if __name__ == "__main__":
    main()
