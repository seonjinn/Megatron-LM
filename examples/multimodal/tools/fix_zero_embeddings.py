#!/usr/bin/env python3
"""
Fix zero embeddings in Megatron-LM checkpoints.

This script identifies embedding rows that have zero norm (never trained special tokens)
and initializes them with small random values scaled to match the typical embedding norm.

Checkpoint structure: mp_rank_{tp_rank}_{ep_rank}
- tp_rank: tensor parallel rank
- ep_rank: expert parallel rank

Embedding sharding: The embeddings are COLUMN-SHARDED across TP ranks (hidden dimension split),
not row-sharded (vocabulary split). This means:
- Each TP rank has the full vocabulary (same number of rows)
- Each TP rank has a portion of the hidden dimension (different columns)
- Token IDs are GLOBAL (token ID 10 is at row 10 on ALL TP ranks)

The script processes ALL checkpoint files and uses the same random seed for all ranks
to ensure the initialized values are consistent (when concatenated across TP ranks,
they form a coherent embedding vector).

Usage:
    python examples/multimodal/tools/fix_zero_embeddings.py \
        --checkpoint-dir /path/to/checkpoints/iter_XXXXX \
        --output-dir /path/to/output/iter_XXXXX \
        [--token-ids 10 11 19 20 23 24 25 26]  # Optional: only fix specific token IDs
        [--dry-run]  # Just report, don't modify
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add megatron-lm to path so we can unpickle checkpoint objects
# Script is at examples/multimodal/tools/, so megatron root is 3 levels up
script_dir = Path(__file__).resolve().parent
megatron_root = script_dir.parent.parent.parent
if str(megatron_root) not in sys.path:
    sys.path.insert(0, str(megatron_root))

import torch


def parse_mp_rank_dir(dir_name: str) -> Optional[Tuple[int, int]]:
    """Parse mp_rank_{tp_rank}_{ep_rank} directory name.
    
    Returns (tp_rank, ep_rank) or None if parsing fails.
    """
    match = re.match(r'mp_rank_(\d+)_(\d+)', dir_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def find_zero_embedding_rows(embedding_weight: torch.Tensor, threshold: float = 1e-6) -> List[int]:
    """Find rows in embedding table that have near-zero norm."""
    norms = embedding_weight.norm(dim=-1)
    zero_mask = norms < threshold
    zero_indices = zero_mask.nonzero(as_tuple=True)[0].tolist()
    return zero_indices


def initialize_zero_embeddings(
    embedding_weight: torch.Tensor,
    zero_indices: List[int],
    seed: int = 42,
    tp_rank: int = 0,
) -> torch.Tensor:
    """Initialize zero-norm embedding rows with scaled random values.
    
    The initialization uses the mean norm of non-zero embeddings to scale
    the random values, ensuring they're in a similar range.
    
    Since embeddings are column-sharded across TP ranks (each rank has full vocab
    but only part of hidden dimension), we use tp_rank to adjust the seed so that
    different TP ranks get different random values. This ensures that when the
    embeddings are concatenated across TP ranks, each token gets a unique random
    vector (not duplicated across TP shards).
    """
    if not zero_indices:
        return embedding_weight
    
    # Compute statistics from non-zero embeddings
    norms = embedding_weight.norm(dim=-1)
    non_zero_mask = norms > 1e-6
    
    if non_zero_mask.any():
        mean_norm = norms[non_zero_mask].mean().item()
        std_norm = norms[non_zero_mask].std().item()
    else:
        # Fallback if all embeddings are zero (shouldn't happen)
        mean_norm = 1.0
        std_norm = 0.1
    
    print(f"  Non-zero embedding stats: mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}")
    
    # Initialize with random values scaled to match typical norm
    hidden_size = embedding_weight.shape[1]
    
    # Use a fixed seed for reproducibility, adjusted by tp_rank
    generator = torch.Generator()
    generator.manual_seed(seed + tp_rank * 1000)
    
    # Create random embeddings with unit norm, then scale to mean_norm
    for idx in zero_indices:
        random_emb = torch.randn(hidden_size, generator=generator, dtype=embedding_weight.dtype)
        random_emb = random_emb / random_emb.norm() * mean_norm
        embedding_weight[idx] = random_emb
    
    return embedding_weight


def process_checkpoint_file(
    input_path: str,
    output_path: str,
    token_ids: Optional[List[int]] = None,
    dry_run: bool = False,
    seed: int = 42,
    tp_rank: int = 0,
) -> dict:
    """Process a single checkpoint file and fix zero embeddings.
    
    Returns a dict with statistics about what was found/fixed.
    """
    embedding_key = 'language_model.embedding.word_embeddings.weight'
    
    print(f"\nLoading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    if 'model' not in checkpoint:
        print("  No 'model' key found, skipping")
        return {'skipped': True, 'reason': 'no model key'}
    
    if embedding_key not in checkpoint['model']:
        print(f"  No '{embedding_key}' found, skipping")
        return {'skipped': True, 'reason': 'no embedding key'}
    
    embedding_weight = checkpoint['model'][embedding_key]
    print(f"  Embedding shape: {embedding_weight.shape}")
    
    # Find zero embeddings
    zero_indices = find_zero_embedding_rows(embedding_weight)
    print(f"  Found {len(zero_indices)} zero-norm embedding rows")
    
    if zero_indices:
        print(f"  Zero embedding token IDs (local): {zero_indices[:50]}{'...' if len(zero_indices) > 50 else ''}")
    
    # Filter to specific token IDs if requested
    # Since embeddings are column-sharded (not row-sharded), token_ids are GLOBAL
    # indices that apply directly to all TP ranks
    if token_ids is not None:
        zero_indices = [idx for idx in zero_indices if idx in token_ids]
        print(f"  After filtering to specified token IDs: {len(zero_indices)} rows to fix")
    
    stats = {
        'total_zero': len(find_zero_embedding_rows(embedding_weight)),
        'to_fix': len(zero_indices),
        'fixed_ids': zero_indices,
    }
    
    if not zero_indices:
        print("  No embeddings to fix")
        return stats
    
    if dry_run:
        print("  [DRY RUN] Would fix these embeddings")
        return stats
    
    # Fix the embeddings
    print(f"  Initializing {len(zero_indices)} zero embeddings...")
    embedding_weight = initialize_zero_embeddings(
        embedding_weight, zero_indices, seed=seed, tp_rank=tp_rank
    )
    checkpoint['model'][embedding_key] = embedding_weight
    
    # Verify the fix
    remaining_zeros = find_zero_embedding_rows(embedding_weight)
    remaining_in_fixed = [idx for idx in remaining_zeros if idx in zero_indices]
    if remaining_in_fixed:
        print(f"  WARNING: {len(remaining_in_fixed)} embeddings still zero after fix!")
    else:
        print(f"  Successfully initialized all {len(zero_indices)} embeddings")
    
    # Save the checkpoint
    print(f"  Saving to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    
    stats['fixed'] = len(zero_indices) - len(remaining_in_fixed)
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fix zero embeddings in Megatron-LM checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Path to checkpoint directory (e.g., iter_XXXXX)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--token-ids', type=int, nargs='+', default=None,
                        help='Only fix specific token IDs (default: fix all zero embeddings)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just report what would be fixed, without modifying')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible initialization')
    parser.add_argument('--copy-other-files', action='store_true',
                        help='Copy non-model files (optimizer state, etc.) to output')
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Token IDs to fix: {args.token_ids if args.token_ids else 'all zero embeddings'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Random seed: {args.seed}")
    
    # Find all mp_rank directories
    mp_rank_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('mp_rank_')])
    
    if not mp_rank_dirs:
        raise ValueError(f"No mp_rank_* directories found in {checkpoint_dir}")
    
    print(f"\nFound {len(mp_rank_dirs)} mp_rank directories")
    
    # Parse directory structure: mp_rank_{tp_rank}_{ep_rank}
    # Group by tp_rank
    tp_groups = {}
    for mp_dir in mp_rank_dirs:
        parsed = parse_mp_rank_dir(mp_dir.name)
        if parsed:
            tp_rank, ep_rank = parsed
            if tp_rank not in tp_groups:
                tp_groups[tp_rank] = []
            tp_groups[tp_rank].append((ep_rank, mp_dir))
    
    print(f"Tensor parallel ranks: {sorted(tp_groups.keys())}")
    for tp_rank in sorted(tp_groups.keys()):
        ep_ranks = [ep for ep, _ in tp_groups[tp_rank]]
        print(f"  TP rank {tp_rank}: {len(ep_ranks)} EP ranks ({min(ep_ranks)}-{max(ep_ranks)})")
    
    all_stats = []
    
    # Process all checkpoint files
    # Embeddings are sharded across TP ranks, replicated across EP ranks
    # We need to apply the same fix to all EP ranks for each TP rank
    for tp_rank in sorted(tp_groups.keys()):
        print(f"\n{'='*60}")
        print(f"Processing TP rank {tp_rank}")
        print(f"{'='*60}")
        
        for ep_rank, mp_dir in sorted(tp_groups[tp_rank]):
            input_file = mp_dir / 'model_optim_rng.pt'
            if not input_file.exists():
                print(f"\nSkipping {mp_dir.name}: no model_optim_rng.pt found")
                continue
            
            output_file = output_dir / mp_dir.name / 'model_optim_rng.pt'
            
            stats = process_checkpoint_file(
                str(input_file),
                str(output_file),
                token_ids=args.token_ids,
                dry_run=args.dry_run,
                seed=args.seed,
                tp_rank=tp_rank,
            )
            stats['mp_rank'] = mp_dir.name
            stats['tp_rank'] = tp_rank
            stats['ep_rank'] = ep_rank
            all_stats.append(stats)
    
    # Copy other files if requested
    if args.copy_other_files and not args.dry_run:
        print("\nCopying other checkpoint files...")
        for mp_dir in mp_rank_dirs:
            for file in mp_dir.iterdir():
                if file.name != 'model_optim_rng.pt':
                    output_file = output_dir / mp_dir.name / file.name
                    os.makedirs(output_file.parent, exist_ok=True)
                    if not output_file.exists():
                        shutil.copy2(file, output_file)
                        print(f"  Copied {mp_dir.name}/{file.name}")
        
        # Copy top-level files
        for file in checkpoint_dir.iterdir():
            if file.is_file():
                output_file = output_dir / file.name
                if not output_file.exists():
                    shutil.copy2(file, output_file)
                    print(f"  Copied {file.name}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Group stats by TP rank
    for tp_rank in sorted(tp_groups.keys()):
        tp_stats = [s for s in all_stats if s.get('tp_rank') == tp_rank]
        fixed_count = sum(s.get('fixed', 0) for s in tp_stats)
        to_fix_count = sum(s.get('to_fix', 0) for s in tp_stats)
        skipped_count = sum(1 for s in tp_stats if s.get('skipped'))
        
        print(f"\nTP rank {tp_rank}:")
        print(f"  Files processed: {len(tp_stats) - skipped_count}")
        print(f"  Files skipped: {skipped_count}")
        if to_fix_count > 0:
            # Show stats from first EP rank (they should all be the same)
            first_stats = next((s for s in tp_stats if not s.get('skipped')), None)
            if first_stats:
                print(f"  Zero embeddings per file: {first_stats.get('total_zero', 0)}")
                print(f"  Embeddings to fix per file: {first_stats.get('to_fix', 0)}")
                if first_stats.get('fixed_ids'):
                    print(f"  Fixed token IDs: {first_stats['fixed_ids'][:20]}{'...' if len(first_stats['fixed_ids']) > 20 else ''}")
    
    total_files = len([s for s in all_stats if not s.get('skipped')])
    total_fixed = sum(s.get('fixed', 0) for s in all_stats)
    
    print(f"\nTotal files modified: {total_fixed > 0 and total_files or 0}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply fixes.")


if __name__ == '__main__':
    main()
