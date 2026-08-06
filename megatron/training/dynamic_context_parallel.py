"""Compatibility helpers for the pinned multimodal DynamicCP path."""

from argparse import Namespace


def normalize_dynamic_context_parallel_args(args: Namespace) -> Namespace:
    """Map the public DynamicCP flag onto the pinned multimodal scheduler.

    The pinned ``vlm2_rebase_super_vlm`` branch still owns the multimodal
    Energon scheduler under the legacy ``hybrid_context_parallel`` name.  The
    upstream DynamicCP spelling is accepted as a compatibility alias here so
    that callers can use the current option without silently selecting a
    different generic dataloader implementation.
    """

    dynamic = bool(getattr(args, "dynamic_context_parallel", False))
    hybrid = bool(getattr(args, "hybrid_context_parallel", False))
    if dynamic and hybrid:
        raise ValueError(
            "Cannot set both dynamic_context_parallel and hybrid_context_parallel. "
            "Please use dynamic_context_parallel only."
        )
    if dynamic:
        args.hybrid_context_parallel = True
    return args
