#!/usr/bin/env python3
"""Initialize and load the multimodal model without constructing datasets."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MULTIMODAL_DIR = REPO_ROOT / "examples" / "multimodal"
sys.path.insert(0, str(MULTIMODAL_DIR))
sys.path.insert(0, str(REPO_ROOT))

from multimodal_args import add_multimodal_extra_args  # noqa: E402
from train import (  # noqa: E402
    forward_step,
    llava_embedding_ranks,
    llava_position_embedding_ranks,
    model_provider,
    run_online_eval,
    write_online_eval_to_tensorboard,
)

from megatron.core.enums import ModelType  # noqa: E402
from megatron.training import pretrain  # noqa: E402
from megatron.training.argument_utils import pretrain_cfg_container_from_args  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402


def no_data_provider(_train_valid_test_num_samples):
    """Return no datasets; model initialization and checkpoint loading run first."""
    return None, None, None


no_data_provider.is_distributed = True


def main() -> None:
    args = parse_and_validate_args(
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_multimodal_extra_args,
    )

    # The production recipe uses --train-full-dataset, which would inspect the
    # real Energon recipe before model construction. Override it only in this
    # dedicated smoke-test entrypoint so setup proceeds through model creation
    # and checkpoint loading, then exits cleanly with no optimizer or data.
    args.train_full_dataset = False
    args.train_iters = 0
    args.train_samples = None
    args.skip_train = True
    args.eval_iters = 0
    args.full_validation = False

    full_config = pretrain_cfg_container_from_args(args)
    pretrain(
        full_config,
        no_data_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
        non_loss_data_func=run_online_eval,
    )


if __name__ == "__main__":
    main()
