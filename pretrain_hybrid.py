# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain and SFT Hybrid."""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import Any, List, Optional, Tuple

import torch

from hybrid_builders import hybrid_builder
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import (
    PackedSeqParams,
    get_thd_padding_kwargs,
    pad_sequence_for_thd,
    packed_thd_exceeds_capacity,
    resolve_thd_tail_padding_policy,
)
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import (
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset
from megatron.training.training import (
    update_packed_sequence_stats,
    update_seqlen_stats_from_cu_seqlens,
)
from megatron.training.utils import get_blend_and_blend_per_split, is_first_or_last_pipeline_stage
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

_PreparedPackedTHDBatch = Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    PackedSeqParams,
    Optional[torch.Tensor],
]


def _sample_lengths_for_packed_stats(batch: dict[str, torch.Tensor | None]) -> torch.Tensor | None:
    sample_lengths = batch.get("sample_lengths")
    if sample_lengths is not None:
        return sample_lengths

    cu_seqlens = batch.get("cu_seqlens")
    if cu_seqlens is None:
        return None
    if cu_seqlens.dim() == 1:
        return cu_seqlens[1:] - cu_seqlens[:-1]
    return cu_seqlens[:, 1:] - cu_seqlens[:, :-1]


def _prepare_packed_thd_batch(
    tokens: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor],
    max_seqlen: torch.Tensor,
    local_cp_size: Optional[torch.Tensor],
    hybrid_cp_group: Optional[torch.distributed.ProcessGroup],
    config: ModelParallelConfig,
    pad_token_id: int,
    context_parallel_size: int,
) -> _PreparedPackedTHDBatch:
    """Build and optionally pad the external packed-THD batch boundary."""
    if context_parallel_size != 1:
        raise ValueError(
            "External packed-THD padding with context parallelism requires DynamicCP "
            "integration, which is a separate follow-up."
        )

    physical_cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
    token_like_tensors = (tokens, labels, loss_mask, position_ids)
    total_tokens = next(
        (int(tensor.shape[-1]) for tensor in token_like_tensors if tensor is not None),
        int(physical_cu_seqlens[-1].item()),
    )
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen.item()),
        max_seqlen_kv=int(max_seqlen.item()),
        local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
        cp_group=hybrid_cp_group,
        total_tokens=total_tokens,
        tokens_per_sample=total_tokens,
    )

    pad_packed_seq_alignment = config.pad_packed_seq_alignment
    if pad_packed_seq_alignment is None:
        return (tokens, labels, loss_mask, position_ids, packed_seq_params, None)

    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        pad_packed_seq_alignment,
        config.max_seqlen_per_dp_cp_rank,
        getattr(config, "thd_max_packed_sequences", None),
        getattr(config, "cuda_graph_impl", "none") != "none",
    )
    if (
        getattr(config, "cuda_graph_impl", "none") != "none"
        and getattr(config, "thd_overflow_policy", "error") == "eager"
        and packed_thd_exceeds_capacity(
            packed_seq_params,
            token_like_tensors,
            target_len,
            max_num_seqs,
        )
    ):
        return (tokens, labels, loss_mask, position_ids, packed_seq_params, None)
    (
        padded_tokens,
        padded_labels,
        padded_loss_mask,
        padded_position_ids,
        padded_seq_params,
        padding_mask,
    ) = pad_sequence_for_thd(
        tokens,
        labels,
        loss_mask,
        position_ids,
        packed_seq_params,
        alignment=alignment,
        target_len=target_len,
        max_num_seqs=max_num_seqs,
        tail_padding_policy=resolve_thd_tail_padding_policy(config),
        cp_size=context_parallel_size,
    )

    if padded_tokens is not None:
        padded_tokens = padded_tokens.masked_fill(padding_mask, pad_token_id)
    if padded_labels is not None:
        padded_labels = padded_labels.masked_fill(padding_mask, IGNORE_INDEX)
    if padded_loss_mask is not None:
        padded_loss_mask = padded_loss_mask.masked_fill(padding_mask, 0)
    padded_seq_params.tokens_per_sample = padded_seq_params.total_tokens

    return (
        padded_tokens,
        padded_labels,
        padded_loss_mask,
        padded_position_ids,
        padded_seq_params,
        padding_mask,
    )


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""

    BATCH_KEYS = ["attention_mask", "cu_seqlens", "cu_seqlens_padded", "hybrid_cp_group", "labels", "local_cp_size", "loss_mask", "max_seqlen", "position_ids", "tokens"]
    DATALOADER_BATCH_KEYS = BATCH_KEYS + ["sample_lengths"]

    args = get_args()
    config = core_transformer_config_from_args(args)

    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_sft = args.sft
    create_attention_mask_in_dataloader = args.create_attention_mask_in_dataloader
    mtp_on_this_rank = mtp_on_this_rank_func(layout=config.pipeline_model_parallel_layout, mtp_num_layers=config.mtp_num_layers, ignore_virtual=False, vp_stage=vp_stage)
    is_hybrid_cp = args.hybrid_context_parallel

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank and not is_sft:
        return [None for _ in BATCH_KEYS]

    batch = {}
    if tp_rank == 0:
        batch = next(data_iterator)
        for key in DATALOADER_BATCH_KEYS:
            batch[key] = batch[key].cuda(non_blocking=True) if key in batch and batch[key] is not None else None
        if is_sft and getattr(args, "log_packed_sequence_stats", False):
            update_packed_sequence_stats(
                _sample_lengths_for_packed_stats(batch),
                batch.get("loss_mask"),
            )
        batch = {key: batch[key] for key in BATCH_KEYS}

    batch = get_batch_on_this_tp_rank(batch, broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(), broadcast_group=mpu.get_tensor_model_parallel_group(), is_sft=is_sft, is_hybrid_cp=is_hybrid_cp, create_attention_mask_in_dataloader=create_attention_mask_in_dataloader, cp_size=cp_size, tp_rank=tp_rank, micro_batch_size=args.micro_batch_size, seq_length=args.seq_length, mtp_on_this_rank=mtp_on_this_rank, pipeline_model_parallel_size=args.pipeline_model_parallel_size, is_pipeline_first_stage=mpu.is_pipeline_first_stage(), is_pipeline_last_stage=mpu.is_pipeline_last_stage())

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank:
        assert is_sft
        return None, batch['cu_seqlens'], batch['cu_seqlens_padded'], None, None, None, None, batch['max_seqlen'], None, None

    batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=is_hybrid_cp, cp_group=get_context_parallel_group(), hybrid_cp_group_func=get_hybrid_data_context_parallel_groups)

    # Return values in BATCH_KEYS order so callers can unpack into the fixed
    # names regardless of any provenance fields wrappers like BlendedDataset
    # add (e.g. "dataset_id"). The for-loop above already populates every
    # BATCH_KEYS entry on tp_rank 0; other tp_ranks receive a fresh dict from
    # get_batch_on_this_tp_rank. BATCH_KEYS is already alphabetical, matching
    # the historical sorted(batch.keys()) order.
    return [batch[key] for key in BATCH_KEYS]


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[HybridModel] = None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: HybridModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (HybridModel): The Hybrid Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()

    global stimer

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            attention_mask,
            cu_seqlens,
            cu_seqlens_padded,
            hybrid_cp_group,
            labels,
            local_cp_size,
            loss_mask,
            max_seqlen,
            position_ids,
            tokens,
        ) = get_batch(data_iterator, vp_stage)

    packed_seq_params = None
    padding_mask = None
    if cu_seqlens is not None:
        # cu_seqlens / cu_seqlens_padded carry the dataloader's batch dim (1, n).
        # PackedSeqParams (and TE attention) expect 1-D, so squeeze before use.
        cu_seqlens = cu_seqlens[0]
        if cu_seqlens_padded is not None:
            cu_seqlens_padded = cu_seqlens_padded[0]
        # Use real (unpadded) cu_seqlens to feed the FLOPs accounting: varlen
        # attention only computes work for real tokens within each chunk.
        update_seqlen_stats_from_cu_seqlens(cu_seqlens)
        args = get_args()
        config = core_transformer_config_from_args(args)
        (tokens, labels, loss_mask, position_ids, packed_seq_params, padding_mask) = (
            _prepare_packed_thd_batch(
                tokens=tokens,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                cu_seqlens_padded=cu_seqlens_padded,
                max_seqlen=max_seqlen,
                local_cp_size=local_cp_size,
                hybrid_cp_group=hybrid_cp_group,
                config=config,
                pad_token_id=int(get_tokenizer().pad),
                context_parallel_size=args.context_parallel_size,
            )
        )

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
        )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank_func(layout=config.pipeline_model_parallel_layout, mtp_num_layers=config.mtp_num_layers, ignore_virtual=False, vp_stage=vp_stage)
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    is_packed_sequence = False
    if args.sft:
        dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence),
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    model_cfg = hybrid_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(full_config,
             train_valid_test_datasets_provider,
             partial(model_provider, hybrid_builder),
             ModelType.encoder_or_decoder,
             forward_step,
             store=store,
             )
