#!/usr/bin/env python3
"""
Script to create YAML inference configuration from Megatron checkpoint.

This script loads args from a Megatron checkpoint, converts it to a dictionary,
updates it with inference-specific parameters, and saves as a YAML config file.

Usage:
    python create_yaml_inference_config.py --ckpt_path /path/to/checkpoint.pt --output_config /path/to/config.yaml
    # Alternatively:
    # Read from `/lustre/fsw/portfolios/llmservice/users/<username>/workspace/output/model_name/`
    python create_yaml_inference_config.py --model_name model_name
"""

import argparse
import json
import os
import torch
import yaml
import sys
from pathlib import Path

# Add megatron to the path.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
)

# =============================================================================
# INFERENCE PARAMETER OVERRIDES
# Define the inference parameters that will overwrite the checkpoint args
# =============================================================================

# Checkpoint args -> config.json entry (dot path). Value from checkpoint is written at the given entry.
# Example: "vision_config.args.min_num_patches" writes to json config["vision_config"]["args"]["min_num_patches"].
HF_OVERRIDES = {
    "dynamic_resolution_min_patches": "vision_config.min_num_patches",
    "dynamic_resolution_max_patches": "vision_config.max_num_patches",
    "video_target_num_patches": "vision_config.video_target_num_patches",
    "video_target_img_size": "vision_config.video_target_img_size",
    "video_maintain_aspect_ratio": "vision_config.video_maintain_aspect_ratio",
    "video_temporal_patch_size": "vision_config.video_temporal_patch_size",
    "video_prompt_version": "vision_config.video_prompt_version",
    "separate_video_embedder": "vision_config.separate_video_embedder",
}

# config.json fields where a token string has a companion *_id field that must
# agree with the tokenizer.  Maps  token-name key  ->  token-id key.
TOKEN_ID_FIELDS = {
    "img_context_token": "img_context_token_id",
    "sound_context_token": "sound_context_token_id",
    "video_context_token": "video_context_token_id",
}


def _validate_tokenizer_compatibility(hf_dir, checkpoint_args):
    """Cross-check the HF template tokenizer against the megatron checkpoint.

    Compares vocab size, special-token count, and individual special-token
    presence/IDs so that a stale or wrong template is caught early instead of
    silently producing a broken HF model.
    """
    tokenizer_path = os.path.join(hf_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("  Warning: tokenizer.json not found, skipping compatibility check")
        return

    with open(tokenizer_path) as f:
        tok = json.load(f)

    added_tokens = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    base_vocab_size = len(tok.get("model", {}).get("vocab", {}))
    num_added = len(added_tokens)

    errors = []

    padded_vocab = getattr(checkpoint_args, "padded_vocab_size", None)
    if padded_vocab is not None and padded_vocab != base_vocab_size:
        errors.append(
            f"padded_vocab_size mismatch: checkpoint={padded_vocab}, "
            f"tokenizer base vocab={base_vocab_size}"
        )

    num_special = getattr(checkpoint_args, "tiktoken_num_special_tokens", None)
    if num_special is not None and num_special != num_added:
        errors.append(
            f"tiktoken_num_special_tokens mismatch: checkpoint={num_special}, "
            f"tokenizer added_tokens={num_added}"
        )

    ckpt_special = getattr(checkpoint_args, "special_tokens", None)
    if ckpt_special:
        for tok_str in ckpt_special:
            if tok_str not in added_tokens:
                errors.append(
                    f"checkpoint special token {tok_str!r} missing from "
                    f"tokenizer.json added_tokens"
                )

    # If the checkpoint records which tokenizer it was trained with, load it
    # and compare every added-token ID against the template.
    ckpt_tokenizer_model = getattr(checkpoint_args, "tokenizer_model", None)
    if ckpt_tokenizer_model:
        train_tok_path = os.path.join(ckpt_tokenizer_model, "tokenizer.json")
        if os.path.exists(train_tok_path):
            with open(train_tok_path) as f:
                train_tok = json.load(f)
            train_added = {t["content"]: t["id"] for t in train_tok.get("added_tokens", [])}
            id_mismatches = []
            for content, train_id in train_added.items():
                template_id = added_tokens.get(content)
                if template_id is not None and template_id != train_id:
                    id_mismatches.append((content, train_id, template_id))
            if id_mismatches:
                for content, train_id, template_id in id_mismatches:
                    errors.append(
                        f"token {content!r} ID mismatch: "
                        f"training tokenizer={train_id}, template={template_id}"
                    )

    if errors:
        print("\n" + "=" * 60)
        print("ERROR: Tokenizer compatibility check FAILED")
        print("=" * 60)
        if ckpt_tokenizer_model:
            print(f"  Checkpoint tokenizer_model: {ckpt_tokenizer_model}")
        print(f"  Template tokenizer:         {tokenizer_path}")
        for e in errors:
            print(f"  - {e}")
        print("The template tokenizer does not match the checkpoint. "
              "Update the template or use the correct tokenizer.")
        sys.exit(1)
    else:
        print(f"  Tokenizer compatibility check passed "
              f"(vocab={base_vocab_size}, added_tokens={num_added}"
              + (f", special_tokens={len(ckpt_special)}" if ckpt_special else "")
              + ")")


def _resolve_token_ids(hf_config, hf_dir):
    """Resolve *_context_token_id fields in hf_config from the tokenizer.

    For every (token_name_key, token_id_key) pair in TOKEN_ID_FIELDS, look up
    the actual integer ID of the token string in tokenizer.json and overwrite
    the ID in hf_config so the two are always consistent.
    """
    tokenizer_path = os.path.join(hf_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"  Warning: tokenizer.json not found at {tokenizer_path}, skipping token-ID resolution")
        return

    with open(tokenizer_path) as f:
        tok = json.load(f)

    token_to_id = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}

    for name_key, id_key in TOKEN_ID_FIELDS.items():
        token_str = hf_config.get(name_key)
        if token_str is None:
            continue

        resolved_id = token_to_id.get(token_str)
        old_id = hf_config.get(id_key)

        if resolved_id is None:
            print(f"  Warning: token {token_str!r} ({name_key}) not found in tokenizer.json added_tokens")
            continue

        if old_id != resolved_id:
            print(f"  Fixing {id_key}: {old_id} -> {resolved_id}  "
                  f"(token {token_str!r} is ID {resolved_id} in tokenizer)")
        else:
            print(f"  Verified {id_key}: {resolved_id} matches {token_str!r}")

        hf_config[id_key] = resolved_id


def _set_config_at_path(config, path, value):
    """Set config[path] = value where path is dot-separated (e.g. 'vision_config.args.min_num_patches')."""
    parts = path.split(".")
    obj = config
    for part in parts[:-1]:
        if part not in obj or not isinstance(obj[part], dict):
            obj[part] = {}
        obj = obj[part]
    obj[parts[-1]] = value


INFERENCE_PARAMS = {
    # Sampling parameters
    "temperature": 1.0,              # Sampling temperature
    "top_k": 1,                      # Top-k sampling

    # Sequence length parameters
    "out_seq_length": 1024,             # Output sequence length (will become out-seq-length)
    "inference_max_seq_length": 131072,  # Maximum sequence length for inference
    "max_tokens_to_oom": 131072,         # Maximum tokens before OOM
    "decoder_seq_length": 131072,        # Decoder sequence length
    "max_position_embeddings": 131072,   # Maximum position embeddings

    # Dropout parameters
    "attention_dropout": 0.0,        # Attention dropout for inference
    "hidden_dropout": 0.0,           # Hidden dropout for inference

    # Batch processing
    "micro_batch_size": 1,           # Micro batch size for inference

    # Precision settings
    "bf16": True,                    # Use bfloat16 precision

    # Model loading settings
    "no_load_rng": True,             # Don't load RNG state
    "no_load_optim": True,           # Don't load optimizer state

    # NOTE: eos_id will be set dynamically based on prompt format

    # Vision/multimodal settings
    "max_num_tiles": 12,             # Maximum number of image tiles

    # Backend settings - Note: this might conflict with the skipped parameter
    # "attention_backend": "flash",    # Use Flash Attention backend - commented out since it causes issues
    "flash_decode": True,            # Enable flash decode
    "attention_backend": "flash"
}

# =============================================================================
# PROMPT FORMAT MAPPINGS
# =============================================================================

def map_prompt_format_for_inference(prompt_format):
    """
    Map training prompt formats to their inference equivalents.

    Args:
        prompt_format (str): The original prompt format from checkpoint

    Returns:
        str: The mapped prompt format for inference
    """
    format_mappings = {
        "nemotron-h-5p5-reasoning": "nemotron-h-5p5-reasoning-inference",
        # Add more mappings here as needed
        # "training-format": "inference-format",
    }

    mapped_format = format_mappings.get(prompt_format, prompt_format)

    if mapped_format != prompt_format:
        print(f"  Mapping prompt format: {prompt_format} -> {mapped_format}")

    return mapped_format


def get_eos_id_for_prompt_format(prompt_format):
    """
    Determine the appropriate eos_id based on the prompt format.

    Args:
        prompt_format (str): The prompt format string

    Returns:
        int: The appropriate eos token ID
    """
    # Common eos token IDs for different model families
    eos_id_mappings = {
        # Nemotron family models
        "nemotron5": 15,
        "nemotron-h-reasoning": 11,
        "nemotron-h-5p5-reasoning": 12,
        "nemotron-h-5p5-reasoning-inference": 12,
    }

    eos_id = eos_id_mappings.get(prompt_format, None)  # Default fallback

    print(f"  Setting eos_id={eos_id} for prompt format '{prompt_format}'")

    return eos_id


EXCLUDED_PARAMS = {
    # =================================================================
    # INTERNAL PYTHON OBJECTS & STATE
    # =================================================================
    '__class__', '__dict__', '__doc__', '__module__', '__weakref__',
    'attention_backend', 'model_type',

    # Internal state variables
    'consumed_train_samples', 'consumed_valid_samples', 'iteration', 'rank', 'world_size',
    'skipped_train_samples', 'local_rank', 'data_parallel_size', 'iterations_to_skip',

    # Computed/derived values
    'padded_vocab_size', 'params_dtype', 'main_grads_dtype', 'main_params_dtype',
    'exp_avg_dtype', 'exp_avg_sq_dtype',

    # Internal flags that shouldn't be CLI args
    'add_position_embedding', 'align_grad_reduce', 'barrier_with_L1_time',
    'bert_binary_head', 'bias_swiglu_fusion', 'check_for_nan_in_loss_and_grad',
    'clone_scatter_output_in_embedding', 'create_attention_mask_in_dataloader',
    'data_sharding', 'enable_gloo_process_groups', 'enable_msc', 'enable_one_logger',
    'fp8_wgrad', 'gradient_accumulation_fusion', 'gradient_reduce_div_fusion',
    'log_loss_scale_to_tensorboard', 'manual_gc_eval', 'mmap_bin_files',
    'pin_cpu_grads', 'pin_cpu_params', 'retro_verify_neighbor_count',
    'scatter_gather_tensors_in_pipeline', 'torch_fsdp2_reshard_after_forward',
    'tp_comm_bulk_dgrad', 'tp_comm_bulk_wgrad', 'tp_comm_overlap_ag',
    'tp_comm_overlap_rs', 'tp_comm_split_ag', 'tp_comm_split_rs',
    'transformer_pipeline_model_parallel_size', 'use_tokenizer_model_from_checkpoint_args',
    'inference_max_batch_size',

    # Encoder-specific (usually not needed for decoder-only inference)
    'encoder_num_layers', 'encoder_pipeline_model_parallel_size',
    'encoder_seq_length', 'encoder_tensor_model_parallel_size',

    # =================================================================
    # OPTIMIZER & LEARNING RATE PARAMETERS
    # =================================================================
    'adam_beta1', 'adam_beta2', 'adam_eps', 'lr', 'lr_decay_samples', 'lr_decay_style',
    'lr_warmup_init', 'lr_warmup_iters', 'lr_warmup_samples', 'lr_wsd_decay_style',
    'weight_decay', 'weight_decay_incr_style', 'start_weight_decay', 'end_weight_decay',
    'clip_grad', 'sgd_momentum', 'optimizer', 'optimizer_cpu_offload',
    'optimizer_offload_fraction', 'min_lr',

    # =================================================================
    # TRAINING DATA & BATCHING
    # =================================================================
    'global_batch_size', 'train_samples', 'train_full_dataset', 'split', 'num_workers',
    'num_dataset_builder_threads', 'dataloader_type', 'sample_rate', 'mask_prob',
    'short_seq_prob', 'classes_fraction', 'data_per_class_fraction',
    'data_parallel_random_init', 'data_parallel_sharding_strategy',

    # =================================================================
    # CHECKPOINTING & SAVING
    # =================================================================
    'save', 'save_interval', 'no_save_optim', 'no_save_rng', 'ckpt_step',
    'ckpt_fully_parallel_save', 'ckpt_fully_parallel_save_deprecated',
    'ckpt_assume_constant_structure', 'pretrained_checkpoint', 'finetune',
    'sft', 'sft_tokenizer_prompt_format',

    # =================================================================
    # LOGGING & MONITORING
    # =================================================================
    'log_interval', 'log_energy', 'log_memory_to_tensorboard', 'log_num_zeros_in_grad',
    'log_params_norm', 'log_progress', 'log_straggler', 'log_throughput',
    'log_timers_to_tensorboard', 'log_validation_ppl_to_tensorboard',
    'log_world_size_to_tensorboard', 'tensorboard_dir', 'tensorboard_log_interval', 'tensorboard_queue_size',
    'wandb_exp_name', 'wandb_project', 'wandb_save_dir', 'one_logger_async',
    'one_logger_project',

    # =================================================================
    # TRAINING INFRASTRUCTURE
    # =================================================================
    'eval_interval', 'eval_iters', 'skip_train', 'test_mode', 'exit_duration_in_mins',
    'exit_signal_handler', 'manual_gc', 'manual_gc_interval', 'profile', 'profile_ranks',
    'profile_step_end', 'profile_step_start', 'record_memory_history', 'memory_snapshot_path',

    # =================================================================
    # TRAINING-SPECIFIC OPTIMIZATIONS
    # =================================================================
    'accumulate_allreduce_grads_in_fp32', 'grad_reduce_in_bf16', 'overlap_grad_reduce',
    'overlap_param_gather', 'overlap_param_gather_with_optimizer_step',
    'overlap_cpu_optimizer_d2h_h2d', 'use_distributed_optimizer', 'calculate_per_token_loss',
    'loss_scale_window', 'initial_loss_scale', 'min_loss_scale', 'hysteresis',
    'align_param_gather', 'ddp_average_in_collective', 'ddp_pad_buckets_for_high_nccl_busbw',
    'defer_embedding_wgrad_compute', 'delay_wgrad_compute', 'wgrad_deferral_limit',

    # =================================================================
    # VISION TRAINING SPECIFIC
    # =================================================================
    'vision_pretraining', 'vision_pretraining_type', 'head_lr_mult', 'iter_per_epoch',
    'dino_bottleneck_size', 'dino_freeze_last_layer', 'dino_head_hidden_size',
    'dino_local_crops_number', 'dino_local_img_size', 'dino_norm_last_layer',
    'dino_teacher_temp', 'dino_warmup_teacher_temp', 'dino_warmup_teacher_temp_epochs',
    'freeze_LM', 'freeze_ViT', 'mask_type', 'mask_factor',

    # =================================================================
    # SYSTEM/INFRASTRUCTURE (mostly training-related)
    # =================================================================
    'straggler_ctrlr_port', 'straggler_minmax_count', 'disable_straggler_on_startup',
    'inprocess_active_world_size', 'inprocess_barrier_timeout', 'inprocess_completion_timeout',
    'inprocess_empty_cuda_cache', 'inprocess_granularity', 'inprocess_hard_timeout',
    'inprocess_heartbeat_interval', 'inprocess_heartbeat_timeout', 'inprocess_last_call_wait',
    'inprocess_monitor_process_interval', 'inprocess_monitor_thread_interval',
    'inprocess_progress_watchdog_interval', 'inprocess_restart', 'inprocess_soft_timeout',
    'inprocess_termination_grace_time', 'error_injection_rate', 'error_injection_type',
    'rerun_mode', 'adlr_autoresume', 'adlr_autoresume_interval',

    # =================================================================
    # BIENCODER/RETRIEVAL (usually not needed for standard inference)
    # =================================================================
    'biencoder_projection_dim', 'biencoder_shared_query_context_model',
    'retriever_score_scaling', 'retriever_seq_length', 'query_in_block_prob',
    'retro_add_retriever', 'retro_attention_gate', 'retro_encoder_attention_dropout',
    'retro_encoder_hidden_dropout', 'retro_encoder_layers', 'retro_num_neighbors',
    'retro_num_retrieved_chunks', 'indexer_batch_size', 'indexer_log_interval',

    # =================================================================
    # MOE TRAINING SPECIFIC
    # =================================================================
    'moe_apply_probs_on_input', 'moe_aux_loss_coeff', 'moe_deepep_num_sms',
    'moe_enable_deepep', 'moe_layer_recompute', 'moe_pad_expert_input_to_capacity',
    'moe_per_layer_logging', 'moe_permute_fusion', 'moe_router_bias_update_rate',
    'moe_router_enable_expert_bias', 'moe_router_force_load_balancing',
    'moe_router_padding_for_fp8', 'moe_shared_expert_overlap', 'moe_token_drop_policy',
    'moe_upcycling_granularity', 'moe_use_upcycling',

    # =================================================================
    # MISCELLANEOUS TRAINING/UNUSED
    # =================================================================
    'app_tag_run_version', 'calc_ft_timeouts', 'config_logger_dir', 'decrease_batch_size_if_needed',
    'deprecated_use_mcore_models', 'eod_mask_loss', 'enable_ft_package', 'enable_te_ce',
    'fsdp_double_buffer', 'init_method_xavier_uniform', 'mid_level_dataset_surplus',
    'non_persistent_local_ckpt_algo', 'num_channels', 'num_classes', 'output_bert_embeddings',
    'override_opt_param_scheduler', 'perform_initialization', 'replication', 'replication_factor',
    'reset_attention_mask', 'reset_position_ids', 'run_workload_inspector_server',
    'tiktoken_num_special_tokens', 'timing_log_level', 'timing_log_option',
    'use_checkpoint_opt_param_scheduler', 'use_one_sent_docs', 'use_persistent_ckpt_worker',
    'use_pytorch_profiler', 'variable_seq_lengths', 'vocab_extra_ids',

    # =================================================================
    # MOE TRAINING/UNUSED
    # =================================================================
    'moe_extended_tp', 'moe_grouped_gemm', 'moe_layer_freq',
    'moe_router_load_balancing_type', 'moe_router_pre_softmax',
    'moe_router_score_function', 'moe_router_topk',
    'moe_token_dispatcher_type', 'moe_use_legacy_grouped_gemm',

    # =================================================================
    # FP8 TRAINING/UNUSED
    # =================================================================
    'fp8_amax_compute_algo', 'fp8_amax_history_len', 'fp8_interval',
    'fp8_margin', 'fp8_param_gather', 'fp8_recipe', "fp8",
    'fp16', 'fp16_lm_cross_entropy', 'fp32_residual_connection',
    'first_last_layers_bf16',

    # =================================================================
    # OTHER TRAINING/UNUSED
    # =================================================================
    'use_cpu_initialization', 'recompute_vision', "do_train", "do_valid",
    "do_test", "async_tensor_model_parallel_allreduce", "bias_dropout_fusion",
    "curr_iteration", "num_floating_point_operations_so_far",
}

# Some args need to be renamed:
RENAME_PARAMS = {
    "top-k": "top_k",
}


def load_checkpoint_args(ckpt_path):
    """
    Load checkpoint and extract the args namespace.

    Args:
        ckpt_path (str): Path to the checkpoint file

    Returns:
        argparse.Namespace: The args namespace from the checkpoint
    """
    print(f"Loading checkpoint from: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    if 'args' not in ckpt:
        print("Error: Checkpoint does not contain 'args' key")
        sys.exit(1)

    args_namespace = ckpt['args']
    print(f"Successfully loaded args namespace with {len(vars(args_namespace))} parameters")

    return args_namespace


def namespace_to_dict(namespace):
    """
    Convert argparse.Namespace to dictionary, handling nested objects.

    Args:
        namespace: argparse.Namespace object

    Returns:
        dict: Dictionary representation of the namespace
    """
    def convert_value(value, depth=0, max_depth=3):
        # Prevent infinite recursion
        if depth > max_depth:
            return str(value)

        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif hasattr(value, '__dict__') and not isinstance(value, type):
            # Handle nested namespace objects, but avoid types/classes
            try:
                return {k: convert_value(v, depth+1, max_depth) for k, v in vars(value).items()}
            except:
                return str(value)
        elif isinstance(value, (list, tuple)):
            # Handle lists and tuples
            try:
                output = [convert_value(item, depth+1, max_depth) for item in value]
                if len(output) == 0:
                    return None
                return output
            except:
                return str(value)
        elif isinstance(value, dict):
            # Handle dictionaries
            try:
                return {k: convert_value(v, depth+1, max_depth) for k, v in value.items()}
            except:
                return str(value)
        elif hasattr(value, 'dtype') and hasattr(value, 'tolist'):
            # Handle numpy arrays or torch tensors
            try:
                return value.tolist()
            except:
                return str(value)
        elif callable(value) or isinstance(value, type):
            # Handle functions/callables/types
            return str(value)
        else:
            # Handle other types
            try:
                # Try to convert to basic types
                if hasattr(value, '__dict__'):
                    return str(value)
                else:
                    return value
            except:
                return str(value)

    result = {}
    for key, value in vars(namespace).items():
        if key in EXCLUDED_PARAMS:
            print(f"  Skipping complex parameter: {key}")
            continue

        try:
            if convert_value(value) is not None:
                converted_value = convert_value(value)
                # Test if the value can be serialized to YAML
                yaml.dump({key: converted_value}, stream=None)
                result[key] = converted_value
        except Exception as e:
            print(f"Warning: Could not convert parameter '{key}': {e}")
            # Try to store as string
            try:
                result[key] = str(value)
                # Test YAML serialization again
                yaml.dump({key: result[key]}, stream=None)
            except:
                print(f"  Skipping parameter '{key}' - cannot serialize to YAML")
                continue

    return result


def convert_param_name_for_yaml(param_name):
    """
    Convert parameter name from Python format (underscores) to YAML/CLI format (dashes).

    Args:
        param_name (str): Parameter name with underscores

    Returns:
        str: Parameter name with dashes for YAML/CLI compatibility
    """
    param_name = param_name.replace('_', '-')
    if param_name in RENAME_PARAMS:
        param_name = RENAME_PARAMS[param_name]
    return param_name


def save_yaml_config(config_dict, output_path, inference_params_keys):
    """
    Save configuration dictionary as YAML file with inference params at the top.

    Args:
        config_dict (dict): Configuration to save
        output_path (str): Output YAML file path
        inference_params_keys (list): List of inference parameter keys to put at top
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean and convert parameter names for YAML
    clean_config = {}

    for key, value in config_dict.items():
        try:
            # Convert parameter name to YAML format (underscores to dashes)
            yaml_key = convert_param_name_for_yaml(key)

            # Test if this key-value pair can be serialized
            yaml.dump({yaml_key: value}, stream=None)
            clean_config[yaml_key] = value
        except Exception as e:
            print(f"Warning: Skipping parameter '{key}' due to YAML serialization error: {e}")
            # Try to convert to string as fallback
            try:
                yaml_key = convert_param_name_for_yaml(key)
                clean_config[yaml_key] = str(value)
                yaml.dump({yaml_key: clean_config[yaml_key]}, stream=None)
            except:
                print(f"  Cannot serialize '{key}' even as string, skipping")
                continue

    # Convert inference parameter keys to YAML format
    yaml_inference_keys = [convert_param_name_for_yaml(k) for k in inference_params_keys]

    # Create ordered config with inference params first
    ordered_config = {}

    # Add inference parameters first (in the order they were defined)
    print("\nInference parameters (at top of YAML):")
    for param_key in yaml_inference_keys:
        if param_key in clean_config:
            ordered_config[param_key] = clean_config[param_key]
            print(f"  {param_key}: {clean_config[param_key]}")

    # Add all other parameters (sorted alphabetically)
    print(f"\nOther parameters from checkpoint: {len(clean_config) - len(ordered_config)} params")
    for key in sorted(clean_config.keys()):
        if key not in yaml_inference_keys:
            ordered_config[key] = clean_config[key]

    try:
        with open(output_path, 'w') as f:
            # Write inference parameters section
            f.write("# ===== Inference Parameters =====\n")
            inference_config = {k: v for k, v in ordered_config.items() if k in yaml_inference_keys}
            yaml.dump(inference_config, f, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)

            # Write separator
            f.write("\n# ===== Checkpoint Parameters =====\n")

            # Write checkpoint parameters
            checkpoint_config = {k: v for k, v in ordered_config.items() if k not in yaml_inference_keys}
            yaml.dump(checkpoint_config, f, default_flow_style=False, indent=2, sort_keys=True, allow_unicode=True)

        print(f"Successfully saved configuration to: {output_path}")
        print(f"Final config contains {len(clean_config)} parameters")
        print(f"  - Inference parameters: {len(inference_config)}")
        print(f"  - Checkpoint parameters: {len(checkpoint_config)}")
    except Exception as e:
        print(f"Error saving configuration: {e}")
        sys.exit(1)


def update_inference_params(config_dict, inference_params):
    """
    Update configuration dictionary with inference parameters.

    Args:
        config_dict (dict): Configuration dictionary from checkpoint
        inference_params (dict): Inference parameters to override

    Returns:
        dict: Updated configuration dictionary
    """
    print("Updating configuration with inference parameters:")

    updated_config = config_dict.copy()

    # Handle prompt format mapping
    if 'tokenizer_prompt_format' in updated_config:
        original_format = updated_config['tokenizer_prompt_format']
        mapped_format = map_prompt_format_for_inference(original_format)
        updated_config['tokenizer_prompt_format'] = mapped_format

        # Determine eos_id based on the mapped prompt format
        eos_id = get_eos_id_for_prompt_format(mapped_format)
        if eos_id is not None:
            updated_config['eos_id'] = eos_id

    for key, value in inference_params.items():
        if key in updated_config:
            old_value = updated_config[key]
            print(f"  Overriding {key}: {old_value} -> {value}")
        else:
            print(f"  Adding {key}: {value}")

        updated_config[key] = value

    # Not used in inference.
    if "sequence_parallel" in updated_config:
        updated_config["sequence_parallel"] = False

    if "context_parallel_size" in updated_config:
        updated_config["context_parallel_size"] = 1

    # For MoE models, keep TP size the same as the checkpoint.
    updated_config["pipeline_model_parallel_size"] = 1
    updated_config["tensor_model_parallel_size"] = 1
    updated_config["expert_model_parallel_size"] = 1
    updated_config["expert_tensor_parallel_size"] = 1

    # Not allowed in inference.
    updated_config.pop("allow_missing_vision_projection_checkpoint", None)
    updated_config.pop("allow_missing_sound_projection_checkpoint", None)
    updated_config.pop("allow_missing_sound_model_checkpoint", None)
    updated_config.pop("allow_missing_conv_merge_checkpoint", None)
    updated_config.pop("masked_softmax_fusion", None)

    # Not used in inference.
    updated_config.pop("tensorboard-dir", None)

    return updated_config


def main():
    """Main function to create YAML inference configuration."""
    parser = argparse.ArgumentParser(
        description="Create YAML inference configuration from Megatron checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Name of the model"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        help="Path to the Megatron checkpoint file (.pt)"
    )

    parser.add_argument(
        "--output_config",
        type=str,
        required=False,
        help="Output path for the YAML configuration file"
    )

    parser.add_argument(
        "--model_base_path",
        type=str,
        required=False,
        help="Base path for the model (parent of checkpoints/). Overrides the default path derived from --model_name."
    )

    parser.add_argument(
        "--show_params",
        action="store_true",
        help="Show all parameters that will be included in the config"
    )

    parser.add_argument(
        "--update_hf_config",
        type=str,
        default=None,
        metavar="HF_PATH",
        help="Path to HF model dir (containing config.json). Generate config.yaml there and add/overwrite params from checkpoint into config.json (see HF_OVERRIDES)."
    )

    args = parser.parse_args()

    if args.model_name is not None:
        if args.model_base_path is not None:
            path = args.model_base_path
        else:
            user = os.environ.get("SLURM_JOB_USER", os.environ.get("USER"))
            path = f"/lustre/fsw/portfolios/llmservice/users/{user}/workspace/output/{args.model_name}"
        some_iter = int(open(f"{path}/checkpoints/latest_checkpointed_iteration.txt").read().strip())
        ckpt_path = f"{path}/checkpoints/iter_{some_iter:07d}/mp_rank_00/model_optim_rng.pt"
        ckpt_path2 = f"{path}/checkpoints/iter_{some_iter:07d}/mp_rank_00_000/model_optim_rng.pt"
        # Check if it's MoE.
        if not os.path.exists(ckpt_path) and os.path.exists(ckpt_path2):
            ckpt_path = ckpt_path2
        args.ckpt_path = ckpt_path
        args.output_config = f"{path}/config.yaml"

    assert args.ckpt_path is not None, "Checkpoint path is required"
    assert args.output_config is not None, "Output config path is required"

    # Validate input file exists
    if not Path(args.ckpt_path).exists():
        print(f"Error: Checkpoint file does not exist: {args.ckpt_path}")
        sys.exit(1)

    print("=" * 60)
    print("Creating YAML Inference Configuration")
    print("=" * 60)

    # Step 1: Load checkpoint args
    checkpoint_args = load_checkpoint_args(args.ckpt_path)

    # Step 2: Convert namespace to dictionary
    print("\nConverting args namespace to dictionary...")
    config_dict = namespace_to_dict(checkpoint_args)

    # Step 3: Update with inference parameters
    print(f"\nUpdating with {len(INFERENCE_PARAMS)} inference parameters...")
    final_config = update_inference_params(config_dict, INFERENCE_PARAMS)

    # Step 4: Show parameters if requested
    if args.show_params:
        print(f"\nFinal configuration contains {len(final_config)} parameters:")
        for key in sorted(final_config.keys()):
            print(f"  {key}: {final_config[key]}")

    # Step 5: Save YAML configuration with inference params at top
    print(f"\nSaving configuration to: {args.output_config}")
    inference_keys = list(INFERENCE_PARAMS.keys()) + ['tokenizer_prompt_format', 'eos_id']
    save_yaml_config(final_config, args.output_config, inference_keys)

    # Step 6: If --update_hf_config, add/overwrite params from checkpoint into config.json
    if args.update_hf_config is not None:
        config_json_path = os.path.join(args.update_hf_config, "config.json")
        if not Path(config_json_path).exists():
            print(f"Error: config.json not found at {config_json_path}")
            sys.exit(1)
        with open(config_json_path) as f:
            hf_config = json.load(f)
        for args_name, config_path in HF_OVERRIDES.items():
            val = getattr(checkpoint_args, args_name, None)
            if val is None:
                continue
            _set_config_at_path(hf_config, config_path, val)
            print(f"  Set config.json {config_path!r} = {val!r} (from checkpoint {args_name!r})")

        _validate_tokenizer_compatibility(args.update_hf_config, checkpoint_args)
        _resolve_token_ids(hf_config, args.update_hf_config)

        # Strip sound_config when the checkpoint has no sound weights.
        # The template config may include sound_config (for omni models), but
        # vision-only checkpoints won't have the weights, causing vLLM to crash.
        if "sound_config" in hf_config:
            index_path = os.path.join(args.update_hf_config, "model.safetensors.index.json")
            has_sound_weights = False
            if Path(index_path).exists():
                with open(index_path) as f:
                    weight_map = json.load(f).get("weight_map", {})
                has_sound_weights = any("sound" in k for k in weight_map)
            if not has_sound_weights:
                for key in ["sound_config", "sound_context_token_id", "sound_context_token"]:
                    hf_config.pop(key, None)
                print("  Stripped sound_config from config.json (no sound weights in checkpoint)")
            else:
                print("  Kept sound_config (sound weights found in checkpoint)")

        with open(config_json_path, "w") as f:
            json.dump(hf_config, f, indent=2)
        print(f"Updated {config_json_path} with checkpoint params.")

    print("\n" + "=" * 60)
    print("Configuration creation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
