# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import gc
import json
import os
import re
import sys
import types
import torch
from datetime import timedelta

from utils import (
    combine_conv1d,
    combine_in_proj,
    install_converter_fake_process_groups,
    print_memory_usage,
)


class MegatronCheckpointLoaderBase:
    """Orchestrates loading a Megatron checkpoint and sending
    model parameters over a given multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
    """

    def __init__(self, args, queue, build_tokenizer=False):
        self.args = args
        self.queue = queue
        self.build_tokenizer = build_tokenizer
        self.margs = None            # Will hold Megatron's main args
        self.checkpoint_args = None  # Will hold additional checkpoint args
        self.all_models = None       # Model sharded over different parallelism
        self.md = None               # Metadata sent to the saver
        self.consumed_train_samples = None
        self.consumed_valid_samples = None


    def get_local_model(self, pp_rank=None, vp_rank=None, ep_rank=None, tp_rank=None):
        """
        Method used to get the local model for a certain (pp,ep,tp).
        If a value is None, will use retrieve a model without any consideration of that parallelism.
        Defaults to returning pp_rank=0, vp_rank=0, tp_rank=0 and a working ep_rank.
        """
        assert self.all_models is not None, "all_models is not set"
        if pp_rank is None:
            pp_rank = 0
        if vp_rank is None:
            vp_rank = 0
        if tp_rank is None:
            tp_rank = 0
        if ep_rank is None:
            ep_rank = 0
            # If MoE, holding all the other values static, find ep_rank where we can get a model with weights for relevant parallelism.
            # Deals with scenarios where etp=1 in MoE for example.
            is_moe = getattr(self.margs, 'num_experts', None) is not None and self.margs.num_experts > 0 and self.args.model_type == "hybrid"
            if is_moe:
                ep_rank = tp_rank // self.margs.expert_tensor_parallel_size
                tp_rank = tp_rank % self.margs.expert_tensor_parallel_size

        return self.all_models[pp_rank][vp_rank][ep_rank][tp_rank]


    def get_assembled_tensor_parallel_models(self, pp_rank=0, vp_rank=0):
        """
        Loop with get_local_model to handle MoE expert-tensor parallelism
        """
        assembled_models_tp = []
        for tp_rank in range(self.margs.tensor_model_parallel_size):
            assembled_models_tp.append(self.get_local_model(pp_rank=pp_rank, vp_rank=vp_rank, tp_rank=tp_rank))
        return assembled_models_tp

    def _maybe_parse_additional_megatron_args(self, margs, checkpoint_args):
        """
        Method used to optionally add arguments from the checkpoint to the main args.
        For instance, using margs.some_arg = checkpoint_args.some_arg
        """
        return margs

    def parse_megatron_args(self):
        """
        Parse Megatron arguments by forcibly overwriting sys.argv.
        Populates self.margs and self.checkpoint_args.
        """
        # Ensure we can import Megatron
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

        try:
            from megatron.training.arguments import parse_args, validate_args
            from megatron.training.checkpointing import load_args_from_checkpoint

        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            self.queue.put("exit")
            sys.exit(1)

        # Overwrite sys.argv
        sys.argv = self.build_sys_argv()

        margs = parse_args()
        margs, checkpoint_args = load_args_from_checkpoint(margs)

        # Adjust world size so validation doesn't fail
        margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

        # Copy data types from checkpoint
        margs.fp16 = checkpoint_args.fp16
        margs.bf16 = checkpoint_args.bf16

        # Ensure expert tensor parallel size reflects checkpoint value when present
        if hasattr(checkpoint_args, 'expert_tensor_parallel_size') and \
           getattr(checkpoint_args, 'expert_tensor_parallel_size') is not None:
            margs.expert_tensor_parallel_size = checkpoint_args.expert_tensor_parallel_size

        # Expert parallelism requires sequence parallelism
        if margs.expert_model_parallel_size > 1:
            margs.sequence_parallel = True

        margs = self._maybe_parse_additional_megatron_args(margs, checkpoint_args)

        # Validate final arguments
        try:
            from megatron.training.arguments import validate_args
            margs = validate_args(margs)
        except Exception as e:
            print(f"Error validating Megatron arguments: {e}")
            self.queue.put("exit")
            sys.exit(1)

        margs.use_legacy_models = False
        margs.transformer_impl = self.args.loader_transformer_impl
        if self.args.loader_transformer_impl == "local" and margs.normalization == "RMSNorm":
            margs.no_persist_layer_norm = True

        if self.args.ckpt_step is not None:
            margs.ckpt_step = self.args.ckpt_step
            margs.iteration = self.args.ckpt_step

        self.margs = margs
        self.checkpoint_args = checkpoint_args

    def _maybe_ensure_additional_required_arguments(self):
        """
        Can be used to ensure some expected args are present.
        For instance, use self.check_for_arg('some_arg')
        """
        pass

    def check_for_arg(self, arg_name, default=None):
        if getattr(self.margs, arg_name, None) is None:
            if default is not None:
                setattr(self.margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify argument {arg_name}. Exiting.")
                print(f"Arguments: {self.margs}")
                self.queue.put("exit")
                sys.exit(1)

    def ensure_required_arguments(self):
        """
        Ensure that certain Megatron arguments (from checkpoint) are present.
        If missing, either set defaults or exit.
        """

        self.check_for_arg('tensor_model_parallel_size')
        self.check_for_arg('pipeline_model_parallel_size')
        self.check_for_arg('num_layers')
        self.check_for_arg('hidden_size')
        self.check_for_arg('seq_length')
        self.check_for_arg('num_attention_heads')
        self.check_for_arg('max_position_embeddings')
        self.check_for_arg('position_embedding_type')
        self.check_for_arg('tokenizer_type')
        self.check_for_arg('iteration')
        self.check_for_arg('bert_binary_head')
        self.check_for_arg('disable_bias_linear', False)
        self.check_for_arg('params_dtype')
        self.check_for_arg('swiglu', False)

        self._maybe_ensure_additional_required_arguments()

    def initialize_megatron_env(self):
        """
        Initialize Megatron global variables and fused kernels.
        """
        try:
            from megatron.training.global_vars import set_global_variables
            from megatron.core import mpu
            from megatron.core.tensor_parallel import get_cuda_rng_tracker
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            self.queue.put("exit")
            sys.exit(1)

        try:
            from megatron.legacy import fused_kernels
        except ModuleNotFoundError:
            fused_kernels = None

        set_global_variables(self.margs, build_tokenizer=self.build_tokenizer)
        mpu.set_tensor_model_parallel_world_size(self.margs.tensor_model_parallel_size)
        mpu.set_expert_tensor_parallel_world_size(self.margs.expert_tensor_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(self.margs.pipeline_model_parallel_size)
        mpu.set_virtual_pipeline_model_parallel_world_size(self.margs.virtual_pipeline_model_parallel_size)
        mpu.set_expert_model_parallel_world_size(self.margs.expert_model_parallel_size)

        # Model construction reads a broad set of mpu process-group globals even
        # during single-process checkpoint conversion.
        install_converter_fake_process_groups(
            mpu,
            tensor_model_parallel_size=self.margs.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.margs.pipeline_model_parallel_size,
            expert_model_parallel_size=self.margs.expert_model_parallel_size,
            expert_tensor_parallel_size=self.margs.expert_tensor_parallel_size,
            context_parallel_size=self.margs.context_parallel_size,
        )

        # Match seed offsets from model_parallel_cuda_manual_seed in random.py
        get_cuda_rng_tracker().add('data-parallel-rng', self.margs.seed)
        get_cuda_rng_tracker().add('model-parallel-rng', self.margs.seed + 2718)
        get_cuda_rng_tracker().add('expert-parallel-rng', self.margs.seed + 1024)

        if fused_kernels is not None:
            fused_kernels.load(self.margs)

    def compute_true_vocab_size(self):
        """Determine the 'true' (non-padded) vocab size."""
        if self.args.true_vocab_size is not None:
            return self.args.true_vocab_size
        elif self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            return len(vocab)
        else:
            return None

    def verify_vocabs_match(self, true_vocab_size):
        """
        If both --true-vocab-size and --vocab-file are specified, verify they match.
        Return False (and exit) if they don't match; True otherwise.
        """
        if self.args.true_vocab_size is not None and self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            if len(vocab) != self.args.true_vocab_size:
                print("Both --true-vocab-size and --vocab-file specified but vocab sizes do not match. Aborting.")
                return False
        return True

    def load_model_shards(self, model_provider, dtype):
        """
        Build and load model shards for each tensor-parallel rank, returning:
          - A nested list of loaded models by [pipeline_rank][virtual_pipeline_rank].
          - consumed_train_samples, consumed_valid_samples
        """
        from megatron.core import mpu
        from megatron.training.checkpointing import load_checkpoint

        consumed_train_samples = None
        consumed_valid_samples = None
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        ep_size = self.margs.expert_model_parallel_size or 1
        etp_size = self.margs.expert_tensor_parallel_size or 1
        is_moe = getattr(self.margs, 'num_experts', None) is not None and self.margs.num_experts > 0 and self.args.model_type == "hybrid"
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # all_models[pp][vp][ep] -> list across TP
        all_models = []

        def get_models_for_pipeline_stage(tp_count, ep_count, dtype):
            # [vp][ep] each contains list across TP
            local_models_for_stage = [[[] for _ in range(ep_count)] for _ in range(vp_size)]

            for ep_rank in range(ep_count):
                # Set EP rank in fake group and parallel state
                if is_moe:
                    fake_ep_group = mpu.get_expert_model_parallel_group()
                    if hasattr(fake_ep_group, 'set_rank'):
                        fake_ep_group.set_rank(ep_rank)
                    try:
                        mpu.set_expert_model_parallel_rank(ep_rank)
                    except Exception:
                        pass

                for tp_rank in range(tp_count):
                    # TODO: check correctness, maybe not correct when tp > etp?
                    if is_moe:
                        diff_tp_rank = tp_rank - ((ep_rank * etp_size) % tp_size)
                        if diff_tp_rank >= etp_size or diff_tp_rank < 0:
                            continue;
                    fake_tp_group = mpu.get_tensor_model_parallel_group()
                    if hasattr(fake_tp_group, 'set_rank'):
                        fake_tp_group.set_rank(tp_rank)
                    mpu.set_tensor_model_parallel_rank(tp_rank)

                    model_list = []
                    for i in range(vp_size):
                        mpu.set_virtual_pipeline_model_parallel_rank(i)
                        pre_process = mpu.is_pipeline_first_stage()
                        post_process = mpu.is_pipeline_last_stage()
                        this_model = model_provider(pre_process=pre_process,
                                                    post_process=post_process).to(dtype)
                        model_list.append(this_model)

                    # Reset counters and load this shard
                    self.margs.consumed_train_samples = 0
                    self.margs.skipped_train_samples = 0
                    self.margs.consumed_valid_samples = 0
                    self.margs.exit_on_missing_checkpoint = True
                    load_checkpoint(model_list, None, None)

                    # Validate that train/valid samples match across ranks
                    nonlocal consumed_train_samples, consumed_valid_samples
                    if consumed_train_samples is not None:
                        assert self.margs.consumed_train_samples == consumed_train_samples
                    else:
                        consumed_train_samples = self.margs.consumed_train_samples

                    if consumed_valid_samples is not None:
                        assert self.margs.consumed_valid_samples == consumed_valid_samples
                    else:
                        consumed_valid_samples = self.margs.consumed_valid_samples

                    for vp_rank in range(vp_size):
                        local_models_for_stage[vp_rank][ep_rank].append(model_list[vp_rank])

                    # Print memory usage (use combined count to reflect TP progress)
                    print_memory_usage("loader", tp_rank, tp_count)

            return local_models_for_stage

        # Load shards for each pipeline rank
        mpu.set_virtual_pipeline_model_parallel_rank(0)
        for pp_rank in range(pp_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            all_models.append(get_models_for_pipeline_stage(tp_size, ep_size, dtype))

        return all_models, consumed_train_samples, consumed_valid_samples

    def _compact_expert_memory(self):
        """Free dispensable EP shard model objects, retaining only expert weight refs.

        When loading checkpoints with many EP shards (e.g. EP=64), all shards are held
        in memory simultaneously (~765 GB for the Super model). The saver subprocess
        then needs additional memory to build the target model, causing OOM.

        This method identifies which EP shards are needed intact for non-expert data
        (used by get_assembled_tensor_parallel_models), then for the remaining shards:
        stores Python references to their expert weight tensors (fc1/fc2) in
        self.expert_cache, and deletes the model objects. Non-expert parameters
        (attention, mamba, norms, embeddings) that are duplicated across those EP shards
        are freed, dramatically reducing memory.
        """
        ep_size = self.margs.expert_model_parallel_size or 1
        if ep_size <= 1:
            return

        tp_size = self.margs.tensor_model_parallel_size
        etp_size = self.margs.expert_tensor_parallel_size or 1
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # EP shards accessed by get_assembled_tensor_parallel_models must be kept:
        # get_local_model maps global tp_rank → ep_rank = tp_rank // ETP
        needed_ep_ranks = set(tp_rank // etp_size for tp_rank in range(tp_size))
        freeable_ep_ranks = [ep for ep in range(ep_size) if ep not in needed_ep_ranks]

        if not freeable_ep_ranks:
            return

        # Match expert weight parameters in three naming conventions:
        # 1) CoreMoETESchema (local_experts): ...layers.N.mlp.experts.local_experts.E.linear_fcK.weight
        # 2) CoreHybridMoETESchema (fused):   ...layers.N.mlp.experts.linear_fcK.weightE
        # 3) 3D fused tensor:                 ...layers.N.mlp.experts.linear_fcK.weight  (shape [E, out, in])
        pattern_local = re.compile(
            r'.*layers\.(\d+)\.mlp\.experts\.local_experts\.(\d+)\.linear_fc([12])\.weight$'
        )
        pattern_fused_individual = re.compile(
            r'.*layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$'
        )
        pattern_fused_3d = re.compile(
            r'.*layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight$'
        )

        def cache_tensor(tensor):
            # Keep cached expert weights independent of the module object that is freed below.
            return tensor.detach().cpu().clone()

        self.expert_cache = {}

        for pp_rank in range(pp_size):
            for vp_rank in range(vp_size):
                for ep_rank in freeable_ep_ranks:
                    tp_models = self.all_models[pp_rank][vp_rank][ep_rank]
                    for tp_rank in range(len(tp_models)):
                        model = tp_models[tp_rank]
                        if model is None:
                            continue

                        cache_key = (pp_rank, vp_rank, ep_rank, tp_rank)
                        layer_data = {}

                        for name, param in model.named_parameters():
                            # Skip MTP parameters: their nested "layers.N" indices
                            # collide with decoder layer indices in the cache dict.
                            if '.mtp.' in name:
                                continue

                            # Convention 1: local_experts.E.linear_fcK.weight
                            m = pattern_local.match(name)
                            if m:
                                layer_idx = int(m.group(1))
                                expert_idx = int(m.group(2))
                                fc_num = int(m.group(3))
                                if layer_idx not in layer_data:
                                    layer_data[layer_idx] = {}
                                if expert_idx not in layer_data[layer_idx]:
                                    layer_data[layer_idx][expert_idx] = {}
                                layer_data[layer_idx][expert_idx][fc_num] = cache_tensor(param)
                                continue

                            # Convention 2: linear_fcK.weightE (individual per-expert params)
                            m = pattern_fused_individual.match(name)
                            if m:
                                layer_idx = int(m.group(1))
                                fc_num = int(m.group(2))
                                expert_idx = int(m.group(3))
                                if layer_idx not in layer_data:
                                    layer_data[layer_idx] = {}
                                if expert_idx not in layer_data[layer_idx]:
                                    layer_data[layer_idx][expert_idx] = {}
                                layer_data[layer_idx][expert_idx][fc_num] = cache_tensor(param)
                                continue

                            # Convention 3: linear_fcK.weight as 3D [E, out, in]
                            m = pattern_fused_3d.match(name)
                            if m and param.data.dim() == 3:
                                layer_idx = int(m.group(1))
                                fc_num = int(m.group(2))
                                if layer_idx not in layer_data:
                                    layer_data[layer_idx] = {}
                                for expert_idx in range(param.data.shape[0]):
                                    if expert_idx not in layer_data[layer_idx]:
                                        layer_data[layer_idx][expert_idx] = {}
                                    layer_data[layer_idx][expert_idx][fc_num] = cache_tensor(
                                        param[expert_idx]
                                    )

                        self.expert_cache[cache_key] = layer_data

                        self.all_models[pp_rank][vp_rank][ep_rank][tp_rank] = None
                        del model

                gc.collect()

        print(f"Expert memory compaction: freed {len(freeable_ep_ranks)}/{ep_size} EP shard models, "
              f"kept {len(needed_ep_ranks)} intact, "
              f"cached expert data for {len(self.expert_cache)} (ep, tp) combinations.")

    def send_metadata_over_queue(self):
        # Let the consumer know the overall metadata:
        self.md.consumed_train_samples = self.consumed_train_samples
        self.md.consumed_valid_samples = self.consumed_valid_samples
        self.queue.put(self.md)

    def queue_put(self, name, msg):
        print(f"sending {name}")
        msg["name"] = name
        self.queue.put(msg)

    def _send_attention_layer(self, models, layer_idx, schema):
        """
        Extract attention layer parameters and return message dictionary.
        """
        tp_size = self.margs.tensor_model_parallel_size
        layer = schema.get_layer(models[0], layer_idx)
        message = {}

        # Non-parallel params
        message["input norm weight"] = layer["self_attn_norm_weight"]
        if self.md.norm_has_bias:
            message["input norm bias"] = layer["self_attn_norm_bias"]
        if self.md.linear_bias:
            message["dense bias"] = layer["self_attn_proj_bias"]

        # Collect parallel parameters
        qkv_weight, qkv_bias = [], []
        dense_weight = []

        for model_tp in models:
            layer_p = schema.get_layer(model_tp, layer_idx)
            qkv_weight.append(layer_p["self_attn_qkv_weight"])
            dense_weight.append(layer_p["self_attn_proj_weight"])
            if self.md.qkv_bias:
                qkv_bias.append(layer_p["self_attn_qkv_bias"])

        # Standard concatenations
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)

        if self.md.qkv_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)

        return message

    def _send_mlp_layer(self, models, layer_idx, schema):
        """
        Extract MLP layer parameters and return message dictionary.
        """
        tp_size = self.margs.tensor_model_parallel_size
        layer = schema.get_layer(models[0], layer_idx)
        message = {}

        # Non-parallel params
        message["post norm weight"] = layer["mlp_norm_weight"]
        if self.md.norm_has_bias:
            message["post norm bias"] = layer["mlp_norm_bias"]
        if self.md.linear_bias:
            message["mlp l1 bias"] = layer["mlp_fc2_bias"]

        # Collect parallel parameters
        mlp_l0_weight, mlp_l0_bias = [], []
        mlp_l1_weight = []

        for model_tp in models:
            layer_p = schema.get_layer(model_tp, layer_idx)
            mlp_l0_weight.append(layer_p["mlp_fc1_weight"])
            mlp_l1_weight.append(layer_p["mlp_fc2_weight"])
            if self.md.linear_bias:
                mlp_l0_bias.append(layer_p["mlp_fc1_bias"])

        # If we are using SwiGLU, chunk each mlp_l0_weight
        if self.md.swiglu:
            for i in range(tp_size):
                mlp_l0_weight[i] = torch.chunk(mlp_l0_weight[i], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

        if self.md.linear_bias:
            if self.md.swiglu:
                for i in range(tp_size):
                    mlp_l0_bias[i] = torch.chunk(mlp_l0_bias[i], 2, dim=0)
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

        return message

    def _get_expert_stacks_from_cache(self, pp_rank, vp_rank, ep_rank, tp_rank, layer_idx, num_local_experts):
        """Retrieve fc1/fc2 expert weight stacks from the compacted expert cache."""
        cache_key = (pp_rank, vp_rank, ep_rank, tp_rank)
        cached_layers = self.expert_cache[cache_key]
        fc1_stack = torch.stack([
            cached_layers[layer_idx][expert_idx][1] for expert_idx in range(num_local_experts)
        ], dim=0)
        fc2_stack = torch.stack([
            cached_layers[layer_idx][expert_idx][2] for expert_idx in range(num_local_experts)
        ], dim=0)
        return fc1_stack, fc2_stack

    def _send_moe_layer(self, models_by_ep, layer_idx, schema, pp_rank=0, vp_rank=0):
        """
        MoE version: aggregate experts across EP ranks and TP shards into a single message.
        models_by_ep: List[List[Module]] shaped [ep_size][tp_size]
        pp_rank, vp_rank: needed to look up expert_cache for compacted ep>0 models.
        """
        ep_size = self.margs.expert_model_parallel_size or 1
        tp_size = self.margs.tensor_model_parallel_size
        etp_size = self.margs.expert_tensor_parallel_size or 1

        # Non-parallel params from ep=0 reference model
        ref_layer = schema.get_layer(models_by_ep[0][0], layer_idx)
        message = {
            "pre mlp norm weight": ref_layer["pre_mlp_norm_weight"],
        }
        if self.md.norm_has_bias:
            message["pre mlp norm bias"] = ref_layer["pre_mlp_norm_bias"]

        message["router weight"] = ref_layer["router_weight"]
        message["router bias"] = ref_layer["router_bias"]

        # MoE latent projections (duplicated mode, not sharded across TP)
        if self.md.moe_latent_size:
            message["fc1 latent proj weight"] = ref_layer["fc1_latent_proj_weight"]
            message["fc2 latent proj weight"] = ref_layer["fc2_latent_proj_weight"]
            if self.md.linear_bias:
                message["fc1 latent proj bias"] = ref_layer["fc1_latent_proj_bias"]
                message["fc2 latent proj bias"] = ref_layer["fc2_latent_proj_bias"]

        # Assemble shared experts across TP (uses ep=0 models only)
        shared_l0_tp = []
        shared_l1_tp = []
        assembled_models_tp = self.get_assembled_tensor_parallel_models(pp_rank=pp_rank, vp_rank=vp_rank)
        for tp_rank in range(tp_size):
            layer_p = schema.get_layer(assembled_models_tp[tp_rank], layer_idx)
            shared_l0_tp.append(layer_p["mlp_shared_fc1_weight"])  # column-parallel combine
            shared_l1_tp.append(layer_p["mlp_shared_fc2_weight"])  # row-parallel combine
        message["shared mlp l0 weight"] = torch.cat(shared_l0_tp, dim=0)
        message["shared mlp l1 weight"] = torch.cat(shared_l1_tp, dim=1)

        # Build per-EP, TP-merged expert weights
        num_local_experts = self.margs.num_experts // (self.margs.expert_model_parallel_size or 1)
        use_cache = hasattr(self, 'expert_cache') and self.expert_cache

        fc1_ep_concat = []  # list of [local_E, out, in] merged across TP per EP
        fc2_ep_concat = []
        for ep_rank in range(ep_size):
            # Gather TP shards for this EP
            fc1_tp = []
            fc2_tp = []
            for etp_rank in range(etp_size):
                model = models_by_ep[ep_rank][etp_rank]
                if model is not None:
                    layer_p = schema.get_layer(model, layer_idx)
                    fc1_stack = torch.stack([
                        layer_p[f"mlp_fc1_weight.{expert_idx}"] for expert_idx in range(num_local_experts)
                    ], dim=0)
                    fc2_stack = torch.stack([
                        layer_p[f"mlp_fc2_weight.{expert_idx}"] for expert_idx in range(num_local_experts)
                    ], dim=0)
                elif use_cache:
                    fc1_stack, fc2_stack = self._get_expert_stacks_from_cache(
                        pp_rank, vp_rank, ep_rank, etp_rank, layer_idx, num_local_experts
                    )
                else:
                    raise RuntimeError(
                        f"Model is None for ep={ep_rank} tp={etp_rank} and no expert cache available"
                    )
                fc1_tp.append(fc1_stack)
                fc2_tp.append(fc2_stack)

            # Combine across TP: fc1 column-parallel -> concat dim=1; fc2 row-parallel -> concat dim=2
            if self.md.swiglu:
                fc1_W = [torch.chunk(t, 2, dim=1)[0] for t in fc1_tp]
                fc1_V = [torch.chunk(t, 2, dim=1)[1] for t in fc1_tp]
                fc1_merged = torch.cat([torch.cat(fc1_W, dim=1), torch.cat(fc1_V, dim=1)], dim=1)
            else:
                fc1_merged = torch.cat(fc1_tp, dim=1)
            fc2_merged = torch.cat(fc2_tp, dim=2)

            fc1_ep_concat.append(fc1_merged)
            fc2_ep_concat.append(fc2_merged)

        # Concatenate experts across EP ranks along expert dimension (dim=0)
        fc1_all = torch.cat(fc1_ep_concat, dim=0)
        fc2_all = torch.cat(fc2_ep_concat, dim=0)

        if self.md.swiglu:
            # Split back into W/V for transport if needed by saver
            message["mlp l0 weight W"] = torch.chunk(fc1_all, 2, dim=1)[0]
            message["mlp l0 weight V"] = torch.chunk(fc1_all, 2, dim=1)[1]
        else:
            message["mlp l0 weight"] = fc1_all
        message["mlp l1 weight"] = fc2_all

        return message

    def _send_mamba_layer(self, models, layer_idx, schema):
        """
        Extract Mamba layer parameters and return message dictionary.
        """
        tp_size = self.margs.tensor_model_parallel_size
        layer = schema.get_layer(models[0], layer_idx)
        message = {}

        # Non-parallel params
        message["in proj norm weight"] = layer["mixer_in_proj_layer_norm_weight"]

        # Collect parallel parameters
        dt_bias = []
        D = []
        A_log = []
        in_proj_weight = []
        conv_1d_weight, conv_1d_bias = [], []
        norm_weight = []
        out_proj_weight = []

        for model_tp in models:
            layer_p = schema.get_layer(model_tp, layer_idx)
            dt_bias.append(layer_p["mixer_dt_bias"])
            D.append(layer_p["mixer_D"])
            A_log.append(layer_p["mixer_A_log"])
            in_proj_weight.append(layer_p["mixer_in_proj_weight"])
            conv_1d_weight.append(layer_p["mixer_conv1d_weight"])
            conv_1d_bias.append(layer_p["mixer_conv1d_bias"])
            norm_weight.append(layer_p["mixer_norm_weight"])
            out_proj_weight.append(layer_p["mixer_out_proj_weight"])

        # Concatenate parallel parameters
        message["dt bias"] = torch.cat(dt_bias, dim=0)
        message["D"] = torch.cat(D, dim=0)
        message["A log"] = torch.cat(A_log, dim=0)

        # Combine specialized parameters
        if self.margs.mamba_num_heads is not None:
            nheads = self.margs.mamba_num_heads
            d_inner = nheads * self.margs.mamba_head_dim
        else:
            d_inner = self.md.hidden_size * 2  # TODO: can I know expansion factor?
            nheads = d_inner // self.margs.mamba_head_dim
        ngroups = self.margs.mamba_num_groups
        d_state = self.md.mamba_state_dim
        message["in proj weight"] = combine_in_proj(in_proj_weight, d_inner, ngroups, d_state, nheads, tp_size=tp_size)
        message["conv1d weight"] = combine_conv1d(conv_1d_weight, "weight", d_inner, ngroups, d_state, tp_size=tp_size)
        message["conv1d bias"] = combine_conv1d(conv_1d_bias, "bias", d_inner, ngroups, d_state, tp_size=tp_size)
        message["norm weight"] = torch.cat(norm_weight, dim=0)
        message["out proj weight"] = torch.cat(out_proj_weight, dim=1)

        return message

    def _send_mtp_layer(self, models, mtp_layer_idx, mtp_schema, main_schema):
        """
        Extract MTP layer parameters and return message dictionary.

        Args:
            models: List of models across TP ranks
            mtp_layer_idx: Index of the MTP layer
            mtp_schema: MTP schema for parameter extraction
            main_schema: Main model schema (for transformer layer params)
        """
        tp_size = self.margs.tensor_model_parallel_size
        message = {}

        # Get first model's MTP layer for non-parallel params
        first_mtp_params = mtp_schema.get_mtp_layer(models[0], mtp_layer_idx)

        # Non-parallel params (norms)
        message["enorm weight"] = first_mtp_params["enorm_weight"]
        if self.md.norm_has_bias and first_mtp_params.get("enorm_bias") is not None:
            message["enorm bias"] = first_mtp_params["enorm_bias"]

        message["hnorm weight"] = first_mtp_params["hnorm_weight"]
        if self.md.norm_has_bias and first_mtp_params.get("hnorm_bias") is not None:
            message["hnorm bias"] = first_mtp_params["hnorm_bias"]

        message["final layernorm weight"] = first_mtp_params["final_layernorm_weight"]
        if self.md.norm_has_bias and first_mtp_params.get("final_layernorm_bias") is not None:
            message["final layernorm bias"] = first_mtp_params["final_layernorm_bias"]

        # Collect eh_proj weight across TP (column-parallel)
        eh_proj_weights = []
        for model_tp in models:
            mtp_params = mtp_schema.get_mtp_layer(model_tp, mtp_layer_idx)
            eh_proj_weights.append(mtp_params["eh_proj_weight"])
        message["eh proj weight"] = torch.cat(eh_proj_weights, dim=0)

        # For hybrid MTP (e.g. MTPHybridSchema) the transformer layer schema is empty;
        # sub-layer params are gathered by _send_mtp_hybrid_sublayers instead.
        if not mtp_schema._mtp_transformer_layer_schema:
            return message

        # Transformer layer params within mtp_model_layer
        # These follow the same parallel strategy as main transformer layers

        # Self-attention norm (non-parallel)
        if first_mtp_params.get("self_attn_norm_weight") is not None:
            message["mtp attn norm weight"] = first_mtp_params["self_attn_norm_weight"]
        if self.md.norm_has_bias and first_mtp_params.get("self_attn_norm_bias") is not None:
            message["mtp attn norm bias"] = first_mtp_params.get("self_attn_norm_bias")

        # QKV weights (column-parallel)
        qkv_weights = []
        qkv_biases = []
        for model_tp in models:
            mtp_params = mtp_schema.get_mtp_layer(model_tp, mtp_layer_idx)
            if mtp_params.get("self_attn_qkv_weight") is not None:
                qkv_weights.append(mtp_params["self_attn_qkv_weight"])
            if self.md.qkv_bias and mtp_params.get("self_attn_qkv_bias") is not None:
                qkv_biases.append(mtp_params["self_attn_qkv_bias"])

        if qkv_weights:
            message["mtp qkv weight"] = torch.cat(qkv_weights, dim=0)
        if qkv_biases:
            message["mtp qkv bias"] = torch.cat(qkv_biases, dim=0)

        # Projection weights (row-parallel)
        proj_weights = []
        for model_tp in models:
            mtp_params = mtp_schema.get_mtp_layer(model_tp, mtp_layer_idx)
            if mtp_params.get("self_attn_proj_weight") is not None:
                proj_weights.append(mtp_params["self_attn_proj_weight"])

        if proj_weights:
            message["mtp dense weight"] = torch.cat(proj_weights, dim=1)
        if self.md.linear_bias and first_mtp_params.get("self_attn_proj_bias") is not None:
            message["mtp dense bias"] = first_mtp_params["self_attn_proj_bias"]

        # MLP norm (non-parallel)
        if first_mtp_params.get("mlp_norm_weight") is not None:
            message["mtp mlp norm weight"] = first_mtp_params["mlp_norm_weight"]
        if self.md.norm_has_bias and first_mtp_params.get("mlp_norm_bias") is not None:
            message["mtp mlp norm bias"] = first_mtp_params.get("mlp_norm_bias")

        # MLP fc1 weights (column-parallel)
        mlp_l0_weights = []
        mlp_l0_biases = []
        for model_tp in models:
            mtp_params = mtp_schema.get_mtp_layer(model_tp, mtp_layer_idx)
            if mtp_params.get("mlp_fc1_weight") is not None:
                mlp_l0_weights.append(mtp_params["mlp_fc1_weight"])
            if self.md.linear_bias and mtp_params.get("mlp_fc1_bias") is not None:
                mlp_l0_biases.append(mtp_params["mlp_fc1_bias"])

        if mlp_l0_weights:
            if self.md.swiglu:
                # Chunk each weight and separate W and V
                for i in range(tp_size):
                    mlp_l0_weights[i] = torch.chunk(mlp_l0_weights[i], 2, dim=0)
                message["mtp mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weights], dim=0)
                message["mtp mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weights], dim=0)
            else:
                message["mtp mlp l0 weight"] = torch.cat(mlp_l0_weights, dim=0)

        if mlp_l0_biases:
            if self.md.swiglu:
                for i in range(tp_size):
                    mlp_l0_biases[i] = torch.chunk(mlp_l0_biases[i], 2, dim=0)
                message["mtp mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_biases], dim=0)
                message["mtp mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_biases], dim=0)
            else:
                message["mtp mlp l0 bias"] = torch.cat(mlp_l0_biases, dim=0)

        # MLP fc2 weights (row-parallel)
        mlp_l1_weights = []
        for model_tp in models:
            mtp_params = mtp_schema.get_mtp_layer(model_tp, mtp_layer_idx)
            if mtp_params.get("mlp_fc2_weight") is not None:
                mlp_l1_weights.append(mtp_params["mlp_fc2_weight"])

        if mlp_l1_weights:
            message["mtp mlp l1 weight"] = torch.cat(mlp_l1_weights, dim=1)
        if self.md.linear_bias and first_mtp_params.get("mlp_fc2_bias") is not None:
            message["mtp mlp l1 bias"] = first_mtp_params["mlp_fc2_bias"]

        return message

    def _send_mtp_hybrid_sublayers(self, mtp_schema, main_schema, pp_rank, vp_rank, mtp_layer_idx):
        """
        For hybrid MTP patterns (e.g. '*E'), extract internal sub-layer params and return
        them as a dict to be merged into the "mtp layer {i}" queue message.

        This supplements the outer params (enorm/hnorm/eh_proj/final_layernorm) that
        _send_mtp_layer already sends via MTPHybridSchema.get_mtp_layer.
        """
        from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols

        mtp_pattern = getattr(self.md, 'mtp_hybrid_override_pattern', None)
        if not mtp_pattern:
            return {}

        layer_type_list = [c for c in mtp_pattern if c != LayerSymbols.PIPE]
        ep_size = self.margs.expert_model_parallel_size or 1
        tp_size = self.margs.tensor_model_parallel_size

        message = {}

        # TP models (ep_rank=0) for this PP/VP stage
        models_tp = self.get_assembled_tensor_parallel_models(pp_rank=pp_rank, vp_rank=vp_rank)
        internal_by_tp = [mtp_schema.get_mtp_model_layers(m, mtp_layer_idx) for m in models_tp]
        if internal_by_tp[0] is None:
            return message

        for sub_idx, layer_type in enumerate(layer_type_list):
            if layer_type == LayerSymbols.ATTENTION:
                # Non-parallel norm from TP rank 0
                ref = main_schema._get(main_schema["layer"], internal_by_tp[0][sub_idx])
                if ref.get("self_attn_norm_weight") is not None:
                    message["input norm weight"] = ref["self_attn_norm_weight"]

                # Column-parallel QKV: concatenate across TP
                qkv_list = [
                    main_schema._get(main_schema["layer"], internal_by_tp[tp][sub_idx]).get("self_attn_qkv_weight")
                    for tp in range(tp_size)
                ]
                qkv_list = [q for q in qkv_list if q is not None]
                if qkv_list:
                    message["qkv weight"] = torch.cat(qkv_list, dim=0)

                # Row-parallel proj: concatenate across TP
                dense_list = [
                    main_schema._get(main_schema["layer"], internal_by_tp[tp][sub_idx]).get("self_attn_proj_weight")
                    for tp in range(tp_size)
                ]
                dense_list = [d for d in dense_list if d is not None]
                if dense_list:
                    message["dense weight"] = torch.cat(dense_list, dim=1)

            elif layer_type == LayerSymbols.MOE:
                # Non-parallel params from EP=0, TP=0 reference layer
                ref = main_schema._get(main_schema["layer"], internal_by_tp[0][sub_idx])
                for msg_key, param_key in [
                    ("pre mlp norm weight", "pre_mlp_norm_weight"),
                    ("router weight", "router_weight"),
                    ("router bias", "router_bias"),
                    ("fc1 latent proj weight", "fc1_latent_proj_weight"),
                    ("fc2 latent proj weight", "fc2_latent_proj_weight"),
                ]:
                    if ref.get(param_key) is not None:
                        message[msg_key] = ref[param_key]

                # Shared experts (column-parallel fc1, row-parallel fc2)
                shared_fc1 = [
                    main_schema._get(main_schema["layer"], internal_by_tp[tp][sub_idx]).get("mlp_shared_fc1_weight")
                    for tp in range(tp_size)
                ]
                shared_fc1 = [s for s in shared_fc1 if s is not None]
                if shared_fc1:
                    message["shared mlp l0 weight"] = torch.cat(shared_fc1, dim=0)
                shared_fc2 = [
                    main_schema._get(main_schema["layer"], internal_by_tp[tp][sub_idx]).get("mlp_shared_fc2_weight")
                    for tp in range(tp_size)
                ]
                shared_fc2 = [s for s in shared_fc2 if s is not None]
                if shared_fc2:
                    message["shared mlp l1 weight"] = torch.cat(shared_fc2, dim=1)

                # Expert weights: EP-sharded (gather across EP ranks)
                num_local_experts = self.margs.num_experts // ep_size
                fc1_ep_parts, fc2_ep_parts = [], []
                for ep_rank in range(ep_size):
                    ep_model = self.all_models[pp_rank][vp_rank][ep_rank][0]
                    ep_internal = mtp_schema.get_mtp_model_layers(ep_model, mtp_layer_idx)
                    if ep_internal is None:
                        continue
                    ep_params = main_schema._get(main_schema["layer"], ep_internal[sub_idx])
                    fc1_w = [ep_params.get(f"mlp_fc1_weight.{e}") for e in range(num_local_experts)]
                    fc2_w = [ep_params.get(f"mlp_fc2_weight.{e}") for e in range(num_local_experts)]
                    fc1_w = [w for w in fc1_w if w is not None]
                    fc2_w = [w for w in fc2_w if w is not None]
                    if fc1_w:
                        fc1_ep_parts.append(torch.stack(fc1_w, dim=0))
                    if fc2_w:
                        fc2_ep_parts.append(torch.stack(fc2_w, dim=0))

                if fc1_ep_parts:
                    message["mlp l0 weight"] = torch.cat(fc1_ep_parts, dim=0)
                if fc2_ep_parts:
                    message["mlp l1 weight"] = torch.cat(fc2_ep_parts, dim=0)

        return message

    def send_mtp_over_queue(self, mtp_schema, main_schema):
        """
        Send MTP block parameters over the queue.
        Only called on the last pipeline stage where MTP layers exist.
        """
        # Check if MTP is enabled
        if self.md.mtp_num_layers is None or self.md.mtp_num_layers == 0:
            return

        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # MTP layers only exist on the last pipeline stage
        last_pp_rank = pp_size - 1
        last_vp_rank = vp_size - 1

        # Get models for the last pipeline stage
        models = self.get_assembled_tensor_parallel_models(pp_rank=last_pp_rank, vp_rank=last_vp_rank)

        # Determine number of physical MTP layers
        # When mtp_use_repeated_layer is True, there's only 1 physical layer
        if self.md.mtp_use_repeated_layer:
            num_physical_layers = 1
        else:
            num_physical_layers = self.md.mtp_num_layers

        is_hybrid = getattr(self.md, 'mtp_hybrid_override_pattern', None) is not None

        # Send each MTP layer
        for mtp_layer_idx in range(num_physical_layers):
            message = self._send_mtp_layer(models, mtp_layer_idx, mtp_schema, main_schema)
            # For hybrid MTP, also include internal sub-layer params (attention, MoE)
            if is_hybrid:
                sublayer_params = self._send_mtp_hybrid_sublayers(
                    mtp_schema, main_schema, last_pp_rank, last_vp_rank, mtp_layer_idx
                )
                message.update(sublayer_params)
            self.queue_put(f"mtp layer {mtp_layer_idx}", message)

    def send_llm_over_queue(self, schema, schema_prefix=""):
        """
        Using self.all_models, extract model parameters and send them over the queue.
        schema_prefix: dotted prefix for the LM sub-model (e.g. "language_model." for LLaVA).
        """
        # 2) Transformer layers
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # We'll start with pipeline=0, vp=0, ep=0 for embeddings/final norm
        # Loop with get_local_model to handle MoE expert-tensor parallelism
        first_pipeline_models = self.get_assembled_tensor_parallel_models(pp_rank=0, vp_rank=0)

        # 1) Embeddings
        embeddings = [schema.get("embeddings", m) for m in first_pipeline_models]
        message = {
            "word embeddings": torch.cat([e["word"] for e in embeddings], dim=0)
        }
        if self.md.position_embedding_type == 'learned_absolute':
            # Only send one set from rank 0
            message["position embeddings"] = embeddings[0]["pos"]
        else:
            assert embeddings[0]["pos"] is None
        self.queue_put("embeddings", message)

        if self.md.model_type == "hybrid":
            from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols

            layer_type_list = [
                c for c in self.md.hybrid_override_pattern if c != LayerSymbols.PIPE
            ]
            assert len(layer_type_list) == self.md.num_layers, (
                f"Hybrid layer pattern has {len(layer_type_list)} layers, "
                f"but metadata expects {self.md.num_layers}."
            )

            total_layer_num = 0
            for vp_rank in range(vp_size):
                for pp_rank in range(pp_size):
                    models = self.get_assembled_tensor_parallel_models(pp_rank=pp_rank, vp_rank=vp_rank)
                    num_layers = schema.get_num_layers(self.all_models[pp_rank][vp_rank][0][0])
                    for layer_idx in range(num_layers):
                        layer_type = layer_type_list[total_layer_num]

                        if layer_type == LayerSymbols.MAMBA:
                            message = self._send_mamba_layer(models, layer_idx, schema)
                        elif layer_type == LayerSymbols.ATTENTION:
                            message = self._send_attention_layer(models, layer_idx, schema)
                        elif layer_type == LayerSymbols.MLP:
                            message = self._send_mlp_layer(models, layer_idx, schema)
                        elif layer_type == LayerSymbols.MOE:
                            message = self._send_moe_layer(self.all_models[pp_rank][vp_rank], layer_idx, schema, pp_rank=pp_rank, vp_rank=vp_rank)

                        self.queue_put(f"transformer layer {total_layer_num}", message)
                        total_layer_num += 1
        else:
            total_layer_num = 0
            for vp_rank in range(vp_size):
                for pp_rank in range(pp_size):
                    # Non-hybrid path: use ep=0 models across TP
                    models = self.all_models[pp_rank][vp_rank][0]
                    num_layers = schema.get_num_layers(models[0])
                    for layer_idx in range(num_layers):
                        # Combine attention and MLP layer parameters
                        attention_message = self._send_attention_layer(models, layer_idx, schema)
                        mlp_message = self._send_mlp_layer(models, layer_idx, schema)

                        # Merge both messages
                        message = {**attention_message, **mlp_message}

                        self.queue_put(f"transformer layer {total_layer_num}", message)
                        total_layer_num += 1

        # 3) MTP (Multi-Token Prediction) layers - sent before final layer outputs
        if self.md.mtp_num_layers is not None and self.md.mtp_num_layers > 0:
            from schema_core import get_mtp_schema
            is_hybrid = self.md.mtp_hybrid_override_pattern is not None
            mtp_schema = get_mtp_schema(
                self.margs.transformer_impl,
                is_hybrid=is_hybrid,
                prefix=schema_prefix,
            )
            self.send_mtp_over_queue(mtp_schema, schema)

        # 4) Final norm
        final_norm = schema.get("final_norm", models[0])
        message = {"weight": final_norm["weight"]}
        if self.md.norm_has_bias:
            message["bias"] = final_norm["bias"]
        self.queue_put("final norm", message)

        # 5) Output layer
        if self.md.output_layer:
            output_layers = [schema.get("output_layer", m) for m in models]
            message = {
                "weight": torch.cat([layer["weight"] for layer in output_layers], dim=0),
            }
            self.queue_put("output layer", message)

        # 6) BERT-specific parameters
        if self.md.model_type == 'BERT':
            # Pooler
            pooler = schema.get("pooler", models[0])
            message = {
                "weight": pooler["weight"],
                "bias": pooler["bias"],
            }
            self.queue_put("pooler", message)

            # LM head
            lm_head = schema.get("lm_head", models[0])
            message = {
                "dense weight": lm_head["dense_weight"],
                "dense bias": lm_head["dense_bias"],
                "norm weight": lm_head["norm_weight"],
            }
            if self.md.norm_has_bias:
                message["norm bias"] = lm_head["norm_bias"]
            self.queue_put("lm head", message)

            # Binary head
            if self.md.bert_binary_head:
                binary_head = schema.get("binary_head", models[0])
                message = {
                    "weight": binary_head["weight"],
                    "bias": binary_head["bias"],
                }
                self.queue_put("binary head", message)

        # Done
        self.queue.put("done")

    def load(self):
        """
        Orchestrate the entire flow of loading the Megatron checkpoint.
        """
        # 1) Parse Megatron arguments
        self.parse_megatron_args()

        # 2) Ensure required arguments are present
        self.ensure_required_arguments()

        # 3) Import the correct model provider (GPT or BERT)
        model_provider = self.import_model_provider()

        # 4) Initialize the Megatron environment
        self.initialize_megatron_env()

        # 5) Determine the true vocab size and verify if both sources match
        true_vocab_size = self.compute_true_vocab_size()
        if not self.verify_vocabs_match(true_vocab_size):
            self.queue.put("exit")
            sys.exit(1)

        # 6) Build metadata
        self.md = self.build_checkpoint_metadata(true_vocab_size)

        # 7) Load all model shards
        self.all_models, self.consumed_train_samples, self.consumed_valid_samples = self.load_model_shards(
            model_provider,
            self.md.params_dtype
        )

        # 7.5) Free ep>0 model objects, keeping only expert weight refs in cache
        self._compact_expert_memory()

        # 8) Send model over the queue
        self.send_model_over_queue()

    def _get_main_hybrid_layer_pattern(self):
        pattern = getattr(self.margs, 'hybrid_layer_pattern', None)
        if pattern is None:
            pattern = getattr(self.margs, 'hybrid_override_pattern', None)
        if pattern is None:
            pattern = getattr(self.checkpoint_args, 'hybrid_layer_pattern', None)
        if pattern is None:
            pattern = getattr(self.checkpoint_args, 'hybrid_override_pattern', None)
        if pattern is None:
            return None
        return pattern.split('/')[0]

    def build_checkpoint_metadata(self, true_vocab_size):
        """
        Construct a simple namespace for all relevant model metadata.
        """
        norm_has_bias = True
        if hasattr(self.checkpoint_args, 'normalization'):
            # For older models, normalization was always "LayerNorm".
            norm_has_bias = (self.checkpoint_args.normalization == "LayerNorm")

        md = types.SimpleNamespace()
        md.model_type = self.args.model_type
        md.num_layers = self.margs.num_layers
        md.hidden_size = self.margs.hidden_size
        md.seq_length = self.margs.seq_length
        md.num_attention_heads = self.margs.num_attention_heads
        md.num_query_groups = self.margs.num_query_groups
        md.num_experts = self.margs.num_experts
        md.moe_latent_size = getattr(self.margs, 'moe_latent_size', None)
        md.max_position_embeddings = self.margs.max_position_embeddings
        md.tokenizer_type = self.margs.tokenizer_type
        md.iteration = self.margs.iteration
        md.params_dtype = self.margs.params_dtype
        md.bert_binary_head = self.margs.bert_binary_head
        md.output_layer = self.margs.untie_embeddings_and_output_weights
        md.position_embedding_type = self.margs.position_embedding_type
        md.linear_bias = self.margs.add_bias_linear
        md.qkv_bias = self.margs.add_qkv_bias
        md.norm_has_bias = norm_has_bias
        md.swiglu = self.margs.swiglu
        md.previous_tensor_parallel_size = self.margs.tensor_model_parallel_size
        md.previous_pipeline_parallel_size = self.margs.pipeline_model_parallel_size
        md.vocab_size = true_vocab_size
        md.true_vocab_size = true_vocab_size
        md.make_vocab_size_divisible_by = self.margs.make_vocab_size_divisible_by
        md.checkpoint_args = self.checkpoint_args
        md.use_legacy_models = self.margs.use_legacy_models
        if self.args.model_type == "hybrid":
            md.hybrid_override_pattern = self._get_main_hybrid_layer_pattern()
            md.mamba_state_dim = self.margs.mamba_state_dim
        if self.args.ckpt_step is not None:
            md.ckpt_step = self.args.ckpt_step
            md.iteration = self.args.ckpt_step

        # Multi-Token Prediction (MTP) metadata
        md.mtp_num_layers = getattr(
            self.margs, 'mtp_num_layers', getattr(self.checkpoint_args, 'mtp_num_layers', None)
        )
        md.mtp_use_repeated_layer = getattr(
            self.margs, 'mtp_use_repeated_layer',
            getattr(self.checkpoint_args, 'mtp_use_repeated_layer', False)
        )
        md.mtp_hybrid_override_pattern = (
            getattr(self.margs, 'mtp_hybrid_override_pattern', None)
            or getattr(self.checkpoint_args, 'mtp_hybrid_override_pattern', None)
        )

        return md

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        This centralizes the hack of overwriting sys.argv.
        """

        my_argv = [
            'script.py',
            '--no-masked-softmax-fusion',
            '--no-bias-gelu-fusion',
            '--no-bias-dropout-fusion',
            '--use-cpu-initialization',
            '--micro-batch-size', '1',
            '--no-load-optim',
            '--no-load-rng',
            '--no-save-optim',
            '--no-save-rng',
            '--no-initialization',
            '--mock-data',  # To pass the "blend data checks" in arguments.py
            '--load', self.args.load_dir,
            '--exit-on-missing-checkpoint',
            '--use-mp-args-from-checkpoint-args',
            '--no-one-logger',
        ]
        if self.args.ckpt_step is not None:
            my_argv.extend(['--ckpt-step', str(self.args.ckpt_step)])
        return my_argv

    def import_model_provider(self):
        """Return the correct model_provider function depending on GPT vs. BERT."""
        raise NotImplementedError

    def send_model_over_queue(self):
        """Creates model schema and sends the model over the queue"""
        raise NotImplementedError
