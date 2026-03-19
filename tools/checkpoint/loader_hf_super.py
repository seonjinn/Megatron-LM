# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import importlib
import json
import os
import sys
import types
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from loader_hf_moe import HuggingFaceCheckpointLoaderMoE
from loader_base import MegatronCheckpointLoaderBase


def _stub_mamba_ssm_if_needed():
    """Stub out mamba_ssm CUDA extensions if they fail to import.

    The HF NemotronH model file imports mamba_ssm CUDA ops at module level,
    but we only need the model on CPU for weight extraction.  If the real
    mamba_ssm is already importable we leave it alone.
    """
    try:
        import mamba_ssm  # noqa: F401
        return  # real package works fine
    except (ImportError, ValueError, RuntimeError, OSError):
        pass

    def _make(name, is_pkg=False, attrs=None):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_pkg)
        if is_pkg:
            m.__path__ = []
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m

    _make('selective_scan_cuda')
    _make('causal_conv1d_cuda')
    _make('causal_conv1d', is_pkg=True,
          attrs={'causal_conv1d_fn': None, 'causal_conv1d_update': None})
    _make('mamba_ssm', is_pkg=True, attrs={'__version__': '2.2.2'})
    _make('mamba_ssm.ops', is_pkg=True)
    _make('mamba_ssm.ops.selective_scan_interface',
          attrs={'selective_scan_fn': None, 'mamba_inner_fn': None})
    _make('mamba_ssm.ops.triton', is_pkg=True)
    _make('mamba_ssm.ops.triton.selective_state_update',
          attrs={'selective_state_update': None})
    _make('mamba_ssm.ops.triton.ssd_combined',
          attrs={'mamba_chunk_scan_combined': None,
                 'mamba_split_conv1d_scan_combined': None})
    # Provide a real (subclassable) dummy class for RMSNorm so that
    # Megatron's ExtendedRMSNorm(RMSNormGated) can be defined.
    class _DummyRMSNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # Accept any init args (d_model, eps, group_size, etc.)
            d = args[0] if args else kwargs.get('normalized_shape', 1)
            self.weight = torch.nn.Parameter(torch.ones(d))

    _make('mamba_ssm.ops.triton.layernorm_gated',
          attrs={'RMSNorm': _DummyRMSNorm, 'rmsnorm_fn': None})


# NOTE: Do NOT call _stub_mamba_ssm_if_needed() at module level!
# convert.py imports this module before forking the saver subprocess.
# If real mamba_ssm import is attempted here, it partially initializes
# CUDA, causing "Cannot re-initialize CUDA in forked subprocess" in
# the saver.  The stub is called lazily in load_checkpoint() instead.


def add_arguments(parser):
    """Add command-line arguments relevant to HuggingFace model loading."""
    group = parser.add_argument_group(title='HuggingFace loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original size of vocab; if specified, trims padding from embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')


def _derive_hybrid_pattern(block_types):
    """Convert HF layers_block_type list to Megatron pattern string."""
    mapping = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
    return "".join(mapping[t] for t in block_types)


def _resolve_torch_dtype(hf_config):
    """Resolve torch dtype from HF config, handling missing torch_dtype field."""
    if hasattr(hf_config, 'torch_dtype') and hf_config.torch_dtype is not None:
        if isinstance(hf_config.torch_dtype, torch.dtype):
            return hf_config.torch_dtype
        # Could be a string like "bfloat16"
        return getattr(torch, str(hf_config.torch_dtype))
    # Fallback to dtype field
    if hasattr(hf_config, 'dtype') and hf_config.dtype is not None:
        return getattr(torch, hf_config.dtype)
    return torch.bfloat16


class HuggingFaceCheckpointLoaderSuper(HuggingFaceCheckpointLoaderMoE):
    def __init__(self, args, queue, build_tokenizer=False):
        super().__init__(args, queue, build_tokenizer)

    def load_model_shards(self, model_provider, dtype):
        """Load the HuggingFace model on CPU without device_map (avoids meta tensor issues)."""
        print("Loading HuggingFace model...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.args.load_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        self.hf_model.eval()
        all_models = [[[self.hf_model]]]
        return all_models, 0, 0

    def parse_megatron_args(self):
        """Parse Megatron arguments for Super model."""
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

        try:
            from megatron.training.arguments import parse_args, validate_args
        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            self.queue.put("exit")
            sys.exit(1)

        # Load HF config
        self.hf_config = AutoConfig.from_pretrained(self.args.load_dir, trust_remote_code=True)

        # Derive hybrid patterns from block type lists
        self._hybrid_override_pattern = _derive_hybrid_pattern(self.hf_config.layers_block_type)
        self._mtp_hybrid_override_pattern = _derive_hybrid_pattern(self.hf_config.mtp_layers_block_type)

        # Resolve dtype
        self._params_dtype = _resolve_torch_dtype(self.hf_config)

        # Build sys.argv based on HF config
        sys.argv = self.build_sys_argv()

        margs = parse_args()

        # Create fake checkpoint args based on HF config
        checkpoint_args = types.SimpleNamespace()
        checkpoint_args.fp16 = self._params_dtype == torch.float16
        checkpoint_args.bf16 = self._params_dtype == torch.bfloat16
        checkpoint_args.normalization = "RMSNorm"
        checkpoint_args.sequence_parallel = False
        checkpoint_args.apply_query_key_layer_scaling = False
        checkpoint_args.ffn_hidden_size = self.hf_config.intermediate_size
        checkpoint_args.num_attention_heads = self.hf_config.num_attention_heads
        checkpoint_args.num_query_groups = self.hf_config.num_key_value_heads
        checkpoint_args.kv_channels = self.hf_config.head_dim
        checkpoint_args.group_query_attention = checkpoint_args.num_query_groups < self.hf_config.num_attention_heads
        checkpoint_args.position_embedding_type = "none"
        if self.hf_config.mlp_hidden_act == "relu2":
            checkpoint_args.squared_relu = True
        if self.args.model_type == "hybrid":
            checkpoint_args.spec = ['megatron.core.models.mamba.mamba_layer_specs', 'mamba_stack_spec']
            checkpoint_args.hybrid_attention_ratio = 0.0
            checkpoint_args.hybrid_mlp_ratio = 0.0
            checkpoint_args.hybrid_override_pattern = self._hybrid_override_pattern
            checkpoint_args.mamba_state_dim = self.hf_config.ssm_state_size
            checkpoint_args.mamba_num_groups = self.hf_config.n_groups
            checkpoint_args.mamba_head_dim = self.hf_config.mamba_head_dim
            checkpoint_args.mamba_num_heads = self.hf_config.mamba_num_heads
            checkpoint_args.is_hybrid_model = True
        checkpoint_args.num_experts = self.hf_config.n_routed_experts
        checkpoint_args.moe_router_topk = self.hf_config.num_experts_per_tok
        checkpoint_args.moe_shared_expert_intermediate_size = self.hf_config.moe_shared_expert_intermediate_size
        checkpoint_args.moe_router_topk_scaling_factor = self.hf_config.routed_scaling_factor
        checkpoint_args.moe_router_enable_expert_bias = True
        checkpoint_args.moe_router_score_function = "sigmoid"
        checkpoint_args.moe_latent_size = self.hf_config.moe_latent_size
        checkpoint_args.expert_tensor_parallel_size = 1
        checkpoint_args.language_model_type = "nemotron6-super"

        # MTP checkpoint args
        checkpoint_args.mtp_num_layers = 2
        checkpoint_args.mtp_use_repeated_layer = True
        checkpoint_args.mtp_hybrid_override_pattern = self._mtp_hybrid_override_pattern
        checkpoint_args.mtp_spec = ['megatron.core.models.mamba.mamba_layer_specs', 'mamba_stack_spec']
        checkpoint_args.keep_mtp_spec_in_bf16 = True

        # Set key attributes from HF config
        margs.num_layers = self.hf_config.num_hidden_layers
        margs.hidden_size = self.hf_config.hidden_size
        margs.ffn_hidden_size = self.hf_config.intermediate_size
        margs.num_attention_heads = self.hf_config.num_attention_heads
        margs.num_query_groups = self.hf_config.num_key_value_heads
        margs.kv_channels = self.hf_config.head_dim
        margs.group_query_attention = checkpoint_args.group_query_attention
        margs.seq_length = 2048
        margs.max_position_embeddings = self.hf_config.max_position_embeddings
        margs.iteration = 1
        margs.params_dtype = self._params_dtype
        margs.add_bias_linear = self.hf_config.use_bias
        margs.add_qkv_bias = self.hf_config.attention_bias
        margs.swiglu = self.hf_config.mlp_hidden_act == "swiglu"
        margs.untie_embeddings_and_output_weights = not self.hf_config.tie_word_embeddings
        margs.bert_binary_head = False
        margs.tokenizer_type = "NullTokenizer"
        margs.position_embedding_type = "none"
        margs.make_vocab_size_divisible_by = 128
        margs.vocab_size = self.hf_config.vocab_size
        margs.padded_vocab_size = self.hf_config.vocab_size
        margs.moe_latent_size = self.hf_config.moe_latent_size

        # Adjust world size so validation doesn't fail
        margs.world_size = 1
        margs.tensor_model_parallel_size = 1
        margs.pipeline_model_parallel_size = 1
        margs.expert_model_parallel_size = 1
        margs.expert_tensor_parallel_size = 1
        margs.data_parallel_size = 1
        margs.context_parallel_size = 1
        margs.micro_batch_size = 1
        margs.global_batch_size = 1
        margs.virtual_pipeline_model_parallel_size = None

        margs.use_legacy_models = False
        margs.transformer_impl = "local"
        margs.no_persist_layer_norm = True
        margs.use_cpu_initialization = False

        self.margs = margs
        self.checkpoint_args = checkpoint_args

    def build_sys_argv(self):
        """Construct a sys.argv list for Megatron's argument parser."""
        base_args = MegatronCheckpointLoaderBase.build_sys_argv(self)

        hybrid_args = [
            '--position-embedding-type', 'none',
            '--hybrid-override-pattern', self._hybrid_override_pattern,
            '--mamba-state-dim', str(self.hf_config.ssm_state_size),
            '--mamba-num-groups', str(self.hf_config.n_groups),
            '--mamba-head-dim', str(self.hf_config.mamba_head_dim),
            '--mamba-num-heads', str(self.hf_config.mamba_num_heads),
        ]

        super_args = [
            '--moe-latent-size', str(self.hf_config.moe_latent_size),
            '--expert-tensor-parallel-size', '1',
            '--num-experts', str(self.hf_config.n_routed_experts),
            '--moe-shared-expert-intermediate-size', str(self.hf_config.moe_shared_expert_intermediate_size),
            '--moe-router-topk', str(self.hf_config.num_experts_per_tok),
            '--moe-router-enable-expert-bias',
            '--moe-router-score-function', 'sigmoid',
            '--moe-router-topk-scaling-factor', str(self.hf_config.routed_scaling_factor),
            '--squared-relu',
            '--kv-channels', str(self.hf_config.head_dim),
            '--mtp-num-layers', '2',
            '--mtp-use-repeated-layer',
            '--mtp-hybrid-override-pattern', self._mtp_hybrid_override_pattern,
            '--mtp-spec', 'megatron.core.models.mamba.mamba_layer_specs', 'mamba_stack_spec',
            '--keep-mtp-spec-in-bf16',
        ]

        return base_args + hybrid_args + super_args

    def build_checkpoint_metadata(self, true_vocab_size):
        """Construct metadata based on HuggingFace config."""
        md = types.SimpleNamespace()
        md.model_type = "hybrid"
        md.num_layers = self.hf_config.num_hidden_layers
        md.hidden_size = self.hf_config.hidden_size
        md.seq_length = 2048
        md.decoder_seq_length = 16384
        md.num_attention_heads = self.hf_config.num_attention_heads
        md.kv_channels = self.hf_config.head_dim
        md.num_query_groups = self.hf_config.num_key_value_heads
        md.max_position_embeddings = self.hf_config.max_position_embeddings
        md.tokenizer_type = "NullTokenizer"
        md.iteration = 1
        md.params_dtype = self._params_dtype
        md.bert_binary_head = False
        md.output_layer = not self.hf_config.tie_word_embeddings
        md.position_embedding_type = "none"
        md.linear_bias = self.hf_config.use_bias
        md.qkv_bias = self.hf_config.attention_bias
        md.norm_has_bias = False
        md.swiglu = False
        md.previous_tensor_parallel_size = 1
        md.previous_pipeline_parallel_size = 1
        md.true_vocab_size = true_vocab_size or self.hf_config.vocab_size
        md.make_vocab_size_divisible_by = 128
        md.padded_vocab_size = self.hf_config.vocab_size
        md.vocab_size = md.true_vocab_size
        md.language_model_type = "nemotron6-super"
        md.checkpoint_args = self.checkpoint_args
        md.use_legacy_models = False
        md.use_cpu_initialization = False

        # Hybrid-specific metadata
        if self.args.model_type == "hybrid":
            md.hybrid_attention_ratio = None
            md.hybrid_mlp_ratio = None
            md.hybrid_override_pattern = self._hybrid_override_pattern
            md.mamba_state_dim = self.hf_config.ssm_state_size
            md.mamba_num_groups = self.hf_config.n_groups
            md.mamba_head_dim = self.hf_config.mamba_head_dim
            md.mamba_num_heads = self.hf_config.mamba_num_heads

        # MoE metadata
        md.num_experts = self.hf_config.n_routed_experts
        md.moe_router_topk = self.hf_config.num_experts_per_tok
        md.moe_shared_expert_intermediate_size = self.hf_config.moe_shared_expert_intermediate_size
        md.moe_router_topk_scaling_factor = self.hf_config.routed_scaling_factor
        md.moe_router_dtype = None
        md.moe_router_padding_for_fp8 = False
        md.moe_router_num_groups = self.hf_config.n_group
        md.moe_router_group_topk = self.hf_config.num_experts_per_tok
        md.moe_router_pre_softmax = False
        md.moe_router_enable_expert_bias = True
        md.moe_router_score_function = "sigmoid"

        # Super-specific metadata
        md.moe_latent_size = self.hf_config.moe_latent_size

        # MTP metadata
        md.mtp_num_layers = 2
        md.mtp_use_repeated_layer = True
        md.mtp_hybrid_override_pattern = self._mtp_hybrid_override_pattern

        return md

    def send_model_over_queue(self):
        """Send the HuggingFace model over the queue in Megatron format."""
        self.send_metadata_over_queue()
        self.send_hf_super_lm_over_queue()
        self.queue.put("done")

    def send_hf_super_lm_over_queue(self):
        """Convert HuggingFace Super model weights to Megatron format and send over queue."""
        model = self.hf_model

        # 1) Embeddings
        word_embeddings = model.model.embeddings.weight
        message = {"word embeddings": word_embeddings}
        self.queue_put("embeddings", message)

        # 2) Determine layer types by inspecting actual model weights
        layer_types = []
        for i in range(self.hf_config.num_hidden_layers):
            layer_weights = model.model.layers[i].mixer

            if hasattr(layer_weights, 'A_log'):
                layer_types.append('MAMBA')
            elif hasattr(layer_weights, 'q_proj'):
                layer_types.append('ATTENTION')
            elif hasattr(layer_weights, 'up_proj'):
                layer_types.append('MLP')
            elif hasattr(layer_weights, 'gate'):
                layer_types.append('MOE')
            else:
                raise ValueError(f"Couldn't detect layer type of layer {i}")

        # 3) Send transformer layers
        for layer_idx in range(self.hf_config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            layer_type = layer_types[layer_idx]
            message = {}

            if layer_type == 'MAMBA':
                message["in proj norm weight"] = layer.norm.weight
                message["dt bias"] = layer.mixer.dt_bias
                message["D"] = layer.mixer.D
                message["A log"] = layer.mixer.A_log
                message["in proj weight"] = layer.mixer.in_proj.weight
                message["conv1d weight"] = layer.mixer.conv1d.weight
                message["conv1d bias"] = layer.mixer.conv1d.bias
                message["norm weight"] = layer.mixer.norm.weight
                message["out proj weight"] = layer.mixer.out_proj.weight

            elif layer_type == 'ATTENTION':
                message["input norm weight"] = layer.norm.weight

                q_weight = layer.mixer.q_proj.weight
                k_weight = layer.mixer.k_proj.weight
                v_weight = layer.mixer.v_proj.weight
                head_dim = self.hf_config.head_dim
                qkv_weight = self.combine_hf_qkv_weight(
                    q_weight, k_weight, v_weight,
                    self.hf_config.num_attention_heads,
                    self.hf_config.num_key_value_heads,
                    head_dim, self.args.target_tensor_parallel_size)

                message["qkv weight"] = qkv_weight
                message["dense weight"] = layer.mixer.o_proj.weight

                if self.hf_config.attention_bias:
                    q_bias = getattr(layer.mixer.q_proj, 'bias', None)
                    k_bias = getattr(layer.mixer.k_proj, 'bias', None)
                    v_bias = getattr(layer.mixer.v_proj, 'bias', None)
                    if q_bias is not None and k_bias is not None and v_bias is not None:
                        qkv_bias = self.combine_hf_qkv_bias(
                            q_bias, k_bias, v_bias,
                            self.hf_config.num_attention_heads,
                            self.hf_config.num_key_value_heads,
                            head_dim, self.args.target_tensor_parallel_size)
                        message["qkv bias"] = qkv_bias
                if hasattr(layer.mixer.o_proj, 'bias') and layer.mixer.o_proj.bias is not None:
                    message["dense bias"] = layer.mixer.o_proj.bias

            elif layer_type == 'MLP':
                message["post norm weight"] = layer.norm.weight
                message["mlp l0 weight"] = layer.mixer.up_proj.weight
                message["mlp l1 weight"] = layer.mixer.down_proj.weight
                if hasattr(layer.mixer.up_proj, 'bias') and layer.mixer.up_proj.bias is not None:
                    message["mlp l0 bias"] = layer.mixer.up_proj.bias
                if hasattr(layer.mixer.down_proj, 'bias') and layer.mixer.down_proj.bias is not None:
                    message["mlp l1 bias"] = layer.mixer.down_proj.bias

            elif layer_type == 'MOE':
                message["pre mlp norm weight"] = layer.norm.weight
                if hasattr(layer.norm, 'bias') and layer.norm.bias is not None:
                    message["pre mlp norm bias"] = layer.norm.bias
                message["router weight"] = layer.mixer.gate.weight
                message["router bias"] = layer.mixer.gate.e_score_correction_bias
                message["shared mlp l0 weight"] = layer.mixer.shared_experts.up_proj.weight
                message["shared mlp l1 weight"] = layer.mixer.shared_experts.down_proj.weight

                # Latent projections
                if hasattr(layer.mixer, 'fc1_latent_proj'):
                    message["fc1 latent proj weight"] = layer.mixer.fc1_latent_proj.weight
                    message["fc2 latent proj weight"] = layer.mixer.fc2_latent_proj.weight

                experts_up = []
                experts_down = []
                for expert_idx in range(self.hf_config.n_routed_experts):
                    experts_up.append(layer.mixer.experts[expert_idx].up_proj.weight)
                    experts_down.append(layer.mixer.experts[expert_idx].down_proj.weight)
                message["mlp l0 weight"] = torch.stack(experts_up, dim=0)
                message["mlp l1 weight"] = torch.stack(experts_down, dim=0)

            message = {k: v.detach() for k, v in message.items()}
            self.queue_put(f"transformer layer {layer_idx}", message)

        # 4) MTP layers - load directly from safetensors since HF model ignores them
        self.send_mtp_over_queue()

        # 5) Final norm
        message = {"weight": model.model.norm_f.weight}
        self.queue_put("final norm", message)

        # 6) Output layer
        if self.md.output_layer:
            lm_head_weight = model.lm_head.weight
            # The HF model class may leave lm_head.weight as meta;
            # load directly from safetensors if so.
            if lm_head_weight.is_meta:
                from safetensors import safe_open
                index_path = os.path.join(self.args.load_dir, 'model.safetensors.index.json')
                with open(index_path) as f:
                    index = json.load(f)
                shard = index['weight_map']['lm_head.weight']
                with safe_open(os.path.join(self.args.load_dir, shard), framework="pt", device="cpu") as f:
                    lm_head_weight = f.get_tensor('lm_head.weight')
                print(f"Loaded lm_head.weight from safetensors (was meta)")
            message = {"weight": lm_head_weight.detach()}
            self.queue_put("output layer", message)

    def send_mtp_over_queue(self):
        """Load MTP weights from safetensors and send over queue."""
        from safetensors import safe_open

        index_path = os.path.join(self.args.load_dir, 'model.safetensors.index.json')
        with open(index_path) as f:
            index = json.load(f)

        # Collect MTP keys and their shard files
        mtp_keys = {k: v for k, v in index['weight_map'].items() if k.startswith('mtp.')}

        # Group keys by shard
        shards = {}
        for key, shard in mtp_keys.items():
            if shard not in shards:
                shards[shard] = []
            shards[shard].append(key)

        # Load all MTP tensors
        mtp_tensors = {}
        for shard_name, keys in shards.items():
            shard_path = os.path.join(self.args.load_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    mtp_tensors[key] = f.get_tensor(key)

        print(f"Loaded {len(mtp_tensors)} MTP tensors from {len(shards)} shard(s)")

        # Build MTP message
        # The MTP pattern is "*E" (attention + MoE), mapped to 2 HF layers:
        #   mtp.layers.0 = attention layer (the "*")
        #   mtp.layers.1 = MoE layer (the "E")
        # MTP-specific params come from layer 0 (enorm, hnorm, eh_proj) and layer 1 (final_layernorm)
        message = {}

        # MTP-specific params
        message["enorm weight"] = mtp_tensors["mtp.layers.0.enorm.weight"]
        message["hnorm weight"] = mtp_tensors["mtp.layers.0.hnorm.weight"]
        message["eh proj weight"] = mtp_tensors["mtp.layers.0.eh_proj.weight"]
        message["final layernorm weight"] = mtp_tensors["mtp.layers.1.final_layernorm.weight"]

        # Attention sub-layer (mtp.layers.0 = the "*" in "*E")
        message["input norm weight"] = mtp_tensors["mtp.layers.0.norm.weight"]
        q_weight = mtp_tensors["mtp.layers.0.mixer.q_proj.weight"]
        k_weight = mtp_tensors["mtp.layers.0.mixer.k_proj.weight"]
        v_weight = mtp_tensors["mtp.layers.0.mixer.v_proj.weight"]
        head_dim = self.hf_config.head_dim
        qkv_weight = self.combine_hf_qkv_weight(
            q_weight, k_weight, v_weight,
            self.hf_config.num_attention_heads,
            self.hf_config.num_key_value_heads,
            head_dim, self.args.target_tensor_parallel_size)
        message["qkv weight"] = qkv_weight
        message["dense weight"] = mtp_tensors["mtp.layers.0.mixer.o_proj.weight"]

        # MoE sub-layer (mtp.layers.1 = the "E" in "*E")
        message["pre mlp norm weight"] = mtp_tensors["mtp.layers.1.norm.weight"]
        message["router weight"] = mtp_tensors["mtp.layers.1.mixer.gate.weight"]
        message["router bias"] = mtp_tensors["mtp.layers.1.mixer.gate.e_score_correction_bias"]

        # Latent projections
        message["fc1 latent proj weight"] = mtp_tensors["mtp.layers.1.mixer.fc1_latent_proj.weight"]
        message["fc2 latent proj weight"] = mtp_tensors["mtp.layers.1.mixer.fc2_latent_proj.weight"]

        # Shared experts
        message["shared mlp l0 weight"] = mtp_tensors["mtp.layers.1.mixer.shared_experts.up_proj.weight"]
        message["shared mlp l1 weight"] = mtp_tensors["mtp.layers.1.mixer.shared_experts.down_proj.weight"]

        # Stack routed experts
        n_experts = self.hf_config.n_routed_experts
        experts_up = []
        experts_down = []
        for expert_idx in range(n_experts):
            experts_up.append(mtp_tensors[f"mtp.layers.1.mixer.experts.{expert_idx}.up_proj.weight"])
            experts_down.append(mtp_tensors[f"mtp.layers.1.mixer.experts.{expert_idx}.down_proj.weight"])
        message["mlp l0 weight"] = torch.stack(experts_up, dim=0)
        message["mlp l1 weight"] = torch.stack(experts_down, dim=0)

        # Detach all tensors
        message = {k: v.detach() for k, v in message.items()}
        self.queue_put("mtp layer 0", message)


def load_checkpoint(queue, args):
    """
    Required top-level function that creates the loader,
    calls its .load(), and handles exceptions by signaling 'exit'.
    """
    # Install mamba_ssm stubs now — after the saver subprocess has been forked,
    # so we don't pollute its CUDA state.
    _stub_mamba_ssm_if_needed()

    loader = HuggingFaceCheckpointLoaderSuper(args, queue)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e
