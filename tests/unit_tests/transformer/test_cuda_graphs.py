# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import sys

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    TECudaGraphHelper,
    _CudagraphGlobalRecord,
    create_cudagraphs,
)
from megatron.core.transformer.enums import (
    AttnBackend,
    CudaGraphModule,
    CudaGraphScope,
    InferenceCudaGraphScope,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from megatron.training import arguments as training_arguments
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

fp8_available, _ = check_fp8_support()


def _base_cuda_graph_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)


def _validated_cuda_graph_cli_args(monkeypatch, cli_args=None, **overrides):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    warning_messages = []
    print_messages = []

    monkeypatch.setattr(
        training_arguments, "warn_rank_0", lambda msg, *args, **kwargs: warning_messages.append(msg)
    )
    monkeypatch.setattr(
        training_arguments, "print_rank_0", lambda msg, *args, **kwargs: print_messages.append(msg)
    )
    monkeypatch.setattr(sys, "argv", ["test_cuda_graphs.py", *(cli_args or [])])

    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    args = validate_args(args)
    return args, warning_messages, print_messages


class TestCudaGraphConfigAndArguments:
    def test_local_impl_defaults_to_layer_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='local')
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_local_impl_allows_expert_activation_offload_scope(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
            fine_grained_activation_offloading=True,
            offload_modules=['expert_fc1', 'moe_act'],
            num_moe_experts=4,
        )

        assert cfg.cuda_graph_impl == 'local'
        assert CudaGraphModule.attn in cfg.cuda_graph_modules
        assert CudaGraphModule.moe_router in cfg.cuda_graph_modules
        assert CudaGraphModule.moe_preprocess in cfg.cuda_graph_modules

    def test_local_impl_rejects_unsupported_activation_offload_scope(self):
        with pytest.raises(
            AssertionError,
            match=(
                "fine-grained activation offloading with cuda_graph_impl='local'.*"
                "Unsupported offload_modules: \\['qkv_linear'\\]"
            ),
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.attn],
                fine_grained_activation_offloading=True,
                offload_modules=['qkv_linear'],
            )

    def test_local_impl_rejects_full_layer_graph_with_activation_offload(self):
        with pytest.raises(
            AssertionError, match="not supported with whole-layer CUDA graph capture"
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[],
                fine_grained_activation_offloading=True,
                offload_modules=['expert_fc1'],
            )

    def test_local_impl_rejects_moe_router_graph_with_mlp_norm_offload(self):
        with pytest.raises(
            AssertionError,
            match=(
                "fine-grained activation offloading with cuda_graph_impl='local'.*"
                "Unsupported offload_modules: \\['mlp_norm'\\]"
            ),
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.moe_router],
                fine_grained_activation_offloading=True,
                offload_modules=['mlp_norm'],
                num_moe_experts=4,
            )

    def test_full_iteration_impl_requires_empty_scope(self):
        with pytest.raises(
            AssertionError,
            match='cuda_graph_modules must be empty when cuda_graph_impl="full_iteration"',
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='full_iteration', cuda_graph_modules=[CudaGraphModule.attn]
            )

    def test_full_iteration_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration'
            )
        assert cfg.cuda_graph_impl == 'full_iteration'
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration_inference'
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_noops_without_local_impl(self):
        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(cuda_graph_modules='full_iteration_inference')
        assert cfg.cuda_graph_impl == 'none'
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_full_iteration_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration' cannot be combined with "
            "inference_cuda_graph_scope='block'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration',
                inference_cuda_graph_scope='block',
            )

    def test_deprecated_full_iteration_inference_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration_inference',
                inference_cuda_graph_scope='layer',
            )

    def test_enable_cuda_graph_flag_migrates_to_local_impl(self, monkeypatch):
        args, _, print_messages = _validated_cuda_graph_cli_args(
            monkeypatch, ['--enable-cuda-graph']
        )
        assert args.cuda_graph_impl == 'local'
        assert any("--enable-cuda-graph is deprecated" in msg for msg in print_messages)

    def test_full_iteration_inference_scope_cli_migrates_to_block_scope(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch,
            ['--cuda-graph-impl', 'local', '--cuda-graph-modules', 'full_iteration_inference'],
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert args.cuda_graph_modules == []
        assert any(
            "--cuda-graph-modules 'full_iteration_inference' is deprecated" in msg
            for msg in warning_messages
        )

    def test_full_iteration_inference_scope_cli_noops_without_local_impl(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-scope', 'full_iteration_inference']
        )
        assert args.cuda_graph_impl == 'none'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert args.cuda_graph_modules == []
        assert any("has no effect when --cuda-graph-impl=none" in msg for msg in warning_messages)

    def test_full_iteration_inference_scope_cli_rejects_conflicting_new_scope(self, monkeypatch):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _validated_cuda_graph_cli_args(
                monkeypatch,
                [
                    '--cuda-graph-impl',
                    'local',
                    '--cuda-graph-modules',
                    'full_iteration_inference',
                    '--inference-cuda-graph-scope',
                    'layer',
                ],
            )

    def test_new_scope_cli_accepts_block(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'block']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block

    def test_new_scope_cli_accepts_layer(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'layer']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_removed_module_scoped_scope_name_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                'test_cuda_graphs.py',
                '--cuda-graph-impl',
                'local',
                '--inference-cuda-graph-scope',
                'module_scoped',
            ],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_removed_old_inference_bool_flag_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys, "argv", ['test_cuda_graphs.py', '--inference-use-full-iteration-cuda-graph']
        )
        with pytest.raises(SystemExit):
            parse_args()

    # --- Backward compat: cuda_graph_scope → cuda_graph_modules rename ---

    def test_deprecated_cuda_graph_scope_kwarg_migrates_to_modules(self):
        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=['attn'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_new_cuda_graph_modules_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_modules=['attn', 'mlp'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn, CudaGraphModule.mlp]
        assert cfg.cuda_graph_scope is None

    def test_new_full_iteration_impl_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='full_iteration', cuda_graph_modules=[])
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_cli_migrates_to_modules(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--cuda-graph-scope', 'attn']
        )
        assert args.cuda_graph_modules == [CudaGraphModule.attn]
        assert any('--cuda-graph-scope is deprecated' in msg for msg in warning_messages)

    def test_cuda_graph_scope_is_standalone_class_for_pickle_compat(self):
        from megatron.core.transformer.enums import CudaGraphScope

        # CudaGraphScope is preserved as a standalone class (not an alias) so that
        # pre-refactor checkpoints can be deserialized without value-collision errors.
        assert CudaGraphScope is not CudaGraphModule
        assert CudaGraphScope.attn.value == 2  # original ordinals preserved
        assert CudaGraphScope.mamba.value == 7

    def test_cuda_graph_scope_and_inference_scope_in_safe_globals(self):
        from megatron.core.safe_globals import SAFE_GLOBALS
        from megatron.core.transformer.enums import CudaGraphScope

        assert CudaGraphScope in SAFE_GLOBALS
        assert InferenceCudaGraphScope in SAFE_GLOBALS

    def test_deprecated_cuda_graph_scope_enum_instance_migrates_to_modules(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.attn])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_enum_migrates_to_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.full_iteration])
        assert cfg.cuda_graph_impl == "full_iteration"
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_enum_migrates_to_scope(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl="local", cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_noops_without_local_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(
                cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.cuda_graph_impl == "none"
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None


class TestParallelTransformerBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

        # initialize transformer model
        num_layers = 8
        hidden_size = 64
        self.transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_layer_with_transformer_engine_spec()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_transformer_block = self.parallel_transformer_block
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_transformer_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_transformer_block.layers[0], "cudagraph_manager")
            assert (
                len(parallel_transformer_block.layers[0].cudagraph_manager.cudagraph_runners) == 1
            )
            del (
                parallel_transformer_block.layers[_]
                .cudagraph_manager.cudagraph_runners[0]
                .fwd_graph
            )


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestPackedSeqCudagraphs:
    """Training CUDA graphs over thd input with padding between sequences.

    The padded cu_seqlens describe a slot layout that differs from the actual lengths,
    and pad_between_seqs is set explicitly so TE does spend a GPU sync inferring it.
    cp_size == 2 additionally captures TE's ring-P2P context-parallel attention inside the graphs.
    """

    SEQ_LENGTHS = [7, 5]
    SLOT_STARTS = [0, 8, 16]  # slot layout aligned to 2 * cp_size for every cp_size tested
    BIN_SIZE = 32
    NVTE_ENV_VARS = (
        "NVTE_FLASH_ATTN",
        "NVTE_FUSED_ATTN",
        "NVTE_UNFUSED_ATTN",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    )

    def setup_method(self, method):
        self.original_nvte_env = {name: os.environ.get(name) for name in self.NVTE_ENV_VARS}
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
            _CudagraphGlobalRecord.cudagraph_created = False
            _CudagraphGlobalRecord.cudagraph_record = []
            CudaGraphManager.global_mempool = None
        finally:
            for name, value in self.original_nvte_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _build_packed_seq_params(self, device):
        # Actual boundaries: each sequence's real tokens inside its slot; the trailing bin
        # padding [SLOT_STARTS[-1], BIN_SIZE) forms a ghost slot of pad tokens.
        boundaries = [0]
        for length in self.SEQ_LENGTHS:
            boundaries.append(boundaries[-1] + length)
        boundaries.append(boundaries[-1] + self.BIN_SIZE - self.SLOT_STARTS[-1])
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
        cu_seqlens_padded = torch.tensor(
            self.SLOT_STARTS + [self.BIN_SIZE], dtype=torch.int32, device=device
        )
        return PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=self.BIN_SIZE,
            max_seqlen_kv=self.BIN_SIZE,
            pad_between_seqs=True,
        )

    @pytest.mark.parametrize("cp_size", [1, 2])
    def test_thd_capture_with_pad_between_seqs(self, cp_size):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        model_parallel_cuda_manual_seed(123)
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=cp_size,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            attention_backend=AttnBackend.fused,
            deterministic_mode=True,
            cuda_graph_impl="local",
            cuda_graph_warmup_steps=1,
            use_cpu_initialization=True,
        )
        block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec()).cuda()
        block.train()
        # CUDA-graphed backward assumes DDP-style grad accumulation buffers.
        for param in block.parameters():
            param.main_grad = torch.zeros_like(param)

        packed_seq_params = self._build_packed_seq_params(torch.device('cuda'))
        # Each CP rank holds its 1/cp_size share of the bin's tokens.
        hidden_states = torch.randn(
            (self.BIN_SIZE // cp_size, 1, config.hidden_size),
            dtype=torch.bfloat16,
            device='cuda',
            requires_grad=True,
        )

        eager_out = block(
            hidden_states=hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        hidden_states_metadata = hidden_states.cg_buffer_metadata
        assert hidden_states_metadata.is_cudagraph_input
        assert hidden_states_metadata.is_saved_for_backward

        # The second layer's TE input layernorm saves the first layer's output for backward.
        # This naturally exercises a CUDA graph output whose pool buffer must stay alive until
        # backward capture.
        first_runner = block.layers[0].cudagraph_manager.cudagraph_runners[0]
        first_runner_record = next(
            record
            for record in _CudagraphGlobalRecord.cudagraph_record
            if record[0] is first_runner and record[1] == "fwd"
        )
        recorded_outputs = first_runner_record[4]
        output_metadata = first_runner.get_arg_metas(recorded_outputs)[0].cg_buffer_metadata
        output_metadata_state = (
            f"input={output_metadata.is_cudagraph_input}, "
            f"output={output_metadata.is_cudagraph_output}, "
            f"saved={output_metadata.is_saved_for_backward}"
        )
        assert output_metadata.is_cudagraph_input, output_metadata_state
        assert output_metadata.is_cudagraph_output, output_metadata_state
        assert output_metadata.is_saved_for_backward, output_metadata_state

        # The q/kv aliases for each offsets tensor must share one metadata object while recording
        # every graph-input use for replay-buffer sharing.
        actual_cu_seqlens_metadata = packed_seq_params.cu_seqlens_q.cg_buffer_metadata
        padded_cu_seqlens_metadata = packed_seq_params.cu_seqlens_q_padded.cg_buffer_metadata
        assert packed_seq_params.cu_seqlens_kv.cg_buffer_metadata is actual_cu_seqlens_metadata
        assert (
            packed_seq_params.cu_seqlens_kv_padded.cg_buffer_metadata is padded_cu_seqlens_metadata
        )
        assert actual_cu_seqlens_metadata.is_cudagraph_input
        assert padded_cu_seqlens_metadata.is_cudagraph_input
        eager_out.sum().backward()

        # This is the primary function under test.
        create_cudagraphs()

        runners = []
        for layer in block.layers:
            layer_runners = layer.cudagraph_manager.cudagraph_runners
            assert len(layer_runners) == 1
            assert layer_runners[0].fwd_graph is not None
            runners.extend(layer_runners)

        # There are four cu_seqlens arguments per layer: q/kv pairs for the real and padded
        # offsets. Each pair and every later layer should alias one of two shared buffers. Within
        # each buffer group, only its first graph-input occurrence performs the replay copy.
        cu_seqlens_buffers = [
            tensor
            for runner in runners
            for tensor in runner.fwd_graph_input_surface[: runner.num_dgrads]
            if tensor.dtype == torch.int32 and tensor.shape == packed_seq_params.cu_seqlens_q.shape
        ]
        assert len(cu_seqlens_buffers) == 4 * len(runners)
        buffers_by_ptr = {}
        for tensor in cu_seqlens_buffers:
            buffers_by_ptr.setdefault(tensor.data_ptr(), []).append(tensor)
        assert len(buffers_by_ptr) == 2
        for shared_buffers in buffers_by_ptr.values():
            assert sum(not tensor.can_skip_replay_copy for tensor in shared_buffers) == 1

        graphed_out = block(
            hidden_states=hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        assert torch.equal(graphed_out, eager_out), (
            "CUDA graph replay output is not bitwise equal to eager output: "
            f"max_abs_diff={(graphed_out.float() - eager_out.float()).abs().max().item()}"
        )
        graphed_out.sum().backward()

        # Destroy captured graphs deterministically before parallel-state teardown.
        for layer in block.layers:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph
        torch.cuda.synchronize()


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
@pytest.mark.parametrize(
    "total_num_layers, pp, vpp, account_for_embedding_in_pipeline_split, account_for_loss_in_pipeline_split, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage, pp_layout, first_layer_numbers_golden, last_layer_numbers_golden",
    [
        (4, 1, None, False, False, None, None, None, [1], [4]),
        (8, 2, None, False, False, None, None, None, [1, 5], [4, 8]),
        (8, 2, 2, False, False, None, None, None, [1, 3, 5, 7], [2, 4, 6, 8]),
        (14, 4, None, True, True, None, None, None, [1, 4, 8, 12], [3, 7, 11, 14]),
        (
            14,
            4,
            2,
            True,
            True,
            None,
            None,
            None,
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
        (12, 4, None, False, False, 2, 2, None, [1, 3, 7, 11], [2, 6, 10, 12]),
        (
            12,
            4,
            2,
            False,
            False,
            2,
            2,
            None,
            [1, 2, 4, 6, 7, 8, 10, 12],
            [1, 3, 5, 6, 7, 9, 11, 12],
        ),
        (
            14,
            4,
            2,
            False,
            False,
            None,
            None,
            [
                ["embedding", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "loss"],
            ],
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
    ],
)
def test_cuda_graph_determine_first_last_layer_logic(
    total_num_layers,
    pp,
    vpp,
    account_for_embedding_in_pipeline_split,
    account_for_loss_in_pipeline_split,
    num_layers_in_first_pipeline_stage,
    num_layers_in_last_pipeline_stage,
    pp_layout,
    first_layer_numbers_golden,
    last_layer_numbers_golden,
):
    # Initialize RNG tracker
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    # Initialize parallel state
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=vpp
    )

    # initialize model
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    hidden_size = 128
    transformer_config = TransformerConfig(
        num_layers=total_num_layers,
        hidden_size=hidden_size,
        num_attention_heads=1,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_size=pp,
        deallocate_pipeline_outputs=True,
        cuda_graph_impl="local",
        use_te_rng_tracker=True,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        pipeline_model_parallel_layout=pp_layout,
    )
    model = []
    for i in range(vpp or 1):
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=1024,
            position_embedding_type="rope",
            vp_stage=i,
        ).cuda()
        model.append(this_model)

    # create runner by running a fake forward pass
    sequence_length, micro_batch_size = 32, 1
    hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size)).cuda()
    attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
    for m in model:
        _ = m(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=hidden_states,
        )

    # Check if cuda graph is correctly setting is first/last layer
    for m in model:
        for l in m.decoder.layers:
            assert hasattr(l, "cudagraph_manager")
            assert (
                len(l.cudagraph_manager.cudagraph_runners) == 1
            ), "Cuda graph runner should be created"
            runner = l.cudagraph_manager.cudagraph_runners[0]
            assert runner.is_first_layer is not None and runner.is_last_layer is not None
            assert runner.is_first_layer == (l.layer_number in first_layer_numbers_golden)
            assert runner.is_last_layer == (l.layer_number in last_layer_numbers_golden)

            del l.cudagraph_manager.cudagraph_runners[0].fwd_graph

    # Destroy all captured graphs deterministically
    for m in model:
        for l in m.decoder.layers:
            for runner in getattr(l.cudagraph_manager, "cudagraph_runners", []):
                # Safely delete both graphs if present
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph

    # Ensure all pending work is complete and graph destruction runs now
    torch.cuda.synchronize()

    # Teardown
    Utils.destroy_model_parallel()
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    CudaGraphManager.global_mempool = None
    CudaGraphManager.fwd_mempools = None
    CudaGraphManager.bwd_mempools = None


class TestLLaVACudaGraph:
    """Test CUDA graphs with LLaVA model focusing on is_last_layer logic for encoder/decoder transitions."""

    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )
        model_parallel_cuda_manual_seed(123)

        from copy import deepcopy

        from megatron.core.models.multimodal.llava_model import LLaVAModel
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec,
        )

        # Create language transformer config with CUDA graphs enabled
        self.language_hidden_size = 64
        self.language_num_attention_heads = 4
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs
        )

        # Create vision transformer config
        vision_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs for vision model too
        )

        # Create vision projection config
        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )

        # Get layer specs
        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)

        # Set vision model type
        vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        # Create LLaVA model with both encoder and decoder
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_llava_cudagraph_is_last_layer_logic(self):
        """Test that is_last_layer logic correctly resets prev_bwd_hidden_state_inputgrad for LLaVA models."""

        # Move model to CUDA
        self.llava_model.cuda()
        # Cudagraph backward capture assumes the model has DDP so create main_grads for params
        for param in self.llava_model.parameters():
            param.main_grad = torch.zeros_like(param)

        set_current_microbatch(self.llava_model.vision_model, 1)
        set_current_microbatch(self.llava_model.language_model, 1)

        # Create test inputs
        batch_size = 2
        seq_length = 1024
        num_images = 1

        images = torch.ones((num_images, 3, 336, 336), dtype=torch.float32).cuda()

        # Create text input with image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        # Insert image token (using default image token index)
        input_ids[0, 5] = self.llava_model.image_token_index

        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = None

        # Create labels and loss mask for training
        labels = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float32).cuda()

        # Create num_image_tiles
        num_image_tiles = torch.ones(num_images, dtype=torch.int).cuda()

        # First forward pass - this should record the CUDA graphs
        output1, loss_mask1 = self.llava_model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # Verify that CUDA graph managers were created
        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Vision model layers should have CUDA graph managers"

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Language model layers should have CUDA graph managers"

                    # Verify that CUDA graphs were created successfully
                    for runner in layer.cudagraph_manager.cudagraph_runners:
                        assert hasattr(runner, 'fwd_graph')
                        assert hasattr(runner, 'bwd_graph')

        # Perform backward pass to trigger backward graph recording
        if isinstance(output1, tuple):
            loss = output1[0].sum()
        else:
            loss = output1.sum()
        loss.backward()

        # Import the CUDA graph creation function
        from megatron.core.transformer.cuda_graphs import create_cudagraphs

        # Create the CUDA graphs - this is where the is_last_layer logic is tested
        create_cudagraphs()

        # Verify that CUDA graphs were created successfully
        assert _CudagraphGlobalRecord.cudagraph_created, "CUDA graphs should be created"

        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph


class TestParallelHybridBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        # Ensure that this test is capturing to a fresh memory pool.
        CudaGraphManager.global_mempool = None

        def get_pg_collection():
            return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

        def get_mamba_block(hybrid_layer_pattern):
            layer_type_list = validate_segment_layers(hybrid_layer_pattern)
            transformer_config = TransformerConfig(
                hidden_size=256,  # The Mamba layer places several constraints on this
                # Need to specify num_attention_heads and num_layers or TransformerConfig
                # will generate errors.
                num_layers=len(layer_type_list),
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="local",
            )
            modules = hybrid_stack_spec.submodules
            return HybridStack(
                transformer_config,
                modules,
                layer_type_list=layer_type_list,
                pp_layer_offset=0,
                pg_collection=get_pg_collection(),
            )

        self.mamba_block = get_mamba_block(hybrid_layer_pattern="M-M*-")
        self.transformer_config = self.mamba_block.config

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_mamba_block = self.mamba_block
        parallel_mamba_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_mamba_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_mamba_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_mamba_block.layers[0], "cudagraph_manager")
            assert len(parallel_mamba_block.layers[0].cudagraph_manager.cudagraph_runners) == 1

            del parallel_mamba_block.layers[_].cudagraph_manager.cudagraph_runners[0].fwd_graph


# Global storage for comparing unique buffer counts across different num_microbatches,
# keyed by (pp_size, vpp_size)
_unique_buffer_counts = {}


class TestTECudaGraphHelper:
    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        # Note: _unique_buffer_counts is intentionally NOT cleared here so we can
        # compare values across parametrized test runs

    @pytest.mark.parametrize("num_microbatches", [16, 64, 256])
    @pytest.mark.parametrize("pp_size", [1, 2, 4])
    @pytest.mark.parametrize("vpp_size", [None, 2])
    def test_get_cuda_graph_input_data(self, num_microbatches, pp_size, vpp_size):
        """Test _get_cuda_graph_input_data function in TECudaGraphHelper."""

        if vpp_size and pp_size == 1:
            pytest.skip("vpp_size must be None when pp_size is 1")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
        )

        # Set up test configuration
        seq_length = 128
        micro_batch_size = 2
        num_layers = 8
        vocab_size = 1024
        hidden_size = 64
        num_attention_heads = 4

        # Initialize num_microbatches calculator
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=micro_batch_size * num_microbatches,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        # Create transformer config directly
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            use_te_rng_tracker=True,
            bf16=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
            pipeline_dtype=torch.bfloat16,
            context_parallel_size=1,
        )

        # Create model
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        model = []
        for i in range(vpp_size or 1):
            this_model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=vocab_size,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
                vp_stage=i if vpp_size else None,
            ).cuda()
            model.append(this_model)

        # Initialize TECudaGraphHelper
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )

        # Call _get_cuda_graph_input_data (which internally calls _get_sample_arguments)
        sample_args, make_graphed_callables_kwargs = (
            cuda_graph_helper._get_cuda_graph_input_data(num_microbatches=num_microbatches)
        )

        # Extract sample_kwargs from the kwargs dict
        # For TE >= 1.10.0, sample_kwargs should always be present
        assert (
            'sample_kwargs' in make_graphed_callables_kwargs
        ), "sample_kwargs should be present in make_graphed_callables_kwargs for TE >= 1.10.0"
        sample_kwargs = make_graphed_callables_kwargs['sample_kwargs']

        # Basic checks
        num_graphable_layers = len(cuda_graph_helper.flattened_callables)
        if pp_size > 1:
            expected_length = num_graphable_layers * num_microbatches
        else:
            expected_length = num_graphable_layers
        assert len(sample_args) == expected_length, (
            f"sample_args length mismatch: expected {expected_length}, " f"got {len(sample_args)}"
        )
        assert len(sample_kwargs) == expected_length, (
            f"sample_kwargs length mismatch: expected {expected_length}, "
            f"got {len(sample_kwargs)}"
        )

        # Check that all elements are not None
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            assert args_item is not None, f"sample_args[{i}] is None"
            assert kwargs_item is not None, f"sample_kwargs[{i}] is None"
            assert isinstance(args_item, tuple), f"sample_args[{i}] should be a tuple"
            assert isinstance(kwargs_item, dict), f"sample_kwargs[{i}] should be a dict"
            assert len(args_item) > 0, f"sample_args[{i}] should not be empty"
            # Check that hidden_states is present
            assert "hidden_states" in kwargs_item or (
                len(args_item) > 0 and torch.is_tensor(args_item[0])
            ), f"sample_args[{i}] or sample_kwargs[{i}] should contain hidden_states"

        # Check tensor properties
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get hidden_states from args or kwargs
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                hidden_states = args_item[0]
            elif "hidden_states" in kwargs_item:
                hidden_states = kwargs_item["hidden_states"]
            else:
                continue

            assert torch.is_tensor(hidden_states), f"hidden_states at index {i} should be a tensor"
            # Check shape matches expected (accounting for TP/CP)
            expected_seq_len = seq_length // transformer_config.context_parallel_size
            if transformer_config.sequence_parallel:
                expected_seq_len = expected_seq_len // transformer_config.tensor_model_parallel_size
            assert hidden_states.shape[0] == expected_seq_len, (
                f"hidden_states seq_len mismatch at index {i}: "
                f"expected {expected_seq_len}, got {hidden_states.shape[0]}"
            )
            assert hidden_states.shape[1] == micro_batch_size, (
                f"hidden_states batch_size mismatch at index {i}: "
                f"expected {micro_batch_size}, got {hidden_states.shape[1]}"
            )
            assert hidden_states.shape[2] == transformer_config.hidden_size, (
                f"hidden_states hidden_size mismatch at index {i}: "
                f"expected {transformer_config.hidden_size}, got {hidden_states.shape[2]}"
            )

        # Memory optimization check: verify that buffers with same signature are reused
        # Create a mapping of sample_keys to indices
        sample_keys_to_indices = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Create sample_keys similar to the function
            args_keys = tuple((t.shape, t.dtype, t.layout) for t in args_item if torch.is_tensor(t))
            kwargs_keys = tuple(
                (k, v.shape, v.dtype, v.layout)
                for k, v in sorted(kwargs_item.items())
                if torch.is_tensor(v)
            )
            sample_keys = args_keys + kwargs_keys

            if sample_keys not in sample_keys_to_indices:
                sample_keys_to_indices[sample_keys] = []
            sample_keys_to_indices[sample_keys].append(idx)

        # Check that buffers with same signature share references (memory optimization)
        # The optimization reuses buffers when:
        # 1. They have the same signature (shape, dtype, layout)
        # 2. The backward pass of the original buffer has completed
        # 3. A new forward pass with matching signature needs a buffer
        # Count how many times each tensor is reused
        unique_tensors = set()
        tensor_reuse_count = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get the first tensor from args (hidden_states)
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                tensor_ptr = args_item[0].data_ptr()
                unique_tensors.add(tensor_ptr)
                tensor_reuse_count[tensor_ptr] = tensor_reuse_count.get(tensor_ptr, 0) + 1

        # With memory optimization, we should see some buffers reused
        # (i.e., some tensors should appear multiple times)
        max_reuse = max(tensor_reuse_count.values()) if tensor_reuse_count else 0
        total_entries = len(sample_args)
        unique_buffer_count = len(unique_tensors)

        # Verify that memory optimization is working:
        # - The number of unique buffers should be <= total entries
        # - With the 1F1B schedule and multiple microbatches, we should see some buffer reuse
        # - The number of unique buffers should be bounded as num_microbatches grows.
        assert unique_buffer_count <= total_entries, (
            f"Memory optimization check: unique_buffer_count ({unique_buffer_count}) "
            f"should be <= total_entries ({total_entries})"
        )
        global _unique_buffer_counts
        # Use (pp_size, vpp_size) as key to track unique buffer counts per configuration
        config_key = (pp_size, vpp_size)
        if config_key not in _unique_buffer_counts:
            _unique_buffer_counts[config_key] = unique_buffer_count
        else:
            assert unique_buffer_count == _unique_buffer_counts[config_key], (
                f"Unique buffer count mismatch: expected {_unique_buffer_counts[config_key]}, "
                f"got {unique_buffer_count}"
            )

        # Verify that buffers with the same signature can potentially be reused
        # (the actual reuse depends on the schedule, but the mechanism should work)
        if expected_length > 1:
            # Check that we have multiple entries with the same signature
            has_duplicate_signatures = any(
                len(indices) > 1 for indices in sample_keys_to_indices.values()
            )
            assert has_duplicate_signatures, (
                "Memory optimization: expected duplicate signatures for buffer reuse, "
                "but all signatures are unique"
            )

            # We tested with a large number of microbatches, so we should see some buffer reuse.
            if pp_size > 1:
                assert max_reuse > 1, "Expected some buffer reuse"

        # Verify that make_graphed_callables_kwargs contains expected keys
        assert (
            '_order' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain '_order'"
        assert (
            'num_warmup_iters' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'num_warmup_iters'"
        assert (
            'allow_unused_input' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'allow_unused_input'"

        # Verify the order in kwargs matches expectations
        order = make_graphed_callables_kwargs['_order']
        num_model_chunks = cuda_graph_helper.num_model_chunks
        forward_count = sum(1 for chunk_id in order if chunk_id > 0)
        if pp_size > 1:
            # Verify that all forward passes in order have corresponding entries in sample_args
            assert forward_count == num_microbatches * num_model_chunks, (
                f"Forward count mismatch: expected {num_microbatches * num_model_chunks}, "
                f"got {forward_count}"
            )
            expected_order_length = num_microbatches * num_model_chunks * 2
        else:
            assert num_model_chunks == 1, "Expected only one model chunk for pp_size == 1"
            assert forward_count == 1, "Expected only one forward pass for pp_size == 1"
            expected_order_length = 2
        assert (
            len(order) == expected_order_length
        ), f"Order length mismatch: expected {expected_order_length}, got {len(order)}"


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def is_nccl_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    return HAVE_TE_EP


class TestPartialCudaGraph:
    """Test that CUDA graph outputs match non-CUDA graph outputs for various scopes."""

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.tp_size = 2
        self.cp_size = 2
        self.cuda_graph_helper = None
        # Store original environment variable values
        self.original_env = {
            'CUDA_DEVICE_MAX_CONNECTIONS': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
            'NVTE_ALLOW_NONDETERMINISTIC_ALGO': os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO'),
        }
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'

    def teardown_method(self, method):
        # Restore original environment variable values
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_decoder_block_spec,
        **config_kwargs,
    ):
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            config,
            use_transformer_engine=True,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
        )
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=True
            )
        else:
            mtp_block_spec = None
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            mtp_block_spec=mtp_block_spec,
        )

    def create_test_args(
        self, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_cuda_graphs.py']
        args = parse_args()
        args.num_layers = 4
        args.mtp_num_layers = 1
        args.vocab_size = 1024
        args.hidden_size = 512
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.global_batch_size = self.micro_batch_size * 8 // self.tp_size // self.cp_size
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = self.tp_size
        args.sequence_parallel = True if self.tp_size > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = self.cp_size
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = True
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0

        # MoE settings
        args.num_experts = 4
        args.expert_model_parallel_size = ep_size
        args.expert_tensor_parallel_size = 1 if ep_size > 1 else self.tp_size
        args.moe_shared_expert_intermediate_size = 1024
        args.moe_layer_freq = [0, 0, 1, 1]
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2
        args.moe_router_dtype = "fp32"

        # CUDA graph settings
        args.cuda_graph_impl = cuda_graph_impl
        args.cuda_graph_modules = cuda_graph_modules
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps

        # fp8 settings
        if fp8_available:
            args.fp8 = "e4m3"
            args.fp8_recipe = "tensorwise"
            args.first_last_layers_bf16 = True
            args.num_layers_at_start_in_bf16 = 1
            args.num_layers_at_end_in_bf16 = 1

        for key, value in kwargs.items():
            assert hasattr(args, key)
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size, cp_size):
        data = list(range(seq_length // cp_size))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length // cp_size, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length // cp_size).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_test_helper(
        self, ep_size, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, **kwargs
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
        )

        set_args(args)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size, self.cp_size
        )

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        if cuda_graph_impl == "transformer_engine":
            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        loss_list = []

        for i in range(100):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            # Capture CUDA graphs after warmup if helper is provided
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                self.cuda_graph_helper.create_cudagraphs()

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length // self.cp_size

            # Verify gradients
            loss = output.mean()
            loss.backward()

            for param in gpt_model[0].parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        return torch.tensor(loss_list)

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    @pytest.mark.parametrize("moe_dropless_dispatcher", [False, True])
    @pytest.mark.parametrize("moe_dispatcher_type", ["alltoall", "deepep", "hybridep", "ncclep"])
    def test_moe_partial_cudagraph(self, ep_size, moe_dropless_dispatcher, moe_dispatcher_type):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {}
        if moe_dispatcher_type == "deepep":
            if not is_deep_ep_available():
                pytest.skip("Deep EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
        elif moe_dispatcher_type == "hybridep":
            if not is_hybrid_ep_available():
                pytest.skip("Hybrid EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "hybridep"
        elif moe_dispatcher_type == "ncclep":
            if not is_nccl_ep_available():
                pytest.skip("NCCL EP is not available")
            if ep_size < 2:
                pytest.skip("NCCL EP requires expert_model_parallel_size >= 2 (ep_bootstrap)")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "ncclep"
            # ncclep sizes a per-rank recv buffer from this and overflow hard-traps; size generously.
            extra_kwargs["moe_expert_rank_capacity_factor"] = 8.0
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if not moe_dropless_dispatcher:
            if moe_dispatcher_type in ("deepep", "ncclep"):
                pytest.skip(f"{moe_dispatcher_type} doesn't support drop&pad MoE")
            extra_kwargs["moe_expert_capacity_factor"] = 1.0
            extra_kwargs["moe_pad_expert_input_to_capacity"] = True

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            None,
            [CudaGraphModule.attn],
            [CudaGraphModule.moe],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            if (moe_dropless_dispatcher or moe_dispatcher_type in ("hybridep", "ncclep")) and (
                cuda_graph_modules is None or CudaGraphModule.moe in cuda_graph_modules
            ):
                # Dropless MoE or a dynamic-shape flex backend (Hybrid EP / NCCL EP) can't be
                # captured at the "moe" scope (the dispatch does a device-to-host sync). Skip;
                # the surrounding compute submodules are still graphed.
                continue
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref)

        if moe_dispatcher_type == "hybridep":
            reset_hybrid_ep_buffer()
        if moe_dispatcher_type == "ncclep":
            from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize

            nccl_ep_finalize()
        Utils.destroy_model_parallel()


class _SimpleModule(MegatronModule):
    """Minimal MegatronModule for testing CudaGraphManager with function_name."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def my_op(self, x):
        return self.linear(x)


class _SimpleNonModule:
    """non-nn.Module base_module for testing the function_name= form of `CudaGraphManager`."""

    def __init__(self, config):
        self.weight = torch.randn(config.hidden_size, config.hidden_size, device="cuda")

    def my_op(self, x):
        return x @ self.weight


def test_moe_replay_state_is_paired_with_exact_graph_index() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_expert_capacity_factor=None,
        moe_expert_rank_capacity_factor=None,
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    dispatcher.tp_size = 1
    dispatcher.ep_size = 1
    dispatcher.router_topk = 2
    dispatcher.num_experts = 4
    dispatcher.num_local_experts = 4
    dispatcher.drop_and_pad = False

    layer = MoETransformerLayer.__new__(MoETransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.mlp = SimpleNamespace(token_dispatcher=dispatcher)
    layer._te_cuda_graph_dispatcher_replay_states = ()

    graph_input_0 = torch.empty((2, 1, 4))
    dispatcher.hidden_shape = graph_input_0.shape
    dispatcher.hidden_shape_before_permute = torch.Size((2, 4))
    dispatcher.capacity = None
    dispatcher.num_out_tokens = 4
    layer._record_te_cuda_graph_dispatcher_replay_state(0, graph_input_0, torch.empty((4, 4)))

    graph_input_1 = torch.empty((3, 1, 4))
    dispatcher.hidden_shape = graph_input_1.shape
    dispatcher.hidden_shape_before_permute = torch.Size((3, 4))
    dispatcher.capacity = None
    dispatcher.num_out_tokens = 6
    layer._record_te_cuda_graph_dispatcher_replay_state(1, graph_input_1, torch.empty((6, 4)))

    state_0 = layer._restore_te_cuda_graph_dispatcher_replay_state(
        0, graph_input_0, torch.empty((4, 4))
    )
    assert dispatcher.hidden_shape == graph_input_0.shape
    assert dispatcher.num_out_tokens == 4
    assert state_0 is layer._te_cuda_graph_dispatcher_replay_states[0]

    state_1 = layer._restore_te_cuda_graph_dispatcher_replay_state(
        1, graph_input_1, torch.empty((6, 4))
    )
    assert dispatcher.hidden_shape == graph_input_1.shape
    assert dispatcher.num_out_tokens == 6
    assert state_1 is layer._te_cuda_graph_dispatcher_replay_states[1]
    assert len(layer._te_cuda_graph_dispatcher_replay_states) == 2


def test_te_graph_bank_guard_runs_before_forward_and_backward_selection() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.module import GraphableMegatronModule

    selections = []

    class _Graph:
        def __call__(self, *args, **kwargs):
            selections.append("forward")

        def backward_dw(self):
            selections.append("backward")

    def reject(layer, graphs, microbatch_index):
        raise ValueError("runtime num_microbatches mismatch")

    layer = SimpleNamespace(
        cuda_graphs=[_Graph()],
        cuda_graph_manual_hooks=[],
        current_microbatch=0,
        _te_cuda_graph_bank_replay_guard=reject,
        _get_te_cuda_graph_replay_args=lambda *args, **kwargs: (args, kwargs),
    )
    with pytest.raises(ValueError, match="runtime num_microbatches"):
        GraphableMegatronModule._te_cuda_graph_replay(layer, torch.empty(1))
    with pytest.raises(ValueError, match="runtime num_microbatches"):
        GraphableMegatronModule._te_cuda_graph_backward_dw_graph(layer, 0)
    assert selections == []


def test_te_graph_bank_guard_index_is_used_for_forward_and_backward() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.module import GraphableMegatronModule

    selections = []

    class _Graph:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args, **kwargs):
            selections.append(f"forward-{self.name}")
            return self.name

        def backward_dw(self):
            selections.append(f"backward-{self.name}")

    graphs = [_Graph("zero"), _Graph("one")]
    layer = SimpleNamespace(
        cuda_graphs=graphs,
        cuda_graph_manual_hooks=[],
        current_microbatch=3,
        _get_te_cuda_graph_replay_args=lambda *args, **kwargs: (args, kwargs),
    )
    manager, bank, _ = _make_task9_active_bank(layer, graphs)

    assert GraphableMegatronModule._te_cuda_graph_replay(layer, torch.empty(1)) == "one"
    GraphableMegatronModule._te_cuda_graph_backward_dw_graph(layer, 3)
    assert selections == ["forward-one", "backward-one"]
    bank.reset()
    manager.close()


class _Task9ReplayGraph:
    def __init__(self, *, tuple_output: bool) -> None:
        self.tuple_output = tuple_output
        self.fail_launch = False
        self.calls = 0
        self.backward_dw_calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.fail_launch:
            raise RuntimeError("graph launch failed")
        output = args[0]
        return (output,) if self.tuple_output else output

    def backward_dw(self):
        self.backward_dw_calls += 1


def _make_task9_active_bank(layer, graph, *, cuda_graph_modules=(), setup=None):
    from types import SimpleNamespace

    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    graphs = list(graph) if isinstance(graph, (list, tuple)) else [graph]
    runtime = {"count": len(graphs)}
    manager = TECudaGraphBankManager(
        [layer],
        cuda_graph_modules=cuda_graph_modules,
        graph_reset_supported=False,
        synchronize=lambda: None,
        runtime_num_microbatches=lambda: runtime["count"],
    )
    helper = SimpleNamespace(
        flattened_callables=[layer],
        config=SimpleNamespace(cuda_graph_modules=cuda_graph_modules),
        num_microbatches=None,
        _capture_attempted=False,
        _capture_finished=False,
        _graphs_created=False,
    )

    def capture(*, num_microbatches):
        assert num_microbatches == len(graphs)
        if setup is not None:
            setup()
        layer.cuda_graphs.extend(graphs)
        helper.num_microbatches = num_microbatches
        helper._capture_finished = True
        helper._graphs_created = True
        return ((layer, tuple(layer.cuda_graphs)),)

    helper._capture_cuda_graph_lists = capture
    layer.assert_te_cuda_graph_bank_drained = lambda: None
    layer.snapshot_te_cuda_graph_bank_references = lambda: None
    layer.restore_te_cuda_graph_bank_references = lambda _snapshot: None
    layer.clear_te_cuda_graph_bank_references = lambda: None
    layer.te_cuda_graph_bank_schema = lambda: ()
    bank = manager.capture(helper, num_microbatches=len(graphs))
    bank.activate()
    return manager, bank, runtime


def _make_task9_compatibility_helper(layer, graph):
    from types import SimpleNamespace

    manager, bank, _ = _make_task9_active_bank(layer, graph)
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper._compatibility_bank_manager = manager
    helper._compatibility_bank = bank
    helper._graphs_created = True
    helper.tp_group = SimpleNamespace()
    helper.dp_cp_group = SimpleNamespace()
    return helper, manager, bank


def test_execution_counter_counts_three_eager_warmups_and_checkpoint_recompute(monkeypatch) -> None:
    from types import SimpleNamespace

    from torch.utils.checkpoint import checkpoint

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    state = {"capturing": False, "warmup": False}
    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: state["capturing"])
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: state["warmup"])
    layer = _make_task7_transformer_leaf(moe=False)
    layer.config = SimpleNamespace(cuda_graph_impl="transformer_engine")
    layer.training = True
    layer.forward = lambda hidden_states: hidden_states
    layer._te_cuda_graph_capture = lambda hidden_states: hidden_states
    manager = TECudaGraphBankManager(
        [layer],
        graph_reset_supported=False,
        synchronize=lambda: None,
        runtime_num_microbatches=lambda: 1,
    )
    hidden_states = torch.ones(1)

    for _ in range(3):
        GraphableMegatronModule.__call__(layer, hidden_states)
    state["warmup"] = True
    GraphableMegatronModule.__call__(layer, hidden_states)
    state["warmup"] = False
    state["capturing"] = True
    GraphableMegatronModule.__call__(layer, hidden_states)
    state["capturing"] = False
    layer.training = False
    GraphableMegatronModule.__call__(layer, hidden_states)
    layer.training = True
    checkpointed_input = torch.ones(1, requires_grad=True)
    checkpoint(
        lambda value: GraphableMegatronModule.__call__(layer, value),
        checkpointed_input,
        use_reentrant=True,
    ).sum().backward()

    snapshot = manager.snapshot_execution_counters()
    assert (snapshot.eligible_calls, snapshot.graph_calls) == (5, 0)
    manager.close()


def test_execution_counter_eligible_call_ignores_instance_method_override(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: False)
    layer = _make_task7_transformer_leaf(moe=False)
    layer.config = SimpleNamespace(cuda_graph_impl="transformer_engine")
    layer.training = True
    layer.forward = lambda hidden_states: hidden_states
    manager = TECudaGraphBankManager(
        [layer],
        graph_reset_supported=False,
        synchronize=lambda: None,
        runtime_num_microbatches=lambda: 1,
    )
    tracker = layer._te_cuda_graph_execution_counter
    tracker.record_eligible_call = lambda: None
    hidden_states = torch.ones(1)

    assert GraphableMegatronModule.__call__(layer, hidden_states) is hidden_states
    assert manager.snapshot_execution_counters().eligible_calls == 1

    manager.close()


def test_transformer_execution_counter_counts_once_after_double_guard_and_preparation(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs

    state = {"capturing": False, "warmup": False}
    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: state["capturing"])
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: state["warmup"])
    layer = _make_task7_transformer_leaf(moe=False)
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
    )
    layer.training = True
    layer.current_microbatch = 0
    layer._flatten_te_cuda_graph_packed_seq_params = lambda _kwargs: None
    layer._rebuild_te_cuda_graph_packed_seq_params = lambda _kwargs: None
    layer._get_te_cuda_graph_replay_args = lambda *args, **kwargs: (args, kwargs)
    graph = _Task9ReplayGraph(tuple_output=True)
    manager, bank, runtime = _make_task9_active_bank(layer, graph)
    tracker = layer._te_cuda_graph_execution_counter
    tracker.record_graph_call = lambda: None
    guard_calls = 0
    assert_replay_ready = manager._assert_replay_ready

    def count_guard_calls(*args, **kwargs):
        nonlocal guard_calls
        guard_calls += 1
        return assert_replay_ready(*args, **kwargs)

    monkeypatch.setattr(manager, "_assert_replay_ready", count_guard_calls)
    hidden_states = torch.ones(1)

    output, context = layer(hidden_states)
    assert output is hidden_states and context is None
    assert guard_calls == 2
    assert manager.snapshot_execution_counters().graph_calls == 1

    state["warmup"] = True
    output, context = layer(hidden_states)
    assert output is hidden_states and context is None
    state["warmup"] = False
    state["capturing"] = True
    output, context = layer(hidden_states)
    assert output is hidden_states and context is None
    state["capturing"] = False
    assert manager.snapshot_execution_counters().graph_calls == 1

    layer._get_te_cuda_graph_replay_args = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("replay preparation failed")
    )
    with pytest.raises(RuntimeError, match="replay preparation failed"):
        layer(hidden_states)
    assert manager.snapshot_execution_counters().graph_calls == 1
    layer._get_te_cuda_graph_replay_args = lambda *args, **kwargs: (args, kwargs)

    runtime["count"] = 2
    with pytest.raises(ValueError, match="runtime num_microbatches"):
        layer(hidden_states)
    assert manager.snapshot_execution_counters().graph_calls == 1
    runtime["count"] = 1

    layer.cuda_graph_manual_hooks.append(
        (lambda: (_ for _ in ()).throw(RuntimeError("manual hook failed")), ())
    )
    with pytest.raises(RuntimeError, match="manual hook failed"):
        layer(hidden_states)
    assert manager.snapshot_execution_counters().graph_calls == 1
    layer.cuda_graph_manual_hooks.clear()

    graph.fail_launch = True
    with pytest.raises(RuntimeError, match="graph launch failed"):
        layer(hidden_states)
    snapshot = manager.snapshot_execution_counters()
    assert (snapshot.eligible_calls, snapshot.graph_calls) == (5, 2)

    graph.fail_launch = False
    bank.reset()
    manager.close()


def test_active_moe_bank_launches_all_padding_ownership_once_without_fallback(monkeypatch) -> None:
    from types import MethodType, SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.packed_seq_params import (
        MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
        split_moe_packed_seq_params_for_cuda_graph,
    )
    from megatron.core.transformer.module import GraphableMegatronModule

    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: False)
    layer = _make_task7_transformer_leaf(moe=True)
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.moe],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
    )
    layer.is_moe_layer = True
    layer.training = True
    layer.current_microbatch = 0
    layer._forward_attention = lambda hidden_states, **_kwargs: (hidden_states, None)
    fallback_calls = 0

    def reject_fallback(*_args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("eager MLP fallback must not run")

    layer._forward_mlp = reject_fallback
    layer._get_te_cuda_graph_replay_args = MethodType(
        GraphableMegatronModule._get_te_cuda_graph_replay_args, layer
    )

    class _RecordingGraph:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return (args[0],)

    graph = _RecordingGraph()
    captured = PackedSeqParams(
        seq_aux_loss_sample_ids=torch.arange(8, dtype=torch.int64).remainder(2),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=4,
    )

    def install_contract() -> None:
        tensor_kwargs, static_metadata = split_moe_packed_seq_params_for_cuda_graph(captured)
        layer._set_te_cuda_graph_moe_packed_seq_params_static_metadata(
            static_metadata, tensor_kwargs
        )

    manager, bank, _ = _make_task9_active_bank(
        layer,
        graph,
        cuda_graph_modules=[CudaGraphModule.moe],
        setup=install_contract,
    )
    replay = PackedSeqParams(
        seq_aux_loss_sample_ids=torch.zeros(8, dtype=torch.int64),
        seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.int64),
        seq_aux_loss_max_samples=4,
    )
    hidden_states = torch.zeros((8, 1, 4))
    padding_mask = torch.ones((1, 8), dtype=torch.bool)

    output, context = layer(
        hidden_states,
        packed_seq_params=replay,
        padding_mask=padding_mask,
    )

    assert output is hidden_states
    assert context is None
    assert fallback_calls == 0
    assert len(graph.calls) == 1
    graph_args, graph_kwargs = graph.calls[0]
    assert len(graph_args) == 1
    assert graph_args[0] is hidden_states
    assert graph_kwargs["padding_mask"] is padding_mask
    assert graph_kwargs[
        f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_sample_ids"
    ] is replay.seq_aux_loss_sample_ids
    assert graph_kwargs[
        f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_num_samples"
    ] is replay.seq_aux_loss_num_samples
    assert not graph_kwargs[
        f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_sample_ids"
    ].any()
    assert graph_kwargs[
        f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_num_samples"
    ].item() == 1
    assert graph_kwargs["padding_mask"].all()
    snapshot = manager.snapshot_execution_counters()
    assert (snapshot.eligible_calls, snapshot.graph_calls) == (1, 1)

    with pytest.raises(ValueError, match="unexpected"):
        layer._te_cuda_graph_replay(
            hidden_states,
            packed_seq_params=replay,
            padding_mask=padding_mask,
            _moe_packed_seq_params_unexpected=torch.zeros((), dtype=torch.int64),
        )

    assert len(graph.calls) == 1
    assert manager.snapshot_execution_counters() == snapshot
    bank.reset()
    manager.close()


def test_active_te_replay_uses_unbound_prelaunch_validation_after_graph_swap(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs

    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: False)
    layer = _make_task7_transformer_leaf(moe=False)
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
    )
    layer.training = True
    layer.current_microbatch = 0
    layer._flatten_te_cuda_graph_packed_seq_params = lambda _kwargs: None
    layer._rebuild_te_cuda_graph_packed_seq_params = lambda _kwargs: None
    canonical_graph = _Task9ReplayGraph(tuple_output=True)
    foreign_graph = _Task9ReplayGraph(tuple_output=True)
    manager, bank, _ = _make_task9_active_bank(layer, canonical_graph)

    def swap_selected_graph(*args, **kwargs):
        layer.cuda_graphs[0] = foreign_graph
        return args, kwargs

    layer._get_te_cuda_graph_replay_args = swap_selected_graph
    manager._validate_graph_call = lambda *_args, **_kwargs: None
    hidden_states = torch.ones(1)

    with pytest.raises(RuntimeError, match="selected callable changed before launch"):
        layer(hidden_states)
    assert manager.snapshot_execution_counters().graph_calls == 0
    assert canonical_graph.calls == 0
    assert foreign_graph.calls == 0

    layer.cuda_graphs[0] = canonical_graph
    bank.reset()
    manager.close()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_tracker",
        "foreign_tracker",
        "missing_guard",
        "foreign_guard",
        "guard_subclass",
        "both_missing",
    ],
)
def test_active_te_replay_rejects_missing_or_foreign_counter_ownership(
    monkeypatch, corruption
) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.te_cuda_graph_bank import _BankReplayGuard

    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: False)
    layer = _make_task7_transformer_leaf(moe=False)
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[],
        delay_offload_until_cuda_graph=False,
        overlap_moe_expert_parallel_comm=False,
        fine_grained_activation_offloading=False,
    )
    layer.training = True
    layer.current_microbatch = 0
    layer._flatten_te_cuda_graph_packed_seq_params = lambda _kwargs: None
    layer._rebuild_te_cuda_graph_packed_seq_params = lambda _kwargs: None
    layer._get_te_cuda_graph_replay_args = lambda *args, **kwargs: (args, kwargs)
    graph = _Task9ReplayGraph(tuple_output=True)
    manager, bank, _ = _make_task9_active_bank(layer, graph)
    tracker = layer._te_cuda_graph_execution_counter
    guard = layer._te_cuda_graph_bank_replay_guard

    class _ForeignGuard:
        def __call__(self, _layer, _graphs, _microbatch):
            return 0

        def record_graph_call(self, _layer, _graphs, _index, _counter):
            return None

    class _GuardSubclass(_BankReplayGuard):
        def __call__(self, _layer, _graphs, _microbatch):
            return 0

        def record_graph_call(self, _layer, _graphs, _index, _counter):
            return None

        def validate_graph_call(self, _layer, _graphs, _index, _counter):
            return None

    if corruption == "missing_tracker":
        del layer._te_cuda_graph_execution_counter
    elif corruption == "foreign_tracker":
        layer._te_cuda_graph_execution_counter = object()
    elif corruption == "missing_guard":
        del layer._te_cuda_graph_bank_replay_guard
    elif corruption == "foreign_guard":
        layer._te_cuda_graph_bank_replay_guard = _ForeignGuard()
    elif corruption == "guard_subclass":
        layer._te_cuda_graph_bank_replay_guard = _GuardSubclass(guard._manager, guard._bank)
    else:
        del layer._te_cuda_graph_execution_counter
        del layer._te_cuda_graph_bank_replay_guard

    with pytest.raises(RuntimeError, match="TE CUDA graph"):
        layer._te_cuda_graph_replay(torch.ones(1))
    assert graph.calls == 0

    layer._te_cuda_graph_execution_counter = tracker
    layer._te_cuda_graph_bank_replay_guard = guard
    assert manager.snapshot_execution_counters().graph_calls == 0
    bank.reset()
    manager.close()


@pytest.mark.parametrize("internal_state", ["capturing", "warmup"])
def test_internal_te_replay_allows_tracker_before_guard_installation(
    monkeypatch, internal_state
) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    state = {"capturing": False, "warmup": False}
    state[internal_state] = True
    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: state["capturing"])
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: state["warmup"])
    layer = _make_task7_transformer_leaf(moe=False)
    layer.config = SimpleNamespace(fine_grained_activation_offloading=False)
    layer.training = True
    layer.current_microbatch = 0
    layer._get_te_cuda_graph_replay_args = lambda *args, **kwargs: (args, kwargs)
    graph = _Task9ReplayGraph(tuple_output=False)
    layer.cuda_graphs = [graph]
    manager = TECudaGraphBankManager(
        [layer],
        graph_reset_supported=False,
        synchronize=lambda: None,
        runtime_num_microbatches=lambda: 1,
    )

    hidden_states = torch.ones(1)
    assert GraphableMegatronModule._te_cuda_graph_replay(layer, hidden_states) is hidden_states
    snapshot = manager.snapshot_execution_counters()
    assert (snapshot.eligible_calls, snapshot.graph_calls) == (0, 0)

    manager.close()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_tracker",
        "foreign_tracker",
        "missing_guard",
        "foreign_guard",
        "guard_subclass",
        "both_missing",
    ],
)
def test_active_te_backward_dw_replay_rejects_missing_or_foreign_counter_ownership(
    corruption,
) -> None:
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.te_cuda_graph_bank import _BankReplayGuard

    layer = _make_task7_transformer_leaf(moe=False)
    layer.current_microbatch = 0
    graph = _Task9ReplayGraph(tuple_output=True)
    manager, bank, _ = _make_task9_active_bank(layer, graph)
    tracker = layer._te_cuda_graph_execution_counter
    guard = layer._te_cuda_graph_bank_replay_guard

    class _ForeignGuard:
        def __call__(self, _layer, _graphs, _microbatch):
            return 0

        def record_graph_call(self, _layer, _graphs, _index, _counter):
            return None

    class _GuardSubclass(_BankReplayGuard):
        def __call__(self, _layer, _graphs, _microbatch):
            return 0

        def record_graph_call(self, _layer, _graphs, _index, _counter):
            return None

        def validate_graph_call(self, _layer, _graphs, _index, _counter):
            return None

    if corruption == "missing_tracker":
        del layer._te_cuda_graph_execution_counter
    elif corruption == "foreign_tracker":
        layer._te_cuda_graph_execution_counter = object()
    elif corruption == "missing_guard":
        del layer._te_cuda_graph_bank_replay_guard
    elif corruption == "foreign_guard":
        layer._te_cuda_graph_bank_replay_guard = _ForeignGuard()
    elif corruption == "guard_subclass":
        layer._te_cuda_graph_bank_replay_guard = _GuardSubclass(guard._manager, guard._bank)
    else:
        del layer._te_cuda_graph_execution_counter
        del layer._te_cuda_graph_bank_replay_guard

    with pytest.raises(RuntimeError, match="TE CUDA graph"):
        GraphableMegatronModule._te_cuda_graph_backward_dw_graph(layer, 0)
    assert graph.backward_dw_calls == 0

    layer._te_cuda_graph_execution_counter = tracker
    layer._te_cuda_graph_bank_replay_guard = guard
    bank.reset()
    manager.close()


def test_mamba_execution_counter_uses_common_replay_boundary(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs

    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: False)
    layer = _make_task7_mamba_leaf()
    layer.config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[],
        fine_grained_activation_offloading=False,
    )
    layer.training = True
    layer.current_microbatch = 0
    layer._flatten_te_cuda_graph_mamba_packed_seq_params = lambda _kwargs: None
    graph = _Task9ReplayGraph(tuple_output=False)
    manager, bank, _ = _make_task9_active_bank(layer, graph)
    hidden_states = torch.ones(1)

    assert layer(hidden_states) is hidden_states
    snapshot = manager.snapshot_execution_counters()
    assert (snapshot.eligible_calls, snapshot.graph_calls) == (1, 1)

    bank.reset()
    manager.close()


@pytest.mark.parametrize("cuda_graph_modules", [[], [CudaGraphModule.mamba]])
def test_real_mamba_helper_and_replay_route_flattened_packed_inputs(cuda_graph_modules) -> None:
    from types import SimpleNamespace

    from megatron.core.packed_seq_params import (
        CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
        MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX,
    )
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.cuda_graphs import _GraphableTELayerDescriptor

    layer = MambaLayer.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        context_parallel_size=1,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        hidden_size=8,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=cuda_graph_modules,
        fine_grained_activation_offloading=False,
    )
    layer.cuda_graphs = []
    layer.cuda_graph_manual_hooks = []
    chunk = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]), mtp=SimpleNamespace(layers=[]))
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda"),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv_padded=torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda"),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=8,
    )
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.config = SimpleNamespace(
        cuda_graph_modules=cuda_graph_modules, multi_latent_attention=False
    )
    helper.seq_length = 8
    helper.micro_batch_size = 1
    helper.sample_packed_seq_params = packed
    helper.num_model_chunks = 1
    helper.num_microbatches = 1
    helper.num_layers_per_chunk = [1]
    helper.flattened_callables = [layer]
    helper.callables_per_chunk = [[layer]]
    helper.chunks_with_decoder = [chunk]
    descriptor = _GraphableTELayerDescriptor(layer=layer, is_mtp=False, mtp_owner=None)
    helper.layer_descriptors_per_chunk = [(descriptor,)]
    helper.flattened_layer_descriptors = [descriptor]

    sample_args, sample_kwargs = helper._get_sample_arguments([1, -1])

    assert sample_args[0][0].shape == (8, 1, 8)
    assert any(key.startswith(CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX) for key in sample_kwargs[0])
    seq_idx_key = f"{MAMBA_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_idx"
    assert seq_idx_key in sample_kwargs[0]
    assert not any(key.startswith("_moe_packed_seq_params_") for key in sample_kwargs[0])
    assert all(not isinstance(value, PackedSeqParams) for value in sample_kwargs[0].values())
    seq_idx = packed.seq_idx
    observed = {}

    def replay(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return args[0]

    bank_manager, bank, _ = _make_task9_active_bank(layer, replay)
    layer.current_microbatch = 0
    hidden_states = torch.empty((8, 1, 8), dtype=torch.bfloat16, device="cuda")
    layer._te_cuda_graph_replay(hidden_states, packed_seq_params=packed)

    assert observed["args"] == (hidden_states,)
    assert observed["kwargs"][seq_idx_key] is seq_idx
    assert observed["kwargs"]["is_first_microbatch"] is True
    assert all(not isinstance(value, PackedSeqParams) for value in observed["kwargs"].values())
    with pytest.raises(AssertionError, match="inference_context"):
        layer._te_cuda_graph_replay(
            hidden_states, packed_seq_params=packed, inference_context=object()
        )
    bank.reset()
    bank_manager.close()


def test_te_helper_abort_restores_capture_globals_and_partial_graphs(monkeypatch) -> None:
    from types import ModuleType, SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs

    class _Graph:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    graph = _Graph()
    layer = SimpleNamespace(cuda_graphs=[])
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.flattened_callables = [layer]
    helper.config = SimpleNamespace(fine_grained_activation_offloading=True)
    helper._capture_finished = False
    helper._graphs_created = False
    helper._capture_gc_frozen = False
    te_capture_end_calls = []
    offload_reset_calls = []
    gc_unfreeze_calls = []
    capture_state_cleanup_calls = []

    def start_capture():
        cuda_graphs._set_capture_start()
        cuda_graphs._set_warmup_start()
        helper._capture_gc_frozen = True
        return 0.0

    def fail_inputs(*, num_microbatches):
        assert num_microbatches == 2
        layer.cuda_graphs.append(graph)
        raise RuntimeError("input failure")

    class _OffloadInterface:
        @staticmethod
        def reset():
            offload_reset_calls.append(None)

    offload_module = ModuleType("megatron.core.pipeline_parallel.fine_grained_activation_offload")
    offload_module.FineGrainedActivationOffloadingInterface = _OffloadInterface
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.pipeline_parallel.fine_grained_activation_offload",
        offload_module,
    )
    monkeypatch.setattr(helper, "_start_capturing", start_capture)
    monkeypatch.setattr(helper, "_get_cuda_graph_input_data", fail_inputs)

    def fail_capture_state_cleanup():
        capture_state_cleanup_calls.append(None)
        raise RuntimeError("capture-state cleanup failed")

    monkeypatch.setattr(helper, "_reset_after_capture", fail_capture_state_cleanup)
    monkeypatch.setattr(
        cuda_graphs, "te_set_capture_end", lambda: te_capture_end_calls.append(None)
    )
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cuda_graphs.gc, "unfreeze", lambda: gc_unfreeze_calls.append(None))
    monkeypatch.setattr(cuda_graphs.gc, "collect", lambda: 0)

    with pytest.raises(RuntimeError, match="input failure"):
        helper._capture_cuda_graph_lists(num_microbatches=2)

    assert not cuda_graphs.is_graph_capturing()
    assert not cuda_graphs.is_graph_warmup()
    assert te_capture_end_calls == [None]
    assert offload_reset_calls == [None]
    assert gc_unfreeze_calls == [None]
    assert capture_state_cleanup_calls == [None]
    assert graph.reset_calls == 1
    assert layer.cuda_graphs == []
    assert helper._capture_gc_frozen is False
    assert helper._capture_finished is False
    assert helper._graphs_created is False


def test_te_helper_manual_hook_setup_refreshes_owning_bank() -> None:
    from types import SimpleNamespace

    layer = SimpleNamespace()
    hook_list = [(object(), (layer,))]
    layer.setup_manual_hooks = lambda make_hook: setattr(
        layer, "cuda_graph_manual_hooks", hook_list
    )
    bank = object()
    refresh_calls = []
    manager = SimpleNamespace(refresh_manual_hooks=lambda target: refresh_calls.append(target))
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.callables_per_chunk = [[layer]]
    helper.layer_descriptors_per_chunk = [[SimpleNamespace(layer=layer)]]
    helper.model = [SimpleNamespace(_make_forward_pre_hook=object())]
    helper._compatibility_bank_manager = manager
    helper._compatibility_bank = bank

    helper.cuda_graph_set_manual_hooks()

    assert layer.cuda_graph_manual_hooks is hook_list
    assert refresh_calls == [bank]


def test_te_helper_delete_closes_counter_manager_and_allows_recreation(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    monkeypatch.setattr(cuda_graphs, "log_on_each_pipeline_stage", lambda **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr("megatron.core.utils.is_te_min_version", lambda _version: False)
    layer = _make_task7_transformer_leaf(moe=False)
    helper, manager, _ = _make_task9_compatibility_helper(
        layer, _Task9ReplayGraph(tuple_output=True)
    )

    helper.delete_cuda_graphs()

    assert not hasattr(layer, "_te_cuda_graph_execution_counter")
    assert helper._compatibility_bank is None
    assert helper._compatibility_bank_manager is None
    assert helper._graphs_created is False
    with pytest.raises(RuntimeError, match="manager is closed"):
        manager.snapshot_execution_counters()

    recreated_helper = SimpleNamespace(
        flattened_callables=[layer],
        pp_group=SimpleNamespace(size=lambda: 1),
        config=SimpleNamespace(overlap_moe_expert_parallel_comm=False, cuda_graph_modules=()),
    )
    replacement = TECudaGraphBankManager.from_helper(recreated_helper)
    replacement.close()


def test_te_helper_delete_retains_ownership_when_bank_reset_fails(monkeypatch) -> None:
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    monkeypatch.setattr(cuda_graphs, "log_on_each_pipeline_stage", lambda **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    layer = _make_task7_transformer_leaf(moe=False)
    helper, manager, bank = _make_task9_compatibility_helper(
        layer, _Task9ReplayGraph(tuple_output=True)
    )
    layer.clear_te_cuda_graph_bank_references = lambda: (_ for _ in ()).throw(
        RuntimeError("bank reset failed")
    )

    with pytest.raises(RuntimeError, match="bank reset failed"):
        helper.delete_cuda_graphs()

    assert helper._compatibility_bank is bank
    assert helper._compatibility_bank_manager is manager
    assert helper._graphs_created is True
    assert manager.active_bank is bank
    manager.snapshot_execution_counters()

    layer.clear_te_cuda_graph_bank_references = lambda: None
    helper.delete_cuda_graphs()


def test_te_helper_delete_retains_ownership_when_manager_close_fails(monkeypatch) -> None:
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    monkeypatch.setattr(cuda_graphs, "log_on_each_pipeline_stage", lambda **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    layer = _make_task7_transformer_leaf(moe=False)
    helper, manager, bank = _make_task9_compatibility_helper(
        layer, _Task9ReplayGraph(tuple_output=True)
    )
    tracker = layer._te_cuda_graph_execution_counter
    layer._te_cuda_graph_execution_counter = object()

    with pytest.raises(ValueError, match="ownership changed"):
        helper.delete_cuda_graphs()

    assert helper._compatibility_bank is bank
    assert helper._compatibility_bank_manager is manager
    assert helper._graphs_created is True
    assert manager.active_bank is None
    assert manager.registered_bank_count == 0

    layer._te_cuda_graph_execution_counter = tracker
    helper.delete_cuda_graphs()


def test_te_helper_delete_clears_closed_ownership_before_logging(monkeypatch) -> None:
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    monkeypatch.setattr(
        cuda_graphs,
        "log_on_each_pipeline_stage",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("logging failed")),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    layer = _make_task7_transformer_leaf(moe=False)
    helper, manager, _ = _make_task9_compatibility_helper(
        layer, _Task9ReplayGraph(tuple_output=True)
    )

    with pytest.raises(RuntimeError, match="logging failed"):
        helper.delete_cuda_graphs()

    assert not hasattr(layer, "_te_cuda_graph_execution_counter")
    assert helper._compatibility_bank is None
    assert helper._compatibility_bank_manager is None
    assert helper._graphs_created is False
    with pytest.raises(RuntimeError, match="manager is closed"):
        manager.snapshot_execution_counters()


def test_transformer_layer_restores_exact_moe_references_after_detach_rollback() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.transformer_layer import TransformerLayer

    class _TensorStore:
        hidden_states = None
        probs = None
        routing_map = None
        shared_expert_output = None

        def clear(self):
            self.hidden_states = None
            self.probs = None
            self.routing_map = None
            self.shared_expert_output = None

    first = torch.empty(1)
    second = torch.empty(2)
    dispatcher = SimpleNamespace(
        valid_cudagraph_attrs=["first", "nested.second"],
        first=first,
        nested=SimpleNamespace(second=second),
    )
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.is_moe_layer = True
    layer.mlp = SimpleNamespace(token_dispatcher=dispatcher, cudagraph_tensor_store=_TensorStore())

    snapshot = layer.snapshot_te_cuda_graph_bank_references()
    layer.clear_te_cuda_graph_bank_references()
    assert dispatcher.first is None
    assert dispatcher.nested.second is None

    layer.restore_te_cuda_graph_bank_references(snapshot)

    assert dispatcher.first is first
    assert dispatcher.nested.second is second


def test_vision_te_helper_preserves_manager_owned_graph_list(monkeypatch) -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.cuda_graphs import VisionTECudaGraphHelper

    owned_graph_list = [lambda: (torch.empty(1), None)]
    layer = SimpleNamespace(cuda_graphs=owned_graph_list)
    helper = VisionTECudaGraphHelper.__new__(VisionTECudaGraphHelper)
    helper.flattened_callables = [layer]
    finish_calls = []
    monkeypatch.setattr(
        TECudaGraphHelper,
        "_finish_capturing",
        lambda self, start_time: finish_calls.append(start_time),
    )

    helper._finish_capturing(1.0)

    assert layer.cuda_graphs is owned_graph_list
    assert len(owned_graph_list) == 1
    assert finish_calls == [1.0]


def test_partial_moe_capture_routes_only_stream_captures_to_exact_runner_index(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.transformer_layer as transformer_layer_module
    from megatron.core.transformer.transformer_layer import TransformerLayer

    layer = SimpleNamespace(
        config=SimpleNamespace(
            cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess]
        ),
        offload_module_in_cuda_graph=False,
        is_moe_layer=True,
        _te_cuda_graph_capture_num_microbatches=2,
        _te_cuda_graph_capture_cursor=0,
        _rebuild_te_cuda_graph_packed_seq_params=lambda kwargs: None,
        _forward_attention=lambda *args, **kwargs: (args[0], None),
    )
    capturing = iter((False, False, True, False, True, True))
    monkeypatch.setattr(
        transformer_layer_module, "_is_te_cuda_graph_stream_capturing", lambda: next(capturing)
    )
    residuals = [torch.empty(1) for _ in range(6)]
    preprocessed = [torch.empty(1) for _ in range(6)]
    calls = []
    invocation = {"index": 0}

    def forward_mlp(hidden_states, **kwargs):
        index = invocation["index"]
        invocation["index"] += 1
        return [preprocessed[index], torch.empty(1), residuals[index]]

    layer._forward_mlp = forward_mlp
    layer._record_te_cuda_graph_dispatcher_replay_state = (
        lambda index, graph_input, output: calls.append((index, graph_input, output))
    )

    for residual in residuals[:5]:
        TransformerLayer._te_cuda_graph_capture(layer, residual)
    with pytest.raises(RuntimeError, match="more committed forwards"):
        TransformerLayer._te_cuda_graph_capture(layer, residuals[5])

    assert calls == [(0, residuals[2], preprocessed[2]), (1, residuals[4], preprocessed[4])]


def test_dropless_alltoall_partial_capture_accepts_static_output_geometry() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.drop_and_pad = False
    dispatcher.config = SimpleNamespace(
        moe_expert_capacity_factor=None, moe_router_padding_for_quantization=False
    )
    layer = MoETransformerLayer.__new__(MoETransformerLayer)
    layer.mlp = SimpleNamespace(token_dispatcher=dispatcher)
    layer.config = SimpleNamespace(overlap_moe_expert_parallel_comm=False)

    layer._validate_te_cuda_graph_dispatcher_replay_capability()


@pytest.mark.parametrize(("capacity_factor", "router_padding"), [(1.0, False), (None, True)])
def test_partial_moe_capture_rejects_cuda_scalar_snapshot_before_capture(
    capacity_factor, router_padding
) -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.drop_and_pad = False
    dispatcher.config = SimpleNamespace(
        moe_expert_capacity_factor=capacity_factor,
        moe_router_padding_for_quantization=router_padding,
    )
    layer = MoETransformerLayer.__new__(MoETransformerLayer)
    layer.mlp = SimpleNamespace(token_dispatcher=dispatcher)
    layer.config = SimpleNamespace(overlap_moe_expert_parallel_comm=False)

    with pytest.raises(RuntimeError, match="CUDA scalar.*actual stream capture"):
        layer._validate_te_cuda_graph_dispatcher_replay_capability()


def test_partial_hybridep_capture_rejects_uneven_input_gpu_sync_before_capture() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher._comm_manager = SimpleNamespace(drop_and_pad=True)
    dispatcher.config = SimpleNamespace(
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_rank_capacity_factor=None,
        moe_hybridep_pad_uneven_dispatch_inputs=True,
    )
    layer = MoETransformerLayer.__new__(MoETransformerLayer)
    layer.mlp = SimpleNamespace(token_dispatcher=dispatcher)
    layer.config = SimpleNamespace(overlap_moe_expert_parallel_comm=False)

    with pytest.raises(RuntimeError, match="uneven-input.*GPU-to-host"):
        layer._validate_te_cuda_graph_dispatcher_replay_capability()


def test_moe_partial_te_replay_rejects_overlap_before_capture() -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.transformer_layer import MoETransformerLayer

    layer = MoETransformerLayer.__new__(MoETransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(overlap_moe_expert_parallel_comm=True)
    layer.mlp = SimpleNamespace(
        token_dispatcher=MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    )

    with pytest.raises(RuntimeError, match="overlap_moe_expert_parallel_comm"):
        layer._validate_te_cuda_graph_dispatcher_replay_capability()


def test_transformer_layer_drain_check_rejects_pending_backward_dw_and_allows_idle() -> None:
    import queue
    from types import SimpleNamespace

    from megatron.core.models.common.utils import _BackwardDWWrapper
    from megatron.core.transformer.transformer_layer import TransformerLayer

    class _DelayedAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wgrad_store = SimpleNamespace(context=queue.Queue())

        def backward_dw(self) -> None:
            self.wgrad_store.context.get_nowait()

        def need_backward_dw(self) -> bool:
            # Pinned TE reports whether delayed-wgrad mode is enabled, not queue occupancy.
            return True

    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.attn], moe_shared_expert_overlap=False
    )
    layer.is_moe_layer = False
    layer.self_attention = _DelayedAttention()
    layer.cuda_graphs = [object()]
    layer.backward_dw_wrapper = _BackwardDWWrapper(layer)

    layer.assert_te_cuda_graph_bank_drained()

    graph_calls = []
    layer.backward_dw_wrapper.set_graphed_backward_dw_callable(
        lambda: graph_calls.append("current")
    )
    with pytest.raises(RuntimeError, match="_BackwardDWWrapper"):
        layer.assert_te_cuda_graph_bank_drained()
    layer.backward_dw_wrapper.backward_dw()
    assert graph_calls == ["current"]
    layer.assert_te_cuda_graph_bank_drained()

    layer.init_backward_dw_wrapper()
    older_wrapper = layer.backward_dw_wrapper
    older_wrapper.set_graphed_backward_dw_callable(lambda: graph_calls.append("older"))
    layer.init_backward_dw_wrapper()
    with pytest.raises(RuntimeError, match="_BackwardDWWrapper"):
        layer.assert_te_cuda_graph_bank_drained()
    older_wrapper.backward_dw()
    assert graph_calls == ["current", "older"]
    layer.assert_te_cuda_graph_bank_drained()

    layer.cuda_graphs = []
    layer.self_attention.wgrad_store.context.put(object())
    with pytest.raises(RuntimeError, match="delayed weight-gradient"):
        layer.assert_te_cuda_graph_bank_drained()

    layer.backward_dw_wrapper.backward_dw()
    layer.assert_te_cuda_graph_bank_drained()

    layer.backward_dw_wrapper = None
    layer.self_attention.wgrad_store.context.put(object())
    with pytest.raises(RuntimeError, match="delayed weight-gradient"):
        layer.assert_te_cuda_graph_bank_drained()
    layer.self_attention.wgrad_store.context.get_nowait()

    fused_owner = torch.nn.Module()
    layer.fused_owner = fused_owner
    for attribute in ("_fused_ops", "_fused_grouped_swiglu_ops"):
        fused_linear = _DelayedAttention()
        setattr(fused_owner, attribute, (torch.nn.Sequential(fused_linear),))
        fused_linear.wgrad_store.context.put(object())
        with pytest.raises(RuntimeError, match="delayed weight-gradient"):
            layer.assert_te_cuda_graph_bank_drained()
        fused_linear.wgrad_store.context.get_nowait()
        delattr(fused_owner, attribute)

    unverifiable = _DelayedAttention()
    unverifiable.wgrad_store.context = object()
    layer.unverifiable = unverifiable
    with pytest.raises(RuntimeError, match="Cannot verify.*queue drainage"):
        layer.assert_te_cuda_graph_bank_drained()


def _make_task7_transformer_leaf(*, moe: bool) -> TransformerLayer:
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.moe_layer import MoELayer

    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attention = IdentityOp()
    layer.cross_attention = IdentityOp()
    if moe:
        layer.mlp = MoELayer.__new__(MoELayer)
        torch.nn.Module.__init__(layer.mlp)
        layer.mlp.num_local_experts = 2
    else:
        layer.mlp = MLP.__new__(MLP)
        torch.nn.Module.__init__(layer.mlp)
    layer.cuda_graphs = []
    layer.cuda_graph_manual_hooks = []
    return layer


def _make_task7_mamba_leaf():
    from megatron.core.ssm.mamba_layer import MambaLayer

    layer = MambaLayer.__new__(MambaLayer)
    torch.nn.Module.__init__(layer)
    layer.cuda_graphs = []
    layer.cuda_graph_manual_hooks = []
    return layer


def _make_task7_hybrid_mtp_model():
    from types import SimpleNamespace

    decoder = _make_task7_transformer_leaf(moe=False)
    moe = _make_task7_transformer_leaf(moe=True)
    mamba = _make_task7_mamba_leaf()
    dense = _make_task7_transformer_leaf(moe=False)
    stack = HybridStack.__new__(HybridStack)
    torch.nn.Module.__init__(stack)
    stack.layers = torch.nn.ModuleList([moe, mamba, dense])
    owner = SimpleNamespace(mtp_model_layer=stack)
    config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.mamba, CudaGraphModule.mlp],
        multi_latent_attention=False,
    )
    chunk = SimpleNamespace(
        config=config,
        decoder=SimpleNamespace(layers=[decoder]),
        mtp=SimpleNamespace(layers=[owner]),
    )
    return chunk, stack, (decoder, moe, mamba, dense)


def _make_task5_ordered_hybrid_mtp_model(mtp_depth: int):
    """Build fresh ordered Hybrid leaves for the decoder and every MTP depth."""

    from types import SimpleNamespace

    def make_ordered_leaves():
        return (
            _make_task7_transformer_leaf(moe=True),
            _make_task7_mamba_leaf(),
            _make_task7_transformer_leaf(moe=False),
        )

    decoder_leaves = make_ordered_leaves()
    mtp_owners = []
    mtp_leaves = []
    for _ in range(mtp_depth):
        leaves = make_ordered_leaves()
        stack = HybridStack.__new__(HybridStack)
        torch.nn.Module.__init__(stack)
        stack.layers = torch.nn.ModuleList(leaves)
        mtp_owners.append(SimpleNamespace(mtp_model_layer=stack))
        mtp_leaves.append(leaves)

    config = SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.moe_router, CudaGraphModule.mamba, CudaGraphModule.mlp],
        multi_latent_attention=False,
    )
    chunk = SimpleNamespace(
        config=config,
        decoder=SimpleNamespace(layers=list(decoder_leaves)),
    )
    if mtp_depth:
        chunk.mtp = SimpleNamespace(layers=mtp_owners)
    return chunk, (decoder_leaves, *mtp_leaves)


def _make_task7_te_helper(monkeypatch):
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    chunk, stack, leaves = _make_task7_hybrid_mtp_model()
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.model = [chunk]
    helper.config = chunk.config
    helper.num_model_chunks = 1
    helper.tp_group = None
    helper.dp_cp_group = None
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(cuda_graphs, "log_on_each_pipeline_stage", lambda **_kwargs: None)
    helper._discover_layers()
    return helper, chunk, stack, leaves


def test_te_discovery_preserves_ordered_hybrid_mtp_leaf_descriptors(monkeypatch) -> None:
    from types import SimpleNamespace

    from megatron.core.transformer.cuda_graphs import (
        _iter_graphable_te_leaves,
        _map_te_graphs_to_layers,
    )
    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    helper, _, stack, leaves = _make_task7_te_helper(monkeypatch)
    decoder, moe, mamba, dense = leaves

    assert list(_iter_graphable_te_leaves(stack, helper.config)) == [moe, mamba, dense]
    assert helper.callables_per_chunk == [[decoder, moe, mamba, dense]]
    assert helper.flattened_callables == [decoder, moe, mamba, dense]
    assert helper.num_layers_per_chunk == [4]
    assert helper.callables_per_chunk_is_mtp == [[False, True, True, True]]
    assert helper.flattened_callables_is_mtp == [False, True, True, True]
    assert [descriptor.layer for descriptor in helper.layer_descriptors_per_chunk[0]] == [
        decoder,
        moe,
        mamba,
        dense,
    ]
    assert [descriptor.mtp_owner for descriptor in helper.layer_descriptors_per_chunk[0]] == [
        None,
        helper.chunks_with_decoder[0].mtp.layers[0],
        helper.chunks_with_decoder[0].mtp.layers[0],
        helper.chunks_with_decoder[0].mtp.layers[0],
    ]
    assert [descriptor.layer for descriptor in helper.flattened_layer_descriptors] == [
        decoder,
        moe,
        mamba,
        dense,
    ]

    helper.pp_group = SimpleNamespace(size=lambda: 1)
    helper.config.overlap_moe_expert_parallel_comm = False
    monkeypatch.setattr("megatron.core.utils.is_te_min_version", lambda _version: False)
    manager = TECudaGraphBankManager.from_helper(helper)
    assert manager.layers == (decoder, moe, mamba, dense)

    graphs = [object() for _ in range(8)]
    for overlap, expected in (
        (False, [[graphs[index], graphs[4 + index]] for index in range(4)]),
        (True, [[graphs[2 * index], graphs[2 * index + 1]] for index in range(4)]),
    ):
        owned_graph_lists = [[] for _ in leaves]
        _map_te_graphs_to_layers(
            graphs,
            callables_per_chunk=helper.callables_per_chunk,
            owned_graph_lists=owned_graph_lists,
            num_microbatches=2,
            overlap_moe_expert_parallel_comm=overlap,
        )
        assert owned_graph_lists == expected


@pytest.mark.parametrize("mtp_depth", (0, 1, 2))
def test_te_descriptors_assign_moe_ownership_only_to_fresh_inner_leaves(
    monkeypatch, mtp_depth
) -> None:
    """Canonical descriptors never assign MoE ownership to Hybrid or non-MoE owners."""

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.cuda_graphs import (
        _add_moe_packed_seq_params_to_te_cuda_graph_sample_kwargs,
    )
    from megatron.core.transformer.moe.moe_layer import MoELayer

    chunk, ordered_groups = _make_task5_ordered_hybrid_mtp_model(mtp_depth)
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.model = [chunk]
    helper.config = chunk.config
    helper.num_model_chunks = 1
    helper.tp_group = None
    helper.dp_cp_group = None
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(cuda_graphs, "log_on_each_pipeline_stage", lambda **_kwargs: None)

    helper._discover_layers()

    descriptors = helper.layer_descriptors_per_chunk[0]
    expected_leaves = [leaf for group in ordered_groups for leaf in group]
    assert len(descriptors) == 3 * (1 + mtp_depth)
    assert [descriptor.layer for descriptor in descriptors] == expected_leaves
    assert len({id(descriptor.layer) for descriptor in descriptors}) == len(descriptors)
    assert [descriptor.is_mtp for descriptor in descriptors] == [False] * 3 + [
        True
    ] * (3 * mtp_depth)

    packed = PackedSeqParams(
        seq_aux_loss_sample_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )
    moe_owners = []
    for descriptor in descriptors:
        sample_kwargs = {}
        _add_moe_packed_seq_params_to_te_cuda_graph_sample_kwargs(
            descriptor, chunk.config, sample_kwargs, packed
        )
        owns_moe_namespace = any(
            key.startswith("_moe_packed_seq_params_") for key in sample_kwargs
        )
        if owns_moe_namespace:
            moe_owners.append(descriptor.layer)
        expected_owner = isinstance(descriptor.layer, TransformerLayer) and isinstance(
            descriptor.layer.mlp, MoELayer
        )
        assert owns_moe_namespace is expected_owner

    assert moe_owners == [group[0] for group in ordered_groups]
    for descriptor in descriptors:
        if descriptor.mtp_owner is not None:
            assert descriptor.layer is not descriptor.mtp_owner
            assert descriptor.layer is not descriptor.mtp_owner.mtp_model_layer


def test_execution_counter_attaches_to_every_ordered_hybrid_mtp_leaf(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs
    from megatron.core.transformer.module import GraphableMegatronModule
    from megatron.core.transformer.te_cuda_graph_bank import TECudaGraphBankManager

    helper, _, _, leaves = _make_task7_te_helper(monkeypatch)
    helper.pp_group = SimpleNamespace(size=lambda: 1)
    helper.config.overlap_moe_expert_parallel_comm = False
    monkeypatch.setattr("megatron.core.utils.is_te_min_version", lambda _version: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(cuda_graphs, "is_graph_warmup", lambda: False)
    manager = TECudaGraphBankManager.from_helper(helper)
    trackers = [leaf._te_cuda_graph_execution_counter for leaf in leaves]

    assert all(tracker is trackers[0] for tracker in trackers)
    for leaf in leaves:
        leaf.config = SimpleNamespace(cuda_graph_impl="transformer_engine")
        leaf.training = True
        leaf._should_call_local_cudagraph = lambda *_args, **_kwargs: False
        leaf.forward = lambda hidden_states: hidden_states
        GraphableMegatronModule.__call__(leaf, torch.ones(1))

    snapshot = manager.snapshot_execution_counters()
    assert (snapshot.eligible_calls, snapshot.graph_calls) == (4, 0)
    manager.close()


def test_te_overlap_input_preparation_preserves_canonical_hybrid_topology(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs

    helper, _, _, leaves = _make_task7_te_helper(monkeypatch)
    observed = {}
    canonical_descriptors = helper.layer_descriptors_per_chunk
    canonical_callables = helper.callables_per_chunk
    canonical_counts = helper.num_layers_per_chunk

    helper.pp_group = SimpleNamespace(size=lambda: 2)
    helper.p2p_communicator = object()
    helper.config.overlap_moe_expert_parallel_comm = True
    helper.config.delay_wgrad_compute = False
    helper.config.moe_shared_expert_intermediate_size = None
    helper.config.moe_shared_expert_overlap = False
    helper.config.microbatch_group_size_per_vp_stage = 1
    helper.config.cuda_graph_retain_backward_graph = False
    helper.config.cuda_graph_warmup_steps = 3
    helper.config.fp8 = None
    helper.config.fp4 = None
    helper.config.fine_grained_activation_offloading = False

    monkeypatch.setattr(cuda_graphs, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda _version: True)
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.schedules.get_pp_rank_microbatches",
        lambda *_args, **_kwargs: (None, None, 0, None),
    )
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.schedules.get_schedule_table",
        lambda *_args, **_kwargs: [(0, 0), (1, 0)],
    )

    def get_sample_arguments(
        order, chunk_id_list, *, schedule_num_layers_per_chunk, schedule_num_model_chunks
    ):
        observed["order"] = order
        observed["chunk_id_list"] = chunk_id_list
        observed["num_layers_per_chunk"] = schedule_num_layers_per_chunk
        observed["num_model_chunks"] = schedule_num_model_chunks
        return [()] * 8, [{} for _ in range(8)]

    monkeypatch.setattr(helper, "_get_sample_arguments", get_sample_arguments)

    _, kwargs = helper._get_cuda_graph_input_data(num_microbatches=2)

    assert observed["num_layers_per_chunk"] == [1, 1, 1, 1]
    assert observed["num_model_chunks"] == 4
    assert kwargs["_num_layers_per_chunk"] is observed["num_layers_per_chunk"]
    assert helper.layer_descriptors_per_chunk is canonical_descriptors
    assert helper.callables_per_chunk is canonical_callables
    assert helper.num_layers_per_chunk is canonical_counts
    assert helper.num_model_chunks == 1
    assert helper.num_layers_per_chunk == [4]
    assert len(helper.layer_descriptors_per_chunk) == len(helper.callables_per_chunk) == 1
    for descriptors, callables, count in zip(
        helper.layer_descriptors_per_chunk, helper.callables_per_chunk, helper.num_layers_per_chunk
    ):
        assert [descriptor.layer for descriptor in descriptors] == callables
        assert len(descriptors) == count
    assert helper.flattened_callables == list(leaves)


def test_te_capture_uses_requested_runtime_schedule_when_global_disagrees(monkeypatch) -> None:
    from types import SimpleNamespace

    import megatron.core.transformer.cuda_graphs as cuda_graphs

    requested_count = 3
    global_count = 2
    observed_schedule_counts = []
    observed_capture_kwargs = []
    layer = SimpleNamespace(cuda_graphs=[])
    helper = TECudaGraphHelper.__new__(TECudaGraphHelper)
    helper.flattened_callables = [layer]
    helper.callables_per_chunk = [[layer]]
    helper.num_layers_per_chunk = [1]
    helper.num_model_chunks = 1
    helper.pp_group = SimpleNamespace(size=lambda: 2)
    helper.p2p_communicator = object()
    helper.tp_group = None
    helper.dp_cp_group = None
    helper.config = SimpleNamespace(
        overlap_moe_expert_parallel_comm=False,
        microbatch_group_size_per_vp_stage=1,
        cuda_graph_retain_backward_graph=False,
        cuda_graph_warmup_steps=1,
        sequence_parallel=False,
        fp8=None,
        fp4=None,
        fine_grained_activation_offloading=False,
    )
    helper._graphs_created = False
    helper._capture_finished = False
    helper._capture_gc_frozen = False

    monkeypatch.setattr(cuda_graphs, "get_num_microbatches", lambda: global_count)
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda _version: True)
    monkeypatch.setattr(helper, "_start_capturing", lambda: 0.0)
    monkeypatch.setattr(helper, "_finish_capturing", lambda _start_time: None)
    monkeypatch.setattr(helper, "_abort_capturing", lambda: None)
    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.schedules.get_pp_rank_microbatches",
        lambda num_microbatches, *_args, **_kwargs: (
            observed_schedule_counts.append(num_microbatches) or None,
            None,
            0,
            None,
        ),
    )

    def get_schedule_table(num_microbatches, *_args, **_kwargs):
        observed_schedule_counts.append(num_microbatches)
        return [(microbatch, 0) for microbatch in range(num_microbatches)]

    monkeypatch.setattr(
        "megatron.core.pipeline_parallel.schedules.get_schedule_table",
        get_schedule_table,
    )
    monkeypatch.setattr(
        helper,
        "_get_sample_arguments",
        lambda *_args, **_kwargs: (
            [()] * requested_count,
            [{} for _ in range(requested_count)],
        ),
    )

    def make_graphs(_callables, sample_args, **kwargs):
        assert len(sample_args) == requested_count
        observed_capture_kwargs.append(kwargs)
        return tuple(object() for _ in range(requested_count))

    monkeypatch.setattr(cuda_graphs, "make_graphed_callables", make_graphs)

    captured = helper._capture_cuda_graph_lists(num_microbatches=requested_count)

    assert observed_schedule_counts == [requested_count, requested_count]
    assert observed_capture_kwargs[0]["num_warmup_iters"] == 3
    assert helper.num_microbatches == requested_count
    assert captured == ((layer, tuple(layer.cuda_graphs)),)


def test_te_hybrid_mtp_static_inputs_are_owned_by_inner_leaves(monkeypatch) -> None:
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    helper, _, _, leaves = _make_task7_te_helper(monkeypatch)
    packed = PackedSeqParams(
        seq_aux_loss_sample_ids=torch.tensor([0, 0], dtype=torch.int64),
        seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.int64),
        seq_aux_loss_max_samples=2,
    )
    calls = []

    for leaf in leaves:
        if leaf is leaves[2]:

            def get_static_inputs(seq_length, micro_batch_size, packed_seq_params, leaf=leaf):
                calls.append((leaf, seq_length, micro_batch_size, packed_seq_params))
                return {"hidden_states": torch.ones((2, 1, 4))}

        else:

            def get_static_inputs(seq_length, micro_batch_size, leaf=leaf):
                calls.append((leaf, seq_length, micro_batch_size, None))
                return {"hidden_states": torch.ones((2, 1, 4))}

        leaf.get_layer_static_inputs = get_static_inputs

    helper.seq_length = 2
    helper.micro_batch_size = 1
    helper.sample_packed_seq_params = packed
    helper.num_microbatches = 1
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda _version: True)

    sample_args, sample_kwargs = helper._get_sample_arguments([1, -1])

    assert len(sample_args) == len(leaves)
    assert len(sample_kwargs) == len(leaves)
    assert [call[0] for call in calls] == list(leaves)
    assert calls[2][3] is packed
    assert not any(key.startswith("_moe_packed_seq_params_") for key in sample_kwargs[0])
    assert {
        key for key in sample_kwargs[1] if key.startswith("_moe_packed_seq_params_")
    } == {
        "_moe_packed_seq_params_seq_aux_loss_sample_ids",
        "_moe_packed_seq_params_seq_aux_loss_num_samples",
    }
    assert not any(key.startswith("_moe_packed_seq_params_") for key in sample_kwargs[2])
    assert not any(key.startswith("_moe_packed_seq_params_") for key in sample_kwargs[3])


@pytest.mark.parametrize(
    ("cuda_graph_modules", "expected"),
    [
        ([], True),
        ([CudaGraphModule.moe], True),
        ([CudaGraphModule.moe_router], True),
        ([CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess], True),
        ([CudaGraphModule.attn], False),
        ([CudaGraphModule.mamba], False),
    ],
)
def test_te_moe_sample_ownership_follows_canonical_inner_leaf_descriptor(
    monkeypatch, cuda_graph_modules, expected
) -> None:
    from types import SimpleNamespace

    from megatron.core.packed_seq_params import MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX
    from megatron.core.transformer.cuda_graphs import (
        _add_moe_packed_seq_params_to_te_cuda_graph_sample_kwargs,
    )
    from megatron.core.transformer.identity_op import IdentityOp

    helper, _, stack, leaves = _make_task7_te_helper(monkeypatch)
    descriptor = helper.layer_descriptors_per_chunk[0][1]
    assert descriptor.layer is leaves[1]
    assert descriptor.mtp_owner is not None
    assert descriptor.layer is not descriptor.mtp_owner
    assert descriptor.layer is not stack
    assert isinstance(descriptor.layer.self_attention, IdentityOp)
    packed = PackedSeqParams(
        seq_aux_loss_sample_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64),
        seq_aux_loss_max_samples=3,
    )
    sample_kwargs = {}
    config = SimpleNamespace(cuda_graph_modules=cuda_graph_modules)

    _add_moe_packed_seq_params_to_te_cuda_graph_sample_kwargs(
        descriptor, config, sample_kwargs, packed
    )

    moe_keys = {
        key for key in sample_kwargs if key.startswith(MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX)
    }
    if expected:
        assert moe_keys == {
            f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_sample_ids",
            f"{MOE_CUDA_GRAPH_PACKED_SEQ_PARAMS_PREFIX}seq_aux_loss_num_samples",
        }
    else:
        assert not moe_keys


def test_set_current_microbatch_targets_each_hybrid_mtp_leaf_graph(monkeypatch) -> None:
    from megatron.core.transformer.module import GraphableMegatronModule

    _, chunk, stack, leaves = _make_task7_te_helper(monkeypatch)
    selections = []

    class FakeGraph:
        def __init__(self, leaf_index, graph_index):
            self.name = (leaf_index, graph_index)

        def __call__(self, *_args, **_kwargs):
            selections.append(("forward", self.name))
            return self.name

        def backward_dw(self):
            selections.append(("backward_dw", self.name))

    class FakeBackwardDWWrapper:
        def __init__(self):
            self.callable = None

        def set_graphed_backward_dw_callable(self, callable_):
            self.callable = callable_

    managers_and_banks = []
    for leaf_index, leaf in enumerate(leaves):
        leaf.cuda_graph_manual_hooks = []
        leaf._get_te_cuda_graph_replay_args = lambda *args, **kwargs: (args, kwargs)
        manager, bank, _ = _make_task9_active_bank(
            leaf, [FakeGraph(leaf_index, 0), FakeGraph(leaf_index, 1)]
        )
        managers_and_banks.append((manager, bank))

    stack.current_microbatch = 101
    stack.backward_dw_wrapper = object()
    set_current_microbatch(chunk, 3)

    for leaf_index, leaf in enumerate(leaves):
        assert leaf.current_microbatch == 3
        assert GraphableMegatronModule._te_cuda_graph_replay(leaf, torch.ones(1)) == (leaf_index, 1)

    for leaf in (leaves[0], leaves[1], leaves[3]):
        leaf.backward_dw_wrapper = FakeBackwardDWWrapper()
        GraphableMegatronModule.set_te_cuda_graph_backward_dw_wrapper(leaf)
        leaf.backward_dw_wrapper.callable()

    assert stack.current_microbatch == 101
    assert selections == [
        ("forward", (0, 1)),
        ("forward", (1, 1)),
        ("forward", (2, 1)),
        ("forward", (3, 1)),
        ("backward_dw", (0, 1)),
        ("backward_dw", (1, 1)),
        ("backward_dw", (3, 1)),
    ]
    assert not hasattr(leaves[2], "backward_dw_wrapper")
    for manager, bank in managers_and_banks:
        bank.reset()
        manager.close()


def _make_simple_module(config):
    return _SimpleModule(config).cuda().eval()


def _make_simple_non_module(config):
    return _SimpleNonModule(config)


class TestInlineCaptureManager:
    """Tests for CudaGraphManager with inline_capture, function_name, eager, and cache_key."""

    def _make_config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            inference_rng_tracker=True,
        )

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "make_module",
        [
            pytest.param(_make_simple_module, id="nn_module"),
            pytest.param(_make_simple_non_module, id="plain_class"),
        ],
    )
    @torch.inference_mode()
    def test_inline_capture_matches_eager(self, make_module):
        """Inline-captured graph output must match eager execution."""
        config = self._make_config()
        module = make_module(config)

        # Get eager reference before wrapping
        x = torch.randn(4, config.hidden_size, device="cuda")
        eager_out = module.my_op(x).clone()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        # First call captures, second replays
        graph_out_1 = module.my_op(x)
        graph_out_2 = module.my_op(x)
        assert torch.equal(eager_out, graph_out_1)
        assert torch.equal(eager_out, graph_out_2)
        assert len(mgr.cudagraph_runners) == 1
        assert mgr.cudagraph_runners[0].fwd_graph_recorded

    @torch.inference_mode()
    def test_eager_bypass(self):
        """eager=True must bypass graph capture entirely."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        _ = module.my_op(x, eager=True)
        _ = module.my_op(x, eager=True)
        assert len(mgr.cudagraph_runners) == 0, "eager=True should not create runners"

    @torch.inference_mode()
    def test_cache_key_routing(self):
        """Different cache_keys must create separate runners."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="key_a")
        module.my_op(x, cache_key="key_b")

        assert len(mgr.cudagraph_runners) == 2
        assert mgr.custom_cudagraphs_lookup_table["key_a"] is not None
        assert mgr.custom_cudagraphs_lookup_table["key_b"] is not None
        assert (
            mgr.custom_cudagraphs_lookup_table["key_a"]
            is not mgr.custom_cudagraphs_lookup_table["key_b"]
        )

        # Same key reuses the runner
        module.my_op(x, cache_key="key_a")
        assert len(mgr.cudagraph_runners) == 2

    @torch.inference_mode()
    def test_num_warmup_steps_override(self):
        """num_warmup_steps on the manager must override the config value on runners."""
        config = self._make_config()
        config.cuda_graph_warmup_steps = 3
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="test")

        runner = mgr.cudagraph_runners[0]
        assert (
            runner.num_warmup_steps == 0
        ), f"Expected 0 warmup steps (manager override), got {runner.num_warmup_steps}"


class TestSkipFp8WeightUpdateTensor:
    """Regression test for the TE 2.15 ``set_skip_fp8_weight_update_tensor`` removal."""

    @staticmethod
    def _read_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        getter = getattr(FP8GlobalStateManager, "get_skip_fp8_weight_update_tensor", None)
        if getter is not None:
            return getter()
        return FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor

    @staticmethod
    def _reset_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        if "skip_fp8_weight_update_tensor" in vars(FP8GlobalStateManager):
            FP8GlobalStateManager.skip_fp8_weight_update_tensor = None
        qstate = getattr(FP8GlobalStateManager, "quantization_state", None)
        if qstate is not None and hasattr(qstate, "skip_fp8_weight_update_tensor"):
            qstate.skip_fp8_weight_update_tensor = None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sets_value_in_place(self):
        """Helper writes the right value and reuses the same storage across calls."""
        from megatron.core.transformer.cuda_graphs import _set_skip_fp8_weight_update_tensor

        self._reset_skip_tensor()
        try:
            _set_skip_fp8_weight_update_tensor(True)
            t = self._read_skip_tensor()
            assert t.shape == (1,) and t.dtype == torch.float32 and t.is_cuda
            assert t.item() == 1.0

            # data_ptr must stay stable so captured cudagraphs read the same address.
            ptr = t.data_ptr()
            _set_skip_fp8_weight_update_tensor(False)
            assert self._read_skip_tensor().data_ptr() == ptr
            assert self._read_skip_tensor().item() == 0.0
        finally:
            self._reset_skip_tensor()


if __name__ == "__main__":

    test = TestParallelTransformerBlockCudagraphs()
    test.setup_method(method=None)
    test.test_gpu_cudagraph()
    test.teardown_method(method=None)

    llava_test = TestLLaVACudaGraph()
    llava_test.setup_method(method=None)
    llava_test.test_llava_cudagraph_is_last_layer_logic()
    llava_test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_moe_partial_cudagraph(4, True, "alltoall")
    test.teardown_method(method=None)
