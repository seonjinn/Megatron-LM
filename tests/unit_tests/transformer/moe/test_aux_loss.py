# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import dataclasses

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    clear_aux_losses_tracker,
    get_default_pg_collection,
    get_moe_layer_wise_logging_tracker,
)
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer

try:
    # Check availability of TE fused router aux ops
    from megatron.core.extensions.transformer_engine import (
        fused_compute_score_for_moe_aux_loss as _fused_compute_score_for_moe_aux_loss,
    )
    from megatron.core.extensions.transformer_engine import (
        fused_moe_aux_loss as _fused_moe_aux_loss,
    )

    HAVE_ROUTER_FUSION = (
        _fused_compute_score_for_moe_aux_loss is not None and _fused_moe_aux_loss is not None
    )
except Exception:  # pragma: no cover - defensive
    HAVE_ROUTER_FUSION = False


class _SingleRankProcessGroup:
    """Minimal process-group surface for CPU-only router validation tests."""

    def size(self) -> int:
        """Return the single-rank world size."""
        return 1


class _RejectTensorToHostScalarMode(TorchDispatchMode):
    """Reject Tensor-to-Python scalar conversions in graph-safe validation tests."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("Tensor ownership validation synchronized with the host")
        return func(*args, **(kwargs or {}))


def _new_cpu_fixed_capacity_router() -> TopKRouter:
    """Build a real single-rank router without initializing CUDA or distributed state."""
    group = _SingleRankProcessGroup()
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = group
    pg_collection.cp = group
    pg_collection.tp_cp = group
    pg_collection.tp_dp_cp = group
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        num_moe_experts=4,
        use_cpu_initialization=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_topk=2,
        moe_aux_loss_coeff=1.0,
        moe_router_dtype="fp32",
        params_dtype=torch.float32,
        add_bias_linear=False,
    )
    router = TopKRouter(config=config, pg_collection=pg_collection)
    router.set_layer_number(0)
    return router


class AuxlossTestContainer(MoEModelTestContainer):
    def partition_input(self, input):
        partitioned_input = input.chunk(
            parallel_state.get_tensor_and_context_parallel_world_size(), dim=0
        )[parallel_state.get_tensor_and_context_parallel_rank()]
        output = partitioned_input.clone().detach()
        output.requires_grad = True
        return output

    @pytest.mark.internal
    def aux_loss_test(self, input, baseline_grad, loss_name):
        partitioned_input = self.partition_input(input)
        moe_layer = self.moe_layer
        probs, indices = apply_module(moe_layer.router)(partitioned_input)
        probs.sum().mul_(0).backward()
        aux_loss_grad = partitioned_input.grad
        torch.distributed.barrier()
        ans = self.partition_input(baseline_grad)
        assert torch.allclose(aux_loss_grad, ans), f"Diff: {(aux_loss_grad/ans).mean()}"
        loss = get_moe_layer_wise_logging_tracker()[loss_name]['values']
        assert loss > 0, "Loss should be greater than 0"
        clear_aux_losses_tracker()

        with torch.no_grad():
            probs, indices = apply_module(moe_layer.router)(partitioned_input)
            loss = get_moe_layer_wise_logging_tracker()[loss_name]['values']
            assert loss == 0, "Loss should be 0"
            clear_aux_losses_tracker()


class TestAuxLoss:
    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(self.input)
        probs.sum().mul_(0).backward()  # zero out the main gradients
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_allgather_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "load_balancing_loss")

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_a2a_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "load_balancing_loss")


class TestSeqAuxLoss:
    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(self.input)
        probs.sum().mul_(0).backward()  # zero out the main gradients
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_a2a_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "seq_load_balancing_loss")


class TestPerTokenAuxLoss:
    """Regression test for the aux_loss TP/CP scaling fix under
    --calculate-per-token-loss. Computes a baseline aux-loss input
    gradient at (tp=1, cp=1) and asserts that each parametrized
    (tp, ep, cp) config produces a matching gradient on each rank's
    local input slice. Without the fix, the per-rank scale on aux_loss
    would shrink with tp_cp_size and the assertion would fail at any
    config with tp_size > 1 or cp_size > 1.
    """

    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
            calculate_per_token_loss=True,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(self.input)
        probs.sum().mul_(0).backward()
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_per_token_aux_loss_invariant_to_tp_cp(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
            calculate_per_token_loss=True,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "load_balancing_loss")


class TestRouterAuxLoss:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        # Default configuration
        self.default_transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=8,
            num_moe_experts=32,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=8,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )

    def new_router(self, **kwargs):
        """Create a new router with updated configuration.

        Args:
            **kwargs: Configuration parameters to update in the default config.

        Returns:
            Router: A new router instance with the specified configuration.
        """
        pg_collection = get_default_pg_collection()
        # Create a new config with updated parameters
        new_transformer_config = dataclasses.replace(self.default_transformer_config, **kwargs)

        # Create the router with the updated config
        router = TopKRouter(config=new_transformer_config, pg_collection=pg_collection)
        router.set_layer_number(0)
        return router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_seq_aux_loss(self, tp_size, ep_size, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(42)

        # Test that with batch_size=1, aux_loss and seq_aux_loss should be the same
        aux_loss_router = self.new_router(
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()
        seq_aux_loss_router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        # Set identical weights for fair comparison
        with torch.no_grad():
            seq_aux_loss_router.weight.copy_(aux_loss_router.weight)

        ### MBS=1 case: results should be identical ###
        clear_aux_losses_tracker()
        seq_len = 32
        batch_size = 1
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn(
                (seq_len, batch_size, aux_loss_router.config.hidden_size),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

        # Forward pass for aux_loss router
        aux_loss_router.weight.grad = None
        scores1, routing_map1 = aux_loss_router(hidden_states)
        loss1 = scores1.sum()
        loss1.backward()
        grad1 = aux_loss_router.weight.grad.clone()

        # Forward pass for seq_aux_loss router
        seq_aux_loss_router.weight.grad = None
        scores2, routing_map2 = seq_aux_loss_router(hidden_states)
        loss2 = scores2.sum()
        loss2.backward()
        grad2 = seq_aux_loss_router.weight.grad.clone()

        # For batch_size=1, they should produce the same results
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss = tracker["load_balancing_loss"]["values"][0]
        seq_aux_loss = tracker["seq_load_balancing_loss"]["values"][0]

        reduce_from_tensor_model_parallel_region(aux_loss, aux_loss_router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(seq_aux_loss, aux_loss_router.tp_cp_group)

        assert torch.equal(routing_map1, routing_map2)
        assert torch.equal(grad1, grad2)
        assert torch.equal(scores1, scores2)
        assert aux_loss == seq_aux_loss, f"aux_loss: {aux_loss}, seq_aux_loss: {seq_aux_loss}"

        ### MBS=2 case ###
        clear_aux_losses_tracker()
        batch_size = 2
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn(
                (seq_len, batch_size, aux_loss_router.config.hidden_size),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

        # Forward pass for aux_loss router
        aux_loss_router.weight.grad = None
        scores_first_batch, _ = aux_loss_router(hidden_states[:, 0:1, :])
        scores_second_batch, _ = aux_loss_router(hidden_states[:, 1:, :])

        # setting grad to 0 to only backward aux loss
        (scores_first_batch + scores_second_batch).backward(torch.zeros_like(scores_first_batch))

        grad1 = aux_loss_router.weight.grad.clone()

        # Forward pass for seq_aux_loss router
        seq_aux_loss_router.weight.grad = None
        scores2, routing_map2 = seq_aux_loss_router(hidden_states)
        # setting grad to 0 to only backward aux loss
        scores2.backward(torch.zeros_like(scores2))
        grad2 = seq_aux_loss_router.weight.grad.clone() * 2

        aux_loss = tracker["load_balancing_loss"]["values"][0] / 2
        seq_aux_loss = tracker["seq_load_balancing_loss"]["values"][0]
        reduce_from_tensor_model_parallel_region(aux_loss, aux_loss_router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(seq_aux_loss, aux_loss_router.tp_cp_group)

        torch.testing.assert_close(aux_loss, seq_aux_loss)
        torch.testing.assert_close(grad1, grad2)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("with_padding", [False, True])
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_seq_aux_loss_mbs_invariant_per_token_loss(
        self, tp_size, ep_size, cp_size, with_padding
    ):
        """seq_aux_loss gradient must be invariant to MBS under --calculate-per-token-loss.

        The same global batch is processed as N micro-batches of size 1 (MBS=1) and as one
        micro-batch of size N (MBS=N). Both cover the same tokens, so the finalize-time
        1/total_tokens normalization is an identical constant and the accumulated
        router-weight aux gradients must match. Before the fix (valid_token_count dropped the
        bsz factor), the MBS=N gradient is scaled by 1/N and the assertion fails. The padding
        case additionally checks the correction uses valid (non-padded) token counts.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(42)
        clear_aux_losses_tracker()

        router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            calculate_per_token_loss=True,
            # fp32 weights so the MBS=1 gradient (accumulated over N backward passes)
            # is not degraded by bf16 rounding relative to the single MBS=N backward.
            params_dtype=torch.float32,
            bf16=False,
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        seq_len = 32
        num_seqs = 4
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn(
                (seq_len, num_seqs, router.config.hidden_size),
                device=torch.device("cuda"),
                dtype=torch.float32,
            )
        padding_mask = None
        if with_padding:
            # True marks padding tokens (second half of each sequence).
            padding_mask = torch.zeros((seq_len, num_seqs), dtype=torch.bool, device="cuda")
            padding_mask[seq_len // 2 :, :] = True

        def run(indices):
            pmask = None if padding_mask is None else padding_mask[:, indices]
            scores, _ = router(hidden_states[:, indices, :].contiguous(), padding_mask=pmask)
            scores.backward(torch.zeros_like(scores))  # isolate the aux-loss gradient
            clear_aux_losses_tracker()

        # MBS=1: N micro-batches of size 1, accumulating the aux-loss gradient.
        router.weight.grad = None
        for b in range(num_seqs):
            run(slice(b, b + 1))
        grad_mbs1 = router.weight.grad.clone()

        # MBS=N: a single micro-batch of size N.
        router.weight.grad = None
        run(slice(0, num_seqs))
        grad_mbsN = router.weight.grad.clone()

        torch.testing.assert_close(grad_mbs1, grad_mbsN)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="CUDA or TE fused router ops not available",
    )
    @pytest.mark.parametrize("aux_type", ["aux_loss", "seq_aux_loss", "global_aux_loss"])
    def test_aux_loss_fusion_equivalence(self, aux_type):
        # Compare fused vs unfused aux loss path to ensure numerical equivalence
        router_ref = self.new_router(
            moe_router_load_balancing_type=aux_type, moe_aux_loss_coeff=1.0, moe_router_dtype="fp32"
        ).cuda()
        router_fused = self.new_router(
            moe_router_load_balancing_type=aux_type, moe_aux_loss_coeff=1.0, moe_router_dtype="fp32"
        ).cuda()

        with torch.no_grad():
            router_fused.weight.copy_(router_ref.weight)

        hidden_states = torch.randn((32, 2, router_ref.config.hidden_size)).cuda().bfloat16()

        # Map aux type to its tracker key
        loss_name_map = {
            "aux_loss": "load_balancing_loss",
            "seq_aux_loss": "seq_load_balancing_loss",
            "global_aux_loss": "global_load_balancing_loss",
        }
        loss_name = loss_name_map[aux_type]

        # Unfused
        router_ref.config.moe_router_fusion = False
        clear_aux_losses_tracker()
        router_ref.weight.grad = None
        scores_ref, routing_ref = router_ref(hidden_states)
        # Backward zeros to isolate aux-loss-only gradient contribution
        scores_ref.backward(torch.zeros_like(scores_ref))
        grad_ref = router_ref.weight.grad.clone()
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss_ref = tracker[loss_name]["values"][0]
        reduce_from_tensor_model_parallel_region(aux_loss_ref, router_ref.tp_cp_group)

        # Fused
        router_fused.config.moe_router_fusion = True
        clear_aux_losses_tracker()
        router_fused.weight.grad = None
        scores_fused, routing_fused = router_fused(hidden_states)
        scores_fused.backward(torch.zeros_like(scores_fused))
        grad_fused = router_fused.weight.grad.clone()
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss_fused = tracker[loss_name]["values"][0]
        reduce_from_tensor_model_parallel_region(aux_loss_fused, router_fused.tp_cp_group)

        # Checks
        assert torch.equal(routing_ref, routing_fused)
        torch.testing.assert_close(scores_ref, scores_fused, rtol=2.0e-2, atol=1.0e-3)
        torch.testing.assert_close(aux_loss_ref, aux_loss_fused)
        torch.testing.assert_close(grad_ref, grad_fused)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_global_aux_loss(self, tp_size, ep_size, cp_size):
        clear_aux_losses_tracker()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )

        router = self.new_router(
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=1.0,
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        seq_len = 32
        # Verify global tokens tracker initialized
        assert router.global_tokens_per_expert is not None
        assert router.ga_steps == 0

        # First microbatch
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn((seq_len, 2, router.config.hidden_size)).cuda().bfloat16()
        num_local_tokens = seq_len * 2
        scores, routing_map = router(hidden_states)
        # Check that global tokens were counted
        assert torch.all(router.global_tokens_per_expert >= 0)
        assert (
            router.global_tokens_per_expert.sum()
            == num_local_tokens * router.tp_dp_cp_group.size() * router.ga_steps * router.topk
        )
        global_aux_loss_1 = get_moe_layer_wise_logging_tracker()["global_load_balancing_loss"][
            "values"
        ][0]
        reduce_from_tensor_model_parallel_region(global_aux_loss_1, router.tp_dp_cp_group)
        assert global_aux_loss_1 >= 1

        # When DP size is 1, the global aux loss should match the aux loss
        # for the first microbatch
        if get_default_pg_collection().tp_dp_cp.size() == tp_size:
            ref_router = self.new_router(
                moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=1.0
            ).cuda()
            with torch.no_grad():
                ref_router.weight.copy_(router.weight)
            ref_scores, ref_routing_map = ref_router(hidden_states)
            aux_loss = get_moe_layer_wise_logging_tracker()["load_balancing_loss"]["values"][0]
            reduce_from_tensor_model_parallel_region(aux_loss, router.tp_cp_group)

            assert torch.equal(
                aux_loss, global_aux_loss_1
            ), f"aux_loss: {aux_loss}, global_aux_loss_1: {global_aux_loss_1}"

        clear_aux_losses_tracker()

        # Get current tokens count to verify accumulation
        current_per_expert = router.global_tokens_per_expert.clone()

        # Second microbatch - should accumulate
        hidden_states = torch.randn((seq_len, 2, router.config.hidden_size)).cuda().bfloat16()
        scores, routing_map = router(hidden_states)
        global_aux_loss_2 = get_moe_layer_wise_logging_tracker()["global_load_balancing_loss"][
            "values"
        ][0]
        reduce_from_tensor_model_parallel_region(global_aux_loss_2, router.tp_dp_cp_group)
        assert torch.all(global_aux_loss_2 >= 1), f"global_aux_loss_2: {global_aux_loss_2}"

        # Verify tokens were accumulated
        assert router.ga_steps == 2
        assert torch.any(router.global_tokens_per_expert > current_per_expert)
        clear_aux_losses_tracker()

        # Reset global tracker
        router.reset_global_aux_loss_tracker()
        assert router.ga_steps == 0
        assert torch.all(router.global_tokens_per_expert == 0)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_combined_aux_loss(self, tp_size, ep_size, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        clear_aux_losses_tracker()

        # Test combined aux loss types
        router = self.new_router(
            moe_router_load_balancing_type=["aux_loss", "seq_aux_loss", "global_aux_loss"],
            moe_aux_loss_coeff=[0.5, 1.0, 2.0],
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        # Verify all aux loss trackers initialized
        assert router.global_tokens_per_expert is not None
        assert router.ga_steps == 0

        # Execute forward pass
        hidden_states = torch.randn((32, 2, router.config.hidden_size)).cuda().bfloat16()
        router.weight.grad = None
        scores, routing_map = router(hidden_states)
        loss = scores.sum()
        loss.backward()

        aux_loss = get_moe_layer_wise_logging_tracker()["load_balancing_loss"]["values"][0]
        seq_aux_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
        global_aux_loss = get_moe_layer_wise_logging_tracker()["global_load_balancing_loss"][
            "values"
        ][0]

        reduce_from_tensor_model_parallel_region(aux_loss, router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(seq_aux_loss, router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(global_aux_loss, router.tp_dp_cp_group)

        assert aux_loss >= 1
        assert seq_aux_loss >= 1
        assert global_aux_loss >= 1

        # Verify gradient is non-zero (aux losses are being applied)
        assert router.weight.grad.abs().sum() > 0

        # Verify method to get aux loss coeffs works properly
        assert router.get_aux_loss_coeff("aux_loss") == 0.5
        assert router.get_aux_loss_coeff("seq_aux_loss") == 1.0
        assert router.get_aux_loss_coeff("global_aux_loss") == 2.0
        assert router.get_aux_loss_coeff("non_existent_type") == 0.0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_force_balanced_aux_loss(self, tp_size, ep_size, cp_size):
        """Test if aux loss is 1.0 when using uniform routing"""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        clear_aux_losses_tracker()
        seq_len = 32
        batch_size = 2

        # Create router with each aux loss type
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            router = self.new_router(
                moe_router_load_balancing_type=aux_loss_type,
                moe_aux_loss_coeff=1.0,
                moe_router_dtype="fp32",
                tensor_model_parallel_size=tp_size,
                expert_tensor_parallel_size=ep_size,
                context_parallel_size=cp_size,
            ).cuda()
            # create uniform weights
            with torch.no_grad():
                router.weight.copy_(torch.ones_like(router.weight) / router.weight.numel())

            # Create uniform logits (all experts equally likely)
            hidden_size = router.config.hidden_size
            num_experts = router.config.num_moe_experts

            loss_name = {
                "aux_loss": "load_balancing_loss",
                "seq_aux_loss": "seq_load_balancing_loss",
                "global_aux_loss": "global_load_balancing_loss",
            }[aux_loss_type]

            hidden_states = torch.randn(
                (seq_len, batch_size, hidden_size),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

            # Get routing scores and map
            scores, routing_map = router(hidden_states)
            aux_loss = get_moe_layer_wise_logging_tracker()[loss_name]["values"][0]
            if aux_loss_type == "global_aux_loss":
                reduce_from_tensor_model_parallel_region(aux_loss, router.tp_dp_cp_group)
            else:
                reduce_from_tensor_model_parallel_region(aux_loss, router.tp_cp_group)
            assert aux_loss.item() == 1, f"{aux_loss_type}: {aux_loss.item()}"
            clear_aux_losses_tracker()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_seq_aux_loss_flattened_packed_sequences(self):
        """Test that TransformerLayer reshapes flattened packed sequences for MoE.

        When inter-document masking flattens MBS > 1 into [mbs*S, 1, H],
        TransformerLayer._maybe_reshape_for_moe should restore [S, mbs, H] so
        the router computes seq_aux_loss per sample. This test runs a forward
        pass through a real TransformerLayer with an MoE MLP and verifies that passing
        packed_seq_params with the flattened input produces the same
        seq_load_balancing_loss as the un-flattened input.
        """
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.transformer.transformer_layer import TransformerLayer

        seq_len = 128
        batch_size = 4
        hidden_size = 12

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=32,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=1.0,
            moe_ffn_hidden_size=64,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=32, moe_grouped_gemm=False)
        layer = TransformerLayer(transformer_config, submodules).cuda().bfloat16()
        assert layer.is_moe_layer

        hidden_states = torch.randn(
            (seq_len, batch_size, hidden_size), device=torch.device("cuda"), dtype=torch.bfloat16
        )

        def _get_seq_aux_loss(hidden_states, packed_seq_params=None):
            clear_aux_losses_tracker()
            layer._forward_mlp(hidden_states, packed_seq_params=packed_seq_params)
            return get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]

        # Baseline: forward with the original [seq_len, mbs, H] shape.
        loss_baseline = _get_seq_aux_loss(hidden_states)

        # Flatten to [mbs*seq_len, 1, H] the same way the dataloader does.
        flattened = hidden_states.transpose(0, 1).reshape(batch_size * seq_len, 1, -1)

        # With packed_seq_params, _maybe_reshape_for_moe restores [S, mbs, H]
        # before the router, recovering the correct per-sample loss.
        packed_seq_params = PackedSeqParams(tokens_per_sample=seq_len)
        loss_with_implicit_reshape = _get_seq_aux_loss(
            flattened, packed_seq_params=packed_seq_params
        )

        torch.testing.assert_close(loss_with_implicit_reshape, loss_baseline)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("topk", (1, 2), ids=("top1", "top2"))
    @pytest.mark.parametrize("score_function", ("softmax", "sigmoid", "sqrtsoftplus"))
    @pytest.mark.parametrize(
        "logical_lengths,physical_lengths,max_samples,dummy_tail",
        (
            pytest.param((3, 5), (8, 8), 4, 0, id="two_equal_physical_unused_rows"),
            pytest.param((3, 5), (4, 8), 4, 0, id="two_unequal_physical_unused_rows"),
            pytest.param((1, 7, 2), (1, 7, 2), 3, 0, id="three_exact_n_eq_capacity"),
            pytest.param((1, 7, 2), (4, 8, 4), 5, 4, id="three_aligned_dummy_tail"),
            pytest.param((5,), (5,), 1, 0, id="one_exact_n_eq_capacity"),
        ),
    )
    def test_variable_length_packed_seq_aux_loss_matches_padded_router_oracle(
        self,
        logical_lengths,
        physical_lengths,
        max_samples,
        dummy_tail,
        score_function,
        topk,
    ):
        """Static segmented ownership preserves padded seq_aux routing and gradients."""
        router = self.new_router(
            num_moe_experts=8,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_topk=topk,
            moe_router_score_function=score_function,
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            params_dtype=torch.float32,
            bf16=False,
        ).cuda()
        hidden_size = router.config.hidden_size
        samples = [
            torch.randn((length, hidden_size), dtype=torch.float32, device="cuda")
            for length in logical_lengths
        ]

        def run(hidden_states, padding_mask, **ownership):
            clear_aux_losses_tracker()
            router.weight.grad = None
            hidden_states = hidden_states.detach().clone().requires_grad_(True)
            probs, routing_map = router(hidden_states, padding_mask=padding_mask, **ownership)
            probs.backward(torch.zeros_like(probs))
            aux_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"][
                "values"
            ][0].clone()
            return (
                probs.detach().clone(),
                routing_map.detach().clone(),
                aux_loss,
                hidden_states.grad.detach().clone(),
                router.weight.grad.detach().clone(),
            )

        max_length = max(logical_lengths)
        padded = torch.zeros(
            (max_length, len(samples), hidden_size), dtype=torch.float32, device="cuda"
        )
        padded_mask = torch.ones(
            (max_length, len(samples)), dtype=torch.bool, device="cuda"
        )
        for sample_id, sample in enumerate(samples):
            padded[: sample.shape[0], sample_id] = sample
            padded_mask[: sample.shape[0], sample_id] = False
        padded_result = run(padded, padded_mask)

        packed_parts = []
        packed_masks = []
        sample_ids = []
        for sample_id, (sample, physical_length) in enumerate(zip(samples, physical_lengths)):
            packed_parts.append(
                torch.cat(
                    (
                        sample,
                        torch.zeros(
                            (physical_length - sample.shape[0], hidden_size),
                            dtype=sample.dtype,
                            device=sample.device,
                        ),
                    )
                )
            )
            packed_masks.append(
                torch.arange(physical_length, device="cuda") >= sample.shape[0]
            )
            sample_ids.extend([sample_id] * physical_length)
        if dummy_tail:
            packed_parts.append(
                torch.zeros((dummy_tail, hidden_size), dtype=torch.float32, device="cuda")
            )
            packed_masks.append(torch.ones(dummy_tail, dtype=torch.bool, device="cuda"))
            sample_ids.extend([0] * dummy_tail)
        packed = torch.cat(packed_parts).unsqueeze(1)
        packed_mask = torch.cat(packed_masks).unsqueeze(1)
        packed_result = run(
            packed,
            packed_mask,
            seq_aux_loss_sample_ids=torch.tensor(sample_ids, dtype=torch.long, device="cuda"),
            seq_aux_loss_num_samples=torch.tensor(len(samples), dtype=torch.long, device="cuda"),
            seq_aux_loss_max_samples=max_samples,
        )

        padded_probs, padded_map, padded_loss, padded_grad, padded_weight_grad = padded_result
        packed_probs, packed_map, packed_loss, packed_grad, packed_weight_grad = packed_result
        padded_valid = ~padded_mask.transpose(0, 1)
        packed_valid = ~packed_mask[:, 0]
        torch.testing.assert_close(
            packed_probs[packed_valid],
            padded_probs.view(max_length, len(samples), -1).transpose(0, 1)[padded_valid],
        )
        assert torch.equal(
            packed_map[packed_valid],
            padded_map.view(max_length, len(samples), -1).transpose(0, 1)[padded_valid],
        )
        assert torch.equal(
            packed_map[packed_valid].sum(dim=-1),
            torch.full(
                (sum(logical_lengths),), topk, dtype=torch.long, device=packed_map.device
            ),
        )
        torch.testing.assert_close(packed_loss, padded_loss)
        torch.testing.assert_close(
            packed_grad[:, 0][packed_valid], padded_grad.transpose(0, 1)[padded_valid]
        )
        torch.testing.assert_close(packed_weight_grad, padded_weight_grad)
        assert torch.count_nonzero(padded_grad[padded_mask]) == 0
        assert torch.count_nonzero(packed_grad[:, 0][~packed_valid]) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,cp_size,sequence_parallel",
        (
            pytest.param(1, 1, False, id="tp1_cp1"),
            pytest.param(2, 1, True, id="tp2_cp1_sp"),
            pytest.param(1, 2, False, id="tp1_cp2"),
            pytest.param(2, 2, True, id="tp2_cp2_sp"),
        ),
    )
    def test_variable_length_packed_seq_aux_loss_tp_cp_sp_matches_padded_oracle(
        self, tp_size: int, cp_size: int, sequence_parallel: bool
    ) -> None:
        """TP/CP/SP production sharding preserves the literal padded router oracle."""

        required_world_size = tp_size * cp_size
        world_size = torch.distributed.get_world_size()
        if world_size < required_world_size or world_size % required_world_size:
            pytest.skip(
                f"requires a world size divisible by TP*CP={required_world_size}, got {world_size}"
            )

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
        )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        router = self.new_router(
            num_moe_experts=4,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_topk=2,
            moe_router_score_function="softmax",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            params_dtype=torch.float32,
            bf16=False,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=sequence_parallel,
        ).cuda()
        with torch.no_grad():
            router.weight.copy_(
                torch.arange(
                    router.weight.numel(), dtype=torch.float32, device=router.weight.device
                ).reshape_as(router.weight)
                / router.weight.numel()
            )

        hidden_size = router.config.hidden_size
        padded_global = (
            torch.arange(8 * 2 * hidden_size, dtype=torch.float32, device="cuda")
            .reshape(8, 2, hidden_size)
            .div_(100.0)
        )
        padded_mask_global = torch.ones((8, 2), dtype=torch.bool, device="cuda")
        padded_mask_global[:3, 0] = False
        padded_mask_global[:5, 1] = False
        padded_tags_global = torch.full((8, 2), -1, dtype=torch.int64, device="cuda")
        padded_tags_global[:3, 0] = torch.arange(3, device="cuda")
        padded_tags_global[:5, 1] = 8 + torch.arange(5, device="cuda")

        packed_global = torch.zeros((16, 1, hidden_size), dtype=torch.float32, device="cuda")
        packed_global[:3, 0] = padded_global[:3, 0]
        packed_global[4:9, 0] = padded_global[:5, 1]
        packed_mask_global = torch.ones((16, 1), dtype=torch.bool, device="cuda")
        packed_mask_global[:3, 0] = False
        packed_mask_global[4:9, 0] = False
        packed_tags_global = torch.full((16, 1), -1, dtype=torch.int64, device="cuda")
        packed_tags_global[:3, 0] = torch.arange(3, device="cuda")
        packed_tags_global[4:9, 0] = 8 + torch.arange(5, device="cuda")
        packed_ids_global = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            dtype=torch.int64,
            device="cuda",
        )

        def cp_zigzag_shard(tensor: torch.Tensor) -> torch.Tensor:
            if cp_size == 1:
                return tensor
            shard_width = tensor.shape[0] // (2 * cp_size)
            first = tensor.narrow(0, cp_rank * shard_width, shard_width)
            mirrored_rank = 2 * cp_size - cp_rank - 1
            second = tensor.narrow(0, mirrored_rank * shard_width, shard_width)
            return torch.cat((first, second), dim=0)

        def maybe_sequence_parallel_shard(tensor: torch.Tensor) -> torch.Tensor:
            if not sequence_parallel:
                return tensor.contiguous()
            shard_width = tensor.shape[0] // tp_size
            return tensor.narrow(0, tp_rank * shard_width, shard_width).contiguous()

        padded = maybe_sequence_parallel_shard(
            torch.stack(
                [cp_zigzag_shard(padded_global[:, sample]) for sample in range(2)], dim=1
            )
        )
        padded_mask = maybe_sequence_parallel_shard(
            torch.stack(
                [cp_zigzag_shard(padded_mask_global[:, sample]) for sample in range(2)], dim=1
            )
        )
        padded_tags = maybe_sequence_parallel_shard(
            torch.stack(
                [cp_zigzag_shard(padded_tags_global[:, sample]) for sample in range(2)], dim=1
            )
        )

        physical_slices = (slice(0, 4), slice(4, 12), slice(12, 16))

        def shard_packed_segments(tensor: torch.Tensor) -> torch.Tensor:
            return maybe_sequence_parallel_shard(
                torch.cat([cp_zigzag_shard(tensor[part]) for part in physical_slices], dim=0)
            )

        packed = shard_packed_segments(packed_global)
        packed_mask = shard_packed_segments(packed_mask_global)
        packed_tags = shard_packed_segments(packed_tags_global)
        packed_ids = shard_packed_segments(packed_ids_global)

        def run(
            hidden_states: torch.Tensor,
            padding_mask: torch.Tensor,
            **ownership: object,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            clear_aux_losses_tracker()
            router.weight.grad = None
            local_input = hidden_states.detach().clone().requires_grad_(True)
            probs, routing_map = router(local_input, padding_mask=padding_mask, **ownership)
            MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=probs.device))
            probs.backward(torch.zeros_like(probs))
            tracker_loss = get_moe_layer_wise_logging_tracker()[
                "seq_load_balancing_loss"
            ]["values"][0].detach().clone()
            router_grad = router.weight.grad.detach().clone()
            torch.distributed.all_reduce(tracker_loss, group=router.tp_cp_group)
            torch.distributed.all_reduce(router_grad, group=router.tp_cp_group)
            return (
                probs.detach(),
                routing_map.detach(),
                tracker_loss,
                local_input.grad.detach(),
                router_grad,
            )

        padded_result = run(padded, padded_mask)
        packed_result = run(
            packed,
            packed_mask,
            seq_aux_loss_sample_ids=packed_ids,
            seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.int64, device="cuda"),
            seq_aux_loss_max_samples=3,
        )

        def gather_logical_valid(
            local_tensor: torch.Tensor, local_tags: torch.Tensor
        ) -> torch.Tensor:
            flattened_tags = local_tags.reshape(-1)
            flattened_values = local_tensor.reshape(flattened_tags.numel(), -1)
            valid = flattened_tags >= 0
            local_payload = (
                flattened_tags[valid].detach().cpu(),
                flattened_values[valid].detach().cpu(),
            )
            gathered_payloads = [None] * torch.distributed.get_world_size(router.tp_cp_group)
            torch.distributed.all_gather_object(
                gathered_payloads, local_payload, group=router.tp_cp_group
            )
            tags = torch.cat([payload[0] for payload in gathered_payloads])
            values = torch.cat([payload[1] for payload in gathered_payloads])
            order = torch.argsort(tags)
            tags = tags[order]
            assert torch.equal(tags, torch.tensor([0, 1, 2, 8, 9, 10, 11, 12]))
            return values[order]

        padded_probs, padded_map, padded_loss, padded_grad, padded_router_grad = padded_result
        packed_probs, packed_map, packed_loss, packed_grad, packed_router_grad = packed_result
        torch.testing.assert_close(
            gather_logical_valid(packed_probs, packed_tags),
            gather_logical_valid(padded_probs, padded_tags),
        )
        assert torch.equal(
            gather_logical_valid(packed_map, packed_tags),
            gather_logical_valid(padded_map, padded_tags),
        )
        torch.testing.assert_close(packed_loss, padded_loss)
        torch.testing.assert_close(
            gather_logical_valid(packed_grad, packed_tags),
            gather_logical_valid(padded_grad, padded_tags),
        )
        torch.testing.assert_close(packed_router_grad, padded_router_grad)
        padded_padding_grad = padded_grad.reshape(-1, hidden_size)[padded_tags.reshape(-1) < 0]
        packed_padding_grad = packed_grad.reshape(-1, hidden_size)[packed_tags.reshape(-1) < 0]
        assert torch.count_nonzero(padded_padding_grad) == 0
        assert torch.count_nonzero(packed_padding_grad) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "missing_field",
        (
            "seq_aux_loss_sample_ids",
            "seq_aux_loss_num_samples",
            "seq_aux_loss_max_samples",
        ),
    )
    def test_fixed_capacity_seq_aux_moe_route_rejects_missing_ownership(
        self, missing_field
    ):
        """The real MoELayer.route API rejects an incomplete ownership tuple."""
        container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        )
        hidden_states = torch.randn((8, 1, container.config.hidden_size), device="cuda")
        ownership = {
            "seq_aux_loss_sample_ids": torch.zeros(8, dtype=torch.long, device="cuda"),
            "seq_aux_loss_num_samples": torch.tensor(1, dtype=torch.long, device="cuda"),
            "seq_aux_loss_max_samples": 2,
        }
        ownership.pop(missing_field)

        with pytest.raises(ValueError, match="must be provided together"):
            container.moe_layer.route(hidden_states, **ownership)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "invalid_case,error_match",
        (
            pytest.param("max_samples_zero", "at least 1", id="max_samples_zero"),
            pytest.param("max_samples_bool", "at least 1", id="max_samples_bool"),
            pytest.param("ids_not_tensor", "sample_ids.*Tensor", id="ids_not_tensor"),
            pytest.param("ids_rank", "sample_ids.*shape", id="ids_rank"),
            pytest.param("ids_length", "sample_ids.*shape", id="ids_length"),
            pytest.param("ids_dtype", "sample_ids.*torch.int64", id="ids_dtype"),
            pytest.param("ids_device", "sample_ids.*same device", id="ids_device"),
            pytest.param("count_not_tensor", "num_samples.*Tensor", id="count_not_tensor"),
            pytest.param("count_rank", "num_samples.*scalar", id="count_rank"),
            pytest.param("count_dtype", "num_samples.*torch.int64", id="count_dtype"),
            pytest.param("count_device", "num_samples.*same device", id="count_device"),
        ),
    )
    def test_fixed_capacity_seq_aux_moe_route_rejects_static_metadata_errors(
        self, invalid_case, error_match
    ):
        """The real MoELayer.route API validates static ownership metadata without Tensor reads."""
        container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        )
        hidden_states = torch.randn((8, 1, container.config.hidden_size), device="cuda")
        ownership = {
            "seq_aux_loss_sample_ids": torch.zeros(8, dtype=torch.long, device="cuda"),
            "seq_aux_loss_num_samples": torch.tensor(1, dtype=torch.long, device="cuda"),
            "seq_aux_loss_max_samples": 2,
        }
        if invalid_case == "max_samples_zero":
            ownership["seq_aux_loss_max_samples"] = 0
        elif invalid_case == "max_samples_bool":
            ownership["seq_aux_loss_max_samples"] = True
        elif invalid_case == "ids_not_tensor":
            ownership["seq_aux_loss_sample_ids"] = [0] * 8
        elif invalid_case == "ids_rank":
            ownership["seq_aux_loss_sample_ids"] = torch.zeros(
                (8, 1), dtype=torch.long, device="cuda"
            )
        elif invalid_case == "ids_length":
            ownership["seq_aux_loss_sample_ids"] = torch.zeros(
                7, dtype=torch.long, device="cuda"
            )
        elif invalid_case == "ids_dtype":
            ownership["seq_aux_loss_sample_ids"] = torch.zeros(
                8, dtype=torch.int32, device="cuda"
            )
        elif invalid_case == "ids_device":
            ownership["seq_aux_loss_sample_ids"] = torch.zeros(8, dtype=torch.long)
        elif invalid_case == "count_not_tensor":
            ownership["seq_aux_loss_num_samples"] = 1
        elif invalid_case == "count_rank":
            ownership["seq_aux_loss_num_samples"] = torch.tensor(
                [1], dtype=torch.long, device="cuda"
            )
        elif invalid_case == "count_dtype":
            ownership["seq_aux_loss_num_samples"] = torch.tensor(
                1, dtype=torch.int32, device="cuda"
            )
        elif invalid_case == "count_device":
            ownership["seq_aux_loss_num_samples"] = torch.tensor(1, dtype=torch.long)

        with pytest.raises(ValueError, match=error_match):
            container.moe_layer.route(hidden_states, **ownership)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_capacity_seq_aux_fusion_fails_closed(self, monkeypatch):
        """Variable packed seq_aux rejects router fusion before any unfused fallback."""
        import megatron.core.transformer.moe.router as router_module

        container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        )
        container.moe_layer.router.config.moe_router_fusion = True
        hidden_states = torch.randn((8, 1, container.config.hidden_size), device="cuda")

        def reject_unfused_fallback(*args, **kwargs):
            raise AssertionError("variable packed seq_aux silently called the loss fallback")

        monkeypatch.setattr(
            router_module, "switch_load_balancing_loss_func", reject_unfused_fallback
        )
        with pytest.raises(ValueError, match="variable packed seq_aux_loss.*moe_router_fusion"):
            container.moe_layer.route(
                hidden_states,
                seq_aux_loss_sample_ids=torch.zeros(8, dtype=torch.long, device="cuda"),
                seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.long, device="cuda"),
                seq_aux_loss_max_samples=2,
            )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_capacity_seq_aux_recompute_closure_forwards_ownership(self, monkeypatch):
        """MoELayer recomputation routes with the same explicit dynamic ownership tensors."""
        container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        )
        moe_layer = container.new_moe_layer(
            recompute_granularity="selective", recompute_modules=["moe"]
        )
        sample_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device="cuda")
        num_samples = torch.tensor(2, dtype=torch.long, device="cuda")
        packed_seq_params = PackedSeqParams(
            seq_aux_loss_sample_ids=sample_ids,
            seq_aux_loss_num_samples=num_samples,
            seq_aux_loss_max_samples=4,
        )
        route_calls = []
        original_route = moe_layer.route

        def record_route(
            hidden_states,
            padding_mask=None,
            *,
            seq_aux_loss_sample_ids=None,
            seq_aux_loss_num_samples=None,
            seq_aux_loss_max_samples=None,
        ):
            route_calls.append(
                (
                    seq_aux_loss_sample_ids,
                    seq_aux_loss_num_samples,
                    seq_aux_loss_max_samples,
                )
            )
            return original_route(
                hidden_states,
                padding_mask,
                seq_aux_loss_sample_ids=seq_aux_loss_sample_ids,
                seq_aux_loss_num_samples=seq_aux_loss_num_samples,
                seq_aux_loss_max_samples=seq_aux_loss_max_samples,
            )

        monkeypatch.setattr(moe_layer, "route", record_route)
        hidden_states = torch.randn(
            (8, 1, container.config.hidden_size), device="cuda", requires_grad=True
        )
        padding_mask = torch.zeros((1, 8), dtype=torch.bool, device="cuda")
        clear_aux_losses_tracker()

        output, _ = moe_layer(
            hidden_states,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )
        output.backward(torch.zeros_like(output))

        assert len(route_calls) == 2
        for routed_ids, routed_count, routed_capacity in route_calls:
            assert routed_ids is sample_ids
            assert routed_count is num_samples
            assert routed_capacity == 4
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        assert moe_layer.router.weight.grad is not None
        assert torch.isfinite(moe_layer.router.weight.grad).all()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_capacity_seq_aux_inference_router_fallback_forwards_ownership(self):
        """InferenceTopKRouter preserves ownership when it falls back to the training router."""
        config = dataclasses.replace(
            self.default_transformer_config,
            num_moe_experts=4,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            params_dtype=torch.float32,
            bf16=False,
        )
        router = InferenceTopKRouter(
            config=config, pg_collection=get_default_pg_collection()
        ).cuda()
        router.set_layer_number(0)
        hidden_states = torch.randn((8, 1, config.hidden_size), device="cuda")
        clear_aux_losses_tracker()

        probs, _ = router(
            hidden_states,
            seq_aux_loss_sample_ids=torch.zeros(8, dtype=torch.long, device="cuda"),
            seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.long, device="cuda"),
            seq_aux_loss_max_samples=3,
        )
        probs.backward(torch.zeros_like(probs))

        aux_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
        assert torch.isfinite(aux_loss)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_capacity_seq_aux_deprecated_combined_route_fails_explicitly(self):
        """The deprecated combined route never silently drops variable packed ownership."""
        container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
        )
        hidden_states = torch.randn((8, 1, container.config.hidden_size), device="cuda")

        with pytest.raises(ValueError, match="router_and_preprocess.*variable packed"):
            container.moe_layer.router_and_preprocess(
                hidden_states,
                seq_aux_loss_sample_ids=torch.zeros(8, dtype=torch.long, device="cuda"),
                seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.long, device="cuda"),
                seq_aux_loss_max_samples=2,
            )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fixed_capacity_seq_aux_all_padding_has_exact_zero_loss_and_gradients(
        self, monkeypatch
    ):
        """A scheduler-only all-padding pack remains finite with exactly zero aux gradients."""
        router = self.new_router(
            num_moe_experts=4,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            params_dtype=torch.float32,
            bf16=False,
            calculate_per_token_loss=True,
        ).cuda()
        hidden_states = torch.randn(
            (8, 1, router.config.hidden_size), dtype=torch.float32, device="cuda", requires_grad=True
        )
        padding_mask = torch.ones((8, 1), dtype=torch.bool, device="cuda")
        clear_aux_losses_tracker()
        valid_token_counts = []
        original_attach = router.attach_and_log_load_balancing_loss

        def capture_valid_token_count(*args, **kwargs):
            if args[3] == "seq_load_balancing_loss":
                valid_token_counts.append(kwargs["valid_token_count"])
            return original_attach(*args, **kwargs)

        monkeypatch.setattr(
            router, "attach_and_log_load_balancing_loss", capture_valid_token_count
        )

        probs, _ = router(
            hidden_states,
            padding_mask=padding_mask,
            seq_aux_loss_sample_ids=torch.zeros(8, dtype=torch.long, device="cuda"),
            seq_aux_loss_num_samples=torch.tensor(1, dtype=torch.long, device="cuda"),
            seq_aux_loss_max_samples=3,
        )
        probs.backward(torch.zeros_like(probs))
        aux_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]

        assert torch.equal(aux_loss, torch.zeros_like(aux_loss))
        assert torch.isfinite(aux_loss)
        assert len(valid_token_counts) == 1
        assert torch.equal(valid_token_counts[0], torch.zeros_like(valid_token_counts[0]))
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        assert torch.count_nonzero(hidden_states.grad) == 0
        assert router.weight.grad is not None
        assert torch.isfinite(router.weight.grad).all()
        assert torch.count_nonzero(router.weight.grad) == 0


class TestFixedCapacitySeqAuxLossValidation:
    @pytest.mark.parametrize("sample_ids", ((-1, 0), (0, 2)), ids=("negative", "at_num_samples"))
    def test_fixed_capacity_seq_aux_ids_use_async_device_validation(self, sample_ids):
        """Dynamic ownership bounds fail through _assert_async without a host scalar read."""
        router = _new_cpu_fixed_capacity_router()
        logits = torch.randn((2, 1, router.config.num_moe_experts), requires_grad=True)

        with _RejectTensorToHostScalarMode(), pytest.raises(
            RuntimeError, match="seq_aux_loss_sample_ids"
        ):
            router.routing(
                logits,
                seq_aux_loss_sample_ids=torch.tensor(sample_ids, dtype=torch.long),
                seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.long),
                seq_aux_loss_max_samples=2,
            )

    @pytest.mark.parametrize("num_samples", (0, 3), ids=("zero", "above_capacity"))
    def test_fixed_capacity_seq_aux_num_samples_uses_async_device_validation(
        self, num_samples
    ):
        """Dynamic sample-count bounds fail through _assert_async without a host scalar read."""
        router = _new_cpu_fixed_capacity_router()
        logits = torch.randn((2, 1, router.config.num_moe_experts), requires_grad=True)

        with _RejectTensorToHostScalarMode(), pytest.raises(
            RuntimeError, match="seq_aux_loss_num_samples"
        ):
            router.routing(
                logits,
                seq_aux_loss_sample_ids=torch.zeros(2, dtype=torch.long),
                seq_aux_loss_num_samples=torch.tensor(num_samples, dtype=torch.long),
                seq_aux_loss_max_samples=2,
            )

    @pytest.mark.parametrize("num_samples", (0, 3), ids=("zero", "above_capacity"))
    def test_fixed_capacity_seq_aux_invalid_count_safe_denominator(
        self, num_samples, monkeypatch
    ):
        """Invalid dynamic counts cannot create NaN/Inf if asynchronous assertion is deferred."""
        router = _new_cpu_fixed_capacity_router()
        logits = torch.randn((2, 1, router.config.num_moe_experts), requires_grad=True)
        assertions = []
        monkeypatch.setattr(
            MoEAuxLossAutoScaler,
            "main_loss_backward_scale",
            torch.tensor(1.0, device=logits.device),
        )

        def record_assertion(condition, message):
            assertions.append((condition.detach().clone(), message))

        monkeypatch.setattr(torch, "_assert_async", record_assertion)
        clear_aux_losses_tracker()
        with _RejectTensorToHostScalarMode():
            probs, _ = router.routing(
                logits,
                seq_aux_loss_sample_ids=torch.zeros(2, dtype=torch.long),
                seq_aux_loss_num_samples=torch.tensor(num_samples, dtype=torch.long),
                seq_aux_loss_max_samples=2,
            )
            probs.backward(torch.zeros_like(probs))
        aux_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]

        assert assertions
        assert torch.equal(assertions[0][0], torch.tensor(False))
        assert "seq_aux_loss_num_samples" in assertions[0][1]
        assert torch.equal(aux_loss, torch.zeros_like(aux_loss))
        assert torch.isfinite(aux_loss)
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad) == 0

    def test_fixed_capacity_seq_aux_invalid_ids_are_clamped_before_scatter(
        self, monkeypatch
    ):
        """Deferred ownership assertion cannot expose scatter_add_ to out-of-range IDs."""
        router = _new_cpu_fixed_capacity_router()
        logits = torch.randn((2, 1, router.config.num_moe_experts), requires_grad=True)
        assertions = []

        def record_assertion(condition, message):
            assertions.append((condition.detach().clone(), message))

        monkeypatch.setattr(torch, "_assert_async", record_assertion)
        clear_aux_losses_tracker()
        with _RejectTensorToHostScalarMode():
            probs, _ = router.routing(
                logits,
                seq_aux_loss_sample_ids=torch.tensor((-5, 100), dtype=torch.long),
                seq_aux_loss_num_samples=torch.tensor(2, dtype=torch.long),
                seq_aux_loss_max_samples=2,
            )

        assert len(assertions) == 2
        assert torch.equal(assertions[0][0], torch.tensor(True))
        assert torch.equal(assertions[1][0], torch.tensor(False))
        assert "seq_aux_loss_sample_ids" in assertions[1][1]
        assert torch.isfinite(probs).all()


class TestPaddingMaskAuxLoss:
    """Test padding mask support in various aux loss types."""

    def setup_model_parallel(self, tp_size=1, ep_size=1, cp_size=1, sequence_parallel=False):
        """Initialize model parallel with given configuration.

        Args:
            tp_size: Tensor parallel size.
            ep_size: Expert parallel size.
            cp_size: Context parallel size.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        # Store parallel configuration
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.cp_size = cp_size

        # Default configuration
        self.default_transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=8,
            num_moe_experts=32,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=8,
            moe_aux_loss_coeff=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            sequence_parallel=sequence_parallel and tp_size > 1,
        )

    def new_router(self, **kwargs):
        """Create a new router with updated configuration."""
        pg_collection = get_default_pg_collection()
        new_transformer_config = dataclasses.replace(self.default_transformer_config, **kwargs)
        router = TopKRouter(config=new_transformer_config, pg_collection=pg_collection)
        router.set_layer_number(0)
        return router

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("aux_loss_type", ["aux_loss", "seq_aux_loss", "global_aux_loss"])
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_padding_mask_removes_padding_tokens(self, aux_loss_type, tp_size, ep_size, cp_size):
        """Test that padding tokens are correctly excluded from aux loss calculation."""
        # Initialize model parallel with given configuration
        self.setup_model_parallel(tp_size=tp_size, ep_size=ep_size, cp_size=cp_size)

        try:
            clear_aux_losses_tracker()

            router = self.new_router(
                moe_router_load_balancing_type=aux_loss_type,
                moe_aux_loss_coeff=1.0,
                moe_router_dtype="fp64",
            ).cuda()

            seq_len = 32
            batch_size = 2
            hidden_size = router.config.hidden_size

            # Create input with padding
            hidden_states_full = torch.randn(
                (seq_len, batch_size, hidden_size), dtype=torch.bfloat16, device='cuda'
            )

            # Create padding mask: first half valid, second half padding
            padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
            padding_mask[seq_len // 2 :, :] = True

            # Test with padding mask
            router.weight.grad = None
            scores_with_mask, routing_map_with_mask = router(
                hidden_states_full, padding_mask=padding_mask
            )
            scores_with_mask.backward(torch.zeros_like(scores_with_mask))

            loss_name = {
                "aux_loss": "load_balancing_loss",
                "seq_aux_loss": "seq_load_balancing_loss",
                "global_aux_loss": "global_load_balancing_loss",
            }[aux_loss_type]

            tracker = get_moe_layer_wise_logging_tracker()
            aux_loss_with_mask = tracker[loss_name]["values"][0].clone()
            grad_with_mask = router.weight.grad.clone()

            # Test without padding (with only half of the tokens)
            clear_aux_losses_tracker()
            router.weight.grad = None
            hidden_states_valid = hidden_states_full[: seq_len // 2, :, :]
            scores_without_mask, routing_map_without_mask = router(hidden_states_valid)
            scores_without_mask.backward(torch.zeros_like(scores_without_mask))

            aux_loss_without_mask = tracker[loss_name]["values"][0].clone()
            grad_without_mask = router.weight.grad.clone()

            # The aux loss with mask should be equal to the aux loss without mask
            assert torch.equal(aux_loss_with_mask, aux_loss_without_mask)
            assert torch.equal(grad_with_mask, grad_without_mask)

            clear_aux_losses_tracker()
        finally:
            # Always cleanup model parallel
            Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_padding_mask_with_z_loss(self, tp_size, ep_size, cp_size):
        """Test that padding mask works correctly with z_loss."""
        # Initialize model parallel with given configuration
        self.setup_model_parallel(tp_size=tp_size, ep_size=ep_size, cp_size=cp_size)

        try:
            clear_aux_losses_tracker()

            router = self.new_router(
                moe_router_load_balancing_type="aux_loss",
                moe_aux_loss_coeff=0.0,
                moe_z_loss_coeff=1.0,
                moe_router_dtype="fp32",
            ).cuda()

            seq_len = 32
            batch_size = 2
            hidden_size = router.config.hidden_size

            # Create input
            hidden_states_full = torch.randn(
                (seq_len, batch_size, hidden_size), dtype=torch.bfloat16, device='cuda'
            )

            # Create padding mask: first half valid, second half padding
            padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
            padding_mask[seq_len // 2 :, :] = True

            # Test with padding mask
            router.weight.grad = None
            scores_with_mask, _ = router(hidden_states_full, padding_mask=padding_mask)
            scores_with_mask.sum().backward()

            tracker = get_moe_layer_wise_logging_tracker()
            z_loss_with_mask = tracker["z_loss"]["values"][0].clone()
            grad_with_mask = router.weight.grad.clone()

            # Test without padding (with only half of the tokens)
            clear_aux_losses_tracker()
            router.weight.grad = None
            hidden_states_valid = hidden_states_full[: seq_len // 2, :, :]
            scores_without_mask, _ = router(hidden_states_valid)
            scores_without_mask.sum().backward()

            z_loss_without_mask = tracker["z_loss"]["values"][0].clone()
            grad_without_mask = router.weight.grad.clone()

            # The z_loss with mask should be close to the z_loss without mask
            assert torch.equal(z_loss_with_mask, z_loss_without_mask)
            assert torch.equal(grad_with_mask, grad_without_mask)

            clear_aux_losses_tracker()
        finally:
            # Always cleanup model parallel
            Utils.destroy_model_parallel()
