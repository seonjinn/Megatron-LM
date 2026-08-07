import unittest
from collections import Counter
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.pipeline_parallel.hybrid_cp_schedule import (
    BalancedCPScheduler,
    _validate_hybrid_cp_runtime_model_calls,
    consume_hybrid_cp_iteration_stats,
    hybrid_context_parallel_forward_backward,
    iter_hybrid_cp_rank_waves,
    record_hybrid_cp_iteration_stats,
    summarize_hybrid_cp_schedule,
)


class _Group:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _per_wave_counts(sample_id_groups: list[list[list[int]]]) -> list[list[int]]:
    return [[len(sample_ids) for sample_ids in wave] for wave in sample_id_groups]


def _participant_counts(sample_id_groups: list[list[list[int]]]) -> Counter[int]:
    return Counter(
        sample_id
        for wave in sample_id_groups
        for rank_ids in wave
        for sample_id in rank_ids
    )


def _wire_sample(token: int) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([token], dtype=torch.int64),
        "labels": torch.tensor([0, token], dtype=torch.int64),
        "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 1], dtype=torch.int32),
        "max_seqlen": torch.tensor(1, dtype=torch.int32),
        "sample_lengths": torch.tensor([1], dtype=torch.int32),
        "samples_seen": torch.tensor(1, dtype=torch.int32),
        "imgs": torch.tensor([[0.0]], dtype=torch.float32),
        "imgs_sizes": torch.tensor([[0, 0]], dtype=torch.int32),
        "vision_cu_lengths": torch.tensor([0], dtype=torch.int32),
        "vision_max_lengths": torch.tensor([0], dtype=torch.int32),
        "num_tiles": torch.tensor([0], dtype=torch.int32),
        "num_frames": torch.tensor([0], dtype=torch.int32),
        "has_pad_img": torch.tensor(False),
        "sound_clips": torch.tensor([[0.0]], dtype=torch.float32),
        "sound_length": torch.tensor([[0]], dtype=torch.int64),
        "sound_timestamps": torch.tensor([[0.0]], dtype=torch.float32),
        "num_sound_clips": torch.tensor([[0]], dtype=torch.int64),
    }


class HybridCPScheduleTest(unittest.TestCase):
    def test_runtime_model_calls_validator_is_disabled_by_default(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.tensor"
            ) as tensor,
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.all_reduce"
            ) as all_reduce,
        ):
            _validate_hybrid_cp_runtime_model_calls(2, 2)

        tensor.assert_not_called()
        all_reduce.assert_not_called()

    def test_runtime_model_calls_validator_logs_matching_world_counts(self) -> None:
        output = StringIO()
        with (
            patch.dict(
                "os.environ", {"MEGATRON_HYBRID_CP_VALIDATE_MODEL_CALLS": "1"}
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.cuda.current_device",
                return_value="cpu",
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.all_reduce"
            ) as all_reduce,
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.get_rank",
                return_value=0,
            ),
            redirect_stdout(output),
        ):
            _validate_hybrid_cp_runtime_model_calls(2, 2)

        self.assertEqual(all_reduce.call_count, 2)
        self.assertEqual(
            output.getvalue().strip(),
            "[HYBRID_CP_VALIDATOR] passed model_calls=2 world_min=2 world_max=2",
        )

    def test_runtime_model_calls_validator_rejects_world_mismatch(self) -> None:
        def fake_all_reduce(value, op):
            if op == torch.distributed.ReduceOp.MIN:
                value.fill_(1)
            elif op == torch.distributed.ReduceOp.MAX:
                value.fill_(2)

        with (
            patch.dict(
                "os.environ", {"MEGATRON_HYBRID_CP_VALIDATE_MODEL_CALLS": "1"}
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.cuda.current_device",
                return_value="cpu",
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.all_reduce",
                side_effect=fake_all_reduce,
            ) as all_reduce,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "world_min=1.*world_max=2.*expected=2"
            ):
                _validate_hybrid_cp_runtime_model_calls(2, 2)

        self.assertEqual(all_reduce.call_count, 2)

    def test_runtime_model_calls_validator_rejects_expected_mismatch(self) -> None:
        with (
            patch.dict(
                "os.environ", {"MEGATRON_HYBRID_CP_VALIDATE_MODEL_CALLS": "1"}
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.cuda.current_device",
                return_value="cpu",
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.all_reduce"
            ) as all_reduce,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "world_min=2.*world_max=2.*expected=3"
            ):
                _validate_hybrid_cp_runtime_model_calls(2, 3)

        self.assertEqual(all_reduce.call_count, 2)

    def test_minimum_cp_size_prevents_cp1_samples(self) -> None:
        scheduler = BalancedCPScheduler(65_536, _Group(4), min_cp_size=2)
        samples = [(0, 41_184), (1, 9_952)]

        _, waves = scheduler.get_groups_and_subsamples(samples, config=None)
        stats = summarize_hybrid_cp_schedule(samples, waves)

        self.assertEqual(scheduler.gpus_needed(9_952), 2)
        self.assertEqual(stats["hybrid_cp/cp1_samples"], 0)
        self.assertEqual(stats["hybrid_cp/cp2_samples"], 2)

    def test_minimum_cp_size_must_be_a_supported_power_of_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            BalancedCPScheduler(65_536, _Group(4), min_cp_size=3)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            BalancedCPScheduler(65_536, _Group(4), min_cp_size=8)

    def test_schedule_summary_counts_unique_work_instead_of_participants(self) -> None:
        scheduler = BalancedCPScheduler(65_536, _Group(4))
        samples = [(0, 71_264), (1, 41_184), (2, 9_952), (3, 8_000)]
        _, waves = scheduler.get_groups_and_subsamples(samples, config=None)

        stats = summarize_hybrid_cp_schedule(samples, waves)

        self.assertEqual(stats["hybrid_cp/unique_samples"], 4)
        self.assertEqual(stats["hybrid_cp/total_real_tokens"], 130_400)
        self.assertEqual(stats["hybrid_cp/waves"], 2)
        self.assertEqual(stats["hybrid_cp/cp1_samples"], 2)
        self.assertEqual(stats["hybrid_cp/cp2_samples"], 1)
        self.assertEqual(stats["hybrid_cp/cp4_samples"], 1)
        self.assertEqual(stats["hybrid_cp/cp_size_min"], 1)
        self.assertEqual(stats["hybrid_cp/cp_size_max"], 4)
        self.assertEqual(stats["hybrid_cp/cp_size_mean"], 2)

    def test_schedule_summary_rejects_duplicate_logical_sample(self) -> None:
        samples = [(0, 1_024)]
        duplicate_waves = [[[0], [0]], [[0], [0]]]

        with self.assertRaisesRegex(RuntimeError, "appears in multiple waves"):
            summarize_hybrid_cp_schedule(samples, duplicate_waves)

    def test_iteration_stats_are_consumed_once(self) -> None:
        consume_hybrid_cp_iteration_stats()
        expected = {"hybrid_cp/waves": 3.0}

        record_hybrid_cp_iteration_stats(expected)

        self.assertEqual(consume_hybrid_cp_iteration_stats(), expected)
        self.assertIsNone(consume_hybrid_cp_iteration_stats())

    def test_mixed_cp_sizes_emit_one_invocation_per_rank_per_wave(self) -> None:
        scheduler = BalancedCPScheduler(65_536, _Group(4))
        samples = [(0, 71_264), (1, 41_184), (2, 9_952), (3, 8_000)]

        _, sample_id_groups = scheduler.get_groups_and_subsamples(samples, config=None)

        self.assertEqual(len(sample_id_groups), 2)
        self.assertEqual(
            _per_wave_counts(sample_id_groups), [[1, 1, 1, 1], [1, 1, 1, 1]]
        )
        coverage = _participant_counts(sample_id_groups)
        self.assertEqual(set(coverage), {0, 1, 2, 3})
        self.assertEqual(sorted(coverage.values()), [1, 1, 2, 4])

    def test_uniform_waves_cover_every_logical_sample_once(self) -> None:
        scheduler = BalancedCPScheduler(65_536, _Group(4))
        samples = [(10, 71_264), (11, 41_184), (12, 9_952), (13, 8_000)]

        _, waves = scheduler.get_groups_and_subsamples(samples, config=None)
        expected_lengths = dict(samples)
        seen: set[int] = set()

        for wave in waves:
            participants: dict[int, list[int]] = {}
            for rank, rank_ids in enumerate(wave):
                self.assertEqual(len(rank_ids), 1)
                participants.setdefault(rank_ids[0], []).append(rank)
            for sample_id, ranks in participants.items():
                self.assertNotIn(sample_id, seen)
                seen.add(sample_id)
                self.assertGreaterEqual(
                    len(ranks), scheduler.gpus_needed(expected_lengths[sample_id])
                )
                self.assertEqual(len(ranks) & (len(ranks) - 1), 0)
                self.assertEqual(ranks, list(range(ranks[0], ranks[0] + len(ranks))))

        self.assertEqual(seen, set(expected_lengths))

    def test_validator_rejects_rank_with_two_invocations(self) -> None:
        scheduler = BalancedCPScheduler(65_536, _Group(4))
        malformed = [[[0], [0], [1], [2, 3]]]

        with self.assertRaisesRegex(
            RuntimeError, r"wave=0.*counts=\[1, 1, 1, 2\]"
        ):
            scheduler.validate_collective_safe_groups(
                [(0, 71_264), (1, 41_184), (2, 9_952), (3, 8_000)], malformed
            )

    def test_packed_mode_reuses_same_cp_subgroup_within_capacity(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(4), min_cp_size=2)
        samples = [(0, 6), (1, 6), (2, 6), (3, 6)]

        _, waves = scheduler.get_groups_and_subsamples(
            samples,
            config=None,
            padded_seqlens={0: 6, 1: 6, 2: 6, 3: 6},
            pack_payloads=True,
        )

        self.assertEqual(waves, [[[0, 2], [0, 2], [1, 3], [1, 3]]])

    def test_packed_mode_starts_new_wave_when_padded_capacity_is_full(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(2), min_cp_size=2)
        samples = [(0, 6), (1, 6)]

        _, waves = scheduler.get_groups_and_subsamples(
            samples,
            config=None,
            padded_seqlens={0: 12, 1: 12},
            pack_payloads=True,
        )

        self.assertEqual(waves, [[[0], [0]], [[1], [1]]])

    def test_default_mode_preserves_one_sample_per_rank_payload(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(4), min_cp_size=2)
        samples = [(0, 6), (1, 6), (2, 6), (3, 6)]

        _, waves = scheduler.get_groups_and_subsamples(samples, config=None)

        self.assertEqual(
            waves,
            [[[0], [0], [1], [1]], [[2], [2], [3], [3]]],
        )

    def test_packed_mode_does_not_mix_different_required_cp_sizes(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(4), min_cp_size=2)
        samples = [(0, 6), (1, 21)]

        _, waves = scheduler.get_groups_and_subsamples(
            samples,
            config=None,
            padded_seqlens={0: 6, 1: 21},
            pack_payloads=True,
        )

        self.assertEqual(waves, [[[1], [1], [1], [1]], [[0], [0], [0], [0]]])

    def test_packed_validator_rejects_payload_over_capacity(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(2), min_cp_size=2)

        with self.assertRaisesRegex(RuntimeError, "padded_tokens=24"):
            scheduler.validate_collective_safe_groups(
                [(0, 6), (1, 6)],
                [[[0, 1], [0, 1]]],
                padded_seqlens={0: 12, 1: 12},
                pack_payloads=True,
            )

    def test_packed_validator_rejects_payload_with_mixed_cp_requirements(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(4), min_cp_size=2)

        with self.assertRaisesRegex(RuntimeError, r"required=\[2, 4\]"):
            scheduler.validate_collective_safe_groups(
                [(0, 6), (1, 21)],
                [[[0, 1], [0, 1], [0, 1], [0, 1]]],
                padded_seqlens={0: 6, 1: 21},
                pack_payloads=True,
            )

    def test_packed_validator_reports_unknown_sample_id(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(2), min_cp_size=2)

        with self.assertRaisesRegex(RuntimeError, "unknown sample_id=9"):
            scheduler.validate_collective_safe_groups(
                [(0, 6)],
                [[[9], [9]]],
                padded_seqlens={0: 6},
                pack_payloads=True,
            )

    def test_packed_mode_rejects_missing_padded_length(self) -> None:
        scheduler = BalancedCPScheduler(10, _Group(2), min_cp_size=2)

        with self.assertRaisesRegex(ValueError, r"missing sample_ids=\[1\]"):
            scheduler.get_groups_and_subsamples(
                [(0, 6), (1, 6)],
                config=None,
                padded_seqlens={0: 6},
                pack_payloads=True,
            )

    def test_rank_wave_iterator_yields_one_payload_per_global_wave(self) -> None:
        waves = [
            [[0, 2], [0, 2], [1], [1]],
            [[3], [3], [4, 5], [4, 5]],
        ]

        rank_waves = list(iter_hybrid_cp_rank_waves(waves, hdp_rank=0))

        self.assertEqual(rank_waves, [([0, 2], 2), ([3], 2)])

    def test_schedule_summary_reports_packing_metrics(self) -> None:
        samples = [(0, 6), (1, 6), (2, 6), (3, 6)]
        waves = [[[0, 2], [0, 2], [1, 3], [1, 3]]]

        stats = summarize_hybrid_cp_schedule(
            samples, waves, padded_seqlens={0: 6, 1: 6, 2: 6, 3: 6}
        )

        self.assertEqual(stats["hybrid_cp/packed_calls"], 2.0)
        self.assertEqual(stats["hybrid_cp/logical_samples_per_call_mean"], 2.0)
        self.assertEqual(stats["hybrid_cp/logical_samples_per_call_max"], 2.0)
        self.assertEqual(stats["hybrid_cp/padded_language_tokens"], 24.0)
        self.assertEqual(stats["hybrid_cp/language_packing_efficiency"], 1.0)

    def test_forward_backward_executes_one_model_call_per_wave(self) -> None:
        batch = {sample_id: _wire_sample(sample_id) for sample_id in range(4)}
        waves = [
            [[0, 1], [0, 1]],
            [[2, 3], [2, 3]],
        ]
        data_iterator = iter([(batch, waves, 4, {})])
        forward_data_store: list[dict[str, torch.Tensor]] = []
        forwarded_tokens: list[list[list[int]]] = []
        backward_calls = 0
        barrier_calls = 0

        def fake_forward_step(*args, **kwargs):
            payload = next(args[1])
            forwarded_tokens.append(payload["tokens"].tolist())
            forward_data_store.append({})
            return torch.tensor(0.0), torch.tensor(payload["tokens"].numel())

        def fake_backward_step(*args, **kwargs):
            nonlocal backward_calls
            backward_calls += 1

        def fake_barrier(*args, **kwargs):
            nonlocal barrier_calls
            barrier_calls += 1

        with (
            patch(
                "megatron.core.pipeline_parallel.schedules.forward_step",
                side_effect=fake_forward_step,
            ),
            patch(
                "megatron.core.pipeline_parallel.schedules.backward_step",
                side_effect=fake_backward_step,
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.parallel_state.get_data_parallel_rank",
                return_value=0,
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.parallel_state.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.parallel_state.get_tensor_model_parallel_src_rank",
                return_value=0,
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.parallel_state.get_tensor_model_parallel_group",
                return_value=object(),
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.parallel_state.get_data_parallel_group",
                return_value=object(),
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.broadcast"
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.distributed.barrier",
                side_effect=fake_barrier,
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule.torch.cuda.current_device",
                return_value="cpu",
            ),
            patch(
                "megatron.core.pipeline_parallel.hybrid_cp_schedule._validate_hybrid_cp_runtime_model_calls"
            ) as validate_model_calls,
        ):
            _, total_num_tokens = hybrid_context_parallel_forward_backward(
                forward_step_func=None,
                data_iterator=data_iterator,
                model=None,
                num_microbatches=1,
                input_tensor=None,
                output_tensor_grad=None,
                forward_data_store=forward_data_store,
                config=SimpleNamespace(max_seqlen_per_dp_cp_rank=10),
                collect_non_loss_data=False,
                first_val_step=None,
                forward_only=False,
                no_sync_func=nullcontext,
                total_num_tokens=0,
                check_first_val_step=lambda *args: False,
                model_type=None,
            )

        self.assertEqual(forwarded_tokens, [[[0, 1]], [[2, 3]]])
        self.assertEqual(backward_calls, 2)
        self.assertEqual(barrier_calls, 1)
        self.assertEqual(total_num_tokens, 4)
        validate_model_calls.assert_called_once_with(2, 2)
        self.assertEqual(
            forward_data_store[-1]["_hybrid_cp_global_samples_seen"].item(), 4
        )


if __name__ == "__main__":
    unittest.main()
