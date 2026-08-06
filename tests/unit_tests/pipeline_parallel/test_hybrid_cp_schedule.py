import unittest
from collections import Counter

from megatron.core.pipeline_parallel.hybrid_cp_schedule import (
    BalancedCPScheduler,
    consume_hybrid_cp_iteration_stats,
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


class HybridCPScheduleTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
