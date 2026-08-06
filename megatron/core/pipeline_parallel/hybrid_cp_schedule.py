# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import os
from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable, List, Optional, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.rerun_state_machine import RerunDataIterator

HYBRID_CP_LOG_STAT_KEYS = (
    "hybrid_cp/unique_samples",
    "hybrid_cp/total_real_tokens",
    "hybrid_cp/waves",
    "hybrid_cp/cp_size_min",
    "hybrid_cp/cp_size_mean",
    "hybrid_cp/cp_size_max",
    "hybrid_cp/cp1_samples",
    "hybrid_cp/cp2_samples",
    "hybrid_cp/cp4_samples",
    "hybrid_cp/cp8_samples",
    "hybrid_cp/cp16_samples",
    "hybrid_cp/cp32_samples",
    "hybrid_cp/cp64_samples",
    "hybrid_cp/estimated_sample_work",
    "hybrid_cp/estimated_rank_work_min",
    "hybrid_cp/estimated_rank_work_mean",
    "hybrid_cp/estimated_rank_work_max",
    "hybrid_cp/estimated_rank_work_imbalance",
    "hybrid_cp/samples_with_vision",
    "hybrid_cp/samples_with_video",
    "hybrid_cp/samples_with_audio",
    "hybrid_cp/vision_tiles",
    "hybrid_cp/video_frames",
    "hybrid_cp/audio_clips",
)

_hybrid_cp_iteration_stats: Optional[dict[str, float]] = None


def _hybrid_cp_debug(message: str) -> None:
    """Emit rank-local HybridCP execution diagnostics when explicitly enabled."""

    if os.environ.get("MEGATRON_HYBRID_CP_DEBUG") != "1":
        return
    rank = "?"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())
    print(f"[HYBRID_CP_DEBUG][rank={rank}] {message}", flush=True)


def summarize_hybrid_cp_schedule(
    sample_id_seqlens: List[Tuple[int, int]],
    sample_id_groups: List[List[List[int]]],
) -> dict[str, float]:
    """Summarize unique DynamicCP work without counting CP participants as samples."""

    expected_lengths = dict(sample_id_seqlens)
    cp_sizes: dict[int, int] = {}
    rank_work = [0.0 for _ in sample_id_groups[0]] if sample_id_groups else []

    for wave_index, wave in enumerate(sample_id_groups):
        if len(wave) != len(rank_work):
            raise RuntimeError(
                f"DynamicCP wave {wave_index} has {len(wave)} ranks, expected {len(rank_work)}"
            )
        participants: dict[int, list[int]] = {}
        for rank, sample_ids in enumerate(wave):
            for sample_id in sample_ids:
                participants.setdefault(sample_id, []).append(rank)

        for sample_id, ranks in participants.items():
            if sample_id not in expected_lengths:
                raise RuntimeError(f"DynamicCP schedule contains unknown sample_id={sample_id}")
            if sample_id in cp_sizes:
                raise RuntimeError(f"DynamicCP sample_id={sample_id} appears in multiple waves")
            cp_size = len(ranks)
            cp_sizes[sample_id] = cp_size
            estimated_work = expected_lengths[sample_id] ** 2 / cp_size
            for rank in ranks:
                rank_work[rank] += estimated_work

    missing = sorted(set(expected_lengths) - set(cp_sizes))
    if missing:
        raise RuntimeError(f"DynamicCP schedule omitted sample_ids={missing}")

    cp_histogram = {
        cp_size: sum(value == cp_size for value in cp_sizes.values())
        for cp_size in (1, 2, 4, 8, 16, 32, 64)
    }
    local_cp_sizes = list(cp_sizes.values())
    rank_work_mean = sum(rank_work) / len(rank_work) if rank_work else 0.0
    stats = {
        "hybrid_cp/unique_samples": float(len(expected_lengths)),
        "hybrid_cp/total_real_tokens": float(sum(expected_lengths.values())),
        "hybrid_cp/waves": float(len(sample_id_groups)),
        "hybrid_cp/cp_size_min": float(min(local_cp_sizes, default=0)),
        "hybrid_cp/cp_size_mean": (
            float(sum(local_cp_sizes) / len(local_cp_sizes)) if local_cp_sizes else 0.0
        ),
        "hybrid_cp/cp_size_max": float(max(local_cp_sizes, default=0)),
        "hybrid_cp/estimated_sample_work": float(
            sum(expected_lengths[sample_id] ** 2 / cp_size for sample_id, cp_size in cp_sizes.items())
        ),
        "hybrid_cp/estimated_rank_work_min": float(min(rank_work, default=0.0)),
        "hybrid_cp/estimated_rank_work_mean": float(rank_work_mean),
        "hybrid_cp/estimated_rank_work_max": float(max(rank_work, default=0.0)),
        "hybrid_cp/estimated_rank_work_imbalance": (
            float(max(rank_work) / rank_work_mean) if rank_work_mean else 0.0
        ),
    }
    stats.update(
        {
            f"hybrid_cp/cp{cp_size}_samples": float(count)
            for cp_size, count in cp_histogram.items()
        }
    )
    return stats


def record_hybrid_cp_iteration_stats(stats: dict[str, float]) -> None:
    """Record rank-local DynamicCP stats for the current optimizer iteration."""

    global _hybrid_cp_iteration_stats
    _hybrid_cp_iteration_stats = dict(stats)


def consume_hybrid_cp_iteration_stats() -> Optional[dict[str, float]]:
    """Return and reset rank-local DynamicCP stats for the current iteration."""

    global _hybrid_cp_iteration_stats
    stats = _hybrid_cp_iteration_stats
    _hybrid_cp_iteration_stats = None
    return stats


class BalancedCPScheduler:
    """
    This class provides the functionality to form groups of sub-samples
    such that all DPxCP ranks have a roughly balanced workload in the group.
    """

    def __init__(self, max_seq_len_per_rank: int, dp_cp_group: torch.distributed.ProcessGroup):
        self.max_seq_len_per_rank = max_seq_len_per_rank
        self.num_subsamples = 0
        self.num_subsamples_processed = 0
        self.free_resources = []
        self.total_hdp_gpus = dp_cp_group.size()

    @lru_cache(maxsize=128)
    def get_total_workload(self, seq_length: int, cp_size: Optional[int] = None):
        """
        seq_length: sequence length of a sub-sample
        cp_size: total number of CP ranks working on this sub-sample

        Note:
        This function is used to estimate the relative workload intensity
        of a sub-sample. This is not meant to be an accurate flops calculator.

        Returns: workload of a sub-sample
        """
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        """
        Calculates the number of GPUs needed for a given sequence length
        and max sequence length per CP rank.
        This is used to determine the CP size of a sub-sample.

        The number is rounded up to the next power of 2 to match the available
        hybrid context parallel process group sizes.
        """
        return max(1, 2 ** ceil(log2((seq_len / self.max_seq_len_per_rank))))

    def make_buckets_equal(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
    ) -> List[deque]:
        """
        Makes as many buckets as unique CP sizes needed.
        This keeps sample IDs tethered to their sequence lengths throughout the bucketing process.
        """
        # Extract just the sequence lengths for determining k
        seqlens = [seq_len for _, seq_len in sample_seqlens]

        # Determine k based on unique GPU categories needed
        k = len({self.gpus_needed(L) for L in seqlens})

        # Create a work target for each bucket
        # This is the total work divided by the number of buckets
        work = []
        for _, s in sample_seqlens:
            cp_size = self.gpus_needed(s)
            work.append(compute_estimator(s, cp_size))
        total_work = sum(work)
        target = total_work / k
        buckets, cur, cur_work = [], [], 0.0
        remaining_work = total_work
        remaining_k = k

        for i, (sample_id, seq_len) in enumerate(sample_seqlens):
            work = compute_estimator(seq_len)
            projected = cur_work + work

            # Check if we should close this bucket
            if cur and (
                projected > target * 1.1  # Too much work
                or len(sample_seqlens) - i <= remaining_k - len(buckets)
            ):  # Need to save sequences for remaining buckets
                buckets.append(deque(cur))
                cur, cur_work = [], 0.0
                remaining_work -= sum(compute_estimator(seq_len) for _, seq_len in cur)
                remaining_k -= 1

            cur.append((sample_id, seq_len))
            cur_work += work

        if cur:
            buckets.append(deque(cur))

        return buckets

    def _available_group_ids(
        self,
        needed: int,
        group_members: dict[int, List[int]],
        group_size: dict[int, int],
        sample_ids_per_gpu: List[List[int]],
    ) -> List[int]:
        """Return compatible subgroups whose ranks have no work in this wave."""

        return [
            gid
            for gid, size in group_size.items()
            if size == needed
            and all(not sample_ids_per_gpu[rank] for rank in group_members[gid])
        ]

    def next_hdp_group(
        self,
        sample_seqlens: List[Tuple[int, int]],  # List of (sample_id, sequence_length) tuples
        compute_estimator: Callable[[int], float],
        total_gpus: int,
        delta: float = 0.05,  # balance slack (e.g. 5 %)
        strategy: str = "dp",  # "dp" or "pp"
        eps_bucket: float = 0.10,  # ε target for bucket balance
    ) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
        """
        Given a list of (sample_id, sequence_length) tuples, this function aims to assign
        sequences in a group such that all GPUs in the DPxCP group have a roughly balanced
        workload. Once each group is roughly balanced, we exit and return the
        group and the leftover sequences.

        The function performs the following passes in order to form a balanced microbatch:
        1. We create buckets of sequences that are roughly balanced.
        We try to create as many buckets as possible CP sizes.
        2. Given a bucket has sequences available, we assign the sample
            a. To a new set of GPUs if there are enough free GPUs.
            b. To an existing set of GPUs with the lowest load.
        3. We check if the group is balanced whenever we need to move onto a new CP size
        in the same set of GPUs.
        4. We trim the group if removing the last added sequence helps improve balance.
        5. If we run out of sequences to assign and there are empty GPUs,
        we redistribute work to empty GPUs by recursively increasing the CP size of a
        sample until no empty GPUs are left.

        Returns (micro_batches, leftover_sample_seqlens, exec_times, sample_ids_per_gpu).
        """
        if not sample_seqlens:
            return (
                [[] for _ in range(total_gpus)],
                [],
                [0.0 for _ in range(total_gpus)],
                [[] for _ in range(total_gpus)],
            )

        # Get buckets of sequences with balanced work
        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)

        # Initialize tracking structures
        micro_batches = [[] for _ in range(total_gpus)]
        exec_times = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu = [[] for _ in range(total_gpus)]

        gpu_group_id = [None] * total_gpus
        group_members = {}
        group_size = {}
        next_gid = 0

        pp_cursor = 0
        prev_needed = None
        check_balance = False

        while buckets:
            # ---- Step 1 – pick the next sequence we COULD place ------------------
            sample_seq_tuple = bucket_idx = None
            needed = None

            scan_order = (
                range(len(buckets))
                if strategy == "dp"
                else [(pp_cursor + i) % len(buckets) for i in range(len(buckets))]
            )

            for idx in scan_order:
                if not buckets[idx]:
                    continue
                cand_tuple = buckets[idx][0]  # This is now (sample_id, seq_len)
                cand_seq_len = cand_tuple[1]
                needed = self.gpus_needed(cand_seq_len)

                # (a) Do we have an *existing* group of size `needed`?
                candidate_gids = self._available_group_ids(
                    needed, group_members, group_size, sample_ids_per_gpu
                )

                # (b) Or enough completely free GPUs to start a new group?
                free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
                if candidate_gids or len(free_ranks) >= needed:
                    sample_seq_tuple, bucket_idx = cand_tuple, idx
                    break

            # No place to put any remaining sequence – finish this micro‑batch
            if sample_seq_tuple is None:
                break

            # TODO[pmannan]: PP not yet supported. Add PP scheduling.
            if strategy == "pp":
                pp_cursor = (bucket_idx + 1) % len(buckets)

            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            if prev_needed is None:
                prev_needed = needed

            # (a)  Existing groups of exactly this size
            candidate_gids = self._available_group_ids(
                needed, group_members, group_size, sample_ids_per_gpu
            )
            if candidate_gids:
                best_gid, best_load = min(
                    (
                        (gid, max(exec_times[r] for r in group_members[gid]))
                        for gid in candidate_gids
                    ),
                    key=lambda t: t[1],
                )
            else:
                best_gid, best_load = None, float("inf")

            # (b)  Hypothetical **new** group from completely free GPUs
            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if len(free_ranks) >= needed:
                free_sorted = sorted(free_ranks, key=lambda r: exec_times[r])
                new_members = free_sorted[:needed]
                new_load = exec_times[new_members[-1]]

                if new_load < best_load:
                    best_gid = None
                    chosen_members = new_members
                else:
                    chosen_members = group_members[best_gid]
            else:
                chosen_members = group_members[best_gid]

            # ---- Step 2 – if we decided to create a fresh group ----------------
            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid] = needed
                for r in chosen_members:
                    gpu_group_id[r] = best_gid

            # ---- Step 3 – assign the sequence to every member of that group ------
            per_gpu_cost = compute_estimator(seq_len)

            for r in chosen_members:
                micro_batches[r].append(seq_len)
                exec_times[r] += per_gpu_cost
                sample_ids_per_gpu[r].append(sample_id)

            # Remove the sequence definitively from its bucket
            buckets[bucket_idx].popleft()

            # ---- Step 4 – tidy, balance‑check, maybe early‑exit ------------------
            while buckets and not buckets[0]:
                buckets.pop(0)
                pp_cursor %= max(1, len(buckets))

            # TODO: Removing this helps reduce the number of groups when we have
            # lots of samples with same CP size.
            # But because we don't exit as soon as we get balanced,
            # even if there is one group available that can take the next sample,
            # we will keep adding samples to the same group.
            # trim_overload() does not help because it only checks if removing the
            # last added sample helps.
            # We cannot check after adding every sample because there will always be imbalance
            # if we don't wait for future scheduling.

            # IMPORTANT: So we need a solution here
            if needed < prev_needed:
                # When we get into a lower CP size in the same group,
                # we can start checking for balance. There is still a gotcha here.
                # Let's say we have a group of 3 GPU 0-2, then we move onto group of 2.
                # We keep assigning group of 2 as we do in descending order but GPU 7/15
                # never sees a microbatch assigned to it
                # until we run out of samples with CP2.
                # This means we are never balanced as min(exec_times) will always be 0.
                # We need a smart way of identifying that we have run out of big samples
                # and if we are having to assign work to a GPU already working,
                # is it because there are empty GPUs?
                # Would assigning work to empty GPUs first by moving onto next CP bucket help?
                # But we need to remember to come back to this CP size bucket and then
                # check for balance. Maybe the scheduling algorithm should look at empty
                # GPUs and find work rather than going sequence by sequence.
                check_balance = True

            if (
                check_balance
                and buckets
                and max(exec_times) - min(exec_times) <= delta * max(exec_times)
            ):
                break

        # Gather leftovers (flatten remaining buckets, preserve order)
        leftovers = []
        for b in buckets:
            for sample_seq_tuple in b:
                leftovers.append(sample_seq_tuple)

        # ---------------------------------------------------------------------------
        def trim_overload():
            """
            Iteratively pop the most‑recent sequence from the *most‑loaded group*
            whenever doing so reduces the global slack.
            """
            while True:
                cur_max = max(exec_times)
                cur_min = min(exec_times)
                cur_slack = cur_max - cur_min
                if cur_slack <= delta * cur_max:
                    # Slack is already within limit.
                    break
                if cur_min == 0:
                    # There are empty GPUs that will be
                    # handled in the next step.
                    break

                max_r = exec_times.index(cur_max)
                gid = gpu_group_id[max_r]
                members = group_members[gid]

                if not micro_batches[max_r] or len(micro_batches[max_r]) <= 1:
                    break

                seq = micro_batches[max_r][-1]
                need = group_size[gid]
                per_gpu_cost = compute_estimator(seq)

                proj_times = exec_times[:]
                for r in members:
                    proj_times[r] -= per_gpu_cost

                proj_slack = max(proj_times) - min(proj_times)

                # Check if trimming the workload helps imbalance
                if proj_slack < cur_slack:
                    sample_id_to_remove = sample_ids_per_gpu[max_r][-1]
                    for r in members:
                        micro_batches[r].pop()
                        exec_times[r] -= per_gpu_cost
                        sample_ids_per_gpu[r].pop()
                    leftovers.append((sample_id_to_remove, seq))
                else:
                    break

        trim_overload()

        # Track samples in this group before redistribution to empty GPUs
        total_work_before = sum(len(mb) for mb in micro_batches)

        # Check for empty GPUs and redistribute work
        def fill_empty_gpus(
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
        ):
            """
            Recursively check for empty GPUs and redistribute work by increasing
            the number of GPUs sharing samples. This ensures all GPUs have work.
            GPUs must be allocated consecutively so we may need to push existing
            work to other ranks in order to expand samples.
            """
            # Find empty GPUs
            empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
            if not empty_gpus:
                return (
                    micro_batches,
                    exec_times,
                    sample_ids_per_gpu,
                    group_members,
                    group_size,
                )  # No empty GPUs, we're done

            # Find the smallest group size that exists
            existing_group_sizes = set(group_size.values())
            assert (
                existing_group_sizes
            ), "There should be at least one group existing, cannot reditribute, "
            "try to increase 'max-seqlen-per-cp-rank'."

            min_group_size = min(existing_group_sizes)
            # We have Hybrid DPxCP groups for every power of 2 of GPUs or the entire DPxCP group.
            next_power = min(min_group_size * 2, total_gpus)

            # Find the first group of min_group_size that can be expanded
            expandable_gid = None
            expandable_members = None
            expandable_new_gpus = None

            for gid, size in group_size.items():
                if size == min_group_size:
                    members = group_members[gid]
                    needed_count = next_power - min_group_size
                    group_start_gpu = members[0]
                    group_end_gpu = members[-1]
                    empty_gpu = [idx for idx, work in enumerate(micro_batches) if not work][0]
                    assert not all(
                        work for work in micro_batches[empty_gpu : empty_gpu + needed_count]
                    ), f"Empty GPUs were detected but not enough to expand."
                    work_to_push = micro_batches[
                        group_end_gpu + 1 : empty_gpu
                    ]  # This is work of all other subsequent sub-samples
                    exec_times_to_push = exec_times[group_end_gpu + 1 : empty_gpu]
                    sample_ids_to_push = sample_ids_per_gpu[group_end_gpu + 1 : empty_gpu]

                    new_micro_batches = [[]] * len(micro_batches)
                    new_exec_times = [0.0] * len(exec_times)
                    new_sample_ids_per_gpu = [[]] * len(sample_ids_per_gpu)

                    # No change in work until the group selected for expansion
                    for i in range(group_start_gpu):
                        new_micro_batches[i] = micro_batches[i]
                        new_exec_times[i] = exec_times[i]
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[i]

                    # The work is distributed across the expanded group
                    for i in range(group_start_gpu, group_end_gpu + needed_count + 1):
                        new_micro_batches[i] = micro_batches[group_end_gpu]
                        new_exec_times[i] = self.get_total_workload(
                            micro_batches[group_end_gpu][0], next_power
                        )
                        new_sample_ids_per_gpu[i] = sample_ids_per_gpu[group_end_gpu]

                    # Any assigned work on expanded GPUs is pushed
                    for i, work in enumerate(work_to_push):
                        new_micro_batches[group_end_gpu + needed_count + 1 + i] = work
                        new_exec_times[group_end_gpu + needed_count + 1 + i] = exec_times_to_push[i]
                        new_sample_ids_per_gpu[group_end_gpu + needed_count + 1 + i] = (
                            sample_ids_to_push[i]
                        )

                    group_size[gid] = next_power
                    group_members[gid] = list(range(members[0], members[-1] + needed_count + 1))
                    for pushed_gid in group_size.keys():
                        if pushed_gid > gid:
                            group_members[pushed_gid] = [
                                x + needed_count for x in group_members[pushed_gid]
                            ]

                    return (
                        new_micro_batches,
                        new_exec_times,
                        new_sample_ids_per_gpu,
                        group_members,
                        group_size,
                    )

        empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])
        while empty_gpus:
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size = (
                fill_empty_gpus(
                    micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
                )
            )
            empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])

        # Assert that no sample has been completely removed
        total_work_after = sum(len(mb) for mb in micro_batches)
        assert (
            total_work_after >= total_work_before
        ), f"Samples were removed: {total_work_before} -> {total_work_after}"

        return micro_batches, leftovers, exec_times, sample_ids_per_gpu

    def validate_collective_safe_groups(
        self,
        sample_id_seqlens: List[Tuple[int, int]],
        sample_id_groups: List[List[List[int]]],
    ) -> None:
        """Validate that every DynamicCP wave keeps model collectives in lockstep."""

        expected_lengths = dict(sample_id_seqlens)
        seen = {}
        for wave_index, wave in enumerate(sample_id_groups):
            counts = [len(rank_ids) for rank_ids in wave]
            if len(wave) != self.total_hdp_gpus or any(count != 1 for count in counts):
                raise RuntimeError(
                    "DynamicCP collective-unsafe schedule: "
                    f"wave={wave_index} counts={counts} expected_one_invocation_per_rank"
                )

            participants = {}
            for rank, rank_ids in enumerate(wave):
                participants.setdefault(rank_ids[0], []).append(rank)

            for sample_id, ranks in participants.items():
                if sample_id not in expected_lengths:
                    raise RuntimeError(
                        f"DynamicCP schedule contains unknown sample_id={sample_id} "
                        f"wave={wave_index}"
                    )
                if sample_id in seen:
                    raise RuntimeError(
                        f"DynamicCP sample_id={sample_id} appears in waves "
                        f"{seen[sample_id]} and {wave_index}"
                    )

                cp_size = len(ranks)
                required = self.gpus_needed(expected_lengths[sample_id])
                contiguous = ranks == list(range(ranks[0], ranks[0] + cp_size))
                power_of_two = cp_size > 0 and cp_size & (cp_size - 1) == 0
                if not contiguous or not power_of_two or cp_size < required:
                    raise RuntimeError(
                        "DynamicCP invalid subgroup: "
                        f"wave={wave_index} sample_id={sample_id} ranks={ranks} "
                        f"cp_size={cp_size} required={required}"
                    )
                seen[sample_id] = wave_index

        missing = sorted(set(expected_lengths) - set(seen))
        if missing:
            raise RuntimeError(f"DynamicCP schedule omitted sample_ids={missing}")

    def get_groups_and_subsamples(self, sample_id_seqlens, config):
        """
        This function recursively forms groups of sub-samples such that all DPxCP ranks
        have a roughly balanced workload in the group.
        """
        groups = []
        sample_id_groups = []
        all_sample_id_seqlens = list(sample_id_seqlens)
        # We assign a sample_id to each sub-sample in order to track assignment to each GPU.
        sample_id_seqlens = sorted(sample_id_seqlens, key=lambda x: x[1], reverse=True)
        while sample_id_seqlens:
            mb, sample_id_seqlens, exec_times, sample_ids = self.next_hdp_group(
                sample_id_seqlens, self.get_total_workload, self.total_hdp_gpus
            )
            groups.append(mb)
            if len(sample_ids) < self.total_hdp_gpus:
                sample_ids.extend([] * (self.total_hdp_gpus - len(sample_ids)))
            sample_id_groups.append(sample_ids)

        self.validate_collective_safe_groups(all_sample_id_seqlens, sample_id_groups)
        return groups, sample_id_groups


def hybrid_context_parallel_forward_backward(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
    model_type,
):
    """
    Scheduler for Hybrid Context Parallel.

    This function performs the packed sample scheduling and determines
    1. The number of microbatches to schedule for each CP rank
    2. The number of groups each CP rank should execute
    3. The number of sub-samples per group each CP rank should execute

    A group is defined by a set of samples that can run across the CP domain without any barrier.
    There are many reasons why we may not be able to run endless samples within a single group.
    For example, if we have 8 GPUs,
    if GPU 0-5 are assigned a long sample that requires CP6,
    GPU 6-7 are assigned a short sample that requires CP2,
    The next sample which requires CP4 can be assigned GPU 4-7.
    But GPU 6-7 will finish first and get deadlocked if GPU 4-5 are not participating in the group.
    """
    from .schedules import backward_step, forward_step

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    def _broadcast_cp_group_size(cp_group_size):
        """Make the active sample CP size available on every TP rank."""

        cp_group_size_tensor = torch.tensor(
            [0 if cp_group_size is None else int(cp_group_size)],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        _broadcast(cp_group_size_tensor)
        cp_group_size = int(cp_group_size_tensor.item())
        if cp_group_size < 1:
            raise RuntimeError(f"HybridCP scheduled an invalid CP group size: {cp_group_size}")
        return cp_group_size

    def _broadcast_num_samples_this_group(num_samples_this_group):
        dev = torch.cuda.current_device()
        torch.distributed.barrier()

        n = 0 if num_samples_this_group is None else int(num_samples_this_group.numel())
        n = torch.tensor([n], dtype=torch.int64, device=dev)

        _broadcast(n)
        n = int(n.item())

        assert n > 0, "there should be at least 1 sub samples in the group"
        num_samples_this_group_broadcast = (
            torch.empty(n, dtype=torch.int32, device=dev)
            if num_samples_this_group is None
            else num_samples_this_group
        )
        _broadcast(num_samples_this_group_broadcast)
        return num_samples_this_group_broadcast

    def _broadcast_global_sample_count(global_sample_count):
        """Broadcast the unique sample count for this scheduled global batch."""

        dev = torch.cuda.current_device()
        count = torch.tensor(
            [0 if global_sample_count is None else int(global_sample_count)],
            dtype=torch.int32,
            device=dev,
        )
        _broadcast(count)
        if int(count.item()) < 1:
            raise RuntimeError("HybridCP scheduled global batch contains no samples")
        return int(count.item())

    def _broadcast_iteration_stats(iteration_stats):
        """Broadcast scheduler and modality diagnostics to every TP rank."""

        values = torch.tensor(
            [
                0.0 if iteration_stats is None else iteration_stats.get(key, 0.0)
                for key in HYBRID_CP_LOG_STAT_KEYS
            ],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )
        _broadcast(values)
        return {
            key: float(values[index].item())
            for index, key in enumerate(HYBRID_CP_LOG_STAT_KEYS)
        }

    def _get_new_data_iterator(sample_id_in_group, group_id):
        if is_first_tp_rank:
            sub_sample_id = sample_ids_this_group[sample_id_in_group]
            sample = batch[sub_sample_id]
            partner_cp_size = len(
                [True for sample_ids in sample_id_groups[group_id] if sub_sample_id in sample_ids]
            )
            if "cu_seqlens" in sample and "imgs" in sample:
                # Energon multimodal samples are routed in a tensor-only wire
                # format. Restore the batch-shaped dict expected by
                # examples/multimodal/train.py only after scheduling/routing.
                from megatron.core.datasets.data_schedule import restore_multimodal_hybrid_cp_sample

                sample = restore_multimodal_hybrid_cp_sample(sample, partner_cp_size)
            else:
                sample["local_cp_size"] = torch.tensor(partner_cp_size, dtype=torch.int32)
            new_data_iterator = RerunDataIterator(iter([sample]))
            return new_data_iterator
        else:
            return None

    # We get data once per global batch and schedule the sub-samples.
    # TODO(pmannan): Should we wrap the data_iterator here instead of the training.py file?
    hdp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    is_first_tp_rank = parallel_state.get_tensor_model_parallel_rank() == 0

    if is_first_tp_rank:
        data = next(data_iterator)
        sample_id_groups = data[1]
        batch = data[0]
        global_sample_count = data[2]
        iteration_stats = data[3] if len(data) > 3 else None
        _hybrid_cp_debug(
            f"schedule input hdp_rank={hdp_rank} global_sample_count={global_sample_count} "
            f"received_ids={sorted(int(gid) for gid in batch)} "
            f"groups={[[len(ids) for ids in group] for group in sample_id_groups]}"
        )
    else:
        data, sample_id_groups, batch, global_sample_count, iteration_stats = (
            None,
            None,
            None,
            None,
            None,
        )

    global_sample_count = _broadcast_global_sample_count(global_sample_count)
    record_hybrid_cp_iteration_stats(_broadcast_iteration_stats(iteration_stats))

    num_samples_this_group = None
    if is_first_tp_rank:
        num_samples_this_group = torch.tensor(
            [len(group[hdp_rank]) for group in sample_id_groups], dtype=torch.int32, device='cuda'
        )

    num_samples_this_group = _broadcast_num_samples_this_group(num_samples_this_group)
    num_samples_this_group = num_samples_this_group.cpu().numpy()
    num_total_groups = num_samples_this_group.shape[0]
    _hybrid_cp_debug(
        f"schedule group_counts={num_samples_this_group.tolist()} num_total_groups={num_total_groups}"
    )

    current_microbatch = 0

    # Upto last group, we don't need any sync.
    with no_sync_func():
        for j in range(num_total_groups - 1):
            sample_ids_this_group = sample_id_groups[j][hdp_rank] if is_first_tp_rank else None
            for i in range(num_samples_this_group[j]):
                # Call forward step for each sub-sample
                new_data_iterator = _get_new_data_iterator(i, j)
                sub_sample_id = sample_ids_this_group[i] if is_first_tp_rank else None
                partner_cp_size = (
                    len(
                        [
                            True
                            for sample_ids in sample_id_groups[j]
                            if sub_sample_id in sample_ids
                        ]
                    )
                    if is_first_tp_rank
                    else None
                )
                active_cp_group_size = _broadcast_cp_group_size(partner_cp_size)
                _hybrid_cp_debug(
                    f"before_forward group={j} index={i} gid={sub_sample_id} "
                    f"cp_size={active_cp_group_size} current_microbatch={current_microbatch}"
                )
                # TODO: Find the usage of current_microbatch and is_first_microbatch and
                # how that may affect my usage.
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    new_data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    cp_group_size=active_cp_group_size,
                    collect_non_loss_data=collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(
                        first_val_step, forward_only, current_microbatch == 0
                    ),
                    current_microbatch=current_microbatch,
                )
                _hybrid_cp_debug(
                    f"after_forward group={j} index={i} gid={sub_sample_id} "
                    f"num_tokens={int(num_tokens.item())} loss_entries={len(forward_data_store)}"
                )
                current_microbatch += 1
                total_num_tokens += num_tokens.item()
                if not forward_only:
                    _hybrid_cp_debug(
                        f"before_backward group={j} index={i} gid={sub_sample_id}"
                    )
                    backward_step(
                        input_tensor, output_tensor, output_tensor_grad, config
                    )
                    _hybrid_cp_debug(
                        f"after_backward group={j} index={i} gid={sub_sample_id}"
                    )

            # Create a barrier at end of each group.
            # This barrier ensures that all ranks are prepared to change assigned CP group sizes and
            # no rank is starting a sub-sample ahead of it's partner ranks.
            _hybrid_cp_debug(f"before_barrier group={j}")
            torch.distributed.barrier(
                parallel_state.get_data_parallel_group(with_context_parallel=True)
            )
            _hybrid_cp_debug(f"after_barrier group={j}")

    # For the last group, we need to run the last sub-sample out of the context handler.
    with no_sync_func():
        sample_ids_this_group = sample_id_groups[-1][hdp_rank] if is_first_tp_rank else None
        for i in range(num_samples_this_group[-1] - 1):
            new_data_iterator = _get_new_data_iterator(i, -1)
            sub_sample_id = sample_ids_this_group[i] if is_first_tp_rank else None
            partner_cp_size = (
                len(
                    [
                        True
                        for sample_ids in sample_id_groups[-1]
                        if sub_sample_id in sample_ids
                    ]
                )
                if is_first_tp_rank
                else None
            )
            active_cp_group_size = _broadcast_cp_group_size(partner_cp_size)
            _hybrid_cp_debug(
                f"before_forward group={num_total_groups - 1} index={i} gid={sub_sample_id} "
                f"cp_size={active_cp_group_size} current_microbatch={current_microbatch}"
            )
            # Call forward step for each sub-sample
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                new_data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size=active_cp_group_size,
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=check_first_val_step(
                    first_val_step, forward_only, current_microbatch == 0
                ),
                current_microbatch=current_microbatch,
            )
            _hybrid_cp_debug(
                f"after_forward group={num_total_groups - 1} index={i} gid={sub_sample_id} "
                f"num_tokens={int(num_tokens.item())} loss_entries={len(forward_data_store)}"
            )
            current_microbatch += 1
            total_num_tokens += num_tokens.item()
            if not forward_only:
                _hybrid_cp_debug(
                    f"before_backward group={num_total_groups - 1} index={i} gid={sub_sample_id}"
                )
                backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                _hybrid_cp_debug(
                    f"after_backward group={num_total_groups - 1} index={i} gid={sub_sample_id}"
                )

    # The last sub-sample of the last group of the last microbatch is
    # run out of the context handler.
    new_data_iterator = _get_new_data_iterator(-1, -1)
    sub_sample_id = sample_ids_this_group[-1] if is_first_tp_rank else None
    partner_cp_size = (
        len(
            [
                True
                for sample_ids in sample_id_groups[-1]
                if sub_sample_id in sample_ids
            ]
        )
        if is_first_tp_rank
        else None
    )
    active_cp_group_size = _broadcast_cp_group_size(partner_cp_size)
    _hybrid_cp_debug(
        f"before_forward group={num_total_groups - 1} index=last gid={sub_sample_id} "
        f"cp_size={active_cp_group_size} current_microbatch={current_microbatch}"
    )
    # Call forward step for each sub-sample
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        new_data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        cp_group_size=active_cp_group_size,
        collect_non_loss_data=collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, current_microbatch == 0
        ),
        current_microbatch=current_microbatch,
    )
    _hybrid_cp_debug(
        f"after_forward group={num_total_groups - 1} index=last gid={sub_sample_id} "
        f"num_tokens={int(num_tokens.item())} loss_entries={len(forward_data_store)}"
    )
    total_num_tokens += num_tokens.item()
    if not forward_only:
        _hybrid_cp_debug(
            f"before_backward group={num_total_groups - 1} index=last gid={sub_sample_id}"
        )
        backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        _hybrid_cp_debug(
            f"after_backward group={num_total_groups - 1} index=last gid={sub_sample_id}"
        )

    # Every rank may execute a different CP group's sample, so the local
    # ``_samples_seen`` list is not a reliable global-batch count.  Preserve
    # the scheduler's unique count for the training loop's global accounting.
    if forward_data_store and isinstance(forward_data_store[-1], dict):
        forward_data_store[-1]["_hybrid_cp_global_samples_seen"] = torch.tensor(
            global_sample_count, dtype=torch.float32, device=torch.cuda.current_device()
        )

    return forward_data_store, total_num_tokens
