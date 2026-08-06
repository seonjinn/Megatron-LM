# DynamicCP Global-MoE-Safe Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Nemotron 3.5 packed multimodal DynamicCP execute HybridEP, global MoE, and DDP collectives in lockstep, then prove the fix with matched Nano smoke and performance runs.

**Architecture:** `BalancedCPScheduler` will emit full execution waves in which every HDP rank participates in exactly one real sample. Any spare ranks will enlarge an assigned sample's power-of-two CP subgroup instead of creating dummy SFT work. A pure validator will reject malformed schedules before distributed model execution, while a distinct pipeline profile enables HybridEP uneven-input padding for mixed local CP sizes.

**Tech Stack:** Python 3.12, PyTorch distributed, Megatron Core, Transformer Engine 2.14, Bash, Pyxis/enroot, SLURM, pytest, W&B, static HTML experiment reporting.

## Global Constraints

- Work only in `/Users/sna/Nemotron_3.5_Super/megatron-lm/.worktrees/megatron-cg-hybridep-dynamiccp-latest` for Megatron changes.
- Work only in `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft` for launcher/profile changes.
- Preserve the meeting command's TP8, CP4, EP16, GBS32, 512K production reference and use the existing 131K, GBS1 profile only as a matched diagnostic surface.
- Mount the exact pushed Lustre Megatron checkout at `/opt/Megatron-LM`; never test the image-bundled source accidentally.
- Use `/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/containers/nano35-te214-hybridep-20260806.sqsh` with SHA256 `b4be967502d1212c10951a3bf2c631bbd98c28498ae80d336ea4f4aedf4d0901`.
- Count each real sample exactly once regardless of its expanded local CP size.
- Keep every HDP rank on the same forward/backward invocation number in every wave.
- Add no new direct `parallel_state.get_*_group()` reads in `megatron/core`; pass explicit process groups through existing APIs.
- Do not mix the scheduler correction with CUDA Graph capture changes; eager DynamicCP+HybridEP must pass first.
- Disable checkpoint writes in performance jobs, use warmup 3 and 20 total steps, and preserve W&B logging.
- Use `sna_super_3.5_sft` for Super 3.5 W&B runs; retain `nt-post-ci` for the matched Nano meeting runs.
- Run `render`, then SLURM `--test-only`, then submit; monitor every submitted job for at least five minutes.
- Commit with both SSH signature and Signed-off-by trailer, push before remote execution, and update the HTML result table after every completed or failed run.

---

### Task 1: Add the failing uniform-wave scheduler regression

**Files:**
- Create: `tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py`
- Reference: `megatron/core/pipeline_parallel/hybrid_cp_schedule.py`

**Interfaces:**
- Consumes: `BalancedCPScheduler(max_seq_len_per_rank: int, dp_cp_group: ProcessGroup)` and `get_groups_and_subsamples(sample_id_seqlens, config)`.
- Produces: an executable regression for the observed CP requirements `[2, 1, 1, 1]` and reusable schedule-invariant assertions.

- [ ] **Step 1: Write a dependency-light fake process group and schedule helpers**

```python
from collections import Counter

import pytest

from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler


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
```

- [ ] **Step 2: Write the observed lockstep regression**

```python
def test_mixed_cp_sizes_emit_one_invocation_per_rank_per_wave() -> None:
    scheduler = BalancedCPScheduler(65_536, _Group(4))
    samples = [(0, 71_264), (1, 41_184), (2, 9_952), (3, 8_000)]

    _, sample_id_groups = scheduler.get_groups_and_subsamples(samples, config=None)

    assert len(sample_id_groups) == 2
    assert _per_wave_counts(sample_id_groups) == [[1, 1, 1, 1], [1, 1, 1, 1]]
    coverage = _participant_counts(sample_id_groups)
    assert set(coverage) == {0, 1, 2, 3}
    assert sorted(coverage.values()) == [1, 1, 2, 4]
```

- [ ] **Step 3: Write coverage and minimum-CP assertions that do not depend on sample order**

```python
def test_uniform_waves_cover_every_logical_sample_once() -> None:
    scheduler = BalancedCPScheduler(65_536, _Group(4))
    samples = [(10, 71_264), (11, 41_184), (12, 9_952), (13, 8_000)]

    _, waves = scheduler.get_groups_and_subsamples(samples, config=None)
    expected_lengths = dict(samples)
    seen: set[int] = set()

    for wave in waves:
        participants: dict[int, list[int]] = {}
        for rank, rank_ids in enumerate(wave):
            assert len(rank_ids) == 1
            participants.setdefault(rank_ids[0], []).append(rank)
        for sample_id, ranks in participants.items():
            assert sample_id not in seen
            seen.add(sample_id)
            assert len(ranks) >= scheduler.gpus_needed(expected_lengths[sample_id])
            assert len(ranks) & (len(ranks) - 1) == 0
            assert ranks == list(range(ranks[0], ranks[0] + len(ranks)))

    assert seen == set(expected_lengths)
```

- [ ] **Step 4: Run the new test in the TE 2.14 container and verify RED**

Run inside the mounted container checkout:

```bash
cd /opt/Megatron-LM
python -m pytest -q tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py
```

Expected: `test_mixed_cp_sizes_emit_one_invocation_per_rank_per_wave` fails because the current scheduler emits a wave with counts `[1, 1, 1, 2]`.

- [ ] **Step 5: Commit and push the RED regression**

```bash
git add tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py
git -c gpg.format=ssh -c user.signingkey=/Users/sna/.ssh/id_ed25519_seonjinn commit -S -s -m "test: reproduce DynamicCP collective imbalance"
git push fork HEAD:sna/cg-hybridep-dynamiccp-latest-20260806
```

### Task 2: Emit full uniform waves and validate the schedule

**Files:**
- Modify: `megatron/core/pipeline_parallel/hybrid_cp_schedule.py`
- Modify: `tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py`

**Interfaces:**
- Consumes: `sample_id_seqlens: list[tuple[int, int]]`, per-sample minimum CP from `gpus_needed()`, and the existing empty-rank CP expansion.
- Produces: `_available_group_ids(...) -> list[int]` and `validate_collective_safe_groups(...) -> None`; `get_groups_and_subsamples()` returns validated, full, one-invocation-per-rank waves.

- [ ] **Step 1: Add a failing validator diagnostic test**

```python
def test_validator_rejects_rank_with_two_invocations() -> None:
    scheduler = BalancedCPScheduler(65_536, _Group(4))
    malformed = [[[0], [0], [1], [2, 3]]]

    with pytest.raises(RuntimeError, match=r"wave=0.*counts=\[1, 1, 1, 2\]"):
        scheduler.validate_collective_safe_groups(
            [(0, 71_264), (1, 41_184), (2, 9_952), (3, 8_000)], malformed
        )
```

- [ ] **Step 2: Run the validator test and verify RED**

Run:

```bash
cd /opt/Megatron-LM
python -m pytest -q tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py::test_validator_rejects_rank_with_two_invocations
```

Expected: FAIL with `AttributeError` because `validate_collective_safe_groups` does not exist.

- [ ] **Step 3: Restrict subgroup reuse to unassigned ranks**

Add the following method to `BalancedCPScheduler`:

```python
def _available_group_ids(
    self,
    needed: int,
    group_members: dict[int, list[int]],
    group_size: dict[int, int],
    sample_ids_per_gpu: list[list[int]],
) -> list[int]:
    return [
        gid
        for gid, size in group_size.items()
        if size == needed
        and all(not sample_ids_per_gpu[rank] for rank in group_members[gid])
    ]
```

Replace both existing `candidate_gids` comprehensions in `next_hdp_group()` with calls to `_available_group_ids`. This forces a wave to close when every compatible subgroup has already received one sample, leaving remaining samples for the next wave. Keep `fill_empty_gpus()` enabled so a partial final wave expands real samples to cover all ranks.

- [ ] **Step 4: Add the pure collective-safety validator**

```python
def validate_collective_safe_groups(
    self,
    sample_id_seqlens: list[tuple[int, int]],
    sample_id_groups: list[list[list[int]]],
) -> None:
    expected_lengths = dict(sample_id_seqlens)
    seen: dict[int, int] = {}
    for wave_index, wave in enumerate(sample_id_groups):
        counts = [len(rank_ids) for rank_ids in wave]
        if len(wave) != self.total_hdp_gpus or any(count != 1 for count in counts):
            raise RuntimeError(
                "DynamicCP collective-unsafe schedule: "
                f"wave={wave_index} counts={counts} expected_one_invocation_per_rank"
            )

        participants: dict[int, list[int]] = {}
        for rank, rank_ids in enumerate(wave):
            participants.setdefault(rank_ids[0], []).append(rank)

        for sample_id, ranks in participants.items():
            if sample_id not in expected_lengths:
                raise RuntimeError(
                    f"DynamicCP schedule contains unknown sample_id={sample_id} wave={wave_index}"
                )
            if sample_id in seen:
                raise RuntimeError(
                    f"DynamicCP sample_id={sample_id} appears in waves {seen[sample_id]} and {wave_index}"
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
```

Call the validator once at the end of `get_groups_and_subsamples()` before returning `groups, sample_id_groups`.

- [ ] **Step 5: Run the focused suite and verify GREEN**

Run:

```bash
cd /opt/Megatron-LM
python -m pytest -q \
  tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py \
  tests/unit_tests/data/test_hybrid_cp_metadata_smoke.py \
  tests/unit_tests/data/test_hybrid_cp_multimodal.py \
  tests/unit_tests/test_dynamic_cp_packed_seq_contract.py
```

Expected: all tests pass; the four-sample regression emits two waves with rank counts `[1, 1, 1, 1]` in each wave.

- [ ] **Step 6: Run formatting and static checks**

Run:

```bash
uv run isort megatron/core/pipeline_parallel/hybrid_cp_schedule.py tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py
git diff --check
python -m compileall -q megatron/core/pipeline_parallel/hybrid_cp_schedule.py tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py
```

Expected: zero exit status. If `uv` is unavailable locally, run `isort` in the TE 2.14 container and confirm it makes no diff.

- [ ] **Step 7: Commit and push the scheduler correction**

```bash
git add megatron/core/pipeline_parallel/hybrid_cp_schedule.py tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py
git -c gpg.format=ssh -c user.signingkey=/Users/sna/.ssh/id_ed25519_seonjinn commit -S -s -m "fix: keep DynamicCP MoE collectives in lockstep"
git push fork HEAD:sna/cg-hybridep-dynamiccp-latest-20260806
```

### Task 3: Add RED profile tests for the collective-safe HybridEP configuration

**Files:**
- Modify: `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft/tests/meeting/test_profiles.sh`
- Modify: `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft/tests/meeting/test_wrapper.sh`

**Interfaces:**
- Consumes: the existing `hybridep_dynamic_context_parallel_tiny_gbs1_131k_perf` profile and meeting wrapper.
- Produces: assertions for two distinct profiles, `hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_smoke` and `hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_perf`.

- [ ] **Step 1: Add profile assertions before creating the profiles**

Append tests that source/render both new names and assert:

```bash
test "$GBS" = 1
test "$PACKING_SEQ_LEN" = 131072
test "$MAX_SEQLEN_PER_DP_CP_RANK" = 65536
[[ "$EXTRA_ARGS" == *"--dynamic-context-parallel"* ]]
[[ "$EXTRA_ARGS" == *"--moe-flex-dispatcher-backend hybridep"* ]]
[[ "$EXTRA_ARGS" == *"--moe-hybridep-pad-uneven-dispatch-inputs"* ]]
[[ "$EXTRA_ARGS" != *"--cuda-graph"* ]]
```

The smoke profile must assert `TRAIN_ITERS=3`, `MEGATRON_HYBRID_CP_DEBUG=1`, and `SFT_CONTAINER_ENV` contains `MEGATRON_HYBRID_CP_DEBUG`. The performance profile must assert `TRAIN_ITERS=20`, `LR_WARMUP_ITERS=3`, and no debug environment variable.

- [ ] **Step 2: Run the pipeline tests and verify RED**

Run:

```bash
bash tests/meeting/test_profiles.sh
bash tests/meeting/test_wrapper.sh
```

Expected: both scripts fail because the wrapper rejects the new profile names or their files are absent.

- [ ] **Step 3: Commit and push the RED profile contract**

```bash
git add tests/meeting/test_profiles.sh tests/meeting/test_wrapper.sh
git -c gpg.format=ssh -c user.signingkey=/Users/sna/.ssh/id_ed25519_seonjinn commit -S -s -m "test: define collective-safe DynamicCP profiles"
git push origin HEAD:sna/nemotron-3p5-unified-sft
```

### Task 4: Implement the matched smoke and performance profiles

**Files:**
- Create: `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft/experiments/nemotron_3p5_sft/meeting/knobs/hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_smoke.env`
- Create: `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft/experiments/nemotron_3p5_sft/meeting/knobs/hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_perf.env`
- Modify: `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft/experiments/nemotron_3p5_sft/meeting/run_meeting_unified_sft.sh`
- Modify: `/Users/sna/Nemotron_3.5_Super/pipeline/.worktrees/pipeline-guyueh-unified-sft/experiments/nemotron_3p5_sft/meeting/source_contract.env`

**Interfaces:**
- Consumes: the pushed Megatron scheduler SHA, existing 131K GBS1 DynamicCP profile, exact `run_unified_sft.sh`, and TE 2.14 image.
- Produces: reusable three-step diagnostic and 20-step performance profiles with HybridEP uneven-input padding and exact source pinning.

- [ ] **Step 1: Create the common collective-safe performance overlay**

The performance file must source the existing performance profile and append exactly one new flag:

```bash
KNOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$KNOB_DIR/hybridep_dynamic_context_parallel_tiny_gbs1_131k_perf.env"
SFT_JOB_NAME=unified-sft-dynamiccp-hybridep-collective-safe-131k-perf
EXTRA_ARGS="${EXTRA_ARGS:-} --moe-hybridep-pad-uneven-dispatch-inputs"
unset MEGATRON_HYBRID_CP_DEBUG
SFT_CONTAINER_ENV="${SFT_CONTAINER_ENV//MEGATRON_HYBRID_CP_DEBUG/}"
SFT_CONTAINER_ENV="${SFT_CONTAINER_ENV#,}"
SFT_CONTAINER_ENV="${SFT_CONTAINER_ENV%,}"
unset KNOB_DIR
```

- [ ] **Step 2: Create the three-step diagnostic overlay**

The smoke file must source the new performance file, shorten the schedule, and enable diagnostics:

```bash
KNOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$KNOB_DIR/hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_perf.env"
TRAIN_ITERS=3
LR_WARMUP_ITERS=1
LR_DECAY_ITERS=2
EXIT_DURATION_MINS=25
SFT_JOB_NAME=unified-sft-dynamiccp-hybridep-collective-safe-131k-smoke
MEGATRON_HYBRID_CP_DEBUG=1
SFT_CONTAINER_ENV="${SFT_CONTAINER_ENV:+${SFT_CONTAINER_ENV},}MEGATRON_HYBRID_CP_DEBUG"
unset KNOB_DIR
```

- [ ] **Step 3: Register both profiles in wrapper usage, validation, and dispatch**

Add both exact names to the help text and the two case lists in `run_meeting_unified_sft.sh`. They use the generic `source "$MEETING_DIR/knobs/${PROFILE}.env"` branch; no bespoke launcher code is needed.

- [ ] **Step 4: Pin the final pushed Megatron SHA**

Set `MEGATRON_EXPECTED_SHA` in `source_contract.env` to the output of:

```bash
git -C /Users/sna/Nemotron_3.5_Super/megatron-lm/.worktrees/megatron-cg-hybridep-dynamiccp-latest rev-parse HEAD
```

Do not change `MEGATRON_BASE_SHA`, the TE 2.14 image hash, data path, tokenizer path, or reference W&B run ID.

- [ ] **Step 5: Run all meeting contract tests and verify GREEN**

Run:

```bash
bash tests/meeting/test_source_contract.sh
bash tests/meeting/test_profiles.sh
bash tests/meeting/test_wrapper.sh
```

Expected: all three scripts print their `passed` line and exit zero. Confirm rendered configs contain `/opt/Megatron-LM`, the exact Megatron SHA, `SAVE_CHECKPOINTS=0`, and no `WANDB_API_KEY`.

- [ ] **Step 6: Commit and push the profile implementation**

```bash
git add \
  experiments/nemotron_3p5_sft/meeting/knobs/hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_smoke.env \
  experiments/nemotron_3p5_sft/meeting/knobs/hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_perf.env \
  experiments/nemotron_3p5_sft/meeting/run_meeting_unified_sft.sh \
  experiments/nemotron_3p5_sft/meeting/source_contract.env
git -c gpg.format=ssh -c user.signingkey=/Users/sna/.ssh/id_ed25519_seonjinn commit -S -s -m "perf: add collective-safe DynamicCP profiles"
git push origin HEAD:sna/nemotron-3p5-unified-sft
```

### Task 5: Verify source parity and run the Nano eager gates

**Files:**
- Verify: remote Megatron checkout at `/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/megatron-lm-cg-hybridep-dynamiccp-latest`
- Verify: remote pipeline checkout at `/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/pipeline-unified-sft-latest`
- Produce: rendered run configs and SLURM logs under the configured `meeting-runs` root

**Interfaces:**
- Consumes: both pushed branches, TE 2.14 image, exact meeting data/checkpoint/tokenizer paths, and OCI-HSG exclusion list.
- Produces: one three-step correctness run and one 20-step eager performance run, each with W&B and exact provenance.

- [ ] **Step 1: Pull the two remote clean checkouts and verify exact SHA parity**

Run one SSH command that performs only lightweight Git operations:

```bash
ssh oci-hsg 'set -Eeuo pipefail
git -C /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/megatron-lm-cg-hybridep-dynamiccp-latest pull --ff-only
git -C /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/pipeline-unified-sft-latest pull --ff-only
git -C /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/megatron-lm-cg-hybridep-dynamiccp-latest status --porcelain
git -C /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/pipeline-unified-sft-latest status --porcelain
git -C /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/megatron-lm-cg-hybridep-dynamiccp-latest rev-parse HEAD
git -C /lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sna/Nemotron_3.5_Super/pipeline-unified-sft-latest rev-parse HEAD'
```

Expected: both status outputs are empty and both SHAs match the local pushed heads.

- [ ] **Step 2: Run the focused Megatron tests in a one-node TE 2.14 SLURM container job**

Inside the container, run the exact focused command from Task 2 Step 5 with the Lustre checkout mounted to `/opt/Megatron-LM`. Record the SLURM job ID, image path/hash, source SHA, pytest count, exit code, and log path.

- [ ] **Step 3: Render and perform a SLURM test-only submission for the smoke profile**

Run from the remote pipeline checkout:

```bash
./experiments/nemotron_3p5_sft/meeting/run_meeting_unified_sft.sh \
  --model-arch nano-3p5 \
  --profile hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_smoke \
  --action render \
  --run-id nano-dynamiccp-hybridep-collective-safe-smoke

./experiments/nemotron_3p5_sft/meeting/run_meeting_unified_sft.sh \
  --model-arch nano-3p5 \
  --profile hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_smoke \
  --action test-only \
  --run-id nano-dynamiccp-hybridep-collective-safe-smoke
```

Expected: the render records the exact Megatron/pipeline SHAs and the test-only request is accepted with 8 nodes, 4 GPUs per node, and the configured exclusion list.

- [ ] **Step 4: Submit the three-step smoke and monitor at least five minutes**

Submit the same run ID with `--action submit`. Poll `squeue`, then inspect only targeted log tails and error matches. The gate passes only if all three iterations finish, LM loss is finite, skipped/NaN counts are zero, and every debug wave reports uniform per-rank invocation counts.

- [ ] **Step 5: Render, test-only, submit, and monitor the 20-step performance profile**

Use run ID `nano-dynamiccp-hybridep-collective-safe-perf` and profile `hybridep_dynamic_context_parallel_tiny_gbs1_131k_collective_safe_perf`. The gate passes only if all 20 iterations complete, steps 4–20 have finite losses, and the job has no collective timeout or idle-GPU stall.

- [ ] **Step 6: Compare against the matched static CP4 control**

Compare the new W&B run with `ybny3du1` from job `5917477`. Record mean and median step time and TFLOP/s/GPU for steps 4–20, an outlier-filtered value only when the exclusion rule is stated, LM loss range, skipped/NaN counts, peak allocated memory, and samples/tokens per second. Do not compare the 131K diagnostic against the 512K production run as if they were matched.

### Task 6: Record results and define the CUDA Graph follow-on gate

**Files:**
- Modify: `/Users/sna/Nemotron_3.5_Super/.worktrees/pipeline-nemotron-3p5-sft/experiments/nemotron_3p5_sft/HANDOFF_2026-08-06.md`
- Modify: `/Users/sna/Nemotron_3.5_Super/.worktrees/pipeline-nemotron-3p5-sft/experiments/nemotron_3p5_sft/runs/*.json`
- Regenerate: `/Users/sna/Nemotron_3.5_Super/.worktrees/pipeline-nemotron-3p5-sft/public/data/runs.json`
- Regenerate: `/Users/sna/Nemotron_3.5_Super/.worktrees/pipeline-nemotron-3p5-sft/public/status.html`

**Interfaces:**
- Consumes: smoke/performance SLURM logs, W&B run metadata, exact source/image provenance, and static control metrics.
- Produces: an updated experiment table and a precise go/no-go decision for TE partial CUDA Graph work.

- [ ] **Step 1: Add run records for both new jobs**

Each JSON record must include job ID, state, exit code, node/GPU topology, W&B URL and project, profile, source SHAs, image hash, sequence/packing settings, TP/CP/EP degrees, HybridEP and DynamicCP flags, checkpoint-write state, step window, throughput, iteration time, TFLOP/s/GPU, loss, memory, and failure/fallback reason when applicable.

- [ ] **Step 2: Update the handoff root-cause section**

Record that MR !49 supplied runtime-CP and lockstep reference contracts, that the Megatron implementation chose real-sample full waves over dummy SFT batches, and whether the `[1,1,1,2]` reproduction became two uniform waves. Preserve the prior job evidence rather than overwriting it.

- [ ] **Step 3: Regenerate and test the report**

Run:

```bash
python3 -m pytest -q tests/nemotron_3p5_sft/test_render_report.py
python3 experiments/nemotron_3p5_sft/tools/render_report.py \
  --experiment-dir experiments/nemotron_3p5_sft \
  --output public/status.html
git diff --check
```

Expected: report tests pass and the performance table contains static CP4, failed pre-fix DynamicCP, collective-safe smoke, and collective-safe 20-step rows.

- [ ] **Step 4: Apply the CUDA Graph go/no-go gate**

If the eager 20-step run passes, write the next design around fixed THD shape buckets and start with `attn`, followed by `mamba`, `moe_router`, and then combined scopes. If eager does not pass, do not enable CUDA Graph; use the first divergent wave/collective evidence to revise the scheduler contract.

- [ ] **Step 5: Commit, push, and verify the served HTML**

```bash
git add experiments/nemotron_3p5_sft public tests/nemotron_3p5_sft/test_render_report.py
git -c gpg.format=ssh -c user.signingkey=/Users/sna/.ssh/id_ed25519_seonjinn commit -S -s -m "docs: record collective-safe DynamicCP results"
git push origin HEAD:sna/nemotron-3p5-sft-tuning
curl --fail --silent http://127.0.0.1:8787/status.html | rg 'collective-safe|DynamicCP|5917477'
```

Expected: the pushed report is clean and the local page server exposes the new status and result rows.
