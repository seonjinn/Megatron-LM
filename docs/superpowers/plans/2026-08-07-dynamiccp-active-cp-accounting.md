# DynamicCP Active-CP Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct DynamicCP FLOP, packed-sequence, and trained-token metrics for mixed active CP groups without changing the validated per-token training computation.

**Architecture:** Additive `sum(L)` and `sum(L^2)` statistics are divided by their active replication factor at update time and de-duplicated only across TP and PP for HybridCP. Non-additive sequence distributions select one representative per active CP group. Packed-wave DynamicCP is rejected when per-token loss normalization is disabled.

**Tech Stack:** Python, PyTorch distributed, pytest, Megatron Core HybridCP, TE 2.14 HybridEP container.

## Global Constraints

- Work only in the linked worktree on `sna/omni-dynamiccp-packed-waves-20260806`.
- Preserve the existing single two-element FLOP-stat all-reduce and packed-stat object gather.
- Do not change scheduler placement, model outputs, optimizer state, or checkpoint format.
- Use `--calculate-per-token-loss` for packed-wave DynamicCP production runs.
- Commit with SSH signature and Signed-off-by trailer.

---

### Task 1: Add active-CP FLOP accounting regression tests

**Files:**
- Modify: `tests/unit_tests/test_num_floating_point_operations.py`

**Interfaces:**
- Consumes: `update_seqlen_stats_from_cu_seqlens(cu_seqlens, local_cp_size=None)` and `consume_seqlen_stats_in_iteration(is_hybrid_cp=False)`.
- Produces: tests that distinguish static CP replication from mixed active-CP replication.

- [ ] **Step 1: Write the failing tests**

Add a static compatibility test and a mixed active-CP test. The mixed test
simulates rank-local updates for CP4, CP8, and CP16 payloads and expects the
hand-derived unique totals after HybridCP consumption.

- [ ] **Step 2: Run tests to verify RED**

Run in the TE 2.14 test environment:

```bash
python -m pytest -q \
  tests/unit_tests/test_num_floating_point_operations.py \
  -k 'hybrid_cp or seqlen_stats'
```

Expected: the new test fails because the current function has no
`local_cp_size`/HybridCP-aware de-duplication contract.

- [ ] **Step 3: Commit the RED tests**

```bash
git add tests/unit_tests/test_num_floating_point_operations.py
git commit -S -s -m "test: expose DynamicCP metric undercount"
```

### Task 2: Implement exact additive statistics

**Files:**
- Modify: `megatron/training/training.py`
- Modify: `examples/multimodal/train.py`
- Test: `tests/unit_tests/test_num_floating_point_operations.py`

**Interfaces:**
- `update_seqlen_stats_from_cu_seqlens(cu_seqlens, local_cp_size: Optional[int] = None)` weights HybridCP contributions by `1 / local_cp_size`.
- `consume_seqlen_stats_in_iteration(is_hybrid_cp: bool = False)` excludes static CP from the HybridCP de-duplication factor.

- [ ] **Step 1: Implement the minimal production change**

Pass the routed `local_cp_size_value` from multimodal `get_batch`, validate it
is positive, weight both accumulator elements, and pass
`args.hybrid_context_parallel` to the consumer.

- [ ] **Step 2: Run focused tests to verify GREEN**

```bash
python -m pytest -q \
  tests/unit_tests/test_num_floating_point_operations.py \
  -k 'hybrid_cp or seqlen_stats'
```

Expected: static and mixed active-CP tests pass.

- [ ] **Step 3: Commit additive accounting**

```bash
git add megatron/training/training.py examples/multimodal/train.py \
  tests/unit_tests/test_num_floating_point_operations.py
git commit -S -s -m "fix: account DynamicCP FLOPs by active group"
```

### Task 3: Correct packed distribution and trained-token statistics

**Files:**
- Modify: `megatron/training/training.py`
- Modify: `examples/multimodal/train.py`
- Test: `tests/unit_tests/test_num_floating_point_operations.py`

**Interfaces:**
- `update_packed_sequence_stats(sample_lengths, loss_mask, local_cp_size=None, cp_group=None)` chooses one active-group representative.
- Existing `consume_packed_sequence_stats_in_iteration()` output keys and meanings remain unchanged.

- [ ] **Step 1: Write failing representative-selection tests**

Cover non-HybridCP static CP rank zero, DynamicCP CP1, active-group rank zero,
and active-group nonzero rank. Assert full sequence lengths and trained tokens
appear once per logical payload.

- [ ] **Step 2: Run tests to verify RED**

```bash
python -m pytest -q \
  tests/unit_tests/test_num_floating_point_operations.py \
  -k 'packed_sequence_stats'
```

Expected: DynamicCP groups outside static CP rank zero are dropped by the
current implementation.

- [ ] **Step 3: Implement active-group representative selection**

Pass `packed_seq_params.local_cp_size` and `packed_seq_params.cp_group` from
multimodal `get_batch`. Retain TP-zero/last-PP gating, use static CP rank zero
for non-HybridCP, accept CP1 directly, and otherwise use active-group rank zero.

- [ ] **Step 4: Run tests to verify GREEN**

```bash
python -m pytest -q \
  tests/unit_tests/test_num_floating_point_operations.py \
  -k 'packed_sequence_stats'
```

Expected: all packed-stat tests pass.

- [ ] **Step 5: Commit packed statistics**

```bash
git add megatron/training/training.py examples/multimodal/train.py \
  tests/unit_tests/test_num_floating_point_operations.py
git commit -S -s -m "fix: log each DynamicCP payload once"
```

### Task 4: Guard the unsupported legacy loss path

**Files:**
- Modify: `megatron/training/dynamic_context_parallel.py`
- Test: `tests/unit_tests/test_dynamic_context_parallel_compat.py`

**Interfaces:**
- HybridCP validation accepts `calculate_per_token_loss=True`.
- HybridCP validation raises a clear `ValueError` when `calculate_per_token_loss=False`.

- [ ] **Step 1: Write the failing validation test**

Create arguments with HybridCP enabled and per-token loss disabled. Expect a
message that packed-wave DynamicCP requires per-token loss normalization.

- [ ] **Step 2: Run test to verify RED**

```bash
python -m pytest -q tests/unit_tests/test_dynamic_context_parallel_compat.py
```

Expected: current validation accepts the unsupported configuration.

- [ ] **Step 3: Add the narrow fail-fast validation**

Add the check beside existing DynamicCP compatibility normalization. Do not
change non-HybridCP validation.

- [ ] **Step 4: Run test to verify GREEN**

```bash
python -m pytest -q tests/unit_tests/test_dynamic_context_parallel_compat.py
```

Expected: all compatibility tests pass.

- [ ] **Step 5: Commit loss-path safety**

```bash
git add megatron/training/dynamic_context_parallel.py \
  tests/unit_tests/test_dynamic_context_parallel_compat.py
git commit -S -s -m "fix: require per-token loss for DynamicCP"
```

### Task 5: Verify, review, and record results

**Files:**
- Modify only if review finds a defect: files from Tasks 2--4.
- Update after verification: experiment report in the pipeline report worktree.

**Interfaces:**
- Produces: tested Megatron commit and a report that distinguishes corrected metrics from pre-fix observations.

- [ ] **Step 1: Run focused regression suites in TE 2.14**

```bash
python -m pytest -q \
  tests/unit_tests/test_num_floating_point_operations.py \
  tests/unit_tests/pipeline_parallel/test_hybrid_cp_schedule.py \
  tests/unit_tests/data/test_hybrid_cp_multimodal.py \
  tests/unit_tests/data/test_hybrid_cp_metadata_smoke.py \
  tests/unit_tests/test_dynamic_context_parallel_compat.py \
  tests/unit_tests/test_pretrain_hybrid_thd_padding.py
```

Expected: zero failures.

- [ ] **Step 2: Run formatting and diff checks**

```bash
uv run isort examples/multimodal/train.py megatron/training/training.py \
  tests/unit_tests/test_num_floating_point_operations.py \
  tests/unit_tests/test_dynamic_context_parallel_compat.py
git diff --check
```

Expected: clean output.

- [ ] **Step 3: Ask Claude Fable to review the final diff**

Provide the base and head diff, the approved design, and focused test results.
Resolve every Critical and Important finding or document why it is not valid.

- [ ] **Step 4: Push the Megatron branch and update the report**

Push only to the personal fork branch. Record the corrected implementation
commit, test evidence, and the requirement for a deterministic performance
replay. Do not claim a speedup until the post-fix W&B TFLOP/s/GPU run exists.
