# Task 1 Execution Report

## Scope and Baseline

- Worktree: `/Users/sna/Nemotron_Super_NemoRL_MLM_PerfGap/megatron-lm-worktrees/sft-common-wandb-6856424`
- Required base: `6856424ca2a7eebc51a5e492b628792c52e4073e`
- Initial `git status --short`: no output.
- Initial `git rev-parse HEAD`: `6856424ca2a7eebc51a5e492b628792c52e4073e`.

## Applied Commit Series

Applied exactly, in the requested order:

1. `11a3bf6c5`
2. `820f5c48b`
3. `88d776e9b`
4. `f49648727`

Cherry-pick completed without conflicts. The resulting local commit range is:

`6856424ca2a7eebc51a5e492b628792c52e4073e..865aaf3639aa2c889bd255ec074311fcd055d256`

Local commits, in order:

1. `956b05fc7 feat: log common SFT comparison metrics`
2. `741448e40 fix: enforce exact SFT comparison metrics`
3. `dd0176d35 fix: capture exact SFT comparison events`
4. `865aaf363 fix: harden SFT comparison metric boundaries`

## Boundary Verification

Ran the required metric-key search. All required keys are present in
`megatron/training/sft_comparison_metrics.py`:

- `performance/train_step_time_s`
- `performance/validation_time_s`
- `performance/e2e_step_time_s`
- `accuracy/main_lm_loss`
- `accuracy/validation_loss`
- `accuracy/grad_norm`
- `accuracy/learning_rate`
- `context/is_validation_step`

`git merge-base --is-ancestor f49648727 HEAD` succeeded.

`git merge-base --is-ancestor ee058c399 HEAD && exit 1 || true` completed successfully, confirming the excluded throughput-accounting commit is not an ancestor.

## Focused Verification

Executed every command required by the task brief:

```bash
python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q
ruff check megatron/training/sft_comparison_metrics.py megatron/training/training.py megatron/training/utils/common_utils.py megatron/training/arguments.py tests/unit_tests/training/test_sft_comparison_metrics.py
python3 -m compileall -q megatron/training/sft_comparison_metrics.py megatron/training/training.py megatron/training/utils/common_utils.py megatron/training/arguments.py
git diff --check 6856424..HEAD
```

Results:

- Pytest: `61 passed in 0.39s`.
- Ruff: passed. Ruff emitted only its existing top-level-settings deprecation warning for `pyproject.toml`.
- Compileall: passed.
- `git diff --check`: passed with no whitespace errors.

## Diff Scope Inspection

The resulting implementation diff modifies only the expected feature files:

- `megatron/training/arguments.py`
- `megatron/training/sft_comparison_metrics.py`
- `megatron/training/training.py`
- `megatron/training/utils/common_utils.py`
- `tests/unit_tests/training/test_sft_comparison_metrics.py`

The transplanted history also contains `.superpowers/sdd/task-3-report.md`. This is an out-of-scope documentation artifact relative to the Task 1 file list, but it came from the required reviewed commit series and was not independently added or changed.

## Timer and W&B Semantics

Source inspection confirms that training duration is calculated from deltas of the native cumulative `interval-time.active_time()` value. The timer remains subject to its existing stop/start boundaries around in-loop evaluation, so validation time is excluded from `performance/train_step_time_s`. A separate `time.perf_counter()` interval covers the in-loop evaluation call for `performance/validation_time_s`; validation events set E2E time to the sum of train and validation durations. Dummy skipped iterations invalidate and rebaseline the comparison timer before the next emitted event.

The comparison state is instantiated only on the rank with a W&B writer. Common payloads use explicit `step=iteration` and `commit=False`, which matches the surrounding native W&B deferred-commit pattern and preserves the custom `comparison/step` axis. Completed in-loop validation emits one combined payload; incomplete validation and final validation emit none.

## Concerns

- No real torch/CUDA distributed run or live W&B history smoke test was available. The focused suite and source-level checks cover the intended timer and deferred-commit behavior, but a cluster run should confirm one merged W&B history row per completed event, especially around validation and the final-validation step.
- The required commit series includes the unrelated prior `.superpowers/sdd/task-3-report.md` artifact. It is retained to comply with the instruction to cherry-pick exactly the specified commits.

## Task-Review Fixes

### Findings Addressed

- Moved `validation_time_s` measurement from around `evaluate()` to the outer
  in-loop validation branch. The interval now starts before validation setup and
  ends after evaluation result processing, native logging/non-loss output, timer
  accounting, hook restoration, energy resume, and MoE tracker cleanup.
- Gated comparison configuration validation with
  `_sft_comparison_metrics_enabled(args)`, so non-SFT and RL runs are not
  constrained when comparison logging is effectively disabled.
- Restored the native CUDA tensor path for lm-loss accumulation, TensorBoard/W&B
  values, averaging, and reset. The separately materialized Python scalar is now
  used only by the comparison payload.

### Changed Files

- `megatron/training/training.py`
- `megatron/training/sft_comparison_metrics.py`
- `tests/unit_tests/training/test_sft_comparison_metrics.py`
- `.superpowers/sdd/task-1-report.md`

### RED Evidence

Added one focused regression test per review finding, then ran:

```bash
python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py::test_pretrain_validates_only_when_comparison_metrics_are_enabled tests/unit_tests/training/test_sft_comparison_metrics.py::test_validation_timer_brackets_the_complete_in_loop_event tests/unit_tests/training/test_sft_comparison_metrics.py::test_training_log_preserves_native_lm_loss_paths -q
```

Result: `3 failed in 0.10s`, for the expected reasons:

- Validation received `bool(getattr(args, 'log_comparison_metrics', False))`
  instead of `_sft_comparison_metrics_enabled(args)`.
- The in-loop validation branch contained zero `time.perf_counter()` calls.
- Native TensorBoard lm-loss logging used the normalized Python scalar instead
  of `loss_dict[key]`.

### GREEN Evidence

After the minimal production changes, reran the same command.

Result: `3 passed in 0.09s`.

Ran the complete focused file:

```bash
python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q
```

Result: `64 passed in 0.35s`.

### Verification

```bash
ruff check megatron/training/sft_comparison_metrics.py megatron/training/training.py megatron/training/utils/common_utils.py megatron/training/arguments.py tests/unit_tests/training/test_sft_comparison_metrics.py
python3 -m compileall -q megatron/training/sft_comparison_metrics.py megatron/training/training.py megatron/training/utils/common_utils.py megatron/training/arguments.py tests/unit_tests/training/test_sft_comparison_metrics.py
pyright megatron/training/training.py megatron/training/sft_comparison_metrics.py tests/unit_tests/training/test_sft_comparison_metrics.py
pyright megatron/training/sft_comparison_metrics.py
git diff --check
```

Results:

- Ruff: passed; only the existing top-level-settings deprecation warning was
  emitted.
- Compileall: passed.
- Broad Pyright: ran because Pyright is installed, but reported `724 errors, 7
  warnings`; the output is dominated by unresolved optional dependencies such
  as `torch` and the existing untyped `training.py` surface, so it is not a clean
  gate in this checkout.
- Standalone adapter Pyright: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: passed.

### Side-Effect Inspection

Inspected the zero-context production diff and compared source call counts
against pre-fix `HEAD`:

- No new collective, barrier, CUDA synchronization, model operation, or native
  timer stop was added.
- `wandb_writer.log(...)` calls in `training.py` remain `18`; the changed line
  restores the native loss value and does not add a call.
- `writer.log(...)` calls in `sft_comparison_metrics.py` remain `1`, with the
  existing `commit=False` behavior unchanged.
- `.item()` calls in `training.py` remain `13`; no extra host materialization was
  introduced.

### Review-Fix Concerns

- The broad Pyright command is nonzero in the current local environment as
  described above. The directly changed standalone adapter is Pyright-clean.
- No GPU/distributed or live W&B smoke test was available locally; focused tests
  and source inspection cover the requested behavioral boundaries.
