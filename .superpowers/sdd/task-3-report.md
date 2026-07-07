# Task 3: Megatron-LM Live Common SFT Metrics

## Scope

- Worktree: `/Users/sna/Nemotron_Super_NemoRL_MLM_PerfGap/megatron-lm-worktrees/sft-live-wandb-6856424`
- Base: `6856424ca2a7eebc51a5e492b628792c52e4073e`
- Implemented only approved-plan Task 3.

## Changed Files

- `megatron/training/sft_comparison_metrics.py`: frozen typed observation and pure training/validation payload builders with finite-value validation.
- `megatron/training/arguments.py`: opt-in `--log-comparison-metrics` SFT flag, default false.
- `megatron/training/training.py`: custom W&B axis setup, native scalar capture, in-loop validation wall timing, coherent per-event payload logging, and final-validation exclusion.
- `tests/unit_tests/training/test_sft_comparison_metrics.py`: pure adapter, CLI, SFT-only gating, W&B axis/merge, and duplicate-final-evaluation regression tests.
- `.superpowers/sdd/task-3-report.md`: this report.

## Exact Logging Semantics

- Enabled only when `--sft` and `--log-comparison-metrics` are set, W&B is configured on the last rank, and RL mode is not active.
- `comparison/step`: one-based Megatron-LM training iteration.
- `performance/train_step_time_s`: existing `interval-time` elapsed iteration average in seconds. Validation is excluded by the native timer stop/start around evaluation.
- `performance/validation_time_s`: last-rank `time.perf_counter()` wall duration around the existing in-loop `evaluate(...)` call. Multiple validation sets use the sum of those call durations.
- `performance/e2e_step_time_s`: training time on non-validation events; training time plus validation wall duration on in-loop validation events.
- `accuracy/main_lm_loss`: existing native interval-averaged `lm loss` scalar already materialized for stdout logging.
- `accuracy/validation_loss`: existing native `lm loss` validation scalar for a single validation set. It is omitted for multiple validation sets because no unambiguous common scalar exists.
- `accuracy/grad_norm` and `accuracy/learning_rate`: existing native training scalars; unavailable values are omitted.
- `context/is_validation_step`: `0` for training-only events and `1` for in-loop validation events.
- One common payload is sent per native log event. W&B 0.28.0 receives `step=iteration, commit=False`, matching native Megatron-LM calls and merging the common keys into the current history row.
- `comparison/step` is defined as the custom axis for `performance/*`, `accuracy/*`, and `context/*`, exactly matching NeMo-RL.
- The post-training final validation and test calls explicitly set `collect_comparison_metrics=False`. Their native logs remain unchanged and can merge at step 200 without a stale-step conflict; no second common performance row is emitted.
- No collective, barrier, CUDA synchronization, or timer synchronization was added. The only new measurement is `perf_counter()` around the existing in-loop evaluation, and existing `.item()` scalar conversions are reused (validation conversions were reduced from repeated calls to one).

Comparison rows are available at native Megatron-LM log events. A validation event whose iteration is not a native log event has no native elapsed iteration scalar, so the common row is omitted rather than using a stale or synthesized training time.

## TDD Evidence

### RED

1. Initial adapter run: 10 failures because `megatron/training/sft_comparison_metrics.py` did not exist.

   `python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q`

2. CLI/axis/wiring run: 3 failures for the missing CLI destination, missing `define_metric` calls, and missing comparison log call.

3. Finite/commit/final-validation run: 3 failures for non-finite combined E2E acceptance, implicit W&B commit behavior, and implicit final-validation opt-out.

4. SFT-only gate run: 5 failures because `_sft_comparison_metrics_enabled` did not exist.

5. Step-200 merge regression run: 1 failure because the comparison call used a committed implicit step instead of explicit `step=iteration, commit=False`.

### GREEN

Final pure suite:

`python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q`

Result: `20 passed in 0.15s` in the final verification run.

The step-20 test proves a single payload contains train time `55.28`, validation time `58.645`, combined E2E time `113.925`, both losses, grad norm, learning rate, and validation marker `1`.

## Commands and Results

- Baseline/full training test: `python3 -m pytest tests/unit_tests/test_training.py -q`
  - Blocked during collection: `ModuleNotFoundError: No module named 'torch'` from `tests/unit_tests/__init__.py`.
- Required combined pytest command cannot collect for the same missing-`torch` reason. The pure suite above bypasses the torch-dependent parent package and conftest while executing the real adapter and parsing the real argument/training source.
- `pyright megatron/training/sft_comparison_metrics.py`
  - `0 errors, 0 warnings, 0 informations`.
- Full changed-file Pyright command
  - Environment/pre-existing blocker: 728 errors headed by unresolved `torch`, `torch.distributed`, `torch_memory_saver`, `modelopt`, `flashinfer`, and NVIDIA resiliency packages, plus existing typing cascades in the large legacy files.
- `ruff check megatron/training/sft_comparison_metrics.py megatron/training/training.py megatron/training/arguments.py tests/unit_tests/training/test_sft_comparison_metrics.py`
  - No issues found.
- `uv run --isolated --with isort isort --check-only ...`
  - Passed.
- `python3 -m compileall -q ...`
  - Passed.
- `git diff --check`
  - Passed.

## W&B API Investigation

- `get_wandb_writer()` returns the imported `wandb` module on the last rank, not a NeMo logger facade or a `Run` wrapper.
- This commit initializes it with `wandb.init(...)` and uses module-level `wandb.log(payload, iteration)` calls.
- `uv.lock` pins W&B `0.28.0`. Its `Run.log` contract defaults explicit-step calls to `commit=False`; no-step calls default to `commit=True` and advance the local W&B step.
- The common payload therefore uses the native explicit-step/deferred-commit pattern. This avoids advancing past iteration 200 before the preserved final validation logs at iteration 200.

## Remaining Concerns

- No torch/GPU/distributed integration test or real W&B smoke run was possible in this local environment.
- Final cluster validation should confirm the W&B history has one common row at each native log event and that step 200 retains both the in-loop common payload and native final-validation metrics.
