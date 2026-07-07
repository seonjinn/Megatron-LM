# Task 3: Megatron-LM Live Common SFT Metrics

## Scope

- Worktree: `/Users/sna/Nemotron_Super_NemoRL_MLM_PerfGap/megatron-lm-worktrees/sft-live-wandb-6856424`
- Base: `6856424ca2a7eebc51a5e492b628792c52e4073e`
- Implemented only approved-plan Task 3.

## Changed Files

- `megatron/training/sft_comparison_metrics.py`: frozen typed observations, strict scalar normalization, configuration validation, pure payload builders, and event-scoped W&B emission.
- `megatron/training/arguments.py`: opt-in `--log-comparison-metrics` SFT flag, default false.
- `megatron/training/training.py`: custom W&B axis setup, native scalar capture, in-loop validation wall timing, coherent per-event payload logging, and final-validation exclusion.
- `tests/unit_tests/training/test_sft_comparison_metrics.py`: pure adapter, configuration, scalar contract, executable fake-writer event behavior, CLI, W&B axis/merge, and final/incomplete-evaluation regression tests.
- `.superpowers/sdd/task-3-report.md`: this report.

## Exact Logging Semantics

- Enabled only when `--sft` and `--log-comparison-metrics` are set, W&B is configured on the last rank, and RL mode is not active.
- `--log-comparison-metrics` requires `--log-interval 1`; any larger interval raises `ValueError` immediately after argument initialization and before model/training setup.
- `comparison/step`: one-based Megatron-LM training iteration.
- `performance/train_step_time_s`: existing `interval-time` elapsed iteration value in seconds for exactly one step. Validation is excluded by the native timer stop/start around evaluation.
- `performance/validation_time_s`: last-rank `time.perf_counter()` wall duration around the existing in-loop `evaluate(...)` call. Multiple validation sets use the sum of those call durations.
- `performance/e2e_step_time_s`: training time on non-validation events; training time plus validation wall duration on in-loop validation events.
- `accuracy/main_lm_loss`: existing native `lm loss` scalar already materialized for stdout logging; the required log interval makes this a one-step average.
- `accuracy/validation_loss`: existing native `lm loss` validation scalar for a single validation set. It is omitted for multiple validation sets because no unambiguous common scalar exists.
- `accuracy/grad_norm` and `accuracy/learning_rate`: existing native training scalars; unavailable values are omitted.
- `context/is_validation_step`: `0` for training-only events and `1` for in-loop validation events.
- One common payload is sent per completed training event. W&B 0.28.0 receives `step=iteration, commit=False`, matching native Megatron-LM calls and merging the common keys into the current history row.
- `comparison/step` is defined as the custom axis for `performance/*`, `accuracy/*`, and `context/*`, exactly matching NeMo-RL.
- An in-loop validation outcome explicitly records whether evaluation was attempted and completed. Attempted-but-incomplete validation, including the timelimit path, suppresses the entire common row instead of emitting a misleading training-only row.
- The post-training final validation and test calls explicitly set `collect_comparison_metrics=False`. Their native logs remain unchanged and can merge at step 200 without a stale-step conflict; no second common performance row is emitted.
- The adapter accepts only exact built-in Python `int`/`float` measurements and an exact built-in `int` step. It rejects booleans, `Decimal`, non-Python numeric scalar objects, and non-finite values; every emitted measurement is normalized to built-in `float` and step/context values remain built-in `int`.
- No collective, barrier, CUDA synchronization, or timer synchronization was added. The only new measurement is `perf_counter()` around the existing in-loop evaluation, and existing `.item()` scalar conversions are reused (validation conversions were reduced from repeated calls to one).

Comparison rows are exact per-step because enabling the feature requires native per-step logging.

## TDD Evidence

### RED

1. Initial adapter run: 10 failures because `megatron/training/sft_comparison_metrics.py` did not exist.

   `python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q`

2. CLI/axis/wiring run: 3 failures for the missing CLI destination, missing `define_metric` calls, and missing comparison log call.

3. Finite/commit/final-validation run: 3 failures for non-finite combined E2E acceptance, implicit W&B commit behavior, and implicit final-validation opt-out.

4. SFT-only gate run: 5 failures because `_sft_comparison_metrics_enabled` did not exist.

5. Step-200 merge regression run: 1 failure because the comparison call used a committed implicit step instead of explicit `step=iteration, commit=False`.

6. Review follow-up configuration run: 7 failures because exact per-step configuration validation did not exist; the pretrain wiring check also failed before the validator call was added.

7. Review follow-up scalar run: 16 failures because booleans, `Decimal`, and float steps were accepted and integer measurements were not normalized to float.

8. Review follow-up event run: 5 failures because the typed validation result/event logger did not exist and `train()` still logged directly. Supplemental wiring tests then failed until timelimit returned an explicit incomplete result and the loop delegated to the pure event helper.

### GREEN

Final pure suite:

`python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q`

Result before the follow-up review: `20 passed in 0.15s`.

Latest follow-up pure run:

`python3 -m pytest --rootdir=tests/unit_tests/training --confcutdir=tests/unit_tests/training --import-mode=prepend tests/unit_tests/training/test_sft_comparison_metrics.py -q`

Result: `49 passed in 0.18s` in the final follow-up verification run.

Executable fake-writer tests prove that step 20 produces one call containing train time `55.28`, validation time `58.645`, combined E2E time `113.925`, both losses, grad norm, learning rate, validation marker `1`, `step=20`, and `commit=False`. They also prove training-only emission, final-validation omission, and timelimit/incomplete-validation suppression. AST tests remain only as supplemental production-wiring checks.

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
- `uv run --isolated --with isort isort --check-only megatron/training/sft_comparison_metrics.py megatron/training/training.py tests/unit_tests/training/test_sft_comparison_metrics.py`
  - Passed for every Python file edited by the follow-up.
- Adding the unchanged `megatron/training/arguments.py` to the isort command reports broad pre-existing import-order drift throughout that file. The follow-up leaves it untouched; its Task 3 change is the two-line CLI option and `ruff check` passes it.
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
