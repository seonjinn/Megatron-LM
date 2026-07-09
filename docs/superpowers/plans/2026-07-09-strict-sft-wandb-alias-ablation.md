# Strict SFT W&B Alias Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two zero-collective SFT W&B aliases and run a same-allocation 64-node A/B/C ablation against the `0823c73` strict baseline.

**Architecture:** A pure helper builds the optional alias payload from already-computed scalars. `training_log` calls it only when SFT, W&B, and the new flag are enabled. A cluster launcher runs original/no-W&B, patched/native-W&B, and patched/alias-W&B sequentially on the same nodes.

**Tech Stack:** Python 3.12, Megatron-LM argument parser and training loop, unittest, Bash, SLURM, W&B.

## Global Constraints

- Base commit is exactly `0823c731ed7d793aef047b6a64f2dbbf32bf6e2c`.
- No token counting, collective, CUDA synchronization, validation, checkpoint, model, or optimizer changes.
- Strict runtime uses 64 nodes / 512 H100, TP8, CP16, EP32, ETP1, PP1, DP4, GBS64, MBS1, sequence length 262144.
- Packed input SHA-256 is `3c034311b84b717f233b0c79831187fefbd9e635bea7721464c4f6d0c11cc4a0`.

---

### Task 1: Add the metric payload helper

**Files:**
- Create: `megatron/training/sft_metric_aliases.py`
- Test: `tests/standalone_tests/test_sft_metric_aliases.py`

**Interfaces:**
- Produces: `build_sft_metric_aliases(enabled: bool, e2e_step_time_s: float, main_lm_loss: T | None) -> dict[str, float | T]`

- [ ] Write a unittest proving disabled mode returns an empty payload and enabled mode preserves the loss object by identity.
- [ ] Run the test and verify it fails because the module does not exist.
- [ ] Implement the pure helper without importing PyTorch.
- [ ] Run the test and verify it passes.

### Task 2: Wire the aliases into strict SFT logging

**Files:**
- Modify: `megatron/training/arguments.py`
- Modify: `megatron/training/training.py`
- Test: `tests/standalone_tests/test_sft_metric_alias_integration.py`

**Interfaces:**
- Consumes: `build_sft_metric_aliases` from Task 1.
- Produces: `--log-comparison-metrics` and one W&B log call containing the two aliases.

- [ ] Write an AST/source test requiring the disabled-by-default argument and guarded W&B call.
- [ ] Run the test and verify it fails on the missing argument and integration.
- [ ] Add `--log-comparison-metrics` to the SFT argument group.
- [ ] In `training_log`, call the helper only when `args.sft`, the new flag, and `wandb_writer` are truthy, passing existing `elapsed_time_per_iteration` and `loss_dict.get("lm loss")`.
- [ ] Run standalone tests and formatting checks.
- [ ] Commit and push the isolated branch.

### Task 3: Build and submit the paired strict experiment

**Files:**
- Create in investigator repository: `launchers/submit_megatron_superv3_strict_wandb_alias_ablation.sh`

**Interfaces:**
- Consumes: original and patched M-LM worktrees plus the audited strict launcher.
- Produces: one SLURM job ID and per-variant manifests/logs/W&B runs.

- [ ] Add preflight checks for commits, clean trees, input SHA, image, topology, and W&B credentials.
- [ ] Run `sbatch --test-only` and inspect account/fair-share status.
- [ ] Commit and push the launcher before submission.
- [ ] Submit the 64-node allocation and monitor it for five minutes.
- [ ] Parse steps 6-20 and report baseline, native-W&B, and alias-W&B mean/median step times and loss equality.
