# Strict SFT W&B Alias Ablation Design

## Goal

Measure whether adding only `performance/e2e_step_time_s` and
`accuracy/main_lm_loss` changes the 64-node Super V3 Megatron-LM strict
baseline measured at commit `0823c731ed7d793aef047b6a64f2dbbf32bf6e2c`.

## Scope

The implementation adds one disabled-by-default SFT flag and one W&B payload
containing aliases of values Megatron-LM already computes. It must not add token
counting, collectives, CUDA synchronization, validation behavior, checkpointing,
or model configuration changes.

## Experiment

One 64-node allocation runs three 20-step variants sequentially on the same
nodes:

1. Original `0823c73`, W&B disabled.
2. Metrics branch, native W&B enabled, common aliases disabled.
3. Metrics branch, native W&B enabled, common aliases enabled.

Every variant uses the audited strict topology, packed input SHA, checkpoint,
container, seed, and performance knobs. Steps 6-20 form the steady-state window.
The comparison separates baseline drift, native W&B overhead, and alias overhead.

## Correctness Gates

- The alias loss must be the exact native `loss_dict["lm loss"]` object.
- The alias step time must be the existing `elapsed_time_per_iteration` value.
- The alias payload is emitted only for SFT when the new flag and W&B are enabled.
- The clean baseline remains free of metric code changes by using the original
  `0823c73` worktree.
- All three variants must report the same resolved topology and input provenance.
