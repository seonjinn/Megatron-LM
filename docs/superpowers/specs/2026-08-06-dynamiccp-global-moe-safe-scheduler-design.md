# DynamicCP Global-MoE-Safe Scheduler Design

## Context

Nemotron 3.5 packed multimodal SFT can run with static CP4 and HybridEP, but the
current DynamicCP scheduler can deadlock before reaching steady state. The
instrumented reproduction, job `5917373`, scheduled four real samples with
per-HDP-rank model invocation counts `[1, 1, 1, 2]`. All 32 TP-expanded ranks
entered the first language-model forward, but none returned from it. The matched
static CP4 control, job `5917477`, completed 20 finite steps on the same source,
image, data, topology, and 131K sequence-length profile.

This evidence isolates two collective contracts that the current DynamicCP path
violates:

1. HybridEP receives different physical token-row counts on ranks in the same
   EP group when a wave mixes local CP sizes.
2. A rank can execute more model invocations than its global-MoE and DDP peers,
   so collective call order diverges.

## Reference: NeMo-RL MR !49

The implementation merged by
`jseppanen/nemo-rl!49` at merge commit
`9b8be8164a45931f143d0c27ca6b9058bf1084f5` is a useful behavioral reference,
not a source-level cherry-pick candidate. Its source commit
`777eaebba1c8e03b6a278cbb6be8fa8c88eaf32c` establishes the following contracts:

- Carry the runtime `cp_group` and `local_cp_size` with each packed microbatch.
- Use the runtime group for THD attention, RoPE, multimodal split/gather, Mamba
  head/state partitioning, loss normalization, and output reconstruction.
- Clamp padded cumulative sequence boundaries to the physical local token count
  before constructing Mamba sequence indices.
- Keep static Megatron collectives in lockstep when local CP subgroups differ.

The first three contracts have already informed the current Megatron branch,
including the packed-boundary regression fix at `a1b5863b3c`. The MR's lockstep
implementation uses NeMo-RL-owned head-node scheduling, worker metadata, dummy
microbatches, RL loss normalization, and vLLM orchestration. Those layers do not
map directly to Megatron's in-tree SFT pipeline.

## Goals

- Make DynamicCP scheduling collective-safe with global MoE, HybridEP, and DDP.
- Preserve every real sample exactly once per global batch.
- Preserve the minimum local CP size required by each sample while allowing a
  larger CP group when spare ranks are available.
- Keep all HDP ranks on the same forward/backward invocation number in every
  execution wave.
- Retain runtime CP-group behavior for packed THD attention, Mamba, and
  multimodal preprocessing.
- Fail before model execution if a schedule cannot satisfy the collective
  contract.
- Establish an eager DynamicCP baseline before adding TE partial CUDA Graph
  fixed-shape buckets.

## Non-goals

- Port NeMo-RL's Ray, vLLM, policy-worker, or RL loss code into Megatron.
- Add arbitrary non-power-of-two local CP groups.
- Change the global batch's real-sample count or loss normalization.
- Use dummy multimodal samples as the primary SFT scheduling mechanism.
- Combine the scheduler correction with CUDA Graph capture changes in one patch.

## Proposed Architecture

### 1. Emit full, uniform execution waves

`BalancedCPScheduler.next_hdp_group()` will treat each returned group as one
collective execution wave. A physical HDP rank may be assigned to only one
sample in that wave. Existing CP subgroups are therefore reusable only when none
of their ranks already has a sample in the current wave.

The scheduler continues to place samples in descending required-CP-size order.
When no remaining sample fits in the unassigned ranks, it closes the wave. If
the final placement leaves ranks unused, the existing power-of-two expansion
mechanism increases one or more assigned samples' local CP sizes until every
rank participates. Expansion changes only the amount of parallelism used by a
sample; it does not duplicate the sample or change its token/loss accounting.

For the observed requirements `[CP2, CP1, CP1, CP1]` on four HDP ranks, the
result is:

- Wave 0: `CP2 + CP1 + CP1`, one invocation on every rank.
- Wave 1: the remaining `CP1` sample expanded to `CP4`, one invocation on every
  rank.

Each `sample_id_groups[wave][rank]` entry must contain exactly one sample ID.
The same sample ID appears on every member of its local CP subgroup, which still
represents one logical sample execution.

### 2. Validate the collective contract before dispatch

A pure scheduler validator will check every emitted wave before data routing:

- the wave contains exactly `total_hdp_gpus` rank assignments;
- every rank has exactly one sample ID;
- each real sample belongs to exactly one contiguous, power-of-two subgroup;
- the subgroup size is at least `gpus_needed(sequence_length)`;
- every input sample appears in exactly one wave;
- no unknown sample ID appears.

Violations raise a `RuntimeError` containing the wave index, per-rank invocation
counts, and offending sample IDs. This converts a distributed hang into a
deterministic host-side error.

### 3. Equalize HybridEP input rows within each wave

DynamicCP plus HybridEP will enable
`moe_hybridep_pad_uneven_dispatch_inputs`. HybridEP then all-reduces the maximum
input-row count in its communication group, pads routing probabilities and
hidden states to the aligned maximum, and trims the combined output back to the
rank-local row count.

This padding solves tensor-shape equality inside a wave. It does not replace the
uniform-wave rule, because equal tensor shapes cannot repair different numbers
or orders of collectives.

### 4. Keep DDP synchronization wave-global

The existing forward/backward loop may retain `no_sync` for every wave except
the final wave. The scheduler invariant guarantees that all ranks execute one
forward and one backward per wave, so every rank exits `no_sync` on the same
logical invocation. The global sample count remains the count of unique real
samples, independent of CP expansion.

### 5. Preserve runtime packed-THD metadata

The current branch's runtime `cp_group`, `local_cp_size`, and clamped packed
sequence boundaries remain authoritative. The scheduler change must not add new
direct reads of global process groups in `megatron/core`; explicit groups will
continue to flow from the caller through `PackedSeqParams` and model APIs.

## Data Flow

1. Energon produces a packed global batch and per-sample sequence lengths.
2. `BalancedCPScheduler` computes each sample's minimum power-of-two CP size.
3. The scheduler packs samples into full, uniform waves and expands local CP
   groups to consume otherwise idle ranks.
4. The validator checks real-sample coverage, subgroup structure, and one
   invocation per rank per wave.
5. Data routing sends each sample to the ranks in its runtime CP subgroup.
6. Every rank executes one model forward/backward in the same wave order.
7. HybridEP pads uneven physical rows inside each collective and trims outputs.
8. The training loop accounts for each real sample once and synchronizes DDP on
   the final wave.

## Testing Strategy

### Pure scheduler tests

- Reproduce the failing four-sample pattern and verify the old `[1,1,1,2]`
  assignment becomes two uniform waves.
- Verify every rank has one invocation in every wave.
- Verify each logical sample is covered exactly once after collapsing repeated
  IDs within a CP subgroup.
- Verify each assigned CP size is power-of-two and no smaller than required.
- Verify a single short leftover sample expands to the full HDP group.
- Verify malformed schedules fail with actionable diagnostics.

### Regression tests

- Run the existing DynamicCP packed-sequence and Mamba boundary tests.
- Run the relevant HybridEP dispatcher and multimodal tests in the TE 2.14
  HybridEP image.
- Render the DynamicCP+HybridEP experiment profile and assert that uneven-input
  padding is enabled.

### Distributed gates

1. A three-step Nano debug run on eight nodes, with schedule diagnostics enabled.
2. A 20-step Nano eager run with warmup 3, compared with static CP4 control
   `5917477` for finite loss, skipped iterations, throughput, and step time.
3. A matched Nano run with TE partial CUDA Graph fixed-shape buckets.
4. Baseline, HybridEP, and HybridEP+CUDA-Graph validation on Super 3.5.

Each submitted job must use the exact pushed Megatron SHA mounted at
`/opt/Megatron-LM`, disable checkpoint writes for performance measurements, log
to the configured W&B project, and be monitored through at least the first five
minutes.

## Alternatives Considered

### NeMo-RL-style zero-loss dummy microbatches

This guarantees lockstep even when a rank has no real subgroup assignment and is
valuable as a reference and possible fallback. It is not the primary design for
SFT because a dummy forward can still interact with router auxiliary losses,
expert-bias state, multimodal preprocessing, and CUDA Graph shape capture. Full
real-sample waves avoid those extra semantics.

### Runtime TP/CP/EP process-group reconstruction

Rebuilding all model-parallel groups per sample would align every collective to
the local CP group, but it is invasive, expensive, and conflicts with static
parameter placement. It is unnecessary when global-MoE collectives can remain
lockstep.

### Padding only or fail-fast only

HybridEP padding alone leaves invocation counts unequal. A validator without a
new scheduling invariant improves diagnostics but does not run the workload.
Both are necessary supporting mechanisms, not complete solutions.

## Success Criteria

- The four-sample regression emits uniform full waves with complete real-sample
  coverage.
- Nano DynamicCP+HybridEP completes 20 finite steps without a collective stall,
  skipped iteration, or NaN.
- The DynamicCP run reports the same unique global sample count and comparable LM
  loss to its static CP4 control.
- The report records median and mean step time, TFLOP/s/GPU, throughput, loss,
  and CUDA Graph eager-fallback rate for every validated configuration.
- CUDA Graph work begins only after the eager DynamicCP+HybridEP gate passes.
