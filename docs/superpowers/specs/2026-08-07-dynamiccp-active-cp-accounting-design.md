# DynamicCP Active-CP Accounting Design

## Goal

Make Megatron's packed-sequence FLOP and W&B metrics exact when one static
context-parallel domain executes mixed DynamicCP groups, while preserving the
validated Super 3.5 per-token loss and gradient behavior.

## Current behavior

The Super 3.5 job uses static CP16 and schedules packed payloads on active CP4,
CP8, or CP16 groups. Every rank in an active group observes the same packed
`cu_seqlens`. `update_seqlen_stats_from_cu_seqlens` therefore records each
logical payload once per active CP rank. `consume_seqlen_stats_in_iteration`
world-reduces those values and currently divides by static `TP * CP * PP`.
This undercounts CP4 work by four and CP8 work by two.

`update_packed_sequence_stats` has a related but different problem. It accepts
only static CP rank zero. DynamicCP can place several active groups inside one
static CP domain, so distributions and trained-token counts from groups that
do not include static CP rank zero are omitted.

The production Super recipe enables `--calculate-per-token-loss`. The reviewed
per-token path accumulates sharded `num_tokens` across the static domain and is
not changed by this design. The legacy non-per-token path scales losses with
the active CP size but divides by the static microbatch count; it is not
qualified for packed-wave DynamicCP.

## Design

### Additive FLOP statistics

`update_seqlen_stats_from_cu_seqlens` accepts an optional
`local_cp_size`. HybridCP callers pass the routed active group size. Each rank
adds `sum(L) / local_cp_size` and `sum(L^2) / local_cp_size` to the device
accumulator. After the existing world all-reduce, HybridCP consumption divides
only by `TP * PP`. Non-HybridCP callers retain unweighted updates and the
existing `TP * CP * PP` de-duplication.

This is exact because every logical payload is replicated on exactly
`local_cp_size` ranks. It supports mixed CP1, CP2, CP4, CP8, and CP16 waves
without leader election or another collective. The rank-side `cu_seqlens`
remain authoritative because they contain the post-multimodal-expansion packed
subsequence boundaries needed for the attention `sum(L^2)` term.

### Non-additive packed-sequence distributions

Packed sequence lengths cannot be divided by the replication count because
their median, minimum, and maximum must describe whole logical sequences.
`update_packed_sequence_stats` therefore selects one representative from each
active CP group:

- TP rank zero and the last PP stage remain the model-parallel representative.
- Non-HybridCP retains static CP rank zero.
- DynamicCP CP1 contributes directly.
- DynamicCP groups larger than one contribute only from rank zero within the
  routed CP process group.

The selected representative holds the full pre-sharding `sample_lengths` and
`loss_mask`, so the existing object gather produces exact whole-sequence and
trained-token statistics without adding a collective.

### Loss safety

Packed-wave DynamicCP requires per-token loss normalization. Argument
validation fails before training when HybridCP is requested without
`calculate_per_token_loss`. This prevents an unqualified legacy path from
silently changing gradient scale with the active CP mix or packed-wave count.
The current Super and Ultra recipes already satisfy this requirement.

## Correctness audit disposition

Existing tests cover exactly-once logical sample scheduling, one model call per
rank per wave, contiguous power-of-two active groups, CP1 metadata, Mamba/MTP
group resolution, packed real and padded THD boundaries, label reconstruction,
and vision index rebasing. The completed 20-step Super runs additionally show
finite LM/MTP losses and no skipped or NaN iterations.

The following risks remain outside the metric patch and receive focused
regression tests or explicit documentation rather than speculative production
changes:

- media tensors are not part of the language-token packing capacity model;
- audio tensors with unequal non-leading dimensions need an explicit contract;
- validation variants that return non-dict data do not carry training sample
  accounting in the same channel;
- full multimodal performance qualification still requires deterministic
  image, video, and audio batches.

## Tests

The regression suite proves:

1. Static CP de-duplication remains unchanged.
2. Mixed active CP groups return exact global `sum(L)` and `sum(L^2)`.
3. CP1, CP4, CP8, and CP16 contributions compose in one iteration.
4. Packed sequence distributions include one copy from every active group.
5. Trained tokens are neither dropped nor duplicated.
6. HybridCP argument validation rejects the non-per-token loss path.
7. Existing scheduler, multimodal packing, Mamba, MTP, and FLOP tests remain
   green in the TE 2.14 HybridEP environment.

## Performance and compatibility

The change keeps the existing two-element world all-reduce and object gather.
It adds only two scalar divisions per packed payload and one active-group-rank
check for optional packed-sequence logging. It changes reporting, not model
outputs, optimizer state, checkpoint layout, scheduling, or communication
topology.
