# Multimodal Language CUDA Graph Support Design

## Goal

Enable Transformer Engine per-layer CUDA Graph capture for the language-model
decoder nested inside the multimodal `LLaVAModel` wrapper. The first delivery
targets Nano 3.5 and Super 3.5 language-layer scopes (`attn`, `mamba`, and MoE
router/preprocess) while keeping vision and multimodal preprocessing eager.

## Current failure

`TECudaGraphHelper._discover_layers()` resolves only a direct `decoder`
attribute through the existing `.module` wrapper chain. A multimodal model has
the decoder at `language_model.decoder`, so discovery returns no callable
layers. The helper then skips capture. In the current experiment branch, the
memory reporter subsequently receives `graphs_created=False` and
`graph_count=0`, which produces a secondary `ValueError`.

## Recommended approach

Add a CUDA-Graph-specific resolver inside `cuda_graphs.py` rather than changing
the global `get_attr_wrapped_model()` contract. The resolver will preserve the
existing direct `decoder` behavior and additionally handle these paths:

1. `model_chunk.decoder`
2. `model_chunk.module.decoder` (recursive wrapper unwrapping)
3. `model_chunk.language_model.decoder`
4. `model_chunk.module.language_model.decoder`

The resolver returns the object owning `decoder`, so the existing layer and MTP
discovery code can remain unchanged. The vision-specific helper remains
separate and is not changed in this phase.

## Capture lifecycle behavior

Ranks or pipeline chunks with no graphable language layers must complete the
capture synchronization without raising a misleading error. The lifecycle
reporter will distinguish `capture_skipped` (zero graphable callables) from
`capture_complete` (one or more graphs created). This does not hide real TE
capture failures; it only prevents the zero-layer condition from being reported
as a successful capture or as an unrelated invariant failure.

## Scope and static-input policy

- `attn`: first validation target; MoE expert dispatch remains eager.
- `mamba`: validate separately after attention capture works.
- `moe_router` and `moe_preprocess`: validate together, subject to static
  routing/preprocess shapes and HybridEP constraints.
- Whole-layer and full-iteration capture are out of scope for the first patch.
- Vision encoder, image-token expansion, projector, and multimodal merge stay
  eager in the first patch.

## Tests

Add CPU-only structural tests for the resolver:

- direct `decoder` model continues to resolve;
- `language_model.decoder` resolves;
- `.module.language_model.decoder` resolves;
- missing decoder returns no owner without raising an unrelated exception.

Retain existing CUDA Graph lifecycle and vision tests. After unit tests pass,
run the Nano 20-step / warmup-3 `attn` smoke test using the TE 2.14 image. Then
run a Super baseline followed by the same language-only `attn` scope. Promotion
to Mamba and MoE scopes requires separate evidence and is not implied by the
resolver test.

## Non-goals

- Replacing the upstream CUDA Graph implementation.
- Changing global model attribute resolution.
- Capturing arbitrary dynamic multimodal batches with one graph.
- Claiming full multimodal or full-iteration CUDA Graph support before static
  input and numerical-equivalence validation.
