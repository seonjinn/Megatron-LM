# Multimodal Language CUDA Graph Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Transformer Engine per-layer CUDA Graph discovery work for the language decoder nested inside Nano/Super multimodal `LLaVAModel` wrappers, while reporting zero-layer ranks explicitly.

**Architecture:** Keep the global `get_attr_wrapped_model()` contract unchanged. Add a local resolver in `cuda_graphs.py` that preserves direct `decoder` discovery and adds recursive `.module` plus `.language_model` traversal. Keep vision discovery and multimodal preprocessing eager in this phase; only language-layer partial capture is enabled.

**Tech Stack:** Python, PyTorch, Transformer Engine `make_graphed_callables()`, pytest, Megatron Core CUDA Graph helpers.

## Global Constraints

- All source and test changes are confined to `/Users/sna/Nemotron_3.5_Super/.worktrees/megatron-nemotron-3p5-sft`.
- Do not change global `get_attr_wrapped_model()` behavior.
- Do not enable full-iteration or vision capture in this patch.
- `moe_preprocess` remains valid only when paired with `moe_router`.
- Run `uv run isort` after changing Python imports.
- Run focused tests before broader CUDA Graph tests.

---

### Task 1: Add failing structural tests for language decoder resolution

**Files:**
- Modify: `tests/unit_tests/transformer/test_cuda_graphs.py`
- Read: `megatron/core/transformer/cuda_graphs.py:2042-2209`

**Interfaces:**
- Consumes: the new private resolver to be named `_get_cuda_graph_decoder_owner(model_chunk)`.
- Produces: tests defining direct, nested-language, wrapped-language, and missing-decoder behavior.

- [ ] **Step 1: Inspect existing CUDA Graph test fixtures and imports**

Run:

```bash
rg -n "TECudaGraphHelper|_layer_is_graphable|get_attr_wrapped_model|SimpleNamespace|decoder" \
  tests/unit_tests/transformer/test_cuda_graphs.py
```

Use the existing test module's import style and avoid importing CUDA-only dependencies for structural tests.

- [ ] **Step 2: Write the failing tests**

Add a small model fixture using `types.SimpleNamespace` and four tests:

```python
def test_cuda_graph_decoder_owner_resolves_direct_decoder():
    decoder = SimpleNamespace(layers=[])
    owner = SimpleNamespace(decoder=decoder)
    assert _get_cuda_graph_decoder_owner(owner) is owner


def test_cuda_graph_decoder_owner_resolves_language_model_decoder():
    decoder = SimpleNamespace(layers=[])
    language_model = SimpleNamespace(decoder=decoder)
    model = SimpleNamespace(language_model=language_model)
    assert _get_cuda_graph_decoder_owner(model) is language_model


def test_cuda_graph_decoder_owner_resolves_wrapped_language_model_decoder():
    decoder = SimpleNamespace(layers=[])
    language_model = SimpleNamespace(decoder=decoder)
    model = SimpleNamespace(module=SimpleNamespace(language_model=language_model))
    assert _get_cuda_graph_decoder_owner(model) is language_model


def test_cuda_graph_decoder_owner_returns_none_without_decoder():
    assert _get_cuda_graph_decoder_owner(SimpleNamespace()) is None
```

Import `_get_cuda_graph_decoder_owner` from `megatron.core.transformer.cuda_graphs`.

- [ ] **Step 3: Run the tests and verify the expected RED state**

Run:

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "decoder_owner" -q
```

Expected: collection succeeds and all four tests fail because the resolver does not yet exist.

- [ ] **Step 4: Commit the RED tests**

```bash
git add tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -m "test: cover multimodal cuda graph decoder discovery"
```

### Task 2: Implement the LLaVA-aware decoder resolver

**Files:**
- Modify: `megatron/core/transformer/cuda_graphs.py:2042-2150`
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py`

**Interfaces:**
- Consumes: the four failing resolver tests from Task 1.
- Produces: `_get_cuda_graph_decoder_owner(model_chunk) -> object | None`.

- [ ] **Step 1: Implement the minimal resolver**

Add this helper before `_layer_is_graphable`:

```python
def _get_cuda_graph_decoder_owner(model_chunk):
    """Return the model object that owns the language decoder, if present."""
    current = model_chunk
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if getattr(current, "decoder", None) is not None:
            return current
        module = getattr(current, "module", None)
        if module is not None:
            current = module
            continue
        language_model = getattr(current, "language_model", None)
        if language_model is not None:
            current = language_model
            continue
        break
    return None
```

Update `TECudaGraphHelper._discover_layers()` to use this resolver. Preserve the existing model-owner semantics by setting `chunk_with_decoder` to the returned owner and treating `None` as the current no-layer case. Do not modify `get_attr_wrapped_model()`.

- [ ] **Step 2: Run the focused tests**

Run:

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "decoder_owner" -q
```

Expected: all four resolver tests pass.

- [ ] **Step 3: Run existing structural CUDA Graph tests**

Run:

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py -q
```

Expected: the full existing module remains green.

- [ ] **Step 4: Format imports if needed and commit**

```bash
uv run isort megatron/core/transformer/cuda_graphs.py tests/unit_tests/transformer/test_cuda_graphs.py
git add megatron/core/transformer/cuda_graphs.py tests/unit_tests/transformer/test_cuda_graphs.py
git commit -s -m "feat: discover multimodal language cuda graph layers"
```

### Task 3: Add a failing lifecycle test for zero graphable layers

**Files:**
- Modify: `tests/unit_tests/transformer/test_cuda_graphs.py:819-1067`.
- Read: `megatron/training/training.py:3357-3366` and `megatron/core/transformer/cuda_graphs.py:119-150`.

**Interfaces:**
- Consumes: the existing `CudaGraphMemoryReporter` lifecycle API.
- Produces: a regression test proving that a zero-callable capture is reported as skipped rather than passed to `capture_complete()`.

- [ ] **Step 1: Locate the existing lifecycle fixture**

Run:

```bash
rg -n "CudaGraphMemoryReporter|capture_start|capture_complete|graphs_created" \
  tests/unit_tests/transformer tests/unit_tests/training
```

Use the existing fixture rather than creating a second reporter abstraction.

- [ ] **Step 2: Write the failing lifecycle test**

Add a test that calls the training capture wrapper with a helper whose `create_cudagraphs()` completes without graphs and whose `graphs_created()` and `graph_count()` return `False` and `0`. Assert that no `ValueError` is raised and that the reporter emits a distinct `capture_skipped` record with zero graphs.

- [ ] **Step 3: Run the test and verify RED**

Run:

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "capture_wrapper_skips_zero_graphs" -q
```

Expected: failure at `capture_complete()` with the current `capture_complete requires a graph profile with at least one created graph` message.

### Task 4: Implement explicit zero-graph capture reporting

**Files:**
- Modify: `megatron/core/transformer/cuda_graphs.py:100-150` and `megatron/training/training.py:3357-3366`.
- Test: `tests/unit_tests/transformer/test_cuda_graphs.py:819-1067`.

**Interfaces:**
- Consumes: the existing reporter `_emit()` method and helper graph-count API.
- Produces: `capture_skipped` telemetry for zero graphable layers; `capture_complete` only for positive graph counts.

- [ ] **Step 1: Add the reporter method**

Implement:

```python
def capture_skipped(self, *, graph_count: int = 0) -> None:
    if self._already_emitted("capture_skipped"):
        return
    if not self.graph_profile:
        raise ValueError("capture_skipped is only valid for a CUDA Graph profile")
    self._emit("capture_skipped", graphs_created=False, graph_count=graph_count)
```

- [ ] **Step 2: Guard the training lifecycle call**

Update `_capture_transformer_engine_cuda_graphs()`:

```python
cuda_graph_helper.create_cudagraphs()
graphs_created = cuda_graph_helper.graphs_created()
graph_count = cuda_graph_helper.graph_count()
if graphs_created and graph_count > 0:
    memory_reporter.capture_complete(
        graphs_created=True,
        graph_count=graph_count,
    )
else:
    memory_reporter.capture_skipped(graph_count=graph_count)
```

This keeps real capture exceptions visible and only changes the zero-graph branch.

- [ ] **Step 3: Run focused lifecycle tests**

Run:

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py \
  -k "cuda_graph_memory or capture_wrapper" -q
```

Expected: all lifecycle tests pass, including the new zero-graph regression test.

- [ ] **Step 4: Run the broader CUDA Graph unit suite**

Run:

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_vision_cuda_graphs.py \
  tests/unit_tests/transformer/test_thd_cuda_graph.py \
  tests/unit_tests/transformer/test_full_cuda_graph.py -q
```

Expected: no regressions in language, vision, THD, or full-iteration unit tests.

- [ ] **Step 5: Format and commit lifecycle changes**

```bash
uv run isort megatron/core/transformer/cuda_graphs.py megatron/training/training.py
git add megatron/core/transformer/cuda_graphs.py megatron/training/training.py tests/unit_tests/transformer
git commit -s -m "fix: report skipped cuda graph capture explicitly"
```

### Task 5: Validate Nano and prepare Super promotion

**Files:**
- Read: `experiments/nemotron_3p5_sft/HANDOFF_2026-08-03.md`
- Read: pipeline configuration and launcher files only; no pipeline changes in this task.

**Interfaces:**
- Consumes: the committed Megatron changes and existing TE 2.14 image/mount.
- Produces: runtime evidence that Nano `attn` creates graphs and that zero-layer ranks no longer fail with the secondary `ValueError`.

- [ ] **Step 1: Run the full local unit gate**

```bash
uv run pytest tests/unit_tests/transformer/test_cuda_graphs.py \
  tests/unit_tests/transformer/test_vision_cuda_graphs.py \
  tests/unit_tests/transformer/test_thd_cuda_graph.py \
  tests/unit_tests/transformer/test_full_cuda_graph.py -q
```

- [ ] **Step 2: Push the Megatron branch to the existing fork**

```bash
git push fork HEAD:sna/nemotron-3p5-sft-tuning
```

- [ ] **Step 3: Submit one Nano 8-node `attn` diagnostic**

Use the existing pipeline launch configuration with `TRAIN_ITERS=20`, `CUDA_GRAPH_WARMUP_STEPS=3`, `CUDA_GRAPH_IMPL=transformer_engine`, and `CUDA_GRAPH_MODULES=attn`. Do not save checkpoints. Require runtime-probe success, `graph_count > 0` on language-bearing ranks, and no `No graphable layers found` message for ranks that contain language decoder layers.

- [ ] **Step 4: Verify the remote log and telemetry**

Record the job ID, Megatron SHA, image SHA, graph count, warmup/capture timings, and tokens-per-second in the existing handoff/run manifest. Treat any TE capture failure, shape mismatch, or numerical divergence as a separate follow-up task.

- [ ] **Step 5: Run Super baseline before Super CUDA Graph**

Run the existing Super baseline topology first. Only after baseline startup and loss progression pass, submit Super `attn` with the same warmup and iteration settings. Do not infer Super performance from Nano.
