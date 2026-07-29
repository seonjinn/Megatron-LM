# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import gc
import importlib
import importlib.util
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest


def _load_bank_module() -> ModuleType:
    spec = importlib.util.find_spec("megatron.core.transformer.te_cuda_graph_bank")
    assert spec is not None
    return importlib.import_module(spec.name)


class _FakeGraph:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeLayer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cuda_graphs = []
        self.cuda_graph_manual_hooks = []


class _FakeHelper:
    def __init__(
        self,
        layers: list[_FakeLayer],
        graphs_by_layer: list[list[_FakeGraph]],
        *,
        modules: tuple[str, ...] = ("attn",),
        fail_capture: bool = False,
        install_packed_contract: bool = False,
    ) -> None:
        self.flattened_callables = layers
        self.config = SimpleNamespace(
            cuda_graph_modules=modules,
            overlap_moe_expert_parallel_comm=False,
        )
        self._graphs_by_layer = graphs_by_layer
        self._fail_capture = fail_capture
        self._install_packed_contract = install_packed_contract
        self.capture_finished = False
        self.graphs_created = False
        self._capture_finished = False
        self._graphs_created = False
        self._capture_attempted = False
        self.saw_empty_graph_lists = False
        self.manual_hook_setup_calls = 0

    def _capture_cuda_graph_lists(
        self, *, num_microbatches: int
    ) -> tuple[tuple[_FakeLayer, tuple[_FakeGraph, ...]], ...]:
        self.saw_empty_graph_lists = all(
            layer.cuda_graphs == [] for layer in self.flattened_callables
        )
        assert self.saw_empty_graph_lists
        if self._fail_capture:
            self.flattened_callables[0].cuda_graphs.append(_FakeGraph("partial"))
            raise RuntimeError("capture failed")

        for layer, graphs in zip(self.flattened_callables, self._graphs_by_layer):
            assert len(graphs) == num_microbatches
            layer.cuda_graphs.extend(graphs)
            if self._install_packed_contract:
                layer._te_cuda_graph_packed_seq_params_static_metadata = {}
                layer._te_cuda_graph_packed_seq_params_tensor_kwarg_names = ()
        self.capture_finished = True
        self.graphs_created = True
        self._capture_finished = True
        self._graphs_created = True
        return tuple(
            (layer, tuple(layer.cuda_graphs)) for layer in self.flattened_callables
        )

    def cuda_graph_set_manual_hooks(self) -> None:
        self.manual_hook_setup_calls += 1
        for layer in self.flattened_callables:
            layer.cuda_graph_manual_hooks = [(f"hook-{layer.name}", (layer,))]


def _make_manager(
    layers: list[_FakeLayer],
    *,
    modules: tuple[str, ...] = ("attn",),
    drained=lambda: True,
    runtime_num_microbatches=lambda: 2,
):
    bank_module = _load_bank_module()
    return bank_module.TECudaGraphBankManager(
        layers,
        cuda_graph_modules=modules,
        assert_model_drained=drained,
        graph_reset_supported=True,
        synchronize=lambda: None,
        runtime_num_microbatches=runtime_num_microbatches,
    )


def _capture(
    manager,
    layers: list[_FakeLayer],
    prefix: str,
    *,
    num_microbatches: int = 2,
    modules: tuple[str, ...] = ("attn",),
    install_packed_contract: bool = False,
):
    graphs = [
        [
            _FakeGraph(f"{prefix}-l{layer_index}-g{graph_index}")
            for graph_index in range(num_microbatches)
        ]
        for layer_index in range(len(layers))
    ]
    helper = _FakeHelper(
        layers,
        graphs,
        modules=modules,
        install_packed_contract=install_packed_contract,
    )
    return manager.capture(helper, num_microbatches=num_microbatches), helper, graphs


def test_direct_manager_requires_runtime_microbatch_provider() -> None:
    bank_module = _load_bank_module()

    with pytest.raises(TypeError, match="runtime_num_microbatches"):
        bank_module.TECudaGraphBankManager([_FakeLayer("0")])


def test_capture_uses_empty_owned_lists_and_activation_installs_the_same_lists() -> (
    None
):
    layers = [_FakeLayer("0"), _FakeLayer("1")]
    old_lists = [[_FakeGraph("old-0")], [_FakeGraph("old-1")]]
    for layer, old_list in zip(layers, old_lists):
        layer.cuda_graphs = old_list
    manager = _make_manager(layers)
    shared_hooks = [[("hook-0", ())], [("hook-1", ())]]
    for layer, hooks in zip(layers, shared_hooks):
        layer.cuda_graph_manual_hooks = hooks

    bank, helper, graphs = _capture(manager, layers, "new")

    assert helper.saw_empty_graph_lists
    assert helper.manual_hook_setup_calls == 0
    assert [layer.cuda_graphs for layer in layers] == old_lists
    bank.activate()
    for layer_index, (layer, expected_graphs) in enumerate(zip(layers, graphs)):
        assert layer.cuda_graphs is bank._owned_graph_lists[layer_index]
        assert tuple(layer.cuda_graphs) == tuple(expected_graphs)
        assert layer.cuda_graph_manual_hooks is shared_hooks[layer_index]
        assert bank.graphs_by_layer[layer_index] == (layer, tuple(expected_graphs))


def test_activation_rejects_bank_from_another_manager() -> None:
    layers = [_FakeLayer("0")]
    owner = _make_manager(layers)
    foreign = _make_manager(layers)
    bank, _, _ = _capture(owner, layers, "owned")

    with pytest.raises(ValueError, match="different TECudaGraphBankManager"):
        foreign.activate(bank)


def test_replay_guard_rejects_wrong_num_microbatches_before_graph_selection() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank", num_microbatches=2)
    bank.activate()

    with pytest.raises(ValueError, match="num_microbatches"):
        manager.get_graph(bank, layers[0], microbatch_index=0, num_microbatches=3)


def test_resetting_inactive_bank_leaves_active_graphs_untouched() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    active_bank, _, active_graphs = _capture(manager, layers, "active")
    active_bank.activate()
    inactive_bank, _, inactive_graphs = _capture(manager, layers, "inactive")

    inactive_bank.reset()

    layer_graphs = layers[0].cuda_graphs
    assert layer_graphs
    assert layer_graphs is active_bank._owned_graph_lists[0]
    assert [graph.reset_calls for graph in active_graphs[0]] == [0, 0]
    assert [graph.reset_calls for graph in inactive_graphs[0]] == [1, 1]


def test_capture_exception_restores_previous_bank_contract() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    active_bank, _, _ = _capture(
        manager, layers, "active", install_packed_contract=True
    )
    active_bank.activate()
    installed_list = layers[0].cuda_graphs
    installed_hooks = layers[0].cuda_graph_manual_hooks
    installed_metadata = layers[0]._te_cuda_graph_packed_seq_params_static_metadata
    failing_helper = _FakeHelper(
        layers,
        [[_FakeGraph("unused-0"), _FakeGraph("unused-1")]],
        fail_capture=True,
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        manager.capture(failing_helper, num_microbatches=2)

    assert layers[0].cuda_graphs is installed_list
    assert layers[0].cuda_graph_manual_hooks is installed_hooks
    assert (
        layers[0]._te_cuda_graph_packed_seq_params_static_metadata is installed_metadata
    )
    assert manager.active_bank is active_bank


def test_manager_validation_failure_marks_helper_uncreated_and_resets_graphs() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    active_bank, _, _ = _capture(manager, layers, "active")
    active_bank.activate()
    rejected_graphs = [[_FakeGraph("rejected-0"), _FakeGraph("rejected-1")]]
    rejected_helper = _FakeHelper(
        layers,
        rejected_graphs,
        install_packed_contract=True,
    )

    with pytest.raises(ValueError, match="packed_input_signature"):
        manager.capture(rejected_helper, num_microbatches=2)

    assert rejected_helper._capture_finished is False
    assert rejected_helper._graphs_created is False
    assert [graph.reset_calls for graph in rejected_graphs[0]] == [1, 1]
    assert layers[0].cuda_graphs is active_bank._owned_graph_lists[0]
    assert manager.active_bank is active_bank


def test_reset_is_idempotent_and_resets_each_owned_graph_identity_once() -> None:
    layers = [_FakeLayer("0"), _FakeLayer("1")]
    shared_graph = _FakeGraph("shared")
    helper = _FakeHelper(
        layers,
        [[shared_graph, _FakeGraph("l0")], [shared_graph, _FakeGraph("l1")]],
    )
    manager = _make_manager(layers)
    bank = manager.capture(helper, num_microbatches=2)

    bank.reset()
    bank.reset()

    assert shared_graph.reset_calls == 1
    assert helper._graphs_by_layer[0][1].reset_calls == 1
    assert helper._graphs_by_layer[1][1].reset_calls == 1
    assert bank.graphs_by_layer == ()
    assert bank._owned_graph_lists == ()
    assert bank._layer_contracts == ()
    assert manager.registered_bank_count == 0


def test_mutable_bank_reset_fields_cannot_bypass_canonical_reset() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, graphs = _capture(manager, layers, "bank")
    bank._is_reset = True
    bank._reset_graph_ids = {id(graph) for graph in graphs[0]}

    bank.reset()
    bank.reset()

    assert [graph.reset_calls for graph in graphs[0]] == [1, 1]
    assert manager.registered_bank_count == 0
    assert bank.graphs_by_layer == ()


def test_reset_and_external_cache_drop_release_graph_references() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    graph = _FakeGraph("owned")
    helper = _FakeHelper(layers, [[graph, _FakeGraph("other")]])
    graph_reference = weakref.ref(graph)
    bank = manager.capture(helper, num_microbatches=2)
    del helper
    del graph

    bank.reset()
    gc.collect()

    assert graph_reference() is None


def test_reset_bank_cannot_be_reactivated_or_replayed() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()
    bank.reset()

    with pytest.raises(ValueError, match="reset"):
        bank.activate()
    with pytest.raises(ValueError, match="reset"):
        manager.get_graph(
            bank,
            layers[0],
            microbatch_index=0,
            num_microbatches=2,
        )


def test_refresh_manual_hooks_updates_active_bank_contract() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()
    refreshed_hooks = [("fresh", (layers[0],))]
    layers[0].cuda_graph_manual_hooks = refreshed_hooks

    manager.refresh_manual_hooks(bank)
    layers[0].cuda_graph_manual_hooks = []
    bank.activate()

    assert layers[0].cuda_graph_manual_hooks is refreshed_hooks


def test_helper_is_one_shot_and_reuse_preserves_active_bank() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, helper, _ = _capture(manager, layers, "bank")
    bank.activate()
    installed_list = layers[0].cuda_graphs

    with pytest.raises(ValueError, match="one-shot"):
        manager.capture(helper, num_microbatches=2)

    assert helper._capture_finished is True
    assert helper._graphs_created is True
    assert layers[0].cuda_graphs is installed_list
    assert manager.active_bank is bank


def test_registration_rejects_mutated_public_graph_views() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.graphs_by_layer = ((layers[0], (_FakeGraph("forged"),) * 2),)

    with pytest.raises(ValueError, match="registration"):
        bank.activate()


def test_registration_rejects_graph_identity_shared_by_live_bank() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    shared = _FakeGraph("shared")
    manager.capture(_FakeHelper(layers, [[shared, _FakeGraph("first")]]), num_microbatches=2)

    with pytest.raises(ValueError, match="shared"):
        manager.capture(
            _FakeHelper(layers, [[shared, _FakeGraph("second")]]),
            num_microbatches=2,
        )


def test_installed_replay_guard_rejects_runtime_topology_change() -> None:
    runtime = {"count": 2}
    layers = [_FakeLayer("0")]
    manager = _make_manager(
        layers, runtime_num_microbatches=lambda: runtime["count"]
    )
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()

    assert (
        layers[0]._te_cuda_graph_bank_replay_guard(
            layers[0], layers[0].cuda_graphs, 0
        )
        == 0
    )
    runtime["count"] = 3
    with pytest.raises(ValueError, match="runtime num_microbatches"):
        layers[0]._te_cuda_graph_bank_replay_guard(
            layers[0], layers[0].cuda_graphs, 0
        )


def test_replay_guard_hot_path_does_not_run_full_bank_validation() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()
    manager._validate_bank = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("full validation reached replay hot path")
    )

    layers[0]._te_cuda_graph_bank_replay_guard(
        layers[0], layers[0].cuda_graphs, 0
    )
    assert (
        manager.get_graph(
            bank,
            layers[0],
            microbatch_index=0,
            num_microbatches=2,
        )
        is layers[0].cuda_graphs[0]
    )


def test_get_graph_rejects_runtime_change_despite_stale_supplied_count() -> None:
    runtime = {"count": 2}
    layers = [_FakeLayer("0")]
    manager = _make_manager(
        layers, runtime_num_microbatches=lambda: runtime["count"]
    )
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()
    runtime["count"] = 3

    with pytest.raises(ValueError, match="runtime provider"):
        manager.get_graph(
            bank,
            layers[0],
            microbatch_index=0,
            num_microbatches=2,
        )


def test_replay_guard_rejects_graph_list_swapped_between_layers() -> None:
    layers = [_FakeLayer("0"), _FakeLayer("1")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()
    layers[0].cuda_graphs = layers[1].cuda_graphs

    with pytest.raises(ValueError, match="does not match"):
        layers[0]._te_cuda_graph_bank_replay_guard(
            layers[0], layers[0].cuda_graphs, 0
        )


def test_replay_guard_rejects_same_length_in_place_callable_mutation() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.activate()
    layers[0].cuda_graphs[0] = _FakeGraph("mutated")

    with pytest.raises(ValueError, match="canonical registration"):
        layers[0]._te_cuda_graph_bank_replay_guard(
            layers[0], layers[0].cuda_graphs, 0
        )


def test_graphable_module_calls_guard_before_forward_and_backward_selection() -> None:
    import torch

    from megatron.core.transformer.module import GraphableMegatronModule

    selections: list[str] = []

    class _Graph:
        def __call__(self, *args, **kwargs):
            selections.append("forward")

        def backward_dw(self) -> None:
            selections.append("backward_dw")

    def _reject(layer, graphs, microbatch_index) -> int:
        raise ValueError("runtime num_microbatches mismatch")

    layer = SimpleNamespace(
        cuda_graphs=[_Graph()],
        current_microbatch=0,
        _te_cuda_graph_bank_replay_guard=_reject,
    )
    with pytest.raises(ValueError, match="runtime num_microbatches"):
        GraphableMegatronModule._te_cuda_graph_replay(layer, torch.empty(1))
    with pytest.raises(ValueError, match="runtime num_microbatches"):
        GraphableMegatronModule._te_cuda_graph_backward_dw_graph(layer, 0)
    assert selections == []


def test_graphable_module_uses_validated_index_for_forward_and_backward() -> None:
    import torch

    from megatron.core.transformer.module import GraphableMegatronModule

    selections: list[str] = []

    class _CallableGraph(_FakeGraph):
        def __call__(self, *args, **kwargs) -> str:
            selections.append(f"forward-{self.name}")
            return self.name

        def backward_dw(self) -> None:
            selections.append(f"backward-{self.name}")

    layer = _FakeLayer("0")
    graphs = [_CallableGraph("g0"), _CallableGraph("g1")]
    manager = _make_manager([layer])
    bank = manager.capture(_FakeHelper([layer], [graphs]), num_microbatches=2)
    bank.activate()
    layer.current_microbatch = 1
    layer._get_te_cuda_graph_replay_args = lambda *args, **kwargs: (args, kwargs)

    assert (
        GraphableMegatronModule._te_cuda_graph_replay(layer, torch.empty(1))
        == "g1"
    )
    GraphableMegatronModule._te_cuda_graph_backward_dw_graph(layer, 1)

    assert selections == ["forward-g1", "backward-g1"]


@pytest.mark.parametrize("operation", ["activate", "reset"])
def test_live_delayed_work_rejects_activation_and_eviction(operation: str) -> None:
    drain_state = {"drained": True}
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers, drained=lambda: drain_state["drained"])
    bank, _, graphs = _capture(manager, layers, "bank")
    drain_state["drained"] = False

    with pytest.raises(RuntimeError, match="not drained"):
        getattr(bank, operation)()

    assert [graph.reset_calls for graph in graphs[0]] == [0, 0]
    assert layers[0].cuda_graphs == []


def test_live_delayed_work_rejects_capture_before_lists_are_uninstalled() -> None:
    layers = [_FakeLayer("0")]
    installed_list = [_FakeGraph("active")]
    layers[0].cuda_graphs = installed_list
    helper = _FakeHelper(
        layers,
        [[_FakeGraph("new-0"), _FakeGraph("new-1")]],
    )
    manager = _make_manager(layers, drained=lambda: False)

    with pytest.raises(RuntimeError, match="not drained"):
        manager.capture(helper, num_microbatches=2)

    assert not helper.saw_empty_graph_lists
    assert layers[0].cuda_graphs is installed_list


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("layer_ids", (123,)),
        ("graph_counts", (3,)),
        ("cuda_graph_modules", ("mamba",)),
        ("packed_input_signature", (("present", ("different",)),)),
        ("moe_attribute_schema", ((123, ("tensor_store",)),)),
    ],
)
def test_activation_rejects_fingerprint_mismatch(
    field: str, replacement: object
) -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    bank, _, _ = _capture(manager, layers, "bank")
    bank.fingerprint = dataclasses.replace(bank.fingerprint, **{field: replacement})

    with pytest.raises(ValueError, match=field):
        bank.activate()


def test_activation_clears_packed_contract_absent_from_target_bank() -> None:
    layers = [_FakeLayer("0")]
    manager = _make_manager(layers)
    unpacked_bank, _, _ = _capture(manager, layers, "unpacked")

    layers[0]._te_cuda_graph_packed_seq_params_static_metadata = {}
    layers[0]._te_cuda_graph_packed_seq_params_tensor_kwarg_names = ()
    assert hasattr(layers[0], "_te_cuda_graph_packed_seq_params_static_metadata")
    unpacked_bank.activate()

    assert not hasattr(layers[0], "_te_cuda_graph_packed_seq_params_static_metadata")
    assert not hasattr(layers[0], "_te_cuda_graph_packed_seq_params_tensor_kwarg_names")


def test_vision_wrapping_preserves_owned_graph_list_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core.transformer.cuda_graphs import TECudaGraphHelper, VisionTECudaGraphHelper

    layer = _FakeLayer("vision")
    layer.cuda_graphs = [_FakeGraph("vision-0"), _FakeGraph("vision-1")]
    owned_list = layer.cuda_graphs
    helper = VisionTECudaGraphHelper.__new__(VisionTECudaGraphHelper)
    helper.flattened_callables = [layer]
    monkeypatch.setattr(
        TECudaGraphHelper,
        "_finish_capturing",
        lambda self, start_time: None,
    )

    helper._finish_capturing(0.0)

    assert layer.cuda_graphs is owned_list
    assert len(layer.cuda_graphs) == 2
    assert all(callable(graph) for graph in layer.cuda_graphs)


def test_real_helper_abort_clears_capture_state_and_resets_partial_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    partial_graph = _FakeGraph("partial")
    layer = _FakeLayer("0")
    helper = cuda_graphs.TECudaGraphHelper.__new__(cuda_graphs.TECudaGraphHelper)
    helper.flattened_callables = [layer]
    helper.callables_per_chunk = [[layer]]
    helper.config = SimpleNamespace(
        cuda_graph_modules=(),
        sequence_parallel=False,
        fine_grained_activation_offloading=True,
    )
    helper._capture_finished = False
    helper._graphs_created = False
    helper._capture_gc_frozen = False
    te_capture_end_calls: list[None] = []
    offload_reset_calls: list[None] = []
    gc_unfreeze_calls: list[None] = []

    def _start_capturing() -> float:
        cuda_graphs._set_capture_start()
        cuda_graphs._set_warmup_start()
        helper._capture_gc_frozen = True
        return 1.0

    def _fail_input_creation():
        layer.cuda_graphs.append(partial_graph)
        raise RuntimeError("input creation failed")

    class _OffloadInterface:
        @staticmethod
        def reset() -> None:
            offload_reset_calls.append(None)

    fake_offload_module = ModuleType(
        "megatron.core.pipeline_parallel.fine_grained_activation_offload"
    )
    fake_offload_module.FineGrainedActivationOffloadingInterface = _OffloadInterface
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.pipeline_parallel.fine_grained_activation_offload",
        fake_offload_module,
    )
    monkeypatch.setattr(helper, "_start_capturing", _start_capturing)
    monkeypatch.setattr(helper, "_get_cuda_graph_input_data", _fail_input_creation)
    monkeypatch.setattr(
        cuda_graphs, "te_set_capture_end", lambda: te_capture_end_calls.append(None)
    )
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(
        cuda_graphs.gc, "unfreeze", lambda: gc_unfreeze_calls.append(None)
    )
    monkeypatch.setattr(cuda_graphs.gc, "collect", lambda: 0)
    monkeypatch.setattr(cuda_graphs.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(cuda_graphs.torch.cuda, "empty_cache", lambda: None)
    bank_module = _load_bank_module()
    manager = bank_module.TECudaGraphBankManager(
        [layer],
        graph_reset_supported=True,
        synchronize=lambda: None,
        runtime_num_microbatches=lambda: 2,
    )

    with pytest.raises(RuntimeError, match="input creation failed"):
        manager.capture(helper, num_microbatches=2)

    assert not cuda_graphs.is_graph_capturing()
    assert not cuda_graphs.is_graph_warmup()
    assert te_capture_end_calls == [None]
    assert offload_reset_calls == [None]
    assert gc_unfreeze_calls == [None]
    assert partial_graph.reset_calls == 1
    assert helper._capture_finished is False
    assert helper._graphs_created is False
    assert layer.cuda_graphs == []


@pytest.mark.parametrize("cleanup_failure", ["synchronize", "reset", "offload"])
def test_abort_cleanup_preserves_original_error_and_clears_state(
    monkeypatch: pytest.MonkeyPatch, cleanup_failure: str
) -> None:
    import megatron.core.transformer.cuda_graphs as cuda_graphs

    class _FailingGraph(_FakeGraph):
        def reset(self) -> None:
            if cleanup_failure == "reset":
                raise RuntimeError("cleanup reset failed")
            super().reset()

    graph = _FailingGraph("partial")
    layer = _FakeLayer("0")
    helper = cuda_graphs.TECudaGraphHelper.__new__(cuda_graphs.TECudaGraphHelper)
    helper.flattened_callables = [layer]
    helper.config = SimpleNamespace(
        sequence_parallel=False,
        fine_grained_activation_offloading=True,
    )
    helper._capture_finished = False
    helper._graphs_created = False
    helper._capture_gc_frozen = True
    monkeypatch.setattr(
        helper,
        "_start_capturing",
        lambda: (cuda_graphs._set_capture_start(), 1.0)[1],
    )

    def _fail_capture():
        layer.cuda_graphs.append(graph)
        raise RuntimeError("original capture failure")

    monkeypatch.setattr(helper, "_get_cuda_graph_input_data", _fail_capture)
    monkeypatch.setattr(cuda_graphs, "is_te_min_version", lambda version: True)
    monkeypatch.setattr(cuda_graphs, "te_set_capture_end", lambda: None)
    monkeypatch.setattr(cuda_graphs.gc, "unfreeze", lambda: None)
    monkeypatch.setattr(cuda_graphs.gc, "collect", lambda: 0)
    monkeypatch.setattr(cuda_graphs.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        cuda_graphs.torch.cuda,
        "synchronize",
        lambda: (
            (_ for _ in ()).throw(RuntimeError("cleanup synchronize failed"))
            if cleanup_failure == "synchronize"
            else None
        ),
    )

    class _OffloadInterface:
        @staticmethod
        def reset() -> None:
            if cleanup_failure == "offload":
                raise RuntimeError("cleanup offload failed")

    fake_offload_module = ModuleType(
        "megatron.core.pipeline_parallel.fine_grained_activation_offload"
    )
    fake_offload_module.FineGrainedActivationOffloadingInterface = _OffloadInterface
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.pipeline_parallel.fine_grained_activation_offload",
        fake_offload_module,
    )

    with pytest.raises(RuntimeError, match="original capture failure"):
        helper._capture_cuda_graph_lists(num_microbatches=2)

    assert layer.cuda_graphs == []
    assert helper._capture_finished is False
    assert helper._graphs_created is False
    assert helper._capture_gc_frozen is False
    assert not cuda_graphs.is_graph_capturing()


@pytest.mark.parametrize(
    ("ep_overlap", "expected_names"),
    [
        (False, (("g0", "g2"), ("g1", "g3"))),
        (True, (("g0", "g1"), ("g2", "g3"))),
    ],
)
def test_two_layer_two_microbatch_mapping(
    ep_overlap: bool, expected_names: tuple[tuple[str, ...], ...]
) -> None:
    from megatron.core.transformer.cuda_graphs import _map_te_graphs_to_layers

    layers = [_FakeLayer("0"), _FakeLayer("1")]
    owned_lists = [[], []]
    graphs = [_FakeGraph(f"g{index}") for index in range(4)]

    _map_te_graphs_to_layers(
        graphs,
        callables_per_chunk=[layers],
        owned_graph_lists=owned_lists,
        num_microbatches=2,
        overlap_moe_expert_parallel_comm=ep_overlap,
    )

    assert tuple(
        tuple(graph.name for graph in graph_list) for graph_list in owned_lists
    ) == (expected_names)
