# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import gc
import importlib
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_replay_module() -> ModuleType:
    module_name = "_standalone_moe_cuda_graph_replay"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "cuda_graph_replay.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REPLAY = _load_replay_module()
sys.modules.setdefault("megatron.core.transformer.moe.cuda_graph_replay", _REPLAY)
AlltoAllCudaGraphState = _REPLAY.AlltoAllCudaGraphState
MoECudaGraphReplayState = _REPLAY.MoECudaGraphReplayState
TensorReplaySignature = _REPLAY.TensorReplaySignature


def _load_router_replay_module() -> ModuleType:
    module_name = "_standalone_router_replay"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "router_replay.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RouterReplayCudaGraphInputSignature = (
    _load_router_replay_module().RouterReplayCudaGraphInputSignature
)


def _load_enums_module() -> ModuleType:
    module_name = "_standalone_transformer_enums"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = (
        Path(__file__).resolve().parents[3] / "megatron" / "core" / "transformer" / "enums.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_bank_module() -> ModuleType:
    cache_name = "_test_te_cuda_graph_bank"
    if cache_name in sys.modules:
        return sys.modules[cache_name]
    try:
        spec = importlib.util.find_spec("megatron.core.transformer.te_cuda_graph_bank")
    except ModuleNotFoundError:
        module_name = "_standalone_te_cuda_graph_bank"
        if module_name in sys.modules:
            return sys.modules[module_name]
        module_path = (
            Path(__file__).resolve().parents[3]
            / "megatron"
            / "core"
            / "transformer"
            / "te_cuda_graph_bank.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    if spec.name == "_standalone_te_cuda_graph_bank":
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[cache_name] = module
        return module
    module = importlib.import_module(spec.name)
    sys.modules[cache_name] = module
    return module


class _FakeGraph:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset_calls = 0

    def __call__(self, *args: object, **kwargs: object) -> str:
        return self.name

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeLayer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cuda_graphs: list[_FakeGraph] = []
        self.cuda_graph_manual_hooks: list[object] = []


class _TransactionalClearLayer(_FakeLayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.bank_reference = object()
        self.fail_clear = False

    def snapshot_te_cuda_graph_bank_references(self) -> object:
        return self.bank_reference

    def restore_te_cuda_graph_bank_references(self, reference: object) -> None:
        self.bank_reference = reference

    def clear_te_cuda_graph_bank_references(self) -> None:
        self.bank_reference = None
        if self.fail_clear:
            raise RuntimeError(f"{self.name} detach failed")


class _FakeTensorStore:
    def __init__(self) -> None:
        self.hidden_states = None
        self.probs = None
        self.routing_map = None
        self.shared_expert_output = None

    def is_empty(self) -> bool:
        return all(value is None for value in vars(self).values())

    def clear(self) -> None:
        for name in vars(self):
            setattr(self, name, None)


class _FakeMoELayer(_FakeLayer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.is_moe_layer = True
        self.dispatcher = SimpleNamespace(
            valid_cudagraph_attrs=["z_output", "nested.a_output"],
            z_output=torch.empty(1),
            nested=SimpleNamespace(a_output=torch.empty(1)),
            handle=None,
            _buffer=None,
        )
        self.mlp = SimpleNamespace(
            token_dispatcher=self.dispatcher,
            cudagraph_tensor_store=_FakeTensorStore(),
            experts=None,
        )
        self._te_cuda_graph_dispatcher_replay_states: tuple[MoECudaGraphReplayState, ...] = ()

    def te_cuda_graph_bank_schema(self) -> tuple[str, ...]:
        return tuple(self.dispatcher.valid_cudagraph_attrs)

    def assert_te_cuda_graph_bank_drained(self) -> None:
        if not self.mlp.cudagraph_tensor_store.is_empty():
            raise RuntimeError("partial MoE continuation is live")

    def clear_te_cuda_graph_bank_references(self) -> None:
        self.mlp.cudagraph_tensor_store.clear()


def _make_alltoall_state(tokens: int) -> MoECudaGraphReplayState:
    signature = TensorReplaySignature(
        shape=torch.Size((tokens, 1, 8)),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layout=torch.strided,
        stride=(8, 8, 1),
    )
    return MoECudaGraphReplayState(
        dispatcher_kind="alltoall",
        input_signature=signature,
        flattened_input_shape=torch.Size((tokens, 8)),
        topology_fingerprint=(("tp_size", 1), ("ep_size", 2), ("router_topk", 2)),
        backend_state=AlltoAllCudaGraphState(
            hidden_shape=signature.shape,
            hidden_shape_before_permute=torch.Size((tokens, 8)),
            capacity=tokens,
            num_out_tokens=tokens,
        ),
    )


class _FakeHelper:
    def __init__(
        self,
        layers: list[_FakeLayer],
        graphs_by_layer: list[list[_FakeGraph]],
        *,
        modules: tuple[object, ...] = ("attn",),
        normalized_count: int | None = None,
        fail_capture: bool = False,
        setup: object | None = None,
    ) -> None:
        self.flattened_callables = layers
        self.config = SimpleNamespace(
            cuda_graph_modules=modules, overlap_moe_expert_parallel_comm=False
        )
        self.num_microbatches = normalized_count
        self._graphs_by_layer = graphs_by_layer
        self._fail_capture = fail_capture
        self._setup = setup
        self._capture_attempted = False
        self._capture_finished = False
        self._graphs_created = False
        self.saw_empty_graph_lists = False

    def _capture_cuda_graph_lists(
        self, *, num_microbatches: int
    ) -> tuple[tuple[_FakeLayer, tuple[_FakeGraph, ...]], ...]:
        self.saw_empty_graph_lists = all(
            layer.cuda_graphs == [] for layer in self.flattened_callables
        )
        assert self.saw_empty_graph_lists
        self.num_microbatches = (
            num_microbatches if self.num_microbatches is None else self.num_microbatches
        )
        if self._fail_capture:
            self.flattened_callables[0].cuda_graphs.append(_FakeGraph("partial"))
            raise RuntimeError("capture failed")
        if callable(self._setup):
            self._setup()
        for layer in self.flattened_callables:
            if getattr(layer, "_te_cuda_graph_dispatcher_replay_states", ()):
                layer._te_cuda_graph_capture_cursor = num_microbatches
        for layer, graphs in zip(self.flattened_callables, self._graphs_by_layer):
            layer.cuda_graphs.extend(graphs)
        self._capture_finished = True
        self._graphs_created = True
        return tuple((layer, tuple(layer.cuda_graphs)) for layer in self.flattened_callables)


def _make_manager(
    layers: list[_FakeLayer],
    *,
    modules: tuple[object, ...] = ("attn",),
    drained=lambda: True,
    synchronize=lambda: None,
    runtime_num_microbatches=lambda: 2,
):
    return _load_bank_module().TECudaGraphBankManager(
        layers,
        cuda_graph_modules=modules,
        assert_model_drained=drained,
        graph_reset_supported=True,
        synchronize=synchronize,
        runtime_num_microbatches=runtime_num_microbatches,
    )


def _graphs(prefix: str, count: int, layers: int = 1) -> list[list[_FakeGraph]]:
    return [
        [_FakeGraph(f"{prefix}-l{layer_index}-g{index}") for index in range(count)]
        for layer_index in range(layers)
    ]


def _install_fake_moe_packed_contract(
    layer: _FakeLayer,
    *,
    sample_ids: torch.Tensor,
    num_samples: torch.Tensor,
    max_samples: int,
    tokens_per_sample: int | None = None,
) -> None:
    tensor_signature = _load_bank_module().tensor_signature
    layer._te_cuda_graph_moe_packed_seq_params_static_metadata = {
        "seq_aux_loss_max_samples": max_samples,
        "tokens_per_sample": tokens_per_sample,
    }
    layer._te_cuda_graph_moe_packed_seq_params_tensor_signatures = {
        "_moe_packed_seq_params_seq_aux_loss_sample_ids": tensor_signature(sample_ids),
        "_moe_packed_seq_params_seq_aux_loss_num_samples": tensor_signature(num_samples),
    }


def _router_replay_signature(tokens: int) -> object:
    return RouterReplayCudaGraphInputSignature(
        shape=(tokens, 2),
        dtype=torch.long,
        device_type="cpu",
        topk=2,
        num_experts=4,
    )


def test_graph_bank_fingerprint_owns_router_replay_input_signature() -> None:
    layer = _FakeLayer("router")
    signature = _router_replay_signature(8)

    def setup() -> None:
        layer._te_cuda_graph_router_replay_input_signature = signature

    manager = _make_manager([layer], modules=("moe_router",))
    bank = manager.capture(
        _FakeHelper([layer], _graphs("router", 2), modules=("moe_router",), setup=setup),
        num_microbatches=2,
    )

    assert bank.fingerprint.router_replay_input_signatures == ((id(layer), signature),)
    assert not hasattr(layer, "_te_cuda_graph_router_replay_input_signature")

    bank.activate()

    assert layer._te_cuda_graph_router_replay_input_signature is signature

    bank.reset()

    assert not hasattr(layer, "_te_cuda_graph_router_replay_input_signature")


def test_activation_records_post_success_graph_identity_and_copy_generation() -> None:
    bank_module = _load_bank_module()
    layer = _FakeLayer("router")
    graph = _FakeGraph("router")
    manager = _make_manager([layer], modules=("moe_router",), runtime_num_microbatches=lambda: 1)
    bank = manager.capture(
        _FakeHelper([layer], [[graph]], modules=("moe_router",)), num_microbatches=1
    )
    bank.activate()

    bank_module._validate_and_record_te_cuda_graph_launch(
        layer, layer.cuda_graphs, 0, record=True
    )
    graph()
    first = bank_module._record_te_cuda_graph_launch_success(layer, layer.cuda_graphs, 0)
    bank_module._validate_and_record_te_cuda_graph_launch(
        layer, layer.cuda_graphs, 0, record=True
    )
    graph()
    second = bank_module._record_te_cuda_graph_launch_success(layer, layer.cuda_graphs, 0)

    assert first.bank_id == second.bank_id == id(bank)
    assert first.graph_index == second.graph_index == 0
    assert second.copy_generation == first.copy_generation + 1
    assert layer._te_cuda_graph_last_launch_record is second

    bank.reset()
    manager.close()


def test_graph_bank_capture_owns_exact_lists_and_contracts() -> None:
    generic = _FakeLayer("attention")
    mamba = _FakeLayer("mamba")
    layers = [generic, mamba]
    old_lists = [[_FakeGraph("old-attn")], [_FakeGraph("old-mamba")]]
    old_hooks = [[object()], [object()]]
    for layer, graph_list, hooks in zip(layers, old_lists, old_hooks):
        layer.cuda_graphs = graph_list
        layer.cuda_graph_manual_hooks = hooks
    mask = torch.empty_strided((3, 5), (7, 1), dtype=torch.bool)
    mamba_seq_idx = torch.empty_strided((15,), (2,), dtype=torch.int32)

    def setup() -> None:
        generic._te_cuda_graph_packed_seq_params_static_metadata = {"qkv_format": "thd"}
        generic._te_cuda_graph_packed_seq_params_tensor_kwarg_names = (
            "_packed_seq_params_cu_seqlens_q",
        )
        generic._te_cuda_graph_padding_mask_signature = _load_bank_module().tensor_signature(mask)
        mamba._te_cuda_graph_mamba_packed_seq_params_static_metadata = {"total_tokens": 15}
        mamba._te_cuda_graph_mamba_packed_seq_params_tensor_signatures = {
            "_mamba_packed_seq_params_seq_idx": _load_bank_module().tensor_signature(mamba_seq_idx)
        }

    manager = _make_manager(layers)
    helper_graphs = _graphs("bank", 2, layers=2)
    bank = manager.capture(_FakeHelper(layers, helper_graphs, setup=setup), num_microbatches=2)

    assert [layer.cuda_graphs for layer in layers] == old_lists
    assert bank.fingerprint.graph_identities == tuple(
        tuple(id(graph) for graph in layer_graphs) for layer_graphs in helper_graphs
    )
    assert bank.fingerprint.padding_mask_signatures == (
        (id(generic), _load_bank_module().tensor_signature(mask)),
        (id(mamba), None),
    )
    generic_packed = dict(bank.fingerprint.packed_input_signatures)[id(generic)]
    mamba_packed = dict(bank.fingerprint.packed_input_signatures)[id(mamba)]
    assert generic_packed[0][0] == "generic"
    assert all(entry[0] != "mamba" for entry in generic_packed)
    assert mamba_packed[0][0] == "mamba"
    assert all(entry[0] != "generic" for entry in mamba_packed)

    bank.activate()
    for index, layer in enumerate(layers):
        assert layer.cuda_graphs is bank._owned_graph_lists[index]
        assert tuple(layer.cuda_graphs) == tuple(helper_graphs[index])
        assert layer.cuda_graph_manual_hooks is old_hooks[index]


def test_graph_bank_fingerprints_and_installs_moe_sample_ownership() -> None:
    layer = _FakeMoELayer("moe")
    sample_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    num_samples = torch.tensor(2, dtype=torch.int64)

    def setup() -> None:
        _install_fake_moe_packed_contract(
            layer,
            sample_ids=sample_ids,
            num_samples=num_samples,
            max_samples=3,
        )

    manager = _make_manager([layer], modules=("moe_router",))
    bank = manager.capture(
        _FakeHelper([layer], _graphs("moe", 2), modules=("moe_router",), setup=setup),
        num_microbatches=2,
    )

    packed_signature = dict(bank.fingerprint.packed_input_signatures)[id(layer)]
    assert packed_signature == (
        (
            "moe",
            (("seq_aux_loss_max_samples", 3), ("tokens_per_sample", None)),
            (
                (
                    "_moe_packed_seq_params_seq_aux_loss_num_samples",
                    _load_bank_module().tensor_signature(num_samples),
                ),
                (
                    "_moe_packed_seq_params_seq_aux_loss_sample_ids",
                    _load_bank_module().tensor_signature(sample_ids),
                ),
            ),
        ),
    )

    bank.activate()

    assert layer._te_cuda_graph_moe_packed_seq_params_static_metadata == {
        "seq_aux_loss_max_samples": 3,
        "tokens_per_sample": None,
    }
    assert layer._te_cuda_graph_moe_packed_seq_params_tensor_signatures[
        "_moe_packed_seq_params_seq_aux_loss_sample_ids"
    ] == _load_bank_module().tensor_signature(sample_ids)

    bank.reset()

    assert not hasattr(layer, "_te_cuda_graph_moe_packed_seq_params_static_metadata")
    assert not hasattr(layer, "_te_cuda_graph_moe_packed_seq_params_tensor_signatures")


def test_failed_activation_restores_padding_dispatcher_and_moe_contract() -> None:
    layer = _FakeMoELayer("moe")
    manager = _make_manager([layer], modules=("moe_router",))
    first_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    second_ids = torch.tensor([0, 1, 2, 2], dtype=torch.int64)

    def setup_first() -> None:
        _install_fake_moe_packed_contract(
            layer,
            sample_ids=first_ids,
            num_samples=torch.tensor(2, dtype=torch.int64),
            max_samples=3,
        )
        layer._te_cuda_graph_padding_mask_signature = _load_bank_module().tensor_signature(
            torch.zeros((1, 4), dtype=torch.bool)
        )
        layer._te_cuda_graph_router_replay_input_signature = _router_replay_signature(4)

    def setup_second() -> None:
        _install_fake_moe_packed_contract(
            layer,
            sample_ids=second_ids,
            num_samples=torch.tensor(3, dtype=torch.int64),
            max_samples=4,
        )
        layer._te_cuda_graph_padding_mask_signature = _load_bank_module().tensor_signature(
            torch.zeros((1, 8), dtype=torch.bool)
        )
        layer._te_cuda_graph_router_replay_input_signature = _router_replay_signature(8)

    first = manager.capture(
        _FakeHelper(
            [layer], _graphs("first", 2), modules=("moe_router",), setup=setup_first
        ),
        num_microbatches=2,
    )
    second = manager.capture(
        _FakeHelper(
            [layer], _graphs("second", 2), modules=("moe_router",), setup=setup_second
        ),
        num_microbatches=2,
    )
    first.activate()
    installation = (
        layer.cuda_graphs,
        layer.cuda_graph_manual_hooks,
        layer._te_cuda_graph_padding_mask_signature,
        layer._te_cuda_graph_router_replay_input_signature,
        layer._te_cuda_graph_dispatcher_replay_states,
        layer._te_cuda_graph_moe_packed_seq_params_static_metadata,
        layer._te_cuda_graph_moe_packed_seq_params_tensor_signatures,
        layer._te_cuda_graph_bank_replay_guard,
    )
    real_install = manager._install_bank

    def fail_install(bank) -> None:
        real_install(bank)
        raise RuntimeError("installation failed")

    manager._install_bank = fail_install
    with pytest.raises(RuntimeError, match="installation failed"):
        second.activate()

    assert manager.active_bank is first
    assert (
        layer.cuda_graphs,
        layer.cuda_graph_manual_hooks,
        layer._te_cuda_graph_padding_mask_signature,
        layer._te_cuda_graph_router_replay_input_signature,
        layer._te_cuda_graph_dispatcher_replay_states,
        layer._te_cuda_graph_moe_packed_seq_params_static_metadata,
        layer._te_cuda_graph_moe_packed_seq_params_tensor_signatures,
        layer._te_cuda_graph_bank_replay_guard,
    ) == installation


def test_graph_bank_activation_restores_dispatcher_states() -> None:
    runtime = {"count": 5}
    layer = _FakeMoELayer("moe")
    manager = _make_manager(
        [layer],
        modules=("moe_router", "moe_preprocess"),
        runtime_num_microbatches=lambda: runtime["count"],
    )
    first_states = tuple(_make_alltoall_state(16) for _ in range(5))
    first = manager.capture(
        _FakeHelper(
            [layer],
            _graphs("first", 5),
            modules=("moe_router", "moe_preprocess"),
            setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", first_states),
        ),
        num_microbatches=5,
    )
    runtime["count"] = 3
    second_states = tuple(_make_alltoall_state(24) for _ in range(3))
    second = manager.capture(
        _FakeHelper(
            [layer],
            _graphs("second", 3),
            modules=("moe_router", "moe_preprocess"),
            setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", second_states),
        ),
        num_microbatches=3,
    )

    runtime["count"] = 5
    first.activate()
    assert layer._te_cuda_graph_dispatcher_replay_states is first_states
    runtime["count"] = 3
    second.activate()
    assert layer._te_cuda_graph_dispatcher_replay_states is second_states
    runtime["count"] = 5
    first.activate()
    assert layer._te_cuda_graph_dispatcher_replay_states is first_states


def test_real_enum_scope_preserves_exact_dispatcher_state_identities() -> None:
    CudaGraphModule = _load_enums_module().CudaGraphModule

    layer = _FakeMoELayer("moe")
    modules = (CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess)
    states = tuple(_make_alltoall_state(16 + index) for index in range(2))
    manager = _make_manager([layer], modules=modules)
    bank = manager.capture(
        _FakeHelper(
            [layer],
            _graphs("enum", 2),
            modules=modules,
            setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", states),
        ),
        num_microbatches=2,
    )

    bank.activate()

    assert bank.fingerprint.cuda_graph_modules == ("moe_preprocess", "moe_router")
    assert layer._te_cuda_graph_dispatcher_replay_states is states
    assert all(
        actual is expected
        for actual, expected in zip(layer._te_cuda_graph_dispatcher_replay_states, states)
    )


def test_activation_rejects_runtime_count_before_mutating_active_bank() -> None:
    runtime = {"count": 5}
    layer = _FakeLayer("layer")
    manager = _make_manager([layer], runtime_num_microbatches=lambda: runtime["count"])
    first = manager.capture(_FakeHelper([layer], _graphs("five", 5)), num_microbatches=5)
    first.activate()
    installed = layer.cuda_graphs
    runtime["count"] = 3

    with pytest.raises(ValueError, match="runtime num_microbatches"):
        first.activate()

    assert manager.active_bank is first
    assert layer.cuda_graphs is installed


def test_capture_rejects_helper_normalized_count_and_restores_previous_bank() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    active = manager.capture(_FakeHelper([layer], _graphs("active", 2)), num_microbatches=2)
    active.activate()
    installed = layer.cuda_graphs
    rejected = _graphs("rejected", 2)

    with pytest.raises(ValueError, match="normalized num_microbatches"):
        manager.capture(_FakeHelper([layer], rejected, normalized_count=3), num_microbatches=2)

    assert manager.active_bank is active
    assert layer.cuda_graphs is installed
    assert [graph.reset_calls for graph in rejected[0]] == [1, 1]


def test_capture_failure_restores_every_active_contract() -> None:
    layer = _FakeMoELayer("moe")
    states = tuple(_make_alltoall_state(8) for _ in range(2))
    manager = _make_manager([layer], modules=("moe_router", "moe_preprocess"))

    def setup() -> None:
        layer._te_cuda_graph_dispatcher_replay_states = states
        layer._te_cuda_graph_router_replay_input_signature = _router_replay_signature(8)

    active = manager.capture(
        _FakeHelper(
            [layer],
            _graphs("active", 2),
            modules=("moe_router", "moe_preprocess"),
            setup=setup,
        ),
        num_microbatches=2,
    )
    active.activate()
    installation = (
        layer.cuda_graphs,
        layer.cuda_graph_manual_hooks,
        layer._te_cuda_graph_router_replay_input_signature,
        layer._te_cuda_graph_dispatcher_replay_states,
        layer._te_cuda_graph_bank_replay_guard,
    )
    failing = _FakeHelper(
        [layer], _graphs("unused", 2), modules=("moe_router", "moe_preprocess"), fail_capture=True
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        manager.capture(failing, num_microbatches=2)

    assert manager.active_bank is active
    assert (
        layer.cuda_graphs,
        layer.cuda_graph_manual_hooks,
        layer._te_cuda_graph_router_replay_input_signature,
        layer._te_cuda_graph_dispatcher_replay_states,
        layer._te_cuda_graph_bank_replay_guard,
    ) == installation
    assert failing._capture_attempted
    with pytest.raises(ValueError, match="one-shot"):
        manager.capture(failing, num_microbatches=2)


def test_activation_failure_rolls_back_active_bank_transaction() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    active = manager.capture(_FakeHelper([layer], _graphs("active", 2)), num_microbatches=2)
    inactive = manager.capture(_FakeHelper([layer], _graphs("inactive", 2)), num_microbatches=2)
    active.activate()
    installed = (
        layer.cuda_graphs,
        layer.cuda_graph_manual_hooks,
        layer._te_cuda_graph_bank_replay_guard,
    )
    real_install = manager._install_bank

    def fail_install(bank) -> None:
        real_install(bank)
        raise RuntimeError("installation failed")

    manager._install_bank = fail_install
    with pytest.raises(RuntimeError, match="installation failed"):
        inactive.activate()

    assert manager.active_bank is active
    assert (
        layer.cuda_graphs,
        layer.cuda_graph_manual_hooks,
        layer._te_cuda_graph_bank_replay_guard,
    ) == installed


def test_refresh_manual_hooks_updates_bank_identity() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    bank = manager.capture(_FakeHelper([layer], _graphs("bank", 2)), num_microbatches=2)
    bank.activate()
    refreshed = [(object(), (layer,))]
    layer.cuda_graph_manual_hooks = refreshed

    manager.refresh_manual_hooks(bank)
    layer.cuda_graph_manual_hooks = []
    bank.activate()

    assert layer.cuda_graph_manual_hooks is refreshed


def test_reset_is_inactive_safe_idempotent_unique_and_releases_references() -> None:
    layer = _FakeLayer("layer")
    shared = _FakeGraph("shared")
    manager = _make_manager([layer])
    active = manager.capture(
        _FakeHelper([layer], [[shared, _FakeGraph("active")]]), num_microbatches=2
    )
    active.activate()
    inactive_graph = _FakeGraph("inactive")
    reference = weakref.ref(inactive_graph)
    helper = _FakeHelper([layer], [[inactive_graph, inactive_graph]])
    inactive = manager.capture(helper, num_microbatches=2)
    installed = layer.cuda_graphs
    del helper
    del inactive_graph

    inactive.reset()
    inactive.reset()
    gc.collect()

    assert manager.active_bank is active
    assert layer.cuda_graphs is installed
    assert shared.reset_calls == 0
    assert reference() is None
    assert inactive.graphs_by_layer == ()


@pytest.mark.parametrize("operation", ["uninstall", "reset"])
def test_active_detach_failure_rolls_back_exact_installation_and_allows_retry(
    operation: str,
) -> None:
    first = _TransactionalClearLayer("first")
    second = _TransactionalClearLayer("second")
    manager = _make_manager([first, second])
    graph_lists = _graphs("bank", 2, layers=2)
    bank = manager.capture(_FakeHelper([first, second], graph_lists), num_microbatches=2)
    bank.activate()
    installations = tuple(
        (
            layer.cuda_graphs,
            layer.cuda_graph_manual_hooks,
            layer._te_cuda_graph_bank_replay_guard,
            layer.bank_reference,
        )
        for layer in (first, second)
    )
    second.fail_clear = True

    with pytest.raises(RuntimeError, match="second detach failed"):
        bank.reset() if operation == "reset" else manager.uninstall(bank)

    assert manager.active_bank is bank
    assert manager.registered_bank_count == 1
    assert (
        tuple(
            (
                layer.cuda_graphs,
                layer.cuda_graph_manual_hooks,
                layer._te_cuda_graph_bank_replay_guard,
                layer.bank_reference,
            )
            for layer in (first, second)
        )
        == installations
    )

    second.fail_clear = False
    if operation == "reset":
        bank.reset()
        assert manager.active_bank is None
        assert manager.registered_bank_count == 0
        assert bank.graphs_by_layer == ()
        assert all(graph.reset_calls == 1 for graphs in graph_lists for graph in graphs)
    else:
        manager.uninstall(bank)
        assert manager.active_bank is None
        assert manager.registered_bank_count == 1
        bank.activate()
        assert manager.active_bank is bank


@pytest.mark.parametrize("operation", ["capture", "activate", "reset", "uninstall"])
def test_live_work_blocks_every_graph_bank_transition(operation: str) -> None:
    drain = {"ready": True}
    layer = _FakeLayer("layer")
    manager = _make_manager([layer], drained=lambda: drain["ready"])
    bank = manager.capture(_FakeHelper([layer], _graphs("bank", 2)), num_microbatches=2)
    if operation == "uninstall":
        bank.activate()
    drain["ready"] = False

    with pytest.raises(RuntimeError, match="not drained"):
        if operation == "capture":
            manager.capture(_FakeHelper([layer], _graphs("new", 2)), num_microbatches=2)
        elif operation == "activate":
            bank.activate()
        elif operation == "reset":
            bank.reset()
        else:
            manager.uninstall(bank)


def test_layer_partial_moe_continuation_blocks_transition_after_cuda_sync() -> None:
    events: list[str] = []
    layer = _FakeMoELayer("moe")
    manager = _make_manager(
        [layer],
        modules=("moe_router", "moe_preprocess"),
        drained=lambda: events.append("drained") or True,
        synchronize=lambda: events.append("synchronized"),
    )
    states = tuple(_make_alltoall_state(8) for _ in range(2))
    bank = manager.capture(
        _FakeHelper(
            [layer],
            _graphs("bank", 2),
            modules=("moe_router", "moe_preprocess"),
            setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", states),
        ),
        num_microbatches=2,
    )
    layer.mlp.cudagraph_tensor_store.probs = torch.empty(1)
    events.clear()

    with pytest.raises(RuntimeError, match="partial MoE"):
        bank.activate()

    assert events == ["drained", "synchronized"]


def test_replay_guard_checks_exact_selected_callable_in_constant_work() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    bank = manager.capture(_FakeHelper([layer], _graphs("bank", 2)), num_microbatches=2)
    bank.activate()
    manager._validate_bank = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("full validation reached hot path")
    )

    assert layer._te_cuda_graph_bank_replay_guard(layer, layer.cuda_graphs, 3) == 1
    layer.cuda_graphs[1] = _FakeGraph("forged")
    with pytest.raises(ValueError, match="selected CUDA graph callable"):
        layer._te_cuda_graph_bank_replay_guard(layer, layer.cuda_graphs, 3)


def test_activation_rejects_mutated_exact_fingerprint_fields() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    bank = manager.capture(_FakeHelper([layer], _graphs("bank", 2)), num_microbatches=2)

    for field, replacement in (
        ("layer_ids", (123,)),
        ("graph_identities", ((123, 456),)),
        ("graph_counts", (3,)),
        ("cuda_graph_modules", ("mamba",)),
        ("packed_input_signatures", ((123, ()),)),
        ("padding_mask_signatures", ((123, None),)),
        ("moe_attribute_schema", ((123, ("x",)),)),
        ("dispatcher_state_signatures", ((123, (None, None)),)),
    ):
        original = bank.fingerprint
        bank.fingerprint = dataclasses.replace(original, **{field: replacement})
        with pytest.raises(ValueError, match=field):
            bank.activate()
        bank.fingerprint = original


def test_dispatcher_state_count_must_match_graph_identity_count() -> None:
    layer = _FakeMoELayer("moe")
    states = (_make_alltoall_state(8),)
    manager = _make_manager([layer], modules=("moe_router", "moe_preprocess"))

    with pytest.raises(ValueError, match="dispatcher state count"):
        manager.capture(
            _FakeHelper(
                [layer],
                _graphs("bank", 2),
                modules=("moe_router", "moe_preprocess"),
                setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", states),
            ),
            num_microbatches=2,
        )


def test_moe_schema_is_ordered_and_requires_tensor_leaves() -> None:
    layer = _FakeMoELayer("moe")
    states = tuple(_make_alltoall_state(8) for _ in range(2))
    manager = _make_manager([layer], modules=("moe_router", "moe_preprocess"))
    bank = manager.capture(
        _FakeHelper(
            [layer],
            _graphs("bank", 2),
            modules=("moe_router", "moe_preprocess"),
            setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", states),
        ),
        num_microbatches=2,
    )
    assert bank.fingerprint.moe_attribute_schema == ((id(layer), ("z_output", "nested.a_output")),)
    layer.dispatcher.valid_cudagraph_attrs.reverse()
    with pytest.raises(ValueError, match="moe_attribute_schema"):
        bank.activate()

    layer.dispatcher.valid_cudagraph_attrs.reverse()
    layer.dispatcher.z_output = "not a Tensor"
    with pytest.raises(ValueError, match="Tensor"):
        manager.capture(
            _FakeHelper(
                [layer],
                _graphs("invalid", 2),
                modules=("moe_router", "moe_preprocess"),
                setup=lambda: setattr(layer, "_te_cuda_graph_dispatcher_replay_states", states),
            ),
            num_microbatches=2,
        )


def test_execution_counter_snapshot_delta_and_owner_validation() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    initial = manager.snapshot_execution_counters()
    tracker = layer._te_cuda_graph_execution_counter

    tracker.record_eligible_call()
    tracker.record_eligible_call()
    tracker.record_graph_call()
    current = manager.snapshot_execution_counters()

    assert (initial.eligible_calls, initial.graph_calls) == (0, 0)
    assert (current.eligible_calls, current.graph_calls) == (2, 1)
    delta = manager.execution_counter_delta(initial)
    assert (delta.eligible_calls, delta.graph_calls) == (2, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        initial.eligible_calls = 1

    other_layer = _FakeLayer("other")
    other_manager = _make_manager([other_layer])
    assert other_manager.snapshot_execution_counters() != initial
    with pytest.raises(TypeError, match="invalid type"):
        manager.execution_counter_delta(object())
    with pytest.raises(ValueError, match="different TECudaGraphBankManager"):
        other_manager.execution_counter_delta(initial)
    with pytest.raises(ValueError, match="monotonic"):
        manager.execution_counter_delta(current, initial)

    other_manager.close()
    manager.close()


def test_execution_counter_endpoints_ignore_instance_snapshot_override() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    tracker = layer._te_cuda_graph_execution_counter
    start = manager.snapshot_execution_counters()

    tracker.record_eligible_call()
    tracker.record_graph_call()
    tracker.snapshot = lambda: dataclasses.replace(start, eligible_calls=777, graph_calls=888)

    current = manager.snapshot_execution_counters()
    delta = manager.execution_counter_delta(start)
    assert (current.eligible_calls, current.graph_calls) == (1, 1)
    assert (delta.eligible_calls, delta.graph_calls) == (1, 1)

    manager.close()


def test_execution_counter_owner_collision_and_identity_safe_close() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    tracker = layer._te_cuda_graph_execution_counter

    with pytest.raises(ValueError, match="already owned"):
        _make_manager([layer])
    assert layer._te_cuda_graph_execution_counter is tracker

    foreign_tracker = object()
    layer._te_cuda_graph_execution_counter = foreign_tracker
    with pytest.raises(ValueError, match="ownership changed"):
        manager.close()
    assert layer._te_cuda_graph_execution_counter is foreign_tracker

    layer._te_cuda_graph_execution_counter = tracker
    manager.close()
    assert not hasattr(layer, "_te_cuda_graph_execution_counter")

    replacement = _make_manager([layer])
    replacement_tracker = layer._te_cuda_graph_execution_counter
    manager.close()
    assert layer._te_cuda_graph_execution_counter is replacement_tracker
    replacement.close()


def test_execution_counter_dead_owner_does_not_block_a_new_manager() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    manager_reference = weakref.ref(manager)
    stale_tracker = layer._te_cuda_graph_execution_counter

    del manager
    gc.collect()
    assert manager_reference() is None

    replacement = _make_manager([layer])
    assert layer._te_cuda_graph_execution_counter is not stale_tracker
    replacement.close()


def test_execution_counter_foreign_dead_owner_spoof_is_rejected() -> None:
    layer = _FakeLayer("layer")
    foreign_tracker = SimpleNamespace(_owner_ref=lambda: None)
    layer._te_cuda_graph_execution_counter = foreign_tracker

    with pytest.raises(ValueError, match="already owned"):
        _make_manager([layer])

    assert layer._te_cuda_graph_execution_counter is foreign_tracker


@pytest.mark.parametrize("mutation", ["delete", "replace"])
@pytest.mark.parametrize("endpoint", ["snapshot", "current_delta", "explicit_delta"])
def test_execution_counter_endpoints_reject_changed_layer_attachment(
    mutation: str, endpoint: str
) -> None:
    first = _FakeLayer("first")
    second = _FakeLayer("second")
    manager = _make_manager([first, second])
    start = manager.snapshot_execution_counters()
    tracker = second._te_cuda_graph_execution_counter
    tracker.record_eligible_call()
    end = manager.snapshot_execution_counters()

    if mutation == "delete":
        del second._te_cuda_graph_execution_counter
    else:
        second._te_cuda_graph_execution_counter = object()

    with pytest.raises(ValueError, match="ownership changed"):
        if endpoint == "snapshot":
            manager.snapshot_execution_counters()
        elif endpoint == "current_delta":
            manager.execution_counter_delta(start)
        else:
            manager.execution_counter_delta(start, end)

    second._te_cuda_graph_execution_counter = tracker
    manager.close()


def test_execution_counters_survive_uninstall_reset_and_eviction() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    tracker = layer._te_cuda_graph_execution_counter
    tracker.record_eligible_call()
    tracker.record_graph_call()
    before_transitions = manager.snapshot_execution_counters()

    first = manager.capture(_FakeHelper([layer], _graphs("first", 2)), num_microbatches=2)
    second = manager.capture(_FakeHelper([layer], _graphs("second", 2)), num_microbatches=2)
    first.activate()
    manager.uninstall(first)
    second.activate()
    first.reset()  # An outer LRU eviction resets an inactive bank.
    manager.uninstall(second)
    second.reset()

    after_transitions = manager.snapshot_execution_counters()
    assert after_transitions == before_transitions
    tracker.record_eligible_call()
    assert manager.execution_counter_delta(before_transitions).eligible_calls == 1
    manager.close()


def test_execution_counter_close_rejects_live_banks() -> None:
    layer = _FakeLayer("layer")
    manager = _make_manager([layer])
    bank = manager.capture(_FakeHelper([layer], _graphs("bank", 2)), num_microbatches=2)

    with pytest.raises(RuntimeError, match="registered TE CUDA graph banks"):
        manager.close()

    bank.reset()
    manager.close()
