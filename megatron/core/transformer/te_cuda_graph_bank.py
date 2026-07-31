# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transactional ownership for schedule-specific Transformer Engine CUDA graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Iterable, Sequence
from weakref import WeakSet, ref

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.moe.cuda_graph_replay import (
        MoECudaGraphReplayState,
        TensorReplaySignature,
    )


_PACKED_REPLAY_ATTRIBUTES = (
    "_te_cuda_graph_packed_seq_params_static_metadata",
    "_te_cuda_graph_packed_seq_params_tensor_kwarg_names",
    "_te_cuda_graph_mamba_packed_seq_params_static_metadata",
    "_te_cuda_graph_mamba_packed_seq_params_tensor_signatures",
)
_PADDING_MASK_SIGNATURE_ATTRIBUTE = "_te_cuda_graph_padding_mask_signature"
_DISPATCHER_STATES_ATTRIBUTE = "_te_cuda_graph_dispatcher_replay_states"
_EXECUTION_COUNTER_ATTRIBUTE = "_te_cuda_graph_execution_counter"


def _freeze_signature(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_signature(item))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_signature(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _normalize_cuda_graph_modules(modules: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        sorted(module.name if isinstance(module, Enum) else str(module) for module in modules)
    )


def tensor_signature(tensor: torch.Tensor) -> TensorReplaySignature:
    """Return Task 5's exact Tensor replay surface."""

    from megatron.core.transformer.moe.cuda_graph_replay import get_tensor_replay_signature

    return get_tensor_replay_signature(tensor)


@dataclass(frozen=True)
class TECudaGraphBankFingerprint:
    """Immutable capture provenance for a TE CUDA graph bank."""

    num_microbatches: int
    layer_ids: tuple[int, ...]
    graph_identities: tuple[tuple[int, ...], ...]
    graph_counts: tuple[int, ...]
    cuda_graph_modules: tuple[str, ...]
    packed_input_signatures: tuple[tuple[int, tuple[object, ...]], ...]
    padding_mask_signatures: tuple[tuple[int, TensorReplaySignature | None], ...]
    moe_attribute_schema: tuple[tuple[int, tuple[str, ...]], ...]
    dispatcher_state_signatures: tuple[tuple[int, tuple[MoECudaGraphReplayState | None, ...]], ...]


@dataclass(frozen=True)
class TECudaGraphExecutionCounterSnapshot:
    """Immutable execution-counter values owned by one graph-bank manager."""

    eligible_calls: int
    graph_calls: int
    _owner: object = field(repr=False)


class _TECudaGraphExecutionCounter:
    def __init__(self, owner: TECudaGraphBankManager) -> None:
        self._owner_ref = ref(owner)
        self.eligible_calls = 0
        self.graph_calls = 0

    def record_eligible_call(self) -> None:
        self.eligible_calls += 1

    def record_graph_call(self) -> None:
        self.graph_calls += 1

    def snapshot(self) -> TECudaGraphExecutionCounterSnapshot:
        return TECudaGraphExecutionCounterSnapshot(
            eligible_calls=self.eligible_calls, graph_calls=self.graph_calls, _owner=self
        )


@dataclass(frozen=True)
class _LayerReplayContract:
    manual_hooks: object
    packed_attributes: tuple[tuple[str, bool, object], ...]
    padding_mask_signature: TensorReplaySignature | None
    dispatcher_states: tuple[MoECudaGraphReplayState | None, ...]


@dataclass(frozen=True)
class _LayerInstallation:
    graph_list: object
    manual_hooks: object
    packed_attributes: tuple[tuple[str, bool, object], ...]
    padding_mask_signature_present: bool
    padding_mask_signature: object
    dispatcher_states_present: bool
    dispatcher_states: object
    replay_guard_present: bool
    replay_guard: object
    bank_references_supported: bool
    bank_references: object


@dataclass(frozen=True)
class _BankRegistration:
    bank: TECudaGraphBank
    fingerprint: TECudaGraphBankFingerprint
    layer_ids: tuple[int, ...]
    layer_index_by_id: dict[int, int]
    owned_graph_lists: tuple[list[object], ...]
    graph_tuples: tuple[tuple[object, ...], ...]
    contracts: tuple[_LayerReplayContract, ...]
    replay_guard: _BankReplayGuard
    reset_graph_ids: set[int]


class _BankReplayGuard:
    def __init__(self, manager: TECudaGraphBankManager, bank: TECudaGraphBank) -> None:
        self._manager = manager
        self._bank = bank

    def __call__(
        self, layer: object, installed_graph_list: list[object], microbatch_index: int
    ) -> int:
        return self._manager._assert_replay_ready(
            self._bank, layer, installed_graph_list, microbatch_index
        )

    def record_graph_call(
        self,
        layer: object,
        installed_graph_list: list[object],
        selected_index: int,
        counter: object,
    ) -> None:
        """Record one launch after revalidating this guard's selected callable."""

        self._manager._record_graph_call(
            self._bank, self, layer, installed_graph_list, selected_index, counter
        )


@dataclass(eq=False)
class TECudaGraphBank:
    """One terminally-owned set of TE graph callables for a schedule geometry."""

    fingerprint: TECudaGraphBankFingerprint
    graphs_by_layer: tuple[tuple[object, tuple[object, ...]], ...]
    _manager: TECudaGraphBankManager = field(repr=False)
    _owned_graph_lists: tuple[list[object], ...] = field(repr=False)
    _layer_contracts: tuple[_LayerReplayContract, ...] = field(repr=False)

    def activate(self) -> None:
        """Atomically install this bank when its runtime schedule is active."""

        self._manager.activate(self)

    def reset(self) -> None:
        """Terminally release and, when supported, reset this bank's callables."""

        self._manager.reset(self)


class TECudaGraphBankManager:
    """Own and transactionally switch TE graph banks for one ordered layer topology."""

    def __init__(
        self,
        layers: Sequence[object],
        *,
        cuda_graph_modules: Iterable[object] = (),
        assert_model_drained: Callable[[], object] | None = None,
        graph_reset_supported: bool | None = None,
        synchronize: Callable[[], None] | None = None,
        runtime_num_microbatches: Callable[[], int],
    ) -> None:
        self.layers = tuple(layers)
        self._layer_ids = tuple(id(layer) for layer in self.layers)
        self._cuda_graph_modules = _normalize_cuda_graph_modules(cuda_graph_modules)
        self._assert_model_drained_callback = assert_model_drained or (lambda: True)
        if graph_reset_supported is None:
            from megatron.core.utils import is_te_min_version

            graph_reset_supported = is_te_min_version("2.10.0")
        self._graph_reset_supported = graph_reset_supported
        self._synchronize = synchronize or torch.cuda.synchronize
        if not callable(runtime_num_microbatches):
            raise TypeError("runtime_num_microbatches must be a callable provider")
        self._runtime_num_microbatches = runtime_num_microbatches
        self._registrations: dict[int, _BankRegistration] = {}
        self._terminal_banks: WeakSet[TECudaGraphBank] = WeakSet()
        self.active_bank: TECudaGraphBank | None = None
        self._execution_counter = _TECudaGraphExecutionCounter(self)
        self._execution_counter_closed = False
        self._attach_execution_counter()

    @property
    def registered_bank_count(self) -> int:
        """Return the number of live bank registrations."""

        return len(self._registrations)

    def snapshot_execution_counters(self) -> TECudaGraphExecutionCounterSnapshot:
        """Return the current monotonic execution counters."""

        self._assert_execution_counter_open()
        return self._execution_counter.snapshot()

    def execution_counter_delta(
        self,
        start: TECudaGraphExecutionCounterSnapshot,
        end: TECudaGraphExecutionCounterSnapshot | None = None,
    ) -> TECudaGraphExecutionCounterSnapshot:
        """Return the non-negative counter delta between two owned snapshots."""

        self._assert_execution_counter_open()
        self._validate_execution_counter_snapshot(start)
        if end is None:
            end = self._execution_counter.snapshot()
        else:
            self._validate_execution_counter_snapshot(end)
        eligible_calls = end.eligible_calls - start.eligible_calls
        graph_calls = end.graph_calls - start.graph_calls
        if eligible_calls < 0 or graph_calls < 0:
            raise ValueError("TE CUDA graph execution counters must be monotonic")
        return TECudaGraphExecutionCounterSnapshot(
            eligible_calls=eligible_calls, graph_calls=graph_calls, _owner=self._execution_counter
        )

    def close(self) -> None:
        """Detach this manager's execution tracker after every bank is released."""

        if self._execution_counter_closed:
            return
        if self.active_bank is not None or self._registrations:
            raise RuntimeError(
                "Cannot close execution counters while registered TE CUDA graph banks remain"
            )
        layers = self._unique_layers()
        if any(
            getattr(layer, _EXECUTION_COUNTER_ATTRIBUTE, None) is not self._execution_counter
            for layer in layers
        ):
            raise ValueError("TE CUDA graph execution counter ownership changed before close")
        for layer in layers:
            delattr(layer, _EXECUTION_COUNTER_ATTRIBUTE)
        self._execution_counter_closed = True

    @classmethod
    def from_helper(
        cls, helper: object, *, assert_model_drained: Callable[[], object] | None = None
    ) -> TECudaGraphBankManager:
        """Construct a manager from an uncaptured helper's fixed topology."""

        pp_size = helper.pp_group.size()
        overlap = helper.config.overlap_moe_expert_parallel_comm

        def runtime_num_microbatches() -> int:
            if pp_size == 1 and not overlap:
                return 1
            from megatron.core.num_microbatches_calculator import get_num_microbatches

            return get_num_microbatches()

        return cls(
            helper.flattened_callables,
            cuda_graph_modules=getattr(helper.config, "cuda_graph_modules", ()),
            assert_model_drained=assert_model_drained,
            runtime_num_microbatches=runtime_num_microbatches,
        )

    def capture(self, helper: object, *, num_microbatches: int) -> TECudaGraphBank:
        """Consume one helper and register a bank without disturbing the active bank."""

        self._assert_execution_counter_open()
        self._assert_model_drained()
        self._validate_helper(helper)
        if num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")
        if self._runtime_num_microbatches() != num_microbatches:
            raise ValueError("runtime num_microbatches does not match the requested capture")

        previous_installations = self._snapshot_installations()
        previous_active_bank = self.active_bank
        owned_graph_lists = tuple([] for _ in self.layers)
        capture_index_state = tuple(
            (
                hasattr(layer, "_te_cuda_graph_capture_num_microbatches"),
                getattr(layer, "_te_cuda_graph_capture_num_microbatches", None),
                hasattr(layer, "_te_cuda_graph_capture_cursor"),
                getattr(layer, "_te_cuda_graph_capture_cursor", None),
            )
            for layer in self.layers
        )
        helper._capture_attempted = True
        try:
            for layer, graph_list, installation in zip(
                self.layers, owned_graph_lists, previous_installations
            ):
                self._clear_replay_contract(layer)
                if hasattr(layer, "_te_cuda_graph_bank_replay_guard"):
                    delattr(layer, "_te_cuda_graph_bank_replay_guard")
                layer.cuda_graph_manual_hooks = installation.manual_hooks
                layer.cuda_graphs = graph_list
                layer._te_cuda_graph_capture_num_microbatches = num_microbatches
                layer._te_cuda_graph_capture_cursor = 0

            captured_pairs = helper._capture_cuda_graph_lists(num_microbatches=num_microbatches)
            if getattr(helper, "num_microbatches", None) != num_microbatches:
                raise ValueError(
                    "helper normalized num_microbatches does not match the requested capture"
                )
            self._validate_capture_result(captured_pairs, owned_graph_lists, num_microbatches)
            contracts = tuple(
                self._snapshot_contract(
                    layer, len(graph_list), manual_hooks=installation.manual_hooks
                )
                for layer, graph_list, installation in zip(
                    self.layers, owned_graph_lists, previous_installations
                )
            )
            graph_tuples = tuple(tuple(graph_list) for graph_list in owned_graph_lists)
            fingerprint = TECudaGraphBankFingerprint(
                num_microbatches=num_microbatches,
                layer_ids=self._layer_ids,
                graph_identities=tuple(
                    tuple(id(graph) for graph in graph_tuple) for graph_tuple in graph_tuples
                ),
                graph_counts=tuple(len(graph_tuple) for graph_tuple in graph_tuples),
                cuda_graph_modules=self._cuda_graph_modules,
                packed_input_signatures=self._packed_input_signatures(contracts),
                padding_mask_signatures=tuple(
                    (id(layer), contract.padding_mask_signature)
                    for layer, contract in zip(self.layers, contracts)
                ),
                moe_attribute_schema=self._moe_attribute_schema(require_tensors=True),
                dispatcher_state_signatures=tuple(
                    (id(layer), contract.dispatcher_states)
                    for layer, contract in zip(self.layers, contracts)
                ),
            )
            graphs_by_layer = tuple(
                (layer, graph_tuple) for layer, graph_tuple in zip(self.layers, graph_tuples)
            )
            bank = TECudaGraphBank(
                fingerprint=fingerprint,
                graphs_by_layer=graphs_by_layer,
                _manager=self,
                _owned_graph_lists=owned_graph_lists,
                _layer_contracts=contracts,
            )
            self._validate_bank(bank, require_registered=False)
            self._register_bank(bank)
            return bank
        except BaseException:
            if hasattr(helper, "_capture_finished"):
                helper._capture_finished = False
            if hasattr(helper, "_graphs_created"):
                helper._graphs_created = False
            try:
                self._synchronize()
            except Exception:
                pass
            else:
                try:
                    self._reset_graph_identities(
                        owned_graph_lists, already_reset=self._live_graph_ids()
                    )
                except Exception:
                    pass
            raise
        finally:
            for layer, state in zip(self.layers, capture_index_state):
                count_present, count, cursor_present, cursor = state
                self._restore_optional_attribute(
                    layer, "_te_cuda_graph_capture_num_microbatches", count_present, count
                )
                self._restore_optional_attribute(
                    layer, "_te_cuda_graph_capture_cursor", cursor_present, cursor
                )
            self._restore_installations(previous_installations)
            self.active_bank = previous_active_bank

    def activate(self, bank: TECudaGraphBank) -> None:
        """Install all replay surfaces for ``bank`` as one rollback-safe transaction."""

        self._assert_model_drained()
        self._validate_bank(bank)
        runtime_count = self._runtime_num_microbatches()
        if bank.fingerprint.num_microbatches != runtime_count:
            raise ValueError(
                "runtime num_microbatches does not match the captured TE CUDA graph bank"
            )
        previous_installations = self._snapshot_installations()
        previous_active_bank = self.active_bank
        try:
            if previous_active_bank is not None and previous_active_bank is not bank:
                self._clear_installed_bank(previous_active_bank)
            self._install_bank(bank)
            self.active_bank = bank
        except BaseException:
            self._restore_installations(previous_installations)
            self.active_bank = previous_active_bank
            raise

    def uninstall(self, bank: TECudaGraphBank | None = None) -> None:
        """Detach the active bank without resetting its graph callables."""

        target = self.active_bank if bank is None else bank
        if target is None:
            return
        self._assert_model_drained()
        self._validate_bank(target)
        if target is not self.active_bank:
            raise ValueError("Only the active TE CUDA graph bank can be uninstalled")
        previous_installations = self._snapshot_installations()
        try:
            self._clear_installed_bank(target)
        except BaseException:
            self._restore_installations(previous_installations)
            raise
        self.active_bank = None

    def reset(self, bank: TECudaGraphBank) -> None:
        """Reset each unique callable owned by ``bank`` once and unregister it."""

        if bank in self._terminal_banks:
            return
        self._assert_model_drained()
        self._validate_bank(bank)
        registration = self._registrations[id(bank)]
        if self.active_bank is bank:
            previous_installations = self._snapshot_installations()
            try:
                self._clear_installed_bank(bank)
            except BaseException:
                self._restore_installations(previous_installations)
                raise
            self.active_bank = None
        try:
            self._synchronize()
            self._reset_graph_identities(
                registration.graph_tuples, already_reset=registration.reset_graph_ids
            )
        finally:
            self._registrations.pop(id(bank), None)
            self._terminal_banks.add(bank)
            bank.graphs_by_layer = ()
            bank._owned_graph_lists = ()
            bank._layer_contracts = ()

    def refresh_manual_hooks(self, bank: TECudaGraphBank) -> None:
        """Refresh exact hook-list identities after legacy hook setup."""

        self._validate_bank(bank)
        if bank is not self.active_bank or not self._bank_is_installed(bank):
            raise ValueError("Manual hooks can only be refreshed for the active bank")
        registration = self._registrations[id(bank)]
        contracts = tuple(
            _LayerReplayContract(
                manual_hooks=layer.cuda_graph_manual_hooks,
                packed_attributes=contract.packed_attributes,
                padding_mask_signature=contract.padding_mask_signature,
                dispatcher_states=contract.dispatcher_states,
            )
            for layer, contract in zip(self.layers, registration.contracts)
        )
        bank._layer_contracts = contracts
        self._registrations[id(bank)] = _BankRegistration(
            bank=bank,
            fingerprint=registration.fingerprint,
            layer_ids=registration.layer_ids,
            layer_index_by_id=registration.layer_index_by_id,
            owned_graph_lists=registration.owned_graph_lists,
            graph_tuples=registration.graph_tuples,
            contracts=contracts,
            replay_guard=registration.replay_guard,
            reset_graph_ids=registration.reset_graph_ids,
        )

    def get_graph(
        self, bank: TECudaGraphBank, layer: object, *, microbatch_index: int, num_microbatches: int
    ) -> object:
        """Return the exact registered callable after constant-work replay checks."""

        if self._runtime_num_microbatches() != num_microbatches:
            raise ValueError("supplied num_microbatches differs from the runtime provider")
        index = self._assert_replay_ready(
            bank, layer, getattr(layer, "cuda_graphs", None), microbatch_index
        )
        return layer.cuda_graphs[index]

    def _assert_replay_ready(
        self,
        bank: TECudaGraphBank,
        layer: object,
        installed_graph_list: list[object],
        microbatch_index: int,
    ) -> int:
        if bank in self._terminal_banks:
            raise ValueError("TE CUDA graph bank has already been reset")
        if bank is not self.active_bank:
            raise ValueError("TE CUDA graph bank is not active")
        registration = self._registrations.get(id(bank))
        if (
            registration is None
            or registration.bank is not bank
            or bank.fingerprint is not registration.fingerprint
        ):
            raise ValueError("TE CUDA graph bank registration is missing or forged")
        layer_index = registration.layer_index_by_id.get(id(layer))
        if (
            layer_index is None
            or layer_index >= len(self.layers)
            or self.layers[layer_index] is not layer
        ):
            raise ValueError("Layer is not registered with the active bank")
        canonical_list = registration.owned_graph_lists[layer_index]
        if installed_graph_list is not canonical_list:
            raise ValueError("Layer CUDA graph list does not match its bank registration")
        runtime_count = self._runtime_num_microbatches()
        if (
            runtime_count != registration.fingerprint.num_microbatches
            or len(canonical_list) != runtime_count
            or registration.fingerprint.graph_counts[layer_index] != runtime_count
        ):
            raise ValueError(
                "runtime num_microbatches or graph count does not match the active TE CUDA graph bank"
            )
        selected_index = microbatch_index % runtime_count
        if (
            canonical_list[selected_index]
            is not registration.graph_tuples[layer_index][selected_index]
        ):
            raise ValueError(
                "selected CUDA graph callable does not match its canonical registration"
            )
        return selected_index

    def _record_graph_call(
        self,
        bank: TECudaGraphBank,
        replay_guard: _BankReplayGuard,
        layer: object,
        installed_graph_list: list[object],
        selected_index: int,
        counter: object,
    ) -> None:
        if (
            counter is not self._execution_counter
            or getattr(layer, _EXECUTION_COUNTER_ATTRIBUTE, None) is not self._execution_counter
        ):
            raise RuntimeError("TE CUDA graph execution counter owner does not match the layer")
        registration = self._registrations.get(id(bank))
        if (
            bank is not self.active_bank
            or registration is None
            or registration.replay_guard is not replay_guard
            or getattr(layer, "_te_cuda_graph_bank_replay_guard", None) is not replay_guard
        ):
            raise RuntimeError("TE CUDA graph replay guard owner changed before launch")
        layer_index = registration.layer_index_by_id.get(id(layer))
        if (
            layer_index is None
            or layer_index >= len(self.layers)
            or self.layers[layer_index] is not layer
        ):
            raise RuntimeError("TE CUDA graph layer owner changed before launch")
        canonical_list = registration.owned_graph_lists[layer_index]
        if installed_graph_list is not canonical_list:
            raise RuntimeError("TE CUDA graph list owner changed before launch")
        if not 0 <= selected_index < len(canonical_list) or (
            canonical_list[selected_index]
            is not registration.graph_tuples[layer_index][selected_index]
        ):
            raise RuntimeError("TE CUDA graph selected callable changed before launch")
        self._execution_counter.record_graph_call()

    def _attach_execution_counter(self) -> None:
        layers = self._unique_layers()
        for layer in layers:
            existing = getattr(layer, _EXECUTION_COUNTER_ATTRIBUTE, None)
            if existing is None:
                continue
            owner_ref = getattr(existing, "_owner_ref", None)
            if (
                not isinstance(existing, _TECudaGraphExecutionCounter)
                or not callable(owner_ref)
                or owner_ref() is not None
            ):
                raise ValueError(
                    "TE CUDA graph execution counter is already owned by another manager"
                )
        for layer in layers:
            setattr(layer, _EXECUTION_COUNTER_ATTRIBUTE, self._execution_counter)

    def _assert_execution_counter_open(self) -> None:
        if self._execution_counter_closed:
            raise RuntimeError("TE CUDA graph execution counter manager is closed")

    def _validate_execution_counter_snapshot(
        self, snapshot: TECudaGraphExecutionCounterSnapshot
    ) -> None:
        if not isinstance(snapshot, TECudaGraphExecutionCounterSnapshot):
            raise TypeError("execution counter snapshot has an invalid type")
        if snapshot._owner is not self._execution_counter:
            raise ValueError(
                "execution counter snapshot belongs to a different TECudaGraphBankManager"
            )

    def _unique_layers(self) -> tuple[object, ...]:
        seen: set[int] = set()
        unique_layers = []
        for layer in self.layers:
            layer_id = id(layer)
            if layer_id not in seen:
                seen.add(layer_id)
                unique_layers.append(layer)
        return tuple(unique_layers)

    def _assert_model_drained(self) -> None:
        if self._assert_model_drained_callback() is False:
            raise RuntimeError(
                "Model is not drained: delayed-wgrad or communication work is still live"
            )
        self._synchronize()
        for layer in self.layers:
            check = getattr(layer, "assert_te_cuda_graph_bank_drained", None)
            if callable(check):
                check()

    def _validate_helper(self, helper: object) -> None:
        if getattr(helper, "_capture_attempted", False) or getattr(
            helper, "_capture_finished", False
        ):
            raise ValueError("TECudaGraphHelper is one-shot and has already been consumed")
        helper_layers = tuple(helper.flattened_callables)
        if len(helper_layers) != len(self.layers) or any(
            actual is not expected for actual, expected in zip(helper_layers, self.layers)
        ):
            raise ValueError("helper layer topology differs from TECudaGraphBankManager")
        modules = _normalize_cuda_graph_modules(getattr(helper.config, "cuda_graph_modules", ()))
        if modules != self._cuda_graph_modules:
            raise ValueError("helper cuda_graph_modules differ from TECudaGraphBankManager")

    def _validate_capture_result(
        self,
        captured_pairs: object,
        owned_graph_lists: tuple[list[object], ...],
        num_microbatches: int,
    ) -> None:
        pairs = tuple(captured_pairs)
        if len(pairs) != len(self.layers):
            raise ValueError("Captured TE CUDA graph layer topology is incomplete")
        for index, ((layer, graphs), expected_layer, owned_list) in enumerate(
            zip(pairs, self.layers, owned_graph_lists)
        ):
            graph_tuple = tuple(graphs)
            if layer is not expected_layer:
                raise ValueError(f"Captured TE CUDA graph layer differs at index {index}")
            if getattr(layer, "cuda_graphs", None) is not owned_list:
                raise ValueError(f"Captured TE CUDA graph replaced its owned list at index {index}")
            if len(owned_list) != num_microbatches:
                raise ValueError("captured graph count does not match num_microbatches")
            if len(graph_tuple) != len(owned_list) or any(
                actual is not expected for actual, expected in zip(graph_tuple, owned_list)
            ):
                raise ValueError(f"Captured TE CUDA graph contents differ at index {index}")

    def _validate_bank(self, bank: TECudaGraphBank, *, require_registered: bool = True) -> None:
        if bank._manager is not self:
            raise ValueError("Bank belongs to a different TECudaGraphBankManager")
        if bank in self._terminal_banks:
            raise ValueError("TE CUDA graph bank has already been reset")
        registration = self._registrations.get(id(bank))
        if require_registered:
            if registration is None or registration.bank is not bank:
                raise ValueError("TE CUDA graph bank registration is missing or forged")
            self._validate_registration(bank, registration)
        fingerprint = bank.fingerprint
        if fingerprint.layer_ids != self._layer_ids:
            raise ValueError("layer_ids differ from TECudaGraphBankManager")
        graph_tuples = tuple(tuple(graph_list) for graph_list in bank._owned_graph_lists)
        graph_identities = tuple(
            tuple(id(graph) for graph in graph_tuple) for graph_tuple in graph_tuples
        )
        if fingerprint.graph_identities != graph_identities:
            raise ValueError("graph_identities differ from owned CUDA graph lists")
        if fingerprint.graph_counts != tuple(map(len, graph_tuples)):
            raise ValueError("graph_counts differ from owned CUDA graph lists")
        if any(count != fingerprint.num_microbatches for count in fingerprint.graph_counts):
            raise ValueError("graph_counts differ from num_microbatches")
        if fingerprint.cuda_graph_modules != self._cuda_graph_modules:
            raise ValueError("cuda_graph_modules differ from TECudaGraphBankManager")
        contracts = bank._layer_contracts
        if fingerprint.packed_input_signatures != self._packed_input_signatures(contracts):
            raise ValueError("packed_input_signatures differ from bank contracts")
        padding = tuple(
            (id(layer), contract.padding_mask_signature)
            for layer, contract in zip(self.layers, contracts)
        )
        if fingerprint.padding_mask_signatures != padding:
            raise ValueError("padding_mask_signatures differ from bank contracts")
        if fingerprint.moe_attribute_schema != self._moe_attribute_schema(require_tensors=False):
            raise ValueError("moe_attribute_schema differs from the current ordered schema")
        dispatcher = tuple(
            (id(layer), contract.dispatcher_states)
            for layer, contract in zip(self.layers, contracts)
        )
        if fingerprint.dispatcher_state_signatures != dispatcher:
            raise ValueError("dispatcher_state_signatures differ from bank contracts")

    def _register_bank(self, bank: TECudaGraphBank) -> None:
        graph_ids = {id(graph) for graph_list in bank._owned_graph_lists for graph in graph_list}
        for registration in self._registrations.values():
            registered_ids = {
                id(graph) for graph_tuple in registration.graph_tuples for graph in graph_tuple
            }
            if graph_ids & registered_ids:
                raise ValueError("CUDA graph identity is shared by another live bank")
        guard = _BankReplayGuard(self, bank)
        self._registrations[id(bank)] = _BankRegistration(
            bank=bank,
            fingerprint=bank.fingerprint,
            layer_ids=self._layer_ids,
            layer_index_by_id={id(layer): index for index, layer in enumerate(self.layers)},
            owned_graph_lists=bank._owned_graph_lists,
            graph_tuples=tuple(tuple(graph_list) for graph_list in bank._owned_graph_lists),
            contracts=bank._layer_contracts,
            replay_guard=guard,
            reset_graph_ids=set(),
        )

    def _validate_registration(
        self, bank: TECudaGraphBank, registration: _BankRegistration
    ) -> None:
        for field_name in TECudaGraphBankFingerprint.__dataclass_fields__:
            if getattr(bank.fingerprint, field_name) != getattr(
                registration.fingerprint, field_name
            ):
                raise ValueError(f"TE CUDA graph bank registration {field_name} was mutated")
        if bank.fingerprint is not registration.fingerprint:
            raise ValueError("TE CUDA graph bank registration fingerprint identity was mutated")
        if (
            bank._owned_graph_lists is not registration.owned_graph_lists
            or bank._layer_contracts is not registration.contracts
            or len(bank.graphs_by_layer) != len(registration.layer_ids)
        ):
            raise ValueError("TE CUDA graph bank registration structure was mutated")
        for index, ((layer, public_graphs), graph_list, graph_tuple) in enumerate(
            zip(bank.graphs_by_layer, registration.owned_graph_lists, registration.graph_tuples)
        ):
            if (
                id(layer) != registration.layer_ids[index]
                or graph_list is not bank._owned_graph_lists[index]
                or len(public_graphs) != len(graph_tuple)
                or any(
                    actual is not expected for actual, expected in zip(public_graphs, graph_tuple)
                )
                or len(graph_list) != len(graph_tuple)
                or any(actual is not expected for actual, expected in zip(graph_list, graph_tuple))
            ):
                raise ValueError("TE CUDA graph bank registration contents were mutated")

    def _snapshot_installations(self) -> tuple[_LayerInstallation, ...]:
        installations = []
        for layer in self.layers:
            snapshot_references = getattr(layer, "snapshot_te_cuda_graph_bank_references", None)
            references_supported = callable(snapshot_references)
            installations.append(
                _LayerInstallation(
                    graph_list=getattr(layer, "cuda_graphs", None),
                    manual_hooks=getattr(layer, "cuda_graph_manual_hooks", None),
                    packed_attributes=self._snapshot_packed_attributes(layer),
                    padding_mask_signature_present=hasattr(
                        layer, _PADDING_MASK_SIGNATURE_ATTRIBUTE
                    ),
                    padding_mask_signature=getattr(layer, _PADDING_MASK_SIGNATURE_ATTRIBUTE, None),
                    dispatcher_states_present=hasattr(layer, _DISPATCHER_STATES_ATTRIBUTE),
                    dispatcher_states=getattr(layer, _DISPATCHER_STATES_ATTRIBUTE, None),
                    replay_guard_present=hasattr(layer, "_te_cuda_graph_bank_replay_guard"),
                    replay_guard=getattr(layer, "_te_cuda_graph_bank_replay_guard", None),
                    bank_references_supported=references_supported,
                    bank_references=snapshot_references() if references_supported else None,
                )
            )
        return tuple(installations)

    def _restore_installations(self, installations: tuple[_LayerInstallation, ...]) -> None:
        for layer, installation in zip(self.layers, installations):
            self._install_packed_attributes(layer, installation.packed_attributes)
            self._restore_optional_attribute(
                layer,
                _PADDING_MASK_SIGNATURE_ATTRIBUTE,
                installation.padding_mask_signature_present,
                installation.padding_mask_signature,
            )
            self._restore_optional_attribute(
                layer,
                _DISPATCHER_STATES_ATTRIBUTE,
                installation.dispatcher_states_present,
                installation.dispatcher_states,
            )
            layer.cuda_graph_manual_hooks = installation.manual_hooks
            layer.cuda_graphs = installation.graph_list
            self._restore_optional_attribute(
                layer,
                "_te_cuda_graph_bank_replay_guard",
                installation.replay_guard_present,
                installation.replay_guard,
            )
            if installation.bank_references_supported:
                restore_references = getattr(layer, "restore_te_cuda_graph_bank_references", None)
                if not callable(restore_references):
                    raise RuntimeError(
                        "Layer can snapshot TE CUDA graph references but cannot restore them"
                    )
                restore_references(installation.bank_references)

    def _snapshot_contract(
        self, layer: object, graph_count: int, *, manual_hooks: object
    ) -> _LayerReplayContract:
        raw_states = getattr(layer, _DISPATCHER_STATES_ATTRIBUTE, None)
        requires_dispatcher_states = bool(
            getattr(layer, "is_moe_layer", False) and "moe_preprocess" in self._cuda_graph_modules
        )
        if not requires_dispatcher_states:
            dispatcher_states = (None,) * graph_count
        else:
            if raw_states is None:
                raise ValueError("partial MoE capture did not expose dispatcher replay states")
            dispatcher_states = tuple(raw_states)
            if len(dispatcher_states) != graph_count:
                raise ValueError(
                    "dispatcher state count must match the exact captured graph identity count"
                )
            capture_cursor = getattr(layer, "_te_cuda_graph_capture_cursor", None)
            if capture_cursor != graph_count:
                raise ValueError(
                    "dispatcher states were not recorded by exactly one committed forward "
                    "capture per owned graph identity"
                )
        return _LayerReplayContract(
            manual_hooks=manual_hooks,
            packed_attributes=self._snapshot_packed_attributes(layer),
            padding_mask_signature=getattr(layer, _PADDING_MASK_SIGNATURE_ATTRIBUTE, None),
            dispatcher_states=dispatcher_states,
        )

    @staticmethod
    def _snapshot_packed_attributes(layer: object) -> tuple[tuple[str, bool, object], ...]:
        return tuple(
            (attribute, hasattr(layer, attribute), getattr(layer, attribute, None))
            for attribute in _PACKED_REPLAY_ATTRIBUTES
        )

    @staticmethod
    def _install_packed_attributes(
        layer: object, attributes: tuple[tuple[str, bool, object], ...]
    ) -> None:
        for attribute, present, value in attributes:
            TECudaGraphBankManager._restore_optional_attribute(layer, attribute, present, value)

    @staticmethod
    def _restore_optional_attribute(
        target: object, name: str, present: bool, value: object
    ) -> None:
        if present:
            setattr(target, name, value)
        elif hasattr(target, name):
            delattr(target, name)

    @staticmethod
    def _clear_replay_contract(layer: object) -> None:
        for attribute in (*_PACKED_REPLAY_ATTRIBUTES, _PADDING_MASK_SIGNATURE_ATTRIBUTE):
            if hasattr(layer, attribute):
                delattr(layer, attribute)
        if hasattr(layer, _DISPATCHER_STATES_ATTRIBUTE):
            setattr(layer, _DISPATCHER_STATES_ATTRIBUTE, ())

    def _packed_input_signatures(
        self, contracts: tuple[_LayerReplayContract, ...]
    ) -> tuple[tuple[int, tuple[object, ...]], ...]:
        signatures = []
        for layer, contract in zip(self.layers, contracts):
            attributes = {
                name: (present, value) for name, present, value in contract.packed_attributes
            }
            layer_signature = []
            generic_static = attributes["_te_cuda_graph_packed_seq_params_static_metadata"]
            generic_keys = attributes["_te_cuda_graph_packed_seq_params_tensor_kwarg_names"]
            if generic_static[0] or generic_keys[0]:
                layer_signature.append(
                    (
                        "generic",
                        _freeze_signature(generic_static[1]) if generic_static[0] else None,
                        tuple(generic_keys[1]) if generic_keys[0] else (),
                    )
                )
            mamba_static = attributes["_te_cuda_graph_mamba_packed_seq_params_static_metadata"]
            mamba_tensors = attributes["_te_cuda_graph_mamba_packed_seq_params_tensor_signatures"]
            if mamba_static[0] or mamba_tensors[0]:
                layer_signature.append(
                    (
                        "mamba",
                        _freeze_signature(mamba_static[1]) if mamba_static[0] else None,
                        _freeze_signature(mamba_tensors[1]) if mamba_tensors[0] else (),
                    )
                )
            signatures.append((id(layer), tuple(layer_signature)))
        return tuple(signatures)

    @staticmethod
    def _resolve_dotted_attribute(root: object, path: str) -> object:
        value = root
        for component in path.split("."):
            value = getattr(value, component)
        return value

    def _moe_attribute_schema(
        self, *, require_tensors: bool
    ) -> tuple[tuple[int, tuple[str, ...]], ...]:
        schema = []
        for layer in self.layers:
            get_schema = getattr(layer, "te_cuda_graph_bank_schema", None)
            names = tuple(get_schema()) if callable(get_schema) else ()
            if require_tensors and names:
                dispatcher = layer.mlp.token_dispatcher
                for name in names:
                    value = self._resolve_dotted_attribute(dispatcher, name)
                    if not torch.is_tensor(value):
                        raise ValueError(
                            f"MoE CUDA graph attribute {name!r} must resolve to a Tensor"
                        )
            schema.append((id(layer), names))
        return tuple(schema)

    def _install_bank(self, bank: TECudaGraphBank) -> None:
        registration = self._registrations[id(bank)]
        for layer, graph_list, contract in zip(
            self.layers, registration.owned_graph_lists, registration.contracts
        ):
            self._install_packed_attributes(layer, contract.packed_attributes)
            self._restore_optional_attribute(
                layer,
                _PADDING_MASK_SIGNATURE_ATTRIBUTE,
                contract.padding_mask_signature is not None,
                contract.padding_mask_signature,
            )
            if hasattr(layer, _DISPATCHER_STATES_ATTRIBUTE) or any(
                state is not None for state in contract.dispatcher_states
            ):
                setattr(layer, _DISPATCHER_STATES_ATTRIBUTE, contract.dispatcher_states)
            layer.cuda_graph_manual_hooks = contract.manual_hooks
            layer.cuda_graphs = graph_list
            layer._te_cuda_graph_bank_replay_guard = registration.replay_guard

    def _bank_is_installed(self, bank: TECudaGraphBank) -> bool:
        return all(
            getattr(layer, "cuda_graphs", None) is graph_list
            for layer, graph_list in zip(self.layers, bank._owned_graph_lists)
        )

    def _clear_installed_bank(self, bank: TECudaGraphBank) -> None:
        if bank is not self.active_bank:
            return
        if not self._bank_is_installed(bank):
            raise ValueError("Active TE CUDA graph bank is not fully installed")
        registration = self._registrations[id(bank)]
        for layer, graph_list in zip(self.layers, registration.owned_graph_lists):
            clear_references = getattr(layer, "clear_te_cuda_graph_bank_references", None)
            if callable(clear_references):
                clear_references()
            self._clear_replay_contract(layer)
            layer.cuda_graph_manual_hooks = []
            layer.cuda_graphs = []
            if (
                getattr(layer, "_te_cuda_graph_bank_replay_guard", None)
                is registration.replay_guard
            ):
                delattr(layer, "_te_cuda_graph_bank_replay_guard")

    def _live_graph_ids(self) -> set[int]:
        return {
            id(graph)
            for registration in self._registrations.values()
            for graph_tuple in registration.graph_tuples
            for graph in graph_tuple
        }

    def _reset_graph_identities(
        self, graph_lists: Sequence[Sequence[object]], *, already_reset: set[int]
    ) -> None:
        first_error = None
        for graph in (graph for graph_list in graph_lists for graph in graph_list):
            graph_id = id(graph)
            if graph_id in already_reset:
                continue
            already_reset.add(graph_id)
            if self._graph_reset_supported and hasattr(graph, "reset"):
                try:
                    graph.reset()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error
