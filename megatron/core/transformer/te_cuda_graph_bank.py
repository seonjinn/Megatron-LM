# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owned Transformer Engine CUDA graph banks.

``layer.cuda_graphs`` remains the legacy replay surface.  This module makes
the mutable list installed there an ownership token, so cached graph sets can
be switched and reset without operating on a different bank by accident.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

_PACKED_REPLAY_ATTRIBUTES = (
    "_te_cuda_graph_packed_seq_params_static_metadata",
    "_te_cuda_graph_packed_seq_params_tensor_kwarg_names",
    "_te_cuda_graph_mamba_packed_seq_params_static_metadata",
    "_te_cuda_graph_mamba_packed_seq_params_tensor_signatures",
)
_UNSET = object()


def _freeze_signature(value: object) -> object:
    """Convert replay metadata to a stable, hashable fingerprint value."""
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
    return tuple(sorted(str(getattr(module, "value", module)) for module in modules))


@dataclass(frozen=True)
class TECudaGraphBankFingerprint:
    """Capture provenance that must match before a bank can be replayed."""

    num_microbatches: int
    layer_ids: tuple[int, ...]
    graph_counts: tuple[int, ...]
    cuda_graph_modules: tuple[str, ...]
    packed_input_signature: tuple[tuple[str, object], ...]
    moe_attribute_schema: tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(frozen=True)
class _LayerReplayContract:
    manual_hooks: object
    packed_attributes: tuple[tuple[str, bool, object], ...]


@dataclass(frozen=True)
class _LayerInstallation:
    graph_list: object
    manual_hooks: object
    packed_attributes: tuple[tuple[str, bool, object], ...]
    replay_guard_present: bool
    replay_guard: object


@dataclass(frozen=True)
class _BankRegistration:
    bank: "TECudaGraphBank"
    fingerprint: TECudaGraphBankFingerprint
    layer_ids: tuple[int, ...]
    owned_graph_lists: tuple[list[object], ...]
    graph_tuples: tuple[tuple[object, ...], ...]
    contracts: tuple[_LayerReplayContract, ...]
    replay_guard: object


class _BankReplayGuard:
    def __init__(
        self, manager: "TECudaGraphBankManager", bank: "TECudaGraphBank"
    ) -> None:
        self._manager = manager
        self._bank = bank

    def __call__(self, layer: object, installed_graph_list: list[object]) -> None:
        self._manager._assert_replay_ready(self._bank, layer, installed_graph_list)


@dataclass(eq=False)
class TECudaGraphBank:
    """One manager-owned TE CUDA graph set."""

    fingerprint: TECudaGraphBankFingerprint
    graphs_by_layer: tuple[tuple[object, tuple[object, ...]], ...]
    _manager: "TECudaGraphBankManager" = field(repr=False)
    _owned_graph_lists: tuple[list[object], ...] = field(repr=False)
    _layer_contracts: tuple[_LayerReplayContract, ...] = field(repr=False)
    _reset_graph_ids: set[int] = field(default_factory=set, init=False, repr=False)
    _is_reset: bool = field(default=False, init=False, repr=False)

    def activate(self) -> None:
        self._manager.activate(self)

    def reset(self) -> None:
        if self._is_reset:
            return
        self._manager.reset(self)


class TECudaGraphBankManager:
    """Own and atomically install TE CUDA graph banks for one model topology."""

    def __init__(
        self,
        layers: Sequence[object],
        *,
        cuda_graph_modules: Iterable[object] = (),
        packed_input_signature: tuple[tuple[str, object], ...] | None = None,
        moe_attribute_schema: tuple[tuple[int, tuple[str, ...]], ...] | None = None,
        assert_model_drained: Callable[[], object] | None = None,
        graph_reset_supported: bool | None = None,
        synchronize: Callable[[], None] | None = None,
        runtime_num_microbatches: Callable[[], int] | None = None,
    ) -> None:
        self.layers = tuple(layers)
        self._layer_ids = tuple(id(layer) for layer in self.layers)
        self._cuda_graph_modules = _normalize_cuda_graph_modules(cuda_graph_modules)
        self._expected_packed_input_signature: object = (
            _UNSET if packed_input_signature is None else tuple(packed_input_signature)
        )
        self._expected_moe_attribute_schema: object = (
            _UNSET if moe_attribute_schema is None else tuple(moe_attribute_schema)
        )
        self._assert_model_drained_callback = assert_model_drained or (lambda: True)
        if graph_reset_supported is None:
            from megatron.core.utils import is_te_min_version

            graph_reset_supported = is_te_min_version("2.10.0")
        self._graph_reset_supported = graph_reset_supported
        if synchronize is None:
            import torch

            synchronize = torch.cuda.synchronize
        self._synchronize = synchronize
        self._runtime_num_microbatches = runtime_num_microbatches
        self._registrations: dict[int, _BankRegistration] = {}
        self.active_bank: TECudaGraphBank | None = None

    @property
    def registered_bank_count(self) -> int:
        return len(self._registrations)

    @classmethod
    def from_helper(
        cls,
        helper: object,
        *,
        assert_model_drained: Callable[[], object] | None = None,
    ) -> "TECudaGraphBankManager":
        """Build a manager from a one-shot helper's fixed model provenance."""
        pp_size = helper.pp_group.size()
        overlap_moe_expert_parallel_comm = (
            helper.config.overlap_moe_expert_parallel_comm
        )

        def runtime_num_microbatches() -> int:
            if pp_size == 1 and not overlap_moe_expert_parallel_comm:
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
        """Capture a bank while preserving the currently installed bank."""
        self._assert_model_drained()
        self._validate_helper(helper)
        if num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")

        previous_installations = self._snapshot_installations()
        previous_active_bank = self.active_bank
        owned_graph_lists = tuple([] for _ in self.layers)
        attempt_started = False
        try:
            helper._capture_attempted = True
            attempt_started = True
            for layer, graph_list, installation in zip(
                self.layers, owned_graph_lists, previous_installations
            ):
                self._clear_replay_contract(layer)
                if hasattr(layer, "_te_cuda_graph_bank_replay_guard"):
                    delattr(layer, "_te_cuda_graph_bank_replay_guard")
                layer.cuda_graph_manual_hooks = installation.manual_hooks
                layer.cuda_graphs = graph_list
            captured_pairs = helper._capture_cuda_graph_lists(
                num_microbatches=num_microbatches
            )
            self._validate_capture_result(captured_pairs, owned_graph_lists)
            contracts = tuple(
                self._snapshot_contract(
                    layer, manual_hooks=installation.manual_hooks
                )
                for layer, installation in zip(self.layers, previous_installations)
            )
            graphs_by_layer = tuple(
                (layer, tuple(graph_list))
                for layer, graph_list in zip(self.layers, owned_graph_lists)
            )
            fingerprint = TECudaGraphBankFingerprint(
                num_microbatches=num_microbatches,
                layer_ids=self._layer_ids,
                graph_counts=tuple(len(graphs) for _, graphs in graphs_by_layer),
                cuda_graph_modules=self._cuda_graph_modules,
                packed_input_signature=self._packed_input_signature(contracts),
                moe_attribute_schema=self._moe_attribute_schema(),
            )
            bank = TECudaGraphBank(
                fingerprint=fingerprint,
                graphs_by_layer=graphs_by_layer,
                _manager=self,
                _owned_graph_lists=owned_graph_lists,
                _layer_contracts=contracts,
            )
            self._validate_bank(
                bank,
                establish_expected_fingerprint=True,
                require_registered=False,
            )
            self._register_bank(bank)
            return bank
        except BaseException:
            if attempt_started and hasattr(helper, "_capture_finished"):
                helper._capture_finished = False
            if attempt_started and hasattr(helper, "_graphs_created"):
                helper._graphs_created = False
            try:
                self._synchronize()
            except Exception:
                pass
            else:
                try:
                    self._reset_graph_identities(
                        owned_graph_lists,
                        already_reset=self._live_graph_ids(),
                    )
                except Exception:
                    pass
            raise
        finally:
            self._restore_installations(previous_installations)
            self.active_bank = previous_active_bank

    def activate(
        self,
        bank: TECudaGraphBank,
        *,
        num_microbatches: int | None = None,
    ) -> None:
        """Install a compatible bank's exact owned lists on every layer."""
        self._assert_model_drained()
        self._validate_bank(bank)
        runtime_num_microbatches = self._get_runtime_num_microbatches(num_microbatches)
        if bank.fingerprint.num_microbatches != runtime_num_microbatches:
            raise ValueError(
                "runtime num_microbatches does not match the captured TE CUDA graph bank"
            )
        previous_installations = self._snapshot_installations()
        previous_active_bank = self.active_bank
        try:
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
        self._clear_installed_bank(target)
        self.active_bank = None

    def reset(self, bank: TECudaGraphBank) -> None:
        """Reset only graph identities owned by ``bank``."""
        if bank._is_reset:
            return
        self._assert_model_drained()
        self._validate_bank(bank)
        registration = self._registrations[id(bank)]
        self._clear_installed_bank(bank)
        if self.active_bank is bank:
            self.active_bank = None
        identities = {
            id(graph) for graph_list in bank._owned_graph_lists for graph in graph_list
        }
        pending_identities = identities - bank._reset_graph_ids
        try:
            if pending_identities:
                self._synchronize()
                self._reset_graph_identities(
                    bank._owned_graph_lists,
                    already_reset=bank._reset_graph_ids,
                )
        finally:
            self._registrations.pop(id(bank), None)
            bank.graphs_by_layer = ()
            bank._owned_graph_lists = ()
            bank._layer_contracts = ()
            bank._is_reset = True
            del registration

    def refresh_manual_hooks(self, bank: TECudaGraphBank) -> None:
        """Refresh the authorized manual-hook objects for an active bank."""
        self._validate_bank(bank)
        if bank is not self.active_bank or not self._bank_is_installed(bank):
            raise ValueError("Manual hooks can only be refreshed for the active bank")
        registration = self._registrations[id(bank)]
        contracts = tuple(
            _LayerReplayContract(
                manual_hooks=layer.cuda_graph_manual_hooks,
                packed_attributes=contract.packed_attributes,
            )
            for layer, contract in zip(self.layers, bank._layer_contracts)
        )
        bank._layer_contracts = contracts
        self._registrations[id(bank)] = _BankRegistration(
            bank=bank,
            fingerprint=registration.fingerprint,
            layer_ids=registration.layer_ids,
            owned_graph_lists=registration.owned_graph_lists,
            graph_tuples=registration.graph_tuples,
            contracts=contracts,
            replay_guard=registration.replay_guard,
        )

    def get_graph(
        self,
        bank: TECudaGraphBank,
        layer: object,
        *,
        microbatch_index: int,
        num_microbatches: int,
    ) -> object:
        """Validate replay geometry before selecting the legacy modulo graph."""
        self._validate_bank(bank)
        if bank is not self.active_bank or not self._bank_is_installed(bank):
            raise ValueError("TE CUDA graph bank is not active")
        if num_microbatches != bank.fingerprint.num_microbatches:
            raise ValueError(
                "num_microbatches does not match the active TE CUDA graph bank"
            )
        try:
            layer_index = next(
                index
                for index, expected_layer in enumerate(self.layers)
                if expected_layer is layer
            )
        except StopIteration as exc:
            raise ValueError(
                "layer is not owned by this TECudaGraphBankManager"
            ) from exc
        graphs = bank._owned_graph_lists[layer_index]
        return graphs[microbatch_index % num_microbatches]

    def _get_runtime_num_microbatches(self, supplied: int | None = None) -> int:
        if supplied is not None:
            if (
                self._runtime_num_microbatches is not None
                and self._runtime_num_microbatches() != supplied
            ):
                raise ValueError(
                    "supplied num_microbatches differs from the runtime provider"
                )
            return supplied
        if self._runtime_num_microbatches is None:
            raise ValueError(
                "runtime num_microbatches must be supplied when no provider is configured"
            )
        return self._runtime_num_microbatches()

    def _assert_replay_ready(
        self,
        bank: TECudaGraphBank,
        layer: object,
        installed_graph_list: list[object],
    ) -> None:
        self._validate_bank(bank)
        if bank is not self.active_bank:
            raise ValueError("TE CUDA graph bank is not active")
        registration = self._registrations[id(bank)]
        try:
            layer_index = next(
                index
                for index, expected_layer in enumerate(self.layers)
                if layer is expected_layer
            )
        except StopIteration as exc:
            raise ValueError("Layer is not registered with the active bank") from exc
        if installed_graph_list is not registration.owned_graph_lists[layer_index]:
            raise ValueError("Layer CUDA graph list does not match its bank registration")
        runtime_num_microbatches = self._get_runtime_num_microbatches()
        if runtime_num_microbatches != bank.fingerprint.num_microbatches:
            raise ValueError(
                "runtime num_microbatches does not match the active TE CUDA graph bank"
            )

    def _assert_model_drained(self) -> None:
        drained = self._assert_model_drained_callback()
        if drained is False:
            raise RuntimeError(
                "Model is not drained: delayed-wgrad or communication work is still live"
            )

    def _validate_helper(self, helper: object) -> None:
        if getattr(helper, "_capture_attempted", False) or getattr(
            helper, "_capture_finished", False
        ):
            raise ValueError("TECudaGraphHelper is one-shot and has already been consumed")
        helper_layers = tuple(helper.flattened_callables)
        if len(helper_layers) != len(self.layers) or any(
            actual is not expected
            for actual, expected in zip(helper_layers, self.layers)
        ):
            raise ValueError(
                "helper layer topology differs from TECudaGraphBankManager"
            )
        helper_modules = _normalize_cuda_graph_modules(
            getattr(helper.config, "cuda_graph_modules", ())
        )
        if helper_modules != self._cuda_graph_modules:
            raise ValueError(
                "helper cuda_graph_modules differ from TECudaGraphBankManager"
            )

    def _validate_capture_result(
        self,
        captured_pairs: object,
        owned_graph_lists: tuple[list[object], ...],
    ) -> None:
        pairs = tuple(captured_pairs)
        if len(pairs) != len(self.layers):
            raise ValueError("Captured TE CUDA graph layer topology is incomplete")
        for index, ((layer, graphs), expected_layer, owned_list) in enumerate(
            zip(pairs, self.layers, owned_graph_lists)
        ):
            if layer is not expected_layer:
                raise ValueError(
                    f"Captured TE CUDA graph layer topology differs at index {index}"
                )
            if getattr(layer, "cuda_graphs", None) is not owned_list:
                raise ValueError(
                    f"Captured TE CUDA graph replaced its owned list at index {index}"
                )
            captured_graphs = tuple(graphs)
            if len(captured_graphs) != len(owned_list) or any(
                captured is not owned
                for captured, owned in zip(captured_graphs, owned_list)
            ):
                raise ValueError(
                    f"Captured TE CUDA graph contents differ from owned list at index {index}"
                )

    def _validate_bank(
        self,
        bank: TECudaGraphBank,
        *,
        establish_expected_fingerprint: bool = False,
        allow_reset: bool = False,
        require_registered: bool = True,
    ) -> None:
        if bank._manager is not self:
            raise ValueError("Bank belongs to a different TECudaGraphBankManager")
        if bank._is_reset and not allow_reset:
            raise ValueError("TE CUDA graph bank has already been reset")
        registration = self._registrations.get(id(bank))
        if require_registered:
            if registration is None or registration.bank is not bank:
                raise ValueError("TE CUDA graph bank registration is missing or forged")
            self._validate_registration(bank, registration)
        fingerprint = bank.fingerprint
        if fingerprint.layer_ids != self._layer_ids:
            raise ValueError("layer_ids differ from TECudaGraphBankManager")
        actual_counts = tuple(len(graphs) for graphs in bank._owned_graph_lists)
        if fingerprint.graph_counts != actual_counts or any(
            count != fingerprint.num_microbatches for count in fingerprint.graph_counts
        ):
            raise ValueError("graph_counts differ from owned CUDA graph lists")
        if fingerprint.cuda_graph_modules != self._cuda_graph_modules:
            raise ValueError("cuda_graph_modules differ from TECudaGraphBankManager")

        expected_packed_input_signature = self._expected_packed_input_signature
        if expected_packed_input_signature is _UNSET:
            expected_packed_input_signature = fingerprint.packed_input_signature
        if fingerprint.packed_input_signature != expected_packed_input_signature:
            raise ValueError(
                "packed_input_signature differs from TECudaGraphBankManager"
            )

        expected_moe_attribute_schema = self._expected_moe_attribute_schema
        if expected_moe_attribute_schema is _UNSET:
            expected_moe_attribute_schema = fingerprint.moe_attribute_schema
        if fingerprint.moe_attribute_schema != expected_moe_attribute_schema:
            raise ValueError("moe_attribute_schema differs from TECudaGraphBankManager")
        if establish_expected_fingerprint:
            self._expected_packed_input_signature = expected_packed_input_signature
            self._expected_moe_attribute_schema = expected_moe_attribute_schema

    def _register_bank(self, bank: TECudaGraphBank) -> None:
        graph_ids = {
            id(graph) for graph_list in bank._owned_graph_lists for graph in graph_list
        }
        for registration in self._registrations.values():
            registered_ids = {
                id(graph)
                for graph_tuple in registration.graph_tuples
                for graph in graph_tuple
            }
            if graph_ids & registered_ids:
                raise ValueError("CUDA graph identity is shared by another live bank")
        guard = _BankReplayGuard(self, bank)
        self._registrations[id(bank)] = _BankRegistration(
            bank=bank,
            fingerprint=bank.fingerprint,
            layer_ids=tuple(id(layer) for layer, _ in bank.graphs_by_layer),
            owned_graph_lists=bank._owned_graph_lists,
            graph_tuples=tuple(tuple(graph_list) for graph_list in bank._owned_graph_lists),
            contracts=bank._layer_contracts,
            replay_guard=guard,
        )

    def _live_graph_ids(self) -> set[int]:
        return {
            id(graph)
            for registration in self._registrations.values()
            for graph_tuple in registration.graph_tuples
            for graph in graph_tuple
        }

    def _validate_registration(
        self, bank: TECudaGraphBank, registration: _BankRegistration
    ) -> None:
        for field_name in (
            "num_microbatches",
            "layer_ids",
            "graph_counts",
            "cuda_graph_modules",
            "packed_input_signature",
            "moe_attribute_schema",
        ):
            if getattr(bank.fingerprint, field_name) != getattr(
                registration.fingerprint, field_name
            ):
                raise ValueError(
                    f"TE CUDA graph bank registration {field_name} was mutated"
                )
        if (
            bank._owned_graph_lists is not registration.owned_graph_lists
            or bank._layer_contracts is not registration.contracts
            or len(bank.graphs_by_layer) != len(registration.layer_ids)
            or len(bank._owned_graph_lists) != len(registration.layer_ids)
            or len(bank._layer_contracts) != len(registration.layer_ids)
        ):
            raise ValueError("TE CUDA graph bank registration structure was mutated")
        for index, ((layer, graph_view), graph_list, graph_tuple) in enumerate(
            zip(
                bank.graphs_by_layer,
                registration.owned_graph_lists,
                registration.graph_tuples,
            )
        ):
            if (
                id(layer) != registration.layer_ids[index]
                or len(graph_view) != len(graph_tuple)
                or any(actual is not expected for actual, expected in zip(graph_view, graph_tuple))
                or len(graph_list) != len(graph_tuple)
                or any(actual is not expected for actual, expected in zip(graph_list, graph_tuple))
                or graph_list is not bank._owned_graph_lists[index]
            ):
                raise ValueError("TE CUDA graph bank registration contents were mutated")

    def _snapshot_installations(self) -> tuple[_LayerInstallation, ...]:
        return tuple(
            _LayerInstallation(
                graph_list=getattr(layer, "cuda_graphs", None),
                manual_hooks=getattr(layer, "cuda_graph_manual_hooks", None),
                packed_attributes=self._snapshot_packed_attributes(layer),
                replay_guard_present=hasattr(
                    layer, "_te_cuda_graph_bank_replay_guard"
                ),
                replay_guard=getattr(
                    layer, "_te_cuda_graph_bank_replay_guard", None
                ),
            )
            for layer in self.layers
        )

    def _restore_installations(
        self, installations: tuple[_LayerInstallation, ...]
    ) -> None:
        for layer, installation in zip(self.layers, installations):
            self._install_packed_attributes(layer, installation.packed_attributes)
            layer.cuda_graph_manual_hooks = installation.manual_hooks
            layer.cuda_graphs = installation.graph_list
            if installation.replay_guard_present:
                layer._te_cuda_graph_bank_replay_guard = installation.replay_guard
            elif hasattr(layer, "_te_cuda_graph_bank_replay_guard"):
                delattr(layer, "_te_cuda_graph_bank_replay_guard")

    def _snapshot_contract(
        self, layer: object, *, manual_hooks: object | None = None
    ) -> _LayerReplayContract:
        return _LayerReplayContract(
            manual_hooks=(
                getattr(layer, "cuda_graph_manual_hooks", None)
                if manual_hooks is None
                else manual_hooks
            ),
            packed_attributes=self._snapshot_packed_attributes(layer),
        )

    @staticmethod
    def _snapshot_packed_attributes(
        layer: object,
    ) -> tuple[tuple[str, bool, object], ...]:
        return tuple(
            (
                attribute,
                hasattr(layer, attribute),
                getattr(layer, attribute, None),
            )
            for attribute in _PACKED_REPLAY_ATTRIBUTES
        )

    @staticmethod
    def _install_packed_attributes(
        layer: object,
        attributes: tuple[tuple[str, bool, object], ...],
    ) -> None:
        for attribute, present, value in attributes:
            if present:
                setattr(layer, attribute, value)
            elif hasattr(layer, attribute):
                delattr(layer, attribute)

    @staticmethod
    def _clear_replay_contract(layer: object) -> None:
        for attribute in _PACKED_REPLAY_ATTRIBUTES:
            if hasattr(layer, attribute):
                delattr(layer, attribute)

    def _packed_input_signature(
        self, contracts: tuple[_LayerReplayContract, ...]
    ) -> tuple[tuple[str, object], ...]:
        entries = []
        for layer_index, contract in enumerate(contracts):
            for attribute, present, value in contract.packed_attributes:
                state = (
                    ("present", _freeze_signature(value)) if present else ("absent",)
                )
                entries.append((f"{layer_index}:{attribute}", state))
        return tuple(entries)

    def _moe_attribute_schema(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        schema = []
        for layer in self.layers:
            get_schema = getattr(layer, "te_cuda_graph_bank_schema", None)
            attributes = tuple(sorted(get_schema())) if callable(get_schema) else ()
            schema.append((id(layer), attributes))
        return tuple(schema)

    def _install_bank(self, bank: TECudaGraphBank) -> None:
        replay_guard = self._registrations[id(bank)].replay_guard
        for layer, graph_list, contract in zip(
            self.layers,
            bank._owned_graph_lists,
            bank._layer_contracts,
        ):
            self._install_packed_attributes(layer, contract.packed_attributes)
            layer.cuda_graph_manual_hooks = contract.manual_hooks
            layer.cuda_graphs = graph_list
            layer._te_cuda_graph_bank_replay_guard = replay_guard

    def _bank_is_installed(self, bank: TECudaGraphBank) -> bool:
        return all(
            getattr(layer, "cuda_graphs", None) is graph_list
            for layer, graph_list in zip(self.layers, bank._owned_graph_lists)
        )

    def _clear_installed_bank(self, bank: TECudaGraphBank) -> None:
        registration = self._registrations.get(id(bank))
        replay_guard = None if registration is None else registration.replay_guard
        for layer, graph_list in zip(self.layers, bank._owned_graph_lists):
            if getattr(layer, "cuda_graphs", None) is graph_list:
                self._clear_replay_contract(layer)
                layer.cuda_graph_manual_hooks = []
                layer.cuda_graphs = []
                if (
                    getattr(layer, "_te_cuda_graph_bank_replay_guard", None)
                    is replay_guard
                ):
                    delattr(layer, "_te_cuda_graph_bank_replay_guard")

    def _reset_graph_identities(
        self,
        graph_lists: Sequence[Sequence[object]],
        *,
        already_reset: set[int],
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
