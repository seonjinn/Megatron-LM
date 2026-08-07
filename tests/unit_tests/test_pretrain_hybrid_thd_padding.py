# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import pretrain_hybrid
from megatron.core.packed_seq_params import get_thd_padding_kwargs


class _NoOpTimer:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _NoOpStragglerTimer:
    def __call__(self, **_kwargs: Any) -> "_NoOpStragglerTimer":
        return self

    def __enter__(self) -> "_NoOpStragglerTimer":
        return self

    def __exit__(self, *_args: Any) -> None:
        pass


class _RecordingModel:
    vp_stage = None

    def __init__(self) -> None:
        self.args: tuple[Any, ...] | None = None
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.args = args
        self.kwargs = kwargs
        return torch.zeros_like(kwargs["labels"], dtype=torch.float32)


def _static_config() -> SimpleNamespace:
    return SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        max_seqlen_per_dp_cp_rank=12,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=4,
        thd_tail_padding_policy="extend_last",
    )


def _packed_inputs() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]], dtype=torch.int64),
        "labels": torch.tensor([[21, 22, 23, 24, 25, 26, 27, 28]], dtype=torch.int64),
        "loss_mask": torch.tensor([[1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], dtype=torch.int64),
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor(5, dtype=torch.int32),
    }


def test_static_numeric_alignment_resolves_to_fixed_target() -> None:
    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        12, max_seqlen_per_dp_cp_rank=12, thd_max_packed_sequences=4, cuda_graph_static=True
    )

    assert alignment is None
    assert target_len == 12
    assert max_num_seqs == 4


def test_static_alignment_smaller_than_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="fixed target"):
        get_thd_padding_kwargs(
            8, max_seqlen_per_dp_cp_rank=12, thd_max_packed_sequences=4, cuda_graph_static=True
        )


def test_static_cp_padding_resolves_global_token_capacity() -> None:
    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        "max", max_seqlen_per_dp_cp_rank=12, thd_max_packed_sequences=4,
        cuda_graph_static=True, cp_size=16
    )

    assert alignment is None
    assert target_len == 192
    assert max_num_seqs == 4


def test_static_cp_local_numeric_alignment_resolves_global_token_capacity() -> None:
    alignment, target_len, max_num_seqs = get_thd_padding_kwargs(
        12,
        max_seqlen_per_dp_cp_rank=12,
        thd_max_packed_sequences=4,
        cuda_graph_static=True,
        cp_size=16,
    )

    assert alignment is None
    assert target_len == 192
    assert max_num_seqs == 4


def _prepare(
    config: SimpleNamespace | None = None,
    context_parallel_size: int = 1,
    inputs: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[Any, ...]]:
    if inputs is None:
        inputs = _packed_inputs()
    result = pretrain_hybrid._prepare_packed_thd_batch(
        tokens=inputs["tokens"],
        labels=inputs["labels"],
        loss_mask=inputs["loss_mask"],
        position_ids=inputs["position_ids"],
        cu_seqlens=inputs["cu_seqlens"],
        cu_seqlens_padded=inputs["cu_seqlens_padded"],
        max_seqlen=inputs["max_seqlen"],
        local_cp_size=None,
        hybrid_cp_group=None,
        config=config or _static_config(),
        pad_token_id=91,
        context_parallel_size=context_parallel_size,
    )
    return inputs, result


def test_static_batch_keeps_real_and_graph_boundaries_in_distinct_coordinates() -> None:
    _, (_, _, _, _, packed, _) = _prepare()

    assert packed.cu_seqlens_q.tolist() == [0, 3, 5, 5, 5]
    assert packed.cu_seqlens_kv.tolist() == [0, 3, 5, 5, 5]
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 12, 12, 12]
    assert packed.cu_seqlens_kv_padded.tolist() == [0, 4, 12, 12, 12]
    assert packed.total_tokens == 12
    assert packed.tokens_per_sample == 12


def test_static_batch_neutralizes_every_appended_position() -> None:
    _, (tokens, labels, loss_mask, position_ids, _, padding_mask) = _prepare()

    assert tokens.tolist() == [[11, 12, 13, 14, 15, 16, 17, 18, 91, 91, 91, 91]]
    assert labels.tolist() == [[21, 22, 23, 24, 25, 26, 27, 28, -100, -100, -100, -100]]
    assert loss_mask.tolist() == [[1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]]
    assert position_ids.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0]]
    assert padding_mask.dtype == torch.bool
    assert padding_mask.tolist() == [[False] * 8 + [True] * 4]


def test_static_batch_does_not_mutate_caller_owned_tensors() -> None:
    inputs = _packed_inputs()
    originals = {name: tensor.clone() for name, tensor in inputs.items()}

    pretrain_hybrid._prepare_packed_thd_batch(
        tokens=inputs["tokens"],
        labels=inputs["labels"],
        loss_mask=inputs["loss_mask"],
        position_ids=inputs["position_ids"],
        cu_seqlens=inputs["cu_seqlens"],
        cu_seqlens_padded=inputs["cu_seqlens_padded"],
        max_seqlen=inputs["max_seqlen"],
        local_cp_size=None,
        hybrid_cp_group=None,
        config=_static_config(),
        pad_token_id=91,
        context_parallel_size=1,
    )

    for name, tensor in inputs.items():
        assert torch.equal(tensor, originals[name]), name


def test_context_parallel_packed_batch_points_to_dynamic_cp_follow_up() -> None:
    with pytest.raises(ValueError, match="DynamicCP.*separate follow-up"):
        _prepare(context_parallel_size=2)


def test_static_thd_overflow_policy_eager_preserves_unpadded_batch() -> None:
    """An over-capacity packed batch can run eagerly instead of aborting the step."""
    inputs = _packed_inputs()
    inputs["cu_seqlens"] = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    inputs["cu_seqlens_padded"] = inputs["cu_seqlens"].clone()
    inputs["tokens"] = torch.arange(5, dtype=torch.int64).unsqueeze(0)
    inputs["labels"] = torch.arange(5, dtype=torch.int64).unsqueeze(0)
    inputs["loss_mask"] = torch.ones((1, 5))
    inputs["position_ids"] = torch.arange(5, dtype=torch.int64).unsqueeze(0)
    config = _static_config()
    config.thd_overflow_policy = "eager"

    _, (tokens, labels, loss_mask, position_ids, packed, padding_mask) = _prepare(
        config=config, inputs=inputs
    )

    assert torch.equal(tokens, inputs["tokens"])
    assert torch.equal(labels, inputs["labels"])
    assert torch.equal(loss_mask, inputs["loss_mask"])
    assert torch.equal(position_ids, inputs["position_ids"])
    assert packed.cu_seqlens_q.tolist() == [0, 1, 2, 3, 4, 5]
    assert packed.cuda_graph_eligible is False
    assert padding_mask is None


def test_eager_packed_batch_preserves_existing_shapes_and_values() -> None:
    config = SimpleNamespace(
        cuda_graph_impl="none",
        max_seqlen_per_dp_cp_rank=None,
        pad_packed_seq_alignment=None,
        thd_max_packed_sequences=None,
        thd_tail_padding_policy=None,
    )

    inputs, (tokens, labels, loss_mask, position_ids, packed, padding_mask) = _prepare(
        config=config
    )

    assert torch.equal(tokens, inputs["tokens"])
    assert torch.equal(labels, inputs["labels"])
    assert torch.equal(loss_mask, inputs["loss_mask"])
    assert torch.equal(position_ids, inputs["position_ids"])
    assert packed.cu_seqlens_q.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_kv.tolist() == [0, 3, 5]
    assert packed.cu_seqlens_q_padded.tolist() == [0, 4, 8]
    assert packed.cu_seqlens_kv_padded.tolist() == [0, 4, 8]
    assert packed.total_tokens == 8
    assert packed.tokens_per_sample == 8
    assert padding_mask is None


def _patch_forward_dependencies(
    monkeypatch: pytest.MonkeyPatch, batch: tuple[Any, ...], config: SimpleNamespace
) -> None:
    monkeypatch.setattr(pretrain_hybrid, "get_batch", lambda *_args: batch)
    monkeypatch.setattr(
        pretrain_hybrid, "get_timers", lambda: lambda *_args, **_kwargs: _NoOpTimer()
    )
    monkeypatch.setattr(
        pretrain_hybrid, "get_attr_wrapped_model", lambda model, name: getattr(model, name)
    )
    monkeypatch.setattr(
        pretrain_hybrid, "get_args", lambda: SimpleNamespace(context_parallel_size=1)
    )
    monkeypatch.setattr(pretrain_hybrid, "core_transformer_config_from_args", lambda _args: config)
    monkeypatch.setattr(pretrain_hybrid, "get_tokenizer", lambda: SimpleNamespace(pad=91))
    monkeypatch.setattr(pretrain_hybrid, "update_seqlen_stats_from_cu_seqlens", lambda _cu: None)
    monkeypatch.setattr(pretrain_hybrid, "stimer", _NoOpStragglerTimer())


def test_forward_step_propagates_boolean_padding_and_zero_loss_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _packed_inputs()
    batch = (
        None,
        inputs["cu_seqlens"].unsqueeze(0),
        inputs["cu_seqlens_padded"].unsqueeze(0),
        None,
        inputs["labels"],
        None,
        inputs["loss_mask"],
        inputs["max_seqlen"],
        inputs["position_ids"],
        inputs["tokens"],
    )
    _patch_forward_dependencies(monkeypatch, batch, _static_config())
    monkeypatch.setattr(
        pretrain_hybrid,
        "loss_func",
        lambda loss_mask, _output_tensor, model=None: (loss_mask, model),
    )
    model = _RecordingModel()

    output, loss_closure = pretrain_hybrid.forward_step(iter(()), model)

    assert model.kwargs is not None
    assert model.kwargs["padding_mask"].dtype == torch.bool
    assert model.kwargs["padding_mask"].tolist() == [[False] * 8 + [True] * 4]
    assert model.kwargs["loss_mask"].tolist()[0][-4:] == [0.0, 0.0, 0.0, 0.0]
    callback_loss_mask, callback_model = loss_closure(output)
    assert torch.equal(callback_loss_mask, model.kwargs["loss_mask"])
    assert callback_model is model


def test_forward_step_preserves_non_packed_model_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    labels = torch.tensor([[2, 3, 4]], dtype=torch.int64)
    loss_mask = torch.ones((1, 3))
    position_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    batch = (None, None, None, None, labels, None, loss_mask, None, position_ids, tokens)
    _patch_forward_dependencies(monkeypatch, batch, _static_config())
    model = _RecordingModel()

    pretrain_hybrid.forward_step(iter(()), model)

    assert model.args is not None
    assert model.kwargs is not None
    assert torch.equal(model.args[0], tokens)
    assert torch.equal(model.args[1], position_ids)
    assert model.kwargs["packed_seq_params"] is None
    assert model.kwargs["padding_mask"] is None
    assert torch.equal(model.kwargs["loss_mask"], loss_mask)
