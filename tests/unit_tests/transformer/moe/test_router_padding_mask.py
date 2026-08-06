import pytest
import torch

from megatron.core.transformer.moe.router import (
    _prepare_padding_mask_for_routing,
)


@pytest.mark.parametrize("mask_shape", [(8,), (4, 2), (1, 8)])
def test_router_padding_mask_is_column_broadcastable(mask_shape):
    padding_mask = torch.zeros(mask_shape, dtype=torch.bool)
    routing_map = torch.zeros((8, 128), dtype=torch.bool)

    prepared = _prepare_padding_mask_for_routing(padding_mask, routing_map)

    assert prepared.shape == (8, 1)
    assert prepared.dtype is torch.bool


def test_router_padding_mask_rejects_token_count_mismatch():
    padding_mask = torch.zeros((7,), dtype=torch.bool)
    routing_map = torch.zeros((8, 128), dtype=torch.bool)

    with pytest.raises(ValueError, match="padding_mask has 7 tokens"):
        _prepare_padding_mask_for_routing(padding_mask, routing_map)
