# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from examples.multimodal.data_loading.task_encoder import _normalize_thinking_trace


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>reasoning</think>answer", "<think>\nreasoning</think>answer"),
        (
            "<think>\n  multi-line\nreasoning  \n</think>\n\nanswer",
            "<think>\nmulti-line\nreasoning</think>answer",
        ),
        ("<think>  </think>\nanswer", "<think></think>answer"),
    ],
)
def test_normalize_thinking_trace_ultra(content, expected):
    assert (
        _normalize_thinking_trace(
            content,
            prompt_format="nemotron6-moe",
            thinking_trace_format="ultra",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("prompt_format", "expected_separator"),
    [
        ("nemotron6-moe", "\n"),
        ("nemotron-h-5p5-reasoning", "\n\n"),
    ],
)
def test_normalize_thinking_trace_preserves_existing_format(
    prompt_format, expected_separator
):
    assert _normalize_thinking_trace(
        "<think>reasoning</think>answer",
        prompt_format=prompt_format,
        thinking_trace_format="normalized",
    ) == f"<think>\nreasoning\n</think>{expected_separator}answer"
