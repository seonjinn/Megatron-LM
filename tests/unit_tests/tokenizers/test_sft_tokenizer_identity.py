# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


IDENTITY_TEMPLATE = (
    """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
)
IGNORE_INDEX = -100


class FakeLegacyTokenizer:
    """Minimal base class needed to instantiate SFTTokenizer."""

    def __init__(self, *tokenizer_paths: str, **tokenizer_options: Any) -> None:
        pass


@dataclass
class FakePromptConfig:
    assistant_prefix_len: int
    pad_token_id: int
    custom_chat_template: str
    has_bos: bool
    has_system_role: bool
    force_system_message: bool = False
    system_default: dict[str, Any] | None = None


class FakeHuggingFaceTokenizer:
    unk_token_id = 17
    bos_token_id = None
    eos_token_id = 2

    def __len__(self) -> int:
        return 256

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<unk>"
        return self.unk_token_id

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        chat_template: str,
        add_generation_prompt: bool = False,
        return_assistant_token_mask: bool = False,
        return_tensors: str | None = None,
    ) -> list[int] | np.ndarray:
        assert tokenize
        assert chat_template == IDENTITY_TEMPLATE
        rendered = "".join(message["content"] for message in conversation)
        tokens = [ord(character) for character in rendered]
        if return_tensors == "np":
            return np.array([tokens], dtype=np.int64)
        return tokens

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def get_added_vocab(self) -> dict[str, int]:
        return {}


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(**kwargs: Any) -> FakeHuggingFaceTokenizer:
        assert kwargs == {"pretrained_model_name_or_path": "unused"}
        return FakeHuggingFaceTokenizer()


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def sft_tokenizer_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for package_name in (
        "megatron",
        "megatron.core",
        "megatron.core.datasets",
        "megatron.training",
        "megatron.training.datasets",
        "megatron.training.tokenizer",
    ):
        package = _stub_module(monkeypatch, package_name)
        package.__path__ = []

    legacy_module = _stub_module(
        monkeypatch, "megatron.core.datasets.megatron_tokenizer"
    )
    legacy_module.MegatronLegacyTokenizer = FakeLegacyTokenizer

    dataset_module = _stub_module(monkeypatch, "megatron.training.datasets.sft_dataset")
    dataset_module.IGNORE_INDEX = IGNORE_INDEX

    multimodal_module = _stub_module(
        monkeypatch, "megatron.training.tokenizer.multimodal_tokenizer"
    )
    multimodal_module.PromptConfig = FakePromptConfig

    transformers_module = _stub_module(monkeypatch, "transformers")
    transformers_module.AutoTokenizer = FakeAutoTokenizer

    module_path = (
        Path(__file__).parents[3] / "megatron/training/tokenizer/sft_tokenizer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sft_tokenizer_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_tokenizer(sft_tokenizer_module: ModuleType) -> Any:
    return sft_tokenizer_module.SFTTokenizer("unused", "identity")


def test_identity_initialization_uses_reference_prompt_config(
    sft_tokenizer_module: ModuleType,
) -> None:
    identity_tokenizer = _identity_tokenizer(sft_tokenizer_module)
    config = identity_tokenizer._prompt_config

    assert config.assistant_prefix_len == 0
    assert config.pad_token_id == FakeHuggingFaceTokenizer.unk_token_id
    assert config.custom_chat_template == IDENTITY_TEMPLATE
    assert config.has_bos is False
    assert config.has_system_role is True


def test_identity_concatenates_exact_message_content(
    sft_tokenizer_module: ModuleType,
) -> None:
    identity_tokenizer = _identity_tokenizer(sft_tokenizer_module)
    conversation = [
        {"role": "system", "content": " system\n"},
        {"role": "user", "content": "user "},
        {"role": "assistant", "content": " assistant"},
    ]

    tokens = identity_tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=False
    )

    expected_text = " system\nuser  assistant"
    np.testing.assert_array_equal(
        tokens, [ord(character) for character in expected_text]
    )


def test_identity_masks_system_and_user_targets(
    sft_tokenizer_module: ModuleType,
) -> None:
    identity_tokenizer = _identity_tokenizer(sft_tokenizer_module)
    conversation = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "answer"},
    ]

    tokens, targets = identity_tokenizer.tokenize_conversation(
        conversation, return_target=True, add_generation_prompt=False
    )

    expected_tokens = [ord(character) for character in "syspromptanswer"]
    expected_targets = [IGNORE_INDEX] * len("sysprompt") + [
        ord(character) for character in "answer"
    ]
    np.testing.assert_array_equal(tokens, expected_tokens)
    np.testing.assert_array_equal(targets, expected_targets)


def test_identity_adds_no_generation_prompt(sft_tokenizer_module: ModuleType) -> None:
    identity_tokenizer = _identity_tokenizer(sft_tokenizer_module)
    conversation = [{"role": "user", "content": "prompt"}]

    without_generation_prompt = identity_tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=False
    )
    with_generation_prompt = identity_tokenizer.tokenize_conversation(
        conversation, return_target=False, add_generation_prompt=True
    )

    np.testing.assert_array_equal(with_generation_prompt, without_generation_prompt)
    np.testing.assert_array_equal(
        with_generation_prompt, [ord(character) for character in "prompt"]
    )
