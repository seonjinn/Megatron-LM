# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal import context_parallel, llava_model
from megatron.core.models.multimodal.llava_model import (
    LLaVAModel,
    _split_image_tiles_by_sample,
    pad_sequence_lengths_for_context_parallel,
)
from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from megatron.training.global_vars import set_args
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
def test_pad_sequence_lengths_for_context_parallel_rounds_each_sample():
    """HybridCP must pad every routed sample to the CP shard requirement."""
    lengths = torch.tensor([13, 16, 17], dtype=torch.int32)

    padded = pad_sequence_lengths_for_context_parallel(lengths, shard_factor=8)

    assert torch.equal(padded, torch.tensor([16, 16, 24], dtype=torch.int32))
    assert padded.dtype == lengths.dtype
    assert padded.device == lengths.device


@pytest.mark.internal
def test_split_image_tiles_by_sample_preserves_marker_alignment():
    per_sample = _split_image_tiles_by_sample(
        torch.tensor([2, 1, 3], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int64),
    )

    assert [tiles.tolist() for tiles in per_sample] == [[2], [1, 3]]


@pytest.mark.internal
def test_split_image_tiles_by_sample_rejects_misaligned_metadata():
    with pytest.raises(
        ValueError,
        match="3 num_image_tiles entries for 2 image markers",
    ):
        _split_image_tiles_by_sample(
            torch.tensor([2, 1, 3], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int64),
        )


@pytest.mark.internal
def test_process_embedding_token_parallel_keeps_standard_cp1_layout(monkeypatch):
    """CP1 preprocessing must not transpose the already sequence-first input again."""
    model = LLaVAModel.__new__(LLaVAModel)
    model.pre_process = True
    model.post_process = False
    model.context_parallel_lm = 1
    model.sequence_parallel_lm = True
    model.tensor_model_parallel_size_lm = 8
    model.tp_comm_overlap_lm = False
    model.cp_group = None

    combined_embeddings = torch.ones(8, 2, 4)
    monkeypatch.setattr(
        tensor_parallel,
        "scatter_to_sequence_parallel_region",
        lambda tensor: tensor,
    )

    result = model._process_embedding_token_parallel(
        combined_embeddings,
        None,
        None,
        None,
        None,
        None,
    )

    assert result[0].shape == (8, 2, 4)


@pytest.mark.internal
def test_compact_packed_media_graph_inputs_removes_only_raw_token_tail():
    tokens = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    labels = torch.arange(100, 109, dtype=torch.int64).unsqueeze(0)
    original_tokens = tokens.clone()
    original_labels = labels.clone()

    compact_tokens, compact_labels = llava_model.compact_packed_multimodal_graph_inputs(
        tokens,
        labels,
        [[2, 2]],
    )

    assert {
        "tokens": compact_tokens.tolist(),
        "labels": compact_labels.tolist(),
        "tokens_contiguous": compact_tokens.is_contiguous(),
        "labels_contiguous": compact_labels.is_contiguous(),
        "caller_tokens": tokens.tolist(),
        "caller_labels": labels.tolist(),
    } == {
        "tokens": [[0, 1, 2, 3]],
        "labels": [[100, 101, 102, 103, 104]],
        "tokens_contiguous": True,
        "labels_contiguous": True,
        "caller_tokens": original_tokens.tolist(),
        "caller_labels": original_labels.tolist(),
    }


@pytest.mark.internal
@pytest.mark.parametrize(
    "token_shape,label_shape,sample_token_lengths,error_match",
    [
        ((1, 8), (1, 9), [[]], "cannot be empty"),
        ((1, 8), (1, 9), [[2, 0]], "must be positive"),
        ((1, 8), (1, 9), [[5, 4]], "exceed the raw token width"),
        ((1, 8), (1, 8), [[2, 2]], "one more token than tokens"),
    ],
)
def test_compact_packed_media_graph_inputs_rejects_invalid_raw_boundaries(
    token_shape: tuple[int, int],
    label_shape: tuple[int, int],
    sample_token_lengths: list[list[int]],
    error_match: str,
):
    with pytest.raises(ValueError, match=error_match):
        llava_model.compact_packed_multimodal_graph_inputs(
            torch.zeros(token_shape, dtype=torch.int64),
            torch.zeros(label_shape, dtype=torch.int64),
            sample_token_lengths,
        )


@pytest.mark.internal
def test_metadata_only_static_padding_retains_compact_media_inputs():
    compact_cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=compact_cu_seqlens,
        cu_seqlens_kv=compact_cu_seqlens.clone(),
        cu_seqlens_q_padded=compact_cu_seqlens.clone(),
        cu_seqlens_kv_padded=compact_cu_seqlens.clone(),
        max_seqlen_q=6,
        max_seqlen_kv=6,
        total_tokens=6,
    )

    tokens, labels, loss_mask, position_ids, padded, padding_mask = pad_sequence_for_thd(
        None,
        None,
        None,
        None,
        packed_seq_params,
        target_len=8,
        max_num_seqs=3,
        tail_padding_policy="extend_last",
        cp_size=1,
    )

    assert {
        "token_tensors": (tokens, labels, loss_mask, position_ids),
        "cu_seqlens_q": padded.cu_seqlens_q.tolist(),
        "cu_seqlens_q_padded": padded.cu_seqlens_q_padded.tolist(),
        "total_tokens": padded.total_tokens,
        "seq_idx_tokens": padded.seq_idx.numel(),
        "padding_mask": padding_mask.tolist(),
    } == {
        "token_tensors": (None, None, None, None),
        "cu_seqlens_q": [0, 6, 6, 6],
        "cu_seqlens_q_padded": [0, 8, 8, 8],
        "total_tokens": 8,
        "seq_idx_tokens": 8,
        "padding_mask": [[False, False, False, False, False, False, True, True]],
    }


class TestLLaVAModel:
    @pytest.mark.internal  # The model is under active development and its methods may change.
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.language_hidden_size = 64
        self.language_num_attention_heads = 4

        self.model = self._build_model()

    def _build_model(self, allow_llm_only_checkpoint=False):
        language_config = TransformerConfig(
            num_layers=3,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=False,
        )
        vision_config = TransformerConfig(
            num_layers=2, hidden_size=16, num_attention_heads=2, use_cpu_initialization=False
        )
        vision_projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=False,
        )

        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        vision_layer_spec = ModuleSpec(
            module=TransformerLayer, submodules=deepcopy(language_layer_submodules)
        )
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)

        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "clip"
        model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            allow_llm_only_checkpoint=allow_llm_only_checkpoint,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )
        return model

    @pytest.mark.internal
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.model, LLaVAModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1488736

    @pytest.mark.internal
    def test_thd_graph_token_capacity(self):
        """Static TE THD graphs retain the full language token surface."""
        original_language_model = self.model.language_model
        self.model.language_model = SimpleNamespace(
            config=SimpleNamespace(
                cuda_graph_impl="transformer_engine",
                pad_packed_seq_alignment="max",
                max_seqlen_per_dp_cp_rank=1024,
            )
        )
        self.model.context_parallel_lm = 2

        assert (
            self.model._get_thd_graph_token_capacity(PackedSeqParams(qkv_format="thd")) == 2048
        )
        assert (
            self.model._get_thd_graph_token_capacity(
                PackedSeqParams(qkv_format="thd", cuda_graph_eligible=False)
            )
            is None
        )
        assert self.model._get_thd_graph_token_capacity(PackedSeqParams(qkv_format="sbhd")) is None

        self.model.language_model.config.pad_packed_seq_alignment = 128
        assert self.model._get_thd_graph_token_capacity(PackedSeqParams(qkv_format="thd")) is None

        self.model.language_model = original_language_model

    @pytest.mark.internal
    def test_set_input_tensor(self):
        expected_shape = (1, 2, 3, 4)
        input_tensor = torch.zeros(expected_shape)
        self.model.set_input_tensor(input_tensor)
        assert self.model.vision_model.decoder.input_tensor.shape == expected_shape

    @pytest.mark.internal
    def test_preprocess_data(self):
        self.model.cuda()

        hidden_size = 72

        # 3 images with 1 tile and 2 image with 2 tiles = 7 tiles.
        image_embeddings = (
            torch.arange(577 * 7 * hidden_size, dtype=torch.float)
            .reshape(577, 7, hidden_size)
            .cuda()
        )

        image_token_index = self.model.image_token_index
        input_ids = torch.arange(1024).expand(5, 1024).cuda()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image
        input_ids[4, 50] = image_token_index  # two images in between
        input_ids[4, 150] = image_token_index

        # Using negative sign to distinguish from image embeddings.
        language_embeddings = (
            -torch.arange(5 * 1024 * hidden_size, dtype=torch.float)
            .reshape(5, 1024, hidden_size)
            .cuda()
        )

        # Labels are input_ids shifted to left by one.
        labels = torch.arange(1, 1025, dtype=torch.int).expand(5, 1024).cuda()
        # labels[0] - image token got dropped by shift to left by one.
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index
        # labels[3] - no image.
        labels[4, 49] = image_token_index
        labels[4, 149] = image_token_index

        loss_mask = torch.ones((5, 1024), dtype=torch.float).cuda()
        # Mask some text inputs (the text mask should carry over)
        loss_mask[:2, :10] = 0.0
        loss_mask[:2, 110:120] = 0.0

        # Number of tiles for each image in the batch.
        num_image_tiles = torch.tensor([1, 2, 1, 2, 1], dtype=torch.int).cuda()

        use_inference_kv_cache = False
        inference_context = None

        embeddings, labels, loss_mask = self.model._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            inference_context,
            image_token_index,
            num_image_tiles,
        )

        img_seq_len = 577
        # The fifth sample has 2 images with 3 tiles and 1024 text tokens.
        max_seq_len = 3 * img_seq_len - 2 + 1024

        assert embeddings.shape == torch.Size((max_seq_len, 5, hidden_size))
        assert labels.shape == torch.Size((5, max_seq_len))
        assert loss_mask.shape == labels.shape

        # First sample where image is before text (index 0).
        expected_embeddings = torch.empty(max_seq_len, hidden_size).cuda()
        expected_embeddings[:577] = image_embeddings[:, 0]
        expected_embeddings[577:1600] = language_embeddings[0, 1:]
        expected_embeddings[1600:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:576] = -100  # image
        expected_labels[576:1600] = torch.arange(1, 1025, dtype=torch.int)
        expected_labels[1600:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:577] = 0
        expected_loss_mask[577:586] = 0
        expected_loss_mask[586:686] = 1
        expected_loss_mask[686:696] = 0
        expected_loss_mask[696:1600] = 1
        expected_loss_mask[1600:] = 0

        assert torch.allclose(embeddings[:, 0], expected_embeddings)
        assert torch.allclose(labels[0], expected_labels)
        assert torch.allclose(loss_mask[0], expected_loss_mask)

        # Second sample where image is in between (index 100). The image has 2 tiles.
        expected_embeddings = torch.empty(max_seq_len, hidden_size).cuda()
        expected_embeddings[:100] = language_embeddings[1, :100]
        expected_embeddings[100:677] = image_embeddings[:, 1]
        expected_embeddings[677:1254] = image_embeddings[:, 2]
        expected_embeddings[1254:2177] = language_embeddings[1, 101:]
        expected_embeddings[2177:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:99] = torch.arange(1, 100)
        expected_labels[99:1253] = -100  # image
        expected_labels[1253:2177] = torch.arange(101, 1025)
        expected_labels[2177:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:10] = 0
        expected_loss_mask[10:99] = 1
        # Last text position before the image is not required to predict the first image embedding.
        expected_loss_mask[99] = 0
        expected_loss_mask[100:1254] = 0
        expected_loss_mask[1254:1263] = 1
        expected_loss_mask[1263:1273] = 0
        expected_loss_mask[1273:2177] = 1
        expected_loss_mask[2177:] = 0  # padding

        assert torch.allclose(embeddings[:, 1], expected_embeddings)
        assert torch.allclose(labels[1], expected_labels)
        assert torch.allclose(loss_mask[1], expected_loss_mask)

        # Third sample where image is at the end.
        expected_embeddings = torch.empty(max_seq_len, hidden_size).cuda()
        expected_embeddings[:1023] = language_embeddings[2, :1023]
        expected_embeddings[1023:1600] = image_embeddings[:, 3]
        expected_embeddings[1600:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:1022] = torch.arange(1, 1023)
        expected_labels[1022:1599] = -100
        expected_labels[1599] = 1024
        expected_labels[1600:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:1022] = 1
        # Last text position before the image is not required to predict the first image embedding.
        expected_loss_mask[1022] = 0
        expected_loss_mask[1023:1600] = 0
        expected_loss_mask[1600:] = 0  # padding

        assert torch.allclose(embeddings[:, 2], expected_embeddings)
        assert torch.allclose(labels[2], expected_labels)
        assert torch.allclose(loss_mask[2], expected_loss_mask)

        # Fourth sample where there is no image.
        expected_embeddings = torch.empty(max_seq_len, hidden_size).cuda()
        expected_embeddings[:1024] = language_embeddings[3]
        expected_embeddings[1024:] = 0  # padding

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:1024] = torch.arange(1, 1025)
        expected_labels[1024:] = -100  # padding

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:1024] = 1
        expected_loss_mask[1024:] = 0  # padding

        assert torch.allclose(embeddings[:, 3], expected_embeddings)
        assert torch.allclose(labels[3], expected_labels)
        assert torch.allclose(loss_mask[3], expected_loss_mask)

        # Fifth sample has two images in between (indices 50 and 150). The first image has two tiles.
        expected_embeddings = torch.empty(max_seq_len, hidden_size).cuda()
        expected_embeddings[:50] = language_embeddings[4, :50]
        expected_embeddings[50:627] = image_embeddings[:, 4]  # two tiles
        expected_embeddings[627:1204] = image_embeddings[:, 5]
        expected_embeddings[1204:1303] = language_embeddings[4, 51:150]
        expected_embeddings[1303:1880] = image_embeddings[:, 6]
        expected_embeddings[1880:] = language_embeddings[4, 151:]

        expected_labels = torch.empty(max_seq_len, dtype=torch.int).cuda()
        expected_labels[:49] = torch.arange(1, 50)
        expected_labels[49:1203] = -100  # image
        expected_labels[1203:1302] = torch.arange(51, 150)
        expected_labels[1302:1879] = -100  # image
        expected_labels[1879:] = torch.arange(151, 1025)

        expected_loss_mask = torch.empty(max_seq_len, dtype=torch.float).cuda()
        expected_loss_mask[:49] = 1
        expected_loss_mask[49:1204] = 0
        expected_loss_mask[1204:1302] = 1
        expected_loss_mask[1302:1880] = 0
        expected_loss_mask[1880:] = 1

        assert torch.allclose(embeddings[:, 4], expected_embeddings)
        assert torch.allclose(labels[4], expected_labels)
        assert torch.allclose(loss_mask[4], expected_loss_mask)

    @pytest.mark.internal
    def test_forward(self):
        self.model.cuda()

        # 3 images with 1 tile and 2 images with 2 tiles.
        img = torch.randn((7, 3, 336, 336)).cuda()

        image_token_index = self.model.image_token_index
        input_ids = torch.randint(0, 2048, (5, 1024)).cuda()
        input_ids[0, 0] = image_token_index  # image before text
        input_ids[1, 100] = image_token_index  # image in between
        input_ids[2, -1] = image_token_index  # image at the end
        # input_ids[3] - no image
        input_ids[4, 50] = image_token_index
        input_ids[4, 150] = image_token_index

        position_ids = torch.arange(0, 1024, dtype=torch.int).expand(5, 1024).cuda()

        loss_mask = torch.ones((5, 1024)).cuda()

        attention_mask = None  # Causal.

        labels = torch.randint(0, 2048, (5, 1024)).cuda()
        labels[1, 99] = image_token_index
        labels[2, -2] = image_token_index

        num_image_tiles = torch.tensor([1, 2, 1, 2, 1], dtype=torch.int).cuda()

        # Try with labels.
        loss, new_loss_mask = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # The maximum sequence length is given by the sample with 2 images in 3 tiles, minus two image token indices, plus other text tokens.
        img_seq_len = 577
        max_seq_len = img_seq_len * 3 - 2 + 1024
        assert loss.shape == new_loss_mask.shape == torch.Size((5, max_seq_len))

        # Try with labels and PackedSeqParams. Only micro batch size 1 is supported in this mode.
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(
                [0, 512, 1024, 1600], dtype=torch.int32
            ).cuda(),  # Just example values.
            cu_seqlens_kv=torch.tensor([0, 512, 1024, 1600], dtype=torch.int32).cuda(),
            max_seqlen_q=1600,
            max_seqlen_kv=1600,
        )

        # NOTE: Packing is only supported with BF16. Use BF16 here and switch back to default.
        self.model.to(torch.bfloat16)
        loss, new_loss_mask = self.model.forward(
            img[:1].to(torch.bfloat16),
            input_ids[:1],
            position_ids[:1],
            attention_mask,
            labels[:1],
            loss_mask[:1],
            num_image_tiles=num_image_tiles[:1],
            packed_seq_params=packed_seq_params,
        )
        self.model.to(torch.float32)

        # 1600 = 577 (img_seq_len) + 1024 (text tokens in the first sample) - 1 (image token).
        assert loss.shape == new_loss_mask.shape == torch.Size((1, 1600))

        # Try text-only input.
        loss, new_loss_mask = self.model.forward(
            torch.tensor([], dtype=torch.float).cuda(),
            torch.randint(0, 2048, (5, 1024)).cuda(),
            position_ids,
            attention_mask,
            torch.randint(0, 2048, (5, 1024)).cuda(),
            loss_mask,
            num_image_tiles=torch.tensor([], dtype=torch.int).cuda(),
        )

        assert loss.shape == new_loss_mask.shape == torch.Size((5, 1024))

        # Try without labels and without inference params.
        logits, _ = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels=None,
            loss_mask=None,
            num_image_tiles=num_image_tiles,
        )
        assert logits.shape == torch.Size((5, max_seq_len, 8192))

        # Try without labels and with inference params.
        inference_context = StaticInferenceContext(5, max_seq_len)
        with InferenceMode.active():
            logits, _ = self.model.forward(
                img,
                input_ids,
                position_ids,
                attention_mask,
                labels=None,
                loss_mask=None,
                num_image_tiles=num_image_tiles,
                inference_context=inference_context,
                runtime_gather_output=True,
            )
        # StaticInferenceContext always sets materialize_only_last_token_logits=True.
        assert logits.shape == torch.Size((5, 1, 8192))

        # Check KV cache got populated correctly.
        kv_dict = inference_context.key_value_memory_dict

        assert kv_dict["image_tokens_count"] == 577 * 7
        for layer_no in range(1, 4):  # 3 layers in the model.
            layer_kv = kv_dict[layer_no]
            # Expected shape is [sequence_len, batch_size, num_heads, hidden_size_per_head]
            assert (
                layer_kv[0].shape
                == layer_kv[1].shape
                == torch.Size((max_seq_len, 5, self.language_num_attention_heads, 16))
            )

    @pytest.mark.internal
    def test_forward_fsdp(self):
        """Test FSDP workaround for text-only data.

        FSDP can hang with text-only data. As a workaround, we run the vision model with a dummy image,
        but then effectively discard the image embeddings.
        """
        self.model.cuda()

        # Dummy image for the FSDP workaround but not image tiles.
        img = torch.zeros((1, 3, 336, 336)).cuda()
        num_image_tiles = torch.tensor([], dtype=torch.int).cuda()

        # No image tag in the input ids (text-only sample).
        image_token_index = self.model.image_token_index
        input_ids = torch.arange(1024, device="cuda").unsqueeze(0)
        assert (
            torch.sum(input_ids == image_token_index) == 0
        ), "expected no image tag in the input ids"

        position_ids = torch.arange(1024, device="cuda").unsqueeze(0)

        loss_mask = torch.ones((1, 1024), device="cuda")

        attention_mask = None  # Causal.

        labels = torch.arange(1, 1025, device="cuda").unsqueeze(0)

        # Mock the FSDP attribute.
        self.model.vision_model._is_fsdp_managed_module = True
        loss, new_loss_mask = self.model.forward(
            img,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            num_image_tiles=num_image_tiles,
        )
        self.model.vision_model._is_fsdp_managed_module = False

        assert loss.shape == new_loss_mask.shape == torch.Size((1, 1024))

    @pytest.mark.internal
    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))

    @pytest.mark.internal
    def test_llm_only_checkpoint_sharded_key_mapping_is_load_only(self):
        model = self._build_model(allow_llm_only_checkpoint=True)

        save_state_dict = model.sharded_state_dict()
        assert (
            save_state_dict["language_model.embedding.word_embeddings.weight"].key
            == "language_model.embedding.word_embeddings.weight"
        )

        load_state_dict = model.sharded_state_dict(
            metadata={"load_from_llm_only_checkpoint": True}
        )
        assert (
            load_state_dict["language_model.embedding.word_embeddings.weight"].key
            == "embedding.word_embeddings.weight"
        )
        assert "vision_model.class_token" not in load_state_dict

    @pytest.mark.internal
    def test_llm_only_checkpoint_allows_missing_vision_modules(self):
        model = self._build_model(allow_llm_only_checkpoint=True)
        language_state_dict = {
            f"language_model.{name}": tensor
            for name, tensor in model.language_model.state_dict().items()
        }

        model.load_state_dict(language_state_dict, strict=True)

    @pytest.mark.internal
    def test_freeze(self):
        self.model.freeze(
            freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=False
        )

        for module in [self.model.language_model, self.model.vision_model]:
            for param in module.parameters():
                assert not param.requires_grad

        for param in self.model.vision_projection.parameters():
            assert param.requires_grad


@pytest.fixture(scope='class', params=["siglip", "radio-g"])
def setup_and_teardown_llava_model(request):
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    language_config = TransformerConfig(
        num_layers=3, hidden_size=128, num_attention_heads=8, use_cpu_initialization=False
    )
    vision_config = TransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=False
    )
    vision_projection_config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=72,
        num_attention_heads=1,
        use_cpu_initialization=False,
    )

    language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
    vision_layer_spec = ModuleSpec(
        module=TransformerLayer, submodules=deepcopy(language_layer_submodules)
    )
    vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))

    language_config.language_model_type = "dummy"
    vision_model_type = request.param
    vision_config.vision_model_type = vision_model_type
    model = LLaVAModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=language_layer_submodules
        ),
        language_vocab_size=2048,
        language_max_sequence_length=4096,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_layer_spec,
        drop_vision_class_token=False,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_spec,
        img_h=336,
        img_w=336,
        patch_dim=14,
    )

    yield model, vision_model_type

    Utils.destroy_model_parallel()


class TestLLaVAModelVisionEncoders:
    num_weights_by_encoder = {"siglip": 1832456, "radio-g": 2844552}

    @pytest.mark.internal
    def test_constructor(self, setup_and_teardown_llava_model):
        model, vision_model_type = setup_and_teardown_llava_model
        assert isinstance(model, LLaVAModel)

        num_weights = sum([p.numel() for p in model.parameters()])
        assert num_weights == self.num_weights_by_encoder[vision_model_type]

    @pytest.mark.internal
    def test_set_input_tensor(self, setup_and_teardown_llava_model):
        model, _ = setup_and_teardown_llava_model
        expected_shape = (1, 2, 3, 4)
        input_tensor = torch.zeros(expected_shape)
        model.set_input_tensor(input_tensor)
        assert model.vision_model.decoder.input_tensor.shape == expected_shape


def create_test_args(cp_size, sequence_parallel):
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.context_parallel_size = cp_size
    args.sequence_parallel = sequence_parallel

    return args


class TestLLaVAModelTokenParallel:

    def _init_llava_model(self, cp_size, tp_size, sequence_parallel):
        language_hidden_size = 64
        language_num_attention_heads = 16

        language_config = TransformerConfig(
            num_layers=3,
            hidden_size=language_hidden_size,
            num_attention_heads=language_num_attention_heads,
            use_cpu_initialization=False,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sequence_parallel,
            context_parallel_size=cp_size,
        )
        # SP and CP are not yet supported for the Vision Backbone
        vision_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=8,
            use_cpu_initialization=False,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=False,
            context_parallel_size=1,
        )
        vision_projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=language_hidden_size,
            ffn_hidden_size=128,
            num_attention_heads=8,
            use_cpu_initialization=False,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=False,
            context_parallel_size=1,
        )

        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        # SP/CP either requires user to ensure token lengths do not require padding OR change mask type to padding
        if (
            language_layer_submodules.self_attention.params.get('attn_mask_type', '')
            == AttnMaskType.causal
        ):
            language_layer_submodules.self_attention.params['attn_mask_type'] = (
                AttnMaskType.padding_causal
            )
        elif (
            language_layer_submodules.self_attention.params.get('attn_mask_type', '')
            == AttnMaskType.no_mask
        ):
            language_layer_submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding

        vision_layer_spec = ModuleSpec(
            module=TransformerLayer, submodules=deepcopy(language_layer_submodules)
        )
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))

        language_config.language_model_type = "dummy"
        vision_config.vision_model_type = "clip"
        model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )

        return model

    def _prepare_inputs(self, cp_size, tp_size, sequence_parallel, padding):
        self.batch_size = 2
        if padding:
            self.combined_valid_seqlen = 2049
            self.combined_padded_seqlen = 2064
        else:
            self.combined_valid_seqlen = 2048
            self.combined_padded_seqlen = 2048

        if cp_size > 1:
            combined_embeddings = torch.ones(
                [self.batch_size, self.combined_padded_seqlen, 4096],
                device='cuda',
                dtype=torch.bfloat16,
            )  # [B, S, H]
        else:
            combined_embeddings = torch.ones(
                [self.combined_padded_seqlen, self.batch_size, 4096],
                device='cuda',
                dtype=torch.bfloat16,
            )  # [S, B, H]
        new_labels = torch.ones(
            [self.batch_size, self.combined_padded_seqlen], device='cuda', dtype=torch.bfloat16
        )  # [B, S]
        new_loss_mask = torch.ones(
            [self.batch_size, self.combined_padded_seqlen], device='cuda', dtype=torch.bfloat16
        )  # [B, S]

        cu_seqlens = torch.arange(
            0,
            (self.batch_size + 1) * (self.combined_valid_seqlen),
            step=(self.combined_valid_seqlen),
            dtype=torch.int32,
            device=combined_embeddings.device,
        )
        cu_seqlens_padded = torch.arange(
            0,
            (self.batch_size + 1) * (self.combined_padded_seqlen),
            step=(self.combined_padded_seqlen),
            dtype=torch.int32,
            device=combined_embeddings.device,
        )

        qkv_format = 'sbhd'  # Default format when not using padding
        if cp_size > 1 and padding:
            # Reshape from [B,S] to [1,T]
            combined_embeddings = (
                combined_embeddings.contiguous()
                .view(combined_embeddings.shape[0] * combined_embeddings.shape[1], -1)
                .unsqueeze(0)
            )
            new_labels = new_labels.view(new_labels.shape[0] * new_labels.shape[1]).unsqueeze(0)
            new_loss_mask = new_loss_mask.view(
                new_loss_mask.shape[0] * new_loss_mask.shape[1]
            ).unsqueeze(0)
            qkv_format = 'thd'

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=self.combined_padded_seqlen,
            max_seqlen_kv=self.combined_padded_seqlen,
            qkv_format=qkv_format,
        )

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

    @pytest.mark.internal
    def setup_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "cp_size,tp_size,sequence_parallel,padding",
        [(1, 8, True, True), (2, 4, False, True), (2, 4, True, False), (2, 4, True, True)],
    )
    def test_process_embedding_token_parallel(self, cp_size, tp_size, sequence_parallel, padding):
        """Test _process_embedding_token_parallel.

        Note: This test requires TE version >= 1.10.0 to run properly.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # TE version must be at least 1.10.0 if using context parallelism. Exit otherwise.
        ctx = (
            nullcontext()
            if (is_te_min_version("1.10.0") or cp_size <= 1)
            else pytest.raises(AssertionError)
        )
        model = None
        with ctx:
            model = self._init_llava_model(cp_size, tp_size, sequence_parallel)

        if model is None:
            return

        model.cuda()

        args = create_test_args(cp_size, sequence_parallel)
        set_args(args)

        combined_embeddings, new_labels, new_loss_mask, packed_seq_params = self._prepare_inputs(
            cp_size, tp_size, sequence_parallel, padding
        )

        combined_embeddings, new_labels, new_loss_mask, packed_seq_params = (
            model._process_embedding_token_parallel(
                combined_embeddings, new_labels, new_loss_mask, packed_seq_params
            )
        )

        # Check if output shape is as expected
        if cp_size > 1 and sequence_parallel:
            if padding:
                # THD format
                assert combined_embeddings.shape[0] == self.batch_size * (
                    self.combined_padded_seqlen / (tp_size * cp_size)
                )
                assert combined_embeddings.shape[1] == 1
            else:
                # SBHD format
                assert combined_embeddings.shape[0] == (
                    self.combined_padded_seqlen / (tp_size * cp_size)
                )
                assert combined_embeddings.shape[1] == self.batch_size
        elif cp_size > 1:
            if padding:
                # THD format
                assert combined_embeddings.shape[0] == self.batch_size * (
                    self.combined_padded_seqlen / cp_size
                )
                assert combined_embeddings.shape[1] == 1
            else:
                # SBHD format
                assert combined_embeddings.shape[0] == (self.combined_padded_seqlen / cp_size)
                assert combined_embeddings.shape[1] == self.batch_size
        else:
            # SBHD format
            assert combined_embeddings.shape[0] == self.combined_padded_seqlen / tp_size
            assert combined_embeddings.shape[1] == self.batch_size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.internal
@pytest.mark.parametrize(
    "cp_size, tp_size, has_sp, seq_len, fp8_enabled, expected_padding",
    [
        (1, 1, False, 99, False, 0),
        (2, 2, True, 99, False, 5),
        (2, 2, False, 99, False, 1),
        (1, 4, False, 99, True, 13),
    ],
)
def test_get_padding(cp_size, tp_size, has_sp, seq_len, fp8_enabled, expected_padding):
    """Test calculating padding for context parallel."""
    padding = context_parallel.get_padding(
        seq_len, cp_size, tp_size, has_sp, fp8_enabled=fp8_enabled
    )

    assert padding == expected_padding


@pytest.mark.internal
@pytest.mark.parametrize(
    "tokens, img_seq_len, padding_needed, cp_size, expected_seq_len",
    [(torch.ones((1, 100)), 100, 0, 2, 200), (torch.ones((1, 100)), 128, 1, 2, 227)],
)
def test_get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, expected_seq_len):
    """Test creating PackedSeqParams for context parallel."""
    packed_seq_params = context_parallel.get_packed_seq_params(
        tokens, img_seq_len, padding_needed, cp_size
    )

    assert torch.equal(
        packed_seq_params.cu_seqlens_q, torch.tensor([0, expected_seq_len], dtype=torch.int32)
    )

    if padding_needed > 0:
        padded_seq_len = tokens.shape[1] + img_seq_len
        assert torch.equal(
            packed_seq_params.cu_seqlens_q_padded,
            torch.tensor([0, padded_seq_len], dtype=torch.int32),
        )
        assert packed_seq_params.max_seqlen_q == padded_seq_len
