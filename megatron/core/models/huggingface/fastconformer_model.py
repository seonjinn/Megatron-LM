# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from megatron.core.models.huggingface import HuggingFaceModule
from megatron.core.models.huggingface.fastconformer.modeling_fastconformer import FastConformerModel
from megatron.core.models.huggingface.fastconformer.feature_extraction_fastconformer import FastConformerFeatureExtractor
import torch

class ParakeetHuggingFaceModel(HuggingFaceModule):
    """
    Wrapper for Parakeet based on HF FastConformer model
    """

    def __init__(self, config):
        super().__init__(config)
        # TODO(jbarker): This is a hack to load the model from a local directory.
        # We should load from an openly available source.
        self.feature_extractor = FastConformerFeatureExtractor.from_pretrained(config.sound_model_type.split("hf://")[1])
        self.model = FastConformerModel.from_pretrained(config.sound_model_type.split("hf://")[1])

    def forward(self, *args, **kwargs):
        """Forward function"""
        # This is the sampling rate of the input audio file, not a target sampling rate
        # so it's correct that it's fixed for this model.
        # device = self.
        # model.device
        # x = self.feature_extractor(*args, **kwargs, return_tensors="pt", sampling_rate=16000)
        x = self.feature_extractor(*args, **kwargs, return_tensors="pt", sampling_rate=16000, return_attention_mask=True)
        x = self.model(x.input_features.to(torch.bfloat16), x.attention_mask)
        return x.last_hidden_state