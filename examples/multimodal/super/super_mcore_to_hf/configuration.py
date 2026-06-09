# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .configuration_nemotron_h import NemotronHConfig
from .configuration_radio import RADIOConfig

logger = logging.get_logger(__name__)


class SoundConfig(PretrainedConfig):
    """Configuration for the sound/audio model (Parakeet encoder + projection)."""
    model_type = "parakeet"

    def __init__(
        self,
        # Parakeet encoder config
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 24,
        intermediate_size: int = 4096,
        conv_kernel_size: int = 9,
        feat_in: int = 128,  # Mel features
        num_mel_bins: int = None,
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        subsampling_conv_kernel_size: int = 3,
        subsampling_conv_stride: int = 2,
        # Projection config
        projection_hidden_size: int = 4096,
        projection_bias: bool = True,
        # Audio processing
        sampling_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.feat_in = feat_in
        self.num_mel_bins = feat_in if num_mel_bins is None else num_mel_bins
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride
        self.projection_hidden_size = projection_hidden_size
        self.projection_bias = projection_bias
        self.sampling_rate = sampling_rate


class NemotronH_Super_Omni_Reasoning_V3_Config(PretrainedConfig):
    model_type = 'NemotronH_Super_Omni_Reasoning_V3'
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        sound_config=None,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        ps_version='v1',
        image_tag_type="internvl",
        projector_hidden_size=20480,
        vit_hidden_size=1280,
        attn_implementation="flash_attention_2",
        video_pruning_rate: float = 0.0,
        video_temporal_patch_size: int = 2,
        # Sound/audio settings
        sound_context_token_id: int = None,
        sound_context_token: str = "<so_embedding>",
        **kwargs
    ):
        super().__init__(**kwargs)

        if vision_config is not None:
            self.vision_config = RADIOConfig(**vision_config)
        else:
            self.vision_config = RADIOConfig()

        # Handle both cases: when loading from JSON (llm_config is dict) and when called internally by transformers (llm_config is None)
        if llm_config is not None:
            self.llm_config = NemotronHConfig(**llm_config)
        else:
            self.llm_config = NemotronHConfig()

        # Sound/audio model configuration
        if sound_config is not None:
            self.sound_config = SoundConfig(**sound_config)
        else:
            self.sound_config = None  # Sound model is optional

        # Assign configuration values
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template  # TODO move out of here and into the tokenizer
        self.ps_version = ps_version  # Pixel shuffle version
        self.image_tag_type = image_tag_type # TODO: into the tokenizer too?
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.video_pruning_rate = video_pruning_rate
        self.video_temporal_patch_size = video_temporal_patch_size

        # Sound/audio token settings
        self.sound_context_token_id = sound_context_token_id
        self.sound_context_token = sound_context_token

        self._attn_implementation = attn_implementation
        self.vision_config.use_flash_attn = self._attn_implementation is not None and "flash_attention" in self._attn_implementation
        self.llm_config._attn_implementation = self._attn_implementation

    # Need to alias llm_config as text_config for speculative decoding / MTP
    @property
    def text_config(self):
        return self.llm_config

    # vLLM's spec-decode setup (llm_base_proposer.py) reads AND writes
    # `target_model.config.image_token_index` using the standard HF VLM
    # convention. Our config stores it as `img_context_token_id`; expose
    # the standard name as a settable alias.
    @property
    def image_token_index(self):
        return self.img_context_token_id

    @image_token_index.setter
    def image_token_index(self, value):
        self.img_context_token_id = value

    def __getattr__(self, name):
        # Fallback so the wrapper config transparently exposes LM fields
        # (e.g. `num_hidden_layers`, `hidden_size`) used by vLLM's MTP draft-model
        # loader, which reads `vllm_config.model_config.hf_config.<lm_field>` on the
        # outer wrapper instead of the inner LM config.
        # __getattr__ runs only when normal lookup fails, so this doesn't interfere
        # with the wrapper's own attributes.
        try:
            llm_config = super().__getattribute__('llm_config')
        except AttributeError:
            llm_config = None
        if llm_config is not None and hasattr(llm_config, name):
            return getattr(llm_config, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
