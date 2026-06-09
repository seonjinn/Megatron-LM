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
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration import NemotronH_Super_Omni_Reasoning_V3_Config
from .modeling_nemotron_h import NemotronHForCausalLM
from .evs import EfficientVideoSampling
from .audio_model import SoundEncoder, SoundProjection

logger = logging.get_logger(__name__)


"""
The following code is adapted from the
https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B/blob/main/modeling_internvl_chat.py repository

The chat function is adapted to handle NVLM 1-D tile-tagging design for dynamic high-resolution images.
"""


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def _load_packaged_radio_model(model_path):
    import hashlib
    import importlib.util
    import sys
    from pathlib import Path

    model_path = Path(model_path)
    hf_model_path = model_path / "hf_model.py"
    if not hf_model_path.exists():
        return None

    package_name = "_megatron_local_radio_" + hashlib.sha1(str(model_path).encode()).hexdigest()[:12]
    if package_name not in sys.modules:
        package_spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
        package = importlib.util.module_from_spec(package_spec)
        package.__path__ = [str(model_path)]
        sys.modules[package_name] = package

    module_name = f"{package_name}.hf_model"
    module = sys.modules.get(module_name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(module_name, hf_model_path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    return module.RADIOModel


def _build_vision_model(config):
    if getattr(config, "_name_or_path", None):
        radio_model_cls = _load_packaged_radio_model(config._name_or_path)
        if radio_model_cls is not None:
            return radio_model_cls(config.vision_config)
    return AutoModel.from_config(config.vision_config, trust_remote_code=True)


class NemotronH_Super_Omni_Reasoning_V3(PreTrainedModel):
    config_class = NemotronH_Super_Omni_Reasoning_V3_Config
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _no_split_modules = ['NemotronHBlock']

    def __init__(self, config: NemotronH_Super_Omni_Reasoning_V3_Config):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.36.2', 'ge')
        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type
        self.img_context_token_id = config.img_context_token_id
        self.video_context_token_id = config.video_context_token_id

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')

        # Instantiate LM directly to avoid Hugging Face dynamic module lookup requiring a repo id.
        self.language_model = NemotronHForCausalLM(config.llm_config)
        self.vision_model = _build_vision_model(config)
        self.vision_model.model._initialize_weights = self.vision_model.model._init_weights  # WAR for transformers issue 38358
        self.vision_model.radio_model.make_preprocessor_external()

        # Attach a separate 3D patch projection for video frames. The RADIO ViT ships with only a 2D
        # `embedder` (shape `[embed_dim, C·P²]`); this repo's checkpoint also carries a
        # `video_embedder` (shape `[embed_dim, T·C·P²]`) used for temporally-packed video patches,
        # so we construct the module here to make the weight bind. `T = video_temporal_patch_size`
        # is the number of frames collapsed into each temporal patch.
        self.video_temporal_patch_dim = config.video_temporal_patch_size
        pg = self.vision_model.radio_model.model.patch_generator
        pg.video_embedder = nn.Linear(
            in_features=self.video_temporal_patch_dim * 3 * pg.patch_size * pg.patch_size,
            out_features=pg.embed_dim,
            bias=False,
        )

        # Align CPE position-embedding interpolation with Megatron training + vLLM inference.
        # The `nvidia/C-RADIOv2-H` remote code uses `align_corners=True` in eval mode, but the V3
        # checkpoint was trained against `align_corners=False` (see Megatron's `radio.py`). That
        # single-flag mismatch shifts every pos_embed by a fraction of a cell, which compounds
        # through 52 ViT layers and is the main cause of HF/vLLM divergence for video (where CPE
        # mode is active — dynamic-res tubelets don't match the model's native 2048-sized grid).
        self._patch_cpe_align_corners(pg)

        self.vision_model = self.vision_model.to(self.language_model.config.torch_dtype)

        self.drop_vision_class_token = True

        # Construct the vision projection.
        # Default
        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.video_pruning_rate = config.video_pruning_rate

        self.mlp1 = nn.Sequential(
            RMSNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, eps=1e-5),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, vision_projection_hidden_size, bias=False),
            SquaredReLU(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False)
        )
        self.mlp1 = self.mlp1.to(self.language_model.config.torch_dtype)

        # Sound/audio model components (optional - only if sound_config is provided)
        self.sound_context_token_id = getattr(config, 'sound_context_token_id', None)
        if config.sound_config is not None:
            sound_config = config.sound_config
            sound_hidden_size = sound_config.hidden_size
            sound_projection_hidden_size = sound_config.projection_hidden_size

            # Initialize sound feature extractor for converting raw audio to mel spectrograms
            from transformers import ParakeetFeatureExtractor
            sampling_rate = getattr(sound_config, 'sampling_rate', 16000)
            feature_size = getattr(sound_config, 'num_mel_bins', 128)
            self.sound_feature_extractor = ParakeetFeatureExtractor(
                sampling_rate=sampling_rate,
                feature_size=feature_size,
            )
            logger.info(f'Sound feature extractor initialized with sampling_rate={sampling_rate}, feature_size={feature_size}')

            # Initialize sound encoder - wraps Parakeet from transformers
            self.sound_encoder = SoundEncoder(config=sound_config)
            self.sound_encoder = self.sound_encoder.to(self.language_model.config.torch_dtype)

            # Initialize sound projection MLP
            self.sound_projection = SoundProjection(
                sound_hidden_size=sound_hidden_size,
                projection_hidden_size=sound_projection_hidden_size,
                llm_hidden_size=llm_hidden_size,
                bias=sound_config.projection_bias,
            )
            self.sound_projection = self.sound_projection.to(self.language_model.config.torch_dtype)

            logger.info(f'Sound model initialized with hidden_size={sound_hidden_size}')
        else:
            self.sound_encoder = None
            self.sound_projection = None
            self.sound_feature_extractor = None

        self.all_tied_weights_keys = {}

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            inputs_embeds = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_flags = image_flags.squeeze(-1)

        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        vit_batch_size = pixel_values.shape[0]
        vit_embeds = self.extract_feature(pixel_values)

        del pixel_values

        if torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        vit_embeds = vit_embeds[image_flags == 1]
        try:
            inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]

        del vit_embeds

        inputs_embeds = inputs_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _patch_cpe_align_corners(patch_generator) -> None:
        """Monkey-patch `patch_generator._get_pos_embeddings` so the CPE-mode eval-path interpolation
        uses `align_corners=False` (Megatron training + vLLM inference convention) instead of the
        `align_corners=True` that the `nvidia/C-RADIOv2-H` remote code ships with.
        """
        import math
        import torch.nn.functional as F

        orig_method = patch_generator._get_pos_embeddings.__func__ if hasattr(
            patch_generator._get_pos_embeddings, "__func__"
        ) else patch_generator._get_pos_embeddings

        def _get_pos_embeddings_aligned(self, batch_size, input_dims):
            if (self.num_rows, self.num_cols) == input_dims:
                return self.pos_embed
            pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)

            def window_select(pe):
                if input_dims[0] < pe.shape[-2]:
                    pe = pe[..., :input_dims[0], :]
                if input_dims[1] < pe.shape[-1]:
                    pe = pe[..., :, :input_dims[1]]
                return pe

            if self.cpe_mode:
                if self.training:
                    # Keep the original training-time jitter path (grid_sample + align_corners=True);
                    # only patch the eval branch, which is what Megatron/vLLM use and where the bug is.
                    return orig_method(self, batch_size, input_dims)
                max_dim = max(input_dims)
                pos_embed = F.interpolate(
                    pos_embed.float(), size=(max_dim, max_dim), align_corners=False, mode="bilinear"
                ).to(pos_embed.dtype)
                pos_embed = window_select(pos_embed)
            else:
                pos_embed = window_select(pos_embed)

            if pos_embed.shape[-2:] != input_dims:
                pos_embed = F.interpolate(
                    pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
                ).to(pos_embed.dtype)

            pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
            return pos_embed

        import types
        patch_generator._get_pos_embeddings = types.MethodType(_get_pos_embeddings_aligned, patch_generator)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """Run the ViT on a batch of image tiles.

        Handles two layouts:
        - A single 4D tensor `(B, 3, H, W)` with all tiles sharing the same spatial size (legacy
          fixed-tile path **or** dynamic-resolution path when every image in the batch resizes to
          the same target).
        - A list of 4D tensors `[(1, 3, H_i, W_i), …]` when dynamic resolution picks different
          target sizes per image. Each is run through the ViT independently and the output tokens
          are concatenated along the sequence dim.

        The patch grid `(h, w)` is computed from the actual input shape, not assumed square — this
        is required for dynamic resolution where the tile aspect ratio matches the original image.
        """
        if isinstance(pixel_values, (list, tuple)):
            outs = [self._extract_feature_single(pv) for pv in pixel_values]
            return torch.cat(outs, dim=0)
        return self._extract_feature_single(pixel_values)

    def _extract_feature_single(self, pixel_values):
        vit_embeds = self.vision_model(pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        # Compute patch grid from the input tile dims; pixel-shuffle needs the real (h, w).
        patch_size = self.vision_model.radio_model.model.patch_generator.patch_size
        B, _, H, W = pixel_values.shape
        h = H // patch_size
        w = W // patch_size
        vit_embeds = vit_embeds.reshape(B, h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(B, -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_video_feature(self, pixel_values_videos):
        """
        Extract features from video frames using the 3D `video_embedder`.

        Consecutive `T = video_temporal_patch_dim` frames are packed into a single temporal patch
        before the ViT, so the output has `N_frames // T` temporal units (each with the usual number
        of spatial tokens) instead of one ViT output per frame.

        Implementation trick: RADIO's patch_generator uses a channel-agnostic `Im2Patches` rearrange
        followed by `self.embedder(patches)`. If we stack the T temporal frames into the channel
        dim — `(N_frames, C, H, W)` → `(N_frames/T, T·C, H, W)` — the rearrange produces patches of
        shape `(·, num_patches, T·C·P²)`, which is exactly what `video_embedder` expects. Temporarily
        swapping `embedder ↔ video_embedder` lets us reuse the full ViT forward without duplicating
        the transformer blocks, pos-embed handling, cls_token, etc.
        """
        pg = self.vision_model.radio_model.model.patch_generator
        T = self.video_temporal_patch_dim
        N, C, H, W = pixel_values_videos.shape

        # Pad to a multiple of T by repeating the last frame so frame pairs align cleanly.
        if N % T != 0:
            pad = pixel_values_videos[-1:].expand(T - (N % T), -1, -1, -1)
            pixel_values_videos = torch.cat([pixel_values_videos, pad], dim=0)
            N = pixel_values_videos.shape[0]
        num_groups = N // T

        # Stack T frames into the channel dim. `.view` here preserves the (frame,channel) row-major
        # layout → per-patch feature order is [t=0,c=0..C-1, t=1,c=0..C-1, ...], matching how the
        # `video_embedder` weights are stored in the checkpoint.
        x = pixel_values_videos.reshape(num_groups, T * C, H, W)

        orig_embedder = pg.embedder
        pg.embedder = pg.video_embedder
        try:
            vit_embeds = self.vision_model(x).features
        finally:
            pg.embedder = orig_embedder

        # Same spatial post-processing as `extract_feature`. Compute `(h, w)` from the reshaped
        # input so dynamic-res video frames (non-square patch grid) are handled correctly.
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        patch_size = pg.patch_size
        h = H // patch_size
        w = W // patch_size
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_sound_feature(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract and project sound features from audio input.

        Args:
            input_features: Mel spectrogram features [batch, seq_len, feature_dim]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Sound embeddings projected to LLM hidden size [batch, encoded_seq_len, llm_hidden_size]
        """
        if self.sound_encoder is None:
            raise RuntimeError("Sound encoder not initialized. Check if sound_config is provided.")

        # Encode audio features
        sound_embeds = self.sound_encoder(input_features, attention_mask)
        sound_embeds = sound_embeds.to(dtype=torch.bfloat16)

        # Project to LLM hidden size
        sound_embeds = self.sound_projection(sound_embeds)

        return sound_embeds

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            sound_clips: Optional[torch.FloatTensor] = None,
            sound_length: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        """Generate text given images, videos, and/or audio.

        Args:
            pixel_values: Image pixel values [num_tiles, C, H, W]
            pixel_values_videos: Video pixel values [num_frames, C, H, W]
            sound_clips: Raw audio waveforms. Can be:
                - A list of numpy arrays or torch tensors (one per audio clip)
                - A single numpy array or torch tensor for a single audio clip
                - Pre-extracted mel spectrogram features [batch, seq_len, num_mel_bins]
            sound_length: Length of each audio clip in samples (optional, used for batched audio)
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            generation_config: Generation configuration
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            **generate_kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        assert self.img_context_token_id is not None

        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None
        has_sound = sound_clips is not None and self.sound_encoder is not None

        if has_images or has_videos or has_sound:
            image_vit_embeds, video_vit_embeds, sound_embeds = None, None, None

            # Process images
            if has_images:
                pixel_values = pixel_values.to(dtype=self.vision_model.config.torch_dtype)
                image_vit_embeds = self.extract_feature(pixel_values)

            # Process videos
            if has_videos:
                pixel_values_videos = pixel_values_videos.to(dtype=self.vision_model.config.torch_dtype)
                video_vit_embeds = self.extract_video_feature(pixel_values_videos)

            # Process sound/audio
            if has_sound:
                # Extract features from raw audio using the feature extractor
                # Handle different input types:
                # - list/tuple of waveforms
                # - 1D tensor/array (single waveform)
                # - 2D tensor [batch, samples] (batched raw waveforms)
                # - 3D tensor [batch, seq_len, num_mel_bins] (pre-extracted features)
                import numpy as np

                is_raw_waveform = False
                if isinstance(sound_clips, (list, tuple)):
                    # List of audio clips (waveforms)
                    is_raw_waveform = True
                    waveforms = sound_clips
                elif isinstance(sound_clips, np.ndarray):
                    # Numpy array - raw waveform
                    is_raw_waveform = True
                    waveforms = [sound_clips.squeeze()] if sound_clips.ndim > 1 else [sound_clips]
                elif isinstance(sound_clips, torch.Tensor):
                    if sound_clips.dim() == 1:
                        # 1D tensor - single raw waveform
                        is_raw_waveform = True
                        waveforms = [sound_clips.cpu().numpy()]
                    elif sound_clips.dim() == 2:
                        # 2D tensor [batch, samples] - batched raw waveforms
                        is_raw_waveform = True
                        waveforms = [clip.cpu().numpy() for clip in sound_clips]
                    else:
                        # 3D tensor [batch, seq_len, num_mel_bins] - pre-extracted features
                        is_raw_waveform = False
                else:
                    is_raw_waveform = False

                if is_raw_waveform:
                    # Convert raw waveforms to mel spectrogram features
                    audio_inputs = self.sound_feature_extractor(
                        waveforms,
                        sampling_rate=self.sound_feature_extractor.sampling_rate,
                        return_tensors="pt",
                    )
                    sound_input_features = audio_inputs.input_features
                    sound_attention_mask = audio_inputs.get("attention_mask", None)
                else:
                    # Already extracted features
                    sound_input_features = sound_clips
                    sound_attention_mask = None

                # Move to correct device and dtype
                target_device = self.sound_encoder.encoder.subsampling.linear.weight.device
                target_dtype = self.language_model.config.torch_dtype

                sound_input_features = sound_input_features.to(dtype=target_dtype, device=target_device)
                if sound_attention_mask is not None:
                    sound_attention_mask = sound_attention_mask.to(device=target_device)

                sound_embeds = self.extract_sound_feature(sound_input_features, sound_attention_mask)

            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids_copy = input_ids.reshape(B * N)

            # Replace image tokens with image embeddings
            if image_vit_embeds is not None:
                image_mask = (input_ids_copy == self.img_context_token_id)
                assert image_mask.sum() != 0, "No image tokens found in input_ids"
                inputs_embeds[image_mask] = image_vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)

            # Replace video tokens with video embeddings. The tokenizer has no distinct `<video>`
            # token (`video_context_token_id` in config doesn't decode to any printable string), so
            # the processor uses `<image>` (id = `img_context_token_id`) as the placeholder for
            # video positions too. We rely on the caller passing `pixel_values_videos` (not
            # `pixel_values`) to signal video vs. image — both share the same token id in the prompt.
            if video_vit_embeds is not None:
                if B > 1:
                    raise NotImplementedError("Video is not supported for batch size > 1")
                video_mask = (input_ids_copy == self.img_context_token_id)
                assert video_mask.sum() != 0, "No video tokens found in input_ids"
                inputs_embeds[video_mask] = video_vit_embeds.reshape(-1, C).to(inputs_embeds.device, inputs_embeds.dtype)

            # Replace sound tokens with sound embeddings.
            # `sound_embeds` has shape `(B_sound, T_out_max, C)` where `T_out_max`
            # is the encoder output length for the longest clip in the batch.
            # When `B_sound > 1` the shorter clips have padding at the tail, so
            # we must gather only the valid positions per row before scattering
            # into `sound_mask`. The encoder's `_get_subsampling_output_length`
            # converts each input mel-frame count (from the feature extractor's
            # attention_mask) to its post-subsampling token count.
            if sound_embeds is not None and self.sound_context_token_id is not None:
                sound_mask = (input_ids_copy == self.sound_context_token_id)
                assert sound_mask.sum() != 0, "No sound tokens found in input_ids"
                if sound_embeds.dim() == 3 and sound_embeds.shape[0] > 1 and sound_attention_mask is not None:
                    # `attention_mask.sum() = L_i // hop` per row, but
                    # `ParakeetFeatureExtractor` pads each row to `1 + L_i // hop`
                    # mel frames in single-call mode (the trailing frame comes
                    # from STFT center padding) — and the existing batch=1 path
                    # consumes that frame's embed too. Add 1 here to match.
                    natural_input_lengths = sound_attention_mask.sum(-1) + 1
                    output_lengths = self.sound_encoder.encoder._get_subsampling_output_length(natural_input_lengths)
                    flat = torch.cat(
                        [sound_embeds[i, : int(n)] for i, n in enumerate(output_lengths.tolist())],
                        dim=0,
                    )
                else:
                    flat = sound_embeds.reshape(-1, C)
                assert sound_mask.sum().item() == flat.shape[0], (
                    f"sound token count ({sound_mask.sum().item()}) != encoder output count ({flat.shape[0]})"
                )
                inputs_embeds[sound_mask] = flat.to(inputs_embeds.device, inputs_embeds.dtype)

            # Apply video pruning (EVS) if enabled
            if video_vit_embeds is not None and self.video_pruning_rate > 0:  # EVS
                h = w = int(video_vit_embeds.shape[1] ** 0.5)  # assumption here (and everywhere else) is that shape is square
                evs_mask = EfficientVideoSampling.compute_retention_mask(
                    video_embeds=video_vit_embeds,
                    thw=(video_vit_embeds.shape[0], h, w),
                    spatial_merge_size=1,  # we already work on vision embeddings, so no downsampling to follow
                    q=self.video_pruning_rate,
                )
                print(f"pruning rate: {self.video_pruning_rate}, EVS mask: {evs_mask.sum().item()} tokens retained out of {evs_mask.numel()} total video tokens ({evs_mask.sum().item() / evs_mask.numel() * 100:.2f}%)")

                retention_mask = torch.ones_like(input_ids_copy, dtype=torch.bool)
                retention_mask[video_mask] = evs_mask.view(-1)
                inputs_embeds = inputs_embeds[retention_mask].unsqueeze(0)  # adding batch=1
                if attention_mask is not None:
                    attention_mask = attention_mask[:, retention_mask].contiguous()
                if input_ids is not None:
                    input_ids = input_ids[:, retention_mask].contiguous()
            else:
                inputs_embeds = inputs_embeds.reshape(B, N, C)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
