# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import torch
from .conversation_sample import AudioMedia
import librosa
from transformers import AutoFeatureExtractor


@dataclass
class AudioParams:
    num_embeddings: int

    num_clips: int
    audio_length: int
    timestamps: tuple[int, int]

    media: AudioMedia


class AudioPreprocessingStrategy(ABC):
    """Audio preprocessing strategy."""

    @abstractmethod
    def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:
        """Compute the media transform parameters."""
        ...

    @abstractmethod
    def apply_params(self, params: AudioParams) -> torch.Tensor:
        """Apply the preprocessing parameters to the audio."""
        ...


class _ResampleAudioTransformStrategy(AudioPreprocessingStrategy):

    def __init__(self, target_freq: int, embedding_size: int):
        self._target_freq = target_freq
        self._embedding_size = embedding_size

    def _get_audio_resampled(self, params: AudioParams) -> torch.Tensor:
        audio = torch.stack(params.media.value.get_audio().audio_clips, dim=0)

        # Convert to float32 for processing. For stereo audio, we average the channels.

        # Convert integer values to float by dividing by the max value

        # max_value = audio.max()
        # audio = audio / max_value
        audio = audio.to(torch.float32)
        audio = audio.mean(dim=1, keepdim=True)
        if params.media.audio_samples_per_second != self._target_freq:
            audio = librosa.resample(
                audio.numpy(),
                orig_sr=params.media.audio_samples_per_second,
                target_sr=self._target_freq
            )
            audio = torch.from_numpy(audio)
        return audio


class AudioTransformStrategy(_ResampleAudioTransformStrategy):
    """Audio transformation."""

    def __init__(self, sound_model_type: str, target_freq: int, embedding_size: int):
        super().__init__(target_freq, embedding_size)
        self._clip_duration = 30    # seconds
        self.sound_model_type = sound_model_type
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(sound_model_type.split("hf://")[1])

    def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:
        params_list = []

        # import os
        # if int(os.environ.get("RANK", 0)) == 0:
        #     breakpoint()
        # else:
        #     import time
        #     time.sleep(10000)

        for media in media_list:
            # Compute the final number of tokens
            # Will be resampled to target_freq
            audio = media.value.get().get_audio().audio_clips
            num_clips = math.ceil(audio[0].shape[1] / self._clip_duration / media.audio_samples_per_second)

            #num_samples = int(media.audio_duration * self._target_freq)
            clip_samples = self._clip_duration * self._target_freq
            #num_clips = math.ceil(num_samples / clip_samples)

            params_list.append(AudioParams(
                num_embeddings=num_clips * self._embedding_size,
                audio_length=torch.tensor([num_clips * clip_samples], dtype=torch.long),
                num_clips=num_clips,
                timestamps=(0, num_clips * self._clip_duration),
                media=media,
            ))
        return params_list

    def apply_params(self, params: AudioParams) -> torch.Tensor:
        """Apply the preprocessing parameters to the audio."""
        audio = self._get_audio_resampled(params)

        clip_samples = self._clip_duration * self._target_freq
        audio = audio.squeeze(1)
        if audio.shape[1] != clip_samples:
            # Pad or batch to expected clip length.
            if audio.shape[1] < clip_samples:
                audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
            else:
                clips = list(torch.split(audio, clip_samples, dim=1))
                clips[-1] = torch.nn.functional.pad(clips[-1], (0, clip_samples - clips[-1].shape[1]))
                audio = torch.stack(clips)
                audio = audio.squeeze()
        # How to get batching to work?
        audio_features = []

        for a in audio:
            audio_features.append(self.feature_extractor(a, return_tensors="pt", sampling_rate=self._target_freq).input_features)
        audio = torch.cat(audio_features, dim=0)
        return audio


class AudioTransformParakeetStrategy(_ResampleAudioTransformStrategy):
    """Audio transformation."""

    def __init__(self, sound_model_type: str, target_freq: int, embedding_size: int):
        super().__init__(target_freq, embedding_size)
        self._clip_duration = 30    # seconds
        self.sound_model_type = sound_model_type
        assert 'parakeet' in sound_model_type.lower(), "Parakeet is the only supported model type for now."
        from megatron.core.models.huggingface.fastconformer.feature_extraction_fastconformer import FastConformerFeatureExtractor
        self.feature_extractor = FastConformerFeatureExtractor.from_pretrained(sound_model_type.split("hf://")[1])

    def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:
        params_list = []
        for media in media_list:
            # Compute the final number of tokens
            # Will be resampled to target_freq
            params_list.append(AudioParams(
                num_embeddings=self._embedding_size,
                audio_length=torch.tensor([int(media.audio_duration * self._target_freq)], dtype=torch.long),
                num_clips=1,
                timestamps=(0, self._clip_duration),
                media=media,
            ))
        return params_list

    def apply_params(self, params: AudioParams) -> torch.Tensor:
        """Apply the preprocessing parameters to the audio."""
        audio = self._get_audio_resampled(params)

        audio = audio.squeeze(1)
        clip_samples = self._clip_duration * 2 * self._target_freq
        audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
        return audio
