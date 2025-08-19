# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import torch
from .conversation_sample import AudioMedia
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
            import librosa
            audio = librosa.resample(
                audio.numpy(),
                orig_sr=params.media.audio_samples_per_second,
                target_sr=self._target_freq
            )
            audio = torch.from_numpy(audio)
        return audio


class AudioTransformStrategy(_ResampleAudioTransformStrategy):
    """Audio transformation."""

    def __init__(self, sound_model_type: str, target_freq: int, embedding_size: int, clip_duration: int):
        super().__init__(target_freq, embedding_size)
        assert clip_duration == 30, "Only 30 second clips are supported in Whisper."
        self._clip_duration = clip_duration    # seconds
        self.sound_model_type = sound_model_type
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(sound_model_type.split("hf://")[1])

    def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:
        params_list = []

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
        audio_length = params.audio_length

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

        return audio, audio_length


class AudioTransformParakeetStrategy(_ResampleAudioTransformStrategy):
    """Audio transformation."""

    def __init__(self, sound_model_type: str, target_freq: int, embedding_size: int, clip_duration: int):
        super().__init__(target_freq, embedding_size)
        self.use_nemo = sound_model_type.startswith("nemo://")
        if sound_model_type.startswith("hf://"):
            assert clip_duration == 60, "Only 60 second clips are supported in HF Parakeet."
        elif sound_model_type.startswith("nemo://"):
            assert clip_duration % 60 == 0, "Only clip durations that are multiples of 60 seconds are supported in Nemo Parakeet."
        self._clip_duration = clip_duration    # seconds
        self.sound_model_type = sound_model_type
        assert 'parakeet' in sound_model_type.lower(), "Parakeet is the only supported model type for now."

    def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:
        params_list = []

        for media in media_list:
            # Compute the final number of tokens
            # Will be resampled to target_freq
            audio = media.value.get().get_audio().audio_clips
            num_clips = math.ceil(audio[0].shape[1] / self._clip_duration / media.audio_samples_per_second)
            if not self.use_nemo:
                # HF implementation is restricted to 60s clips.
                num_clips = 1

            clip_samples = self._clip_duration * self._target_freq
            audio_length = num_clips * clip_samples

            params_list.append(AudioParams(
                num_embeddings=self._embedding_size * num_clips * (self._clip_duration // 60) + 1,
                audio_length=torch.tensor([audio_length for _ in range(num_clips)], dtype=torch.long),
                num_clips=num_clips,
                timestamps=(0, int(audio_length / self._target_freq)),
                media=media,
            ))
        return params_list

    def apply_params(self, params: AudioParams) -> torch.Tensor:
        """Apply the preprocessing parameters to the audio."""
        audio = self._get_audio_resampled(params)
        audio_length = []

        if self.use_nemo:
            clip_samples = self._clip_duration * self._target_freq
            audio = audio.squeeze(1)
            if audio.shape[1] != clip_samples:
                # Pad or batch to expected clip length.
                if audio.shape[1] < clip_samples:
                    audio_length.append(audio.shape[1])
                    audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
                else:
                    clips = list(torch.split(audio, clip_samples, dim=1))
                    audio_length = [c.shape[1] for c in clips]
                    clips[-1] = torch.nn.functional.pad(clips[-1], (0, clip_samples - clips[-1].shape[1]))
                    audio = torch.stack(clips)
                    audio = audio.squeeze()
            else:
                audio_length.append(audio.shape[1])
        else:
            # Force cap to 60s.
            audio = audio.squeeze(1)
            clip_samples = self._clip_duration * self._target_freq

            audio_length.append(audio.shape[1])
            audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))

        return audio, torch.tensor(audio_length, dtype=torch.long)
