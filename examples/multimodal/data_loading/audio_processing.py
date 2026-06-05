# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
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

    samples_per_clip: list[int] = None


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
        val = params.media.value
        val = [val] if isinstance(val, torch.Tensor) else val.get_audio().audio_clips
        audio = torch.stack(val, dim=0)

        # Convert to float32 for processing. For stereo audio, we average the channels.
        # Normalize integer PCM to [-1.0, 1.0] using per-dtype constants.
        if audio.dtype == torch.int16:
            audio = audio.to(torch.float32) / 32768.0
        elif audio.dtype == torch.int32:
            audio = audio.to(torch.float32) / 2147483648.0
        else:
            audio = audio.to(torch.float32)
        # Fallback: normalize if values are outside [-1, 1] (e.g., float32 with int-scale values)
        max_val = audio.abs().max()
        if max_val > 1.0:
            audio = audio / max_val
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
            # Reduce the audio duration by 0.1 seconds. i.e. do not use a clip that is <0.1sec.
            num_samples = int((media.audio_duration - 0.1) * self._target_freq)
            clip_samples = int(self._clip_duration * self._target_freq)
            num_clips = math.ceil(num_samples / clip_samples)
            assert num_clips > 0, f"Expected at least 1 clip, got {num_clips}"
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

        clip_samples = int(self._clip_duration * self._target_freq)
        audio = audio.squeeze(1)
        if audio.shape[1] != clip_samples:
            # Pad or batch to expected clip length.
            if audio.shape[1] < clip_samples:
                assert params.num_clips == 1, f"Expected 1 clip, got {params.num_clips}"
                audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
            else:
                clips = list(torch.split(audio, clip_samples, dim=1))
                # Optionally truncate if the last clip is very short.
                assert len(clips) in (params.num_clips, params.num_clips + 1), f"Expected at {params.num_clips}(+1) clips, got {len(clips)}"
                clips = clips[:params.num_clips]
                if clips[-1].shape[1] < clip_samples:
                    clips[-1] = torch.nn.functional.pad(clips[-1], (0, clip_samples - clips[-1].shape[1]))
                audio = torch.stack(clips)
                audio = audio.squeeze(1)
        # How to get batching to work?
        audio_features = []

        for a in audio:
            audio_features.append(self.feature_extractor(a, return_tensors="pt", sampling_rate=self._target_freq).input_features)
        audio = torch.cat(audio_features, dim=0)

        return audio, audio_length


class AudioTransformParakeetStrategy(_ResampleAudioTransformStrategy):
    """Audio transformation."""

    def __init__(
        self,
        sound_model_type: str,
        target_freq: int,
        embedding_size: int,
        clip_duration: int,
        min_duration: float = 0.1,
        pad_to_clip_duration: bool = False,  # True => old setup with fixed shapes
    ):
        super().__init__(target_freq, embedding_size)
        self.use_nemo = sound_model_type.startswith("nemo://")
        if not self.use_nemo:
            assert clip_duration == 60, "Only 60 second clips are supported in HF Parakeet."
        self._clip_duration = clip_duration    # seconds
        self.sound_model_type = sound_model_type
        self._clip_duration = clip_duration
        self._clip_samples = round(self._clip_duration * self._target_freq)
        self.min_duration = min_duration  # will pad if less
        self.min_audio_samples = int(round(self.min_duration * self._target_freq))
        self.pad_to_clip_duration = pad_to_clip_duration
        assert 'parakeet' in sound_model_type.lower(), "Parakeet is the only supported model type for now."

    def compute_params(self, media_list: list[AudioMedia]) -> list[AudioParams]:

        def seconds_to_samples(seconds: float) -> int:
            return round(seconds * self._target_freq)

        params_list = []
        for media in media_list:
            # Compute the final number of tokens
            # Will be resampled to target_freq
            val = media.value
            audio = [val] if isinstance(val, torch.Tensor) else val.get().get_audio().audio_clips
            orig_sr = media.audio_samples_per_second
            orig_audio_length = audio[0].shape[-1]
            audio_duration = max(self.min_duration, orig_audio_length / orig_sr)
            audio_length = seconds_to_samples(audio_duration)

            num_clips = math.ceil(audio_length / self._clip_samples)

            remainder = audio_length % self._clip_samples
            last_clip_size = self._clip_samples if remainder == 0 else max(remainder, self.min_audio_samples)
            clip_samples = [self._clip_samples] * (num_clips - 1) + [last_clip_size]
            if (tot := sum(clip_samples)) > audio_length:  # we applied some padding to the very short last clip
                audio_length = tot
                audio_duration = audio_length / self._target_freq

            if self.pad_to_clip_duration:
                # +1 because fastconformer outputs one more frame than the actual input length due to STFT padding
                num_embeddings = num_clips * (estimate_audio_num_embeddings(self._clip_samples) + 1)
            else:
                num_embeddings = sum(estimate_audio_num_embeddings(num_samples) for num_samples in clip_samples)

            params_list.append(AudioParams(
                num_embeddings=num_embeddings,
                audio_length=audio_length,
                num_clips=num_clips,
                samples_per_clip=clip_samples,
                timestamps=(0, audio_duration),
                media=media,
            ))
        return params_list

    def apply_params(self, params: AudioParams) -> torch.Tensor:
        """Apply the preprocessing parameters to the audio."""
        audio = self._get_audio_resampled(params).squeeze(1)

        def _maybe_pad_to_min_length(audio: torch.Tensor, min_length: int) -> torch.Tensor:
            # Parakeet requires a minimum of hop length audio length; empirically padding to min 0.1s works well
            if audio.shape[1] < min_length:
                audio = torch.nn.functional.pad(audio, (0, min_length - audio.shape[1]))
            return audio

        if self.pad_to_clip_duration:
            min_signal_length = self._clip_samples * params.num_clips  # samples_per_clip was set to clip_duration * target_freq for each clip in compute_params()
        else:
            min_signal_length = self.min_audio_samples

        audio = _maybe_pad_to_min_length(audio, min_length=min_signal_length)
        audio_length = torch.tensor([params.audio_length], dtype=torch.long)
        # note(pzelasko): the resampling transform results in a different than expected number of audio samples
        #                 we have to truncate/pad to the expected number to stay faithful to num_embeddings computed
        #                 at compute_params() step
        if audio.shape[1] < audio_length.item():
            audio = torch.nn.functional.pad(audio, (0, audio_length.item() - audio.shape[1]))
        elif not self.pad_to_clip_duration and audio.shape[1] > audio_length.item():
            audio = audio[:, :audio_length.item()]

        # Splitting into self._clip_duration clips
        if audio.shape[1] > self._clip_samples:
            clips = list(torch.split(audio, self._clip_samples, dim=1))
            assert len(clips) == params.num_clips, f"Expected {params.num_clips} clips, got {len(clips)}. Something went wrong with compute_params() step or audio decoding/transforms."
            audio_length = torch.tensor(params.samples_per_clip, dtype=torch.long)
            clips[-1] = torch.nn.functional.pad(clips[-1], (0, self._clip_samples - clips[-1].shape[1]))
            audio = torch.stack(clips)
            audio = audio.squeeze()

        # Note(pzelasko): The assertions below will result in some data being throwing away, for which we weren't able to accurately estimate the number of clips and audio lengths.
        # I've found such (very rare) edge cases but couldn't easily reproduce the issue.
        assert audio_length.sum().item() == params.audio_length, f"Expected {params.audio_length} audio samples, got {audio_length.sum().item()}. Something went wrong with compute_params() step or audio decoding/transforms."
        assert audio.shape[0] == params.num_clips, f"Expected {params.num_clips} audio clips, got {audio.shape[0]}. Something went wrong with compute_params() step or audio decoding/transforms."
        assert audio_length.tolist() == params.samples_per_clip, f"Expected {params.samples_per_clip} audio samples per clip, got {audio_length.tolist()}. Something went wrong with compute_params() step or audio decoding/transforms."

        return audio, audio_length


@dataclass
class AudioFrameConfig:
    """Settings necessary to accurately estimate sequence length after audio encoder. Defaults for Parakeet TDT V2."""

    # Feature extrator.
    # (n_audio_samples) -> (n_frames)
    stft_pad_amount: int = None
    n_fft: int = 512
    hop_length: int = 160

    # Convolutional subsampling frontend.
    # (n_frames) -> (n_subsampled_frames)
    left_padding: int = 1
    right_padding: int = 1
    kernel_size: int = 3
    stride: int = 2
    ceil_mode: bool = False
    repeat_num: int = 3


def estimate_conv_subsampling_length(lengths: torch.Tensor, config: AudioFrameConfig = AudioFrameConfig()) -> torch.Tensor:
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad = config.left_padding + config.right_padding - config.kernel_size
    for i in range(config.repeat_num):
        lengths = torch.div(lengths.float() + add_pad, config.stride) + 1.0
        if config.ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.long()


def estimate_fbank_length(seq_len: int | torch.Tensor, config: AudioFrameConfig = AudioFrameConfig()) -> torch.Tensor:
    seq_len = torch.as_tensor(seq_len)
    # Assuming that center is True is stft_pad_amount = 0
    pad_amount = (
        config.stft_pad_amount * 2
        if config.stft_pad_amount is not None
        else config.n_fft // 2 * 2
    )
    seq_len_unfixed = torch.floor_divide((seq_len + pad_amount - config.n_fft), config.hop_length)
    # fix for seq_len = 0 for streaming; if size was 0, it is always padded to 1, and normalizer fails
    seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)
    return seq_len.to(dtype=torch.long)


def estimate_audio_num_embeddings(seq_len: int | torch.Tensor, config: AudioFrameConfig = AudioFrameConfig()) -> torch.Tensor:
    return estimate_conv_subsampling_length(estimate_fbank_length(seq_len, config), config)


def test():
    from nemo.core.classes import typecheck
    from nemo.collections.asr.models import ASRModel

    typecheck.set_typecheck_enabled(False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").eval().to(device)
    print(model.preprocessor.featurizer.__dict__)
    print(model.encoder.pre_encode.__dict__)

    for l in (1000, 6000, 8000, 16000, 64123, 849815, 848216, 4688216):
        for ll in (l-200, l-100, l-1, l, l+1, l+100, l+200):
            if ll < 0:
                continue
            audio = torch.randn(1, ll, dtype=torch.float32, device=device)
            length = torch.as_tensor([ll], dtype=torch.long, device=device)

            _, expected = model.preprocessor(audio, length)
            actual = estimate_fbank_length(length)
            assert torch.allclose(actual, expected), (ll, expected, actual)
            print(ll, "audio samples =>", actual, "audio fbank frames === OK")

            feats, feats_len = model.preprocessor(audio, length)
            h, expected = model.encoder(feats, feats_len)
            actual = estimate_audio_num_embeddings(length)
            assert torch.allclose(actual, expected), (ll, expected, actual)
            print(ll, "audio samples =>", actual, "audio embedding tokens === OK")


if __name__ == "__main__":
    test()
