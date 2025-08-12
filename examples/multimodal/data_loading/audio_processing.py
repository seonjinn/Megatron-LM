# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor


class AudioTransform:
    """Audio transformation."""

    def __init__(self, sound_model_type, target_freq):
        self._target_freq = target_freq
        self._clip_duration = 30    # seconds
        self.sound_model_type = sound_model_type
        if 'parakeet' in sound_model_type.lower():
            from megatron.core.models.huggingface.fastconformer.feature_extraction_fastconformer import FastConformerFeatureExtractor
            self.feature_extractor = FastConformerFeatureExtractor.from_pretrained(sound_model_type.split("hf://")[1])
        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(sound_model_type.split("hf://")[1])

    def __call__(self, audio, orig_freq):
        # Convert to float32 for processing. For stereo audio, we average the channels.

        # Convert integer values to float by dividing by the max value

        # max_value = audio.max()
        # audio = audio / max_value
        audio = audio.to(torch.float32)
        audio = audio.mean(dim=1, keepdim=True)
        if orig_freq != self._target_freq:
            audio = librosa.resample(
                audio.numpy(),
                orig_sr=orig_freq,
                target_sr=self._target_freq
            )
            audio = torch.from_numpy(audio)

        if 'parakeet' in self.sound_model_type:
            audio = audio.squeeze(1)
            audio_length = torch.tensor([audio.shape[1]], dtype=torch.long)
            clip_samples = self._clip_duration * 2 * self._target_freq
            timestamps = (0, len(audio) * self._clip_duration)
            audio = torch.nn.functional.pad(audio, (0, clip_samples - audio.shape[1]))
        else:
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
            timestamps = (0, len(audio) * self._clip_duration)
            audio_features = []

            for a in audio:
                audio_features.append(self.feature_extractor(a, return_tensors="pt", sampling_rate=self._target_freq).input_features)
            audio = torch.cat(audio_features, dim=0)
            audio_length = torch.tensor([len(audio)], dtype=torch.long)
        return audio,audio_length, timestamps


def preprocess_whisper(sound_file, sample_rate=16000):
    import whisper
    if sound_file is None:
        return None
    sound_outputs = []
    CHUNK_LIM_SEC = 30
    CHUNK_LIM = 30 * 16000
    num_audio_frames = 0
    try:
        if not isinstance(sound_file, np.ndarray):
            sound = whisper.load_audio(sound_file,sr=sample_rate)
        else:
            sound = sound_file
        duration = float(len(sound) / sample_rate)
        # if smaller than 30 sec, move on
        if duration <= CHUNK_LIM_SEC:
            sound = whisper.pad_or_trim(sound)
            sound = sound.reshape(1, -1)
            sound_outputs.append(torch.tensor(sound))
            num_audio_frames = 1
        # if larger than 30 sec, chunk it and pad last piece
        else:
            for i in range(0, len(sound), CHUNK_LIM):
                chunk = sound[i : i + CHUNK_LIM]
                chunk_index = float(len(chunk) / sample_rate)
                if chunk_index <= CHUNK_LIM_SEC:
                    chunk = whisper.pad_or_trim(chunk)
                    chunk = chunk.reshape(1, -1)
                    sound_outputs.append(torch.tensor(chunk))
                    num_audio_frames+=1
                if num_audio_frames == 5:
                    break
    except:
        raise ValueError("Error in preprocess_whisper")
    return torch.stack(sound_outputs, dim=0)


def preprocess_parakeet(sound_path):
    # librosa loading is better for Parakeet.TODO: test for nv-whisper
    import librosa
    sound, sr = librosa.load(sound_path, sr=16000)
    # audio_tensor1 = torch.from_numpy(test_audio1).float().unsqueeze(0)  # (1, time1)
    # audio_length1 = torch.tensor([len(test_audio1)], dtype=torch.long)
    # sound = whisper.load_audio(sound_path,sr=16000)
    # sound = whisper.pad_or_trim(sound)

    sound = sound.reshape(1, -1)
    sound = torch.tensor(sound).unsqueeze(0)
    return sound
