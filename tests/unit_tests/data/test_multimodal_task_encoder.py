# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib
import io
from types import SimpleNamespace

import pytest

from examples.multimodal.data_loading import task_encoder
from examples.multimodal.data_loading.task_encoder import (
    MultiModalTaskEncoder,
    _normalize_thinking_trace,
)


def _encoder(thread_count=8):
    encoder = object.__new__(MultiModalTaskEncoder)
    encoder.args = SimpleNamespace(video_decode_thread_count=thread_count)
    return encoder


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


class _FakeVideoStream:
    def __init__(self):
        self.codec_context = SimpleNamespace(thread_count=None, thread_type=None)
        self.type = "video"


class _FakeStreams:
    def __init__(self, streams):
        self.video = [stream for stream in streams if stream.type == "video"]
        self._streams = streams

    def __iter__(self):
        return iter(self._streams)


class _FakeContainer:
    def __init__(self, video_stream=None):
        self.closed = False
        self.video_stream = video_stream or _FakeVideoStream()
        self.streams = _FakeStreams([self.video_stream])

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _decoder_module():
    return importlib.import_module(task_encoder.AVDecoder.__module__)


def _fake_decoder(decoder_module, *, frames=None, open_container=True, calls=None):
    frames = ["frame"] if frames is None else frames

    class FakeEnergonDecoder:
        def get_clips(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            if open_container:
                with decoder_module.av_open(io.BytesIO(b"video")):
                    pass
            return SimpleNamespace(video_clips=[[frame] for frame in frames])

    return FakeEnergonDecoder()


def test_energon_exact_decode_configures_video_threads_and_restores_hook(monkeypatch):
    container = _FakeContainer()
    audio_context = SimpleNamespace(thread_count=None, thread_type=None)
    audio_stream = SimpleNamespace(type="audio", codec_context=audio_context)
    container.streams = _FakeStreams([container.video_stream, audio_stream])
    decoder_module = _decoder_module()
    calls = []

    def original_av_open(*args, **kwargs):
        return container

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)
    monkeypatch.setattr(task_encoder, "tensor_to_pil", lambda image: image)

    images = _encoder()._decode_video_frames_with_energon(
        _fake_decoder(decoder_module, frames=["frame-0", "frame-1"], calls=calls),
        [1.5, 0.25],
        thread_count=8,
    )

    assert images == ["frame-0", "frame-1"]
    assert calls == [{"video_clip_ranges": [(1.5, 1.5), (0.25, 0.25)], "video_unit": "seconds"}]
    assert container.video_stream.codec_context.thread_count == 8
    assert container.video_stream.codec_context.thread_type == "FRAME"
    assert audio_context.thread_count is None
    assert audio_context.thread_type is None
    assert decoder_module.av_open is original_av_open
    assert container.closed


def test_energon_exact_decode_disabled_preserves_original_open(monkeypatch):
    container = _FakeContainer()
    decoder_module = _decoder_module()

    def original_av_open(*args, **kwargs):
        return container

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)
    monkeypatch.setattr(task_encoder, "tensor_to_pil", lambda image: image)

    images = _encoder(thread_count=0)._decode_video_frames(_fake_decoder(decoder_module), [0.0])

    assert images == ["frame"]
    assert container.video_stream.codec_context.thread_count is None
    assert container.video_stream.codec_context.thread_type is None


def test_energon_exact_decode_retries_unthreaded_when_clips_are_missing(monkeypatch):
    decoder_module = _decoder_module()
    containers = []

    def original_av_open(*args, **kwargs):
        container = _FakeContainer()
        containers.append(container)
        return container

    class ShortThreadedDecoder:
        def get_clips(self, **kwargs):
            with decoder_module.av_open(io.BytesIO(b"video")) as container:
                threaded = container.video_stream.codec_context.thread_type == "FRAME"
            frames = ["frame-0"] if threaded else ["frame-0", "frame-1"]
            return SimpleNamespace(video_clips=[[frame] for frame in frames])

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)
    monkeypatch.setattr(task_encoder, "tensor_to_pil", lambda image: image)

    images = _encoder()._decode_video_frames_with_energon(
        ShortThreadedDecoder(), [0.0, 1.0], thread_count=8
    )

    assert images == ["frame-0", "frame-1"]
    assert len(containers) == 2
    assert containers[0].video_stream.codec_context.thread_type == "FRAME"
    assert containers[1].video_stream.codec_context.thread_type is None
    assert decoder_module.av_open is original_av_open


def test_energon_exact_decode_restores_hook_when_decode_raises(monkeypatch):
    decoder_module = _decoder_module()

    def original_av_open(*args, **kwargs):
        return _FakeContainer()

    class FailingDecoder:
        def get_clips(self, **kwargs):
            with decoder_module.av_open(io.BytesIO(b"video")):
                raise RuntimeError("decode failed")

    monkeypatch.setattr(decoder_module, "av_open", original_av_open, raising=False)

    with pytest.raises(RuntimeError, match="decode failed"):
        _encoder()._decode_video_frames_with_energon(FailingDecoder(), [0.0], thread_count=8)

    assert decoder_module.av_open is original_av_open


def test_video_decode_empty_targets_does_not_open_container():
    decoder = SimpleNamespace(
        get_clips=lambda **kwargs: pytest.fail("empty targets must not decode")
    )
    assert _encoder()._decode_video_frames(decoder, []) == []


def test_video_decode_rejects_negative_thread_count():
    decoder = SimpleNamespace(
        get_clips=lambda **kwargs: pytest.fail("invalid config must fail before decode")
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        _encoder(thread_count=-1)._decode_video_frames(decoder, [0.0])


def test_load_media_decodes_multiple_videos_and_preserves_frame_order(monkeypatch):
    class FakeAVDecoder:
        def __init__(self, name):
            self.name = name
            self.suppress_warnings = False

    class FakeLazy:
        def __init__(self, value):
            self.value = value

        def get(self, sample):
            return self.value

    first_decoder = FakeAVDecoder("first")
    second_decoder = FakeAVDecoder("second")
    first_lazy = FakeLazy(first_decoder)
    second_lazy = FakeLazy(second_decoder)
    frames = [
        SimpleNamespace(media=SimpleNamespace(value=first_lazy, timestamp=2.5)),
        SimpleNamespace(media=SimpleNamespace(value=second_lazy, timestamp=1.0)),
        SimpleNamespace(media=SimpleNamespace(value=first_lazy, timestamp=0.5)),
    ]
    sample = SimpleNamespace(images=frames, audio=[])
    encoder = _encoder()
    decode_calls = []

    def decode_video_frames(decoder, targets):
        decode_calls.append((decoder.name, targets))
        return [f"{decoder.name}-{timestamp}" for timestamp in targets]

    monkeypatch.setattr(task_encoder, "AVDecoder", FakeAVDecoder)
    monkeypatch.setattr(encoder, "_decode_video_frames", decode_video_frames)

    encoder._load_media(sample)

    assert decode_calls == [("first", [2.5, 0.5]), ("second", [1.0])]
    assert [frame.media.value for frame in frames] == ["first-2.5", "second-1.0", "first-0.5"]
    assert first_decoder.suppress_warnings
    assert second_decoder.suppress_warnings
