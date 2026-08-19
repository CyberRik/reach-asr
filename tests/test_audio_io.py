"""Tests for the upload decode path.

These deliberately avoid loading Whisper. Everything here is a property of the
decoder -- rate, channel count, duration bounds -- and those are exactly the
things that fail silently in production: a 48 kHz stereo phone recording fed
through unchanged produces a confident, completely wrong transcript rather than
an error, so there is nothing downstream that would catch it.
"""

from __future__ import annotations

import io
import math

import pytest
import soundfile as sf
import torch

from reach_asr.audio_io import MAX_SECONDS, decode
from reach_asr.telephony import WHISPER_SR


def wav_bytes(wave: torch.Tensor, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    # soundfile wants (frames, channels); torch holds (channels, frames).
    sf.write(buffer, wave.transpose(0, 1).numpy(), sample_rate, format="WAV")
    return buffer.getvalue()


def tone(seconds: float, sample_rate: int, freq: float = 440.0, channels: int = 1) -> torch.Tensor:
    t = torch.arange(int(seconds * sample_rate), dtype=torch.float32) / sample_rate
    wave = 0.5 * torch.sin(2 * math.pi * freq * t)
    if channels == 1:
        return wave
    return wave.unsqueeze(0).repeat(channels, 1)


def test_passthrough_16k_mono_is_unchanged_in_rate_and_length():
    data = wav_bytes(tone(1.0, WHISPER_SR), WHISPER_SR)
    result = decode(data, "clip.wav")
    assert result.sample_rate == WHISPER_SR
    assert result.waveform.dim() == 1
    assert result.waveform.shape[0] == pytest.approx(WHISPER_SR, rel=0.01)
    assert result.duration_s == pytest.approx(1.0, abs=0.01)


def test_48k_is_resampled_to_whisper_rate_preserving_duration():
    """The failure this guards: 48 kHz frames read as 16 kHz play 3x fast.

    Duration is the tell. If resampling were skipped, the sample count would
    stay at 48000 and be *interpreted* as 3 seconds of 16 kHz audio.
    """
    data = wav_bytes(tone(1.0, 48000), 48000)
    result = decode(data, "phone.wav")
    assert result.sample_rate == WHISPER_SR
    assert result.duration_s == pytest.approx(1.0, abs=0.02)


def test_stereo_is_downmixed_to_one_channel():
    data = wav_bytes(tone(1.0, WHISPER_SR, channels=2), WHISPER_SR)
    result = decode(data, "stereo.wav")
    assert result.waveform.dim() == 1


def test_downmix_averages_rather_than_dropping_a_channel():
    """A caller on one channel and silence on the other must survive.

    Taking channel 0 instead of the mean would transcribe silence whenever the
    recorder happened to put the voice on the right channel -- intermittent by
    device, and invisible in any test using a correlated stereo signal.
    """
    voice = tone(1.0, WHISPER_SR)
    stereo = torch.stack([torch.zeros_like(voice), voice])
    result = decode(wav_bytes(stereo, WHISPER_SR), "one-sided.wav")
    assert result.waveform.abs().max() > 0.1


def test_missing_extension_still_decodes_a_wav():
    """Browsers and curl both send blobs with no filename at all."""
    data = wav_bytes(tone(0.5, WHISPER_SR), WHISPER_SR)
    result = decode(data, None)
    assert result.duration_s == pytest.approx(0.5, abs=0.02)


def test_audio_longer_than_the_whisper_window_is_refused_not_truncated():
    data = wav_bytes(tone(MAX_SECONDS + 5, WHISPER_SR), WHISPER_SR)
    with pytest.raises(ValueError, match="window"):
        decode(data, "long.wav")


def test_too_short_is_refused():
    data = wav_bytes(tone(0.02, WHISPER_SR), WHISPER_SR)
    with pytest.raises(ValueError, match="too short"):
        decode(data, "blip.wav")


def test_empty_upload_is_refused():
    with pytest.raises(ValueError, match="empty"):
        decode(b"", "nothing.wav")


def test_garbage_bytes_raise_rather_than_returning_silence():
    """A decode failure must not become an empty waveform.

    Silence transcribes to "" with no error, which would show a dispatcher a
    blank transcript for a recording that actually contains speech.
    """
    with pytest.raises((ValueError, RuntimeError)):
        decode(b"not audio at all, just some bytes" * 100, "junk.wav")
