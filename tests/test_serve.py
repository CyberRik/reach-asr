"""HTTP contract tests for the serving layer.

No model is loaded. Every test here stubs `transcribe_bytes`, because what is
being checked is the contract REACH depends on -- status codes, field names,
error propagation -- and none of that involves Whisper. Requiring a 290 MB
download to test a 422 response would mean these never run.

The startup hook is bypassed for the same reason: `lifespan` loads the model,
so `TestClient` is used without entering its context manager.
"""

from __future__ import annotations

import io
import math

import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from reach_asr import serve
from reach_asr.telephony import WHISPER_SR
from reach_asr.transcribe import Transcript


@pytest.fixture
def client() -> TestClient:
    # No `with`: entering the context runs lifespan, which loads the model.
    return TestClient(serve.app)


def wav_upload(seconds: float = 1.0) -> bytes:
    t = torch.arange(int(seconds * WHISPER_SR), dtype=torch.float32) / WHISPER_SR
    wave = (0.5 * torch.sin(2 * math.pi * 440 * t)).unsqueeze(1)
    buffer = io.BytesIO()
    sf.write(buffer, wave.numpy(), WHISPER_SR, format="WAV")
    return buffer.getvalue()


def test_health_reports_configuration_without_loading_a_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "ffmpeg" in body


def test_transcribe_returns_the_documented_field_names(monkeypatch, client):
    """REACH's route reads these exact keys; a rename here breaks it silently."""
    monkeypatch.setattr(
        serve,
        "transcribe_bytes",
        lambda *a, **k: Transcript(
            text="he is still unresponsive",
            duration_s=1.0,
            latency_ms=812.0,
            model="openai/whisper-base",
            adapter="runs/whisper-lora/adapter",
            device="cpu",
        ),
    )
    response = client.post("/transcribe", files={"file": ("clip.wav", wav_upload(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "he is still unresponsive"
    assert set(body) == {"text", "duration_s", "latency_ms", "model", "adapter", "device"}


def test_bad_audio_is_422_with_the_reason_intact(monkeypatch, client):
    """The message has to survive the hop.

    "audio is 42.0s; Whisper's window is 30s" tells a responder to re-record
    shorter. A generic 422 tells them nothing, and REACH forwards this string
    straight to the UI.
    """

    def boom(*args, **kwargs):
        raise ValueError("audio is 42.0s; Whisper's window is 30s.")

    monkeypatch.setattr(serve, "transcribe_bytes", boom)
    response = client.post("/transcribe", files={"file": ("long.wav", wav_upload(), "audio/wav")})
    assert response.status_code == 422
    assert "42.0s" in response.json()["detail"]


def test_infrastructure_failure_is_503_not_422(monkeypatch, client):
    """A missing ffmpeg or a CUDA OOM is ours, not the caller's.

    The distinction is load-bearing: REACH retries a 503 and shows a degraded
    banner, but treats a 422 as final and asks the user to re-record. Getting
    these backwards makes an outage look like the responder's mistake.
    """

    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg is not on PATH")

    monkeypatch.setattr(serve, "transcribe_bytes", boom)
    response = client.post("/transcribe", files={"file": ("clip.webm", wav_upload(), "audio/webm")})
    assert response.status_code == 503


def test_oversized_upload_is_rejected_before_transcription(monkeypatch, client):
    called = False

    def spy(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not reach the model")

    monkeypatch.setattr(serve, "transcribe_bytes", spy)
    monkeypatch.setattr(serve, "MAX_UPLOAD_BYTES", 1024)
    response = client.post("/transcribe", files={"file": ("big.wav", wav_upload(2.0), "audio/wav")})
    assert response.status_code == 413
    assert not called


def test_missing_file_field_is_422(client):
    assert client.post("/transcribe").status_code == 422
