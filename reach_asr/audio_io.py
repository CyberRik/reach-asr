"""Decode arbitrary uploaded audio to the one format Whisper accepts.

Whisper's feature extractor wants mono float32 at 16 kHz. Nothing that arrives
over HTTP is in that form, and the gap is wider than it looks:

* A browser's `MediaRecorder` produces **WebM/Opus** on Chrome and Firefox, and
  **MP4/AAC** on Safari. Neither is a format `torchaudio.load` can open --
  libsndfile handles WAV/FLAC/OGG and stops there. Discovering that only in
  production is the usual way a "just send the blob" design fails.
* Phones record at 44.1 or 48 kHz, and often in stereo. Feeding 48 kHz frames to
  a feature extractor configured for 16 kHz does not error; it silently
  transposes everything up in pitch by a factor of three, and the transcript
  comes back as confident nonsense. That failure mode is far worse than a crash.

So this module is deliberately strict: one entry point, one output contract, and
an explicit ffmpeg fallback for the container formats libsndfile refuses. The
fallback is checked for at import time rather than at request time, because a
missing ffmpeg should fail the health check, not the first real emergency call.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass

import soundfile as sf
import torch
import torchaudio.functional as AF

from reach_asr.telephony import WHISPER_SR

# Formats libsndfile opens directly. Anything else takes the ffmpeg path.
NATIVE_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".aiff", ".aif"})

# Whisper's window. Longer input is not an error -- the feature extractor
# truncates silently -- which is precisely why it is worth refusing loudly.
MAX_SECONDS = 30.0


@dataclass(frozen=True)
class DecodedAudio:
    waveform: torch.Tensor  # 1-D float32, mono, WHISPER_SR
    sample_rate: int
    duration_s: float


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _read_native(data: bytes) -> tuple[torch.Tensor, int]:
    """Decode via libsndfile.

    soundfile rather than `torchaudio.load`: recent torchaudio routes file-like
    inputs through TorchCodec, which is a separate optional install and raises
    ImportError when it is missing. soundfile reads a BytesIO directly and is
    already required for writing the corpus, so it adds nothing to the install.

    It returns (frames, channels); torch convention here is (channels, frames).
    """
    array, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return torch.from_numpy(array).transpose(0, 1), int(sample_rate)


def _decode_with_ffmpeg(data: bytes) -> tuple[torch.Tensor, int]:
    """Transcode anything ffmpeg understands to 16 kHz mono PCM.

    Reads from stdin and writes to stdout so nothing touches the filesystem --
    an upload endpoint that writes temp files needs a cleanup story and a disk
    quota, and here there is no reason to have either.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "this audio needs ffmpeg to decode (browser recordings are WebM/Opus "
            "or MP4/AAC, which libsndfile cannot open) and ffmpeg is not on PATH"
        )
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "wav", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(WHISPER_SR),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"could not decode audio: {detail or 'ffmpeg failed'}")
    return _read_native(process.stdout)


def decode(data: bytes, filename: str | None = None) -> DecodedAudio:
    """Bytes in, mono 16 kHz float32 out.

    `filename` only picks which decoder to *try* first; a wrong or absent
    extension still works, because a libsndfile failure falls through to ffmpeg
    rather than propagating. Clients lie about content types constantly.
    """
    if not data:
        raise ValueError("empty audio upload")

    suffix = ""
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()

    wave: torch.Tensor | None = None
    sample_rate = 0
    if suffix in NATIVE_SUFFIXES or not suffix:
        try:
            wave, sample_rate = _read_native(data)
        except Exception:  # noqa: BLE001 - any libsndfile failure means "try ffmpeg"
            wave = None
    if wave is None:
        wave, sample_rate = _decode_with_ffmpeg(data)

    # Downmix before resampling: averaging channels is cheaper at the original
    # rate, and resampling two channels to discard one afterwards is wasted work.
    if wave.dim() == 2 and wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    wave = wave.reshape(-1).to(torch.float32)

    if sample_rate != WHISPER_SR:
        wave = AF.resample(wave, sample_rate, WHISPER_SR)

    duration = wave.shape[-1] / WHISPER_SR
    if duration > MAX_SECONDS:
        # Refused rather than truncated. Whisper's extractor would silently drop
        # everything past 30 s, so a 90 s clip would return a transcript of its
        # first third with no indication that two thirds were discarded.
        raise ValueError(
            f"audio is {duration:.1f}s; Whisper's window is {MAX_SECONDS:.0f}s. "
            "Split it into segments before sending."
        )
    if duration < 0.1:
        raise ValueError(f"audio is {duration:.2f}s -- too short to transcribe")

    return DecodedAudio(waveform=wave, sample_rate=WHISPER_SR, duration_s=duration)
