"""HTTP service exposing the fine-tuned model to the REACH app.

    uvicorn reach_asr.serve:app --port 8081
    REACH_ASR_ADAPTER=runs/whisper-lora/adapter uvicorn reach_asr.serve:app --port 8081

Why a separate service rather than transcribing inside the Next.js app: the
model is a 290 MB PyTorch graph that wants a GPU and holds it for the process
lifetime. Next.js route handlers are serverless-shaped -- they scale to zero and
spin up per request -- which is the opposite of what a resident model needs. Two
processes with an HTTP boundary means REACH can deploy on Vercel unchanged and
this can sit on whatever has the GPU.

The boundary also degrades honestly. If this service is down, REACH's own route
returns a 503 and the dispatcher sees "transcription unavailable" next to a
still-playable recording -- the audio is never lost to an ASR failure, which is
the only acceptable behaviour when the recording is someone's emergency report.

CONCURRENCY
-----------
One model, one GPU, and `generate()` is not reentrant on a shared module in any
way worth relying on. Requests are serialised through a semaphore rather than
allowed to interleave: a queue of 3 requests that each take 800 ms is strictly
better than 3 that thrash a 4 GB card into an OOM at 2.4 s. Raise
REACH_ASR_CONCURRENCY only if you have measured headroom.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from reach_asr.audio_io import ffmpeg_available
from reach_asr.transcribe import DEFAULT_MODEL, load_pipeline, resolve_device, transcribe_bytes

logger = logging.getLogger("reach_asr.serve")

MODEL_NAME = os.environ.get("REACH_ASR_MODEL", DEFAULT_MODEL)
ADAPTER = os.environ.get("REACH_ASR_ADAPTER") or None
DEVICE = os.environ.get("REACH_ASR_DEVICE", "auto")
CONCURRENCY = int(os.environ.get("REACH_ASR_CONCURRENCY", "1"))
# 30 s of 16-bit 48 kHz stereo is ~5.8 MB; 25 MB leaves room for a wasteful
# container without letting an unbounded upload sit in memory.
MAX_UPLOAD_BYTES = int(os.environ.get("REACH_ASR_MAX_BYTES", str(25 * 1024 * 1024)))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("REACH_ASR_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

_semaphore = asyncio.Semaphore(CONCURRENCY)


class TranscriptResponse(BaseModel):
    text: str
    duration_s: float
    latency_ms: float
    model: str
    adapter: str | None
    device: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load at startup, not on the first request.
    #
    # A cold load is 5-20 s. Paying it lazily means the first real transcription
    # request times out at the client while the model is still coming off disk,
    # and an orchestrator's readiness probe reports healthy throughout. Loading
    # here means the process is not accepting traffic until it can serve it.
    if not ffmpeg_available():
        logger.warning(
            "ffmpeg is not on PATH -- WebM/Opus and MP4/AAC uploads will be rejected. "
            "Browser recordings are one of those two on every major engine."
        )
    logger.info("loading %s (adapter=%s) ...", MODEL_NAME, ADAPTER or "none")
    await asyncio.to_thread(load_pipeline, MODEL_NAME, ADAPTER, DEVICE)
    logger.info("model ready on %s", resolve_device(DEVICE))
    yield


app = FastAPI(title="reach-asr", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "adapter": ADAPTER,
        "device": resolve_device(DEVICE),
        "ffmpeg": ffmpeg_available(),
        "concurrency": CONCURRENCY,
    }


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptResponse:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload is {len(data) / 1e6:.1f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
        )

    async with _semaphore:
        try:
            # to_thread, not a direct call: generate() is synchronous and CPU/GPU
            # bound, and running it on the event loop would block every other
            # connection for the duration -- including /health, which is how a
            # busy service gets killed by its own liveness probe.
            result = await asyncio.to_thread(
                transcribe_bytes, data, file.filename, MODEL_NAME, ADAPTER, DEVICE
            )
        except ValueError as exc:
            # Bad audio: the client's problem, and the message is specific
            # enough to act on ("42s exceeds the 30s window").
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            # Missing ffmpeg, CUDA OOM: ours, not theirs.
            logger.exception("transcription failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TranscriptResponse(**result.__dict__)
