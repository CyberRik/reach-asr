"""Load a checkpoint once, transcribe many times.

The model that `train.py` produces is a LoRA adapter, not a full checkpoint --
about 7 MB of low-rank deltas against `openai/whisper-base`. Serving it has one
decision worth making deliberately:

**Merge the adapter, don't keep it as a PEFT wrapper.** `merge_and_unload()`
folds the low-rank deltas into the base weights, giving a plain Whisper model at
inference time. A live `PeftModel` computes `Wx + BAx` as two extra matmuls per
attention projection on every forward pass -- pure overhead once training is
done, since the adapter is frozen and there is nothing left to switch between.
The merge costs a few seconds at startup and nothing per request.

Loading is lazy and cached: the first call pays it, and a process that never
transcribes never loads a 290 MB model. That matters because the health check
should be able to answer "am I up" without a GPU allocation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from reach_asr.audio_io import decode
from reach_asr.telephony import WHISPER_SR

DEFAULT_MODEL = "openai/whisper-base"

# Model loading is not thread-safe in the way an ASGI server needs: two
# concurrent first-requests would each pull a copy into memory. lru_cache alone
# does not prevent that -- it caches the result, not the in-flight call.
_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class Transcript:
    text: str
    duration_s: float
    latency_ms: float
    model: str
    adapter: str | None
    device: str


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_pipeline(
    model_name: str = DEFAULT_MODEL,
    adapter: str | None = None,
    device: str = "auto",
) -> tuple[Any, Any, str]:
    """Return (model, processor, device). Cached per (model, adapter, device)."""
    with _LOAD_LOCK:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        resolved = resolve_device(device)
        processor = WhisperProcessor.from_pretrained(
            model_name, language="english", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if resolved == "cuda" else torch.float32,
        )

        # Same three lines as evaluate_wer: Whisper ships a forced decoder
        # prefix meant for zero-shot language detection, and leaving it in place
        # makes the fine-tuned model decode under constraints it was not trained
        # under. Serving must match evaluation or the served WER is not the
        # measured WER.
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"

        if adapter:
            adapter_path = Path(adapter)
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"adapter not found: {adapter_path}\n"
                    "Train one with `python -m reach_asr.train`, or serve the base "
                    "model by leaving --adapter unset."
                )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()

        model = model.to(resolved).eval()
        return model, processor, resolved


@torch.no_grad()
def transcribe_bytes(
    data: bytes,
    filename: str | None = None,
    model_name: str = DEFAULT_MODEL,
    adapter: str | None = None,
    device: str = "auto",
    max_new_tokens: int = 200,
) -> Transcript:
    audio = decode(data, filename)
    model, processor, resolved = load_pipeline(model_name, adapter, device)

    started = time.perf_counter()
    features = processor.feature_extractor(
        audio.waveform.numpy(), sampling_rate=WHISPER_SR, return_tensors="pt"
    ).input_features.to(resolved)
    if resolved == "cuda":
        features = features.half()

    predicted = model.generate(features, max_new_tokens=max_new_tokens)
    text = processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()
    latency_ms = (time.perf_counter() - started) * 1000

    return Transcript(
        text=text,
        duration_s=round(audio.duration_s, 3),
        latency_ms=round(latency_ms, 1),
        model=model_name,
        adapter=adapter,
        device=resolved,
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Transcribe one audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = transcribe_bytes(
        args.audio.read_bytes(),
        filename=args.audio.name,
        model_name=args.model,
        adapter=args.adapter,
        device=args.device,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
