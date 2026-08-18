"""Build a paired (degraded audio, transcript) corpus on disk.

Streams a bounded number of utterances from LibriSpeech and a bank of
environmental noise from ESC-50, applies `telephony.degrade`, and writes WAV
files plus a JSONL manifest. Nothing is held in memory beyond one utterance, so
this runs in a few hundred MB regardless of how many samples are requested.

Why write files instead of an in-memory HF dataset: the degraded audio is the
experiment. Having it on disk means it can be listened to, re-measured, and
diffed between runs -- and it means the eval set is a fixed artefact rather than
something regenerated (and therefore potentially different) each time a
baseline is measured.

The train and eval splits get *disjoint* seed ranges and disjoint noise clips.
Sharing a noise bank across splits would let the model memorise the specific
siren recordings in eval, which inflates the WER improvement without any real
robustness gain -- the classic way an augmentation experiment lies to you.

    python -m reach_asr.build_dataset --train 2000 --eval 300
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torchaudio

from reach_asr.telephony import WHISPER_SR, DegradationConfig, degrade

# Noise classes that plausibly sit behind an emergency call. ESC-50 has 50
# categories; most (chainsaw, washing machine, church bells) would just be
# arbitrary interference. These are chosen for face validity -- an interviewer
# should be able to see why each one is here.
EMERGENCY_NOISE_CATEGORIES = {
    "siren",
    "car_horn",
    "crying_baby",
    "helicopter",
    "engine",
    "crackling_fire",
    "thunderstorm",
    "rain",
    "wind",
    "footsteps",
    "door_wood_knock",
    "clock_alarm",
    "glass_breaking",
    "coughing",
    "breathing",
}


def load_noise_bank(limit: int) -> list[tuple[str, torch.Tensor]]:
    """ESC-50 clips, resampled to Whisper's rate, filtered to the classes above."""
    from datasets import load_dataset

    print(f"loading ESC-50 noise bank (target {limit} clips)...")
    dataset = load_dataset("ashraq/esc50", split="train")

    bank: list[tuple[str, torch.Tensor]] = []
    for row in dataset:
        category = row.get("category")
        if category not in EMERGENCY_NOISE_CATEGORIES:
            continue
        audio = row["audio"]
        wave = torch.tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
        if sr != WHISPER_SR:
            wave = torchaudio.functional.resample(wave, sr, WHISPER_SR)
        if wave.abs().max() < 1e-6:
            continue  # a silent clip contributes no noise at any SNR
        bank.append((category, wave))
        if len(bank) >= limit:
            break

    if not bank:
        raise RuntimeError("ESC-50 yielded no usable clips in the selected categories")
    print(f"  {len(bank)} noise clips across {len({c for c, _ in bank})} categories")
    return bank


def stream_librispeech(split: str, limit: int, min_seconds: float, max_seconds: float):
    """Yield (id, waveform, transcript) from LibriSpeech, streamed.

    Streaming rather than downloading: train.clean.100 is ~6 GB and this needs a
    couple of thousand utterances. Length bounds keep every clip inside
    Whisper's 30 s window without truncation (which would silently make the
    reference transcript wrong for the audio, poisoning both training and WER).
    """
    from datasets import load_dataset

    # No trust_remote_code: datasets 4.x removed script-based loading entirely,
    # and passing it now raises on an unexpected kwarg. LibriSpeech is
    # parquet-converted on the Hub, so plain streaming is the supported path.
    dataset = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
    taken = 0
    for row in dataset:
        audio = row["audio"]
        wave = torch.tensor(audio["array"], dtype=torch.float32)
        sr = int(audio["sampling_rate"])
        if sr != WHISPER_SR:
            wave = torchaudio.functional.resample(wave, sr, WHISPER_SR)
        seconds = wave.shape[-1] / WHISPER_SR
        if not (min_seconds <= seconds <= max_seconds):
            continue
        yield row["id"], wave, row["text"]
        taken += 1
        if taken >= limit:
            return


def build_split(
    name: str,
    hf_split: str,
    count: int,
    noise_bank: list[tuple[str, torch.Tensor]],
    config: DegradationConfig,
    out_root: Path,
    seed_base: int,
    keep_clean: bool,
) -> Path:
    audio_dir = out_root / name / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / name / "manifest.jsonl"

    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (utt_id, wave, text) in enumerate(
            stream_librispeech(hf_split, count, min_seconds=1.0, max_seconds=25.0)
        ):
            seed = seed_base + index
            category, noise = noise_bank[index % len(noise_bank)]
            degraded, snr_db = degrade(wave, WHISPER_SR, config, noise, seed)

            rel = f"audio/{utt_id}.wav"
            torchaudio.save(str(out_root / name / rel), degraded.unsqueeze(0), WHISPER_SR)

            record = {
                "id": utt_id,
                "audio": rel,
                "text": text,
                "snr_db": round(snr_db, 2),
                "noise_category": category,
                "seed": seed,
                "duration_s": round(wave.shape[-1] / WHISPER_SR, 3),
            }

            if keep_clean:
                # The undegraded original, kept only for eval. It is what makes
                # "how much of the WER gap is the channel" answerable, instead
                # of leaving the degradation an unquantified confound.
                clean_rel = f"audio/{utt_id}.clean.wav"
                torchaudio.save(str(out_root / name / clean_rel), wave.unsqueeze(0), WHISPER_SR)
                record["clean_audio"] = clean_rel

            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  {name}: {written}/{count}")

    print(f"{name}: wrote {written} utterances -> {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--eval", type=int, default=300)
    parser.add_argument("--noise-clips", type=int, default=180)
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--packet-loss", type=float, default=0.02)
    args = parser.parse_args()

    config = DegradationConfig(
        snr_db_min=args.snr_min,
        snr_db_max=args.snr_max,
        packet_loss_rate=args.packet_loss,
    )

    bank = load_noise_bank(args.noise_clips)
    # Disjoint noise: the eval set never hears a clip the training set used.
    split_at = max(1, int(len(bank) * 0.8))
    train_bank, eval_bank = bank[:split_at], bank[split_at:]
    if not eval_bank:
        raise RuntimeError("noise bank too small to split; raise --noise-clips")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "degradation": asdict(config),
                "train_utterances": args.train,
                "eval_utterances": args.eval,
                "noise_clips_train": len(train_bank),
                "noise_clips_eval": len(eval_bank),
                "source_corpus": "openslr/librispeech_asr (clean)",
                "noise_corpus": "ashraq/esc50",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    build_split("train", "train.100", args.train, train_bank, config, args.out, 1_000_000, False)
    # keep_clean on eval only: it doubles disk for the split it is useful on.
    build_split("eval", "validation", args.eval, eval_bank, config, args.out, 9_000_000, True)


if __name__ == "__main__":
    main()
