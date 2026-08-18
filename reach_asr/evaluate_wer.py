"""Measure WER: zero-shot Whisper vs the fine-tuned adapter, on degraded audio.

Reports three numbers, because two of them would be misleading alone:

* **clean / zero-shot** -- the ceiling. What this checkpoint scores on the same
  utterances before the channel touched them. Without it, "12% WER" is
  unreadable: it could be a hard corpus or a hard channel and there is no way to
  tell which.
* **degraded / zero-shot** -- the baseline the fine-tune has to beat. The gap
  between this and the ceiling is the cost of the channel, and it is the only
  thing fine-tuning can recover.
* **degraded / fine-tuned** -- the result.

Quoting only the third against the first would attribute the entire clean-to-
degraded gap to the fine-tune, which is the standard way this experiment is
oversold.

Text is compared through Whisper's own EnglishTextNormalizer. LibriSpeech
references are uppercase and unpunctuated while Whisper emits cased, punctuated
text, so raw WER would be ~100% for reasons that have nothing to do with
recognition. The normalizer is the same one OpenAI reports Whisper's published
WER with, so these numbers are comparable to the model card rather than to a
bespoke scoring function.

    python -m reach_asr.evaluate_wer --adapter runs/whisper-lora/adapter
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@torch.no_grad()
def transcribe_all(
    model: Any, processor: Any, root: Path, records: list[dict], key: str, device: str
) -> list[str]:
    import torchaudio

    outputs: list[str] = []
    for index, record in enumerate(records):
        wave, sr = torchaudio.load(str(root / record[key]))
        features = processor.feature_extractor(
            wave.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device)
        if device == "cuda":
            features = features.half()
        predicted = model.generate(features, max_new_tokens=200)
        outputs.append(processor.batch_decode(predicted, skip_special_tokens=True)[0])
        if (index + 1) % 25 == 0:
            print(f"    {index + 1}/{len(records)}")
    return outputs


def score(references: list[str], hypotheses: list[str], normalizer: Any) -> float:
    import jiwer

    refs = [normalizer(text) for text in references]
    hyps = [normalizer(text) for text in hypotheses]
    # An empty reference makes WER undefined (division by zero words). Dropping
    # the pair is the honest handling; silently scoring it as 0 or 100 would
    # bias the mean in whichever direction was chosen.
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r.strip()]
    if not pairs:
        return float("nan")
    return float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))


def wer_by_snr(
    records: list[dict], references: list[str], hypotheses: list[str], normalizer: Any
) -> dict[str, float]:
    """WER bucketed by the SNR each utterance was actually mixed at.

    A single mean WER hides whether the fine-tune helped uniformly or only
    rescued the loudest cases -- and on emergency audio the low-SNR bucket is
    the one that matters, since a caller in a quiet room was never the problem.
    """
    buckets: dict[str, list[int]] = {"5-10 dB": [], "10-15 dB": [], "15-20 dB": []}
    for index, record in enumerate(records):
        snr = record.get("snr_db", float("inf"))
        if snr < 10:
            buckets["5-10 dB"].append(index)
        elif snr < 15:
            buckets["10-15 dB"].append(index)
        else:
            buckets["15-20 dB"].append(index)

    out: dict[str, float] = {}
    for name, indices in buckets.items():
        if indices:
            out[name] = score(
                [references[i] for i in indices], [hypotheses[i] for i in indices], normalizer
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/wer.json"))
    parser.add_argument("--skip-clean", action="store_true")
    args = parser.parse_args()

    # Checked before the model imports, and long before the expensive passes.
    # The two zero-shot passes are ~10 min of autoregressive generation, and
    # discovering a missing adapter only at step [3/3] throws all of it away.
    # Not hypothetical: a PEFT/torchao version clash killed training on Kaggle
    # while the eval command in the same cell ran on regardless -- `!a` then
    # `!b` in one notebook cell does not short-circuit on failure.
    if args.adapter is not None and not args.adapter.exists():
        raise SystemExit(
            f"adapter not found: {args.adapter}\n"
            "Training did not produce one -- check the train step's output for an "
            "error before rerunning.\nTo measure only the zero-shot baselines, drop "
            "--adapter."
        )

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_root = args.data / "eval"
    records = load_manifest(eval_root / "manifest.jsonl")
    if args.limit:
        records = records[: args.limit]
    references = [record["text"] for record in records]
    print(f"eval utterances: {len(records)}  device: {device}")

    processor = WhisperProcessor.from_pretrained(args.model, language="english", task="transcribe")
    normalizer = EnglishTextNormalizer(processor.tokenizer.english_spelling_normalizer)

    def fresh_base() -> Any:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        return model.to(device).eval()

    results: dict[str, Any] = {
        "model": args.model,
        "n_utterances": len(records),
        "mean_snr_db": round(
            statistics.fmean(r["snr_db"] for r in records if r.get("snr_db") != float("inf")), 2
        ),
    }

    base = fresh_base()

    if not args.skip_clean and "clean_audio" in records[0]:
        print("[1/3] zero-shot on CLEAN audio (the ceiling)...")
        hyps = transcribe_all(base, processor, eval_root, records, "clean_audio", device)
        results["wer_clean_zeroshot"] = score(references, hyps, normalizer)
        print(f"      WER {results['wer_clean_zeroshot']:.4f}")

    print("[2/3] zero-shot on DEGRADED audio (the baseline)...")
    hyps_base = transcribe_all(base, processor, eval_root, records, "audio", device)
    results["wer_degraded_zeroshot"] = score(references, hyps_base, normalizer)
    results["wer_degraded_zeroshot_by_snr"] = wer_by_snr(records, references, hyps_base, normalizer)
    print(f"      WER {results['wer_degraded_zeroshot']:.4f}")

    if args.adapter is not None:
        from peft import PeftModel

        print("[3/3] fine-tuned on DEGRADED audio...")
        tuned = PeftModel.from_pretrained(fresh_base(), str(args.adapter))
        tuned = tuned.merge_and_unload().to(device).eval()
        hyps_tuned = transcribe_all(tuned, processor, eval_root, records, "audio", device)
        results["wer_degraded_finetuned"] = score(references, hyps_tuned, normalizer)
        results["wer_degraded_finetuned_by_snr"] = wer_by_snr(
            records, references, hyps_tuned, normalizer
        )
        base_wer = results["wer_degraded_zeroshot"]
        tuned_wer = results["wer_degraded_finetuned"]
        results["relative_wer_reduction"] = (base_wer - tuned_wer) / base_wer if base_wer else 0.0
        print(f"      WER {tuned_wer:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 58)
    for key in ("wer_clean_zeroshot", "wer_degraded_zeroshot", "wer_degraded_finetuned"):
        if key in results:
            print(f"{key:32s} {results[key] * 100:6.2f}%")
    if "relative_wer_reduction" in results:
        print(f"{'relative WER reduction':32s} {results['relative_wer_reduction'] * 100:6.2f}%")
    print("=" * 58)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
