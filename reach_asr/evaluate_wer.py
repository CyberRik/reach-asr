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


def require_jiwer() -> None:
    """Fail before the generation passes, not after them.

    jiwer is only used at the very end, so importing it where it is used means a
    missing dependency surfaces after ~10 min of autoregressive decoding has
    already been thrown away. Kaggle's image does not ship it. Same reasoning as
    the adapter check below: everything cheap that can fail should fail first.
    """
    try:
        import jiwer  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit("jiwer is not installed -- `pip install jiwer`") from exc


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
    import jiwer  # noqa: PLC0415  -- checked eagerly in main(); see require_jiwer

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
    records: list[dict],
    references: list[str],
    hypotheses: list[str],
    normalizer: Any,
    n_buckets: int = 3,
) -> dict[str, float]:
    """WER bucketed by the SNR each utterance was actually mixed at.

    A single mean WER hides whether the fine-tune helped uniformly or only
    rescued the loudest cases -- and on emergency audio the low-SNR bucket is
    the one that matters, since a caller in a quiet room was never the problem.

    The edges come from the run's own SNR values (`stats.derive_buckets`). They
    used to be hardcoded at 5-10/10-15/15-20 dB, which silently broke the moment
    the channel was hardened: the reported run used -5 to 10 dB, so every
    utterance fell in the bottom bucket and this function returned the corpus
    mean under a label claiming it was a per-condition breakdown. A diagnostic
    that degrades into the number it is supposed to decompose is worse than one
    that is absent, because nothing about the output says it has stopped working.
    """
    from reach_asr.stats import derive_buckets

    snrs = [record.get("snr_db", float("inf")) for record in records]
    out: dict[str, float] = {}
    for bucket in derive_buckets(snrs, n_buckets=n_buckets):
        out[bucket.label] = score(
            [references[i] for i in bucket.indices],
            [hypotheses[i] for i in bucket.indices],
            normalizer,
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
    parser.add_argument(
        "--passes",
        default="all",
        help=(
            "comma-separated subset of clean_zeroshot, degraded_zeroshot, "
            "degraded_finetuned, clean_finetuned. A subset merges into an existing "
            "--out rather than overwriting it, so the missing cell of a finished run "
            "can be filled in with one generation pass instead of four."
        ),
    )
    parser.add_argument("--snr-buckets", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    require_jiwer()

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

    # The design is a 2x2: audio condition against model.
    #
    #                 zero-shot                fine-tuned
    #     clean       the ceiling              THE SPECIALISATION CHECK
    #     degraded    the baseline             the result
    #
    # Only three of these were measured originally, and the missing one is the
    # only cell that can distinguish two very different outcomes:
    #
    #   "it learned to handle phone audio"       clean/fine-tuned ~= clean/zero-shot
    #   "it learned to ONLY handle phone audio"  clean/fine-tuned >> clean/zero-shot
    #
    # A LoRA trained exclusively on one narrow degraded channel can buy its
    # degraded-audio gain by giving up the wideband case, and at low rank that is
    # a routine outcome rather than an exotic one. Without the fourth cell the
    # headline improvement is not wrong, but it is unaudited: nobody can tell
    # whether the model got better or merely got narrower. It costs one more
    # generation pass and no retraining, which is a poor reason to leave the
    # question open.
    PASS_SPECS = {
        # name: (audio key, needs adapter, human label)
        "clean_zeroshot": ("clean_audio", False, "zero-shot on CLEAN audio (the ceiling)"),
        "degraded_zeroshot": ("audio", False, "zero-shot on DEGRADED audio (the baseline)"),
        "degraded_finetuned": ("audio", True, "fine-tuned on DEGRADED audio (the result)"),
        "clean_finetuned": (
            "clean_audio",
            True,
            "fine-tuned on CLEAN audio (the specialisation check)",
        ),
    }

    has_clean = "clean_audio" in records[0]
    selected = [name.strip() for name in args.passes.split(",")] if args.passes != "all" else None
    if selected:
        unknown = set(selected) - set(PASS_SPECS)
        if unknown:
            raise SystemExit(f"unknown pass(es): {', '.join(sorted(unknown))}")

    wanted: list[str] = []
    for name, (audio_key, needs_adapter, _) in PASS_SPECS.items():
        if selected is not None and name not in selected:
            continue
        if needs_adapter and args.adapter is None:
            continue
        if audio_key == "clean_audio" and (args.skip_clean or not has_clean):
            continue
        wanted.append(name)

    if not wanted:
        raise SystemExit("no passes to run -- check --passes, --adapter and --skip-clean")

    # Running a subset means the other cells live in a previous run's file.
    # Overwriting it with only the new pass would silently destroy three numbers
    # that cost half an hour of GPU to produce, so a partial run merges instead.
    if selected is not None and args.out.exists():
        results.update(json.loads(args.out.read_text(encoding="utf-8")))
        print(f"merging into existing {args.out}")

    models: dict[bool, Any] = {}

    def model_for(needs_adapter: bool) -> Any:
        if needs_adapter not in models:
            if needs_adapter:
                from peft import PeftModel

                tuned = PeftModel.from_pretrained(fresh_base(), str(args.adapter))
                models[True] = tuned.merge_and_unload().to(device).eval()
            else:
                models[False] = fresh_base()
        return models[needs_adapter]

    hyps: dict[str, list[str]] = {}
    for step, name in enumerate(wanted, start=1):
        audio_key, needs_adapter, label = PASS_SPECS[name]
        print(f"[{step}/{len(wanted)}] {label}...")
        hyps[name] = transcribe_all(
            model_for(needs_adapter), processor, eval_root, records, audio_key, device
        )
        results[f"wer_{name}"] = score(references, hyps[name], normalizer)
        print(f"      WER {results[f'wer_{name}']:.4f}")

    for name in ("degraded_zeroshot", "degraded_finetuned"):
        if name in hyps:
            results[f"wer_{name}_by_snr"] = wer_by_snr(
                records, references, hyps[name], normalizer, n_buckets=args.snr_buckets
            )

    base_wer = results.get("wer_degraded_zeroshot")
    tuned_wer = results.get("wer_degraded_finetuned")
    if base_wer is not None and tuned_wer is not None:
        results["relative_wer_reduction"] = (base_wer - tuned_wer) / base_wer if base_wer else 0.0

    # --- the two comparisons, each with its own interval -------------------
    #
    # Both are paired: every pass scores the same 300 utterances, and for the
    # clean pair it is literally the same audio file through two models.
    from reach_asr.stats import count_pair, paired_bootstrap

    def bootstrap_between(baseline_name: str, system_name: str) -> Any:
        if baseline_name not in hyps or system_name not in hyps:
            return None
        rows = [
            (normalizer(ref), normalizer(a), normalizer(b))
            for ref, a, b in zip(references, hyps[baseline_name], hyps[system_name], strict=True)
            if normalizer(ref).strip()
        ]
        return paired_bootstrap(
            [count_pair(ref, a) for ref, a, _ in rows],
            [count_pair(ref, b) for ref, _, b in rows],
            n_resamples=args.resamples,
            seed=args.seed,
        )

    boot = bootstrap_between("degraded_zeroshot", "degraded_finetuned")
    if boot is not None:
        results["bootstrap"] = boot.as_dict()
        print(
            f"\n95% CI on the degraded-audio reduction: "
            f"[{boot.absolute_reduction.low * 100:.2f},"
            f" {boot.absolute_reduction.high * 100:.2f}] pp"
        )
        if boot.absolute_reduction.low <= 0.0:
            print("  NOTE: the interval includes zero -- report it that way.")

    # Sign convention: this is deliberately NOT a "reduction". A positive
    # absolute_reduction here would mean the fine-tune improved clean audio too;
    # the expected and worrying case is negative, i.e. clean WER got worse. The
    # key is named for what it measures so nobody reads a regression as a gain.
    spec = bootstrap_between("clean_zeroshot", "clean_finetuned")
    if spec is not None:
        clean_zero = results["wer_clean_zeroshot"]
        clean_tuned = results["wer_clean_finetuned"]
        cost = clean_tuned - clean_zero
        results["specialisation"] = {
            "clean_wer_change_pp": cost,
            "bootstrap": spec.as_dict(),
        }
        print(
            f"\nspecialisation check -- clean WER {clean_zero * 100:.2f}% -> "
            f"{clean_tuned * 100:.2f}% ({cost * 100:+.2f} pp)"
        )
        print(
            f"  95% CI on the change: "
            f"[{-spec.absolute_reduction.high * 100:+.2f},"
            f" {-spec.absolute_reduction.low * 100:+.2f}] pp"
        )
        if spec.absolute_reduction.low <= 0.0 <= spec.absolute_reduction.high:
            print("  Clean-audio ability is intact within the interval.")
        elif cost > 0:
            print(
                "  Clean audio got WORSE. The fine-tune bought its degraded-audio\n"
                "  gain by specialising; report both numbers together."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Dump every hypothesis next to its reference, normalised and raw.
    #
    # A WER number tells you the fine-tune failed; it does not tell you HOW.
    # Truncation, hallucinated continuations, and a systematic formatting shift
    # all look identical in the aggregate. The first run cost a full re-read of
    # the training manifest to work out that the model had learned to emit ALL
    # CAPS -- which one glance at ten hypotheses would have shown immediately.
    #
    # Field names keep the legacy `zeroshot_*` / `finetuned_*` spelling for the
    # degraded pair so that analyze.py and any previously saved file still line
    # up; the clean pair gets its own prefix.
    predictions = args.out.parent / "predictions.jsonl"
    legacy = {"degraded_zeroshot": "zeroshot", "degraded_finetuned": "finetuned"}
    with predictions.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            row = {
                "id": record["id"],
                "snr_db": record.get("snr_db"),
                "noise_category": record.get("noise_category"),
                "reference_raw": references[index],
                "reference_norm": normalizer(references[index]),
            }
            for name, values in hyps.items():
                prefix = legacy.get(name, name)
                row[f"{prefix}_raw"] = values[index]
                row[f"{prefix}_norm"] = normalizer(values[index])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"per-utterance predictions -> {predictions}")

    print("\n" + "=" * 58)
    for key in (
        "wer_clean_zeroshot",
        "wer_clean_finetuned",
        "wer_degraded_zeroshot",
        "wer_degraded_finetuned",
    ):
        if key in results:
            print(f"{key:32s} {results[key] * 100:6.2f}%")
    if "relative_wer_reduction" in results:
        print(f"{'relative WER reduction':32s} {results['relative_wer_reduction'] * 100:6.2f}%")
    if "specialisation" in results:
        change = results["specialisation"]["clean_wer_change_pp"] * 100
        print(f"{'clean-audio change (pp)':32s} {change:+6.2f}")
    print("=" * 58)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
