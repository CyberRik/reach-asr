"""Re-analyse a finished eval run from `predictions.jsonl`. No model, no GPU.

`evaluate_wer.py` already dumps every hypothesis next to its reference, both raw
and normalised, with the per-utterance SNR and noise category. Everything needed
to put a confidence interval on the result and to break it down by condition is
therefore already on disk after the run -- what was missing was the code to do
it, not the data.

That matters more than it sounds. Re-running the eval costs three passes of
autoregressive generation on a GPU this project does not have day to day; this
runs on a laptop in seconds and produces numbers identical to what a re-run would
give, because it is scoring the same text the same way. Any analysis that can be
expressed over stored predictions should live here rather than behind another
generation pass.

    python -m reach_asr.analyze --predictions results/predictions.jsonl

Add --out to write the summary next to the run's own `wer.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reach_asr.stats import (
    WordCounts,
    corpus_wer,
    count_pair,
    derive_buckets,
    paired_bootstrap,
)

# The fields evaluate_wer.py writes. Normalised, not raw: the raw pair scores
# ~100% WER because LibriSpeech references are uppercase and unpunctuated while
# Whisper emits cased punctuated text, which is the whole reason the normaliser
# is in the scoring path to begin with.
REFERENCE_FIELD = "reference_norm"
ZEROSHOT_FIELD = "zeroshot_norm"
FINETUNED_FIELD = "finetuned_norm"


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    if REFERENCE_FIELD not in rows[0]:
        raise SystemExit(
            f"{path} has no '{REFERENCE_FIELD}' field -- this file was not written by "
            "evaluate_wer.py, or predates the normalised dump."
        )
    return rows


def score_rows(rows: list[dict[str, Any]], field: str) -> list[WordCounts]:
    """Per-utterance counts for one system, dropping empty references.

    An empty reference makes WER undefined. `evaluate_wer.score` drops those
    pairs; dropping them here too is what keeps the two paths comparable -- if
    this filtered differently, the re-analysis would disagree with the run's own
    headline number for a reason that has nothing to do with the model.
    """
    return [
        count_pair(row[REFERENCE_FIELD], row.get(field, ""))
        for row in rows
        if row[REFERENCE_FIELD].strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="write the summary as JSON")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-buckets", type=int, default=3)
    args = parser.parse_args()

    rows = load_predictions(args.predictions)
    scored = [row for row in rows if row[REFERENCE_FIELD].strip()]
    dropped = len(rows) - len(scored)

    has_finetuned = FINETUNED_FIELD in rows[0]

    zeroshot = score_rows(rows, ZEROSHOT_FIELD)
    summary: dict[str, Any] = {
        "predictions": str(args.predictions),
        "n_utterances": len(scored),
        "n_dropped_empty_reference": dropped,
        "wer_degraded_zeroshot": corpus_wer(zeroshot),
    }

    note = f"  ({dropped} dropped, empty reference)" if dropped else ""
    print(f"utterances: {len(scored)}{note}")
    print(f"degraded / zero-shot   {corpus_wer(zeroshot) * 100:6.2f}%")

    if not has_finetuned:
        print("\nno fine-tuned hypotheses in this file -- nothing to compare against.")
    else:
        finetuned = score_rows(rows, FINETUNED_FIELD)
        boot = paired_bootstrap(
            zeroshot,
            finetuned,
            n_resamples=args.resamples,
            confidence=args.confidence,
            seed=args.seed,
        )
        summary["wer_degraded_finetuned"] = corpus_wer(finetuned)
        summary["bootstrap"] = boot.as_dict()

        pct = int(args.confidence * 100)
        print(f"degraded / fine-tuned  {corpus_wer(finetuned) * 100:6.2f}%")
        print(f"\npaired bootstrap, {boot.n_resamples:,} resamples, {pct}% percentile interval")
        print(
            f"  absolute reduction   {boot.absolute_reduction.point * 100:6.2f} pp"
            f"   [{boot.absolute_reduction.low * 100:.2f},"
            f" {boot.absolute_reduction.high * 100:.2f}]"
        )
        print(
            f"  relative reduction   {boot.relative_reduction.point * 100:6.2f} %"
            f"   [{boot.relative_reduction.low * 100:.2f},"
            f" {boot.relative_reduction.high * 100:.2f}]"
        )
        print(f"  resamples with no improvement: {boot.p_no_improvement * 100:.2f}%")
        if boot.absolute_reduction.low <= 0.0:
            print(
                "\n  The interval includes zero. On this eval set the improvement is not\n"
                "  separable from sampling noise -- report it that way."
            )

    # --- per-condition breakdown ------------------------------------------
    snrs = [row.get("snr_db") for row in scored]
    buckets = derive_buckets(
        [s if s is not None else float("inf") for s in snrs], n_buckets=args.snr_buckets
    )
    if buckets:
        print(f"\nby SNR ({len(buckets)} buckets, edges from this run's own range)")
        header = f"  {'band':>18}  {'n':>4}  {'zero-shot':>10}"
        if has_finetuned:
            header += f"  {'fine-tuned':>11}  {'delta':>8}"
        print(header)

        rows_out: list[dict[str, Any]] = []
        for bucket in buckets:
            member_rows = [scored[i] for i in bucket.indices]
            base = score_rows(member_rows, ZEROSHOT_FIELD)
            base_wer = corpus_wer(base)
            entry: dict[str, Any] = {
                "band": bucket.label,
                "n": len(bucket.indices),
                "wer_zeroshot": base_wer,
            }
            line = f"  {bucket.label:>18}  {len(bucket.indices):>4}  {base_wer * 100:9.2f}%"
            if has_finetuned:
                tuned_wer = corpus_wer(score_rows(member_rows, FINETUNED_FIELD))
                entry["wer_finetuned"] = tuned_wer
                entry["absolute_reduction"] = base_wer - tuned_wer
                line += f"  {tuned_wer * 100:10.2f}%  {(base_wer - tuned_wer) * 100:7.2f}pp"
            rows_out.append(entry)
            print(line)
        summary["by_snr"] = rows_out

    # --- by noise category ------------------------------------------------
    # The other axis the manifest already carries. A fine-tune that only helps on
    # steady noise and not on transients is a different result from one that
    # helps uniformly, and neither the mean nor the SNR split would show it.
    categories = sorted({row.get("noise_category") for row in scored} - {None})
    if len(categories) > 1:
        print("\nby noise category")
        cat_out: list[dict[str, Any]] = []
        for category in categories:
            member_rows = [row for row in scored if row.get("noise_category") == category]
            base_wer = corpus_wer(score_rows(member_rows, ZEROSHOT_FIELD))
            entry = {"category": category, "n": len(member_rows), "wer_zeroshot": base_wer}
            line = f"  {category:>18}  {len(member_rows):>4}  {base_wer * 100:9.2f}%"
            if has_finetuned:
                tuned_wer = corpus_wer(score_rows(member_rows, FINETUNED_FIELD))
                entry["wer_finetuned"] = tuned_wer
                entry["absolute_reduction"] = base_wer - tuned_wer
                line += f"  {tuned_wer * 100:10.2f}%  {(base_wer - tuned_wer) * 100:7.2f}pp"
            cat_out.append(entry)
            print(line)
        summary["by_noise_category"] = cat_out

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
