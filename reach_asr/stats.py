"""Confidence intervals and SNR bucketing for WER results.

Two gaps this closes, both of which made the headline result weaker than it
needed to be.

**A WER delta without an interval is not a result.** The reported run moved
degraded WER from 23.76% to 21.20% on 300 utterances -- one seed, one training
run. Quoted bare, 2.56 points invites exactly the question it cannot answer: is
that the fine-tune, or is it 300 draws from a distribution whose spread nobody
measured? `paired_bootstrap` answers it by resampling utterances with
replacement and recomputing corpus WER for both systems on each resample.

The pairing is the part that matters. Both systems are scored on byte-identical
audio (`telephony.degrade` is seed-deterministic precisely so they are), so the
same resampled indices must be applied to both. An unpaired bootstrap would
throw that away and widen the interval with variance that the experiment design
already eliminated -- it would understate the evidence, which is the same kind
of error as overstating it, just in the safer direction.

**A bucket range that does not match the run tells you nothing.** The original
breakdown hardcoded 5-10/10-15/15-20 dB while the reported run used -5 to 10 dB,
so all 300 utterances landed in one bucket and the per-condition diagnostic --
the thing the breakdown exists for -- produced a single number identical to the
mean. `derive_buckets` takes the edges from the SNR values actually present.

Nothing here imports torch, transformers or peft. Everything operates on
reference/hypothesis text that `evaluate_wer.py` has already normalised and
written to `predictions.jsonl`, which is what makes re-analysis of a finished run
a CPU job with no model load and no GPU.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

# Corpus WER is a ratio of sums, not a mean of ratios:
#
#     WER = (S + D + I) / (S + D + H)
#
# summed over the corpus. That is what jiwer.wer(list, list) computes and what
# Whisper's published numbers use. Keeping the per-utterance numerator and
# denominator separately is what makes the bootstrap cheap: each resample is a
# sum over sampled pairs, not a re-alignment. Re-running the Levenshtein
# alignment 10,000 times would take minutes; summing precomputed integers takes
# milliseconds, and the two are numerically identical.


@dataclass(frozen=True)
class WordCounts:
    """Edit count and reference length for one utterance."""

    edits: int
    ref_words: int


def count_pair(reference: str, hypothesis: str) -> WordCounts:
    """Align one reference/hypothesis pair and return its WER numerator and denominator.

    Text is used exactly as given. `predictions.jsonl` stores both a raw and a
    `_norm` field; callers must pass the normalised one, or the numbers will not
    match the run they are re-analysing.
    """
    import jiwer

    out = jiwer.process_words([reference], [hypothesis])
    return WordCounts(
        edits=out.substitutions + out.deletions + out.insertions,
        ref_words=out.substitutions + out.deletions + out.hits,
    )


def corpus_wer(counts: Sequence[WordCounts]) -> float:
    """Corpus-level WER over a set of per-utterance counts."""
    denominator = sum(c.ref_words for c in counts)
    if denominator == 0:
        return float("nan")
    return sum(c.edits for c in counts) / denominator


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile confidence interval."""

    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}


@dataclass(frozen=True)
class BootstrapResult:
    """Paired bootstrap over two systems scored on the same utterances."""

    n_utterances: int
    n_resamples: int
    confidence: float
    baseline: Interval
    system: Interval
    absolute_reduction: Interval
    relative_reduction: Interval
    p_no_improvement: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_utterances": self.n_utterances,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
            "baseline_wer": self.baseline.as_dict(),
            "system_wer": self.system.as_dict(),
            "absolute_wer_reduction": self.absolute_reduction.as_dict(),
            "relative_wer_reduction": self.relative_reduction.as_dict(),
            "p_no_improvement": self.p_no_improvement,
        }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    baseline: Sequence[WordCounts],
    system: Sequence[WordCounts],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap over utterances, resampling both systems together.

    `baseline[i]` and `system[i]` must be the same utterance scored by the two
    models -- that is the pairing, and swapping the order of one list silently
    produces a wider interval that looks like a legitimate result.

    Percentile rather than BCa: BCa's bias and acceleration corrections matter
    most for small samples and skewed statistics, and at n=300 with a ratio of
    sums the difference is well inside the width being reported. Percentile is
    also the one a reader can reconstruct from the description alone, which for a
    number that exists to be checked is worth more than the third decimal place.
    """
    if len(baseline) != len(system):
        raise ValueError(
            f"paired bootstrap needs matched utterances: got {len(baseline)} baseline "
            f"and {len(system)} system counts"
        )
    if not baseline:
        raise ValueError("no utterances to bootstrap")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n = len(baseline)
    rng = random.Random(seed)

    baseline_wers: list[float] = []
    system_wers: list[float] = []
    absolute: list[float] = []
    relative: list[float] = []
    not_improved = 0

    for _ in range(n_resamples):
        # One index list, applied to both systems. This is the pairing.
        indices = [rng.randrange(n) for _ in range(n)]

        base_edits = base_words = sys_edits = sys_words = 0
        for i in indices:
            base_edits += baseline[i].edits
            base_words += baseline[i].ref_words
            sys_edits += system[i].edits
            sys_words += system[i].ref_words

        # A resample can in principle draw only empty references. Skipping it is
        # the honest handling; scoring it as 0 or 1 would drag the interval.
        if base_words == 0 or sys_words == 0:
            continue

        base_wer = base_edits / base_words
        sys_wer = sys_edits / sys_words
        baseline_wers.append(base_wer)
        system_wers.append(sys_wer)
        absolute.append(base_wer - sys_wer)
        relative.append((base_wer - sys_wer) / base_wer if base_wer else 0.0)
        if sys_wer >= base_wer:
            not_improved += 1

    if not absolute:
        raise ValueError("every resample was degenerate; check that references are non-empty")

    tail = (1.0 - confidence) / 2.0

    def interval(point: float, draws: list[float]) -> Interval:
        ordered = sorted(draws)
        return Interval(
            point=point,
            low=_percentile(ordered, tail),
            high=_percentile(ordered, 1.0 - tail),
        )

    point_baseline = corpus_wer(baseline)
    point_system = corpus_wer(system)
    point_absolute = point_baseline - point_system
    point_relative = point_absolute / point_baseline if point_baseline else 0.0

    return BootstrapResult(
        n_utterances=n,
        n_resamples=n_resamples,
        confidence=confidence,
        baseline=interval(point_baseline, baseline_wers),
        system=interval(point_system, system_wers),
        absolute_reduction=interval(point_absolute, absolute),
        relative_reduction=interval(point_relative, relative),
        # One-sided: the fraction of resamples in which the fine-tune did not
        # beat the baseline. Not a p-value from a null model -- it is the
        # bootstrap's own answer to "how often does this comparison come out the
        # other way", which is the question actually being asked.
        p_no_improvement=not_improved / len(absolute),
    )


@dataclass(frozen=True)
class Bucket:
    """One SNR band, with the half-open range it covers."""

    label: str
    low: float
    high: float
    indices: list[int]


def derive_buckets(
    snr_values: Sequence[float], n_buckets: int = 3, decimals: int = 0
) -> list[Bucket]:
    """Bucket utterances by SNR using edges taken from the data.

    The previous version hardcoded 5-10/10-15/15-20 dB. Any run outside that
    window collapsed into one bucket and the breakdown reported the corpus mean
    three times over -- which is worse than reporting nothing, because it looks
    like a per-condition result.

    Equal-width bins over the observed range rather than quantiles: the label has
    to mean something acoustically ("-5 to 0 dB" is a condition; "the bottom
    third of this run" is not), and comparing buckets across runs requires edges
    that do not move with the sample. Empty buckets are dropped rather than
    reported as 0%, since an empty bucket is a gap in the corpus, not a score.

    Utterances with a non-finite SNR -- what `degrade` returns when noise is
    disabled -- are excluded rather than binned at an arbitrary edge.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    finite = [(i, float(v)) for i, v in enumerate(snr_values) if math.isfinite(float(v))]
    if not finite:
        return []

    low = min(v for _, v in finite)
    high = max(v for _, v in finite)

    # A run at a single fixed SNR is legitimate -- it just has one bucket.
    if math.isclose(low, high):
        return [
            Bucket(
                label=f"{low:.{decimals}f} dB",
                low=low,
                high=high,
                indices=[i for i, _ in finite],
            )
        ]

    width = (high - low) / n_buckets
    buckets: list[Bucket] = []
    for b in range(n_buckets):
        edge_low = low + b * width
        # Last bucket takes the upper edge so the maximum lands somewhere.
        edge_high = high if b == n_buckets - 1 else low + (b + 1) * width
        members = [
            i
            for i, v in finite
            if (edge_low <= v < edge_high) or (b == n_buckets - 1 and v == high)
        ]
        if not members:
            continue
        buckets.append(
            Bucket(
                label=f"{edge_low:.{decimals}f} to {edge_high:.{decimals}f} dB",
                low=edge_low,
                high=edge_high,
                indices=members,
            )
        )
    return buckets
