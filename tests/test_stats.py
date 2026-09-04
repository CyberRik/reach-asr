"""Tests for the confidence interval and the SNR bucketing.

Same standard as `test_telephony.py`: assert measurable properties, not that the
code runs. A bootstrap that silently drops its pairing, or a bucketer that
collapses every utterance into one bin, both return well-formed output that is
wrong in exactly the way that produces a confident false claim -- which is the
failure mode this project already got caught by once.
"""

from __future__ import annotations

import math

import pytest

from reach_asr.stats import (
    WordCounts,
    corpus_wer,
    count_pair,
    derive_buckets,
    paired_bootstrap,
)

# --- scoring ---------------------------------------------------------------


def test_count_pair_matches_hand_computed_edits() -> None:
    counts = count_pair("the cat sat on the mat", "the cat sit on the mat")
    assert counts.edits == 1
    assert counts.ref_words == 6


def test_corpus_wer_is_a_ratio_of_sums_not_a_mean_of_ratios() -> None:
    """The distinction that makes the bootstrap valid.

    A 1-word utterance with 1 error and a 99-word utterance with 0 errors is 1%
    corpus WER, not the 50% a mean of per-utterance rates would give. jiwer and
    Whisper's published numbers both use the former; if this ever drifted to the
    latter the bootstrap would be resampling a different statistic than the
    headline.
    """
    counts = [WordCounts(edits=1, ref_words=1), WordCounts(edits=0, ref_words=99)]
    assert corpus_wer(counts) == pytest.approx(0.01)


def test_corpus_wer_of_empty_reference_set_is_nan_not_zero() -> None:
    assert math.isnan(corpus_wer([WordCounts(edits=0, ref_words=0)]))


# --- bootstrap -------------------------------------------------------------


def _counts(pairs: list[tuple[int, int]]) -> list[WordCounts]:
    return [WordCounts(edits=e, ref_words=w) for e, w in pairs]


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    baseline = _counts([(3, 10)] * 50)
    system = _counts([(2, 10)] * 50)
    result = paired_bootstrap(baseline, system, n_resamples=500, seed=1)

    assert result.absolute_reduction.low <= result.absolute_reduction.point
    assert result.absolute_reduction.point <= result.absolute_reduction.high


def test_a_uniform_improvement_never_reverses_under_resampling() -> None:
    """Every utterance improves, so no resample can show a regression.

    This is the test that fails if the pairing is dropped: resampling the two
    systems independently would let a resample draw good baseline utterances
    against bad system ones and report a spurious regression.
    """
    baseline = _counts([(5, 10)] * 40)
    system = _counts([(1, 10)] * 40)
    result = paired_bootstrap(baseline, system, n_resamples=500, seed=2)

    assert result.p_no_improvement == 0.0
    assert result.absolute_reduction.low > 0.0


def test_identical_systems_produce_a_zero_delta_and_an_interval_containing_zero() -> None:
    counts = _counts([(2, 10)] * 40)
    result = paired_bootstrap(counts, list(counts), n_resamples=500, seed=3)

    assert result.absolute_reduction.point == pytest.approx(0.0)
    assert result.absolute_reduction.low <= 0.0 <= result.absolute_reduction.high


def test_a_delta_driven_by_one_outlier_has_an_interval_that_includes_zero() -> None:
    """The case the whole thing exists for.

    Thirty-nine utterances are identical between the two systems and one is
    dramatically better. The point estimate shows an improvement; an honest
    interval has to admit that resampling without that one utterance shows
    nothing. If this ever passed with a strictly positive lower bound, the CI
    would be decorative.
    """
    baseline = _counts([(2, 10)] * 39 + [(40, 40)])
    system = _counts([(2, 10)] * 39 + [(0, 40)])
    result = paired_bootstrap(baseline, system, n_resamples=2000, seed=4)

    assert result.absolute_reduction.point > 0.0
    assert result.absolute_reduction.low <= 0.0


def test_bootstrap_is_deterministic_for_a_given_seed() -> None:
    baseline = _counts([(3, 10), (1, 8), (5, 12), (0, 9)] * 10)
    system = _counts([(2, 10), (1, 8), (3, 12), (1, 9)] * 10)

    first = paired_bootstrap(baseline, system, n_resamples=300, seed=7)
    second = paired_bootstrap(baseline, system, n_resamples=300, seed=7)
    third = paired_bootstrap(baseline, system, n_resamples=300, seed=8)

    assert first.absolute_reduction.low == second.absolute_reduction.low
    assert first.absolute_reduction.low != third.absolute_reduction.low


def test_wider_confidence_gives_a_wider_interval() -> None:
    baseline = _counts([(3, 10), (4, 10), (2, 10), (5, 10)] * 10)
    system = _counts([(2, 10), (3, 10), (2, 10), (4, 10)] * 10)

    narrow = paired_bootstrap(baseline, system, n_resamples=1000, confidence=0.80, seed=5)
    wide = paired_bootstrap(baseline, system, n_resamples=1000, confidence=0.99, seed=5)

    narrow_width = narrow.absolute_reduction.high - narrow.absolute_reduction.low
    wide_width = wide.absolute_reduction.high - wide.absolute_reduction.low
    assert wide_width > narrow_width


def test_mismatched_lengths_are_refused_rather_than_zipped_short() -> None:
    with pytest.raises(ValueError, match="matched utterances"):
        paired_bootstrap(_counts([(1, 10)] * 5), _counts([(1, 10)] * 4), n_resamples=10)


# --- SNR bucketing ---------------------------------------------------------


def test_buckets_cover_the_range_actually_present_not_a_hardcoded_one() -> None:
    """The regression this replaces.

    A run at -5 to 10 dB used to collapse into the single hardcoded 5-10 dB
    bucket. Derived edges must spread it across the requested number of bands.
    """
    snrs = [-5.0, -2.0, 0.0, 3.0, 6.0, 9.0, 10.0]
    buckets = derive_buckets(snrs, n_buckets=3)

    assert len(buckets) == 3
    assert sum(len(b.indices) for b in buckets) == len(snrs)


def test_every_utterance_lands_in_exactly_one_bucket() -> None:
    snrs = [-5.0, -5.0, 0.0, 2.5, 7.5, 10.0, 10.0]
    buckets = derive_buckets(snrs, n_buckets=4)

    assigned = [i for bucket in buckets for i in bucket.indices]
    assert sorted(assigned) == list(range(len(snrs)))


def test_the_maximum_value_is_not_dropped_off_the_top_edge() -> None:
    """Half-open bins would silently lose the loudest utterance."""
    snrs = [0.0, 5.0, 10.0]
    buckets = derive_buckets(snrs, n_buckets=2)
    assigned = {i for bucket in buckets for i in bucket.indices}
    assert 2 in assigned


def test_a_single_fixed_snr_yields_one_bucket_rather_than_dividing_by_zero() -> None:
    buckets = derive_buckets([8.0] * 6, n_buckets=3)
    assert len(buckets) == 1
    assert len(buckets[0].indices) == 6


def test_non_finite_snr_is_excluded_rather_than_binned_at_an_edge() -> None:
    """`degrade` returns inf when noise is disabled; that is not a condition."""
    buckets = derive_buckets([0.0, 5.0, 10.0, float("inf")], n_buckets=2)
    assigned = {i for bucket in buckets for i in bucket.indices}
    assert 3 not in assigned
    assert assigned == {0, 1, 2}


def test_no_finite_snr_yields_no_buckets_rather_than_an_empty_label() -> None:
    assert derive_buckets([float("inf")] * 4, n_buckets=3) == []


def test_empty_bands_are_dropped_rather_than_reported_as_zero_percent() -> None:
    """A gap in the corpus is not a score of 0% WER."""
    snrs = [0.0, 0.5, 1.0, 20.0]
    buckets = derive_buckets(snrs, n_buckets=4)
    assert all(bucket.indices for bucket in buckets)
    assert len(buckets) < 4
