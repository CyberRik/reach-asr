# Evaluation audit

*An audit of this repo's own WER methodology, September 2026. It reports one
labelling defect found in the eval, what it does and does not invalidate, the
fix, and the limits that remain.*

---

## What this is

`reach-asr` fine-tunes Whisper for telephony-band noisy speech. There is no
public corpus of real emergency-call audio, so the acoustic condition is
*constructed* — a 300–3400 Hz passband, G.711 μ-law companding, ESC-50
environmental noise at a controlled SNR, and packet loss, applied to LibriSpeech
read speech. A LoRA adapter is trained on that channel and the result is served
over HTTP.

The evaluation is a 2×2 of audio condition against model — clean and degraded
audio, each through the stock checkpoint and the fine-tuned one — reported with
paired bootstrap intervals, plus a breakdown of WER against the SNR each
utterance was mixed at. This document audits that evaluation.

## What the audit found

**The per-utterance SNR label is measured before the channel, and the audio is
scored after it.** `degrade()` computes the speech-to-noise ratio at 16 kHz
full-band, then band-limits, resamples and μ-law encodes the mixture — so the
number written into the manifest is not the condition the file was scored under.

Because the filtering is linear and different noises put their energy in
different places, the drift is not a constant offset: at a labelled 10 dB, the
true post-channel SNR ranges from **2 dB to 33 dB** depending only on which noise
clip was drawn (measured with a synthetic speech-like probe — see *Reproducing
the measurements*). The WER-vs-SNR breakdown was therefore not sorting by condition.

---

## The mechanism

A band-limiting filter is linear, so it acts on the speech and the noise
independently. How much it removes from each depends on where that sound's energy
sits relative to the 300–3400 Hz passband. Noise with most of its energy outside
the band is largely deleted and the effective SNR *rises*; noise sitting inside
the band survives while the speech loses its own out-of-band energy, and the
effective SNR *falls*.

Measured on this pipeline — a synthetic harmonic speech-like signal mixed at a
labelled 10 dB, then each component passed separately through `band_limit()`:

| noise | labelled | after band-limiting | drift |
|---|---|---|---|
| wind / rumble (low-frequency) | 10.00 dB | 19.96 dB | **+9.96** |
| hiss (>4 kHz) | 10.00 dB | 33.41 dB | **+23.41** |
| siren (~1 kHz tonal) | 10.00 dB | 2.02 dB | **−7.98** |
| white | 10.00 dB | 7.41 dB | **−2.59** |

Every one of those files carries `"snr_db": 10.0` in the manifest. A siren at
"10 dB" is a harder condition than white noise at "5 dB"; wind at "10 dB" is
effectively clean.

**A second, independent mechanism: global versus active-speech level.**
`mix_noise` scales noise to hit a target ratio of *mean* power over the whole
utterance. LibriSpeech clips contain leading, trailing and inter-phrase silence
that noise fills, so the SNR during the speech itself is higher than the label —
measured at **13.0 dB for a labelled 10 dB** on an utterance that is 50% silence.
The same applies on the noise side: an ESC-50 clip that is a few transients in
five seconds of silence has its mean scaled to target, leaving it nearly absent
while the speech is actually happening.

**This is visible in the results, not just in theory.** In the by-category
breakdown, `door_wood_knock` is the largest category (n=69) and by far the
easiest — 8.88% zero-shot against a 23.76% corpus mean, and a door knock is
exactly the three-transients-in-five-seconds case just described. That is
consistent with the mechanism rather than proof of it: confirming it means
computing the effective SNR for those 69 utterances, which is the fix below. But
it is the shape the defect predicts, in the largest category of the run.

---

## The measured filter response

The passband is documented as 300–3400 Hz. `band_limit()` implements it as two
cascaded highpass biquads and three cascaded lowpass biquads. Cascading *N*
identical sections multiplies the response, so each nominal −3 dB corner becomes
−3*N* dB. Measured, tone by tone:

| frequency | attenuation |
|---|---|
| 100 Hz | −38.31 dB |
| 200 Hz | −15.67 dB |
| 300 Hz | **−6.02 dB** (nominal corner) |
| 500 Hz | −1.06 dB |
| 1000 Hz | −0.12 dB |
| 2000 Hz | −0.96 dB |
| 3000 Hz | −5.43 dB |
| 3400 Hz | **−9.04 dB** (nominal corner) |
| 4000 Hz | −16.65 dB |
| 6000 Hz | −57.94 dB |

The real passband is roughly **500–2500 Hz at −1 dB**, not 300–3400. A telephone
channel is close to flat to 3400 Hz and then falls off sharply; this one is
already −5 dB at 3 kHz, attenuating the F3/F4 region and part of the band that
distinguishes /s/ from /ʃ/ *inside* the range it claims to preserve. The cascade
was added to fix a genuine defect — a single biquad left ~10% of a 6 kHz tone
standing — but it overshot into the passband rather than steepening only the
edge.

A designed filter (elliptic or Chebyshev via `scipy.signal.iirfilter` →
`sosfilt`) with the corner specified once is the correct construction. A stack of
first-choice biquads tuned against one out-of-band test point is not.

---

## What this does and does not invalidate

**Untouched — the corpus-level result and its interval.**

| | zero-shot | fine-tuned | change |
|---|---|---|---|
| clean | 4.37% | 5.24% | +0.87 pp, CI [+0.35, +1.40] |
| degraded | 23.76% | 21.20% | −2.56 pp, CI [−3.85, −1.31] |

Both systems were scored on byte-identical audio — `degrade()` is
seed-deterministic precisely so that they are — and the paired bootstrap resamples
per-utterance edit counts and reference lengths. It never reads `snr_db`. The
defect is a **mislabelling of a covariate**, not an error in the audio, the
transcripts, or the scoring. Relabelling every utterance would not move a single
edit count, so the headline numbers and both intervals stand exactly as reported.

**Rebuilt — the per-bucket breakdown.** `wer_by_snr` and the by-category table
sort on the defective label, so the bucket *edges* and the assignment of
utterances to buckets both change under the fix. The monotone shape is expected
to survive, since the drift is bounded and the ordering is dominated by the
sampled SNR range, but that is a prediction, not a result, until it is rerun.

**Unaffected — the training set.** The model trained on the audio, not on the
labels. Nothing about the fine-tune changes.

---

## The fix

Band-limiting is linear, so the two branches can be measured separately without
re-mixing:

1. Pass the scaled noise through the same channel the speech goes through.
2. Measure the ratio on the **post-channel** signals, over **active speech only**
   (ITU-T P.56 active speech level, rather than a mean over the whole clip).
3. Write both numbers to the manifest — `snr_db_mixed` (what was requested, kept
   for provenance and reproducibility) and `snr_db_effective` (what the file was
   actually scored at).
4. Bucket, report and plot on `snr_db_effective`.

Keeping both is the point. Dropping `snr_db_mixed` would make old runs
unreproducible; dropping `snr_db_effective` is the current bug.

---

## The 2×2 and what the fine-tune cost

The eval originally measured three of the four cells — clean/zero-shot,
degraded/zero-shot, degraded/fine-tuned. The missing one, **clean audio through
the fine-tuned model**, is the only cell that separates a model that *gained*
robustness from one that merely *narrowed*.

Measured: **4.37% → 5.24%, +0.87 pp, 95% CI [+0.35, +1.40].** The interval
excludes zero, so the regression is real — and it is mild. A LoRA that had
collapsed onto the narrow channel would be expected to put clean WER in the
teens; this one did not.

Read honestly: the channel cost 19.39 points (23.76 − 4.37), the fine-tune
recovered 2.56 of them — about 13% of the gap — and gave up 0.87 points on clean
speech to do it. Roughly 3:1 in favour of the target condition in absolute terms;
in relative terms it inverts, −10.8% on degraded against **+19.9% on clean**,
because the clean baseline is small enough that a small absolute regression is a
large relative one. Both framings are true and quoting only the first would be
overselling it.

The specialisation is not a separate finding. The per-band relative gain is
monotone — 14.8% / 8.7% / 3.1% across the SNR bands — and the clean regression is
that curve's endpoint at infinite SNR. Those bands are bucketed on the defective
label, so the exact figures will move under the fix; the drift is bounded and the
ordering is dominated by the sampled SNR range, so the direction should survive,
but that is the claim to check first when it is rerun. The adapter reallocates capacity along the
SNR axis; it gains most where the noise is worst and gives it back where there is
none. The standard remedy, not yet run, is multi-condition training: mix 10–30%
undegraded utterances into the training set.

---

## Known limits

Things this evaluation does not establish, listed because they are the questions
worth asking of it.

- **The bootstrap covers utterance sampling only.** One seed, one training run.
  The interval says how much the *eval set* could have moved the number; it says
  nothing about how much a different training seed would. Bounding that needs
  n independent fine-tunes, which was not run.
- **There is no dev split.** `train.py` has no eval loop. Epoch count and
  hyperparameters were not selected against held-out data — and if they had been,
  the only held-out data is the 300-utterance eval set, which would make it no
  longer held out.
- **Packet loss is memoryless.** Independent Bernoulli drops at 2–5%, where real
  IP and cellular loss is bursty (the standard model is Gilbert–Elliott
  two-state). Isolated 20 ms gaps are close to the best case, and are what a
  codec's packet-loss concealment hides most easily; real damage arrives in runs
  of 3–10 frames. This channel is optimistic on that axis.
- **Dropped packets are zeroed hard.** A rectangular gate produces broadband
  splatter at both edges, injecting energy above 3.4 kHz *after* the stage that
  removed it. A short raised-cosine taper, or actual concealment, is the fix.
- **μ-law is peak-normalised per utterance.** Dividing by the peak before
  encoding makes the codec level-invariant, which defeats the property being
  modelled: companding exists precisely so that quantisation SNR stays roughly
  constant across level, and a quiet caller on a real line is measurably worse
  off. Peak rather than RMS also lets one impulsive sample set the scale for the
  whole utterance.
- **The clean condition is undegraded LibriSpeech.** So the specialisation check
  is the mildest possible version of that test: it bounds what the adapter did to
  clean read speech of the same kind it trained on. It says nothing about
  accented, spontaneous, or far-field speech, which is the form the question
  usually takes.
- **The eval noise mix is an artifact of iteration order.** `load_noise_bank`
  walks ESC-50 in dataset order until it has 180 clips and the eval bank is the
  last 20%, so category counts range from 69 to 8. The by-category breakdown is
  underpowered — five of eleven categories have n ≤ 9 and none has an interval —
  and should not be read as a per-category finding.
- **No per-bucket intervals.** The SNR-band deltas are point estimates. At n≈100
  the 5.67 pp bottom-bucket gain is almost certainly real and the 0.41 pp
  top-bucket delta almost certainly is not resolvable from zero, but neither has
  been bootstrapped.
- **One model size.** Everything here is `whisper-base`. Whether the same trade
  appears at `whisper-small` or `whisper-large` is untested; larger models have
  more capacity to spend and might specialise less.

---

## Reproducing the measurements

The filter response and SNR-drift tables above come from `band_limit()` and
`mix_noise()` in `reach_asr/telephony.py`, driven directly: a pure tone per
frequency for the response, and a synthetic harmonic stack with a 3 Hz amplitude
envelope for the drift, with each component passed separately through the filter
so the two branches can be compared. The speech-like signal is synthetic, not
LibriSpeech — the drift depends on the spectral overlap between speech and noise,
so the exact figures shift with real speech while the signs and rough magnitudes
do not.

The result numbers come from `results/wer.json` and `results/analysis.json` of
the reported run: `whisper-base`, 2000 train / 300 eval utterances, LoRA r=32
alpha=64 on `q_proj`/`v_proj`, 2 epochs, channel at −5 to 10 dB SNR with 5%
packet loss.
