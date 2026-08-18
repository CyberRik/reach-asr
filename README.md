# reach-asr — Whisper for emergency-call audio

Fine-tunes Whisper for the acoustic conditions of an emergency phone call, and
measures the result as WER against a zero-shot baseline on the same utterances.

## The honest framing

**There is no public corpus of real emergency-call audio.** 911 recordings are
legally restricted almost everywhere, for good privacy reasons. So this does not
train on emergency calls, and no claim here should be read that way.

What it does instead is *construct* the acoustic condition from a corpus that is
freely available, and state exactly what was constructed:

| Stage | What it models | Why it matters for ASR |
|---|---|---|
| 300–3400 Hz bandpass, 6th-order | The telephone passband | Removes the fricative band; /s/ and /f/ live above 4 kHz, which is why "six"/"fix" is the classic phone confusion |
| 8 kHz + G.711 μ-law | The PSTN codec | 8-bit logarithmic quantisation, coarser in loud passages |
| ESC-50 noise at 5–20 dB SNR | Sirens, traffic, crying, alarms | The things actually audible behind an emergency call |
| 2% packet loss, 20 ms frames | A cellular/VoIP link under poor signal | Signal *absent*, not corrupted — Whisper tends to hallucinate fluent text across a gap rather than degrade gracefully |

Source speech is LibriSpeech (clean); noise is ESC-50, filtered to the 15
categories with face validity for an emergency scene.

So the defensible claim is **"fine-tuned Whisper for telephony-band noisy
speech"**, not "trained on emergency calls". The second would be a lie an
interviewer could catch by asking one question about the dataset.

## Results

Three numbers, because any one alone is misleading:

- **clean / zero-shot** — the ceiling. Without it, a WER figure could be a hard
  corpus or a hard channel and there is no way to tell which.
- **degraded / zero-shot** — the baseline. The gap to the ceiling is the cost of
  the channel, and it is the only thing fine-tuning can recover.
- **degraded / fine-tuned** — the result.

Quoting the third against the *first* would attribute the whole channel cost to
the fine-tune. That is the standard way this experiment gets oversold.

WER is scored through Whisper's own `EnglishTextNormalizer` — the same one
OpenAI reports Whisper's published WER with — so the numbers are comparable to
the model card rather than to a bespoke scoring function. Without it, LibriSpeech's
uppercase unpunctuated references against Whisper's cased punctuated output
score ~100% WER for reasons unrelated to recognition.

Results also break down by SNR bucket. A single mean hides whether the fine-tune
helped uniformly or only rescued the loudest cases — and the low-SNR bucket is
the one that matters, since a caller in a quiet room was never the problem.

## Running it

**Train on Kaggle, not locally.** See `kaggle/run_kaggle.py` for setup. A 4 GB
laptop GPU forces batch size 1 plus gradient checkpointing (~2 h); a Kaggle
T4/P100 runs batch 8 with no checkpointing (~30 min). Evaluation is what makes
the small card painful — three passes of autoregressive generation that cannot
be usefully batched at 4 GB.

Locally, CPU is enough for the part worth developing carefully:

```bash
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch torchaudio soundfile jiwer pytest \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ -q
```

## Why the tests matter more than usual

An augmentation pipeline is uniquely easy to get silently wrong. A mis-scaled
noise mix or a filter that does nothing still produces audio that sounds
plausible and trains without error — the only symptom is a WER result that means
something other than what you claim.

So the tests assert measurable signal properties, not that the code runs:

- `mix_noise` lands within 0.5 dB of the requested SNR at 0/5/10/20 dB. Every
  result is labelled with an SNR, so a scale error here mislabels everything and
  nothing else would fail.
- The passband actually removes the band it claims to. **This caught a real
  bug**: a single `lowpass_biquad` is 2nd-order (12 dB/octave) and left ~10% of
  a 6 kHz tone standing. The filter is now three cascaded sections (~36
  dB/octave). Understating the degradation would have overstated the WER gap it
  produces.
- `degrade` is deterministic given a seed. The baseline and the fine-tuned model
  must be scored on byte-identical audio, or part of the WER difference is just
  a different random mix.
- Train and eval use **disjoint noise clips**. Sharing them would let the model
  memorise the specific siren recordings in eval — the classic way an
  augmentation experiment lies to you.

## Layout

```
reach_asr/telephony.py       the degradation chain (pure signal processing, CPU)
reach_asr/build_dataset.py   streams LibriSpeech + ESC-50, writes WAVs + manifest
reach_asr/train.py           LoRA fine-tune, sized for a 4 GB card
reach_asr/evaluate_wer.py    three-way WER with an SNR breakdown
kaggle/run_kaggle.py         build -> train -> eval, tuned for 16 GB
tests/test_telephony.py      15 signal-property tests
```
