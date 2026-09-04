# reach-asr

Speech recognition for telephony-band noisy audio: a LoRA fine-tune of Whisper,
measured as WER against a zero-shot baseline on the same utterances, and served
over HTTP to the [R.E.A.C.H.](https://github.com/CyberRik/reach-app) emergency
response platform.

```
browser MediaRecorder ──▶ reach-app /api/transcribe ──▶ reach-asr /transcribe
   (WebM/Opus or MP4/AAC)      (Next.js, Vercel)          (FastAPI + GPU)
```

| WER | zero-shot | fine-tuned | change |
|---|---|---|---|
| **clean audio** | **4.37%** — the ceiling | **5.24%** | **+0.87 pp** [+0.35, +1.40] |
| **degraded audio** | **23.76%** — the baseline | **21.20%** — the result | **-2.56 pp** [-3.85, -1.31] |

`whisper-base`, 2000 training utterances, 300 eval utterances, 95% paired
bootstrap intervals over 10,000 resamples. Read it honestly: the fine-tune
recovered **2.56 of the 19.39 points** the channel cost, about 13% of the gap,
**and gave up 0.87 points on clean speech to do it**. Both effects are resolved
— neither interval contains zero.

**The trade is the result, not a footnote.** In absolute terms it is roughly 3:1
in favour of the target condition. In relative terms it inverts: -10.8% on
degraded audio against **+19.9% on clean**, because the clean baseline is small
enough that a small absolute regression is a large relative one. Both framings
are true and quoting only the first is how this gets oversold. For a system
whose entire deployment surface is telephone audio the trade is worth making
— but that is a deployment argument, not a measurement one, and it is stated
here as such.

---

## The honest framing

**There is no public corpus of real emergency-call audio.** 911 recordings are
legally restricted almost everywhere, for good privacy reasons. So this does not
train on emergency calls, and no claim here should be read that way.

What it does instead is *construct* the acoustic condition from corpora that are
freely available, and state exactly what was constructed:

| Stage | What it models | Why it matters for ASR |
|---|---|---|
| 300–3400 Hz bandpass, 6th-order | The telephone passband | Removes the fricative band; /s/ and /f/ live above 4 kHz, which is why "six"/"fix" is the classic phone confusion |
| 8 kHz + G.711 μ-law | The PSTN codec | 8-bit logarithmic quantisation, coarser in loud passages |
| ESC-50 noise at a controlled SNR | Sirens, traffic, crying, alarms | The things actually audible behind an emergency call |
| Packet loss, 20 ms frames | A cellular/VoIP link under poor signal | Signal *absent*, not corrupted — Whisper hallucinates fluent text across a gap rather than degrading gracefully |

Source speech is LibriSpeech (clean); noise is ESC-50, filtered to the 15
categories with face validity for an emergency scene. SNR and loss rate are
flags — the defaults are 5–20 dB and 2%, and the results above used a
deliberately harsher **−5 to 10 dB at 5% loss** (see
[Choosing the channel](#choosing-the-channel-headroom-first)).

So the defensible claim is **"fine-tuned Whisper for telephony-band noisy
speech"**, not "trained on emergency calls". The second would be a lie an
interviewer could catch by asking one question about the dataset.

## Results

Four numbers are reported — the full 2x2 of audio condition against model
— because any subset of them misleads:

- **clean / zero-shot** — the ceiling. Without it, a WER figure could be a hard
  corpus or a hard channel and there is no way to tell which.
- **degraded / zero-shot** — the baseline. The gap to the ceiling is the cost of
  the channel, and it is the only thing fine-tuning can recover.
- **degraded / fine-tuned** — the result.
- **clean / fine-tuned** — the specialisation check. What the result cost.

Quoting the third against the *first* would attribute the whole channel cost to
the fine-tune. Quoting the first three without the fourth leaves "it learned to
handle phone audio" and "it learned to *only* handle phone audio"
indistinguishable. Both are standard ways this experiment gets oversold.

Configuration behind the table above: LoRA r=32, alpha=64 on `q_proj`/`v_proj`,
2 epochs, lr 1e-4, batch 16 on a Kaggle T4. Channel: 300–3400 Hz passband, G.711
companding, ESC-50 noise at −5 to 10 dB SNR, 5% packet loss. Raw output is in
`results/wer.json`, and every hypothesis is dumped next to its reference in
`results/predictions.jsonl` — a WER number tells you a run failed, not *how*.

WER is scored through Whisper's own `EnglishTextNormalizer` — the same one
OpenAI reports Whisper's published WER with — so these numbers are comparable to
the model card rather than to a bespoke scoring function. Without it,
LibriSpeech's uppercase unpunctuated references against Whisper's cased
punctuated output score ~100% WER for reasons unrelated to recognition.

### The fourth cell, and what it cost

The eval is a 2x2 — audio condition against model — and only three cells were
originally measured. The missing one, clean audio through the fine-tuned model,
is the only one that can tell a model that *gained* robustness from one that
merely *narrowed*.

It has now been measured, and the model narrowed: **4.37% to 5.24%, +0.87 pp,
95% CI [+0.35, +1.40]**. The interval excludes zero, so this is a real
regression rather than sampling noise — but it is a mild one. A LoRA that had
genuinely collapsed onto the narrow channel would put clean WER in the teens,
not at 5.24%.

`evaluate_wer` reports the clean pair as a **change**, not a reduction — a
regression there is the expected failure, and naming it a "reduction" would
invite exactly the misreading the check exists to prevent. A subset of
`--passes` merges into an existing `wer.json` *and* into `predictions.jsonl`
(keyed by utterance id), so the missing cell of a finished run can be filled in
without re-running the passes that already cost GPU time:

```bash
python -m reach_asr.evaluate_wer --model openai/whisper-base     --adapter runs/whisper-lora/adapter --passes clean_zeroshot,clean_finetuned
```

Both clean passes go in **one** invocation. The paired bootstrap on the clean
pair needs two sets of hypotheses in the same run; asking for `clean_finetuned`
alone produces the number without an interval, which is most of the point.

### One curve, not two effects

The SNR breakdown is monotone, and it stays monotone in *relative* terms —
which is the stronger statement, since it is not merely an artifact of harder
buckets having more room to improve:

| band | n | zero-shot | fine-tuned | delta | relative |
|---|---|---|---|---|---|
| —5 to 0 dB | 99 | 38.19% | 32.53% | 5.67 pp | **14.8%** |
| 0 to 5 dB | 104 | 20.93% | 19.11% | 1.82 pp | **8.7%** |
| 5 to 10 dB | 97 | 13.16% | 12.75% | 0.41 pp | **3.1%** |
| clean | 300 | 4.37% | 5.24% | —0.87 pp | **—19.9%** |

Read the clean regression as the fourth row and the result is one curve rather
than two unrelated effects: the adapter reallocates capacity along the SNR axis,
gaining most where the noise is worst and giving it back where there is no noise
at all. It also lands where the premise says it should — the low-SNR bucket is
the one that matters for emergency audio, and it is the one that improved most.

The per-bucket deltas do **not** have intervals. At n~100 the 5.67 pp is almost
certainly real and the 0.41 pp almost certainly is not distinguishable from
zero, but "almost certainly" is what the bootstrap exists to replace, and it has
not been run per bucket.

The by-category breakdown that `analyze.py` also prints is **underpowered and
should not be read as a finding**: five of eleven categories have n <= 9, none
has an interval, and every negative delta sits in that group. Two artifacts in
it are worth knowing about, because both are real flaws rather than noise:

- `door_wood_knock` is the largest category (n=69) and by far the easiest
  (8.88% zero-shot). That is the SNR-labelling flaw showing up in the results:
  `mix_noise` scales noise to a target **mean** power over the whole clip, so a
  clip that is three transients in five seconds of silence is nearly absent
  during the speech. Those 69 utterances carry an `snr_db` label that is not the
  condition they were scored under, and they pull the corpus mean down. The fix
  is an active-speech level measurement (ITU-T P.56) on both signals, and
  measuring the ratio *after* the channel rather than before it — band-limiting
  is linear and moves speech and noise by different amounts.
- The category counts are uneven (69 knocks, 8 crying babies) because
  `load_noise_bank` walks ESC-50 in dataset order until it has 180 clips and
  `build_dataset` takes the last 20% as the eval bank. The eval noise mix is
  therefore an artifact of iteration order, not a balanced design.

### The interval, and the axis

Two more things were wrong with how the table above was originally reported,
both fixed in `reach_asr/stats.py`.

**A delta without an interval is not a result.** 23.76% → 21.20% is 2.56 points
from 300 utterances, one seed, one training run. Quoted bare it invites exactly
the question it cannot answer. `evaluate_wer` now reports a **paired percentile
bootstrap** over utterances alongside the point estimate — paired because both
systems are scored on byte-identical audio, which `telephony.degrade` guarantees
by seed, so resampling them independently would widen the interval with variance
the experiment design already removed.

**Buckets that don't match the run report the mean three times.** The SNR
breakdown was hardcoded at 5–10/10–15/15–20 dB. The reported run used −5 to
10 dB, so all 300 utterances landed in the bottom bucket and the per-condition
diagnostic silently returned the corpus mean under a label claiming otherwise —
worse than reporting nothing, because the output looks fine. Edges now come from
the run's own SNR values (`--snr-buckets`, default 3).

### Re-analysing a finished run without a GPU

`predictions.jsonl` already carries every reference and hypothesis, normalised,
with per-utterance SNR and noise category. So the interval and both breakdowns
are pure post-processing on data the run already produced — no model load, no
generation pass, seconds on a laptop:

```bash
python -m reach_asr.analyze --predictions results/predictions.jsonl
```

It prints the corpus WERs, the bootstrap interval on the absolute and relative
reduction, the fraction of resamples in which the fine-tune did not win, and
breakdowns by SNR band **and by noise category** — the second axis the manifest
was already recording and nothing was reading. A fine-tune that helps on rain and
not on sirens is a different result from one that helps uniformly, and neither
the mean nor the SNR split would show it.

If the interval includes zero, it says so rather than leaving the reader to
notice.

### The run before this one failed, and why

The first attempt used `whisper-small` on the milder default channel and made
things **worse** — 5.52% → 11.89%, WER more than doubled in every SNR bucket,
while training loss fell steadily from 1.82 to 0.71. Two independent causes:

1. **Targets were LibriSpeech raw.** Those references are ALL CAPS with no
   punctuation; Whisper emits cased, punctuated text. The model spent its
   capacity learning a formatting change — an expensive one, since BPE splits
   uppercase into far more tokens than the same words in normal case. The eval
   normalizer then strips case and punctuation, so it got no credit for what it
   learned and paid in full for the damage. It was learning to shout, not to
   hear. `ManifestDataset` now normalises targets with the same
   `EnglishTextNormalizer` that scores them.
2. **There was almost nothing to win.** `whisper-small` scored 3.06% clean and
   5.52% degraded on that channel — 2.5 points of headroom in total. No
   fine-tune recovers a gap that isn't there.

The falling loss is the part worth sitting with: it was a *correct* measurement
of the model getting better at the objective it was given. The objective was
wrong. A loss curve cannot detect that; only the three-way WER split did.

### Choosing the channel: headroom first

The fix for cause (2) is to measure headroom *before* spending an hour of GPU:

```bash
python -m reach_asr.build_dataset --out data_probe --train 50 --eval 200 \
    --snr-min -5 --snr-max 10 --packet-loss 0.05
python -m reach_asr.evaluate_wer --data data_probe --model openai/whisper-base --limit 200
```

Seven minutes, no training. If clean-vs-degraded is **15+ points**, a fine-tune
has something to recover. `whisper-base` at −5 to 10 dB showed 19.2, which is
what made the full run worth doing. Under ~10 points, harden the channel or drop
to a smaller checkpoint rather than training into a gap that isn't there.

## Running it

**Train on Kaggle, not locally.** `kaggle/reach_asr_kaggle.ipynb` is the actual
notebook that produced the numbers above, outputs included — upload the repo as
a Kaggle Dataset, open that notebook, and run it. `kaggle/run_kaggle.py`'s
docstring carries the full setup and a one-command variant.

A 4 GB laptop GPU forces batch size 1 plus gradient checkpointing (~2 h); a
Kaggle T4 runs batch 8–16 with no checkpointing (~30 min). Evaluation is what
makes the small card painful — three passes of autoregressive generation that
cannot be usefully batched at 4 GB.

Two things that will otherwise cost you a session:

- **Use Save Version → Save & Run All (Commit), not an interactive notebook.**
  An interactive session loses everything in `/kaggle/working` on a disconnect,
  an idle timeout, or the power icon. A committed run executes headless and
  persists the output. Learned expensively: a completed 20-minute fine-tune and
  a built corpus, both gone.
- **`pip uninstall -y torchao` first.** Kaggle ships 0.10.0; PEFT's
  `is_torchao_available()` *raises* on an incompatible version rather than
  returning False, so training dies at `get_peft_model()` before step 1. Nothing
  here uses torchao, so removing it is the clean fix.

Locally, CPU is enough for the part worth developing carefully:

```bash
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch torchaudio soundfile jiwer pytest \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ -q
```

## Serving it

```bash
uv pip install -e ".[serve]"
REACH_ASR_ADAPTER=runs/whisper-lora/adapter uvicorn reach_asr.serve:app --port 8081
```

`POST /transcribe` takes a multipart `file` and returns the text plus duration,
latency, and which checkpoint produced it. `GET /health` reports readiness, so
the UI can show the mic's real state instead of discovering an outage after a
responder has already recorded twenty seconds. One file at a time:

```bash
python -m reach_asr.transcribe clip.wav --adapter runs/whisper-lora/adapter
```

| Variable | Default | |
|---|---|---|
| `REACH_ASR_MODEL` | `openai/whisper-base` | base checkpoint |
| `REACH_ASR_ADAPTER` | *(none)* | LoRA adapter; unset serves the base model |
| `REACH_ASR_DEVICE` | `auto` | `cuda` when available |
| `REACH_ASR_CONCURRENCY` | `1` | in-flight transcriptions |
| `REACH_ASR_ORIGINS` | `http://localhost:3000` | CORS allow-list |

Four decisions in there that are not obvious:

- **A separate process, not a Next.js route.** The model is a resident ~290 MB
  graph that holds a GPU for the process lifetime; route handlers scale to zero
  and spin up per request. Splitting them lets REACH deploy on Vercel unchanged
  while this sits on whatever has the GPU.
- **The adapter is merged, not wrapped.** `merge_and_unload()` folds the
  low-rank deltas into the base weights. A live `PeftModel` computes `Wx + BAx`
  as two extra matmuls per attention projection on every forward pass — pure
  overhead once the adapter is frozen and there is nothing left to switch
  between.
- **Requests are serialised, not interleaved.** One model, one GPU. Three
  queued requests at 800 ms each beat three that thrash a card into an OOM at
  2.4 s.
- **Failure is degraded, never lost.** If this service is down, REACH attaches
  the recording untranscribed and the dispatcher plays it. An ASR outage must
  not be why an emergency report disappears.

Browser recordings arrive as WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari),
neither of which libsndfile opens, so `audio_io` falls back to ffmpeg and the
health check reports whether ffmpeg is on PATH. Phone recordings arrive at 44.1
or 48 kHz and often in stereo; feeding those to a 16 kHz feature extractor does
not error — it transposes everything up in pitch by a factor of three and
returns confident nonsense. That is why the decode path is tested on duration
rather than on "did it run".

## Why the tests matter more than usual

An augmentation pipeline is uniquely easy to get silently wrong. A mis-scaled
noise mix or a filter that does nothing still produces audio that sounds
plausible and trains without error — the only symptom is a WER result that means
something other than what you claim. The serving path has the same shape: a
wrong sample rate returns a fluent transcript of the wrong thing.

So the tests assert measurable signal properties, not that the code runs.
**49 tests, all passing** (`pytest tests/ -q`):

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
- Decoding a 48 kHz clip preserves its *duration*, not merely its sample count —
  the one assertion that catches a skipped resample.
- Stereo is **averaged**, not truncated to channel 0, so a caller recorded on
  one side survives.
- 422 (bad audio) and 503 (service broken) stay distinct across the HTTP
  boundary. REACH retries one and asks the user to re-record on the other, so
  swapping them makes an outage look like the responder's mistake.
- A **uniform improvement never reverses under resampling**. This is the test
  that fails if the bootstrap's pairing is ever dropped -- independent resampling
  would draw good baseline utterances against bad fine-tuned ones and report a
  regression that the data does not contain.
- A delta driven by **one outlier utterance** produces an interval that includes
  zero. Without this, the CI would be decorative: it has to be able to say the
  result is not separable from noise, or it is not measuring anything.
- A clean-audio **regression reports a negative reduction**, and `p_no_improvement`
  goes to 1.0. This pins the sign convention on the specialisation check, where
  reporting a regression as a gain is the whole failure being guarded against.
- Every utterance lands in **exactly one** SNR bucket, the maximum is not dropped
  off the top edge, and a run at a single fixed SNR yields one bucket rather than
  dividing by zero.

## Layout

```
reach_asr/telephony.py       the degradation chain (pure signal processing, CPU)
reach_asr/build_dataset.py   streams LibriSpeech + ESC-50, writes WAVs + manifest
reach_asr/train.py           LoRA fine-tune, sized for a 4 GB card
reach_asr/evaluate_wer.py    the 2x2, bootstrap CIs, SNR breakdown
reach_asr/stats.py           paired bootstrap and data-derived SNR buckets
reach_asr/analyze.py         re-analyse a finished run from predictions.jsonl (CPU)
reach_asr/audio_io.py        upload decoding: any container -> mono 16 kHz float32
reach_asr/transcribe.py      inference core; merges the adapter, caches the load
reach_asr/serve.py           FastAPI service the REACH app calls
kaggle/run_kaggle.py         build -> train -> eval, tuned for 16 GB
kaggle/reach_asr_kaggle.ipynb the committed run behind the reported WER, with outputs
tests/test_telephony.py      15 signal-property tests
tests/test_audio_io.py        9 decode-path tests
tests/test_serve.py           6 HTTP contract tests (no model loaded)
tests/test_stats.py          19 bootstrap, bucketing and sign-convention tests
```

## License

MIT.
