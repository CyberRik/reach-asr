"""One-shot Kaggle runner: build the corpus, fine-tune, measure WER.

Defaults are tuned for a Kaggle T4/P100 (16 GB), not the 4 GB laptop the local
code is sized for. That difference is the whole reason to run here:

    laptop RTX 3050 (4 GB)      batch 1  + gradient checkpointing, ~2 h
    Kaggle T4/P100 (16 GB)      batch 8, no checkpointing,        ~30 min

Evaluation, not training, is what makes the small card painful: scoring is three
passes of autoregressive generation over the eval split (clean, degraded
baseline, degraded fine-tuned) and generation cannot be batched usefully at 4 GB.

SETUP
-----
1. Zip the `reach_asr/` package and upload it as a Kaggle Dataset named
   `reach-asr` (Datasets -> New Dataset -> drag the folder in). No GitHub
   involved, which avoids the CyberRik/Rik0411 credential mismatch entirely.
2. New Notebook -> Settings: Accelerator **GPU T4 x2** (or P100), and
   Internet **On** -- the corpora stream from Hugging Face at runtime.
3. Add the `reach-asr` dataset to the notebook.
4. In one cell:

       !pip install -q jiwer peft
       !cp -r /kaggle/input/reach-asr/reach_asr /kaggle/working/
       %cd /kaggle/working
       !python kaggle/run_kaggle.py --train 2000 --eval 300

5. Download `results/wer.json` and `runs/whisper-lora/adapter/` from the
   notebook output when it finishes.

Kaggle sessions are capped at 12 h and the GPU quota is 30 h/week, so a run this
size costs well under 3% of a week's allowance -- there is room to iterate on
the augmentation settings rather than treating one run as the answer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str], produces: list[Path] | None = None) -> None:
    """Run a step, judging it by what it produced rather than only by its code.

    A step can do all of its work and still exit non-zero: HF's streaming
    downloader and torchaudio leave non-Python threads that can abort the
    interpreter during finalization (SIGABRT, -6) *after* every output is
    safely on disk. Failing the pipeline there throws away a completed corpus
    build and makes the user redo twenty minutes of streaming for nothing.

    So when a step declares its expected artifacts and all of them exist, a
    non-zero exit is reported as a teardown warning rather than a failure. A
    step with no declared artifacts still fails on a non-zero code -- the
    leniency is scoped to cases where completion is independently checkable,
    not applied blindly, because "ignore the exit code" as a general rule is
    how a genuinely broken step gets to look successful.
    """
    print(f"\n$ {' '.join(command)}\n", flush=True)
    started = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - started

    if result.returncode != 0:
        missing = [path for path in (produces or []) if not path.exists()]
        if produces and not missing:
            print(
                f"\n[warning] exited {result.returncode} but every expected output is "
                "present -- treating as a teardown crash, not a failure",
                flush=True,
            )
        else:
            detail = f"; missing {[str(p) for p in missing]}" if missing else ""
            raise SystemExit(f"step failed ({result.returncode}){detail}: {' '.join(command)}")

    print(f"\n[done in {elapsed:.0f}s]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--eval", type=int, default=300)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--epochs", type=float, default=2.0)
    # 8 fits whisper-small on a 16 GB card with fp16 and no checkpointing. Raise
    # to 16 for whisper-base; drop to 2 and add --grad-accum if you move to
    # whisper-medium, which needs LoRA to fit at all.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    data = Path("data")

    if not args.skip_build:
        run(
            [
                python, "-m", "reach_asr.build_dataset",
                "--train", str(args.train),
                "--eval", str(args.eval),
                "--snr-min", str(args.snr_min),
                "--snr-max", str(args.snr_max),
            ],
            produces=[data / "train" / "manifest.jsonl", data / "eval" / "manifest.jsonl"],
        )
    else:
        print("skipping corpus build (--skip-build)")

    if not (data / "train" / "manifest.jsonl").exists():
        raise SystemExit("no training manifest; run without --skip-build first")

    run([
        python, "-m", "reach_asr.train",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
    ])

    run([
        python, "-m", "reach_asr.evaluate_wer",
        "--model", args.model,
        "--adapter", "runs/whisper-lora/adapter",
    ])

    results = Path("results/wer.json")
    if results.exists():
        print("\n" + "=" * 60)
        print(results.read_text(encoding="utf-8"))
        print("=" * 60)
        print("Download results/wer.json and runs/whisper-lora/adapter/ before")
        print("the session expires -- /kaggle/working is not persisted otherwise.")


if __name__ == "__main__":
    main()
