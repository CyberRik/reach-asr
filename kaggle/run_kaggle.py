"""One-shot Kaggle runner: build the corpus, fine-tune, measure WER.

Defaults are tuned for a Kaggle T4/P100 (16 GB), not the 4 GB laptop the local
code is sized for. That difference is the whole reason to run here:

    laptop RTX 3050 (4 GB)      batch 1  + gradient checkpointing, ~2 h
    Kaggle T4/P100 (16 GB)      batch 8, no checkpointing,        ~30 min

Evaluation, not training, is what makes the small card painful: scoring is three
passes of autoregressive generation over the eval split (clean, degraded
baseline, degraded fine-tuned) and generation cannot be batched usefully at 4 GB.

RUN THIS AS A BATCH JOB, NOT INTERACTIVELY
------------------------------------------
Use **Save Version -> Save & Run All (Commit)**. Do not babysit it in an
interactive session.

This is not a style preference, it is the difference between finishing and
starting over. The full pipeline is ~50 minutes, and an interactive Kaggle
session loses everything in /kaggle/working the moment it ends -- a kernel hang,
a browser disconnect, an idle timeout, or hitting the power icon (which stops
the session and tears down the container; only an in-place *restart* preserves
the filesystem, and not dependably). A committed run executes headless on
Kaggle's side and persists /kaggle/working as the version's output, downloadable
afterwards. Learned the expensive way: a completed 20-minute fine-tune and a
built corpus were both lost to a stopped session.

SETUP
-----
1. Zip the **whole repo** -- `reach_asr/` AND `kaggle/` -- and upload it as a
   Kaggle Dataset (Datasets -> New Dataset -> drag the folder in). Note the
   slug Kaggle assigns; it strips the hyphen ("reach-asr" -> "reachasr"). No
   GitHub involved, which sidesteps the CyberRik/Rik0411 credential mismatch.
2. New Notebook -> Settings: Accelerator **GPU T4 x2** (or P100), Internet
   **On** -- the corpora stream from Hugging Face at runtime. Internet needs
   phone verification on some accounts; check before starting.
3. Add the dataset to the notebook.
4. Cell 1 -- environment and code:

       !pip install -q jiwer peft
       !pip uninstall -y -q torchao   # Kaggle ships 0.10.0; PEFT demands >0.16
       import pathlib, shutil
       root = next(pathlib.Path('/kaggle/input').rglob('run_kaggle.py')).parent.parent
       for item in root.iterdir():
           dst = pathlib.Path('/kaggle/working') / item.name
           if dst.is_dir(): shutil.rmtree(dst)
           elif dst.exists(): dst.unlink()
           (shutil.copytree if item.is_dir() else shutil.copy2)(item, dst)
       print(sorted(p.name for p in pathlib.Path('/kaggle/working').iterdir()))

   The rglob is deliberate: it finds the repo root whatever depth the zip
   nested it at, instead of guessing the mount path.

5. Cell 2 -- the pipeline:

       %cd /kaggle/working
       !nvidia-smi --query-gpu=name,memory.total --format=csv
       !python kaggle/run_kaggle.py --train 2000 --eval 300

6. **Save Version -> Save & Run All (Commit).** Close the tab; it runs without
   you. When it finishes, open the version and download `results/wer.json` and
   `runs/whisper-lora/adapter/` from its Output tab.

The uninstall in step 4 is load-bearing: PEFT's LoRA dispatcher calls
is_torchao_available(), which *raises* on an incompatible version rather than
returning False, so training dies at get_peft_model() before a single step. We
use no torchao at all -- this is plain fp16 LoRA -- so removing it is the clean
fix rather than fighting a version bump into the image.

Kaggle allows 9 h GPU per session and 30 h/week, so a ~50 min run costs under 3%
of a week's allowance. There is room to iterate on the augmentation settings
rather than treating one run as the answer.
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
