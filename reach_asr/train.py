"""LoRA fine-tune Whisper on the degraded corpus.

Sized for a 4 GB laptop GPU (RTX 3050), which is the binding constraint on
everything below and the reason for each of these choices:

* **LoRA on the attention projections, not a full fine-tune.** whisper-small is
  244M parameters; full fine-tuning needs optimiser state for all of them
  (~3 GB in fp32 Adam moments alone) before a single activation is stored. LoRA
  on q_proj/v_proj trains ~1% of that.
* **Batch size 1 with gradient accumulation.** Whisper always sees a padded
  30 s window, so activation memory per sample is constant and large regardless
  of how short the utterance is -- there is no packing win to be had.
* **Gradient checkpointing.** Trades ~30% step time for a large activation
  saving. On a 4 GB card that is not a tuning knob, it is the difference
  between running and not.
* **fp16 with the GradScaler the Trainer manages.** Not bf16: consumer Ampere
  (SM 8.6) supports bf16 but the throughput advantage is on datacentre parts,
  and fp16 with a scaler is better tested on this path.

    python -m reach_asr.train --model openai/whisper-small --epochs 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class Utterance:
    audio_path: Path
    text: str


class ManifestDataset(Dataset):
    """Reads the JSONL manifest written by build_dataset.py.

    Features are computed lazily per item rather than precomputed for the whole
    corpus: a log-mel spectrogram for a padded 30 s window is 80x3000 floats
    (~960 KB in fp32), so 2000 utterances would be ~2 GB held in RAM to save an
    operation that is not the bottleneck next to the backward pass.
    """

    def __init__(self, manifest: Path, processor: Any, max_items: int | None = None) -> None:
        self.root = manifest.parent
        self.processor = processor
        self.items: list[Utterance] = []
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.items.append(Utterance(self.root / record["audio"], record["text"]))
                if max_items is not None and len(self.items) >= max_items:
                    break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torchaudio

        item = self.items[index]
        wave, sr = torchaudio.load(str(item.audio_path))
        features = self.processor.feature_extractor(
            wave.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
        )
        labels = self.processor.tokenizer(text=item.text).input_ids
        return {
            "input_features": features.input_features[0],
            "labels": labels,
        }


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = torch.stack([item["input_features"] for item in batch])
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in batch], return_tensors="pt"
        )
        # -100 is the ignore index for cross-entropy: padding must not
        # contribute loss, or the model is rewarded for predicting padding.
        labels = label_batch.input_ids.masked_fill(label_batch.attention_mask.ne(1), -100)
        # The tokenizer prepends the decoder start token, and the model prepends
        # it again when it shifts labels right to build decoder_input_ids.
        # Leaving both in trains the model on a duplicated BOS.
        if (labels[:, 0] == self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
            labels = labels[:, 1:]
        return {"input_features": features, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--out", type=Path, default=Path("runs/whisper-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(f"device: {'cuda: ' + torch.cuda.get_device_name(0) if use_cuda else 'cpu'}")

    processor = WhisperProcessor.from_pretrained(args.model, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # Whisper ships decoding constraints that make sense for zero-shot use and
    # actively fight fine-tuning: forced_decoder_ids pins the language/task
    # prefix and suppress_tokens blocks a token set chosen for the pretrained
    # distribution. Both have to go, or the loss is computed against a sequence
    # the model is not free to produce.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = ManifestDataset(args.data / "train" / "manifest.jsonl", processor, args.max_train)
    print(f"train utterances: {len(train_ds)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        fp16=use_cuda,
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,  # our dataset yields tensors the Trainer cannot introspect
        label_names=["labels"],
        dataloader_num_workers=0,  # Windows: workers respawn the interpreter per epoch
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=Collator(processor),
    )

    # Required with gradient checkpointing on a PEFT model: without it the
    # checkpointed segments have no input requiring grad, so autograd records
    # nothing and every LoRA gradient silently arrives as None.
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "adapter"))
    processor.save_pretrained(str(args.out / "adapter"))
    print(f"adapter saved -> {args.out / 'adapter'}")


if __name__ == "__main__":
    main()
