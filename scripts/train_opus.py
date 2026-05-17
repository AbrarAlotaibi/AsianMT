"""Fine-tune a Helsinki-NLP opus-mt bilingual model for one direction.

Run twice for both directions:
  python scripts/train_opus.py --direction ar-en
  python scripts/train_opus.py --direction en-ar

Models used:
  ar-en -> Helsinki-NLP/opus-mt-ar-en
  en-ar -> Helsinki-NLP/opus-mt-en-ar
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DEFAULT_MODEL = {
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
}


def load_parallel(data_dir: Path, split: str, src: str, tgt: str) -> list[dict]:
    s_lines = (data_dir / f"{split}.{src}").read_text(encoding="utf-8").splitlines()
    t_lines = (data_dir / f"{split}.{tgt}").read_text(encoding="utf-8").splitlines()
    assert len(s_lines) == len(t_lines), f"{split}: {len(s_lines)} vs {len(t_lines)}"
    return [{"src": s, "tgt": t} for s, t in zip(s_lines, t_lines)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--model", default=None, help="HF model id; default picked by --direction")
    ap.add_argument("--data-dir", default="data/clean", type=Path)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, tgt = args.direction.split("-")
    model_name = args.model or DEFAULT_MODEL[args.direction]
    output_dir = args.output_dir or Path(f"checkpoints/opus-{args.direction}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_rows = load_parallel(args.data_dir, "train", src, tgt)
    dev_rows = load_parallel(args.data_dir, "dev", src, tgt)
    print(f"[{args.direction}] train: {len(train_rows)}  dev: {len(dev_rows)}")

    def tokenize_batch(batch: dict) -> dict:
        enc = tokenizer(
            batch["src"], text_target=batch["tgt"],
            max_length=args.max_length, truncation=True,
        )
        return enc

    train_ds = Dataset.from_list(train_rows).map(
        tokenize_batch, batched=True, remove_columns=["src", "tgt"], desc="tokenize train",
    )
    dev_ds = Dataset.from_list(dev_rows).map(
        tokenize_batch, batched=True, remove_columns=["src", "tgt"], desc="tokenize dev",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        predict_with_generate=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    print(f"saved best model to {output_dir / 'best'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
