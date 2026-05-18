"""Fine-tune NLLB-200-distilled-600M for both ar<->en directions in one model.

Each parallel pair contributes two training examples (ar->en and en->ar), with the
correct NLLB language tags applied per direction at tokenization time.

Tuned for a single 12 GB GPU: fp16, batch 4, grad-accum 8 (effective 32),
gradient checkpointing.
"""
from __future__ import annotations

import argparse
import os
import sys
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

# torch < 2.6 forces use_safetensors=True (CVE-2025-32434), which triggers a meta-tensor
# bug in some transformers releases. Require torch >= 2.6 to keep the loading path clean.
_TORCH_VER = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _TORCH_VER < (2, 6):
    sys.exit(
        f"torch {torch.__version__} detected; this script requires torch >= 2.6.\n"
        f"Fix:\n"
        f"  conda install -n asianmt -c pytorch -c nvidia pytorch=2.6 pytorch-cuda=12.1"
    )

LANG = {"ar": "arb_Arab", "en": "eng_Latn"}


def load_parallel(data_dir: Path, split: str) -> list[dict]:
    ar = (data_dir / f"{split}.ar").read_text(encoding="utf-8").splitlines()
    en = (data_dir / f"{split}.en").read_text(encoding="utf-8").splitlines()
    assert len(ar) == len(en), f"{split}: {len(ar)} vs {len(en)}"
    rows = []
    for a, e in zip(ar, en):
        rows.append({"src": a, "tgt": e, "src_lang": "ar", "tgt_lang": "en"})
        rows.append({"src": e, "tgt": a, "src_lang": "en", "tgt_lang": "ar"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="Allow training on CPU. Without this flag, the script exits if no GPU is found.")
    ap.add_argument("--data-dir", default="Ar-En/clean", type=Path)
    ap.add_argument("--output-dir", default="checkpoints/nllb-Ar-En", type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="If >0, stop after this many optimizer steps (overrides epochs). For smoke tests.")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB  "
              f"torch: {torch.__version__}")
    elif args.allow_cpu:
        print("WARNING: training on CPU (--allow-cpu set). This will be ~10-30x slower.")
    else:
        sys.exit(
            "ERROR: no GPU detected by PyTorch. NLLB-600M on CPU is impractically slow.\n"
            "Sanity-check:\n"
            "  python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "If False, reinstall CUDA-enabled PyTorch:\n"
            "  conda install -n asianmt -c pytorch -c nvidia pytorch=2.6 pytorch-cuda=12.1\n"
            "Or pass --allow-cpu to override (not recommended)."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_rows = load_parallel(args.data_dir, "train")
    dev_rows = load_parallel(args.data_dir, "dev")
    print(f"train examples: {len(train_rows)}  dev examples: {len(dev_rows)}")

    def tokenize_batch(batch: dict) -> dict:
        # All rows in a batch share the same src/tgt direction iff the dataset is
        # pre-sorted; instead handle each row independently for correctness.
        input_ids, attention_mask, labels = [], [], []
        for src, tgt, sl, tl in zip(batch["src"], batch["tgt"], batch["src_lang"], batch["tgt_lang"]):
            tokenizer.src_lang = LANG[sl]
            tokenizer.tgt_lang = LANG[tl]
            enc = tokenizer(
                src, text_target=tgt,
                max_length=args.max_length, truncation=True,
            )
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            labels.append(enc["labels"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_ds = Dataset.from_list(train_rows).map(
        tokenize_batch, batched=True, batch_size=64, remove_columns=["src", "tgt", "src_lang", "tgt_lang"],
        desc="tokenize train",
    )
    dev_ds = Dataset.from_list(dev_rows).map(
        tokenize_batch, batched=True, batch_size=64, remove_columns=["src", "tgt", "src_lang", "tgt_lang"],
        desc="tokenize dev",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    smoke = args.max_steps and args.max_steps > 0
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=2,
        load_best_model_at_end=not smoke,
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
        processing_class=tokenizer,
    )

    trainer.train()
    if smoke:
        print(f"smoke test completed {args.max_steps} steps (no checkpoint saved)")
    else:
        trainer.save_model(str(args.output_dir / "best"))
        tokenizer.save_pretrained(str(args.output_dir / "best"))
        print(f"saved best model to {args.output_dir / 'best'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
