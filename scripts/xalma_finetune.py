"""QLoRA fine-tune X-ALMA-13B on our in-domain Ar<->En parallel data.

Approach:
  - Load X-ALMA-13B-Pretrain in 4-bit nf4 (no Group-8: we use the pretrain backbone
    so a fresh small LoRA can adapt it freely to our domain without PEFT
    multi-adapter complexity).
  - Add a fresh LoRA (r=16, all-linear) -> trainable adapter, tiny memory footprint.
  - Train on both directions simultaneously (each pair becomes 2 instruction examples).
  - Loss masked to target tokens only (prompt is set to -100).
  - paged_adamw_8bit + gradient checkpointing keeps memory well under 24 GB.

Inference after training:
  python scripts/xalma_translate.py \\
      --backbone facebook-or-pretrain \\
      --group checkpoints/xalma-ft/best \\
      --direction ar-en --input ... --output ...
  (the saved LoRA replaces Group-8 as the adapter on the pretrain backbone)
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

LANG_NAME = {"ar": "Arabic", "en": "English"}


def load_pairs(data_dir: Path, split: str, direction: str) -> list[tuple[str, str, str, str]]:
    ar = (data_dir / f"{split}.ar").read_text(encoding="utf-8").splitlines()
    en = (data_dir / f"{split}.en").read_text(encoding="utf-8").splitlines()
    assert len(ar) == len(en)
    rows: list[tuple[str, str, str, str]] = []
    for a, e in zip(ar, en):
        if direction in ("ar-en", "both"):
            rows.append((a, e, "ar", "en"))
        if direction in ("en-ar", "both"):
            rows.append((e, a, "en", "ar"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="haoranxu/X-ALMA-13B-Pretrain")
    ap.add_argument("--data-dir", default="Ar-En/clean", type=Path)
    ap.add_argument("--output-dir", default="checkpoints/xalma-ft", type=Path)
    ap.add_argument("--direction", choices=["ar-en", "en-ar", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optim", default="paged_adamw_8bit",
                    help="Optimizer; switch to adamw_bnb_8bit or adamw_torch if "
                         "paged_adamw_8bit crashes from pinned-memory contention.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU detected; X-ALMA QLoRA needs a CUDA GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB  "
          f"torch: {torch.__version__}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print(f"loading {args.backbone} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, quantization_config=bnb,
        device_map={"": 0}, dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(example: dict) -> dict:
        src, tgt = example["src"], example["tgt"]
        sl, tl = LANG_NAME[example["src_lang"]], LANG_NAME[example["tgt_lang"]]
        prompt = f"Translate this from {sl} to {tl}:\n{sl}: {src}\n{tl}:"
        chat = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # Full sequence = prompt + " " + target + eos. Loss masked to target only.
        full = prompt_str + " " + tgt + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][: args.max_length]
        labels = list(full_ids)
        plen = min(len(prompt_ids), len(labels))
        for i in range(plen):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    rows = load_pairs(args.data_dir, "train", args.direction)
    print(f"train examples: {len(rows)}")
    raw = Dataset.from_list([{"src": s, "tgt": t, "src_lang": sl, "tgt_lang": tl}
                             for s, t, sl, tl in rows])
    train_ds = raw.map(encode, remove_columns=raw.column_names, desc="tokenize")

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=True,
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    best_dir = args.output_dir / "best"
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"saved LoRA adapter -> {best_dir}")
    print(f"inference: xalma_translate.py --backbone {args.backbone} --group {best_dir}")


if __name__ == "__main__":
    main()
