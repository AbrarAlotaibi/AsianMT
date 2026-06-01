"""QLoRA fine-tune ALLAM-7B-Instruct for Arabic↔English translation.

Follows the same recipe as xalma_finetune.py:
  - 4-bit NF4 base + fresh r=16 LoRA on all linear layers
  - Instruction format: system + user (source) + assistant (target)
  - Loss masked to assistant tokens only
  - paged_adamw_8bit + gradient checkpointing

Usage:
  python scripts/allam_finetune.py \\
      --src-file Ar-En/clean/train.en --tgt-file Ar-En/clean/train.ar \\
      --direction en-ar --output-dir checkpoints/allam-ft-en-ar
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)

LANG_NAME = {"ar": "Arabic", "en": "English"}
DEFAULT_MODEL = "ALLaM-AI/ALLaM-7B-Instruct-preview"
SYSTEM_PROMPT = (
    "You are a professional translator. Translate the given text accurately "
    "and naturally. Output only the translation with no explanation."
)
IGNORE_INDEX = -100


def build_prompt(tokenizer, src_name: str, tgt_name: str,
                 src_text: str, tgt_text: str):
    """Return input_ids with labels masked to assistant turn only."""
    chat_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following {src_name} text to {tgt_name}:\n{src_text}"},
    ]
    prompt_str = tokenizer.apply_chat_template(
        chat_prompt, tokenize=False, add_generation_prompt=True
    )
    full_str = prompt_str + tgt_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--src-file", required=True, type=Path)
    ap.add_argument("--tgt-file", required=True, type=Path)
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="If >0, cap training examples (smoke test).")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--flash-attn", action="store_true", default=True,
                    help="Use Flash Attention 2 if available (faster + less memory on long seqs).")
    args = ap.parse_args()

    src_lang, tgt_lang = args.direction.split("-")
    src_name, tgt_name = LANG_NAME[src_lang], LANG_NAME[tgt_lang]

    src_lines = args.src_file.read_text(encoding="utf-8").splitlines()
    tgt_lines = args.tgt_file.read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(tgt_lines), "src/tgt line count mismatch"
    if args.max_samples > 0:
        src_lines, tgt_lines = src_lines[: args.max_samples], tgt_lines[: args.max_samples]
    print(f"training pairs: {len(src_lines)}  direction: {args.direction}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print(f"loading {args.model} in 4-bit ...")
    fa2_kwargs = {}
    try:
        if args.flash_attn:
            import flash_attn  # noqa: F401
            fa2_kwargs["attn_implementation"] = "flash_attention_2"
            print("Flash Attention 2 enabled.")
    except ImportError:
        print("flash-attn not installed; falling back to standard attention.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0}, **fa2_kwargs,
    )
    torch.cuda.empty_cache()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("tokenizing ...")
    raw = [build_prompt(tokenizer, src_name, tgt_name, s, t)
           for s, t in zip(src_lines, tgt_lines)]

    # Truncate to max_length
    raw = [r for r in raw if len(r["input_ids"]) <= args.max_length]
    print(f"examples after length filter: {len(raw)}")

    dataset = Dataset.from_list(raw)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True,
                                      label_pad_token_id=IGNORE_INDEX)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    print("training ...")
    trainer.train()

    best = args.output_dir / "best"
    model.save_pretrained(best)
    tokenizer.save_pretrained(best)
    print(f"saved adapter -> {best}")


if __name__ == "__main__":
    main()
