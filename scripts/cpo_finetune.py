"""CPO (Contrastive Preference Optimization) fine-tune of X-ALMA, ALMA-R style.

Starts from our SFT LoRA (xalma-ft) and continues training it on preference
triples from build_preference_data.py. CPO (Xu et al., ICML 2024) needs NO
separate reference model -- it regularizes with the chosen-NLL term -- so the
4-bit base + one trainable LoRA fits a 24 GB card comfortably.

Inference afterwards is identical to the SFT model, just point --group/--adapter
at the CPO output:
  python scripts/xalma_translate.py --backbone haoranxu/X-ALMA-13B-Pretrain \\
      --group checkpoints/xalma-cpo/best --direction ar-en --input ... --output ...

Example:
  python scripts/cpo_finetune.py \\
      --backbone haoranxu/X-ALMA-13B-Pretrain \\
      --sft-adapter checkpoints/xalma-ft/best \\
      --data data/pref.ar-en.jsonl data/pref.en-ar.jsonl \\
      --output-dir checkpoints/xalma-cpo
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import CPOConfig, CPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="haoranxu/X-ALMA-13B-Pretrain")
    ap.add_argument("--sft-adapter", default="checkpoints/xalma-ft/best",
                    help="The SFT LoRA to continue from (our xalma-ft).")
    ap.add_argument("--data", required=True, nargs="+", type=Path,
                    help="One or more preference JSONL files (prompt/chosen/rejected).")
    ap.add_argument("--output-dir", default="checkpoints/xalma-cpo", type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--cpo-alpha", type=float, default=1.0, help="Weight of the chosen-NLL term.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-prompt-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU detected; CPO on X-ALMA needs a CUDA GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print(f"loading {args.backbone} in 4-bit + SFT adapter {args.sft_adapter} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Continue training the existing SFT LoRA (is_trainable=True) -> no fresh adapter.
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.config.use_cache = False
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=[str(p) for p in args.data], split="train")
    print(f"preference triples: {len(ds)}")

    cfg = CPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        cpo_alpha=args.cpo_alpha,
        loss_type="sigmoid",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )

    # trl 0.11/0.12 CPOTrainer uses tokenizer=; trl >=1.0 renamed to processing_class=.
    try:
        trainer = CPOTrainer(
            model=model,
            args=cfg,
            train_dataset=ds,
            tokenizer=tokenizer,
        )
    except TypeError:
        trainer = CPOTrainer(
            model=model,
            args=cfg,
            train_dataset=ds,
            processing_class=tokenizer,
        )
    trainer.train()

    best = args.output_dir / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    print(f"saved CPO adapter -> {best}")


if __name__ == "__main__":
    main()
