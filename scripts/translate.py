"""Translate an input file with a fine-tuned NLLB or opus-mt checkpoint.

Examples:
  # NLLB checkpoint, ar -> en
  python scripts/translate.py --model checkpoints/nllb-Ar-En/best \\
      --kind nllb --direction ar-en \\
      --input Ar-En/clean/devtest.ar --output predictions/nllb_ar-en.devtest.en

  # opus-mt checkpoint (already direction-locked at training time)
  python scripts/translate.py --model checkpoints/opus-Ar-En-ar-en/best \\
      --kind opus --direction ar-en \\
      --input Ar-En/clean/devtest.ar --output predictions/opus_ar-en.devtest.en
"""
from __future__ import annotations

import argparse
import os
import sys

# Avoid the conda-MKL + pip-torch libiomp5 collision on Windows when running this
# script outside the PowerShell wrappers.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_LANG = {"ar": "arb_Arab", "en": "eng_Latn"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kind", choices=["nllb", "opus"], required=True)
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0,
                    help="Length penalty: >1.0 favors longer outputs (useful for morphologically "
                         "rich target languages like Arabic). Default 1.0.")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="Allow inference on CPU. Without this flag, the script exits if no GPU is found.")
    args = ap.parse_args()

    src, tgt = args.direction.split("-")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB  "
              f"torch: {torch.__version__}")
    elif args.allow_cpu:
        device = "cpu"
        print("WARNING: inference on CPU (--allow-cpu set). Beam search will be very slow.")
    else:
        sys.exit(
            "ERROR: no GPU detected by PyTorch. Beam-search inference on CPU is ~100x slower.\n"
            "If this is unexpected, sanity-check:\n"
            "  python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "Pass --allow-cpu to override (not recommended)."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model)).to(device).eval()

    gen_kwargs = {
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "max_length": args.max_length,
    }
    if args.kind == "nllb":
        tokenizer.src_lang = NLLB_LANG[src]
        forced = tokenizer.convert_tokens_to_ids(NLLB_LANG[tgt])
        gen_kwargs["forced_bos_token_id"] = forced

    lines = args.input.read_text(encoding="utf-8").splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), args.batch_size), desc="translate"):
            batch = lines[i : i + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_length).to(device)
            out = model.generate(**enc, **gen_kwargs)
            out_lines.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out_lines)} lines -> {args.output}")


if __name__ == "__main__":
    main()
