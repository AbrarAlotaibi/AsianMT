"""Translate an input file with a fine-tuned NLLB or opus-mt checkpoint.

Examples:
  # NLLB checkpoint, ar -> en
  python scripts/translate.py --model checkpoints/nllb-aren/best \\
      --kind nllb --direction ar-en \\
      --input data/clean/devtest.ar --output predictions/nllb_ar-en.devtest.en

  # opus-mt checkpoint (already direction-locked at training time)
  python scripts/translate.py --model checkpoints/opus-ar-en/best \\
      --kind opus --direction ar-en \\
      --input data/clean/devtest.ar --output predictions/opus_ar-en.devtest.en
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    src, tgt = args.direction.split("-")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model)).to(device).eval()

    gen_kwargs = {"num_beams": args.num_beams, "max_length": args.max_length}
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
