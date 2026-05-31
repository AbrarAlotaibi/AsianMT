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
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                    help="Inference dtype. fp16 halves VRAM (essential for NLLB-3.3B). Default fp16.")
    ap.add_argument("--model-parallel", action="store_true",
                    help="Shard the model across all visible GPUs (device_map='auto' + capped "
                         "max_memory). Use if a single GPU OOMs at generation time.")
    ap.add_argument("--num-samples", type=int, default=1,
                    help="If >1, draw N sampled hypotheses per source (for MBR / preference data). "
                         "Writes <output>.sampleK files (K=0..N-1) instead of a single beam output.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
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
    dtype = torch.float16 if args.dtype == "fp16" and device == "cuda" else torch.float32
    if args.model_parallel and device == "cuda":
        n_gpu = torch.cuda.device_count()
        max_memory = {i: "8GiB" for i in range(n_gpu)}
        max_memory["cpu"] = "200GiB"
        print(f"sharding across {n_gpu} GPU(s) in {dtype}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(args.model), dtype=dtype, device_map="auto", max_memory=max_memory,
        ).eval()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(args.model), dtype=dtype,
        ).to(device).eval()
    model.config.use_cache = True  # KV cache speeds up generation

    sampling = args.num_samples > 1
    gen_kwargs = {
        "max_length": args.max_length,
    }
    if sampling:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_return_sequences": args.num_samples,
        })
    else:
        gen_kwargs.update({
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "early_stopping": True,  # stop a beam group once num_beams finished hyps are found
        })
    if args.kind == "nllb":
        tokenizer.src_lang = NLLB_LANG[src]
        forced = tokenizer.convert_tokens_to_ids(NLLB_LANG[tgt])
        gen_kwargs["forced_bos_token_id"] = forced

    lines = args.input.read_text(encoding="utf-8").splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    k = args.num_samples if sampling else 1
    cols: list[list[str]] = [[] for _ in range(k)]
    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), args.batch_size), desc="translate"):
            batch = lines[i : i + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_length).to(device)
            out = model.generate(**enc, **gen_kwargs)
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            # rows are grouped per source: [s0_k0, s0_k1, ..., s1_k0, ...]
            for j, text in enumerate(dec):
                cols[j % k].append(text)

    if sampling:
        for ki, col in enumerate(cols):
            p = args.output.with_suffix(args.output.suffix + f".sample{ki}")
            p.write_text("\n".join(col) + "\n", encoding="utf-8")
        print(f"wrote {k} sample files -> {args.output}.sample0..{k-1}")
    else:
        args.output.write_text("\n".join(cols[0]) + "\n", encoding="utf-8")
        print(f"wrote {len(cols[0])} lines -> {args.output}")


if __name__ == "__main__":
    main()
