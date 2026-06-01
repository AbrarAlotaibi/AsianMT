"""Translate with ALLAM-7B-Instruct (SDAIA × IBM Arabic LLM).

ALLAM is an Arabic-native 7B instruction model — stronger than X-ALMA on
en→ar due to its Arabic-focused pretraining. Loaded in 4-bit NF4 (~5 GB),
fits comfortably on a single 3090.

Examples:
  python scripts/allam_translate.py --direction en-ar \\
      --input Ar-En/clean/dev.en --output predictions/allam-ft.en-ar.dev.ar
  python scripts/allam_translate.py --direction ar-en \\
      --input Ar-En/clean/dev.ar --output predictions/allam-ft.ar-en.dev.en
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

LANG_NAME = {"ar": "Arabic", "en": "English"}
DEFAULT_MODEL = "ALLaM-AI/ALLaM-7B-Instruct-preview"

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the given text accurately "
    "and naturally. Output only the translation with no explanation."
)


def build_prompt(tokenizer, src_name: str, tgt_name: str, sentence: str) -> str:
    user = f"Translate the following {src_name} text to {tgt_name}:\n{sentence}"
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def clean_output(text: str) -> str:
    """Strip any preamble the model may prepend (e.g. 'Translation: ...')."""
    for prefix in ("Translation:", "ترجمة:", "الترجمة:", "Here is", "Here's"):
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):].strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model ID or local path (base or fine-tuned).")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter path stacked on top of --model.")
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Reduce to 1-2 when --num-samples > 1 (K seqs generated per source).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=1,
                    help="If >1, draw N sampled hypotheses per source. "
                         "Writes <output>.sampleK files.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU detected.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")

    src, tgt = args.direction.split("-")
    src_name, tgt_name = LANG_NAME[src], LANG_NAME[tgt]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print(f"loading {args.model} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
    )
    if args.adapter:
        print(f"loading adapter {args.adapter} ...")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lines = args.input.read_text(encoding="utf-8").splitlines()
    if args.limit > 0:
        lines = lines[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sampling = args.num_samples > 1
    if sampling and args.batch_size > 2:
        print(f"WARNING: --num-samples {args.num_samples} x batch {args.batch_size} "
              f"= {args.num_samples * args.batch_size} sequences; consider --batch-size 2 to avoid OOM.")
    cols: list[list[str]] = [[] for _ in range(args.num_samples if sampling else 1)]

    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), args.batch_size), desc="allam"):
            batch = lines[i: i + args.batch_size]
            prompts = [build_prompt(tokenizer, src_name, tgt_name, s) for s in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            if sampling:
                gen = model.generate(
                    **enc, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, num_return_sequences=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                gen = model.generate(
                    **enc, do_sample=False, num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            dec = tokenizer.batch_decode(new, skip_special_tokens=True)
            k = args.num_samples if sampling else 1
            for j, text in enumerate(dec):
                cols[j % k].append(clean_output(text))

    if sampling:
        for k, col in enumerate(cols):
            p = args.output.with_suffix(args.output.suffix + f".sample{k}")
            p.write_text("\n".join(col) + "\n", encoding="utf-8")
        print(f"wrote {args.num_samples} sample files -> {args.output}.sample0..{args.num_samples-1}")
    else:
        args.output.write_text("\n".join(cols[0]) + "\n", encoding="utf-8")
        print(f"wrote {len(cols[0])} lines -> {args.output}")


if __name__ == "__main__":
    main()
