"""Translate with X-ALMA-13B (Arabic is in language Group 8), loaded in 4-bit for 16 GB GPUs.

X-ALMA = pretrain backbone + a language-group PEFT module. We load the backbone in
4-bit (nf4) and attach Group-8 (which contains Arabic). An optional extra --adapter
(e.g. our own QLoRA fine-tune) can be stacked on top.

Prompt + chat template + generation follow the official X-ALMA usage.

Examples:
  python scripts/xalma_translate.py --direction ar-en \\
      --input Ar-En/clean/dev.ar --output predictions/xalma.ar-en.dev.en
  python scripts/xalma_translate.py --direction en-ar \\
      --input Ar-En/clean/dev.en --output predictions/xalma.en-ar.dev.ar
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

LANG_NAME = {"ar": "Arabic", "en": "English"}


def build_prompt(tokenizer, src_name: str, tgt_name: str, sentence: str) -> str:
    prompt = f"Translate this from {src_name} to {tgt_name}:\n{src_name}: {sentence}\n{tgt_name}:"
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="haoranxu/X-ALMA-13B-Pretrain")
    ap.add_argument("--group", default="haoranxu/X-ALMA-13B-Group8",
                    help="PEFT group module containing the target language (Group 8 = Arabic).")
    ap.add_argument("--adapter", default=None,
                    help="Optional extra LoRA adapter (e.g. our fine-tune) stacked on top.")
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=1,
                    help="1 = greedy (fast, low memory). beam>1 needs more VRAM; on a 16 GB "
                         "card use --batch-size 1 if you raise this.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--allow-cpu-offload", action="store_true",
                    help="Permit accelerate to offload layers to CPU. Off by default because "
                         "offloading makes generation ~100x slower; better to fail fast.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only translate the first N lines (smoke test).")
    ap.add_argument("--num-samples", type=int, default=1,
                    help="If >1, draw N sampled hypotheses per source (for MBR / preference data). "
                         "Writes <output>.sampleK files (K=0..N-1) instead of a single beam output.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU detected; X-ALMA-13B 4-bit needs a CUDA GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")

    src, tgt = args.direction.split("-")
    src_name, tgt_name = LANG_NAME[src], LANG_NAME[tgt]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print("loading X-ALMA backbone in 4-bit ...")
    # Force all weights onto GPU 0 (device_map={"":0}); "auto" silently offloads to
    # CPU when it doesn't fit, which crawls. If this OOMs, the model genuinely
    # doesn't fit -> lower --batch-size / --num-beams (or pass --allow-cpu-offload).
    device_map = "auto" if args.allow_cpu_offload else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, quantization_config=bnb, device_map=device_map, dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, args.group)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.group, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lines = args.input.read_text(encoding="utf-8").splitlines()
    if args.limit > 0:
        lines = lines[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sampling = args.num_samples > 1
    # One output column per sample (sampling) or a single column (beam/greedy).
    cols: list[list[str]] = [[] for _ in range(args.num_samples if sampling else 1)]
    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), args.batch_size), desc="xalma"):
            batch = lines[i : i + args.batch_size]
            prompts = [build_prompt(tokenizer, src_name, tgt_name, s) for s in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            if sampling:
                gen = model.generate(
                    **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    num_return_sequences=args.num_samples, max_new_tokens=args.max_new_tokens,
                )
            else:
                gen = model.generate(**enc, num_beams=args.num_beams,
                                     max_new_tokens=args.max_new_tokens, do_sample=False)
            new = gen[:, enc["input_ids"].shape[1]:]
            dec = tokenizer.batch_decode(new, skip_special_tokens=True)
            # generate() returns rows grouped as [s0_k0, s0_k1, ..., s1_k0, ...].
            k = args.num_samples if sampling else 1
            for j, text in enumerate(dec):
                cols[j % k].append(text.strip())

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
