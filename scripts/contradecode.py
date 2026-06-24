"""Source/language-contrastive decoding for NLLB (Sennrich et al., EACL 2024).

Standard decoding picks argmax log P(y | x). Contrastive decoding instead scores

    (1 + lambda) * log P(y | x)  -  lambda * log P(y | x_contrast)

which down-weights tokens that are probable *regardless* of the true input, i.e.
off-target-language tokens and hallucinations. Two contrast types (combinable):

  * language-contrastive: same source, but the contrast stream is conditioned on
    the WRONG target-language tag (e.g. eng_Latn while really translating to
    arb_Arab). Directly penalizes English/wrong-script leakage into Arabic
    output -- our en->ar weak spot.
  * source-contrastive: contrast stream sees a shuffled source, penalizing
    tokens that are fluent but input-agnostic (hallucinations).

Greedy implementation with a shared decoder prefix and KV cache. Built for
seq2seq NLLB checkpoints (our fine-tunes).

Example:
  python scripts/contradecode.py --model checkpoints/nllb-Ar-En/best \\
      --direction en-ar --mode language --lam 0.5 \\
      --input Ar-En/clean/dev.en --output predictions/contra.en-ar.dev.ar
"""
from __future__ import annotations

import argparse
import os
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_LANG = {"ar": "arb_Arab", "en": "eng_Latn"}


def shuffle_words(s: str) -> str:
    w = s.split()
    random.shuffle(w)
    return " ".join(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=["source", "language", "both"], default="language")
    ap.add_argument("--lam", type=float, default=0.5, help="Contrastive strength lambda.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU detected; contrastive decoding needs a CUDA GPU.")
    random.seed(args.seed)
    device = "cuda"
    src, tgt = args.direction.split("-")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model)).to(device).eval()
    model.config.use_cache = True

    tgt_id = tokenizer.convert_tokens_to_ids(NLLB_LANG[tgt])
    wrong_id = tokenizer.convert_tokens_to_ids(NLLB_LANG[src])  # wrong tgt tag for lang-contrast
    eos_id = tokenizer.eos_token_id
    start_id = model.config.decoder_start_token_id
    if start_id is None:
        start_id = eos_id

    use_lang = args.mode in ("language", "both")
    use_src = args.mode in ("source", "both")

    lines = args.input.read_text(encoding="utf-8").splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def encode(sentences: list[str], lang_tag: str):
        tokenizer.src_lang = lang_tag
        return tokenizer(sentences, return_tensors="pt", padding=True, truncation=True,
                         max_length=args.max_length).to(device)

    def encoder_out(enc):
        return model.get_encoder()(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"], return_dict=True)

    out_lines: list[str] = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(lines), args.batch_size), desc="contra"):
            batch = lines[i : i + args.batch_size]
            b = len(batch)

            # Build the parallel streams: [real, lang-contrast?, src-contrast?].
            # Each stream has its own encoder inputs and its own forced first token.
            # Weights sum to 1: real = (1+lam), contrasts share -lam.
            streams = []  # [encoder_outputs, attention_mask, first_tok, weight]
            real_enc = encode(batch, NLLB_LANG[src])
            real_eo = encoder_out(real_enc)
            streams.append([real_eo, real_enc["attention_mask"], tgt_id, 1.0 + args.lam])
            n_contrast = int(use_lang) + int(use_src)
            neg = -args.lam / n_contrast
            if use_lang:
                streams.append([real_eo, real_enc["attention_mask"], wrong_id, neg])
            if use_src:
                sc = encode([shuffle_words(s) for s in batch], NLLB_LANG[src])
                streams.append([encoder_out(sc), sc["attention_mask"], tgt_id, neg])

            # Decoder prefix per stream: [start, first_tok]. The generated tokens
            # after position 1 are SHARED across streams.
            dec = [torch.full((b, 1), start_id, device=device) for _ in streams]
            firsts = [torch.full((b, 1), s[2], device=device) for s in streams]
            dec = [torch.cat([d, f], dim=1) for d, f in zip(dec, firsts)]

            past = [None] * len(streams)
            gen_tokens = [[] for _ in range(b)]
            finished = torch.zeros(b, dtype=torch.bool, device=device)

            cur_in = dec  # first forward consumes the 2-token prefix
            for step in range(args.max_length):
                combined = None
                for si, (eo, amask, _ft, w) in enumerate(streams):
                    res = model(
                        encoder_outputs=eo, attention_mask=amask,
                        decoder_input_ids=cur_in[si], past_key_values=past[si],
                        use_cache=True, return_dict=True,
                    )
                    past[si] = res.past_key_values
                    logp = F.log_softmax(res.logits[:, -1, :], dim=-1)
                    combined = w * logp if combined is None else combined + w * logp

                nxt = combined.argmax(dim=-1)  # [b]
                nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
                for r in range(b):
                    if not finished[r]:
                        gen_tokens[r].append(nxt[r].item())
                finished = finished | nxt.eq(eos_id)
                if finished.all():
                    break
                step_in = nxt.unsqueeze(1)
                cur_in = [step_in for _ in streams]  # feed shared token to every stream

            for r in range(b):
                toks = [t for t in gen_tokens[r] if t != eos_id]
                out_lines.append(tokenizer.decode(toks, skip_special_tokens=True))

    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out_lines)} lines -> {args.output}")


if __name__ == "__main__":
    main()
