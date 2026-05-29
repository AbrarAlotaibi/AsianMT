"""Build CPO/ALMA-R preference data from sampled candidates + gold references.

For each training pair we form a candidate pool = {gold reference} U {K sampled
hypotheses}, score every candidate with COMET-Kiwi (reference-free quality
estimation), and emit a preference triple:

    chosen   = highest-KIWI candidate
    rejected = lowest-KIWI candidate   (must differ from chosen)

This is the ALMA-R construction (Xu et al., ICML 2024): including the human
reference in the pool means CPO can learn to *prefer* it over fluent-but-wrong
model samples, and to reject its own low-quality samples.

Prerequisite: sampled candidate files written by
    scripts/xalma_translate.py --num-samples K  (-> <out>.sample0..K-1)

Output: JSONL with {"prompt", "chosen", "rejected"} ready for cpo_finetune.py.
`prompt` is already chat-templated so training and inference prompts match.

Example:
  python scripts/build_preference_data.py \\
      --src Ar-En/clean/train.ar --ref Ar-En/clean/train.en --direction ar-en \\
      --samples predictions/cand.ar-en.train.en.sample* \\
      --tokenizer haoranxu/X-ALMA-13B-Group8 \\
      --out data/pref.ar-en.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch

LANG_NAME = {"ar": "Arabic", "en": "English"}


def load(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in Path(p).read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--samples", required=True, nargs="+", type=Path,
                    help="Sampled candidate files (each line-aligned to --src).")
    ap.add_argument("--tokenizer", default="haoranxu/X-ALMA-13B-Group8")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--qe-model", default="Unbabel/wmt22-cometkiwi-da")
    ap.add_argument("--max-pairs", type=int, default=0, help="If >0, cap number of source pairs.")
    ap.add_argument("--min-margin", type=float, default=0.02,
                    help="Skip a pair if best-worst KIWI gap is below this (uninformative).")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    src_lang, tgt_lang = args.direction.split("-")
    srcs, refs = load(args.src), load(args.ref)
    samples = [load(f) for f in args.samples]
    assert len(refs) == len(srcs), "src/ref length mismatch"
    # Candidate files may cover only a prefix of train (we sample a capped subset),
    # so process the largest aligned prefix all files share.
    n = min([len(srcs), len(refs)] + [len(s) for s in samples])
    if args.max_pairs > 0:
        n = min(n, args.max_pairs)
    print(f"aligned pairs: {n}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def make_prompt(sentence: str) -> str:
        sl, tl = LANG_NAME[src_lang], LANG_NAME[tgt_lang]
        p = f"Translate this from {sl} to {tl}:\n{sl}: {sentence}\n{tl}:"
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    # Pool per source = {ref} U {samples}, deduped.
    pools: list[list[str]] = []
    for i in range(n):
        seen, pool = set(), []
        for cand in [refs[i]] + [s[i] for s in samples]:
            cand = cand.strip()
            if cand and cand not in seen:
                seen.add(cand)
                pool.append(cand)
        pools.append(pool)

    from comet import download_model, load_from_checkpoint
    gpus = 1 if torch.cuda.is_available() else 0
    print(f"scoring with {args.qe_model} (gpus={gpus}) ...")
    qe = load_from_checkpoint(download_model(args.qe_model))
    data, index = [], []
    for i in range(n):
        for j, h in enumerate(pools[i]):
            data.append({"src": srcs[i], "mt": h})
            index.append((i, j))
    scores = qe.predict(data, batch_size=args.batch_size, gpus=gpus)["scores"]
    by_pool: list[list[float]] = [[0.0] * len(p) for p in pools]
    for (i, j), s in zip(index, scores):
        by_pool[i][j] = s

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i in range(n):
            if len(pools[i]) < 2:
                continue
            best = max(range(len(pools[i])), key=lambda j: by_pool[i][j])
            worst = min(range(len(pools[i])), key=lambda j: by_pool[i][j])
            if best == worst:
                continue
            if by_pool[i][best] - by_pool[i][worst] < args.min_margin:
                continue
            fh.write(json.dumps({
                "prompt": make_prompt(srcs[i]),
                "chosen": " " + pools[i][best],
                "rejected": " " + pools[i][worst],
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept}/{n} preference triples -> {args.out}")


if __name__ == "__main__":
    main()
