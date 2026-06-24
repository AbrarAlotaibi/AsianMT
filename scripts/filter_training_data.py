"""Filter parallel training data by COMET score, keeping the top-K% quality pairs.

Scores each (src, mt, ref) triple with COMET-22 (reference-based) and writes
the top --keep fraction to output files. Fine-tuning on high-quality pairs
consistently beats training on all data when the corpus contains noise.

Usage:
  python scripts/filter_training_data.py \\
      --src Ar-En/clean/train.ar --ref Ar-En/clean/train.en \\
      --keep 0.4 --direction ar-en \\
      --out-src data/filtered.ar --out-ref data/filtered.en
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--direction", required=True, choices=["ar-en", "en-ar"])
    ap.add_argument("--out-src", required=True, type=Path)
    ap.add_argument("--out-ref", required=True, type=Path)
    ap.add_argument("--keep", type=float, default=0.4,
                    help="Fraction of pairs to keep (e.g. 0.4 = top 40%%).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--chunk-size", type=int, default=100_000,
                    help="Score this many pairs at a time to limit peak RAM.")
    args = ap.parse_args()

    src_lines = args.src.read_text(encoding="utf-8").splitlines()
    ref_lines = args.ref.read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(ref_lines), "src/ref line count mismatch"
    n = len(src_lines)
    print(f"{n} pairs  direction={args.direction}  keep={args.keep:.0%}")

    gpus = 1 if torch.cuda.is_available() else 0
    if gpus:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from comet import download_model, load_from_checkpoint
    print(f"loading {args.model} ...")
    model = load_from_checkpoint(download_model(args.model))

    # Score in chunks to avoid loading everything into GPU at once.
    src_lang, tgt_lang = args.direction.split("-")
    all_scores: list[float] = []
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        data = [
            {"src": src_lines[i], "mt": ref_lines[i], "ref": ref_lines[i]}
            for i in range(start, end)
        ]
        scores = model.predict(data, batch_size=args.batch_size, gpus=gpus)["scores"]
        all_scores.extend(scores)
        print(f"  scored {end}/{n}  mean so far: {sum(all_scores)/len(all_scores):.4f}")

    keep_n = max(1, int(n * args.keep))
    # argsort descending
    ranked = sorted(range(n), key=lambda i: all_scores[i], reverse=True)
    keep_idx = set(ranked[:keep_n])

    filtered_src = [src_lines[i] for i in range(n) if i in keep_idx]
    filtered_ref = [ref_lines[i] for i in range(n) if i in keep_idx]

    args.out_src.parent.mkdir(parents=True, exist_ok=True)
    args.out_ref.parent.mkdir(parents=True, exist_ok=True)
    args.out_src.write_text("\n".join(filtered_src) + "\n", encoding="utf-8")
    args.out_ref.write_text("\n".join(filtered_ref) + "\n", encoding="utf-8")

    threshold = all_scores[ranked[keep_n - 1]]
    print(f"kept {keep_n}/{n} pairs  score threshold: {threshold:.4f}")
    print(f"  -> {args.out_src}")
    print(f"  -> {args.out_ref}")


if __name__ == "__main__":
    main()
