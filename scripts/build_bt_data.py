"""Assemble a per-direction training set: real parallel + back-translated synthetic.

Back-translation rule: synthetic data is only ever used in the direction where the
TARGET side is real human text. The SOURCE side is the (synthetic) machine translation.

For --direction en-ar  (we want to improve en->ar; target ar must be real):
    --mono   = real monolingual Arabic        (becomes target, train.ar)
    --synth  = back-translation of that Arabic via ar->en  (becomes source, train.en)

For --direction ar-en  (we want to improve ar->en; target en must be real):
    --mono   = real monolingual English       (becomes target, train.en)
    --synth  = back-translation of that English via en->ar (becomes source, train.ar)

--mono and --synth must have the same number of lines, aligned by index.

Output: <out-dir>/train.{ar,en} = real_parallel (optionally oversampled) + synthetic,
plus copies of dev.{ar,en} from --real-dir so train_nllb.py can evaluate.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["ar-en", "en-ar"], required=True)
    ap.add_argument("--real-dir", default="Ar-En/clean", type=Path)
    ap.add_argument("--mono", required=True, type=Path, help="real monolingual TARGET-side text")
    ap.add_argument("--synth", required=True, type=Path, help="back-translated SOURCE-side text (aligned to --mono)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--oversample-real", type=int, default=1,
                    help="Repeat real parallel data this many times to balance against synthetic.")
    args = ap.parse_args()

    src, tgt = args.direction.split("-")  # en-ar -> src=en, tgt=ar

    mono = read(args.mono)    # real, target language
    synth = read(args.synth)  # synthetic, source language
    if len(mono) != len(synth):
        raise SystemExit(f"--mono ({len(mono)}) and --synth ({len(synth)}) must align line-for-line")

    real_src = read(args.real_dir / f"train.{src}")
    real_tgt = read(args.real_dir / f"train.{tgt}")
    if len(real_src) != len(real_tgt):
        raise SystemExit(f"real train mismatch: {len(real_src)} vs {len(real_tgt)}")

    out_src = real_src * args.oversample_real + synth
    out_tgt = real_tgt * args.oversample_real + mono
    assert len(out_src) == len(out_tgt)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"train.{src}").write_text("\n".join(out_src) + "\n", encoding="utf-8")
    (args.out_dir / f"train.{tgt}").write_text("\n".join(out_tgt) + "\n", encoding="utf-8")
    for lang in ("ar", "en"):
        shutil.copyfile(args.real_dir / f"dev.{lang}", args.out_dir / f"dev.{lang}")

    print(f"[{args.direction}] real={len(real_src)} x{args.oversample_real} + synth={len(synth)} "
          f"-> {len(out_src)} pairs in {args.out_dir}")


if __name__ == "__main__":
    main()
