"""Filter parallel ar/en data: dedupe pairs, drop length-ratio and length outliers.

Reads Ar-En/{train,dev,devtest}.{ar,en} and writes Ar-En/clean/ with the same names.
Dev and devtest are kept as-is (we never filter eval data) — only train is cleaned.
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path


def normalize(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    return " ".join(line.split())


def clean_train(src_path: Path, tgt_path: Path, out_src: Path, out_tgt: Path,
                max_tokens: int, min_tokens: int, max_ratio: float) -> None:
    with src_path.open(encoding="utf-8") as fs, tgt_path.open(encoding="utf-8") as ft:
        pairs = [(normalize(s), normalize(t)) for s, t in zip(fs, ft)]

    seen: set[tuple[str, str]] = set()
    kept: list[tuple[str, str]] = []
    dropped = {"empty": 0, "dup": 0, "length": 0, "ratio": 0}

    for s, t in pairs:
        if not s or not t:
            dropped["empty"] += 1
            continue
        key = (s, t)
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        ns, nt = len(s.split()), len(t.split())
        if ns < min_tokens or nt < min_tokens or ns > max_tokens or nt > max_tokens:
            dropped["length"] += 1
            continue
        ratio = max(ns, nt) / max(min(ns, nt), 1)
        if ratio > max_ratio:
            dropped["ratio"] += 1
            continue
        kept.append((s, t))

    out_src.parent.mkdir(parents=True, exist_ok=True)
    out_src.write_text("\n".join(s for s, _ in kept) + "\n", encoding="utf-8")
    out_tgt.write_text("\n".join(t for _, t in kept) + "\n", encoding="utf-8")

    print(f"input pairs:   {len(pairs)}")
    print(f"kept:          {len(kept)}")
    for k, v in dropped.items():
        print(f"dropped {k:7s} {v}")


def copy_through(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [normalize(l) for l in src.read_text(encoding="utf-8").splitlines()]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="Ar-En", type=Path)
    ap.add_argument("--out-dir", default="Ar-En/clean", type=Path)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    args = ap.parse_args()

    print(">>> cleaning train")
    clean_train(
        args.raw_dir / "train.ar", args.raw_dir / "train.en",
        args.out_dir / "train.ar", args.out_dir / "train.en",
        args.max_tokens, args.min_tokens, args.max_ratio,
    )
    for split in ("dev", "devtest"):
        for lang in ("ar", "en"):
            print(f">>> passthrough {split}.{lang}")
            copy_through(args.raw_dir / f"{split}.{lang}", args.out_dir / f"{split}.{lang}")


if __name__ == "__main__":
    main()
