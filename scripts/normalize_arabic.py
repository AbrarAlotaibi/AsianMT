"""Arabic text normalization for MT preprocessing.

Applies standard normalizations before feeding Arabic source to any MT model:
  - Remove tatweel/kashida (ـ)
  - Remove diacritics/harakat (tashkeel)
  - Unify Alef variants (أإآٱ → ا)
  - Normalize Alef maqsura (ى → ي)

Usage:
  python scripts/normalize_arabic.py input.ar output.ar
  python scripts/normalize_arabic.py --check input.ar   # show diff stats only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Alef variants → bare Alef
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا"})

# Alef maqsura → ya
_YA = str.maketrans({"ى": "ي"})

# Tatweel (kashida elongation mark)
_TATWEEL = str.maketrans({"ـ": ""})

# Diacritics: harakat + shadda + sukun + superscript alef (U+064B–U+065F, U+0670)
_DIACRITICS = re.compile(r"[ً-ٰٟ]")

# Repeated whitespace
_SPACES = re.compile(r"  +")


def normalize(text: str) -> str:
    text = text.translate(_TATWEEL)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ALEF)
    text = text.translate(_YA)
    text = _SPACES.sub(" ", text).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", nargs="?", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="Print change statistics without writing output.")
    args = ap.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    normed = [normalize(l) for l in lines]

    changed = sum(a != b for a, b in zip(lines, normed))
    chars_removed = sum(len(a) - len(b) for a, b in zip(lines, normed))
    print(f"{args.input}: {len(lines)} lines, {changed} changed, {chars_removed} chars removed")

    if args.check:
        for i, (a, b) in enumerate(zip(lines, normed)):
            if a != b:
                print(f"  line {i+1}: {a!r} -> {b!r}")
        return

    if args.output is None:
        sys.exit("ERROR: provide output path or use --check")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(normed) + "\n", encoding="utf-8")
    print(f"wrote {len(normed)} lines -> {args.output}")


if __name__ == "__main__":
    main()
