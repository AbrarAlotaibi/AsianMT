"""Download monolingual sentences from the Leipzig Corpora Collection and filter them.

Leipzig packages are clean, deduplicated, one-sentence-per-line corpora — ideal for
back-translation. Browse available corpora and exact names at:
  https://wortschatz.uni-leipzig.de/en/download

Corpus names look like  <lang3>_<source>_<year>_<size>  e.g.:
  ara_news_2022_300K   eng_news_2024_300K   ara_wikipedia_2021_300K

Example:
  python scripts/fetch_mono.py --corpus ara_news_2022_300K --lang ar \\
      --n 100000 --output Ar-En/mono/mono.ar
  python scripts/fetch_mono.py --corpus eng_news_2024_300K --lang en \\
      --n 100000 --output Ar-En/mono/mono.en

If a download 404s, the corpus name/year/size doesn't exist — pick a valid one
from the portal above.
"""
from __future__ import annotations

import argparse
import io
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

BASE = "https://downloads.wortschatz-leipzig.de/corpora/"
ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RE = re.compile(r"[A-Za-z]")


def script_ratio(text: str, pattern: re.Pattern) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if pattern.match(c)) / len(chars)


def iter_sentences(corpus: str):
    url = f"{BASE}{corpus}.tar.gz"
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith("-sentences.txt")), None)
        if member is None:
            raise SystemExit("no *-sentences.txt found in archive; is this a valid Leipzig corpus?")
        fh = tar.extractfile(member)
        assert fh is not None
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t", 1)  # format: <id>\t<sentence>
            if len(parts) == 2:
                yield parts[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Leipzig corpus name, e.g. ara_news_2022_300K")
    ap.add_argument("--lang", choices=["ar", "en"], required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--min-script-ratio", type=float, default=0.5)
    args = ap.parse_args()

    pattern = ARABIC_RE if args.lang == "ar" else LATIN_RE
    seen: set[str] = set()
    kept: list[str] = []
    scanned = 0

    for raw in iter_sentences(args.corpus):
        scanned += 1
        text = " ".join(unicodedata.normalize("NFKC", raw).split())
        if not text:
            continue
        ntok = len(text.split())
        if ntok < args.min_tokens or ntok > args.max_tokens:
            continue
        if script_ratio(text, pattern) < args.min_script_ratio:
            continue
        if text in seen:
            continue
        seen.add(text)
        kept.append(text)
        if len(kept) >= args.n:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"scanned {scanned} sentences, wrote {len(kept)} -> {args.output}")


if __name__ == "__main__":
    main()
