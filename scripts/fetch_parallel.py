"""Download curated ar-en parallel corpora from OPUS and merge with the official data.

Resolves each corpus's latest 'moses' download via the OPUS API (no hardcoded
versions), then:
  - normalizes (NFKC + whitespace)
  - DROPS any pair whose ar or en side appears in dev/devtest (test-contamination guard)
  - dedups against the official training set and within the external data
  - applies the same length / length-ratio filters as clean.py

Output is a ready-to-train directory: train.{ar,en} = official + external (deduped),
plus copies of dev/devtest from --official-dir.

Example (the curated clean mix):
  python scripts/fetch_parallel.py \\
      --corpora News-Commentary GlobalVoices TED2020 Tatoeba \\
      --official-dir Ar-En/clean \\
      --out-dir Ar-En/unconstrained

Browse corpus names at https://opus.nlpl.eu/ . If the OPUS API is unreachable,
re-run later; it occasionally rate-limits.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

OPUS_API = "https://opus.nlpl.eu/opusapi/"


def normalize(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).split())


def resolve_url(corpus: str, src: str, tgt: str) -> str | None:
    q = urllib.parse.urlencode({
        "corpus": corpus, "source": src, "target": tgt,
        "preprocessing": "moses", "version": "latest",
    })
    req = urllib.request.Request(f"{OPUS_API}?{q}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    entries = [c for c in data.get("corpora", [])
               if c.get("url", "").endswith(".zip") and "/moses/" in c.get("url", "")]
    return entries[0]["url"] if entries else None


def download_pairs(url: str, src: str, tgt: str) -> list[tuple[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        src_name = next((n for n in names if n.endswith(f".{src}")), None)
        tgt_name = next((n for n in names if n.endswith(f".{tgt}")), None)
        if not src_name or not tgt_name:
            return []
        with z.open(src_name) as fs:
            s_lines = io.TextIOWrapper(fs, encoding="utf-8").read().splitlines()
        with z.open(tgt_name) as ft:
            t_lines = io.TextIOWrapper(ft, encoding="utf-8").read().splitlines()
    return list(zip(s_lines, t_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+",
                    default=["News-Commentary", "GlobalVoices", "TED2020", "Tatoeba"])
    ap.add_argument("--official-dir", default="Ar-En/clean", type=Path)
    ap.add_argument("--out-dir", default="Ar-En/unconstrained", type=Path)
    ap.add_argument("--src", default="ar")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    ap.add_argument("--max-extra", type=int, default=None,
                    help="Cap on total EXTERNAL pairs (randomly sampled across all corpora). "
                         "Official pairs are always kept. Omit to keep everything.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # Contamination blocklist: every ar/en line in dev + devtest.
    block: set[str] = set()
    for split in ("dev", "devtest"):
        for lang in (args.src, args.tgt):
            p = args.official_dir / f"{split}.{lang}"
            if p.exists():
                block.update(normalize(x) for x in p.read_text(encoding="utf-8").splitlines())
    print(f"contamination blocklist: {len(block)} unique eval lines")

    # Seed output with official training data; dedup external against it.
    off_src = [normalize(x) for x in (args.official_dir / f"train.{args.src}").read_text(encoding="utf-8").splitlines()]
    off_tgt = [normalize(x) for x in (args.official_dir / f"train.{args.tgt}").read_text(encoding="utf-8").splitlines()]
    seen: set[tuple[str, str]] = set(zip(off_src, off_tgt))
    print(f"official train pairs: {len(off_src)}")

    extra: list[tuple[str, str]] = []
    for corpus in args.corpora:
        try:
            url = resolve_url(corpus, args.src, args.tgt)
        except Exception as e:  # noqa: BLE001
            print(f"  {corpus}: API error ({e}); skipping")
            continue
        if not url:
            print(f"  {corpus}: no {args.src}-{args.tgt} moses release found; skipping")
            continue
        pairs = download_pairs(url, args.src, args.tgt)
        kept = 0
        drop = {"empty": 0, "contam": 0, "dup": 0, "length": 0, "ratio": 0}
        for s, t in pairs:
            ns, nt = normalize(s), normalize(t)
            if not ns or not nt:
                drop["empty"] += 1; continue
            if ns in block or nt in block:
                drop["contam"] += 1; continue
            if (ns, nt) in seen:
                drop["dup"] += 1; continue
            a, b = len(ns.split()), len(nt.split())
            if a < args.min_tokens or b < args.min_tokens or a > args.max_tokens or b > args.max_tokens:
                drop["length"] += 1; continue
            if max(a, b) / max(min(a, b), 1) > args.max_ratio:
                drop["ratio"] += 1; continue
            seen.add((ns, nt))
            extra.append((ns, nt))
            kept += 1
        print(f"  {corpus}: downloaded {len(pairs)}, kept {kept}  (drop {drop})")

    if args.max_extra is not None and len(extra) > args.max_extra:
        print(f"sampling {args.max_extra} of {len(extra)} external pairs (seed {args.seed})")
        extra = random.sample(extra, args.max_extra)

    out_src = list(off_src) + [s for s, _ in extra]
    out_tgt = list(off_tgt) + [t for _, t in extra]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"train.{args.src}").write_text("\n".join(out_src) + "\n", encoding="utf-8")
    (args.out_dir / f"train.{args.tgt}").write_text("\n".join(out_tgt) + "\n", encoding="utf-8")
    import shutil
    for split in ("dev", "devtest"):
        for lang in (args.src, args.tgt):
            p = args.official_dir / f"{split}.{lang}"
            if p.exists():
                shutil.copyfile(p, args.out_dir / f"{split}.{lang}")

    print(f"\ntotal training pairs: {len(out_src)}  (was {len(off_src)} official)")
    print(f"wrote -> {args.out_dir}/train.{{{args.src},{args.tgt}}}")


if __name__ == "__main__":
    main()
