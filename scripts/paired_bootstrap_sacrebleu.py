"""Paired bootstrap (BLEU / chrF / TER) via sacreBLEU's own PairedTest API,
written as full-precision JSON.

Equivalent to
  sacrebleu REF -i BASELINE SYS1 SYS2 ... -m bleu chrf ter --paired-bs --paired-bs-n N
(same metrics with default settings, same seed handling via SACREBLEU_SEED,
default 12345) -- the CLI's text table rounds to one decimal and its
``-f json`` crashes on numpy float32 in sacrebleu 2.6.0.

Example:
  python scripts/paired_bootstrap_sacrebleu.py --ref Ar-En/dev.en \\
      --baseline predictions/mbr5pub.ar-en.dev.en \\
      --systems predictions/allam-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en \\
      --names allam-ft xalma-cpo --n-resamples 1000 \\
      --out results/camera_ready/paired-bs.ar-en.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF, TER
from sacrebleu.significance import PairedTest


def load(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in p.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--systems", required=True, nargs="+", type=Path)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = args.names or [p.name.split(".")[0] for p in args.systems]
    assert len(names) == len(args.systems), "--names must match --systems"
    base_name = args.baseline.name.split(".")[0]

    refs = load(args.ref)
    named = [(base_name, load(args.baseline))]
    for name, f in zip(names, args.systems):
        named.append((name, load(f)))
    for name, h in named:
        assert len(h) == len(refs), f"{name}: {len(h)} lines, ref has {len(refs)}"

    metrics = {"BLEU": BLEU(), "chrF": CHRF(), "TER": TER()}
    test = PairedTest(named, metrics, [refs], test_type="bs", n_samples=args.n_resamples)
    signatures, scores = test()
    # PairedTest keys its output by the metrics' own names (BLEU, chrF2, TER),
    # in the order the metrics were given.
    keys = dict(zip(metrics, [k for k in scores if k != "System"]))
    signatures = {m: signatures[k] for m, k in keys.items()}

    base = {m: scores[k][0] for m, k in keys.items()}
    systems = []
    for i, (name, f) in enumerate(zip(names, args.systems), start=1):
        entry = {"name": name, "file": str(f)}
        for m, k in keys.items():
            r = scores[k][i]
            entry[m] = {
                "score": round(float(r.score), 4),
                "delta": round(float(r.score - base[m].score), 4),
                "bs_mean": round(float(r.mean), 4),
                "ci95": round(float(r.ci), 4),
                "p": round(float(r.p_value), 4),
            }
        systems.append(entry)

    report = {
        "n": len(refs),
        "n_resamples": args.n_resamples,
        "seed": str(test._seed),
        "signatures": {m: str(sig) for m, sig in signatures.items()},
        "baseline": {"name": base_name, "file": str(args.baseline),
                     **{m: round(float(base[m].score), 4) for m in metrics}},
        "systems": systems,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
