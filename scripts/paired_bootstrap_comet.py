"""Paired bootstrap resampling on sentence-level COMET scores.

Mirrors sacreBLEU's ``--paired-bs`` convention (Koehn 2004 / Moses
bootstrap-hypothesis-difference-significance.pl, as implemented in
sacrebleu.significance):

  * corpus score  = mean of sentence-level COMET scores
  * delta         = system - baseline   (on the full set)
  * N resamples of the sentence indices WITH replacement, shared across
    baseline and system (paired); per resample d_b = |sys_b - base_b|
  * p = (#{ (d_b - mean(d_b)) > |delta| } + 1) / (N + 1)

Example:
  python scripts/paired_bootstrap_comet.py --src Ar-En/dev.ar --ref Ar-En/dev.en \\
      --baseline predictions/mbr5pub.ar-en.dev.en \\
      --systems predictions/allam-ft.ar-en.dev.en predictions/xalma-cpo.ar-en.dev.en \\
      --names allam-ft xalma-cpo --n-resamples 1000 --seed 12345 \\
      --sentence-scores results/camera_ready/comet-sentence-scores.ar-en.json \\
      --out results/camera_ready/paired-bs-comet.ar-en.json

``--sentence-scores`` caches the per-sentence COMET scores; if the file already
exists and covers every requested hypothesis file, COMET is not re-run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in p.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--systems", required=True, nargs="+", type=Path)
    ap.add_argument("--names", nargs="*", default=None,
                    help="display names for --systems (default: file stem)")
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sentence-scores", type=Path, default=None,
                    help="JSON cache of sentence-level scores keyed by hypothesis path")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = args.names or [p.name.split(".")[0] for p in args.systems]
    assert len(names) == len(args.systems), "--names must match --systems"

    srcs, refs = load(args.src), load(args.ref)
    n = len(srcs)
    assert len(refs) == n, f"src/ref length mismatch: {n} vs {len(refs)}"
    hyp_files = [args.baseline, *args.systems]
    hyps = {str(f): load(f) for f in hyp_files}
    for f, h in hyps.items():
        assert len(h) == n, f"{f}: {len(h)} lines, expected {n}"

    cache: dict = {}
    if args.sentence_scores and args.sentence_scores.exists():
        cache = json.loads(args.sentence_scores.read_text(encoding="utf-8"))
        if cache.get("model") != args.model:
            cache = {}
    scores: dict[str, list[float]] = dict(cache.get("scores", {}))
    todo = [f for f in hyps if f not in scores or len(scores[f]) != n]

    if todo:
        import torch
        from comet import download_model, load_from_checkpoint
        gpus = 1 if torch.cuda.is_available() else 0
        print(f"scoring {len(todo)} hypothesis file(s) with {args.model} (gpus={gpus})")
        model = load_from_checkpoint(download_model(args.model))
        for f in todo:
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps[f], refs)]
            out = model.predict(data, batch_size=args.batch_size, gpus=gpus)
            scores[f] = [float(x) for x in out["scores"]]
            print(f"  {f}: system_score={out['system_score']:.4f}")
        if args.sentence_scores:
            args.sentence_scores.parent.mkdir(parents=True, exist_ok=True)
            args.sentence_scores.write_text(
                json.dumps({"model": args.model, "n": n, "scores": scores}, ensure_ascii=False),
                encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n, size=(args.n_resamples, n), replace=True)

    base = np.asarray(scores[str(args.baseline)], dtype=np.float64)
    base_score = float(base.mean())
    base_bs = base[idxs].mean(axis=1)

    results = []
    for name, f in zip(names, args.systems):
        sy = np.asarray(scores[str(f)], dtype=np.float64)
        sys_score = float(sy.mean())
        delta = sys_score - base_score
        sys_bs = sy[idxs].mean(axis=1)
        sample_diffs = np.abs(sys_bs - base_bs)
        stats = sample_diffs - sample_diffs.mean()
        c = int(np.sum(stats > abs(delta)))
        p = (c + 1) / (args.n_resamples + 1)
        d_bs = sys_bs - base_bs
        lo, hi = np.percentile(d_bs, [2.5, 97.5])
        results.append({
            "name": name, "file": str(f),
            "COMET": round(sys_score, 4),
            "delta": round(delta, 4),
            "delta_bs_mean": round(float(d_bs.mean()), 4),
            "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(p, 4),
        })

    report = {
        "model": args.model, "n": n,
        "n_resamples": args.n_resamples, "seed": args.seed,
        "baseline": {"name": args.baseline.name.split(".")[0],
                     "file": str(args.baseline), "COMET": round(base_score, 4)},
        "systems": results,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
