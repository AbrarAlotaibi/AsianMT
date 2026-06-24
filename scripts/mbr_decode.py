"""COMET-MBR decoding / system combination over candidate files.

Given a source file and several candidate hypothesis files (all line-aligned to
the source), pick, per sentence, the candidate that maximizes a COMET-based
objective. Two roles, same script:

  * System combination: pass the single best output of each system as candidate
    files -> per-sentence "ensemble" that takes the best system for each input.
  * Sampling-MBR: pass N sampled hypotheses from one (or several) systems
    -> classic Minimum Bayes Risk decoding.

Pipeline (Eikema & Aziz 2020; "Chasing COMET" 2024):
  1. Pool candidates per source (dedup identical strings).
  2. Optional QE pre-pruning: score each (src, cand) with COMET-Kiwi
     (reference-free) and keep the top --prune-k. Cuts the O(K^2) MBR cost.
  3. MBR: for each source pick
         argmax_h  (1/|P|) sum_{h' in P} COMET22(src, mt=h, ref=h')
     i.e. the candidate most "central" under COMET to the candidate pool.

Examples:
  # System combination of three best-1 outputs (cheap, no sampling needed)
  python scripts/mbr_decode.py --src Ar-En/clean/dev.ar \\
      --candidates predictions/xalma-ft.ar-en.dev.en \\
                   predictions/nllb33b.ar-en.dev.en \\
                   predictions/opus.ar-en.dev.en \\
      --output predictions/mbr.ar-en.dev.en

  # Sampling-MBR over 32 samples (files written by translate.py --num-samples)
  python scripts/mbr_decode.py --src Ar-En/clean/dev.ar \\
      --candidates predictions/xalma.ar-en.dev.en.sample* \\
      --prune-k 16 --output predictions/mbr.ar-en.dev.en
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch


def load_lines(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in Path(p).read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--candidates", required=True, nargs="+", type=Path,
                    help="One or more candidate files, each line-aligned to --src.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--qe-model", default="Unbabel/wmt22-cometkiwi-da",
                    help="Reference-free model for QE pre-pruning.")
    ap.add_argument("--mbr-model", default="Unbabel/wmt22-comet-da",
                    help="Reference-based model used as the MBR utility.")
    ap.add_argument("--prune-k", type=int, default=0,
                    help="If >0, QE-prune each candidate pool to the top-K before MBR. "
                         "0 = no pruning (fine for small pools like system combination).")
    ap.add_argument("--method", choices=["mbr", "qe"], default="mbr",
                    help="'mbr' = COMET-MBR over the (pruned) pool. "
                         "'qe' = just take the top QE candidate (QE reranking).")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    gpus = 1 if torch.cuda.is_available() else 0
    if gpus:
        print(f"GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")
    else:
        print("WARNING: no GPU; COMET scoring on CPU will be slow.")

    from comet import download_model, load_from_checkpoint

    srcs = load_lines(args.src)
    cand_files = []
    kept_paths = []
    for f in args.candidates:
        c = load_lines(f)
        if len(c) != len(srcs):
            print(f"WARNING: {f} has {len(c)} lines, src has {len(srcs)}; "
                  f"dropping this candidate from the pool.")
            continue
        cand_files.append(c)
        kept_paths.append(f)
    if not cand_files:
        sys.exit("ERROR: no candidate files matched the source line count.")
    n = len(srcs)
    print(f"{n} sources x {len(cand_files)} candidate files (kept after length check)")

    # Per-source deduped candidate pool (order-stable).
    pools: list[list[str]] = []
    for i in range(n):
        seen, pool = set(), []
        for cf in cand_files:
            h = cf[i]
            if h not in seen:
                seen.add(h)
                pool.append(h)
        pools.append(pool)

    need_qe = args.method == "qe" or args.prune_k > 0
    qe_by_pool: list[list[float]] = [[] for _ in range(n)]
    qe_ok = False
    if need_qe:
        print(f"loading QE model {args.qe_model} ...")
        try:
            qe = load_from_checkpoint(download_model(args.qe_model))
            qe_data, qe_index = [], []
            for i, pool in enumerate(pools):
                for j, h in enumerate(pool):
                    qe_data.append({"src": srcs[i], "mt": h})
                    qe_index.append((i, j))
            scores = qe.predict(qe_data, batch_size=args.batch_size, gpus=gpus)["scores"]
            for (i, j), s in zip(qe_index, scores):
                qe_by_pool[i].append(s)
            qe_ok = True
        except Exception as e:
            print(f"WARNING: QE model load/predict failed ({type(e).__name__}: {e})")
            if args.method == "qe":
                sys.exit("--method qe requires the QE model; aborting.")
            print(f"falling back: deterministic head-of-pool pruning to {args.prune_k} candidates.")

    # Prune each pool to the top-K candidates (by QE if available, else by file order).
    if args.prune_k > 0:
        for i in range(n):
            if qe_ok:
                order = sorted(range(len(pools[i])), key=lambda j: qe_by_pool[i][j], reverse=True)
            else:
                order = list(range(len(pools[i])))
            keep = order[: args.prune_k]
            pools[i] = [pools[i][j] for j in keep]
            if qe_ok:
                qe_by_pool[i] = [qe_by_pool[i][j] for j in keep]

    if args.method == "qe":
        chosen = [pools[i][max(range(len(pools[i])), key=lambda j: qe_by_pool[i][j])]
                  for i in range(n)]
    else:
        # MBR: score every (hypothesis, pseudo-ref) pair in each pool with COMET-22,
        # then pick the hypothesis with the highest mean utility over the pool.
        print(f"loading MBR utility model {args.mbr_model} ...")
        mbr = load_from_checkpoint(download_model(args.mbr_model))
        data, index = [], []
        for i, pool in enumerate(pools):
            for a, hyp in enumerate(pool):
                for b, ref in enumerate(pool):
                    data.append({"src": srcs[i], "mt": hyp, "ref": ref})
                    index.append((i, a, b))
        print(f"scoring {len(data)} (hyp,pseudo-ref) pairs ...")
        scores = mbr.predict(data, batch_size=args.batch_size, gpus=gpus)["scores"]

        # Accumulate utility[i][a] = sum_b COMET(src_i, pool[a], pool[b]).
        util = [[0.0] * len(pool) for pool in pools]
        for (i, a, _b), s in zip(index, scores):
            util[i][a] += s
        chosen = [pools[i][max(range(len(pools[i])), key=lambda a: util[i][a])]
                  for i in range(n)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"wrote {len(chosen)} lines -> {args.output}")


if __name__ == "__main__":
    main()
