"""Score a hypothesis file against a reference file with BLEU, chrF, TER.

Optionally also COMET (requires unbabel-comet and a model download, ~2 GB):
  python scripts/evaluate.py --hyp pred.en --ref devtest.en --src devtest.ar --comet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in p.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--src", type=Path, default=None, help="needed for COMET")
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    args = ap.parse_args()

    import sacrebleu

    hyps = load(args.hyp)
    refs = load(args.ref)
    assert len(hyps) == len(refs), f"len mismatch: hyp={len(hyps)} ref={len(refs)}"

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    ter = sacrebleu.corpus_ter(hyps, [refs])

    results = {
        "n": len(hyps),
        "BLEU": round(bleu.score, 2),
        "chrF": round(chrf.score, 2),
        "TER": round(ter.score, 2),
        "BLEU_signature": bleu.get_signature().format(),
    }

    if args.comet:
        assert args.src is not None, "--src is required for COMET"
        from comet import download_model, load_from_checkpoint
        srcs = load(args.src)
        assert len(srcs) == len(hyps)
        model_path = download_model(args.comet_model)
        model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = model.predict(data, batch_size=16, gpus=1 if __import__("torch").cuda.is_available() else 0)
        results["COMET"] = round(out["system_score"], 4)
        results["COMET_model"] = args.comet_model

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
