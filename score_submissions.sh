#!/usr/bin/env bash
# Reference-free QE scoring of all 6 submission files with COMET-Kiwi.
# The challenge test is blind (no human references), so this is the closest
# we can get to predicting how the systems will rank on the official metric.
#
# Outputs a single results/submissions_qe.json with all six scores.

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

TEST_AR="Challenge_Test/Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt"
TEST_EN="Challenge_Test/Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt"

exec > >(tee results/score_submissions.log) 2>&1
echo "=== score_submissions $(date -u +%FT%TZ)  GPU=$CUDA_VISIBLE_DEVICES ==="

python3 <<PY
import json
from pathlib import Path
import torch
from comet import download_model, load_from_checkpoint

print("loading Unbabel/wmt22-cometkiwi-da ...")
qe = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
gpus = 1 if torch.cuda.is_available() else 0

TEST_AR = "$TEST_AR"
TEST_EN = "$TEST_EN"
src_ar = Path(TEST_AR).read_text(encoding="utf-8").splitlines()
src_en = Path(TEST_EN).read_text(encoding="utf-8").splitlines()

submissions = [
    ("primary",       "ar-en", "results/submission-primary.ar-en.test.en"),
    ("primary",       "en-ar", "results/submission-primary.en-ar.test.ar"),
    ("contrastive1-allam",        "ar-en", "results/submission-contrastive1-allam.ar-en.test.en"),
    ("contrastive1-allam",        "en-ar", "results/submission-contrastive1-allam.en-ar.test.ar"),
    ("contrastive2-sampling-mbr", "ar-en", "results/submission-contrastive2-sampling-mbr.ar-en.test.en"),
    ("contrastive2-sampling-mbr", "en-ar", "results/submission-contrastive2-sampling-mbr.en-ar.test.ar"),
]

out = []
for label, direction, hyp_path in submissions:
    hyp = Path(hyp_path).read_text(encoding="utf-8").splitlines()
    src = src_ar if direction == "ar-en" else src_en
    assert len(hyp) == len(src), f"len mismatch on {hyp_path}: {len(hyp)} vs {len(src)}"
    data = [{"src": s, "mt": m} for s, m in zip(src, hyp)]
    print(f"scoring {label} {direction} ({len(data)} pairs) ...")
    res = qe.predict(data, batch_size=32, gpus=gpus)
    score = float(res["system_score"])
    print(f"  COMET-Kiwi = {score:.4f}")
    out.append({"label": label, "direction": direction, "file": hyp_path,
                "cometkiwi_da": round(score, 4), "n": len(hyp)})

Path("results/submissions_qe.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print()
print("=== ranking by COMET-Kiwi ===")
for direction in ("ar-en", "en-ar"):
    print(f"\\n{direction}:")
    rows = sorted([r for r in out if r["direction"] == direction],
                  key=lambda r: -r["cometkiwi_da"])
    for r in rows:
        print(f"  {r['cometkiwi_da']:.4f}  {r['label']}")
PY

bash push_results.sh "QE scoring of 6 submission files (COMET-Kiwi)" 2>&1 | tail -3 || true
echo "done."
