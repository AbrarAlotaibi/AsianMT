#!/usr/bin/env bash
# Zero-shot evaluation of X-ALMA-13B (Arabic = Group 8) on dev, both directions.
# See run_xalma_eval.ps1 for notes. Prereq: pip install peft bitsandbytes

set -euo pipefail

# Unbuffer Python stdout/stderr so tqdm progress bars (and other prints) flow
# through tee in real time instead of pooling in the pipe buffer.
export PYTHONUNBUFFERED=1

DATA_DIR="Ar-En/clean"
PRED_DIR="predictions"
RES_DIR="results"
mkdir -p "$PRED_DIR" "$RES_DIR"
# Tee ALL output to a log so it can be pushed to the repo and reviewed remotely.
exec > >(tee "$RES_DIR/xalma_eval.log") 2>&1
echo "=== run_xalma_eval $(date -u +%FT%TZ) ==="

echo ">>> smoke: 20-line ar->en sanity check"
python scripts/xalma_translate.py --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/xalma.smoke.en" --limit 20
echo "---- first lines of smoke output ----"
head -5 "$PRED_DIR/xalma.smoke.en"

echo ">>> full dev ar->en"
python scripts/xalma_translate.py --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/xalma.ar-en.dev.en"

echo ">>> full dev en->ar"
python scripts/xalma_translate.py --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/xalma.en-ar.dev.ar"

echo ">>> scoring"
python scripts/evaluate.py --hyp "$PRED_DIR/xalma.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/xalma.ar-en.dev.json"
python scripts/evaluate.py --hyp "$PRED_DIR/xalma.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/xalma.en-ar.dev.json"

echo
echo "X-ALMA zero-shot. Compare to NLLB-1.3B official (ar->en 30.79, en->ar 20.29)."
