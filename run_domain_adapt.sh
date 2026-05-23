#!/usr/bin/env bash
# Domain adaptation stage 2: fine-tune the unconstrained checkpoint on official
# in-domain data only. See run_domain_adapt.ps1 for rationale.

set -euo pipefail

BASE="checkpoints/nllb-unconstrained/best"
CKPT="checkpoints/nllb-unconstrained-ft/best"
DATA_DIR="Ar-En/clean"
PRED_DIR="predictions"
RES_DIR="results"

if [ ! -d "$BASE" ]; then
  echo "$BASE not found. Run bash run_unconstrained.sh first." >&2
  exit 1
fi
mkdir -p "$PRED_DIR" "$RES_DIR"

echo ">>> [1/3] fine-tuning unconstrained checkpoint on official in-domain data"
python scripts/train_nllb.py \
  --model "$BASE" \
  --data-dir "$DATA_DIR" \
  --output-dir checkpoints/nllb-unconstrained-ft \
  --epochs 2 \
  --batch-size 16 \
  --grad-accum 2 \
  --lr 1e-5

echo ">>> [2/3] translating dev set"
python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/unconstrained-ft.ar-en.dev.en"
python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/unconstrained-ft.en-ar.dev.ar"

echo ">>> [3/3] scoring dev translations"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/unconstrained-ft.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/unconstrained-ft.ar-en.dev.json"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/unconstrained-ft.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/unconstrained-ft.en-ar.dev.json"

echo
echo "compare to: 600M official (27.49 / 17.27), 600M+external (25.19 / 16.08)."
