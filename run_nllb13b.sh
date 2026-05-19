#!/usr/bin/env bash
# Bidirectional fine-tune of NLLB-200-distilled-1.3B for Ar<->En.
# See run_nllb13b.ps1 for full notes. Expected wall time: ~7-10 h overnight.

set -euo pipefail

DATA_DIR="Ar-En/clean"
CKPT="checkpoints/nllb13b-Ar-En/best"
PRED_DIR="predictions"
RES_DIR="results"

mkdir -p "$PRED_DIR" "$RES_DIR"

if [ ! -f "$DATA_DIR/train.ar" ]; then
  echo ">>> [1/4] cleaning training data"
  python scripts/clean.py
else
  echo ">>> [1/4] cleaned data already present, skipping clean.py"
fi

echo ">>> [2/4] fine-tuning NLLB-200-distilled-1.3B (both directions)"
python scripts/train_nllb.py \
  --model facebook/nllb-200-distilled-1.3B \
  --output-dir checkpoints/nllb13b-Ar-En \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 3e-5 \
  --optim adamw_bnb_8bit

echo ">>> [3/4] translating dev set"
python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/nllb13b.ar-en.dev.en"

python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/nllb13b.en-ar.dev.ar"

echo ">>> [4/4] scoring dev translations"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/nllb13b.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/nllb13b.ar-en.dev.json"

python scripts/evaluate.py \
  --hyp "$PRED_DIR/nllb13b.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/nllb13b.en-ar.dev.json"

echo
echo "done. compare to 600M baseline (ar->en 27.49, en->ar 17.27)."
