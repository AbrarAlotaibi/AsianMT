#!/usr/bin/env bash
# Unconstrained system: fine-tune on official + external OPUS data
# (Ar-En/unconstrained from scripts/fetch_parallel.py). See run_unconstrained.ps1
# for notes. Defaults to 600M; switch MODEL to the 1.3B (much slower) if desired.

set -euo pipefail

MODEL="facebook/nllb-200-distilled-600M"
DATA_DIR="Ar-En/unconstrained"
CKPT="checkpoints/nllb-unconstrained/best"
PRED_DIR="predictions"
RES_DIR="results"

if [ ! -f "$DATA_DIR/train.ar" ]; then
  echo "$DATA_DIR/train.ar not found. Run scripts/fetch_parallel.py first." >&2
  exit 1
fi
mkdir -p "$PRED_DIR" "$RES_DIR"

echo ">>> [1/3] fine-tuning $MODEL on official+external (both directions)"
python scripts/train_nllb.py \
  --model "$MODEL" \
  --data-dir "$DATA_DIR" \
  --output-dir checkpoints/nllb-unconstrained \
  --epochs 2 \
  --batch-size 32 \
  --grad-accum 2 \
  --lr 3e-5

echo ">>> [2/3] translating dev set"
python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/unconstrained.ar-en.dev.en"
python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/unconstrained.en-ar.dev.ar"

echo ">>> [3/3] scoring dev translations"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/unconstrained.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/unconstrained.ar-en.dev.json"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/unconstrained.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/unconstrained.en-ar.dev.json"

echo
echo "done. compare to 600M official baseline (ar->en 27.49, en->ar 17.27)."
