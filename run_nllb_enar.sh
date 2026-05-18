#!/usr/bin/env bash
# Per-direction fine-tune for en->ar (Sub-Task 1A), starting from the bidirectional
# checkpoint produced by run_nllb.sh. See run_nllb_enar.ps1 for full notes.

set -euo pipefail

BASE_CKPT="checkpoints/nllb-Ar-En/best"
OUT_CKPT="checkpoints/nllb-Ar-En-enar/best"
DATA_DIR="Ar-En/clean"
PRED_DIR="predictions"
RES_DIR="results"

if [ ! -d "$BASE_CKPT" ]; then
  echo "Base checkpoint not found at $BASE_CKPT. Run bash run_nllb.sh first." >&2
  exit 1
fi

mkdir -p "$PRED_DIR" "$RES_DIR"

echo ">>> [1/3] en->ar fine-tune from bidir checkpoint"
python scripts/train_nllb.py \
  --model "$BASE_CKPT" \
  --direction en-ar \
  --epochs 2 \
  --lr 1e-5 \
  --batch-size 8 \
  --grad-accum 4 \
  --output-dir checkpoints/nllb-Ar-En-enar

echo ">>> [2/3] translating dev (en->ar)"
python scripts/translate.py \
  --model "$OUT_CKPT" --kind nllb --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/nllb-enar.en-ar.dev.ar"

echo ">>> [3/3] scoring dev (en->ar)"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/nllb-enar.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/nllb-enar.en-ar.dev.json"

echo
echo "done. compare to baseline nllb.en-ar.dev.json (BLEU 17.27)."
