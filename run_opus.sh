#!/usr/bin/env bash
# Contrastive system: fine-tune Helsinki-NLP opus-mt for both directions, then
# translate dev and score. See run_opus.ps1 for notes.

set -euo pipefail

DATA_DIR="Ar-En/clean"
PRED_DIR="predictions"
RES_DIR="results"
mkdir -p "$PRED_DIR" "$RES_DIR"

for dir in ar-en en-ar; do
  echo ">>> training opus-mt $dir"
  python scripts/train_opus.py --direction "$dir" --epochs 5 --batch-size 16 --grad-accum 2

  ckpt="checkpoints/opus-Ar-En-$dir/best"
  if [ "$dir" = "ar-en" ]; then
    inp="$DATA_DIR/dev.ar"; ref="$DATA_DIR/dev.en"; out="$PRED_DIR/opus.ar-en.dev.en"
  else
    inp="$DATA_DIR/dev.en"; ref="$DATA_DIR/dev.ar"; out="$PRED_DIR/opus.en-ar.dev.ar"
  fi

  echo ">>> translating dev ($dir)"
  python scripts/translate.py --model "$ckpt" --kind opus --direction "$dir" --input "$inp" --output "$out"

  echo ">>> scoring dev ($dir)"
  python scripts/evaluate.py --hyp "$out" --ref "$ref" | tee "$RES_DIR/opus.$dir.dev.json"
done

echo
echo "done. compare to NLLB-1.3B (ar->en 30.79, en->ar 20.29)."
