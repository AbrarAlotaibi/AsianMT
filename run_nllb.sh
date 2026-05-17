#!/usr/bin/env bash
# Full NLLB-600M training run for Ar<->En, then translate dev and score.
# Expected wall time on a 12 GB GPU: ~1.5-3 hours for 3 epochs.
#
# Outputs:
#   checkpoints/nllb-Ar-En/best/                       fine-tuned model
#   predictions/nllb.ar-en.dev.en                      Ar->En dev translations
#   predictions/nllb.en-ar.dev.ar                      En->Ar dev translations
#   results/nllb.ar-en.dev.json                        BLEU/chrF/TER (ar->en)
#   results/nllb.en-ar.dev.json                        BLEU/chrF/TER (en->ar)
#
# We score on DEV (not devtest) — keeping devtest pristine for final submission
# so we don't accidentally overfit to it during iteration.

set -euo pipefail

DATA_DIR="Ar-En/clean"
CKPT="checkpoints/nllb-Ar-En/best"
PRED_DIR="predictions"
RES_DIR="results"

mkdir -p "$PRED_DIR" "$RES_DIR"

# 1. Clean data once
if [ ! -f Ar-En/clean/train.ar ]; then
  echo ">>> [1/4] cleaning training data"
  python scripts/clean.py
else
  echo ">>> [1/4] cleaned data already present, skipping clean.py"
fi

# 2. Train (3 epochs, fp16, grad-accum 8 for 12 GB GPU)
echo ">>> [2/4] fine-tuning NLLB-200-distilled-600M (both directions)"
python scripts/train_nllb.py \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 3e-5

# 3. Translate dev both directions
echo ">>> [3/4] translating dev set"
python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/nllb.ar-en.dev.en"

python scripts/translate.py \
  --model "$CKPT" --kind nllb --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/nllb.en-ar.dev.ar"

# 4. Score
echo ">>> [4/4] scoring dev translations"
python scripts/evaluate.py \
  --hyp "$PRED_DIR/nllb.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/nllb.ar-en.dev.json"

python scripts/evaluate.py \
  --hyp "$PRED_DIR/nllb.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/nllb.en-ar.dev.json"

echo
echo "done. dev scores are above. send them back so we can decide next steps."
