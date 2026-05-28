#!/usr/bin/env bash
# NLLB-200 3.3B bidirectional fine-tune for Ar<->En.
# Bigger than the 1.3B baseline (3.3B params, not distilled). Expected:
# +2 to +4 BLEU per direction.
#
# Tuned for a single 24 GB 3090: fp16 + 8-bit AdamW + gradient checkpointing.
# batch=4 grad_accum=8 -> effective batch 32 (same as 1.3B run).
#
# Expected wall time: ~5-8 h for 2 epochs on a 3090.
#
# Run on a specific GPU when paralleling other experiments:
#   CUDA_VISIBLE_DEVICES=0 bash run_nllb33b.sh

set -euo pipefail
export PYTHONUNBUFFERED=1

DATA_DIR="Ar-En/clean"
CKPT="checkpoints/nllb33b-Ar-En/best"
PRED_DIR="predictions"
RES_DIR="results"
mkdir -p "$PRED_DIR" "$RES_DIR"
exec > >(tee "$RES_DIR/nllb33b.log") 2>&1
echo "=== run_nllb33b $(date -u +%FT%TZ) ==="

if [ ! -f "$DATA_DIR/train.ar" ]; then
  python scripts/clean.py
fi

echo ">>> [1/4] fine-tuning NLLB-200-3.3B (both directions)"
python scripts/train_nllb.py \
  --model facebook/nllb-200-3.3B \
  --output-dir checkpoints/nllb33b-Ar-En \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 3e-5 \
  --optim adamw_bnb_8bit

echo ">>> [2/4] translating dev (ar->en)"
python scripts/translate.py --model "$CKPT" --kind nllb --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/nllb33b.ar-en.dev.en"

echo ">>> [3/4] translating dev (en->ar)"
python scripts/translate.py --model "$CKPT" --kind nllb --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/nllb33b.en-ar.dev.ar"

echo ">>> [4/4] scoring"
python scripts/evaluate.py --hyp "$PRED_DIR/nllb33b.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/nllb33b.ar-en.dev.json"
python scripts/evaluate.py --hyp "$PRED_DIR/nllb33b.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/nllb33b.en-ar.dev.json"

echo
echo "done. compare to NLLB-1.3B (ar->en 30.79, en->ar 20.29)."
