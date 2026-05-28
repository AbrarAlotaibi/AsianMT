#!/usr/bin/env bash
# QLoRA fine-tune X-ALMA-13B-Pretrain on Ar<->En official data, then translate
# dev with the fine-tuned adapter and score.
#
# Tuned for a single 24 GB 3090: 4-bit base + fresh r=16 LoRA + paged_adamw_8bit
# + gradient checkpointing. Fits with plenty of headroom (~12 GB used).
#
# Expected wall time: ~10-14 h on a 3090 for 2 epochs (42k examples).
#
# Run on a specific GPU when paralleling other experiments:
#   CUDA_VISIBLE_DEVICES=1 bash run_xalma_finetune.sh

set -euo pipefail
export PYTHONUNBUFFERED=1

DATA_DIR="Ar-En/clean"
BACKBONE="haoranxu/X-ALMA-13B-Pretrain"
CKPT="checkpoints/xalma-ft/best"
PRED_DIR="predictions"
RES_DIR="results"
mkdir -p "$PRED_DIR" "$RES_DIR"
exec > >(tee "$RES_DIR/xalma_finetune.log") 2>&1
echo "=== run_xalma_finetune $(date -u +%FT%TZ) ==="

if [ ! -f "$DATA_DIR/train.ar" ]; then
  python scripts/clean.py
fi

echo ">>> [1/4] QLoRA fine-tuning X-ALMA-13B-Pretrain (both directions)"
python scripts/xalma_finetune.py \
  --backbone "$BACKBONE" \
  --data-dir "$DATA_DIR" \
  --output-dir checkpoints/xalma-ft \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-4

echo ">>> [2/4] translating dev (ar->en)"
python scripts/xalma_translate.py \
  --backbone "$BACKBONE" --group "$CKPT" \
  --direction ar-en \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/xalma-ft.ar-en.dev.en"

echo ">>> [3/4] translating dev (en->ar)"
python scripts/xalma_translate.py \
  --backbone "$BACKBONE" --group "$CKPT" \
  --direction en-ar \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/xalma-ft.en-ar.dev.ar"

echo ">>> [4/4] scoring"
python scripts/evaluate.py --hyp "$PRED_DIR/xalma-ft.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/xalma-ft.ar-en.dev.json"
python scripts/evaluate.py --hyp "$PRED_DIR/xalma-ft.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/xalma-ft.en-ar.dev.json"

echo
echo "done. compare to NLLB-1.3B (ar->en 30.79, en->ar 20.29)"
echo "and X-ALMA zero-shot (ar->en 24.67, en->ar 14.94)."
