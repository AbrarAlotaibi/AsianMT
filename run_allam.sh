#!/usr/bin/env bash
# Fine-tune ALLAM-7B-Instruct for Arabic↔English, translate dev, score.
#
# ALLAM (SDAIA × IBM) is an Arabic-native LLM — strongest for en→ar.
# Fits in ~5 GB NF4 on a single 3090; faster than X-ALMA-13B.
#
# Usage:  CUDA_VISIBLE_DEVICES=2 bash run_allam.sh
# Skip training (eval only): SKIP_TRAIN=1 bash run_allam.sh
# Monitor: tail -f results/allam.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA="Ar-En/clean"
PRED="predictions"
RES="results"
MODEL="${ALLAM_MODEL:-sdaia/allam-1-7b-instruct}"
CKPT_ENAR="checkpoints/allam-ft-en-ar/best"
CKPT_AREN="checkpoints/allam-ft-ar-en/best"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/allam.log") 2>&1
echo "=== run_allam $(date -u +%FT%TZ)  model=$MODEL ==="

# --- normalize Arabic source before fine-tuning and inference ---------------
echo ">>> normalizing Arabic sources"
python scripts/normalize_arabic.py "$DATA/train.ar" "$DATA/train.norm.ar"
python scripts/normalize_arabic.py "$DATA/dev.ar"   "$DATA/dev.norm.ar"

# --- fine-tune en->ar (ALLAM's strongest direction) -------------------------
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo ">>> [1/4] fine-tune en->ar"
  python scripts/allam_finetune.py \
    --model "$MODEL" \
    --src-file "$DATA/train.en" --tgt-file "$DATA/train.ar" \
    --direction en-ar --output-dir checkpoints/allam-ft-en-ar

  echo ">>> [2/4] fine-tune ar->en (normalized Arabic source)"
  python scripts/allam_finetune.py \
    --model "$MODEL" \
    --src-file "$DATA/train.norm.ar" --tgt-file "$DATA/train.en" \
    --direction ar-en --output-dir checkpoints/allam-ft-ar-en
fi

# --- translate dev -----------------------------------------------------------
echo ">>> [3/4] translate dev en->ar"
python scripts/allam_translate.py --model "$MODEL" --adapter "$CKPT_ENAR" \
  --direction en-ar \
  --input "$DATA/dev.en" --output "$PRED/allam-ft.en-ar.dev.ar"
python scripts/evaluate.py \
  --hyp "$PRED/allam-ft.en-ar.dev.ar" --ref "$DATA/dev.ar" \
  --src "$DATA/dev.en" --comet | tee "$RES/allam-ft.en-ar.dev.json"

echo ">>> [4/4] translate dev ar->en (normalized source)"
python scripts/allam_translate.py --model "$MODEL" --adapter "$CKPT_AREN" \
  --direction ar-en \
  --input "$DATA/dev.norm.ar" --output "$PRED/allam-ft.ar-en.dev.en"
python scripts/evaluate.py \
  --hyp "$PRED/allam-ft.ar-en.dev.en" --ref "$DATA/dev.en" \
  --src "$DATA/dev.ar" --comet | tee "$RES/allam-ft.ar-en.dev.json"

bash push_results.sh "ALLAM-7B fine-tuned (ar-en, en-ar)" 2>&1 | tail -3 || true
echo "done."
