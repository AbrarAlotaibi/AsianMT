#!/usr/bin/env bash
# Quality-filtered re-fine-tune: score all training pairs with COMET,
# keep the top KEEP fraction, then re-run X-ALMA and/or NLLB fine-tune.
#
# Research finding: fine-tuning on top-40% quality pairs typically beats
# training on all data when the corpus contains noise (wrong language,
# misaligned segments, low-quality MT outputs mixed with human translations).
#
# Usage:  CUDA_VISIBLE_DEVICES=0 bash run_quality_finetune.sh
# Keep ratio: KEEP=0.5 bash run_quality_finetune.sh  (default 0.4)
# Skip filtering (reuse cached filtered data): SKIP_FILTER=1 ...
# Monitor: tail -f results/quality_finetune.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA="Ar-En/clean"
FDATA="data/filtered"
RES="results"
PRED="predictions"
KEEP="${KEEP:-0.4}"
BACKBONE="haoranxu/X-ALMA-13B-Pretrain"
mkdir -p "$RES" "$PRED" "$FDATA"
exec > >(tee "$RES/quality_finetune.log") 2>&1
echo "=== run_quality_finetune $(date -u +%FT%TZ)  keep=$KEEP ==="

# --- Step 1: score and filter training data ---------------------------------
if [ "${SKIP_FILTER:-0}" != "1" ]; then
  echo ">>> [1/5] scoring + filtering ar->en training pairs (keep=$KEEP)"
  python scripts/filter_training_data.py \
    --src "$DATA/train.ar" --ref "$DATA/train.en" \
    --direction ar-en --keep "$KEEP" \
    --out-src "$FDATA/train.ar" --out-ref "$FDATA/train.en"

  echo ">>> [2/5] scoring + filtering en->ar training pairs (keep=$KEEP)"
  python scripts/filter_training_data.py \
    --src "$DATA/train.en" --ref "$DATA/train.ar" \
    --direction en-ar --keep "$KEEP" \
    --out-src "$FDATA/train.en" --out-ref "$FDATA/train.ar"
else
  echo ">>> skipping filter (SKIP_FILTER=1)"
fi

# --- Step 2: re-fine-tune X-ALMA on filtered data ---------------------------
echo ">>> [3/5] X-ALMA QLoRA fine-tune on filtered data"
python scripts/xalma_finetune.py \
  --backbone "$BACKBONE" \
  --src-file "$FDATA/train.ar" --tgt-file "$FDATA/train.en" \
  --direction ar-en \
  --output-dir checkpoints/xalma-filtered-ft

# --- Step 3: translate dev + score ------------------------------------------
echo ">>> [4/5] translate dev ar->en"
python scripts/xalma_translate.py \
  --backbone "$BACKBONE" \
  --group checkpoints/xalma-filtered-ft/best \
  --direction ar-en \
  --input "$DATA/dev.ar" --output "$PRED/xalma-filtered.ar-en.dev.en"
python scripts/evaluate.py \
  --hyp "$PRED/xalma-filtered.ar-en.dev.en" \
  --ref "$DATA/dev.en" --src "$DATA/dev.ar" --comet \
  | tee "$RES/xalma-filtered.ar-en.dev.json"

echo ">>> [5/5] translate dev en->ar"
python scripts/xalma_translate.py \
  --backbone "$BACKBONE" \
  --group checkpoints/xalma-filtered-ft/best \
  --direction en-ar \
  --input "$DATA/dev.en" --output "$PRED/xalma-filtered.en-ar.dev.ar"
python scripts/evaluate.py \
  --hyp "$PRED/xalma-filtered.en-ar.dev.ar" \
  --ref "$DATA/dev.ar" --src "$DATA/dev.en" --comet \
  | tee "$RES/xalma-filtered.en-ar.dev.json"

bash push_results.sh "quality-filtered X-ALMA fine-tune (keep=$KEEP)" 2>&1 | tail -3 || true
echo "done. compare to baseline xalma-ft: ar->en 31.76 BLEU / 0.8449 COMET"
