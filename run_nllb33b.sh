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
#
# If a single 24 GB card still OOMs, shard the model across GPUs (no memory
# ceiling) by exposing 2+ GPUs and setting MODEL_PARALLEL=1:
#   CUDA_VISIBLE_DEVICES=0,1 MODEL_PARALLEL=1 bash run_nllb33b.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
# Full FT of 3.3B leaves only ~3.7 GB for activations on a 24 GB card; batch=4
# OOM'd by 32 MB. expandable_segments removes the fragmentation waste the error
# flagged, and batch=2/grad_accum=16 halves activation memory (same effective
# batch 32). Falls within budget with ~1.5 GB headroom.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: spread layers across all visible GPUs to eliminate OOM risk.
# With the model sharded there's room for a bigger micro-batch.
MP_ARGS=""
BATCH=2; ACCUM=16
if [ "${MODEL_PARALLEL:-0}" = "1" ]; then
  MP_ARGS="--model-parallel"
  BATCH="${BATCH_OVERRIDE:-6}"; ACCUM="${ACCUM_OVERRIDE:-5}"
fi

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

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo ">>> [1/4] SKIP_TRAIN=1 -- skipping training, using existing $CKPT"
  if [ ! -d "$CKPT" ]; then
    echo "ERROR: SKIP_TRAIN=1 but $CKPT doesn't exist." >&2
    exit 1
  fi
else
  echo ">>> [1/4] fine-tuning NLLB-200-3.3B (both directions)"
  python scripts/train_nllb.py \
    --model facebook/nllb-200-3.3B \
    --output-dir checkpoints/nllb33b-Ar-En \
    --epochs 2 \
    --batch-size "$BATCH" \
    --grad-accum "$ACCUM" \
    --lr 3e-5 \
    --optim adamw_bnb_8bit $MP_ARGS
fi

# 3.3B inference needs fp16 (~6.6 GB) instead of fp32 (~13 GB) to fit a 24 GB
# card with KV cache. --model-parallel kept as a safety net.
echo ">>> [2/4] translating dev (ar->en)"
python scripts/translate.py --model "$CKPT" --kind nllb --direction ar-en \
  --dtype fp16 \
  --input "$DATA_DIR/dev.ar" --output "$PRED_DIR/nllb33b.ar-en.dev.en"

echo ">>> [3/4] translating dev (en->ar)"
python scripts/translate.py --model "$CKPT" --kind nllb --direction en-ar \
  --dtype fp16 \
  --input "$DATA_DIR/dev.en" --output "$PRED_DIR/nllb33b.en-ar.dev.ar"

echo ">>> [4/4] scoring"
python scripts/evaluate.py --hyp "$PRED_DIR/nllb33b.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  | tee "$RES_DIR/nllb33b.ar-en.dev.json"
python scripts/evaluate.py --hyp "$PRED_DIR/nllb33b.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  | tee "$RES_DIR/nllb33b.en-ar.dev.json"

echo
echo "done. compare to NLLB-1.3B (ar->en 30.79, en->ar 20.29)."
