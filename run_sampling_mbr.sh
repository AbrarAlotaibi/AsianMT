#!/usr/bin/env bash
# Sampling-MBR: generate K diverse sampled hypotheses from the strongest NLLB
# checkpoint, then COMET-MBR over the samples. Usually delivers a bigger gain
# than system-combination MBR because the candidate pool is genuinely diverse.
#
# Defaults: K=32 samples, QE pre-prune to top 16, NLLB-3.3B model. Same model
# can be reused (no expensive download), so this runs on a single GPU and is
# cache-friendly.
#
# Usage:        bash run_sampling_mbr.sh
# Override K:   K=64 bash run_sampling_mbr.sh
# Other model:  CKPT=checkpoints/nllb-Ar-En/best bash run_sampling_mbr.sh
# Monitor:      tail -f results/sampling_mbr.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA="Ar-En/clean"
PRED="predictions"
RES="results"
CKPT="${CKPT:-checkpoints/nllb33b-Ar-En/best}"
K="${K:-32}"
PRUNE_K="${PRUNE_K:-16}"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/sampling_mbr.log") 2>&1
echo "=== run_sampling_mbr $(date -u +%FT%TZ)  ckpt=$CKPT K=$K prune=$PRUNE_K ==="

if [ ! -d "$CKPT" ]; then
  echo "ERROR: $CKPT not found." >&2; exit 1
fi

# Sampling needs more KV cache than beam search; smaller batch keeps it safe.
echo ">>> [1/4] sampling ar->en (K=$K)"
python scripts/translate.py --model "$CKPT" --kind nllb --direction ar-en \
  --dtype fp16 --num-samples "$K" --batch-size 4 --temperature 0.8 --top-p 0.95 \
  --input "$DATA/dev.ar" --output "$PRED/sampling.ar-en.dev.en"

echo ">>> [2/4] sampling en->ar (K=$K)"
python scripts/translate.py --model "$CKPT" --kind nllb --direction en-ar \
  --dtype fp16 --num-samples "$K" --batch-size 4 --temperature 0.8 --top-p 0.95 \
  --input "$DATA/dev.en" --output "$PRED/sampling.en-ar.dev.ar"

echo ">>> [3/4] COMET-MBR over ar->en samples"
python scripts/mbr_decode.py --src "$DATA/dev.ar" \
  --candidates "$PRED"/sampling.ar-en.dev.en.sample* \
  --prune-k "$PRUNE_K" --output "$PRED/sampling-mbr.ar-en.dev.en"
python scripts/evaluate.py --hyp "$PRED/sampling-mbr.ar-en.dev.en" \
  --ref "$DATA/dev.en" --src "$DATA/dev.ar" --comet \
  | tee "$RES/sampling-mbr.ar-en.dev.json"

echo ">>> [4/4] COMET-MBR over en->ar samples"
python scripts/mbr_decode.py --src "$DATA/dev.en" \
  --candidates "$PRED"/sampling.en-ar.dev.ar.sample* \
  --prune-k "$PRUNE_K" --output "$PRED/sampling-mbr.en-ar.dev.ar"
python scripts/evaluate.py --hyp "$PRED/sampling-mbr.en-ar.dev.ar" \
  --ref "$DATA/dev.ar" --src "$DATA/dev.en" --comet \
  | tee "$RES/sampling-mbr.en-ar.dev.json"

bash push_results.sh "sampling-MBR over $K NLLB-3.3B samples" 2>&1 | tail -3 || true
echo "run_sampling_mbr done."
