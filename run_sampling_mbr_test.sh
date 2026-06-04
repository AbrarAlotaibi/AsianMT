#!/usr/bin/env bash
# Sampling-MBR on the challenge test set: K diverse samples from NLLB-3.3B,
# then COMET-MBR over them. This is contrastive submission #2 -- our best
# COMET on dev (ar->en 0.8524, en->ar 0.8862).
#
# Cost: ~K * seq2seq decode time * 2 directions. With K=32, batch=4 on a
# 3090 this runs ~2-3 hours.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 bash run_sampling_mbr_test.sh
# Smaller pool: K=16 PRUNE_K=8 bash run_sampling_mbr_test.sh
# Monitor: tail -f results/sampling_mbr_test.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

PRED="predictions"
RES="results"
CKPT="${CKPT:-checkpoints/nllb33b-Ar-En/best}"
K="${K:-32}"
BSZ="${BSZ:-4}"            # sampling batch (effective K*BSZ seqs in parallel)
PRUNE_K="${PRUNE_K:-16}"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/sampling_mbr_test.log") 2>&1
echo "=== run_sampling_mbr_test $(date -u +%FT%TZ)  ckpt=$CKPT K=$K prune=$PRUNE_K GPU=$CUDA_VISIBLE_DEVICES ==="

[ -d "$CKPT" ] || { echo "ERROR: $CKPT not found." >&2; exit 1; }

# Auto-detect test files (matches run_test.sh logic).
pick() { for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done; echo ""; }
TEST_AR="${TEST_AR:-$(pick \
  Ar-En/test.ar Ar-En/clean/test.ar \
  "Challenge_Test/Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt" \
  Ar-En/devtest.ar)}"
TEST_EN="${TEST_EN:-$(pick \
  Ar-En/test.en Ar-En/clean/test.en \
  "Challenge_Test/Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt" \
  Ar-En/devtest.en)}"
[ -n "$TEST_AR" ] || { echo "ERROR: test ar source not found" >&2; exit 1; }
[ -n "$TEST_EN" ] || { echo "ERROR: test en source not found" >&2; exit 1; }

# --- Step 1: sample K hypotheses ar->en -------------------------------------
if [ ! -f "$PRED/sampling.ar-en.test.en.sample0" ]; then
  echo ">>> [1/4] sampling ar->en (K=$K)"
  python scripts/translate.py --model "$CKPT" --kind nllb --direction ar-en \
    --dtype fp16 --num-samples "$K" --batch-size "$BSZ" --temperature 0.8 --top-p 0.95 \
    --input "$TEST_AR" --output "$PRED/sampling.ar-en.test.en"
fi

# --- Step 2: sample K hypotheses en->ar -------------------------------------
if [ ! -f "$PRED/sampling.en-ar.test.ar.sample0" ]; then
  echo ">>> [2/4] sampling en->ar (K=$K)"
  python scripts/translate.py --model "$CKPT" --kind nllb --direction en-ar \
    --dtype fp16 --num-samples "$K" --batch-size "$BSZ" --temperature 0.8 --top-p 0.95 \
    --input "$TEST_EN" --output "$PRED/sampling.en-ar.test.ar"
fi

# --- Step 3: COMET-MBR ar->en -----------------------------------------------
echo ">>> [3/4] COMET-MBR over ar->en samples"
python scripts/mbr_decode.py --src "$TEST_AR" \
  --candidates "$PRED"/sampling.ar-en.test.en.sample* \
  --prune-k "$PRUNE_K" --output "$PRED/sampling-mbr.ar-en.test.en"

# --- Step 4: COMET-MBR en->ar -----------------------------------------------
echo ">>> [4/4] COMET-MBR over en->ar samples"
python scripts/mbr_decode.py --src "$TEST_EN" \
  --candidates "$PRED"/sampling.en-ar.test.ar.sample* \
  --prune-k "$PRUNE_K" --output "$PRED/sampling-mbr.en-ar.test.ar"

bash push_results.sh "sampling-MBR on test (K=$K NLLB-3.3B)" 2>&1 | tail -3 || true
echo "done."
