#!/usr/bin/env bash
# One-shot resume for the sampling-MBR test run after the en->ar OOM.
#
# Preserves the ar->en samples that already finished, deletes any partial
# en->ar samples, and re-runs sampling-MBR with a safer batch (BSZ=2) so
# the K*BSZ KV cache fits in 24 GB on the longer test sequences.
#
# Usage: just run it. The script picks a free GPU if you don't pin one.
#   bash resume_sampling_mbr.sh
# Or pin a specific GPU:
#   CUDA_VISIBLE_DEVICES=1 bash resume_sampling_mbr.sh
# Need even more safety? Smaller K:
#   K=16 PRUNE_K=8 bash resume_sampling_mbr.sh

set -euo pipefail
cd "$(dirname "$0")"

echo ">>> sync repo"
git pull origin main || true

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=1
  echo "CUDA_VISIBLE_DEVICES not set; defaulting to 1"
fi
export K="${K:-32}"
export BSZ="${BSZ:-2}"
export PRUNE_K="${PRUNE_K:-16}"

echo ">>> state of sampling files before cleanup"
shopt -s nullglob
ar_files=(predictions/sampling.ar-en.test.en.sample*)
en_files=(predictions/sampling.en-ar.test.ar.sample*)
echo "ar->en samples present: ${#ar_files[@]}"
echo "en->ar samples present (will be deleted): ${#en_files[@]}"

echo ">>> deleting partial en->ar samples (ar->en preserved)"
rm -f predictions/sampling.en-ar.test.ar.sample*

echo ">>> running sampling-MBR (K=$K BSZ=$BSZ PRUNE_K=$PRUNE_K GPU=$CUDA_VISIBLE_DEVICES)"
bash run_sampling_mbr_test.sh
