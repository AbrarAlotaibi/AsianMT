#!/usr/bin/env bash
# One-command rerun of just the NLLB-200-3.3B fine-tune (the only track that
# OOM'd; X-ALMA QLoRA and opus-mt already succeeded). Pulls latest code, runs,
# and auto-pushes results/ to main when done. Survives SSH disconnect.
#
# The 3.3B full fine-tune does NOT fit on one 24 GB card (OOMs even at batch=2),
# so this shards the model across all visible GPUs by DEFAULT. With 3x 3090 it
# spreads over all three and trains comfortably.
#
# To pin to specific GPUs:   MP_GPUS=0,1 bash retry_nllb.sh
# To force single-GPU (will OOM on 3.3B; only for smaller models):
#                            MODEL_PARALLEL=0 bash retry_nllb.sh
#
# Usage:  bash retry_nllb.sh
# Monitor: tail -f results/nllb33b.log

set -uo pipefail
trap '' HUP
cd "$(dirname "$0")"

# Activate asianmt env + sync to main (after pulling, so our own log writes
# below don't block the rebase).
source "$(dirname "$0")/scripts/preflight.sh"

mkdir -p results
echo "=== retry_nllb $(date -u +%FT%TZ) ===" | tee -a results/run_all.log

# Model-parallel ON by default (the 3.3B won't fit on one card). Set
# MODEL_PARALLEL=0 to force single-GPU. MP_GPUS picks which GPUs to shard over
# (default: all visible).
if [ "${MODEL_PARALLEL:-1}" != "0" ]; then
  GPUS="${MP_GPUS:-$(python -c 'import torch;print(",".join(str(i) for i in range(torch.cuda.device_count())))')}"
  echo "retry_nllb: model-parallel across GPUs $GPUS" | tee -a results/run_all.log
  CUDA_VISIBLE_DEVICES="$GPUS" MODEL_PARALLEL=1 bash run_nllb33b.sh
else
  echo "retry_nllb: single-GPU (GPU 0) -- 3.3B will likely OOM" | tee -a results/run_all.log
  CUDA_VISIBLE_DEVICES=0 bash run_nllb33b.sh
fi
R=$?
echo "retry_nllb: NLLB-3.3B exit=$R at $(date -u +%FT%TZ)" | tee -a results/run_all.log

bash push_results.sh "rerun NLLB-3.3B (batch=2, expandable_segments)" 2>&1 | tail -3 || true
echo "retry_nllb done." | tee -a results/run_all.log
