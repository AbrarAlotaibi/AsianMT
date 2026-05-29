#!/usr/bin/env bash
# One-command rerun of just the NLLB-200-3.3B fine-tune (the only track that
# OOM'd; X-ALMA QLoRA and opus-mt already succeeded). Pulls latest code, runs,
# and auto-pushes results/ to main when done. Survives SSH disconnect.
#
# Default: single GPU 0 with the batch=2 + expandable_segments fix (frees GPUs
# 1-2 for run_improvements.sh in parallel).
# Safer: shard across two GPUs (no OOM possible, bigger batch) ->
#   MODEL_PARALLEL=1 bash retry_nllb.sh
#
# Usage:  bash retry_nllb.sh
# Monitor: tail -f results/nllb33b.log

set -uo pipefail
trap '' HUP
cd "$(dirname "$0")"

mkdir -p results
echo "=== retry_nllb $(date -u +%FT%TZ) ===" | tee -a results/run_all.log
# Operate on main so results land there (checkpoints/ are gitignored, so
# switching branches won't touch the trained models).
git checkout main 2>/dev/null || git checkout -b main origin/main
git pull --rebase origin main 2>&1 | tail -2 || true

if [ "${MODEL_PARALLEL:-0}" = "1" ]; then
  # Shard the 3.3B across GPUs 0+1; leaves GPU 2 for other work.
  echo "retry_nllb: model-parallel across GPUs 0,1" | tee -a results/run_all.log
  CUDA_VISIBLE_DEVICES="${MP_GPUS:-0,1}" MODEL_PARALLEL=1 bash run_nllb33b.sh
else
  CUDA_VISIBLE_DEVICES=0 bash run_nllb33b.sh
fi
R=$?
echo "retry_nllb: NLLB-3.3B exit=$R at $(date -u +%FT%TZ)" | tee -a results/run_all.log

bash push_results.sh "rerun NLLB-3.3B (batch=2, expandable_segments)" 2>&1 | tail -3 || true
echo "retry_nllb done." | tee -a results/run_all.log
