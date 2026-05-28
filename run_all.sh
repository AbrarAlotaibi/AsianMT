#!/usr/bin/env bash
# One-shot launcher for the three parallel training tracks (one per 3090):
#   GPU 0: NLLB-200-3.3B fine-tune          (run_nllb33b.sh)
#   GPU 1: X-ALMA-13B QLoRA fine-tune       (run_xalma_finetune.sh)
#   GPU 2: opus-mt contrastive (both dirs)  (run_opus.sh)
#
# When all three finish (or error), commits & pushes results/ to the repo.
#
# Survives SSH disconnect (HUP is ignored, so children outlive the shell).
#
# Usage:  bash run_all.sh
# Monitor: tail -f results/*.log
# Status:  pidof python  or  nvidia-smi

set -uo pipefail
trap '' HUP

cd "$(dirname "$0")"

mkdir -p results
LOG="results/run_all.log"
echo "=== run_all $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# Pull latest code (non-fatal if it fails or there are local changes).
git pull --rebase origin main 2>&1 | tee -a "$LOG" || true

CUDA_VISIBLE_DEVICES=0 bash run_nllb33b.sh        &
P1=$!
CUDA_VISIBLE_DEVICES=1 bash run_xalma_finetune.sh &
P2=$!
CUDA_VISIBLE_DEVICES=2 bash run_opus.sh           &
P3=$!

{
  echo "launched:"
  echo "  GPU 0  NLLB-3.3B FT       pid=$P1   log=results/nllb33b.log"
  echo "  GPU 1  X-ALMA QLoRA FT    pid=$P2   log=results/xalma_finetune.log"
  echo "  GPU 2  opus-mt contrastive pid=$P3  log=results/opus.log"
  echo "monitor: tail -f results/*.log"
} | tee -a "$LOG"

# Wait for each, capture exit code, keep going on failure so a single bad run
# doesn't stop the others' results from being pushed.
wait "$P1"; R1=$?
wait "$P2"; R2=$?
wait "$P3"; R3=$?

{
  echo "all finished at $(date -u +%FT%TZ)"
  echo "  NLLB-3.3B exit=$R1"
  echo "  X-ALMA FT exit=$R2"
  echo "  opus-mt  exit=$R3"
} | tee -a "$LOG"

bash push_results.sh "parallel: NLLB-3.3B + X-ALMA QLoRA + opus-mt" 2>&1 | tee -a "$LOG" || true

echo "run_all done." | tee -a "$LOG"
