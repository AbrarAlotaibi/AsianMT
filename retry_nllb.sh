#!/usr/bin/env bash
# One-command rerun of just the NLLB-200-3.3B fine-tune on GPU 0 (the only track
# that OOM'd; X-ALMA QLoRA and opus-mt already succeeded). Pulls latest code,
# runs, and auto-pushes results/ when done. Survives SSH disconnect.
#
# Usage:  bash retry_nllb.sh
# Monitor: tail -f results/nllb33b.log

set -uo pipefail
trap '' HUP
cd "$(dirname "$0")"

mkdir -p results
echo "=== retry_nllb $(date -u +%FT%TZ) ===" | tee -a results/run_all.log
git pull --rebase origin main 2>&1 | tail -2 || true

CUDA_VISIBLE_DEVICES=0 bash run_nllb33b.sh
R=$?
echo "retry_nllb: NLLB-3.3B exit=$R at $(date -u +%FT%TZ)" | tee -a results/run_all.log

bash push_results.sh "rerun NLLB-3.3B (batch=2, expandable_segments)" 2>&1 | tail -3 || true
echo "retry_nllb done." | tee -a results/run_all.log
