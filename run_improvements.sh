#!/usr/bin/env bash
# Fill GPUs 1 and 2 with the post-training improvements while NLLB-3.3B reruns
# on GPU 0 (retry_nllb.sh). Pulls main, runs both tracks in parallel, then
# auto-pushes results to main. Survives SSH disconnect.
#
#   GPU 1: CPO (ALMA-R) preference fine-tune of X-ALMA   (run_cpo.sh)
#   GPU 2: source/language-contrastive decoding on NLLB  (run_contradecode.sh)
#
# COMET-MBR (run_mbr.sh) is intentionally NOT here: it combines all systems'
# dev outputs and is best run once the NLLB-3.3B rerun has produced its
# predictions. It's cheap (no training) -- run it after.
#
# Usage:  bash run_improvements.sh
# Monitor: tail -f results/cpo.log results/contradecode.log

set -uo pipefail
trap '' HUP
cd "$(dirname "$0")"

mkdir -p results
echo "=== run_improvements $(date -u +%FT%TZ) ===" | tee -a results/run_all.log
git checkout main 2>/dev/null || git checkout -b main origin/main
git pull --rebase origin main 2>&1 | tail -2 || true

CUDA_VISIBLE_DEVICES=1 bash run_cpo.sh          & P1=$!
CUDA_VISIBLE_DEVICES=2 bash run_contradecode.sh & P2=$!
echo "launched: CPO pid=$P1 (GPU1), ContraDecode pid=$P2 (GPU2)" | tee -a results/run_all.log

wait "$P1"; R1=$?
wait "$P2"; R2=$?
echo "run_improvements finished: CPO exit=$R1  ContraDecode exit=$R2" | tee -a results/run_all.log

bash push_results.sh "improvements: CPO + ContraDecode" 2>&1 | tail -3 || true
echo "run_improvements done." | tee -a results/run_all.log
