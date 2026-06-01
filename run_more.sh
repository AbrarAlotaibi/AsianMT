#!/usr/bin/env bash
# One-shot orchestrator that fires every queued improvement in parallel:
#
#   GPU 0:  score_dev_comet.sh  (~5 min, models already cached)
#           -> run_sampling_mbr.sh on NLLB-3.3B (~30-60 min)
#           -> run_test.sh test-set translation (depends on test file)
#   GPU 1:  run_cpo.sh           (~6-10 h, longest job)
#   GPU 2:  run_contradecode.sh  (~10 min)
#
# Each script auto-pushes results to main on success. Survives SSH disconnect.
#
# Usage:  bash run_more.sh
# Skip a track: SKIP_CPO=1 / SKIP_TEST=1 / SKIP_CONTRA=1 / SKIP_SAMPLING_MBR=1
# Monitor: tail -f results/*.log

set -uo pipefail
trap '' HUP
cd "$(dirname "$0")"

source "$(dirname "$0")/scripts/preflight.sh"

mkdir -p results
{
  echo "=== run_more $(date -u +%FT%TZ) ==="
  echo "GPUs: $(python -c 'import torch;print(torch.cuda.device_count())')"
} | tee -a results/run_all.log

# GPU 0 chain: cheap COMET re-score, then sampling-MBR, then test-set translation.
(
  if [ "${SKIP_COMET:-0}" != "1" ]; then
    echo "[GPU0] score_dev_comet"; CUDA_VISIBLE_DEVICES=0 bash score_dev_comet.sh
  fi
  if [ "${SKIP_SAMPLING_MBR:-0}" != "1" ]; then
    echo "[GPU0] sampling_mbr"; CUDA_VISIBLE_DEVICES=0 bash run_sampling_mbr.sh
  fi
  if [ "${SKIP_TEST:-0}" != "1" ]; then
    echo "[GPU0] run_test"; CUDA_VISIBLE_DEVICES=0 bash run_test.sh
  fi
) & P0=$!

# GPU 1: CPO (longest single job).
if [ "${SKIP_CPO:-0}" != "1" ]; then
  ( CUDA_VISIBLE_DEVICES=1 bash run_cpo.sh ) & P1=$!
else
  P1=""
fi

# GPU 2: ContraDecode (fast).
if [ "${SKIP_CONTRA:-0}" != "1" ]; then
  ( CUDA_VISIBLE_DEVICES=2 bash run_contradecode.sh ) & P2=$!
else
  P2=""
fi

{
  echo "launched:"
  echo "  GPU 0 chain (COMET-rescore -> sampling-MBR -> test) pid=$P0"
  [ -n "${P1:-}" ] && echo "  GPU 1 CPO                                       pid=$P1"
  [ -n "${P2:-}" ] && echo "  GPU 2 ContraDecode                              pid=$P2"
  echo "monitor: tail -f results/*.log"
} | tee -a results/run_all.log

wait "$P0"; R0=$?
R1=0; R2=0
[ -n "${P1:-}" ] && { wait "$P1"; R1=$?; }
[ -n "${P2:-}" ] && { wait "$P2"; R2=$?; }
echo "run_more finished: GPU0 chain=$R0 CPO=$R1 Contra=$R2 at $(date -u +%FT%TZ)" | tee -a results/run_all.log

# Final push (each sub-script also pushes; this catches anything they missed).
bash push_results.sh "run_more wrap-up" 2>&1 | tail -3 || true
echo "run_more done." | tee -a results/run_all.log
