#!/usr/bin/env bash
# Resume the NLLB-3.3B pipeline from step 2 (translate dev) using the existing
# checkpoint. Skips the 4.7 h training entirely. Targets the OOM-on-generate
# crash that happened with fp32 loading on a single 24 GB card -- now uses fp16
# (~6.6 GB) so the dev pass fits cleanly.
#
# Usage:  bash resume_nllb.sh
# Monitor: tail -f results/nllb33b.log

set -uo pipefail
trap '' HUP
cd "$(dirname "$0")"

source "$(dirname "$0")/scripts/preflight.sh"

if [ ! -d checkpoints/nllb33b-Ar-En/best ]; then
  echo "ERROR: checkpoints/nllb33b-Ar-En/best not found -- training never finished." >&2
  exit 1
fi

mkdir -p results
echo "=== resume_nllb $(date -u +%FT%TZ) ===" | tee -a results/run_all.log

# fp16 inference on a single card is enough; pin to GPU 0 so the other cards
# can run improvements in parallel. (Set MODEL_PARALLEL=1 to shard if needed.)
SKIP_TRAIN=1 CUDA_VISIBLE_DEVICES=0 bash run_nllb33b.sh
R=$?
echo "resume_nllb: NLLB-3.3B inference+score exit=$R at $(date -u +%FT%TZ)" | tee -a results/run_all.log

bash push_results.sh "NLLB-3.3B dev (fp16 inference, resumed from saved checkpoint)" 2>&1 | tail -3 || true
echo "resume_nllb done." | tee -a results/run_all.log
