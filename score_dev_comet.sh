#!/usr/bin/env bash
# Re-score every existing dev prediction with BLEU + chrF + TER + COMET-22.
# Cheap (COMET-22 already cached after run_mbr.sh). Writes/overwrites
# results/<system>.<direction>.dev.json with the COMET field added.
#
# Essential for the paper: we currently have BLEU/chrF/TER for single systems
# but only MBR has COMET. This pass gives apples-to-apples numbers.
#
# Usage:  bash score_dev_comet.sh
# Monitor: tail -f results/score_dev_comet.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA="Ar-En/clean"
PRED="predictions"
RES="results"
mkdir -p "$RES"
exec > >(tee "$RES/score_dev_comet.log") 2>&1
echo "=== score_dev_comet $(date -u +%FT%TZ) ==="

shopt -s nullglob
for pred in "$PRED"/*.ar-en.dev.en "$PRED"/*.en-ar.dev.ar; do
  base=$(basename "$pred")
  system=$(echo "$base" | cut -d. -f1)
  direction=$(echo "$base" | cut -d. -f2)
  src_lang=${direction%-*}
  tgt_lang=${direction#*-}
  src_file="$DATA/dev.$src_lang"
  ref_file="$DATA/dev.$tgt_lang"
  result_file="$RES/$system.$direction.dev.json"

  if [ ! -f "$src_file" ] || [ ! -f "$ref_file" ]; then
    echo "skip $base (missing src/ref)"; continue
  fi
  echo ">>> $system $direction"
  python scripts/evaluate.py --hyp "$pred" --ref "$ref_file" --src "$src_file" --comet \
    | tee "$result_file"
done

bash push_results.sh "COMET re-score all dev systems" 2>&1 | tail -3 || true
echo "score_dev_comet done."
