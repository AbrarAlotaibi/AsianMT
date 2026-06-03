#!/usr/bin/env bash
# Stage the three submission files per direction (primary + 2 contrastive)
# in results/, with clear names, and push.
#
# Run AFTER:
#   bash run_final_mbr.sh         (primary + ALLAM-FT test outputs)
#   bash run_sampling_mbr_test.sh (sampling-MBR test outputs)
#
# Outputs in results/:
#   submission-primary.{ar-en,en-ar}.test.{en,ar}     -- final MBR ensemble
#   submission-contrastive1-allam.{ar-en,en-ar}.test.{en,ar}
#   submission-contrastive2-sampling-mbr.{ar-en,en-ar}.test.{en,ar}

set -euo pipefail
cd "$(dirname "$0")"

PRED="predictions"
RES="results"
mkdir -p "$RES"

stage() {  # stage <src-pred-file> <dst-submission-file>
  if [ -f "$1" ]; then
    cp "$1" "$2"
    echo "  staged $2 ($(wc -l < "$2") lines)"
  else
    echo "  MISSING: $1 -- run the prerequisite script first" >&2
  fi
}

echo "Primary (final MBR ensemble):"
stage "$PRED/final-mbr.ar-en.test.en" "$RES/submission-primary.ar-en.test.en"
stage "$PRED/final-mbr.en-ar.test.ar" "$RES/submission-primary.en-ar.test.ar"

echo "Contrastive 1 (ALLAM-FT):"
stage "$PRED/allam-ft.ar-en.test.en" "$RES/submission-contrastive1-allam.ar-en.test.en"
stage "$PRED/allam-ft.en-ar.test.ar" "$RES/submission-contrastive1-allam.en-ar.test.ar"

echo "Contrastive 2 (Sampling-MBR over NLLB-3.3B):"
stage "$PRED/sampling-mbr.ar-en.test.en" "$RES/submission-contrastive2-sampling-mbr.ar-en.test.en"
stage "$PRED/sampling-mbr.en-ar.test.ar" "$RES/submission-contrastive2-sampling-mbr.en-ar.test.ar"

echo ""
echo "Submission files:"
ls -la "$RES"/submission-*.test.* 2>/dev/null || echo "  (none yet)"

bash push_results.sh "staged 3-system submissions (primary + 2 contrastive)" 2>&1 | tail -3 || true
echo "done."
