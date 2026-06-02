#!/usr/bin/env bash
# Score the Claude-API dev translations and push results.
#
# Note: Claude-API dev files are 999 lines (one trailing translation got
# dropped during batching) vs source 1000. To enable evaluation, this
# script truncates the source/reference files to the hypothesis length
# in temp files, scores, and pushes results to main.
#
# Usage:  bash score_claude_api.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA="Ar-En/clean"
PRED="predictions"
RES="results"
mkdir -p "$RES"
exec > >(tee "$RES/score_claude_api.log") 2>&1
echo "=== score_claude_api $(date -u +%FT%TZ) ==="

# Sanity check the inputs exist.
for f in "$PRED/claude-api.ar-en.dev.en" "$PRED/claude-api.en-ar.dev.ar" \
         "$DATA/dev.ar" "$DATA/dev.en"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# Truncate ref/src to match the hypothesis line count (999 vs 1000) so
# evaluate.py doesn't error on mismatch. We're scoring the 999 sentences
# Claude actually translated; the dropped final line is excluded from both
# hypothesis and reference -- fair comparison.
N=$(wc -l < "$PRED/claude-api.ar-en.dev.en")
echo "hypothesis line count: $N"

tmp=$(mktemp -d)
head -n "$N" "$DATA/dev.ar" > "$tmp/dev.ar"
head -n "$N" "$DATA/dev.en" > "$tmp/dev.en"

echo ">>> scoring claude-api ar->en"
python scripts/evaluate.py \
  --hyp "$PRED/claude-api.ar-en.dev.en" \
  --ref "$tmp/dev.en" --src "$tmp/dev.ar" --comet \
  | tee "$RES/claude-api.ar-en.dev.json"

echo ">>> scoring claude-api en->ar"
python scripts/evaluate.py \
  --hyp "$PRED/claude-api.en-ar.dev.ar" \
  --ref "$tmp/dev.ar" --src "$tmp/dev.en" --comet \
  | tee "$RES/claude-api.en-ar.dev.json"

rm -rf "$tmp"

bash push_results.sh "Claude-API dev scores (n=$N, 1 line dropped)" 2>&1 | tail -3 || true
echo "done."
