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

# Verify line counts match using Python's splitlines (wc -l misses lines
# without trailing newline; the claude-api files happen to lack one).
python3 -c "
import sys
pairs = [
    ('predictions/claude-api.ar-en.dev.en', 'Ar-En/clean/dev.en'),
    ('predictions/claude-api.en-ar.dev.ar', 'Ar-En/clean/dev.ar'),
]
for hyp, ref in pairs:
    h = open(hyp, encoding='utf-8').read().splitlines()
    r = open(ref, encoding='utf-8').read().splitlines()
    print(f'{hyp}: hyp={len(h)} ref={len(r)}')
    if len(h) != len(r):
        sys.exit(f'ERROR: line mismatch {hyp}')
"

echo ">>> scoring claude-api ar->en"
python scripts/evaluate.py \
  --hyp "$PRED/claude-api.ar-en.dev.en" \
  --ref "$DATA/dev.en" --src "$DATA/dev.ar" --comet \
  | tee "$RES/claude-api.ar-en.dev.json"

echo ">>> scoring claude-api en->ar"
python scripts/evaluate.py \
  --hyp "$PRED/claude-api.en-ar.dev.ar" \
  --ref "$DATA/dev.ar" --src "$DATA/dev.en" --comet \
  | tee "$RES/claude-api.en-ar.dev.json"

bash push_results.sh "Claude-API dev scores" 2>&1 | tail -3 || true
echo "done."
