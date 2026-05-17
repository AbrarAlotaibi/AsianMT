#!/usr/bin/env bash
# Smoke test: verify the NLLB training pipeline end-to-end with ~50 steps.
# Should take 2-5 minutes on a 12 GB GPU. No checkpoint is saved.
#
# What this catches:
#   - missing deps / wrong torch+CUDA build
#   - data path or tokenization bugs
#   - OOM at the configured batch size
#   - language-tag plumbing errors
#
# What this does NOT catch: model quality. Use run_nllb.sh for that.

set -euo pipefail

# Clean data if not already done
if [ ! -f Ar-En/clean/train.ar ]; then
  echo ">>> running scripts/clean.py (one-time)"
  python scripts/clean.py
fi

echo ">>> NLLB smoke test (50 steps)"
python scripts/train_nllb.py \
  --max-steps 50 \
  --batch-size 4 \
  --grad-accum 8 \
  --output-dir checkpoints/_smoke

echo
echo "smoke test OK — proceed with: bash run_nllb.sh"
