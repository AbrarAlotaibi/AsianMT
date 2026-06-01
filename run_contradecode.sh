#!/usr/bin/env bash
# Source/language-contrastive decoding on the best NLLB checkpoint, both
# directions, scored on dev. Targets the en->ar off-target/hallucination gap.
# No training; ~minutes on a GPU.
#
# Uses the NLLB-1.3B fine-tune by default (best en->ar system). Override with
# NLLB_CKPT=... bash run_contradecode.sh
#
# Usage:  bash run_contradecode.sh
# Monitor: tail -f results/contradecode.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA_DIR="Ar-En/clean"
PRED="predictions"
RES="results"
# Auto-pick the best available NLLB checkpoint (3.3B preferred for en->ar,
# 1.3B otherwise). NLLB_CKPT=... overrides.
if [ -n "${NLLB_CKPT:-}" ]; then
  CKPT="$NLLB_CKPT"
elif [ -d checkpoints/nllb33b-Ar-En/best ]; then
  CKPT="checkpoints/nllb33b-Ar-En/best"
elif [ -d checkpoints/nllb-Ar-En/best ]; then
  CKPT="checkpoints/nllb-Ar-En/best"
else
  echo "ERROR: no NLLB checkpoint found." >&2
  exit 1
fi
LAM="${LAM:-0.5}"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/contradecode.log") 2>&1
echo "=== run_contradecode $(date -u +%FT%TZ)  ckpt=$CKPT lam=$LAM ==="

echo ">>> en->ar (language-contrastive, the weak direction)"
python scripts/contradecode.py --model "$CKPT" --direction en-ar \
  --mode language --lam "$LAM" \
  --input "$DATA_DIR/dev.en" --output "$PRED/contra.en-ar.dev.ar"
python scripts/evaluate.py --hyp "$PRED/contra.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  --src "$DATA_DIR/dev.en" --comet | tee "$RES/contra.en-ar.dev.json"

echo ">>> ar->en (language-contrastive)"
python scripts/contradecode.py --model "$CKPT" --direction ar-en \
  --mode language --lam "$LAM" \
  --input "$DATA_DIR/dev.ar" --output "$PRED/contra.ar-en.dev.en"
python scripts/evaluate.py --hyp "$PRED/contra.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  --src "$DATA_DIR/dev.ar" --comet | tee "$RES/contra.ar-en.dev.json"

bash push_results.sh "ContraDecode dev (ckpt=$(basename $(dirname "$CKPT")) lam=$LAM)" 2>&1 | tail -3 || true
echo "done. compare to plain beam: NLLB-1.3B en->ar 20.29, ar->en 30.79 BLEU."
