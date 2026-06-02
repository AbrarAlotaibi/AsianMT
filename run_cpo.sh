#!/usr/bin/env bash
# Full CPO (ALMA-R) pipeline on top of our X-ALMA SFT adapter:
#   1. sample K candidates per train source (both directions) from xalma-ft
#   2. build preference triples (chosen/rejected) scored by COMET-Kiwi vs gold
#   3. CPO fine-tune the SFT LoRA on those triples
#   4. translate dev with the CPO adapter and score (incl. COMET)
#
# Expected wall time: ~6-10 h on a 3090 (sampling dominates).
#
# Run on its own GPU:  CUDA_VISIBLE_DEVICES=1 bash run_cpo.sh
# Monitor: tail -f results/cpo.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA_DIR="Ar-En/clean"
PRED="predictions"
RES="results"
BACKBONE="haoranxu/X-ALMA-13B-Pretrain"
SFT="checkpoints/xalma-ft/best"
CPO="checkpoints/xalma-cpo/best"
K="${K:-4}"            # samples per source
N="${N:-4000}"         # train pairs per direction used for preferences
mkdir -p "$RES" "$PRED" data
exec > >(tee "$RES/cpo.log") 2>&1
echo "=== run_cpo $(date -u +%FT%TZ)  K=$K N=$N ==="

if [ ! -d "$SFT" ]; then
  echo "ERROR: SFT adapter $SFT not found; run run_xalma_finetune.sh first." >&2
  exit 1
fi

# Sampling K candidates multiplies the effective batch by K (num_return_sequences),
# so the per-call batch must be small or the 13B + KV cache overflows 24 GB.
SAMP_BS="${SAMP_BS:-2}"
if [ "${SKIP_SAMPLE:-0}" != "1" ]; then
  echo ">>> [1/6] sampling candidates ar->en (batch=$SAMP_BS x K=$K)"
  python scripts/xalma_translate.py --backbone "$BACKBONE" --group "$SFT" \
    --direction ar-en --num-samples "$K" --limit "$N" --batch-size "$SAMP_BS" \
    --input "$DATA_DIR/train.ar" --output "$PRED/cand.ar-en.train.en"

  echo ">>> [2/6] sampling candidates en->ar (batch=$SAMP_BS x K=$K)"
  python scripts/xalma_translate.py --backbone "$BACKBONE" --group "$SFT" \
    --direction en-ar --num-samples "$K" --limit "$N" --batch-size "$SAMP_BS" \
    --input "$DATA_DIR/train.en" --output "$PRED/cand.en-ar.train.ar"
else
  echo ">>> [1-2/6] SKIP_SAMPLE=1, reusing existing $PRED/cand.*.train.* samples"
fi

if [ "${SKIP_PREF:-0}" != "1" ]; then
  echo ">>> [3/6] building preference data"
  python scripts/build_preference_data.py --direction ar-en --max-pairs "$N" \
    --src "$DATA_DIR/train.ar" --ref "$DATA_DIR/train.en" --tokenizer "$SFT" \
    --samples "$PRED"/cand.ar-en.train.en.sample* --out data/pref.ar-en.jsonl
  python scripts/build_preference_data.py --direction en-ar --max-pairs "$N" \
    --src "$DATA_DIR/train.en" --ref "$DATA_DIR/train.ar" --tokenizer "$SFT" \
    --samples "$PRED"/cand.en-ar.train.ar.sample* --out data/pref.en-ar.jsonl
else
  echo ">>> [3/6] SKIP_PREF=1, reusing existing data/pref.*.jsonl"
fi

echo ">>> [4/6] CPO fine-tune"
python scripts/cpo_finetune.py --backbone "$BACKBONE" --sft-adapter "$SFT" \
  --data data/pref.ar-en.jsonl data/pref.en-ar.jsonl \
  --output-dir checkpoints/xalma-cpo

echo ">>> [5/6] translate dev with CPO adapter"
python scripts/xalma_translate.py --backbone "$BACKBONE" --group "$CPO" \
  --direction ar-en --input "$DATA_DIR/dev.ar" --output "$PRED/xalma-cpo.ar-en.dev.en"
python scripts/xalma_translate.py --backbone "$BACKBONE" --group "$CPO" \
  --direction en-ar --input "$DATA_DIR/dev.en" --output "$PRED/xalma-cpo.en-ar.dev.ar"

echo ">>> [6/6] scoring"
python scripts/evaluate.py --hyp "$PRED/xalma-cpo.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
  --src "$DATA_DIR/dev.ar" --comet | tee "$RES/xalma-cpo.ar-en.dev.json"
python scripts/evaluate.py --hyp "$PRED/xalma-cpo.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
  --src "$DATA_DIR/dev.en" --comet | tee "$RES/xalma-cpo.en-ar.dev.json"

bash push_results.sh "CPO (ALMA-R) on X-ALMA-FT (K=$K N=$N)" 2>&1 | tail -3 || true
echo "done. compare to X-ALMA SFT (ar->en 31.76, en->ar 17.38)."
