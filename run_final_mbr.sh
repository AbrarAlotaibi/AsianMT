#!/usr/bin/env bash
# Final MBR ensemble on the challenge test set, combining all systems that
# proved competitive on dev (BLEU or COMET).
#
# Pool composition (based on dev scores):
#   ar->en: NLLB-3.3B + X-ALMA-FT + X-ALMA-CPO + ALLAM-FT + opus-mt
#           (excludes Claude-API and X-ALMA-filtered as both regress)
#   en->ar: NLLB-3.3B + X-ALMA-FT + X-ALMA-CPO + ALLAM-FT + opus-mt + Claude-API
#           (Claude-API earns its slot here -- 20.45 BLEU / 0.8746 COMET)
#
# Steps:
#   1. Translate challenge test with X-ALMA-CPO (both dirs)  ~20 min
#   2. Translate challenge test with ALLAM-FT  (both dirs)   ~15 min
#   3. COMET-MBR over the full pool                          ~10 min
#   4. Stage submission files + push                          fast
#
# Usage:  CUDA_VISIBLE_DEVICES=0 bash run_final_mbr.sh
# Skip retranslation (reuse cached outputs): SKIP_TRANSLATE=1 ...
# Monitor: tail -f results/final_mbr.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

PRED="predictions"
RES="results"
BACKBONE="haoranxu/X-ALMA-13B-Pretrain"
CPO_ADAPTER="checkpoints/xalma-cpo/best"
ALLAM_MODEL="ALLaM-AI/ALLaM-7B-Instruct-preview"
ALLAM_ENAR="checkpoints/allam-ft-en-ar/best"
ALLAM_AREN="checkpoints/allam-ft-ar-en/best"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/final_mbr.log") 2>&1
echo "=== run_final_mbr $(date -u +%FT%TZ)  GPU=$CUDA_VISIBLE_DEVICES ==="

# Locate test sources (same auto-detect as run_test.sh).
pick() { for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done; echo ""; }
TEST_AR="${TEST_AR:-$(pick \
  Ar-En/test.ar Ar-En/clean/test.ar \
  "Challenge_Test/Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt" \
  Ar-En/devtest.ar)}"
TEST_EN="${TEST_EN:-$(pick \
  Ar-En/test.en Ar-En/clean/test.en \
  "Challenge_Test/Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt" \
  Ar-En/devtest.en)}"
[ -n "$TEST_AR" ] || { echo "ERROR: test ar source not found" >&2; exit 1; }
[ -n "$TEST_EN" ] || { echo "ERROR: test en source not found" >&2; exit 1; }
echo "test ar source: $TEST_AR"
echo "test en source: $TEST_EN"

# --- Step 1: translate test with X-ALMA-CPO ---------------------------------
if [ "${SKIP_TRANSLATE:-0}" != "1" ] && [ -d "$CPO_ADAPTER" ]; then
  if [ ! -f "$PRED/xalma-cpo.ar-en.test.en" ]; then
    echo ">>> [1/4] X-ALMA-CPO ar->en (test)"
    python scripts/xalma_translate.py --backbone "$BACKBONE" \
      --group "$CPO_ADAPTER" --direction ar-en \
      --input "$TEST_AR" --output "$PRED/xalma-cpo.ar-en.test.en"
  fi
  if [ ! -f "$PRED/xalma-cpo.en-ar.test.ar" ]; then
    echo ">>> [1/4] X-ALMA-CPO en->ar (test)"
    python scripts/xalma_translate.py --backbone "$BACKBONE" \
      --group "$CPO_ADAPTER" --direction en-ar \
      --input "$TEST_EN" --output "$PRED/xalma-cpo.en-ar.test.ar"
  fi
fi

# --- Step 2: translate test with ALLAM-FT -----------------------------------
if [ "${SKIP_TRANSLATE:-0}" != "1" ]; then
  if [ -d "$ALLAM_AREN" ] && [ ! -f "$PRED/allam-ft.ar-en.test.en" ]; then
    echo ">>> [2/4] ALLAM-FT ar->en (test, normalized source)"
    # Match training: normalize Arabic source first.
    python scripts/normalize_arabic.py "$TEST_AR" "$PRED/test.norm.ar"
    python scripts/allam_translate.py --model "$ALLAM_MODEL" \
      --adapter "$ALLAM_AREN" --direction ar-en \
      --input "$PRED/test.norm.ar" --output "$PRED/allam-ft.ar-en.test.en"
  fi
  if [ -d "$ALLAM_ENAR" ] && [ ! -f "$PRED/allam-ft.en-ar.test.ar" ]; then
    echo ">>> [2/4] ALLAM-FT en->ar (test)"
    python scripts/allam_translate.py --model "$ALLAM_MODEL" \
      --adapter "$ALLAM_ENAR" --direction en-ar \
      --input "$TEST_EN" --output "$PRED/allam-ft.en-ar.test.ar"
  fi
fi

# --- Step 3: build MBR pools and run COMET-MBR ------------------------------
echo ">>> [3/4] MBR ensemble ar->en"
ar_en=()
for f in \
  "$PRED/nllb33b.ar-en.test.en" \
  "$PRED/xalma-ft.ar-en.test.en" \
  "$PRED/xalma-cpo.ar-en.test.en" \
  "$PRED/allam-ft.ar-en.test.en" \
  "$PRED/opus.ar-en.test.en"; do
  [ -f "$f" ] && ar_en+=("$f")
done
echo "ar->en pool (${#ar_en[@]}):"; printf '  %s\n' "${ar_en[@]}"
python scripts/mbr_decode.py --src "$TEST_AR" --candidates "${ar_en[@]}" \
  --output "$PRED/final-mbr.ar-en.test.en"

echo ">>> [3/4] MBR ensemble en->ar"
en_ar=()
for f in \
  "$PRED/nllb33b.en-ar.test.ar" \
  "$PRED/xalma-ft.en-ar.test.ar" \
  "$PRED/xalma-cpo.en-ar.test.ar" \
  "$PRED/allam-ft.en-ar.test.ar" \
  "$PRED/opus.en-ar.test.ar" \
  "$PRED/claude-api.en-ar.test.ar"; do
  [ -f "$f" ] && en_ar+=("$f")
done
echo "en->ar pool (${#en_ar[@]}):"; printf '  %s\n' "${en_ar[@]}"
python scripts/mbr_decode.py --src "$TEST_EN" --candidates "${en_ar[@]}" \
  --output "$PRED/final-mbr.en-ar.test.ar"

# --- Step 4: stage submission files -----------------------------------------
echo ">>> [4/4] staging submission files"
cp "$PRED/final-mbr.ar-en.test.en" "$RES/submission.ar-en.test.en"
cp "$PRED/final-mbr.en-ar.test.ar" "$RES/submission.en-ar.test.ar"
echo "submission files:"
ls -la "$RES/submission".*

bash push_results.sh "final MBR ensemble (ALLAM + CPO + NLLB + X-ALMA-FT + opus [+ Claude en->ar])" 2>&1 | tail -3 || true
echo "done."
