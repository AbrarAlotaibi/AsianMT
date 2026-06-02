#!/usr/bin/env bash
# Produce submission outputs for the released test set with every available
# system, then MBR-ensemble them per direction. Final outputs:
#   predictions/mbr.ar-en.test.en    <- submit this for ar->en
#   predictions/mbr.en-ar.test.ar    <- submit this for en->ar
# Plus the per-system files for ablation tables in the paper.
#
# Test set location (first that exists, can be overridden):
#   $TEST_AR / $TEST_EN        explicit (env vars)
#   Ar-En/test.{ar,en}         newly released test
#   Ar-En/clean/test.{ar,en}   if you've cleaned the test
#   Ar-En/devtest.{ar,en}      fallback (held-out from start)
#
# Usage:                bash run_test.sh
# Custom test path:     TEST_AR=path/to/test.ar TEST_EN=path/to/test.en bash run_test.sh
# Skip MBR (just per-system outputs): SKIP_MBR=1 bash run_test.sh
# Monitor:              tail -f results/run_test.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

PRED="predictions"
RES="results"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/run_test.log") 2>&1
echo "=== run_test $(date -u +%FT%TZ) ==="

# Locate test sources (auto-detect with overrides).
pick() {  # pick FIRST_EXISTING fallback1 fallback2 ...
  for c in "$@"; do [ -f "$c" ] && { echo "$c"; return; }; done
  echo ""
}
TEST_AR="${TEST_AR:-$(pick \
  Ar-En/test.ar Ar-En/clean/test.ar \
  "Challenge_Test/Sub-Task 2A_Arabic-to-English/Ar-to-En/test_challenge_ar_ar-to-en.txt" \
  Ar-En/devtest.ar Ar-En/clean/devtest.ar)}"
TEST_EN="${TEST_EN:-$(pick \
  Ar-En/test.en Ar-En/clean/test.en \
  "Challenge_Test/Sub-Task 1A_ English-to- Arabic/En-to-Ar/test_challenge_en_en-to-ar.txt" \
  Ar-En/devtest.en Ar-En/clean/devtest.en)}"
if [ -z "$TEST_AR" ] || [ -z "$TEST_EN" ]; then
  echo "ERROR: no test source files found. Set TEST_AR=... TEST_EN=... ." >&2
  exit 1
fi
n_ar=$(wc -l < "$TEST_AR")
n_en=$(wc -l < "$TEST_EN")
echo "test ar source: $TEST_AR ($n_ar lines)"
echo "test en source: $TEST_EN ($n_en lines)"

ar_en=()  # candidate files for MBR ar->en
en_ar=()

# --- NLLB-3.3B ----------------------------------------------------------------
if [ -d checkpoints/nllb33b-Ar-En/best ]; then
  echo ">>> NLLB-3.3B ar->en"
  python scripts/translate.py --model checkpoints/nllb33b-Ar-En/best --kind nllb \
    --direction ar-en --dtype fp16 \
    --input "$TEST_AR" --output "$PRED/nllb33b.ar-en.test.en"
  ar_en+=("$PRED/nllb33b.ar-en.test.en")
  echo ">>> NLLB-3.3B en->ar"
  python scripts/translate.py --model checkpoints/nllb33b-Ar-En/best --kind nllb \
    --direction en-ar --dtype fp16 \
    --input "$TEST_EN" --output "$PRED/nllb33b.en-ar.test.ar"
  en_ar+=("$PRED/nllb33b.en-ar.test.ar")
fi

# --- NLLB-1.3B ----------------------------------------------------------------
if [ -d checkpoints/nllb-Ar-En/best ]; then
  echo ">>> NLLB-1.3B ar->en"
  python scripts/translate.py --model checkpoints/nllb-Ar-En/best --kind nllb \
    --direction ar-en --dtype fp16 \
    --input "$TEST_AR" --output "$PRED/nllb.ar-en.test.en"
  ar_en+=("$PRED/nllb.ar-en.test.en")
  echo ">>> NLLB-1.3B en->ar"
  python scripts/translate.py --model checkpoints/nllb-Ar-En/best --kind nllb \
    --direction en-ar --dtype fp16 \
    --input "$TEST_EN" --output "$PRED/nllb.en-ar.test.ar"
  en_ar+=("$PRED/nllb.en-ar.test.ar")
fi

# --- opus-mt ------------------------------------------------------------------
for d in ar-en en-ar; do
  ckpt="checkpoints/opus-Ar-En-${d}/best"
  src_lang=${d%-*}; tgt_lang=${d#*-}
  inp="$TEST_AR"; [ "$src_lang" = "en" ] && inp="$TEST_EN"
  out="$PRED/opus.${d}.test.${tgt_lang}"
  if [ -d "$ckpt" ]; then
    echo ">>> opus-mt $d"
    python scripts/translate.py --model "$ckpt" --kind opus --direction "$d" \
      --input "$inp" --output "$out"
    if [ "$d" = "ar-en" ]; then ar_en+=("$out"); else en_ar+=("$out"); fi
  fi
done

# --- X-ALMA fine-tuned --------------------------------------------------------
if [ -d checkpoints/xalma-ft/best ]; then
  echo ">>> X-ALMA-FT ar->en"
  python scripts/xalma_translate.py --backbone haoranxu/X-ALMA-13B-Pretrain \
    --group checkpoints/xalma-ft/best --direction ar-en \
    --input "$TEST_AR" --output "$PRED/xalma-ft.ar-en.test.en"
  ar_en+=("$PRED/xalma-ft.ar-en.test.en")
  echo ">>> X-ALMA-FT en->ar"
  python scripts/xalma_translate.py --backbone haoranxu/X-ALMA-13B-Pretrain \
    --group checkpoints/xalma-ft/best --direction en-ar \
    --input "$TEST_EN" --output "$PRED/xalma-ft.en-ar.test.ar"
  en_ar+=("$PRED/xalma-ft.en-ar.test.ar")
fi

# --- X-ALMA after CPO ---------------------------------------------------------
if [ -d checkpoints/xalma-cpo/best ]; then
  echo ">>> X-ALMA-CPO ar->en"
  python scripts/xalma_translate.py --backbone haoranxu/X-ALMA-13B-Pretrain \
    --group checkpoints/xalma-cpo/best --direction ar-en \
    --input "$TEST_AR" --output "$PRED/xalma-cpo.ar-en.test.en"
  ar_en+=("$PRED/xalma-cpo.ar-en.test.en")
  echo ">>> X-ALMA-CPO en->ar"
  python scripts/xalma_translate.py --backbone haoranxu/X-ALMA-13B-Pretrain \
    --group checkpoints/xalma-cpo/best --direction en-ar \
    --input "$TEST_EN" --output "$PRED/xalma-cpo.en-ar.test.ar"
  en_ar+=("$PRED/xalma-cpo.en-ar.test.ar")
fi

echo
echo "ar->en candidates: ${ar_en[*]:-none}"
echo "en->ar candidates: ${en_ar[*]:-none}"

# --- MBR ensemble (submission file) ------------------------------------------
if [ "${SKIP_MBR:-0}" != "1" ] && [ "${#ar_en[@]}" -ge 2 ]; then
  echo ">>> MBR ensemble ar->en"
  python scripts/mbr_decode.py --src "$TEST_AR" --candidates "${ar_en[@]}" \
    --output "$PRED/mbr.ar-en.test.en"
fi
if [ "${SKIP_MBR:-0}" != "1" ] && [ "${#en_ar[@]}" -ge 2 ]; then
  echo ">>> MBR ensemble en->ar"
  python scripts/mbr_decode.py --src "$TEST_EN" --candidates "${en_ar[@]}" \
    --output "$PRED/mbr.en-ar.test.ar"
fi

# --- Scoring (only if test files are TRULY parallel: same line count AND
#     same path stem, i.e. dev/devtest with matched ar/en references).
#     Challenge_Test files happen to share a line count but are independent
#     test sets (no parallel references) -> skip scoring entirely.
ar_stem="$(basename "${TEST_AR%.*}")"
en_stem="$(basename "${TEST_EN%.*}")"
if [ "$n_ar" = "$n_en" ] && [ "$ar_stem" = "$en_stem" ]; then
  echo ">>> scoring (test files share stem '$ar_stem' -> parallel references)"
  shopt -s nullglob
  for hyp in "$PRED"/*.ar-en.test.en; do
    name=$(basename "$hyp" .ar-en.test.en)
    python scripts/evaluate.py --hyp "$hyp" --ref "$TEST_EN" --src "$TEST_AR" --comet \
      | tee "$RES/$name.ar-en.test.json"
  done
  for hyp in "$PRED"/*.en-ar.test.ar; do
    name=$(basename "$hyp" .en-ar.test.ar)
    python scripts/evaluate.py --hyp "$hyp" --ref "$TEST_AR" --src "$TEST_EN" --comet \
      | tee "$RES/$name.en-ar.test.json"
  done
else
  echo "test files are blind (stems '$ar_stem' vs '$en_stem') -- skipping scoring."
  # Remove stale invalid scores from previous runs that mis-scored blind test.
  rm -f "$RES"/*.ar-en.test.json "$RES"/*.en-ar.test.json
fi

# Stage submission copies of the MBR outputs at predictable paths.
[ -f "$PRED/mbr.ar-en.test.en" ] && cp "$PRED/mbr.ar-en.test.en" "$RES/submission.ar-en.test.en"
[ -f "$PRED/mbr.en-ar.test.ar" ] && cp "$PRED/mbr.en-ar.test.ar" "$RES/submission.en-ar.test.ar"

bash push_results.sh "test set: per-system + MBR ensemble" 2>&1 | tail -3 || true
echo "run_test done."
echo "Submission files (copies of the MBR ensemble): "
echo "  $RES/submission.ar-en.test.en"
echo "  $RES/submission.en-ar.test.ar"
