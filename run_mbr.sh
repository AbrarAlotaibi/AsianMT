#!/usr/bin/env bash
# COMET-MBR system combination over whatever dev predictions exist.
# Picks, per sentence, the MBR-optimal candidate across all systems -> a single
# combined output per direction, then scores it (incl. COMET).
#
# Cheap: no model training, just COMET scoring. Run after the systems' dev
# predictions are in predictions/. Re-run freely as more systems land.
#
# Usage:  bash run_mbr.sh
# Monitor: tail -f results/mbr.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

DATA_DIR="Ar-En/clean"
PRED="predictions"
RES="results"
mkdir -p "$RES"
exec > >(tee "$RES/mbr.log") 2>&1
echo "=== run_mbr $(date -u +%FT%TZ) ==="

# Collect candidate files that actually exist, per direction.
# Also accept the NLLB-1.3B baseline that's saved without a "13b" suffix.
ar_en=()
for f in "$PRED"/xalma-ft.ar-en.dev.en "$PRED"/nllb33b.ar-en.dev.en \
         "$PRED"/nllb13b.ar-en.dev.en "$PRED"/nllb.ar-en.dev.en \
         "$PRED"/opus.ar-en.dev.en; do
  [ -f "$f" ] && ar_en+=("$f")
done
en_ar=()
for f in "$PRED"/xalma-ft.en-ar.dev.ar "$PRED"/nllb33b.en-ar.dev.ar \
         "$PRED"/nllb13b.en-ar.dev.ar "$PRED"/nllb.en-ar.dev.ar \
         "$PRED"/opus.en-ar.dev.ar; do
  [ -f "$f" ] && en_ar+=("$f")
done

echo "ar->en candidates: ${ar_en[*]:-none}"
echo "en->ar candidates: ${en_ar[*]:-none}"

if [ "${#ar_en[@]}" -ge 2 ]; then
  echo ">>> MBR ar->en"
  python scripts/mbr_decode.py --src "$DATA_DIR/dev.ar" \
    --candidates "${ar_en[@]}" --output "$PRED/mbr.ar-en.dev.en"
  python scripts/evaluate.py --hyp "$PRED/mbr.ar-en.dev.en" --ref "$DATA_DIR/dev.en" \
    --src "$DATA_DIR/dev.ar" --comet | tee "$RES/mbr.ar-en.dev.json"
else
  echo "skip ar->en (need >=2 candidate files)"
fi

if [ "${#en_ar[@]}" -ge 2 ]; then
  echo ">>> MBR en->ar"
  python scripts/mbr_decode.py --src "$DATA_DIR/dev.en" \
    --candidates "${en_ar[@]}" --output "$PRED/mbr.en-ar.dev.ar"
  python scripts/evaluate.py --hyp "$PRED/mbr.en-ar.dev.ar" --ref "$DATA_DIR/dev.ar" \
    --src "$DATA_DIR/dev.en" --comet | tee "$RES/mbr.en-ar.dev.json"
else
  echo "skip en->ar (need >=2 candidate files)"
fi

echo "done. compare MBR to the best single system per direction."

# Auto-push so the scores land in main without a second command.
bash push_results.sh "MBR ensemble (XALMA-FT + NLLB-3.3B + opus + NLLB-1.3B if present)" 2>&1 | tail -3 || true
