#!/usr/bin/env bash
# Re-run dev-set MBR system combination with the SAME 5-system public pool
# used at test time, for BOTH directions. Replaces the old 3-system dev MBR
# scores in the paper with faithful 5-system numbers.
#
# Pool (identical both dirs, NO Claude-API):
#   NLLB-3.3B FT, X-ALMA-FT, X-ALMA-CPO, ALLaM-FT, opus-mt
#
# Prereqs:
#   - Ar-En/clean/dev.{ar,en} must exist (run: python scripts/clean.py)
#   - Per-system dev predictions in predictions/ (10 files, see REQUIRED below)
#   - GPU for COMET-22 scoring
#
# Usage:  CUDA_VISIBLE_DEVICES=0 bash run_mbr5_dev.sh
# Monitor: tail -f results/mbr5_dev.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source "$(dirname "$0")/scripts/preflight.sh"

DATA="Ar-En/clean"
PRED="predictions"
RES="results"
mkdir -p "$RES" "$PRED"
exec > >(tee "$RES/mbr5_dev.log") 2>&1
echo "=== run_mbr5_dev $(date -u +%FT%TZ)  GPU=${CUDA_VISIBLE_DEVICES:-auto} ==="

# ---- Verify clean dev exists ------------------------------------------------
if [ ! -f "$DATA/dev.ar" ] || [ ! -f "$DATA/dev.en" ]; then
  echo "Ar-En/clean/dev.* missing — regenerating via clean.py ..."
  python scripts/clean.py
fi

# ---- Verify all 10 per-system dev predictions exist -------------------------
REQUIRED_AREN=(
  "$PRED/nllb33b.ar-en.dev.en"
  "$PRED/xalma-ft.ar-en.dev.en"
  "$PRED/xalma-cpo.ar-en.dev.en"
  "$PRED/allam-ft.ar-en.dev.en"
  "$PRED/opus.ar-en.dev.en"
)
REQUIRED_ENAR=(
  "$PRED/nllb33b.en-ar.dev.ar"
  "$PRED/xalma-ft.en-ar.dev.ar"
  "$PRED/xalma-cpo.en-ar.dev.ar"
  "$PRED/allam-ft.en-ar.dev.ar"
  "$PRED/opus.en-ar.dev.ar"
)
ok=true
for f in "${REQUIRED_AREN[@]}" "${REQUIRED_ENAR[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f" >&2
    ok=false
  else
    n=$(awk 'END{print NR}' "$f")
    echo "  ok  $f  ($n lines)"
  fi
done
if [ "$ok" != "true" ]; then
  echo "ERROR: one or more dev prediction files are missing. Cannot run MBR." >&2
  echo "       Generate them first, then re-run this script." >&2
  exit 1
fi

# ---- ar->en MBR (5-system, no Claude-API) -----------------------------------
echo ""
echo ">>> MBR 5-system ar->en"
python scripts/mbr_decode.py --src "$DATA/dev.ar" \
  --candidates "${REQUIRED_AREN[@]}" \
  --output "$PRED/mbr5pub.ar-en.dev.en"

echo ">>> scoring ar->en"
python scripts/evaluate.py \
  --hyp "$PRED/mbr5pub.ar-en.dev.en" \
  --ref "$DATA/dev.en" --src "$DATA/dev.ar" --comet \
  | tee "$RES/mbr5pub.ar-en.dev.json"

# ---- en->ar MBR (5-system, no Claude-API) -----------------------------------
echo ""
echo ">>> MBR 5-system en->ar"
python scripts/mbr_decode.py --src "$DATA/dev.en" \
  --candidates "${REQUIRED_ENAR[@]}" \
  --output "$PRED/mbr5pub.en-ar.dev.ar"

echo ">>> scoring en->ar"
python scripts/evaluate.py \
  --hyp "$PRED/mbr5pub.en-ar.dev.ar" \
  --ref "$DATA/dev.ar" --src "$DATA/dev.en" --comet \
  | tee "$RES/mbr5pub.en-ar.dev.json"

# ---- Summary ----------------------------------------------------------------
echo ""
echo "=========================================="
echo "5-system public MBR dev results"
echo "=========================================="
echo "ar->en:"; cat "$RES/mbr5pub.ar-en.dev.json"
echo ""
echo "en->ar:"; cat "$RES/mbr5pub.en-ar.dev.json"
echo ""
echo "Pool (both dirs): NLLB-3.3B FT, X-ALMA-FT, X-ALMA-CPO, ALLaM-FT, opus-mt"
echo "NO Claude-API in either direction."
echo ""
echo "Replace the old MBR sys-combo rows in system_description.md with these."
echo ""

# Auto-push results so scores land in the repo.
bash push_results.sh "5-system public MBR dev scores (replaces 3-system)" 2>&1 | tail -3 || true
echo "done."
