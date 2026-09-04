#!/usr/bin/env bash
# Camera-ready experiments requested by the reviewers, on the official DEV set
# (n=1000), both directions, same 5-system public pool and metric config as
# the paper:
#
#   TASK A  leave-one-out ablation of the MBR candidate pool
#   TASK B  paired bootstrap significance (sacreBLEU --paired-bs for
#           BLEU/chrF/TER; sentence-level wmt22-comet-da bootstrap for COMET)
#
# Pool (both dirs): nllb33b, xalma-ft, xalma-cpo, allam-ft, opus
# MBR utility:      Unbabel/wmt22-comet-da   (scripts/mbr_decode.py default)
#
# Needs a GPU (COMET-22 scoring: 5 leave-one-out pools + full pool per dir).
# Inputs: predictions/<sys>.{ar-en,en-ar}.dev.{en,ar} for the 5 pool systems
#         plus sampling-mbr, and Ar-En/dev.{ar,en}. Missing inputs abort.
#
# Usage:   CUDA_VISIBLE_DEVICES=0 bash run_camera_ready.sh
# Monitor: tail -f results/camera_ready/run.log
# Output:  results/camera_ready/*.json  +  camera_ready_experiments.md

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

if ! python -c "import torch, comet, sacrebleu" 2>/dev/null; then
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate asianmt
fi
python -c "import torch, comet, sacrebleu" || { echo "ERROR: torch/comet/sacrebleu not importable" >&2; exit 1; }

OUT="results/camera_ready"
LOO="predictions/camera_ready"
mkdir -p "$OUT" "$LOO"
exec > >(tee "$OUT/run.log") 2>&1
echo "=== run_camera_ready $(date -u +%FT%TZ)  GPU=${CUDA_VISIBLE_DEVICES:-auto} ==="

SYS=(nllb33b xalma-ft xalma-cpo allam-ft opus)
N_BS=1000

run_dir() {
  local dir=$1 src=$2 ref=$3 ext=$4
  echo ""; echo "################ $dir ################"

  local missing=0
  for f in "$src" "$ref"; do [ -f "$f" ] || { echo "MISSING: $f"; missing=1; }; done
  for s in "${SYS[@]}" sampling-mbr; do
    f="predictions/$s.$dir.dev.$ext"
    if [ -f "$f" ]; then echo "  ok  $f ($(awk 'END{print NR}' "$f") lines)"; else echo "MISSING: $f"; missing=1; fi
  done
  [ "$missing" = 0 ] || { echo "ERROR: missing inputs for $dir; aborting." >&2; exit 1; }

  # ---- TASK A: full pool (reference row) ------------------------------------
  local full="predictions/mbr5pub.$dir.dev.$ext"
  local cands=()
  for s in "${SYS[@]}"; do cands+=("predictions/$s.$dir.dev.$ext"); done
  if [ ! -f "$full" ]; then
    echo ">>> [$dir] full 5-system MBR -> $full"
    python scripts/mbr_decode.py --src "$src" --candidates "${cands[@]}" --output "$full"
  fi
  echo ">>> [$dir] score full pool"
  python scripts/evaluate.py --hyp "$full" --ref "$ref" --src "$src" --comet \
    | tee "$OUT/loo-none.$dir.dev.json"

  # ---- TASK A: leave-one-out pools -----------------------------------------
  for drop in "${SYS[@]}"; do
    local pool=()
    for s in "${SYS[@]}"; do [ "$s" != "$drop" ] && pool+=("predictions/$s.$dir.dev.$ext"); done
    local out="$LOO/mbr-minus-$drop.$dir.dev.$ext"
    if [ ! -f "$out" ]; then
      echo ">>> [$dir] MBR without $drop (${#pool[@]} systems) -> $out"
      python scripts/mbr_decode.py --src "$src" --candidates "${pool[@]}" --output "$out"
    fi
    echo ">>> [$dir] score pool without $drop"
    python scripts/evaluate.py --hyp "$out" --ref "$ref" --src "$src" --comet \
      | tee "$OUT/loo-minus-$drop.$dir.dev.json"
  done

  # ---- TASK B: paired bootstrap, primary (full pool) vs 4 systems ----------
  local others=(
    "predictions/allam-ft.$dir.dev.$ext"
    "predictions/xalma-cpo.$dir.dev.$ext"
    "predictions/nllb33b.$dir.dev.$ext"
    "predictions/sampling-mbr.$dir.dev.$ext"
  )
  echo ">>> [$dir] sacreBLEU paired bootstrap (BLEU/chrF/TER, n=$N_BS, seed=default 12345)"
  # CLI text table kept as the human-readable record (its -f json crashes in
  # sacrebleu 2.6.0 with --paired-bs); the API wrapper writes the same test as JSON.
  sacrebleu "$ref" -i "$full" "${others[@]}" -m bleu chrf ter \
    --paired-bs --paired-bs-n "$N_BS" -f text > "$OUT/paired-bs.$dir.txt"
  python scripts/paired_bootstrap_sacrebleu.py --ref "$ref" --baseline "$full" \
    --systems "${others[@]}" --names allam-ft xalma-cpo nllb33b sampling-mbr \
    --n-resamples "$N_BS" --out "$OUT/paired-bs.$dir.json"

  echo ">>> [$dir] COMET paired bootstrap (sentence-level wmt22-comet-da, n=$N_BS)"
  python scripts/paired_bootstrap_comet.py --src "$src" --ref "$ref" \
    --baseline "$full" --systems "${others[@]}" \
    --names allam-ft xalma-cpo nllb33b sampling-mbr \
    --n-resamples "$N_BS" --seed 12345 \
    --sentence-scores "$OUT/comet-sentence-scores.$dir.json" \
    --out "$OUT/paired-bs-comet.$dir.json"
}

run_dir ar-en Ar-En/dev.ar Ar-En/dev.en en
run_dir en-ar Ar-En/dev.en Ar-En/dev.ar ar

echo ""
echo ">>> building camera_ready_experiments.md"
python scripts/camera_ready_report.py

git add camera_ready_experiments.md predictions/mbr5pub.*.dev.* 2>/dev/null || true
bash push_results.sh "camera-ready: LOO pool ablation + paired bootstrap (dev)" 2>&1 | tail -3 || true
echo "done."
