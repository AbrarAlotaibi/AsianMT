#!/usr/bin/env bash
# Prepare the OpenBracket WMT26 submission zip:
#   OpenBracket.zip/
#     OpenBracket_primary_Sub-Task-1A.txt        (en->ar primary)
#     OpenBracket_primary_Sub-Task-2A.txt        (ar->en primary)
#     OpenBracket_contrastive1_Sub-Task-1A.txt   (en->ar contrastive 1: ALLAM-FT)
#     OpenBracket_contrastive1_Sub-Task-2A.txt   (ar->en contrastive 1: ALLAM-FT)
#     OpenBracket_contrastive2_Sub-Task-1A.txt   (en->ar contrastive 2: Sampling-MBR)
#     OpenBracket_contrastive2_Sub-Task-2A.txt   (ar->en contrastive 2: Sampling-MBR)
#     OpenBracket_system_description.pdf         (mandatory abstract)

set -euo pipefail
cd "$(dirname "$0")"

TEAM="OpenBracket"
STAGE="submission_package/$TEAM"
rm -rf "submission_package"
mkdir -p "$STAGE"

echo ">>> staging submission files"
cp results/submission-primary.en-ar.test.ar                  "$STAGE/${TEAM}_primary_Sub-Task-1A.txt"
cp results/submission-primary.ar-en.test.en                  "$STAGE/${TEAM}_primary_Sub-Task-2A.txt"
cp results/submission-contrastive1-allam.en-ar.test.ar       "$STAGE/${TEAM}_contrastive1_Sub-Task-1A.txt"
cp results/submission-contrastive1-allam.ar-en.test.en       "$STAGE/${TEAM}_contrastive1_Sub-Task-2A.txt"
cp results/submission-contrastive2-sampling-mbr.en-ar.test.ar "$STAGE/${TEAM}_contrastive2_Sub-Task-1A.txt"
cp results/submission-contrastive2-sampling-mbr.ar-en.test.en "$STAGE/${TEAM}_contrastive2_Sub-Task-2A.txt"

echo ">>> verifying line counts (should all be 1500)"
for f in "$STAGE"/*.txt; do
  n=$(python3 -c "print(len(open('$f',encoding='utf-8').read().splitlines()))")
  echo "  $n  $(basename "$f")"
done

echo ">>> generating system description PDF"
# Use our reportlab-based markdown -> PDF script. No LaTeX or pandoc needed.
# Falls back to plain markdown if reportlab is missing.
if python3 -c "import reportlab" 2>/dev/null; then
  python3 scripts/md_to_pdf.py system_description.md \
    "$STAGE/${TEAM}_system_description.pdf"
else
  echo "  reportlab not installed; copying markdown."
  cp system_description.md "$STAGE/${TEAM}_system_description.md"
  echo "  pip install reportlab  # then re-run for PDF"
fi

echo ">>> creating zip"
cd submission_package
zip -r "${TEAM}.zip" "$TEAM" -q
mv "${TEAM}.zip" ..
cd ..

echo ""
echo "=== package contents ==="
unzip -l "${TEAM}.zip"
echo ""
echo "Submission archive: ${TEAM}.zip"
