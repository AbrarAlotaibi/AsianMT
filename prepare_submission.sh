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

echo ">>> generating system description"
cp system_description.md "$STAGE/${TEAM}_system_description.md"
# Convert to PDF if pandoc is available; otherwise leave markdown.
if command -v pandoc >/dev/null 2>&1; then
  pandoc "$STAGE/${TEAM}_system_description.md" \
    -o "$STAGE/${TEAM}_system_description.pdf" \
    --pdf-engine=xelatex -V geometry:margin=1in 2>/dev/null \
    || pandoc "$STAGE/${TEAM}_system_description.md" \
         -o "$STAGE/${TEAM}_system_description.pdf" 2>/dev/null \
    || echo "  pandoc available but PDF engine missing -- leaving .md"
  if [ -f "$STAGE/${TEAM}_system_description.pdf" ]; then
    rm "$STAGE/${TEAM}_system_description.md"
    echo "  generated ${TEAM}_system_description.pdf"
  fi
elif command -v libreoffice >/dev/null 2>&1; then
  libreoffice --headless --convert-to pdf "$STAGE/${TEAM}_system_description.md" \
    --outdir "$STAGE" 2>/dev/null && rm "$STAGE/${TEAM}_system_description.md" \
    && echo "  generated PDF via libreoffice"
else
  echo "  no pandoc/libreoffice; leaving .md (convert manually before submitting)"
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
