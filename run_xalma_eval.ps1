# Zero-shot evaluation of X-ALMA-13B (Arabic = Group 8) on dev, both directions.
# 13B loaded in 4-bit (nf4) -> fits the 16 GB 4080 SUPER. No training.
#
# Prereq (once):
#   pip install peft bitsandbytes
#
# First run downloads the 13B backbone (~26 GB) + Group-8 module. Generation with
# beam=5 on a 4-bit 13B is slow-ish; budget ~30-60 min for 1000 dev sentences.
#
# Outputs:
#   predictions\xalma.ar-en.dev.en   results\xalma.ar-en.dev.json
#   predictions\xalma.en-ar.dev.ar   results\xalma.en-ar.dev.json

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$DataDir = "Ar-En\clean"
$PredDir = "predictions"
$ResDir  = "results"
New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

# Smoke check first: 20 lines ar->en to validate prompt/loading before the full run.
Write-Host ">>> smoke: 20-line ar->en sanity check"
python scripts/xalma_translate.py --direction ar-en `
    --input "$DataDir\dev.ar" --output "$PredDir\xalma.smoke.en" --limit 20
if ($LASTEXITCODE -ne 0) { throw "smoke failed" }
Write-Host "---- first lines of smoke output ----"
Get-Content "$PredDir\xalma.smoke.en" -TotalCount 5

Write-Host ">>> full dev ar->en"
python scripts/xalma_translate.py --direction ar-en `
    --input "$DataDir\dev.ar" --output "$PredDir\xalma.ar-en.dev.en"
if ($LASTEXITCODE -ne 0) { throw "ar->en failed" }

Write-Host ">>> full dev en->ar"
python scripts/xalma_translate.py --direction en-ar `
    --input "$DataDir\dev.en" --output "$PredDir\xalma.en-ar.dev.ar"
if ($LASTEXITCODE -ne 0) { throw "en->ar failed" }

Write-Host ">>> scoring"
python scripts/evaluate.py --hyp "$PredDir\xalma.ar-en.dev.en" --ref "$DataDir\dev.en" `
    | Tee-Object -FilePath "$ResDir\xalma.ar-en.dev.json"
python scripts/evaluate.py --hyp "$PredDir\xalma.en-ar.dev.ar" --ref "$DataDir\dev.ar" `
    | Tee-Object -FilePath "$ResDir\xalma.en-ar.dev.json"

Write-Host ""
Write-Host "X-ALMA zero-shot. Compare to NLLB-1.3B official (ar->en 30.79, en->ar 20.29)."
