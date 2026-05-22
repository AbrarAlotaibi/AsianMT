# Contrastive system: fine-tune Helsinki-NLP opus-mt bilingual models for both
# directions, then translate dev and score. Independent of NLLB -> good contrastive.
#
# Two small (~298M) models, ~30-60 min each on a 16 GB GPU.
#
# Outputs:
#   checkpoints\opus-Ar-En-ar-en\best\   checkpoints\opus-Ar-En-en-ar\best\
#   predictions\opus.{dir}.dev.*         results\opus.{dir}.dev.json

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$DataDir = "Ar-En\clean"
$PredDir = "predictions"
$ResDir  = "results"
New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

foreach ($dir in @("ar-en", "en-ar")) {
    Write-Host ">>> training opus-mt $dir"
    python scripts/train_opus.py --direction $dir --epochs 5 --batch-size 16 --grad-accum 2
    if ($LASTEXITCODE -ne 0) { throw "opus $dir training failed" }

    $ckpt = "checkpoints\opus-Ar-En-$dir\best"
    if ($dir -eq "ar-en") { $inp = "$DataDir\dev.ar"; $ref = "$DataDir\dev.en"; $out = "$PredDir\opus.ar-en.dev.en" }
    else                  { $inp = "$DataDir\dev.en"; $ref = "$DataDir\dev.ar"; $out = "$PredDir\opus.en-ar.dev.ar" }

    Write-Host ">>> translating dev ($dir)"
    python scripts/translate.py --model $ckpt --kind opus --direction $dir --input $inp --output $out
    if ($LASTEXITCODE -ne 0) { throw "opus $dir translate failed" }

    Write-Host ">>> scoring dev ($dir)"
    python scripts/evaluate.py --hyp $out --ref $ref | Tee-Object -FilePath "$ResDir\opus.$dir.dev.json"
}

Write-Host ""
Write-Host "done. compare to NLLB-1.3B (ar->en 30.79, en->ar 20.29)."
