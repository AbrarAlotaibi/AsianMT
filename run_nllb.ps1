# Full NLLB-600M training run for Ar<->En, then translate dev and score.
# Expected wall time on a 12 GB GPU: ~1.5-3 hours for 3 epochs.
#
# Outputs:
#   checkpoints\nllb-Ar-En\best\                      fine-tuned model
#   predictions\nllb.ar-en.dev.en                     Ar->En dev translations
#   predictions\nllb.en-ar.dev.ar                     En->Ar dev translations
#   results\nllb.ar-en.dev.json                       BLEU/chrF/TER (ar->en)
#   results\nllb.en-ar.dev.json                       BLEU/chrF/TER (en->ar)
#
# We score on DEV (not devtest) -- keeping devtest pristine for final submission
# so we don't accidentally overfit to it during iteration.
#
# Run from PowerShell (after `conda activate asianmt`):
#   .\run_nllb.ps1
# If you get an execution policy error, run once in the same session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = 'Stop'

$DataDir = "Ar-En\clean"
$Ckpt    = "checkpoints\nllb-Ar-En\best"
$PredDir = "predictions"
$ResDir  = "results"

New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

# 1. Clean data once
if (-not (Test-Path "$DataDir\train.ar")) {
    Write-Host ">>> [1/4] cleaning training data"
    python scripts/clean.py
    if ($LASTEXITCODE -ne 0) { throw "clean.py failed" }
} else {
    Write-Host ">>> [1/4] cleaned data already present, skipping clean.py"
}

# 2. Train (3 epochs, fp16, grad-accum 8 for 12 GB GPU)
Write-Host ">>> [2/4] fine-tuning NLLB-200-distilled-600M (both directions)"
python scripts/train_nllb.py `
    --epochs 3 `
    --batch-size 4 `
    --grad-accum 8 `
    --lr 3e-5
if ($LASTEXITCODE -ne 0) { throw "training failed" }

# 3. Translate dev both directions
Write-Host ">>> [3/4] translating dev set"
python scripts/translate.py `
    --model $Ckpt --kind nllb --direction ar-en `
    --input "$DataDir\dev.ar" --output "$PredDir\nllb.ar-en.dev.en"
if ($LASTEXITCODE -ne 0) { throw "translate ar->en failed" }

python scripts/translate.py `
    --model $Ckpt --kind nllb --direction en-ar `
    --input "$DataDir\dev.en" --output "$PredDir\nllb.en-ar.dev.ar"
if ($LASTEXITCODE -ne 0) { throw "translate en->ar failed" }

# 4. Score
Write-Host ">>> [4/4] scoring dev translations"
python scripts/evaluate.py `
    --hyp "$PredDir\nllb.ar-en.dev.en" --ref "$DataDir\dev.en" `
    | Tee-Object -FilePath "$ResDir\nllb.ar-en.dev.json"

python scripts/evaluate.py `
    --hyp "$PredDir\nllb.en-ar.dev.ar" --ref "$DataDir\dev.ar" `
    | Tee-Object -FilePath "$ResDir\nllb.en-ar.dev.json"

Write-Host ""
Write-Host "done. dev scores are above. send them back so we can decide next steps."
