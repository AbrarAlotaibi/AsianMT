# Bidirectional fine-tune of NLLB-200-distilled-1.3B for Ar<->En.
# Bigger model than the 600M baseline -> expected +3 to +5 BLEU per direction.
#
# Memory: 1.3B params need fp16 + gradient checkpointing on a 16 GB GPU.
# Tuned for RTX 4080 (Super) class cards. Effective batch 32 (batch 2 x accum 16),
# same as the 600M run for an apples-to-apples comparison.
#
# Expected wall time: ~7-10 hours for 3 epochs. Plan it as an overnight run.
#
# If you OOM at batch=2, edit this script to use --batch-size 1 --grad-accum 32.
#
# Outputs:
#   checkpoints\nllb13b-Ar-En\best\                fine-tuned model
#   predictions\nllb13b.ar-en.dev.en               Ar->En dev translations
#   predictions\nllb13b.en-ar.dev.ar               En->Ar dev translations
#   results\nllb13b.ar-en.dev.json                 BLEU/chrF/TER (ar->en)
#   results\nllb13b.en-ar.dev.json                 BLEU/chrF/TER (en->ar)

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$DataDir = "Ar-En\clean"
$Ckpt    = "checkpoints\nllb13b-Ar-En\best"
$PredDir = "predictions"
$ResDir  = "results"

New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

if (-not (Test-Path "$DataDir\train.ar")) {
    Write-Host ">>> [1/4] cleaning training data"
    python scripts/clean.py
    if ($LASTEXITCODE -ne 0) { throw "clean.py failed" }
} else {
    Write-Host ">>> [1/4] cleaned data already present, skipping clean.py"
}

Write-Host ">>> [2/4] fine-tuning NLLB-200-distilled-1.3B (both directions)"
python scripts/train_nllb.py `
    --model facebook/nllb-200-distilled-1.3B `
    --output-dir checkpoints/nllb13b-Ar-En `
    --epochs 3 `
    --batch-size 2 `
    --grad-accum 16 `
    --lr 3e-5
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host ">>> [3/4] translating dev set"
python scripts/translate.py `
    --model $Ckpt --kind nllb --direction ar-en `
    --input "$DataDir\dev.ar" --output "$PredDir\nllb13b.ar-en.dev.en"
if ($LASTEXITCODE -ne 0) { throw "translate ar->en failed" }

python scripts/translate.py `
    --model $Ckpt --kind nllb --direction en-ar `
    --input "$DataDir\dev.en" --output "$PredDir\nllb13b.en-ar.dev.ar"
if ($LASTEXITCODE -ne 0) { throw "translate en->ar failed" }

Write-Host ">>> [4/4] scoring dev translations"
python scripts/evaluate.py `
    --hyp "$PredDir\nllb13b.ar-en.dev.en" --ref "$DataDir\dev.en" `
    | Tee-Object -FilePath "$ResDir\nllb13b.ar-en.dev.json"

python scripts/evaluate.py `
    --hyp "$PredDir\nllb13b.en-ar.dev.ar" --ref "$DataDir\dev.ar" `
    | Tee-Object -FilePath "$ResDir\nllb13b.en-ar.dev.json"

Write-Host ""
Write-Host "done. compare to 600M baseline (ar->en 27.49, en->ar 17.27)."
