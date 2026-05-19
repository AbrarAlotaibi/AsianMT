# Bidirectional fine-tune of NLLB-200-distilled-1.3B for Ar<->En.
# Bigger model than the 600M baseline -> expected +3 to +5 BLEU per direction.
#
# Memory: 1.3B needs 8-bit AdamW (bitsandbytes) on a 16 GB GPU to free up
# enough VRAM for a usable microbatch. With batch=4, grad_accum=8, effective
# batch is 32 (same as the 600M run).
#
# Prereq (once):
#   pip install bitsandbytes
#
# Expected wall time: ~6-10 h for 3 epochs. Plan it as an overnight run.
#
# If you still OOM at batch=4, drop to batch=2 grad_accum=16. If batch=2 still
# OOMs (it shouldn't with adamw_bnb_8bit), go to batch=1 grad_accum=32.
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
    --batch-size 4 `
    --grad-accum 8 `
    --lr 3e-5 `
    --optim adamw_bnb_8bit
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
