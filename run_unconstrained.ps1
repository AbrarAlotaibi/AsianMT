# Unconstrained system: fine-tune on official + external OPUS parallel data
# (produced by scripts/fetch_parallel.py -> Ar-En/unconstrained).
#
# Defaults to the 600M model: with ~300-400k merged pairs, the 1.3B would take
# multiple days, while the 600M finishes in roughly half a day and lets us
# compare against the 600M official baseline (ar->en 27.49, en->ar 17.27) to
# measure the pure data effect.
#
# To use the 1.3B instead (much slower; plan for days), set:
#   $Model     = "facebook/nllb-200-distilled-1.3B"
#   and add    --optim adamw_bnb_8bit  with --batch-size 4 --grad-accum 8
#   in the training command below.
#
# More data needs fewer epochs (2) and a larger effective batch (16 x 4 = 64).

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Model   = "facebook/nllb-200-distilled-600M"
$DataDir = "Ar-En\unconstrained"
$Ckpt    = "checkpoints\nllb-unconstrained\best"
$PredDir = "predictions"
$ResDir  = "results"

if (-not (Test-Path "$DataDir\train.ar")) {
    throw "$DataDir\train.ar not found. Run scripts/fetch_parallel.py first."
}
New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

Write-Host ">>> [1/3] fine-tuning $Model on official+external (both directions)"
python scripts/train_nllb.py `
    --model $Model `
    --data-dir $DataDir `
    --output-dir checkpoints/nllb-unconstrained `
    --epochs 2 `
    --batch-size 32 `
    --grad-accum 2 `
    --lr 3e-5
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host ">>> [2/3] translating dev set"
python scripts/translate.py `
    --model $Ckpt --kind nllb --direction ar-en `
    --input "$DataDir\dev.ar" --output "$PredDir\unconstrained.ar-en.dev.en"
if ($LASTEXITCODE -ne 0) { throw "translate ar->en failed" }

python scripts/translate.py `
    --model $Ckpt --kind nllb --direction en-ar `
    --input "$DataDir\dev.en" --output "$PredDir\unconstrained.en-ar.dev.ar"
if ($LASTEXITCODE -ne 0) { throw "translate en->ar failed" }

Write-Host ">>> [3/3] scoring dev translations"
python scripts/evaluate.py `
    --hyp "$PredDir\unconstrained.ar-en.dev.en" --ref "$DataDir\dev.en" `
    | Tee-Object -FilePath "$ResDir\unconstrained.ar-en.dev.json"

python scripts/evaluate.py `
    --hyp "$PredDir\unconstrained.en-ar.dev.ar" --ref "$DataDir\dev.ar" `
    | Tee-Object -FilePath "$ResDir\unconstrained.en-ar.dev.json"

Write-Host ""
Write-Host "done. compare to 600M official baseline (ar->en 27.49, en->ar 17.27)."
