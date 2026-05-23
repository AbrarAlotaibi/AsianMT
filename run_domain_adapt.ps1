# Domain adaptation stage 2: take the unconstrained checkpoint (trained on
# official + external, but pulled off-domain) and fine-tune it on the official
# in-domain data only. Classic "broad pretrain -> in-domain finetune" recipe.
#
# Goal: recover/exceed the official baseline by combining external breadth with
# in-domain specialization. Cheap (~1 h) since official is only ~21k pairs.
#
# Prereq: checkpoints\nllb-unconstrained\best must exist (run run_unconstrained.ps1).

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Base    = "checkpoints\nllb-unconstrained\best"
$Ckpt    = "checkpoints\nllb-unconstrained-ft\best"
$DataDir = "Ar-En\clean"
$PredDir = "predictions"
$ResDir  = "results"

if (-not (Test-Path $Base)) {
    throw "$Base not found. Run .\run_unconstrained.ps1 first."
}
New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

Write-Host ">>> [1/3] fine-tuning unconstrained checkpoint on official in-domain data"
python scripts/train_nllb.py `
    --model $Base `
    --data-dir $DataDir `
    --output-dir checkpoints/nllb-unconstrained-ft `
    --epochs 2 `
    --batch-size 16 `
    --grad-accum 2 `
    --lr 1e-5
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host ">>> [2/3] translating dev set"
python scripts/translate.py `
    --model $Ckpt --kind nllb --direction ar-en `
    --input "$DataDir\dev.ar" --output "$PredDir\unconstrained-ft.ar-en.dev.en"
if ($LASTEXITCODE -ne 0) { throw "translate ar->en failed" }

python scripts/translate.py `
    --model $Ckpt --kind nllb --direction en-ar `
    --input "$DataDir\dev.en" --output "$PredDir\unconstrained-ft.en-ar.dev.ar"
if ($LASTEXITCODE -ne 0) { throw "translate en->ar failed" }

Write-Host ">>> [3/3] scoring dev translations"
python scripts/evaluate.py `
    --hyp "$PredDir\unconstrained-ft.ar-en.dev.en" --ref "$DataDir\dev.en" `
    | Tee-Object -FilePath "$ResDir\unconstrained-ft.ar-en.dev.json"

python scripts/evaluate.py `
    --hyp "$PredDir\unconstrained-ft.en-ar.dev.ar" --ref "$DataDir\dev.ar" `
    | Tee-Object -FilePath "$ResDir\unconstrained-ft.en-ar.dev.json"

Write-Host ""
Write-Host "compare to: 600M official (27.49 / 17.27), 600M+external (25.19 / 16.08)."
