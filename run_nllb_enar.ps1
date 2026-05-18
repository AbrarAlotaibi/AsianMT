# Per-direction fine-tune for en->ar (Sub-Task 1A), starting from the bidirectional
# checkpoint produced by run_nllb.ps1. Targets the weakest direction without
# disturbing ar->en performance.
#
# Strategy: lower LR (1e-5), fewer epochs (2), only en->ar examples (~21k).
# Expected wall time: ~45-90 min on a 16 GB GPU.
#
# Outputs:
#   checkpoints\nllb-Ar-En-enar\best\           fine-tuned model (en->ar only)
#   predictions\nllb-enar.en-ar.dev.ar          new dev translations
#   results\nllb-enar.en-ar.dev.json            BLEU/chrF/TER

$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$BaseCkpt = "checkpoints\nllb-Ar-En\best"
$OutCkpt  = "checkpoints\nllb-Ar-En-enar\best"
$DataDir  = "Ar-En\clean"
$PredDir  = "predictions"
$ResDir   = "results"

if (-not (Test-Path $BaseCkpt)) {
    throw "Base checkpoint not found at $BaseCkpt. Run .\run_nllb.ps1 first."
}

New-Item -ItemType Directory -Force -Path $PredDir, $ResDir | Out-Null

Write-Host ">>> [1/3] en->ar fine-tune from bidir checkpoint"
python scripts/train_nllb.py `
    --model $BaseCkpt `
    --direction en-ar `
    --epochs 2 `
    --lr 1e-5 `
    --batch-size 8 `
    --grad-accum 4 `
    --output-dir checkpoints/nllb-Ar-En-enar
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host ">>> [2/3] translating dev (en->ar)"
python scripts/translate.py `
    --model $OutCkpt --kind nllb --direction en-ar `
    --input "$DataDir\dev.en" --output "$PredDir\nllb-enar.en-ar.dev.ar"
if ($LASTEXITCODE -ne 0) { throw "translate failed" }

Write-Host ">>> [3/3] scoring dev (en->ar)"
python scripts/evaluate.py `
    --hyp "$PredDir\nllb-enar.en-ar.dev.ar" --ref "$DataDir\dev.ar" `
    | Tee-Object -FilePath "$ResDir\nllb-enar.en-ar.dev.json"

Write-Host ""
Write-Host "done. compare to baseline nllb.en-ar.dev.json (BLEU 17.27)."
