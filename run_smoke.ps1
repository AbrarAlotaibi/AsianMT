# Smoke test: verify the NLLB training pipeline end-to-end with ~50 steps.
# Should take 2-5 minutes on a 12 GB GPU. No checkpoint is saved.
#
# What this catches:
#   - missing deps / wrong torch+CUDA build
#   - data path or tokenization bugs
#   - OOM at the configured batch size
#   - language-tag plumbing errors
#
# What this does NOT catch: model quality. Use run_nllb.ps1 for that.
#
# Run from PowerShell (after `conda activate asianmt`):
#   .\run_smoke.ps1
# If you get an execution policy error, run once in the same session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = 'Stop'

# Workaround for "OMP: Error #15" caused by conda MKL + pip torch shipping
# duplicate libiomp5md.dll on Windows.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

if (-not (Test-Path "Ar-En\clean\train.ar")) {
    Write-Host ">>> running scripts/clean.py (one-time)"
    python scripts/clean.py
    if ($LASTEXITCODE -ne 0) { throw "clean.py failed" }
}

Write-Host ">>> NLLB smoke test (50 steps)"
python scripts/train_nllb.py `
    --max-steps 50 `
    --batch-size 4 `
    --grad-accum 8 `
    --output-dir checkpoints/_smoke
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

Write-Host ""
Write-Host "smoke test OK -- proceed with: .\run_nllb.ps1"
