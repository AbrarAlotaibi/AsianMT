#!/usr/bin/env bash
# Shared preflight for the run_*/retry_* launchers. Source it, don't execute:
#   source "$(dirname "$0")/scripts/preflight.sh"
#
# Does three things, in order:
#   1. Activate the asianmt conda env if torch isn't already importable
#      (the deps live in `asianmt`, not `base`).
#   2. Hard-fail with a clear message if torch still can't import, instead of
#      crawling into a training run that dies on `import torch`.
#   3. Sync to origin/main without letting the self-written logs block a rebase
#      (stashes any stray working-tree changes first).

# 1 + 2: ensure the ML env is active.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "preflight: torch not importable in current env; activating 'asianmt' ..."
  # shellcheck disable=SC1091
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate asianmt 2>/dev/null || true
fi
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "preflight ERROR: 'torch' is not installed in the active Python env." >&2
  echo "  Run:  conda activate asianmt" >&2
  echo "  If the env doesn't exist yet:  conda env create -f environment.yml" >&2
  echo "  Verify: python -c \"import torch; print(torch.__version__, torch.cuda.device_count())\"" >&2
  exit 1
fi
echo "preflight: $(python -c 'import torch; print("torch", torch.__version__, "| GPUs", torch.cuda.device_count())')"

# 3: sync to main cleanly. Stash first so logs/edits don't block the rebase.
git checkout main 2>/dev/null || git checkout -b main origin/main
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "preflight: stashing local working-tree changes before pull"
  git stash push -m "preflight-autostash $(date -u +%FT%TZ)" >/dev/null 2>&1 || true
fi
git pull --rebase origin main 2>&1 | tail -2 || true
