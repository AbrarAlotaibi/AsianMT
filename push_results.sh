#!/usr/bin/env bash
# Commit and push results/ (score JSONs + run logs) so they can be reviewed
# remotely (e.g. by an assistant that can read the GitHub repo but not your server).
#
# Usage:
#   bash push_results.sh "xalma zero-shot dev"
#
# Pushes the current branch. Make sure git auth (token/SSH key) is set up on the
# server first. Only results/ is added — large predictions/ and checkpoints/ stay local.

set -euo pipefail

MSG="${1:-results update}"

git add results
if git diff --cached --quiet; then
  echo "no new results to commit"
  exit 0
fi

git -c user.email="${GIT_EMAIL:-results-bot@local}" \
    -c user.name="${GIT_NAME:-results-bot}" \
    commit -m "results: ${MSG}"

# Everything lives in main now: publish there regardless of the checked-out
# branch. Rebase first so a concurrent push doesn't reject this one.
git pull --rebase origin main 2>&1 | tail -3 || true

# Retry push a few times in case of transient network errors.
for i in 1 2 3 4; do
  if git push origin HEAD:main; then
    echo "pushed to main."
    exit 0
  fi
  echo "push failed (attempt $i); retrying in $((2**i))s"
  sleep $((2**i))
done
echo "push failed after retries" >&2
exit 1
