#!/bin/bash
# 2026-10-03 close studies: pull a finished Hoffman2 run of the trees pre-tune study back to the
# laptop (Git Bash, repo root) and write its report locally (the scorer needs the deck, which is not
# shipped).  Copies results/close_studies_2026-10-03/trees_pretune/_work/<tag>/ (trial records,
# Optuna journals, walk forecasts, retune logs, task logs) over h2dtn, md5-checked, then runs the
# report stage for each model the run holds.
#
#   bash cluster/close_pretune_pull_h2.sh <tag>
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: close_pretune_pull_h2.sh <tag>}
SSH=${SSH:-/c/Windows/System32/OpenSSH/ssh.exe}
H2=${H2:-/u/scratch/j/jamesdc1/harxhar-close-pretune}
REL=results/close_studies_2026-10-03/trees_pretune/_work
PY=${PY:-/c/Users/james/miniconda3/envs/285J/python.exe}
mkdir -p "$REL"
"$SSH" -o BatchMode=yes hoffman2 "bash -lc 'cd $H2/$REL && ls $TAG/*/WALK_DONE_r* && find $TAG -type f | sort | xargs md5sum'" > "$REL/${TAG}_remote.md5"
"$SSH" -o BatchMode=yes h2dtn "cd $H2/$REL && tar czf - $TAG" | tar xzf - -C "$REL"
( cd "$REL" && grep -v WALK_DONE "${TAG}_remote.md5" | grep -E '^[0-9a-f]{32} ' | md5sum -c --quiet ) || { echo "PULL FAILED: md5"; exit 1; }
echo "pulled $REL/$TAG ($(find "$REL/$TAG" -type f | wc -l) files, md5 OK)"
"$PY" experiments/close_trees_pretune.py gate
for d in "$REL/$TAG"/*/; do
  m=$(basename "$d")
  CTP_TAG=$TAG CTP_MODEL=$m "$PY" experiments/close_trees_pretune.py report
done
