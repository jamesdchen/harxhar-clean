#!/usr/bin/env bash
# The orchestrator (std worktree) on the 13-bar 2012 w1000 late-start configuration (run B's runner_cfg).
# usage: run_orch.sh <out_dir (Windows path)> <stage> [more orchestrator options]
set -euo pipefail
WT=/c/Users/james/AppData/Local/Temp/std_wt
PY=/c/Users/james/AppData/Local/Temp/w1500_venv/Scripts/python.exe
OUT=$1; STAGE=$2; shift 2
unset OPENBLAS_CORETYPE  # the KKT agent: setting it changes the last bits
export TQDM_DISABLE=1
cd "$WT"
t0=$(date +%s)
echo "[$(date +%H:%M:%S)] $STAGE out=$OUT $*"
"$PY" experiments/pooled_linear_rerun_2026_10_19_13.py "$STAGE" --window-days 1000 --first-day 2012-01-03 \
  --ex-flag LATE_START_WARMUP=1 --out "$OUT" "$@"
echo "[$(date +%H:%M:%S)] $STAGE done in $(( $(date +%s) - t0 )) s"
