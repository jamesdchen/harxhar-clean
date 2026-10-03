#!/bin/bash
# 2026-10-03 close studies: ship the trees pre-tune study to Hoffman2 (run on the laptop, Git Bash,
# from the repo root).  Fills the deployment root $H2 with the study's code, the tree specs it
# executes, src/, its cluster scripts and the two design caches (the 16:00-bar and last-hour
# all_features designs, ~27 MB: copied over the data-transfer node h2dtn), writes
# close_pretune_manifest.md5 (the md5 of every shipped file, computed here) and FAILS unless every
# file on Hoffman2 matches it.  Re-run after a scratch purge; nothing is submitted.
#
#   bash cluster/close_pretune_ship_h2.sh
#   DRY=1 STAGE_DIR=<dir> bash cluster/close_pretune_ship_h2.sh   # build the bundle only (local smoke test)
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=${SSH:-/c/Windows/System32/OpenSSH/ssh.exe}
H2=${H2:-/u/scratch/j/jamesdc1/harxhar-close-pretune}
STAGE_DIR=${STAGE_DIR:-${TMPDIR:-/tmp}/ship_close_pretune}
mkdir -p "$STAGE_DIR"
FILES=(experiments/close_trees_pretune.py experiments/close_trees_pretune_jobs.py
       specs/causal_tune_trees.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py
       specs/causal_tune_linear.py
       cluster/close_pretune_task.sh cluster/submit_close_pretune_h2.sh
       cluster/close_pretune_tasks_pretune.txt cluster/close_pretune_tasks_walk.txt
       results/close_design/_work/design_bar1600_all_features.npz
       results/close_design/_work/design_last30_all_features.npz)
mapfile -t SRC < <(find src -name '*.py' -not -path '*/__pycache__/*' | sort)
FILES+=("${SRC[@]}")
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }; done
md5sum "${FILES[@]}" > "$STAGE_DIR/close_pretune_manifest.md5"
tar czf "$STAGE_DIR/close_pretune.tgz" "${FILES[@]}" -C "$STAGE_DIR" close_pretune_manifest.md5
if [ "${DRY:-0}" = 1 ]; then echo "DRY=1: bundle at $STAGE_DIR/close_pretune.tgz (nothing shipped)"; exit 0; fi
echo "shipping ${#FILES[@]} files ($(du -h "$STAGE_DIR/close_pretune.tgz" | cut -f1)) -> h2dtn:$H2"
"$SSH" -o BatchMode=yes h2dtn "mkdir -p $H2/logs && cd $H2 && tar xzf -" < "$STAGE_DIR/close_pretune.tgz"
"$SSH" -o BatchMode=yes hoffman2 "bash -lc 'cd $H2 && md5sum -c --quiet close_pretune_manifest.md5 && echo md5 OK: \$(wc -l < close_pretune_manifest.md5) files'" \
  || { echo "SHIP FAILED: md5 mismatch on Hoffman2"; exit 1; }
"$SSH" -o BatchMode=yes hoffman2 "bash -lc 'test -x /u/scratch/j/jamesdc1/harxhar-optuna/carc_env/harxhar/bin/python3.11 && echo runtime: CARC stack present || echo runtime: CARC stack absent, the tasks will use conda hpc-pi'"
echo "ship OK -> hoffman2:$H2"
