#!/bin/bash
# Pull the scored TUNED per-bar tree campaign (specs/causal_tune_trees_tuned.py,
# CARC root results/linear_subsection_trees_tuned) into the same local path,
# WITHOUT the npz files and WITHOUT the per-chunk folders (chunks/):
#   1. ship experiments/a2b_extract_tree_importance.py (tar over native Windows
#      OpenSSH, md5-checked) and run it on the login node (reads the merged npz
#      files, writes $R/importance/: per-refit native importance and the TreeSHAP
#      rows of the SHAP bar as parquet tables, plus a manifest) -- so the
#      importance leaves CARC as tables, never as npz;
#   2. list, on CARC, the files to pull: every regular file directly under $R
#      (score tables, logs, flags), every merged arm's small files
#      ($R/<bucket>/<seg>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/:
#      results_*.csv, results_qsel_*.csv, tune_trace_*.csv, tune_candidates_*.csv,
#      grid_*.json, *_feature_health.csv, MERGED) and $R/importance/*;
#   3. tar them over ssh into the local repo root, then FAIL unless every pulled
#      file's local md5 equals its remote md5.
# Refuses to run before the scorer has written $R/SCORED (override: FORCE=1, for a
# partial pull of what has merged so far).
# Run from Git Bash locally:
#   bash cluster/slurm/a2b_pull_treestuned_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
R=${RESULTS_ROOT:-results/linear_subsection_trees_tuned}
TW=${TW:-2000}
EXTRACT=experiments/a2b_extract_tree_importance.py

# md5sum output -> "<path> <md5>" lines (Git Bash may prefix a binary-mode '*')
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

if [ "${FORCE:-0}" != 1 ]; then
  "$SSH" -o BatchMode=yes usc-discovery "test -f $REMOTE/$R/SCORED" \
    || { echo "no $R/SCORED on CARC yet (FORCE=1 pulls what exists)"; exit 1; }
fi

# 1. ship the extractor, check it, run it on the login node
L=$(md5sum "$EXTRACT" | sums)
RM=$(tar czf - "$EXTRACT" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum $EXTRACT" | sums)
[ "$L" = "$RM" ] || { echo "SHIP FAILED for $EXTRACT: local '$L' remote '$RM'"; exit 1; }
echo "shipped $EXTRACT (md5 ${L##* })"
XOUT=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && python -u $EXTRACT --root $R --tw $TW" 2>&1) \
  || { printf '%s\n' "$XOUT"; echo "EXTRACT FAILED on CARC"; exit 1; }
printf '%s\n' "$XOUT" | grep -v -e 'have been reloaded' -e 'python/3' || true
printf '%s\n' "$XOUT" | grep -q 'wrote importance_manifest.csv' || { echo "EXTRACT wrote no manifest"; exit 1; }

# 2. the file list, built on CARC
LIST_CMD="cd $REMOTE && { find $R -maxdepth 1 -type f;
  find $R -mindepth 8 -maxdepth 8 -type f -path '$R/*/*/*/tw$TW/causal_tune_trees/*/*/*' \
    \\( -name 'results_*.csv' -o -name 'tune_trace_*.csv' -o -name 'tune_candidates_*.csv' \
       -o -name 'grid_*.json' -o -name 'MERGED' \\) | grep -v '/chunks/';
  find $R -mindepth 10 -maxdepth 10 -type f -path '$R/*/*/*/tw$TW/chunks/c*/causal_tune_trees/*/*/*' \
    \\( -name 'tune_trace_*.csv' -o -name 'tune_candidates_*.csv' \\);
  find $R/importance -maxdepth 1 -type f 2>/dev/null; } | sort"
mapfile -t FILES < <("$SSH" -o BatchMode=yes usc-discovery "$LIST_CMD" | tr -d '\r')
[ "${#FILES[@]}" -gt 0 ] || { echo "nothing to pull under $R"; exit 1; }
echo "${#FILES[@]} files to pull"
printf '%s\n' "${FILES[@]}" > "$R.pull_list.txt.tmp" 2>/dev/null || { mkdir -p "$R"; printf '%s\n' "${FILES[@]}" > "$R.pull_list.txt.tmp"; }

# 3. tar over ssh (the list travels on stdin of a second ssh), then md5 both sides
mkdir -p "$R"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - -T -" < "$R.pull_list.txt.tmp" | tar xzf -
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs -d '\n' md5sum" < "$R.pull_list.txt.tmp" | sums)
LOCAL_SUMS=$(tr -d '\r' < "$R.pull_list.txt.tmp" | xargs -d '\n' md5sum | sums)
rm -f "$R.pull_list.txt.tmp"
if [ "$REMOTE_SUMS" != "$LOCAL_SUMS" ]; then
  echo "PULL FAILED: md5 differs for:"
  diff <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS") | head -20
  exit 1
fi
echo "pull OK: ${#FILES[@]} files, local md5 == remote md5"
