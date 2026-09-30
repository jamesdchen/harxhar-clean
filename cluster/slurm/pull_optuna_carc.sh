#!/bin/bash
# Pull the merged Optuna per-bar tree campaign (specs/causal_tune_trees_optuna.py, CARC root
# results/linear_subsection_trees_optuna_mask) into the same local path, WITHOUT the per-chunk
# folders (chunks/, canary_*):
#   every regular file directly under $R (reduce gates and logs, flags); per arm the merged
#   stage-1 records ($R/stage1/<bucket>/<seg>/<model>/tw<TW>/trials_<seg>.npz, STAGE1_COMPLETE);
#   per path and arm the merged untuned layout ($R/paths/<path>/.../results_<seg>.csv,
#   trees_<seg>.npz, MERGED).  Tar over native Windows OpenSSH, then FAIL unless every
#   pulled file's local md5 equals its remote md5.
# Refuses to run before the stage-2 merge wrote $R/STAGE2_MERGED (FORCE=1 pulls what exists).
# Run from Git Bash locally:
#   bash cluster/slurm/pull_optuna_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}
TW=${TW:-2000}

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

if [ "${FORCE:-0}" != 1 ]; then
  "$SSH" -o BatchMode=yes usc-discovery "test -f $REMOTE/$R/STAGE2_MERGED" \
    || { echo "no $R/STAGE2_MERGED on CARC yet (FORCE=1 pulls what exists)"; exit 1; }
fi
LIST_CMD="cd $REMOTE && { find $R -maxdepth 1 -type f; find $R/xclass -maxdepth 1 -type f 2>/dev/null;
  find $R/stage1 -mindepth 5 -maxdepth 5 -type f \\( -name 'trials_*.npz' -o -name 'STAGE1_COMPLETE' \\) 2>/dev/null;
  find $R/paths -mindepth 9 -maxdepth 9 -type f -path '$R/paths/*/*/*/*/tw$TW/causal_tune_trees/*/*/*' \
    \\( -name 'results_*.csv' -o -name 'trees_*.npz' -o -name 'MERGED' \\) 2>/dev/null; } | sort"
mapfile -t FILES < <("$SSH" -o BatchMode=yes usc-discovery "$LIST_CMD" | tr -d '\r')
[ "${#FILES[@]}" -gt 0 ] || { echo "nothing to pull under $R"; exit 1; }
echo "${#FILES[@]} files to pull"
mkdir -p "$R"
LIST="$R/.pull_list.txt.tmp"
printf '%s\n' "${FILES[@]}" > "$LIST"
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar czf - -T -" < "$LIST" | tar xzf -
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs -d '\n' md5sum" < "$LIST" | sums)
LOCAL_SUMS=$(tr -d '\r' < "$LIST" | xargs -d '\n' md5sum | sums)
rm -f "$LIST"
if [ "$REMOTE_SUMS" != "$LOCAL_SUMS" ]; then
  echo "PULL FAILED: md5 differs for:"
  diff <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS") | head -20
  exit 1
fi
echo "pull OK: ${#FILES[@]} files, local md5 == remote md5"
