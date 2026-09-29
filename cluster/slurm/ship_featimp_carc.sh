#!/bin/bash
# Ship the 16:00-bar feature-importance script, its three input files (the per-bar
# design + realized variance + permutation units of each bucket, built locally by
# `python experiments/feature_importance_1530.py design` then `inputs`), the Slurm
# scripts and the task files to the CARC root (tar over native Windows OpenSSH), and
# FAIL if any shipped file's remote md5 differs from the local one; then check on the
# login node that ./pylib_trees/shap exists (installed by the untuned tree campaign;
# not installed here), that the tree stack imports, and that the shipped scripts parse.
# The trees stage reads nothing but its input file (no src/ import, no data/ read).
# Run from Git Bash locally:
#   bash cluster/slurm/ship_featimp_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  experiments/feature_importance_1530_trees.py
  cluster/slurm/featimp_pack.sbatch
  cluster/slurm/featimp_collect.sbatch
  cluster/slurm/submit_featimp.sh
  cluster/featimp_tasks_canary.txt
  cluster/featimp_tasks.txt
  results/feature_importance_1530/_work/input_baseline.npz
  results/feature_importance_1530/_work/input_live_feasible.npz
  results/feature_importance_1530/_work/input_all_features.npz
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
show() { awk '{printf "%s  %s\n", substr($2, 1, 12), $1}'; }

LOCAL_SHIP=$(md5sum "${FILES[@]}" | sums)
REMOTE_SHIP=$(tar czf - "${FILES[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" | sums) \
  || { echo "SHIP FAILED: tar / ssh / remote md5sum returned non-zero (see above)"; exit 1; }
echo "local:";  printf '%s\n' "$LOCAL_SHIP" | show
echo "remote:"; printf '%s\n' "$REMOTE_SHIP" | show
SHIP_BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' \
  <(printf '%s\n' "$REMOTE_SHIP") <(printf '%s\n' "$LOCAL_SHIP"))
if [ -n "$SHIP_BAD" ]; then
  echo "SHIP FAILED: remote md5 differs from local (or has no remote line) for:"
  printf '  %s\n' $SHIP_BAD
  exit 1
fi

"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
if [ ! -d $REMOTE/pylib_trees/shap ]; then
  echo 'FATAL: $REMOTE/pylib_trees/shap is missing (bash cluster/slurm/ship_trees_carc.sh installs it); this script does not'
  exit 1
fi
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn, numpy, scipy
print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)
print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)
\"
python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('parses:', sys.argv[1])\" experiments/feature_importance_1530_trees.py
for s in cluster/slurm/featimp_pack.sbatch cluster/slurm/featimp_collect.sbatch cluster/slurm/submit_featimp.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
  echo \"bash -n ok: \$s\"
done"
echo "ship OK: ${#FILES[@]} files"
