#!/bin/bash
# Ship the per-bar TREE spec, its Slurm scripts, task files, scorer and yhat
# builder to the CARC root (tar over native Windows OpenSSH); then, WITHOUT
# shipping, compare local vs remote md5 of every module the spec imports and
# every data/*.parquet the loader reads (SAME / DIFF per file); then install
# shap into $REMOTE/pylib_trees if absent (isolated, --no-deps; the harxhar env
# carries lightgbm, xgboost, sklearn, numba, cloudpickle) and check on the
# login node that the tree stack imports and the shipped scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_trees_carc.sh
# The canary/fleet time and cpu defaults live at the top of submit_trees.sh
# (overridable by env on CARC, e.g. RF_CPUS=4 RF_TIME=12:00:00 bash ...).
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  specs/causal_tune_trees.py
  cluster/slurm/trees_pack.sbatch
  cluster/slurm/trees_score.sbatch
  cluster/slurm/submit_trees.sh
  cluster/trees_tasks_canary.txt
  cluster/trees_tasks_gb.txt
  cluster/trees_tasks_rf.txt
  experiments/score_trees_subsection.py
  experiments/build_subsection_tree_yhat.py
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done

# 1. ship
echo "local:";  md5sum "${FILES[@]}" | cut -c1-12,34-
tar czf - "${FILES[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && echo remote: && md5sum ${FILES[*]} | cut -c1-12,34-"

# 2. compare (not shipped): the modules the spec imports and the data it reads
DEPS=(
  src/backtest/executor.py
  src/backtest/multi_stage.py
  src/backtest/segmentation.py
  src/data/loading.py
  src/data/options_features.py
  src/diagnostics.py
  src/evaluation/metrics.py
)
mapfile -t FEAT < <(find src/features -name '*.py' -not -path '*/__pycache__/*' | sort)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' | sort)
DEPS+=("${FEAT[@]}" "${DATA[@]}")
echo
echo "dependencies, local vs remote md5 (${#DEPS[@]} files, not shipped):"
LOCAL_SUMS=$(md5sum "${DEPS[@]}" | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && for f in ${DEPS[*]}; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo MISSING \"\$f\"; fi; done" \
  | awk '{p = $2; sub(/^\*/, "", p); print p, $1}')
awk 'NR == FNR {r[$1] = $2; next}
     {
       if (!($1 in r)) s = "DIFF (no remote line)";
       else if (r[$1] == "MISSING") s = "DIFF (missing remote)";
       else if (r[$1] == $2) s = "SAME";
       else s = "DIFF";
       if (s != "SAME") bad++;
       printf "  %-22s %s\n", s, $1
     }
     END {printf "  %d of %d differ\n", bad + 0, FNR}' \
  <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS")

# 3. remote: shap into pylib_trees (if absent), import check, syntax checks
echo
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
if [ ! -d $REMOTE/pylib_trees/shap ]; then
  echo 'installing shap==0.51.0 slicer==0.0.8 into $REMOTE/pylib_trees'
  python -m pip install --target $REMOTE/pylib_trees --no-deps shap==0.51.0 slicer==0.0.8
else
  echo 'pylib_trees/shap present'
fi
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn, numba, cloudpickle
print('python', sys.version.split()[0])
print('shap', shap.__version__, shap.__file__)
print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numba', numba.__version__)
\"
python -c \"import ast; ast.parse(open('specs/causal_tune_trees.py').read()); print('spec parses')\"
for s in cluster/slurm/trees_pack.sbatch cluster/slurm/trees_score.sbatch cluster/slurm/submit_trees.sh; do
  bash -n \$s && echo \"bash -n ok: \$s\"
done"
