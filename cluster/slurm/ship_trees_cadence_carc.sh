#!/bin/bash
# Ship the 16:00 tree cadence campaign (checklist I2: T10, T1, RS10, RS1) to the CARC
# root (tar over native Windows OpenSSH) and FAIL if any shipped file's remote md5
# differs from the local one.  Shipped: the untuned and tuned tree specs (+ the tuned
# pool-worker module and the linear spec the tuned spec reads constants from),
# src/backtest/executor.py (the de-duplicated per-bar design of commit 47f7f9c: every
# 16:00 arm of this campaign is fitted on it), the gates, the two chunk reducers, the
# Slurm scripts and the task files.  Then, WITHOUT shipping, compare local vs remote md5
# of every other module the specs import and every data/*.parquet the loader reads
# (SAME / DIFF; a DIFF is a warning, the caller decides), check that ./pylib_trees/shap
# exists (installed by cluster/slurm/ship_trees_carc.sh), that the tree stack imports
# on the login node, and that the shipped scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_trees_cadence_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/backtest/executor.py
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_linear.py
  experiments/trees_cadence_gates.py
  experiments/trees_cadence_reduce_chunks.py
  experiments/reduce_trees_tuned_chunks.py
  cluster/slurm/trees_cadence_gates.sbatch
  cluster/slurm/trees_cadence_pack.sbatch
  cluster/slurm/treestuned_cadence_pack.sbatch
  cluster/slurm/trees_cadence_merge.sbatch
  cluster/slurm/submit_trees_cadence.sh
  cluster/trees_cadence_tasks_untuned.txt
  cluster/trees_cadence_tasks_rs10.txt
  cluster/trees_cadence_tasks_rs1.txt
  cluster/trees_cadence_tasks_rs_canary.txt
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done
# the shipped design and specs must be the committed ones
if ! git diff --quiet HEAD -- src/backtest/executor.py specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py \
     specs/causal_tune_trees_tuned_jobs.py specs/causal_tune_linear.py; then
  echo "SHIP REFUSED: a spec or the executor differs from HEAD (commit or restore it first)"
  exit 1
fi

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
show() { awk '{printf "%s  %s\n", substr($2, 1, 12), $1}'; }

# 1. ship, then compare the remote md5 of every shipped file with the local one
LOCAL_SHIP=$(md5sum "${FILES[@]}" | sums)
REMOTE_SHIP=$(tar czf - "${FILES[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" | sums) \
  || { echo "SHIP FAILED: tar / ssh / remote md5sum returned non-zero (see above)"; exit 1; }
echo "remote (= local):"; printf '%s\n' "$REMOTE_SHIP" | show
SHIP_BAD=$(awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' \
  <(printf '%s\n' "$REMOTE_SHIP") <(printf '%s\n' "$LOCAL_SHIP"))
if [ -n "$SHIP_BAD" ]; then
  echo "SHIP FAILED: remote md5 differs from local (or has no remote line) for:"
  printf '  %s\n' $SHIP_BAD
  exit 1
fi

# 2. compare (not shipped): the other modules the specs import, and the data the loader reads
DEPS=(
  src/backtest/multi_stage.py
  src/backtest/segmentation.py
  src/data/loading.py
  src/data/options_features.py
  src/diagnostics.py
  src/evaluation/metrics.py
  src/models/reclasso_har.py
)
mapfile -t FEAT < <(find src/features -name '*.py' -not -path '*/__pycache__/*' | sort)
# the option-chain and spot parquets are the notebooks' inputs, not the loader's
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
DEPS+=("${FEAT[@]}" "${DATA[@]}")
LOCAL_SUMS=$(md5sum "${DEPS[@]}" | sums)
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && for f in ${DEPS[*]}; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo MISSING \"\$f\"; fi; done" | sums)
DEP_TABLE=$(awk 'NR == FNR {r[$1] = $2; next}
     {
       if (!($1 in r)) s = "DIFF (no remote line)";
       else if (r[$1] == "MISSING") s = "DIFF (missing remote)";
       else if (r[$1] == $2) s = "SAME";
       else s = "DIFF";
       printf "%s\t%s\n", s, $1
     }' <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
N_SAME=$(printf '%s\n' "$DEP_TABLE" | awk -F '\t' '$1 == "SAME" {n++} END {print n + 0}')
N_DIFF=$(printf '%s\n' "$DEP_TABLE" | awk -F '\t' '$1 != "SAME" {n++} END {print n + 0}')
printf '%s\n' "$DEP_TABLE" | awk -F '\t' '$1 != "SAME" {printf "WARNING: dependency %s on CARC (not shipped): %s\n", $1, $2}'
echo "dependencies: $N_SAME SAME, $N_DIFF DIFF (of ${#DEPS[@]})"

# 3. remote: shap present, the tree stack imports, the shipped scripts parse
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
[ -d $REMOTE/pylib_trees/shap ] || { echo 'FATAL: pylib_trees/shap missing'; exit 1; }
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn
print('python', sys.version.split()[0], 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)
\"
for p in specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py experiments/trees_cadence_gates.py experiments/trees_cadence_reduce_chunks.py experiments/reduce_trees_tuned_chunks.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/trees_cadence_gates.sbatch cluster/slurm/trees_cadence_pack.sbatch cluster/slurm/treestuned_cadence_pack.sbatch cluster/slurm/trees_cadence_merge.sbatch cluster/slurm/submit_trees_cadence.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | grep -v 'reloaded with a version change\|python/3'
echo "ship OK: ${#FILES[@]} files shipped (md5 match), $N_SAME deps SAME, $N_DIFF deps DIFF"
