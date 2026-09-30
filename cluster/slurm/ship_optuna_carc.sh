#!/bin/bash
# Ship the Optuna per-bar tree campaign (specs/causal_tune_trees_optuna.py + its
# worker module, and the specs it execs / reads constants from), its Slurm
# scripts, task files and chunk reducer to the CARC root (tar over native Windows
# OpenSSH), and FAIL if any shipped file's remote md5 differs from the local one;
# then, WITHOUT shipping, compare local vs remote md5 of every module the spec
# imports and every data/*.parquet the loader reads (SAME / DIFF per file; a DIFF
# is a warning, the caller decides).  src/backtest/executor.py IS shipped: it
# carries the de-duplicated per-bar design of commit 47f7f9c (the committed file,
# the one every 16:00 re-run of this campaign needs); then check on the
# login node that optuna / shap (./pylib_trees) / the tree stack import and that
# the shipped scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_optuna_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  src/backtest/executor.py
  src/models/window_mask.py
  specs/causal_tune_trees_optuna.py
  specs/causal_tune_trees_optuna_jobs.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees.py
  specs/causal_tune_linear.py
  cluster/slurm/optuna_pack.sbatch
  cluster/slurm/optuna_merge.sbatch
  cluster/slurm/submit_optuna.sh
  cluster/optuna_tasks_canary.txt
  cluster/optuna_tasks_s1.txt
  cluster/optuna_tasks_s2.txt
  experiments/reduce_trees_optuna.py
  experiments/optuna_chunk_hosts.py
  experiments/optuna_xclass_compare.py
  cluster/slurm/submit_optuna_finish.sh
  cluster/slurm/submit_optuna_xclass.sh
  cluster/optuna_tasks_xclass.txt
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
show() { awk '{printf "%s  %s\n", substr($2, 1, 12), $1}'; }

# 1. ship, then compare the remote md5 of every shipped file with the local one
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

# 2. compare (not shipped): the modules the spec imports and the data it reads
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
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' | sort)
DEPS+=("${FEAT[@]}" "${DATA[@]}")
echo
echo "dependencies, local vs remote md5 (${#DEPS[@]} files, not shipped):"
LOCAL_SUMS=$(md5sum "${DEPS[@]}" | sums)
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && for f in ${DEPS[*]}; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo MISSING \"\$f\"; fi; done" \
  | sums)
DEP_TABLE=$(awk 'NR == FNR {r[$1] = $2; next}
     {
       if (!($1 in r)) s = "DIFF (no remote line)";
       else if (r[$1] == "MISSING") s = "DIFF (missing remote)";
       else if (r[$1] == $2) s = "SAME";
       else s = "DIFF";
       printf "%s\t%s\n", s, $1
     }' \
  <(printf '%s\n' "$REMOTE_SUMS") <(printf '%s\n' "$LOCAL_SUMS"))
printf '%s\n' "$DEP_TABLE" | awk -F '\t' '{printf "  %-22s %s\n", $1, $2}'
N_SAME=$(printf '%s\n' "$DEP_TABLE" | awk -F '\t' '$1 == "SAME" {n++} END {print n + 0}')
N_DIFF=$(printf '%s\n' "$DEP_TABLE" | awk -F '\t' '$1 != "SAME" {n++} END {print n + 0}')
DEP_WARN=$(printf '%s\n' "$DEP_TABLE" \
  | awk -F '\t' '$1 != "SAME" {printf "WARNING: dependency %s on CARC (not shipped; the caller decides): %s\n", $1, $2}')
echo "  $N_DIFF of ${#DEPS[@]} differ"

# 3. remote: imports, syntax checks
echo
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
[ -d $REMOTE/pylib_trees/shap ] || { echo 'FATAL: $REMOTE/pylib_trees/shap is missing'; exit 1; }
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, optuna, shap, lightgbm, xgboost, sklearn
print('python', sys.version.split()[0], 'optuna', optuna.__version__, 'shap', shap.__version__)
print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)
\"
for p in specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py experiments/reduce_trees_optuna.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('parses:', sys.argv[1])\" \$p
done
for s in cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch cluster/slurm/submit_optuna.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
  echo \"bash -n ok: \$s\"
done"

echo
[ -z "$DEP_WARN" ] || printf '%s\n' "$DEP_WARN"
echo "ship OK: ${#FILES[@]} files, $N_SAME deps SAME, $N_DIFF deps DIFF"
