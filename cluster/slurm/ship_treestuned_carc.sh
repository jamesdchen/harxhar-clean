#!/bin/bash
# Ship the per-bar TUNED TREE spec (with its pool-worker module and the two
# specs it execs / reads constants from), its Slurm scripts, task files, chunk
# reducer and scorer to the CARC root
# (tar over native Windows OpenSSH), and FAIL if any shipped file's remote md5
# differs from the local one; then, WITHOUT shipping, compare local vs remote
# md5 of every module the spec and the scorer import and every data/*.parquet
# the loader reads (SAME / DIFF per file; a DIFF is a warning, the caller
# decides); then check on the login node that ./pylib_trees/shap exists (the
# untuned campaign installed shap 0.51.0 there; this script does NOT install
# it), that the tree stack imports, and that the shipped scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_treestuned_carc.sh
# The canary/fleet time and cpu defaults live at the top of
# submit_treestuned.sh (overridable by env on CARC, e.g. POOL_CPUS=8
# GB_TIME=1:00:00 bash ...).
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
# causal_tune_trees.py / causal_tune_linear.py are already on CARC and
# unchanged; shipping them byte-identical is harmless and pins the pair the
# tuned spec execs / reads constants from.
FILES=(
  specs/causal_tune_trees_tuned.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees.py
  specs/causal_tune_linear.py
  cluster/slurm/treestuned_pack.sbatch
  cluster/slurm/treestuned_score.sbatch
  cluster/slurm/submit_treestuned.sh
  cluster/treestuned_tasks_canary.txt
  cluster/treestuned_tasks_gb.txt
  cluster/treestuned_tasks_rf.txt
  experiments/reduce_trees_tuned_chunks.py
  experiments/score_trees_tuned.py
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done

# md5sum output -> "<path> <md5>" lines (Git Bash may prefix a binary-mode '*')
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
# "<path> <md5>" lines -> the display form "<first 12 hex of md5>  <path>"
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

# 2. compare (not shipped): the modules the spec and the scorer import and the
# data the spec reads
DEPS=(
  src/backtest/executor.py
  src/backtest/multi_stage.py
  src/backtest/segmentation.py
  src/data/loading.py
  src/data/options_features.py
  src/diagnostics.py
  src/evaluation/metrics.py
  src/evaluation/diebold_mariano.py
  src/models/reclasso_har.py
  experiments/score_trees_subsection.py
  experiments/build_subsection_tree_yhat.py
  experiments/score_linear_subsection.py
  experiments/score_linear_subsection_causal.py
  experiments/build_subsection_yhat.py
  notebooks/atm_straddle_lib.py
)
# (notebooks/atm_straddle_lib.py: imported by experiments/score_linear_subsection.py.
#  specs/causal_tune_linear.py's own imports are not listed: the tuned spec
#  reads its constants with ast, it does not import it.)
mapfile -t FEAT < <(find src/features -name '*.py' -not -path '*/__pycache__/*' | sort)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' | sort)
DEPS+=("${FEAT[@]}" "${DATA[@]}")
echo
echo "dependencies, local vs remote md5 (${#DEPS[@]} files, not shipped):"
LOCAL_SUMS=$(md5sum "${DEPS[@]}" | sums)
REMOTE_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && for f in ${DEPS[*]}; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo MISSING \"\$f\"; fi; done" \
  | sums)
# "<status>\t<path>" per dependency
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

# 3. remote: shap must already be in pylib_trees; import check, syntax checks
echo
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
if [ ! -d $REMOTE/pylib_trees/shap ]; then
  echo 'FATAL: $REMOTE/pylib_trees/shap is missing -- the untuned campaign installs it (bash cluster/slurm/ship_trees_carc.sh); this script does not'
  exit 1
fi
echo 'pylib_trees/shap present'
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn, numba, cloudpickle
print('python', sys.version.split()[0])
print('shap', shap.__version__, shap.__file__)
print('lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numba', numba.__version__)
\"
for p in specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py experiments/reduce_trees_tuned_chunks.py experiments/score_trees_tuned.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('parses:', sys.argv[1])\" \$p
done
for s in cluster/slurm/treestuned_pack.sbatch cluster/slurm/treestuned_score.sbatch cluster/slurm/submit_treestuned.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
  echo \"bash -n ok: \$s\"
done"

echo
[ -z "$DEP_WARN" ] || printf '%s\n' "$DEP_WARN"
echo "ship OK: ${#FILES[@]} files, $N_SAME deps SAME, $N_DIFF deps DIFF"
