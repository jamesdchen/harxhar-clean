#!/bin/bash
# Ship the per-bar LSTM spec (with its pool-worker module), its Slurm scripts,
# task files, chunk reducer, stacker and scorer to the CARC root (tar over
# native Windows OpenSSH), and FAIL if any shipped file's remote md5 differs
# from the local one.  The three files the spec READS but the running tuned-tree
# campaign also uses (specs/causal_tune_linear.py: TUNE_PER / VAL_TAIL / EMBARGO
# / SEED / HORIZON / DATA_PATH; specs/causal_tune_trees.py: REFIT_EVERY;
# specs/causal_tune_trees_tuned_jobs.py: val_losses) are NOT re-shipped (a
# re-extract under a running fleet is a needless race): their remote md5 must
# EQUAL the local one, or the script fails.  Then, WITHOUT shipping, compare
# local vs remote md5 of every module the spec and the scorer import and every
# bar-keyed data/*.parquet the loader reads (SAME / DIFF per file; a DIFF is a
# warning, the caller decides).  Then check on the login node that torch
# imports in the harxhar env -- if it does not, install the CPU wheel into
# ./pylib_lstm (pip --target; lstm_pack.sbatch puts it on PYTHONPATH) -- and
# that the shipped scripts parse.
# Run from Git Bash locally:
#   bash cluster/slurm/ship_lstm_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-subsection
FILES=(
  specs/causal_tune_lstm.py
  specs/causal_tune_lstm_jobs.py
  cluster/slurm/lstm_pack.sbatch
  cluster/slurm/lstm_score.sbatch
  cluster/slurm/submit_lstm.sh
  cluster/lstm_tasks_canary.txt
  cluster/lstm_tasks_fleet.txt
  experiments/reduce_lstm_chunks.py
  experiments/build_subsection_lstm_yhat.py
  experiments/score_lstm_subsection.py
)
MUST_MATCH=(
  specs/causal_tune_linear.py
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned_jobs.py
)
for f in "${FILES[@]}" "${MUST_MATCH[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }
done

sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}'; }
show() { awk '{printf "%s  %s\n", substr($2, 1, 12), $1}'; }
# "<path> <md5>" remote vs local -> the paths that differ or are missing remotely
differ() { awk 'NR == FNR {r[$1] = $2; next} !($1 in r) || r[$1] != $2 {print $1}' <(printf '%s\n' "$1") <(printf '%s\n' "$2"); }

# 0. the files the spec reads from the running campaign must already match
LOCAL_MM=$(md5sum "${MUST_MATCH[@]}" | sums)
REMOTE_MM=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${MUST_MATCH[*]}" | sums) \
  || { echo "FAILED: remote md5sum of the must-match files"; exit 1; }
MM_BAD=$(differ "$REMOTE_MM" "$LOCAL_MM")
echo "must match (not shipped):"; printf '%s\n' "$LOCAL_MM" | show
if [ -n "$MM_BAD" ]; then
  echo "FAILED: these files differ on CARC (the spec reads them; not re-shipped under the running fleet):"
  printf '  %s\n' $MM_BAD
  exit 1
fi

# 1. ship, then compare the remote md5 of every shipped file with the local one
LOCAL_SHIP=$(md5sum "${FILES[@]}" | sums)
REMOTE_SHIP=$(tar czf - "${FILES[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" | sums) \
  || { echo "SHIP FAILED: tar / ssh / remote md5sum returned non-zero (see above)"; exit 1; }
echo "local:";  printf '%s\n' "$LOCAL_SHIP" | show
echo "remote:"; printf '%s\n' "$REMOTE_SHIP" | show
SHIP_BAD=$(differ "$REMOTE_SHIP" "$LOCAL_SHIP")
if [ -n "$SHIP_BAD" ]; then
  echo "SHIP FAILED: remote md5 differs from local (or has no remote line) for:"
  printf '  %s\n' $SHIP_BAD
  exit 1
fi

# 2. compare (not shipped): the modules the spec, stacker and scorer import and
# the bar-keyed data the loader reads (the option-chain exports are skipped by
# the loader and not compared)
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
mapfile -t FEAT < <(find src/features -name '*.py' -not -path '*/__pycache__/*' | sort)
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
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

# 3. remote: torch (env first, ./pylib_lstm fallback), then syntax checks
echo
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
if python -c 'import torch' 2>/dev/null; then
  echo 'torch in the harxhar env:'
else
  echo 'torch missing from the harxhar env: installing the CPU wheel into ./pylib_lstm'
  pip install --quiet --target ./pylib_lstm torch --index-url https://download.pytorch.org/whl/cpu
fi
PYTHONPATH=\$PWD:\$PWD/pylib_lstm python -c \"
import sys, torch, numpy, pandas
print('python', sys.version.split()[0], 'torch', torch.__version__, torch.__file__)
print('numpy', numpy.__version__, 'pandas', pandas.__version__)
\"
for p in specs/causal_tune_lstm.py specs/causal_tune_lstm_jobs.py experiments/reduce_lstm_chunks.py experiments/build_subsection_lstm_yhat.py experiments/score_lstm_subsection.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read()); print('parses:', sys.argv[1])\" \$p
done
for s in cluster/slurm/lstm_pack.sbatch cluster/slurm/lstm_score.sbatch cluster/slurm/submit_lstm.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
  echo \"bash -n ok: \$s\"
done"

echo
[ -z "$DEP_WARN" ] || printf '%s\n' "$DEP_WARN"
echo "ship OK: ${#FILES[@]} files, ${#MUST_MATCH[@]} must-match files SAME, $N_SAME deps SAME, $N_DIFF deps DIFF"
