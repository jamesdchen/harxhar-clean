#!/bin/bash
# Ship the moments_vix Optuna run to its Hoffman2 root (/u/scratch/j/jamesdc1/harxhar-momvix):
#   1. remote, first time only: data/*.parquet (the 11 bar-panel files) copied from agent D's root
#      (harxhar-optuna, whose data md5 equals CARC's); agent D's runtime (CARC's Python stack in
#      harxhar-optuna, used read-only by cluster/momvix_optuna_h2_task.sh) checked against its
#      md5 manifest (scratch is purged)
#   2. local -> remote (tar over native OpenSSH): the same code bytes as the CARC root
#      (cluster/slurm/ship_momvix_optuna_carc.sh) + the Hoffman2 scripts and task files; the gate's
#      stored references (the canonical run's merged live_feasible stage-1 records and paths, the
#      local copies whose md5 the CARC ship compared with the CARC main root) into refs/;
#      FAIL unless every remote md5 equals the local one
#   3. data md5 local vs remote; scripts parse; the runtime imports and resolves the bucket
# Nothing is written into agent D's root.
# Run from Git Bash locally:  bash cluster/ship_momvix_optuna_h2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
H2=/u/scratch/j/jamesdc1/harxhar-momvix
RT=/u/scratch/j/jamesdc1/harxhar-optuna
REF=results/linear_subsection_trees_optuna_mask
DATA=(core_stats ewstock_stats releases spxw_ivslice spxw_mfiv spy_and_sentiment time_categories
      vix_and_voldemand vix_representations vix_rvol vwstock_stats)
mapfile -t SRC < <(git ls-files 'src/*.py' | sort)
FILES=("${SRC[@]}"
  specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees.py specs/causal_tune_linear.py
  experiments/reduce_trees_optuna.py experiments/momvix_optuna_gate.py
  cluster/momvix_optuna_h2_task.sh cluster/momvix_optuna_h2_gate.sh cluster/submit_momvix_optuna_h2.sh
  cluster/momvix_optuna_tasks_gate.txt cluster/momvix_optuna_tasks_h2audit.txt cluster/momvix_optuna_tasks_s1_h2.txt
  cluster/momvix_optuna_tasks_s1.txt cluster/momvix_optuna_tasks_s2.txt)
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }; done
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

# 1. remote root, data, runtime check
"$SSH" -o BatchMode=yes hoffman2 "set -e; mkdir -p $H2/data $H2/logs $H2/results $H2/refs
cd $H2
for d in ${DATA[*]}; do [ -f data/\$d.parquet ] || cp -p $RT/data/\$d.parquet data/; done
echo \"data files: \$(ls data | wc -l)\"
cd $RT && md5sum -c --quiet runtime_manifest.md5 && echo \"runtime $RT: \$(wc -l < runtime_manifest.md5) files match their manifest\""

# 2. code + references
LOCAL_SHIP=$(md5sum "${FILES[@]}" | sums)
REMOTE_SHIP=$(tar czf - "${FILES[@]}" \
  | "$SSH" -o BatchMode=yes hoffman2 "cd $H2 && tar xzf - && md5sum ${FILES[*]}" | sums) \
  || { echo "SHIP FAILED: tar / ssh / remote md5sum returned non-zero"; exit 1; }
[ "$LOCAL_SHIP" = "$REMOTE_SHIP" ] || { echo "SHIP FAILED: remote md5 != local:"; diff <(printf '%s\n' "$LOCAL_SHIP") <(printf '%s\n' "$REMOTE_SHIP") | head; exit 1; }
echo "code: ${#FILES[@]} files, Hoffman2 md5 == local"
REFS=()
for m in lgbm xgb rf; do
  REFS+=("stage1/live_feasible/bar1600/$m/tw2000/trials_bar1600.npz")
  for p in tp1 tp5 tp25 tp250 tp1_k10 tp1_k25 tp25_k10 tp25_k25 tp25_q; do
    REFS+=("paths/$p/live_feasible/bar1600/$m/tw2000/causal_tune_trees/$m/live_feasible/trees_bar1600.npz")
  done
done
L_REF=$(cd "$REF" && md5sum "${REFS[@]}" | sums)
R_REF=$( (cd "$REF" && tar czf - "${REFS[@]}") \
  | "$SSH" -o BatchMode=yes hoffman2 "cd $H2/refs && tar xzf - && md5sum ${REFS[*]}" | sums)
[ "$L_REF" = "$R_REF" ] || { echo "REFERENCE SHIP FAILED"; diff <(printf '%s\n' "$L_REF") <(printf '%s\n' "$R_REF") | head; exit 1; }
echo "references: ${#REFS[@]} files in $H2/refs, md5 == local"

# 3. data md5, syntax, runtime import of the shipped code (shap / numba / llvmlite are not imported
#    here: llvmlite does not load under the login node's limits; every task checks them in-process)
DF=()
for d in "${DATA[@]}"; do DF+=("data/$d.parquet"); done
L_DATA=$(md5sum "${DF[@]}" | sums)
R_DATA=$("$SSH" -o BatchMode=yes hoffman2 "cd $H2 && md5sum ${DF[*]}" | sums)
[ "$L_DATA" = "$R_DATA" ] && echo "data: ${#DF[@]} files, Hoffman2 md5 == local" \
  || { echo "DATA DIFF:"; diff <(printf '%s\n' "$L_DATA") <(printf '%s\n' "$R_DATA"); exit 1; }
"$SSH" -o BatchMode=yes hoffman2 "cd $H2 && set -e
for s in cluster/momvix_optuna_h2_task.sh cluster/momvix_optuna_h2_gate.sh cluster/submit_momvix_optuna_h2.sh; do bash -n \$s && echo \"bash -n ok: \$s\"; done
env -i HOME=\$HOME PATH=$RT/carc_env/harxhar/bin:/usr/bin:/bin PYTHONUSERBASE=$RT/carc_usersite PYTHONPATH=$H2:$RT/pylib_trees \
  $RT/carc_env/harxhar/bin/python3.11 -c \"
import numpy, pandas, sklearn, lightgbm, xgboost, optuna
print('runtime:', 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'optuna', optuna.__version__)
from src.data.loading import get_bucket
print('moments_vix:', get_bucket('moments_vix'))
\""
echo "ship OK"
