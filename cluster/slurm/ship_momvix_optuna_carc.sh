#!/bin/bash
# Ship the moments_vix Optuna run to an ISOLATED CARC root (/scratch1/jc_905/harxhar-momvix), so
# nothing other workstreams ship to the main root (/scratch1/jc_905/harxhar-subsection) can change
# this run's code or data while it is queued:
#   1. remote, first time only: data/*.parquet (the 11 bar-panel files the canonical masked run's
#      loader read) and ./pylib_trees (shap 0.51.0) copied from the main root
#   2. local -> remote (tar over native OpenSSH): every tracked src/**/*.py (src/data/loading.py
#      with the moments_vix bucket), the 5 specs the Optuna spec execs / reads, the chunk reducer,
#      the pack / merge sbatch scripts (the canonical run's bytes) and this run's own scripts and
#      task files; FAIL unless every remote md5 equals the local one
#   3. compare (not shipped): local vs the MAIN root's md5 of the same code files (every one must
#      be SAME except src/data/loading.py, whose only change is the new bucket), the 11 data files
#      local vs remote, and the stored references the gate reads (main root's merged live_feasible
#      stage-1 records and paths) vs the local copies the master table used
#   4. remote: the harxhar env imports, the bucket resolves, scripts parse
# Run from Git Bash locally:  bash cluster/slurm/ship_momvix_optuna_carc.sh
# Second / third buckets (one root): PREFIX="momvolvix leanvolvix" REMOTE=/scratch1/jc_905/harxhar-momvolvix #   bash cluster/slurm/ship_momvix_optuna_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
MAIN=/scratch1/jc_905/harxhar-subsection
PREFIX=${PREFIX:-momvix}
REMOTE=${REMOTE:-/scratch1/jc_905/harxhar-momvix}
REF=results/linear_subsection_trees_optuna_mask
DATA=(core_stats ewstock_stats releases spxw_ivslice spxw_mfiv spy_and_sentiment time_categories
      vix_and_voldemand vix_representations vix_rvol vwstock_stats)
mapfile -t SRC < <(git ls-files 'src/*.py' | sort)
FILES=("${SRC[@]}"
  specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees.py specs/causal_tune_linear.py
  experiments/reduce_trees_optuna.py experiments/momvix_optuna_gate.py
  cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch
  cluster/slurm/momvix_gate.sbatch cluster/slurm/submit_momvix_optuna.sh
  $(for p in $PREFIX; do ls cluster/${p}_optuna_tasks_{canary,s1,s1_carc,s1_h2,s2,gate}.txt 2>/dev/null; done || true)
  $(ls cluster/slurm/submit_momvolvix_lean.sh 2>/dev/null || true))
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "missing locally: $f"; exit 1; }; done
for f in "${FILES[@]}"; do
  if grep -q $'\r' "$f" && [[ "$f" == cluster/* ]]; then echo "CRLF in $f (shell scripts / task files must be LF)"; exit 1; fi
done
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }

# 1. remote root, data and pylib_trees (first time only; never overwritten)
"$SSH" -o BatchMode=yes usc-discovery "set -e; mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results
cd $REMOTE
for d in ${DATA[*]}; do [ -f data/\$d.parquet ] || cp -p $MAIN/data/\$d.parquet data/; done
[ -d pylib_trees ] || cp -a $MAIN/pylib_trees .
ls data | wc -l | sed 's/^/data files: /'"

# 2. ship code + compare remote md5 with local
LOCAL_SHIP=$(md5sum "${FILES[@]}" | sums)
REMOTE_SHIP=$(tar czf - "${FILES[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf - && md5sum ${FILES[*]}" | sums) \
  || { echo "SHIP FAILED: tar / ssh / remote md5sum returned non-zero"; exit 1; }
if [ "$LOCAL_SHIP" != "$REMOTE_SHIP" ]; then
  echo "SHIP FAILED: remote md5 != local:"; diff <(printf '%s\n' "$LOCAL_SHIP") <(printf '%s\n' "$REMOTE_SHIP") | head; exit 1
fi
echo "shipped ${#FILES[@]} files to $REMOTE, remote md5 == local"

# 3. comparisons (main root code, data, stored references)
CODE=("${SRC[@]}" specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py
      specs/causal_tune_trees.py specs/causal_tune_linear.py experiments/reduce_trees_optuna.py
      cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch)
MAIN_SUMS=$("$SSH" -o BatchMode=yes usc-discovery "cd $MAIN && for f in ${CODE[*]}; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo MISSING \"\$f\"; fi; done" | sums)
LOCAL_CODE=$(md5sum "${CODE[@]}" | sums)
echo "code, local (= shipped) vs the main root:"
join <(printf '%s\n' "$LOCAL_CODE") <(printf '%s\n' "$MAIN_SUMS") \
  | awk '{s = ($2 == $3) ? "SAME" : "DIFF"; n[s]++; if (s != "SAME") print "  DIFF  " $1} END {printf "  %d SAME, %d DIFF\n", n["SAME"], n["DIFF"]}'
DF=()
for d in "${DATA[@]}"; do DF+=("data/$d.parquet"); done
L_DATA=$(md5sum "${DF[@]}" | sums)
R_DATA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && md5sum ${DF[*]}" | sums)
[ "$L_DATA" = "$R_DATA" ] && echo "data: ${#DF[@]} files, remote md5 == local" \
  || { echo "DATA DIFF:"; diff <(printf '%s\n' "$L_DATA") <(printf '%s\n' "$R_DATA"); exit 1; }
REFS=()
for m in lgbm xgb rf; do
  REFS+=("$REF/stage1/live_feasible/bar1600/$m/tw2000/trials_bar1600.npz")
  for p in tp1 tp5 tp25 tp250 tp1_k10 tp1_k25 tp25_k10 tp25_k25 tp25_q; do
    REFS+=("$REF/paths/$p/live_feasible/bar1600/$m/tw2000/causal_tune_trees/$m/live_feasible/trees_bar1600.npz")
  done
done
L_REF=$(md5sum "${REFS[@]}" | sums)
R_REF=$("$SSH" -o BatchMode=yes usc-discovery "cd $MAIN && md5sum ${REFS[*]}" | sums)
[ "$L_REF" = "$R_REF" ] && echo "stored references: ${#REFS[@]} files, main-root md5 == local" \
  || { echo "REFERENCE DIFF:"; diff <(printf '%s\n' "$L_REF") <(printf '%s\n' "$R_REF") | head; exit 1; }

# 4. remote checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import optuna, shap, lightgbm, xgboost, sklearn, numpy, pandas
print('optuna', optuna.__version__, 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__)
from src.data.loading import get_bucket
print('moments_vix:', get_bucket('moments_vix'))
\"
for p in specs/causal_tune_trees_optuna.py experiments/reduce_trees_optuna.py experiments/momvix_optuna_gate.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p && echo \"parses: \$p\"
done
for s in cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch cluster/slurm/momvix_gate.sbatch cluster/slurm/submit_momvix_optuna.sh; do
  bash -n \$s && echo \"bash -n ok: \$s\"
done" 2>&1 | grep -v -E 'reloaded with a version change|python/3.11.9 => python/3.13.2|^$'
echo "ship OK"
