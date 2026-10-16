#!/bin/bash
# xgbtop (user decision 2026-10-07): ship the per-bar XGBoost re-run with the top dev-span entry (xgb_opt_00) -- the
# fixed-settings rungs (cluster/slurm/submit_trees_xgbtop.sh) and the re-tuning rungs (random search and Optuna) -- to a
# CARC deployment root of its own, /scratch1/jc_905/harxhar-xgbtop.  Copy of cluster/slurm/ship_trees_rfdev_carc.sh /
# ship_trees_rfp163_carc.sh with this run's files:
#   * the campaign's code from the COMMIT (git archive HEAD, LF blobs): src/ (but src/data/loading.py), the tree, tuned,
#     Optuna and linear specs and their job modules, the unedited reducers / kept-count script; REFUSES when one differs
#     from HEAD in the worktree;
#   * from the WORKTREE: src/data/loading.py (added comment / SUBGROUPS lines only: checked), the generator and the two
#     generated specs (specs/causal_tune_trees_xgbtop.py from cluster/xgbtop_xgb_config.json = xgb_opt_00;
#     specs/causal_tune_trees_xgb06.py from cluster/xgbtop_shipped_xgb.json = xgb_opt_06, for the gate), the wrappers,
#     gate / pack / merge / submit scripts and task files (the phase-2 ones when present);
#   * ref_tables/: results/spxw_pnl/'s published per-bar XGBoost tables (yhat_subtree_daily_<set>_xgb,
#     yhat_subtree_xgb_<set>; the gate's table-cell check), read only;
#   * data copied on the cluster from harxhar-mask where the md5 equals the local file; pylib_trees likewise.
# FAILS unless every shipped file's remote md5 equals the local source.  Re-running it adds / refreshes files; a file a
# queued job uses must not change between ships (the md5 manifest of each ship is kept in logs/).
# Run from Git Bash locally (no conda env needed):   bash cluster/slurm/ship_trees_xgbtop_carc.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=/c/Windows/System32/OpenSSH/ssh.exe
REMOTE=/scratch1/jc_905/harxhar-xgbtop
MASK=/scratch1/jc_905/harxhar-mask

mapfile -t SRC < <(git ls-tree -r --name-only HEAD src | grep -v '/__pycache__/' | grep -v -x 'src/data/loading.py')
FROM_HEAD=(
  "${SRC[@]}"
  specs/causal_tune_trees.py
  specs/causal_tune_trees_tuned.py
  specs/causal_tune_trees_tuned_jobs.py
  specs/causal_tune_trees_optuna.py
  specs/causal_tune_trees_optuna_jobs.py
  specs/causal_tune_linear.py
  experiments/trees_cadence_reduce_chunks.py
  experiments/reduce_trees_tuned_chunks.py
  experiments/reduce_trees_optuna.py
  experiments/trees_mask_kept_counts.py
)
FROM_WORKTREE=(
  src/data/loading.py
  experiments/trees_momvix_gate.py
  experiments/trees_xgbtop_make_specs.py
  experiments/trees_xgbtop_gate.py
  experiments/trees_xgbtop_xcpu_compare.py
  specs/causal_tune_trees_xgbtop.py
  specs/causal_tune_trees_xgb06.py
  cluster/xgbtop_xgb_config.json
  cluster/xgbtop_shipped_xgb.json
  cluster/slurm/trees_xgbtop_pack.sbatch
  cluster/slurm/trees_xgbtop_gates.sbatch
  cluster/slurm/trees_xgbtop_merge.sbatch
  cluster/slurm/submit_trees_xgbtop.sh
  cluster/slurm/ship_trees_xgbtop_carc.sh
  cluster/trees_xgbtop_gate_lines.txt
  cluster/trees_xgbtop_tasks_untuned.txt
  cluster/trees_xgbtop_tasks_xcpu.txt
  cluster/trees_rfdev_tasks_none.txt
)
# phase 2 (re-tuning rungs), shipped when present
for f in specs/causal_tune_trees_tuned_xgbtop.py specs/causal_tune_trees_optuna_xgbtop.py \
         experiments/trees_xgbtop_tuned_gate.py experiments/trees_xgbtop_optuna_gate.py \
         cluster/slurm/treestuned_xgbtop_pack.sbatch cluster/slurm/trees_xgbtop_tuned_gates.sbatch \
         cluster/slurm/trees_xgbtop_tuned_merge.sbatch cluster/slurm/optuna_xgbtop_pack.sbatch \
         cluster/slurm/optuna_xgbtop_gate.sbatch cluster/slurm/optuna_xgbtop_merge.sbatch \
         cluster/slurm/submit_trees_xgbtop_optuna.sh \
         cluster/trees_xgbtop_tasks_rs10.txt cluster/trees_xgbtop_tasks_rs1.txt cluster/trees_xgbtop_tasks_rs_canary.txt \
         cluster/trees_xgbtop_tuned_gate_lines.txt \
         cluster/xgbtop_optuna_tasks_gate.txt cluster/xgbtop_optuna_tasks_canary.txt \
         cluster/xgbtop_optuna_tasks_s1.txt cluster/xgbtop_optuna_tasks_s2.txt; do
  [ -f "$f" ] && FROM_WORKTREE+=("$f")
done
TABLES=()
for b in baseline all_features lean_vol_vix; do TABLES+=("yhat_subtree_daily_${b}_xgb.parquet" "yhat_subtree_xgb_${b}.parquet"); done
for f in "${FROM_HEAD[@]}"; do
  git cat-file -e "HEAD:$f" 2>/dev/null || { echo "SHIP REFUSED: $f is not in HEAD"; exit 1; }
done
if ! git diff --quiet HEAD -- "${FROM_HEAD[@]}"; then
  echo "SHIP REFUSED: a file shipped from HEAD differs from HEAD in the worktree:"
  git diff --stat HEAD -- "${FROM_HEAD[@]}"
  exit 1
fi
NUM=$(git diff --numstat HEAD -- src/data/loading.py | cut -f2)
[ "$NUM" = 0 ] || { echo "SHIP REFUSED: src/data/loading.py removes $NUM lines of HEAD"; exit 1; }
BADADD=$(git diff -U0 HEAD -- src/data/loading.py | grep '^+[^+]' | grep -v -E '^\+#|^\+SUBGROUPS\["[a-z_]+"\] = ' || true)
[ -z "$BADADD" ] || { echo "SHIP REFUSED: added loading.py lines other than comments / SUBGROUPS lines:"; echo "$BADADD"; exit 1; }
for f in "${FROM_WORKTREE[@]}"; do
  [ -f "$f" ] || { echo "SHIP REFUSED: $f missing"; exit 1; }
  case "$f" in src/data/loading.py) continue ;; esac  # the worktree's CRLF copy: shipped as LF below
  ! grep -q $'\r' "$f" || { echo "SHIP REFUSED: $f has CRLF line endings"; exit 1; }
done
for t in "${TABLES[@]}"; do [ -f "results/spxw_pnl/$t" ] || { echo "SHIP REFUSED: results/spxw_pnl/$t missing"; exit 1; }; done
mapfile -t DATA < <(find data -maxdepth 1 -name '*.parquet' -not -name 'spxw_chain.parquet' -not -name 'spxw_spot.parquet' | sort)
sums() { awk '{p = $2; sub(/^\*/, "", p); print p, $1}' | sort; }
quiet() { grep -v "reloaded with a version change\|python/3\|^$" || true; }
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
# src/data/loading.py as LF (the worktree copy may be CRLF under core.autocrlf)
mkdir -p "$STAGE/wt/src/data" "$STAGE/ref_tables"
tr -d '\r' < src/data/loading.py > "$STAGE/wt/src/data/loading.py"
for t in "${TABLES[@]}"; do cp -p "results/spxw_pnl/$t" "$STAGE/ref_tables/$t"; done

"$SSH" -o BatchMode=yes usc-discovery "mkdir -p $REMOTE/data $REMOTE/logs $REMOTE/results $REMOTE/ref_tables" 2>&1 | quiet

# 1. data: copy on the cluster where harxhar-mask's md5 matches, upload the rest
LOCAL_DATA=$(md5sum "${DATA[@]}" | sums)
OLD_DATA=$("$SSH" -o BatchMode=yes usc-discovery "cd $MASK && md5sum data/*.parquet 2>/dev/null" | sums || true)
UPLOAD=()
CP=""
for f in "${DATA[@]}"; do
  l=$(printf '%s\n' "$LOCAL_DATA" | awk -v f="$f" '$1 == f {print $2}')
  o=$(printf '%s\n' "$OLD_DATA" | awk -v f="$f" '$1 == f {print $2}')
  if [ -n "$o" ] && [ "$o" = "$l" ]; then CP="$CP [ -f $REMOTE/$f ] || cp -p $MASK/$f $REMOTE/$f;"; else UPLOAD+=("$f"); fi
done
[ -z "$CP" ] || "$SSH" -o BatchMode=yes usc-discovery "$CP true"
if [ "${#UPLOAD[@]}" -gt 0 ]; then
  echo "uploading ${#UPLOAD[@]} data files: ${UPLOAD[*]}"
  tar czf - "${UPLOAD[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xzf -"
fi
"$SSH" -o BatchMode=yes usc-discovery "[ -d $REMOTE/pylib_trees/shap ] || cp -rp $MASK/pylib_trees $REMOTE/pylib_trees" 2>&1 | quiet

# 2. code: HEAD blobs, then the worktree files (loading.py from the LF stage), then the reference tables
git -c core.autocrlf=false -c core.eol=lf archive --format=tar HEAD "${FROM_HEAD[@]}" \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
WT=()
for f in "${FROM_WORKTREE[@]}"; do [ "$f" = src/data/loading.py ] || WT+=("$f"); done
tar cf - "${WT[@]}" | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
tar cf - -C "$STAGE/wt" src/data/loading.py | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"
tar cf - -C "$STAGE" ref_tables | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && tar xf -"

# 3. md5: HEAD files against HEAD's blobs, worktree files / tables / data against the local files
LOCAL_ALL=$( { for f in "${FROM_HEAD[@]}"; do printf '%s %s\n' "$f" "$(git show "HEAD:$f" | md5sum | cut -d' ' -f1)"; done
               md5sum "${WT[@]}" "${DATA[@]}" | sums
               printf 'src/data/loading.py %s\n' "$(md5sum < "$STAGE/wt/src/data/loading.py" | cut -d' ' -f1)"
               for t in "${TABLES[@]}"; do printf 'ref_tables/%s %s\n' "$t" "$(md5sum < "$STAGE/ref_tables/$t" | cut -d' ' -f1)"; done; } | sort)
REMOTE_ALL=$( { printf '%s\n' "${FROM_HEAD[@]}" "${FROM_WORKTREE[@]}" "${DATA[@]}"; for t in "${TABLES[@]}"; do echo "ref_tables/$t"; done; } \
  | "$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && xargs md5sum" | sums)
BAD=$(diff <(printf '%s\n' "$LOCAL_ALL") <(printf '%s\n' "$REMOTE_ALL") || true)
if [ -n "$BAD" ]; then
  echo "SHIP FAILED: remote md5 differs:"; printf '%s\n' "$BAD"; exit 1
fi
echo "md5 OK: $(printf '%s\n' "$LOCAL_ALL" | wc -l) files (${#FROM_HEAD[@]} from HEAD $(git rev-parse --short HEAD), ${#FROM_WORKTREE[@]} from the worktree, ${#TABLES[@]} tables, ${#DATA[@]} data)"
mkdir -p logs
printf '%s\n' "$LOCAL_ALL" > "logs/ship_trees_xgbtop_carc_manifest_$(date +%Y%m%d_%H%M%S).txt"
EXTRA=$("$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && ls data/*.parquet" | grep -v -x -F -f <(printf '%s\n' "${DATA[@]}") || true)
[ -z "$EXTRA" ] || { echo "FAILED: remote data/ holds files the local set does not: $EXTRA"; exit 1; }
# against the masked run's deployment (informational; the gate decides)
MASK_SUMS=$(printf '%s\n' "${FROM_HEAD[@]}" src/data/loading.py | "$SSH" -o BatchMode=yes usc-discovery "cd $MASK && xargs md5sum 2>/dev/null" | sums || true)
echo "files differing from $MASK (or absent there):"
join -a1 <(printf '%s\n' "$REMOTE_ALL") <(printf '%s\n' "$MASK_SUMS") | awk 'NF < 3 || $2 != $3 {print "  " $1}' \
  | grep -v -e '^  data/' -e 'xgbtop' -e 'xgb06' -e 'ref_tables/' || true

# 4. login node: libraries, the bucket, the generated specs, parse checks
"$SSH" -o BatchMode=yes usc-discovery "cd $REMOTE && module load conda >/dev/null 2>&1 && source activate harxhar && set -e
PYTHONPATH=\$PWD:\$PWD/pylib_trees python -c \"
import sys, shap, lightgbm, xgboost, sklearn, optuna
from src.data.loading import get_bucket
print('python', sys.version.split()[0], 'shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__, 'optuna', optuna.__version__)
print('lean_vol_vix:', get_bucket('lean_vol_vix'))
\"
python experiments/trees_xgbtop_make_specs.py --check
python experiments/trees_xgbtop_make_specs.py --config cluster/xgbtop_shipped_xgb.json --token xgb06 --check
for p in specs/*.py experiments/*.py; do
  python -c \"import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())\" \$p
done
for s in cluster/slurm/*xgbtop*.sbatch cluster/slurm/submit_trees_xgbtop*.sh; do
  bash -n \$s || { echo \"bash -n FAILED: \$s\"; exit 1; }
done
echo 'remote checks ok'" 2>&1 | quiet
echo "ship OK -> usc-discovery:$REMOTE"
