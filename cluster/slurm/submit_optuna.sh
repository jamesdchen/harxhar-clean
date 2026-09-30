#!/bin/bash
# The Optuna per-bar tree campaign at the 16:00 bar (specs/causal_tune_trees_optuna.py,
# checklist I3): LightGBM / XGBoost / random forest x baseline / live_feasible /
# all_features, SEGMENT = bar1600, TRAIN_WIN = 2000, the de-duplicated per-bar design, and
# (WINDOW_MASK=1, the default since the user decision of 2026-09-29 "Restart with a
# per-window mask") the per-window column mask of src/models/window_mask.py on every fit.
#
#   canary   an array of 3 tasks (the canary bucket x lgbm / xgb / rf, stage 1 on the arm's
#            first CANARY rows) with CANARY_CHECKS=1: the chunk re-run with a different
#            pool size must give bit-identical records, and stage 2 must run on the
#            chunk's rows from its own records.  Each task writes $R/canary_ok_<model>
#            (WRITE_FLAG with %m).  The canary chunks are the fleet's chunk 0 of those arms
#            (DONE -> skipped).
#   s1       stage 1 (a tuning point at every forecast row, 50 TPE trials each), held
#            afterany on the canary and refusing to run unless all three canary flags exist;
#            with S1_ONEWEEK=N the first N task lines (N <= 50, oneweek's per-user submit
#            limit) go to a second array that may also run on the oneweek partition
#   m1       the stage-1 merge (experiments/reduce_trees_optuna.py --stage 1: the
#            completeness gate writes each complete arm's STAGE1_COMPLETE), afterany on s1
#   s2       stage 2 (a refit every session along the 9 paths), afterany on m1; a task
#            refuses to run unless its arm's STAGE1_COMPLETE exists; S2_ONEWEEK as S1_ONEWEEK
#   m2       the stage-2 merge (the untuned layout per path under $R/paths/), afterany on s2
# (afterany + flags rather than afterok: a failed upstream job then makes the held tasks
# exit at once instead of pending forever.)
#
# HISTORY.  The unmasked run (WINDOW_MASK=0, root results/linear_subsection_trees_optuna:
# canary 12479664, s1 12479667, m1 12479668, s2 12479669, m2 12479670) was stopped by the
# user decision above; its outputs are not used.  Two things learned there are built in:
# the canary lacked WRITE_FLAG (the flags were touched by hand after sacct showed all three
# canary tasks COMPLETED with "canary repeat OK" / "canary stage 2 OK"), and 40G per task
# kept the tasks off the memory-bound nodes (every node's memory allocated, cores idle);
# the tasks use ~9 GB (parent 1.5 GB + 20 workers x ~0.35 GB), so MEM is 16G.
#
# Task lines: "<bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>", W = 2000 rows,
# start = W + first OOS row, end = W + last OOS row + 1 (-1 for an arm's last chunk),
# halo = W.  The task files are written by cluster/optuna_make_tasks.py (its header has
# the cost model and the sizing; rows per chunk differ by arm so every task carries about
# the same expected work).
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_optuna.sh
# Override by env on CARC, e.g. S1_ONEWEEK=40 bash cluster/slurm/submit_optuna.sh.
# RESUME_ONLY=1 skips the canary (its flags must exist) -- for resubmitting the stages.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

POOL_CPUS=${POOL_CPUS:-20}         # cores per task = the pool of single-threaded fits (the campaign file: ~20)
MEM=${MEM:-16G}                    # measured ~9 GB RSS per task (see HISTORY)
WINDOW_MASK=${WINDOW_MASK:-1}      # the per-window column mask (user decision 2026-09-29)
# expected walls (cluster/optuna_make_tasks.py cost model, CARC-measured unmasked costs; the
# mask drops columns, so these are upper estimates): stage-1 task ~30 min, stage-2 ~4 min
CANARY_TIME=${CANARY_TIME:-1:30:00}
S1_TIME=${S1_TIME:-2:30:00}        # >= 3x the longest expected stage-1 task; stage 1 resumes from its saved points
S2_TIME=${S2_TIME:-1:00:00}
MERGE_TIME=${MERGE_TIME:-2:00:00}
S1_ONEWEEK=${S1_ONEWEEK:-0}
S2_ONEWEEK=${S2_ONEWEEK:-0}

R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}
CAN=cluster/optuna_tasks_canary.txt
S1T=cluster/optuna_tasks_s1.txt
S2T=cluster/optuna_tasks_s2.txt
for f in specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_trees.py specs/causal_tune_linear.py src/backtest/executor.py src/models/window_mask.py \
         experiments/reduce_trees_optuna.py cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch \
         "$CAN" "$S1T" "$S2T"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_optuna_carc.sh, locally)"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
[ "$S1_ONEWEEK" -le 50 ] && [ "$S2_ONEWEEK" -le 50 ] || { echo "S1_ONEWEEK / S2_ONEWEEK must be <= 50"; exit 1; }
mkdir -p "$R"
FLAGS="$R/canary_ok_lgbm $R/canary_ok_xgb $R/canary_ok_rf"
rm -f "$R/STAGE1_MERGED" "$R/STAGE2_MERGED"
SUBMIT=sbatch
COMMON="RESULTS_ROOT=$R,WINDOW_MASK=$WINDOW_MASK"
DEP_CAN=""
if [ "${RESUME_ONLY:-0}" != 1 ]; then
  rm -f $FLAGS
  CANJ=$($SUBMIT --parsable -J op_canary --array=1-"$(wc -l < "$CAN")" --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
        --time="$CANARY_TIME" \
        --export=ALL,TASKFILE=$CAN,STAGE=tune,$COMMON,CANARY_CHECKS=1,REPEAT_CPUS=10,WRITE_FLAG=$R/canary_ok_%m \
        cluster/slurm/optuna_pack.sbatch)
  DEP_CAN="--dependency=afterany:$CANJ"
else
  CANJ=none
  for f in $FLAGS; do [ -f "$f" ] || { echo "RESUME_ONLY=1 but $f is missing"; exit 1; }; done
fi

# one stage's arrays: the first N lines on main,oneweek (a task file of their own), the rest on main
submit_stage() {  # $1 name, $2 task file, $3 stage, $4 time, $5 oneweek lines, $6 dependency, $7 extra export
  local name=$1 tf=$2 st=$3 tl=$4 n1=$5 dep=$6 extra=$7 ids="" n
  n=$(wc -l < "$tf")
  if [ "$n1" -gt 0 ]; then
    head -n "$n1" "$tf" > "$R/tasks_${name}_oneweek.txt"
    tail -n +"$((n1 + 1))" "$tf" > "$R/tasks_${name}_main.txt"
    ids=$($SUBMIT --parsable -J "op_$name" -p main,oneweek --array=1-"$n1" $dep --cpus-per-task="$POOL_CPUS" \
          --mem="$MEM" --time="$tl" "--export=ALL,TASKFILE=$R/tasks_${name}_oneweek.txt,STAGE=$st,$COMMON$extra" \
          cluster/slurm/optuna_pack.sbatch)
    if [ "$n" -gt "$n1" ]; then
      ids="$ids:$($SUBMIT --parsable -J "op_$name" --array=1-"$((n - n1))" $dep --cpus-per-task="$POOL_CPUS" \
            --mem="$MEM" --time="$tl" "--export=ALL,TASKFILE=$R/tasks_${name}_main.txt,STAGE=$st,$COMMON$extra" \
            cluster/slurm/optuna_pack.sbatch)"
    fi
  else
    ids=$($SUBMIT --parsable -J "op_$name" --array=1-"$n" $dep --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$tl" "--export=ALL,TASKFILE=$tf,STAGE=$st,$COMMON$extra" cluster/slurm/optuna_pack.sbatch)
  fi
  echo "$ids"
}
S1=$(submit_stage s1 "$S1T" tune "$S1_TIME" "$S1_ONEWEEK" "$DEP_CAN" ",CANARY_FLAG=$FLAGS")
M1=$($SUBMIT --parsable -J op_m1 --dependency=afterany:"$S1" --time="$MERGE_TIME" \
      --export=ALL,MERGE_STAGE=1,TASKFILE=$S1T,RESULTS_ROOT=$R cluster/slurm/optuna_merge.sbatch)
S2=$(submit_stage s2 "$S2T" refit "$S2_TIME" "$S2_ONEWEEK" "--dependency=afterany:$M1" ",HOLD_FLAG=$R/STAGE1_MERGED")
M2=$($SUBMIT --parsable -J op_m2 --dependency=afterany:"$S2" --time="$MERGE_TIME" \
      --export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R cluster/slurm/optuna_merge.sbatch)
echo "canary=$CANJ s1=$S1 m1=$M1 s2=$S2 m2=$M2 window_mask=$WINDOW_MASK root=$R ($(wc -l < "$S1T") stage-1 tasks, oneweek $S1_ONEWEEK; $(wc -l < "$S2T") stage-2 tasks, oneweek $S2_ONEWEEK; $POOL_CPUS cpus / $MEM each)" \
  | tee -a logs/submitted_optuna_carc.txt
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.5C %P %R" | grep -E "op_|JOBID" | head -14
