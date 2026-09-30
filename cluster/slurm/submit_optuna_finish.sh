#!/bin/bash
# Finish a partial submission of the Optuna campaign (cluster/slurm/submit_optuna.sh).
#
# 2026-09-29 22:06: `S1_ONEWEEK=40 bash cluster/slurm/submit_optuna.sh` submitted the masked
# canary (12480992) and the main-partition stage-1 array (12480993 = lines 41..150 of
# cluster/optuna_tasks_s1.txt, task file $R/tasks_s1_main.txt), but the main,oneweek array
# of lines 1..40 was refused (QOSMaxSubmitJobPerUserLimit on oneweek) and the script stopped
# before the merges and stage 2.  This script submits what is missing, on main only:
#   s1x  the stage-1 lines of EXTRA_TASKFILE (default $R/tasks_s1_oneweek.txt = lines 1..40),
#        held afterany on the canary CANJ, refusing to run without the three canary flags
#   m1   the stage-1 merge over the FULL stage-1 task file, afterany on S1_IDS and s1x
#   s2   stage 2 (all lines of cluster/optuna_tasks_s2.txt), afterany on m1
#   m2   the stage-2 merge, afterany on s2
# Required env: CANJ (the canary job id), S1_IDS (the stage-1 job id(s) already queued,
# colon-separated).  The settings (MEM, POOL_CPUS, times, WINDOW_MASK, root) are
# submit_optuna.sh's defaults.
#
#   cd /scratch1/jc_905/harxhar-subsection && CANJ=12480992 S1_IDS=12480993 bash cluster/slurm/submit_optuna_finish.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL_CPUS=${POOL_CPUS:-20}
MEM=${MEM:-16G}
WINDOW_MASK=${WINDOW_MASK:-1}
S1_TIME=${S1_TIME:-2:30:00}
S2_TIME=${S2_TIME:-1:00:00}
MERGE_TIME=${MERGE_TIME:-2:00:00}
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}
S1T=cluster/optuna_tasks_s1.txt
S2T=cluster/optuna_tasks_s2.txt
EXTRA=${EXTRA_TASKFILE:-$R/tasks_s1_oneweek.txt}
: "${CANJ:?CANJ (the canary job id) is required}"
: "${S1_IDS:?S1_IDS (the queued stage-1 job ids, colon-separated) is required}"
for f in "$S1T" "$S2T" "$EXTRA" cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch; do
  [ -f "$f" ] || { echo "$f missing"; exit 1; }
done
# the extra lines plus the queued array's lines must be the whole stage-1 task file
cat "$EXTRA" "$R/tasks_s1_main.txt" | cmp -s - "$S1T" || { echo "$EXTRA + $R/tasks_s1_main.txt != $S1T"; exit 1; }
FLAGS="$R/canary_ok_lgbm $R/canary_ok_xgb $R/canary_ok_rf"
COMMON="RESULTS_ROOT=$R,WINDOW_MASK=$WINDOW_MASK"
X=$(sbatch --parsable -J op_s1 --array=1-"$(wc -l < "$EXTRA")" --dependency=afterany:"$CANJ" \
      --cpus-per-task="$POOL_CPUS" --mem="$MEM" --time="$S1_TIME" \
      "--export=ALL,TASKFILE=$EXTRA,STAGE=tune,$COMMON,CANARY_FLAG=$FLAGS" cluster/slurm/optuna_pack.sbatch)
M1=$(sbatch --parsable -J op_m1 --dependency=afterany:"$S1_IDS:$X" --time="$MERGE_TIME" \
      "--export=ALL,MERGE_STAGE=1,TASKFILE=$S1T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
S2=$(sbatch --parsable -J op_s2 --array=1-"$(wc -l < "$S2T")" --dependency=afterany:"$M1" \
      --cpus-per-task="$POOL_CPUS" --mem="$MEM" --time="$S2_TIME" \
      "--export=ALL,TASKFILE=$S2T,STAGE=refit,$COMMON,HOLD_FLAG=$R/STAGE1_MERGED" cluster/slurm/optuna_pack.sbatch)
M2=$(sbatch --parsable -J op_m2 --dependency=afterany:"$S2" --time="$MERGE_TIME" \
      "--export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
echo "finish: canary=$CANJ s1=$S1_IDS:$X m1=$M1 s2=$S2 m2=$M2 window_mask=$WINDOW_MASK root=$R" \
  | tee -a logs/submitted_optuna_carc.txt
squeue -u "$USER" -o "%.14i %.12j %.4t %.10M %.5C %P %R" | grep -E "op_|JOBID" | head -12
