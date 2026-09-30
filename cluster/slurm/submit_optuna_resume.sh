#!/bin/bash
# Resume part of the Optuna campaign after a stage-1 task hit its time limit.
#
# 2026-09-30 00:20: the masked run's stage-1 chunk all_features x lgbm c10 (40 tuning points,
# task 12480998_39, time limit lowered to 1:00 while pending) had 39 points saved and one
# heavy-tail point (the chunk's slowest had taken 2,653 s) still running at 57 min, so it
# times out; that arm then has no STAGE1_COMPLETE and its stage-2 tasks refuse.  This script
# re-runs only what is missing, chained after the first submission so nothing runs twice:
#   s1r  the lines of S1R (the unfinished chunk; the spec resumes from its saved points, so
#        one point is computed), afterany on the first submission's stage-1 arrays
#   m1r  the stage-1 merge over the FULL stage-1 task file, afterany on s1r
#   s2r  the lines of S2R (that arm's stage-2 chunks), afterany on m1r and on the first
#        stage-2 array (a first-run task of the same chunk either did it -- DONE, skipped --
#        or refused, so no chunk runs twice at once)
#   m2r  the stage-2 merge over the FULL stage-2 task file, afterany on s2r and the first m2
# Required env: AFTER_S1 (first stage-1 job ids, colon-separated), AFTER_S2 (first stage-2
# job id), AFTER_M2 (first stage-2 merge job id).  S1_CPUS: cores of the resumed stage-1
# task (default 4: one tuning point is left, and a smaller task schedules sooner).
#   cd /scratch1/jc_905/harxhar-subsection && AFTER_S1=12480993:12480998 AFTER_S2=12481000 \
#     AFTER_M2=12481001 bash cluster/slurm/submit_optuna_resume.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL_CPUS=${POOL_CPUS:-20}
S1_CPUS=${S1_CPUS:-4}
MEM=${MEM:-16G}
WINDOW_MASK=${WINDOW_MASK:-1}
S1_TIME=${S1_TIME:-2:00:00}
S2_TIME=${S2_TIME:-1:00:00}
MERGE_TIME=${MERGE_TIME:-2:00:00}
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask}
S1T=cluster/optuna_tasks_s1.txt
S2T=cluster/optuna_tasks_s2.txt
S1R=${S1R:-cluster/optuna_tasks_s1_resume.txt}
S2R=${S2R:-cluster/optuna_tasks_s2_resume.txt}
: "${AFTER_S1:?}" "${AFTER_S2:?}" "${AFTER_M2:?}"
for f in "$S1T" "$S2T" "$S1R" "$S2R" cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch; do
  [ -f "$f" ] || { echo "$f missing"; exit 1; }
done
grep -qxFf "$S1R" "$S1T" && grep -qxFf "$S2R" "$S2T" || { echo "resume lines are not lines of the task files"; exit 1; }
COMMON="RESULTS_ROOT=$R,WINDOW_MASK=$WINDOW_MASK"
S1=$(sbatch --parsable -J op_s1r --array=1-"$(wc -l < "$S1R")" --dependency=afterany:"$AFTER_S1" \
      --cpus-per-task="$S1_CPUS" --mem="$MEM" --time="$S1_TIME" \
      "--export=ALL,TASKFILE=$S1R,STAGE=tune,$COMMON" cluster/slurm/optuna_pack.sbatch)
M1=$(sbatch --parsable -J op_m1r --dependency=afterany:"$S1" --time="$MERGE_TIME" \
      "--export=ALL,MERGE_STAGE=1,TASKFILE=$S1T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
S2=$(sbatch --parsable -J op_s2r --array=1-"$(wc -l < "$S2R")" --dependency=afterany:"$M1:$AFTER_S2" \
      --cpus-per-task="$POOL_CPUS" --mem="$MEM" --time="$S2_TIME" \
      "--export=ALL,TASKFILE=$S2R,STAGE=refit,$COMMON,HOLD_FLAG=$R/STAGE1_MERGED" cluster/slurm/optuna_pack.sbatch)
M2=$(sbatch --parsable -J op_m2r --dependency=afterany:"$S2:$AFTER_M2" --time="$MERGE_TIME" \
      "--export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
echo "resume: s1r=$S1 m1r=$M1 s2r=$S2 m2r=$M2 (after s1 $AFTER_S1, s2 $AFTER_S2, m2 $AFTER_M2)" | tee -a logs/submitted_optuna_carc.txt
squeue -u "$USER" -o "%.14i %.10j %.4t %.10M %.5C %P %R" | grep -E "op_|JOBID" | head -12
