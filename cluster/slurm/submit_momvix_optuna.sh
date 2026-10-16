#!/bin/bash
# The Optuna per-bar trees on the moments_vix bucket at the 16:00 bar (user request 2026-10-06):
# the masked Optuna campaign (cluster/slurm/submit_optuna.sh, root
# results/linear_subsection_trees_optuna_mask) re-run on one new input set, in this ISOLATED root
# (/scratch1/jc_905/harxhar-momvix, filled by cluster/slurm/ship_momvix_optuna_carc.sh: the same
# spec / pack / merge bytes as the canonical run, plus src/data/loading.py with the bucket), every
# task pinned to the canonical CPU class (--constraint=epyc-7513), WINDOW_MASK=1, 50 trials.
#
#   bash cluster/slurm/submit_momvix_optuna.sh start
#       gate    array of 3 (cluster/slurm/momvix_gate.sbatch): stored live_feasible tuning points
#               and refits re-computed here and compared bit for bit -> $R/GATE_OK_<model>
#       canary  array of 3 (moments_vix x lgbm / xgb / rf, stage 1 on OOS rows [0, 20) = the
#               arm's chunk 0) with CANARY_CHECKS=1 (a repeat at 10 processes must be
#               bit-identical; stage 2 on the chunk's own records) -> $R/canary_ok_<model>
#       s1_<m>  per model, the CARC stage-1 lines of cluster/momvix_optuna_tasks_s1_carc.txt,
#               held afterany on gate + canary and refusing to run unless BOTH of that model's
#               flags exist (afterany + flags: a failed gate makes the held tasks exit at once
#               instead of pending forever -- jobs cannot be cancelled from this session)
#   TASKS=<file> bash cluster/slurm/submit_momvix_optuna.sh extra
#       more stage-1 lines on CARC (e.g. the Hoffman2 share when that gate fails), per model,
#       each requiring its model's two flags (which must exist already)
#   [DEP=afterany:<s1 job ids>] S2_MODELS="lgbm xgb rf" bash cluster/slurm/submit_momvix_optuna.sh finish
#       m1 (stage-1 merge over the FULL task file cluster/momvix_optuna_tasks_s1.txt: CARC and
#       Hoffman2 chunks, the latter relayed into this tree) -> s2 (the S2_MODELS lines of
#       cluster/momvix_optuna_tasks_s2.txt; HOLD_FLAG STAGE1_MERGED) -> m2 (stage-2 merge over
#       the full s2 task file)
#   bash cluster/slurm/submit_momvix_optuna.sh merge2
#       the stage-2 merge alone (after Hoffman2 stage-2 chunks were relayed in)
# Every mode refuses a second submission of itself (logs/submitted_<prefix>_<mode>*.txt).
# Second bucket (2026-10-06): PREFIX=momvolvix BUCKET=moments_vol_vix JP=mvv in the root
# /scratch1/jc_905/harxhar-momvolvix (task files cluster/momvolvix_optuna_tasks_*.txt, results root
# results/linear_subsection_trees_optuna_mask_momvolvix); the defaults are the moments_vix run's.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE=${1:?usage: submit_momvix_optuna.sh start|extra|finish|merge2}
# 2026-10-06 17:26: every moments_vol_vix task failed in seconds ("No module named 'optuna'") because
# the submitting shell had the harxhar conda env active: --export=ALL handed it to the jobs, whose own
# `module load conda; source activate harxhar` then left a python without optuna.  Submit from a
# clean shell only.
if [ -n "${CONDA_DEFAULT_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "a conda env is active in this shell (${CONDA_DEFAULT_ENV:-}${CONDA_PREFIX:+ $CONDA_PREFIX}): sbatch --export=ALL would hand it to the jobs; submit from a clean shell"
  exit 1
fi
CLASS=${CLASS:-epyc-7513}
POOL_CPUS=${POOL_CPUS:-20}
MEM=${MEM:-16G}
GATE_TIME=${GATE_TIME:-1:30:00}
CANARY_TIME=${CANARY_TIME:-1:30:00}
S1_TIME=${S1_TIME:-2:30:00}
S2_TIME=${S2_TIME:-1:30:00}
MERGE_TIME=${MERGE_TIME:-2:00:00}
PREFIX=${PREFIX:-momvix}
BUCKET=${BUCKET:-moments_vix}
JP=${JP:-mv}   # job-name prefix
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask_$PREFIX}
REF=/scratch1/jc_905/harxhar-subsection/results/linear_subsection_trees_optuna_mask
CAN=cluster/${PREFIX}_optuna_tasks_canary.txt
S1T=cluster/${PREFIX}_optuna_tasks_s1.txt
S1C=cluster/${PREFIX}_optuna_tasks_s1_carc.txt
S2T=cluster/${PREFIX}_optuna_tasks_s2.txt
GT=cluster/${PREFIX}_optuna_tasks_gate.txt
[ "$PWD" != /scratch1/jc_905/harxhar-subsection ] || { echo "run from the isolated root, not the main one"; exit 1; }
for f in specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_trees.py specs/causal_tune_linear.py src/backtest/executor.py src/models/window_mask.py \
         src/data/loading.py experiments/reduce_trees_optuna.py experiments/momvix_optuna_gate.py \
         cluster/slurm/optuna_pack.sbatch cluster/slurm/optuna_merge.sbatch cluster/slurm/momvix_gate.sbatch \
         "$CAN" "$S1T" "$S1C" "$S2T" "$GT"; do
  [ -f "$f" ] || { echo "$f missing: run cluster/slurm/ship_momvix_optuna_carc.sh locally first"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
grep -q "\"$BUCKET\"" src/data/loading.py || { echo "src/data/loading.py has no $BUCKET bucket"; exit 1; }
awk -v b="$BUCKET" '$1 != b {bad = 1} END {exit bad}' "$CAN" "$S1T" "$S1C" "$S2T" || { echo "a task line is not of bucket $BUCKET"; exit 1; }
STAMP="logs/submitted_${PREFIX}_${MODE}${TASKS:+_$(basename "$TASKS" .txt)}${S2_MODELS:+_${S2_MODELS// /_}}.txt"
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
mkdir -p "$R"
COMMON="RESULTS_ROOT=$R,WINDOW_MASK=1"
PIN="--constraint=$CLASS"

s1_arrays() {  # $1 task file, $2 dependency ("" or --dependency=...): one array per model
  local tf=$1 dep=$2 ids="" m n j sub
  for m in lgbm xgb rf; do
    sub="$R/tasks_$(basename "$tf" .txt)_$m.txt"
    awk -v m="$m" '$2 == m' "$tf" > "$sub"
    n=$(wc -l < "$sub")
    [ "$n" -gt 0 ] || continue
    j=$(sbatch --parsable -J "${JP}_s1_$m" $PIN --array=1-"$n" $dep --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$S1_TIME" "--export=ALL,TASKFILE=$sub,STAGE=tune,$COMMON,CANARY_FLAG=$R/canary_ok_$m $R/GATE_OK_$m" \
          cluster/slurm/optuna_pack.sbatch)
    ids="$ids $m:$j($n)"
  done
  echo "$ids"
}

case "$MODE" in
  start)
    rm -f "$R"/canary_ok_* "$R"/GATE_OK_*
    G=$(sbatch --parsable -J ${JP}_gate $PIN --array=1-"$(wc -l < "$GT")" --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$GATE_TIME" "--export=ALL,RESULTS_ROOT=$R,REF_ROOT=$REF,GATE_TASKFILE=$GT" cluster/slurm/momvix_gate.sbatch)
    C=$(sbatch --parsable -J ${JP}_canary $PIN --array=1-"$(wc -l < "$CAN")" --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$CANARY_TIME" \
          "--export=ALL,TASKFILE=$CAN,STAGE=tune,$COMMON,CANARY_CHECKS=1,REPEAT_CPUS=10,WRITE_FLAG=$R/canary_ok_%m" \
          cluster/slurm/optuna_pack.sbatch)
    S1=$(s1_arrays "$S1C" "--dependency=afterany:$G:$C")
    echo "start: gate=$G canary=$C s1:$S1 class=$CLASS root=$R ($POOL_CPUS cpus / $MEM) $(date)" | tee "$STAMP"
    ;;
  extra)
    TF=${TASKS:?TASKS (a stage-1 task file) must be set}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    grep -qvxFf "$S1T" "$TF" && { echo "$TF has a line that is not in $S1T"; exit 1; }
    for m in $(awk '{print $2}' "$TF" | sort -u); do
      [ -f "$R/canary_ok_$m" ] && [ -f "$R/GATE_OK_$m" ] || { echo "flags of $m missing; refusing"; exit 1; }
    done
    S1=$(s1_arrays "$TF" "")
    echo "extra $TF: s1:$S1 class=$CLASS root=$R $(date)" | tee "$STAMP"
    ;;
  finish)
    MODELS=${S2_MODELS:-lgbm xgb rf}
    S2SUB="$R/tasks_s2_${MODELS// /_}.txt"
    : > "$S2SUB"
    for m in $MODELS; do awk -v m="$m" '$2 == m' "$S2T" >> "$S2SUB"; done
    rm -f "$R/STAGE1_MERGED" "$R/STAGE2_MERGED"
    M1=$(sbatch --parsable -J ${JP}_m1 $PIN ${DEP:+--dependency=$DEP} --time="$MERGE_TIME" \
          "--export=ALL,MERGE_STAGE=1,TASKFILE=$S1T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
    S2=none
    if [ -s "$S2SUB" ]; then
      S2=$(sbatch --parsable -J ${JP}_s2 $PIN --array=1-"$(wc -l < "$S2SUB")" --dependency=afterany:"$M1" \
            --cpus-per-task="$POOL_CPUS" --mem="$MEM" --time="$S2_TIME" \
            "--export=ALL,TASKFILE=$S2SUB,STAGE=refit,$COMMON,HOLD_FLAG=$R/STAGE1_MERGED" cluster/slurm/optuna_pack.sbatch)
      DEP2="--dependency=afterany:$S2"
    else
      DEP2="--dependency=afterany:$M1"
    fi
    M2=$(sbatch --parsable -J ${JP}_m2 $PIN $DEP2 --time="$MERGE_TIME" \
          "--export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
    echo "finish: m1=$M1 s2=$S2 [$MODELS] m2=$M2 class=$CLASS root=$R $(date)" | tee "$STAMP"
    ;;
  merge2)
    rm -f "$R/STAGE2_MERGED"
    M2=$(sbatch --parsable -J ${JP}_m2 $PIN --time="$MERGE_TIME" \
          "--export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R" cluster/slurm/optuna_merge.sbatch)
    echo "merge2: m2=$M2 root=$R $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be start, extra, finish or merge2"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.12j %.4t %.10M %.5C %P %R" | grep -E "${JP}_|JOBID" | head -20
