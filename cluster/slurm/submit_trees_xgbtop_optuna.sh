#!/bin/bash
# xgbtop (2026-10-07): the Optuna rung (OP) of the per-bar 16:00 XGBoost re-centred on the top dev-span entry xgb_opt_00
# (trial 0 of every study; reg_lambda axis [1e-5, 1e4]), PARAMETERIZED by TOKEN (default xgbtop) and XGB_CONFIG (default,
# for xgbtop, cluster/xgbtop_xgb_config.json) -- cluster/slurm/submit_trees_rfp163_optuna.sh for XGBoost: the masked
# Optuna campaign re-run for XGBoost only on baseline, lean_vol_vix and all_features, WINDOW_MASK=1, bar1600, TRAIN_WIN
# 2000, 50 TPE trials per tuning point at every session, the 9 stage-2 paths (tp1, tp5, tp25, tp250, tp1_k10, tp1_k25,
# tp25_k10, tp25_k25, tp25_q = optunaq_tp25), single-threaded fits in SPAWNED pools, every task pinned to CLASS (default
# epyc-7513; CLASSES widens the fleets, e.g. "epyc-7513|epyc-7542", once the cross-CPU cell matched byte for byte), through
#   specs/causal_tune_trees_optuna_xgbtop.py  (specs/causal_tune_trees_optuna.py unchanged; TREE_SPEC ->
#                                              specs/causal_tune_trees_<TOKEN>.py)
#   cluster/slurm/optuna_xgbtop_pack.sbatch   (the campaign's pack script for the wrapper)
#   cluster/slurm/optuna_xgbtop_merge.sbatch  (the unedited reducer on this root)
#   cluster/slurm/optuna_xgbtop_gate.sbatch   (the identity gate vs the fixed-settings XGBoost)
# in the deployment root /scratch1/jc_905/harxhar-xgbtop (cluster/slurm/ship_trees_xgbtop_carc.sh), results root
# R = results/linear_subsection_trees_optuna_mask_<TOKEN>, task files cluster/<TOKEN>_optuna_tasks_*.txt
# (cluster/xgbtop_optuna_make_tasks.py).  Submit from a shell WITHOUT a conda env active.
#
#   bash cluster/slurm/submit_trees_xgbtop_optuna.sh gate
#       array of 3 (one line per bucket): IDENTITY=1 through both stages on a few rows must reproduce the fixed-settings
#       XGBoost (specs/causal_tune_trees_<TOKEN>.py) bit for bit, + a 3-trial TPE smoke -> $R/GATE_OK_<bucket>
#   [AFTER=<gate job>] bash cluster/slurm/submit_trees_xgbtop_optuna.sh canary
#       array of 3 (stage 1 on OOS rows [0, 20) of each bucket = its fleet chunk 0) with CANARY_CHECKS=1 (a repeat at
#       REPEAT_CPUS=10 processes must be bit-identical; stage 2 on the chunk's own records) -> $R/canary_ok_<bucket>
#   [AFTER=<gate job>:<canary job>] bash cluster/slurm/submit_trees_xgbtop_optuna.sh s1
#       the stage-1 fleet: per bucket, an array of the s1 lines that are not canary lines; every task refuses to run
#       unless its bucket's canary_ok AND GATE_OK flags exist; with AFTER the arrays are held on it
#   TASKS=<file> bash cluster/slurm/submit_trees_xgbtop_optuna.sh extra      stage-1 lines again (resume)
#   [AFTER=<s1 job ids, colon-separated>] bash cluster/slurm/submit_trees_xgbtop_optuna.sh finish
#       m1 (stage-1 merge over the FULL s1 file) -> s2 (held afterany on m1 + HOLD_FLAG STAGE1_MERGED; each task requires
#       its arm's STAGE1_COMPLETE) -> m2 (stage-2 merge, afterany on s2)
# AFTER holds the mode's first job with --dependency=$DEP_TYPE:$AFTER (DEP_TYPE default afterok) and
# --kill-on-invalid-dep=yes.  Every mode refuses a second submission of itself (logs/submitted_<TOKEN>_optuna_<mode>[...].txt).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE=${1:?usage: submit_trees_xgbtop_optuna.sh gate|canary|s1|extra|finish}
if [ -n "${CONDA_DEFAULT_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "a conda env is active in this shell (${CONDA_DEFAULT_ENV:-}${CONDA_PREFIX:+ $CONDA_PREFIX}): sbatch --export=ALL would hand it to the jobs; submit from a clean shell"
  exit 1
fi
TOKEN=${TOKEN:-xgbtop}
if [ "$TOKEN" = xgbtop ]; then XGB_CONFIG=${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}; else XGB_CONFIG=${XGB_CONFIG:?XGB_CONFIG must name the configuration of token $TOKEN}; fi
DEPLOY=/scratch1/jc_905/harxhar-xgbtop
[ "$PWD" = "$DEPLOY" ] || [ "${ROOT_OK:-0}" = 1 ] || { echo "run from the deployment root $DEPLOY (ROOT_OK=1 overrides), not $PWD"; exit 1; }
CLASS=${CLASS:-epyc-7513}
CLASSES=${CLASSES:-$CLASS}  # the stage-1 / stage-2 fleets; the gate and the canary stay on CLASS
POOL_CPUS=${POOL_CPUS:-20}
MEM=${MEM:-24G}
GATE_TIME=${GATE_TIME:-1:30:00}
CANARY_TIME=${CANARY_TIME:-2:30:00}
S1_TIME=${S1_TIME:-3:30:00}
S2_TIME=${S2_TIME:-2:30:00}
MERGE_TIME=${MERGE_TIME:-2:00:00}
JP=${JP:-op_$TOKEN}  # job-name prefix
R=${RESULTS_ROOT:-results/linear_subsection_trees_optuna_mask_$TOKEN}
TREE_SPEC=specs/causal_tune_trees_$TOKEN.py
BUCKETS=${BUCKETS:-baseline lean_vol_vix all_features}  # s1 / extra: a subset submits those buckets only
GT=cluster/${TOKEN}_optuna_tasks_gate.txt
CAN=cluster/${TOKEN}_optuna_tasks_canary.txt
S1T=cluster/${TOKEN}_optuna_tasks_s1.txt
S2T=cluster/${TOKEN}_optuna_tasks_s2.txt
for f in specs/causal_tune_trees_optuna.py specs/causal_tune_trees_optuna_jobs.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_trees.py specs/causal_tune_linear.py "$TREE_SPEC" \
         specs/causal_tune_trees_optuna_xgbtop.py src/backtest/executor.py src/models/window_mask.py src/data/loading.py \
         experiments/reduce_trees_optuna.py experiments/trees_xgbtop_make_specs.py experiments/trees_xgbtop_optuna_gate.py \
         cluster/slurm/optuna_xgbtop_pack.sbatch cluster/slurm/optuna_xgbtop_merge.sbatch cluster/slurm/optuna_xgbtop_gate.sbatch \
         "$XGB_CONFIG" "$GT" "$CAN" "$S1T" "$S2T"; do
  [ -f "$f" ] || { echo "$f missing: run cluster/slurm/ship_trees_xgbtop_carc.sh locally first"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
grep -q '"lean_vol_vix"' src/data/loading.py || { echo "src/data/loading.py has no lean_vol_vix bucket"; exit 1; }
awk '$2 != "xgb" || $3 != 2000 || $4 != "bar1600" || $8 != 2000 {bad = 1} END {exit bad}' "$GT" "$CAN" "$S1T" "$S2T" \
  || { echo "a task line is not xgb / tw2000 / bar1600 / halo 2000"; exit 1; }
grep -qvxFf "$S1T" "$CAN" && { echo "$CAN has a line that is not in $S1T"; exit 1; }
TAG=${TASKS:+_$(basename "$TASKS" .txt)}
[ "$BUCKETS" = "baseline lean_vol_vix all_features" ] || TAG="${TAG}_${BUCKETS// /_}"
STAMP="logs/submitted_${TOKEN}_optuna_${MODE}${TAG}.txt"
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); move $STAMP aside by hand to resubmit"; exit 1; }
mkdir -p "$R"
PIN="--constraint=$CLASS"
FLEET_PIN="--constraint=$CLASSES"
COMMON="RESULTS_ROOT=$R,WINDOW_MASK=1,TOKEN=$TOKEN,XGB_CONFIG=$XGB_CONFIG,HPC_KW_TREE_SPEC=$TREE_SPEC"
AFTER=${AFTER:-}
DEP_TYPE=${DEP_TYPE:-afterok}
HOLD=()
[ -z "$AFTER" ] || HOLD=(--dependency="$DEP_TYPE:$AFTER" --kill-on-invalid-dep=yes)
flags_of() { echo "$R/canary_ok_$1 $R/GATE_OK_$1"; }
need_flags() {  # $1 = 1: AFTER may stand in for missing flags (the tasks re-check them when they start)
  local b f miss=""
  for b in $BUCKETS; do for f in $(flags_of "$b"); do [ -f "$f" ] || miss="$miss $f"; done; done
  [ -z "$miss" ] && return 0
  if [ "$1" = 1 ] && [ -n "$AFTER" ]; then echo "flags not yet present:$miss -- held $DEP_TYPE:$AFTER"; return 0; fi
  echo "flags missing:$miss; refusing${1:+ (give AFTER=<gate job>:<canary job> to hold on them)}"
  exit 1
}
s1_arrays() {  # $1 task file (lines of $S1T): one array per bucket, each task requiring its bucket's two flags
  local tf=$1 ids="" b n j sub
  for b in $BUCKETS; do
    sub="$R/tasks_$(basename "$tf" .txt)_$b.txt"
    awk -v b="$b" '$1 == b' "$tf" > "$sub"
    n=$(wc -l < "$sub")
    [ "$n" -gt 0 ] || continue
    j=$(sbatch --parsable -J "${JP}_s1_$b" $FLEET_PIN --array=1-"$n" "${HOLD[@]}" --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$S1_TIME" "--export=ALL,TASKFILE=$sub,STAGE=tune,$COMMON,CANARY_FLAG=$(flags_of "$b")" \
          cluster/slurm/optuna_xgbtop_pack.sbatch)
    ids="$ids $b:$j($n)"
  done
  echo "$ids"
}

case "$MODE" in
  gate)
    rm -f "$R"/GATE_OK_*
    G=$(sbatch --parsable -J ${JP}_gate $PIN --array=1-"$(wc -l < "$GT")" "${HOLD[@]}" --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$GATE_TIME" "--export=ALL,RESULTS_ROOT=$R,GATE_TASKFILE=$GT,TOKEN=$TOKEN,XGB_CONFIG=$XGB_CONFIG" \
          cluster/slurm/optuna_xgbtop_gate.sbatch)
    echo "gate: $G${FIXED_NPZ:+ (fixed npz $FIXED_NPZ)}${AFTER:+ held $DEP_TYPE:$AFTER} token=$TOKEN ($XGB_CONFIG) class=$CLASS root=$R ($POOL_CPUS cpus / $MEM) $(date)" | tee "$STAMP"
    ;;
  canary)
    rm -f "$R"/canary_ok_*
    C=$(sbatch --parsable -J ${JP}_canary $PIN --array=1-"$(wc -l < "$CAN")" "${HOLD[@]}" --cpus-per-task="$POOL_CPUS" --mem="$MEM" \
          --time="$CANARY_TIME" \
          "--export=ALL,TASKFILE=$CAN,STAGE=tune,$COMMON,CANARY_CHECKS=1,REPEAT_CPUS=10,WRITE_FLAG=$R/canary_ok_%b" \
          cluster/slurm/optuna_xgbtop_pack.sbatch)
    echo "canary: $C${AFTER:+ held $DEP_TYPE:$AFTER} token=$TOKEN ($XGB_CONFIG) class=$CLASS root=$R ($POOL_CPUS cpus / $MEM) $(date)" | tee "$STAMP"
    ;;
  s1)
    need_flags 1
    FLEET="$R/s1_fleet.txt"
    grep -vxFf "$CAN" "$S1T" > "$FLEET" || true
    S1=$(s1_arrays "$FLEET")
    echo "s1:$S1${AFTER:+ held $DEP_TYPE:$AFTER} token=$TOKEN ($XGB_CONFIG) class=$CLASSES root=$R ($POOL_CPUS cpus / $MEM / $S1_TIME) $(date)" | tee "$STAMP"
    ;;
  extra)
    TF=${TASKS:?TASKS (a stage-1 task file) must be set}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    grep -qvxFf "$S1T" "$TF" && { echo "$TF has a line that is not in $S1T"; exit 1; }
    need_flags ""
    S1=$(s1_arrays "$TF")
    echo "extra $TF: s1:$S1${AFTER:+ held $DEP_TYPE:$AFTER} token=$TOKEN ($XGB_CONFIG) class=$CLASSES root=$R $(date)" | tee "$STAMP"
    ;;
  finish)
    rm -f "$R/STAGE1_MERGED" "$R/STAGE2_MERGED"
    M1=$(sbatch --parsable -J ${JP}_m1 "${HOLD[@]}" --time="$MERGE_TIME" \
          "--export=ALL,MERGE_STAGE=1,TASKFILE=$S1T,RESULTS_ROOT=$R,TOKEN=$TOKEN" cluster/slurm/optuna_xgbtop_merge.sbatch)
    S2=$(sbatch --parsable -J ${JP}_s2 $FLEET_PIN --array=1-"$(wc -l < "$S2T")" --dependency=afterany:"$M1" \
          --cpus-per-task="$POOL_CPUS" --mem="$MEM" --time="$S2_TIME" \
          "--export=ALL,TASKFILE=$S2T,STAGE=refit,$COMMON,HOLD_FLAG=$R/STAGE1_MERGED" cluster/slurm/optuna_xgbtop_pack.sbatch)
    M2=$(sbatch --parsable -J ${JP}_m2 --dependency=afterany:"$S2" --time="$MERGE_TIME" \
          "--export=ALL,MERGE_STAGE=2,TASKFILE=$S2T,RESULTS_ROOT=$R,TOKEN=$TOKEN" cluster/slurm/optuna_xgbtop_merge.sbatch)
    echo "finish: m1=$M1${AFTER:+ (held $DEP_TYPE:$AFTER)} s2=$S2 m2=$M2 token=$TOKEN ($XGB_CONFIG) class=$CLASSES root=$R $(date)" | tee "$STAMP"
    ;;
  *) echo "MODE must be gate, canary, s1, extra or finish"; exit 1 ;;
esac
squeue -u "$USER" -o "%.16i %.14j %.4t %.10M %.5C %P %R" | grep -E "${JP}_|JOBID" | head -20
