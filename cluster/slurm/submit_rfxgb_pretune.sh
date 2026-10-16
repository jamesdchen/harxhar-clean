#!/bin/bash
# Random forest / XGBoost pre-tune study (2026-10-06): submit on CARC from the deployment root that
# cluster/slurm/ship_rfxgb_pretune_carc.sh fills (/scratch1/jc_905/harxhar-rfxgb-pretune):
#
#   bash cluster/slurm/submit_rfxgb_pretune.sh canary            gate + toy pipeline for every cell
#   bash cluster/slurm/submit_rfxgb_pretune.sh fleet [cells]     refuses unless every cell's canary
#                                                                passed; all_features cells first
#   bash cluster/slurm/submit_rfxgb_pretune.sh all [cells]       the canary, and the fleet held
#                                                                afterany on it (one ssh session)
#
# HELD ON THE CANARY ('all'): the fleet's first jobs wait for the canary job to END (afterany), and
# every fleet task refuses to run unless CANARY_OK_<cell> exists (rfxgb_pretune_task.sbatch): a cell
# whose canary failed has its tasks exit at once, so no job is left pending on a dependency that
# can never be satisfied (jobs cannot be cancelled from the laptop).
#
# fleet, for each cell (model x input set) in order:
#   control  array over cluster/slurm/rfxgb_pretune_tasks_control.txt lines of the cell (the shipped
#            configuration's refits, every session and every 10; needs no pre-tune)
#   pretune  array over cluster/slurm/rfxgb_pretune_tasks_pretune.txt lines of the cell (5 Optuna
#            studies a task, 4 single-threaded fold processes each), RXP_TRIALS cap, RXP_TIMEOUT
#   plan     one task held afterany on the cell's pre-tune array (refuses unless every study of the
#            cell finished): merge, light retunes, plan.json
#   arms     array over cluster/slurm/rfxgb_pretune_tasks_arms.txt lines of the cell, held afterany
#            on the plan job (refuses without PLAN_DONE)
# afterany (not afterok) everywhere: a failed upstream task makes its dependants exit at once with a
# message instead of leaving them pending forever (jobs cannot be cancelled from the laptop).
# Every task is pinned to one CPU class (CLASS, default epyc-7513: the class of the stored masked
# runs the control is gated against).  A stamp (logs/submitted_rfxgb_pretune_<mode>_<tag>.txt)
# refuses a second identical submission.
# Env: TAG (default carc), CLASS, TRIALS (1000 a study), TIMEOUT (10800 s a study), PRE_TIME
# (3:50:00), CPUS (20), MEM (32G).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
MODE=${1:?usage: submit_rfxgb_pretune.sh canary|fleet|all|status [cells]}
TAG=${TAG:-carc}
CLASS=${CLASS:-epyc-7513}
TRIALS=${TRIALS:-1000}
TIMEOUT=${TIMEOUT:-10800}
PRE_TIME=${PRE_TIME:-3:50:00}
CPUS=${CPUS:-20}
MEM=${MEM:-32G}
# moments_vix dropped (user, 2026-10-06); its task lines stay (submitted jobs index them), lean_vol_vix appended
ALL="rf_all_features xgb_all_features rf_baseline xgb_baseline rf_lean_vol_vix xgb_lean_vol_vix"
CELLS=${2:-$ALL}
WORK=results/trees_pretune_2026-10-06/_work
TS=cluster/slurm/rfxgb_pretune_tasks
SB=cluster/slurm/rfxgb_pretune_task.sbatch
LOG=logs/submitted_rfxgb_pretune.txt
for f in experiments/close_trees_pretune_rfxgb.py experiments/close_trees_pretune_rfxgb_jobs.py "$SB" \
         "${TS}_pretune.txt" "${TS}_plan.txt" "${TS}_control.txt" "${TS}_arms.txt" \
         specs/causal_tune_trees.py specs/causal_tune_trees_tuned_jobs.py specs/causal_tune_trees_optuna_jobs.py \
         specs/causal_tune_linear.py src/models/window_mask.py "$WORK/design/design_sha256.json"; do
  [ -f "$f" ] || { echo "$f missing: run cluster/slurm/ship_rfxgb_pretune_carc.sh locally first"; exit 1; }
done
[ -f rfxgb_pretune_manifest.md5 ] && { md5sum -c --quiet rfxgb_pretune_manifest.md5 || { echo "shipped files changed since the ship; re-ship"; exit 1; }; }
STAMP="logs/submitted_rfxgb_pretune_${MODE}_${TAG}.txt"
[ "$CELLS" = "$ALL" ] || STAMP="logs/submitted_rfxgb_pretune_${MODE}_${TAG}_${CELLS// /+}.txt"
[ "$MODE" = status ] || { [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }; } || true
COMMON=(--parsable --account=pollok_1603 --partition=main --constraint="$CLASS" --cpus-per-task="$CPUS" --mem="$MEM")
lines_of() {  # 1-based line numbers of a cell in a task file, as an array range list
  grep -n "^$1 $2 " "$3" | cut -d: -f1 | paste -sd, -
}

submit_fleet() {  # $1 = canary job id to hold on ("" = the canary flags must exist now)
  local after=$1 c m b LC LP LA LN E JC JP JN JA DEP=()
  [ -z "$after" ] || DEP=(--dependency="afterany:$after")
  if [ -z "$after" ]; then
    for c in $CELLS; do
      [ -f "$WORK/canary/$c/CANARY_OK_$c" ] || { echo "CANARY_OK_$c missing; refusing the fleet"; exit 1; }
    done
  fi
  OUTL=""
  for c in $CELLS; do
    m=${c%%_*}; b=${c#*_}
    LC=$(lines_of "$m" "$b" "${TS}_control.txt"); LP=$(lines_of "$m" "$b" "${TS}_pretune.txt")
    LA=$(lines_of "$m" "$b" "${TS}_arms.txt"); LN=$(lines_of "$m" "$b" "${TS}_plan.txt")
    [ -n "$LC" ] && [ -n "$LP" ] && [ -n "$LA" ] && [ -n "$LN" ] || { echo "no task lines for $c"; exit 1; }
    E="ALL,RXP_TAG=$TAG,RXP_TRIALS=$TRIALS,RXP_TIMEOUT=$TIMEOUT"
    JC=$(sbatch "${COMMON[@]}" -J "rxp_ctl_$c" --time=2:00:00 --array="$LC" "${DEP[@]}"           --export="$E,STAGE=refit,TASKFILE=${TS}_control.txt" "$SB")
    JP=$(sbatch "${COMMON[@]}" -J "rxp_pre_$c" --time="$PRE_TIME" --array="$LP" "${DEP[@]}"           --export="$E,STAGE=pretune,TASKFILE=${TS}_pretune.txt" "$SB")
    JN=$(sbatch "${COMMON[@]}" -J "rxp_plan_$c" --time=2:00:00 --array="$LN" --dependency="afterany:$JP"           --export="$E,STAGE=plan,TASKFILE=${TS}_plan.txt" "$SB")
    JA=$(sbatch "${COMMON[@]}" -J "rxp_arms_$c" --time=3:00:00 --array="$LA" --dependency="afterany:$JN"           --export="$E,STAGE=refit,TASKFILE=${TS}_arms.txt" "$SB")
    OUTL="$OUTL $c:ctl=$JC,pre=$JP,plan=$JN,arms=$JA"
  done
}
submit_canary() {
  sbatch "${COMMON[@]}" -J rxp_canary --time=1:30:00 --output=logs/%x.%j.out     --export=ALL,STAGE=canary,RXP_TAG=canary,CELLS="${CELLS// /:}" "$SB"
}

case "$MODE" in
  canary)
    J=$(submit_canary)
    echo "$(date +%F_%T) canary=$J cells=[$CELLS] class=$CLASS" | tee -a "$LOG" > "$STAMP"
    ;;
  fleet)
    for c in $CELLS; do
      g="$WORK/canary/$c/gate_$c.csv"
      [ -f "$g" ] && { echo "== $c canary gate"; cut -d, -f1-6 "$g"; }
    done
    submit_fleet ""
    echo "$(date +%F_%T) fleet tag=$TAG class=$CLASS trials=$TRIALS timeout=$TIMEOUT$OUTL" | tee -a "$LOG" > "$STAMP"
    ;;
  all)
    J=$(submit_canary)
    submit_fleet "$J"
    echo "$(date +%F_%T) canary=$J fleet(afterany canary) tag=$TAG class=$CLASS trials=$TRIALS timeout=$TIMEOUT$OUTL" | tee -a "$LOG" > "$STAMP"
    ;;
  status)  # read only: canary gates, progress flags, the latest pre-tune lines, the queue
    for c in $CELLS rf_moments_vix xgb_moments_vix; do
      g="$WORK/canary/$c/gate_$c.csv"
      echo "== $c canary: $(ls "$WORK/canary/$c"/CANARY_OK_* 2>/dev/null | wc -l) flag(s)"
      if [ -f "$g" ]; then python3 -c 'import csv, sys
for r in csv.DictReader(open(sys.argv[1])): print("   %-44.44s | %-44.44s | rel %s | ok %s" % (r["gate"], r["item"], r["max_rel_diff"], r["ok"]))' "$g"; fi
      d="$WORK/$TAG/$c"
      if [ ! -d "$d" ]; then continue; fi
      echo "   $TAG: pretune done $(ls "$d"/PRETUNE_DONE_* 2>/dev/null | wc -l), plan $(ls "$d"/PLAN_DONE 2>/dev/null | wc -l), refit done $(ls "$d"/REFIT_DONE_* 2>/dev/null | wc -l), refit parts $(ls "$d"/refits/*_p*.npz 2>/dev/null | wc -l)"
      for f in "$d"/pretune_s*.log; do
        if [ -f "$f" ]; then echo "   $(basename "$f"): $(grep -E 'trial [0-9]+:|export|Error|Traceback' "$f" | tail -1 | cut -c1-150)"; fi
      done
    done
    squeue -u "$USER" -h -o "%T %j" | grep rxp_ | sort | uniq -c || true
    exit 0
    ;;
  *) echo "MODE must be canary, fleet, all or status"; exit 1 ;;
esac
squeue -u "$USER" -o "%.18i %.22j %.3t %.10M %.5C %R" | grep -E "JOBID|rxp_" | head -40
