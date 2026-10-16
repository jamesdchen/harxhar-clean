#!/bin/bash
# Pooled tree walks (user request 2026-10-07): the bank's walk (refit every bar, 24,000-bar window, seed 42, 8 threads,
# 100 chunks) for lgbm_opt_00 / xgb_opt_00 / the 400-trial forest on new pooled input sets, from the deployment root
# /scratch1/jc_905/harxhar-pooltrees (cluster/slurm/ship_pooltrees_carc.sh).  Results root R = results/pooled_trees_2026-10-07.
#   CLASS=<feature> bash cluster/slurm/submit_pooltrees.sh gate
#        the bank's arms 00 (LightGBM) and 10 (XGBoost) re-walked on the first rows of chunks 0 and 50 must equal the stored
#        bank chunks byte for byte (+ XGBoost at 1 thread vs the 8-thread reference) -> results/pooltree_gates/<CLASS>/GATE_OK
#   AFTER=<gate job> CLASSES=<constraint> bash cluster/slurm/submit_pooltrees.sh canary <est> <set> <rows>
#        chunk 0's first <rows> rows, run twice (bit-identical repeat) -> R/canary_ok_<est>_<set>; the cost probe
#   AFTER=<canary job> CLASSES=<constraint> bash cluster/slurm/submit_pooltrees.sh fleet <est> <set>
#        an array over cluster/pooltree_tasks_<est>_<set>.txt ("<est> <set> <chunk>[,<chunk>...]" per line), each task
#        refusing to run without R/canary_ok_<est>_<set>
#   bash cluster/slurm/submit_pooltrees.sh dump [after job ids]    yhat_pooltree_<est>_<set>.parquet for every complete arm
# REFIT_EVERY=K (default 1 = the bank's walk) runs a cadence-K walk (arms, canary flags and stamps suffixed _r<K>).
# Submit from a shell with no conda env active.  Stamps (logs/submitted_pooltrees_<mode>_<...>.txt) refuse resubmission.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active; submit from a clean shell"; exit 1 ;; esac
[ -z "${CONDA_DEFAULT_ENV:-}" ] || { echo "conda env ${CONDA_DEFAULT_ENV} is active; submit from a clean shell"; exit 1; }
MODE=${1:?usage: submit_pooltrees.sh gate|canary <est> <set> <rows>|fleet <est> <set>|pieces <est> <set> <taskfile>|dump [after]}
R=results/pooled_trees_2026-10-07
CLASS=${CLASS:-epyc-7513}
CLASSES=${CLASSES:-$CLASS}
for f in experiments/pooled_trees_walk.py experiments/pooled_trees_gate.py experiments/dump_unif_yhat.py src/unification.py \
         experiments/run_geometry_local.py cluster/slurm/pooltree_task.sbatch cluster/slurm/pooltree_gate.sbatch \
         cluster/slurm/pooltree_dump.sbatch cache/prep_cache_all_features_b2.npz; do
  [ -f "$f" ] || { echo "$f missing (ship first; the cache is uploaded once)"; exit 1; }
done
LOG=logs/submitted_pooltrees.txt
AFTER=${AFTER:-}
HOLD=()
[ -z "$AFTER" ] || HOLD=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
# SUBMIT_HELD=1: submit with a user hold (a reserve array that a top-up releases later; 2026-10-15)
if [ "${SUBMIT_HELD:-0}" = 1 ]; then HOLD+=(--hold); fi
K=${REFIT_EVERY:-1}   # refit cadence exported to the tasks (1 = the bank's walk); K > 1 -> arms / flags / stamps _r<K>
SFX=""; [ "$K" != 1 ] && SFX="_r$K"
stamp() { local S="logs/submitted_pooltrees_$1.txt"; [ -f "$S" ] && { echo "already submitted ($(cat "$S"))"; exit 1; }; echo "$S"; }
case "$MODE" in
  gate)
    # TAG + LINES (optional): a second gate run with other lines into results/pooltree_gates/<CLASS>_<TAG>
    S=$(stamp "gate_$CLASS${TAG:+_$TAG}")
    G=results/pooltree_gates/$CLASS${TAG:+_$TAG}; mkdir -p "$G"; rm -f "$G/GATE_OK"
    J=$(sbatch --parsable -J ptr_gate "${HOLD[@]}" --constraint="$CLASS" --export=ALL,OUT=$G,LINES=${LINES:-cluster/pooltree_gate_lines.txt} cluster/slurm/pooltree_gate.sbatch)
    echo "$(date +%F_%T) gate_$CLASS${TAG:+_$TAG}=$J (${LINES:-cluster/pooltree_gate_lines.txt})" | tee -a "$LOG" > "$S"
    ;;
  canary)
    EST=${2:?est}; SET=${3:?set}; ROWS=${4:?rows}
    S=$(stamp "canary_${EST}_$SET$SFX")
    TF=$R/canary_tasks_${EST}_$SET$SFX.txt; mkdir -p "$R"; echo "$EST $SET 0 $ROWS" > "$TF"; rm -f "$R/canary_ok_${EST}_$SET$SFX"
    J=$(sbatch --parsable -J "ptr_c_${EST}_$SET" "${HOLD[@]}" --constraint="$CLASSES" --time="${CANARY_TIME:-12:00:00}" \
         --export=ALL,TASKFILE=$TF,RESULTS_ROOT=$R,REPEAT=1,CANARY_DIR=${CANARY_DIR:-canary},REFIT_EVERY=$K,WRITE_FLAG=$R/canary_ok_%e_%s$SFX cluster/slurm/pooltree_task.sbatch)
    echo "$(date +%F_%T) canary_${EST}_$SET$SFX=$J (rows $ROWS, $CLASSES)${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  fleet)
    EST=${2:?est}; SET=${3:?set}
    TF=${TASKS:-cluster/pooltree_tasks_${EST}_$SET.txt}   # TASKS: another task file (e.g. a redo of timed-out chunks; TAG it)
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    S=$(stamp "fleet_${EST}_$SET$SFX${TAG:+_$TAG}")
    N=$(grep -c . "$TF")
    # CPUS (default 8): a forest's trees do not depend on n_jobs (predict is one-threaded), XGBoost gave the same bytes at
    # 1 / 4 / 8 threads; LightGBM must keep the bank's 8 threads
    [ "$EST" = lgbm ] && [ "${CPUS:-8}" != 8 ] && { echo "LightGBM must run with the bank's 8 threads"; exit 1; }
    J=$(sbatch --parsable -J "ptr_${EST}_$SET$SFX" "${HOLD[@]}" --constraint="$CLASSES" --array=1-"$N" --time="${FLEET_TIME:-24:00:00}" \
         --cpus-per-task="${CPUS:-8}" --mem="${MEM:-40G}" \
         --export=ALL,TASKFILE=$TF,RESULTS_ROOT=$R,REFIT_EVERY=$K,CANARY_FLAG=$R/canary_ok_${EST}_$SET$SFX cluster/slurm/pooltree_task.sbatch)
    echo "$(date +%F_%T) fleet_${EST}_$SET$SFX${TAG:+_$TAG}=$J (1-$N, $TF, $CLASSES, ${FLEET_TIME:-24:00:00}, ${MEM:-40G})${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  pieces)
    # row pieces (experiments/pooled_trees_pieces.py task lines), backfill-friendly: CPUS (default 4), MEM (default 8G),
    # TIME (required: ~1.5 x the measured seconds a row x rows a task), PARTITION (main | largemem), ARRAY_MAX (optional
    # %-throttle), ARRAY (an index range of the task file, default all; disjoint ranges may go to other partitions),
    # NICE, RESULTS_ROOT override (RROOT), CANARY_FLAG (files every task requires)
    EST=${2:?est}; SET=${3:?set}; TF=${4:?taskfile}
    [ -f "$TF" ] || { echo "$TF missing"; exit 1; }
    [ "$EST" != lgbm ] || { echo "LightGBM is not run in pieces (the bank's 8 threads, chunk fleets)"; exit 1; }
    S=$(stamp "pieces_$(basename "$TF" .txt)${TAG:+_$TAG}")
    N=$(grep -c . "$TF")
    J=$(sbatch --parsable -J "ptrp_${EST}_$SET" "${HOLD[@]}" --partition="${PARTITION:-main}" --constraint="$CLASSES"          --array="${ARRAY:-1-$N}"${ARRAY_MAX:+%$ARRAY_MAX} ${NICE:+--nice=$NICE} --cpus-per-task="${CPUS:-4}" --mem="${MEM:-8G}" --time="${TIME:?TIME}"          --export=ALL,TASKFILE=$TF,RESULTS_ROOT=${RROOT:-$R}${CANARY_FLAG:+,CANARY_FLAG=$CANARY_FLAG} cluster/slurm/pooltree_piece.sbatch)
    echo "$(date +%F_%T) pieces_$(basename "$TF" .txt)${TAG:+_$TAG}=$J (${ARRAY:-1-$N}, ${PARTITION:-main}, $CLASSES, ${CPUS:-4} cpus, ${MEM:-8G}, $TIME${NICE:+, nice $NICE})${HOLD:+ ${HOLD[*]}}" | tee -a "$LOG" > "$S"
    ;;
  interim)
    # the INTERIM 16:00 all_features table (cluster/slurm/pooltree_interim.sbatch); AFTER=<job ids> optional
    S=$(stamp "interim${TAG:+_$TAG}")
    J=$(sbatch --parsable "${HOLD[@]}" --nice=0 cluster/slurm/pooltree_interim.sbatch)
    echo "$(date +%F_%T) interim${TAG:+_$TAG}=$J${AFTER:+ (afterok $AFTER)}" | tee -a "$LOG" > "$S"
    ;;
  dump)
    A=${2:-}
    S=$(stamp "dump_${A//:/_}${TAG:+_$TAG}")   # TAG: a later dump (e.g. TAG=rf_baseline) after the first one
    DEP=(); [ -z "$A" ] || DEP=(--dependency="afterany:$A")
    J=$(sbatch --parsable -J ptr_dump "${DEP[@]}" --export=ALL,RESULTS_ROOT=$R cluster/slurm/pooltree_dump.sbatch)
    echo "$(date +%F_%T) dump${TAG:+_$TAG}=$J${A:+ (afterany $A)}" | tee -a "$LOG" > "$S"
    ;;
  *) echo "MODE must be gate, canary, fleet or dump"; exit 1 ;;
esac
squeue -u "$USER" -h -o "%i %j %T %R" | grep -E "ptr_" | head -12
