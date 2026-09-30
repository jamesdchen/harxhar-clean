#!/bin/bash
# I9 (2026-09-29 evening; plan writeup/CAMPAIGN_16H_2026-09-29.md, "Restart with a per-window
# mask"): the MASKED 16:00 re-run, WINDOW_MASK=1, SEGMENT bar1600, TRAIN_WIN 2000, de-duplicated
# per-bar design, LightGBM / XGBoost / RF x baseline / live_feasible / all_features + the LSTM:
#   T10     specs/causal_tune_trees.py        REFIT_EVERY 10, IMPORTANCE_EVERY 1
#   T1      specs/causal_tune_trees.py        REFIT_EVERY 1,  IMPORTANCE_EVERY 10
#   RS10    specs/causal_tune_trees_tuned.py  REFIT_EVERY 10 (random search, 32 candidates, TUNE_PER 250)
#   RS1     specs/causal_tune_trees_tuned.py  REFIT_EVERY 1, TreeSHAP off
#   LSTM    specs/causal_tune_lstm.py         REFIT_EVERY 1 (user decision 2026-09-29 evening: the
#           LSTM refits every session, as the trees do; results/linear_subsection_lstm_mask, the
#           canonical yhat_lstm_* tables)
#   LSTM10  specs/causal_tune_lstm.py         REFIT_EVERY 10 (the earlier cadence, masked: the
#           cadence effect for the LSTM; results/linear_subsection_lstm_mask_re10)
# Run in the deployment root /scratch1/jc_905/harxhar-mask (cluster/slurm/ship_trees_mask_carc.sh).
# Stages, each submitted by hand after the previous one was read:
#   bash cluster/slurm/submit_trees_mask.sh gates [cadence,lstm,h_trees,h_lstm]
#        concurrent gate jobs (cluster/slurm/trees_mask_gates.sbatch), default all four
#   bash cluster/slurm/submit_trees_mask.sh canary        (trees) refuses unless cadence + h_trees
#        passed; untuned pack 0 (chunk c0 of all 18 T10 / T1 arms + one chunk re-run alone,
#        compared bit for bit), the tuned canaries (chunk 0 of live_feasible x lgbm,xgb,rf for
#        RS10 and RS1)
#   bash cluster/slurm/submit_trees_mask.sh fleet         (trees) refuses unless the three tree
#        CANARY_OK flags exist; untuned packs 1..K, RS10 and RS1 arrays (one chunk-arm per task),
#        the tree merge job held afterany on them
#   bash cluster/slurm/submit_trees_mask.sh lstm_canary   refuses unless lstm + h_lstm passed;
#        chunk 0 of live_feasible at REFIT_EVERY 1 run twice AT ONCE (pools of half the task's
#        CPUs; byte-identical tables) and at REFIT_EVERY 10 once
#   bash cluster/slurm/submit_trees_mask.sh lstm_fleet    refuses unless both LSTM CANARY_OK flags
#        exist; the REFIT_EVERY 1 and 10 arrays (their canary chunks carry DONE and are skipped),
#        the LSTM merge job held afterany on them
#   bash cluster/slurm/submit_trees_mask.sh kept          the kept-column counts over every root
# Every fleet task also refuses to run without its canary flag.
#
# SIZING (the campaign's rule: ~20-CPU tasks of single-threaded processes, so the 2000-CPU cap
# binds before the 100-job cap).  Untuned: one array task = one pack of <= 20 single-threaded
# chunk processes (the task file of the unmasked I2 run).  Tuned: one array task = one chunk-arm
# with a pool of 20 single-threaded workers.  LSTM: one array task = one chunk (one tuning
# period: the spec's chunks must start on a tuning row), a pool of 20.  Arrays of <= 50 tasks
# may also use the oneweek partition (PART_SMALL; its QOS allows 50 submitted jobs per user);
# the 54-task tuned arrays stay on main.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE="${1:-}"
U=results/linear_subsection_trees_mask
T=results/linear_subsection_trees_tuned_mask
L1=results/linear_subsection_lstm_mask
L10=results/linear_subsection_lstm_mask_re10
G=$U/gates
UT=cluster/trees_mask_tasks_untuned.txt
RS_CANARY=cluster/trees_mask_tasks_rs_canary.txt
LC=cluster/lstm_mask_tasks_canary.txt
LF=cluster/lstm_mask_tasks_fleet.txt
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-80G}
PACK_TIME=${PACK_TIME:-3:00:00}
POOL_CPUS=${POOL_CPUS:-20}
TUNED_MEM=${TUNED_MEM:-32G}
TUNED_TIME=${TUNED_TIME:-3:00:00}
LSTM_MEM=${LSTM_MEM:-32G}
LSTM1_TIME=${LSTM1_TIME:-8:00:00}   # REFIT_EVERY 1: ~10x the refits of the earlier 42 s - 13 min chunks
LSTM10_TIME=${LSTM10_TIME:-2:00:00}
LSTM1_CPUS=${LSTM1_CPUS:-$POOL_CPUS}  # the daily-refit fleet's pool (a chunk = one tuning period, the spec's minimum)
PART_SMALL=${PART_SMALL:-main,oneweek}
for f in specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_linear.py specs/causal_tune_lstm.py specs/causal_tune_lstm_jobs.py src/models/window_mask.py \
         experiments/trees_mask_gates.py experiments/trees_cadence_gates.py experiments/gate_lstm.py \
         experiments/trees_cadence_reduce_chunks.py experiments/reduce_trees_tuned_chunks.py \
         experiments/reduce_lstm_chunks.py experiments/trees_mask_kept_counts.py \
         cluster/slurm/trees_mask_gates.sbatch cluster/slurm/trees_mask_pack.sbatch \
         cluster/slurm/treestuned_mask_pack.sbatch cluster/slurm/lstm_mask_pack.sbatch \
         cluster/slurm/trees_mask_merge.sbatch "$UT" "$RS_CANARY" "$LC" "$LF" \
         cluster/trees_mask_tasks_rs10.txt cluster/trees_mask_tasks_rs1.txt gates_ref/specs/causal_tune_trees.py; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_trees_mask_carc.sh, locally)"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
SUBMIT=sbatch
LOG=logs/submitted_trees_mask_carc.txt
need() { for f in "$@"; do [ -f "$f" ] || { echo "$f missing: not passed yet"; exit 1; }; done; }

case "$MODE" in
  gates)
    mkdir -p "$G"
    IDS=""
    for GS in $(echo "${2:-cadence,lstm,h_trees,h_lstm}" | tr ',' ' '); do
      rm -f "$G/FINISHED_$GS"
      J=$($SUBMIT --parsable -J "tm_gates_$GS" --export=ALL,GATESET=$GS,OUT=$G cluster/slurm/trees_mask_gates.sbatch)
      IDS="$IDS gates_$GS=$J"
    done
    echo "$(date +%F_%T)$IDS" | tee -a "$LOG"
    ;;
  canary)
    need "$G/cadence/GATES_OK" "$G/h_trees/GATES_OK"
    mkdir -p "$U" "$T/rs10" "$T/rs1"
    rm -f "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    CU=$($SUBMIT --parsable -J tm_canary --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=t1:all_features:lgbm:0 \
         cluster/slurm/trees_mask_pack.sbatch)
    C10=$($SUBMIT --parsable -J tm_rs10_canary --array=1 --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
          --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs10,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs10/CANARY_OK \
          cluster/slurm/treestuned_mask_pack.sbatch)
    C1=$($SUBMIT --parsable -J tm_rs1_canary --array=1 --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
         --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs1,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs1/CANARY_OK \
         cluster/slurm/treestuned_mask_pack.sbatch)
    echo "$(date +%F_%T) canary_untuned=$CU canary_rs10=$C10 canary_rs1=$C1" | tee -a "$LOG"
    ;;
  fleet)
    need "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    [ -f "$LOG.fleet" ] && { echo "tree fleet already submitted ($(cat "$LOG.fleet")); delete $LOG.fleet by hand to resubmit (DONE chunks are skipped)"; exit 1; }
    NP=$(awk 'NF {print $1}' "$UT" | sort -n | tail -1)
    FU=$($SUBMIT --parsable -J tm_untuned --partition="$PART_SMALL" --array=1-"$NP" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$U,CANARY_FLAG=$U/CANARY_OK cluster/slurm/trees_mask_pack.sbatch)
    N10=$(grep -c . cluster/trees_mask_tasks_rs10.txt)
    F10=$($SUBMIT --parsable -J tm_rs10 --array=1-"$N10" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
          --export=ALL,TASKFILE=cluster/trees_mask_tasks_rs10.txt,RUNG=rs10,RESULTS_ROOT=$T,CANARY_FLAG=$T/rs10/CANARY_OK \
          cluster/slurm/treestuned_mask_pack.sbatch)
    N1=$(grep -c . cluster/trees_mask_tasks_rs1.txt)
    F1=$($SUBMIT --parsable -J tm_rs1 --array=1-"$N1" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
         --export=ALL,TASKFILE=cluster/trees_mask_tasks_rs1.txt,RUNG=rs1,RESULTS_ROOT=$T,CANARY_FLAG=$T/rs1/CANARY_OK \
         cluster/slurm/treestuned_mask_pack.sbatch)
    M=$($SUBMIT --parsable -J tm_merge --dependency=afterany:"$FU":"$F10":"$F1" \
        --export=ALL,PART=trees cluster/slurm/trees_mask_merge.sbatch)
    echo "$(date +%F_%T) fleet_untuned=$FU (packs 1-$NP) fleet_rs10=$F10 ($N10) fleet_rs1=$F1 ($N1) merge_trees=$M" | tee -a "$LOG" > "$LOG.fleet"
    cat "$LOG.fleet"
    ;;
  lstm_canary)
    need "$G/lstm/GATES_OK" "$G/h_lstm/GATES_OK"
    mkdir -p "$L1" "$L10"
    rm -f "$L1/CANARY_OK" "$L10/CANARY_OK"
    C1=$($SUBMIT --parsable -J lstm_mask_canary --cpus-per-task=$((2 * POOL_CPUS)) --mem=64G --time="$LSTM1_TIME" \
         --export=ALL,TASKFILE=$LC,RESULTS_ROOT=$L1,LSTM_RE=1,WRITE_FLAG=$L1/CANARY_OK,REPEAT_CHECK=1 \
         cluster/slurm/lstm_mask_pack.sbatch)
    C10=$($SUBMIT --parsable -J lstm_mask10_canary --cpus-per-task="$POOL_CPUS" --mem="$LSTM_MEM" --time="$LSTM10_TIME" \
          --export=ALL,TASKFILE=$LC,RESULTS_ROOT=$L10,LSTM_RE=10,WRITE_FLAG=$L10/CANARY_OK \
          cluster/slurm/lstm_mask_pack.sbatch)
    echo "$(date +%F_%T) canary_lstm1=$C1 canary_lstm10=$C10" | tee -a "$LOG"
    ;;
  lstm_fleet)
    need "$L1/CANARY_OK" "$L10/CANARY_OK"
    [ -f "$LOG.lstm_fleet" ] && { echo "LSTM fleet already submitted ($(cat "$LOG.lstm_fleet")); delete $LOG.lstm_fleet by hand to resubmit (DONE chunks are skipped)"; exit 1; }
    NL=$(grep -c . "$LF")
    F1=$($SUBMIT --parsable -J lstm_mask --partition="$PART_SMALL" --array=1-"$NL" --cpus-per-task="$LSTM1_CPUS" --mem="$LSTM_MEM" --time="$LSTM1_TIME" \
         --export=ALL,TASKFILE=$LF,RESULTS_ROOT=$L1,LSTM_RE=1,CANARY_FLAG=$L1/CANARY_OK cluster/slurm/lstm_mask_pack.sbatch)
    F10=$($SUBMIT --parsable -J lstm_mask10 --partition="$PART_SMALL" --array=1-"$NL" --cpus-per-task="$POOL_CPUS" --mem="$LSTM_MEM" --time="$LSTM10_TIME" \
          --export=ALL,TASKFILE=$LF,RESULTS_ROOT=$L10,LSTM_RE=10,CANARY_FLAG=$L10/CANARY_OK cluster/slurm/lstm_mask_pack.sbatch)
    M=$($SUBMIT --parsable -J lm_merge --dependency=afterany:"$F1":"$F10" \
        --export=ALL,PART=lstm cluster/slurm/trees_mask_merge.sbatch)
    echo "$(date +%F_%T) fleet_lstm1=$F1 ($NL) fleet_lstm10=$F10 ($NL) merge_lstm=$M" | tee -a "$LOG" > "$LOG.lstm_fleet"
    cat "$LOG.lstm_fleet"
    ;;
  kept)
    K=$($SUBMIT --parsable -J tm_kept --export=ALL,PART=kept cluster/slurm/trees_mask_merge.sbatch)
    echo "$(date +%F_%T) kept=$K" | tee -a "$LOG"
    ;;
  *)
    echo "usage: bash cluster/slurm/submit_trees_mask.sh gates [suites]|canary|fleet|lstm_canary|lstm_fleet|kept"
    exit 1
    ;;
esac
squeue -u "$USER" -o "%.14i %.18j %.3t %.10M %.5C %R" | grep -E "JOBID|tm_|lstm_mask|lm_" | head -14
