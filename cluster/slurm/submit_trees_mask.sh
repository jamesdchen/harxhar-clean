#!/bin/bash
# I9 (2026-09-29 evening; plan writeup/CAMPAIGN_16H_2026-09-29.md, "Restart with a per-window
# mask"): the MASKED 16:00 re-run, WINDOW_MASK=1, SEGMENT bar1600, TRAIN_WIN 2000, de-duplicated
# per-bar design, LightGBM / XGBoost / RF x baseline / live_feasible / all_features + the LSTM:
#   T10   specs/causal_tune_trees.py        REFIT_EVERY 10, IMPORTANCE_EVERY 1
#   T1    specs/causal_tune_trees.py        REFIT_EVERY 1,  IMPORTANCE_EVERY 10
#   RS10  specs/causal_tune_trees_tuned.py  REFIT_EVERY 10 (random search, 32 candidates, TUNE_PER 250)
#   RS1   specs/causal_tune_trees_tuned.py  REFIT_EVERY 1, TreeSHAP off
#   LSTM  specs/causal_tune_lstm.py         as its spec (tune every 250, refit every 10)
# Run in the deployment root /scratch1/jc_905/harxhar-mask (cluster/slurm/ship_trees_mask_carc.sh).
# Three stages, each submitted by hand after the previous one was read:
#   bash cluster/slurm/submit_trees_mask.sh gates    three concurrent gate jobs (GATESET cadence /
#        lstm / h: cluster/slurm/trees_mask_gates.sbatch)
#   bash cluster/slurm/submit_trees_mask.sh canary   refuses unless the three GATES_OK flags exist;
#        untuned pack 0 (chunk c0 of all 18 T10 / T1 arms + one chunk re-run alone, compared bit
#        for bit), the tuned canaries (chunk 0 of live_feasible x lgbm,xgb,rf for RS10 and RS1),
#        the LSTM canary (chunk 0 of live_feasible, run twice: byte-identical tables)
#   bash cluster/slurm/submit_trees_mask.sh fleet    refuses unless the four CANARY_OK flags exist;
#        untuned packs 1..K, RS10 and RS1 arrays (one chunk-arm per task), the LSTM array (its
#        canary chunk carries DONE and is skipped), the merge job held afterany on all of them
# Every fleet task also refuses to run without its canary flag.
#
# SIZING (the campaign's rule: ~20-CPU tasks of single-threaded processes, so the 2000-CPU cap
# binds before the 100-job cap).  Untuned: one array task = one pack of <= 20 single-threaded
# chunk processes (the task file of the unmasked I2 run: chunks of ~600 s estimated at the
# unmasked refit seconds; a masked fit sees fewer columns).  Tuned: one array task = one
# chunk-arm with a pool of 20 single-threaded workers.  LSTM: one array task = one chunk, a
# pool of 20.  Arrays of <= 50 tasks may also use the oneweek partition (PART_SMALL; the
# oneweek QOS allows 50 submitted jobs per user), the 54-task tuned arrays stay on main.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE="${1:-}"
U=results/linear_subsection_trees_mask
T=results/linear_subsection_trees_tuned_mask
L=results/linear_subsection_lstm_mask
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
LSTM_TIME=${LSTM_TIME:-2:00:00}
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

case "$MODE" in
  gates)
    mkdir -p "$G"
    rm -f "$G"/FINISHED_*
    IDS=""
    for GS in cadence lstm h; do
      J=$($SUBMIT --parsable -J "tm_gates_$GS" --export=ALL,GATESET=$GS,OUT=$G cluster/slurm/trees_mask_gates.sbatch)
      IDS="$IDS $GS=$J"
    done
    echo "$(date +%F_%T) gates$IDS" | tee -a "$LOG"
    ;;
  canary)
    for f in "$G/cadence/GATES_OK" "$G/lstm/GATES_OK" "$G/h/GATES_OK"; do
      [ -f "$f" ] || { echo "$f missing: the gates have not passed"; exit 1; }
    done
    mkdir -p "$U" "$T/rs10" "$T/rs1" "$L"
    rm -f "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK" "$L/CANARY_OK"
    CU=$($SUBMIT --parsable -J tm_canary --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=t1:all_features:lgbm:0 \
         cluster/slurm/trees_mask_pack.sbatch)
    C10=$($SUBMIT --parsable -J tm_rs10_canary --array=1 --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
          --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs10,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs10/CANARY_OK \
          cluster/slurm/treestuned_mask_pack.sbatch)
    C1=$($SUBMIT --parsable -J tm_rs1_canary --array=1 --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
         --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs1,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs1/CANARY_OK \
         cluster/slurm/treestuned_mask_pack.sbatch)
    CL=$($SUBMIT --parsable -J lstm_mask_canary --cpus-per-task="$POOL_CPUS" --mem="$LSTM_MEM" --time="$LSTM_TIME" \
         --export=ALL,TASKFILE=$LC,RESULTS_ROOT=$L,WRITE_FLAG=$L/CANARY_OK,REPEAT_CHECK=1 \
         cluster/slurm/lstm_mask_pack.sbatch)
    echo "$(date +%F_%T) canary untuned=$CU rs10=$C10 rs1=$C1 lstm=$CL" | tee -a "$LOG"
    ;;
  fleet)
    for f in "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK" "$L/CANARY_OK"; do
      [ -f "$f" ] || { echo "$f missing: a canary has not passed"; exit 1; }
    done
    [ -f "$LOG.fleet" ] && { echo "fleet already submitted ($(cat "$LOG.fleet")); delete $LOG.fleet by hand to resubmit (DONE chunks are skipped)"; exit 1; }
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
    NL=$(grep -c . "$LF")
    FL=$($SUBMIT --parsable -J lstm_mask --partition="$PART_SMALL" --array=1-"$NL" --cpus-per-task="$POOL_CPUS" --mem="$LSTM_MEM" --time="$LSTM_TIME" \
         --export=ALL,TASKFILE=$LF,RESULTS_ROOT=$L,CANARY_FLAG=$L/CANARY_OK cluster/slurm/lstm_mask_pack.sbatch)
    M=$($SUBMIT --parsable -J tm_merge --dependency=afterany:"$FU":"$F10":"$F1":"$FL" \
        --export=ALL,UNTUNED_ROOT=$U,TUNED_ROOT=$T,LSTM_ROOT=$L cluster/slurm/trees_mask_merge.sbatch)
    echo "$(date +%F_%T) fleet untuned=$FU (packs 1-$NP) rs10=$F10 ($N10) rs1=$F1 ($N1) lstm=$FL ($NL) merge=$M" | tee -a "$LOG" > "$LOG.fleet"
    cat "$LOG.fleet"
    ;;
  *)
    echo "usage: bash cluster/slurm/submit_trees_mask.sh gates|canary|fleet"
    exit 1
    ;;
esac
squeue -u "$USER" -o "%.14i %.18j %.3t %.10M %.5C %R" | grep -E "JOBID|tm_|lstm_mask" | head -14
