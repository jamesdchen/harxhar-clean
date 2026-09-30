#!/bin/bash
# The 16:00 tree cadence campaign (checklist I2; plan writeup/CAMPAIGN_16H_2026-09-29.md):
# per-bar LightGBM / XGBoost / RF x baseline / live_feasible / all_features at the
# 15:30-16:00 bar, TRAIN_WIN 2000, de-duplicated per-bar design (commit 47f7f9c):
#   T10   specs/causal_tune_trees.py        REFIT_EVERY 10, IMPORTANCE_EVERY 1
#   T1    specs/causal_tune_trees.py        REFIT_EVERY 1,  IMPORTANCE_EVERY 10
#   RS10  specs/causal_tune_trees_tuned.py  REFIT_EVERY 10 (random search, 32 candidates, TUNE_PER 250)
#   RS1   specs/causal_tune_trees_tuned.py  REFIT_EVERY 1, TreeSHAP off
# Three stages, each submitted by hand after the previous one was read:
#   bash cluster/slurm/submit_trees_cadence.sh gates    gates g0-g4 (one 20-cpu job)
#   bash cluster/slurm/submit_trees_cadence.sh canary   untuned pack 0 (chunk c0 of all 18
#        T10/T1 arms at once + one chunk re-run alone, compared bit for bit) and the tuned
#        canaries (chunk 0 of live_feasible x lgbm,xgb,rf for RS10 and for RS1)
#   bash cluster/slurm/submit_trees_cadence.sh fleet    untuned packs 1..K, RS10 and RS1
#        arrays (one chunk-arm per task), the merge job held afterany on all three
# The fleet stage refuses to submit unless the three CANARY_OK flags exist, and every
# fleet task refuses to run without its flag.
#
# SIZING.  Every fit is single-threaded.  Untuned: one array task = one PACK of <= 20
# chunk processes on 20 cores (experiments/trees_cadence_make_tasks.py: chunks of ~600 s
# at the one-thread refit seconds of the earlier 16:00 arms; ~28 core-h in all).  Memory:
# the all_features data preparation peaks at 5.7 GiB and holds 3.2 GiB while fitting
# (measured locally, 2026-09-29), live_feasible less; 20 processes started STAGGER = 10 s
# apart stay under 80 GB.  Tuned: one array task = one chunk-arm (32 candidates + its
# 250 refits, a pool of 20 single-threaded workers).  Tasks of 20 cores let the
# 2000-cpu cap bind before the 100-job cap.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
MODE="${1:-}"
U=results/linear_subsection_trees_dedup
T=results/linear_subsection_trees_tuned_dedup
UT=cluster/trees_cadence_tasks_untuned.txt
RS_CANARY=cluster/trees_cadence_tasks_rs_canary.txt
PACK_CPUS=${PACK_CPUS:-20}
PACK_MEM=${PACK_MEM:-80G}
PACK_TIME=${PACK_TIME:-2:00:00}   # a pack's processes are ~600 s each (estimate); 12x margin
POOL_CPUS=${POOL_CPUS:-20}
TUNED_MEM=${TUNED_MEM:-32G}       # main process (<= 5.7 GiB peak) + 20 workers
TUNED_TIME=${TUNED_TIME:-2:00:00} # an RS1 chunk-arm: ~6k core-s worst case / 20 workers
for f in specs/causal_tune_trees.py specs/causal_tune_trees_tuned.py specs/causal_tune_trees_tuned_jobs.py \
         specs/causal_tune_linear.py src/backtest/executor.py experiments/trees_cadence_gates.py \
         experiments/trees_cadence_reduce_chunks.py experiments/reduce_trees_tuned_chunks.py \
         cluster/slurm/trees_cadence_pack.sbatch cluster/slurm/treestuned_cadence_pack.sbatch \
         cluster/slurm/trees_cadence_merge.sbatch cluster/slurm/trees_cadence_gates.sbatch \
         "$UT" "$RS_CANARY" cluster/trees_cadence_tasks_rs10.txt cluster/trees_cadence_tasks_rs1.txt; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_trees_cadence_carc.sh, locally)"; exit 1; }
done
[ -d pylib_trees/shap ] || { echo "pylib_trees/shap missing"; exit 1; }
SUBMIT=sbatch
LOG=logs/submitted_trees_cadence_carc.txt

case "$MODE" in
  gates)
    G=$($SUBMIT --parsable -J tc_gates cluster/slurm/trees_cadence_gates.sbatch)
    echo "$(date +%F_%T) gates=$G" | tee -a "$LOG"
    ;;
  canary)
    [ -f "$U/gates/GATES_OK" ] || { echo "$U/gates/GATES_OK missing: the gates have not passed"; exit 1; }
    mkdir -p "$U" "$T/rs10" "$T/rs1"
    rm -f "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"
    CU=$($SUBMIT --parsable -J tc_canary --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$U,WRITE_FLAG=$U/CANARY_OK,ALONE_CHECK=t1:all_features:lgbm:0 \
         cluster/slurm/trees_cadence_pack.sbatch)
    C10=$($SUBMIT --parsable -J tc_rs10_canary --array=1 --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
          --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs10,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs10/CANARY_OK \
          cluster/slurm/treestuned_cadence_pack.sbatch)
    C1=$($SUBMIT --parsable -J tc_rs1_canary --array=1 --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
         --export=ALL,TASKFILE=$RS_CANARY,RUNG=rs1,RESULTS_ROOT=$T,WRITE_FLAG=$T/rs1/CANARY_OK \
         cluster/slurm/treestuned_cadence_pack.sbatch)
    echo "$(date +%F_%T) canary untuned=$CU rs10=$C10 rs1=$C1" | tee -a "$LOG"
    ;;
  fleet)
    for f in "$U/CANARY_OK" "$T/rs10/CANARY_OK" "$T/rs1/CANARY_OK"; do
      [ -f "$f" ] || { echo "$f missing: a canary has not passed"; exit 1; }
    done
    NP=$(awk 'NF {print $1}' "$UT" | sort -n | tail -1)
    FU=$($SUBMIT --parsable -J tc_untuned --array=1-"$NP" --cpus-per-task="$PACK_CPUS" --mem="$PACK_MEM" --time="$PACK_TIME" \
         --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$U,CANARY_FLAG=$U/CANARY_OK cluster/slurm/trees_cadence_pack.sbatch)
    N10=$(grep -c . cluster/trees_cadence_tasks_rs10.txt)
    F10=$($SUBMIT --parsable -J tc_rs10 --array=1-"$N10" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
          --export=ALL,TASKFILE=cluster/trees_cadence_tasks_rs10.txt,RUNG=rs10,RESULTS_ROOT=$T,CANARY_FLAG=$T/rs10/CANARY_OK \
          cluster/slurm/treestuned_cadence_pack.sbatch)
    N1=$(grep -c . cluster/trees_cadence_tasks_rs1.txt)
    F1=$($SUBMIT --parsable -J tc_rs1 --array=1-"$N1" --cpus-per-task="$POOL_CPUS" --mem="$TUNED_MEM" --time="$TUNED_TIME" \
         --export=ALL,TASKFILE=cluster/trees_cadence_tasks_rs1.txt,RUNG=rs1,RESULTS_ROOT=$T,CANARY_FLAG=$T/rs1/CANARY_OK \
         cluster/slurm/treestuned_cadence_pack.sbatch)
    M=$($SUBMIT --parsable -J tc_merge --dependency=afterany:"$FU":"$F10":"$F1" \
        --export=ALL,UNTUNED_ROOT=$U,TUNED_ROOT=$T cluster/slurm/trees_cadence_merge.sbatch)
    echo "$(date +%F_%T) fleet untuned=$FU (packs 1-$NP) rs10=$F10 ($N10) rs1=$F1 ($N1) merge=$M" | tee -a "$LOG"
    ;;
  *)
    echo "usage: bash cluster/slurm/submit_trees_cadence.sh gates|canary|fleet"
    exit 1
    ;;
esac
squeue -u "$USER" -o "%.12i %.18j %.3t %.10M %.5C %R" | head -12
