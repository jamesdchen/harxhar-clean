#!/bin/bash
# I1 (2026-09-29): the per-bar LSTM of the 15:30-16:00 bar (specs/causal_tune_lstm.py, read-only;
# pool workers specs/causal_tune_lstm_jobs.py) re-run on the DE-DUPLICATED per-bar design (commit
# 47f7f9c) for the buckets live_feasible, all_features and baseline at tw 2000, global lags,
# production HAR ladder.  Twin of cluster/slurm/submit_lstm.sh; the changes: its own deployment
# root (cluster/slurm/ship_lstm_dedup_carc.sh), the results root results/linear_subsection_lstm_dedup,
# POOL_CPUS = 20 a task (the campaign's sizing: the CPU cap binds, not the job cap; every fit is
# single-threaded, so no number depends on it), and a merge-only final job.
# Each arm is split into TIME CHUNKS of 250 forecast rows (= TUNE_PER), one array task per chunk
# (cluster/lstm_dedup_tasks_fleet.txt = the 18 lines of cluster/lstm_tasks_fleet.txt).
# Canary first: chunk 0 of live_feasible run TWICE (REPEAT_CHECK=1: the two results tables must be
# byte-identical); it writes $ROOT/CANARY_OK only if both succeed and agree.  The fleet is held on it
# with afterok and refuses to run without CANARY_OK (the canary's chunk carries DONE, so the fleet
# skips it).  The merge job is held on the fleet with afterany.
# REFUSES a second submission while logs/submitted_lstm_dedup_carc.txt exists (jobs cannot be
# cancelled from here): delete it by hand to resubmit; DONE chunks are skipped, so it resumes.
#
#   cd /scratch1/jc_905/harxhar-dedup && bash cluster/slurm/submit_lstm_dedup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

POOL_CPUS=${POOL_CPUS:-20}             # 16 configs x 5 seeds = 80 fits per tuning point / 20 = 4 waves
CANARY_TIME=${CANARY_TIME:-1:30:00}    # one live_feasible chunk, run twice (8 CPUs before: 4.6 min)
CANARY_MEM=${CANARY_MEM:-32G}          # 20 workers x (torch + one 2000-row window)
FLEET_TIME=${FLEET_TIME:-1:30:00}      # one chunk (8 CPUs before: 1-13 min)
FLEET_MEM=${FLEET_MEM:-32G}
FLEET_CONC=${FLEET_CONC:-18}           # every chunk at once (18 lines)

CANARY=cluster/lstm_dedup_tasks_canary.txt
FLEET=cluster/lstm_dedup_tasks_fleet.txt
ROOT=${RESULTS_ROOT:-results/linear_subsection_lstm_dedup}
FLAG=$ROOT/CANARY_OK
STAMP=logs/submitted_lstm_dedup_carc.txt
if [ -f "$STAMP" ]; then
  echo "already submitted ($(cat "$STAMP")); refusing a second submission -- delete $STAMP by hand to resubmit"
  exit 1
fi
for f in specs/causal_tune_lstm.py specs/causal_tune_lstm_jobs.py specs/causal_tune_linear.py \
         specs/causal_tune_trees.py specs/causal_tune_trees_tuned_jobs.py src/backtest/executor.py \
         experiments/reduce_lstm_chunks.py cluster/slurm/lstm_dedup_pack.sbatch \
         cluster/slurm/lstm_dedup_score.sbatch "$CANARY" "$FLEET"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_lstm_dedup_carc.sh, locally)"; exit 1; }
done
mkdir -p "$ROOT"
rm -f "$FLAG" "$ROOT/MERGED_ALL"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J lstm_dedup_canary --cpus-per-task="$POOL_CPUS" --mem="$CANARY_MEM" --time="$CANARY_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$FLAG,REPEAT_CHECK=1 cluster/slurm/lstm_dedup_pack.sbatch)
FL=$($SUBMIT --parsable -J lstm_dedup_fleet --array=1-"$(wc -l < "$FLEET")"%"$FLEET_CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task="$POOL_CPUS" --mem="$FLEET_MEM" --time="$FLEET_TIME" \
      --export=ALL,TASKFILE=$FLEET,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/lstm_dedup_pack.sbatch)
S=$($SUBMIT --parsable -J lstm_dedup_merge --dependency=afterany:"$FL" \
      --export=ALL,RESULTS_ROOT=$ROOT cluster/slurm/lstm_dedup_score.sbatch)
echo "canary=$CAN fleet=$FL merge=$S $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.12i %.18j %.4t %.10M %.5C %R" | grep -E "JOBID|lstm_dedup" | head -10
