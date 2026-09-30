#!/bin/bash
# I10 (2026-09-29): stage 0 + stage 1 of the intraday-sequence LSTM (specs/causal_tune_lstm_intraday.py)
# for the buckets live_feasible, all_features, baseline at tw 2000, bar1600.
#   gates A  identity / seq_end / mask_flatten (3 buckets)                 debug, 1 h
#   gates B  determinism / pool / chunk / causal x2 / mask / end-to-end     debug, 1 h
#   canary   the real-budget tuning point live_feasible row 0 (STAGE=tune)  -> CANARY_OK
#   stage 1  every tuning point of every bucket (cluster/lstm_intraday_tasks_tune.txt, 18 lines;
#            the canary's line carries DONE and is skipped) -> $ROOT/tune_cache/<bucket>/tune_R*.npz
# The gates and the canary run at once; stage 1 is held on all three with afterany (no job can be
# cancelled from here, so nothing is left pending forever) and refuses to run unless GATES_A_OK,
# GATES_B_OK and CANARY_OK exist.  Stage 2 (the refit chunks, sized from stage 1's measured fit
# times) is submitted by cluster/slurm/submit_lstm_intraday_stage2.sh once stage 1 is done.
# REFUSES a second submission while logs/submitted_lstm_intraday_carc.txt exists (delete it by hand
# to resubmit; DONE lines are skipped).
#   cd /scratch1/jc_905/harxhar-lstmi && bash cluster/slurm/submit_lstm_intraday.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

POOL_CPUS=${POOL_CPUS:-20}          # 24 configs x 5 seeds = 120 fits per tuning point / 20 = 6 waves
GATE_CPUS=${GATE_CPUS:-20}
TUNE_TIME=${TUNE_TIME:-2:00:00}     # one tuning point (<= 200 epochs per fit, early stopping)
TUNE_MEM=${TUNE_MEM:-40G}           # 20 workers x (window bars + sequences, all_features <= ~0.4 GB) + the panel
PARTS=${PARTS:-main,oneweek}        # arrays of <= 50 tasks may list oneweek (campaign note)

ROOT=${RESULTS_ROOT:-results/linear_subsection_lstm_intraday}
CANARY=cluster/lstm_intraday_tasks_canary.txt
TUNE=cluster/lstm_intraday_tasks_tune.txt
STAMP=logs/submitted_lstm_intraday_carc.txt
if [ -f "$STAMP" ]; then
  echo "already submitted ($(cat "$STAMP")); refusing a second submission -- delete $STAMP by hand to resubmit"
  exit 1
fi
for f in specs/causal_tune_lstm_intraday.py specs/causal_tune_lstm_intraday_jobs.py specs/causal_tune_lstm.py \
         specs/causal_tune_lstm_jobs.py specs/causal_tune_linear.py specs/causal_tune_trees_tuned_jobs.py \
         src/backtest/executor.py src/models/window_mask.py experiments/gate_lstm_intraday.py experiments/gate_lstm.py \
         experiments/reduce_lstm_intraday_chunks.py cluster/slurm/lstm_intraday_pack.sbatch \
         cluster/slurm/lstm_intraday_gates.sbatch "$CANARY" "$TUNE"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_lstm_intraday_carc.sh, locally)"; exit 1; }
done
mkdir -p "$ROOT"
rm -f "$ROOT/GATES_A_OK" "$ROOT/GATES_B_OK" "$ROOT/CANARY_OK"
GA=$(sbatch --parsable -J lstmi_gatesA --cpus-per-task="$GATE_CPUS" \
      --export=ALL,GATE_SET=A,RESULTS_ROOT=$ROOT cluster/slurm/lstm_intraday_gates.sbatch)
GB=$(sbatch --parsable -J lstmi_gatesB --cpus-per-task="$GATE_CPUS" \
      --export=ALL,GATE_SET=B,RESULTS_ROOT=$ROOT cluster/slurm/lstm_intraday_gates.sbatch)
CAN=$(sbatch --parsable -J lstmi_canary --partition="$PARTS" --cpus-per-task="$POOL_CPUS" --mem="$TUNE_MEM" --time="$TUNE_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$ROOT/CANARY_OK cluster/slurm/lstm_intraday_pack.sbatch)
S1=$(sbatch --parsable -J lstmi_tune --partition="$PARTS" --array=1-"$(wc -l < "$TUNE")" \
      --dependency=afterany:"$GA":"$GB":"$CAN" --cpus-per-task="$POOL_CPUS" --mem="$TUNE_MEM" --time="$TUNE_TIME" \
      --export=ALL,TASKFILE=$TUNE,RESULTS_ROOT=$ROOT,REQUIRE_FLAGS="$ROOT/GATES_A_OK $ROOT/GATES_B_OK $ROOT/CANARY_OK" \
      cluster/slurm/lstm_intraday_pack.sbatch)
echo "gatesA=$GA gatesB=$GB canary=$CAN stage1=$S1 $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.14i %.16j %.4t %.10M %.5C %.9P %R" | grep -E "JOBID|lstmi" | head -12
