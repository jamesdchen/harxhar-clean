#!/bin/bash
# Per-bar LSTM forecasts of the 15:30-16:00 bar (specs/causal_tune_lstm.py; its
# pool workers run specs/causal_tune_lstm_jobs.py) for the buckets
# live_feasible, all_features and baseline at tw 2000, global lags, production
# HAR ladder -- the design of the per-bar linear and tuned-tree arms.
# Each arm is split into TIME CHUNKS of CHUNK_ROWS = 250 forecast rows (=
# TUNE_PER, one tuning period per chunk), one array task per chunk, each task a
# process pool of POOL_CPUS single-threaded workers.
# Canary first: ONE line, chunk 0 of live_feasible, run TWICE (REPEAT_CHECK=1:
# the two runs' results tables must be byte-identical -- determinism on this
# hardware); it writes $ROOT/CANARY_OK only if both succeed and agree.  The
# fleet is held on it with afterok and refuses to run unless CANARY_OK exists;
# the canary's chunk carries DONE, so the fleet skips it.  The scorer is held on
# the fleet with afterany: it merges every arm whose chunks are all DONE
# (experiments/reduce_lstm_chunks.py), lists the rest, scores what merged (best
# effort) and touches SCORED.
# REFUSES a second submission while logs/submitted_lstm_carc.txt exists (jobs
# cannot be cancelled from here): delete that file by hand to resubmit;
# DONE chunks are skipped, so a resubmission resumes.
#
# Task line: "<bucket> lstm <tw> <seg> <chunk> <start> <end> <halo>"; W = 2000
# rows; chunk k: start = W + 250 k, end = start + 250 except the last (-1);
# halo = W.  bar1600 has 1469 forecast rows (the per-bar tree and linear arms)
# -> 6 chunks per bucket, 18 lines.  The task files were generated once, locally:
# --- task-file generator ---
#   python - <<'EOF'
#   import math
#   W, CHUNK_ROWS, N_OOS = 2000, 250, 1469
#   BUCKETS = ["live_feasible", "all_features", "baseline"]
#   n_chunks = math.ceil(N_OOS / CHUNK_ROWS)
#   def chunks():
#       for k in range(n_chunks):
#           s = W + CHUNK_ROWS * k
#           yield k, s, (-1 if k == n_chunks - 1 else s + CHUNK_ROWS)
#   with open("cluster/lstm_tasks_fleet.txt", "w", newline="\n") as f:
#       for b in BUCKETS:
#           for k, s, e in chunks():
#               f.write(f"{b} lstm {W} bar1600 {k} {s} {e} {W}\n")
#   k, s, e = next(chunks())
#   with open("cluster/lstm_tasks_canary.txt", "w", newline="\n") as f:
#       f.write(f"live_feasible lstm {W} bar1600 {k} {s} {e} {W}\n")
#   EOF
# --- end task-file generator ---
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_lstm.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

# TIMINGS: see the header of results/linear_subsection_lstm/SUMMARY.md (local
# smoke, experiments/gate_lstm.py) -- the limits below carry a wide margin.
POOL_CPUS=${POOL_CPUS:-8}              # 16 configs x 5 seeds = 80 fits per tuning point / 8 = 10 waves
CANARY_TIME=${CANARY_TIME:-3:00:00}    # one live_feasible chunk, run twice
CANARY_MEM=${CANARY_MEM:-16G}          # 8 workers x (torch + one 2000-row window)
FLEET_TIME=${FLEET_TIME:-2:30:00}      # one chunk (all_features is the widest design)
FLEET_MEM=${FLEET_MEM:-16G}
FLEET_CONC=${FLEET_CONC:-18}           # every chunk at once (18 lines)

CANARY=cluster/lstm_tasks_canary.txt
FLEET=cluster/lstm_tasks_fleet.txt
ROOT=${RESULTS_ROOT:-results/linear_subsection_lstm}
FLAG=$ROOT/CANARY_OK
STAMP=logs/submitted_lstm_carc.txt
if [ -f "$STAMP" ]; then
  echo "already submitted ($(cat "$STAMP")); refusing a second submission -- delete $STAMP by hand to resubmit"
  exit 1
fi
for f in specs/causal_tune_lstm.py specs/causal_tune_lstm_jobs.py specs/causal_tune_linear.py \
         specs/causal_tune_trees.py specs/causal_tune_trees_tuned_jobs.py \
         experiments/reduce_lstm_chunks.py experiments/score_lstm_subsection.py \
         experiments/build_subsection_lstm_yhat.py \
         cluster/slurm/lstm_pack.sbatch cluster/slurm/lstm_score.sbatch "$CANARY" "$FLEET"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_lstm_carc.sh, locally)"; exit 1; }
done
mkdir -p "$ROOT"
rm -f "$FLAG" "$ROOT/SCORED"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J lstm_canary --cpus-per-task="$POOL_CPUS" --mem="$CANARY_MEM" --time="$CANARY_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$FLAG,REPEAT_CHECK=1 cluster/slurm/lstm_pack.sbatch)
FL=$($SUBMIT --parsable -J lstm_fleet --array=1-"$(wc -l < "$FLEET")"%"$FLEET_CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task="$POOL_CPUS" --mem="$FLEET_MEM" --time="$FLEET_TIME" \
      --export=ALL,TASKFILE=$FLEET,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/lstm_pack.sbatch)
S=$($SUBMIT --parsable -J lstm_score --dependency=afterany:"$FL" \
      --export=ALL,RESULTS_ROOT=$ROOT cluster/slurm/lstm_score.sbatch)
echo "canary=$CAN fleet=$FL score=$S $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.6D %R" | grep -E "JOBID|lstm" | head -10
