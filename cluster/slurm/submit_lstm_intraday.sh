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
GATE_PARTS=${GATE_PARTS:-debug,main}  # debug (1 h max) or main, whichever starts first
GATE_MEM=${GATE_MEM:-32G}
TUNE_TIME=${TUNE_TIME:-1:30:00}     # one tuning point (<= 200 epochs per fit, early stopping)
TUNE_MEM=${TUNE_MEM:-40G}           # 20 workers x (window bars + sequences, all_features <= ~0.4 GB) + the panel
PARTS=${PARTS:-main,oneweek}        # arrays of <= 50 tasks may list oneweek (campaign note)

ROOT=${RESULTS_ROOT:-results/linear_subsection_lstm_intraday}
CANARY=cluster/lstm_intraday_tasks_canary.txt
TUNE=cluster/lstm_intraday_tasks_tune.txt
STAMP=logs/submitted_lstm_intraday_carc.txt
# ONLY_STAGE1=1: submit the stage-1 array alone (the first submission's stage-1 sbatch was refused by
# a QOS submit limit on oneweek after the gates and the canary had gone in), on S1_PARTS, held with
# afterany on the job ids in AFTER (colon-separated, e.g. a gates job still running); it still refuses
# to run without the three flags.
if [ "${ONLY_STAGE1:-0}" = "1" ]; then
  S1_PARTS=${S1_PARTS:-main}
  S1_TIME=${S1_TIME:-0:40:00}  # the canary's tuning point took 165 s on 20 CPUs
  S1=$(sbatch --parsable -J lstmi_tune --partition="$S1_PARTS" --array=1-"$(wc -l < "$TUNE")" \
        ${AFTER:+--dependency=afterany:$AFTER} --cpus-per-task="$POOL_CPUS" --mem="$TUNE_MEM" --time="$S1_TIME" \
        --export=ALL,TASKFILE=$TUNE,RESULTS_ROOT=$ROOT,REQUIRE_FLAGS="$ROOT/GATES_A_OK $ROOT/GATES_B_OK $ROOT/CANARY_OK" \
        cluster/slurm/lstm_intraday_pack.sbatch)
  echo "${EARLIER:+first submission: $EARLIER; }stage1=$S1 (submitted alone, after ${AFTER:-nothing}) $(date)" | tee -a "$STAMP"
  exit 0
fi
# PINNED_RERUN=1 CONSTRAINT=<feature>: re-run gates B and stage 1 on ONE CPU class.  The first stage 2
# (12483604) found the executor's bar-level inputs differ in their last bits between Intel (xeon-4116)
# and AMD (epyc-75xx) nodes: the tuning cache's fingerprint (sha256 of the window's bars) refused 5 of 25
# chunks whose tuning point had been computed on the other vendor.  Everything from that mixed-class
# attempt is moved aside with the suffix _$TAG (tune_cache, every arm's chunks / stage_tune / merged
# directories, the merge outputs, GATES_B_OK), never deleted; gates B (determinism, pool, chunk, causal,
# mask, end to end) re-run on the class; stage 1 is held on them.
if [ "${PINNED_RERUN:-0}" = "1" ]; then
  : "${CONSTRAINT:?set CONSTRAINT, e.g. epyc-7513}"
  TAG=${TAG:-mixedclass_$(date +%Y%m%d%H%M)}
  for d in "$ROOT/tune_cache" "$ROOT"/*/bar1600/lstm_intraday/tw2000/chunks "$ROOT"/*/bar1600/lstm_intraday/tw2000/stage_tune \
           "$ROOT"/*/bar1600/lstm_intraday/tw2000/causal_tune_lstm_intraday "$ROOT/MERGED_ALL" "$ROOT/reduce.log" \
           "$ROOT/reduce_gates.csv" "$ROOT/stage2_plan.csv" "$ROOT/GATES_B_OK" "$ROOT/gates/scratch_B" \
           "$ROOT/gates/gate_lstm_intraday_causal_bar_causal_target_chunk_determinism_e2e_mask_pool.csv"; do
    if [ -e "$d" ]; then mv "$d" "${d}_$TAG" && echo "moved aside: $d -> ${d}_$TAG"; fi
  done
  S1_TIME=${S1_TIME:-0:40:00}
  GB=$(sbatch --parsable -J lstmi_gatesB --partition=main --constraint="$CONSTRAINT" --mem="$GATE_MEM" \
        --cpus-per-task="$GATE_CPUS" --export=ALL,GATE_SET=B,RESULTS_ROOT=$ROOT cluster/slurm/lstm_intraday_gates.sbatch)
  S1=$(sbatch --parsable -J lstmi_tune --partition=main --constraint="$CONSTRAINT" --array=1-"$(wc -l < "$TUNE")" \
        --dependency=afterany:"$GB" --cpus-per-task="$POOL_CPUS" --mem="$TUNE_MEM" --time="$S1_TIME" \
        --export=ALL,TASKFILE=$TUNE,RESULTS_ROOT=$ROOT,REQUIRE_FLAGS="$ROOT/GATES_A_OK $ROOT/GATES_B_OK $ROOT/CANARY_OK" \
        cluster/slurm/lstm_intraday_pack.sbatch)
  echo "pinned rerun ($CONSTRAINT, earlier outputs -> *_$TAG): gatesB=$GB stage1=$S1 $(date)" | tee -a "$STAMP"
  exit 0
fi
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
GA=$(sbatch --parsable -J lstmi_gatesA --partition="$GATE_PARTS" --mem="$GATE_MEM" --cpus-per-task="$GATE_CPUS" \
      --export=ALL,GATE_SET=A,RESULTS_ROOT=$ROOT cluster/slurm/lstm_intraday_gates.sbatch)
GB=$(sbatch --parsable -J lstmi_gatesB --partition="$GATE_PARTS" --mem="$GATE_MEM" --cpus-per-task="$GATE_CPUS" \
      --export=ALL,GATE_SET=B,RESULTS_ROOT=$ROOT cluster/slurm/lstm_intraday_gates.sbatch)
CAN=$(sbatch --parsable -J lstmi_canary --partition="$PARTS" --cpus-per-task="$POOL_CPUS" --mem="$TUNE_MEM" --time="$TUNE_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$ROOT/CANARY_OK cluster/slurm/lstm_intraday_pack.sbatch)
S1=$(sbatch --parsable -J lstmi_tune --partition="$PARTS" --array=1-"$(wc -l < "$TUNE")" \
      --dependency=afterany:"$GA":"$GB":"$CAN" --cpus-per-task="$POOL_CPUS" --mem="$TUNE_MEM" --time="$TUNE_TIME" \
      --export=ALL,TASKFILE=$TUNE,RESULTS_ROOT=$ROOT,REQUIRE_FLAGS="$ROOT/GATES_A_OK $ROOT/GATES_B_OK $ROOT/CANARY_OK" \
      cluster/slurm/lstm_intraday_pack.sbatch)
echo "gatesA=$GA gatesB=$GB canary=$CAN stage1=$S1 $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.14i %.16j %.4t %.10M %.5C %.9P %R" | grep -E "JOBID|lstmi" | head -12
