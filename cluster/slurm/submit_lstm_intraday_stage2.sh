#!/bin/bash
# I10 (2026-09-29): stage 2 of the intraday-sequence LSTM -- the refit chunks.  Requires every stage-1
# tuning record (results/.../tune_cache/<bucket>/tune_R*.npz) and the gate / canary flags; plans the
# chunks from the records' measured fit times (cluster/lstm_intraday_make_stage2.py ->
# cluster/lstm_intraday_tasks_fleet.txt, results/.../stage2_plan.csv), submits them as arrays of <= 50
# tasks (main: a oneweek listing was refused by its QOS submit limit on 2026-09-30), each chunk reading its
# tuning point from the cache (REQUIRE_CACHE=1), and the merge job held on all of them with afterany.
# REFUSES a second submission while logs/submitted_lstm_intraday_stage2_carc.txt exists, unless
# RESUBMIT=1 (the stamp is then kept as <stamp>.<epoch>; DONE chunks are skipped).
# The planner runs in a SUBSHELL with the conda env: the jobs are submitted from the plain login
# environment (--export=ALL), because an env already active when the job script activates it again
# is not re-put in front of PATH after its module load (the first stage-2 array, 12482776, failed on
# a python without pandas that way).
#   cd /scratch1/jc_905/harxhar-lstmi && bash cluster/slurm/submit_lstm_intraday_stage2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL_CPUS=${POOL_CPUS:-20}
FLEET_TIME=${FLEET_TIME:-1:00:00}
FLEET_MEM=${FLEET_MEM:-32G}
PARTS=${PARTS:-main}
CONSTRAINT=${CONSTRAINT:-}  # one CPU class for every chunk (see submit_lstm_intraday.sh, PINNED_RERUN)
ARRAY_MAX=50
ROOT=${RESULTS_ROOT:-results/linear_subsection_lstm_intraday}
FLEET=cluster/lstm_intraday_tasks_fleet.txt
STAMP=logs/submitted_lstm_intraday_stage2_carc.txt
if [ -f "$STAMP" ]; then
  if [ "${RESUBMIT:-0}" = "1" ]; then
    mv "$STAMP" "$STAMP.$(date +%s)"
  else
    echo "already submitted ($(cat "$STAMP")); refusing -- RESUBMIT=1 to resubmit (DONE chunks are skipped)"; exit 1
  fi
fi
for f in "$ROOT/GATES_A_OK" "$ROOT/GATES_B_OK" "$ROOT/CANARY_OK"; do
  [ -f "$f" ] || { echo "$f missing: gates / canary not passed"; exit 1; }
done
N_CACHE=$(ls "$ROOT"/tune_cache/*/tune_R*.npz 2>/dev/null | wc -l)
N_TUNE=$(wc -l < cluster/lstm_intraday_tasks_tune.txt)
[ "$N_CACHE" -eq "$N_TUNE" ] || { echo "stage 1 incomplete: $N_CACHE of $N_TUNE tuning records"; exit 1; }
( module load conda >/dev/null 2>&1 && source activate harxhar     && python cluster/lstm_intraday_make_stage2.py --root "$ROOT" --nt "$POOL_CPUS" --out "$FLEET" )
N=$(wc -l < "$FLEET")
rm -f cluster/lstm_intraday_tasks_fleet_part*.txt
split -l "$ARRAY_MAX" -d -a 2 --additional-suffix=.txt "$FLEET" cluster/lstm_intraday_tasks_fleet_part
IDS=()
for part in cluster/lstm_intraday_tasks_fleet_part*.txt; do
  J=$(sbatch --parsable -J lstmi_refit --partition="$PARTS" ${CONSTRAINT:+--constraint=$CONSTRAINT} --array=1-"$(wc -l < "$part")" \
        --cpus-per-task="$POOL_CPUS" --mem="$FLEET_MEM" --time="$FLEET_TIME" \
        --export=ALL,TASKFILE=$part,RESULTS_ROOT=$ROOT,REQUIRE_CACHE=1,REQUIRE_FLAGS="$ROOT/GATES_A_OK $ROOT/GATES_B_OK $ROOT/CANARY_OK" \
        cluster/slurm/lstm_intraday_pack.sbatch)
  IDS+=("$J")
done
DEP=$(IFS=:; echo "${IDS[*]}")
M=$(sbatch --parsable -J lstmi_merge --dependency=afterany:"$DEP" \
      --export=ALL,RESULTS_ROOT=$ROOT,FLEET=$FLEET cluster/slurm/lstm_intraday_merge.sbatch)
echo "stage2 arrays=${IDS[*]} ($N chunks) merge=$M $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.14i %.16j %.4t %.10M %.5C %.9P %R" | grep -E "JOBID|lstmi" | head -12
