#!/bin/bash
# DIRECT rest-of-day variance arms (specs/causal_tune_rest_of_day.py): one
# regression per entry clock whose target is that day's realized variance from
# the clock to the 16:00 close.  13 entry clocks (0930 .. 1530) x buckets
# all_features, baseline, live_feasible x estimators ridge, reclasso,
# reclasticnet at tw 2000 = 117 arms, one single-threaded python process each.
# Canary first (live_feasible x ridge,reclasso,reclasticnet x 2000 x 1530: its
# three arms run serially on 1 cpu, then the identity check -- the 15:30
# direct arm IS the per-bar bar1600 arm, checked against the shipped reference
# table); it writes $ROOT/CANARY_OK only if all three arms AND the identity
# check succeed.  The fleet (114 arms, one per array task, 1 cpu) is held on it
# with afterok and refuses to run unless CANARY_OK exists.  The scorer is held
# on the fleet with afterany (a failed arm does not block scoring the rest; it
# lists NOT DONE).  Every task first checks cluster/restofday_manifest.md5
# (the code + data the local gates ran with) and refuses to run on any
# difference; so does this script, before submitting anything.
#
#   cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_restofday.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results

# Set from the local smoke (2026-09-24, laptop, one core, whole series = one
# chunk, ~1,470 out-of-sample sessions per arm):
#   live_feasible ridge 15:30   23 s wall incl. data prep (identity arm)
#   live_feasible x3 est 12:00  12-17 s each backtest (data prep cached)
#   all_features reclasso 09:30 128 s wall (data prep ~95 s + backtest ~35 s;
#                               the heaviest arm: widest bucket, lasso homotopy)
#   all_features ridge 09:30    58 s wall, peak working set 5.8 GiB
#   identity check              < 5 s
# CARC single core assumed up to ~2x slower -> <= ~5 min per arm.  Limits
# carry a >= 4x margin.  MEM 16G = the per-bar arms' request, 2.7x the
# measured all_features peak.
CANARY_TIME=${CANARY_TIME:-1:00:00}
FLEET_TIME=${FLEET_TIME:-0:30:00}
MEM=${MEM:-16G}
CONC=${CONC:-60}

CANARY=cluster/restofday_tasks_canary.txt
FLT=cluster/restofday_tasks_fleet.txt
MANIFEST=cluster/restofday_manifest.md5
ROOT=results/linear_subsection_restofday
FLAG=$ROOT/CANARY_OK
for f in specs/causal_tune_rest_of_day.py specs/causal_tune_linear.py \
         experiments/score_rest_of_day.py experiments/check_restofday_identity.py \
         experiments/ft_remaining.py \
         cluster/slurm/restofday_pack.sbatch cluster/slurm/restofday_score.sbatch \
         "$CANARY" "$FLT" "$MANIFEST"; do
  [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_restofday_carc.sh, locally)"; exit 1; }
done
for est in ridge lasso enet; do
  for b in all_features baseline live_feasible; do
    f=$ROOT/ref/yhat_sub_${est}_${b}.parquet
    [ -f "$f" ] || { echo "$f missing: run the ship script first (bash cluster/slurm/ship_restofday_carc.sh, locally)"; exit 1; }
  done
done
if ! md5sum -c --quiet "$MANIFEST"; then
  echo "manifest check failed (the files above differ from $MANIFEST): nothing submitted"
  exit 1
fi
echo "manifest ok: $(wc -l < "$MANIFEST") pinned files"
mkdir -p "$ROOT"
rm -f "$FLAG"
SUBMIT=sbatch
CAN=$($SUBMIT --parsable -J rod_canary --cpus-per-task=1 --mem="$MEM" --time="$CANARY_TIME" \
      --export=ALL,TASKFILE=$CANARY,RESULTS_ROOT=$ROOT,WRITE_FLAG=$FLAG cluster/slurm/restofday_pack.sbatch)
FLEET=$($SUBMIT --parsable -J rod_fleet --array=1-"$(wc -l < "$FLT")"%"$CONC" --dependency=afterok:"$CAN" \
      --cpus-per-task=1 --mem="$MEM" --time="$FLEET_TIME" \
      --export=ALL,TASKFILE=$FLT,RESULTS_ROOT=$ROOT,CANARY_FLAG=$FLAG cluster/slurm/restofday_pack.sbatch)
S=$($SUBMIT --parsable -J rod_score --dependency=afterany:"$FLEET" cluster/slurm/restofday_score.sbatch)
echo "canary=$CAN fleet=$FLEET score=$S" | tee logs/submitted_restofday_carc.txt
squeue -u "$USER" -o "%.10i %.12j %.4t %.10M %.6D %R" | head -10
