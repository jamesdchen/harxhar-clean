#!/bin/bash
# Pooled linear models pre-tuned, backbone 1/100 (2026-10-06) on CARC, the backup of the Hoffman2 run
# (cluster/submit_pool_pretune.sh): canary -> gate -> fleet, every job on epyc-7513.  From the root:
#
#   cd /scratch1/jc_905/harxhar-pool-pretune && bash cluster/slurm/submit_pool_pretune.sh [1|f]
#
# stage 1  canary: arm a (r = 1, causal) on the stored-arm slice (solves 199,250 .. 204,316) of
#          all_features / baseline / lean_vol_vix x ridge / lasso / elastic net + the smoke tasks;
#          gate (afterany): cluster/slurm/pool_pretune_check_carc.py gate -> GATE_OK (functional checks;
#          ridge vs the stored Hoffman2 forecasts to 1e-6; lasso / elastic net differences recorded --
#          their warm homotopy branches differently across machines);
#          fleet (afterok the gate, refusing to run without GATE_OK): arm a and arm b over blocks
#          600 .. 817 (tune-aligned chunks) and the pre-tunes, one CPU per task
# stage regate  the gate again on the finished canary, then the fleet afterok it
# stage f  the arm-f walks (cluster/pool_pretune_tasks_f_carc.txt), refusing to run without GATE_OK
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=${1:-1}
mkdir -p logs results/pool_pretune
STAMP=logs/submitted_pool_pretune_carc_$STAGE.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_pretune/GATE_OK
S="sbatch --parsable --constraint=epyc-7513"
if [ "$STAGE" = 1 ]; then
  rm -f "$FLAG"
  C=cluster/pool_pretune_tasks_canary_carc.txt
  F=cluster/pool_pretune_tasks_fleet_carc.txt
  CAN=$($S -J plp_canary --array=1-"$(wc -l < $C)" --time=4:00:00 --export=ALL,MODE=task,TASKFILE=$C \
        cluster/slurm/pool_pretune_carc.sbatch)
  G=$($S -J plp_gate --time=1:00:00 --mem=8G --dependency=afterany:"$CAN" --export=ALL,MODE=check,STEP=gate \
        cluster/slurm/pool_pretune_carc.sbatch)
  FL=$($S -J plp_fleet --array=1-"$(wc -l < $F)"%40 --dependency=afterok:"$G" --kill-on-invalid-dep=yes \
        --export=ALL,MODE=task,TASKFILE=$F,REQUIRE_FLAG=$FLAG cluster/slurm/pool_pretune_carc.sbatch)
  echo "canary=$CAN gate=$G fleet=$FL  $(date)" | tee "$STAMP"
elif [ "$STAGE" = regate ]; then
  # the gate again on the finished canary (after a tolerance change), then the fleet afterok it
  F=cluster/pool_pretune_tasks_fleet_carc.txt
  G=$($S -J plp_gate --time=1:00:00 --mem=8G --export=ALL,MODE=check,STEP=gate cluster/slurm/pool_pretune_carc.sbatch)
  FL=$($S -J plp_fleet --array=1-"$(wc -l < $F)"%40 --dependency=afterok:"$G" --kill-on-invalid-dep=yes \
        --export=ALL,MODE=task,TASKFILE=$F,REQUIRE_FLAG=$FLAG cluster/slurm/pool_pretune_carc.sbatch)
  echo "gate=$G fleet=$FL  $(date)" | tee "$STAMP"
elif [ "$STAGE" = f ]; then
  [ -f "$FLAG" ] || { echo "$FLAG missing: the gate has not passed"; exit 1; }
  F=cluster/pool_pretune_tasks_f_carc.txt
  FJ=$($S -J plp_f --array=1-"$(wc -l < $F)"%40 --export=ALL,MODE=task,TASKFILE=$F,REQUIRE_FLAG=$FLAG \
        cluster/slurm/pool_pretune_carc.sbatch)
  echo "arm f=$FJ  $(date)" | tee "$STAMP"
else
  echo "stage must be 1 or f"; exit 2
fi
squeue -u "$USER" -o "%.14i %.12j %.4t %.10M %R" | grep plp | head -12 || true
