#!/bin/bash
# Pooled twins on the moments_vix bucket on CARC (2026-10-06; the Hoffman2 twin is
# cluster/submit_pool_momvix.sh): canary -> gate -> fleet -> collect, every job on epyc-7513 (the
# class the stored pooled live_feasible arms of the CARC root ran on).  From the deployment root:
#
#   cd /scratch1/jc_905/harxhar-pool-momvix && bash cluster/slurm/submit_pool_momvix.sh
#
# canary   one slice (solves 199,250 .. 204,316, a tune boundary of the whole-series run) of the three
#          stored pooled live_feasible arms (ridge / lasso / elastic net), one CPU each
# gate     afterany the canary: cluster/slurm/pool_momvix_check_carc.py gate -> GATE_OK (exit 1 without)
# fleet    afterok the gate (killed if the gate fails) and refusing to run without GATE_OK: the three
#          moments_vix arms, each as 20 tune-aligned chunks of 10,250 solves on 20 CPUs, merged
# collect  afterany the fleet
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/pool_momvix
STAMP=logs/submitted_pool_momvix_carc.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
FLAG=results/pool_momvix/GATE_OK
rm -f "$FLAG"
S="sbatch --parsable --constraint=epyc-7513"
CAN=$($S -J pmv_canary --array=1-3 --export=ALL,MODE=gate,TASKFILE=cluster/pool_momvix_tasks_canary_carc.txt \
      cluster/slurm/pool_momvix_carc.sbatch)
G=$($S -J pmv_gate --dependency=afterany:"$CAN" --export=ALL,MODE=check,STEP=gate \
      cluster/slurm/pool_momvix_carc.sbatch)
F=$($S -J pmv_fleet --array=1-3 --cpus-per-task=20 --mem=120G --time=3:00:00 \
      --dependency=afterok:"$G" --kill-on-invalid-dep=yes \
      --export=ALL,MODE=fleet,TASKFILE=cluster/pool_momvix_tasks_fleet_carc.txt,REQUIRE_FLAG=$FLAG \
      cluster/slurm/pool_momvix_carc.sbatch)
K=$($S -J pmv_collect --dependency=afterany:"$F" --kill-on-invalid-dep=yes --export=ALL,MODE=check,STEP=collect \
      cluster/slurm/pool_momvix_carc.sbatch)
echo "canary=$CAN gate=$G fleet=$F collect=$K  $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.12i %.12j %.4t %.10M %.5C %R" | grep pmv || true
