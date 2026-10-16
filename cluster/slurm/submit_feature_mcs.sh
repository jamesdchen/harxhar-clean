#!/bin/bash
# Feature Model Confidence Set (2026-10-08) on CARC: canary -> gate -> fleet, every job on epyc-7513 at
# nice 0 (ahead of this account's pooled-forest pieces at nice 100-120).  From the deployment root
# (cluster/slurm/ship_feature_mcs_carc.sh), in a shell with NO conda env active (sbatch --export=ALL would
# hand it to the jobs):
#
#   cd /scratch1/jc_905/harxhar-feature-mcs && bash cluster/slurm/submit_feature_mcs.sh
#
#   canary  4 cpus: the LD_DEBUG binding check, then the two anchors (baseline, all_features) and one
#           leave-one-out / one add-one design (cost probes), cluster/feature_mcs_tasks_canary.txt
#   gate    afterok the canary: the anchors vs the stored arm-f forecasts -> GATE_OK
#   fleet   afterok the gate (killed if the gate fails), each task refusing to run without GATE_OK: the other
#           92 designs, cluster/feature_mcs_tasks_fleet.txt (one line = one array task, 12 processes)
# A stamp (logs/submitted_feature_mcs.txt) refuses resubmission.
#
#   bash cluster/slurm/submit_feature_mcs.sh ridge_lasso
#           (2026-10-08) the pre-tuned ridge and lasso on the same 96 designs in ONE job (16 cpus; the cap is
#           full): MODE=pipeline of cluster/slurm/feature_mcs.sbatch -- binding check, the canary lines of
#           cluster/feature_mcs_tasks2_canary.txt, the gate of each estimator, then the lines of
#           cluster/feature_mcs_tasks2_fleet.txt whose estimator passed.  Stamp logs/submitted_feature_mcs_rl.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active; submit from a clean shell"; exit 1 ;; esac
[ -z "${CONDA_DEFAULT_ENV:-}" ] || { echo "conda env ${CONDA_DEFAULT_ENV} is active; submit from a clean shell"; exit 1; }
R=results/feature_mcs_2026-10-08
FLAG=$R/_work/GATE_OK
for f in experiments/feature_mcs_runner.py cluster/slurm/feature_mcs.sbatch cluster/feature_mcs_tasks_canary.txt \
         cluster/feature_mcs_tasks_fleet.txt $R/_work/inputs/models.csv $R/_work/kernel/close_exogpen_kernel_nofma.so; do
  [ -f "$f" ] || { echo "$f missing (ship first)"; exit 1; }
done
mkdir -p logs
if [ "${1:-}" = buckets ]; then
  # 2026-10-12: the 16 bucket designs (addb_* / loob_*) for ridge, lasso and elastic net in ONE job: MODE=pipeline
  # with an empty canary file (the three estimators' gates passed on 2026-10-08; the pipeline re-runs the ridge /
  # lasso gates on the stored anchors and each fleet line still needs its estimator's flag)
  # (2026-10-13: run from a fresh root as well -- the canary re-runs the six anchors and the job gates all three)
  for f in cluster/feature_mcs_tasks3_canary.txt cluster/feature_mcs_tasks3_fleet.txt; do
    [ -f "$f" ] || { echo "$f missing"; exit 1; }
  done
  STAMP=logs/submitted_feature_mcs_buckets.txt
  [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
  J=$(sbatch --parsable --constraint=epyc-7513 --nice=0 -J fmcs_bk --cpus-per-task=16 --mem=12G --time=0:45:00         --export=ALL,MODE=pipeline,BIND_TAG=_bk,GATE_ESTS=reclasticnet:ridge:reclasso,TASKFILE_CANARY=cluster/feature_mcs_tasks3_canary.txt,TASKFILE_FLEET=cluster/feature_mcs_tasks3_fleet.txt         cluster/slurm/feature_mcs.sbatch)
  echo "buckets pipeline=$J  $(date)" | tee "$STAMP"
  squeue -u "$USER" -o "%.14i %.12j %.4t %.10M %.6y %R" | grep fmcs || true
  exit 0
fi
if [ "${1:-}" = ridge_lasso ]; then
  for f in cluster/feature_mcs_tasks2_canary.txt cluster/feature_mcs_tasks2_fleet.txt $R/_work/inputs/ref_armf.npz; do
    [ -f "$f" ] || { echo "$f missing (ship first)"; exit 1; }
  done
  STAMP=logs/submitted_feature_mcs_rl.txt
  [ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
  J=$(sbatch --parsable --constraint=epyc-7513 --nice=0 -J fmcs_rl --cpus-per-task=16 --mem=12G --time=1:00:00         --export=ALL,MODE=pipeline,TASKFILE_CANARY=cluster/feature_mcs_tasks2_canary.txt,TASKFILE_FLEET=cluster/feature_mcs_tasks2_fleet.txt         cluster/slurm/feature_mcs.sbatch)
  echo "ridge+lasso pipeline=$J  $(date)" | tee "$STAMP"
  squeue -u "$USER" -o "%.14i %.12j %.4t %.10M %.6y %R" | grep fmcs || true
  exit 0
fi
STAMP=logs/submitted_feature_mcs.txt
[ -f "$STAMP" ] && { echo "already submitted ($(cat "$STAMP")); delete $STAMP by hand to resubmit"; exit 1; }
rm -f "$FLAG"
S="sbatch --parsable --constraint=epyc-7513 --nice=0"
C=cluster/feature_mcs_tasks_canary.txt
F=cluster/feature_mcs_tasks_fleet.txt
CAN=$($S -J fmcs_canary --cpus-per-task=4 --mem=4G --time=0:45:00 \
      --export=ALL,MODE=canary,TASKFILE=$C cluster/slurm/feature_mcs.sbatch)
G=$($S -J fmcs_gate --cpus-per-task=1 --mem=2G --time=0:10:00 --dependency=afterok:"$CAN" --kill-on-invalid-dep=yes \
      --export=ALL,MODE=gate cluster/slurm/feature_mcs.sbatch)
FL=$($S -J fmcs_fleet --array=1-"$(wc -l < $F)" --cpus-per-task=12 --mem=8G --time=0:45:00 \
      --dependency=afterok:"$G" --kill-on-invalid-dep=yes \
      --export=ALL,MODE=fleet,TASKFILE=$F,REQUIRE_FLAG=$FLAG cluster/slurm/feature_mcs.sbatch)
echo "canary=$CAN gate=$G fleet=$FL  $(date)" | tee "$STAMP"
squeue -u "$USER" -o "%.14i %.12j %.4t %.10M %.6y %R" | grep fmcs || true
