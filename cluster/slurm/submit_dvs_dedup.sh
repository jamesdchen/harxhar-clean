#!/bin/bash
# Submit the C2 dense-vs-sparse re-run's cluster part (experiments/dvs_dedup_1530.py stage
# carc; cluster/slurm/dvs_dedup.sbatch).  Run on CARC from the project root, after
# cluster/slurm/ship_dvs_dedup_carc.sh:
#   bash cluster/slurm/submit_dvs_dedup.sh            the canary (block 5 of live_feasible, the
#                                                     three models, 3 CPUs), then the fleet held
#                                                     on it with afterok (every other block chunk,
#                                                     a pool of FLEET_CPUS single-threaded workers)
#   MODE=gate bash cluster/slurm/submit_dvs_dedup.sh  one 3-CPU job: this walk WITHOUT the mask on
#                                                     block 5 of live_feasible (three models) vs the
#                                                     stored unmasked T10 run (same platform)
# PARTITION (default main) may be main,oneweek or debug when main is full.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
W=results/dense_vs_sparse_dedup/_work_carc
for b in live_feasible all_features; do
  [ -f "$W/design_$b.npz" ] || { echo "missing $W/design_$b.npz (run the ship script)"; exit 1; }
done
PART=${PARTITION:-main}
FLEET_CPUS=${FLEET_CPUS:-20}
if [ "${MODE:-}" = gate ]; then
  G=$(sbatch --parsable --partition="$PART" --job-name=dvs_dedup_gate --cpus-per-task=3 \
    --mem=8G --time=0:30:00 --export=ALL,DVS_MODE=gate cluster/slurm/dvs_dedup.sbatch)
  echo "$(date '+%F %T') gate $G (partition $PART)" | tee -a "$W/submitted.txt"
  exit 0
fi
C=$(sbatch --parsable --partition="$PART" --job-name=dvs_dedup_canary --cpus-per-task=3 \
  --mem=8G --time=1:00:00 --export=ALL,DVS_MODE=canary cluster/slurm/dvs_dedup.sbatch)
F=$(sbatch --parsable --partition="$PART" --job-name=dvs_dedup_fleet --dependency=afterok:"$C" \
  --cpus-per-task="$FLEET_CPUS" --mem=32G --time=2:00:00 --export=ALL,DVS_MODE=fleet \
  cluster/slurm/dvs_dedup.sbatch)
echo "$(date '+%F %T') canary $C fleet $F (partition $PART, fleet cpus $FLEET_CPUS)" | tee -a "$W/submitted.txt"
