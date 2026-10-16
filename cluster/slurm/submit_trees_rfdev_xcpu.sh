#!/bin/bash
# rfdev cross-CPU test on CARC (2026-10-07): the six stored rfdev cells (rf_opt_00, cluster/rfdev_rf_config.json; T1 / T10 x
# baseline / all_features / lean_vol_vix) re-run on ANOTHER CPU class with the stored run's code (this deployment root,
# /scratch1/jc_905/harxhar-rfdev-trees), chunking (cluster/trees_rfdev_tasks_untuned.txt), pack script
# (cluster/slurm/trees_rfdev_pack.sbatch: single-threaded fits, seed 42, WINDOW_MASK=1), into
# results/trees_rfdev_xcpu/<CLASS>/{t1,t10} -- to be compared byte for byte with the stored epyc-7513 chunks
# (experiments/trees_rfdev_xcpu_compare.py).  Decides whether CLASS can join the forest fleets beside epyc-7513.
#   CLASS=epyc-7542 bash cluster/slurm/submit_trees_rfdev_xcpu.sh
#     pack 0 (chunk c0 of the six arms; writes CANARY_OK) -> the compare job on pack 0 (afterany) -> packs 1..K held afterok on
#     pack 0 (each refuses to run without CANARY_OK) -> merge + full compare (afterany)
# Submit from a shell with no conda env active.  A stamp per class refuses a second submission.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results
case "${CONDA_PREFIX:-}" in */envs/*) echo "conda env ${CONDA_PREFIX} is active in this shell; submit from a clean one"; exit 1 ;; esac
CLASS=${CLASS:?CLASS (e.g. epyc-7542, xeon-4116)}
R=results/trees_rfdev_xcpu/$CLASS
UT=cluster/trees_rfdev_tasks_untuned.txt
CFG=cluster/rfdev_rf_config.json
PARTS=${PARTS:-main}
for f in "$UT" "$CFG" cluster/slurm/trees_rfdev_pack.sbatch cluster/slurm/trees_rfdev_xcpu_merge.sbatch experiments/trees_rfdev_xcpu_compare.py \
         specs/causal_tune_trees_rfdev.py; do
  [ -f "$f" ] || { echo "$f missing"; exit 1; }
done
S=logs/submitted_trees_rfdev_xcpu_$CLASS.txt
[ -f "$S" ] && { echo "already submitted ($(cat "$S"))"; exit 1; }
mkdir -p "$R"
NP=$(awk 'NF {print $1}' "$UT" | sort -n | tail -1)
C=$(sbatch --parsable -J "xc_${CLASS}_p0" --constraint="$CLASS" --partition="$PARTS" --cpus-per-task=20 --mem=80G --time=2:00:00 \
     --export=ALL,TASKFILE=$UT,PACK=0,RESULTS_ROOT=$R,WRITE_FLAG=$R/CANARY_OK,HPC_KW_RF_CONFIG=$CFG cluster/slurm/trees_rfdev_pack.sbatch)
K=$(sbatch --parsable -J "xc_${CLASS}_cmp0" --dependency=afterany:"$C" --export=ALL,RESULTS_ROOT=$R,PART=compare cluster/slurm/trees_rfdev_xcpu_merge.sbatch)
F=$(sbatch --parsable -J "xc_${CLASS}" --dependency=afterok:"$C" --kill-on-invalid-dep=yes --constraint="$CLASS" --partition="$PARTS" --array=1-"$NP" \
     --cpus-per-task=20 --mem=80G --time=2:00:00 --export=ALL,TASKFILE=$UT,RESULTS_ROOT=$R,CANARY_FLAG=$R/CANARY_OK,HPC_KW_RF_CONFIG=$CFG \
     cluster/slurm/trees_rfdev_pack.sbatch)
M=$(sbatch --parsable -J "xc_${CLASS}_merge" --dependency=afterany:"$F" --export=ALL,RESULTS_ROOT=$R,PART=all cluster/slurm/trees_rfdev_xcpu_merge.sbatch)
echo "$(date +%F_%T) $CLASS: pack0=$C compare0=$K fleet=$F (1-$NP) merge=$M" | tee -a logs/submitted_trees_rfdev_xcpu.txt > "$S"
cat "$S"
squeue -u "$USER" -o "%.16i %.22j %.3t %.10M %.5C %R" | grep -E "JOBID|xc_" | head -10
