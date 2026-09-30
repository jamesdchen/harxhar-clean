#!/bin/bash
# Gates g0-g4 of the 16:00 tree cadence campaign (checklist I2) on the cluster the
# fleets run on: experiments/trees_cadence_gates.py (the shipped specs run end to end
# on a live_feasible 16:00 smoke slice, LightGBM / XGBoost / RF).  GATES_OK is written
# under $OUT only if every gate passes.  Exported: OUT (default
# results/linear_subsection_trees_dedup/gates), PROCS, POOL.
#SBATCH --job-name=tc_gates
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments:$PWD/pylib_trees"
OUT="${OUT:-results/linear_subsection_trees_dedup/gates}"
mkdir -p "$OUT"
rm -f "$OUT/GATES_OK"
echo "gates on $(hostname), ${SLURM_CPUS_PER_TASK:-1} cpus  $(date)"
python -u experiments/trees_cadence_gates.py --procs "${PROCS:-16}" --pool "${POOL:-4}" --out "$OUT" \
  > "$OUT/gates.log" 2>&1 || echo "GATES FAILED: $(tail -3 "$OUT/gates.log")"
tail -25 "$OUT/gates.log"
touch "$OUT/FINISHED"
