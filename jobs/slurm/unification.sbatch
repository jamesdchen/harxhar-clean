#!/bin/bash
#SBATCH --job-name=unif
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-99
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_%x_%A_%a.out
# Unification campaign task: one (arm, chunk). Submit with:
#   sbatch --export=ALL,ARM=<arm_name> --job-name=unif_<arm_name> jobs/slurm/unification.sbatch
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# Idempotent resume: a chunk that already produced its npz is done.
OUT=$(printf 'results/unification/%s/chunk_%03d.npz' "$ARM" "$SLURM_ARRAY_TASK_ID")
[ -f "$OUT" ] && { echo "SKIP existing $OUT"; exit 0; }
$PY experiments/run_unification.py --arm "$ARM" --chunk-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir results/unification
