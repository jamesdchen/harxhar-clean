#!/bin/bash
#SBATCH --job-name=var_sidecar
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_%x_%A_%a.out
# Exog-conditional second-moment sidecar: one (arm, chunk). Requires the
# target arm's chunk npzs in results/unification (present on CARC). Early
# chunks exit 0 with a burn-in SKIP. Submit with:
#   sbatch --export=ALL,ARM=<arm> --job-name=var_<arm> jobs/slurm/variance_sidecar.sbatch
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
OUT=$(printf 'results/variance_sidecar/%s/chunk_%03d.npz' "$ARM" "$SLURM_ARRAY_TASK_ID")
[ -f "$OUT" ] && { echo "SKIP existing $OUT"; exit 0; }
$PY experiments/variance_sidecar.py --arm "$ARM" --chunk-index "$SLURM_ARRAY_TASK_ID"
