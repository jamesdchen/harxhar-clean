#!/bin/bash
#SBATCH --job-name=unif_tree
#SBATCH --account=pollok_1603
#SBATCH --partition=main
# 24h, not 6h. Per-bar tree refits do ~2,760 fits per chunk; the 6h that
# the linear arms inherit is not obviously enough, and a task that runs
# out of wall is a total loss (the menu study died exactly this way).
# `main` allows 2-00:00:00 and the `normal` QOS sets no MaxWall, so the
# 6h cap was self-imposed and bought nothing.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-99
#SBATCH --output=/scratch1/jc_905/harxhar-clean/logs/unif_%x_%A_%a.out
# Tree expert bank task: one (arm, chunk). cpus-per-task=8 per author QOS
# ruling (MaxJobsPU=100 binds before the cpu cap; LightGBM saturates ~8-16
# threads on this design) — the arm reads SLURM_CPUS_PER_TASK for LightGBM
# num_threads, and OMP/MKL are exported to match. Submit with:
#   sbatch --export=ALL,ARM=<arm_name> --job-name=unif_<arm_name> jobs/slurm/unification_tree.sbatch
cd /scratch1/jc_905/harxhar-clean
export PYTHONPATH="$PWD:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
export TQDM_DISABLE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=/home1/jc_905/.conda/envs/harxhar/bin/python
# Idempotent resume: a chunk that already produced its npz is done.
OUT=$(printf 'results/unification/%s/chunk_%03d.npz' "$ARM" "$SLURM_ARRAY_TASK_ID")
[ -f "$OUT" ] && { echo "SKIP existing $OUT"; exit 0; }
$PY experiments/run_unification.py --arm "$ARM" --chunk-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir results/unification
