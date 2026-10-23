#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch1/jc_905/trees_final/logs/%x.%j.out
# One python call in the environment of cluster/slurm/pooltree_rerun13.sbatch (env harxhar-r13, threads =
# --cpus-per-task, PYTHONPATH = the submit directory and its experiments/).  Submit from the repo clone:
#   sbatch -J <name> -c <cpus> --mem=<mem> -t <time> [--constraint=epyc-7513] tf_py.sbatch <script> <args...>
set -eo pipefail
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
NT="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export PYTHONPATH="$PWD:$PWD/experiments"
echo "tf_py on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')], $NT CPUs, $("$PY" -V), $(date): $*"
"$PY" "$@"
echo "tf_py done $(date)"
