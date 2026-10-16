#!/bin/bash
# Pooled pre-tune study on CARC (2026-10-06; the Hoffman2 twin is cluster/submit_pool_pretune.sh).
#   MODE=task   array; line SLURM_ARRAY_TASK_ID of TASKFILE -> cluster/slurm/pool_pretune_run1.sh (one CPU)
#   MODE=check  STEP=gate (cluster/slurm/pool_pretune_check_carc.py)
#SBATCH --job-name=plp
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
echo "$MODE on $(hostname) [$(scontrol show node "$(hostname -s)" 2>/dev/null | grep -o 'AvailableFeatures=[^ ]*')]  $(date)"
if [ -n "${REQUIRE_FLAG:-}" ] && [ ! -f "$REQUIRE_FLAG" ]; then
  echo "required flag $REQUIRE_FLAG missing; refusing to run"
  exit 1
fi
case "$MODE" in
  task)
    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")
    echo "$LINE"
    bash cluster/slurm/pool_pretune_run1.sh "$LINE"
    ;;
  check)
    python -u cluster/slurm/pool_pretune_check_carc.py "$STEP"
    ;;
  *) echo "MODE must be task or check"; exit 2 ;;
esac
