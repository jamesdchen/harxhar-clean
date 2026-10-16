#!/bin/bash
# Pooled moments_vix campaign on CARC (2026-10-06): gate slices, chunked fleet arms, checks.
#   MODE=gate   array; TASKFILE line "<bucket> <est> <tw> <start> <end> <halo>": one slice of a stored
#               pooled arm into results/pool_momvix/gate/<bucket>/none/<est>/tw<tw> (one CPU)
#   MODE=fleet  array; TASKFILE line "<bucket> <est> <tw> <n_chunks> <chunk_solves>": the whole-series
#               arm as n tune-aligned chunks (chunk k = solves [k L, (k + 1) L), the last to the end,
#               each with a W-row halo, W = tw x 48), run in parallel one single-threaded process per
#               CPU, then merged (cluster/slurm/pool_momvix_check_carc.py merge); REQUIRE_FLAG must exist
#   MODE=check  STEP=gate | collect [CBUCKET=<fleet bucket>] (cluster/slurm/pool_momvix_check_carc.py)
#SBATCH --job-name=pmv
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
echo "$MODE on $(hostname) [$(scontrol show node "$(hostname -s)" 2>/dev/null | grep -o 'AvailableFeatures=[^ ]*')] cpus ${SLURM_CPUS_PER_TASK:-1}  $(date)"
python -c "import numpy; from threadpoolctl import threadpool_info as t; print('numpy', numpy.__version__, [(i.get('internal_api'), i.get('version'), i.get('architecture')) for i in t()])" || true

if [ -n "${REQUIRE_FLAG:-}" ] && [ ! -f "$REQUIRE_FLAG" ]; then
  echo "required flag $REQUIRE_FLAG missing; refusing to run"
  exit 1
fi

case "$MODE" in
  gate)
    read -r BUCKET EST TW START END HALO <<< "$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")"
    bash cluster/slurm/pool_momvix_run1.sh "results/pool_momvix/gate/$BUCKET/none/$EST/tw$TW" \
      "$BUCKET" "$EST" "$TW" "$START" "$END" "$HALO"
    ;;
  fleet)
    read -r BUCKET EST TW NCH L <<< "$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")"
    W=$(( TW * 48 ))
    BASE="results/pool_momvix/fleet_chunks/$BUCKET/none/$EST/tw$TW"
    LIST=$(for k in $(seq 0 $(( NCH - 1 ))); do
      S=$(( W + k * L )); E=$(( S + L )); [ "$k" -eq $(( NCH - 1 )) ] && E=-1
      printf '%s %s %s %s %s %s %s\n' "$BASE/c$(printf %02d "$k")" "$BUCKET" "$EST" "$TW" "$S" "$E" "$W"
    done)
    echo "$LIST"
    printf '%s\n' "$LIST" | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -L 1 bash cluster/slurm/pool_momvix_run1.sh
    python -u cluster/slurm/pool_momvix_check_carc.py merge "$BUCKET" "$EST"
    ;;
  check)
    python -u cluster/slurm/pool_momvix_check_carc.py "$STEP" ${CBUCKET:-}  # CBUCKET: a fleet bucket other than moments_vix
    ;;
  *) echo "MODE must be gate, fleet or check"; exit 2 ;;
esac
