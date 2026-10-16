#!/bin/bash
# Pooled tree walks, row pieces (2026-10-07): one array task = one line of TASKFILE, "<est> <set> k:a:b[,k:a:b...]"
# (experiments/pooled_trees_pieces.py), run in ONE python process by experiments/pooled_trees_walk.py --pieces (the bank's
# compute() / _walk_tree, refit at every bar on the trailing 24,000 bars, seed 42) with SLURM_CPUS_PER_TASK threads:
# the forest (8-thread fit = 1-thread fit; predict one-threaded) and XGBoost (1 / 4 / 8 threads byte-identical vs bank
# arm 10) may use any count; LightGBM is refused (it keeps the bank's 8 threads in the chunk fleets).
# Output: $RESULTS_ROOT/chunks/pooltree_<est>_<set>/pieces/chunk_<kkk>_<aaaa>-<bbbb>.npz + .meta.json + .node (host, CPU
# model, threads); an existing piece is skipped; a piece listed in $RESULTS_ROOT/revoked_<task file name>.txt is dropped (moved
# to the other cluster before it started) and every piece run leaves a .claim first.  CANARY_FLAG: files the task requires.  Backfill-friendly: the submit
# script sets few CPUs, the measured memory and a tight --time.
#SBATCH --job-name=ptr_piece
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
NT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments" UNIFY_CACHE_DIR="${UNIFY_CACHE_DIR:-cache}"
unset TREE_REFIT_EVERY
R="${RESULTS_ROOT:-results/pooled_trees_2026-10-07}"
for f in ${CANARY_FLAG:-}; do [ -f "$f" ] || { echo "flag $f missing; refusing to run"; exit 1; }; done
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "${TASKFILE:?TASKFILE}")
read -r EST SET PIECES <<< "$LINE"
[ -n "${PIECES:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID:-1}: '$LINE'"; exit 1; }
case "$EST" in
  xgb) CFG=cluster/xgbtop_xgb_config.json ;;
  rf) CFG=cluster/pooltree_rf400_config.json ;;
  *) echo "est '$EST': LightGBM pieces are not run (the bank's 8 threads only, in the chunk fleets)"; exit 1 ;;
esac
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
OUT="$R/chunks"
ARM="pooltree_${EST}_$SET"
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname) [$CPU], $NT threads: $EST $SET pieces $PIECES  $(date)"
mkdir -p "$OUT/$ARM/pieces"
LOGF="$OUT/$ARM/pieces/task_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}.log"
# "<est> <set> k:a:b" lines: pieces of THIS task file moved elsewhere before they started.  One list per task file
# (a moved piece is listed in its old file's list, never in its new one); arrays submitted before 2026-10-08 03:30
# read $R/revoked_pieces.txt instead (their scripts were copied at submission).
REV="$R/revoked_$(basename "$TASKFILE" .txt).txt"
KEEP=""
for P in ${PIECES//,/ }; do
  if [ -f "$REV" ] && grep -qxF "$EST $SET $P" "$REV"; then echo "revoked (moved): $P"; continue; fi
  IFS=: read -r K A B <<< "$P"
  printf '%s %s %s\n' "$(hostname)" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}" "$(date +%s)" > "$(printf '%s/%s/pieces/chunk_%03d_%04d-%04d.claim' "$OUT" "$ARM" "$K" "$A" "$B")"
  KEEP="${KEEP:+$KEEP,}$P"
done
[ -n "$KEEP" ] || { echo "every piece of this task was revoked; nothing to do"; exit 0; }
PIECES=$KEEP
python -u experiments/pooled_trees_walk.py --est "$EST" --set "$SET" --config "$CFG" --pieces "$PIECES" --out "$OUT" > "$LOGF" 2>&1 \
  || { echo "FAILED: $(tail -1 "$LOGF" | cut -c1-200)"; exit 1; }
for P in ${PIECES//,/ }; do
  IFS=: read -r K A B <<< "$P"
  NPZ=$(printf '%s/%s/pieces/chunk_%03d_%04d-%04d.npz' "$OUT" "$ARM" "$K" "$A" "$B")
  [ -f "$NPZ" ] || { echo "missing $NPZ"; exit 1; }
  [ -f "${NPZ%.npz}.node" ] || printf 'host %s\ncpu %s\nthreads %s\njob %s_%s\n' "$(hostname)" "$CPU" "$NT" \
    "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" "${SLURM_ARRAY_TASK_ID:-}" > "${NPZ%.npz}.node"
done
grep -E "DONE|skip" "$LOGF" | tail -3 || true
echo "task done $(date)"
