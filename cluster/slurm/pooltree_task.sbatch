#!/bin/bash
# Pooled tree walks (user request 2026-10-07): one array task = one line of TASKFILE,
#   "<est> <set> <chunk>[,<chunk>...] [<rows>]"
# run one chunk after another through experiments/pooled_trees_walk.py (the bank's own compute() / _walk_tree, refit at
# every bar on the trailing 24,000 bars, seed 42; a random-forest branch for est rf), with SLURM_CPUS_PER_TASK threads
# (8 = the bank's: harxhar-clean/jobs/slurm/unification_tree.sbatch; OMP / MKL exported to match, as the bank did).
#   est lgbm -> cluster/pooltree_lgbm_config.json (lgbm_opt_00 = tree_expert_00), xgb -> cluster/xgbtop_xgb_config.json
#   (xgb_opt_00 = tree_expert_10), rf -> cluster/pooltree_rf400_config.json (the 400-trial dev-span forest, trial 255)
# REFIT_EVERY (exported, default 1 = the bank's walk): the refit cadence; K > 1 writes arm pooltree_<est>_<set>_r<K>.
# Output: $RESULTS_ROOT/chunks/pooltree_<est>_<set>[_r<K>]/chunk_<kkk>.npz + .meta.json + .node (host, CPU model); a chunk whose
# npz exists is skipped (resume).  <rows> (optional) limits every chunk to its first rows into $RESULTS_ROOT/<CANARY_DIR>
# (the canary: run twice, REPEAT=1, and the two npz compared bit for bit -> WRITE_FLAG).  CANARY_FLAG: files the task
# requires.  UNIFY_CACHE_DIR: the folder of the bank's prep cache (cache/).
#SBATCH --job-name=pooltree
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
NT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/experiments"
export UNIFY_CACHE_DIR="${UNIFY_CACHE_DIR:-cache}"
unset TREE_REFIT_EVERY
R="${RESULTS_ROOT:-results/pooled_trees_2026-10-07}"
for f in ${CANARY_FLAG:-}; do [ -f "$f" ] || { echo "flag $f missing; refusing to run"; exit 1; }; done
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "${TASKFILE:?TASKFILE}")
read -r EST SET CHUNKS ROWS <<< "$LINE"
[ -n "${CHUNKS:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID:-1}: '$LINE'"; exit 1; }
case "$EST" in
  lgbm) CFG=cluster/pooltree_lgbm_config.json ;;
  xgb) CFG=cluster/xgbtop_xgb_config.json ;;
  rf) CFG=cluster/pooltree_rf400_config.json ;;
  *) echo "unknown est '$EST'"; exit 1 ;;
esac
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "task ${SLURM_ARRAY_TASK_ID:-1} on $(hostname) [$CPU], $NT threads: $EST $SET chunks $CHUNKS rows ${ROWS:-all}  $(date)"
OUT="$R/chunks"
[ -n "${ROWS:-}" ] && OUT="$R/${CANARY_DIR:-canary}"
RE="${REFIT_EVERY:-1}"   # refit cadence (1 = the bank's walk); K > 1 -> arm pooltree_<est>_<set>_r<K>
SFX=""; [ "$RE" != 1 ] && SFX="_r$RE"
FAIL=0
for C in ${CHUNKS//,/ }; do
  NPZ=$(printf '%s/pooltree_%s_%s%s/chunk_%03d.npz' "$OUT" "$EST" "$SET" "$SFX" "$C")
  [ -f "$NPZ" ] && { echo "  skip (exists) $NPZ"; continue; }
  mkdir -p "$(dirname "$NPZ")"
  a=$(date +%s)
  if python -u experiments/pooled_trees_walk.py --est "$EST" --set "$SET" --config "$CFG" --chunk-index "$C" \
       ${ROWS:+--rows "$ROWS"} --refit-every "$RE" --out "$OUT" > "${NPZ%.npz}.log" 2>&1; then
    printf 'host %s\ncpu %s\nthreads %s\nseconds %s\njob %s_%s\n' "$(hostname)" "$CPU" "$NT" "$(( $(date +%s) - a ))" \
      "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" "${SLURM_ARRAY_TASK_ID:-}" > "${NPZ%.npz}.node"
    echo "  done $EST $SET c$C in $(( $(date +%s) - a ))s  $(date +%T)"
  else
    FAIL=1; echo "  FAILED $EST $SET c$C: $(tail -1 "${NPZ%.npz}.log" | cut -c1-200)"
  fi
  if [ "$FAIL" = 0 ] && [ "${REPEAT:-0}" = 1 ]; then
    REP="$R/${CANARY_DIR:-canary}_repeat"
    mkdir -p "$REP"
    python -u experiments/pooled_trees_walk.py --est "$EST" --set "$SET" --config "$CFG" --chunk-index "$C" \
      ${ROWS:+--rows "$ROWS"} --refit-every "$RE" --out "$REP" > "$REP/$EST.$SET.$C.log" 2>&1 || FAIL=1
    python - "$NPZ" "$(printf '%s/pooltree_%s_%s%s/chunk_%03d.npz' "$REP" "$EST" "$SET" "$SFX" "$C")" <<'EOF' || FAIL=1
import sys
import numpy as np
a, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
same = {k: bool(np.array_equal(a[k], b[k])) for k in ("row_id", "t", "y_fit", "yhat", "rv_raw", "baseline")}
print("repeat vs first run (bit for bit):", same)
sys.exit(0 if all(same.values()) else 1)
EOF
  fi
done
if [ "$FAIL" = 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  WF="${WRITE_FLAG//%e/$EST}"; WF="${WF//%s/$SET}"
  mkdir -p "$(dirname "$WF")"; touch "$WF"; echo "wrote $WF"
fi
exit $FAIL
