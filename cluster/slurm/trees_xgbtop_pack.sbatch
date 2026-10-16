#!/bin/bash
# xgbtop (user decision 2026-10-07): cluster/slurm/trees_rfdev_pack.sbatch for the per-bar XGBoost with the top dev-span
# entry -- every chunk process runs the GENERATED spec specs/causal_tune_trees_$TOKEN.py (specs/causal_tune_trees.py with
# the XGB_PARAMS block replaced by $XGB_CONFIG's params, experiments/trees_xgbtop_make_specs.py; TOKEN xgbtop =
# cluster/xgbtop_xgb_config.json = xgb_opt_00) instead of the spec itself.  Changes from the rfdev pack:
#   * before any process: the generated spec must equal what the generator writes from $XGB_CONFIG (--check)
#   * XGBoost lines only (refuses another model)
#   * MASK (exported, default 1): WINDOW_MASK of every process (1 = the masked campaign's rungs; 0 = the unmasked
#     twin, the runs before the mask decision); a finished chunk's npz must record window_mask = MASK, carry kept_n
#     iff MASK = 1, and record exactly the configuration's per-bar parameters (meta params == expected_params) before
#     DONE is written
# Everything else is the rfdev / masked pack's: TASKFILE holds one
#   "<pack> <rung> <bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>"
# line per chunk PROCESS; this task runs every line of pack PACK (default: the array index) AT ONCE, one
# single-threaded python process per line (SLURM_CPUS_PER_TASK=1 per process: XGBoost n_jobs 1; no fork pool, so no
# XGBoost fit ever runs inside a forked worker), staggered STAGGER seconds:
#   rung t10: REFIT_EVERY 10, IMPORTANCE_EVERY 1   (the shipped cadence)
#   rung t1 : REFIT_EVERY 1,  IMPORTANCE_EVERY 10  (a refit every session)
# TreeSHAP at 16:00 at every recorded refit.  Output of a line: $RESULTS_ROOT/<rung>/<bucket>/<seg>/<model>/tw<tw>/
# chunks/c<chunk> (experiments/trees_cadence_reduce_chunks.py merges an arm's chunks).  DONE = skip.  CANARY_FLAG (a
# file the fleet requires), WRITE_FLAG (touched when every line succeeded and, with ALONE_CHECK
# "<rung>:<bucket>:<model>:<chunk>", that chunk re-run ALONE into <its dir>_alone matched bit for bit).
# CPU CLASS: the submit script pins every task to one class (--constraint, default epyc-7513) and every finished chunk
# records its node and CPU features in NODE beside DONE.
#SBATCH --job-name=trees_xgbtop
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
TOKEN="${TOKEN:-xgbtop}"
if [ "$TOKEN" = xgbtop ]; then XGB_CONFIG="${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}"; else XGB_CONFIG="${XGB_CONFIG:?XGB_CONFIG must name the configuration of token $TOKEN}"; fi
SPEC="specs/causal_tune_trees_$TOKEN.py"
MASK="${MASK:-1}"
case "$MASK" in 0|1) ;; *) echo "MASK must be 0 or 1, got '$MASK'"; exit 1 ;; esac
echo "xgbtop pack: $SPEC from $XGB_CONFIG, WINDOW_MASK=$MASK: $(head -c 300 "$XGB_CONFIG" | tr -d "\n")"

# one thread per fit: the task's processes are the parallelism
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"

if [ -n "${CANARY_FLAG:-}" ] && [ ! -f "$CANARY_FLAG" ]; then
  echo "canary did not finish ($CANARY_FLAG missing); refusing to run"
  exit 1
fi
if ! python -c "import shap, lightgbm, xgboost, sklearn; print('shap', shap.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)"; then
  echo "tree stack import failed (PYTHONPATH=$PYTHONPATH); refusing to run"
  exit 1
fi
if ! python experiments/trees_xgbtop_make_specs.py --config "$XGB_CONFIG" --token "$TOKEN" --check; then
  echo "$SPEC is not the file generated from $XGB_CONFIG; refusing to run"
  exit 1
fi

ROOT="${RESULTS_ROOT:?RESULTS_ROOT}"
PACK="${PACK:-${SLURM_ARRAY_TASK_ID:-0}}"
STAGGER="${STAGGER:-10}"
NCPU="${SLURM_CPUS_PER_TASK:-1}"
mapfile -t LINES < <(awk -v p="$PACK" 'NF && $1 == p' "$TASKFILE")
if [ "${#LINES[@]}" -eq 0 ] || [ "${#LINES[@]}" -gt "$NCPU" ]; then
  echo "pack $PACK of $TASKFILE has ${#LINES[@]} lines for $NCPU cores; refusing to run"
  exit 1
fi
if awk -v p="$PACK" 'NF && $1 == p && $4 != "xgb" {bad = 1} END {exit !bad}' "$TASKFILE"; then
  echo "pack $PACK of $TASKFILE has a non-xgb line; refusing to run"
  exit 1
fi
echo "pack $PACK on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')]: ${#LINES[@]} processes on $NCPU cores, stagger ${STAGGER}s  $(date)"

chunk_dir() {  # rung bucket seg model tw chunk -> the chunk's output dir
  echo "$ROOT/$1/$2/$3/$4/tw$5/chunks/c$6"
}

node_tag() {  # $1 = a finished chunk dir: the node it ran on and that node's CPU features (the class)
  { echo "host $(hostname)"; scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2>/dev/null \
      | grep -o "AvailableFeatures=[^ ]*"; echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"; } > "$1/NODE"
}

run_line() {  # "<line>" [out-dir override]
  local PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO RE IE OUT
  read -r PK RUNG BUCKET MODEL TW SEG CHUNK START END HALO <<< "$1"
  case "$RUNG" in
    t10) RE=10; IE=1 ;;
    t1) RE=1; IE=10 ;;
    *) echo "  unknown rung '$RUNG'"; return 1 ;;
  esac
  OUT="${2:-$(chunk_dir "$RUNG" "$BUCKET" "$SEG" "$MODEL" "$TW" "$CHUNK")}"
  if [ -f "$OUT/DONE" ]; then
    echo "  skip (DONE) $RUNG $BUCKET $MODEL c$CHUNK"
    return 0
  fi
  mkdir -p "$OUT"
  local T0=$SECONDS
  if SLURM_CPUS_PER_TASK=1 \
     HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
     HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_HAR_BASE="" \
     HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
     HPC_KW_REFIT_EVERY="$RE" HPC_KW_IMPORTANCE_EVERY="$IE" HPC_KW_SHAP_SEGMENTS="$SEG" \
     HPC_KW_WINDOW_MASK="$MASK" \
     HPC_RESULT_DIR="$OUT" \
     python -u "$SPEC" > "$OUT/run.log" 2>&1 \
     && python -c "import json, sys, numpy as np; sys.path.insert(0, 'experiments'); import trees_xgbtop_make_specs as mk; z = np.load(sys.argv[1]); m = json.loads(str(z['meta'])); mask = int(sys.argv[3]); sys.exit(0 if m.get('model') == 'xgb' and m.get('window_mask') == mask and ('kept_n' in z.files) == (mask == 1) and m.get('params') == json.loads(json.dumps(mk.expected_params(sys.argv[2], int(sys.argv[4])))) else 1)" \
          "$OUT/causal_tune_trees/$MODEL/$BUCKET/trees_$SEG.npz" "$XGB_CONFIG" "$MASK" "$TW"; then
    node_tag "$OUT"
    touch "$OUT/DONE"
    echo "  done $RUNG $BUCKET $MODEL c$CHUNK [$START, $END) in $((SECONDS - T0))s  $(date +%T)"
  else
    echo "  FAILED $RUNG $BUCKET $MODEL c$CHUNK: $(tail -1 "$OUT/run.log" | cut -c1-200)"
    return 1
  fi
}

PIDS=()
n=0
for L in "${LINES[@]}"; do
  run_line "$L" &
  PIDS+=("$!")
  n=$((n + 1))
  if [ "$n" -lt "${#LINES[@]}" ]; then sleep "$STAGGER"; fi
done
FAIL=0
for p in "${PIDS[@]}"; do
  wait "$p" || FAIL=1
done
echo "pack $PACK: processes finished, fail=$FAIL  $(date)"

if [ "$FAIL" -eq 0 ] && [ -n "${ALONE_CHECK:-}" ]; then
  IFS=: read -r AR AB AM AC <<< "$ALONE_CHECK"
  L=$(awk -v p="$PACK" -v r="$AR" -v b="$AB" -v m="$AM" -v c="$AC" '$1 == p && $2 == r && $3 == b && $4 == m && $7 == c' "$TASKFILE" | head -1)
  [ -n "$L" ] || { echo "ALONE_CHECK '$ALONE_CHECK' is not a line of pack $PACK"; exit 1; }
  read -r _ _ _ _ TW SEG _ _ _ _ <<< "$L"
  A=$(chunk_dir "$AR" "$AB" "$SEG" "$AM" "$TW" "$AC")
  B="${A}_alone"
  rm -rf "$B"
  echo "alone check: $ALONE_CHECK again, the only process of the task"
  run_line "$L" "$B" || FAIL=1
  if [ "$FAIL" -eq 0 ]; then
    python - "$A/causal_tune_trees/$AM/$AB/trees_$SEG.npz" "$B/causal_tune_trees/$AM/$AB/trees_$SEG.npz" <<'EOF' || FAIL=1
import sys
import numpy as np
a, b = (np.load(p, allow_pickle=False) for p in sys.argv[1:3])
keys = ("date", "pred_adj", "true_adj", "true_raw", "baseline", "e2_adj", "refit_row", "importance", "shap", "kept_n", "kept")
same = {k: bool(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")) for k in keys if k in a.files or k in b.files}
print("alone vs in-pack (bit for bit):", same)
sys.exit(0 if all(same.values()) else 1)
EOF
  fi
fi
if [ "$FAIL" -eq 0 ] && [ -n "${WRITE_FLAG:-}" ]; then
  mkdir -p "$(dirname "$WRITE_FLAG")"
  touch "$WRITE_FLAG"
  echo "wrote $WRITE_FLAG"
fi
exit $FAIL
