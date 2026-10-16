#!/bin/bash
# Pooled tree walks on Hoffman2 AVX-only nodes (2026-10-07; coordinator: use every byte-identical CPU class for the
# forest / XGBoost): experiments/pooled_trees_walk.py with CARC's own Python stack (agent D's runtime, read only; glibc
# 2.28 and no shared object from outside the runtime, checked in process), ONE slot (XGBoost gave the same bytes at 1 / 4
# / 8 threads against bank arm 10; a forest's trees do not depend on n_jobs and its predict is one-threaded).  Every task
# refuses a CPU outside ALLOW_CPU and records its CPU model beside the chunk.
#   MODE gate   experiments/pooled_trees_gate.py on GATE_LINES (bank arm 10 / XGBoost rows at 1 thread, the bank chunks
#               copied under ./ref) -> $OUT (GATE_OK only when every line matched byte for byte)
#   MODE chunk  line $SGE_TASK_ID of TASKFILE ("<est> <set> <chunk>[,<chunk>...]"), REFIT_EVERY (default 1), into
#               results/pooled_trees_2026-10-07/chunks/pooltree_<est>_<set>[_r<K>]/chunk_<kkk>.npz (+ .meta.json, .node);
#               an existing npz is skipped.  The chunks are copied to CARC by cluster/transfer_pooltrees_h2_to_carc.sh.
#   MODE pieces line $SGE_TASK_ID of TASKFILE ("<est> <set> k:a:b[,...]", experiments/pooled_trees_pieces.py) through
#               pooled_trees_walk.py --pieces into $RESULTS_ROOT/chunks/pooltree_<est>_<set>/pieces/ (+ .node a piece); pieces
#               listed in $RESULTS_ROOT/revoked_<task file name>.txt are dropped (moved to CARC unstarted); a .claim marks a started one
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
RT=${RUNTIME_ROOT:-/u/scratch/j/jamesdc1/harxhar-optuna}
PY=$RT/carc_env/harxhar/bin/python3.11
[ -x "$PY" ] || { echo "no runtime at $PY"; exit 1; }
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV TREE_REFIT_EVERY
export PATH="$RT/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$RT/carc_usersite"
export PYTHONPATH="$D:$D/experiments" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SLURM_CPUS_PER_TASK=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 UNIFY_CACHE_DIR="$D/cache"
set -u
MODE="${MODE:?MODE must be gate or chunk}"
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "task $TASK mode $MODE on $(hostname) [$CPU] $(date)"
ALLOW_CPU=${ALLOW_CPU:-"E5-2670 0 @|E5-2650 v2 @"}
printf '%s\n' "$CPU" | grep -qE "$ALLOW_CPU" || { echo "CPU '$CPU' not allowed ($ALLOW_CPU); refusing"; exit 1; }
"$PY" - "$RT" <<'EOF' || { echo "runtime check FAILED; refusing to run"; exit 1; }
import sys, ctypes
import numpy, sklearn, lightgbm, xgboost
libc = ctypes.CDLL(None); libc.gnu_get_libc_version.restype = ctypes.c_char_p
g = libc.gnu_get_libc_version().decode()
maps = {l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith("/") and ".so" in l}
out = sorted(p for p in maps if not p.startswith(sys.argv[1] + "/"))
print("runtime: glibc", g, "|", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, sklearn, lightgbm, xgboost)), f"| outside: {out or 'none'}")
sys.exit(0 if g == "2.28" and not out else 1)
EOF
case "$MODE" in
  gate)
    "$PY" -u experiments/pooled_trees_gate.py --lines "${GATE_LINES:?GATE_LINES}" --out "${OUT:?OUT}"
    ;;
  chunk)
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r EST SET CHUNKS <<< "$LINE"
    [ -n "${CHUNKS:-}" ] || { echo "malformed line $TASK: '$LINE'"; exit 1; }
    case "$EST" in
      xgb) CFG=cluster/xgbtop_xgb_config.json ;;
      rf) CFG=cluster/pooltree_rf400_config.json ;;
      *) echo "est '$EST' is not run on Hoffman2 (LightGBM stays on the bank's class)"; exit 1 ;;
    esac
    RE="${REFIT_EVERY:-1}"; SFX=""; [ "$RE" != 1 ] && SFX="_r$RE"
    OUT=results/pooled_trees_2026-10-07/chunks
    for C in ${CHUNKS//,/ }; do
      NPZ=$(printf '%s/pooltree_%s_%s%s/chunk_%03d.npz' "$OUT" "$EST" "$SET" "$SFX" "$C")
      [ -f "$NPZ" ] && { echo "  skip (exists) $NPZ"; continue; }
      mkdir -p "$(dirname "$NPZ")"
      a=$(date +%s)
      "$PY" -u experiments/pooled_trees_walk.py --est "$EST" --set "$SET" --config "$CFG" --chunk-index "$C" \
        --refit-every "$RE" --out "$OUT" > "${NPZ%.npz}.log" 2>&1
      printf 'host %s\ncpu %s\nthreads 1\nseconds %s\njob %s.%s\n' "$(hostname -s)" "$CPU" "$(( $(date +%s) - a ))" "${JOB_ID:-0}" "$TASK" \
        > "${NPZ%.npz}.node"
      echo "  done $EST $SET c$C in $(( $(date +%s) - a ))s $(date +%T)"
    done
    ;;
  pieces)  # one line of TASKFILE, "<est> <set> k:a:b[,k:a:b...]" (experiments/pooled_trees_pieces.py), one process
    LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
    read -r EST SET PIECES <<< "$LINE"
    [ -n "${PIECES:-}" ] || { echo "malformed line $TASK: '$LINE'"; exit 1; }
    case "$EST" in
      xgb) CFG=cluster/xgbtop_xgb_config.json ;;
      rf) CFG=cluster/pooltree_rf400_config.json ;;
      *) echo "est '$EST' is not run on Hoffman2 (LightGBM stays on the bank's class)"; exit 1 ;;
    esac
    R="${RESULTS_ROOT:-results/pooled_trees_2026-10-07}"
    OUT="$R/chunks"
    ARM="pooltree_${EST}_$SET"
    mkdir -p "$OUT/$ARM/pieces"
    LOGF="$OUT/$ARM/pieces/task_${JOB_ID:-0}_${TASK}.log"
    # "<est> <set> k:a:b" lines: pieces of THIS task file moved elsewhere before they started.  One list per task file
    # (a moved piece is listed in its old file's list, never in its new one); arrays submitted before 2026-10-08 03:30
    # read $R/revoked_pieces.txt instead (their scripts were copied at submission).
    REV="$R/revoked_$(basename "$TASKFILE" .txt).txt"
    KEEP=""
    for P in ${PIECES//,/ }; do
      if [ -f "$REV" ] && grep -qxF "$EST $SET $P" "$REV"; then echo "revoked (moved): $P"; continue; fi
      IFS=: read -r K A B <<< "$P"
      printf '%s %s %s\n' "$(hostname)" "${JOB_ID:-0}.$TASK" "$(date +%s)" > "$(printf '%s/%s/pieces/chunk_%03d_%04d-%04d.claim' "$OUT" "$ARM" "$K" "$A" "$B")"
      KEEP="${KEEP:+$KEEP,}$P"
    done
    [ -n "$KEEP" ] || { echo "every piece of this task was revoked; nothing to do"; exit 0; }
    PIECES=$KEEP
    "$PY" -u experiments/pooled_trees_walk.py --est "$EST" --set "$SET" --config "$CFG" --pieces "$PIECES" --out "$OUT" > "$LOGF" 2>&1       || { echo "FAILED: $(tail -1 "$LOGF" | cut -c1-200)"; exit 1; }
    for P in ${PIECES//,/ }; do
      IFS=: read -r K A B <<< "$P"
      NPZ=$(printf '%s/%s/pieces/chunk_%03d_%04d-%04d.npz' "$OUT" "$ARM" "$K" "$A" "$B")
      [ -f "$NPZ" ] || { echo "missing $NPZ"; exit 1; }
      [ -f "${NPZ%.npz}.node" ] || printf 'host %s\ncpu %s\nthreads 1\njob %s.%s\n' "$(hostname -s)" "$CPU" "${JOB_ID:-0}" "$TASK" > "${NPZ%.npz}.node"
    done
    grep -E "DONE|skip" "$LOGF" | tail -3 || true
    ;;
  *) echo "MODE must be gate, chunk or pieces"; exit 1 ;;
esac
