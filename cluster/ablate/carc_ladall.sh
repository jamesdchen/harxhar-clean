#!/bin/bash
# 2026-10-28 addendum K, the CARC side (a job payload; no sbatch, copy or move in here), in the trees' environment of
# the fomcfix / beyond_trees CARC runs (PY = /home1/jc_905/trees_final/venv_skx/bin/python; threads = --cpus-per-task;
# PYTHONPATH = the code tree and its experiments/).  env: MODE, PY, CODE (code tree), FULL (the patched full cache),
# D (staging dir).
#   MODE=gate   TAG, WANT: the rung gate on this node's CPU model (WANT: a regex it must match, else exit 99): X md5 of
#               FULL = 725eb201... (gate F0), ablate_trees.py walk on lrung_<est>_0 / lrung_<est>_46 at the first 10
#               origins for LightGBM and XGBoost, ablate_ladall.py gate -> $D/gate/<TAG>/passed_<est>; exit 3 on a failure
#   MODE=piece  TASKS: line SLURM_ARRAY_TASK_ID ("<est> <arm> <i>/<n> <hours>") -> ablate_trees.py walk ->
#               $D/runs/trees_ladall/pieces (checkpoints every 300 s; resumable by resubmitting the index); exit 98 if
#               this node's CPU model has not passed the gate for the class
#   MODE=merge  TASKS [, GATES]: ablate_ladall.py merge -> $D/runs/trees_ladall/trees_wide.parquet (GATES default: every
#               gate dir under $D/gate; a dir without passed_<est> is ignored by T2)
set -eo pipefail
: "${MODE:?}" "${PY:?}" "${CODE:?}" "${FULL:?}" "${D:?}"
cd "$CODE"
if [ -f "$(dirname "$PY")/activate" ]; then source "$(dirname "$PY")/activate"; fi
export PATH="$(dirname "$PY"):$PATH"
set -u
NT="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
export PYTHONPATH="$PWD:$PWD/experiments"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "carc_ladall $MODE job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} on $(hostname) [$CPU], $NT threads, $("$PY" -V), $(date)"
case "$MODE" in
  gate)
    : "${TAG:?}" "${WANT:?}"
    echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
    m=$(md5sum "$FULL/X.npy" | cut -d' ' -f1)
    [ "$m" = 725eb201044783d3cdc62501307dcc3a ] || { echo "X md5 $m != 725eb201...: exit 2"; exit 2; }
    O=$D/gate/$TAG
    mkdir -p "$O/logs"
    for e in lgbm xgb; do
      for k in 0 46; do
        rc=0
        "$PY" experiments/ablate_trees.py walk --arm lrung_${e}_$k --est $e --first 10 --full "$FULL" --out "$O" \
          --threads "$NT" > "$O/logs/${e}_$k.log" 2>&1 || rc=$?
        echo "$e k=$k walk rc $rc $(date)"
      done
    done
    "$PY" experiments/ablate_ladall.py gate --out "$O" --est lgbm,xgb
    ;;
  piece)
    : "${TASKS:?}"
    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "$TASKS")
    read -r EST ARM PART HOURS <<< "$LINE"
    [ -n "${PART:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID}: '$LINE'"; exit 1; }
    grep -qxF "$CPU" "$D"/gate/*/passed_"$EST" 2> /dev/null || { echo "CPU '$CPU' has no passed gate for $EST: exit 98"; exit 98; }
    echo "line ${SLURM_ARRAY_TASK_ID}: $LINE"
    "$PY" experiments/ablate_trees.py walk --arm "$ARM" --est "$EST" --part "$PART" --full "$FULL" \
      --out "$D/runs/trees_ladall" --threads "$NT"
    ;;
  merge)
    : "${TASKS:?}"
    # GATES: default every gate dir under $D/gate (sbatch --export splits a comma-separated value)
    GATES=${GATES:-$(ls -d "$D"/gate/*/ | sed 's:/$::' | paste -sd, -)}
    echo "gates: $GATES"
    "$PY" experiments/ablate_ladall.py merge --out "$D/runs/trees_ladall" --gates "$GATES" --tasks "$TASKS" \
      --to "$D/runs/trees_ladall/trees_wide.parquet"
    ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "carc_ladall $MODE done $(date)"
