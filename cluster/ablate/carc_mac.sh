#!/bin/bash
# 2026-10-29 addendum L, the CARC side (a job payload; no sbatch, copy or move in here), as carc_ladall.sh, in the trees'
# environment of the fomcfix / beyond_trees / ladall CARC runs (PY = /home1/jc_905/trees_final/venv_skx/bin/python;
# threads = --cpus-per-task; PYTHONPATH = the code tree and its experiments/).  env: MODE, PY, CODE (code tree), FULL (the
# macro cache, bucket all_features_mac), D (staging dir), MACX (its X md5, from gate GM0; default: the X.npy line of
# $FULL/MD5SUMS, the manifest carc_build_mac.sh wrote -- so a gate submitted afterok on the build job needs no value).
#   MODE=gate   TAG, WANT: gate GMT on this node's CPU model (WANT: a regex it must match, else exit 99): X md5 of FULL =
#               MACX, then for LightGBM and XGBoost ablate_trees.py walk on the macro cache: full and har at the first and
#               last 10 origins, full_mac at the first 10; ablate_trees.py gatemac (the pieces read the macro cache;
#               full / har text-identical to fomcfix forecasts_gb / the templates, true_adj and B equal; full_mac finite)
#               -> $D/gate/<TAG>/passed_<est>; exit 3 on a failure
#   MODE=piece  TASKS: line SLURM_ARRAY_TASK_ID ("<est> <arm> <i>/<n> <hours>") -> ablate_trees.py walk ->
#               $D/runs/trees_mac/pieces (checkpoints every 300 s; resumable by resubmitting the index); exit 98 if this
#               node's CPU model has not passed the gate for the class
#   MODE=merge  TASKS [, GATES]: ablate_trees.py mergemac (T2 over the gate dirs) -> $D/runs/trees_mac/trees_wide.parquet
#               (GATES default: every gate dir under $D/gate; a dir without passed_<est> is ignored by T2)
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
echo "carc_mac $MODE job ${SLURM_JOB_ID:-} task ${SLURM_ARRAY_TASK_ID:-} on $(hostname) [$CPU], $NT threads, $("$PY" -V), $(date)"
case "$MODE" in
  gate)
    : "${TAG:?}" "${WANT:?}"
    MACX=${MACX:-$(awk '$2 == "X.npy" {print $1}' "$FULL/MD5SUMS")}
    [ -n "$MACX" ] || { echo "no MACX and no X.npy line in $FULL/MD5SUMS: exit 2"; exit 2; }
    echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
    m=$(md5sum "$FULL/X.npy" | cut -d' ' -f1)
    [ "$m" = "$MACX" ] || { echo "X md5 $m != MACX $MACX: exit 2"; exit 2; }
    O=$D/gate/$TAG
    mkdir -p "$O/logs"
    for e in lgbm xgb; do
      for at in full:ends har:ends full_mac:first; do
        a=${at%%:*}
        sel=--${at##*:}
        rc=0
        "$PY" experiments/ablate_trees.py walk --arm "$a" --est $e $sel 10 --full "$FULL" --out "$O" \
          --threads "$NT" > "$O/logs/${e}_$a.log" 2>&1 || rc=$?
        echo "$e $a $sel 10 walk rc $rc $(date)"
      done
    done
    "$PY" experiments/ablate_trees.py gatemac --out "$O" --est lgbm,xgb \
      --template-gb "$CODE/results/fomcfix_2026-10-26/forecasts/forecasts_gb.parquet"
    ;;
  piece)
    : "${TASKS:?}"
    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" "$TASKS")
    read -r EST ARM PART HOURS <<< "$LINE"
    [ -n "${PART:-}" ] || { echo "malformed line ${SLURM_ARRAY_TASK_ID}: '$LINE'"; exit 1; }
    grep -qxF "$CPU" "$D"/gate/*/passed_"$EST" 2> /dev/null || { echo "CPU '$CPU' has no passed gate for $EST: exit 98"; exit 98; }
    echo "line ${SLURM_ARRAY_TASK_ID}: $LINE"
    "$PY" experiments/ablate_trees.py walk --arm "$ARM" --est "$EST" --part "$PART" --full "$FULL" \
      --out "$D/runs/trees_mac" --threads "$NT"
    ;;
  merge)
    : "${TASKS:?}"
    # GATES: default every gate dir under $D/gate (sbatch --export splits a comma-separated value)
    GATES=${GATES:-$(ls -d "$D"/gate/*/ | sed 's:/$::' | paste -sd, -)}
    echo "gates: $GATES"
    "$PY" experiments/ablate_trees.py mergemac --out "$D/runs/trees_mac" --gates "$GATES" --tasks "$TASKS" \
      --to "$D/runs/trees_mac/trees_wide.parquet"
    ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "carc_mac $MODE done $(date)"
