#!/bin/bash
# 2026-10-03 close studies: Hoffman2 (SGE) array task of the trees pre-tune study
# (experiments/close_trees_pretune.py).  One task = one line of $TASKFILE:
#
#   STAGE=pretune   line "<model> <seed>"                 one Optuna TPE study (sampler seed
#                   20261003 + <seed>), CTP_TRIALS trials, the folds of a trial in a pool of
#                   $NSLOTS spawned processes (single-threaded fits, combined in fold order:
#                   the pool size changes no number).  Its Optuna JournalFileStorage lives in
#                   $CTP_WORK/<tag>/<model>/pretune_s<seed>.journal: a task killed at h_rt (or a
#                   resubmission) resumes from it (the sampler is reseeded with seed + 7919 x
#                   completed trials, recorded in pretune_s<seed>_resumes.json).
#   STAGE=walk      line "<model> <seeds> <refit_every>"  merges the seeds' studies, runs the light
#                   retunes and the walk-forward refits of every arm (pool of $NSLOTS).
#
# Exported by cluster/submit_close_pretune_h2.sh: TASKFILE, STAGE, CTP_TAG, CTP_WORK, CTP_TRIALS,
# CTP_FOLDS, CTP_FOLD_LEN, CTP_TIMEOUT, CTP_RETUNE_TRIALS, CTP_CHECKPOINTS (and CTP_WALK_ROWS for a
# smoke run).  Runtime: CARC's Python stack relayed into $RT by cluster/optuna_h2_runtime_setup.sh
# when present (LightGBM 4.6.0 / XGBoost 3.2.0, the stored campaign's builds), else the cluster's
# conda env hpc-pi (older LightGBM / XGBoost: its numbers then stand on their own, against the
# run's own shipped-config control).  PYBIN=<python> overrides both (local smoke test).
# Scratch is purged: every input is checked (and md5-checked against close_pretune_manifest.md5)
# before anything runs.
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
RT=${RT:-/u/scratch/j/jamesdc1/harxhar-optuna}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
if [ -n "${PYBIN:-}" ]; then
  PY=$PYBIN; RUNTIME="PYBIN ($PYBIN)"
elif [ -x "$RT/carc_env/harxhar/bin/python3.11" ]; then
  PY=$RT/carc_env/harxhar/bin/python3.11
  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV
  export PATH="$RT/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$RT/carc_usersite"
  export PYTHONPATH="$D:$RT/pylib_trees"
  RUNTIME="CARC stack at $RT"
else
  set +u
  # shellcheck disable=SC1091
  source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate hpc-pi
  PY=$(command -v python)
  export PYTHONPATH="$D"
  RUNTIME="conda hpc-pi ($PY)"
fi
set -u
NT="${NSLOTS:-1}"
STAGE="${STAGE:?STAGE must be pretune or walk}"
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
read -r MODEL SEEDS REFIT <<< "$LINE"
[ -n "${MODEL:-}" ] && [ -n "${SEEDS:-}" ] || { echo "malformed line $TASK of $TASKFILE: '$LINE'"; exit 1; }
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
case ",$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' ','),"  in *,avx512f,*) CLASS=AVX-512 ;; *,avx2,*) CLASS=AVX2 ;; *) CLASS=AVX ;; esac
echo "task $TASK on $(hostname) [$CPU | $CLASS] stage $STAGE: $MODEL seeds $SEEDS refit ${REFIT:--} pool $NT; runtime $RUNTIME; $(date)"

# inputs (scratch is purged): the shipped files with their md5, the two design caches
[ -f close_pretune_manifest.md5 ] || { echo "close_pretune_manifest.md5 missing: re-ship (cluster/close_pretune_ship_h2.sh)"; exit 1; }
md5sum -c --quiet close_pretune_manifest.md5 || { echo "shipped files missing or changed: re-ship"; exit 1; }
"$PY" - <<'EOF' || { echo "runtime check FAILED"; exit 1; }
import numpy, pandas, optuna, lightgbm, xgboost, sklearn
print("runtime:", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, pandas, optuna, lightgbm, xgboost, sklearn)))
EOF

W=${CTP_WORK:?CTP_WORK}
OUTD="$W/${CTP_TAG:?CTP_TAG}/$MODEL"
mkdir -p "$OUTD" logs
case "$STAGE" in
  pretune)
    FLAG="$OUTD/PRETUNE_DONE_s$SEEDS"
    [ -f "$FLAG" ] && { echo "already done ($FLAG)"; exit 0; }
    a=$(date +%s)
    CTP_MODEL=$MODEL CTP_SEED=$SEEDS CTP_WORKERS=$NT "$PY" -u experiments/close_trees_pretune.py pretune \
      >> "$OUTD/pretune_s$SEEDS.log" 2>&1
    N=$("$PY" -c "import numpy as np; print(len(np.load('$OUTD/pretune_s$SEEDS.npz')['val_mse']))")
    echo "host=$(hostname -s) class=$CLASS cpu=$CPU seconds=$(( $(date +%s) - a )) pool=$NT trials=$N" >> "$OUTD/pretune_s$SEEDS.host"
    if [ "$N" -ge "${CTP_TRIALS:?}" ]; then touch "$FLAG"; echo "pretune $MODEL s$SEEDS: $N trials, DONE"
    else echo "pretune $MODEL s$SEEDS: $N of $CTP_TRIALS trials (time guard); resubmit to continue"; fi
    ;;
  walk)
    REFIT=${REFIT:?walk lines need <model> <seeds> <refit_every>}
    FLAG="$OUTD/WALK_DONE_r$REFIT"
    [ -f "$FLAG" ] && { echo "already done ($FLAG)"; exit 0; }
    for s in ${SEEDS//,/ }; do
      [ -f "$OUTD/PRETUNE_DONE_s$s" ] || { echo "pre-tune study s$s of $MODEL not complete; refusing"; exit 1; }
    done
    a=$(date +%s)
    CTP_MODEL=$MODEL CTP_SEEDS=$SEEDS CTP_SEED=${SEEDS%%,*} CTP_REFIT_EVERY=$REFIT CTP_WORKERS=$NT \
      "$PY" -u experiments/close_trees_pretune.py walk >> "$OUTD/walk_r$REFIT.log" 2>&1
    echo "host=$(hostname -s) class=$CLASS cpu=$CPU seconds=$(( $(date +%s) - a )) pool=$NT" >> "$OUTD/walk_r$REFIT.host"
    touch "$FLAG"
    echo "walk $MODEL r$REFIT DONE"
    ;;
  *) echo "STAGE must be pretune or walk"; exit 1 ;;
esac
