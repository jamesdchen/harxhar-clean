#!/bin/bash
# I5 (2026-09-29): one SGE array task = experiments/optuna_h2_design_probe.py on one task line
# "<bucket> <model> <tw> <seg> <chunk> <start> <end> <halo>" (TASKFILE), with the spec's env axes
# as cluster/optuna_h2_task.sh passes them, under CARC's runtime (cluster/optuna_h2_runtime_setup.sh).
# Writes $XROOT/probe/<bucket>_<model>_c<chunk>.csv.  CARC twin: cluster/slurm/optuna_h2_probe_carc.sbatch.
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
PY=$D/carc_env/harxhar/bin/python3.11
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONSTARTUP PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="$D/carc_env/harxhar/bin:/usr/bin:/bin" PYTHONUSERBASE="$D/carc_usersite"
export PYTHONPATH="$D:$D/pylib_trees" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${JOB_ID:-0}_${SGE_TASK_ID:-0}" TQDM_DISABLE=1 PYTHONUNBUFFERED=1
set -u
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$(sed -n "${TASK}p" "$TASKFILE")"
echo "probe $TASK on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)]: $BUCKET $MODEL c$CHUNK [$START, $END) mask ${WINDOW_MASK:-0}"
SLURM_CPUS_PER_TASK=1 HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
  HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
  HPC_KW_STAGE=tune HPC_KW_N_TRIALS=50 HPC_KW_IDENTITY=0 HPC_KW_RESUME=0 HPC_KW_WINDOW_MASK="${WINDOW_MASK:-0}" \
  HPC_KW_STAGE1_TRIALS="" HPC_RESULT_DIR="${TMPDIR:-/tmp}/probe_$TASK" \
  "$PY" -u experiments/optuna_h2_design_probe.py --cluster hoffman2 --out "$XROOT/probe/${BUCKET}_${MODEL}_c${CHUNK}_hoffman2_$(hostname -s).csv" \
  --save-x "$XROOT/probe_x/${BUCKET}_c${CHUNK}_hoffman2_$(hostname -s).npz"
