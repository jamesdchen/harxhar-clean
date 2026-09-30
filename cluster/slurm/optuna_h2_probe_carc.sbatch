#!/bin/bash
# I5 (2026-09-29): CARC twin of cluster/optuna_h2_probe.sh -- one array task = the design probe
# (experiments/optuna_h2_design_probe.py: design / target hashes and the per-window kept-column set)
# on one line of TASKFILE, with the env axes C's optuna_pack.sbatch passes to the spec, in C's
# CARC environment (harxhar env, ./pylib_trees).  Writes $XROOT/probe/<bucket>_<model>_c<chunk>_carc.csv.
#SBATCH --job-name=op_h2_probe
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$TASKFILE")"
echo "probe ${SLURM_ARRAY_TASK_ID:-1} on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)]: $BUCKET $MODEL c$CHUNK [$START, $END) mask ${WINDOW_MASK:-0}"
SLURM_CPUS_PER_TASK=1 HPC_KW_MODEL="$MODEL" HPC_KW_EXOG_BUCKET="$BUCKET" HPC_KW_SEGMENT="$SEG" \
  HPC_KW_TRAIN_WIN="$TW" HPC_KW_LAG_SCOPE=global HPC_KW_START="$START" HPC_KW_END="$END" HPC_KW_HALO="$HALO" \
  HPC_KW_STAGE=tune HPC_KW_N_TRIALS=50 HPC_KW_IDENTITY=0 HPC_KW_RESUME=0 HPC_KW_WINDOW_MASK="${WINDOW_MASK:-0}" \
  HPC_KW_STAGE1_TRIALS="" HPC_RESULT_DIR="${TMPDIR:-/tmp}/probe_${SLURM_ARRAY_TASK_ID:-1}" \
  python -u experiments/optuna_h2_design_probe.py --cluster carc --out "$XROOT/probe/${BUCKET}_${MODEL}_c${CHUNK}_carc_$(hostname -s).csv" \
  --save-x "$XROOT/probe_x/${BUCKET}_c${CHUNK}_carc_$(hostname -s).npz"
