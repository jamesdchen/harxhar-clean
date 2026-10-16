#!/bin/bash
# xgbtop (2026-10-07): IDENTITY gate of the re-centred XGBoost Optuna rung on CARC (one array task per line of
# GATE_TASKFILE = cluster/<TOKEN>_optuna_tasks_gate.txt, one per bucket; submitted by cluster/slurm/
# submit_trees_xgbtop_optuna.sh gate, pinned to the run's CPU class) -- cluster/slurm/optuna_rfp_gate.sbatch for XGBoost.
# PARAMETERIZED by TOKEN (default xgbtop) and XGB_CONFIG (default, for xgbtop, cluster/xgbtop_xgb_config.json).
# experiments/trees_xgbtop_optuna_gate.py runs, on the line's rows [START - HALO, END - HALO):
#   fixed     specs/causal_tune_trees_<TOKEN>.py, REFIT_EVERY 1, IMPORTANCE_EVERY 10, WINDOW_MASK 1, one thread (the
#             fixed-settings XGBoost) -- or FIXED_NPZ (a stored chunk npz of that run covering the rows) when exported
#   stage 1   specs/causal_tune_trees_optuna_xgbtop.py IDENTITY=1 (one trial = xgb_opt_00), a spawned pool of
#             SLURM_CPUS_PER_TASK processes
#   stage 2   the same, from those records
# and requires every path's forecasts / targets / importance / kept columns / TreeSHAP to equal the fixed run's bit
# for bit (tp1's results CSV byte for byte), the records to show the patches reached the spawned workers (trial 0 =
# the shipped vector, rounds = 367, the re-centred space, worker pids != the wrapper's) and a SMOKE_TRIALS-trial TPE
# smoke on SMOKE_ROWS rows (trial 0 = the shipped vector, every parameter in the re-centred space, patience / cap of each
# trial's learning rate).  Only then is $RESULTS_ROOT/GATE_OK_<bucket> written -- the flag the bucket's stage-1 fleet
# tasks require.  Output: $RESULTS_ROOT/gate/<bucket>/.
# Exported: RESULTS_ROOT, GATE_TASKFILE, TOKEN, XGB_CONFIG, [FIXED_NPZ (%b = the line's bucket: the phase-1 T1 chunk c0 of
# results/linear_subsection_trees_mask_xgbtop)], [SMOKE_TRIALS (default 3)], [SMOKE_ROWS (default 2)].
#SBATCH --job-name=xgbtop_gate
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
R="${RESULTS_ROOT:?RESULTS_ROOT}"
GT="${GATE_TASKFILE:?GATE_TASKFILE}"
I="${SLURM_ARRAY_TASK_ID:-1}"
LINE=$(sed -n "${I}p" "$GT")
read -r BUCKET MODEL TW SEG CHUNK START END HALO <<< "$LINE"
[ -n "${HALO:-}" ] || { echo "malformed line $I of $GT: '$LINE'"; exit 1; }
[ "$MODEL" = xgb ] && [ "$TW" = 2000 ] && [ "$SEG" = bar1600 ] && [ "$HALO" = 2000 ] || { echo "gate line must be xgb / 2000 / bar1600 / halo 2000: '$LINE'"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-0}_${I}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
TOKEN="${TOKEN:-xgbtop}"
if [ "$TOKEN" = xgbtop ]; then XGB_CONFIG="${XGB_CONFIG:-cluster/xgbtop_xgb_config.json}"; else XGB_CONFIG="${XGB_CONFIG:?XGB_CONFIG must name the configuration of token $TOKEN}"; fi
echo "$TOKEN gate ($XGB_CONFIG) $BUCKET on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')]: $LINE $(date)"
python experiments/trees_xgbtop_make_specs.py --config "$XGB_CONFIG" --token "$TOKEN" --check \
  || { echo "specs/causal_tune_trees_$TOKEN.py is not the file generated from $XGB_CONFIG"; exit 1; }
FIXED_NPZ="${FIXED_NPZ:-}"
FIXED_NPZ="${FIXED_NPZ//%b/$BUCKET}"  # %b = the line's bucket (one fixed chunk per bucket)
[ -z "$FIXED_NPZ" ] || [ -f "$FIXED_NPZ" ] || { echo "FIXED_NPZ $FIXED_NPZ missing"; exit 1; }
GR="$R/gate/$BUCKET"
rm -rf "$GR"
mkdir -p "$GR"
python -u experiments/trees_xgbtop_optuna_gate.py --bucket "$BUCKET" --row0 $((START - HALO)) --rows $((END - START)) \
  --pool "${SLURM_CPUS_PER_TASK:-20}" --out "$GR" ${FIXED_NPZ:+--fixed-npz "$FIXED_NPZ"} --config "$XGB_CONFIG" --token "$TOKEN" \
  --smoke-trials "${SMOKE_TRIALS:-3}" --smoke-rows "${SMOKE_ROWS:-2}" --flag "$R/GATE_OK_$BUCKET"
