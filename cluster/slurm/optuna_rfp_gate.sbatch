#!/bin/bash
# rfp163 (2026-10-07): IDENTITY gate of the Optuna rung with a dev-span shipped forest, on CARC (one array task per
# line of GATE_TASKFILE = cluster/<TOKEN>_optuna_tasks_gate.txt, one per bucket; submitted by
# cluster/slurm/submit_trees_rfp163_optuna.sh gate, pinned to the run's CPU class).  PARAMETERIZED by TOKEN (default
# rfp163) and RF_CONFIG (default, for rfp163, cluster/rfprev163_rf_config.json = the search's trial 163).
# experiments/trees_rfp163_optuna_gate.py runs, on the line's rows [START - HALO, END - HALO):
#   fixed     specs/causal_tune_trees_rfdev.py with HPC_KW_RF_CONFIG=$RF_CONFIG, REFIT_EVERY 1,
#             IMPORTANCE_EVERY 10, WINDOW_MASK 1, one thread (the fixed-settings forest) -- or FIXED_NPZ
#             (a stored chunk npz of that run covering the rows) when exported
#   stage 1   specs/causal_tune_trees_optuna_rfp.py IDENTITY=1 (one trial = the shipped forest), a spawned pool of
#             SLURM_CPUS_PER_TASK processes
#   stage 2   the same, from those records
# and requires every path's forecasts / targets / importance / kept columns / TreeSHAP to equal the fixed run's bit
# for bit (tp1's results CSV byte for byte), the records to show the patches reached the spawned workers (trial 0 =
# the shipped vector, rounds = the shipped n_estimators, worker pids != the wrapper's) and a SMOKE_TRIALS-trial TPE
# smoke on SMOKE_ROWS rows (trial 0 = the shipped vector, every max_depth in the re-centred ladder -- {2, 4, 7, 14, 28,
# None} for rfp163).  Only then is $RESULTS_ROOT/GATE_OK_<bucket>
# written -- the flag the bucket's stage-1 fleet tasks require.  Output: $RESULTS_ROOT/gate/<bucket>/ (not results).
# Exported: RESULTS_ROOT, GATE_TASKFILE, TOKEN, RF_CONFIG, [FIXED_NPZ], [SMOKE_TRIALS (default 3)], [SMOKE_ROWS (default 2)].
#SBATCH --job-name=rfp_gate
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
[ "$MODEL" = rf ] && [ "$TW" = 2000 ] && [ "$SEG" = bar1600 ] && [ "$HALO" = 2000 ] || { echo "gate line must be rf / 2000 / bar1600 / halo 2000: '$LINE'"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-0}_${I}"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/pylib_trees"
TOKEN="${TOKEN:-rfp163}"  # the key token: specs/causal_tune_trees_<token>.py, roots, task files, tables
if [ "$TOKEN" = rfp163 ]; then RF_CONFIG="${RF_CONFIG:-cluster/rfprev163_rf_config.json}"; else RF_CONFIG="${RF_CONFIG:?RF_CONFIG must name the configuration of token $TOKEN}"; fi
echo "$TOKEN gate ($RF_CONFIG) $BUCKET on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')]: $LINE $(date)"
python experiments/trees_rfp163_make_specs.py --config "$RF_CONFIG" --token "$TOKEN" --check \
  || { echo "specs/causal_tune_trees_$TOKEN.py is not the file generated from $RF_CONFIG"; exit 1; }
GR="$R/gate/$BUCKET"
rm -rf "$GR"
mkdir -p "$GR"
python -u experiments/trees_rfp163_optuna_gate.py --bucket "$BUCKET" --row0 $((START - HALO)) --rows $((END - START)) \
  --pool "${SLURM_CPUS_PER_TASK:-20}" --out "$GR" ${FIXED_NPZ:+--fixed-npz "$FIXED_NPZ"} --config "$RF_CONFIG" --token "$TOKEN" \
  --smoke-trials "${SMOKE_TRIALS:-3}" --smoke-rows "${SMOKE_ROWS:-2}" --flag "$R/GATE_OK_$BUCKET"
