#!/bin/bash
# Random-forest dev-span menu study (2026-10-06): one CARC task of experiments/tree_menu_dev_rf.py,
# run from the deployment root /scratch1/jc_905/harxhar-rf-devmenu (cluster/slurm/ship_rf_devmenu_carc.sh).
# STAGE (exported by cluster/slurm/submit_rf_devmenu.sh):
#   canary  gate (lgbm_opt_00 REQUIRED; xgb_opt_06, xgb_opt_00 recorded: each re-scored on the dev matrices
#           against its stored dev_screen_mse, in a fresh pool), cost probe (one fit at 16 Sobol points of
#           the RF box), then 2 RF trials into a canary journal; writes $WORK/$CANARY_TAG/CANARY_OK only when
#           the gate passed and both trials completed (CANARY_TAG, default canary: the first canary, job
#           12776003, hung in multiprocessing.Pool and cannot be cancelled from here, so reruns take a new tag)
#   worker  one TPE worker of the study (jobs/slurm/tree_menu.sbatch's layout: 8 such workers share one
#           journal): N_TRIALS random-forest trials into $WORK/tree_menu_journal_rf.log; refuses without
#           $WORK/$CANARY_TAG/CANARY_OK
#   seqgate tree_menu_dev._screen_score itself (one process, multi-threaded fits, as the menu workers) on
#           the stored entries (ENTRIES): XGBoost segfaults in the pool's forked workers on epyc-7513
# Every fit is single-threaded (the pool of SLURM_CPUS_PER_TASK processes is the parallelism); the submit
# script pins the CPU class (--constraint).
# ENV HYGIENE: the job uses the harxhar env's interpreter by absolute path and drops conda / python
# variables a submitting shell may carry (an env leak through --export=ALL killed jobs on 2026-10-06).
#SBATCH --job-name=rfdm
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE CONDA_PROMPT_MODIFIER \
  PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE _CE_CONDA _CE_M VIRTUAL_ENV
PY=${PYBIN:-/home1/jc_905/.conda/envs/harxhar/bin/python}
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# user site-packages stay ON, as in jobs/slurm/tree_menu.sbatch (its numpy shadowed the env's then too)
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/experiments"
NCPU=${SLURM_CPUS_PER_TASK:-4}
WORK=results/rf_devmenu_2026-10-06/_work
S=experiments/tree_menu_dev_rf.py
CT=${CANARY_TAG:-canary}
"$PY" -c "import sys, numpy, sklearn, lightgbm, xgboost, optuna, scipy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'lightgbm', lightgbm.__version__, 'xgboost', xgboost.__version__, 'optuna', optuna.__version__, 'scipy', scipy.__version__)" \
  || { echo "library import failed; refusing to run"; exit 1; }
CPU=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' || true)
echo "stage ${STAGE:?STAGE} on $(hostname) [$CPU] cpus $NCPU job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-} $(date)"

case "$STAGE" in
  canary)
    mkdir -p "$WORK/$CT"
    [ ! -f "$WORK/$CT/CANARY_OK" ] || { echo "a CANARY_OK already exists in $CT; refusing to re-run the canary"; exit 1; }
    "$PY" "$S" gate --tag "$CT" --procs "$NCPU" --entries lgbm_opt_00
    "$PY" "$S" cost --tag "$CT" --procs "$NCPU"
    "$PY" "$S" worker --journal "$WORK/$CT/journal_rf_canary.log" --n-trials 2 --procs "$NCPU"
    N_OK=$("$PY" - "$WORK/$CT/journal_rf_canary.log" <<'EOF'
import sys
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
st = optuna.load_study(study_name="tree_menu_dev_rf", storage=JournalStorage(JournalFileBackend(sys.argv[1])))
print(sum(t.state.name == "COMPLETE" for t in st.trials))
EOF
)
    [ "$N_OK" -ge 2 ] || { echo "canary: only $N_OK complete trials; no CANARY_OK"; exit 1; }
    date > "$WORK/$CT/CANARY_OK"
    echo "CANARY_OK written"
    # informational, last and time-boxed: the XGBoost entries (the first canary hung on them)
    timeout 2400 "$PY" "$S" gate --tag "${CT}_xgb" --procs "$NCPU" --entries xgb_opt_06,xgb_opt_00       || echo "XGBoost gate entries failed or timed out (informational; see the log above)"
    ;;
  worker)
    [ -f "$WORK/$CT/CANARY_OK" ] || { echo "$CT/CANARY_OK missing: the canary did not pass; refusing"; exit 1; }
    "$PY" "$S" worker --journal "$WORK/tree_menu_journal_rf.log" --n-trials "${N_TRIALS:?N_TRIALS}" --procs "$NCPU" --cap "${CAP:-0}"
    ;;
  seqgate)
    # the study's own screen (tree_menu_dev._screen_score), one process, SLURM_CPUS_PER_TASK threads a fit
    "$PY" experiments/tree_menu_dev_rf_seqgate.py --tag "seq${NCPU}" --entries "${ENTRIES:-lgbm_opt_00,xgb_opt_06,xgb_opt_00}"
    ;;
  *)
    echo "unknown STAGE $STAGE"; exit 1 ;;
esac
echo "done $(date)"
