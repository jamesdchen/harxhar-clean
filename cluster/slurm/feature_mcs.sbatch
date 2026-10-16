#!/bin/bash
# Feature Model Confidence Set (2026-10-08) on CARC: the candidate 16:00-bar elastic nets (arm f of the
# 2026-10-06 linear pre-tune study on each candidate design), experiments/feature_mcs_runner.py.  Every job
# runs on an EPYC 7513 (the submit script's --constraint=epyc-7513; checked again below), so every candidate
# is fitted on one CPU class and one build (the kernel + the renamed reference LAPACK / BLAS 3.12 that
# cluster/slurm/ship_feature_mcs_carc.sh prepared under results/feature_mcs_2026-10-08/_work/kernel/).
#   MODE=canary  the kernel's symbol bindings under LD_DEBUG=bindings (every BLAS / LAPACK call must bind to
#                the renamed reference libraries) -> _work/bindings.csv; then line 1 of TASKFILE (the two
#                anchors + one cost probe of each kind) with SLURM_CPUS_PER_TASK processes
#   MODE=gate    the anchors vs the stored arm-f forecasts -> _work/gate.csv and, on a pass, _work/GATE_OK
#   MODE=fleet   array: line SLURM_ARRAY_TASK_ID of TASKFILE (comma-separated models); refuses to run
#                without REQUIRE_FLAG
#   MODE=pipeline  (ridge and lasso, 2026-10-08: one job, as the 100-job cap is full) canary -> gate -> fleet
#                inside one allocation: the binding check (-> _work/bindings_rl.csv); every line of
#                TASKFILE_CANARY ("<est> <models>": the anchors + one cost probe of each kind); the gate of each
#                estimator (-> _work/gate_<est>.csv, GATE_OK_<est>); then every line of TASKFILE_FLEET whose
#                estimator passed its gate (a line of an estimator without its flag is refused and skipped)
# Output: results/feature_mcs_2026-10-08/_work/models/<model>.npz + .json (an existing model is skipped).
#SBATCH --job-name=fmcs
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0:45:00
#SBATCH --output=logs/%x.%A.%a.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
R=results/feature_mcs_2026-10-08
NT="${SLURM_CPUS_PER_TASK:-1}"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "$MODE on $(hostname) [$CPU] $(scontrol show node "$(hostname -s)" 2>/dev/null | grep -o 'AvailableFeatures=[^ ]*') $NT cpus  $(date)"
echo "python $(command -v python); CONDA_PREFIX=${CONDA_PREFIX:-}"
case "$CPU" in *7513*) ;; *) echo "not an EPYC 7513: refusing (one CPU class for every candidate)"; exit 1 ;; esac
if [ -n "${REQUIRE_FLAG:-}" ] && [ ! -f "$REQUIRE_FLAG" ]; then
  echo "required flag $REQUIRE_FLAG missing; refusing to run"
  exit 1
fi
mkdir -p "$R/_work/logs"
case "$MODE" in
  canary)
    D="$R/_work/logs/lddebug_${SLURM_JOB_ID:-0}"
    LD_DEBUG=bindings LD_DEBUG_OUTPUT="$D" python -u experiments/feature_mcs_runner.py smoke
    python -u experiments/feature_mcs_runner.py bindings --files "$D".*
    rm -f "$D".*
    python -u experiments/feature_mcs_runner.py run --taskfile "${TASKFILE:?TASKFILE}" --line 1 --workers "$NT"
    ;;
  gate)
    python -u experiments/feature_mcs_runner.py gate
    ;;
  fleet)
    python -u experiments/feature_mcs_runner.py run --taskfile "${TASKFILE:?TASKFILE}" \
      --line "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID}" --workers "$NT"
    ;;
  pipeline)
    GATE_ESTS=${GATE_ESTS:-ridge:reclasso}
    D="$R/_work/logs/lddebug_${SLURM_JOB_ID:-0}"
    LD_DEBUG=bindings LD_DEBUG_OUTPUT="$D" python -u experiments/feature_mcs_runner.py smoke
    python -u experiments/feature_mcs_runner.py bindings --tag "${BIND_TAG:-_rl}" --files "$D".*
    rm -f "$D".*
    NC=$(wc -l < "${TASKFILE_CANARY:?TASKFILE_CANARY}")
    for i in $(seq 1 "$NC"); do
      python -u experiments/feature_mcs_runner.py run --taskfile "$TASKFILE_CANARY" --line "$i" --workers "$NT"
    done
    for E in ${GATE_ESTS//:/ }; do  # default ridge:reclasso (the 2026-10-08 run); buckets: all three
      python -u experiments/feature_mcs_runner.py gate --est "$E" || echo "gate $E FAILED"
    done
    NF=$(wc -l < "${TASKFILE_FLEET:?TASKFILE_FLEET}")
    for i in $(seq 1 "$NF"); do
      E=$(sed -n "${i}p" "$TASKFILE_FLEET" | cut -d' ' -f1)
      case "$E" in ridge) F=$R/_work/GATE_OK_ridge ;; reclasso) F=$R/_work/GATE_OK_lasso ;; *) F=$R/_work/GATE_OK ;; esac
      if [ ! -f "$F" ]; then echo "fleet line $i ($E): $F missing, refused"; continue; fi
      python -u experiments/feature_mcs_runner.py run --taskfile "$TASKFILE_FLEET" --line "$i" --workers "$NT"
    done
    ;;
  *) echo "MODE must be canary, gate, fleet or pipeline"; exit 2 ;;
esac
echo "done $MODE  $(date)"
