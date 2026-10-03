#!/bin/bash
# 2026-10-03 close study "trees and training-set size" -- Hoffman2 (SGE) array task: one task runs
# one TIME CHUNK of one tree arm of experiments/close_trees_datasize.py at one refit cadence, with
# the same code and the same captured design as the local run, into
#   results/close_studies_2026-10-03/trees_datasize[/refit<k>]/_work/<arm>.part<c>of<n>.npz
# (the script's own layout; `merge` joins the pieces, `analyze` scores them -- both run on the laptop
# after the pull, because the scorer reads the deck and the stored forecasts, which are not shipped).
#
# Task lines (TASKFILE): "<arm> <refit_every> <chunk> <n_chunks>", e.g. "lgbm_pool_w4000 1 3 8".
# Exported at submit time (cluster/submit_close_trees_datasize_h2.sh): TASKFILE.
# Every fit is single-threaded (the script pins its model threads and BLAS / OpenMP to 1), so a
# number does not depend on the slot count; each refit depends on its own training window only, so
# a chunked arm equals the whole arm (checked by the smoke test, see the submit script).
# A piece already written is skipped (resubmitting resumes).
#$ -cwd
#$ -j y
#$ -o logs/
set -eo pipefail
D=$PWD
# LOCAL_SMOKE=1: the local smoke test (no Hoffman2 conda; the caller's python)
if [ "${LOCAL_SMOKE:-0}" != 1 ]; then
  set +u
  source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
  conda activate hpc-pi
  set -u
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONUNBUFFERED=1
TASK="${SGE_TASK_ID:-1}"; [ "$TASK" = "undefined" ] && TASK=1
LINE=$(sed -n "${TASK}p" "${TASKFILE:?TASKFILE}")
read -r ARM CAD CH NCH <<< "$LINE"
[ -n "${NCH:-}" ] || { echo "malformed line $TASK of $TASKFILE (want 4 fields): '$LINE'"; exit 1; }
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
echo "task $TASK on $(hostname) [$CPU]: $ARM refit every $CAD, chunk $CH of $NCH $(date)"
python - <<'EOF'
import numpy, pandas, sklearn, lightgbm, xgboost
print("runtime:", " ".join(f"{m.__name__} {m.__version__}" for m in (numpy, pandas, sklearn, lightgbm, xgboost)))
EOF
a=$(date +%s)
TDS_REFIT_EVERY="$CAD" TDS_CHUNK="$CH/$NCH" python -u experiments/close_trees_datasize.py run "$ARM"
echo "$ARM refit $CAD chunk $CH/$NCH done in $(( $(date +%s) - a )) s on $(hostname -s) $(date)"
