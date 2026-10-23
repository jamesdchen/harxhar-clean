#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home1/jc_905/trees_final/logs/%x.%A_%a.out
# rf_skx gate, one line of $TASKS:
#   "fold <W> <set>"                                                gate_skx.py fold (the benchmark's engine-B fits)
#   "walk <W> <out dir> <est> <set> <refit> <first> <last> <part>"  a walk piece (pooled_trees_rerun_13.py walk)
#   "time <W>"                                                      the rf_skx time probe (3 refits, 2014-01-02)
# Everything on /home1 (some CARC nodes have a failing /scratch1 mount): python = the venv $HB/venv_skx built on the
# md5-verified copy /home1/jc_905/envs/harxhar-r13 (+ scikit-learn-intelex 2026.2.0), code = $CODE (git archive of
# the branch commit), designs $HB/designs (md5-verified copies).  The array's --constraint picks the CPU model.
set -euo pipefail
HB=/home1/jc_905/trees_final
: "${TASKS:?}" "${CODE:?}"
PY=$HB/venv_skx/bin/python
NT=$SLURM_CPUS_PER_TASK
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMBA_NUM_THREADS=$NT
[ -f /home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf ] && export OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
echo "gate_skx [$LINE] on $(hostname) [$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')] $PY $(date)"
set -- $LINE
case $1 in
  fold) "$PY" results/pooled_rf_skx_2014start_w3000/code/gate_skx.py fold --repo $PWD --design $HB/designs/w$2 \
          --W $2 --set $3 --threads $NT --bench $HB/bench/out --out $HB/gate_skx/fold ;;
  walk) "$PY" experiments/pooled_trees_rerun_13.py walk --design $HB/designs/w$2 --out $3 --window-days $2 \
          --est $4 --set $5 --refit $6 --first-day $7 --last-day $8 --part $9 --threads $NT ;;
  time) "$PY" experiments/pooled_trees_rerun_13.py time --design $HB/designs/w$2 --out $HB/runs_skx/w$2 \
          --window-days $2 --k 3 --threads $NT --span 2014-01-02:2024-04-30 --est rf_skx ;;
esac
echo "gate_skx done $(date)"
