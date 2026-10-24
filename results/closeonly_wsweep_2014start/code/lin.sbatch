#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
# RULE.txt section 2: the linear close arm at each W of $WLIST (colon-separated, in order): derive_close from the W design copy, pretune
# (fold 3), select, the 7 walks ($NT at a time, 1 thread each), export; with GATE=1 (W = 3,000) then gate L against the
# three-way CLOSE arm (exit 3 on a failure).  The wsweep venv with the linear three-way's settings (stage3.sh).
set -euo pipefail
: "${CODE:?}" "${WLIST:?}"
source "$CODE/results/closeonly_wsweep_2014start/code/env.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
sweep_env
PY=$(pyof sweep)
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
"$PY" "$CODE/results/closeonly_wsweep_2014start/code/envinfo.py"
for W in ${WLIST//:/ }; do
  OUT=$BASE/runs/linear/w$W
  mkdir -p $OUT/_work
  hostlog "linear w$W"
  echo "$(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | SLURM_JOB_ID ${SLURM_JOB_ID:-} | $(date)" >> $OUT/_work/hosts.txt
  run() {
    local st=$1 t0
    shift
    t0=$(date +%s)
    echo "[$(date +%T)] w$W $st $*"
    "$PY" experiments/closewin_linear.py "$st" --W "$W" --arm CLOSE --out "$OUT" "$@"
    echo "$st $* $(($(date +%s) - t0)) $(hostname)" >> "$OUT/_work/stage_times.txt"
  }
  run derive_close --from "$DES/RTH13_w$W"
  run pretune --jobs "$NT"
  run select
  pids=()
  for k in 1 2 3 4 5 6 7; do
    run walk1 --task "$k" > "$OUT/_work/walk1.$k.log" 2>&1 &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge "$NT" ]; do sleep 2; done
  done
  for p in "${pids[@]}"; do wait "$p"; done
  tail -n 2 "$OUT"/_work/walk1.*.log
  run export
  if [ "${GATE:-0}" = 1 ]; then
    "$PY" experiments/closewin_linear.py gate --out "$OUT" \
      --ref /scratch1/jc_905/train3way/code/results/linear_train3way_2014start_w3000/CLOSE
  fi
done
echo "[$(date +%T)] done: $WLIST"
