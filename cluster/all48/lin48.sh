#!/bin/bash
# ALL48 Part 2, the linear chain on CARC (results/all48_2026-10-30/RULE.txt sections 2.1-2.2; exploratory), a job
# payload (no sbatch, copy or move in here).  Every stage is experiments/all48_linear.py (or the scorer) through the
# FOMC-fix linear stage environment, results/fomcfix_2026-10-26/code/lin_stage.sh stage "py" (its venv, OpenBLAS kernel
# Prescott, numpy's AVX-512 dispatch off, one thread a runner process; hosts.txt / stage_times.txt under <run>/_work).
# usage: lin48.sh <stage>
#   design    the two 48-interval design caches (2 CPUs, ~96 GB, ~1 h)
#   gate      D1, D2, D3 -> <run>/gate_design.json; exit 3 on a failure
#   pretune   pretune + select -> <run>/frozen_penalties.json, pretune_mse.csv
#   gates     L1 (gate_har) and L2 (gate_seam) -> <run>/gate_har.json, gate_seam.json; exit 3 on a failure
#   walk1     walk job SLURM_ARRAY_TASK_ID (19 jobs: full-set ridge 2, lasso 8, elastic net 5 chunks; HAR set and OLS 1)
#   export    <run>/forecasts_all.parquet (every walked row, no RTH filter)
#   score     experiments/all48_score.py --part 2 on the run's walks (+ $TREES) -> $SCORES
# env: A48 (staging dir holding code/), FF_PY (python; default /scratch1/jc_905/wsweep/venv/bin/python, as
#      cluster/transform/tx_variant.sh), RUN (default $A48/runs/lin), CELLB (default /home1/jc_905/fomcfix_lin/runs/B),
#      REF (train3way's ALL48 run dir; default /scratch1/jc_905/train3way/code/results/linear_train3way_2014start_w3000/ALL48),
#      TREES (score: the trees' merged forecasts.parquet; optional), SCORES (score: default $A48/scores/part2)
set -euo pipefail
: "${A48:?}"
export FF_BASE=$A48 FF_PY=${FF_PY:-/scratch1/jc_905/wsweep/venv/bin/python}
S=$A48/code/results/fomcfix_2026-10-26/code/lin_stage.sh
RUN=${RUN:-$A48/runs/lin}
CELLB=${CELLB:-/home1/jc_905/fomcfix_lin/runs/B}
REF=${REF:-/scratch1/jc_905/train3way/code/results/linear_train3way_2014start_w3000/ALL48}
L=experiments/all48_linear.py
py() { PYARGS="$*" bash "$S" "$RUN" py; }
echo "lin48 ${1:-} on $(hostname) $(date); run $RUN"
case ${1:?stage} in
  design) py $L design --out "$RUN" --jobs 2 ;;
  gate) py $L gate --out "$RUN" --cellb "$CELLB" --ref "$REF" ;;
  pretune)
    py $L pretune --out "$RUN" --jobs 4
    py $L select --out "$RUN" ;;
  gates)
    py $L gate_har --out "$RUN" --ref "$REF" --jobs 4
    py $L gate_seam --out "$RUN" --jobs 8 ;;
  walk1) py $L walk1 --out "$RUN" ;;
  export) py $L export --out "$RUN" ;;
  score)
    X=""
    if [ -n "${TREES:-}" ]; then X="--trees $TREES"; fi
    py experiments/all48_score.py --part 2 --linear-root "$RUN" $X --out "${SCORES:-$A48/scores/part2}" ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
echo "lin48 $1 done $(date)"
