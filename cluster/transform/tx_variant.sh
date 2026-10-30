#!/bin/bash
# 2026-10-30 transform ablation Part A (results/transform_ablation_2026-10-30/RULE.txt), one variant's linear chain on
# CARC, a job payload (no sbatch, copy or move in here), through the environment of the FOMC-fix linear stages
# (results/fomcfix_2026-10-26/code/lin_stage.sh: its venv, OpenBLAS kernel and numpy dispatch):
#   design (cell B + the variant's switches, experiments/transform_design_gate.py flags) -> gate against the cell-B
#   caches (V0: identical md5s; every other variant: only the parts its switches touch change; exit 3 on a failure)
#   -> pretune + select on fold 3 -> walk (refit at every bar, as Table 1) -> score stage (forecasts.parquet with
#   pred_adj and B; its plain / QLIKE are not used for the log or level variants: experiments/transform_score.py
#   applies each variant's inverse map).
# env: V (variant), FF_BASE (staging dir holding code/; runs go to $FF_BASE/runs/$V), REF (cell-B run dir, default
#      /home1/jc_905/fomcfix_lin/runs/B), J (walk jobs, default 10), FF_PY (python; lin_stage.sh's default)
set -euo pipefail
: "${V:?}" "${FF_BASE:?}"
REF=${REF:-/home1/jc_905/fomcfix_lin/runs/B}  # 2026-10-30: the cell-B caches (the /scratch1 copy is empty)
J=${J:-10}
PY=${FF_PY:-/scratch1/jc_905/wsweep/venv/bin/python}
S=$FF_BASE/code/results/fomcfix_2026-10-26/code/lin_stage.sh
cd "$FF_BASE/code"
VFLAGS=$("$PY" experiments/transform_design_gate.py flags --variant "$V" | tail -1)
CELL="FOMC_SCHEDULE_2020_FIX=1,FOMC_PROXIMITY_NO_WINSOR=1,DROP_HOLIDAY_SESSIONS=0"
for kv in $(echo "$VFLAGS" | sed 's/--ex-flag//g'); do CELL="$CELL,$kv"; done
export CELL FF_BASE FF_PY="$PY"
F="--window-days 3000 --first-day 2014-01-02 --ex-flag LATE_START_WARMUP=1"
for kv in ${CELL//,/ }; do F="$F --ex-flag $kv"; done
OUT=../runs/$V
echo "tx_variant $V on $(hostname) $(date); CELL=$CELL"
if [ -f "$OUT/_work/design_baseline/design.json" ] && [ -f "$OUT/_work/design_all_features/design.json" ]; then echo "designs present: kept"; else bash "$S" "$OUT" design 2; fi
for s in baseline all_features; do
  PYARGS="experiments/transform_design_gate.py gate --ref $REF/_work/design_$s --new $OUT/_work/design_$s --variant $V --out $OUT/gate_$s.json" \
    bash "$S" "$OUT" py || { echo "gate $s FAIL: exit 3"; exit 3; }
done
bash "$S" "$OUT" pretune_select 4
PYARGS="experiments/pooled_linear_rerun_2026_10_19_13.py walk --jobs $J $F --first-fold 3 --chunks all_features/ridge=2 --chunks all_features/lasso=4 --chunks all_features/enet=4 --out $OUT" \
  bash "$S" "$OUT" py
PYARGS="experiments/pooled_linear_rerun_2026_10_19_13.py score $F --first-fold 3 --out $OUT" bash "$S" "$OUT" py
echo "tx_variant $V done $(date)"
