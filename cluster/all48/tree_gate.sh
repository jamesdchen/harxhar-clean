#!/bin/bash
# ALL48 Part 2, the tree gates of one CPU model (results/all48_2026-10-30/RULE.txt section 2.3: T1, T2), a job payload
# (no sbatch, copy or move in here); 8 threads, every walk a separate process, one after the other:
#   T1  experiments/all48_trees.py walk --ends 10 on the cell-B 13-bar cache (D13), both sets, each class
#   T2  the same on the 48-interval cache (D48)
#   then all48_trees.py gate (T1 against the main forecasts; T2 against REF's pieces, or this tag is the reference)
#   -> $A48_B/gate/$TAG/passed_<est> (the CPU model) for each class that passes; exit 3 if any fails.
# env: A48_B (staging holding code/), PYSH (cluster/ablate/py_carc.sh on CARC, which reads ABL_B; cluster/all48/py_h2.sh
#      on Hoffman2), TAG (one CPU model a tag), WANT (a regex the node's CPU model must match, else exit 99), D13 (the
#      cell-B 13-bar run dir: _work/design_<set>, md5 checked by the gate), D48 (the 48-interval run dir; default
#      $A48_B/runs/lin), REF (the reference model's gate dir for T2; empty = this tag is the reference, CARC EPYC
#      7513 / 7542), GEST (classes; default lgbm,xgb,rf_skx)
set -uo pipefail
: "${A48_B:?}" "${PYSH:?}" "${TAG:?}" "${WANT:?}" "${D13:?}"
D48=${D48:-$A48_B/runs/lin}
GEST=${GEST:-lgbm,xgb,rf_skx}
export ABL_B=$A48_B
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/ *$//')
echo "tree_gate $TAG job ${SLURM_JOB_ID:-${JOB_ID:-}} on $(hostname) [$CPU] $(date); D13 $D13 D48 $D48 REF ${REF:-none}"
echo "$CPU" | grep -qE "$WANT" || { echo "CPU '$CPU' is not '$WANT': exit 99"; exit 99; }
G=$A48_B/gate/$TAG
mkdir -p "$G/logs"
for e in ${GEST//,/ }; do
  env=gb
  if [ "$e" = rf_skx ]; then env=skx; fi
  for s in baseline all_features; do
    for d in "$D13" "$D48"; do
      lab=d48
      if [ "$d" = "$D13" ]; then lab=d13; fi
      NT=8 bash "$PYSH" $env experiments/all48_trees.py walk --design "$d" --est "$e" --set "$s" --ends 10 \
        --out "$G" --threads 8 > "$G/logs/${e}_${s}_$lab.log" 2>&1
      echo "$e $s $lab walk rc $? $(date)"
    done
  done
done
X=()
if [ -n "${REF:-}" ]; then X+=(--ref "$REF"); fi
NT=1 bash "$PYSH" gb experiments/all48_trees.py gate --out "$G" --est "$GEST" "${X[@]}"
rc=$?
echo "tree_gate $TAG done rc $rc $(date)"
exit $rc
