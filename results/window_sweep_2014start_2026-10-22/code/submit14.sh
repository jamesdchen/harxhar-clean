#!/bin/bash
# 2014-start window sweep, part 1 (designs + trees): per window W the design stage (stage.sh, --first-day 2014-01-02),
# then the trees' fold fits on that design (tree14.sbatch, an array over the fittable folds x 3 trees x 2 sets).
# usage: submit14.sh [W ...]   (default 1000 1500 2000 2500 3000 3500)
set -euo pipefail
B=/scratch1/jc_905/wsweep
WS=${*:-1000 1500 2000 2500 3000 3500}
S=50777  # design row of the first 2014-01-02 bar (folds14.json)
L=$B/logs
mkdir -p $L $B/trees14/tasks $B/trees14/out
cd $B/jobs
for W in $WS; do
  OUT=results/window_sweep_2014start_2026-10-22/w$W
  d=$(sbatch --parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542" \
      --export=ALL,WS_NPY=1,WS_NUMBA=0,WS_FIRST_DAY=2014-01-02 -J wsd_14w$W -o $L/14w$W.design.log -c 2 --mem=32G \
      -t 4:00:00 stage.sh "$OUT" "$W" 0 design 2)
  T=$B/trees14/tasks/w$W.txt
  : > $T
  for est in rf xgb lgbm; do
    for s in all_features baseline; do
      for f in 0 1 2 3; do
        if [ $(( S - (4 - f) * 3250 - 325 - 13 * W )) -ge 0 ]; then echo "$W $f $est $s" >> $T; fi
      done
    done
  done
  n=$(wc -l < $T)
  t=$(sbatch --parsable --dependency=afterok:$d --export=ALL,TASKS=$T -J wst_14w$W -o "$L/14w$W.tree.%a.log" \
      --array=1-$n $B/trees14/tree14.sbatch)
  echo "W=$W design $d trees $t ($n fits)"
done
