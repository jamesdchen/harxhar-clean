#!/bin/bash
# 2016-start learning-curve sweep, part 1 (designs + trees), one pre-test fold (fold 3).
# Designs: W = 1,000 rebuilt (md5 re-check of the first-day independence) and W = 4,000 built with --first-day
# 2016-01-04; W = 1,500 .. 3,500 linked to the 2014-start caches (md5-identical to the 2012-start ones).
# Trees: fold 3 x 3 trees x 2 sets at every W (tree16.sbatch).
set -euo pipefail
B=/scratch1/jc_905/wsweep
R=$B/code/results
D14=$R/window_sweep_2014start_2026-10-22
D16=$R/window_sweep_2016start_2026-10-22
L=$B/logs
mkdir -p $L $B/trees16/tasks $B/trees16/out
cd $B/jobs
for W in 1000 1500 2000 2500 3000 3500 4000; do
  OUT=results/window_sweep_2016start_2026-10-22/w$W
  dep=()
  if [ $W = 1000 ] || [ $W = 4000 ]; then
    d=$(sbatch --parsable --account=pollok_1603 --partition=main --constraint="epyc-7513|epyc-7542" \
        --export=ALL,WS_NPY=1,WS_NUMBA=0,WS_FIRST_DAY=2016-01-04 -J wsd_16w$W -o $L/16w$W.design.log -c 2 --mem=32G \
        -t 4:00:00 stage.sh "$OUT" "$W" 3 design 2)
    dep=(--dependency=afterok:$d)
    echo "W=$W design $d"
  else
    mkdir -p $D16/w$W/_work/logs
    for s in baseline all_features; do ln -sfn $D14/w$W/_work/design_$s $D16/w$W/_work/design_$s; done
    cp $D14/w$W/_work/logs/design_*.log $D16/w$W/_work/logs/
    echo "W=$W designs linked to $D14/w$W"
  fi
  T=$B/trees16/tasks/w$W.txt
  : > $T
  for est in rf xgb lgbm; do for s in all_features baseline; do echo "$W 3 $est $s" >> $T; done; done
  t=$(sbatch --parsable "${dep[@]}" --export=ALL,TASKS=$T -J wst_16w$W -o "$L/16w$W.tree.%a.log" --array=1-6 \
      $B/trees16/tree16.sbatch)
  echo "W=$W trees $t"
done
