#!/bin/bash
# 2026-10-22 pooled ridge-init MinGRU on the 13-bar design: submit the canary, the 16 tune shards (4 folds x {3
# all-features shards, 1 HAR-benchmark shard}), the select job held on them, and the 12 walk jobs held on select and
# the canary.  Run on CARC from /home1/jc_905/pm13 after the prep job (cluster/pooled_mingru13_cpu_carc.sbatch
# MODE=prep) wrote inputs/.  Prints every job id (append them to results/pooled_mingru_rerun13_2026-10-22/STATUS.md).
set -euo pipefail
cd /home1/jc_905/pm13
G=pooled_mingru13_gpu_carc.sbatch
C=pooled_mingru13_cpu_carc.sbatch
can=$(sbatch --parsable --job-name=pm13can --export=NONE,JOBS=canary_walk.txt,SUB=refit,TAG=canary,CONC=4 $G)
echo "canary $can"
ids=""
for s in all_features baseline; do
  if [ $s = all_features ]; then ns=3; conc=6; else ns=1; conc=12; fi
  for k in 0 1 2 3; do
    for ((i = 0; i < ns; i++)); do
      id=$(sbatch --parsable --job-name=pm13t --export=NONE,JOBS=tune_${s}_f$k.s$i.txt,SUB=tune,TAG=tune_${s}_f${k}_s$i,CONC=$conc $G)
      echo "tune $s f$k s$i $id"
      ids="$ids:$id"
    done
  done
done
sel=$(sbatch --parsable --job-name=pm13sel --export=NONE,MODE=select,PARTS=6 --dependency=afterok$ids $C)
echo "select $sel"
for s in all_features baseline; do
  if [ $s = all_features ]; then conc=6; else conc=12; fi
  for p in 0 1 2 3 4 5; do
    id=$(sbatch --parsable --job-name=pm13w --export=NONE,JOBS=walk_$s.p$p.txt,SUB=refit,TAG=walk_${s}_p$p,CONC=$conc --dependency=afterok:$sel:$can $G)
    echo "walk $s p$p $id"
  done
done
