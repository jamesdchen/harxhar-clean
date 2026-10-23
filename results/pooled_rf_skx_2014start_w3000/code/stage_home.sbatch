#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --constraint=epyc-7513
# The final designs (W = 1,000, 2,000, 3,000, 3,500) and the benchmark's forecasts copied from /scratch1 to /home1:
# each copy md5-checked against the MD5SUMS of its /scratch1 copy (itself md5-equal to the window sweep's source).
set -euo pipefail
SB=/scratch1/jc_905/trees_final
HB=/home1/jc_905/trees_final
F="X.npy y.npy b_oos.npy dates_oos.csv design.json"
echo "stage_home on $(hostname) $(date)"
for W in 3000 3500 2000 1000; do
  for s in baseline all_features; do
    S=$SB/designs/w$W/_work/design_$s
    D=$HB/designs/w$W/_work/design_$s
    mkdir -p $D
    for f in $F MD5SUMS MD5SUMS.source; do cp $S/$f $D/; done
    (cd $D && md5sum -c --quiet MD5SUMS) && echo "w$W design_$s: /home1 copy md5 = MD5SUMS"
    cmp $D/MD5SUMS $D/MD5SUMS.source && echo "w$W design_$s: MD5SUMS = the sweep source's"
  done
  cp $SB/designs/w$W/SOURCE $HB/designs/w$W/
done
mkdir -p $HB/bench/out
cp $SB/bench/out/B_*_epyc.parquet $SB/bench/out/B_*_epyc.json $HB/bench/out/
for f in $HB/bench/out/*; do cmp $f $SB/bench/out/$(basename $f); done && echo "bench forecasts copied, byte-equal"
du -sh $HB/designs $HB/bench
echo "stage_home done $(date)"
