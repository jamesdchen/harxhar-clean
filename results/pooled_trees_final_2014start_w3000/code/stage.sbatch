#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=tf_stage
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/jc_905/trees_final/logs/%x.%j.out
# Final-design caches for the tree walks: the 2014-start window sweep's design caches (W = 3,000 and 3,500; built on
# CARC on 6c264b7 = e34 8898522 + --first-fold, LATE_START_WARMUP / UNOBSERVED_NEUTRAL / EXCLUDE_FOMC_RELEASE on), read
# only, copied to /scratch1/jc_905/trees_final/designs/w<W>/_work/design_<set>/ (md5 of the copy = md5 of the source;
# the source re-verified after the copy); the sha256 of the printed 2012-start w1000 design (the gate's input, read in
# place) against results/pooled_trees_rerun13_2026-10-20/SUMMARY.md; then design_check.py on the copies.
set -euo pipefail
BASE=/scratch1/jc_905/trees_final
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
echo "tf_stage on $(hostname) $(date)"
F="X.npy y.npy b_oos.npy dates_oos.csv design.json"
for W in 3000 3500; do
  SRC=/scratch1/jc_905/wsweep/code/results/window_sweep_2014start_2026-10-22/w$W/_work
  for s in baseline all_features; do
    DEST=$BASE/designs/w$W/_work/design_$s
    mkdir -p $DEST
    (cd $SRC/design_$s && md5sum $F) > $DEST/MD5SUMS.source
    for f in $F; do cp $SRC/design_$s/$f $DEST/; done
    (cd $DEST && md5sum $F > MD5SUMS)
    cmp $DEST/MD5SUMS $DEST/MD5SUMS.source && echo "w$W design_$s: md5 of the copy = md5 of the source"
    (cd $SRC/design_$s && md5sum -c --quiet $DEST/MD5SUMS) && echo "w$W design_$s: source unchanged after the copy"
    echo "## w$W design_$s source=$SRC/design_$s"; cat $DEST/MD5SUMS
  done
  mkdir -p $BASE/designs/w$W/_work/logs && cp $SRC/logs/design_*.log $BASE/designs/w$W/_work/logs/
  echo "$SRC" > $BASE/designs/w$W/SOURCE
  grep -H "\[scale\]" $BASE/designs/w$W/_work/logs/design_*.log || true
done
P=/scratch1/jc_905/harxhar-trees13/results/pooled_linear_rerun_2026-10-19_13_w1000/_work
echo "## printed 2012-start w1000 design ($P), sha256"
for s in baseline all_features; do (cd $P/design_$s && sha256sum X.npy y.npy b_oos.npy dates_oos.csv | sed "s|$| design_$s|"); done
cd $BASE/repo
export PYTHONPATH="$PWD:$PWD/experiments"
"$PY" $BASE/repo/results/pooled_trees_final_2014start_w3000/code/design_check.py $BASE/repo $BASE/designs \
  $BASE/designs/design_check.json 3000 3500
echo "tf_stage done $(date)"
