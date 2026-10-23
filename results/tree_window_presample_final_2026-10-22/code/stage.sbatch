#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=twf_stage
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-4
#SBATCH --output=/scratch1/jc_905/tree_wcheck_final/logs/%x_%A_%a.out
# Final-design caches for the trees' pre-sample window check: the window sweep's design caches (built on 9d3fc05 =
# e34 e0f3a7a + --first-fold; LATE_START_WARMUP, UNOBSERVED_NEUTRAL, EXCLUDE_FOMC_RELEASE on), read only, copied to
# this check's directory (md5 of the source = md5 of the copy, source re-verified after the copy); then each set's
# difference from the earlier check's cache of the same window (/home1/jc_905/wsweep_trees/designs/w<W>, e34 45183a0).
set -euo pipefail
WS=(1000 1500 2000 2500 3000)
W=${WS[$SLURM_ARRAY_TASK_ID]}
if [ "$W" = 1000 ]; then SRCD=gate_w1000_nofomcrel_carc; else SRCD=pooled_linear_rerun_2026-10-22_13_2012_w${W}_latestart_nofomcrel; fi
SRC=/scratch1/jc_905/wsweep/code/results/$SRCD/_work
BASE=/scratch1/jc_905/tree_wcheck_final
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
echo "twf_stage W=$W from $SRC on $(hostname) $(date)"
DEST=$BASE/designs/w$W
F="X.npy y.npy b_oos.npy dates_oos.csv design.json"
for s in baseline all_features; do
  mkdir -p $DEST/design_$s
  (cd $SRC/design_$s && md5sum $F) > $DEST/design_$s/MD5SUMS.source
  for f in $F; do cp $SRC/design_$s/$f $DEST/design_$s/; done
  (cd $DEST/design_$s && md5sum $F > MD5SUMS)
  cmp $DEST/design_$s/MD5SUMS $DEST/design_$s/MD5SUMS.source && echo "design_$s: md5 of the copy = md5 of the source"
  (cd $SRC/design_$s && md5sum -c --quiet $DEST/design_$s/MD5SUMS) && echo "design_$s: source unchanged after the copy"
  cat $DEST/design_$s/MD5SUMS
done
mkdir -p $DEST/logs && cp $SRC/logs/design_*.log $DEST/logs/
echo "$SRCD" > $DEST/SOURCE
grep -H "\[scale\]" $DEST/logs/design_*.log || true
$PY $BASE/drivers/design_diff.py $DEST /home1/jc_905/wsweep_trees/designs/w$W $BASE/diff/w$W
echo "twf_stage W=$W done $(date)"
