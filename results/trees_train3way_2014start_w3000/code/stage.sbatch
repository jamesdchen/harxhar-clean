#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# The design caches of the comparison: the linear comparison's RTH13 (rebuilt, = the templates' w3000 design) and
# ALL48 caches copied (read only there) to $BASE/designs/<ARM>/_work/design_<set>/, each copy md5-checked against its
# source after the copy (and the source re-checked); then experiments/trees_train3way.py check -> $BASE/design_check.json.
set -euo pipefail
: "${CODE:?}"
source "$CODE/results/trees_train3way_2014start_w3000/code/env.sh"
L=/scratch1/jc_905/train3way/code/results/linear_train3way_2014start_w3000
F="X.npy y.npy b_oos.npy dates_oos.csv dates_all.csv design.json"
hostlog "stage"
for A in RTH13 ALL48; do
  for s in baseline all_features; do
    S=$L/$A/_work/design_$s
    D=$BASE/designs/$A/_work/design_$s
    mkdir -p $D
    (cd $S && md5sum $F) > $D/MD5SUMS.source
    for f in $F; do cp $S/$f $D/; done
    (cd $D && md5sum $F) > $D/MD5SUMS
    cmp $D/MD5SUMS $D/MD5SUMS.source && echo "$A design_$s: md5 of the copy = md5 of the source"
    (cd $S && md5sum -c --quiet $D/MD5SUMS) && echo "$A design_$s: source unchanged after the copy"
    echo "## $A design_$s ($S)"; cat $D/MD5SUMS
  done
done
cd "$CODE"
export PYTHONPATH="$PWD:$PWD/experiments"
"$(pyof gb)" experiments/trees_train3way.py check --design-root $BASE/designs --linear $L --out $BASE
