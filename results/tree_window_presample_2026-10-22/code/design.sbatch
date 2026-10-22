#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=wsdesign
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --constraint=epyc-7513
#SBATCH --output=/home1/jc_905/wsweep_trees/logs/%x_%A_%a.out
# Fixed-design caches (e34 45183a0: LATE_START_WARMUP on, EXCLUDE_FOMC_RELEASE on) for the trees' pre-sample window
# check: the linear runner's design stage, as the w1000 nofomcrel run's runner_cfg, built on node-local disk and
# copied to /home1 with md5 verification.
set -euo pipefail
WS=(1000 1500 2000 2500 3000)
W=${WS[$SLURM_ARRAY_TASK_ID]}
BASE=/home1/jc_905/wsweep_trees
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
LOCAL=${TMPDIR:-/tmp}/wsd_${SLURM_JOB_ID}
mkdir -p "$LOCAL/repo"
trap 'rm -rf "$LOCAL"' EXIT
echo "wsdesign W=$W on $(hostname) [$(lscpu | grep 'Model name' | sed 's/.*: *//')] $(date)"
df -h "$LOCAL" | tail -1
HEAD=$(git -C $BASE/repo rev-parse HEAD)
echo "code $HEAD"
[ "$HEAD" = 45183a05acc626184003e3ecbc7dc90c9db4a7fc ]
cd $BASE/repo
git archive HEAD $(git ls-tree --name-only HEAD | grep -v -x -e results -e writeup) | tar -x -C "$LOCAL/repo"
cd "$LOCAL/repo"
md5sum -c --quiet $BASE/drivers/data_md5.txt && echo "data md5 OK (laptop's tracked data)"
$PY -c "import numpy, pandas, sklearn; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
OUTREL=results/pooled_linear_rerun_2026-10-22_13_2012_w${W}_latestart_nofomcrel
t0=$(date +%s)
$PY experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 2 --window-days $W --first-day 2012-01-03 \
    --ex-flag LATE_START_WARMUP=1 --out $OUTREL
echo "design stage $(( $(date +%s) - t0 )) s"
DEST=$BASE/designs/w$W
for s in baseline all_features; do
  src=$OUTREL/_work/design_$s
  mkdir -p $DEST/design_$s
  (cd $src && md5sum X.npy y.npy b_oos.npy dates_oos.csv design.json > MD5SUMS)
  cp $src/X.npy $src/y.npy $src/b_oos.npy $src/dates_oos.csv $src/design.json $src/MD5SUMS $DEST/design_$s/
  (cd $DEST/design_$s && md5sum -c MD5SUMS)
done
mkdir -p $DEST/logs && cp $OUTREL/_work/logs/* $DEST/logs/
echo "wsdesign W=$W done $(date)"
