#!/bin/bash
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --job-name=wsfold
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --constraint=epyc-7513
#SBATCH --output=/home1/jc_905/wsweep_trees/logs/%x_%A_%a.out
# One task = one line of tasks.txt: "<window_days> <fold> <est> <set>".  Code (git archive of e34 45183a0), the
# window's design cache and the driver are staged on node-local disk (md5-verified); outputs copied back verified.
set -euo pipefail
BASE=/home1/jc_905/wsweep_trees
PY=/scratch1/jc_905/envs/harxhar-r13/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
read -r W F EST SET <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" $BASE/drivers/tasks.txt)"
TAG=w${W}_f${F}_${EST}_${SET}
LOCAL=${TMPDIR:-/tmp}/wsf_${SLURM_JOB_ID}
mkdir -p "$LOCAL/repo" "$LOCAL/design_$SET" "$LOCAL/out"
trap 'rm -rf "$LOCAL"' EXIT
echo "wsfold $TAG on $(hostname) [$(lscpu | grep 'Model name' | sed 's/.*: *//')] $(date)"
HEAD=$(git -C $BASE/repo rev-parse HEAD)
[ "$HEAD" = 45183a05acc626184003e3ecbc7dc90c9db4a7fc ]
git -C $BASE/repo archive HEAD experiments/pooled_trees_rerun_13.py cluster/pooltree_lgbm_config.json \
    cluster/xgbtop_xgb_config.json cluster/pooltree_rf400_config.json data/core_stats.parquet \
    results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel/frozen_penalties.json \
    | tar -x -C "$LOCAL/repo"
(cd "$LOCAL/repo" && grep ' data/core_stats.parquet$' $BASE/drivers/data_md5.txt | md5sum -c --quiet)
cp $BASE/drivers/fold_fit.py "$LOCAL/"
DZ=$BASE/designs/w$W/design_$SET
cp $DZ/X.npy $DZ/y.npy $DZ/b_oos.npy $DZ/dates_oos.csv $DZ/design.json $DZ/MD5SUMS "$LOCAL/design_$SET/"
(cd "$LOCAL/design_$SET" && md5sum -c --quiet MD5SUMS) && echo "staged design md5 OK"
$PY "$LOCAL/fold_fit.py" --repo "$LOCAL/repo" --design "$LOCAL/design_$SET" --window-days $W --fold $F \
    --est $EST --set $SET --threads $SLURM_CPUS_PER_TASK --out "$LOCAL/out"
(cd "$LOCAL/out" && md5sum $TAG.parquet $TAG.json > $TAG.md5)
cp "$LOCAL/out/$TAG.parquet" "$LOCAL/out/$TAG.json" "$LOCAL/out/$TAG.md5" $BASE/out/
(cd $BASE/out && md5sum -c --quiet $TAG.md5) && echo "wsfold $TAG copied, md5 OK $(date)"
