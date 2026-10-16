#!/bin/bash
# INTERIM 16:00 table of the pooled forest all_features (2026-10-13): experiments/pooled_trees_interim1600.py over the
# main root and the race root; refuses (exit 1) unless every row of cluster/pooltree_interim1600_rows.csv is found.
#SBATCH --job-name=ptr_interim
#SBATCH --account=pollok_1603
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x.%j.out
set -eo pipefail
module load conda >/dev/null 2>&1
source activate harxhar
set -u
export PYTHONPATH="$PWD:$PWD/experiments"
R=results/pooled_trees_2026-10-07
python experiments/pooled_trees_interim1600.py --root $R/chunks --root results/pooled_trees_2026-10-07_race/chunks \
  --arm pooltree_rf_all_features --rows cluster/pooltree_interim1600_rows.csv \
  --out $R/yhat_pooltree_rf_all_features_INTERIM1600.parquet
md5sum $R/yhat_pooltree_rf_all_features_INTERIM1600.parquet
