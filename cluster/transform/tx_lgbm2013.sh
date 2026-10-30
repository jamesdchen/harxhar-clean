#!/bin/bash
# 2026-10-30 transform ablation Part A, section 6 of results/transform_ablation_2026-10-30/RULE.txt: the LightGBM
# selection walk of one variant on the 2013 fold (refit every day, the frozen configuration, full set), merged to
# <runs>/<V>/forecasts_sel2013_lgbm.parquet for experiments/transform_score.py, which picks the three primary
# variants other than V0 with the lowest 2013 QLIKE for the LightGBM confirmation.  A CARC job payload (no sbatch,
# copy or move in here).  env: V (variant), TX (staging dir holding code/ and runs/), NT (threads, default 8)
set -euo pipefail
: "${V:?}" "${TX:?}"
NT=${NT:-8}
R=$TX/runs/$V
P="bash $TX/code/cluster/ablate/py_carc.sh gb"
export ABL_B=$TX
echo "tx_lgbm2013 $V on $(hostname) $(date)"
NT=$NT $P experiments/pooled_trees_rerun_13.py walk --design "$R" --out "$R/lgbm2013" --window-days 3000 --est lgbm \
  --set all_features --refit day --first-day 2013-01-02 --last-day 2013-12-31 --part 1/1 --threads "$NT"
NT=1 $P experiments/pooled_trees_rerun_13.py merge --design "$R" --out "$R/lgbm2013" --window-days 3000 --est lgbm \
  --set all_features --refit day --first-day 2013-01-02 --last-day 2013-12-31 --dest "$R/forecasts_sel2013_lgbm.parquet"
echo "tx_lgbm2013 $V done $(date)"
