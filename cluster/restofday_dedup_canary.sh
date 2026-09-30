#!/bin/bash
# I6 canary (Hoffman2): the direct rest-of-day arm live_feasible x ridge x 2000 sessions x the 15:30
# entry clock on the de-duplicated per-bar design, through cluster/restofday_dedup_pack.sh, pinned
# at submit time to the CPU architecture of its per-bar twin (agent A's bar1600 arm).
# CANARY_OK is written only if
#   * experiments/check_restofday_identity.py passes against agent A's de-duplicated per-bar table
#     results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet (shipped at that path): the 15:30
#     target equals the per-bar 16:00 target bit for bit and the forecast reproduces the table's
#     16:00 rows to its bound (1e-9 relative), with no table row missing;
#   * the design has the 232 columns of the de-duplicated live_feasible design, no
#     har_ma_*_x_open / har_ma_*_x_close column, and the same column list as A's bar1600 arm
#     (its feature-health table, shipped at its results path);
#   * the stamps and the target (true_adj, true_raw) equal A's bar1600 arm bit for bit.
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=16G
set -eo pipefail
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
R=results/linear_subsection_restofday_dedup
mkdir -p "$R"
rm -f "$R/CANARY_OK"
SGE_TASK_ID=1 TASKFILE=cluster/restofday_dedup_tasks_canary.txt CANARY_FLAG="" \
  bash cluster/restofday_dedup_pack.sh
python -u experiments/check_restofday_identity.py --root "$R" \
  --ref results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet | tee "$R/identity_check.log"
python - <<'PY'
import sys

import numpy as np
import pandas as pd

NEW = ("results/linear_subsection_restofday_dedup/live_feasible/rod1530/ridge/tw2000/"
       "causal_tune_rest_of_day/ridge/live_feasible")
TWIN = ("results/linear_subsection_dedup/live_feasible/bar1600/ridge/tw2000/"
        "causal_tune_linear/ridge/live_feasible")
P_DEDUP = 232  # live_feasible columns on the de-duplicated per-bar design (commit 47f7f9c)
TOL = 1e-9  # = check_restofday_identity.IDENTITY_REL
ok = True
cols = list(pd.read_csv(f"{NEW}/results_bar1600_feature_health.csv")["feature"])
twin_cols = list(pd.read_csv(f"{TWIN}/results_bar1600_feature_health.csv")["feature"])
edge = [c for c in cols if c.endswith("_x_open") or c.endswith("_x_close")]
print(f"design: {len(cols)} columns (want {P_DEDUP}), session-edge columns {edge}, "
      f"same list as the per-bar twin {cols == twin_cols}")
ok &= len(cols) == P_DEDUP and not edge and cols == twin_cols
g = pd.read_csv(f"{NEW}/results_bar1600.csv")
r = pd.read_csv(f"{TWIN}/results_bar1600.csv")
same = len(g) == len(r) and (g["date"] == r["date"]).all()
if same:
    dt = float(np.max(np.abs(g["true_adj"] - r["true_adj"])))
    dr = float(np.max(np.abs(g["true_raw"] - r["true_raw"])))
    rel = float(np.max(np.abs(g["pred_adj"] / r["pred_adj"] - 1.0)))
else:
    dt = dr = rel = np.inf
print(f"vs the per-bar twin: {len(g)} rows vs {len(r)}, stamps equal {same}, "
      f"max |d true_adj| {dt:.2e}, max |d true_raw| {dr:.2e}, max rel d pred_adj {rel:.2e}")
ok &= same and dt == 0.0 and dr == 0.0 and rel <= TOL
print("CANARY GATES", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
touch "$R/CANARY_OK"
echo "CANARY_OK written  $(date)"
