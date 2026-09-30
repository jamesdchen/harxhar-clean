#!/bin/bash
# I1 canary (Hoffman2): the baseline bucket x recursive lasso x 2000 sessions x the 15:30-16:00
# bar on the de-duplicated per-bar design, through cluster/linear_dedup_pack.sh.  That run also
# writes the per-bar OLS incumbent.  CANARY_OK is written only if
#   * the design has the 22 columns of the de-duplicated HAR + calendar design and no
#     har_ma_*_x_open / har_ma_*_x_close column (the executor's feature-health table),
#   * the stamps and the target equal the pre-dedup arm's, and
#   * the lasso and OLS forecasts equal the pre-dedup ones (cluster/linear_dedup_ref_*.csv, the
#     arms the master table scores): an L1 or least-squares fit's predictions do not change when a
#     copy of a column is dropped.  The bound is a float-path tolerance, not a model constant
#     (the lasso-fix canary's).
#$ -cwd
#$ -j y
#$ -o logs/
#$ -l h_rt=1:00:00,h_data=16G
set -eo pipefail
source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate hpc-pi
set -u
R=results/linear_subsection_dedup
rm -f "$R/CANARY_OK"
SGE_TASK_ID=1 TASKFILE=cluster/linear_dedup_tasks_canary.txt CANARY_FLAG="" bash cluster/linear_dedup_pack.sh
python - <<'PY'
import sys
import numpy as np, pandas as pd
D = "results/linear_subsection_dedup/baseline/bar1600/reclasso/tw2000/causal_tune_linear"
TOL = 1e-6  # float-path tolerance on the adjusted scale (O(1) values)
ok = True
fh = pd.read_csv(f"{D}/reclasso/baseline/results_bar1600_feature_health.csv")
cols = list(fh["feature"])
edge = [c for c in cols if c.endswith("_x_open") or c.endswith("_x_close")]
print(f"design: {len(cols)} columns, session-edge columns {edge}")
ok &= len(cols) == 22 and not edge
for name, got_f, ref_f in (
    ("lasso", f"{D}/reclasso/baseline/results_bar1600.csv", "cluster/linear_dedup_ref_lasso_baseline_bar1600.csv"),
    ("OLS", f"{D}/incumbent_ols/results_bar1600.csv", "cluster/linear_dedup_ref_ols_baseline_bar1600.csv"),
):
    g, r = pd.read_csv(got_f), pd.read_csv(ref_f)
    same = len(g) == len(r) and (g["date"] == r["date"]).all()
    dt = float(np.max(np.abs(g["true_adj"] - r["true_adj"]))) if same else np.inf
    dp = float(np.max(np.abs(g["pred_adj"] - r["pred_adj"]))) if same else np.inf
    rel = float(np.max(np.abs(g["pred_adj"] / r["pred_adj"] - 1.0))) if same else np.inf
    print(f"{name}: {len(g)} rows vs {len(r)}, stamps equal {same}, max |d true_adj| {dt:.2e}, "
          f"max |d pred_adj| {dp:.2e} (relative {rel:.2e})")
    ok &= same and dt == 0.0 and dp < TOL
sys.exit(0 if ok else 1)
PY
touch "$R/CANARY_OK"
echo "CANARY_OK written  $(date)"
