# Sharpe-ratio tests on the straddle returns (2026-10-17)

Script: `experiments/sharpe_tests_pooled.py` (run with `C:\Users\james\miniconda3\envs\285J\python.exe`; the base
miniconda scipy is ABI-broken against numpy 2 and its pyarrow/pandas cannot read the tz-aware tree tables).
Read-only on every input. Nothing under `writeup/` was touched.

Files: `sharpe_tests_pooled.csv` (one row per test), `block_calibration.csv` (LW block-size calibration coverage),
`notes.txt` (ex-post tree picks, pooled-tree gates), `run.log`.

## Data (same daily series as Table E1, `writeup/generated/rehaul_econ_e1_control.tex`)
`make_rehaul_economics.py` loads `results/close_master_table/master_table_daily.parquet`, column `ret_mid`
(= q R at mid-quotes, q = sign(plain - iv_var)); the short straddle is -R. Keys:

| group | keys | predictor set |
|---|---|---|
| pooled intraday linear (E1 Panel B) | `poolpt_{ridge,lasso,enet}_all_features` | full |
| pooled intraday linear (not in E1) | `poolpt_{ridge,lasso,enet}_baseline` | HAR benchmark |
| OLS, HAR benchmark (E1 Panel A) | `sub_ols_baseline` -- closing-interval; no pooled OLS series exists | HAR benchmark |
| closing trees, fixed daily rung | `subtree_daily_{all_features,baseline}_{lgbm,xgb,rf}` | both |
| closing trees, E1's ex-post picks | `subtree_optuna_tp25_all_features_lgbm`, `subtree_xgb_all_features`, `subtree_optuna_tp1_k10_all_features_rf` | full |
| pooled intraday trees (experiment, never in the paper) | `results/pooled_trees_2026-10-07/yhat_pooltree_<est>_baseline`, `results/spxw_pnl/yhat_tree00` (lgbm full), `yhat_tree10` (xgb full), `yhat_pooltree_rf_all_features_INTERIM1600` (rf full, INTERIM) | both |

The pooled trees are Optuna-tuned bank arms refit at every bar (24,000-bar window). There is no fixed-hyperparameter
pooled tree. They are traded with `master_table_close.trade_days`, exactly as `experiments/pooled_trees_score.py` does,
and each one's 866-day Sharpe equals that scorer's `levels.csv` (gate in `notes.txt`).

Samples: `oos614` = days 253-866, the sample of every Sharpe ratio printed in E1 (primary).
`all866` = all option days, the sample of E1's HAC t-statistics.

## Tests
* Lo (2002): moments (r-mu, (r-mu)^2-sigma^2), Newey-West Bartlett, lag floor(4(T/100)^(2/9)) (5 at T=614, 6 at T=866),
  delta method; z = SR/SE, two-sided normal p. `se_iid` = sqrt((1+SR^2/2)/T). SE is per-period;
  `SE_ann` = SE*sqrt(252) (Lo's eta(q) correction for autocorrelation not applied).
* Ledoit-Wolf (2008), each model against the short straddle and against the OLS HAR benchmark, date-aligned:
  - HAC delta method on v = (mu_i, mu_n, E r_i^2, E r_n^2) with the same Bartlett kernel and lag (LW themselves use a
    prewhitened QS kernel), giving `p_hac`.
  - Studentized circular block bootstrap (LW eq. 9: PV = (#{d*>=d}+1)/(M+1), M = 4999, block-based Psi*),
    giving `p_boot` (b = 5, primary) and `p_boot_b10` (b = 10).
  - LW calibration as in `bootTimeInference::block.size.calibrate` (b in {1,3,6,10}; pseudo-data by block bootstrap
    with b_av = 5; K = 100, below the R default of 1000 for run time; M = 199), giving `b_cal` and `p_boot_bcal`.
    It ran on oos614 only. Coverages are in `block_calibration.csv`; with K = 100 the choice of b is noisy, and the
    p-values barely move across b.
  - Seed 20261017.

## Sanity checks
Every E1 Sharpe ratio is reproduced to rounding on oos614 (column `e1_SR_ann`): short 0.34, OLS 1.64, ridge 1.73,
lasso 1.56, elastic net 1.91, LightGBM 1.86, XGBoost 1.59, random forest 1.38.
