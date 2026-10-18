# Sharpe tests: pooled intraday OLS HAR benchmark (2026-10-17 follow-up)

Script: `experiments/sharpe_tests_pooled_ols.py`. It reuses the functions in `experiments/sharpe_tests_pooled.py`,
unchanged. Python: `miniconda3\envs\285J`. Nothing under `writeup/` or `results/sharpe_tests_pooled_2026-10-17/` was
touched.

## Series
- The pooled OLS HAR is key `a0` in `results/close_master_table/master_table_daily.parquet`, column `ret_mid`.
- That is the master table's trade of the 16:00 rows of `results/spxw_pnl/yhat_a0.parquet`
  (plain = yhat^2 x baseline, q = sign(plain - iv_var)).
- It is the pooled OLS row of tab:rh_oos: `writeup/make_rehaul_oos_table.py` says "a0 for the OLS", and
  `make_rehaul_results.EARLIER` lists "a0".
- master_table.csv labels it "baseline (HAR + calendar OLS)" and gives an 866-day Sharpe of 0.4277; the run reproduces
  0.4277.
- The comparators are the same as in sharpe_tests_pooled.py, and the pooled trees pass the same `levels.csv` gates.

## Tests
- Lo (2002) test against zero, Newey-West Bartlett, lag floor(4(T/100)^(2/9)).
- Ledoit-Wolf HAC delta-method test, plus the studentized circular block bootstrap (b = 5, also b = 10), M = 4999,
  seed 20261018.
- Samples: oos614 (days 253-866) and all866.

## Caveat
`a0` is the earlier draft's pooled OLS, fitted on the earlier panel (results/spxw_pnl/MANIFEST.md). It is not part of
the `poolpt_*` pre-tune walk, and its predictor set is labelled "HAR + calendar" in master_table.csv.

## Data end
- The quote book `results/atm_straddle_0dte_1530/daily_a0.parquet` runs 2020-01-03 .. 2024-04-30 (866 days).
- The forecast tables end at the 30 Apr 2024 ET session.
- Option P&L cannot be computed past 30 Apr 2024 without new option data.
