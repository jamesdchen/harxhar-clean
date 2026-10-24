# Threshold score, QLIKE and sample splits across the paper's model set (plain forecast f²·B)

Written by `experiments/close_murphy_splits_headline.py` (`main`, 2026-10-06: no headline model, user decision) from `scores_modelset.csv`, `murphy_grid_modelset.csv`, `murphy_leader_by_threshold_modelset.csv`, `rank_vs_sharpe_modelset.csv`, `splits_modelset.csv`, `splits_summary_modelset.csv`, `sign_split_modelset.csv` and `gates.csv`. Input: the master table's per-day frame. Model set: 94 forecasts (results/close_master_table/paper_model_set.csv). Every forecast is its plain f²·B; the deck's 15:30 sign(s) trade; 866 days 2020-01-03 .. 2024-04-30; point estimates only.

## Scores by group (medians)

| group | n | QLIKE | QLIKE range | ES(c=1) | EL | same side as RV % | same side as R % | Sharpe mid | Sharpe crossed |
|---|---|---|---|---|---|---|---|---|---|
| paper | 9 | 0.1092 | 0.1078 to 0.1270 | 0.0944 | 0.3705 | 70.1 | 57.3 | 1.28 | 0.80 |
| pooled linear | 6 | 0.1081 | 0.1040 to 0.1098 | 0.0941 | 0.3784 | 71.1 | 56.6 | 1.06 | 0.57 |
| per-bar linear | 7 | 0.0967 | 0.0931 to 0.0975 | 0.0995 | 0.3673 | 68.7 | 56.8 | 1.37 | 0.89 |
| per-bar trees | 72 | 0.1026 | 0.0957 to 0.1100 | 0.0996 | 0.3831 | 69.3 | 55.7 | 0.92 | 0.45 |
| model set | 94 | 0.1030 | 0.0931 to 0.1270 | 0.0991 | 0.3781 | 69.5 | 56.1 | 1.07 | 0.58 |

- Lowest QLIKE: per-bar lasso [all_features], pre-tuned; lowest score at the threshold: LightGBM.
- Murphy grid (33 thresholds, 0.25 to 4): thresholds led by group — paper 14, per-bar linear 8, per-bar trees 8, pooled linear 3; at c = 1 the leader is LightGBM (paper).
- Oracles: told RV, the payoff's side on 62.9 % of days (Sharpe 4.53 / 4.07); told the payoff's sign, 17.28 crossed.
- Spearman of minus the loss with the Sharpe ratio across the set (point): QLIKE vs crossed +0.14; QLIKE vs mid +0.14; ES at the threshold vs crossed +0.51; ES at the threshold vs mid +0.52; EL (affine in the mid P&L) vs crossed +1.00; EL (affine in the mid P&L) vs mid +1.00.

## Sample splits across the set (midpoint)

| split | days | Sharpe > 0 | median Sharpe | range | median t | t > 1.96 | always short |
|---|---|---|---|---|---|---|---|
| all days | 866 | 94 of 94 | 1.07 | 0.29 to 1.92 | 1.97 | 48 | 0.20 |
| through 2020-03-31 (the COVID quarter) | 39 | 51 of 94 | 0.23 | -3.96 to 3.67 | 0.09 | 0 | -3.25 |
| after 2020-03-31 | 827 | 94 of 94 | 1.12 | 0.21 to 2.20 | 2.03 | 48 | 0.42 |
| before 2022-05-16 | 377 | 92 of 94 | 0.79 | -0.51 to 2.12 | 0.96 | 8 | -0.45 |
| from 2022-05-16 (daily SPXW listings) | 489 | 94 of 94 | 1.17 | 0.22 to 2.34 | 1.62 | 34 | 0.73 |
| settlement pins (R = -1) | 193 | 94 of 94 | 7.54 | 4.34 to 14.63 | 6.60 | 94 | -- |
| off pins | 673 | 16 of 94 | -0.36 | -1.94 to 0.68 | -0.59 | 0 | -3.77 |

- Sign split (mean R on buy days minus sell days, HC0 t): median 0.164, positive for 94 of 94, median t 1.80, t > 1.96 for 43.

## Gates
- slice = the deck's iv_var: 0 ≤ 1e-09 on n = 866 — PASS
- realized variance (rv_unclipped) shared by every forecast: 0 ≤ 1e-09 on n = 94 — PASS
- q = the master table's, every forecast and day: 0 ≤ 0 on n = 81404 — PASS
- ret_mid = the master table's: 0 ≤ 1e-10 on n = 81404 — PASS
- ret_crossed = the master table's: 0 ≤ 1e-10 on n = 81404 — PASS
- Sharpe_mid = master_table.csv, every forecast: 2.22e-16 ≤ 1e-09 on n = 94 — PASS
- Sharpe_crossed = master_table.csv, every forecast: 3.33e-15 ≤ 1e-09 on n = 94 — PASS
- QLIKE (plain) = master_table.csv qlike_raw, every forecast: 8.88e-16 ≤ 1e-09 on n = 94 — PASS
- mean(qR) = mean|R| - 2 mean(EL), every forecast: 2.08e-16 ≤ 1e-12 on n = 94 — PASS
- QLIKE = integral ES_c c^-2 dc, day by day, every forecast: 5.7e-08 ≤ 1e-06 on n = 81404 — PASS
- always short collects +1 on pin days: 0 ≤ 1e-09 on n = 193 — PASS
- scores_by_forecast.csv = the committed copy (HEAD = 0be871d), 1 rows that read no forecast: 0 ≤ 1e-09 on n = 11 — PASS
- splits.csv = the committed copy (HEAD = 0be871d), 7 rows that read no forecast: 6.94e-17 ≤ 1e-09 on n = 56 — PASS
