# Threshold score, QLIKE and sample splits across the paper's model set (plain forecast f²·B)

Written by `experiments/close_murphy_splits_headline.py` (`main`, 2026-10-06: no headline model, user decision) from `scores_modelset.csv`, `murphy_grid_modelset.csv`, `murphy_leader_by_threshold_modelset.csv`, `rank_vs_sharpe_modelset.csv`, `splits_modelset.csv`, `splits_summary_modelset.csv`, `sign_split_modelset.csv` and `gates.csv`. Input: the master table's per-day frame. Model set: 112 forecasts (results/close_master_table/paper_model_set.csv). Every forecast is its plain f²·B; the deck's 15:30 sign(s) trade; 866 days 2020-01-03 .. 2024-04-30; point estimates only.

## Scores by group (medians)

| group | n | QLIKE | QLIKE range | ES(c=1) | EL | same side as RV % | same side as R % | Sharpe mid | Sharpe crossed |
|---|---|---|---|---|---|---|---|---|---|
| paper | 9 | 0.1092 | 0.1078 to 0.1270 | 0.0944 | 0.3705 | 70.1 | 57.3 | 1.28 | 0.80 |
| pooled linear | 6 | 0.1081 | 0.1040 to 0.1098 | 0.0941 | 0.3784 | 71.1 | 56.6 | 1.06 | 0.57 |
| per-bar linear | 7 | 0.0967 | 0.0931 to 0.0975 | 0.0995 | 0.3673 | 68.7 | 56.8 | 1.37 | 0.89 |
| per-bar trees | 90 | 0.1026 | 0.0957 to 0.1100 | 0.0995 | 0.3819 | 69.5 | 55.7 | 0.96 | 0.48 |
| model set | 112 | 0.1028 | 0.0931 to 0.1270 | 0.0991 | 0.3778 | 69.5 | 56.0 | 1.08 | 0.59 |

- Lowest QLIKE: per-bar lasso [all_features], pre-tuned; lowest score at the threshold: LightGBM.
- Murphy grid (33 thresholds, 0.25 to 4): thresholds led by group — paper 14, per-bar trees 9, per-bar linear 7, pooled linear 3; at c = 1 the leader is LightGBM (paper).
- Oracles: told RV, the payoff's side on 62.9 % of days (Sharpe 4.53 / 4.07); told the payoff's sign, 17.28 crossed.
- Spearman of minus the loss with the Sharpe ratio across the set (point): QLIKE vs crossed +0.20; QLIKE vs mid +0.20; ES at the threshold vs crossed +0.54; ES at the threshold vs mid +0.55; EL (affine in the mid P&L) vs crossed +1.00; EL (affine in the mid P&L) vs mid +1.00.

## Sample splits across the set (midpoint)

| split | days | Sharpe > 0 | median Sharpe | range | median t | t > 1.96 | always short |
|---|---|---|---|---|---|---|---|
| all days | 866 | 112 of 112 | 1.08 | 0.29 to 1.92 | 1.99 | 58 | 0.20 |
| through 2020-03-31 (the COVID quarter) | 39 | 61 of 112 | 0.31 | -3.96 to 3.67 | 0.12 | 0 | -3.25 |
| after 2020-03-31 | 827 | 112 of 112 | 1.13 | 0.21 to 2.20 | 2.06 | 59 | 0.42 |
| before 2022-05-16 | 377 | 110 of 112 | 0.87 | -0.51 to 2.12 | 1.07 | 13 | -0.45 |
| from 2022-05-16 (daily SPXW listings) | 489 | 112 of 112 | 1.14 | 0.22 to 2.34 | 1.59 | 40 | 0.73 |
| settlement pins (R = -1) | 193 | 112 of 112 | 7.54 | 4.34 to 14.63 | 6.60 | 112 | -- |
| off pins | 673 | 25 of 112 | -0.33 | -1.94 to 0.68 | -0.53 | 0 | -3.77 |

- Sign split (mean R on buy days minus sell days, HC0 t): median 0.166, positive for 112 of 112, median t 1.82, t > 1.96 for 52.

## Gates
- slice = the deck's iv_var: 0 ≤ 1e-09 on n = 866 — PASS
- realized variance (rv_unclipped) shared by every forecast: 0 ≤ 1e-09 on n = 112 — PASS
- q = the master table's, every forecast and day: 0 ≤ 0 on n = 96992 — PASS
- ret_mid = the master table's: 0 ≤ 1e-10 on n = 96992 — PASS
- ret_crossed = the master table's: 0 ≤ 1e-10 on n = 96992 — PASS
- Sharpe_mid = master_table.csv, every forecast: 2.22e-16 ≤ 1e-09 on n = 112 — PASS
- Sharpe_crossed = master_table.csv, every forecast: 3.33e-15 ≤ 1e-09 on n = 112 — PASS
- QLIKE (plain) = master_table.csv qlike_raw, every forecast: 8.88e-16 ≤ 1e-09 on n = 112 — PASS
- mean(qR) = mean|R| - 2 mean(EL), every forecast: 2.08e-16 ≤ 1e-12 on n = 112 — PASS
- QLIKE = integral ES_c c^-2 dc, day by day, every forecast: 5.7e-08 ≤ 1e-06 on n = 96992 — PASS
- always short collects +1 on pin days: 0 ≤ 1e-09 on n = 193 — PASS
- scores_by_forecast.csv = the committed copy (HEAD = 34385fd), 1 rows that read no forecast: 0 ≤ 1e-09 on n = 11 — PASS
- splits.csv = the committed copy (HEAD = 34385fd), 7 rows that read no forecast: 6.94e-17 ≤ 1e-09 on n = 56 — PASS
