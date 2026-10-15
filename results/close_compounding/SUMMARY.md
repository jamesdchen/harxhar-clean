# Compounding at a fixed fraction of wealth across the paper's model set (plain forecast f²·B)

Written by `experiments/close_compounding_headline.py` (`main`, 2026-10-06: no headline model, user decision) from `fixedfrac_modelset.csv`, `fixedfrac_modelset_summary.csv`, `peryear_growth_modelset.csv` and `gates.csv`. Definition: the notebook's section 15 (f = 0.03 of wealth as straddle premium every day, W_T = Π(1 + f q_t R_t), g = 252·mean log(1 + f R'), the drawdown as a fraction of the running peak, the ruin bound 1/|min R'|). Input: the master table's per-day returns. Model set: 112 forecasts (results/close_master_table/paper_model_set.csv).

| group | fill | above 1 | median W_T | range | median g | median DD | largest DD | smallest ruin bound |
|---|---|---|---|---|---|---|---|---|
| paper | mid | 9 of 9 | 6.48 | 1.32 to 9.64 | +0.544 | 41 % | 64 % | 0.10 |
| paper | crossed | 8 of 9 | 2.67 | 0.53 to 3.91 | +0.286 | 51 % | 78 % | 0.09 |
| pooled linear | mid | 6 of 6 | 4.43 | 2.20 to 9.60 | +0.423 | 45 % | 48 % | 0.15 |
| pooled linear | crossed | 5 of 6 | 1.81 | 0.90 to 3.94 | +0.164 | 52 % | 59 % | 0.14 |
| per-bar linear | mid | 7 of 7 | 7.67 | 7.67 to 20.62 | +0.593 | 39 % | 39 % | 0.15 |
| per-bar linear | crossed | 7 of 7 | 3.16 | 3.16 to 8.54 | +0.335 | 55 % | 55 % | 0.14 |
| per-bar trees | mid | 90 of 90 | 3.56 | 1.02 to 17.02 | +0.370 | 45 % | 64 % | 0.10 |
| per-bar trees | crossed | 62 of 90 | 1.47 | 0.41 to 7.12 | +0.113 | 51 % | 74 % | 0.09 |
| model set | mid | 112 of 112 | 4.38 | 1.02 to 20.62 | +0.430 | 43 % | 64 % | 0.10 |
| model set | crossed | 82 of 112 | 1.80 | 0.41 to 8.54 | +0.171 | 51 % | 78 % | 0.09 |
| always short | mid | 0 of 1 | 0.86 | 0.86 to 0.86 | -0.043 | 55 % | 55 % | 0.10 |
| always short | crossed | 0 of 1 | 0.33 | 0.33 to 0.33 | -0.319 | 74 % | 74 % | 0.09 |

- Largest terminal wealth at the midpoint: per-bar elastic net [all_features], pre-tuned (20.62); smallest: Optuna per-bar random forest, TUNE_PER 25, best of 10 [baseline] (1.02).
- Per year, forecasts with positive log-growth (mid): 2020 94, 2021 70, 2022 112, 2023 99, 2024 103 of 112; median g: 2020 +0.38, 2021 +0.24, 2022 +0.54, 2023 +0.37, 2024 +0.65.

## Gates
- Sharpe mid = master_table.csv, every forecast: 2.22e-16 ≤ 1e-09 on n = 112 — PASS
- Sharpe crossed = master_table.csv, every forecast: 3.33e-15 ≤ 1e-09 on n = 112 — PASS
- ret_mid = q R, every forecast: 0 ≤ 1e-12 on n = 112 — PASS
- q = sign(plain - iv_var), every forecast: 0 ≤ 0 on n = 112 — PASS
- the ruin bound exceeds f = 0.03 for every forecast and fill (no wealth factor <= 0): 0 ≤ 0 on n = 226 — PASS
- fixedfrac_headline.csv = the committed copy (HEAD = f452978), always short's rows (no forecast): 7.63e-17 ≤ 1e-09 on n = 14 — PASS
- peryear_growth.csv = the committed copy (HEAD = f452978), always short's rows (no forecast): 5.55e-17 ≤ 1e-09 on n = 20 — PASS
