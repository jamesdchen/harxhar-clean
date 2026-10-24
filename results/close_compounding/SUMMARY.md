# Compounding at a fixed fraction of wealth across the paper's model set (plain forecast f²·B)

Written by `experiments/close_compounding_headline.py` (`main`, 2026-10-06: no headline model, user decision) from `fixedfrac_modelset.csv`, `fixedfrac_modelset_summary.csv`, `peryear_growth_modelset.csv` and `gates.csv`. Definition: the notebook's section 15 (f = 0.03 of wealth as straddle premium every day, W_T = Π(1 + f q_t R_t), g = 252·mean log(1 + f R'), the drawdown as a fraction of the running peak, the ruin bound 1/|min R'|). Input: the master table's daily returns, on the 614 post-warmup days t > 252 of the 866 option-data days (2026-10-21; was all 866). Model set: 94 forecasts (results/close_master_table/paper_model_set.csv).

| group | fill | above 1 | median W_T | range | median g | median DD | largest DD | smallest ruin bound |
|---|---|---|---|---|---|---|---|---|
| paper | mid | 9 of 9 | 5.36 | 1.69 to 8.04 | +0.689 | 26 % | 44 % | 0.10 |
| paper | crossed | 9 of 9 | 3.30 | 1.01 to 4.93 | +0.491 | 31 % | 49 % | 0.09 |
| pooled linear | mid | 6 of 6 | 3.72 | 1.90 to 8.58 | +0.491 | 39 % | 48 % | 0.18 |
| pooled linear | crossed | 6 of 6 | 2.29 | 1.17 to 5.30 | +0.291 | 43 % | 53 % | 0.18 |
| per-bar linear | mid | 7 of 7 | 6.00 | 5.06 to 8.77 | +0.735 | 28 % | 28 % | 0.18 |
| per-bar linear | crossed | 7 of 7 | 3.70 | 3.12 to 5.42 | +0.537 | 30 % | 36 % | 0.18 |
| per-bar trees | mid | 56 of 72 | 1.91 | 0.59 to 8.06 | +0.264 | 44 % | 59 % | 0.10 |
| per-bar trees | crossed | 38 of 72 | 1.17 | 0.36 to 4.97 | +0.062 | 50 % | 68 % | 0.09 |
| model set | mid | 78 of 94 | 2.22 | 0.59 to 8.77 | +0.327 | 40 % | 59 % | 0.10 |
| model set | crossed | 60 of 94 | 1.35 | 0.36 to 5.42 | +0.122 | 45 % | 68 % | 0.09 |
| always short | mid | 1 of 1 | 1.07 | 1.07 to 1.07 | +0.029 | 52 % | 52 % | 0.10 |
| always short | crossed | 0 of 1 | 0.64 | 0.64 to 0.64 | -0.185 | 60 % | 60 % | 0.09 |

- Largest terminal wealth at the midpoint: per-bar elastic net [all_features], pre-tuned (8.77); smallest: Optuna per-bar random forest, TUNE_PER 25, best of 25 [baseline] (0.59).
- Per year, forecasts with positive log-growth (mid): 2021 33, 2022 94, 2023 82, 2024 87 of 94; median g: 2021 -1.24, 2022 +0.54, 2023 +0.38, 2024 +0.65.

## Gates
- Sharpe mid = master_table.csv, every forecast: 2.22e-16 ≤ 1e-09 on n = 94 — PASS
- Sharpe crossed = master_table.csv, every forecast: 3.33e-15 ≤ 1e-09 on n = 94 — PASS
- ret_mid = q R, every forecast: 0 ≤ 1e-12 on n = 94 — PASS
- q = sign(plain - iv_var), every forecast: 0 ≤ 0 on n = 94 — PASS
- the ruin bound exceeds f = 0.03 for every forecast and fill (no wealth factor <= 0): 0 ≤ 0 on n = 190 — PASS
- fixedfrac_headline.csv = the committed copy (HEAD = 0be871d), always short's rows (no forecast), 866-day recomputation: 7.63e-17 ≤ 1e-09 on n = 14 — PASS
- peryear_growth.csv = the committed copy (HEAD = 0be871d), always short's rows (no forecast), 866-day recomputation: 5.55e-17 ≤ 1e-09 on n = 20 — PASS
