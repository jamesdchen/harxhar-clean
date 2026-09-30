# Compounding at a fixed fraction of wealth: the headline forecast under the 16:00-bar recalibration

Written by `experiments/close_compounding_headline.py` from `fixedfrac_headline.csv`, `fixedfrac_all_forecasts.csv`, `peryear_growth.csv` and `gates.csv` in this folder. Definition: the notebook's section 15 (f = 0.03 of wealth deployed as straddle premium every day, W_T = Π(1 + f q_t R_t), g = 252·mean log(1 + f R'), the drawdown as a fraction of the running peak, the worst-day factor 1 + f·min R', the ruin bound 1/|min R'|). Input: the master table's per-day returns of every table-A forecast (`results/close_master_table/master_table_daily.parquet`; the headline's Sharpe gated against `master_table.csv`). The parked exercise (Appendix D.15) is the paper's block-diagonal ridge under the deck's session-bar recalibration and is not set beside these numbers.

## The headline, the reference scored this way, always short

| series | fill | W_T | g (ann. log-growth) | max drawdown | worst-day factor | ruin bound f |
|---|---|---|---|---|---|---|
| headline sign(s) | mid | 20.18 | +0.874 | 34 % | 0.837 | 0.18 |
| headline sign(s) | crossed | 8.55 | +0.624 | 38 % | 0.832 | 0.18 |
| reference sign(s) (this scorer) | mid | 3.89 | +0.395 | 42 % | 0.837 | 0.18 |
| reference sign(s) (this scorer) | crossed | 1.62 | +0.141 | 50 % | 0.832 | 0.18 |
| always short | mid | 0.86 | -0.043 | 55 % | 0.691 | 0.10 |
| always short | crossed | 0.33 | -0.319 | 74 % | 0.671 | 0.09 |

## Every table-A forecast (`fixedfrac_all_forecasts.csv`: 229 forecasts; counted over the master table's rank set of 220, `in_rank_set`: check rows, exact duplicates and always short left out)
- Midpoint: 218 of 220 sign(s) portfolios end above 1 (terminal wealth 0.58 to 20.42); the headline's 20.18 ranks 2.
- Crossed: 208 of 220 end above 1 (0.24 to 8.63); the headline's 8.55 ranks 2.

## Per calendar year, headline sign(s) (annualized from the days traded that year; `peryear_growth.csv`)

| year | days | g mid | g crossed |
|---|---|---|---|
| 2020 | 158 | +0.869 | +0.442 |
| 2021 | 158 | +0.791 | +0.479 |
| 2022 | 219 | -0.010 | -0.191 |
| 2023 | 248 | +1.285 | +1.107 |
| 2024 | 83 | +2.149 | +1.956 |

## Gates
- headline Sharpe mid = master_table.csv: 0 ≤ 1e-09 on n = 866 — PASS
- headline Sharpe crossed = master_table.csv: 0 ≤ 1e-09 on n = 866 — PASS
- headline ret_mid = q R: 0 ≤ 1e-12 on n = 866 — PASS
- fixedfrac_headline.csv = the committed copy (HEAD = e4b83a0), every row, max |new - old| / max(1, |old|): 0 ≤ 1e-09 on n = 42 — PASS
- peryear_growth.csv = the committed copy (HEAD = e4b83a0), every row, max |new - old| / max(1, |old|): 0 ≤ 1e-09 on n = 60 — PASS
- fixedfrac_all_forecasts.csv = the committed copy (HEAD = e4b83a0), the headline's and the reference's rows, max |new - old| / max(1, |old|): 0 ≤ 1e-09 on n = 32 — PASS
