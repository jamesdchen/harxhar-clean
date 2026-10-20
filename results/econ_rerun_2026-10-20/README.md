# Economics re-score on the pooled linear reruns (2026-10-20)

Branch `insurance-selling-2026-09-19-econ-rerun`, from `insurance-selling-2026-09-19` at 3fa819e. This directory holds the sign
strategy of Section 7 (`writeup/rehaul/07_economics.tex`), re-scored on the 16:00 forecasts of the two pooled linear reruns
behind the new Table 1. Nothing under `writeup/rehaul/`, `writeup/paper_rehaul.tex` or the existing `rehaul_econ_*.tex`
files was touched, and no existing generator was run.

## Inputs

### What the paper of record's economics chain reads, and what this checkout holds

| Input | Read by | In this checkout |
|---|---|---|
| `results/atm_straddle_0dte_1530/daily_a0.parquet` (the quote book: R, exit, call and put bids and asks at 15:30, iv_var; 866 option days) | `make_rehaul_economics.py` via `gui._QUOTE_BOOK` | **present** |
| `results/atm_straddle_0dte_1530/daily_blk2.parquet` (the master table's deck) | `experiments/master_table_close.py` | **present**; identical to daily_a0 in R, exit, bids, asks, iv_var, entry (checked) |
| `results/close_master_table/master_table.csv` | `make_rehaul_economics.py` (gates) | **present** |
| `results/close_master_table/master_table_daily.parquet` (daily q, ret_mid, ret_crossed, plain, iv_var, rv of every model) | `make_rehaul_economics.py`, `make_rehaul_results.py`, `experiments/sharpe_tests_pooled*.py` | **missing** |
| `results/close_master_table/paper_model_set.csv` | `make_rehaul_economics.py`, `sharpe_tests_pooled.py` | **missing** |
| `writeup/master_table_gui/` (`rows.py`: `_QUOTE_BOOK`, `_crossed_return`, sizing, `_nn_daily`) | `make_rehaul_economics.py` (imported) | **missing** (the directory is not in the repo) |
| `results/spxw_pnl/yhat_a0.parquet`, `yhat_poolpt_*.parquet`, `yhat_tree00/10.parquet` (the old pooled forecasts) | master table, `sharpe_tests_pooled.py` | **missing** |
| `results/pooled_trees_2026-10-07/` | `make_rehaul_economics.py`, `sharpe_tests_pooled.py` | **missing** |
| `data/spxw_chain.parquet` (raw SPXW chain that `notebooks/_write_0dte_nb.py` turns into `daily_<tag>.parquet`) | upstream of the quote book | **missing**, not needed: the quote book built from it is present |
| `results/pooled_linear_rerun_2026-10-19_48/forecasts/*.parquet`, `results/pooled_linear_rerun_2026-10-19_13/forecasts.parquet` | this re-score | **present** |

So `make_rehaul_economics.py` itself cannot run here (missing master table daily frame, model set and the GUI
package). The re-score does not need any of them: the option data that price the straddle (the quote book) and the new
forecasts are both present, and the trade, cost and test rules are re-implemented from the paper's code (below) and
gated on the printed numbers. No `BLOCKED.md` is needed.

## Design

- **Forecasts.** The 16:00 row (naive ET interval end) of each rerun model: `plain` = pred_adj^2 x B, the 15:30
  forecast of the closing half hour. Panel A: trained on all 48 intervals; Panel B: trained on the 13 regular-hours
  intervals. OLS on the HAR benchmark; ridge, lasso, elastic net on the HAR benchmark and on all features. Every model
  covers all 866 option days.
- **Trade** (as `make_rehaul_economics.py` `load` / `add_pooled_trees`, `master_table_close.trade_days`):
  q = +1 if plain > iv_var, else -1; mid return q R; at quoted bid/ask prices a buy pays the ask, a sale receives the
  bid, spread at entry only. Always short: q = -1. No other costs.
- **Days.** Sharpe ratios on the 614 post-warmup option days (t > 252); mean, hit rate, buys and every HAC t on all
  866 option days.
- **Sharpe ratio.** `asl.sharpe_ann_by_year` (year-by-year annualization, N_y-weighted), the paper's ratio since 2026-10-18.
- **Tests.** (i) HAC t of the daily return differential (`asl.newey_west_t`, Bartlett, lag 14), mid and bid/ask, as
  every dSR of the paper's tables: a test of the mean differential. (ii) Ledoit and Wolf (2008) on the Sharpe
  difference, `experiments/sharpe_tests_pooled.py` unchanged (HAC delta-method p, studentized circular block bootstrap
  with b = 5 and b = 10, M = 4999, and the calibrated block), on 614 and 866 days; Lo (2002) SR = 0 for every series.
  Seed 20261037. LW works on the daily Sharpe ratio (fixed factor), so its `lw_dsr_ann` can differ from `dsh_mid`
  (year-by-year); the p-values do not depend on the annualization.
- **QLIKE** on the 866 option days against the forecast files' actual RV is in `levels.csv` / `pairs.csv` for
  context only (Table 1 scores the 1,082-day span).

## Gates (all pass; `gates.txt`)

Always short reproduces `\rheShortSh` 0.77, `\rheShortShX` 0.42, `\rheShortT` 0.43, `\rheShortHit` 61.8, the master
table's `always_short` row (Sharpe_mid, Sharpe_crossed, mean_mid, hit_rate_mid to 1e-8), and the always-short Lo row
of `results/sharpe_tests_pooled_2026-10-17/sharpe_tests_pooled.csv`. None of the rerun models can be gated against a
stored economics number: no earlier file traded them.

## Results

SR over 614 days; t = HAC t of the daily mid return differential (866 days); LW p = Ledoit-Wolf bootstrap p, b = 5,
614 days. Stars: 10/5/1 % of the HAC t.

### Levels and differences against the short straddle (SR 0.77 mid, 0.42 bid/ask; t of mean 0.43)

| Panel | Model | Set | SR mid | SR bid/ask | t mean | Buys % | dSR vs short | t | LW p |
|---|---|---|---|---|---|---|---|---|---|
| 48 | OLS | HAR | 1.04 | 0.70 | 2.25 | 23.9 | 0.27 | 1.41 | 0.337 |
| 48 | Ridge | HAR | 1.01 | 0.67 | 2.24 | 24.1 | 0.24 | 1.38 | 0.365 |
| 48 | Lasso | HAR | 1.01 | 0.67 | 2.35 | 24.2 | 0.24 | 1.47 | 0.351 |
| 48 | Elastic net | HAR | 0.90 | 0.56 | 1.94 | 25.5 | 0.13 | 1.15 | 0.440 |
| 48 | Ridge | Full | 1.10 | 0.76 | 1.99 | 21.9 | 0.33 | 1.54 | 0.322 |
| 48 | Lasso | Full | 1.48 | 1.13 | 2.91 | 19.9 | 0.71** | 2.14 | 0.065 |
| 48 | Elastic net | Full | 1.42 | 1.08 | 2.69 | 21.2 | 0.66* | 1.93 | 0.067 |
| 13 | OLS | HAR | 1.07 | 0.73 | 2.45 | 23.3 | 0.31 | 1.56 | 0.316 |
| 13 | Ridge / lasso / elastic net | HAR | identical to the OLS on every day | | | | | | |
| 13 | Ridge | Full | 1.49 | 1.14 | 3.40 | 18.9 | 0.73** | 2.45 | 0.048 |
| 13 | Lasso | Full | 1.23 | 0.89 | 2.77 | 18.7 | 0.47** | 2.07 | 0.089 |
| 13 | Elastic net | Full | 1.63 | 1.28 | 3.19 | 20.8 | 0.86** | 2.31 | 0.030 |

The bid/ask dSR against the short straddle is within 0.01 of the mid dSR in every row, with the same stars.

### Against the OLS of the same panel

| Panel | Model | Set | Days differing | dSR mid | t | LW p |
|---|---|---|---|---|---|---|
| 48 | Ridge | HAR | 2 | -0.03 | -0.42 | 0.774 |
| 48 | Lasso | HAR | 3 | -0.03 | 0.58 | 0.774 |
| 48 | Elastic net | HAR | 14 | -0.14 | -1.64 | 0.128 |
| 48 | Ridge | Full | 113 | 0.06 | -0.16 | 0.859 |
| 48 | Lasso | Full | 107 | 0.44 | 0.96 | 0.198 |
| 48 | Elastic net | Full | 113 | 0.38 | 0.72 | 0.197 |
| 13 | Ridge / lasso / elastic net | HAR | 0 | 0 | -- | -- |
| 13 | Ridge | Full | 122 | 0.42 | 1.33 | 0.158 |
| 13 | Lasso | Full | 122 | 0.16 | 0.48 | 0.355 |
| 13 | Elastic net | Full | 102 | 0.55 | 1.18 | 0.046 |

### Full predictor set minus HAR benchmark, same estimator

| Panel | Model | dSR mid | t | LW p |
|---|---|---|---|---|
| 48 | Ridge | 0.09 | -0.13 | 0.888 |
| 48 | Lasso | 0.47 | 0.89 | 0.166 |
| 48 | Elastic net | 0.52 | 1.05 | 0.117 |
| 13 | Ridge | 0.42 | 1.33 | 0.162 |
| 13 | Lasso | 0.16 | 0.48 | 0.337 |
| 13 | Elastic net | 0.55 | 1.18 | 0.047 |

### 48 intervals minus 13 intervals, same model

| Model | Set | Days differing | dSR mid | t | LW p |
|---|---|---|---|---|---|
| OLS | HAR | 33 | -0.04 | -0.40 | 1.000 |
| Ridge | HAR | 35 | -0.07 | -0.48 | 0.927 |
| Lasso | HAR | 34 | -0.07 | -0.18 | 0.926 |
| Elastic net | HAR | 45 | -0.17 | -1.23 | 0.540 |
| Ridge | Full | 66 | -0.39 | -1.43 | 0.226 |
| Lasso | Full | 56 | 0.24 | 0.49 | 0.602 |
| Elastic net | Full | 54 | -0.20 | -1.09 | 0.419 |

No mean differential between two rerun models is significant at 10 % by the HAC t. The rows differing from a
comparator on only 2 or 3 days (48-interval ridge and lasso on the HAR benchmark against the OLS) have a bootstrap with
draws whose standard error is zero; their LW p is not informative and their block calibration is undefined
(`block_calibration.csv`).

### Against the printed Section 7 numbers (old forecasts)

For orientation only (different forecasts): the printed full-set pooled linear SR are 1.56 (elastic net), 1.46
(lasso), 1.44 (ridge), and the printed pooled HAR-benchmark SR 0.74-0.75 (Table E2 / E3). The reruns give, on 48
intervals, 1.42 / 1.48 / 1.10 on all features and 0.90-1.04 on the HAR benchmark; on 13 intervals 1.63 / 1.23 / 1.49 and
1.07.

## Files

- `levels.csv`: one row a series (14 rerun models + always short): SR mid and bid/ask (614, year by year), fixed-factor
  SR (614 and 866), mean and HAC t (mid, bid/ask), hit and buys %, QLIKE (866), Lo z and p (614, 866).
- `pairs.csv`: comparisons `vs_short`, `vs_ols`, `all_vs_har`, `48_vs_13` (a minus b): days differing, dSR mid and
  bid/ask, mean differential and its HAC t (mid, bid/ask), QLIKE change %, LW dSR, z, p_hac, p_boot (b = 5),
  p_boot_b10, b_cal and its p (614), the same on 866 days.
- `block_calibration.csv`: LW block calibration coverage on 614 days.
- `daily.parquet`: key, day, q, mid, x (bid/ask) for every series.
- `gates.txt`, `run.log`.

## Reproduce

```
python experiments/econ_rerun_2026_10_20.py          # about 20 min with the block calibration; --no-calib about 40 s
python writeup/make_rehaul_econ_rerun.py             # writes writeup/generated/rehaul_econ_rerun_{numbers,levels,pairs}.tex
```

The generated tables (`tab:rh_econ_rerun_levels`, `tab:rh_econ_rerun_pairs`) and `\rher...` macros are not
`\input` anywhere; wiring them into Section 7 is the author's call.

## Open points

1. The sizing rules (Table E4), the closing-interval models (E1 Panel A, E3) and the trees are not re-scored: the
   closing-interval and tree forecasts are not in this checkout, and the sizing code lives in the missing
   `writeup/master_table_gui/rows.py`.
2. On 13 intervals the HAR-benchmark ridge, lasso and elastic net take the OLS's position on every day (their frozen
   penalties are at the grid floor), so their economic rows coincide with the OLS row.
3. A pooled-versus-closing-interval comparison like Table E3 needs the closing-interval daily returns
   (`master_table_daily.parquet`, keys `sub_*`), which are missing.
