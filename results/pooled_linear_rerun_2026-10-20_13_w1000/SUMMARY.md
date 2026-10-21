# Pooled linear models, 13 regular-hours intervals, 1,000-day window (2026-10-20)

Branch `insurance-selling-2026-09-19-w1000-check`, from `insurance-selling-2026-09-19` at eb109e3. This is the
committed 13-bar run (`results/pooled_linear_rerun_2026-10-19_13/`, called "the 2,000-day run" below) with one change:
the rolling window is 1,000 trading days instead of 2,000. Nothing under `writeup/` was touched, and no Table 1
generator was edited.

## Configuration

Everything is as in `results/pooled_linear_rerun_2026-10-19_13/SUMMARY.md` except the window:

- **Window.** 1,000 trading days, i.e. 13,000 rows on the 13-interval grid, for every model (the OLS included) and
  for the robust standardization.
- **Pre-tune.** 4 folds × 250 sessions tile the 1,000 sessions before 2020-01-03, each fit on the 13,000 rows before
  a 25-session (325-row) embargo. Validation spans:

  | fold | first | last |
  |---|---|---|
  | 0 | 2016-01-19 13:00 | 2017-01-11 14:30 |
  | 1 | 2017-01-11 15:00 | 2018-01-08 13:00 |
  | 2 | 2018-01-08 13:30 | 2019-01-07 11:00 |
  | 3 | 2019-01-07 11:30 | 2020-01-02 16:00 |

  The validation spans are the same rows as in the 2,000-day run; only the fit windows are shorter.
- **Rolling-window exercise.** The walk starts at the same 250-solve boundary (2019-12-04 16:00) and ends at
  2024-04-30 16:00. Lasso and elastic net on the full set ran in 2 chunks each, with a 13,000-row halo (one window).
- Runner configuration (`runner_cfg` in `frozen_penalties.json`):

  ```
  --sample rth13 --har-base 5 --ex-flag DROP_ACTIVITY_MA=1 --ex-flag CALENDAR_THROUGH_OVERNIGHT=1 --ex-flag DROP_OVERNIGHT_INDICATOR=1 --ex-flag LEAN_GLOBAL_LAGS=1 --window-days 1000
  ```

- Designs: HAR benchmark 12 columns, full set 576 columns, as in the 2,000-day run. `check_design` passed before the
  design, pre-tune and walk stages: the HAR columns are exactly `har_ma_{1,5,25,125,625,3125}`, and every FOMC series
  is present. The identifiability mask zeroes 234–240 columns of the full set in the pre-tune fit windows (224–230 at
  2,000 days).

## Frozen penalties (`frozen_penalties.json`)

| model | HAR benchmark | 2,000-day run | full set | 2,000-day run |
|---|---|---|---|---|
| ridge | 1e-2 (**bottom of grid**) | 1e-2 | 1e3 (**top of grid**) | 1e3 |
| lasso | 1e-3 | 1e-6 | 1e-3 | 1e-3 |
| elastic net | 1e-3 | 1e-6 | 1e-2 (**top of grid**) | 1e-2 |
| OLS | — | — | | |

Grids: ridge 1e-2..1e3, lasso and elastic net 1e-6..1e-2 (l1_ratio 0.5). Mean validation MSE (adjusted scale, all 13
intervals, `pretune_mse.csv`):

- ridge, HAR benchmark: 0.075958 over 1e-2..1, so the criterion is flat at the bottom edge.
- lasso and elastic net, HAR benchmark: 0.075958 at 1e-6, 0.075952 (lasso) and 0.075946 (elastic net) at 1e-3, and
  0.0766 / 0.0763 at 1e-2.
- ridge, full set: 0.0778 at 1e2 and 0.0711 at 1e3, still falling at the top edge.
- elastic net, full set: 0.0711 at 1e-3 and 0.0701 at 1e-2, still falling at the top edge.
- lasso, full set: 0.0707 at 1e-3 and 0.0709 at 1e-2.

Penalty-weight batch solves in the walks: none failed the KKT certification. Lasso on the full set needed 62 active-set
repairs, elastic net 40 active-set repairs and 1 coordinate-descent polish.

## Checks

- `frozen_penalties.json`: `window_days` 1000, `window_rows` 13000.
- The last pre-tune validation span ends on 2020-01-02 16:00.
- `forecasts.parquet`: 7 models × 14,497 rows = 101,479 rows. For each model, the rows cover every interval from
  2019-12-04 16:00 to 2024-04-30 16:00, with no duplicates, and the dates are the same set as in the 2,000-day run.
  `pred_adj`, `plain` and `pred_raw_duan` are finite.
- On the 16:00 rows of the common span, ŷ²·B / RV lies in [0.091, 3.85] for every model; it was [0.102, 3.83] in the
  2,000-day run.
- The chunked walks were not checked against a single walk across the chunk boundary in this run. The 2,000-day run
  made that check (bit for bit, 500 rows), and the chunk logic is unchanged.

## Results: the closing half hour, common span (1,082 days)

`scores.csv`. The forecast is ŷ²·B at the 16:00 row; the truth is the actual RV of the closing half hour; the span is
2020-01-03 to 2024-04-30, early closes dropped.

| model | set | λ | QLIKE | MSE ×1e10 | R²_OOS % |
|---|---|---|---|---|---|
| OLS | HAR benchmark | — | 0.12045 | 3.925 | 79.40 |
| ridge | HAR benchmark | 1e-2 | 0.12045 | 3.925 | 79.40 |
| lasso | HAR benchmark | 1e-3 | 0.11902 | 3.896 | 79.56 |
| elastic net | HAR benchmark | 1e-3 | 0.11965 | 3.911 | 79.48 |
| ridge | full set | 1e3 | 0.12362 | 6.253 | 67.19 |
| lasso | full set | 1e-3 | 0.12292 | 6.807 | 64.28 |
| elastic net | full set | 1e-2 | 0.11940 | 6.485 | 65.97 |

The printed columns of tab:rh_oos are unchanged from the 2,000-day run's `scores.csv` (a0: QLIKE 0.12912, MSE 4.820,
R²_OOS 74.71).

**Diebold–Mariano tests against the 1,000-day pooled OLS** (`dm.csv`, HLN, Newey–West lag 6, T = 1,082; dm < 0
means the model has the lower loss):

| model | set | DM QLIKE | p | mean QLIKE diff | DM sq. error | p |
|---|---|---|---|---|---|---|
| ridge | HAR benchmark | −3.10 | 0.002 | −4.7e-8 | 0.50 | 0.61 |
| lasso | HAR benchmark | −5.51 | <0.001 | −1.4e-3 | −1.31 | 0.19 |
| elastic net | HAR benchmark | −5.74 | <0.001 | −8.1e-4 | −1.19 | 0.24 |
| ridge | full set | 0.61 | 0.54 | 3.2e-3 | 1.34 | 0.18 |
| lasso | full set | 0.54 | 0.59 | 2.5e-3 | 1.29 | 0.20 |
| elastic net | full set | −0.31 | 0.76 | −1.1e-3 | 1.28 | 0.20 |

As in the 2,000-day run, the tests against the printed rows and the closing-interval OLS are not computed: their
forecast files are not in the repo, and `dm.csv` records this.

**Every interval of the sample (secondary)** (`scores_all_bars.csv`, 1,082 days × 13 = 14,066 intervals):

| model | set | QLIKE | MSE ×1e10 |
|---|---|---|---|
| OLS | HAR benchmark | 0.19511 | 2.2067 |
| ridge | HAR benchmark | 0.19511 | 2.2067 |
| lasso | HAR benchmark | 0.19549 | 2.2045 |
| elastic net | HAR benchmark | 0.19531 | 2.2052 |
| ridge | full set | 0.18689 | 3.8949 |
| lasso | full set | 0.18742 | 4.0653 |
| elastic net | full set | 0.18738 | 3.5639 |

## Side by side with the 2,000-day run

**Closing half hour** (`scores.csv`; 1,000 = this run, 2,000 = `results/pooled_linear_rerun_2026-10-19_13/`):

| model | set | λ 1,000 | λ 2,000 | QLIKE 1,000 | QLIKE 2,000 | MSE ×1e10 1,000 | MSE ×1e10 2,000 | R²_OOS % 1,000 | R²_OOS % 2,000 |
|---|---|---|---|---|---|---|---|---|---|
| OLS | HAR benchmark | — | — | 0.12045 | 0.11524 | 3.925 | 4.153 | 79.40 | 78.21 |
| ridge | HAR benchmark | 1e-2 | 1e-2 | 0.12045 | 0.11524 | 3.925 | 4.153 | 79.40 | 78.21 |
| lasso | HAR benchmark | 1e-3 | 1e-6 | 0.11902 | 0.11524 | 3.896 | 4.153 | 79.56 | 78.21 |
| elastic net | HAR benchmark | 1e-3 | 1e-6 | 0.11965 | 0.11524 | 3.911 | 4.153 | 79.48 | 78.21 |
| ridge | full set | 1e3 | 1e3 | 0.12362 | 0.12301 | 6.253 | 6.926 | 67.19 | 63.66 |
| lasso | full set | 1e-3 | 1e-3 | 0.12292 | 0.12122 | 6.807 | 7.821 | 64.28 | 58.96 |
| elastic net | full set | 1e-2 | 1e-2 | 0.11940 | 0.11476 | 6.485 | 7.151 | 65.97 | 62.48 |

**Every interval** (`scores_all_bars.csv`):

| model | set | QLIKE 1,000 | QLIKE 2,000 | MSE ×1e10 1,000 | MSE ×1e10 2,000 |
|---|---|---|---|---|---|
| OLS | HAR benchmark | 0.19511 | 0.19263 | 2.2067 | 2.2087 |
| ridge | HAR benchmark | 0.19511 | 0.19263 | 2.2067 | 2.2087 |
| lasso | HAR benchmark | 0.19549 | 0.19263 | 2.2045 | 2.2087 |
| elastic net | HAR benchmark | 0.19531 | 0.19263 | 2.2052 | 2.2087 |
| ridge | full set | 0.18689 | 0.18601 | 3.8949 | 4.3848 |
| lasso | full set | 0.18742 | 0.18665 | 4.0653 | 4.2816 |
| elastic net | full set | 0.18738 | 0.18592 | 3.5639 | 3.7765 |

**Two splits of the closing-half-hour errors** (16:00 rows of the common span, computed from the two
`forecasts.parquet` files; not part of the score stage's output):

| model | set | share of sq. error on 2020-03-13, 1,000 | 2,000 | QLIKE from 2020-07-01, 1,000 | 2,000 |
|---|---|---|---|---|---|
| OLS | HAR benchmark | 0.513 | 0.515 | 0.12078 | 0.11460 |
| ridge | HAR benchmark | 0.513 | 0.515 | 0.12078 | 0.11460 |
| lasso | HAR benchmark | 0.512 | 0.514 | 0.11925 | 0.11460 |
| elastic net | HAR benchmark | 0.512 | 0.514 | 0.11992 | 0.11460 |
| ridge | full set | 0.458 | 0.400 | 0.11983 | 0.11887 |
| lasso | full set | 0.417 | 0.346 | 0.12056 | 0.11908 |
| elastic net | full set | 0.411 | 0.346 | 0.11772 | 0.11354 |

## Files

- `frozen_penalties.json`, `pretune_mse.csv`, `scores.csv`, `dm.csv`, `scores_all_bars.csv`: as in the 2,000-day run.
- `forecasts.parquet`: every interval of every model, 7 × 14,497 rows, 2019-12-04 16:00 to 2024-04-30 16:00, with the
  same columns as in the 2,000-day run.
- `_work/` (gitignored): design caches, pre-tune folds, walk tables and logs.

## Timings (this container: 4 CPUs, 15 GB, Python 3.12.3)

| step | time |
|---|---|
| design | 268 s (full set 267 s, HAR benchmark 33 s, 2 jobs) |
| pre-tune | 34 s (full set: ridge 4 s, lasso 30 s, elastic net 32 s; HAR benchmark ≤ 1 s) |
| select | < 1 s |
| walks | 138 s (full set: ridge 69 s; lasso 121 + 118 s; elastic net 70 + 67 s; HAR benchmark and OLS 1–4 s) |
| score | 2 s |
| end to end | about 7.5 min wall |

**Library versions.** Python 3.12.3 venv: numpy 1.26.4 (OpenBLAS), pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1,
numba 0.68.0, pyarrow 19.0.1. `src/backtest/multi_stage.py` also imports `tqdm`, and `requirements.txt` lists `pyyaml`.
Neither is named in the 2,000-day run's SUMMARY.md, so both were added to the venv here: tqdm 4.70.1, PyYAML 6.0.3.

## Reproduce

```
OUT=results/pooled_linear_rerun_2026-10-20_13_w1000
python experiments/pooled_linear_rerun_2026_10_19_13.py design  --jobs 2 --window-days 1000 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py pretune --jobs 4 --window-days 1000 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py select  --window-days 1000 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py walk --chunks all_features/lasso=2 --chunks all_features/enet=2 --jobs 4 --window-days 1000 --out $OUT
python experiments/pooled_linear_rerun_2026_10_19_13.py score --window-days 1000 --out $OUT
```

No code was changed for this run.
