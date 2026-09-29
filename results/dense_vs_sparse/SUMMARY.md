# Trees vs linear at the close: is the 16:00 signal weak-and-dense or sparse? (checklist C2)

Written by `experiments/dense_vs_sparse_1530.py` (stages `capture`, `refit`, `analyze`, which writes this file); every number below is read from the CSVs and `numbers.json` of this folder, and the claims the verdict rests on are asserted against those numbers before this file is written.

**What is compared.** The per-bar forecast of the bar ending 16:00 (issued at 15:30; the forecast the last-30-min trade uses) on the per-bar arms' own design, target, 2000-session rolling window and refit cadence: per-bar ridge, lasso and elastic net refit every session with the penalty re-chosen every 250 sessions (`specs/causal_tune_linear.py`, its class run read-only), LightGBM / XGBoost / random forest refit every 10 sessions (`specs/causal_tune_trees.py`, shipped configuration). Two designs: `live_feasible` (244 columns, 18 source series) and `all_features` (640 columns, 49 series); 1,469 forecast sessions (2018-06-25 .. 2024-04-30); the trade and QLIKE use the 866 trade days (2020-01-03 .. 2024-04-30).

**One scorer, the research scorer:** the 16:00 bar recalibrated alone, forecast = (ŷ² + s)·B with s the forecast's own trailing-250-session mean squared error, lagged one session (`score_linear_subsection_causal.causal_forecasts`); QLIKE against the per-bar spec's 16:00 target on the 866 trade days; the trade is the deck's 15:30 **sign(s)** rule (`trade_1530`): buy the **straddle** (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position) when the forecast exceeds the 15:30 implied variance, sell it otherwise, hold to the close. Intervals: 95 %, circular block bootstrap over the trade days (block 21 sessions, 2,000 draws, one set of draws for every model, so differences are paired). The per-bar ridge `live_feasible` scores QLIKE 0.1005 and Sharpe 1.90 mid / 1.44 crossed here, the numbers of the closing-strategy master table.

Series names are the design's column stems: `har_ma_*` = the target's own HAR ladder (realized variance, means of the last 1, 5, 25, 125, 625, 3125 bars), `adj_sumabsret_ma_*` = absolute returns, `adj_sumret4_ma_*` = 4th-power returns, `adj_sumpret2_ma_*` = upside squared returns, `adj_sumbipow_ma_*` = bipower variation, `adj_sumvolume_ma_*` = ES volume, `adj_vix_ma_*` / `adj_vvix_ma_*` / `adj_vix3m_ma_*` = VIX, VVIX, VIX3M, `*_ewstock` / `*_vwstock` = the same statistics on equal- / value-weighted constituent stocks.

## Answer in brief

1. **Ridge's forecast is the dense one.** Its effective number of inputs is 41 of 244 columns (`live_feasible`) and 124 of 640 (`all_features`), against 8–19 and 16–39 for the lasso, the elastic net and the three tree models; it needs 22 / 64 columns to reproduce 95 % of its forecast's variance, every other model at most 20. But 80 % of every model's forecast variance comes from at most 7 columns: the first-order signal is the same few inputs (`har_ma_1`, `har_ma_5` and the recent return-size columns) for all six models, and ridge's density sits in the last fifth.
2. **Where ridge's extra weight goes:** onto the group of recent return-size measures that move with `har_ma` (`adj_sumabsret`, `adj_sumret4`, `adj_sumpret2`, `adj_sumbipow` at `_ma_1`, `_ma_5`): 25 % of ridge's forecast variance vs 8 % of the lasso's and 4–7 % of the trees' (`live_feasible`), while the `har_ma_1`, `har_ma_5` group carries 31 % of ridge's vs 55 % of the lasso's and 39–50 % of the trees'.
3. **The forecastable signal is spread over many inputs, for every model.** Restricted to the top-k inputs of a causal screen, every model's QLIKE is significantly worse than with all columns at every k ≤ 4 on both designs (k ≤ 8 on `live_feasible`), and at every k ≤ 16 every model trades below its all-column Sharpe.
4. **The professor's hypothesis, tested directly (the same inputs for all three models, k = 1, 2, 4, 8, 16, 32, 64):**
   * *On the trade:* not testable at this sample size. With identical inputs the ridge-minus-lasso and ridge-minus-LightGBM Sharpe gaps have intervals that include zero at 27 of the 28 (k, design) pairs, and so does the change of the gap between k inputs and all columns (27 of 28). Even with all columns ridge's trade edge is inside the noise: +0.08 [-0.74, +0.93] over the lasso and +0.12 [-0.61, +0.78] over LightGBM (`live_feasible`); +0.22 [-0.31, +0.80] and +0.24 [-0.58, +0.99] (`all_features`).
   * *On QLIKE (the precise measure):* **half confirmed.** *Against the lasso*, as the hypothesis says: with 4–8 strong, collinear inputs the lasso's selection wins (ridge minus lasso +0.0100 [+0.0025, +0.0227] at k = 4, +0.0105 [+0.0023, +0.0243] at k = 8, `all_features`), and adding the long tail of weak inputs closes the gap (+0.0006 [-0.0055, +0.0070] with all 640 columns); the narrowing is significant at k = 4, 8, 16 on `all_features`, not on `live_feasible`. *Against the trees*, reversed: LightGBM is worst at the sparse end, not best; with 1 or 2 inputs ridge beats it (-0.0066 [-0.0118, -0.0009], -0.0060 [-0.0106, -0.0010]), with all columns they tie (+0.0023 [-0.0031, +0.0084] `live_feasible`, -0.0003 [-0.0082, +0.0067] `all_features`); the change is significant at k = 1, 2 on `live_feasible`.
5. **Collapsing each correlated group to its first principal component** costs both linear models forecast accuracy and does not move the ridge-minus-lasso trade gap beyond noise (change -0.18 [-0.85, +0.46] `live_feasible`, +0.20 [-0.78, +1.07] `all_features`); there is no significant ridge trade edge to explain in the first place.

**Verdict.** The data support "the signal is dense" (every model forecasts and trades better with many inputs) and "ridge uses it densely" (ridge spreads weight over correlated return-size measures that the lasso and the trees mostly leave to `har_ma`). They support "dense favours ridge over the lasso" on forecast accuracy, on the full design. They do **not** support "sparse favours trees": trees lose most when inputs are few. None of it is visible in the trade's Sharpe, whose model-to-model differences (intervals 0.8 to 1.8 wide) are larger than every effect measured here; the trade ranking ridge ≥ lasso ≥ trees on the 866 days is not a significant ranking.

## 1. Signal density: how many inputs does each forecast use?

For each model, the contribution of every design column to every forecast: linear β_j (x_j − window mean_j) (linear SHAP; sums to the forecast minus the window-mean forecast, max relative gap 2.4e-14); trees: the TreeSHAP values the tree campaign stored (additivity ≤ 1.8e-06 relative). Definitions (all 1,469 forecasts; the trade-day values are in `density.csv` and differ little):

* **share_j** = mean |contribution_j| / sum over columns; **N_eff = 1 / Σ share²** (= p if every input carries the same, 1 if one input carries all);
* **k_x** = the fewest columns, taken in share order, whose summed contributions reproduce x of the variance of the input-driven forecast (R² = 1 − var(F − F_k) / var(F)); **greedy k_95** = the same with columns added one at a time to maximize R².

| design | model | columns with any weight | N_eff (columns) | largest share | k_50 | k_80 | k_95 | greedy k_95 | N_eff (series) | N_eff (groups, \|corr\| > 0.8) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| live_feasible | ridge | 138 | **40.7** | 7.4 % | 2 | 6 | **22** | 15 | **7.1** | **13.8** |
|  | lasso | 86 | 10.8 | 23.8 % | 1 | 3 | 7 | 5 | 3.5 | 5.3 |
|  | elastic net | 52 | 7.7 | 28.6 % | 1 | 3 | 5 | 5 | 2.5 | 4.8 |
|  | LightGBM | 150 | 19.1 | 15.2 % | 2 | 4 | 12 | 11 | 2.9 | 7.3 |
|  | XGBoost | 146 | 12.7 | 18.2 % | 2 | 3 | 10 | 9 | 2.3 | 5.1 |
|  | random forest | 174 | 13.2 | 16.1 % | 2 | 3 | 6 | 6 | 2.2 | 5.9 |
| all_features | ridge | 379 | **124.2** | 3.3 % | 3 | 7 | **64** | 25 | **21.5** | **38.1** |
|  | lasso | 204 | 21.3 | 14.4 % | 1 | 3 | 6 | 6 | 6.2 | 9.2 |
|  | elastic net | 238 | 39.4 | 9.5 % | 2 | 3 | 20 | 7 | 9.8 | 14.7 |
|  | LightGBM | 411 | 25.8 | 13.5 % | 2 | 4 | 16 | 14 | 4.1 | 9.7 |
|  | XGBoost | 392 | 16.4 | 17.9 % | 2 | 4 | 15 | 12 | 3.0 | 6.5 |
|  | random forest | 566 | 16.4 | 15.4 % | 2 | 3 | 9 | 8 | 2.8 | 8.5 |

A series is one source (all its HAR lags and its is-present / is-nonzero flags); a group is a set of columns whose every pairwise |correlation| over the forecast rows exceeds 0.8 (complete linkage; 64 groups of the 143 identifiable `live_feasible` columns, 139 of 361 for `all_features`). The lasso keeps a median of 16 (`live_feasible`, range 12–56) and 23 (`all_features`, 18–124) non-zero weights per refit; ridge keeps every identifiable column. k_80 across the six 250-session blocks: 3–13 for ridge, 2–9 for every other model. Figure: `density_curves.png`.

Share of each model's forecast variance carried by a group (the group's covariance with the forecast / the forecast's variance; the four largest groups by ridge's share, `groups.csv`):

| design | group (design columns) | ridge | lasso | LightGBM | XGBoost | random forest |
|---|---|---:|---:|---:|---:|---:|
| live_feasible | `har_ma_1`, `har_ma_5` | 30.8 % | 55.1 % | 46.1 % | 50.2 % | 39.3 % |
|  | `adj_sumabsret`, `adj_sumret4`, `adj_sumpret2`, `adj_sumbipow` at `_ma_1`, `_ma_5` | 25.0 % | 8.3 % | 4.0 % | 6.6 % | 6.1 % |
|  | `har_ma_25`, `har_ma_125` | 17.1 % | 21.4 % | 6.7 % | 7.9 % | 2.1 % |
|  | `adj_sumvolume` at `_ma_1`, `_ma_5` | 6.6 % | 4.3 % | 4.8 % | 4.8 % | 2.1 % |
| all_features | `har_ma_1`, `har_ma_5` | 28.3 % | 50.8 % | 48.0 % | 48.1 % | 34.7 % |
|  | `adj_sumabsret_ewstock`, `adj_sumabsret_vwstock` at `_ma_1`, `_ma_5` | 21.0 % | 16.1 % | 4.0 % | 3.8 % | 3.4 % |
|  | `har_ma_25`, `har_ma_125` | 15.1 % | 19.6 % | 6.9 % | 6.9 % | 1.7 % |
|  | `adj_sumabsret`, `adj_sumret4`, `adj_sumpret2`, `adj_sumbipow` at `_ma_1`, `_ma_5` | 14.2 % | 1.4 % | 3.4 % | 4.9 % | 5.0 % |

## 2. The sparsity sweep: ridge, lasso and LightGBM on the same top-k inputs

**Rule (causal):** at each of the six penalty re-choices (every 250 sessions; first forecasts 2018-06-25, 2019-06-26, 2020-06-23, 2021-06-21, 2022-06-17, 2023-06-16) the k columns with the largest |correlation with the target| over the 2000 sessions before the block's first forecast are kept for that block (constant or duplicated columns are never picked); the same columns go to all three models, which then refit exactly as their specs do. The picks (`screen_picks.csv`): k = 1 is `har_ma_1` in every block; k = 2 is `har_ma_1`, `har_ma_5`; k = 4 is `har_ma_1`, `har_ma_5`, `adj_sumabsret_ma_1`, `adj_sumabsret_ma_5` or `har_ma_1`, `har_ma_5`, `har_ma_25`, `adj_sumabsret_ma_1`; k = 8 in the last block is `har_ma_1`, `har_ma_5`, `har_ma_25`, `har_ma_125`, `adj_sumabsret_ma_1`, `adj_sumabsret_ma_5`, `adj_sumabsret_ma_25`, `adj_sumvolume_ma_5` (`live_feasible`) and `har_ma_1`, `har_ma_5`, `har_ma_25`, `adj_sumabsret_ma_1`, `adj_sumabsret_ma_5`, `adj_sumabsret_ewstock_ma_1`, `adj_sumabsret_vwstock_ma_1`, `adj_sumabsret_vwstock_ma_5` (`all_features`). A second rule (each linear model's own top-k standardized weights, β × window sd, at the block start) is in `sweep.csv` (rule `own`).

QLIKE (866 trade days) and sign(s) Sharpe (mid) against k; * = the paired interval against the same model on all columns excludes zero (`sweep.csv` has the intervals and the crossed Sharpe; figure `sweep_qlike_sharpe_vs_k.png`):

| k | ridge QLIKE | lasso QLIKE | LightGBM QLIKE | ridge Sharpe | lasso Sharpe | LightGBM Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| **live_feasible** | | | | | | |
| 1 | 0.1218 * | 0.1217 * | 0.1284 * | 1.46 | 1.00 | 1.39 |
| 2 | 0.1136 * | 0.1113 * | 0.1197 * | 0.91 * | 0.70 * | 0.59 * |
| 4 | 0.1185 * | 0.1085 * | 0.1148 * | 0.94 * | 1.36 | 0.13 * |
| 8 | 0.1136 * | 0.1063 * | 0.1093 * | 1.16 * | 1.25 | 0.99 * |
| 16 | 0.1081 * | 0.1034 * | 0.1033 | 1.45 | 1.14 | 0.99 * |
| 32 | 0.1034 | 0.1018 * | 0.1047 * | 1.98 | 1.18 | 1.71 |
| 64 | 0.0988 | 0.0977 | 0.0986 | 2.06 | 2.19 | 2.17 |
| all (244) | 0.1005 | 0.0964 | 0.0983 | 1.90 | 1.83 | 1.78 |
| **all_features** | | | | | | |
| 1 | 0.1218 * | 0.1217 * | 0.1284 * | 1.46 | 1.00 | 1.39 |
| 2 | 0.1136 * | 0.1113 * | 0.1197 * | 0.91 | 0.70 | 0.59 * |
| 4 | 0.1185 * | 0.1085 * | 0.1148 * | 0.94 | 1.36 | 0.13 * |
| 8 | 0.1152 * | 0.1047 | 0.1087 * | 1.45 | 1.44 | 1.19 |
| 16 | 0.1159 * | 0.1063 * | 0.1075 * | 1.34 | 1.25 | 1.27 |
| 32 | 0.1073 | 0.1056 * | 0.1039 | 1.25 | 1.15 | 1.39 |
| 64 | 0.1020 | 0.1003 | 0.1042 | 1.44 | 1.14 | 2.01 |
| all (640) | 0.1004 | 0.0998 | 0.1007 | 1.66 | 1.45 | 1.43 |

(k ≤ 4 selects the same columns in both designs, hence the identical rows.)

**The hypothesis test** (`sweep_gaps.csv`, `sweep_gap_counts.csv`): for each pair of models on the same k inputs, the paired gap and the change of the gap between k and all columns (a difference in differences):

* ridge − lasso, QLIKE: lasso better at k = 4 and 8 (`live_feasible` +0.0100 [+0.0025, +0.0227] and +0.0073 [+0.0004, +0.0194]; `all_features` +0.0100 [+0.0025, +0.0227] and +0.0105 [+0.0023, +0.0243]); tied with all columns (+0.0041 [-0.0019, +0.0116] `live_feasible`, +0.0006 [-0.0055, +0.0070] `all_features`); on `all_features` the gap narrows significantly from k = 4 (-0.0094 [-0.0208, -0.0019]), 8 (-0.0098 [-0.0218, -0.0021]), 16 (-0.0090 [-0.0211, -0.0002]) to all columns.
* ridge − LightGBM, QLIKE: ridge better at k = 1, 2 (-0.0066 [-0.0118, -0.0009], -0.0060 [-0.0106, -0.0010]); tied with all columns; on `live_feasible` the gap closes significantly from k = 1 (+0.0089 [+0.0020, +0.0162]), 2 (+0.0083 [+0.0018, +0.0156]).
* lasso − LightGBM, QLIKE: lasso better at k = 1, 2, 4 in 6 of 6 (k, design) cells (e.g. -0.0084 [-0.0121, -0.0043] at k = 2); tied with all columns.
* Sharpe: of the 28 ridge-vs-lasso and ridge-vs-LightGBM gaps at k = 1 … 64, 1 interval excludes zero (`live_feasible` ridge - lasso at k = 32, +0.80 [+0.24, +1.40]); the change of the gap between k and all columns excludes zero in 1 of 28. Across all 42 Sharpe comparisons (the lasso − LightGBM pair included) 2 changes of gap exclude zero, about the rate expected by chance at 5 %.
* Ridge restricted to its own top 16 standardized weights is within noise of its all-column QLIKE on `live_feasible` (+0.0032 [-0.0029, +0.0083]); its top 8 are not (+0.0096 [+0.0036, +0.0170]).

## 3. Correlated-input groups: collapse each group to its first principal component

**Rule:** at each 250-session block, on the 2000 sessions before the block, the identifiable columns are grouped by complete linkage at |correlation| > 0.8; each group of two or more columns is replaced by its first principal component (standardized on the window, loadings from the window, held for the block); ridge and lasso then refit every session on the collapsed design as their spec does. The first component carries a median 94–96 % of a group's window variance (per block and design). The brief's "PCA per refit" is implemented per re-tune block (the spec's own reseed points), so the design only changes where the penalty is re-chosen.

| design | | QLIKE | Sharpe mid | vs the same model on the full design |
|---|---|---:|---:|---|
| live_feasible | ridge, collapsed | 0.1034 | 2.06 | QLIKE +0.0029 [-0.0025, +0.0098], Sharpe +0.16 [-0.35, +0.65] |
|  | lasso, collapsed | 0.1011 | 1.81 | QLIKE +0.0047 [+0.0006, +0.0095], Sharpe -0.02 [-0.63, +0.58] |
| all_features | ridge, collapsed | 0.1119 | 1.19 | QLIKE +0.0115 [+0.0032, +0.0210], Sharpe -0.48 [-1.26, +0.28] |
|  | lasso, collapsed | 0.1063 | 1.17 | QLIKE +0.0066 [+0.0005, +0.0130], Sharpe -0.28 [-1.17, +0.58] |

Ridge − lasso Sharpe gap: `live_feasible` +0.08 [-0.74, +0.93] on the full design, +0.25 [-0.42, +1.01] collapsed (change -0.18 [-0.85, +0.46]); QLIKE gap +0.0041 → +0.0023 (change +0.0018 [-0.0026, +0.0070]); `all_features` +0.22 [-0.31, +0.80] on the full design, +0.02 [-0.93, +1.07] collapsed (change +0.20 [-0.78, +1.07]); QLIKE gap +0.0006 → +0.0055 (change -0.0049 [-0.0137, +0.0024]). Averaging each correlated group into one input loses forecast information for both models and leaves the two estimators as far apart as before.

## Gates and caveats

* The design re-run through the spec's own executor call equals the stored research forecast (max relative difference 9.6e-12 `live_feasible`, 3.5e-11 `all_features`); the local full refits equal the stored per-bar ridge / lasso / elastic net forecasts to ≤ 3.3e-11 relative, except the `all_features` elastic net: up to 9.2e-03 relative on 242 rows of one block (forecast rows 1007–1249); the spec's own continuous run on this machine differs from the stored run by up to 8.6e-03 on 242 rows (the warm homotopy is float-path dependent in that block). The elastic net enters only the density table.
* LightGBM refit locally (1 thread) differs from the stored cluster run (max 5.6e-02 / 7.9e-02 relative; another platform and thread count), so the sweep compares LightGBM with its own local all-column refit (Sharpe 1.78 / 1.43 locally vs 1.69 / 1.35 stored); the density table uses the stored runs' TreeSHAP.
* The per-day trade reproduces `trade_1530` to 4e-15 for every run; near-zero adjusted-scale forecasts need no treatment (the recalibrated forecast is at least s·B).
* Sample: 866 trade days. Sharpe differences between variants carry intervals 0.8–1.8 wide, so the trade cannot separate the estimators at this size; QLIKE differences of about 0.005 are resolvable.
* The screen is one named rule (univariate |correlation| over the training window); its picks are dominated by the HAR ladder and absolute returns. A screen that decorrelates its picks would test a different notion of sparsity and was not run.
