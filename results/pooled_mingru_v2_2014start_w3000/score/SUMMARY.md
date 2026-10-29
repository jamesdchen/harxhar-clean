# MinGRU v2, W = 3,000: merge, seed average, score and comparison (2026-10-28)

**Status.** The HAR-set walk is FINAL. The full-set walk ran on the PRE-FOMC-FIX design caches (pmf_<set> exported
2026-10-22, before the FOMC patch D13). **Every full-set number below, and so every full-vs-HAR difference, gain share
and full-set Sharpe ratio, is a PRE-FOMC-FIX PREVIEW.** Nothing here is in the paper. It was computed on CPU only, from
the four arm workers' merged walks, without refitting anything.

## Inputs

| arm | walk | forecasts (compact, copied here) | checks |
|---|---|---|---|
| HAR (baseline) | 33,970 bars x 5 seeds (169,850 refits), P100 | forecasts_w3000_baseline.parquet | pass, fail = [] (checks_w3000_baseline.json) |
| full (all_features) [PRE-FOMC-FIX PREVIEW] | 33,970 bars x 5 seeds (169,850 refits), P100 | forecasts_w3000_all_features.parquet | pass, fail = [] (checks_w3000_all_features.json) |

- forecast = the 5-seed mean of f; plain = f^2 B (RULE_gru_v2.txt ADDENDUM 2). B matches Table 1's B to within
  9.5e-13 relative. B comes from the design export pmf_rows.csv.
- walk-combine (experiments/pooled_mingru_v2.py walk-combine, PMF_WINDOW_DAYS=3000) of the two arms' walk-merge outputs
  was run by the comparison worker. The result is `forecasts.parquet` (md5 f73fb32ad32a0bd549d6d5f34fab1b41) with
  `forecasts.json`, both here. It equals each arm's compact file exactly (checked).
- Seed-level forecasts are not copied (2.3 MB each). They are in the session scratchpad (`...\scratchpad\mgv2\`):
  `w3000_all_features\tmp\merge\forecasts_seeds_all_features.parquet` (md5 6038487176e23326b2bc13a5de116e16) and
  `w3000_baseline\out\merge\forecasts_seeds_baseline.parquet` (md5 b94e133db001215993a090251bfcb191).
- Files copied from the arms: checks_*.json, scores_*.csv, econ_*.csv / .json, daily_*.parquet (per-day losses),
  econ_daily_*.parquet, forecasts_w3000_*.parquet and DONE_<arm>.txt (the arms' md5 lists). The four arm workers use
  different column layouts and span names.

## Method (experiments/pooled_mingru_v2_compare.py; the main tree read-only)

- Scored set: Table 1's 33,527 bars on 2,579 days (2014-01-02 .. 2024-04-30), built as writeup/make_rehaul_channels.py
  builds it. It is the same in every context run, and the MinGRU forecasts every bar of it. QLIKE = r - ln r - 1,
  r = RV / plain, with RV from core_stats sumret2.
- Full vs HAR within the MinGRU (a = full, b = HAR): the % change of the mean loss, and the DM test (HLN) on the daily
  mean of the day's bar losses, Newey-West lag 8 on every span (the channels script's HAC_LAG). For the EVENT days the
  automatic lag 3 is shown as well. Spans: every bar; the 16:00 bar (2,579 days); the 16:00 bar on 2020-01-03 ..
  2024-04-30 (1,082 days, the span of Table 1's printed 16:00 column); ORDINARY days (2,498); EVENT days (the 81
  scheduled FOMC statement days).
- Gain share: Section 6.2's definition (make_rehaul_channels item 2): 100 x the sum over the EVENT-day bars ending
  14:30 .. 16:00 of (HAR loss - full loss), divided by the same sum over every bar.
- Benchmarks in the context rows: the OLS on the HAR set for ridge, lasso and elastic net (D2, as Section 6). The
  trees and the MinGRU use their own HAR-set fit.
- Context: Table 1's six classes, FOMC-patched (Table 1's runs = ablations scores_fomcfix) and pre-patch (the
  ablations' templates = scores/). The **pre-patch rows are the apples-to-apples comparison for the MinGRU full-set
  preview.**
- Economics: the sign strategy of Section 7 on the 866 option days. The Sharpe ratio is the paper's D19 ratio on the
  614 post-warmup days. Every difference is tested by Ledoit-Wolf (block 10, 4,999 resamples), with the paired
  block-bootstrap p alongside. The short straddle is make_rehaul_economics' always_short. The context economics are the
  pre-patch W = 3,000 rows of results/rehaul_robust_window_final/sharpe.csv, recomputed from the forecasts.
- Gates: 713 checks, 0 failed (one run for both windows; compare_gates.json). They reproduce, from the forecasts:
  - results/channels_2026-10-26: the item3ii joint full-vs-HAR rows on every bar / EVENT / ORDINARY, and the item2
    gain shares;
  - the ablation scorer's metrics.csv / dm.csv (patched and pre-patch);
  - Table 1's metrics.csv (every-bar QLIKE and qlike_1600);
  - results/rehaul_robust_window_final (dm_full_vs_har, metrics, and every full - HAR Sharpe row);
  - each MinGRU arm's own scores and econ files.

## Results

### MinGRU v2, W = 3,000: full set vs HAR set (PRE-FOMC-FIX PREVIEW for every full-set number)

Negative % / DM = the full set more accurate. DM: HLN, daily mean loss, NW lag 8 (EVENT: automatic lag in brackets).

| span | QLIKE full [PREVIEW] | QLIKE HAR | % QLIKE (DM, p) [PREVIEW] | % MSE (DM, p) [PREVIEW] |
|---|---|---|---|---|
| every bar | 0.16225 | 0.17998 | -9.85 (-10.39, <0.001) | +26.81 (+1.10, 0.271) |
| 16:00 bar | 0.11434 | 0.12046 | -5.08 (-2.37, 0.018) | +17.68 (+0.87, 0.387) |
| 16:00 bar 2020-24 (Table 1's column) | 0.10800 | 0.10909 | -1.00 (-0.23, 0.816) | +21.82 (+0.97, 0.334) |
| ORDINARY days | 0.15258 | 0.16278 | -6.27 (-7.88, <0.001) | +29.42 (+1.13, 0.258) |
| EVENT days (81) | 0.46068 | 0.71045 | -35.16 (-7.61, <0.001) [lag 3: -7.57, <0.001] | -10.98 (-1.71, 0.088) |

Gain shares [PRE-FOMC-FIX PREVIEW] (Section 6.2's definition; sum over bars of HAR loss - full loss = 594.20): EVENT days 44.3 %, EVENT afternoons (bars ending 14:30-16:00) 45.2 %, ORDINARY days 55.7 % (EVENT afternoons hold 0.97 % of the bars).

### Context, W = 3,000: full vs HAR, % QLIKE (DM, p); linear classes against the OLS on the HAR set (D2)

| model | design | every bar | 16:00 | 16:00 2020-24 | ORDINARY | EVENT | EVENT pm share % |
|---|---|---|---|---|---|---|---|
| Ridge | patched | -6.04 (-6.28, <0.001) | +1.78 (+0.67, 0.500) | +11.60 (+2.36, 0.018) | -3.24 (-3.51, <0.001) | -22.41 (-6.45, <0.001) | 55.5 |
| Lasso | patched | -8.97 (-6.34, <0.001) | +4.24 (+1.35, 0.178) | +15.17 (+2.58, 0.010) | -2.44 (-2.08, 0.037) | -47.19 (-7.56, <0.001) | 77.5 |
| Elastic net | patched | -6.69 (-7.36, <0.001) | +0.60 (+0.24, 0.807) | +9.50 (+2.01, 0.044) | -3.97 (-4.73, <0.001) | -22.61 (-6.79, <0.001) | 50.9 |
| LightGBM | patched | -14.11 (-10.26, <0.001) | -2.67 (-1.76, 0.079) | -1.36 (-0.48, 0.629) | -6.04 (-10.70, <0.001) | -62.28 (-8.62, <0.001) | 62.7 |
| XGBoost | patched | -14.60 (-9.94, <0.001) | -3.15 (-2.01, 0.044) | -1.69 (-0.63, 0.531) | -6.11 (-10.42, <0.001) | -63.92 (-7.75, <0.001) | 63.3 |
| Random forest | patched | -11.31 (-9.74, <0.001) | -1.59 (-1.03, 0.302) | -0.74 (-0.27, 0.787) | -5.31 (-11.16, <0.001) | -44.80 (-8.56, <0.001) | 58.9 |
| Ridge | pre-patch | -5.36 (-5.63, <0.001) | +0.80 (+0.31, 0.759) | +10.85 (+2.23, 0.026) | -3.17 (-3.26, 0.001) | -18.22 (-7.12, <0.001) | 54.4 |
| Lasso | pre-patch | -6.23 (-4.72, <0.001) | +3.28 (+1.04, 0.300) | +14.43 (+2.42, 0.016) | -1.96 (-1.51, 0.132) | -31.22 (-7.51, <0.001) | 76.4 |
| Elastic net | pre-patch | -6.04 (-6.98, <0.001) | -0.52 (-0.21, 0.832) | +8.53 (+1.84, 0.066) | -4.00 (-4.68, <0.001) | -17.95 (-7.61, <0.001) | 48.1 |
| LightGBM | pre-patch | -13.99 (-10.32, <0.001) | -2.73 (-1.81, 0.071) | -1.08 (-0.39, 0.700) | -6.07 (-10.70, <0.001) | -61.22 (-8.81, <0.001) | 62.7 |
| XGBoost | pre-patch | -13.80 (-9.94, <0.001) | -3.39 (-2.18, 0.030) | -2.07 (-0.76, 0.447) | -6.11 (-10.38, <0.001) | -58.46 (-7.23, <0.001) | 61.8 |
| Random forest | pre-patch | -11.07 (-9.85, <0.001) | -1.42 (-0.92, 0.357) | -0.71 (-0.26, 0.794) | -5.30 (-11.22, <0.001) | -43.32 (-9.17, <0.001) | 58.2 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | -9.85 (-10.39, <0.001) | -5.08 (-2.37, 0.018) | -1.00 (-0.23, 0.816) | -6.27 (-7.88, <0.001) | -35.16 (-7.61, <0.001) | 45.2 |

### QLIKE levels, W = 3,000 (every bar / 16:00)

| model | design | full every bar | full 16:00 | HAR every bar | HAR 16:00 |
|---|---|---|---|---|---|
| OLS | patched | - | - | 0.18927 | 0.12687 |
| Ridge | patched | 0.17783 | 0.12913 | 0.18927 | 0.12687 |
| Lasso | patched | 0.17230 | 0.13225 | 0.18927 | 0.12687 |
| Elastic net | patched | 0.17660 | 0.12764 | 0.18927 | 0.12687 |
| LightGBM | patched | 0.16106 | 0.12021 | 0.18751 | 0.12350 |
| XGBoost | patched | 0.16121 | 0.11954 | 0.18878 | 0.12342 |
| Random forest | patched | 0.16808 | 0.12459 | 0.18950 | 0.12660 |
| OLS | pre-patch | - | - | 0.18927 | 0.12687 |
| Ridge | pre-patch | 0.17912 | 0.12789 | 0.18927 | 0.12687 |
| Lasso | pre-patch | 0.17747 | 0.13104 | 0.18927 | 0.12687 |
| Elastic net | pre-patch | 0.17784 | 0.12622 | 0.18927 | 0.12687 |
| LightGBM | pre-patch | 0.16128 | 0.12013 | 0.18751 | 0.12350 |
| XGBoost | pre-patch | 0.16272 | 0.11924 | 0.18878 | 0.12342 |
| Random forest | pre-patch | 0.16853 | 0.12480 | 0.18950 | 0.12660 |
| MinGRU v2 | v2 (full: PRE-FOMC-FIX PREVIEW) | 0.16225 | 0.11434 | 0.17998 | 0.12046 |

### Economics, W = 3,000: the sign strategy, paper Sharpe ratio (D19) on the 614 post-warmup days; differences tested by Ledoit-Wolf (p; paired block bootstrap p_boot alongside)

Short straddle: mid 0.322, bid/ask -0.032.

| model | design | test | side | SR a | SR b | delta | p (LW) | p_boot |
|---|---|---|---|---|---|---|---|---|
| Ridge | pre-patch | full - HAR | mid | 1.770 | 0.954 | +0.816 | 0.106 | 0.098 |
| Ridge | pre-patch | full - short | mid | 1.770 | 0.322 | +1.449 | 0.034 | 0.019 |
| Ridge | pre-patch | har - short | mid | 0.954 | 0.322 | +0.632 | 0.350 | 0.314 |
| Ridge | pre-patch | full - HAR | bid/ask | 1.414 | 0.602 | +0.812 | 0.106 | 0.098 |
| Ridge | pre-patch | full - short | bid/ask | 1.414 | -0.032 | +1.446 | 0.034 | 0.018 |
| Ridge | pre-patch | har - short | bid/ask | 0.602 | -0.032 | +0.634 | 0.345 | 0.305 |
| Lasso | pre-patch | full - HAR | mid | 1.724 | 0.954 | +0.770 | 0.119 | 0.122 |
| Lasso | pre-patch | full - short | mid | 1.724 | 0.322 | +1.402 | 0.040 | 0.020 |
| Lasso | pre-patch | har - short | mid | 0.954 | 0.322 | +0.632 | 0.350 | 0.314 |
| Lasso | pre-patch | full - HAR | bid/ask | 1.367 | 0.602 | +0.765 | 0.119 | 0.122 |
| Lasso | pre-patch | full - short | bid/ask | 1.367 | -0.032 | +1.399 | 0.040 | 0.019 |
| Lasso | pre-patch | har - short | bid/ask | 0.602 | -0.032 | +0.634 | 0.345 | 0.305 |
| Elastic net | pre-patch | full - HAR | mid | 1.855 | 0.954 | +0.901 | 0.077 | 0.075 |
| Elastic net | pre-patch | full - short | mid | 1.855 | 0.322 | +1.533 | 0.026 | 0.013 |
| Elastic net | pre-patch | har - short | mid | 0.954 | 0.322 | +0.632 | 0.350 | 0.314 |
| Elastic net | pre-patch | full - HAR | bid/ask | 1.498 | 0.602 | +0.896 | 0.076 | 0.075 |
| Elastic net | pre-patch | full - short | bid/ask | 1.498 | -0.032 | +1.530 | 0.026 | 0.012 |
| Elastic net | pre-patch | har - short | bid/ask | 0.602 | -0.032 | +0.634 | 0.345 | 0.305 |
| LightGBM | pre-patch | full - HAR | mid | 1.587 | 0.654 | +0.933 | 0.040 | 0.039 |
| LightGBM | pre-patch | full - short | mid | 1.587 | 0.322 | +1.265 | 0.055 | 0.038 |
| LightGBM | pre-patch | har - short | mid | 0.654 | 0.322 | +0.332 | 0.614 | 0.589 |
| LightGBM | pre-patch | full - HAR | bid/ask | 1.233 | 0.306 | +0.928 | 0.042 | 0.040 |
| LightGBM | pre-patch | full - short | bid/ask | 1.233 | -0.032 | +1.265 | 0.054 | 0.035 |
| LightGBM | pre-patch | har - short | bid/ask | 0.306 | -0.032 | +0.338 | 0.603 | 0.579 |
| XGBoost | pre-patch | full - HAR | mid | 1.472 | 0.981 | +0.491 | 0.269 | 0.273 |
| XGBoost | pre-patch | full - short | mid | 1.472 | 0.322 | +1.150 | 0.090 | 0.065 |
| XGBoost | pre-patch | har - short | mid | 0.981 | 0.322 | +0.659 | 0.315 | 0.293 |
| XGBoost | pre-patch | full - HAR | bid/ask | 1.119 | 0.633 | +0.486 | 0.275 | 0.279 |
| XGBoost | pre-patch | full - short | bid/ask | 1.119 | -0.032 | +1.151 | 0.089 | 0.061 |
| XGBoost | pre-patch | har - short | bid/ask | 0.633 | -0.032 | +0.665 | 0.305 | 0.283 |
| Random forest | pre-patch | full - HAR | mid | 1.882 | 0.318 | +1.564 | 0.020 | 0.002 |
| Random forest | pre-patch | full - short | mid | 1.882 | 0.322 | +1.560 | 0.019 | 0.006 |
| Random forest | pre-patch | har - short | mid | 0.318 | 0.322 | -0.004 | 0.992 | 0.992 |
| Random forest | pre-patch | full - HAR | bid/ask | 1.523 | -0.031 | +1.554 | 0.023 | 0.003 |
| Random forest | pre-patch | full - short | bid/ask | 1.523 | -0.032 | +1.555 | 0.019 | 0.006 |
| Random forest | pre-patch | har - short | bid/ask | -0.031 | -0.032 | +0.001 | 0.998 | 0.998 |
| MinGRU v2 | v2 [final] | har - short | mid | 0.894 | 0.322 | +0.572 | 0.467 | 0.443 |
| MinGRU v2 | v2 [final] | har - short | bid/ask | 0.550 | -0.032 | +0.582 | 0.453 | 0.428 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - short | mid | 1.495 | 0.322 | +1.173 | 0.093 | 0.075 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - short | bid/ask | 1.146 | -0.032 | +1.178 | 0.089 | 0.070 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - HAR | mid | 1.495 | 0.894 | +0.600 | 0.266 | 0.232 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - HAR | bid/ask | 1.146 | 0.550 | +0.596 | 0.271 | 0.239 |

## Read (comparison worker; every full-set number a PRE-FOMC-FIX PREVIEW)

- **FOMC calendar and the statement-day afternoons: fits, but less concentrated than the trees.** EVENT days: -35.2 %
  (DM -7.61). The statement afternoons (0.97 % of the bars) carry 45.2 % of the every-bar gain. That is below the
  pre-patch trees (58-63 %) and the lasso (76 %), and close to ridge and elastic net (48-54 %). One reason: the
  MinGRU's own HAR arm already cuts the EVENT-day loss (0.710 against the OLS-HAR's 0.879). Whether the FOMC column is
  the channel is not tested here, because the MinGRU has no ablation arms.
- **Ordinary days: fits, and is the MinGRU's strength.** ORDINARY -6.27 % (DM -7.88). This is the largest ordinary-day
  gain of any class on the pre-patch design (trees -5.3 to -6.1 %, linear -2.0 to -4.0 %); 55.7 % of the gain falls on
  ordinary days. That fits the realized-moments channel, but without ablations it is not attributed.
- **Every bar and 16:00.** Every bar -9.85 % (DM -10.39): between the linear classes (-5.4 to -6.2 % pre-patch) and
  the trees (-11.1 to -14.0 %). The 16:00 bar over 2014-24 is -5.08 % (p 0.018), the best 16:00 result of any class.
  On Table 1's 2020-24 16:00 column, however, it is -1.00 % (p 0.82), not significant (the trees are -0.7 to -2.1 %
  there).
- **Economics: fits "beats passive selling, not HAR significantly", and is weaker than most classes.**
  - Full D19 Sharpe ratio 1.495 mid / 1.146 bid-ask, against 0.894 / 0.550 for the HAR set and 0.322 / -0.032 for the
    short straddle.
  - Full minus HAR +0.60 (LW p 0.27). Full minus short +1.17 (p 0.093 mid, 0.089 bid-ask): not significant at 5 %.
    By contrast, the pre-patch ridge, lasso, elastic net and forest beat the short straddle at 5 %.
- **HAR set (final).** The MinGRU's HAR-set fit is the best HAR benchmark of all: every bar 0.17998 against 0.18927
  for the OLS-HAR (-4.9 %) and 0.1875-0.1895 for the trees' HAR fits.

## Files (this folder)

- compare_full_vs_har.csv: every comparison, all spans, QLIKE and MSE, lag-8 and automatic-lag DM (label column marks the MinGRU full-set rows PRE-FOMC-FIX PREVIEW)
- compare_gain_shares.csv: Section 6.2 gain shares; compare_levels.csv: QLIKE / MSE of every arm by span; compare_econ.csv: D19 Sharpe ratios and LW / bootstrap p
- compare_gates.json: every check (713, 0 failed) and notes; compare_tables.md: the tables above (generated)
- forecasts.parquet / forecasts.json: walk-combine of both arms (full-set rows: PRE-FOMC-FIX PREVIEW); the arms' own files as listed under Inputs
- Rerun: python experiments/pooled_mingru_v2_compare.py <main tree> <mgv2 scratch> results 3000 3500 (285J env; the main tree read-only)
