# MinGRU v2, W = 3,500: merge, seed average, score and comparison (2026-10-28)

**Status.** The HAR-set walk is FINAL. The full-set walk ran on the PRE-FOMC-FIX design caches (pmf_<set> exported
2026-10-22, before the FOMC patch D13). **Every full-set number below, and so every full-vs-HAR difference, gain share
and full-set Sharpe ratio, is a PRE-FOMC-FIX PREVIEW.** Nothing here is in the paper. It was computed on CPU only, from
the four arm workers' merged walks, without refitting anything.

## Inputs

| arm | walk | forecasts (compact, copied here) | checks |
|---|---|---|---|
| HAR (baseline) | 33,970 bars x 5 seeds (169,850 refits), P100 | forecasts_w3500_baseline.parquet | pass (checks_w3500_baseline.json) |
| full (all_features) [PRE-FOMC-FIX PREVIEW] | 33,970 bars x 5 seeds (169,850 refits), P100 | forecasts_w3500_all_features.parquet | pass, fail = [] (checks_w3500_all_features.json) |

- forecast = the 5-seed mean of f; plain = f^2 B (RULE_gru_v2.txt ADDENDUM 2). B matches Table 1's B to within
  9.5e-13 relative. B comes from the design export pmf_rows.csv.
- walk-combine (experiments/pooled_mingru_v2.py walk-combine, PMF_WINDOW_DAYS=3500) of the two arms' walk-merge outputs
  was run by the comparison worker. The result is `forecasts.parquet` (md5 37d59ea89111c23a9037b7452d8ef93f) with
  `forecasts.json`, both here. It equals each arm's compact file exactly (checked).
- Seed-level forecasts are not copied (2.3 MB each). They are in the session scratchpad (`...\scratchpad\mgv2\`):
  `w3500_all_features\out\walk_merge\forecasts_seeds_all_features.parquet` (md5 6861a4b2669ae0faafabf1db00918fdf) and
  `w3500_baseline\out\merge\forecasts_seeds_baseline.parquet` (md5 3e14c630b1679f5a982a3ec1188ff401).
- Files copied from the arms: checks_*.json, scores_*.csv, econ_*.csv / .json, daily_*.parquet (per-day losses),
  econ_daily_*.parquet, forecasts_w3500_*.parquet and DONE_<arm>.txt (the arms' md5 lists). The four arm workers use
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
- Context: the six classes' W = 3,500 robustness runs (writeup/make_rehaul_robust_window.py's inputs:
  window_sweep_2014start_2026-10-22/w3500, pooled_trees_final_2014start_w3500, pooled_rf_skx_2014start_w3500). They are
  pre-patch, so apples-to-apples with the MinGRU full-set preview. No patched W = 3,500 run exists.
- Economics: the sign strategy of Section 7 on the 866 option days. The Sharpe ratio is the paper's D19 ratio on the
  614 post-warmup days. Every difference is tested by Ledoit-Wolf (block 10, 4,999 resamples), with the paired
  block-bootstrap p alongside. The short straddle is make_rehaul_economics' always_short. The context economics are the
  W = 3,500 rows of results/rehaul_robust_window_final/sharpe.csv, recomputed from the forecasts.
- Gates: 713 checks, 0 failed (one run for both windows; compare_gates.json). They reproduce, from the forecasts:
  - results/channels_2026-10-26: the item3ii joint full-vs-HAR rows on every bar / EVENT / ORDINARY, and the item2
    gain shares;
  - the ablation scorer's metrics.csv / dm.csv (patched and pre-patch);
  - Table 1's metrics.csv (every-bar QLIKE and qlike_1600);
  - results/rehaul_robust_window_final (dm_full_vs_har, metrics, and every full - HAR Sharpe row);
  - each MinGRU arm's own scores and econ files.

## Results

### MinGRU v2, W = 3,500: full set vs HAR set (PRE-FOMC-FIX PREVIEW for every full-set number)

Negative % / DM = the full set more accurate. DM: HLN, daily mean loss, NW lag 8 (EVENT: automatic lag in brackets).

| span | QLIKE full [PREVIEW] | QLIKE HAR | % QLIKE (DM, p) [PREVIEW] | % MSE (DM, p) [PREVIEW] |
|---|---|---|---|---|
| every bar | 0.16154 | 0.18000 | -10.25 (-10.82, <0.001) | +29.26 (+1.09, 0.274) |
| 16:00 bar | 0.11327 | 0.12037 | -5.90 (-3.16, 0.002) | +19.56 (+0.90, 0.368) |
| 16:00 bar 2020-24 (Table 1's column) | 0.10644 | 0.10905 | -2.39 (-0.61, 0.541) | +23.44 (+0.97, 0.333) |
| ORDINARY days | 0.15233 | 0.16275 | -6.41 (-8.46, <0.001) | +32.18 (+1.12, 0.263) |
| EVENT days (81) | 0.44581 | 0.71189 | -37.38 (-7.93, <0.001) [lag 3: -7.67, <0.001] | -9.81 (-1.73, 0.083) |

Gain shares [PRE-FOMC-FIX PREVIEW] (Section 6.2's definition; sum over bars of HAR loss - full loss = 618.79): EVENT days 45.3 %, EVENT afternoons (bars ending 14:30-16:00) 46.2 %, ORDINARY days 54.7 % (EVENT afternoons hold 0.97 % of the bars).

### Context, W = 3,500: full vs HAR, % QLIKE (DM, p); linear classes against the OLS on the HAR set (D2)

| model | design | every bar | 16:00 | 16:00 2020-24 | ORDINARY | EVENT | EVENT pm share % |
|---|---|---|---|---|---|---|---|
| Ridge | pre-patch | -6.33 (-7.06, <0.001) | -3.07 (-1.55, 0.120) | +1.96 (+0.49, 0.627) | -4.11 (-4.70, <0.001) | -19.13 (-7.18, <0.001) | 49.3 |
| Lasso | pre-patch | -7.62 (-6.39, <0.001) | -1.55 (-0.69, 0.488) | +4.10 (+0.87, 0.386) | -3.52 (-3.15, 0.002) | -31.41 (-7.54, <0.001) | 63.7 |
| Elastic net | pre-patch | -6.69 (-8.45, <0.001) | -3.89 (-2.15, 0.032) | +1.06 (+0.30, 0.767) | -4.75 (-6.40, <0.001) | -17.93 (-7.26, <0.001) | 44.2 |
| LightGBM | pre-patch | -13.98 (-10.02, <0.001) | -2.38 (-1.62, 0.106) | -2.13 (-0.81, 0.418) | -5.74 (-10.36, <0.001) | -62.78 (-8.36, <0.001) | 64.6 |
| XGBoost | pre-patch | -13.45 (-9.97, <0.001) | -3.25 (-2.19, 0.029) | -2.86 (-1.02, 0.309) | -5.80 (-10.64, <0.001) | -57.61 (-6.70, <0.001) | 62.6 |
| Random forest | pre-patch | -11.19 (-9.40, <0.001) | -1.81 (-1.15, 0.251) | +0.55 (+0.18, 0.858) | -5.08 (-9.75, <0.001) | -45.21 (-8.84, <0.001) | 60.7 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | -10.25 (-10.82, <0.001) | -5.90 (-3.16, 0.002) | -2.39 (-0.61, 0.541) | -6.41 (-8.46, <0.001) | -37.38 (-7.93, <0.001) | 46.2 |

### QLIKE levels, W = 3,500 (every bar / 16:00)

| model | design | full every bar | full 16:00 | HAR every bar | HAR 16:00 |
|---|---|---|---|---|---|
| OLS | pre-patch | - | - | 0.18967 | 0.12723 |
| Ridge | pre-patch | 0.17767 | 0.12333 | 0.18967 | 0.12723 |
| Lasso | pre-patch | 0.17520 | 0.12525 | 0.18967 | 0.12723 |
| Elastic net | pre-patch | 0.17698 | 0.12229 | 0.18967 | 0.12723 |
| LightGBM | pre-patch | 0.16134 | 0.12037 | 0.18756 | 0.12331 |
| XGBoost | pre-patch | 0.16356 | 0.11918 | 0.18899 | 0.12318 |
| Random forest | pre-patch | 0.16854 | 0.12414 | 0.18977 | 0.12642 |
| MinGRU v2 | v2 (full: PRE-FOMC-FIX PREVIEW) | 0.16154 | 0.11327 | 0.18000 | 0.12037 |

### Economics, W = 3,500: the sign strategy, paper Sharpe ratio (D19) on the 614 post-warmup days; differences tested by Ledoit-Wolf (p; paired block bootstrap p_boot alongside)

Short straddle: mid 0.322, bid/ask -0.032.

| model | design | test | side | SR a | SR b | delta | p (LW) | p_boot |
|---|---|---|---|---|---|---|---|---|
| Ridge | pre-patch | full - HAR | mid | 1.837 | 1.005 | +0.832 | 0.069 | 0.066 |
| Ridge | pre-patch | full - short | mid | 1.837 | 0.322 | +1.515 | 0.026 | 0.012 |
| Ridge | pre-patch | har - short | mid | 1.005 | 0.322 | +0.683 | 0.320 | 0.281 |
| Ridge | pre-patch | full - HAR | bid/ask | 1.481 | 0.653 | +0.828 | 0.068 | 0.065 |
| Ridge | pre-patch | full - short | bid/ask | 1.481 | -0.032 | +1.513 | 0.027 | 0.011 |
| Ridge | pre-patch | har - short | bid/ask | 0.653 | -0.032 | +0.685 | 0.313 | 0.275 |
| Lasso | pre-patch | full - HAR | mid | 1.797 | 1.005 | +0.792 | 0.101 | 0.104 |
| Lasso | pre-patch | full - short | mid | 1.797 | 0.322 | +1.475 | 0.033 | 0.016 |
| Lasso | pre-patch | har - short | mid | 1.005 | 0.322 | +0.683 | 0.320 | 0.281 |
| Lasso | pre-patch | full - HAR | bid/ask | 1.441 | 0.653 | +0.788 | 0.101 | 0.104 |
| Lasso | pre-patch | full - short | bid/ask | 1.441 | -0.032 | +1.473 | 0.033 | 0.014 |
| Lasso | pre-patch | har - short | bid/ask | 0.653 | -0.032 | +0.685 | 0.313 | 0.275 |
| Elastic net | pre-patch | full - HAR | mid | 2.004 | 1.005 | +0.998 | 0.027 | 0.023 |
| Elastic net | pre-patch | full - short | mid | 2.004 | 0.322 | +1.682 | 0.012 | 0.003 |
| Elastic net | pre-patch | har - short | mid | 1.005 | 0.322 | +0.683 | 0.320 | 0.281 |
| Elastic net | pre-patch | full - HAR | bid/ask | 1.648 | 0.653 | +0.995 | 0.027 | 0.022 |
| Elastic net | pre-patch | full - short | bid/ask | 1.648 | -0.032 | +1.680 | 0.012 | 0.003 |
| Elastic net | pre-patch | har - short | bid/ask | 0.653 | -0.032 | +0.685 | 0.313 | 0.275 |
| LightGBM | pre-patch | full - HAR | mid | 1.540 | 0.590 | +0.950 | 0.035 | 0.029 |
| LightGBM | pre-patch | full - short | mid | 1.540 | 0.322 | +1.218 | 0.065 | 0.040 |
| LightGBM | pre-patch | har - short | mid | 0.590 | 0.322 | +0.268 | 0.682 | 0.665 |
| LightGBM | pre-patch | full - HAR | bid/ask | 1.187 | 0.242 | +0.945 | 0.036 | 0.030 |
| LightGBM | pre-patch | full - short | bid/ask | 1.187 | -0.032 | +1.219 | 0.063 | 0.037 |
| LightGBM | pre-patch | har - short | bid/ask | 0.242 | -0.032 | +0.274 | 0.673 | 0.653 |
| XGBoost | pre-patch | full - HAR | mid | 1.347 | 0.777 | +0.570 | 0.151 | 0.129 |
| XGBoost | pre-patch | full - short | mid | 1.347 | 0.322 | +1.025 | 0.106 | 0.078 |
| XGBoost | pre-patch | har - short | mid | 0.777 | 0.322 | +0.455 | 0.499 | 0.472 |
| XGBoost | pre-patch | full - HAR | bid/ask | 0.994 | 0.429 | +0.564 | 0.155 | 0.131 |
| XGBoost | pre-patch | full - short | bid/ask | 0.994 | -0.032 | +1.026 | 0.102 | 0.073 |
| XGBoost | pre-patch | har - short | bid/ask | 0.429 | -0.032 | +0.461 | 0.486 | 0.462 |
| Random forest | pre-patch | full - HAR | mid | 1.828 | 0.802 | +1.026 | 0.010 | 0.009 |
| Random forest | pre-patch | full - short | mid | 1.828 | 0.322 | +1.506 | 0.023 | 0.009 |
| Random forest | pre-patch | har - short | mid | 0.802 | 0.322 | +0.480 | 0.469 | 0.444 |
| Random forest | pre-patch | full - HAR | bid/ask | 1.470 | 0.452 | +1.018 | 0.010 | 0.009 |
| Random forest | pre-patch | full - short | bid/ask | 1.470 | -0.032 | +1.502 | 0.023 | 0.009 |
| Random forest | pre-patch | har - short | bid/ask | 0.452 | -0.032 | +0.484 | 0.461 | 0.435 |
| MinGRU v2 | v2 [final] | har - short | mid | 0.882 | 0.322 | +0.560 | 0.457 | 0.434 |
| MinGRU v2 | v2 [final] | har - short | bid/ask | 0.538 | -0.032 | +0.570 | 0.443 | 0.421 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - short | mid | 1.449 | 0.322 | +1.127 | 0.109 | 0.090 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - short | bid/ask | 1.103 | -0.032 | +1.135 | 0.103 | 0.087 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - HAR | mid | 1.449 | 0.882 | +0.567 | 0.270 | 0.253 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - HAR | bid/ask | 1.103 | 0.538 | +0.565 | 0.273 | 0.256 |

## Read (comparison worker; every full-set number a PRE-FOMC-FIX PREVIEW)

- **FOMC calendar and the statement-day afternoons: fits, but less concentrated than the trees.** EVENT days: -37.4 %
  (DM -7.93). The statement afternoons (0.97 % of the bars) carry 46.2 % of the every-bar gain. That is below the
  W = 3,500 trees (61-65 %) and the lasso (64 %), and close to ridge and elastic net (44-49 %). One reason: the
  MinGRU's own HAR arm already cuts the EVENT-day loss (0.712 against the OLS-HAR's 0.889). There are no MinGRU
  ablations, so the channel is not attributed.
- **Ordinary days: fits, and is the MinGRU's strength.** ORDINARY -6.41 % (DM -8.46). This is the largest ordinary-day
  gain of any class at W = 3,500 (trees -5.1 to -5.8 %, linear -3.5 to -4.8 %); 54.7 % of the gain falls on ordinary
  days.
- **Every bar and 16:00.** Every bar -10.25 % (DM -10.82): between the linear classes (-6.3 to -7.6 %) and the trees
  (-11.2 to -14.0 %). The 16:00 bar over 2014-24 is -5.90 % (p 0.002), the best 16:00 result of any class. On Table 1's
  2020-24 16:00 column it is -2.39 % (p 0.54), not significant (the classes there run from -2.9 to +4.1 %).
- **Economics: fits "beats passive selling, not HAR significantly", and is weaker than most classes.**
  - Full D19 Sharpe ratio 1.449 mid / 1.103 bid-ask, against 0.882 / 0.538 for the HAR set and 0.322 / -0.032 for the
    short straddle.
  - Full minus HAR +0.57 (LW p 0.27). Full minus short +1.13 (p 0.109 mid, 0.103 bid-ask): not significant at 5 %.
    By contrast, the W = 3,500 ridge, lasso, elastic net and forest beat the short straddle at 5 %.
- **HAR set (final).** The MinGRU's HAR-set fit is the best HAR benchmark at W = 3,500: every bar 0.18000 against
  0.18967 for the OLS-HAR (-5.1 %) and 0.1876-0.1898 for the trees' HAR fits.
- **Window.** The MinGRU at 3,500 against 3,000 is close on every span (every-bar full 0.16154 vs 0.16225, HAR 0.18000
  vs 0.17998). The robust-window DM between windows is not computed for the MinGRU.

## Files (this folder)

- compare_full_vs_har.csv: every comparison, all spans, QLIKE and MSE, lag-8 and automatic-lag DM (label column marks the MinGRU full-set rows PRE-FOMC-FIX PREVIEW)
- compare_gain_shares.csv: Section 6.2 gain shares; compare_levels.csv: QLIKE / MSE of every arm by span; compare_econ.csv: D19 Sharpe ratios and LW / bootstrap p
- compare_gates.json: every check (713, 0 failed) and notes; compare_tables.md: the tables above (generated)
- forecasts.parquet / forecasts.json: walk-combine of both arms (full-set rows: PRE-FOMC-FIX PREVIEW); the arms' own files as listed under Inputs
- Rerun: python experiments/pooled_mingru_v2_compare.py <main tree> <mgv2 scratch> results 3000 3500 (285J env; the main tree read-only)
