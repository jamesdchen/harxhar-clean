# Trees vs linear at the close, re-run on the de-duplicated per-bar design with the per-window tree mask (checklist I8)

Written by `experiments/dvs_dedup_1530.py` (stages `capture`, `refit`, `carc`, `analyze`, which writes this file) from this folder's CSVs and `numbers.json` and, for the before / after, the first pass's (`results/dense_vs_sparse/`, `experiments/dense_vs_sparse_1530.py`, whose functions this re-run imports and runs unchanged). Nothing below is asserted: every statement of the first pass is re-evaluated on the new runs and its status recorded (`before_after_claims.csv`).

**What changed, and nothing else.** (1) The design: the 12 session-edge columns `har_ma_k_x_open` (all zero in a one-bar series: 6 of 6) and `har_ma_k_x_close` (byte-copies of `har_ma_k` at the 16:00 bar: 6 of 6) are gone (commit 47f7f9c): `live_feasible` 244 → 232 columns, `all_features` 640 → 628; gate: the new design is the old one minus exactly those 12 columns, bit for bit, same target and dates. (2) Every tree fit uses the per-window column mask (`src/models/window_mask.py`, commit f9a19b6, the user's decision of 2026-09-29): on each refit's own 2000-session training window the constant columns and the byte-copies of an earlier column are dropped before the fit; TreeSHAP is mapped back to all columns with 0 for a dropped column. Kept columns per LightGBM refit: `live_feasible` 136–145 of 232 (median 138); `all_features` 360–378 of 628 (median 369); on the screened top-k subsets the mask drops a column in 3 of 2058 refits (`mask_counts.csv`). The linear models are unchanged in method: their identifiability mask already removed constant and duplicated columns at every tune.

Consequently every ridge and lasso forecast is the first pass's: the largest relative change of any ridge / lasso run (all columns and both top-k sweeps) is 3.3e-12 and the sign(s) side differs on 0 trade days (the elastic net, which enters only the density table, moves by up to 4.6e-04 on 20 rows (`all_features`): the warm-homotopy float path, see the gates). LightGBM on all columns moves: median relative change of the forecast 1.7e-02 (`live_feasible`) / 1.7e-02 (`all_features`), the side differs on 34 / 50 of 866 days; on the screened subsets (k ≤ 64) the largest relative change is 5.3e-02 and the side differs on at most 0 days (`before_after_sweep.csv`).

**Same methodology.** The per-bar forecast of the bar ending 16:00 (issued at 15:30), 2000-session rolling window; ridge / lasso / elastic net refit every session with the penalty re-chosen every 250 sessions; LightGBM / XGBoost / random forest in the shipped configuration refit every 10 sessions, one thread per fit; the research scorer (the 16:00 bar recalibrated alone, (ŷ² + s)·B); QLIKE and the deck's 15:30 **sign(s)** rule on the 866 trade days; 95 % circular block bootstrap (block 21, 2,000 draws, the first pass's seed, one set of draws for every run, the first pass's runs included, so every difference and every new-minus-first-pass change is paired).

**Where it ran** (as the first pass split it: the sweep locally, the density table's TreeSHAP from cluster runs): the linear refits and the LightGBM sweep with its all-column references locally, in the first pass's environment (LightGBM 4.6.0; 4 single-threaded processes on a shared laptop, 4.55 process-hours, 69 min wall); the density table's masked TreeSHAP (LightGBM, XGBoost, random forest on all columns) and the no-mask gate on the cluster (LightGBM 4.6.0, XGBoost 3.2.0, shap 0.51.0): dvs_dedup_canary 12480879 (main, 3 CPUs, 00:03:34) dvs_dedup_fleet 12480880 (main, 20 CPUs, 00:11:24) dvs_dedup_gate 12481354 (debug, 3 CPUs, 00:03:12); 4.14 allocated CPU-hours, peak 20 CPUs, 18 min of job wall time.

The tree spec changed locally since the cluster runs (md5 7696a895d40d now; another campaign's WINDOW_MASK axis): the cluster ran md5 4bb0f7155908: commit 47f7f9c; model section (parameters, make_model, contributions) identical to the local spec.

Series names are the design's column stems: `har_ma_*` = the target's own HAR ladder (realized variance, means of the last 1, 5, 25, 125, 625, 3125 bars), `adj_sumabsret_ma_*` = absolute returns, `adj_sumret4_ma_*` = 4th-power returns, `adj_sumpret2_ma_*` = upside squared returns, `adj_sumbipow_ma_*` = bipower variation, `adj_sumvolume_ma_*` = ES volume, `adj_vix_ma_*` / `adj_vvix_ma_*` / `adj_vix3m_ma_*` = VIX, VVIX, VIX3M, `*_ewstock` / `*_vwstock` = the same statistics on equal- / value-weighted constituent stocks.

## Answer in brief: the first pass's statements, first pass → de-dup + mask

| statement | first pass (old design) | de-dup + mask | same |
|---|---|---|---|
| ridge has the largest N_eff (live_feasible) | yes | yes | yes |
| N_eff ridge / range of the other five (live_feasible) | 40.7 / 8–19 | 40.7 / 7–14 | no |
| k_95 ridge / largest of the other five (live_feasible) | 22 / 12 | 22 / 10 | no |
| ridge has the largest N_eff (all_features) | yes | yes | yes |
| N_eff ridge / range of the other five (all_features) | 124.2 / 16–39 | 124.2 / 8–39 | no |
| k_95 ridge / largest of the other five (all_features) | 64 / 20 | 64 / 20 | yes |
| largest k_80 of any model | 7 | 7 | yes |
| ridge puts the most forecast variance on the recent return-size group (live_feasible) | yes | yes | yes |
| ridge puts the least on the har_ma_1, har_ma_5 group (live_feasible) | yes | yes | yes |
| QLIKE significantly worse than all columns for every model at every k <= K (both designs) | K = 4 | K = 4 | yes |
| the same, live_feasible | K = 8 | K = 8 | yes |
| every model trades below its all-column Sharpe at every k <= 16 | yes | yes | yes |
| ridge beats LightGBM on QLIKE at k = 1, 2 (both designs, intervals exclude 0) | yes | yes | yes |
| with all columns ridge ties the lasso and LightGBM on QLIKE (both designs) | yes | yes | yes |
| the lasso beats ridge on QLIKE at k = 4, 8 (all_features) | yes | yes | yes |
| ridge - lasso QLIKE gap narrows significantly to all columns from k (all_features) | 4, 8, 16 | 4, 8, 16 | yes |
| the same, live_feasible | none | none | yes |
| ridge - LightGBM QLIKE gap closes significantly to all columns from k (live_feasible) | 1, 2 | 1, 2 | yes |
| the same, all_features | none | none | yes |
| ridge-vs-lasso / ridge-vs-LightGBM Sharpe gaps at k <= 64 whose interval excludes 0 | 1 of 28 | 1 of 28 | yes |
| ... whose change between k and all columns excludes 0 | 1 of 28 | 1 of 28 | yes |
| all-column Sharpe gaps (ridge - lasso, ridge - LightGBM) whose interval excludes 0 | none | none | yes |
| PC1 collapse moves the ridge - lasso Sharpe gap significantly | no | no | yes |

Headline (20 of 23 statements read the same; the rest are numbers that moved):

* `live_feasible`, density: the trees' N_eff falls: LightGBM 19.1 → 14.1 (no mask 13.9), XGBoost 12.7 → 9.7 (no mask 9.2), random forest 13.2 → 6.8 (no mask 6.8); ridge's 40.7 → 40.7. Of each tree's change, the de-duplication without the mask already gives 100–116 %.
* `all_features`, density: the trees' N_eff falls: LightGBM 25.8 → 20.4 (no mask 20.4), XGBoost 16.4 → 13.0 (no mask 11.9), random forest 16.4 → 8.4 (no mask 8.5); ridge's 124.2 → 124.2. Of each tree's change, the de-duplication without the mask already gives 99–131 %.
* The `har_ma_1`, `har_ma_5` group's share of the trees' forecast variance rises (`all_features` LightGBM 48 → 56 %, XGBoost 48 → 56 %, random forest 35 → 75 %; `live_feasible` LightGBM 46 → 58 %, XGBoost 50 → 61 %, random forest 39 → 78 %): in the first pass the trees also split on the byte-copies `har_ma_k_x_close`, which carried 9–29 % of their mean |TreeSHAP| and sat outside every group (a copy is not an identifiable column).
* `all_features`, LightGBM on all columns, first pass → de-dup + mask: QLIKE 0.1007 → 0.1009 (change +0.0002 [-0.0012, +0.0016]), sign(s) Sharpe mid 1.43 → 1.63 (change +0.20 [-0.27, +0.77]).
* `live_feasible`, LightGBM on all columns, first pass → de-dup + mask: QLIKE 0.0983 → 0.0987 (change +0.0004 [-0.0012, +0.0020]), sign(s) Sharpe mid 1.78 → 1.82 (change +0.04 [-0.30, +0.39]).
* `live_feasible`, the mask alone (LightGBM on all columns, masked minus unmasked, both local): QLIKE +0.0007 [-0.0008, +0.0021], Sharpe -0.10 [-0.62, +0.38].
* `all_features`, the mask alone (LightGBM on all columns, masked minus unmasked, both local): QLIKE +0.0007 [-0.0008, +0.0024], Sharpe -0.05 [-0.56, +0.43].
* The sweep, hypothesis-test and collapse statements of the first pass (from 'QLIKE significantly worse than all columns for every model at every k <= K (both designs)' on) read the same; the screen picks the same columns; the linear runs are unchanged (≤ 1e-8); LightGBM's subset forecasts change in 1 (design, k) cells (`live_feasible` k = 64), the mask drops a column in 1 (`live_feasible` k = 64).

All-column gaps, first pass → de-dup + mask (ridge minus the other model; QLIKE, then Sharpe mid): `live_feasible` ridge - lasso: +0.0041 [-0.0019, +0.0116] → +0.0041 [-0.0019, +0.0116], +0.08 [-0.74, +0.93] → +0.08 [-0.74, +0.93]; `live_feasible` ridge - LightGBM: +0.0023 [-0.0031, +0.0084] → +0.0019 [-0.0032, +0.0075], +0.12 [-0.61, +0.78] → +0.08 [-0.59, +0.73]; `all_features` ridge - lasso: +0.0006 [-0.0055, +0.0070] → +0.0006 [-0.0055, +0.0070], +0.22 [-0.31, +0.80] → +0.22 [-0.31, +0.80]; `all_features` ridge - LightGBM: -0.0003 [-0.0082, +0.0067] → -0.0006 [-0.0081, +0.0061], +0.24 [-0.58, +0.99] → +0.03 [-0.74, +0.81]. The Sharpe intervals of the k-sweep gaps are 0.8–1.8 wide (first pass 0.8–1.8).

## 1. Signal density: how many inputs does each forecast use?

Contributions as in the first pass: linear β_j (x_j − window mean_j) (linear SHAP; additivity 7.1e-14); trees: TreeSHAP of the masked T10 walk run on the cluster for this study (additivity ≤ 1.9e-06 relative; the first pass read the tree campaign's stored unmasked TreeSHAP; the stored unmasked T10 runs on the new design are the **de-dup, no mask** rows). share_j = mean |contribution_j| / sum; N_eff = 1 / Σ share²; k_x = the fewest columns, in share order, reproducing x of the variance of the input-driven forecast; greedy k_95 = columns added one at a time. All 1,469 forecasts; `before_after_density.csv` has every column, the trade-day sample and the series level.

| design | model | N_eff first pass | N_eff de-dup + mask | (no mask) | k_80 | k_95 | greedy k_95 | columns with any weight | N_eff groups |
|---|---|---:|---:|---:|---|---|---|---|---|
| live_feasible | ridge | 40.7 | **40.7** |  | 6 → 6 | 22 → 22 | 15 → 15 | 138 → 138 | 13.8 → 13.8 |
|  | lasso | 10.8 | **10.8** |  | 3 → 3 | 7 → 7 | 5 → 5 | 86 → 86 | 5.3 → 5.3 |
|  | elastic net | 7.7 | **7.7** |  | 3 → 3 | 5 → 5 | 5 → 5 | 52 → 52 | 4.8 → 4.8 |
|  | LightGBM | 19.1 | **14.1** | 13.9 | 4 → 3 | 12 → 10 | 11 → 9 | 150 → 139 | 7.3 → 6.2 |
|  | XGBoost | 12.7 | **9.7** | 9.2 | 3 → 2 | 10 → 8 | 9 → 7 | 146 → 138 | 5.1 → 4.6 |
|  | random forest | 13.2 | **6.8** | 6.8 | 3 → 2 | 6 → 4 | 6 → 4 | 174 → 147 | 5.9 → 3.3 |
| all_features | ridge | 124.2 | **124.2** |  | 7 → 7 | 64 → 64 | 25 → 25 | 379 → 379 | 38.1 → 38.1 |
|  | lasso | 21.3 | **21.3** |  | 3 → 3 | 6 → 6 | 6 → 6 | 204 → 204 | 9.2 → 9.2 |
|  | elastic net | 39.4 | **39.4** |  | 3 → 3 | 20 → 20 | 7 → 7 | 238 → 238 | 14.7 → 14.7 |
|  | LightGBM | 25.8 | **20.4** | 20.4 | 4 → 3 | 16 → 12 | 14 → 12 | 411 → 367 | 9.7 → 8.6 |
|  | XGBoost | 16.4 | **13.0** | 11.9 | 4 → 3 | 15 → 11 | 12 → 9 | 392 → 358 | 6.5 → 6.0 |
|  | random forest | 16.4 | **8.4** | 8.5 | 3 → 2 | 9 → 6 | 8 → 6 | 566 → 388 | 8.5 → 4.0 |

Why the trees' numbers move: in the first pass each `har_ma_k_x_close` column was a byte-copy of `har_ma_k`, and the trees split credit between the two. Share of mean |TreeSHAP| on the 12 session-edge columns in the first pass's stored tree runs, and the `har_ma` ladder's share (with its copies) then vs now (masked):

| design | model | session-edge share, first pass | `har_ma_*` + copies, first pass | `har_ma_*`, de-dup + mask |
|---|---|---:|---:|---:|
| live_feasible | LightGBM | 13.0 % | 47.9 % | 43.8 % |
| live_feasible | XGBoost | 12.8 % | 57.5 % | 53.6 % |
| live_feasible | random forest | 29.1 % | 58.8 % | 58.5 % |
| all_features | LightGBM | 8.6 % | 39.5 % | 36.4 % |
| all_features | XGBoost | 11.9 % | 50.5 % | 46.5 % |
| all_features | random forest | 27.6 % | 51.8 % | 51.8 % |

Share of each model's forecast variance carried by a correlated group (|corr| > 0.8, complete linkage; 64 groups of the 143 identifiable `live_feasible` columns, 139 of 361 `all_features`; first pass 64 of 143 and 139 of 361), de-dup + mask with the first pass in brackets (`before_after_groups.csv`):

| design | group (design columns) | ridge | lasso | LightGBM | XGBoost | random forest |
|---|---|---:|---:|---:|---:|---:|
| live_feasible | `har_ma_1`, `har_ma_5` | 30.8 % (30.8) | 55.1 % (55.1) | 58.3 % (46.1) | 60.9 % (50.2) | 77.5 % (39.3) |
|  | `adj_sumabsret`, `adj_sumret4`, `adj_sumpret2`, `adj_sumbipow` at `_ma_1`, `_ma_5` | 25.0 % (25.0) | 8.3 % (8.3) | 8.3 % (4.0) | 9.2 % (6.6) | 5.9 % (6.1) |
|  | `har_ma_25`, `har_ma_125` | 17.1 % (17.1) | 21.4 % (21.4) | 9.0 % (6.7) | 9.6 % (7.9) | 4.4 % (2.1) |
|  | `adj_sumvolume` at `_ma_1`, `_ma_5` | 6.6 % (6.6) | 4.3 % (4.3) | 5.4 % (4.8) | 5.7 % (4.8) | 2.2 % (2.1) |
| all_features | `har_ma_1`, `har_ma_5` | 28.3 % (28.3) | 50.8 % (50.8) | 56.4 % (48.0) | 56.2 % (48.1) | 75.1 % (34.7) |
|  | `adj_sumabsret_ewstock`, `adj_sumabsret_vwstock` at `_ma_1`, `_ma_5` | 21.0 % (21.0) | 16.1 % (16.1) | 4.3 % (4.0) | 4.4 % (3.8) | 3.6 % (3.4) |
|  | `har_ma_25`, `har_ma_125` | 15.1 % (15.1) | 19.6 % (19.6) | 8.8 % (6.9) | 9.8 % (6.9) | 3.4 % (1.7) |
|  | `adj_sumabsret`, `adj_sumret4`, `adj_sumpret2`, `adj_sumbipow` at `_ma_1`, `_ma_5` | 14.2 % (14.2) | 1.4 % (1.4) | 6.7 % (3.4) | 8.0 % (4.9) | 4.9 % (5.0) |

## 2. The sparsity sweep: ridge, lasso and LightGBM on the same top-k inputs

The first pass's causal rule, unchanged: at each of the six penalty re-choices the k columns with the largest |correlation with the target| over the 2000 sessions before the block (constant or duplicated columns never picked), the same columns for all three models; LightGBM's fits also apply the per-window mask to those k columns. The screen picks the same columns as in the first pass in 48 of 48 (design, k, block) cells (`before_after_picks.csv`).

QLIKE (866 trade days) and sign(s) Sharpe (mid) against k, de-dup + mask; * = the paired interval against the same model on all columns excludes zero; first pass in brackets:

| k | ridge QLIKE | lasso QLIKE | LightGBM QLIKE | ridge Sharpe | lasso Sharpe | LightGBM Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| **live_feasible** | | | | | | |
| 1 | 0.1218 * (0.1218) | 0.1217 * (0.1217) | 0.1284 * (0.1284) | 1.46 (1.46) | 1.00 (1.00) | 1.39 (1.39) |
| 2 | 0.1136 * (0.1136) | 0.1113 * (0.1113) | 0.1197 * (0.1197) | 0.91 * (0.91) | 0.70 * (0.70) | 0.59 * (0.59) |
| 4 | 0.1185 * (0.1185) | 0.1085 * (0.1085) | 0.1148 * (0.1148) | 0.94 * (0.94) | 1.36 (1.36) | 0.13 * (0.13) |
| 8 | 0.1136 * (0.1136) | 0.1063 * (0.1063) | 0.1093 * (0.1093) | 1.16 * (1.16) | 1.25 (1.25) | 0.99 * (0.99) |
| 16 | 0.1081 * (0.1081) | 0.1034 * (0.1034) | 0.1033 (0.1033) | 1.45 (1.45) | 1.14 (1.14) | 0.99 * (0.99) |
| 32 | 0.1034 (0.1034) | 0.1018 * (0.1018) | 0.1047 * (0.1047) | 1.98 (1.98) | 1.18 (1.18) | 1.71 (1.71) |
| 64 | 0.0988 (0.0988) | 0.0977 (0.0977) | 0.0985 (0.0986) | 2.06 (2.06) | 2.19 (2.19) | 2.17 (2.17) |
| all (232; first pass 244) | 0.1005 (0.1005) | 0.0964 (0.0964) | 0.0987 (0.0983) | 1.90 (1.90) | 1.83 (1.83) | 1.82 (1.78) |
| **all_features** | | | | | | |
| 1 | 0.1218 * (0.1218) | 0.1217 * (0.1217) | 0.1284 * (0.1284) | 1.46 (1.46) | 1.00 (1.00) | 1.39 (1.39) |
| 2 | 0.1136 * (0.1136) | 0.1113 * (0.1113) | 0.1197 * (0.1197) | 0.91 (0.91) | 0.70 (0.70) | 0.59 * (0.59) |
| 4 | 0.1185 * (0.1185) | 0.1085 * (0.1085) | 0.1148 * (0.1148) | 0.94 (0.94) | 1.36 (1.36) | 0.13 * (0.13) |
| 8 | 0.1152 * (0.1152) | 0.1047 (0.1047) | 0.1087 * (0.1087) | 1.45 (1.45) | 1.44 (1.44) | 1.19 (1.19) |
| 16 | 0.1159 * (0.1159) | 0.1063 * (0.1063) | 0.1075 * (0.1075) | 1.34 (1.34) | 1.25 (1.25) | 1.27 (1.27) |
| 32 | 0.1073 (0.1073) | 0.1056 * (0.1056) | 0.1039 (0.1039) | 1.25 (1.25) | 1.15 (1.15) | 1.39 (1.39) |
| 64 | 0.1020 (0.1020) | 0.1003 (0.1003) | 0.1042 (0.1042) | 1.44 (1.44) | 1.14 (1.14) | 2.01 (2.01) |
| all (628; first pass 640) | 0.1004 (0.1004) | 0.0998 (0.0998) | 0.1009 (0.1007) | 1.66 (1.66) | 1.45 (1.45) | 1.63 (1.43) |

LightGBM, the model whose forecasts change: de-dup + mask minus first pass, paired (`before_after_sweep.csv`):

| design | k | QLIKE change | Sharpe change | median rel. forecast change | days the side differs |
|---|---|---|---|---:|---:|
| live_feasible | 1 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | 2 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | 4 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | 8 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | 16 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | 32 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | 64 | -0.0001 [-0.0003, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| live_feasible | all | +0.0004 [-0.0012, +0.0020] | +0.04 [-0.30, +0.39] | 1.7e-02 | 34 |
| all_features | 1 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | 2 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | 4 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | 8 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | 16 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | 32 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | 64 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 0.0e+00 | 0 |
| all_features | all | +0.0002 [-0.0012, +0.0016] | +0.20 [-0.27, +0.77] | 1.7e-02 | 50 |

**The hypothesis test** (`sweep_gaps.csv`; first pass vs now in `before_after_gaps.csv`, with the paired change of each gap):

* ridge - lasso, QLIKE, `live_feasible`: gap excludes zero at k = 4 +0.0100 [+0.0025, +0.0227]; k = 8 +0.0073 [+0.0004, +0.0194]; all columns +0.0041 [-0.0019, +0.0116]; the gap's change from k to all columns excludes zero at no k.
* ridge - lasso, QLIKE, `all_features`: gap excludes zero at k = 4 +0.0100 [+0.0025, +0.0227]; k = 8 +0.0105 [+0.0023, +0.0243]; all columns +0.0006 [-0.0055, +0.0070]; the gap's change from k to all columns excludes zero at k = 4 -0.0094 [-0.0208, -0.0019]; k = 8 -0.0098 [-0.0218, -0.0021]; k = 16 -0.0090 [-0.0211, -0.0002].
* ridge - LightGBM, QLIKE, `live_feasible`: gap excludes zero at k = 1 -0.0066 [-0.0118, -0.0009]; k = 2 -0.0060 [-0.0106, -0.0010]; all columns +0.0019 [-0.0032, +0.0075]; the gap's change from k to all columns excludes zero at k = 1 +0.0085 [+0.0016, +0.0155]; k = 2 +0.0079 [+0.0013, +0.0151].
* ridge - LightGBM, QLIKE, `all_features`: gap excludes zero at k = 1 -0.0066 [-0.0118, -0.0009]; k = 2 -0.0060 [-0.0106, -0.0010]; all columns -0.0006 [-0.0081, +0.0061]; the gap's change from k to all columns excludes zero at no k.
* lasso - LightGBM, QLIKE, `live_feasible`: gap excludes zero at k = 1 -0.0067 [-0.0101, -0.0030]; k = 2 -0.0084 [-0.0121, -0.0043]; k = 4 -0.0064 [-0.0114, -0.0004]; all columns -0.0022 [-0.0070, +0.0028]; the gap's change from k to all columns excludes zero at k = 2 +0.0062 [+0.0003, +0.0118].
* lasso - LightGBM, QLIKE, `all_features`: gap excludes zero at k = 1 -0.0067 [-0.0101, -0.0030]; k = 2 -0.0084 [-0.0121, -0.0043]; k = 4 -0.0064 [-0.0114, -0.0004]; k = 64 -0.0039 [-0.0080, -0.0001]; all columns -0.0012 [-0.0078, +0.0050]; the gap's change from k to all columns excludes zero at k = 2 +0.0072 [+0.0008, +0.0129].
* Sharpe: of the 28 ridge-vs-lasso and ridge-vs-LightGBM gaps at k = 1 … 64, 1 has an interval excluding zero (first pass 1); the change of the gap between k and all columns excludes zero in 1 (first pass 1); across all 42 Sharpe comparisons 3 changes of gap exclude zero (first pass 2).
* Did a gap itself move between the designs? Of the 48 (pair, k, design) gaps, the paired change new minus first pass excludes zero for QLIKE in 0 and for Sharpe in 0.

**LightGBM's all-column references** (`lgbm_references.csv`; research scorer, 866 days):

* `live_feasible`: LightGBM all columns, de-dup + mask (local; the sweep's reference): QLIKE 0.0987, Sharpe 1.82 mid / 1.35 crossed; LightGBM all columns, de-dup, no mask (local): QLIKE 0.0980, Sharpe 1.92 mid / 1.46 crossed; LightGBM all columns, de-dup, no mask (stored T10, cluster): QLIKE 0.0981, Sharpe 1.76 mid / 1.29 crossed; LightGBM all columns, de-dup + mask (cluster; the density table's run): QLIKE 0.0987, Sharpe 1.82 mid / 1.35 crossed; mask effect (both local): QLIKE +0.0007 [-0.0008, +0.0021], Sharpe -0.10 [-0.62, +0.38]; mask effect (both cluster): QLIKE +0.0006 [-0.0008, +0.0019], Sharpe +0.06 [-0.43, +0.51]; local vs cluster (both unmasked): QLIKE -0.0001 [-0.0008, +0.0004], Sharpe +0.16 [-0.06, +0.40]; local vs cluster (both masked): QLIKE +0.0000 [+0.0000, +0.0000], Sharpe +0.00 [+0.00, +0.00].
* `all_features`: LightGBM all columns, de-dup + mask (local; the sweep's reference): QLIKE 0.1009, Sharpe 1.63 mid / 1.16 crossed; LightGBM all columns, de-dup, no mask (local): QLIKE 0.1003, Sharpe 1.68 mid / 1.22 crossed; LightGBM all columns, de-dup, no mask (stored T10, cluster): QLIKE 0.1013, Sharpe 1.75 mid / 1.29 crossed; LightGBM all columns, de-dup + mask (cluster; the density table's run): QLIKE 0.1009, Sharpe 1.63 mid / 1.16 crossed; mask effect (both local): QLIKE +0.0007 [-0.0008, +0.0024], Sharpe -0.05 [-0.56, +0.43]; mask effect (both cluster): QLIKE -0.0004 [-0.0019, +0.0013], Sharpe -0.12 [-0.62, +0.35]; local vs cluster (both unmasked): QLIKE -0.0011 [-0.0019, -0.0003], Sharpe -0.07 [-0.59, +0.35]; local vs cluster (both masked): QLIKE +0.0000 [+0.0000, +0.0000], Sharpe +0.00 [+0.00, +0.00].

## 3. Correlated-input groups: collapse each group to its first principal component

The first pass's rule, unchanged (groups by complete linkage at |corr| > 0.8 on the 2000 sessions before each 250-session block, each multi-column group replaced by its first principal component, ridge and lasso refit on the collapsed design). The first component carries a median 94–96 % of a group's window variance. De-dup + mask, first pass in brackets (`before_after_collapse.csv`):

| design | | QLIKE | Sharpe mid | vs the same model on the full design |
|---|---|---:|---:|---|
| live_feasible | ridge, collapsed | 0.1034 (0.1034) | 2.06 (2.06) | QLIKE +0.0029 [-0.0025, +0.0098], Sharpe +0.16 [-0.35, +0.65] |
|  | lasso, collapsed | 0.1011 (0.1011) | 1.81 (1.81) | QLIKE +0.0047 [+0.0006, +0.0095], Sharpe -0.02 [-0.63, +0.58] |
| all_features | ridge, collapsed | 0.1119 (0.1119) | 1.19 (1.19) | QLIKE +0.0115 [+0.0032, +0.0210], Sharpe -0.48 [-1.26, +0.28] |
|  | lasso, collapsed | 0.1063 (0.1063) | 1.17 (1.17) | QLIKE +0.0066 [+0.0005, +0.0130], Sharpe -0.28 [-1.17, +0.58] |

Ridge − lasso Sharpe gap, full design minus collapsed: `live_feasible` -0.18 [-0.85, +0.46] (first pass -0.18 [-0.85, +0.46]); `all_features` +0.20 [-0.78, +1.07] (first pass +0.20 [-0.78, +1.07]).

## Gates and caveats

* `live_feasible` design re-run through the executor: ridge vs the de-duplicated stored arm: max rel diff 5.5e-13, PASS.
* `live_feasible` new design = the first pass's design minus the 12 session-edge columns (X, target, dates bit for bit): PASS, dropped 6 all-zero x_open, 6 x_close byte-copies of har_ma_k.
* `live_feasible` full ridge refit vs the de-duplicated stored arm: max rel diff 2.8e-12, rows > 1e-6: 0, PASS.
* `live_feasible` the de-duplicated stored ridge arm vs the first pass's stored arm: max rel diff 6.6e-12, rows > 1e-6: 0, the stored arms were fitted on different machines (I1).
* `live_feasible` this ridge refit vs the first pass's refit on this machine (old design): max rel diff 3.3e-12, rows > 1e-6: 0, PASS.
* `live_feasible` full lasso refit vs the de-duplicated stored arm: max rel diff 1.7e-03, rows > 1e-6: 245, forecast rows 1004–1249, differs, a warm-homotopy float path in one tune period: see the next rows.
* `live_feasible` the de-duplicated stored lasso arm vs the first pass's stored arm: max rel diff 1.7e-03, rows > 1e-6: 245, the stored arms were fitted on different machines (I1).
* `live_feasible` this lasso refit vs the first pass's refit on this machine (old design): max rel diff 1.7e-14, rows > 1e-6: 0, PASS.
* `live_feasible` spec continuous run of lasso here vs the stored arm: max rel diff 1.7e-03, rows > 1e-6: 271, this script's block run vs that continuous run 1.4e-03.
* `live_feasible` full elastic net refit vs the de-duplicated stored arm: max rel diff 1.4e-13, rows > 1e-6: 0, PASS.
* `live_feasible` the de-duplicated stored elastic net arm vs the first pass's stored arm: max rel diff 4.1e-14, rows > 1e-6: 0, the stored arms were fitted on different machines (I1).
* `live_feasible` this elastic net refit vs the first pass's refit on this machine (old design): max rel diff 5.6e-16, rows > 1e-6: 0, PASS.
* `live_feasible` LightGBM all columns, no mask, local vs the stored T10 run (cluster): max rel diff 6.0e-02, rows > 1e-6: 1469, not the platform: this walk gives the same LightGBM forecasts locally and on the cluster (next rows).
* `live_feasible` LightGBM all columns, masked, local vs the cluster's masked T10 walk: max rel diff 0.0e+00, rows > 1e-6: 0, PASS, bit-identical: the sweep's reference and the density table's LightGBM are one forecast.
* `live_feasible` per-window mask: kept columns per refit, local = cluster: PASS.
* `live_feasible` LightGBM, this walk without the mask on the cluster vs the stored T10 run (same cluster; forecast rows 1250..1468): max rel diff 3.6e-02, differs, the stored run is read from CSV text (15-16 significant digits).
* `live_feasible` XGBoost, this walk without the mask on the cluster vs the stored T10 run (same cluster; forecast rows 1250..1468): max rel diff 2.2e-16, PASS, the stored run is read from CSV text (15-16 significant digits).
* `live_feasible` random forest, this walk without the mask on the cluster vs the stored T10 run (same cluster; forecast rows 1250..1468): max rel diff 2.2e-16, PASS, the stored run is read from CSV text (15-16 significant digits).
* `live_feasible` LightGBM, this walk without the mask, local vs the cluster (block 5, bit for bit): max abs diff 0.0e+00, PASS.
* `live_feasible` tree run in six block chunks = one unchunked run (lgbm_screen_k4, bit for bit): max abs diff 0.0e+00, PASS.
* `live_feasible` ridge contributions sum to the forecast (de-dup + mask): max rel diff 2.0e-15, PASS.
* `live_feasible` lasso contributions sum to the forecast (de-dup + mask): max rel diff 5.5e-16, PASS.
* `live_feasible` elastic net contributions sum to the forecast (de-dup + mask): max rel diff 8.1e-16, PASS.
* `live_feasible` LightGBM contributions sum to the forecast (de-dup + mask): max rel diff 4.4e-15, PASS.
* `live_feasible` XGBoost contributions sum to the forecast (de-dup + mask): max rel diff 1.4e-06, PASS.
* `live_feasible` random forest contributions sum to the forecast (de-dup + mask): max rel diff 6.2e-12, PASS.
* `live_feasible` LightGBM contributions sum to the forecast (de-dup, no mask): max rel diff 7.6e-08, PASS.
* `live_feasible` XGBoost contributions sum to the forecast (de-dup, no mask): max rel diff 1.9e-06, PASS.
* `live_feasible` random forest contributions sum to the forecast (de-dup, no mask): max rel diff 8.8e-08, PASS.
* `live_feasible` per-day trade vs trade_1530 (Sharpe mid), every run (new and first pass): max rel diff 3.8e-15, PASS.
* `all_features` design re-run through the executor: ridge vs the de-duplicated stored arm: max rel diff 3.9e-11, PASS.
* `all_features` new design = the first pass's design minus the 12 session-edge columns (X, target, dates bit for bit): PASS, dropped 6 all-zero x_open, 6 x_close byte-copies of har_ma_k.
* `all_features` full ridge refit vs the de-duplicated stored arm: max rel diff 3.5e-11, rows > 1e-6: 0, PASS.
* `all_features` the de-duplicated stored ridge arm vs the first pass's stored arm: max rel diff 1.0e-11, rows > 1e-6: 0, the stored arms were fitted on different machines (I1).
* `all_features` this ridge refit vs the first pass's refit on this machine (old design): max rel diff 2.6e-12, rows > 1e-6: 0, PASS.
* `all_features` full lasso refit vs the de-duplicated stored arm: max rel diff 1.1e-03, rows > 1e-6: 23, forecast rows 1101–1123, differs, a warm-homotopy float path in one tune period: see the next rows.
* `all_features` the de-duplicated stored lasso arm vs the first pass's stored arm: max rel diff 1.1e-03, rows > 1e-6: 23, the stored arms were fitted on different machines (I1).
* `all_features` this lasso refit vs the first pass's refit on this machine (old design): max rel diff 5.8e-14, rows > 1e-6: 0, PASS.
* `all_features` spec continuous run of lasso here vs the stored arm: max rel diff 1.2e-03, rows > 1e-6: 71, this script's block run vs that continuous run 1.2e-03.
* `all_features` full elastic net refit vs the de-duplicated stored arm: max rel diff 6.5e-03, rows > 1e-6: 132, forecast rows 1118–1249, differs, a warm-homotopy float path in one tune period: see the next rows.
* `all_features` the de-duplicated stored elastic net arm vs the first pass's stored arm: max rel diff 1.5e-02, rows > 1e-6: 242, the stored arms were fitted on different machines (I1).
* `all_features` this elastic net refit vs the first pass's refit on this machine (old design): max rel diff 4.6e-04, rows > 1e-6: 20, differs.
* `all_features` spec continuous run of elastic net here vs the stored arm: max rel diff 1.1e-02, rows > 1e-6: 386, this script's block run vs that continuous run 1.0e-02.
* `all_features` LightGBM all columns, no mask, local vs the stored T10 run (cluster): max rel diff 5.6e-02, rows > 1e-6: 1469, not the platform: this walk gives the same LightGBM forecasts locally and on the cluster (next rows).
* `all_features` LightGBM all columns, masked, local vs the cluster's masked T10 walk: max rel diff 0.0e+00, rows > 1e-6: 0, PASS, bit-identical: the sweep's reference and the density table's LightGBM are one forecast.
* `all_features` per-window mask: kept columns per refit, local = cluster: PASS.
* `all_features` ridge contributions sum to the forecast (de-dup + mask): max rel diff 7.1e-14, PASS.
* `all_features` lasso contributions sum to the forecast (de-dup + mask): max rel diff 6.9e-16, PASS.
* `all_features` elastic net contributions sum to the forecast (de-dup + mask): max rel diff 1.0e-15, PASS.
* `all_features` LightGBM contributions sum to the forecast (de-dup + mask): max rel diff 5.7e-15, PASS.
* `all_features` XGBoost contributions sum to the forecast (de-dup + mask): max rel diff 1.4e-06, PASS.
* `all_features` random forest contributions sum to the forecast (de-dup + mask): max rel diff 1.3e-12, PASS.
* `all_features` LightGBM contributions sum to the forecast (de-dup, no mask): max rel diff 6.0e-08, PASS.
* `all_features` XGBoost contributions sum to the forecast (de-dup, no mask): max rel diff 1.6e-06, PASS.
* `all_features` random forest contributions sum to the forecast (de-dup, no mask): max rel diff 7.5e-08, PASS.
* `all_features` per-day trade vs trade_1530 (Sharpe mid), every run (new and first pass): max rel diff 4.2e-15, PASS.

* Only the linear arms' own identifiability mask and the trees' new per-window mask decide which columns a fit sees; the screen's eligibility rule (`identifiable` on the block's window) is the first pass's.
* LightGBM without the mask: this study's walk gives bit-identical forecasts locally and on the cluster, and reproduces the stored T10 XGBoost and random forest forecasts on the cluster to the CSV's rounding, but not the stored T10 LightGBM forecasts (the gate rows above: same cluster, same spec, same design); the first pass saw the same LightGBM gap and attributed it to the platform. With the mask, local and cluster LightGBM are bit-identical on every row. Every comparison inside the sweep uses local runs only; the density table uses the cluster's runs only, as in the first pass.
* Sample: 866 trade days. Sharpe differences between variants carry intervals 0.8–1.8 wide; QLIKE differences of about 0.005 are resolvable.

Files: `density.csv` (variant `de-dup + mask` / `de-dup, no mask`), `groups.csv`, `group_concentration.csv`, `sweep.csv`, `sweep_gaps.csv`, `sweep_gap_counts.csv`, `screen_picks.csv`, `collapse.csv`, `lgbm_references.csv`, `mask_counts.csv`, `gates.csv`, `numbers.json`; before / after: `before_after_claims.csv`, `before_after_density.csv`, `before_after_groups.csv`, `before_after_sweep.csv`, `before_after_gaps.csv`, `before_after_collapse.csv`, `before_after_picks.csv`, `before_after_session_edge.csv`; figures `density_curves.png`, `sweep_qlike_sharpe_vs_k.png`, `before_after_sweep_vs_k.png`.
