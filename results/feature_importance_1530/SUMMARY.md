# Feature importance of the 15:30 forecast (16:00 bar) -- the professor's four measures

Models: per-bar ridge and lasso (every-session re-solve, causal penalty), untuned per-bar LightGBM, XGBoost and random forest (refit every 10 sessions) -- the shipped configurations. Window: the 2000 sessions before each refit. Forecasts: 1,469 sessions 2018-06-25 .. 2024-04-30; 147 refits; each refit's CAUSAL held-out tail = the <= 10 sessions its model forecasts, up to the next refit (never seen in its fit). Buckets: live_feasible (the deck's), all_features, baseline = HAR + calendar. Script: `experiments/feature_importance_1530.py` (+ `_trees.py` for the refits, run on the cluster); tables in `results/feature_importance_1530/`, PDF `writeup/feature_importance_1530.pdf`.

## Key (design column names)
`har_ma_*` realized variance of the bar (the target's own HAR ladder; `*` = mean over the last 1, 5, 25, 125, 625, 3125 bars; `har_ma_*_x_close` = the same times the 16:00 close gate, identical at this bar); `adj_sumabsret_ma_*` absolute return; `adj_sumret_ma_*` signed return; `adj_sumret3/4_ma_*` 3rd / 4th-power returns; `adj_sumpret2_ma_*` upside squared returns; `adj_sumbipow_ma_*` bipower variation; `adj_sumautocov_ma_*` return autocovariance; `adj_sumvolume_ma_*` ES volume; `adj_numobs_ma_*` ES prints per bar; `adj_vix_ma_*`, `adj_vvix_ma_*`, `adj_vix3m_ma_*` the Cboe indices; `adj_fomc_*` FOMC flags / distances; `*_avail_ma_*` / `*_active_ma_*` is-present / is-nonzero flags of a source; calendar = `DOW_*`, `is_*`, `hour`, `days_to_opex`. all_features adds the constituent cross-section (`*_ewstock`, `*_vwstock`: moments, turnover, spreads, order-flow imbalance `ofi_*`), Cboe volume (`adj_voldemand_*`) and StockTwits (`adj_stocktwits_*`). A SERIES = all columns of one input (lags, flags, gates); a CLUSTER = columns every pair of which correlates >= 0.8 in absolute value (complete linkage).

## Measures
1. MDI / gain: LightGBM gain, XGBoost total_gain, random-forest impurity decrease; share per refit, averaged over refits.
2. Split count: splits on the column (LightGBM split, XGBoost weight, forest internal nodes); share per refit.
3. Permutation importance on the causal held-out tail, loss = QLIKE (the bar's realized variance vs the plain back-transform yhat^2 x diurnal baseline, no recalibration) and squared error in the fit space; 10 draws per unit and refit, the SAME draws for every model. P1 shuffles the unit's values among the tail's rows (the literal permutation; blind to slow inputs, which barely move within 10 sessions); P2 replaces them with rows drawn from that refit's own training window (causal: past data only). Units: columns, series, clusters (a group's columns move together). Interval = 95 % circular-block bootstrap over refits (block 6).
4. SHAP: TreeSHAP (path-dependent; LightGBM / XGBoost native, shap 0.51 TreeExplainer for the forest) on the same tail rows; mean |phi| share (and signed mean in the CSVs). Linear: beta (x - window mean) = the linear SHAP with the window as background.
Linear extras: |beta x sd| (standardized weight) and drop-column importance (re-solve the anchor's window without the unit, penalty held).

## Library and data type
scikit-learn 1.9.0 (random forest), LightGBM 4.6.0, XGBoost 3.2.0, shap 0.51.0; the permutation and drop-column loops are written out (sklearn's permutation_importance cannot score walk-forward models on their own tails). Every design column is a float. What MDI's cardinality bias keys on is the number of distinct values in each 2000-session window (median over refits): HAR + calendar 34 columns = 12 continuous (> 500 values), 1 with 26-500, 0 with 3-25, 11 binary, 10 constant at 16:00; live-feasible 244 columns = 87 continuous (> 500 values), 21 with 26-500, 15 with 3-25, 26 binary, 95 constant at 16:00; all features 640 columns = 273 continuous (> 500 values), 91 with 26-500, 93 with 3-25, 40 binary, 143 constant at 16:00. So the design is dominated by continuous inputs, but it is NOT all continuous: the calendar dummies and many flag moving averages are binary or few-valued -- exactly the mix where MDI's cardinality bias can bite.

## Does MDI's cardinality bias matter here? Yes, at the column level.
- Rank correlation of a column's importance with its number of distinct values (non-constant columns, tree models): live_feasible MDI 0.84..0.86, split count 0.83..0.87, SHAP 0.82..0.85; permutation P1 0.05..0.17, P2 0.05..0.24. all_features: MDI 0.83..0.90, permutation P2 -0.06..0.17.
- Noise probes (a second fit on every 3rd refit with three pure-noise columns appended): the CONTINUOUS noise column gets a median MDI rank of 32 (LightGBM), 37 (XGBoost), 22 (random forest) among the 247 live_feasible columns -- above 76 %, 68 %, 86 % of the real columns the model uses; by split count the forest ranks it 3 (above 99 % of used real columns). The BINARY noise column ranks 111, 113, 110 by MDI. Permutation P2 gives the continuous probe +0.02 [-0.51, +0.56]e-3, +0.30 [+0.07, +0.56]e-3, -0.37 [-0.74, +0.01]e-3 dQLIKE, against 327e-3 for `har_ma_*` (LightGBM).
- TreeSHAP (path-dependent) is not immune: it credits the continuous probe with a median rank of 45, 54, 44 (above 65 %, 56 %, 71 % of used real columns): it splits the fitted trees, including their fits to noise.
- At the SERIES level the bias washes out of the headline: every tree measure puts `har_ma_*` first. It matters for everything below the leader.
- What permutation says instead (series in the top 5 by MDI or split count that permutation P2 ranks outside its top 10; permutation rank in brackets): live-feasible LightGBM: `adj_sumret3_ma_*` (15); live-feasible XGBoost: `adj_sumret3_ma_*` (18); live-feasible random forest: `adj_sumret3_ma_*` (13), `adj_vix_ma_*` (16); all features LightGBM: `adj_sellturnover_vwstock_ma_*` (35), `adj_stocktwits_sentiment_ma_*` (49), `adj_sumret3_ma_*` (28); all features XGBoost: `adj_sumautocov_ma_*` (11); all features random forest: `adj_voldemand_spx_open_and_close_ma_*` (47), `adj_voldemand_all_open_only_ma_*` (36), `adj_voldemand_spx_open_only_ma_*` (43). These are series the trees split on often that rank low when scrambled on the held-out tail.

## Top series per measure -- live_feasible (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 23 %; `adj_sumabsret_ma_*` 12 %; `adj_sumret_ma_*` 9 % |
| ridge | perm P1 dMSE | `har_ma_*` +69 %; `adj_sumabsret_ma_*` +15 %; `adj_sumret_ma_*` +10 % |
| ridge | perm P2 dMSE | `har_ma_*` +145 %; `adj_sumabsret_ma_*` +42 %; `adj_sumret4_ma_*` +15 % |
| ridge | perm P2 dQLIKE | `adj_sumabsret_ma_*` +26865 %; `adj_sumret4_ma_*` +1534 %; `adj_sumret_ma_*` +724 % |
| ridge | drop-col dMSE | `har_ma_*` +15 %; `calendar` +2 %; `adj_sumvolume_ma_*` +2 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 29 %; `adj_sumabsret_ma_*` 15 %; `adj_sumret_ma_*` 11 % |
| lasso | abs(beta x sd) | `har_ma_*` 50 %; `adj_sumret_ma_*` 10 %; `adj_vix_ma_*` 7 % |
| lasso | perm P1 dMSE | `har_ma_*` +173 %; `adj_sumret_ma_*` +8 %; `calendar` +3 % |
| lasso | perm P2 dMSE | `har_ma_*` +328 %; `adj_vix_ma_*` +18 %; `adj_vix3m_ma_*` +12 % |
| lasso | perm P2 dQLIKE | `har_ma_*` +294 %; `adj_vix3m_ma_*` +66 %; `adj_vix_ma_*` +25 % |
| lasso | drop-col dMSE | `har_ma_*` +23 %; `adj_sumret_ma_*` +3 %; `calendar` +2 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 52 %; `adj_sumret_ma_*` 11 %; `adj_vix_ma_*` 6 % |
| LightGBM | MDI / gain | `har_ma_*` 62 %; `adj_sumret_ma_*` 8 %; `adj_sumabsret_ma_*` 5 % |
| LightGBM | split count | `har_ma_*` 17 %; `adj_sumret_ma_*` 12 %; `adj_sumvolume_ma_*` 8 % |
| LightGBM | perm P1 dQLIKE | `har_ma_*` +130 %; `calendar` +6 %; `adj_sumret_ma_*` +6 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +268 %; `calendar` +8 %; `adj_sumret_ma_*` +7 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +261 %; `calendar` +6 %; `adj_sumvolume_ma_*` +4 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 56 %; `adj_sumret_ma_*` 10 %; `adj_sumvolume_ma_*` 6 % |
| XGBoost | MDI / gain | `har_ma_*` 73 %; `adj_sumret_ma_*` 6 %; `adj_sumabsret_ma_*` 6 % |
| XGBoost | split count | `har_ma_*` 27 %; `adj_sumret_ma_*` 13 %; `adj_sumvolume_ma_*` 8 % |
| XGBoost | perm P1 dQLIKE | `har_ma_*` +123 %; `adj_sumret_ma_*` +5 %; `calendar` +5 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +252 %; `adj_sumvolume_ma_*` +7 %; `calendar` +6 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +259 %; `adj_sumret_ma_*` +7 %; `adj_sumvolume_ma_*` +6 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 63 %; `adj_sumret_ma_*` 10 %; `adj_sumvolume_ma_*` 7 % |
| random forest | MDI / gain | `har_ma_*` 67 %; `adj_sumabsret_ma_*` 6 %; `adj_sumret_ma_*` 5 % |
| random forest | split count | `har_ma_*` 11 %; `adj_numobs_ma_*` 7 %; `adj_sumret_ma_*` 7 % |
| random forest | perm P1 dQLIKE | `har_ma_*` +144 %; `calendar` +4 %; `adj_sumret_ma_*` +4 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +331 %; `calendar` +5 %; `adj_sumvolume_ma_*` +5 % |
| random forest | perm P2 dMSE | `har_ma_*` +292 %; `adj_sumret_ma_*` +5 %; `adj_sumvolume_ma_*` +4 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 64 %; `adj_sumret_ma_*` 7 %; `adj_sumabsret_ma_*` 6 % |

## Top series per measure -- all_features (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 11 %; `adj_sumabsret_ma_*` 5 %; `adj_sumabsret_vwstock_ma_*` 5 % |
| ridge | perm P1 dMSE | `har_ma_*` +63 %; `adj_sumabsret_ma_*` +7 %; `adj_sumabsret_vwstock_ma_*` +7 % |
| ridge | perm P2 dMSE | `har_ma_*` +142 %; `adj_sumabsret_ma_*` +22 %; `adj_sumabsret_vwstock_ma_*` +18 % |
| ridge | perm P2 dQLIKE | `adj_sumautocov_ma_*` +4051120 %; `adj_spread_vwstock_ma_*` +57340 %; `adj_voldemand_spx_open_only_ma_*` +3749 % |
| ridge | drop-col dMSE | `har_ma_*` +17 %; `calendar` +2 %; `adj_sumvolume_ma_*` +2 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 15 %; `adj_sumabsret_ma_*` 6 %; `adj_sumabsret_vwstock_ma_*` 5 % |
| lasso | abs(beta x sd) | `har_ma_*` 46 %; `adj_sumabsret_vwstock_ma_*` 10 %; `adj_sumret_ma_*` 9 % |
| lasso | perm P1 dMSE | `har_ma_*` +145 %; `adj_sumabsret_vwstock_ma_*` +8 %; `adj_sumret_ma_*` +7 % |
| lasso | perm P2 dMSE | `har_ma_*` +290 %; `adj_sumabsret_vwstock_ma_*` +18 %; `adj_vix_ma_*` +15 % |
| lasso | perm P2 dQLIKE | `har_ma_*` +259 %; `adj_vix3m_ma_*` +85 %; `adj_sumbipow_ewstock_ma_*` +48 % |
| lasso | drop-col dMSE | `har_ma_*` +27 %; `adj_sumret_ma_*` +2 %; `adj_sumabsret_vwstock_ma_*` +1 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 46 %; `adj_sumabsret_vwstock_ma_*` 10 %; `adj_sumret_ma_*` 9 % |
| LightGBM | MDI / gain | `har_ma_*` 57 %; `adj_sumret_ma_*` 5 %; `adj_sumabsret_ma_*` 4 % |
| LightGBM | split count | `har_ma_*` 12 %; `adj_sumret_ma_*` 6 %; `adj_sumvolume_ma_*` 4 % |
| LightGBM | perm P1 dQLIKE | `har_ma_*` +111 %; `calendar` +5 %; `adj_sumret_ma_*` +4 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +237 %; `calendar` +6 %; `adj_sumret_ma_*` +4 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +235 %; `calendar` +4 %; `adj_sumret_ma_*` +3 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 47 %; `adj_sumret_ma_*` 6 %; `adj_sumvolume_ma_*` 4 % |
| XGBoost | MDI / gain | `har_ma_*` 69 %; `adj_sumret_ma_*` 4 %; `adj_sumabsret_ma_*` 4 % |
| XGBoost | split count | `har_ma_*` 21 %; `adj_sumret_ma_*` 8 %; `adj_sumvolume_ma_*` 5 % |
| XGBoost | perm P1 dQLIKE | `har_ma_*` +107 %; `adj_sumret_ma_*` +4 %; `calendar` +4 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +224 %; `adj_sumret_ma_*` +4 %; `calendar` +4 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +231 %; `adj_sumret_ma_*` +4 %; `adj_sumvolume_ma_*` +3 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 55 %; `adj_sumret_ma_*` 7 %; `adj_sumvolume_ma_*` 5 % |
| random forest | MDI / gain | `har_ma_*` 62 %; `adj_sumabsret_ma_*` 5 %; `adj_sumret_ma_*` 3 % |
| random forest | split count | `har_ma_*` 4 %; `adj_voldemand_all_open_and_close_ma_*` 3 %; `adj_voldemand_spx_open_and_close_ma_*` 3 % |
| random forest | perm P1 dQLIKE | `har_ma_*` +128 %; `calendar` +2 %; `adj_sumret_ma_*` +2 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +290 %; `calendar` +3 %; `adj_sumabsret_ma_*` +3 % |
| random forest | perm P2 dMSE | `har_ma_*` +252 %; `calendar` +1 %; `adj_sumvolume_ma_*` +1 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 58 %; `adj_sumabsret_ma_*` 5 %; `adj_sumret_ma_*` 4 % |

## Top series per measure -- baseline (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 81 %; `calendar` 19 % |
| ridge | perm P1 dMSE | `har_ma_*` +280 %; `calendar` +8 % |
| ridge | perm P2 dMSE | `har_ma_*` +501 %; `calendar` +9 % |
| ridge | perm P2 dQLIKE | `har_ma_*` +537 %; `calendar` +10 % |
| ridge | drop-col dMSE | `har_ma_*` +247 %; `calendar` +5 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 90 %; `calendar` 10 % |
| lasso | abs(beta x sd) | `har_ma_*` 84 %; `calendar` 16 % |
| lasso | perm P1 dMSE | `har_ma_*` +276 %; `calendar` +7 % |
| lasso | perm P2 dMSE | `har_ma_*` +495 %; `calendar` +8 % |
| lasso | perm P2 dQLIKE | `har_ma_*` +539 %; `calendar` +9 % |
| lasso | drop-col dMSE | `har_ma_*` +244 %; `calendar` +4 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 91 %; `calendar` 9 % |
| LightGBM | MDI / gain | `har_ma_*` 96 %; `calendar` 4 % |
| LightGBM | split count | `har_ma_*` 87 %; `calendar` 13 % |
| LightGBM | perm P1 dQLIKE | `har_ma_*` +225 %; `calendar` +8 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +483 %; `calendar` +8 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +426 %; `calendar` +6 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 91 %; `calendar` 9 % |
| XGBoost | MDI / gain | `har_ma_*` 97 %; `calendar` 3 % |
| XGBoost | split count | `har_ma_*` 82 %; `calendar` 18 % |
| XGBoost | perm P1 dQLIKE | `har_ma_*` +219 %; `calendar` +7 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +457 %; `calendar` +7 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +444 %; `calendar` +5 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 93 %; `calendar` 7 % |
| random forest | MDI / gain | `har_ma_*` 95 %; `calendar` 5 % |
| random forest | split count | `har_ma_*` 82 %; `calendar` 18 % |
| random forest | perm P1 dQLIKE | `har_ma_*` +232 %; `calendar` +7 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +528 %; `calendar` +7 % |
| random forest | perm P2 dMSE | `har_ma_*` +453 %; `calendar` +5 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 93 %; `calendar` 7 % |

## Ranking agreement
Spearman rank correlation of the refit-averaged importance vectors (eligible units = non-constant in at least one window).

Within a tree model (series / column level): live-feasible LightGBM MDI~SHAP 0.89/0.97, MDI~perm P2 0.58/0.20, SHAP~perm P2 0.69/0.21; live-feasible XGBoost MDI~SHAP 0.96/0.97, MDI~perm P2 0.69/0.41, SHAP~perm P2 0.80/0.42; live-feasible random forest MDI~SHAP 0.84/0.98, MDI~perm P2 0.55/0.25, SHAP~perm P2 0.77/0.24; all features LightGBM MDI~SHAP 0.87/0.97, MDI~perm P2 0.22/0.06, SHAP~perm P2 0.25/0.03; all features XGBoost MDI~SHAP 0.85/0.97, MDI~perm P2 0.54/0.29, SHAP~perm P2 0.57/0.28; all features random forest MDI~SHAP 0.86/0.98, MDI~perm P2 0.22/0.12, SHAP~perm P2 0.30/0.13.

Across models, same measure (series level, range over pairs): live-feasible SHAP tree~tree 0.88..0.94, tree~linear 0.51..0.76; perm P2 dMSE tree~linear 0.18..0.66; all features SHAP tree~tree 0.83..0.89, tree~linear 0.33..0.49; perm P2 dMSE tree~linear 0.04..0.45.

## Correlated groups (the dilution caveat)
Joint permutation of a cluster vs the sum of its columns' separate permutations (P2, dMSE x 1e-3, joint vs sum), live_feasible cluster C22 (`har_ma_1 har_ma_5 har_ma_1_x_close har_ma_5_x_close`): ridge 33.8 vs 19.2; lasso 100.3 vs 63.3; LightGBM 112.4 vs 29.5; XGBoost 106.6 vs 39.3; random forest 151.9 vs 35.2. Joint > sum = the columns substitute for each other and permuting one at a time understates the group (dilution); joint < sum = offsetting weights (permuting one breaks an offset the other carries).

## Top-5 stability and regimes
`har_ma_*` is in the per-refit top 5 of every measure of every model in at least 58 % of the 147 refits (live_feasible; lowest: ridge drop-col dMSE) and at least 57 % (all_features; lowest: ridge drop-col dMSE). Top series of each measure that is NOT `har_ma_*`: live_feasible:ridge:perm_p2_qlike -> sumabsret, all_features:ridge:perm_p2_qlike -> sumautocov.
By calendar year (2018: 13, 2019: 26, 2020: 25, 2021: 25, 2022: 25, 2023: 25, 2024: 8 refits) and by causal VIX tercile (VIX at 15:30 vs the 1/3, 2/3 quantiles of the previous 2000 sessions' 15:30 VIX; low 28, mid 38, high 81 refits -- the trailing window includes the calm 2010s, so 2018-2024 is mostly 'high'): Spearman of a stratum's series ranking with the full-sample ranking, median over strata and models (live_feasible / all_features): built-in 0.97 / 0.93 by year, 0.99 / 0.99 by VIX tercile; SHAP 0.95 / 0.83 and 0.98 / 0.93; permutation P2 0.71 / 0.56 and 0.88 / 0.70. The built-in and SHAP rankings barely move across regimes; the permutation ranking below the leader moves more (it is measured on 10-session tails). Tables `stability_by_stratum.csv`, PDF Table 10.

## Linear models: the QLIKE permutation numbers are dominated by single tails
The linear forecasts are unbounded: moving an extreme input (a crash day's 4th-power or absolute return) onto another row can push the fit-space forecast through zero, and QLIKE of yhat^2 x baseline explodes. Example: live_feasible ridge, top P1 dQLIKE `adj_sumret4_ma_*` mean 52.9 [0.0027, 1.6e+02], median refit 0.00094, one refit carries 100 % of the sum. The all_features ridge is the extreme case: its intact tail QLIKE averages 2.54 against 0.122..0.152 for every other model x bucket (the known rare extreme values of the plain back-transform). Handled explicitly: every perm / drop table carries the median refit, the share of refits with a positive change and the largest single-refit share; for the linear models the figures and the cross-model comparisons use the squared-error versions (dMSE in the fit space, no blow-up). Trees cannot extrapolate (their forecasts stay inside the training range of the target), so their QLIKE numbers are well behaved.

## Tuned trees
The causally tuned trees' own extracts (`results/linear_subsection_trees_tuned/importance/`: gain per refit, TreeSHAP rows at 16:00): `har_ma_*` leads both MDI and SHAP in 9 of 9 bucket x model arms; shares tuned vs untuned in `tuned_trees_series.csv` and PDF Table 13. Split count and permutation were not computed for the tuned trees (their per-refit chosen configurations would have to be refitted).

## Gates
- Refits vs the stored forecasts: XGBoost and random forest reproduce them (largest gap 1.2e-03, above 1e-9 on 2 XGBoost and 1 forest refits of the 3 x 147; mean gap 3.0e-06); LightGBM reproduces them bit for bit on HAR + calendar (gap 4.4e-16) but not on live_feasible / all_features: mean |gap| 0.0092 / 0.0093, max 0.056 / 0.058, correlation 0.9996 / 0.9995 (same library version, same params, seed and thread count; the refit is deterministic run to run -- the cluster and a laptop give the same gap -- so the stored LightGBM runs differ in something the input file does not carry; XGBoost on the same input reproduces to 1e-16). The LightGBM importance is that of the refit.
- The design: out-of-sample stamps, targets and column names equal the stored runs'; the linear coefficient captures equal the design rows exactly and the stored research forecasts to <= 7e-11 (relative).
- Drop-column anchors: the re-solve on the window before the tail reproduces the captured coefficients (ridge <= 2e-10; lasso <= 5e-04, above 1e-6 on 0 / 8 / 5 of 147 refits (HAR + calendar / live_feasible / all_features); the re-solve masks constant and duplicate columns of its own window, the walk keeps a between-tune mask). The all_features lasso refits 75..124 (penalty 0.001, 115-120 active columns) ran on the cluster (`cluster/slurm/submit_featimp_linear.sh`), the rest locally; the 53 parts partition the 147 refits (checked).
- Every model of a bucket saw the identical permutation draws (md5 per refit equal across the five models, 147/147).
- TreeSHAP additivity: sum of phi + expected value = forecast to <= 3e-06.

## Files
- `by_measure/imp_<bucket>_<model>_<measure>.csv` (one per measure x model x bucket; rows = columns, series and clusters: value, 95 % interval over refits, rank, top-5 share, % of loss, median refit, largest single-refit share, by-year and by-VIX-tercile means, SHAP signed mean)
- `series_level_all.csv`, `rank_corr_within_model.csv`, `rank_corr_across_models.csv`, `top5_stability.csv`, `stability_by_stratum.csv`, `unique_values.csv`, `cardinality_vs_measure.csv`, `noise_probes.csv`, `clusters.csv`, `cluster_dilution.csv`, `tuned_trees_series.csv`, `gates.csv`
- figures: `fig_ranked_<bucket>.png` (ranked bars per measure), `fig_rank_heatmap_<bucket>_<level>.png` (ranks across measures), `fig_cardinality_live_feasible.png`, `fig_top5_stability_live_feasible.png`
- cluster twins: `cluster/slurm/{ship_featimp_carc.sh, submit_featimp.sh, featimp_pack.sbatch, featimp_collect.sbatch}`, `cluster/featimp_tasks*.txt` (tree refits); `cluster/slurm/{ship_featimp_linear_carc.sh, submit_featimp_linear.sh, featimp_linear.sbatch}`, `cluster/featimp_linear_tasks*.txt` (the slow lasso drop-column refits)
