# Feature importance of the 15:30 forecast (16:00 bar) -- the professor's four measures -- re-run on the de-duplicated per-bar design, every tree fit window-masked

## What changed from the first pass (`results/feature_importance_1530/`)
- The design (commit 47f7f9c): the per-bar design no longer carries the twelve HAR x open / close session-edge interactions. At 16:00 the six `har_ma_k_x_open` columns were all zero and the six `har_ma_k_x_close` columns exact copies of `har_ma_1 .. har_ma_3125` (checked bit for bit, gate below). Columns baseline 34 -> 22, live_feasible 244 -> 232, all_features 640 -> 628.
- The trees (user decision 2026-09-29, commit f9a19b6): every tree refit and probe fit uses only the columns `src/models/window_mask.window_keep` keeps on its own 2000-session training window (constant columns and exact copies of an earlier kept column removed) -- the per-bar linear arms' identifiability mask applied to the trees. MDI / gain, split count and TreeSHAP are computed on the kept columns and mapped back to all p columns with 0 for a dropped column (the model never saw it); permuting a dropped column leaves the forecast unchanged (0, exactly). Kept columns per refit (min / median / max over refits and tree models): baseline 17 / 17 / 17 of 22; live_feasible 136 / 138 / 145 of 232; all_features 360 / 369 / 378 of 628. The linear models are unchanged by the mask: their identifiability mask already removed constant columns and copies (including every session-edge column) at every tune.
- Everything else is the first pass's: the same 147 refits (every 10 sessions), the same causal held-out tails, the same permutation generator and seeds (by bucket and refit), the same five models, parameters, seeds and three designs; one thread per fit. So before / after differs only by the design (and, for the trees, the mask) -- and by the permutation draws of most units: the stream is laid out unit by unit (every unit's P1 draws, then every unit's P2 draws), and the old design's twelve extra columns were its last twelve, so a column's P1 draws are identical in both runs while the P2 draws and the series / cluster units' draws are fresh draws of the same scheme. The ridge fits are the same problem in both runs (coefficients equal to 1e-10), so the ridge rows of the before / after tables measure that Monte Carlo noise alone. The intermediate rung -- de-dup, no mask -- is `results/feature_importance_1530_dedup_nomask/` (same code, FEATIMP_WINDOW_MASK=0).
- Cadence: the campaign's trees now refit every session (`yhat_subtree_daily_*`). At an importance refit the model is identical under either cadence (the same 2000-session window, the same configuration), so the importance recorded at every 10th refit is the daily-refit model's importance on those days (checked on the stored runs, gate below).

Models: per-bar ridge and lasso (every-session re-solve, causal penalty), untuned per-bar LightGBM, XGBoost and random forest (refit every 10 sessions) -- the shipped configurations. Window: the 2000 sessions before each refit. Forecasts: 1,469 sessions 2018-06-25 .. 2024-04-30; 147 refits; each refit's CAUSAL held-out tail = the <= 10 sessions its model forecasts, up to the next refit (never seen in its fit). Buckets: live_feasible (the deck's), all_features, baseline = HAR + calendar. Scripts: `experiments/feature_importance_1530_dedup.py` (the re-run's roots and stages) driving the first pass's `experiments/feature_importance_1530.py` (+ `_trees.py` for the refits, run on the cluster; the per-window mask behind FEATIMP_WINDOW_MASK=1); tables in `results/feature_importance_1530_dedup/`, PDF `writeup/feature_importance_1530_dedup.pdf`.

## Key (design column names)
`har_ma_*` realized variance of the bar (the target's own HAR ladder; `*` = mean over the last 1, 5, 25, 125, 625, 3125 bars; the per-bar design no longer carries the `har_ma_*_x_open` / `_x_close` session-edge interactions); `adj_sumabsret_ma_*` absolute return; `adj_sumret_ma_*` signed return; `adj_sumret3/4_ma_*` 3rd / 4th-power returns; `adj_sumpret2_ma_*` upside squared returns; `adj_sumbipow_ma_*` bipower variation; `adj_sumautocov_ma_*` return autocovariance; `adj_sumvolume_ma_*` ES volume; `adj_numobs_ma_*` ES prints per bar; `adj_vix_ma_*`, `adj_vvix_ma_*`, `adj_vix3m_ma_*` the Cboe indices; `adj_fomc_*` FOMC flags / distances; `*_avail_ma_*` / `*_active_ma_*` is-present / is-nonzero flags of a source; calendar = `DOW_*`, `is_*`, `hour`, `days_to_opex`. all_features adds the constituent cross-section (`*_ewstock`, `*_vwstock`: moments, turnover, spreads, order-flow imbalance `ofi_*`), Cboe volume (`adj_voldemand_*`) and StockTwits (`adj_stocktwits_*`). A SERIES = all columns of one input (lags, flags, gates); a CLUSTER = columns every pair of which correlates >= 0.8 in absolute value (complete linkage).

## Measures
1. MDI / gain: LightGBM gain, XGBoost total_gain, random-forest impurity decrease; share per refit, averaged over refits.
2. Split count: splits on the column (LightGBM split, XGBoost weight, forest internal nodes); share per refit.
3. Permutation importance on the causal held-out tail, loss = QLIKE (the bar's realized variance vs the plain back-transform yhat^2 x diurnal baseline, no recalibration) and squared error in the fit space; 10 draws per unit and refit, the SAME draws for every model. P1 shuffles the unit's values among the tail's rows (the literal permutation; blind to slow inputs, which barely move within 10 sessions); P2 replaces them with rows drawn from that refit's own training window (causal: past data only). Units: columns, series, clusters (a group's columns move together). Interval = 95 % circular-block bootstrap over refits (block 6).
4. SHAP: TreeSHAP (path-dependent; LightGBM / XGBoost native, shap 0.51 TreeExplainer for the forest) on the same tail rows; mean |phi| share (and signed mean in the CSVs). Linear: beta (x - window mean) = the linear SHAP with the window as background.
Linear extras: |beta x sd| (standardized weight) and drop-column importance (re-solve the anchor's window without the unit, penalty held).

## Library and data type
scikit-learn 1.9.0 (random forest), LightGBM 4.6.0, XGBoost 3.2.0, shap 0.51.0; the permutation and drop-column loops are written out (sklearn's permutation_importance cannot score walk-forward models on their own tails). Every design column is a float. What MDI's cardinality bias keys on is the number of distinct values in each 2000-session window (median over refits): HAR + calendar 22 columns = 6 continuous (> 500 values), 1 with 26-500, 0 with 3-25, 11 binary, 4 constant at 16:00; live-feasible 232 columns = 81 continuous (> 500 values), 21 with 26-500, 15 with 3-25, 26 binary, 89 constant at 16:00; all features 628 columns = 267 continuous (> 500 values), 91 with 26-500, 93 with 3-25, 40 binary, 137 constant at 16:00. So the design is dominated by continuous inputs, but it is NOT all continuous: the calendar dummies and many flag moving averages are binary or few-valued -- exactly the mix where MDI's cardinality bias can bite.

## Does MDI's cardinality bias matter here? Yes, at the column level.
- Rank correlation of a column's importance with its number of distinct values (non-constant columns, tree models): live_feasible MDI 0.85..0.87, split count 0.84..0.88, SHAP 0.83..0.86; permutation P1 0.04..0.19, P2 0.04..0.33. all_features: MDI 0.82..0.82, permutation P2 -0.06..0.17.
- Noise probes (a second fit on every 3rd refit with three pure-noise columns appended): the CONTINUOUS noise column gets a median MDI rank of 31 (LightGBM), 35 (XGBoost), 19 (random forest) among the 235 live_feasible columns -- above 76 %, 69 %, 87 % of the real columns the model uses; by split count the forest ranks it 5 (above 97 % of used real columns). The BINARY noise column ranks 106, 106, 105 by MDI. Permutation P2 gives the continuous probe +0.15 [-0.50, +0.78]e-3, +0.27 [-0.01, +0.63]e-3, -0.14 [-0.48, +0.17]e-3 dQLIKE, against 266e-3 for `har_ma_*` (LightGBM).
- TreeSHAP (path-dependent) is not immune: it credits the continuous probe with a median rank of 40, 50, 43 (above 66 %, 58 %, 71 % of used real columns): it splits the fitted trees, including their fits to noise.
- At the SERIES level: `har_ma_*` is the leader of 52 of 54 tree bucket x model x measure rows; not of live-feasible random forest split (leader `numobs`), all features random forest split (leader `voldemand_spx_open_and_close`). The cardinality bias matters for everything below the leader.
- What permutation says instead (series in the top 5 by MDI or split count that permutation P2 ranks outside its top 10; permutation rank in brackets): live-feasible LightGBM: `adj_sumret3_ma_*` (16); live-feasible XGBoost: `adj_sumret3_ma_*` (18); live-feasible random forest: `adj_sumret3_ma_*` (11), `adj_vix_ma_*` (12); all features LightGBM: `adj_sellturnover_vwstock_ma_*` (36), `adj_stocktwits_sentiment_ma_*` (49), `adj_sumret3_ma_*` (42); all features XGBoost: none; all features random forest: `adj_voldemand_spx_open_and_close_ma_*` (34), `adj_sumret3_ewstock_ma_*` (29), `adj_voldemand_all_open_only_ma_*` (39). These are series the trees split on often that rank low when scrambled on the held-out tail.

## Top series per measure -- live_feasible (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 23 %; `adj_sumabsret_ma_*` 12 %; `adj_sumret_ma_*` 9 % |
| ridge | perm P1 dMSE | `har_ma_*` +71 %; `adj_sumabsret_ma_*` +15 %; `adj_sumret_ma_*` +10 % |
| ridge | perm P2 dMSE | `har_ma_*` +144 %; `adj_sumabsret_ma_*` +44 %; `adj_sumret4_ma_*` +13 % |
| ridge | perm P2 dQLIKE | `adj_sumabsret_ma_*` +83528 %; `adj_sumret4_ma_*` +34953 %; `adj_sumret_ma_*` +6841 % |
| ridge | drop-col dMSE | `har_ma_*` +15 %; `calendar` +2 %; `adj_sumvolume_ma_*` +2 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 29 %; `adj_sumabsret_ma_*` 15 %; `adj_sumret_ma_*` 11 % |
| lasso | abs(beta x sd) | `har_ma_*` 50 %; `adj_sumret_ma_*` 10 %; `adj_vix_ma_*` 7 % |
| lasso | perm P1 dMSE | `har_ma_*` +174 %; `adj_sumret_ma_*` +8 %; `calendar` +3 % |
| lasso | perm P2 dMSE | `har_ma_*` +330 %; `adj_vix_ma_*` +16 %; `adj_vix3m_ma_*` +13 % |
| lasso | perm P2 dQLIKE | `adj_vix3m_ma_*` +8868 %; `adj_sumret4_ma_*` +534 %; `har_ma_*` +288 % |
| lasso | drop-col dMSE | `har_ma_*` +23 %; `adj_sumret_ma_*` +3 %; `calendar` +2 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 52 %; `adj_sumret_ma_*` 11 %; `adj_vix_ma_*` 6 % |
| LightGBM | MDI / gain | `har_ma_*` 57 %; `adj_sumabsret_ma_*` 8 %; `adj_sumret_ma_*` 8 % |
| LightGBM | split count | `har_ma_*` 15 %; `adj_sumret_ma_*` 12 %; `adj_sumvolume_ma_*` 9 % |
| LightGBM | perm P1 dQLIKE | `har_ma_*` +104 %; `calendar` +6 %; `adj_sumret_ma_*` +6 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +219 %; `calendar` +8 %; `adj_sumret_ma_*` +8 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +220 %; `calendar` +6 %; `adj_sumvolume_ma_*` +5 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 52 %; `adj_sumret_ma_*` 10 %; `adj_sumvolume_ma_*` 7 % |
| XGBoost | MDI / gain | `har_ma_*` 69 %; `adj_sumabsret_ma_*` 8 %; `adj_sumret_ma_*` 6 % |
| XGBoost | split count | `har_ma_*` 24 %; `adj_sumret_ma_*` 13 %; `adj_sumvolume_ma_*` 9 % |
| XGBoost | perm P1 dQLIKE | `har_ma_*` +103 %; `calendar` +5 %; `adj_sumret_ma_*` +5 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +210 %; `adj_sumvolume_ma_*` +8 %; `adj_sumret_ma_*` +6 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +225 %; `adj_sumret_ma_*` +7 %; `adj_sumvolume_ma_*` +7 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 59 %; `adj_sumret_ma_*` 10 %; `adj_sumvolume_ma_*` 8 % |
| random forest | MDI / gain | `har_ma_*` 67 %; `adj_sumabsret_ma_*` 6 %; `adj_sumret_ma_*` 5 % |
| random forest | split count | `adj_numobs_ma_*` 8 %; `adj_sumret_ma_*` 8 %; `har_ma_*` 7 % |
| random forest | perm P1 dQLIKE | `har_ma_*` +141 %; `calendar` +4 %; `adj_sumret_ma_*` +4 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +320 %; `calendar` +6 %; `adj_sumret_ma_*` +5 % |
| random forest | perm P2 dMSE | `har_ma_*` +284 %; `adj_sumret_ma_*` +5 %; `adj_sumvolume_ma_*` +4 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 64 %; `adj_sumret_ma_*` 7 %; `adj_sumabsret_ma_*` 6 % |

## Top series per measure -- all_features (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 11 %; `adj_sumabsret_ma_*` 5 %; `adj_sumabsret_vwstock_ma_*` 5 % |
| ridge | perm P1 dMSE | `har_ma_*` +69 %; `adj_sumabsret_vwstock_ma_*` +8 %; `adj_sumabsret_ma_*` +7 % |
| ridge | perm P2 dMSE | `har_ma_*` +142 %; `adj_sumabsret_ma_*` +22 %; `adj_sumret4_ma_*` +18 % |
| ridge | perm P2 dQLIKE | `adj_voldemand_spx_open_only_ma_*` +59024 %; `adj_stocktwits_sentiment_ma_*` +4255 %; `adj_sumret3_ma_*` +3480 % |
| ridge | drop-col dMSE | `har_ma_*` +17 %; `calendar` +2 %; `adj_sumvolume_ma_*` +2 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 15 %; `adj_sumabsret_ma_*` 6 %; `adj_sumabsret_vwstock_ma_*` 5 % |
| lasso | abs(beta x sd) | `har_ma_*` 46 %; `adj_sumabsret_vwstock_ma_*` 10 %; `adj_sumret_ma_*` 9 % |
| lasso | perm P1 dMSE | `har_ma_*` +156 %; `adj_sumabsret_vwstock_ma_*` +9 %; `adj_sumret_ma_*` +7 % |
| lasso | perm P2 dMSE | `har_ma_*` +294 %; `adj_sumabsret_vwstock_ma_*` +17 %; `adj_vix_ma_*` +15 % |
| lasso | perm P2 dQLIKE | `adj_effspread_vwstock_ma_*` +569 %; `har_ma_*` +259 %; `adj_sumpret2_vwstock_ma_*` +73 % |
| lasso | drop-col dMSE | `har_ma_*` +27 %; `adj_sumret_ma_*` +2 %; `adj_sumabsret_vwstock_ma_*` +1 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 46 %; `adj_sumabsret_vwstock_ma_*` 10 %; `adj_sumret_ma_*` 9 % |
| LightGBM | MDI / gain | `har_ma_*` 52 %; `adj_sumabsret_ma_*` 6 %; `adj_sumret_ma_*` 5 % |
| LightGBM | split count | `har_ma_*` 10 %; `adj_sumret_ma_*` 6 %; `adj_sumvolume_ma_*` 4 % |
| LightGBM | perm P1 dQLIKE | `har_ma_*` +97 %; `calendar` +5 %; `adj_sumret_ma_*` +4 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +200 %; `calendar` +6 %; `adj_sumret_ma_*` +3 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +204 %; `calendar` +4 %; `adj_sumvolume_ma_*` +3 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 44 %; `adj_sumret_ma_*` 6 %; `adj_sumvolume_ma_*` 5 % |
| XGBoost | MDI / gain | `har_ma_*` 64 %; `adj_sumabsret_ma_*` 7 %; `adj_sumret_ma_*` 4 % |
| XGBoost | split count | `har_ma_*` 19 %; `adj_sumret_ma_*` 8 %; `adj_sumvolume_ma_*` 5 % |
| XGBoost | perm P1 dQLIKE | `har_ma_*` +95 %; `calendar` +4 %; `adj_sumret_ma_*` +3 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +187 %; `calendar` +5 %; `adj_sumvolume_ma_*` +5 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +193 %; `adj_sumvolume_ma_*` +4 %; `calendar` +3 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 51 %; `adj_sumret_ma_*` 7 %; `adj_sumvolume_ma_*` 6 % |
| random forest | MDI / gain | `har_ma_*` 62 %; `adj_sumabsret_ma_*` 5 %; `adj_sumret_ma_*` 3 % |
| random forest | split count | `adj_voldemand_spx_open_and_close_ma_*` 3 %; `har_ma_*` 3 %; `adj_voldemand_all_open_and_close_ma_*` 3 % |
| random forest | perm P1 dQLIKE | `har_ma_*` +134 %; `calendar` +3 %; `adj_sumabsret_ma_*` +2 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +283 %; `calendar` +4 %; `adj_sumabsret_ma_*` +3 % |
| random forest | perm P2 dMSE | `har_ma_*` +246 %; `calendar` +2 %; `adj_sumvolume_ma_*` +1 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 58 %; `adj_sumabsret_ma_*` 5 %; `adj_sumret_ma_*` 4 % |

## Top series per measure -- baseline (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 81 %; `calendar` 19 % |
| ridge | perm P1 dMSE | `har_ma_*` +287 %; `calendar` +8 % |
| ridge | perm P2 dMSE | `har_ma_*` +501 %; `calendar` +9 % |
| ridge | perm P2 dQLIKE | `har_ma_*` +544 %; `calendar` +9 % |
| ridge | drop-col dMSE | `har_ma_*` +247 %; `calendar` +5 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 90 %; `calendar` 10 % |
| lasso | abs(beta x sd) | `har_ma_*` 84 %; `calendar` 16 % |
| lasso | perm P1 dMSE | `har_ma_*` +284 %; `calendar` +7 % |
| lasso | perm P2 dMSE | `har_ma_*` +495 %; `calendar` +7 % |
| lasso | perm P2 dQLIKE | `har_ma_*` +544 %; `calendar` +9 % |
| lasso | drop-col dMSE | `har_ma_*` +244 %; `calendar` +4 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 91 %; `calendar` 9 % |
| LightGBM | MDI / gain | `har_ma_*` 95 %; `calendar` 5 % |
| LightGBM | split count | `har_ma_*` 85 %; `calendar` 15 % |
| LightGBM | perm P1 dQLIKE | `har_ma_*` +224 %; `calendar` +8 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +488 %; `calendar` +8 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +430 %; `calendar` +7 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 90 %; `calendar` 10 % |
| XGBoost | MDI / gain | `har_ma_*` 97 %; `calendar` 3 % |
| XGBoost | split count | `har_ma_*` 79 %; `calendar` 21 % |
| XGBoost | perm P1 dQLIKE | `har_ma_*` +218 %; `calendar` +6 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +457 %; `calendar` +6 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +444 %; `calendar` +4 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 93 %; `calendar` 7 % |
| random forest | MDI / gain | `har_ma_*` 95 %; `calendar` 5 % |
| random forest | split count | `har_ma_*` 76 %; `calendar` 24 % |
| random forest | perm P1 dQLIKE | `har_ma_*` +235 %; `calendar` +6 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +527 %; `calendar` +6 % |
| random forest | perm P2 dMSE | `har_ma_*` +453 %; `calendar` +5 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 92 %; `calendar` 8 % |

## Ranking agreement
Spearman rank correlation of the refit-averaged importance vectors (eligible units = non-constant in at least one window).

Within a tree model (series / column level): live-feasible LightGBM MDI~SHAP 0.89/0.97, MDI~perm P2 0.59/0.19, SHAP~perm P2 0.71/0.22; live-feasible XGBoost MDI~SHAP 0.95/0.97, MDI~perm P2 0.69/0.46, SHAP~perm P2 0.80/0.45; live-feasible random forest MDI~SHAP 0.82/0.98, MDI~perm P2 0.62/0.21, SHAP~perm P2 0.77/0.21; all features LightGBM MDI~SHAP 0.87/0.97, MDI~perm P2 0.21/0.06, SHAP~perm P2 0.27/0.05; all features XGBoost MDI~SHAP 0.88/0.97, MDI~perm P2 0.52/0.26, SHAP~perm P2 0.56/0.25; all features random forest MDI~SHAP 0.86/0.98, MDI~perm P2 0.17/0.03, SHAP~perm P2 0.31/0.03.

Across models, same measure (series level, range over pairs): live-feasible SHAP tree~tree 0.89..0.95, tree~linear 0.56..0.76; perm P2 dMSE tree~linear 0.50..0.85; all features SHAP tree~tree 0.83..0.89, tree~linear 0.32..0.49; perm P2 dMSE tree~linear 0.01..0.49.

## Correlated groups (the dilution caveat)
Joint permutation of a cluster vs the sum of its columns' separate permutations (P2, dMSE x 1e-3, joint vs sum), live_feasible cluster C21 (`har_ma_1 har_ma_5`): ridge 33.5 vs 18.9; lasso 100.0 vs 64.0; LightGBM 92.3 vs 46.6; XGBoost 91.4 vs 51.6; random forest 147.4 vs 80.9. Joint > sum = the columns substitute for each other and permuting one at a time understates the group (dilution); joint < sum = offsetting weights (permuting one breaks an offset the other carries).

## Top-5 stability and regimes
`har_ma_*` is in the per-refit top 5 of every measure of every model in at least 58 % of the 147 refits (live_feasible; lowest: ridge drop-col dMSE) and at least 57 % (all_features; lowest: ridge drop-col dMSE). Top series of each measure that is NOT `har_ma_*`: live_feasible:ridge:perm_p2_qlike -> sumabsret, live_feasible:lasso:perm_p2_qlike -> vix3m, live_feasible:rf:split -> numobs, all_features:ridge:perm_p2_qlike -> voldemand_spx_open_only, all_features:lasso:perm_p2_qlike -> effspread_vwstock, all_features:rf:split -> voldemand_spx_open_and_close.
By calendar year (2018: 13, 2019: 26, 2020: 25, 2021: 25, 2022: 25, 2023: 25, 2024: 8 refits) and by causal VIX tercile (VIX at 15:30 vs the 1/3, 2/3 quantiles of the previous 2000 sessions' 15:30 VIX; low 28, mid 38, high 81 refits -- the trailing window includes the calm 2010s, so 2018-2024 is mostly 'high'): Spearman of a stratum's series ranking with the full-sample ranking, median over strata and models (live_feasible / all_features): built-in 0.98 / 0.94 by year, 0.99 / 0.99 by VIX tercile; SHAP 0.96 / 0.83 and 0.98 / 0.93; permutation P2 0.78 / 0.57 and 0.89 / 0.66. The built-in and SHAP rankings barely move across regimes; the permutation ranking below the leader moves more (it is measured on 10-session tails). Tables `stability_by_stratum.csv`, PDF Table 10.

## Linear models: the QLIKE permutation numbers are dominated by single tails
The linear forecasts are unbounded: moving an extreme input (a crash day's 4th-power or absolute return) onto another row can push the fit-space forecast through zero, and QLIKE of yhat^2 x baseline explodes. Example: live_feasible ridge, top P1 dQLIKE `adj_sumabsret_ma_*` mean 8.35 [-0.0068, 25], median refit 0.0094, one refit carries 100 % of the sum. The all_features ridge is the extreme case: its intact tail QLIKE averages 2.54 against 0.122..0.152 for every other model x bucket (the known rare extreme values of the plain back-transform). Handled explicitly: every perm / drop table carries the median refit, the share of refits with a positive change and the largest single-refit share; for the linear models the figures and the cross-model comparisons use the squared-error versions (dMSE in the fit space, no blow-up). Trees cannot extrapolate (their forecasts stay inside the training range of the target), so their QLIKE numbers are well behaved.

## Tuned trees
The causally tuned trees' own extracts (random search, refit every 10; importance per refit and TreeSHAP rows at 16:00, from the masked tuned runs): `har_ma_*` leads both MDI and SHAP in 9 of 9 bucket x model arms; shares tuned vs untuned in `tuned_trees_series.csv`. Split count and permutation were not computed for the tuned trees.

## Gates
- Design (`gates_design.csv`): baseline p 22 (want 22; old 34), no `_x_` column: True, old minus new = the 12 session-edge columns: True; live_feasible p 232 (want 232; old 244), no `_x_` column: True, old minus new = the 12 session-edge columns: True; all_features p 628 (want 628; old 640), no `_x_` column: True, old minus new = the 12 session-edge columns: True; the same out-of-sample stamps, the same targets and every other column bit for bit as the first pass's design: True; the old x_open columns all zero and the old x_close columns = `har_ma_k` bit for bit: True. Stamps, targets and column names also equal the stored de-dup tree runs'.
- Refits vs the stored forecasts (the reproduction gate, no-mask root vs the stored de-dup T10 runs, unmasked, agent B): bit for bit (<= 1e-9) in 7 of 9 bucket x model arms; not in live-feasible LightGBM (max |gap| 0.056, mean 0.0094, correlation 0.9995, 147 of 147 refits above 1e-9); all features LightGBM (max |gap| 0.054, mean 0.0095, correlation 0.9995, 147 of 147 refits above 1e-9). The LightGBM gap is the first pass's again, now with ONE thread on both sides (the thread count is ruled out), and forcing LightGBM's column-wise or row-wise histograms leaves it unchanged; the refit is deterministic run to run and the cluster and a laptop give the same gap, so the stored LightGBM runs differ in something the input file does not carry. The LightGBM importance is that of the refit.
- The mask's effect on the forecasts (masked refits vs the same unmasked stored runs): max |gap| HAR + calendar LightGBM 0.138, HAR + calendar XGBoost 0.199, HAR + calendar random forest 0.234, live-feasible LightGBM 0.107, live-feasible XGBoost 0.196, live-feasible random forest 0.309, all features LightGBM 0.135, all features XGBoost 0.172, all features random forest 0.338; correlation 0.9974..0.9993.
- Masked refits vs the canonical masked tree tables (`results/spxw_pnl/yhat_subtree_<model>_<bucket>`, agent H): bit for bit (<= 1e-9) in 6 of 9 arms; otherwise max |gap| live-feasible LightGBM 0.0639, all features LightGBM 0.0503, all features random forest 0.00228 (`gates_canonical.csv`).
- Cadence identity -- nomask (9 bucket x model arms, 147 importance refits each): every-session vs every-10 run on the importance-refit days, forecast max |gap| 0.0e+00, native importance max |gap| 0.0e+00; this study's refit vs the every-session run: forecast equal (<= 1e-9) in 7 of 9 arms (not in live-feasible LightGBM: 3.4e-02, all features LightGBM: 4.1e-02); mask (9 bucket x model arms, 147 importance refits each): every-session vs every-10 run on the importance-refit days, forecast max |gap| 0.0e+00, native importance max |gap| 0.0e+00; this study's refit vs the every-session run: forecast equal (<= 1e-9) in 6 of 9 arms (not in live-feasible LightGBM: 6.4e-02, all features LightGBM: 5.0e-02, all features random forest: 2.3e-03).
- The linear coefficient captures (re-run on the new design through the spec's own class): coefficient x row + intercept = the forecast; capture vs the stored de-dup research forecasts (agent A, `results/linear_subsection_dedup/arms_hoffman2`): ridge max rel HAR + calendar 6e-11, live-feasible 5e-13, all features 4e-11; lasso HAR + calendar 6.5e-15, live-feasible 1.7e-03 (273 of 1469 rows above 1e-9), all features 1.2e-03 (71 of 1469 rows above 1e-9). The lasso gaps are the warm-homotopy float path of the spec's lasso: it moves with the CPU architecture (agent A: a re-run on the same architecture is bit-identical, another machine moves 16:00 forecasts by 5e-4..8e-2), and the stored arms ran on the other cluster; the lasso importance is that of this capture. Drop-column anchors reproduce the captures: ridge <= 2e-10, lasso <= 2e-02 (above 1e-6 on 0 / 10 / 9 of 147 refits, HAR + calendar / live-feasible / all features). The all_features lasso refits 75..124 (penalty 0.001) ran on the cluster (`cluster/slurm/submit_featimp_dedup.sh linear`), the rest locally; the parts partition the 147 refits (checked by the merge).
- Every model of a bucket saw the identical permutation draws (md5 per refit equal across the five models): True. TreeSHAP additivity: sum of phi + expected value = forecast to <= 3e-06.

## Before / after: first pass (old design, trees unmasked) vs de-dup + mask
Tables `before_after_ranks_<bucket>_<model>.csv` (every unit of every measure and level: value and rank old / new [/ de-dup no mask for the trees], the rank among the units common to both runs and its change), `before_after_spearman.csv`, `before_after_movers.csv`, `before_after_har_ma.csv`, `before_after_har_ma_columns.csv`, `before_after_clusters.csv` (clusters are renumbered per run; matched by their members once the session-edge columns are removed from the old cluster).

- `har_ma_*` in the trees, live_feasible: LightGBM MDI 62 -> 57 %, SHAP 56 -> 52 %, perm P2 dQLIKE +268 -> +219 % of tail loss (the old `_x_close` copies carried 29 % of the old har_ma column MDI); XGBoost MDI 73 -> 69 %, SHAP 63 -> 59 %, perm P2 dQLIKE +252 -> +210 % of tail loss (the old `_x_close` copies carried 22 % of the old har_ma column MDI); random forest MDI 67 -> 67 %, SHAP 64 -> 64 %, perm P2 dQLIKE +331 -> +320 % of tail loss (the old `_x_close` copies carried 49 % of the old har_ma column MDI).
- `har_ma_*` in the trees, all_features: LightGBM MDI 57 -> 52 %, SHAP 47 -> 44 %, perm P2 dQLIKE +237 -> +200 % of tail loss (the old `_x_close` copies carried 22 % of the old har_ma column MDI); XGBoost MDI 69 -> 64 %, SHAP 55 -> 51 %, perm P2 dQLIKE +224 -> +187 % of tail loss (the old `_x_close` copies carried 22 % of the old har_ma column MDI); random forest MDI 62 -> 62 %, SHAP 58 -> 58 %, perm P2 dQLIKE +290 -> +283 % of tail loss (the old `_x_close` copies carried 54 % of the old har_ma column MDI).
- The draw-noise yardstick: the ridge fits are the same problem in both runs, so its `har_ma_*` change is the Monte Carlo of fresh permutation draws alone: perm P2 dMSE moves by at most 1.7 points of % tail loss over the three buckets (abs(beta x sd), SHAP and drop-column unchanged).
- Spearman old vs new (series level, common units, every bucket x measure): trees median 0.90 (min 0.66), linear median 1.00 (min 0.87); trees old vs de-dup no mask 0.94 (min 0.57), no mask vs mask 0.95 (min 0.66).
- Spearman old vs new (cluster level, common units, every bucket x measure): trees median 0.93 (min 0.59), linear median 1.00 (min 0.77); trees old vs de-dup no mask 0.97 (min 0.56), no mask vs mask 0.97 (min 0.62).
- Spearman old vs new (column level, common units, every bucket x measure): trees median 0.94 (min 0.38), linear median 1.00 (min 0.79); trees old vs de-dup no mask 0.97 (min 0.42), no mask vs mask 0.94 (min 0.40).
- Series leader changed in 7 of 111 bucket x model x measure rows: live_feasible ridge perm P1 QL `sumret4` -> `sumabsret`; live_feasible lasso perm P2 QL `har_ma` -> `vix3m`; live_feasible random forest split `har_ma` -> `numobs`; all_features ridge perm P1 QL `sumret4_ewstock` -> `sumautocov`; all_features ridge perm P2 QL `sumautocov` -> `voldemand_spx_open_only`; all_features lasso perm P2 QL `har_ma` -> `effspread_vwstock`; all_features random forest split `har_ma` -> `voldemand_spx_open_and_close`.
- Clusters baseline: matched 19, removed (session-edge only) 6.
- Clusters live_feasible: matched 135, removed (session-edge only) 6.
- Clusters all_features: matched 230, removed (session-edge only) 6.

### `har_ma_*` once its duplicates are gone (series value old / de-dup no mask / new; permutation and drop-column as % of the model's tail loss; `x_close share` = the old `_x_close` copies' part of the old har_ma column total)

live_feasible:

| model | measure | old | no mask | new | x_close share (old) | rank old | rank new |
|---|---|---|---|---|---|---|---|
| ridge | abs(b sd) | 23% |  | 23% | 0% | 1 | 1 |
| ridge | SHAP | 29% |  | 29% | 0% | 1 | 1 |
| ridge | perm P2 MSE | +145% |  | +144% | 0% | 1 | 1 |
| ridge | drop MSE | +15% |  | +15% | 0% | 1 | 1 |
| lasso | abs(b sd) | 50% |  | 50% | 0% | 1 | 1 |
| lasso | SHAP | 52% |  | 52% | 0% | 1 | 1 |
| lasso | perm P2 MSE | +328% |  | +330% | 0% | 1 | 1 |
| lasso | drop MSE | +23% |  | +23% | 0% | 1 | 1 |
| LightGBM | MDI | 62% | 56% | 57% | 29% | 1 | 1 |
| LightGBM | split | 17% | 15% | 15% | 24% | 1 | 1 |
| LightGBM | SHAP | 56% | 52% | 52% | 27% | 1 | 1 |
| LightGBM | perm P2 QL | +268% | +219% | +219% | 15% | 1 | 1 |
| LightGBM | perm P2 MSE | +261% | +222% | +220% | 7% | 1 | 1 |
| XGBoost | MDI | 73% | 71% | 69% | 22% | 1 | 1 |
| XGBoost | split | 27% | 25% | 24% | 22% | 1 | 1 |
| XGBoost | SHAP | 63% | 61% | 59% | 22% | 1 | 1 |
| XGBoost | perm P2 QL | +252% | +223% | +210% | 12% | 1 | 1 |
| XGBoost | perm P2 MSE | +259% | +238% | +225% | 8% | 1 | 1 |
| random forest | MDI | 67% | 67% | 67% | 49% | 1 | 1 |
| random forest | split | 11% | 7% | 7% | 50% | 1 | 3 |
| random forest | SHAP | 64% | 64% | 64% | 50% | 1 | 1 |
| random forest | perm P2 QL | +331% | +321% | +320% | 49% | 1 | 1 |
| random forest | perm P2 MSE | +292% | +290% | +284% | 47% | 1 | 1 |

all_features:

| model | measure | old | no mask | new | x_close share (old) | rank old | rank new |
|---|---|---|---|---|---|---|---|
| ridge | abs(b sd) | 11% |  | 11% | 0% | 1 | 1 |
| ridge | SHAP | 15% |  | 15% | 0% | 1 | 1 |
| ridge | perm P2 MSE | +142% |  | +142% | 0% | 1 | 1 |
| ridge | drop MSE | +17% |  | +17% | 0% | 1 | 1 |
| lasso | abs(b sd) | 46% |  | 46% | 0% | 1 | 1 |
| lasso | SHAP | 46% |  | 46% | 0% | 1 | 1 |
| lasso | perm P2 MSE | +290% |  | +294% | 0% | 1 | 1 |
| lasso | drop MSE | +27% |  | +27% | 0% | 1 | 1 |
| LightGBM | MDI | 57% | 52% | 52% | 22% | 1 | 1 |
| LightGBM | split | 12% | 10% | 10% | 22% | 1 | 1 |
| LightGBM | SHAP | 47% | 44% | 44% | 22% | 1 | 1 |
| LightGBM | perm P2 QL | +237% | +198% | +200% | 9% | 1 | 1 |
| LightGBM | perm P2 MSE | +235% | +203% | +204% | 4% | 1 | 1 |
| XGBoost | MDI | 69% | 66% | 64% | 22% | 1 | 1 |
| XGBoost | split | 21% | 20% | 19% | 24% | 1 | 1 |
| XGBoost | SHAP | 55% | 53% | 51% | 24% | 1 | 1 |
| XGBoost | perm P2 QL | +224% | +203% | +187% | 13% | 1 | 1 |
| XGBoost | perm P2 MSE | +231% | +210% | +193% | 10% | 1 | 1 |
| random forest | MDI | 62% | 62% | 62% | 54% | 1 | 1 |
| random forest | split | 4% | 3% | 3% | 50% | 1 | 2 |
| random forest | SHAP | 58% | 58% | 58% | 53% | 1 | 1 |
| random forest | perm P2 QL | +290% | +286% | +283% | 54% | 1 | 1 |
| random forest | perm P2 MSE | +252% | +246% | +246% | 58% | 1 | 1 |

baseline:

| model | measure | old | no mask | new | x_close share (old) | rank old | rank new |
|---|---|---|---|---|---|---|---|
| ridge | abs(b sd) | 81% |  | 81% | 0% | 1 | 1 |
| ridge | SHAP | 90% |  | 90% | 0% | 1 | 1 |
| ridge | perm P2 MSE | +501% |  | +501% | 0% | 1 | 1 |
| ridge | drop MSE | +247% |  | +247% | 0% | 1 | 1 |
| lasso | abs(b sd) | 84% |  | 84% | 0% | 1 | 1 |
| lasso | SHAP | 91% |  | 91% | 0% | 1 | 1 |
| lasso | perm P2 MSE | +495% |  | +495% | 0% | 1 | 1 |
| lasso | drop MSE | +244% |  | +244% | 0% | 1 | 1 |
| LightGBM | MDI | 96% | 95% | 95% | 24% | 1 | 1 |
| LightGBM | split | 87% | 84% | 85% | 23% | 1 | 1 |
| LightGBM | SHAP | 91% | 90% | 90% | 23% | 1 | 1 |
| LightGBM | perm P2 QL | +483% | +490% | +488% | 8% | 1 | 1 |
| LightGBM | perm P2 MSE | +426% | +432% | +430% | 2% | 1 | 1 |
| XGBoost | MDI | 97% | 97% | 97% | 30% | 1 | 1 |
| XGBoost | split | 82% | 79% | 79% | 26% | 1 | 1 |
| XGBoost | SHAP | 93% | 93% | 93% | 28% | 1 | 1 |
| XGBoost | perm P2 QL | +457% | +457% | +457% | 18% | 1 | 1 |
| XGBoost | perm P2 MSE | +444% | +442% | +444% | 14% | 1 | 1 |
| random forest | MDI | 95% | 95% | 95% | 46% | 1 | 1 |
| random forest | split | 82% | 76% | 76% | 50% | 1 | 1 |
| random forest | SHAP | 93% | 92% | 92% | 47% | 1 | 1 |
| random forest | perm P2 QL | +528% | +531% | +527% | 45% | 1 | 1 |
| random forest | perm P2 MSE | +453% | +458% | +453% | 41% | 1 | 1 |

Column level, the dilution the copies caused (permutation P2 dQLIKE x 1e-3 of `har_ma_1` / `har_ma_5` alone, old -> new; old = its `_x_close` copy left intact): live-feasible LightGBM 34.1 -> 48.7 / 27.9 -> 44.1; live-feasible XGBoost 34.3 -> 49.7 / 34.2 -> 43.8; live-feasible random forest 25.9 -> 78.3 / 18.0 -> 62.9; all features LightGBM 33.4 -> 47.4 / 30.7 -> 41.0; all features XGBoost 36.0 -> 46.8 / 25.9 -> 38.3; all features random forest 18.6 -> 78.7 / 18.4 -> 61.9.

### Rows that moved most (series level, top-20 zone of either run, rank among common units old -> new)

live_feasible:
- ridge: perm P2 MSE: `fomc_until_inv` 16->17, `fomc_release` 17->16
- lasso: perm P2 MSE: `sumret3` 12->15, `numobs` 14->12
- LightGBM: MDI: `calendar` 9->11, `sumret` 2->3; SHAP: `sumabsret` 6->4, `calendar` 4->5; perm P2 QL: `fomc_since_inv` 9->15, `vix3m` 11->17; perm P2 MSE: `sumabsret` 16->5, `vix` 14->10
- XGBoost: MDI: `fomc_since_inv` 14->16, `sumpret2` 15->13; split: `numobs` 5->6, `sumret3` 6->5; SHAP: `fomc_until_inv` 14->12, `vix` 9->10; perm P2 QL: `sumautocov` 6->9, `numobs` 10->12; perm P2 MSE: `numobs` 8->16, `fomc_day` 12->17
- random forest: MDI: `calendar` 11->13, `sumret4` 12->11; split: `har_ma` 1->3, `numobs` 2->1; SHAP: `numobs` 9->10, `sumret4` 10->9; perm P2 QL: `vix3m` 7->13, `sumbipow` 11->7; perm P2 MSE: `sumpret2` 6->13, `vix3m` 7->14

all_features:
- ridge: perm P2 MSE: `sumbipow_vwstock` 21->18, `sumvolume` 17->19
- lasso: perm P2 MSE: `sumbipow_vwstock` 24->20, `sumautocov` 20->23
- LightGBM: MDI: `sumret` 2->3, `sumabsret` 3->2; split: `sumabsret` 18->9, `stocktwits_sentcount` 13->17; SHAP: `voldemand_spx_open_and_close` 28->20, `sumret3_ewstock` 18->22; perm P2 QL: `sumautocov` 9->40, `buyturnover_ewstock` 12->43; perm P2 MSE: `sumabsret` 33->4, `fomc_until_inv` 36->20
- XGBoost: MDI: `sumbipow` 20->15, `sumret3_ewstock` 16->19; split: `sumabsret` 8->5, `sumret3` 7->9; SHAP: `sumbipow` 17->14, `turnover_vwstock` 14->16; perm P2 QL: `stocktwits_sentcount` 28->14, `sellturnover_ewstock` 31->18; perm P2 MSE: `vix3m` 30->14, `voldemand_all_open_only` 18->31
- random forest: MDI: `stocktwits_sentcount` 17->20, `sumret3_ewstock` 19->16; split: `sumret2_ewstock` 36->12, `vvix` 39->20; SHAP: `buyturnover_ewstock` 15->18, `sumret3_ewstock` 18->15; perm P2 QL: `sumret3_ewstock` 10->29, `ofi_ewstock` 28->9; perm P2 MSE: `ofi_vwstock` 13->39, `vvix` 10->33

## Files
- `results/feature_importance_1530_dedup/by_measure/imp_<bucket>_<model>_<measure>.csv` and the first pass's tables (`series_level_all.csv`, `rank_corr_*.csv`, `top5_stability.csv`, `stability_by_stratum.csv`, `unique_values.csv`, `cardinality_vs_measure.csv`, `noise_probes.csv`, `clusters.csv`, `cluster_dilution.csv`, `tuned_trees_series.csv`, `gates.csv`) on the new design
- the re-run's own: `gates_design.csv`, `gates_cadence.csv`, `gates_canonical.csv`, `before_after_*.csv`, `fig_before_after_*.png`; the no-mask rung in `results/feature_importance_1530_dedup_nomask/` (same tables)
- scripts: `experiments/feature_importance_1530_dedup.py` (roots, stages, gates), `experiments/feature_importance_1530_dedup_compare.py` (before / after), `writeup/make_feature_importance_1530_tex.py --root results/feature_importance_1530_dedup --stem feature_importance_1530_dedup --dedup` (+ `writeup/feature_importance_1530_dedup_sections.py`)
- cluster twins: `cluster/slurm/{ship_featimp_dedup_carc.sh, submit_featimp_dedup.sh, featimp_dedup_pack.sbatch, featimp_dedup_collect.sbatch, featimp_dedup_linear.sbatch, pull_featimp_dedup_carc.sh}`, `cluster/featimp_dedup_tasks*.txt`, `cluster/featimp_dedup_linear_tasks.txt`
