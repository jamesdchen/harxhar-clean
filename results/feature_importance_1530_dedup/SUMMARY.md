# Feature importance of the 15:30 forecast (16:00 bar) -- the professor's four measures -- re-run on the de-duplicated per-bar design, every tree fit window-masked

## What changed from the first pass (`results/feature_importance_1530/`)
- The design (commit 47f7f9c): the per-bar design no longer carries the twelve HAR x open / close session-edge interactions. At 16:00 the six `har_ma_k_x_open` columns were all zero and the six `har_ma_k_x_close` columns exact copies of `har_ma_1 .. har_ma_3125` (checked bit for bit, gate below). Columns all_features 640 -> 628.
- The trees (user decision 2026-09-29, commit f9a19b6): every tree refit and probe fit uses only the columns `src/models/window_mask.window_keep` keeps on its own 2000-session training window (constant columns and exact copies of an earlier kept column removed) -- the per-bar linear arms' identifiability mask applied to the trees. MDI / gain, split count and TreeSHAP are computed on the kept columns and mapped back to all p columns with 0 for a dropped column (the model never saw it); permuting a dropped column leaves the forecast unchanged (0, exactly). Kept columns per refit (min / median / max over refits and tree models): all_features 360 / 369 / 378 of 628. The linear models are unchanged by the mask: their identifiability mask already removed constant columns and copies (including every session-edge column) at every tune.
- Everything else is the first pass's: the same 147 refits (every 10 sessions), the same causal held-out tails, the same permutation generator and seeds (by bucket and refit), the same five models, parameters, seeds and two designs; one thread per fit. So before / after differs only by the design (and, for the trees, the mask) -- and by the permutation draws of most units: the stream is laid out unit by unit (every unit's P1 draws, then every unit's P2 draws), and the old design's twelve extra columns were its last twelve, so a column's P1 draws are identical in both runs while the P2 draws and the series / cluster units' draws are fresh draws of the same scheme. The ridge fits are the same problem in both runs (coefficients equal to 1e-10), so the ridge rows of the before / after tables measure that Monte Carlo noise alone. The intermediate rung -- de-dup, no mask -- is `results/feature_importance_1530_dedup_nomask/` (same code, FEATIMP_WINDOW_MASK=0).
- Cadence: the campaign's trees now refit every session (`yhat_subtree_daily_*`). At an importance refit the model is identical under either cadence (the same 2000-session window, the same configuration), so the importance recorded at every 10th refit is the daily-refit model's importance on those days (checked on the stored runs, gate below).
- Settings changed since the first pass: XGBoost now runs xgb_opt_00 (was xgb_opt_06; 2026-10-07); the random forest now runs trial 255 of its 400-trial dev-span search (was the library defaults; 2026-10-08). For XGBoost and the random forest the before / after comparison mixes the settings change with the design and the mask.

Models: per-bar ridge and lasso (every-session re-solve, causal penalty), untuned per-bar LightGBM, XGBoost and random forest (refit every 10 sessions; settings below). Tree settings (read from the refit files): LightGBM: lgbm_opt_00 (392 rounds, 20 leaves, learning rate 0.0130, min child samples 8; the top entry of the 1998-2002 dev-span search); XGBoost: xgb_opt_00 (367 rounds, depth 5, learning rate 0.0123, min child weight 4.18; the top entry of the 1998-2002 dev-span search (since 2026-10-07; was the 7th-ranked xgb_opt_06)); random forest: trial 255 (485 trees, depth 20, min leaf 1, 90% of the features at a split, bootstrap of 41% of the rows; the top entry of the forest's 400-trial 1998-2002 dev-span search (since 2026-10-08; was the library defaults)). Leaf minimums are scaled from the pooled 24,000-row window to the 2,000-row window as in the per-bar spec. Window: the 2000 sessions before each refit. Forecasts: 1,469 sessions 2018-06-25 .. 2024-04-30; 147 refits; each refit's CAUSAL held-out tail = the <= 10 sessions its model forecasts, up to the next refit (never seen in its fit). Buckets: all_features. Scripts: `experiments/feature_importance_1530_dedup.py` (the re-run's roots and stages) driving the first pass's `experiments/feature_importance_1530.py` (+ `_trees.py` for the refits, run on the cluster; the per-window mask behind FEATIMP_WINDOW_MASK=1); tables in `results/feature_importance_1530_dedup/`, PDF `writeup/feature_importance_1530_dedup.pdf`.

## Key (design column names)
`har_ma_*` realized variance of the bar (the target's own HAR ladder; `*` = mean over the last 1, 5, 25, 125, 625, 3125 bars; the per-bar design no longer carries the `har_ma_*_x_open` / `_x_close` session-edge interactions); `adj_sumabsret_ma_*` absolute return; `adj_sumret_ma_*` signed return; `adj_sumret3/4_ma_*` 3rd / 4th-power returns; `adj_sumpret2_ma_*` upside squared returns; `adj_sumbipow_ma_*` bipower variation; `adj_sumautocov_ma_*` return autocovariance; `adj_sumvolume_ma_*` ES volume; `adj_numobs_ma_*` ES prints per bar; `adj_vix_ma_*`, `adj_vvix_ma_*`, `adj_vix3m_ma_*` the Cboe indices; `adj_fomc_*` FOMC flags / distances; `*_avail_ma_*` / `*_active_ma_*` is-present / is-nonzero flags of a source; calendar = `DOW_*`, `is_*`, `hour`, `days_to_opex`. all_features adds the constituent cross-section (`*_ewstock`, `*_vwstock`: moments, turnover, spreads, order-flow imbalance `ofi_*`), Cboe volume (`adj_voldemand_*`) and StockTwits (`adj_stocktwits_*`). A SERIES = all columns of one input (lags, flags, gates); a CLUSTER = columns every pair of which correlates >= 0.8 in absolute value (complete linkage).

## Measures
1. MDI / gain: LightGBM gain, XGBoost total_gain, random-forest impurity decrease; share per refit, averaged over refits.
2. Split count: splits on the column (LightGBM split, XGBoost weight, forest internal nodes); share per refit.
3. Permutation importance on the causal held-out tail, loss = QLIKE (the bar's realized variance vs the plain back-transform yhat^2 x diurnal baseline, no recalibration) and squared error in the fit space; 10 draws per unit and refit, the SAME draws for every model. P1 shuffles the unit's values among the tail's rows (the literal permutation; blind to slow inputs, which barely move within 10 sessions); P2 replaces them with rows drawn from that refit's own training window (causal: past data only). Units: columns, series, clusters (a group's columns move together). Interval = 95 % circular-block bootstrap over refits (block 6).
4. SHAP: TreeSHAP (path-dependent; LightGBM / XGBoost native, shap 0.51 TreeExplainer for the forest) on the same tail rows; mean |phi| share (and signed mean in the CSVs). Linear: beta (x - window mean) = the linear SHAP with the window as background.
Linear extras: |beta x sd| (standardized weight) and drop-column importance (re-solve the anchor's window without the unit, penalty held).

## Library and data type
scikit-learn 1.9.0 (random forest), LightGBM 4.6.0, XGBoost 3.2.0, shap 0.51.0; the permutation and drop-column loops are written out (sklearn's permutation_importance cannot score walk-forward models on their own tails). Every design column is a float. What MDI's cardinality bias keys on is the number of distinct values in each 2000-session window (median over refits): all features 628 columns = 267 continuous (> 500 values), 91 with 26-500, 93 with 3-25, 40 binary, 137 constant at 16:00. So the design is dominated by continuous inputs, but it is NOT all continuous: the calendar dummies and many flag moving averages are binary or few-valued -- exactly the mix where MDI's cardinality bias can bite.

## Does MDI's cardinality bias matter here? Yes, at the column level.
- Rank correlation of a column's importance with its number of distinct values (non-constant columns, tree models), all_features: MDI 0.82..0.82, split count 0.82..0.83, SHAP 0.79..0.80; permutation P1 -0.02..0.17, P2 -0.04..0.13.
- Noise probes (a second fit on every 3rd refit with three pure-noise columns appended): the CONTINUOUS noise column gets a median MDI rank of 73 (LightGBM), 88 (XGBoost), 52 (random forest) among the 631 all-features columns -- above 77 %, 74 %, 85 % of the real columns the model uses; by split count the forest ranks it 10 (above 97 % of used real columns). The BINARY noise column ranks 315, 310, 308 by MDI. Permutation P2 gives the continuous probe -0.07 [-0.40, +0.25]e-3, +0.04 [-0.33, +0.37]e-3, +0.09 [-0.07, +0.27]e-3 dQLIKE, against 249e-3 for `har_ma_*` (LightGBM).
- TreeSHAP (path-dependent) is not immune: it credits the continuous probe with a median rank of 82, 110, 96 (above 70 %, 64 %, 72 % of used real columns): it splits the fitted trees, including their fits to noise.
- At the SERIES level: `har_ma_*` is the leader of 12 of 12 tree bucket x model x measure rows. The cardinality bias matters for everything below the leader.
- What permutation says instead (series in the top 5 by MDI or split count that permutation P2 ranks outside its top 10; permutation rank in brackets): all features LightGBM: `adj_sellturnover_vwstock_ma_*` (36); all features XGBoost: `adj_sellturnover_vwstock_ma_*` (33); all features random forest: none. These are series the trees split on often that rank low when scrambled on the held-out tail.

## Top series per measure -- all_features (value = share for MDI/split/|beta sd|/SHAP; % of the model's tail loss for permutation / drop-column)

| model | measure | top 3 |
|---|---|---|
| ridge | abs(beta x sd) | `har_ma_*` 11 %; `adj_sumabsret_ma_*` 5 %; `adj_sumabsret_vwstock_ma_*` 5 % |
| ridge | perm P2 dMSE | `har_ma_*` +142 %; `adj_sumabsret_ma_*` +22 %; `adj_sumret4_ma_*` +18 % |
| ridge | drop-col dMSE | `har_ma_*` +17 %; `calendar` +2 %; `adj_sumvolume_ma_*` +2 % |
| ridge | SHAP mean abs(phi) | `har_ma_*` 15 %; `adj_sumabsret_ma_*` 6 %; `adj_sumabsret_vwstock_ma_*` 5 % |
| lasso | abs(beta x sd) | `har_ma_*` 46 %; `adj_sumabsret_vwstock_ma_*` 10 %; `adj_sumret_ma_*` 9 % |
| lasso | perm P2 dMSE | `har_ma_*` +294 %; `adj_sumabsret_vwstock_ma_*` +17 %; `adj_vix_ma_*` +15 % |
| lasso | drop-col dMSE | `har_ma_*` +27 %; `adj_sumret_ma_*` +2 %; `adj_sumabsret_vwstock_ma_*` +1 % |
| lasso | SHAP mean abs(phi) | `har_ma_*` 46 %; `adj_sumabsret_vwstock_ma_*` 10 %; `adj_sumret_ma_*` 9 % |
| LightGBM | MDI / gain | `har_ma_*` 52 %; `adj_sumabsret_ma_*` 6 %; `adj_sumret_ma_*` 5 % |
| LightGBM | perm P2 dQLIKE | `har_ma_*` +200 %; `calendar` +6 %; `adj_sumret_ma_*` +3 % |
| LightGBM | perm P2 dMSE | `har_ma_*` +204 %; `calendar` +4 %; `adj_sumvolume_ma_*` +3 % |
| LightGBM | SHAP mean abs(phi) | `har_ma_*` 44 %; `adj_sumret_ma_*` 6 %; `adj_sumvolume_ma_*` 5 % |
| XGBoost | MDI / gain | `har_ma_*` 54 %; `adj_sumabsret_ma_*` 7 %; `adj_sumret_ma_*` 5 % |
| XGBoost | perm P2 dQLIKE | `har_ma_*` +181 %; `calendar` +6 %; `adj_sumvolume_ma_*` +4 % |
| XGBoost | perm P2 dMSE | `har_ma_*` +175 %; `calendar` +3 %; `adj_sumvolume_ma_*` +3 % |
| XGBoost | SHAP mean abs(phi) | `har_ma_*` 47 %; `adj_sumret_ma_*` 7 %; `adj_sumabsret_ma_*` 5 % |
| random forest | MDI / gain | `har_ma_*` 59 %; `adj_sumabsret_ma_*` 5 %; `adj_sumret_ma_*` 3 % |
| random forest | perm P2 dQLIKE | `har_ma_*` +271 %; `adj_sumvolume_ma_*` +2 %; `calendar` +2 % |
| random forest | perm P2 dMSE | `har_ma_*` +249 %; `adj_sumvolume_ma_*` +2 %; `adj_sumabsret_ma_*` +2 % |
| random forest | SHAP mean abs(phi) | `har_ma_*` 61 %; `adj_sumabsret_ma_*` 5 %; `adj_sumret_ma_*` 3 % |

## Ranking agreement
Spearman rank correlation of the refit-averaged importance vectors (eligible units = non-constant in at least one window).

Within a tree model (series / column level): all features LightGBM MDI~SHAP 0.87/0.97, MDI~perm P2 0.21/0.06, SHAP~perm P2 0.27/0.05; all features XGBoost MDI~SHAP 0.87/0.97, MDI~perm P2 0.16/0.10, SHAP~perm P2 0.20/0.12; all features random forest MDI~SHAP 0.88/0.97, MDI~perm P2 0.09/0.21, SHAP~perm P2 0.29/0.22.

Across models, same measure (series level, range over pairs): all features SHAP tree~tree 0.84..0.97, tree~linear 0.33..0.49; perm P2 dMSE tree~linear -0.08..0.49.

## Correlated groups (the dilution caveat)
Joint permutation of a cluster vs the sum of its columns' separate permutations (P2, dMSE x 1e-3, joint vs sum), all-features cluster C49 (`har_ma_1 har_ma_5`): ridge 33.2 vs 18.3; lasso 85.7 vs 50.0; LightGBM 86.9 vs 45.6; XGBoost 76.6 vs 38.5; random forest 136.7 vs 72.3. Joint > sum = the columns substitute for each other and permuting one at a time understates the group (dilution); joint < sum = offsetting weights (permuting one breaks an offset the other carries).

## Top-5 stability and regimes
`har_ma_*` is in the per-refit top 5 of every measure of every model in at least 57 % of the 147 refits (all_features; lowest: ridge drop-col dMSE). Top series of each measure that is NOT `har_ma_*`: none.
By calendar year (2018: 13, 2019: 26, 2020: 25, 2021: 25, 2022: 25, 2023: 25, 2024: 8 refits) and by causal VIX tercile (VIX at 15:30 vs the 1/3, 2/3 quantiles of the previous 2000 sessions' 15:30 VIX; low 28, mid 38, high 81 refits -- the trailing window includes the calm 2010s, so 2018-2024 is mostly 'high'): Spearman of a stratum's series ranking with the full-sample ranking, median over strata and models (all_features): built-in 0.94 by year, 0.99 by VIX tercile; SHAP 0.83 and 0.95; permutation P2 0.51 and 0.72. The built-in and SHAP rankings barely move across regimes; the permutation ranking below the leader moves more (it is measured on 10-session tails). Tables `stability_by_stratum.csv`, PDF Table 10.

## Linear models: the QLIKE permutation numbers are dominated by single tails
The linear forecasts are unbounded: moving an extreme input (a crash day's 4th-power or absolute return) onto another row can push the fit-space forecast through zero, and QLIKE of yhat^2 x baseline explodes. Example: all-features ridge, top P1 dQLIKE `adj_sumautocov_ma_*` mean 492 [0.00084, 1.5e+03], median refit 0.0011, one refit carries 100 % of the sum. The all_features ridge is the extreme case: its intact tail QLIKE averages 2.54 against 0.124..0.126 for every other model x bucket (the known rare extreme values of the plain back-transform). Handled explicitly: every perm / drop table carries the median refit, the share of refits with a positive change and the largest single-refit share; for the linear models the figures and the cross-model comparisons use the squared-error versions (dMSE in the fit space, no blow-up). Trees cannot extrapolate (their forecasts stay inside the training range of the target), so their QLIKE numbers are well behaved.

## Tuned trees
The causally tuned trees' own extracts (random search, refit every 10; importance per refit and TreeSHAP rows at 16:00, from the masked tuned runs): `har_ma_*` leads both MDI and SHAP in 3 of 3 bucket x model arms; shares tuned vs untuned in `tuned_trees_series.csv`. Split count and permutation were not computed for the tuned trees.

## Gates
- Design (`gates_design.csv`): all_features p 628 (want 628; old 640), no `_x_` column: True, old minus new = the 12 session-edge columns: True; the same out-of-sample stamps, the same targets and every other column bit for bit as the first pass's design: True; the old x_open columns all zero and the old x_close columns = `har_ma_k` bit for bit: True. Stamps, targets and column names also equal the stored de-dup tree runs'.
- Refits vs the stored forecasts (the reproduction gate, no-mask root vs the stored de-dup T10 runs, unmasked, agent B): bit for bit (<= 1e-9) in 0 of 1 bucket x model arms; not in all features LightGBM (max |gap| 0.054, mean 0.0095, correlation 0.9995, 147 of 147 refits above 1e-9). The LightGBM gap is the first pass's again, now with ONE thread on both sides (the thread count is ruled out), and forcing LightGBM's column-wise or row-wise histograms leaves it unchanged; the refit is deterministic run to run and the cluster and a laptop give the same gap, so the stored LightGBM runs differ in something the input file does not carry. The LightGBM importance is that of the refit.
- The mask's effect on the forecasts (masked refits vs the same unmasked stored runs): max |gap| all features LightGBM 0.135; correlation 0.9974..0.9974.
- Masked refits vs the canonical masked tree tables (`results/spxw_pnl/yhat_subtree_<model>_<bucket>`, agent H): bit for bit (<= 1e-9) in 0 of 1 arms; otherwise max |gap| all features LightGBM 0.0503 (`gates_canonical.csv`).
- Cadence identity -- nomask (1 bucket x model arms, 147 importance refits each): every-session vs every-10 run on the importance-refit days, forecast max |gap| 0.0e+00, native importance max |gap| 0.0e+00; this study's refit vs the every-session run: forecast equal (<= 1e-9) in 0 of 1 arms (not in all features LightGBM: 4.1e-02); mask (1 bucket x model arms, 147 importance refits each): every-session vs every-10 run on the importance-refit days, forecast max |gap| 0.0e+00, native importance max |gap| 0.0e+00; this study's refit vs the every-session run: forecast equal (<= 1e-9) in 0 of 1 arms (not in all features LightGBM: 5.0e-02).
- XGBoost re-run on its new settings (xgb_opt_00; CPU not recorded; `gates_xgb10.csv`): asserted gates pass (permutation draws equal the stored lgbm / rf parts' (draw_md5, 147 refits): 147 / 147 equal to lgbm, 147 to rf; parts' params = xgb_config.json per_bar_params_train_win_2000, mask on, 1 thread: every part; TreeSHAP additivity (sum phi + E = forecast): <= 1e-5); reported: refits vs xgbtop T10 masked table (bit for bit, <= 1e-9): 145 of 147 refits bit for bit; differ: refits [136, 138]; mean 7.13e-05, corr 0.999996625994, n 1469. Its stored-run gates (no-mask reproduction, the mask's effect, canonical table, cadence) compare its old settings and are left out here.
- random forest re-run on its new settings (trial 255; CPU AMD EPYC 7513 32-Core Processor; `gates_rf255.csv`): asserted gates pass (default forest by the current code reproduces the stored default-forest parts (refits 0-1 of both input sets, every array, <= 1e-9): all_features: refits [np.int64(0), np.int64(1)] of trees_all_features_rf_000_002.npz -- max |refit - stored| over 18 arrays 0.00e+00 (bit for bit); fit threads 4, cpu b22-28.hpc.usc.edu|AMD EPYC 7542 32-Core Processor; per array: all 0 | all_features: refits [np.int64(0), np.int64(1)] of trees_all_features_rf_000_021.npz -- max |refit - stored| over 18 arrays 0.00e+00 (bit for bit); fit threads 4, cpu b22-28.hpc.usc.edu|AMD EPYC 7542 32-Core Processor; per array: all 0 | baseline: refits [np.int64(0), np.int64(1)] of trees_baseline_rf_000_003.npz -- max |refit - stored| over 18 arrays 0.00e+00 (bit for bit); fit threads 4, cpu b22-28.hpc.usc.edu|AMD EPYC 7542 32-Core Processor; per array: all 0 | baseline: refits [np.int64(0), np.int64(1)] of trees_baseline_rf_000_147.npz -- max |refit - stored| over 18 arrays 0.00e+00 (bit for bit); fit threads 4, cpu b22-28.hpc.usc.edu|AMD EPYC 7542 32-Core Processor; per array: all 0; permutation draws equal the stored lgbm / xgb parts' (draw_md5, 147 refits): 147 / 147 equal to lgbm, 147 / 147 equal to xgb; parts' params = trial 255 (RF255_PARAMS), mask on, tag rf255; every refit's CPU recorded: every part; CPU: AMD EPYC 7513 32-Core Processor 147; TreeSHAP additivity (sum phi + E = forecast): max 2e-14 (<= 1e-5)); reported: refits vs the per-bar campaign's masked T10 table on trial 255 (rf400; bit for bit, <= 1e-9): 146 of 147 refits within 1e-9 (max |gap| 1.4e-03); differ: refits [118]. Its stored-run gates (no-mask reproduction, the mask's effect, canonical table, cadence) compare its old settings and are left out here.
- The linear coefficient captures (re-run on the new design through the spec's own class): coefficient x row + intercept = the forecast; capture vs the stored de-dup research forecasts (agent A, `results/linear_subsection_dedup/arms_hoffman2`): ridge max rel all features 4e-11; lasso all features 1.2e-03 (71 of 1469 rows above 1e-9). The lasso gaps are the warm-homotopy float path of the spec's lasso: it moves with the CPU architecture (agent A: a re-run on the same architecture is bit-identical, another machine moves 16:00 forecasts by 5e-4..8e-2), and the stored arms ran on the other cluster; the lasso importance is that of this capture. Drop-column anchors reproduce the captures: ridge <= 3e-12, lasso <= 2e-02 (above 1e-6 on 9 of 147 refits, all features). The all_features lasso refits 75..124 (penalty 0.001) ran on the cluster (`cluster/slurm/submit_featimp_dedup.sh linear`), the rest locally; the parts partition the 147 refits (checked by the merge).
- Every model of a bucket saw the identical permutation draws (md5 per refit equal across the five models): True. TreeSHAP additivity: sum of phi + expected value = forecast to <= 4e-06.

## Before / after: first pass (old design, trees unmasked) vs de-dup + mask
Tables `before_after_ranks_<bucket>_<model>.csv` (every unit of every measure and level: value and rank old / new [/ de-dup no mask for the trees], the rank among the units common to both runs and its change), `before_after_spearman.csv`, `before_after_movers.csv`, `before_after_har_ma.csv`, `before_after_har_ma_columns.csv`, `before_after_clusters.csv` (clusters are renumbered per run; matched by their members once the session-edge columns are removed from the old cluster).
Settings changed since the first pass: XGBoost now runs xgb_opt_00 (was xgb_opt_06; 2026-10-07); the random forest now runs trial 255 of its 400-trial dev-span search (was the library defaults; 2026-10-08). For XGBoost and the random forest the before / after comparison mixes the settings change with the design and the mask.

- `har_ma_*` in the trees, all_features: LightGBM MDI 57 -> 52 %, SHAP 47 -> 44 %, perm P2 dQLIKE +237 -> +200 % of tail loss (the old `_x_close` copies carried 22 % of the old har_ma column MDI); XGBoost MDI 69 -> 54 %, SHAP 55 -> 47 %, perm P2 dQLIKE +224 -> +181 % of tail loss (the old `_x_close` copies carried 22 % of the old har_ma column MDI); random forest MDI 62 -> 59 %, SHAP 58 -> 61 %, perm P2 dQLIKE +290 -> +271 % of tail loss (the old `_x_close` copies carried 54 % of the old har_ma column MDI).
- The draw-noise yardstick: the ridge fits are the same problem in both runs, so its `har_ma_*` change is the Monte Carlo of fresh permutation draws alone: perm P2 dMSE moves by at most 0.8 points of % tail loss over the 1 buckets (abs(beta x sd), SHAP and drop-column unchanged).
- Spearman old vs new (series level, common units, every bucket x measure): trees median 0.76 (min 0.47), linear median 0.99 (min 0.87); trees old vs de-dup no mask 0.87 (min 0.57), no mask vs mask 0.77 (min 0.46).
- Spearman old vs new (cluster level, common units, every bucket x measure): trees median 0.68 (min 0.43), linear median 0.99 (min 0.77); trees old vs de-dup no mask 0.79 (min 0.56), no mask vs mask 0.68 (min 0.41).
- Spearman old vs new (column level, common units, every bucket x measure): trees median 0.43 (min 0.28), linear median 1.00 (min 0.79); trees old vs de-dup no mask 0.60 (min 0.42), no mask vs mask 0.49 (min 0.24).
- Series leader changed in 4 of 37 bucket x model x measure rows: all_features ridge perm P1 QL `sumret4_ewstock` -> `sumautocov`; all_features ridge perm P2 QL `sumautocov` -> `voldemand_spx_open_only`; all_features lasso perm P2 QL `har_ma` -> `effspread_vwstock`; all_features random forest split `har_ma` -> `voldemand_spx_open_and_close`.
- Clusters all_features: matched 230, removed (session-edge only) 6.

### `har_ma_*` once its duplicates are gone (series value old / de-dup no mask / new; permutation and drop-column as % of the model's tail loss; `x_close share` = the old `_x_close` copies' part of the old har_ma column total)

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
| XGBoost | MDI | 69% | 66% | 54% | 22% | 1 | 1 |
| XGBoost | split | 21% | 20% | 13% | 24% | 1 | 1 |
| XGBoost | SHAP | 55% | 53% | 47% | 24% | 1 | 1 |
| XGBoost | perm P2 QL | +224% | +203% | +181% | 13% | 1 | 1 |
| XGBoost | perm P2 MSE | +231% | +210% | +175% | 10% | 1 | 1 |
| random forest | MDI | 62% | 62% | 59% | 54% | 1 | 1 |
| random forest | split | 4% | 3% | 3% | 50% | 1 | 2 |
| random forest | SHAP | 58% | 58% | 61% | 53% | 1 | 1 |
| random forest | perm P2 QL | +290% | +286% | +271% | 54% | 1 | 1 |
| random forest | perm P2 MSE | +252% | +246% | +249% | 58% | 1 | 1 |

Column level, the dilution the copies caused (permutation P2 dQLIKE x 1e-3 of `har_ma_1` / `har_ma_5` alone, old -> new; old = its `_x_close` copy left intact): all features LightGBM 33.4 -> 47.4 / 30.7 -> 41.0; all features XGBoost 36.0 -> 45.5 / 25.9 -> 38.9; all features random forest 18.6 -> 69.8 / 18.4 -> 62.9.

### Rows that moved most (series level, top-20 zone of either run, rank among common units old -> new)

all_features:
- ridge: perm P2 MSE: `sumbipow_vwstock` 21->18, `sumvolume` 17->19
- lasso: perm P2 MSE: `sumbipow_vwstock` 24->20, `sumautocov` 20->23
- LightGBM: MDI: `sumret` 2->3, `sumabsret` 3->2; split: `sumabsret` 18->9, `stocktwits_sentcount` 13->17; SHAP: `voldemand_spx_open_and_close` 28->20, `sumret3_ewstock` 18->22; perm P2 QL: `sumautocov` 9->40, `buyturnover_ewstock` 12->43; perm P2 MSE: `sumabsret` 33->4, `fomc_until_inv` 36->20
- XGBoost: MDI: `sumbipow` 20->26, `ofi_vwstock` 19->14; split: `sellturnover_vwstock` 16->22, `stocktwits_attention` 15->11; SHAP: `sumbipow` 17->35, `voldemand_all_open_and_close` 33->20; perm P2 QL: `sellturnover_vwstock` 7->33, `turnover_vwstock` 15->41; perm P2 MSE: `sumret` 2->46, `sumabsret` 5->48
- random forest: MDI: `turnover_vwstock` 23->15, `voldemand_all_open_only` 15->21; split: `sumret2_ewstock` 36->12, `vvix` 39->20; SHAP: `buyturnover_vwstock` 22->17, `sumpret2` 17->21; perm P2 QL: `sumret4` 49->14, `sumret3_ewstock` 10->42; perm P2 MSE: `vvix` 10->38, `sumbipow` 18->43

## Files
- `results/feature_importance_1530_dedup/by_measure/imp_<bucket>_<model>_<measure>.csv` and the first pass's tables (`series_level_all.csv`, `rank_corr_*.csv`, `top5_stability.csv`, `stability_by_stratum.csv`, `unique_values.csv`, `cardinality_vs_measure.csv`, `noise_probes.csv`, `clusters.csv`, `cluster_dilution.csv`, `tuned_trees_series.csv`, `gates.csv`) on the new design
- the re-run's own: `gates_design.csv`, `gates_cadence.csv`, `gates_canonical.csv`, `before_after_*.csv`, `fig_before_after_*.png`; the no-mask rung in `results/feature_importance_1530_dedup_nomask/` (same tables)
- scripts: `experiments/feature_importance_1530_dedup.py` (roots, stages, gates), `experiments/feature_importance_1530_dedup_compare.py` (before / after), `writeup/make_feature_importance_1530_tex.py --root results/feature_importance_1530_dedup --stem feature_importance_1530_dedup --dedup` (+ `writeup/feature_importance_1530_dedup_sections.py`)
- cluster twins: `cluster/slurm/{ship_featimp_dedup_carc.sh, submit_featimp_dedup.sh, featimp_dedup_pack.sbatch, featimp_dedup_collect.sbatch, featimp_dedup_linear.sbatch, pull_featimp_dedup_carc.sh}`, `cluster/featimp_dedup_tasks*.txt`, `cluster/featimp_dedup_linear_tasks.txt`
