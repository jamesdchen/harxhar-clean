# Paper rehaul: section outline (2026-10-15)

Bullet outline for the full rewrite. Built from the live paper, the generated macros and the results folders; every
number names its macro or source. Paths are relative to `writeup/`; RCO = `sections/results_close_option.tex`. Parked
text was read as context only. The LSTM and the bucket-by-bucket write-up are parked and left out.

## 1. Introduction

- **Question.** At 15:30, forecast the 15:30-16:00 SPX realized variance. A 0DTE straddle bought or sold at 15:30
  settles on exactly that variance. Trade the sign of s = forecast - IV^2 (RCO:15-30). Sample: `\cmNdays` = 866
  expiration days, 2020-01-03 to 2024-04-30.
- **Why it matters.** Quoted implied variance sits above the forecast on `\cmSellPaperMin`-`\cmSellPaperMax` =
  72-83 % of days (related_works.tex:130-132), so a forecast's value is in finding the buy days.
- **Replace the current frame.** Title, abstract and intro still describe the 24-hour, 30-minute-ahead panel
  (1998-2024: 41 channels, block ridge, benchmark QLIKE `\unifAzeroQLIKE` = 0.23806; main.tex:39, 59-74;
  introduction.tex:8-22).
- **Contribution claims, with what to do about each:**
  - C1, missing / stale-data representation (intro:24-33): keep, restated for the 16:00 design. All features has
    `\cmPNewAll` = 628 columns and HAR + calendar `\cmPNewBase` = 22, after dropping `\cmNDup` = 12 duplicate
    session-edge columns.
  - C2, the plain f^2 B convention (intro:50-54): keep, as one sentence.
  - C3, "dense but weak, recoverable only by shrinkage" (intro:56-78; main.tex:70-74): drop or recast.
  - C4, trees beat linear (intro:87-92; DM -10.7): a 24-hour-panel result only.
  - C5, what tuning buys, measured (intro:213-222): keep, with the 16:00 numbers.
- **Axis 1, all features vs HAR + calendar** (figure macros; RCO:270-326):
  - QLIKE is lower with all features for `\fqsQLower` = 38 of `\fqsPairs` = 51 pairs, `\fqsQLowerSig` = 3
    (was 2) significantly.
  - Sharpe is higher for `\fqsShHigher` = 50 of 51, `\fqsShHigherSig` = 4 significantly.
  - Median changes: QLIKE `\fqsMedDQPct` = -1.7 %, Sharpe `\fqsMedDSh` = +0.54. There are no significant reversals.
  - Model-set medians: QLIKE 0.1033 -> 0.1014 (was 0.1007 -> 0.0993; `\cmMasterMedQBase`, `\cmMasterMedQAll`);
    Sharpe 0.73 -> 1.27.
- **Axis 1 caveat, to state up front.** For the per-bar ridge, all features moves QLIKE `\cmLdInQ` = +0.8 %
  (was +1.2 %; DM +0.18, was +0.26) and Sharpe +0.05 (RCO:419-445).
- **Axis 2, per-bar vs pooled:**
  - On HAR + calendar, per-bar fitting improves QLIKE by `\cmLdPerBarBaseQ` = -12.0 % (was -11.3 %; DM -3.37, was
    -3.42); Sharpe +0.46,
    HAC t 1.05.
  - On all features: QLIKE -6.7 % (was -6.2 %; DM -1.61, was -1.56); Sharpe -0.03.
  - Sharpe ranges: pooled linear 0.91-1.44; per-bar linear 1.37-1.92.
- **Accuracy is not the trade:**
  - The rank correlation of QLIKE with Sharpe is `\cmMasterRankCorr` = -0.20 (was -0.18).
  - None of the 21 single-change steps moves Sharpe significantly (`\cmLdResolved`).
- **The trade:**
  - 112 of 112 forecasts beat always-short, 20 significantly. Always-short: Sharpe 0.20 mid, -0.27 crossed.
  - It is a tail trade: the top 20 days carry a median 99 % of P&L (RCO:744-755).
- **Estimators at 16:00:**
  - Trees vs the per-bar ridge: 46 of 48 QLIKE differences favour the ridge, 21 (was 18) significantly
    (RCO:631-640).
  - Optuna is worse than the shipped settings in 24 of 24 cases.
  - Reference: the per-bar HAR + calendar OLS, QLIKE 0.0967 (was 0.0946; `\rhrOlsQ`), Sharpe 1.37.
- **Roadmap** (rewrite intro:224-237): data -> methods -> results (axis 1, then axis 2) -> the trade -> conclusion.

## 2. Related work

Every key below is in `references.bib`.

- **HAR / realized volatility:** Corsi2009, AndersenBollerslevDieboldLabys2003, BarndorffNielsenShephard2004,
  AndersenBollerslevDiebold2007, CorsiMittnikPigorschPigorsch2008, BollerslevPattonQuaedvlieg2016,
  BekiermanManner2018, BennedsenLundePakkanen2022, HansenHuangShek2012, AndersenBollerslevChristoffersenDiebold2006.
  BollerslevHoodHussLebedev2018 (panel-pooled HAR) anchors the pooled axis; its key says "Lebedev" but its author
  field says Pedersen.
- **Intraday / diurnal:** AndersenBollerslev1997, EngleGallo2006, HizmeriBevilacqua2025. The Hawkes / rough-volatility
  entries are uncited: cite them or drop them.
- **0DTE / volatility risk premium:** only HizmeriBevilacqua2025, BlackScholes1973, Merton1973.
- **Machine learning:** gu2020empirical, christensen2023machine, Ke2017, Chen2016, Akiba2019 (uncited), Lou2013,
  lundberg2017unified, lundberg2018consistent, aas2021explaining.
- **Evaluation:** Patton2011, DieboldMariano1995, HarveyLeybourneNewbold1997, HansenLundeNason2011, Ehm2016,
  MincerZarnowitz1969.
- **Shrinkage:** GiannoneLenzaPrimiceri2021, ShenXiu2025, KozakNagelSantosh2020, DonohoJin2004, HoerlKennard1970,
  Tibshirani1996, ZouHastie2005, Efron2004, vandeWiel2021multipenalty, vanWieringen2023ridge.
- **Back-transform** (one sentence): Duan1983, Manning1998, ManningMullahy2001, DemetrescuGolosnoyTitova2020.
- **Suggested references NOT in the bib:**
  - 0DTE: Bandi-Fusari-Reno 2023; Beckmeyer-Branger-Gayda 2023; Brogaard-Han-Won 2023; Dim-Eraker-Vilkov 2024;
    Vilkov "0DTE Trading Rules"; Almeida-Freire-Hizmeri 2025.
  - Volatility risk premium: Carr-Wu 2009; Bollerslev-Tauchen-Zhou 2009; Bakshi-Kapadia 2003.
  - IV as a predictor: Blair-Poon-Taylor 2001; Busch-Christensen-Nielsen 2011.
  - Intraday: Engle-Sokalska 2012; Patton-Sheppard 2015.
  - ML / NN: Breiman 2001; Bucci 2020; Rahimikia-Poon 2020; Zhang-Zhang-Cucuringu-Qian 2024 (a co-author's paper);
    Feng et al. 2024 (MinGRU); PufferLib.
  - Evaluation: Hansen 2005 (SPA); White 2000; Romano-Wolf 2005 (StepM); Newey-West 1987; Politis-Romano 1994.
  - Shrinkage: Kelly-Malamud-Zhou 2024; Stock-Watson 2012.

## 3. Data

- **Sources.** Every bar file is `data/*.parquet`, keyed on `endbartime` (naive ET, bar end). `load_raw_data`
  outer-merges the bar files (src/data/loading.py:465-511).
  - **core_stats:** the E-mini (ES) near-24-hour tape, as 30-minute sums of 1-minute log returns: `sumret2` (= RV,
    the target), sumret, sumabsret, sumret3, sumret4, sumpret2, sumbipow, sumautocov, sumvolume, numobs. RV prints
    run 1998-01-05 01:30 to 2024-04-30 23:30 (303,444 prints).
  - **ewstock_stats / vwstock_stats:** the equal- and value-weighted constituent cross-section: six return moments,
    turnover, buy and sell turnover, effective spread, quoted spread. 1993-01-04 to 2024-04-30, cash and extended
    session.
  - **spy_and_sentiment:** StockTwits attention (from 2009-03-23); sentiment and sentcount (from 2010-06-01).
  - **vix_and_voldemand:** VIX (from 2006-05-22), VIX3M (from 2009-08-12) and VVIX (from 2012-03-26), all **ending
    2024-02-12**. Four voldemand option open/close volume series, 2011-01-03 to **2023-08-31**.
  - **releases:** only `fomc release` is used: 246 bars at 14:30, 1993-02-03 to **2023-11-01**.
  - **spxw_chain:** the SPXW expiration-day chain, 8.2M rows, 1,291 expirations, 2020-01-03 to 2025-12-31, 30-minute
    UTC stamps. Fields: bid, ask, mid, underlying, IV, delta, hours to expiry, early_close. From data/SPXW.csv
    (src/data/spxw.py).
  - **Merged but never a feature:** the other release flags, spxw_mfiv / ivslice, the VIX representations, the
    OptionMetrics features. There is no VIX9D.
- **Cleaning:**
  - **Grid:** starts at the first RV print, reindexed to 30 minutes.
  - **Weekend cut:** Friday after 20:00, Saturday, and Sunday before 18:30 are dropped.
  - **Dead-session rule:** a bar exists only if RV printed. Holidays and early closes stay in as futures bars; the
    trade filter drops early closes.
  - **OFI:** (buy - sell) / (buy + sell).
  - **FOMC features:** the tail after the last print is set to NaN.
  - **Missing data:**
    - The observed flag is taken before the fill.
    - The forward fill is **unlimited**: FFILL_LIMIT = 26 is not passed (executor.py:476, 736-745).
    - Unobserved rows get the full-sample median.
  - **Diurnal baseline B:** the mean of the last 20 same-slot RV bars, shifted one bar, needing at least 5 observations.
  - **Target:** y = winsorised sqrt(RV / B), with rolling 240-bar 1 / 99 % bounds shifted one bar.
  - **Not scored:** the 65 pooled no-history bars.
  - **Early closes** depress B for 20 sessions afterwards, affecting 94 of the 866 trade days (`\cmBContamDays`).
- **Spans and counts:**
  - **Grid:** 303,442 print-bars after the weekend cut.
  - **Pooled panel:** a 3,125-bar burn-in leaves 300,317 rows from 1998-06-04. With the 24,000-bar window, scoring
    starts about 2001-11. Scored rows `\unifAzeroN` = 273,489; rows in QLIKE `\unifAzeroNQlike` = 272,907.
  - **Per-bar 16:00 design:** 3,469 sessions, 2010-07-12 to 2024-04-30, with W = 2,000. 1,469 forecast rows from
    2018-06-25 (`\campKeptRefits`).
  - **Pre-tune:** the 16:00 rows of the last-hour design back to 2004-04-13 (3,947 sessions), with folds 2016-01-06
    to 2020-01-02.
  - **Test:** 866 trade days (`\cmNdays`) = 872 SPXW expirations to 2024-04-30 minus 6 early closes, out of 1,087
    sessions with a 16:00 row. The 413 later expirations are not scored.

## 4. Features

- **Buckets** (src/data/loading.py:224-255): 8 buckets, 47 series (`\bkNSeries`).
  - moments (7);
  - liquidity (14: volume, numobs, ew / vw turnover, buy / sell turnover, spreads, ofi);
  - market_ew (6) and market_vw (6);
  - sentiment (3);
  - implied_vol (3: vix, vvix, vix3m);
  - vol_demand (4);
  - fomc (4: release, day, until_inv, since_inv).
- **Columns per series:**
  - `adj_x_ma_k` and `x_avail_ma_k`, plus `x_active_ma_k` for the 7 mode-inflated series (numobs, the 4 voldemand,
    fomc_release, fomc_day).
  - Ladders are trailing means shifted one bar: per bar, base 5, k in {1, 5, 25, 125, 625, 3125}; pooled, base 2,
    12 rungs from 1 to 2048.
  - HAR: `har_ma_k` of the adjusted target.
  - Calendar (16): DOW_0-4, hour, is_overnight, is_open, is_close, opex, opex week, quad witch, rebalance close,
    month end, quarter end, days_to_opex.
  - Session-edge `har_ma_k_x_{open,close}` columns appear in the pooled designs only.
- **Transforms, in order** (robust_transform, src/features/transforms/target.py:578-685):
  1. Diurnal adjustment:
     - unsigned series are divided by the per-slot mean; signed series by the per-slot sd (floored);
     - exogenous baselines are banded to [0.5, 2] x the slot's trailing-250 median;
     - excluded: vix*, *sentiment*, fomc_*.
  2. Asinh for signed series whose sd divisor is pinned on 50 % or more of rows (voldemand).
  3. A name-based map:
     - sqrt for ret2, RV, turnover, bipow and effspread;
     - signed sqrt for autocov; cube root for ret3; signed 4th root for ret4;
     - identity for signed series and sumabsret;
     - log otherwise.
  4. Winsorise (240 bars, 1 / 99 %).
  5. Impute, with an indicator.
  6. Hurdle `_active` columns.
  7. Ladders.
  8. Scaling:
     - rolling_robust_scale over the trailing W rows, with the IQR floored at 1 % of its running maximum;
     - a `ref_iqr` floor: a causal, expanding IQR over the available rows;
     - the avail and active MAs are not scaled;
     - pooled only: masked_rolling_scale_col (at least 512 observations, else 0).
- **Filtering:**
  - `window_keep` drops window-constant columns and byte-copies, at every refit and tune.
  - Kept columns: baseline 17 of 22 (`\campKeptBase`); all features a median 369 (range 360-378) of 628
    (`\campKeptAll`). 137 columns are constant at 16:00.
  - Dedup removed the 12 session-edge columns from the per-bar design: baseline 34 -> 22, all features 640 -> 628.
- **Designs:**
  - baseline 22 = 6 HAR + 16 calendar;
  - all_features 628 = 22 + 47 x 6 values + 47 x 6 avail + 7 x 6 active;
  - lean_vol_vix 94 and live_feasible 232 (both parked);
  - pooled 1,144 (pre-FOMC) or 1,264 (with FOMC);
  - the bucket designs are column subsets of the 628.
- **Causal by construction:**
  - one-bar shifts in the ladders, B, the winsor bounds and the band;
  - trailing scalers and window_keep;
  - calendar from the timestamp;
  - the 16:00 row is issued at 15:30.
- **Full-sample elements (disclose):**
  - the neutral median;
  - the signed-sd floor;
  - the asinh routing;
  - the mode-inflated class;
  - the FOMC dead-tail cut;
  - the first window's scaler statistics (warm-up rows only).

### Flags (data / features)

1. **Data bug: `_active` flags are computed after the forward fill** (executor.py:618-627, after :540).
   - Checked in design_bar1600_all_features.npz: fomc_release_active_ma_* and fomc_day_active_ma_* equal 1.0 on
     every 16:00 row from 2023-11-02 to 2024-04-30, i.e. 122 trade days after the FOMC feed ends.
   - voldemand_*_active equals 1 after 2023-08-31.
   - These columns are constant before the dead tail, so window_keep drops them until the tail enters the window.
2. **Stale feeds at the end of the test period.**
   - VIX / VVIX / VIX3M end 2024-02-12, voldemand ends 2023-08-31 and FOMC ends 2023-11-01.
   - Because the forward fill is unlimited, these inputs are carried flat for months into 2024.
   - This may relate to the Jan-Feb 2024 extreme-input days in the MinGRU diagnosis. Not yet checked.
3. **Stale counts in data_prep.tex** against the 47-series, 8-bucket partition: 41 channels (:125-127), 218,934 rows
   (:251), 359 / 529 columns (:193-198), 36 / 78 % / 0.638 (:211-221), 31 of 41 (:243).
4. **Ladder contradiction:** the data_prep equation is base 2, but the prose describes six base-5 rungs. Per-bar uses
   base 5; pooled uses base 2 with 12 rungs.
5. **"No weekend rule"** (data_prep.tex:18-20) is false (loading.py:536-548).
6. **The missing-data text is wrong:**
   - every unobserved bar gets the median, not only those with no prior print;
   - the fill is unlimited;
   - the indicators are taken before the fill, not after (fixed in e9cc5e8).
7. **The 512-observation / 1 % scaling rule** applies to the pooled panel only.
8. **Possible unit mismatch:** the per-bar ref_iqr erosion counts MA windows in sessions, not bars (executor.py:178-188).
   Its impact is unmeasured.
9. **Burn-in placement:** the 3,125-row burn-in is applied after slicing to one bar per session. It discards
   1998-2010, which is why the per-bar design starts in 2010-07 and the pre-tune borrows the last-hour design.
10. **Panel width:** the current code builds 1,264 pooled columns; the panel of record is 1,144 (pre-FOMC).
11. **No source names the vendor** of the ES, cross-section, StockTwits or voldemand files. The chain is called
    "OptionMetrics" only in a comment (loading.py:483).
12. **numobs is described three ways** (data_prep.tex:88-90, 133-134; loading.py:348-349).
13. **FOMC:** the 2020-03-03 exclusion does nothing, and adj_fomc_release is constant.
14. **Docstrings say 5 / 95; the code uses 1 / 99.**
15. **Broken or untraceable:** dvs_dedup_1530.py fails at import, and the 456-of-1,264 prefix audit
    (data_prep.tex:374-389) has no source.
16. **No macros** exist for 303,442; 300,317; 3,125; 24,000; the per-bar span; or 1,087.

## 5. Methods

Keys: M = sections/methods.tex, MC = methods_close_option.tex, A16 = appendix_campaign_16h.tex, DP = data_prep.tex,
CN = generated/close_main_numbers.tex, AN = generated/appendix_campaign_16h_numbers.tex. CLP =
experiments/close_linear_pretune.py, CTL = specs/causal_tune_linear.py, OPT / OJ = specs/causal_tune_trees_optuna(_jobs).py,
PF = experiments/puffer_forecast/, TR = PF/native/forecast_train.cu, R = results/puffer_forecast_2026-10-12/.

### 5.1 Problem formulation: time-series regression run as a causal walk-forward backtest

- **Target:**
  - y = sqrt(RV / B), winsorised on the fit scale. The clip bounds are rolling 1st / 99th percentiles over 240 bars,
    lagged one bar (DP:277-289; src/features/transforms/target.py:96-97, 452, 536-539). Models are fitted, tuned and
    selected on this trimmed target. **Since 2026-10-15 (data audit #10) every out-of-sample score is against the
    actual RV** (`rv_unclipped`), not the trimmed one: the target was cut on 5 of the 866 trade days (kept 0.65-0.96).
  - The per-bar target is the 15:30-16:00 bar: the row is stamped 16:00 and issued at 15:30 (MC:137-142, 278-284).
- **Diurnal baseline B:**
  - B is the same-slot mean of the last 20 observations with slot index j < t, and needs at least 5 (DP:254-275).
    Bars with fewer than 5 earlier same-slot values are not scored (M:186-188).
  - Exogenous series are divided by their own baseline; signed series use the same-slot sd (DP:270; MC:293-295).
- **Back-transform:** the forecast is f^2 B, with no smearing and no level or second-moment map (M:165-182;
  MC:483-497). The Jensen low bias is left in and disclosed (M:199-206).
- **Loss:**
  - Scored on QLIKE in raw variance: scale-invariant, strictly consistent for the variance, and asymmetric (Patton;
    M:62-82).
  - Fit, tuning and selection use MSE in fit space (CTL:455; OPT:26-27; PF/phase2_select.py). The exception is the
    QLIKE-trained MinGRU (2026-10-15).
- **Information set:** HAR sums start at t-1. The ladders are means over k in {1, 5, 25, 125, 625, 3125} bars, plus
  is-present flags and calendar dummies (MC:284-298). B, the clip bounds and the scaling are all trailing. Disclosed
  breaches: the neutral fill is a full-sample median, and the indicators come after the forward fill (DP:377-389).
- **Per-bar vs pooled:**
  - Per-bar: the 16:00 bar, a 2,000-session window, refit every session; 22 columns (baseline) or 628 (all features)
    after the 12 session-edge drops (AN:9, 12; A16:97-108).
  - Pooled twins: all 48 bars of the same 2,000 sessions, i.e. 96,000 bars (MC:470-473).
  - Main-paper pooled arms, the tree bank and the pooled MinGRU: trailing 24,000 bars (M:45-51;
    src/unification.py:135).
- **Spans and splits:**
  - Scored rows run 2018-06-25 to 2024-04-30; the trade sample is 866 days from 2020-01-03 (AN:33).
  - Tuning splits are forward: fit block, a 25-row embargo, then a 125-row validation tail (CTL:157-162).

### 5.2 Models

- **HAR / HAR + calendar OLS:**
  - Pooled: minimum-norm least squares, with exact rank-1 rolling updates (M:41-63, 286-311).
  - Per bar: OLS on the baseline set (MC:318).
- **Ridge / lasso / elastic net:** sklearn units. The lasso and elastic net are exact online homotopies (M:313-367);
  the per-bar elastic net has l1_ratio 0.5. A fixed lasso at alpha 1e-4 is among the paper's eight (MC:58).
- **Block-diagonal Tikhonov:**
  - Pooled: (alpha1, alpha2) = (1, 100), solved via column scaling (M:475-516).
  - Per bar: pf = R_BB = 0.01 on the HAR + calendar backbone, 1 on exogenous columns, 0 on the intercept (CLP:109;
    experiments/close_exogpen.py:249-296).
- **Per-bar LightGBM / XGBoost / random forest:** the same rows and window as the linear models; seed 42 (MC:390-391).
- **Pooled tree bank:**
  - One frozen configuration per arm, refit every bar on the trailing 24,000 bars, on a 300,317 x 1,144 panel.
  - 20 experts; aggregated by selection (best on a 125-bar tail, 25-bar embargo, refreshed every 250 bars) and by
    hedge weights (M:424-450; results.tex:265-286).
  - The pooled forest is trial 255, every bar.
- **MinGRU (PufferLib 5.0 native):**
  - Architecture: encoder, then 1 MinGRU layer with highway gates, then a decoder. Inputs: the last L session rows
    per bar, or L half-hour bars when pooled.
  - Training: Muon (Nesterov 0.95, Newton-Schulz, clip 1.0); cosine learning rate; batch 128; at most 200 epochs;
    patience 20 (TR:1-26).
  - Output link: identity f = mu + sd o, or softplus f = softplus(c + s o).
  - Inputs are the prescaled design (xstd = 0) with the window mask.
  - Variants:
    - wd: decoupled decay (g6);
    - wd_enc: decay on encoder input columns, scaled by pf (exogenous 1, backbone 0.01) (g7);
    - linear shortcut: f = link(c + s[o + v x]), with an L2 penalty wd_lin and Adam on v (g8);
    - ridge init: v starts at each window's closed-form ridge;
    - QLIKE training with es = qlike (g9).

### 5.3 Tuned hyperparameters

- **Linear:**
  - Grids: ridge 1e-2 to 1e3; lasso and elastic net 1e-6 to 1e-2, one value per decade (CTL:168-172).
  - Pre-tune picks are `\cmPtPen{Ridge,Lasso,Enet}{Base,All}` (CN:286-296). All features: ridge 1e3, lasso and
    elastic net 1e-2, all at the top edge. Baseline: the low edge.
  - 21 of 33 picks sit at a grid edge; with the grid widened a decade, 3 of 21 still do
    (results/linear_pretune_2026-10-06/SUMMARY.md).
- **Trees, shipped settings** (1998-2002 pooled development search):
  - LightGBM lgbm_opt_00; XGBoost xgb_opt_00 (xgbtop).
  - Random forest: trial 255 of 400 (485 trees, depth 20, max_features 0.90, max_samples 0.41), in
    `\cmTreeProvRf*` (CN:329-344).
  - Leaf minimums are scaled to the 2,000-row window.
- **Tree Optuna spaces** (OJ:32-47, 93-115):
  - LightGBM: num_leaves 2-128, min_child 1-256, feature_fraction 0.05-1, bagging 0.3-1, l2 1e-3 to 1e4, lr
    0.005-0.1.
  - XGBoost: depth 1-8, min_child_weight 0.1-256, subsample 0.15-1, colsample 0.05-1, lambda 1e-4 to 1e4 (xgbtop
    floor 1e-5).
  - Random forest: max_features 0.02-1, min_leaf 1-256, depth from a ladder.
  - Picks: tab:app_camp_picks.
- **Tree random search:** 32 fixed candidates, GRID_SEED 20260929 (specs/causal_tune_trees_tuned.py:46-53).
- **MinGRU:**
  - Base grid: L {5, 20} x H {16, 64} x lr {0.003, 0.01, 0.03}.
  - Extensions: wd {0, .01, .03, .1, .3, 1}; wd_enc {0, .03, .3, 1, 3, 6, 15}; ridge-init 216 candidates.
  - Pooled grid: H {64, 256} x L {20, 48} x lr {.001, .003} x wd.
  - lr 0.03 is picked in 9-10 of 10 cells.

### 5.4 Tuning procedure, per model

- **Linear pre-tune:** 4 folds x 250 sessions over 2016-01-06 to 2020-01-02. Each fold fits the 2,000 sessions
  before a 25-session embargo, scored on mean validation MSE (CLP:34-44).
- **Linear light retune (margin rule):**
  - At 4 block starts, the two neighbouring grid values challenge the incumbent on two 125-session folds. A
    challenger switches in only if Newey-West t < -z(0.05 / #challengers). 4 of 24 challenges fired.
  - Alpha is frozen between retunes; coefficients are re-solved every session; the mask is recomputed every 250
    sessions.
- **Pooled linear twins:** pre-tuned on 4 x 12,000-bar folds; margin retune every 250 bars on a 125-bar tail.
- **Tree rungs:**
  - T10 / T1: the shipped settings, refit every 10 sessions or every session.
  - RS: the random search every 250 sessions, chosen on validation MSE.
  - Optuna: a fresh TPE study at each point (seed 20260929 + row), 50 sequential trials with trial 0 = shipped,
    TUNE_PER in {1, 5, 25, 250}, best of the first 10 / 25 / 50; tp25_q selects on QLIKE.
- **Window mask:** at every fit, drop window-constant columns and exact byte-copies (src/models/window_mask.py).
- **Seeds:** linear deterministic; trees 42; MinGRU 5 seeds per bar and 3 pooled, with the seed mean of f as the
  forecast.
- **MinGRU walk:**
  - 1,088 origins from 2020-01-02 in 5 blocks of 250. Each block start re-tunes on a 125-session tail with a
    25-session embargo, and selects the lowest validation loss of the seed-averaged forecast. The configuration and
    the epoch counts are frozen per block; every origin gets a cold refit.
  - The pooled walk uses a 2,400-bar tail, a 48-bar embargo and monthly refits.
  - QLIKE variants: option 1 selects on QLIKE; option 2 also trains and early-stops on QLIKE (R/qlike_study).

### 5.5 Statistical tests

- **DM:** daily QLIKE difference, Bartlett Newey-West, lag max(h-1, floor(4(T/100)^(2/9))) = 6, HLN correction
  (src/evaluation/diebold_mariano.py).
- **HAC t of P&L:** Newey-West on the daily mid-return difference, lag floor(1.5 n^(1/3)) = 14
  (notebooks/atm_straddle_lib.py:1234-1261).
- **No intervals:** the bootstrap was disabled (b761b28). MCS / SPA / StepM exist only in the parked buckets appendix.

### Flags (methods)

1. **HAR ladder:** DP:325-331 says base 2; the code, MC:286 and A16:88 use base 5.
2. **QLIKE realisation:** M:183-185 says raw RV, never winsorised, but the close tables' `qlike_raw` uses the
   winsorised y^2 B. **Resolved 2026-10-15 (data audit #10):** `qlike_raw` and every other out-of-sample score now use
   the actual RV; the text and the code agree.
3. **Window size:** M:47-50 says "about 480 sessions"; M:351 says 500.
4. **Model families:** methods.tex describes two GBM families only. The random forest and the MinGRU are absent; the
   MinGRU appears nowhere in sections/ or generated/.
5. **Elastic-net rho:** M:340-345 says rho is free; every close enet fixes 0.5.
6. **"Pre-tuned, as the trees are":** the trees' settings come from the 1998-2002 pooled development search, and the
   16:00 tree pre-tune is not reported.
7. **Margin rule:** MC:369-370 states -z(0.05/2); the code uses z(0.05 / #challengers).
8. **Pooled twins:** "tuned the same way, with a larger validation tail" is false: the tail is 125 bars (about 2.6
   sessions).
9. **Duan back-transform inside QLIKE selection:** the trees' tp25_q and the linear QLIKE picks use it, undisclosed.
   The MinGRU's selection QLIKE is plain.
10. **Undisclosed picks:** the `\cmPtPen*` macros are not cited, edge picks are not mentioned, and the pre-tune folds
    use the last-hour cache.
11. **MCS promised but absent** (M:136-137). The P&L HAC lag is not stated.
12. **Pooled tree bank:** part of the LightGBM / XGBoost bank ran every 10th bar; tree_menu.json is lost; experts are
    scored including the development span.
13. **XGBoost min_child_weight scaling** is not disclosed.
14. **Indicator timing:** DP:230-231 says before the fill, DP:380-382 says after.
15. **Embargo:** the 25-bar pooled embargo covers rungs <= 25 only.
16. **Stale comments:** winsorisation is described as "5/95" while the code uses 1/99; the rfp wrapper says trial
    163, not 255.
17. **Block calendars differ:** the MinGRU blocks start 2020-01-02, the linear retunes 2020-06-23.
18. **MinGRU gaps:**
    - no variant of record is declared;
    - epochs tuned on 1,850 rows are reused on 2,000-row refits;
    - the cosine schedule barely anneals;
    - g7-g9 are not committed.
19. **Column counts are not reconciled:** 523 (M:316), 1,264 / 1,144 (MC:187-191), 628 (AN:12).
20. **B contamination:** early closes bias B low for 20 sessions; disclosed, not repaired (MC:157-177).

## 6. Results (statistical)

### 6.0 Scoring convention, stated once

- The forecast is the plain back-transform f^2 B, with no recalibration (experiments/master_table_close.py:35-38).
- ~~The target is the winsorised 15:30-16:00 RV, `true_raw`.~~ The scoring truth is the **actual** 15:30-16:00 RV,
  `rv_unclipped` (2026-10-15, data audit #10; master_table.csv `qlike_raw`). The trimmed-target QLIKE is kept as the
  diagnostic `qlike_raw_trimmed`; models still fit the trimmed target.
- Scoring covers the 866 trade days, 2020-01-03 to 2024-04-30, with early closes excluded.
- Point estimates only; the bootstrap is off.
- Errors are measured in variance space. Note that the NN CSVs' `mse_fit` is in fit space.

### 6.1 Statistical columns

| column | definition | source |
|---|---|---|
| QLIKE | mean of y/f - log(y/f) - 1, y the actual RV (`rv_unclipped`, since 2026-10-15) | master_table_close.py:1224-1241; GUI columns.py:92 |
| % vs R | 100 mean(d) / mean(q_ref); R = blk2, **not** OLS | master_table_close.py:233, 1250-1251 |
| DM (master) | daily QLIKE vs blk2; Bartlett NW, lag 6, HLN, normal p | :1252-1263; src/evaluation/diebold_mariano.py |
| DM (GUI) | the same, vs the per-bar OLS on HAR + cal. (`sub_ols_baseline`) | rows.py:1585; columns.py:142 |
| MSE / MAE | variance space; GUI only | rows.py:1526-1540 |
| OOS R^2 | 1 - SSE / sum((y - ybar)^2), where ybar is the full-sample mean (look-ahead) | rows.py:1538-1539 |
| RV/F, share above | mean(y)/mean(f); P(y > f) | master_table_close.py:1268-1269 |

There is no MCS or SPA column anywhere in the model table.

### 6.2 Proposed tables

- **T1: HAR + calendar vs all features, within each family.** Paired QLIKE, DM, and Sharpe change. No CSV holds
  these pairs yet: `make_fig_qlike_sharpe.py:288-294` computes them but writes only counts, so add a pairs CSV.

  Actual RV (2026-10-15, data audit #10); the trimmed-target values of the earlier draft in brackets.

  | family (canonical rung) | QLIKE base -> all | change | DM | dSharpe (t) |
  |---|---|---|---|---|
  | per-bar ridge | 0.0967 -> 0.0975 [0.0946 -> 0.0958] | +0.8 % [+1.2] | +0.18 [+0.26] | +0.05 (0.09) |
  | per-bar lasso | 0.0967 -> 0.0931 [0.0946 -> 0.0914] | -3.7 % [-3.4] | -1.35 [-1.22] | +0.50 (1.47) |
  | per-bar enet | 0.0967 -> 0.0958 [0.0946 -> 0.0942] | -0.9 % [-0.4] | -0.20 [-0.10] | +0.55 (1.34) |
  | per-bar LightGBM T1 | 0.1010 -> 0.0961 [0.0986 -> 0.0939] | -4.9 % [-4.7] | -1.45 [-1.43] | +0.54 (1.11) |
  | per-bar XGBoost T1 | 0.1001 -> 0.0957 [0.0976 -> 0.0935] | -4.3 % [-4.2] | -1.45 [-1.41] | +0.16 (0.30) |
  | per-bar RF T1 | 0.0992 -> 0.0991 [0.0968 -> 0.0966] | -0.0 % [-0.2] | -0.01 [-0.05] | +0.18 (0.34) |
  | pooled ridge / lasso / enet | 0.110 -> 0.105 / 0.107 / 0.104 [0.107 -> 0.102 / 0.104 / 0.102] | -4.8 / -2.1 / -5.1 % [-4.3 / -1.7 / -4.7] | -1.45 / -0.63 / -1.57 [-1.29 / -0.50 / -1.42] | +0.5 to +0.6 |
  | pooled LightGBM / XGBoost / RF (interim) | 0.125-0.128 -> 0.109 / 0.109 / 0.110 [0.12 -> 0.106 / 0.106 / 0.107] | -14.6 / -12.7 / -13.0 % [-14.3 / -12.4 / -12.7] | **-4.64 / -4.55 / -4.63** [-4.57 / -4.46 / -4.57] | +0.3 to +0.5 |
  | MinGRU, ridge-init, QLIKE-tuned (current GUI row) | 0.1006 -> 0.0966 [0.0985 -> 0.0948] | -4.0 % [-3.8] | p 0.32 [0.34] | Sharpe 1.04 -> 1.53 |
  | pooled MinGRU, ridge-init, QLIKE-tuned | 0.1177 -> 0.1141 [0.1151 -> 0.1121] | -3.0 % [-2.7] | p 0.47 [0.52] | Sharpe 0.78 -> 0.34 |

- **T2: pooled vs per-bar** (pooled minus per-bar):
  - Linear, HAR + calendar: ridge / lasso / enet +13.6 / +13.0 / +13.4 % (was +12.8 / +12.2 / +12.6; DM 3.37-3.45,
    was about 3.4-3.5).
  - Linear, all features: ridge +7.2 % (DM 1.61), lasso +14.9 % (3.09), enet +8.5 % (2.03) (was +6.6 % (1.56),
    +14.2 % (3.19), +7.8 % (2.00)).
  - Trees, HAR + calendar: +25 to +28 % (DM 4.2-4.6; was +24 to +27 %, DM 4.4-4.8).
  - Trees, all features: +11 to +14 % (DM 2.8-3.0; was DM 2.7-3.0).
  - MinGRU (ridge-init, QLIKE): HAR + calendar +17.0 % (p 0.002), all features +18.1 % (p < 0.001) (was +16.8 / +18.2).
  - Every Sharpe HAC t is below 1.
  - The MinGRU comparison mixes variants; redo it with pooled ridge-init QLIKE vs per-bar ridge-init QLIKE (running).
- **T3: full model set ranked by QLIKE, vs the per-bar OLS on HAR + calendar.** The DM vs OLS must be ported from the
  GUI into master_table_close.py. Current order:
  - per-bar lasso all 0.0931; XGBoost all 0.0957; enet all 0.0958; LightGBM all 0.0961; MinGRU all (QLIKE-tuned)
    0.0966 (was lasso 0.0914, XGBoost 0.0935, LightGBM 0.0939, enet 0.0942, MinGRU 0.0948: enet and LightGBM swap);
  - **OLS 0.0967** (was 0.0946) sits just behind that group, rank 9 (`\rhrOlsRank`), after the per-bar ridge / lasso /
    enet on HAR + calendar (within 0.03 %);
  - then ridge all 0.0975, RF all 0.0991, per-bar trees on baseline 0.099-0.101, the pooled linear on all features
    0.104-0.107, the pooled trees on all features 0.109-0.110, blk2 0.1092, the pooled MinGRU 0.114-0.118, the pooled
    trees on baseline 0.125-0.128, and a0 0.1270 (was 0.0958, 0.0966, 0.097-0.099, 0.102-0.104, 0.106, 0.1068,
    0.113-0.117, 0.121-0.124, 0.1232).
  - Across the 111 forecasts in the model set: 7 have lower QLIKE than the OLS (none significantly); 51 (was 48) are
    significantly worse.
- **T4 (appendix): tree rungs.** Reuse `\cmTrVsRidge*`, `\cmCad*`, `\cmRs*`, `\cmOp*`, `\cmTp*`, `\cmBud*`
  (close_main_numbers.tex:182-212).
- **T5 (appendix): MinGRU variants**, from the record walk through ridge-init QLIKE (qlike_study/scores.csv and
  earlier studies).

### 6.3 What the numbers say

- **Axis 1, QLIKE:**
  - Most pairs improve with all features (38 of 51), but only 3 (was 2) significantly, all tree rungs: LightGBM random
    search with daily refit, -7.7 % (was -7.5), DM -2.11 (was -2.07); LightGBM Optuna tuned every 250 sessions, -8.5 %
    (was -8.0), DM -2.37 (was -2.24); and, new, XGBoost random search with daily refit, -8.1 %, DM -1.99 (p 0.046).
  - The pooled trees are the one clear, significant win (DM about -4.6, was -4.5). Their HAR + calendar versions are
    weak, though.
  - The MinGRU now goes the right way when QLIKE-tuned, but not significantly.
- **Axis 1, MSE / R^2: the sign flips.**
  - All 51 pairs have higher variance-space MSE with all features, none significantly.
  - Five March 2020 days carry 85-92 % (was 83-91 %) of the squared error.
  - Excluding 24 Feb - 30 Apr 2020, the lasso on all features beats the OLS on MSE.
  - In fit space, all features lowers MSE (enet 0.0534 -> 0.0518 against sqrt(actual RV / B); was 0.0520 -> 0.0506).
- **Axis 1, the trade:** Sharpe rises in 50 of 51 pairs (median +0.54), 4 significantly, all tree rungs. This goes in
  the economics section.
- **Axis 2:** per-bar beats pooled on QLIKE wherever both exist.
  - Significant on HAR + calendar in every family.
  - Significant on all features for the lasso, the elastic net and every tree, but not the ridge.
  - No Sharpe difference is significant.
- **Families:** linear is about level with the per-bar trees, and both are ahead of the pooled models. Trees lose to
  the per-bar ridge in 46 of 48 comparisons, 21 (was 18) significantly.

### 6.4 Wording to change in RCO

- :16 cites the pooled `tab:block_ladder`; re-anchor on the per-bar OLS.
- :49-90 and :275-282 use R = blk2. "105 of 111 below R, 24 significantly" (now 99 of 111, 23: `\cmMasterQVsRef*`)
  becomes, vs the OLS: 7 lower, 0 significantly, 51 (was 48) significantly worse.
- :239-260: move the Sharpe-vs-R comparisons to the economics section.
- :291-293: say that 90 of the 112 forecasts are tree rungs.
- :315-320 caption: note that 36 of the 38 improvements are within noise, and that 45 of the 51 pairs are trees.
- :482-488: `\cmLdIn` uses the ridge, the one linear model that gets worse; add the lasso, enet and OLS steps.
- :631-644: compare the trees with the OLS as well, and add the NN.
- sections/results.tex:14-200: the lasso-vs-ridge battery runs on the old pooled panel. Drop it or move it to the
  appendix.

### 6.5 Figures

- **Existing:**
  - figures/fig_qlike_sharpe.pdf (in RCO:300);
  - fig_qlike_sharpe_change.pdf (not included);
  - results/close_master_table/qlike_vs_sharpe.png.
- **New:**
  - F1: forest plot of the all-vs-base QLIKE change per family, with DM (T1);
  - F2: pooled vs per-bar dumbbell;
  - F3: cumulative daily QLIKE difference vs the OLS, to show March 2020;
  - F4: ranked dot plot of T3;
  - F5: squared-error share by day, as the case for QLIKE as the primary loss.

### Flags (results)

- **Axis 1 at 5 % on QLIKE:** fails per-bar and for the pooled linear models. It holds for the pooled trees (weak
  baseline) and for 2 of 45 LightGBM rungs; it holds in point estimate on the trade; MSE and R^2 point the other way.
- **The pooled forest on all features** is still interim at 16:00; the full table is due about Oct 17.
- **The NN is not in master_table.csv or paper_model_set.csv.** Its CSVs store p values, not DM statistics, and
  have no variance-space MSE / MAE / R^2.
- **Missing outputs:**
  - the master table has no DM vs OLS;
  - no CSV holds the all-vs-base pairs;
  - the pooled-tree all-vs-base DM is not in any file.
- **The model set has no MCS / SPA.** The feature MCS / SPA / StepM exists (results/feature_mcs_2026-10-08/). StepM
  never names all features.
- **Convention issues:**
  - OOS R^2 uses the full-sample mean, which is look-ahead;
  - the DM lag is 6 against a HAC lag of 14 (results unchanged at 14);
  - the OLS, ridge, lasso and enet on HAR + calendar are one forecast;
  - there is no OLS on all features.
- The paper's `lgbm` / `xgb` rows are the pooled trees on all features.
- The agent's computations are in scratchpad: pairs.py, af.csv, levels.csv, pooltree.py.

## 7. Economic application: the closing trade

[mine] = recomputed by the outline agent from results/close_master_table/master_table_daily.parquet with the repo's
functions (scratchpad econ_pairs.py, sz_set.py, yr.py).

### 7.1 The trade

- **Instrument:** SPXW 0DTE, the nearest OTM call (smallest K >= S with a live mid) plus the nearest OTM put
  (largest K <= S) at the 15:30 print. It is a true straddle only when S sits on a strike; otherwise it is a strangle
  one strike wide (methods_close_option.tex:14-21; notebooks/atm_straddle_lib.py:1591-1661).
- **Timing:**
  - Enter at 15:30 ET at mid. Hold to the official close, cash-settled at intrinsic, no hedge.
  - R = exit/P - 1 >= -1; a short earns -R. Early closes are dropped. 866 days, 2020-01-03 to 2024-04-30.
- **Signal:** s = f^2 B - IV30^2, with the plain back-transform.
  - IV30 = mean of the two legs' vendor IV / sqrt 2 (methods:540-546).
  - Censored IV nodes are masked; 5 days are recovered by Black-76 inversion.
- **Rules:**
  - Always short: q = -1 every day.
  - sign(s): q = +1 if s > 0, else -1, with ties going short (master_table_close.py:1168-1183).
- **Fills:**
  - Mid: return qR.
  - Crossed: a long returns exit/ask - 1, a short returns 1 - exit/bid; the spread is paid at entry only.
  - No commissions, borrow or impact. Crossing costs a median 2.9 % of premium.

### 7.2 Sizing rules (master_table_gui/rows.py:88-134, 1014-1048)

- Positions use only earlier days; sizing is scored on days 253-866 (614 days).
- **sign(s);**
- **robust clip:** clip(s / m, -1, 1), with m the median |s| over earlier days;
- **full long, scaled short:** +1 if s > 0, else max(s / m, -1). This rule exists in the GUI only.
- **rank of |s|:** sign(s) times the share of earlier days with |s_u| <= |s_t|. The text says "smaller"; the code
  uses <=.

### 7.3 Economic columns

- **Sharpe, mid and crossed:** mean / sd (ddof 1) x sqrt 252 per trade day. The trade runs about 200 days a year, so
  calendar time would be x0.892.
- **Mean mid return.**
- **t(mean):** GUI only; Newey-West t, lag 14. Not in the paper.
- **Hit %:** share of days with mid return > 0. The GUI header says "buy rate".
- **Buys and pct_buy.**
- **dSharpe vs always short / vs R:** its "significance" is the HAC t (lag 14) of the daily mid-return difference,
  which tests the mean difference, not the Sharpe difference.
- **ES(1):** the elementary score at c = 1.
- **Compounding:** f = 0.03 of wealth; growth = 252 x mean log(1 + f R'); the ruin bound is 1/|min R'|.
- **Kelly:** not used (see the flags).

### 7.4 Proposed tables and figures

- **E1. Control and the forecasts:** always short plus the forecasts of record. Columns: mean, t(mean), Sharpe
  mid / crossed, hit %, buys %, dSharpe vs short (HAC t). Currently:
  - always short: 0.20 / -0.27, hit 61.8 %, t(mean) 0.43 [mine];
  - blk2: 1.28 / 0.80, +1.08 vs short (t 1.71);
  - the paper's 8 forecasts: 0.43-1.50, 3 significantly above short.
- **E2. HAR + calendar -> all features, within each family** (needs a pairs CSV):
  - Sharpe rises in 50 of 51 pairs (median +0.54), 4 significantly, all tuned / Optuna tree rungs.
  - Per-bar enet +0.55 (t 1.34), lasso +0.50 (1.47), ridge +0.05 (0.09).
  - Pooled ridge / enet / lasso +0.53 / +0.61 / +0.51.
  - MinGRU ridge-init QLIKE-tuned: 1.04 -> 1.53 (GUI).
- **E3. Pooled vs per-bar, 6 linear pairs** [mine]:
  - QLIKE: per-bar better in all 6 (-7 to -13 %, was -6 to -12 %; DM <= -2.0 in five).
  - Sharpe: no pair significant (the lasso comes closest at t 1.69 / 1.66).
- **E4. Sizing rules over the model set** (614 days) [mine]:
  - Median Sharpe: sign(s) 0.92, clip 1.34, full long / scaled short 1.12, rank 1.34; always short 0.34.
  - Clip beats sign(s) for 98 of 112 forecasts (median +0.45); the gain is largest on baseline.
  - The current appendix reports only 12 forecasts.
- **Best models:**
  - per-bar enet, all features: 1.92 / 1.43 (t 2.56 vs short);
  - per-bar lasso, all: 1.88;
  - LightGBM T10 tuned: 1.81.
- **Figures:**
  - fig_qlike_sharpe.pdf (in the paper);
  - fig_qlike_sharpe_change.pdf (exists, unused);
  - cumulative P&L vs short with the tail days marked (needs regenerating for the model set).
- **Appendix robustness:**
  - shift test (results/close_shift_test/shift_test_by_k.csv);
  - Murphy diagram and splits;
  - compounding;
  - strategy variations;
  - the P&L decomposition (must be rebuilt).

### 7.5 What the numbers say

- **Significant:**
  - 20 of 112 forecasts beat always short, and all 20 read exogenous inputs: 3 paper forecasts and 17 per-bar
    forecasts on all features. No HAR + calendar forecast is significant (best t 1.85) [mine].
  - Per-bar beats pooled on QLIKE.
- **Not significant:**
  - every forecast vs R (0 of 111);
  - every step of Table 8 on Sharpe (0 of 21);
  - 47 of the 51 HAR -> all-features pairs;
  - all 6 pooled -> per-bar pairs.
- **Always short:**
  - Its mean is indistinguishable from zero.
  - By year: 2020 -0.43, 2021 -0.48, 2022 +1.94, 2023 +0.88, 2024 -2.06.
- **Hit %:** every sign(s) forecast (54-60 %) is below always short's 61.8 %, so hit % does not rank the trade.
- **QLIKE vs Sharpe:** rank correlation -0.20 (was -0.18). ES(1) tracks Sharpe at +0.55 (was +0.50;
  `\hmurRhoESMid`).
- **Shift test:** not monotone (k = 0: 1.35; -1: 0.53; -3: 1.40; +5 / +6: 1.48 / 1.58). Swings of about +-1 Sharpe
  are noise.
- **Sample splits:**
  - Median Sharpe 0.87 before 2022-05-16 and 1.14 after.
  - Pin days supply a median 123 % of the return.
- **Tail:** in 102 of 112 forecasts the edge over short turns negative without its 10 best days.
- **Compounding:**
  - Median wealth 4.4 mid and 1.8 crossed; 82 of 112 end above 1 crossed; always short ends at 0.86.
  - Drawdowns reach 64 % (mid) and 78 % (crossed).

### 7.6 Wording to change (RCO, methods)

- :101-104, "small premium": its t is 0.43, and it loses in 3 of 5 years. The same at methods:710-714.
- :127-135, "lifts every one": say in point estimate. The 8 forecasts take the same side on 82-96 % of days.
- :136-138: add "not significant" to R's +1.08.
- :147-149, "highest of the eight": this is selection on noise.
- :157-222: provenance and CPU material belongs in the appendix.
- :270-276, "112 of 112": the rows are dependent. Add that all 20 significant forecasts use all features.
- :294-325: the caption carries the analysis. 45 of the 51 pairs are tree rungs.
- :476-479 and :1043-1048, "not by itself a better trade": the evidence supports "not detectably".
- :744-753, the tail trade: reconcile with "pin days supply 123 %".
- :755-764, "crossed costs 0.48-0.50": true for the tabled rows only. Over the set, the median is 0.59 and 8
  forecasts are negative.
- :804-821: the straddle ranks first for only 4 of 112, so the instrument choice is not shown to be optimal.
- :854-865: process narration; delete.
- :875-879, sizing "turns on the sample": contradicted by E4. The same claim appears at methods:753-771,
  introduction:174-176, conclusion:94 and related_works:140-141.
- :16-30 and methods:107-110: they point to an empty post-2024 appendix.
- methods:739-745, "leans short": this is partly mechanical, because f^2 forecasts low.

### Flags (economics)

- "ATM straddle" is usually a one-strike strangle.
- **The trade depends on the back-transform:** recalibrated vs plain changes a median 83 of 866 positions; the Sharpe
  rank correlation between the two is 0.67.
- **The HAC lag (14)** for the trade t-statistics is never stated.
- **The DM name means two things:** the GUI's DM is vs the per-bar OLS, while the CSV's `dm_vs_ref` is vs blk2.
- **Stale GUI note:** rows.py:137-140 says sizing is on the 13-bar recalibration; it is not.
- **vrp_sized_rules.csv** uses the old `sub_*` rows, not the pre-tuned `subpt_*` ones.
- **Kelly** (experiments/close_kelly_sizing.py) uses the 13-bar recalibration: re-run it or drop it.
- **Stale inputs:** appendix_close_pnl.tex is fully commented out but still input by main.tex:147. close_pnl_decomp
  and strategy_variations are on the old headline forecast.
- **Elementary-score equations** exist only in the parked appendix_running_parked.tex.
- **B contamination** moves a Sharpe by up to 0.38 and flips up to 16 positions, the same size as many reported
  differences.
- **`cmShortT`** is always-short minus R, not always short's own t; it is unused.

## 8. Feature importance (skeleton; no conclusions yet)

- **Scope:** the per-bar 16:00 forecast on the all-features design (628 columns, 49 series). 147 refits, one every 10
  sessions, each scored on its causal held-out tail (2018-2024). Units: column, series, cluster.
- **Measures on disk** (results/feature_importance_1530_dedup/by_measure/, 111 CSVs):
  - trees: MDI / gain, split count, permutation P1 / P2 (QLIKE and fit-space MSE), TreeSHAP;
  - ridge / lasso: |beta sd|, linear SHAP, P1 / P2, drop-column with the penalty held.
- **Known issues:**
  - MDI's cardinality bias: the 628 columns are 267 continuous, 91 with 26-500 values, 93 with 3-25, 40 binary and
    137 constant at 16:00.
  - The noise probe: every third refit adds 3 probe columns, and TreeSHAP credits the continuous probe too.
  - Dedup (640 -> 628) and the per-window mask are confounded with the newer tree settings.
  - Correlated-cluster dilution.
  - The linear QLIKE permutation is driven by single tails.
- **Exists today:**
  - In the paper: only B.16.7 (appendix_campaign_16h.tex:728-778).
  - Standalone PDFs: feature_importance_1530.pdf, feature_importance_1530_dedup.pdf, model_diagnostics_1530.pdf (the
    last built on the out-of-scope live-feasible arm).
- **Open questions:**
  - the 16:00 per-bar models only, or the pooled ones too;
  - the primary measure (P2, SHAP or MDI) and level;
  - the inclusion bar (probe, interval or StepM);
  - re-running the linear side on the paper's pre-tuned arms (the lasso's har_ma share swings 60 -> 15 -> 60 %);
  - whether the dedup before/after belongs in the main text or the appendix.
- **Conflict:** the feature-importance outputs report 95 % intervals, against the paper's no-intervals rule.

## 9. Conclusion

- **Frame:** the 16:00 bar, 0DTE, and the two axes, input set and per-bar vs pooled. This replaces
  conclusion.tex:4-12.
- **Axis 1:** all features helps on both QLIKE and Sharpe, consistently but weakly. Most pairs improve, few
  individually significantly, and the pairs share data.
- **Axis 2:** per-bar fitting significantly improves accuracy on HAR + calendar. The trade effect is not significant.
- **Accuracy vs trade:** a better QLIKE is not a better trade. The trade scores only which side of IV^2 the forecast
  lands on.
- **The trade** is a demonstration, not a trading claim (RCO:1031-1050). Every forecast beats always-short, it is a
  tail trade, and the crossed spread costs Sharpe.
- **Limits:**
  - only 866 days, and the MCS has little power (the top 5 % of days carry 31-43 % of each loss);
  - one asset and one bar;
  - 413 post-2024 expirations unscored;
  - tree forecasts depend on the CPU instruction class;
  - no delta hedge and no market-impact model.
- **Future work:**
  - score the post-2024 days;
  - finish the pooled forest and the pooled MinGRU with exogenous decay;
  - a distributional head (future_work.md AM-05);
  - a second market (AM-08);
  - intraday entry timing (AM-14);
  - real cost calibration (STRAT-07).
- **Delete** the "MCS once per-bar losses are persisted" item (conclusion.tex:159-161); that MCS has been run.

### Flags (intro / related work / conclusion)

1. **Wrong frame.** The title, abstract and conclusion describe the 24-hour panel (main.tex:39, 60-62;
   conclusion.tex:4).
2. **"Statistically significant information"** (intro:57-58; conclusion:7) is unsupported:
   - on the 24-hour panel, tuned ridge DM +1.0, lasso +0.6, elastic net -0.4, fixed block ridge -1.7 (not
     significant);
   - at 16:00, 2 of 51 QLIKE gains are significant.
3. **"Best handled by the two-block ridge"** (main.tex:70-74) is contradicted: elastic net beats tuned ridge
   (DM -2.1), trees beat ridge (DM -10.7), and the top 16:00 Sharpe is the per-bar elastic net.
4. **"Tuning repeatedly worse than constants"** (main.tex:66-69) conflicts with intro:213-222.
5. **The figure's counts are mostly tree rungs** (45 of 51 pairs), so they are not independent evidence.
6. **The feature MCS / SPA / StepM results are not in the paper.** No MCS excludes HAR + calendar; StepM never names
   all features; SPA rejects "nothing beats HAR + calendar" (p 0.016-0.033) only through single added inputs. The
   "together" claim therefore rests on the aggregate pair counts.
7. **MinGRU:** no macros, text or bib entry yet. All features currently worsens its QLIKE in both fits (per-bar
   0.0990 -> 0.1054; pooled 0.1134 -> 0.1174), while improving the pooled Sharpe (1.04 -> 1.83).
8. **Pooled trees are GUI-only.** They are 10-27 % worse than per-bar on QLIKE (DM 1.8-4.8). Within pooled fits,
   all features cuts QLIKE about 13-14 %, which supports both axes. The pooled forest is still running.
9. **Stale reference.** Reference R (24-hour block ridge, QLIKE 0.1068) is a weak anchor at 16:00; the per-bar
   HAR + calendar OLS should replace it.
10. **Obsolete back-transform paragraph** (related_works.tex:83-103). AndersenBollerslev1997 is mis-cited for the
    realized-variance definition (:12). "94-97 % stable" (:26) has no macro.
11. **The tree decomposition claim** (intro:91-92; conclusion:37-38) is not re-checked at 16:00.

## 10. Bibliography

- **State:**
  - 75 entries, no duplicates.
  - 38 keys are cited in live text and all resolve; nothing cited is missing.
  - 37 entries are unused.
  - main.bbl (3 Oct) is stale.
- **Fix:**
  - HansenHuangShek2012 title braces ("GARCH"; the Hawkes / Heston titles have the same problem).
  - ShenXiu2025's `institution` field.
  - lundberg2018consistent and BacryGaiffasMuzy2015 are typed as journal "arXiv preprint".
  - 74 of 75 entries have no DOI; key styles are mixed.
- **Used in the text but not cited:**
  - Optuna (Akiba2019 is in the bib, uncited; TPE / Bergstra 2011 is absent);
  - Mincer-Zarnowitz (in the bib, uncited);
  - Newey-West and Holm (not in the bib);
  - Breiman 2001 (absent).
- **Add:**
  - 0DTE papers: Vilkov; Almeida-Freire-Hizmeri 2025; Busch-Christensen-Nielsen 2011; Clements-Liao-Tang 2022;
    Engle-Sokalska 2012.
  - Feature importance: Strobl 2007; Fisher-Rudin-Dominici 2019.
  - MinGRU (Feng 2024) and PufferLib, for the NN.
- **Live / total by topic:** RV/HAR 10/18; GARCH 0/6; shrinkage 11/11; ML / trees 5/7; SHAP 2/4; evaluation 5/6;
  retransformation 3/4; FWL 0/2; Laplacian 0/2; Hawkes / rough 0/13; Black-Scholes / Merton 2/2.

## 11. Appendix

- **Current inventory** (live inputs; letters as counted):
  - **A. Master table of campaign arms** (pooled; score_unification.py). Live, but it has **no tree rows**, although
    the trees carry 5.2 and the abstract.
  - **B.1-B.11, the running appendix:**
    - re-pick every 30 min;
    - hold to close;
    - Sharpe through the day;
    - rest-of-day forecast;
    - month-end long straddle;
    - per-bar vs pooled;
    - remaining variance;
    - sizing by the gap;
    - beside the S&P 500;
    - strategy variations.
    - **B.5 (post-2024) is a stub** cited 3 times.
  - **B.12 is stale but renders:** retired recalibration, hand-typed numbers, still referenced at RCO:861.
  - **B.13-B.16:** shift test, Murphy and splits, compounding, and the 16:00 re-run (design, mask, cadence, tuning,
    CPUs, master before/after, feature importance, compute).
  - **Parked:** window weighting, back-transformation, convention sensitivity, P&L decomposition, decision-time / VIX,
    the lean set, buckets.
- **Proposed structure:**
  - **A. Data and features:** panel and sources; a feature dictionary (41 channels -> 49 series -> 628 columns; none
    exists today); missing and stale representation; dedup and mask; SPXW and the 866 days; IV root-finding; the FOMC
    table.
  - **B. Tuning per model:** linear arms and block ridges; LightGBM, XGBoost xgb_opt_00 and forest trial 255; Optuna
    budget, picks, tuning frequency, refit cadence; the MinGRU.
  - **C. Robustness of the statistical results:** the master table with tree rows; DM / HLN / HAC conventions; MCS
    plus the feature MCS / SPA / StepM; regime and break tests; per-bar vs pooled; remaining variance; feature-
    importance details.
  - **D. Robustness of the trade:** shift test; Murphy and splits (absorb B.12); P&L decomposition (unpark it on the
    model set); compounding; sizing (including Kelly); strategy variations; other clocks; month-end; beside the S&P
    500; post-sample, only if re-scored.
  - **E. Reproducibility:** REPRODUCE.md, the lock file, design hashes, environment, repro_check (21 pass); BLAS
    settings; CPU classes; compute; a generator-to-macro map. The top-level requirements.txt is not pinned.
  - **F. NN implementation:** the PufferLib 5.0 native MinGRU (@6ffa5b10); the trainer and Muon; the float64 and
    determinism checks; gates 4-9; the walk scripts; a note that the LSTM is parked.
- **Flags:**
  - B.12 must be parked or re-run.
  - B.5 is an empty stub.
  - Table A has no trees.
  - B.16.5 and B.16.7 mix in old tree settings.
  - Ehm2016 is cited only in B.12.
  - The live Retransformation paragraph (related_works.tex:83-89) is obsolete.
  - The "Appendix D.x" numbering in comments and docs is stale.
