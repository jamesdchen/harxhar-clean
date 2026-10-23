# tree_shap_w3000_2014start_2026-10-23

The professor asked how the nonlinear models respond to the dataset's non-standard variables in their
feature-importance measures. This study answers with TreeSHAP of the three final tree models: LightGBM, XGBoost and
the random forest on oneDAL (sklearnex). All three use the frozen configurations, the full set (564 columns),
W = 3,000 days and the 2014 start.

- **Refits:** each model is refitted at the 126 monthly origins of `fi_by_window_2014start`, exactly as the walk
  refits at that bar.
- **SHAP rows:** every one of the 33,970 forecast rows (2014-01-02 .. 2024-04-30) is explained by the refit of its
  own origin. That origin's neighbourhood is the 21 sessions it forecasts.
- **Plan:** the measures, classes and thresholds are in `RULE_shap.txt`. It was committed and pushed (e091357,
  02:05 PDT) before any refit or SHAP value.
- **Performance numbers:** none were computed or used.

**PDF of the figures:** `tree_shap_figures.pdf` (9 pages; the PNGs are in `figures/`).

## Findings

1. **M1: the trees use the non-standard variables, but as a minority input.**
   - The 36 non-standard variables hold **21.8 / 20.6 / 18.6 %** of total mean |SHAP| for LightGBM / XGBoost /
     random forest (95% block-bootstrap intervals [20.4, 23.4] / [19.3, 22.2] / [17.2, 20.3]). That is
     **44–46 %** of the exogenous total.
   - The HAR terms take 52–58 % of the total and the E-mini realized moments 21–24 %.
   - The non-standard weight sits in a few variables, the same ones in all three models: the cross-sectional
     absolute returns sumabsret_ewstock / sumabsret_vwstock (1.9–3.9 % each), fomc_until_inv (2.0 / 2.0 / 1.1 %),
     ofi_vwstock (1.4–1.8 %) and effspread_vwstock (1.3 %).
   - The rest carry little:
     - StockTwits as a bucket: 0.6–0.8 %.
     - Volatility demand (4 series together): 0.2–0.3 %.
     - vvix: 0.2–0.3 %; vix3m: 0.3–0.7 %; fomc_day: 0.1 %.
   - By year, the non-standard share dips in 2018–2021 (LightGBM 18–20 %) and is back at 24–25 % in 2023–24. The
     HAR share moves the other way. All three models show this pattern.
2. **M2: the trees lean on the non-standard variables LESS than the linear models do.**
   - Of the exogenous total, the non-standard share is 43.7–45.9 % for the trees and **67.3 %** for the W = 3,000
     ridge (s.d.-weighted share).
   - D = tree − ridge is **−21.4 / −23.6 / −23.0 pp**, every 95% interval below 0.
   - The verdict is LESS for every model against ridge, lasso (73.9 %) and elastic net (56.0 %), on every
     denominator, and with sumvolume / numobs moved to non-standard.
   - Rank agreement with the ridge is moderate. Spearman over the 46 exogenous variables is +0.59 / +0.59 / +0.55;
     the top-10 overlap is 6 / 6 / 4 of 10.
   - Where they differ: the ridge's top 10 holds the FOMC calendar twice (fomc_until_inv 2nd, fomc_day 4th). The
     trees rank fomc_day near the bottom (0.1 %) and put the E-mini sumabsret first. They also rank the
     cross-sectional absolute returns, ofi_vwstock and effspread_vwstock higher than the ridge does.
   - The trees agree with each other far more than with any linear model. Spearman is LightGBM–XGBoost 0.99 and
     either booster vs the forest 0.89–0.90.
3. **M3: the responses are mostly monotone and one-sided; none is U-shaped.**
   - Rule labels, LightGBM / XGBoost / random forest, for the variables at ≥ 0.5 % (11 / 10 / 13):
     - MONOTONE: 7 / 6 / 10.
     - IRREGULAR: 3 / 3 / 3.
     - STEP: 1 / 0 / 0. FLAT: 0 / 1 / 0. THRESHOLD and U-SHAPED: 0.
     - 9 / 7 / 13 of them carry the ASYMMETRIC flag.
   - Cross-sectional absolute returns (EW / VW) and the spreads (effspread / spread): monotone up and convex, with
     most of the response in the top decile.
     - sumabsret_vwstock moves the forecast from −1.4 % (bottom decile) to +3.9 % (top decile) of the mean forecast
       for LightGBM, +4.3 % for XGBoost and +2.3 % for the forest.
   - ofi_vwstock: monotone down. A higher order-flow imbalance gives a lower forecast, from +0.8 % to −1.1 %.
   - Cross-sectional skewness (sumret3_vwstock / _ewstock): monotone down, with the effect in the most negative
     deciles. Negative skewness gives a higher forecast.
   - fomc_until_inv (dominant column ma_25) and sellturnover_vwstock: the rule labels them IRREGULAR + ASYMMETRIC
     (top-tail) because Spearman fails on their flat lower deciles. In words (post hoc): both are flat through
     nine (eight) deciles and then jump.
     - fomc_until_inv rises to +2.9 / +2.8 / +1.4 % of the mean forecast in the top decile, the approach to a
       statement.
   - vix3m: 75 % of rows sit exactly at the neutral value 0 (82 % at or below it), so there are only 3 bins. STEP
     up for LightGBM, FLAT for XGBoost.
4. **M4: the non-standard variables matter most on FOMC days, at the statement bars, and in calm markets.**
   - **FOMC statement days:** the non-standard share is 41.7 / 40.1 / 28.1 % against 20.9 / 19.8 / 18.3 % on other
     days (+20.8 / +20.3 / +9.8 pp). This is almost all fomc_until_inv: 25.1 / 22.9 / 11.5 % of the total on those
     days, about 1 % or less otherwise.
   - **Bar of the day:**
     - The share peaks at 10:00 (24.0 / 22.7 / 20.2 %) and again at 14:30–15:00 (25.2 / 23.3 / 19.7 %). At those
       bars the FOMC-calendar bucket rises from about 1.5 % to 5 % of the total (forest: 1 % to 3 %).
     - It is lowest at midday and at 16:00 (16:00 − 10:00: −3.5 / −3.1 / −2.3 pp).
   - **VIX regime:** the share is lower in the high-VIX tercile (high − low: −2.9 / −2.8 / −3.3 pp of the total,
     all below 0). The HAR terms gain +3.8 to +4.9 pp there instead.
5. **M5: the non-standard variables act through interactions, mainly with the HAR terms.**
   - Interactions carry **33.1 / 31.0 %** of all attribution (LightGBM / XGBoost, variable level; 45.4 / 42.6 % at
     the column level).
   - They carry **49.7 / 45.7 %** of a non-standard variable's attribution against 26.8 / 26.0 % for a standard
     one.
   - The largest variable pairs involving a non-standard variable:

     | pair | share of attribution (LightGBM / XGBoost) |
     |---|---|
     | har_ma × sumabsret_vwstock | 1.26 / 1.23 % |
     | har_ma × sumabsret_ewstock | 0.86 / 0.83 % |
     | har_ma × fomc_until_inv | 0.25 / 0.40 % |
     | har_ma × sellturnover_vwstock | 0.31 / 0.36 % |
     | har_ma × effspread_vwstock | 0.30 / 0.31 % |
     | har_ma × stocktwits_sentiment | 0.28 / 0.31 % |
     | har_ma × ofi_vwstock | 0.23 / 0.30 % |

     The remaining entries of the top 10 pair the E-mini sumret / sumabsret with the cross-sectional absolute
     returns.
   - The weakest variables are used mostly through interactions: 54–66 % of the attribution of the StockTwits
     series, vvix, vix3m and fomc_day.

## Gate (`gate.csv`, `gate.txt`)

| model | where | origin refit = walk pred_adj | additivity max | other |
|---|---|---|---|---|
| LightGBM | laptop | **126/126 bit-equal** | 8.4e-15 | interaction rows sum to SHAP: 1.7e-15 |
| XGBoost | CARC 13325739, EPYC 7513 | **126/126 bit-equal** | 2.9e-6 (float32; tolerance 1e-5) | 3.6e-7; repeats at k 0 / 62 / 125 identical |
| random forest | CARC 13324940, EPYC 7513 | **31/31 bit-equal**: every origin row the rf_skx walk had written when fetched (03:27) | 4.8e-14 | oneDAL vs the explained trees 2.5e-15 rel.; repeats at k 0 / 62 / 125 identical (forecasts and SHAP) |

The fit rows equal the walk's `fit_row` at every origin. The walk rows were fetched from the CARC pieces in the
staging ssh (LightGBM / XGBoost complete) and again with the results (rf_skx). They are in
`inputs/walk_origin_rows.csv`.

## What changed against the plan, and why (all before any SHAP number was read)

- **Random forest on CARC, not the laptop.**
  - The laptop fit took about 230 s an origin (the laptop was already 65 % loaded), against 32 s on EPYC. TreeSHAP
    was 0.58 s a row, single-threaded. That projected 10–14 h.
  - The coordinator approved one small CARC job:
    - EPYC 7513 | 7542, the /home1 venv_skx read-only, shap 0.49.1 in a `--target` pylib.
    - TreeSHAP split over 16 forked processes, because shap holds the GIL.
  - The RULE's one-in-three-rows contingency was not needed: all rows were explained.
- **RF SHAP route (G2(b) caught it).**
  - shap's scikit-learn path casts X to float32, as scikit-learn trees do. The oneDAL forest routes rows on float64
    values (x ≤ threshold, float64 thresholds). Design values within float32 rounding of a threshold went the other
    way, and the additivity errors reached 1.5e-2 on the first array.
  - That array (13324817) was cancelled and its output set aside.
  - The fix keeps shap.TreeExplainer on the sklearnex estimators_ but sets `explainer.model.input_dtype = float64`
    (`code/shap_run.py`). The G2(b) reference became a float64 traversal of estimators_.
  - With the fix, additivity holds to 5e-14 and oneDAL equals the explained trees to 2.5e-15.
- **XGBoost on CARC: the laptop failed G1.**
  - 0 of 73 laptop refits were bit-equal to the walk (max 0.049, 4.2 % relative). The laptop LightGBM matched
    exactly.
  - The likely cause: XGBoost's row and column subsampling draws through C++ standard-library distributions, which
    differ between MSVC (Windows wheel) and libstdc++ (Linux). So the same seed grows a different model.
  - Following the RULE, the coordinator was told; it approved a 2-task CARC job, which passed 126/126.
  - The laptop XGBoost output is unused. It is kept outside the repository.
- **LightGBM** ran on the laptop in two sessions. The C: drive filled up at 02:28 and the first session stopped at
  origin 26 on the write error. Origins 0–25 lack the G2(a) repeat (it is the forest's gate; LightGBM was repeated
  at k 62 and 125).

## How to read this

- **What the shares measure.** A SHAP share measures how much a variable moves the forecast. It does not measure
  how much the variable improves accuracy. A variable can matter for the forecast without improving it, or the
  reverse.
- **Credit among correlated variables.** TreeSHAP here is path-dependent: the expectation is over each refit's own
  training window. Between correlated variables (EW vs VW, value vs availability averages) the credit follows the
  trees' split choices. The variable-level sum removes the split within a variable, not the split between
  variables.
- **The ridge comparison uses different measures** (|β|·s.d. against mean |SHAP|), so M2 compares rankings and
  shares, not magnitudes. In the ridge the HAR block carries a 0.01 penalty factor. The comparison with the backbone
  in the denominator is therefore secondary.
- **Bin values are model inputs.** M3's x values are the trailing-window robust-scaled design values the trees
  see, not raw levels.
- **The intervals** come from a circular moving-block bootstrap over the 126 origins (block 12, B = 2,000, seed
  20261023), paired across models and the ridge. They describe variation over the span's neighbourhoods, not the
  sampling error of one fit. M5 is descriptive (1,008 fixed rows; no intervals).

## Files

| file | contents |
|---|---|
| `RULE_shap.txt` | plan: models, gate, SHAP variant, classes, measures M1–M5, bootstrap (committed before computing) |
| `tree_shap_figures.pdf`, `figures/*.png` | 9 pages: M1 buckets, M1 top variables, M1 by year, M2 vs ridge, M3 shapes ×3 models, M4, M5 |
| `tables.txt` | every table in text form, gate included |
| `gate.csv`, `gate.txt` | per model × origin: refit and walk forecast (hex), additivity, interaction sums, oneDAL check, repeats, CPU |
| `m1_variables.csv`, `m1_buckets.csv`, `m1_classes.csv`, `m1_by_year.csv` | M1 shares with intervals, the column-level secondary, K_live / K_used |
| `m2_rank.csv`, `m2_nonstandard_D.csv`, `m2_linear_shares.csv` | M2: Spearman and top-10 overlap (all scopes, the linear models, tree vs tree), D with intervals and verdicts, the linear shares |
| `m3_shapes.csv`, `m3_bins.csv` | M3: selection, dominant column, bins, shape class, flag; per-bin means and bands |
| `m4_subsets.csv`, `m4_contrasts.csv`, `m4_origins_vix.csv` | M4: shares by VIX tercile / bar / FOMC day, contrasts with intervals, VIX at each origin |
| `m5_summary.csv`, `m5_variables.csv`, `m5_buckets.csv`, `m5_pairs_nonstandard.csv` | M5: interaction shares, per-variable interaction fraction, bucket × bucket, pairs |
| `inputs/` | the 126 origins and W = 3,000 linear shares (from -fi-window 0c44dca), fi_predictors.csv (linear FI groups), the walk's origin rows and the pieces at fetch time |
| `carc/` | Slurm logs, sacct, job ids and run_info of the two CARC arrays used |
| `code/` | below |

## Code and reproduction

- **Runner:** `code/shap_common.py` holds the paths, groups and classes, and loads the runner
  `experiments/pooled_trees_rerun_13.py` at 3cbb85f from git. `code/shap_run.py` does the refits, SHAP and
  interactions, writing one compressed file per origin.
- **CARC side:**
  - `code/carc_stage.sh` stages and smoke-tests the inputs and fetches the walk rows.
  - `code/carc_job.sbatch` runs the array, with task files `tasks.txt` (forest) and `tasks_xgb.txt`.
  - `code/carc_fetch_origin_rows.py` and `code/export_x.py` are the helpers.
- **Reduction:** `code/shap_reduce.py` computes the gate and M1–M5 (`--gate-only` first). `code/shap_figs.py`
  draws the figures and the PDF.
- **Environments:**
  - Laptop: venv C:\Users\james\AppData\Local\Temp\shap_venv with python 3.12.3, numpy 1.26.4, scikit-learn 1.9.1,
    lightgbm 4.6.0, xgboost 3.2.0, sklearnex 2026.2.0 and shap 0.49.1.
  - CARC: /home1/jc_905/trees_final/venv_skx (python 3.12.15, same versions) plus /home1/jc_905/tree_shap/pylib
    (shap 0.49.1, slicer 0.0.8).
- **Reproduce:**
  - LightGBM: `python code/shap_run.py --est lgbm` (laptop or CARC).
  - XGBoost and the forest on CARC (EPYC 7513 | 7542):
    - `sbatch --array=1-3 code/carc_job.sbatch` for the forest.
    - `sbatch --array=1-2 --export=ALL,TASKFILE=.../tasks_xgb.txt code/carc_job.sbatch` for XGBoost.
  - Then `python code/shap_reduce.py --gate-only`, `python code/shap_reduce.py` and `python code/shap_figs.py`.
    The per-origin SHAP files (about 300 MB) are not committed.
- **The code that ran:** CARC ran the committed `shap_common.py`, `shap_run.py` and `export_x.py` (sha256 in
  `code/carc_stage_out_13324817.txt` for the first upload; the second upload equals the committed files). The
  laptop LightGBM origins 0–25 ran an earlier `shap_run.py`: uncompressed output, no repeat check, the same
  computation.
