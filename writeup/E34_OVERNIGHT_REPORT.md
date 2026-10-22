# E34 overnight report (2026-10-22)

Branch `insurance-selling-2026-09-19-e34`, off `insurance-selling-2026-09-19-econ614` at 51ca15e. Pushed and not
merged. Built on the author's PC with the base conda env (Python 3.11.4, pandas 3.0.2). Before any change, the
unchanged generators reproduced every committed `writeup/generated/*.tex` byte for byte. The figures differed in
bytes only.

## What was done

The pooled tree ensembles are LightGBM, XGBoost and the random forest, each on the HAR benchmark and on the full
predictor set. They are now the base-5 refit from branch `insurance-selling-2026-09-19-trees-results` (6746103):

- **Source:** `results/pooled_trees_rerun13_2026-10-20/w1000/forecasts.parquet` (+ `forecasts.json`,
  `STATUS.md`), copied onto this branch.
- **CARC runs:** walk jobs 13262219, 13265608 and 13265651, merge job 13278970.
- **Design:** the design of Table 1's pooled linear models (13 regular-hours bars, base-5 HAR horizons, FOMC series
  in, 576 columns on the full set).
- **Fitting:** the frozen configurations, refit at every bar on a 1,000-day (13,000-bar) window, from 2012-01-03 to
  2024-04-30.
- **Rows used:** the generators read only the 16:00 rows: the forecast of the 15:30–16:00 variance, made at 15:30.

No cluster compute was needed. All six arms were complete (6 × 40,466 bars), and the 866 option-data days
(2020-01-03 to 2024-04-30) and the 1,082-day accuracy span are both inside the run's span. Both clusters were
checked: CARC (0 jobs) and Hoffman2 are reachable. Hoffman2 has 3 old pending `pp_pooltre…` array tasks from
10/13, jobs 15156515/6/8. I left them untouched.

### Code (commits 853ddc3, 772c82e)

| File | Change |
|---|---|
| `writeup/pooled_trees_13.py` (new) | One loader for every generator. **Gates:** 6 arms × 40,466 bars; `plain = pred_adj² × B` exactly; 13 bar times; span and refit cadence match `forecasts.json`. **Provides:** `at16(est, set)` (one 16:00 row a day: `plain`, `rv`) and `meta()` (window, bars). |
| `writeup/make_rehaul_economics.py` | Full-set and HAR-benchmark pooled trees now come from pt13. New gate: the bar's RV equals the master table's `rv_unclipped` on all 866 days (it does, exactly). Parked: the base-2 gates (`levels.csv`, the master table's `lgbm`/`xgb` returns, `POOLTREE_SR`). Table E1 note a describes the new design; note b ("lacks FOMC") and the interim-forest dagger are parked in E1/E4. New macros: `\rheTreeNBars` (13), `\rheTreeWindowDays` (1,000), `\rheTreeWindowBars` (13,000), `\rheTreeRunStart` (3 January 2012). |
| `writeup/make_rehaul_results.py` (written by a subagent, reviewed) | The six `pooltree_*` keys now come from pt13, for trades (866 days) and accuracy (1,082-day rolling span). New gates: RV equals `rv_unclipped` and `rolling.history()`. The full-set LightGBM/XGBoost no longer alias the master table's `lgbm`/`xgb` rows, which stay in the model set. `INTERIM` is now empty. Base-2 values of record print `E34 CHANGED`. New count macros, each asserted against the p-values it counts: `\rhrAfPoolTreeLowerSig`, `\rhrTreePoolVsCloseHarSig`, `\rhrTreePoolVsCloseFullSig`, `\rhrVsBmPoolTreeFullSig`, `\rhrVsBmPoolHarSig`, `\rhrVsBmPoolHarSigOne`. Also fixed a crash on HEAD in `--span mt866` (`lv` on the 550-day `*_tw500` keys). |
| `writeup/make_rehaul_oos_table.py` (written by a subagent, reviewed) | Pooled-tree keys now come from pt13; there are no `lean_vol_vix` trees in the rerun. No pooled-tree number reaches the files this script writes, so it was not re-run (see Unresolved 5). |

Checks: `ruff check` passes on all four files, and `mypy` reports no new errors (the 3 in the econ generator were
already on HEAD). The generators were not ruff-formatted on HEAD, so I formatted only the changed line ranges
(`ruff format --range`); the AST is identical before and after.

### Outputs (commit dea7126)

The econ and results generators were re-run, which regenerated:

- **Numbers:** `rehaul_econ_numbers`, `rehaul_results_numbers` (115 changed macros + 6 new).
- **Parked tables:** `rehaul_econ_e1_control`, `rehaul_econ_e4_sizing`, `rehaul_results_t1_axis1`, `_t2_pooled`, `_t3_ranking`, `_t3_full`.
- **Robustness table:** `rehaul_app_hac_robust` (15 pooled-tree rows; counts unchanged).
- **Figures:** the generators' figures (f3 and f5 bytes only).

Every number in the text comes from these macros. No generated file was hand-edited.

### Text (commit c979b57), replaced text parked in dated comments

- **Claims that no longer hold with the base-5 trees** (the generator printed `E34 CLAIM`) are rewritten with
  p-values and counts, with no ranking:
  - `06_results.tex`, the tree bullet of "Consistent QLIKE reductions": was "largest relative improvements … each of which is statistically significant". It now prints the three ΔQLIKE, DM and p, and "significant … in `\rhrAfPoolTreeLowerSig` (0) of the three classes".
  - `06_results.tex`, "by model class": was "the penalized linear models consistently achieve lower average QLIKE … on both predictor sets" and "outperforming". It now gives plain ranges. On the HAR benchmark the pooled XGBoost (0.1133) is below the linear range (0.1135–0.1143).
  - `06_results.tex`, vs the closing-interval HAR benchmark: "each significant" became `\rhrVsBmPoolTreeFullSig` (1) of three. "Every difference … at the 1% level" became 5%: `\rhrVsBmPoolHarSig` (6) of 6; 1%: `\rhrVsBmPoolHarSigOne` (4).
  - `01_introduction.tex` and `09_conclusion.tex`: the "tree ensembles exhibit the largest relative improvements" sentences, and "significant for the tree ensembles", now use the same counts.
  - `10_appendix.tex` (closing-interval appendix): "with every difference significant at the 5% level" for the tree pooled-vs-closing gap now prints `\rhrTreePoolVsCloseHarSig` (1) and `\rhrTreePoolVsCloseFullSig` (2) of three.
- **Design of the trees:**
  - `03_data.tex`: the trees' window `\rhdmPooledWindow` became the regular-hours window.
  - `06_results.tex` training scheme: the "earlier specification / preliminary random forest" sentence is parked.
  - `10_appendix.tex`: the schemes bullet; the leaf-size sentence (n_min selected on 24,000 intervals, used unscaled on 13,000 bars, verified in `experiments/pooled_trees_rerun_13.py`); the reproducibility list now names `writeup/pooled_trees_13.py`.
- **Notes:**
  - **E34:** rewritten in `paper_rehaul.tex`; the old note is parked in `\iffalse`. It lists what this pass switched and what stays open.
  - **E15:** the tree clauses are dropped.
  - **E27:** no pooled tree is in the master table now.
  - **E35:** the trees are refit on the 576-column regular-hours design.
  - **E36:** one sentence added: the trees are refit, but not yet in Table 1.
  - **N18:** the six trees now have every regular-hours bar.

### Build (committed with this report)

`pdflatex`, `bibtex`, `pdflatex` ×2 on `writeup/paper_rehaul.tex`: 53 pages, 0 undefined control sequences, 0
undefined citations, 0 undefined references. `make_rehaul_data_methods.py` and `overleaf_git.sh` were not run.

## Key new numbers (old → new)

**Economics (Section 7), sign strategy, 614 post-warmup days, mid / bid–ask Sharpe ratio.**

| Pooled tree | HAR benchmark | Full predictor set | ΔSR (full − HAR) |
|---|---|---|---|
| LightGBM | 1.13 / 0.79 → **0.96 / 0.63** | 1.21 / 0.87 → **1.07 / 0.74** | +0.08 → **+0.11** |
| XGBoost | 1.22 / 0.87 → **1.03 / 0.69** | 1.26 / 0.92 → **1.34 / 1.00** | +0.05 → **+0.31** |
| Random forest | 1.19 / 0.84 → **1.21 / 0.86** | 1.00 / 0.66 → **1.03 / 0.69** | −0.19 → **−0.18** |

- **Unchanged:**
  - Full set raises the pooled SR in `\rhrAfPoolShHigher` = 5 of 6 classes; the random forest is the exception, as before.
  - Ledoit–Wolf significant: 1 of 5.
  - Mean differential significant (fixed-b): 0 of 6.
  - Pooled forecasts above the short straddle: 9 of 12; significant: 0 of 12; t range 0.60–1.71.
  - Tail negative: 12 of 12.
  - Pooled vs closing-interval return differential significant: 0.
- **Ranges and medians:**
  - ΔSR median +0.38 → **+0.50** (range −0.18 to +0.82).
  - Pooled SR range on HAR 0.74–1.22 → **0.74–1.21**; on full 1.00–1.56 → **1.03–1.56**.
- **Rank correlations over the 12 pooled forecasts:** QLIKE vs SR −0.52 → **−0.66**; ES₁ vs SR −0.22 → **−0.13** (still negative and weak, as the text says).
- **Sizing medians (12 pooled):** sign 1.20 → **1.05**, clip 1.34 → **1.44**, rank 1.40 → **1.41**. Clip beats sign 10 → **12** of 12.
- **Intraday LightGBM:** cumulative gain over short +74.9 → **+65.1** premia; without the 10 best days −13.2 → **−10.4**.

**Accuracy (Section 6), 16:00 forecasts on the 1,082-day rolling span.**

- **Mean QLIKE:**
  - HAR benchmark: LightGBM 0.1296 → **0.1144**, XGBoost 0.1279 → **0.1133**, RF 0.1301 → **0.1190**.
  - Full set: 0.1128 → **0.1125**, 0.1126 → **0.1114**, 0.1146 → **0.1140**.
- **Full vs HAR ΔQLIKE:** LightGBM −13.0% → **−1.6%** (p 0.55), XGBoost −12.0% → **−1.7%** (p 0.53), RF −11.9% → **−4.2%** (p 0.083). All three were p < 0.001.
- **Counts in the abstract and introduction:** `\rhrAfPoolLowerSig` 4 → **1** of 6 (only the pooled elastic net). The full set still lowers QLIKE in all 6 classes and still narrows the gap to the closing-interval benchmark in every class (both asserted).
- **Excess over the closing-interval HAR benchmark:** full-set trees +9.0–11.0% → **+7.9–10.4%**, significant for 1 of 3. HAR set +10.0–26.0% → **+9.7–15.3%**.
- **Pooled vs closing-interval trees, T2:** HAR +18.5–21.7% → **+4.6–11.4%**; full +7.5–9.1% → **+7.3–8.0%**.

## Unresolved / for the author

1. **N15c (open, unchanged by E34).** These statements say the Sharpe ratio rises "in every intraday model class":
   - the abstract (`paper_rehaul.tex`, last sentence);
   - `01_introduction.tex` (Economic application);
   - `09_conclusion.tex` (Economic value).

   With the base-5 trees it is still 5 of 6: the pooled random forest has ΔSR = −0.18. The N15c notes in the
   introduction, the conclusion and Section 7 are still accurate and were left for your decision. The pooled XGBoost
   is still above the LightGBM in SR, while the parked cumulative-P&L and by-year figures show the LightGBM.
2. **The model set still holds the earlier draft's pooled 48-interval forecasts.** This is the "paper" family: a0,
   blk2, …, its `lgbm`/`xgb` (base-2 trees, no FOMC), b2lasso. Every statistic over the whole model set includes them:
   - the counts against the short straddle;
   - the sizing medians;
   - the compounding (E27);
   - the interval-shift test (E26/N18).

   Changing that needs a master-table rebuild and is tied to the E25/E27 decisions.
3. **Two designs among the 12 pooled forecasts.** The pooled linear rows (`poolpt_*`) are still the 48-interval runs
   with W = 96,000 intervals (2,000 days). The trees are on the 13 regular-hours bars with a 1,000-day window.
   Like-for-like pooled linear forecasts on the trees' design exist (`results/pooled_linear_rerun_2026-10-20_13_2012_w1000`,
   Table 1, and the CARC check `lin13_2012_w1000_carc` on the trees-results branch). Branch
   `insurance-selling-2026-09-19-prose-numbers` already computes `\rhrr…` counterparts on them. Decision needed.
4. **Tree window in the samples table.** `tab:rh_dm_samples` (live) and the `\rhdm…` macros give the trees' window as
   W = 24,000 intervals. They come from `make_rehaul_data_methods.py`, which is not to be run. Notes E20 and E24
   still describe that window.
5. **`make_rehaul_oos_table.py` stops at a stale gate, unrelated to E34.**
   - **Cause:** it checks every cell against `results/rolling_eval_2026-10-15/metrics.csv` (10-15). The forecasts
     `results/spxw_pnl/yhat_subpt_{ridge,lasso,enet}_baseline.parquet` were regenerated on 10-18; the relative
     differences are up to 1.9e-4 (enet QLIKE). The unchanged HEAD script fails the same way, before writing
     anything.
   - **Effect:** its committed outputs are unchanged and contain no pooled-tree number.
   - **Fix:** exempt those three keys (recompute from data), or re-run
     `experiments/rolling_eval_2026_10_15.py`. Either way, one cell of `tab:rh_oos_last30` would move (elastic
     net, HAR benchmark, QLIKE 0.1032 → 0.1033).
6. **The 1,082-day span still depends on the old tree files.** `experiments/rolling_eval_2026_10_15.py`
   `common_span()` still reads the base-2 tree tables (including INTERIM1600) to define the span. The span is
   unchanged and the new trees are gated to cover every day of it, but `results/pooled_trees_2026-10-07/` must stay
   on disk until that script is switched.
7. **`make_rehaul_results.py --span mt866` (the 866-day reproduction check) still stops, on old non-tree values of
   record:**
   - the T2 elastic net HAR 13.348 vs 13.4;
   - three MinGRU contrast p-values about 1e-6 off;
   - the model-set counts (111, 7, 2, 51) vs (111, 7, 0, 51).

   With those made soft in a scratch copy, it completes. Its pooled-tree values of record print `E34 CHANGED`.
8. **Possible next steps the refit allows (not done):**
   - Add the trees to Table 1, scored on every bar (note E36).
   - Run the shift test on the six trees for j < 0 (N18/E26).
   - Compute compounding on the 12 pooled forecasts (E27). `make_rehaul_economics.py` already builds their daily
     returns.
9. **Smaller items:**
   - Note E22 still mentions the "draft or preliminary flag on the full-set tree rows". Table 1 no longer has tree
     rows (E36), so that note needs your restatement.
   - "(see Table 1)" after `\rhrAfPoolLowerSig` in `06_results.tex` points to a table that no longer holds the
     trees. This was already so before E34 (E36).

## Files

- **Code:**
  - `writeup/pooled_trees_13.py`
  - `writeup/make_rehaul_economics.py`
  - `writeup/make_rehaul_results.py`
  - `writeup/make_rehaul_oos_table.py`
- **Data:** `results/pooled_trees_rerun13_2026-10-20/w1000/{forecasts.parquet, forecasts.json, STATUS.md}`
- **Generated:** `writeup/generated/rehaul_{econ_numbers,econ_e1_control,econ_e4_sizing,results_numbers,results_t1_axis1,results_t2_pooled,results_t3_ranking,results_t3_full,app_hac_robust}.tex`, `writeup/figures/rehaul_econ_*.pdf`, `writeup/figures/rehaul_results_f{1,2,3,5}_*`
- **Text:**
  - `writeup/paper_rehaul.tex`
  - `writeup/rehaul/{01_introduction,03_data,04_features,06_results,07_economics,09_conclusion,10_appendix}.tex`
- **Build:** `writeup/paper_rehaul.{pdf,log,aux,bbl,blg}`
