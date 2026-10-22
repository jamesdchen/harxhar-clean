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

---

# Follow-up, 2026-10-22 (day): one design for the 12 pooled forecasts, Table 1 trees, samples table, merge

Same branch, `insurance-selling-2026-09-19-e34` (not merged). Commits listed under Files below. `overleaf_git.sh`
was not run. `make_rehaul_data_methods.py` was not run in full: the samples and tuning tables were regenerated by
calling only their functions (the rule was lifted mid-task; the targeted run was already done and checked).

## 1. Merge of econ614 into the paper branch (done)

- `insurance-selling-2026-09-19-econ614` merged into `insurance-selling-2026-09-19`: merge commit **e5b580b**,
  normal push (6d9d9f1 → e5b580b). e34 is not merged.
- Only the build files (`paper_rehaul.pdf`, `.log`) conflicted; resolved by rebuilding. The branches changed no
  other file in common. Merged build: 52 pages, 0 errors, 0 undefined references/citations/control sequences.

## 2. The pooled linear rows on the trees' design (the switch you asked for)

**Source.** `results/pooled_linear_rerun_2026-10-20_13_2012_w1000` (13 regular-hours bars, base-5 HAR, FOMC in,
1,000-day = 13,000-bar window, penalties frozen from 4 pre-2012 folds, evaluation from 2012-01-03), the 16:00
rows. New loader **`writeup/pooled_linear_13.py`** (mirrors `pooled_trees_13.py`). Gates: 7 models × 40,497 bars;
`plain = pred_adj² × B` (max rel. diff 0); 13 bar times; each model's penalty equals `frozen_penalties.json`; the
16:00 rows reproduce the run's own `scores.csv` QLIKE to 1e-10 on its 1,082 days; the 16:00 RV equals the master
table's `rv_unclipped` on all 866 option days exactly; window/bars/first day equal the trees' (asserted).

**How it is wired.** The rerun rows are rows of their own, `poolpt13_<est>_<set>`, in place of the master table's
`poolpt_*` in every pooled row, exactly as the trees were handled in E34:

- `make_rehaul_economics.py`: E1 Panel A, E2 Panel B, E3, E4, the figures' intraday elastic net, and every
  12-forecast statistic (rank correlations, worthless-day share, May-2022 split). Before standing in, each master
  row is gated: its position, gap and QLIKE follow from its `plain` by the same rules.
- `make_rehaul_results.py`: `FAMILY_ROWS` → T1 Panel B, T2, T3 (headline and full; the master 48-interval rows stay
  in T3 full, labelled "All \rhdmNIntervals{} intervals"; 118 → 124 forecasts), and every `\rhrPool…`,
  `\rhrAf…Pool…`, `\rhrVsBm…Pool…`, `\rhrPoolVsClose…` macro, both for trades (866 days) and for accuracy (1,082-day
  span).
- The model set itself is unchanged: the master table's 48-interval `poolpt_*` stay in it, so every model-set
  statistic (`\rhePairs*`, `\rheVsShortAbove`, `\rheSigVsShort`, the sizing medians, `\rheTailNeg`, the model-set
  rank correlations, compounding, the shift test) is unchanged.
- **Scope, please check.** This reaches Section 6, not only Section 7: the 12 pooled forecasts of Section 7 *are* T1
  Panel B's, and the `\rhrAfPool…` macros feed the abstract, introduction, Sections 6 and 7 and the conclusion.
  Switching only Section 7 would have left two different "pooled ridge" forecasts in adjacent sections. To undo the
  Section 6 part only, `FAMILY_ROWS` would need a second (accuracy) key set.
- **Prose-numbers branch, checked.** `make_rehaul_rerun_text_numbers.py` there reads the 2,000-day / 2020 run, not
  w1000, and computes accuracy only (no Sharpe ratios). Its closing-half-hour Panel B points the same way as here:
  at 16:00 the full set raises QLIKE for pooled ridge and lasso.

**Economics (Section 7), sign strategy, 614 post-warmup days, Sharpe ratio mid / bid–ask (old 48-interval → new).**

| Pooled linear | HAR benchmark | Full predictor set | ΔSR mid (LW p) | Long days of 866 (HAR / full) |
|---|---|---|---|---|
| Elastic net | 0.74 / 0.41 → **1.30 / 0.96** | 1.56 / 1.23 → **1.13 / 0.79** | +0.82 → **−0.17** (0.75) | 232 / 237 → **166 / 160** |
| Lasso | 0.74 / 0.40 → **1.28 / 0.93** | 1.46 / 1.12 → **1.02 / 0.68** | +0.72 → **−0.26** (≈1.0) | 227 / 205 → **167 / 160** |
| Ridge | 0.75 / 0.41 → **1.30 / 0.96** | 1.44 / 1.10 → **1.09 / 0.75** | +0.69 → **−0.22** (0.96) | 231 / 235 → **166 / 164** |

Over the 12 pooled forecasts (trees unchanged from the E34 section above):

- Full set raises the pooled SR in `\rhrAfPoolShHigher` **5 → 2** of 6 classes (LightGBM, XGBoost); significant
  (Ledoit–Wolf) `\rhrAfPoolShHigherSig` 1 → **0**; mean differential significant 0 → 0.
- ΔSR range −0.18…+0.82 → **−0.26…+0.31**; median +0.50 → **−0.18**.
- SR range, HAR 0.74–1.21 → **0.96–1.30**; full 1.03–1.56 → **1.02–1.34**. Bid–ask: HAR 0.40–0.86 → **0.63–0.96**;
  full 0.69–1.23 → **0.68–1.00**.
- Above the short straddle 9 → **12** of 12; significant 0 → 0; t range 0.60–1.71 → **0.79–1.70**.
- Sizing medians: sign 1.05 → **1.11**, clip 1.44 → **1.40**, rank 1.41 → **1.39**; clip beats sign 12 → **9** of 12.
- Rank correlations: QLIKE vs SR −0.66 → **+0.05**; ES₁ vs SR −0.13 → **+0.31** (the "both negatively
  rank-correlated" sentences in Section 7 and the conclusion are reworded without a sign).
- Worthless-day share of positive returns: median 37% → **39%** (range 34–39% → **36–40%**).
- Intraday elastic net, cumulative gain over short: +79.3 → **+75.1** premia; without its 10 best days −6.3 → **−10.7**.
- Pooled vs closing-interval return differentials significant: 0 → 0 (sentence still asserted). Tail negative: 12 of 12.
- Robustness table (`tab:rh_app_hac_robust`): Ledoit–Wolf significant 2 → **1**; NW lag-5 1 → **0**; lag-12 2 → **1**.

**Accuracy (Section 6), 16:00 rows on the 1,082-day span (old → new).**

| Pooled linear | QLIKE HAR | QLIKE full | ΔQLIKE full vs HAR (DM p) | Excess over closing-interval HAR benchmark, full (p) |
|---|---|---|---|---|
| Elastic net | 0.1142 → **0.1205** | 0.1078 → **0.1194** | −5.6% (0.036) → **−0.9% (0.76)** | +4.4% (0.24) → **+15.6% (0.004)** |
| Ridge | 0.1143 → **0.1205** | 0.1086 → **0.1236** | −5.0% (0.077) → **+2.6% (0.54)** | +5.1% (0.17) → **+19.7% (0.001)** |
| Lasso | 0.1135 → **0.1203** | 0.1101 → **0.1229** | −3.0% (0.26) → **+2.2% (0.56)** | +6.7% (0.11) → **+19.0% (0.001)** |

- Full set lowers the closing-half-hour QLIKE in `\rhrAfPoolLower` **6 → 4** of 6 classes; significant
  `\rhrAfPoolLowerSig` **1 → 0**. MSE higher with the full set 4 → **6** of 6 (none significant).
- Gap to the closing-interval HAR benchmark narrows with the full set in **4** of 6 classes (new
  `\rhrVsBmPoolNarrow`; was all 6); full-set linear gap significant in **3** of 3 (new `\rhrVsBmPoolLinFullSig`; was
  0).
- Pooled vs closing-interval (T2), linear: HAR +10.0–10.7% → **+16.5–16.7%**; full +5.3–11.1% → **+17.7–24.0%**,
  significant in 1 → **3** of 3 (new `\rhrLinPoolVsCloseFullSig`). Pooled forecasts significantly worse than the
  benchmark 7 → **10** of 12.
- **Contrast with Table 1 (every bar, 2012–2024, same forecasts):** there the full set lowers QLIKE significantly
  in all 6 classes (`\rhorAfLowerSig` 6 of 6). The accuracy gain of the exogenous predictors shows on the full
  regular-hours day, not on the 16:00 bar alone.

**Text** (replaced text parked in dated comments; no rankings; counts via macros):

- Abstract: QLIKE sentence → "lower … in `\rhrAfPoolLower` of the `\rhrAfPoolPairs` classes, `\rhrAfPoolLowerSig`
  significant"; "narrows this gap across all model classes" → "in `\rhrVsBmPoolNarrow` of the `\rhrAfPoolPairs`".
  The last sentence ("higher Sharpe ratio … across every intraday model class") is **left for you (N15c)**.
- `01_introduction`, `09_conclusion`: the same QLIKE counts; "the elastic net achieves a statistically significant
  reduction" → `\rhrAfPoolLinLowerSig` (0) of three; "insignificant … for the linear models" →
  `\rhrVsBmPoolLinFullSig` (3) of three; the conclusion's heading "Exogenous predictors reduce QLIKE loss…" →
  "Exogenous predictors and QLIKE loss…"; rank-correlation sentence without a sign.
- `06_results`: training-scheme bullet (one design; the model set's 48-interval runs named); E15 rewritten (the rule
  now holds for the pooled models); the "Consistent QLIKE reductions across all model classes" bullet → "QLIKE: full
  predictor set against the HAR benchmark" with counts; the linear bullet prints Δ%, DM, p and the count; "the three
  most accurate specifications" (ranking) dropped; the full-set linear gap's "none significant" → count; "narrows …
  across every intraday model class" → count; E36's stale clauses updated.
- `07_economics`: rank-correlation sentence without a sign; N15c note restated (2 of 6, four classes below zero,
  options document); final subsection: "(significant for the lasso)" → `\rhrLinPoolVsCloseFullSig` of three, and the
  trees' stale "with every difference significant" (stale since E34) → `\rhrTreePoolVsCloseHarSig` /
  `\rhrTreePoolVsCloseFullSig`; "same forecast origins as Table 1" → the 15:30 forecasts over T_oos (also in the
  appendix).
- `03_data`, `10_appendix`: the windows sentence (model set 96,000; Table 1 and Sections 6–7 13,000); N18 (the
  linear rows now have every bar; no export needed); E22's window; reproducibility list adds
  `writeup/pooled_linear_13.py`; E33 names Table 1's generator.
- `paper_rehaul.tex` E34 note: (ii) and (iii) resolved; (i) the model set still holds earlier designs (the earlier
  draft's pooled 48-interval forecasts and now also the master's 48-interval pooled linear runs).

## 3. Table 1 with the pooled trees (tasks 2 and 3; done by a subagent, reviewed)

- `writeup/make_rehaul_oos_rerun.py` adds the six base-5 trees, scored on every bar exactly like the linear rows.
  The unchanged generator first reproduced the committed outputs byte for byte (34 s). Asserted: the scored set with
  the trees equals the set without them (39,975 bars); each tree's RV is the core_stats truth; its 16:00 rows equal
  `pt13.at16`. **No existing Table 1 value changed.**
- New rows (MSE ×10⁻¹⁰ / QLIKE / R²_OOS %, HAR → full; DM full vs HAR): LightGBM 1.03 / 0.1881 / 62.9 → 1.23 /
  0.1698 / 55.7 (DM −11.67, p < 0.001); XGBoost 1.02 / 0.1876 / 63.4 → 1.11 / 0.1653 / 60.2 (−10.78); random forest
  1.01 / 0.1893 / 63.5 → 1.16 / 0.1765 / 58.5 (−11.86).
- New macros: `\rhorAfPairs` 6, `\rhorAfLower` 6, `\rhorAfLowerSig` 6, `\rhorAfLinLower(Sig)` 3/3,
  `\rhorAfTreeLower(Sig)` 3/3, tree DM/p macros; each count asserted against its p-values.
- Caption now macro-only: "… (\rhorWindowDays-day window, hyperparameters frozen)". `\tabcolsep` 4pt to fit.
- The `(see Table 1)` after `\rhrAfPoolLowerSig` now says the counts are the closing half hour over T_oos and adds
  Table 1's own counts. The "linear models of Table 1" lines (06_results ×3, 03_data, 10_appendix ×2, introduction,
  conclusion, E1 note a, E34 note) now cover the trees.

## 4. Samples table and the tree window (task 4; subagent, reviewed)

- `make_rehaul_data_methods.py`: the regular-hours rows and windows come from `pt13.meta()` / `pl13.meta()`
  (asserted equal; the linear arms' in-span bars equal the trees'). Regenerated by calling only `write_samples`,
  `write_tuning`, `write_numbers` (each matches a scratch run byte for byte; 161 + 88 checks pass).
- `tab:rh_dm_samples`, Panel B: the all-interval row's W 24,000 → **96,000** (the model set's linear runs); new row
  "Regular-hours bars … : the trees and the pooled linear models of Table 1 and Sections 6–7", 85,314 bars,
  W **13,000**; rolling forecasts from 2012-01-03, 40,466 bars (less 26 early closes). Note rewritten.
- Tuning table (not in the paper): tree row 24,000 → 13,000; new row for the 13-bar linear models (4 × 3,250 / 325).
- **No `\rhdm` macro value changed.** `\rhdmPooledWindow` stays 24,000: its remaining uses are the window the tree
  hyperparameters were selected on and the n_min rescaling (correct as is). Four macro comments changed.
- The 4 hand-edit comment lines (E5) in the samples file are gone (a real run drops them).
- E20 and E24 (in `10_appendix.tex`, not `paper_rehaul.tex`) updated to the new window.

## 5. The random forest and the "every intraday model class" story (task B; analysis only, nothing rerun)

`writeup/RF_STORY_OPTIONS.md`. Short version:

- On one design the full set lowers the SR in **four** of six pooled classes (the three linear + RF), so fixing the
  RF alone cannot restore "every class". None of the six ΔSR is significant (LW p 0.18–0.996); the signs rest on a
  handful of days, and for the elastic net and the RF they flip under the fixed-factor annualization used before
  10-18.
- RF diagnosis: its full-set forecast has the same level, bias, long-day share (21.0% vs 21.5%) and dispersion as
  its HAR forecast (no shrinkage); feature subsampling does not starve the HAR columns (519 of 576 per split); the
  mean return differential is about zero, and the −0.18 comes from the per-year weighting (2022–23 carry 76% of the
  weight). The unscaled leaf minimum (24,000 → 13,000) applies to all three trees, not only the RF.
- Options: (A + B, recommended) report counts instead of "every class" and keep the RF as is: no compute.
  (C) pre-registered re-tune of all three trees on the pre-2012 folds (as the linear models), then re-walk: about
  5,800 CPU-h, about a day on CARC. Rejected: re-tuning the RF alone, or anything chosen on 2020–24.
- **Nothing was launched; your decision is needed** (the document ends with four questions).

## 6. Pooled MinGRU rerun on the new design (in progress)

Launched as a background task (CARC / Hoffman2): 13 bars, base 5, both predictor sets, 1,000-day window, tuned on
the pre-2012 folds. Results go to `writeup/MINGRU_RERUN_REPORT.md` when the forecasts are complete; the paper's NN
content stays parked until you decide.

## Commits, checks, build

| Commit | Content |
|---|---|
| 50eba29 | `writeup/pooled_linear_13.py` (new); `make_rehaul_economics.py`, `make_rehaul_results.py` read `poolpt13_*` |
| 8cd73b3 | Table 1 trees: `make_rehaul_oos_rerun.py`, `rehaul_results_oos.tex`, `rehaul_oos_rerun_numbers.tex`, `results/oos_span_check_2026-10-20/table1_2012_w1000/*.csv` |
| 4cb4adf | Samples/tuning tables: `make_rehaul_data_methods.py`, `rehaul_dm_{samples,tuning,numbers}.tex` |
| 63ec5a5 | Text (abstract, sections 1, 3, 5, 6, 7, 9, appendix) and `writeup/RF_STORY_OPTIONS.md` |
| bd82f95 | Regenerated outputs (`rehaul_econ_*`, `rehaul_results_*`, `rehaul_app_hac_robust`, figures) |
| (this commit) | This section and the build |

- `ruff check` passes on every changed `.py` except two F841 in `make_rehaul_data_methods.py` that are on HEAD;
  `mypy` reports no new errors (pre-existing: 3 in the econ generator, 2 in `make_rehaul_oos_rerun.py`). Changed
  ranges ruff-formatted only (AST identical).
- Build: `pdflatex`, `bibtex`, `pdflatex` ×2: **54 pages**, 0 errors, 0 undefined references, citations or control
  sequences, 0 BibTeX warnings, no overfull box wider than 5pt.

## Open / for you

1. **N15c** (abstract, introduction, conclusion: "higher Sharpe ratio … in every intraday model class"): now 2 of 6.
   See `RF_STORY_OPTIONS.md`; the sentences are untouched, the notes give the count.
2. **Scope of the switch** (section 2): Section 6's T1/T2 and accuracy prose moved with it. If you want Section 6 on
   the 48-interval rows, that needs a second key set.
3. **The model set** still holds the earlier designs (E34 note (i)), now including the master's 48-interval pooled
   linear runs; model-set statistics are unchanged by this pass.
4. **Closing half hour vs every bar**: at 16:00 the full set no longer lowers QLIKE for pooled ridge and lasso; on
   every bar (Table 1) it lowers QLIKE significantly in all six classes. Which framing the prose leads with is open
   (as in `RERUN_TEXT_REPORT_2026-10-20.md` on the prose-numbers branch).
5. `\rhdmNColsPooled` (634, the 48-interval design) is still cited as "the full predictor set"; the 13-bar design has
   576 columns (no macro yet).
6. Note E22's question (which procedure produced a Table 1 tree cell) can now be answered: the frozen configuration,
   refit at every bar.
