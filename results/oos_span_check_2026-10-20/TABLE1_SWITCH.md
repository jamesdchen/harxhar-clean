# Table 1 on 2012-2024: what the switch would change (prepared 2026-10-20, not applied)

Nothing in the paper changes until the author decides. With no option, `writeup/make_rehaul_oos_rerun.py` writes the
same bytes as before (checked with `git diff` after running it: no change in `writeup/generated/` or in
`results/oos_rerun_2026-10-20/`).

Preview: `preview_table1_2012.pdf` is `writeup/paper_rehaul.tex` compiled unchanged against a copy of
`writeup/generated/` in which the two files in `preview_generated/` replace the committed ones. Only Table 1 and the
E36 test values change in it; the prose below still reads the 2020 span, which is what the diffs below would fix.

## 1. Generator command

```
python writeup/make_rehaul_oos_rerun.py --span 2012
```

Reads `results/pooled_linear_rerun_2026-10-20_13_2012_w2000` (2,000-day window, penalties chosen once on the four
validation folds 15 January 2008 .. 30 December 2011 and frozen), scores every bar on 2012-01-03 .. 2024-04-30
(3,075 trading days, 26 early closes dropped, 39,975 bars). The 16:00 check reproduces the run's own `scores.csv` QLIKE
on its 1,082 days 2020-01-03 .. 2024-04-30 to 1e-10.

Table 1 on this span:

| Model | HAR: MSE (1e-10) | QLIKE | R2_OOS % | All features: MSE (1e-10) | QLIKE | R2_OOS % | DM (p) |
|---|---|---|---|---|---|---|---|
| OLS | 0.94 | 0.1871 | 66.3 | -- | -- | -- | -- |
| Ridge | 0.94 | 0.1871 | 66.3 | 1.69 | 0.1768 | 39.2 | -7.07 (<0.001) |
| Lasso | 0.94 | 0.1871 | 66.3 | 1.65 | 0.1771 | 40.8 | -7.71 (<0.001) |
| Elastic net | 0.94 | 0.1871 | 66.3 | 1.79 | 0.1769 | 35.5 | -6.95 (<0.001) |

Newey-West lag 8 (6 on 2020-2024 alone, 7 on 2012-2019).

Subperiods (the same bars cut by date; QLIKE, HAR benchmark / all features; DM all features vs HAR benchmark):

| | days | bars | OLS | Ridge | Lasso | Elastic net | DM ridge | DM lasso | DM elastic net |
|---|---|---|---|---|---|---|---|---|---|
| 2012-01-03 .. 2019-12-31 | 1,992 | 25,896 | 0.1841 | 0.1841 / 0.1719 | 0.1842 / 0.1719 | 0.1841 / 0.1720 | -7.67 (<0.001) | -8.14 (<0.001) | -7.34 (<0.001) |
| 2020-01-03 .. 2024-04-30 | 1,082 | 14,066 | 0.1926 | 0.1926 / 0.1860 | 0.1927 / 0.1867 | 0.1926 / 0.1861 | -2.35 (0.019) | -2.53 (0.011) | -2.38 (0.018) |

These agree with `README.md` (rerun_2012, 2,000 days) to the printed digits.

## 2. Generated files that change

| file | change |
|---|---|
| `writeup/generated/rehaul_results_oos.tex` | Table 1 rows above; caption "..., \rhorSpanStart{} to \rhorSpanEnd{} (2{,}000-day window, penalties frozen from 2008--2011)"; the note states the folds 2008--2011, the span from `\rhorSpanStart` to `\rhorSpanEnd`, `$T_{\mathrm{oos}} = \rhorDays$`, Newey--West lag 8 |
| `writeup/generated/rehaul_oos_rerun_numbers.tex` | `\rhorNIntB` 39{,}975, `\rhorDays` 3{,}075, `\rhorHacLag` 8, `\rhorDm/P{Ridge,Lasso,Enet}B` as above; new `\rhorSpanStart`, `\rhorSpanEnd`, `\rhorEarlyCloses` (26), `\rhorPenYears` (2008--2011), and the subperiod macros `\rhorSubOne...` (2012-2019) and `\rhorSubTwo...` (2020-2024): `Start`, `End`, `Days`, `NInt`, `Qlike{Ols,Ridge,Lasso,Enet}{Har,Af}`, `Dm{Ridge,Lasso,Enet}`, `P{Ridge,Lasso,Enet}`. The Panel A macros (`\rhorNIntA`, `\rhorDm/P...A`) are dropped: there is no 48-interval rerun from 2012, and no `.tex` file uses them |
| `results/oos_span_check_2026-10-20/table1_2012/` | `metrics.csv`, `dm.csv`, `subperiods.csv` (already committed on this branch; `results/oos_rerun_2026-10-20/` keeps the 2020 CSVs) |

Unchanged: `writeup/generated/rehaul_oos_numbers.tex` (`\rhoosDays`, `\rhoosSpanStart`, `\rhoosSpanEnd`,
`\rhoosEarlyCloses`, written by `make_rehaul_oos_table.py`). Table 2 (`tab:rh_oos_last30`), the straddle sample and
every `\rhr...` macro stay on 2020-2024. After the switch the paper has two spans: Table 1 on 2012-2024, the rest on
2020-2024. The diffs below say so where a sentence describes the span; they do not introduce a new symbol for it.

To adopt: make `--span 2012` the default of `make_rehaul_oos_rerun.py` (or run it with the option), rerun it, apply
the diffs below, then compile.

## 3. Sentences that quote Table 1 numbers or describe its span

Line numbers are on branch `insurance-selling-2026-09-19` at 7a509ff.

### 3.1 Abstract, `writeup/paper_rehaul.tex:117-119`

```diff
 intervals and evaluate the forecast issued at 15:30 for the closing half hour over the out-of-sample period from
-\rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days). We compare a benchmark
+% 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "\rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days). We compare a benchmark"
+\rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days), and the forecasts of every
+regular-hours bar from \rhorSpanStart{} to \rhorSpanEnd{} (\rhorDays{} trading days). We compare a benchmark
```

The rest of the abstract (lines 121-132) quotes `\rhrAfPoolPairs{}` / `\rhrAfPoolLowerSig{}`, counts from the earlier
runs (E34, E36), not Table 1 numbers; the switch does not change them.

### 3.2 Introduction, `writeup/rehaul/01_introduction.tex:58-60`

```diff
   \item We evaluate predictive accuracy via the average QLIKE loss over the out-of-sample evaluation span
-    $\mathcal T_{\mathrm{oos}}$ from \rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$
-    trading days; see Section~\ref{sec:rh_results_eval}).
+  % 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "$\mathcal T_{\mathrm{oos}}$ from \rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days; see Section~\ref{sec:rh_results_eval})."
+    $\mathcal T_{\mathrm{oos}}$ from \rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$
+    trading days; see Section~\ref{sec:rh_results_eval}); the accuracy of the pooled linear models in
+    Table~\ref{tab:rh_oos} is scored on every regular-hours bar from \rhorSpanStart{} to \rhorSpanEnd{}
+    (\rhorDays{} trading days).
```

Lines 64-68 (main findings) quote `\rhrAfPoolPairs{}` / `\rhrAfPoolLowerSig{}` (E34, E36); not changed by the switch.

### 3.3 Data, `writeup/rehaul/03_data.tex:257-261`

```diff
   \item The primary out-of-sample evaluation span $\mathcal T_{\mathrm{oos}}$ covers \rhoosSpanStart{} to
     \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days on which all rolling forecasts are available,
     after excluding \rhoosEarlyCloses{} early-close sessions). This span begins immediately following the linear
     % 2026-10-17 (user): parked -- closing-interval material (the pre-sample moved to Appendix app:rh_closing_sample). Was: "pre-sample period described above (see"
     models' pre-sample period (Appendix~\ref{app:rh_closing_sample}; see Section~\ref{sec:rh_results_eval}).
+  % 2026-10-20 (user): added -- Table 1 scored from 2012 (TABLE1_SWITCH.md).
+  \item Table~\ref{tab:rh_oos} is scored on a longer span, \rhorSpanStart{} to \rhorSpanEnd{} (\rhorDays{}
+    trading days, after excluding \rhorEarlyCloses{} early-close sessions); its penalties are chosen once on four
+    validation folds covering \rhorPenYears{} and then frozen.
```

### 3.4 Results, experimental setup, `writeup/rehaul/06_results.tex:58-63`

```diff
 \item \textbf{Evaluation span $\mathcal T_{\mathrm{oos}}$} (see Section~\ref{sec:rh_results_eval}). We evaluate
   the rolling out-of-sample forecasts over the period from \rhoosSpanStart{} to \rhoosSpanEnd, which comprises
   $T_{\mathrm{oos}} = |\mathcal T_{\mathrm{oos}}| = \rhoosDays$ trading days on which all competing forecasts are
   available (after excluding \rhoosEarlyCloses{} early-close sessions). This evaluation span begins with the
   pooled intraday linear forecasts immediately following the pre-sample validation period (\cmPtValFirst{} to
   \cmPtValLast) used to select the linear regularization penalties.
+  % 2026-10-20 (user): added -- Table 1 scored from 2012 (TABLE1_SWITCH.md).
+  The pooled linear models of Table~\ref{tab:rh_oos} are scored on a longer span, from \rhorSpanStart{} to
+  \rhorSpanEnd{} (\rhorDays{} trading days, after excluding \rhorEarlyCloses{} early-close sessions), with a
+  rolling window of 2{,}000 trading days and penalties chosen once on four validation folds covering
+  \rhorPenYears{}, before the first forecast, and then frozen.
```

(`\cmPtValFirst` .. `\cmPtValLast`, 6 January 2016 .. 2 January 2020, is the pre-sample of the 2020 runs; it stays
right for the closing-interval models and for Table 2.)

### 3.5 Results, overview, `writeup/rehaul/06_results.tex:140-143`

```diff
 \item \textbf{Overview of forecasting performance.} Table~\ref{tab:rh_oos} reports the out-of-sample MSE, average
   QLIKE loss, and out-of-sample $R^2_{\mathrm{OOS}}$ for each pooled intraday model class under both the HAR
-  benchmark and the full predictor set across the \rhoosDays{} trading days of the evaluation span
-  $\mathcal T_{\mathrm{oos}}$. The corresponding metrics for specifications estimated exclusively on the closing
+  % 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "benchmark and the full predictor set across the \rhoosDays{} trading days of the evaluation span $\mathcal T_{\mathrm{oos}}$. The corresponding metrics for specifications estimated exclusively on the closing"
+  benchmark and the full predictor set across the \rhorDays{} trading days from \rhorSpanStart{} to
+  \rhorSpanEnd. The corresponding metrics for specifications estimated exclusively on the closing
```

### 3.6 Note E36, `writeup/rehaul/06_results.tex:153`

Replace the decision sentence and the test values; the Problem paragraph stays.

```diff
+% 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "... (base-5 HAR horizons, FOMC series in, penalties chosen once on the pre-sample, a 2{,}000-day window), with every bar scored. ... New tests, QLIKE, all features against the HAR benchmark of the same model, $p$-values in parentheses: ridge \rhorDmRidgeB{} (\rhorPRidgeB), lasso \rhorDmLassoB{} (\rhorPLassoB), elastic net \rhorDmEnetB{} (\rhorPEnetB). ..."
-  \editc{orange}{E36}{Decision (user, 2026-10-20): Table~\ref{tab:rh_oos} is like for like. It holds the pooled linear rerun trained and scored on the 13 regular-hours bars (base-5 HAR horizons, FOMC series in, penalties chosen once on the pre-sample, a 2{,}000-day window), with every bar scored. Its tests are ...
+  \editc{orange}{E36}{Decision (user, 2026-10-20): Table~\ref{tab:rh_oos} is like for like. It holds the pooled linear rerun trained and scored on the 13 regular-hours bars (base-5 HAR horizons, FOMC series in, a 2{,}000-day window, penalties chosen once on four validation folds covering \rhorPenYears{} and frozen), with every bar scored from \rhorSpanStart{} to \rhorSpanEnd{} (\rhorDays{} trading days). Its tests are ...
```

and, in the same note,

```diff
-New tests, QLIKE, all features against the HAR benchmark of the same model, $p$-values in parentheses: ridge \rhorDmRidgeB{} (\rhorPRidgeB), lasso \rhorDmLassoB{} (\rhorPLassoB), elastic net \rhorDmEnetB{} (\rhorPEnetB). The like-for-like ...
+New tests, QLIKE, all features against the HAR benchmark of the same model, $p$-values in parentheses: ridge \rhorDmRidgeB{} (\rhorPRidgeB), lasso \rhorDmLassoB{} (\rhorPLassoB), elastic net \rhorDmEnetB{} (\rhorPEnetB). On \rhorSubOneStart{} to \rhorSubOneEnd: ridge \rhorSubOneDmRidge{} (\rhorSubOnePRidge), lasso \rhorSubOneDmLasso{} (\rhorSubOnePLasso), elastic net \rhorSubOneDmEnet{} (\rhorSubOnePEnet). On \rhorSubTwoStart{} to \rhorSubTwoEnd: ridge \rhorSubTwoDmRidge{} (\rhorSubTwoPRidge), lasso \rhorSubTwoDmLasso{} (\rhorSubTwoPLasso), elastic net \rhorSubTwoDmEnet{} (\rhorSubTwoPEnet). The like-for-like ...
```

The test values themselves change with no edit (the `\rhorDm/P...B` macros); the preview PDF already shows -7.07,
-7.71, -6.95 (all p < 0.001) there.

### 3.7 Conclusion, `writeup/rehaul/09_conclusion.tex:34`

```diff
-  \item Our out-of-sample evaluation spans $T_{\mathrm{oos}} = \rhoosDays$ trading days from \rhoosSpanStart{} to \rhoosSpanEnd.
+  % 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "\item Our out-of-sample evaluation spans $T_{\mathrm{oos}} = \rhoosDays$ trading days from \rhoosSpanStart{} to \rhoosSpanEnd."
+  \item Our out-of-sample evaluation spans $T_{\mathrm{oos}} = \rhoosDays$ trading days from \rhoosSpanStart{} to \rhoosSpanEnd; the forecast accuracy of the pooled linear models (Table~\ref{tab:rh_oos}) is scored on \rhorDays{} trading days from \rhorSpanStart{} to \rhorSpanEnd.
```

Line 38 (`post-\rhoosSpanEnd{} option expirations`): the end date is the same on both spans; no change. Line 19 quotes
`\rhrAfPoolPairs{}` / `\rhrAfPoolLowerSig{}` (E34, E36); not changed by the switch.

### 3.8 Appendix, `writeup/rehaul/10_appendix.tex:404` and `:591`

```diff
-\item \textbf{Reported comparisons.} Forecast accuracy on $\mathcal T_{\mathrm{oos}}$ is Table~\ref{tab:rh_oos}.
+% 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "\item \textbf{Reported comparisons.} Forecast accuracy on $\mathcal T_{\mathrm{oos}}$ is Table~\ref{tab:rh_oos}."
+\item \textbf{Reported comparisons.} Forecast accuracy of the pooled linear models is Table~\ref{tab:rh_oos}, from \rhorSpanStart{} to \rhorSpanEnd.
```

```diff
-  \item \texttt{writeup/make\_rehaul\_oos\_table.py}: Table~\ref{tab:rh_oos} and the last-30-minutes split.
+  % 2026-10-20 (user): parked -- Table 1 has its own generator (TABLE1_SWITCH.md). Was: "\item \texttt{writeup/make\_rehaul\_oos\_table.py}: Table~\ref{tab:rh_oos} and the last-30-minutes split."
+  \item \texttt{writeup/make\_rehaul\_oos\_rerun.py --span 2012}: Table~\ref{tab:rh_oos}.
+  \item \texttt{writeup/make\_rehaul\_oos\_table.py}: Table~\ref{tab:rh_oos_last30}.
```

(Line 591 is already out of date on the default span: `make_rehaul_oos_table.py` has written only Table 2 since
Table 1 moved to `make_rehaul_oos_rerun.py`.)

### 3.9 Checked, no change from the switch

- `writeup/rehaul/05_methods.tex:549` defines $\mathcal T_{\mathrm{oos}}$ with no dates or counts.
- `writeup/rehaul/06_results.tex:154-157` ($R^2_{\mathrm{OOS}}$ bullet): benchmark history from `\rhoosBenchStart`
  (5 January 1998) on both spans; its "closing-half-hour" wording is the stale description E36 already names.
- `writeup/rehaul/06_results.tex:313-322` and `:468-474` cite Table~\ref{tab:rh_oos} for `\rhrAfPool...`,
  `\rhrAf...PoolDm` and `\rhrQ...PoolFull` values from `make_rehaul_results.py` (earlier runs, E34, E36); those
  macros do not come from Table 1's generator and the switch leaves them as they are.
- `writeup/paper_rehaul.tex:142` (E34) refers to the tree and OLS rows of the earlier table; unaffected.
- `writeup/rehaul/10_appendix.tex:456-470` (closing-interval sample and its pre-sample): not Table 1.
