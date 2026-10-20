# Table 1 on 2012-2024 with a 1,000-day window (2026-10-20)

Author's decision (relayed from the main session, 2026-10-20): "switch the main window to 1,000 days, not 2,000.
Table 1 should cover 2012-2024 as a single panel, with no 2012-2019 / 2020-2024 split table."

## 1. What is applied on this branch

- **Generator.** `python writeup/make_rehaul_oos_rerun.py` (default `--span 2012`) reads
  `results/pooled_linear_rerun_2026-10-20_13_2012_w1000`: a 1,000-trading-day window (13,000 bars), penalties chosen
  once on the four validation folds 15 January 2008 .. 30 December 2011 and frozen. It scores every bar on 3 January
  2012 .. 30 April 2024: 3,075 trading days, 26 early closes dropped, 39,975 bars, one panel. The 16:00 check
  reproduces the run's own `scores.csv` QLIKE on its 1,082 days 2020-01-03 .. 2024-04-30 to 1e-10. The earlier
  configurations are parked in dated comments. `--span 2020` still writes the earlier 2020-2024 table, and it is
  byte-identical to the committed one (checked with `cmp` against `git show HEAD:`).
- **Generated files changed (by rerunning the generator, not by hand).**
  - `writeup/generated/rehaul_results_oos.tex`: the rows below. The caption now reads "..., \rhorSpanStart{} to
    \rhorSpanEnd{} (1{,}000-day window, penalties frozen from 2008--2011)". The note states the folds 2008--2011,
    the 1{,}000-day window, the span, $T_{\mathrm{oos}} = \rhorDays$ and Newey--West lag 8.
  - `writeup/generated/rehaul_oos_rerun_numbers.tex`: `\rhorNIntB` 39{,}975, `\rhorDays` 3{,}075, `\rhorHacLag` 8,
    and `\rhorDm/P{Ridge,Lasso,Enet}B`. New macros: `\rhorSpanStart`, `\rhorSpanEnd`, `\rhorEarlyCloses` (26),
    `\rhorPenYears` (2008--2011), `\rhorWindowDays` (1{,}000) and `\rhorWindowBars` (13{,}000). The Panel A macros
    are no longer written because there is no 48-interval rerun from 2012, and no `.tex` file used them.
  - CSVs are in `table1_2012_w1000/`. `table1_2012_w2000/` holds the parked 2,000-day version, and
    `results/oos_rerun_2026-10-20/` holds the 2020 table.
- **Note E36** (`writeup/rehaul/06_results.tex:160-161`) now describes the new table: the span, the 1,000-day window,
  and penalties chosen once on the folds 2008-2011 and frozen. The old text is parked on line 160.
- No subperiod table. One sentence on the 2012-2019 / 2020-2024 split (author, 2026-10-20) closes the overview
  bullet, `writeup/rehaul/06_results.tex:149-155`. Its numbers are the `\rhorSubA...` (2012-2019) and
  `\rhorSubB...` (2020-2024) macros: the days, the HAR benchmark's QLIKE and R2_OOS, and the lowest and highest
  QLIKE and R2_OOS of ridge, lasso and elastic net on all features. The rows are in
  `table1_2012_w1000/subperiods.csv` and equal `rerun_2012/metrics.csv` (w1000). On 2020-2024 the four HAR rows
  do not agree to four decimals (lasso 0.19515 prints 0.1952, the others 0.1951), so `\rhorSubBQlikeHar` prints
  0.1951--0.1952. The 2,000-day preview with subperiods has been removed.

| Model | HAR: MSE (1e-10) | QLIKE | R2_OOS % | All features: MSE (1e-10) | QLIKE | R2_OOS % | DM (p) |
|---|---|---|---|---|---|---|---|
| OLS | 0.94 | 0.1879 | 66.4 | -- | -- | -- | -- |
| Ridge | 0.94 | 0.1879 | 66.4 | 1.52 | 0.1774 | 45.4 | -7.86 (<0.001) |
| Lasso | 0.94 | 0.1879 | 66.4 | 1.58 | 0.1773 | 43.4 | -8.07 (<0.001) |
| Elastic net | 0.94 | 0.1879 | 66.4 | 1.40 | 0.1787 | 49.7 | -10.45 (<0.001) |

**Check against `experiments/score_13bar_span.py`.** The reference is `rerun_2012/metrics.csv` and `dm.csv`, run
w1000, span 2012-01-03:2024-04-30. All seven rows match: the same 39,975 bars and 3,075 days; QLIKE and R2_OOS
identical; MSE within 7e-16 (x 1e-10 units); DM and p identical, lag 8.

## 2. Prose: sentences that quote Table 1 or describe its span or window

**Applied 2026-10-20 (approved by the author)**: every diff in 2.1-2.8 below, each replaced passage parked in a dated
comment next to its replacement. The line numbers below are those before applying. Two differences from the diffs
as written: "13 regular-hours bars" and "four validation folds" are the generated macros `\rhorNBars` and
`\rhorPenFolds`, so the new prose types no numbers. The methods window, `writeup/rehaul/10_appendix.tex`
(experimental schemes, "Linear models: ..."), also takes `\rhorWindowBars` / `\rhorWindowDays`. The split sentence
takes its years from `\rhorSubBStartYear`, `\rhorSubALabel` and `\rhorSubBLabel`. Note E36 was updated to
match. Not applied: the items of 2.9, which the switch does not change.

These are proposals for the author to review. Line numbers are on this branch's head. Each diff keeps the
`\rhoos...` span (2020-2024) where the sentence also covers Table 2 or the straddle sample, which stay on that span.
No new symbol is introduced for the second span.

### 2.1 Abstract, `writeup/paper_rehaul.tex:117-119`

```diff
 intervals and evaluate the forecast issued at 15:30 for the closing half hour over the out-of-sample period from
-\rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days). We compare a benchmark
+% 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "\rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days). We compare a benchmark"
+\rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days), and the forecasts of every
+regular-hours bar from \rhorSpanStart{} to \rhorSpanEnd{} (\rhorDays{} trading days). We compare a benchmark
```

Lines 121-132 quote `\rhrAfPoolPairs{}` / `\rhrAfPoolLowerSig{}`. Those counts come from the earlier runs (notes E34,
E36), not from Table 1's generator, so this change leaves them as they are.

### 2.2 Introduction, `writeup/rehaul/01_introduction.tex:58-60`

```diff
   \item We evaluate predictive accuracy via the average QLIKE loss over the out-of-sample evaluation span
-    $\mathcal T_{\mathrm{oos}}$ from \rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$
-    trading days; see Section~\ref{sec:rh_results_eval}).
+  % 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "$\mathcal T_{\mathrm{oos}}$ from \rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$ trading days; see Section~\ref{sec:rh_results_eval})."
+    $\mathcal T_{\mathrm{oos}}$ from \rhoosSpanStart{} to \rhoosSpanEnd{} ($T_{\mathrm{oos}} = \rhoosDays$
+    trading days; see Section~\ref{sec:rh_results_eval}); the accuracy of the pooled linear models in
+    Table~\ref{tab:rh_oos} is scored on every regular-hours bar from \rhorSpanStart{} to \rhorSpanEnd{}
+    (\rhorDays{} trading days, a rolling window of \rhorWindowDays{} trading days).
```

Lines 64-68 quote `\rhrAfPool...` (E34, E36) and are unchanged by this.

### 2.3 Data, `writeup/rehaul/03_data.tex:225-227` (window) and `:257-261` (span)

```diff
   \item The pooled intraday sample comprises \rhdmPooledRows{} half-hour intervals beginning in June 1998, using
     rolling estimation windows of length $W = \rhdmPooledWindow$ intervals for the tree ensembles and
-    $W = \rhdmPooledLinWindow$ intervals for the linear models (see Appendix~\ref{sec:rh_exp_schemes}):
+    % 2026-10-20 (user): parked -- Table 1's linear models use a 1,000-day window of 13 bars (TABLE1_SWITCH.md). Was: "$W = \rhdmPooledLinWindow$ intervals for the linear models (see Appendix~\ref{sec:rh_exp_schemes}):"
+    $W = \rhdmPooledLinWindow$ intervals for the linear models (see Appendix~\ref{sec:rh_exp_schemes}); the
+    linear models of Table~\ref{tab:rh_oos} are trained on the 13 regular-hours bars with
+    $W = \rhorWindowBars$ bars (\rhorWindowDays{} trading days):
```

(`\rhdmPooledLinWindow`, 96{,}000 = 2,000 days x 48, comes from `make_rehaul_data_methods.py`, which is not to be run.
The diff therefore uses the `\rhorWindow...` macros.)

```diff
     models' pre-sample period (Appendix~\ref{app:rh_closing_sample}; see Section~\ref{sec:rh_results_eval}).
+  % 2026-10-20 (user): added -- Table 1 scored from 2012 (TABLE1_SWITCH.md).
+  \item Table~\ref{tab:rh_oos} is scored on a longer span, \rhorSpanStart{} to \rhorSpanEnd{} (\rhorDays{}
+    trading days, after excluding \rhorEarlyCloses{} early-close sessions); its penalties are chosen once on four
+    validation folds covering \rhorPenYears{} and then frozen.
```

### 2.4 Results, training scheme, `writeup/rehaul/06_results.tex:76-79`

```diff
 \item \textbf{Training scheme.} Under the pooled intraday estimation scheme, models are fit across all
   $M = \rhdmNIntervals$ half-hour intervals of each trading day using a rolling window of
   $W = \rhdmPooledWindow$ intervals for the tree ensembles and $W = \rhdmPooledLinWindow$ intervals for the
-  linear specifications. On the full predictor set, ...
+  % 2026-10-20 (user): parked -- Table 1's linear models (TABLE1_SWITCH.md). Was: "linear specifications. On the full predictor set, ..."
+  linear specifications. The linear models of Table~\ref{tab:rh_oos} are fit on the 13 regular-hours bars with a
+  rolling window of \rhorWindowBars{} bars (\rhorWindowDays{} trading days). On the full predictor set, ...
```

### 2.5 Results, experimental setup, `writeup/rehaul/06_results.tex:58-63`

```diff
   pooled intraday linear forecasts immediately following the pre-sample validation period (\cmPtValFirst{} to
   \cmPtValLast) used to select the linear regularization penalties.
+  % 2026-10-20 (user): added -- Table 1 scored from 2012 (TABLE1_SWITCH.md).
+  The pooled linear models of Table~\ref{tab:rh_oos} are scored on a longer span, from \rhorSpanStart{} to
+  \rhorSpanEnd{} (\rhorDays{} trading days, after excluding \rhorEarlyCloses{} early-close sessions), with a
+  rolling window of \rhorWindowDays{} trading days and penalties chosen once on four validation folds covering
+  \rhorPenYears{}, before the first forecast, and then frozen.
```

(`\cmPtValFirst` .. `\cmPtValLast`, 6 January 2016 .. 2 January 2020, remains the pre-sample of the closing-interval
models and of Table 2.)

### 2.6 Results, overview, `writeup/rehaul/06_results.tex:140-143`

```diff
-  benchmark and the full predictor set across the \rhoosDays{} trading days of the evaluation span
-  $\mathcal T_{\mathrm{oos}}$. The corresponding metrics for specifications estimated exclusively on the closing
+  % 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "benchmark and the full predictor set across the \rhoosDays{} trading days of the evaluation span $\mathcal T_{\mathrm{oos}}$. The corresponding metrics for specifications estimated exclusively on the closing"
+  benchmark and the full predictor set across the \rhorDays{} trading days from \rhorSpanStart{} to
+  \rhorSpanEnd. The corresponding metrics for specifications estimated exclusively on the closing
```

### 2.7 Conclusion, `writeup/rehaul/09_conclusion.tex:34`

```diff
-  \item Our out-of-sample evaluation spans $T_{\mathrm{oos}} = \rhoosDays$ trading days from \rhoosSpanStart{} to \rhoosSpanEnd.
+  % 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "\item Our out-of-sample evaluation spans $T_{\mathrm{oos}} = \rhoosDays$ trading days from \rhoosSpanStart{} to \rhoosSpanEnd."
+  \item Our out-of-sample evaluation spans $T_{\mathrm{oos}} = \rhoosDays$ trading days from \rhoosSpanStart{} to \rhoosSpanEnd; the forecast accuracy of the pooled linear models (Table~\ref{tab:rh_oos}) is scored on \rhorDays{} trading days from \rhorSpanStart{} to \rhorSpanEnd.
```

Line 38 (`post-\rhoosSpanEnd{}`) has the same end date on both spans and needs no change. Line 19 quotes
`\rhrAfPool...` (E34, E36) and is unchanged by this.

### 2.8 Appendix, `writeup/rehaul/10_appendix.tex:404` and `:591`

```diff
-\item \textbf{Reported comparisons.} Forecast accuracy on $\mathcal T_{\mathrm{oos}}$ is Table~\ref{tab:rh_oos}.
+% 2026-10-20 (user): parked -- Table 1 scored from 2012 (TABLE1_SWITCH.md). Was: "\item \textbf{Reported comparisons.} Forecast accuracy on $\mathcal T_{\mathrm{oos}}$ is Table~\ref{tab:rh_oos}."
+\item \textbf{Reported comparisons.} Forecast accuracy of the pooled linear models is Table~\ref{tab:rh_oos}, from \rhorSpanStart{} to \rhorSpanEnd.
```

```diff
-  \item \texttt{writeup/make\_rehaul\_oos\_table.py}: Table~\ref{tab:rh_oos} and the last-30-minutes split.
+  % 2026-10-20 (user): parked -- Table 1 has its own generator (TABLE1_SWITCH.md). Was: "\item \texttt{writeup/make\_rehaul\_oos\_table.py}: Table~\ref{tab:rh_oos} and the last-30-minutes split."
+  \item \texttt{writeup/make\_rehaul\_oos\_rerun.py}: Table~\ref{tab:rh_oos}.
+  \item \texttt{writeup/make\_rehaul\_oos\_table.py}: Table~\ref{tab:rh_oos_last30}.
```

### 2.9 Checked: nothing in Table 1's change touches these

- `writeup/rehaul/05_methods.tex:549` defines $\mathcal T_{\mathrm{oos}}$ with no dates or counts.
- `writeup/rehaul/06_results.tex:162-165` ($R^2_{\mathrm{OOS}}$ bullet): the benchmark history starts at
  `\rhoosBenchStart` (5 January 1998) on both spans. Its "closing-half-hour" wording is the stale description that
  E36 already names.
- `writeup/rehaul/06_results.tex:321-330` and `:476-482` cite Table~\ref{tab:rh_oos} for `\rhrAf...Pool...` and
  `\rhrQ...PoolFull` values from `make_rehaul_results.py` (earlier runs, E34, E36). Those values do not come from
  Table 1's generator.
- `writeup/paper_rehaul.tex:142` (E34) refers to the tree and OLS rows of the earlier table.
- `writeup/rehaul/10_appendix.tex:456-470` (the closing-interval sample, its 2,000-day window and its pre-sample) and
  `writeup/rehaul/07_economics.tex:379` (the closing-interval window `\cmPtFitWin`) are not Table 1.
- `writeup/generated/rehaul_dm_samples.tex` (note: "The pooled linear models use $W = \rhdmPooledLinWindow$
  intervals (\cmPtFitWin{} trading days)") is written by `make_rehaul_data_methods.py`, which is not to be run. It
  is out of date for Table 1's models and needs a decision on how that generator is updated.
