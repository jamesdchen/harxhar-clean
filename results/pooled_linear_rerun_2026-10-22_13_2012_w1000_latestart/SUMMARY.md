# Pooled linear models, 13 bars, 2012, 1,000-day window, late-start fix on, fomc_release kept (run A, 2026-10-22)

The published run (`results/pooled_linear_rerun_2026-10-20_13_2012_w1000`) with only `LATE_START_WARMUP = True`
(`src/backtest/executor.py`, commit 67308d8): a value column `adj_<c>_ma_<k>` of a series that starts after the data is
the neutral 0 until the scaler's window [t − W, t) holds W live rows (rows whose k-bar trailing average holds no bar
before the series' first print day). The rule was written down (scratchpad `latestart\RULE.md`, 2026-10-22 09:12 PDT;
addendum 09:18) before any rerun or score. `fomc_release` is kept (`--keep-fomc-release`), so the design has the
published 576 columns; this isolates the fix. Built on the laptop (branch `insurance-selling-2026-09-19-e34-latestart`,
worktree `C:\Users\james\AppData\Local\Temp\ls_wt`), commit 13d8a09.

## Why

Before its first print a series holds its first value (CAUSAL_NEUTRAL_FILL). While that constant fills > 75% of the
scaler window the IQR is exactly 0, the relative floor is 0 and the 1.0 fallback sets the scale; once live rows pass
25% the scale drops to a tiny IQR or ref_iqr (0.0046 for vix3m_ma_3125, 0.0010 for voldemand_ma_3125) while the median
is still the stale constant, so the scaled value jumps (−14.5 on 2010-10-14; 110 in 2012-01).

## Checks

- Flag off (with `--keep-fomc-release`): byte-identical to the eb78186 rebuild; a flag-off rerun on this laptop
  reproduces every published Sharpe ratio, ΔSR, LW p and pooled ΔSR exactly (scratchpad `latestart\run_C0`).
- Future-perturbation test: 0 of 204 columns move before the cutoff (4 scenarios, including moving VVIX's start).
- Fold-2 validation MSE at the smallest penalties: 24.9 / 37.8 / 27.9 → 0.065 / 0.065 / 0.066 (other folds 0.07–0.12).
- Max |scaled| over the evaluation span: vix3m_ma_3125 1.78 → 1.30; voldemand ma_3125 110 / 89 / 47 / 18 → 4.2 / 2.8 /
  2.6 / 3.7. Feature-health FAILs 19 → 14 (no late-start FAIL left; StockTwits attention / sentcount still FAIL on a
  2021-07-05 holiday spike and the era test).
- 61 columns change, all in zero prefixes ending VIX 2010-05/08, StockTwits 2013–14, VIX3M 2013-08/10, volatility
  demand 2014-12 / 2015-03, VVIX 2016-03/06 (`latestart_column_changes.csv`). No 16:00 row of an option day changes;
  the fit windows of the first 67 option days hold 4,226 changed VVIX values, so forecasts differ up to 2020-06-17
  (73 option days, all inside the 252-day warm-up; 0 post-warm-up sign flips).

## Results (16:00 rows; `economics_vs_published.csv`)

| model | penalty HAR / full | SR HAR / full | ΔSR per-year [LW p] | ΔSR pooled 614d | QLIKE Δ% full vs HAR |
|---|---|---|---|---|---|
| elastic net | 1e-6 / **1e-3** (was 1e-2) | 1.30 / 1.05 | −0.26 [0.932] | −0.04 | +2.63 |
| lasso | as published | 1.28 / 1.02 | −0.26 [0.996] | 0.00 | +2.20 |
| ridge | as published | 1.30 / 1.09 | −0.22 [0.963] | −0.02 | +2.64 |

The elastic net's full-set penalty is re-picked (a near tie on the pre-sample criterion: 0.061918 at 1e-3 vs 0.061965 at
1e-2); its change comes from that re-pick, not from the option-day rows. DM p for full vs HAR 0.54–0.76.

Timings (laptop, shared): design 668 s, pre-tune 85 s, walks 2,006 s (4 jobs); about 46 min.
