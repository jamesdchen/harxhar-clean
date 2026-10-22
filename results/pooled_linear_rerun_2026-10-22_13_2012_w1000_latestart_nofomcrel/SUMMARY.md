# Pooled linear models, 13 bars, 2012, 1,000-day window, late-start fix on, fomc_release dropped (run B, 2026-10-22)

As run A (`results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart/SUMMARY.md`: `LATE_START_WARMUP = True`,
everything else as the published w1000 run), with `EXCLUDE_FOMC_RELEASE = True` (the committed design since 0250f44),
so the full set has **564** columns (576 less the 12 `fomc_release` value and availability averages). This is the
design going forward. Commit a707349.

## Results (16:00 rows; `economics_vs_published.csv`)

| model | penalty HAR / full | SR HAR / full | ΔSR per-year [LW p] | ΔSR pooled 614d | QLIKE Δ% full vs HAR |
|---|---|---|---|---|---|
| elastic net | as published | 1.30 / 1.13 | −0.17 [0.754] | +0.11 | −0.87 |
| lasso | as published | 1.28 / 1.04 | −0.24 [0.973] | +0.02 | +2.07 |
| ridge | as published | 1.30 / 1.09 | −0.22 [0.963] | −0.02 | +2.58 |

DM p for full vs HAR 0.54–0.76; bid–ask ΔSR moves as at mid. Checks as in run A (flag-off reproduction, perturbation
test, fold-2 MSE, column changes). Timings: about 45 min on the laptop.

Note: the trees' runner (`experiments/pooled_trees_rerun_13.py`) hard-codes `N_COLS = 576`; a tree rerun on this design
needs 564 there and a design rebuilt with both flags.
