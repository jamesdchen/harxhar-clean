# 1,000-day design: CARC build against the cloud build (2026-10-20)

CARC: AMD EPYC 7513 (AVX2), job 13254885, files on branch insurance-selling-2026-09-19-trees-results (babf665):
`w1000/x_colstats_carc.csv`, `w1000/colsha/colsha_all_features_w1000d.csv`. Cloud: Intel Xeon (AVX-512),
`x_colstats_vm_w1000.csv` and `colsha/colsha_all_features_w1000d*.csv` here. Same code, data and library pins.

- HAR benchmark design, y, b_oos and dates_oos: identical (sha256).
- All-features X: 450 of 576 columns byte-identical; 126 differ (the same count as the cloud build with numpy's
  AVX-512 loops turned off, which matches CARC in 492 columns).
- 575 of the 576 columns: largest relative difference of any column statistic (sum, sum of squares, min, max)
  5.6e-13, i.e. round-off.
- One column, `adj_voldemand_spx_open_only_ma_5`: sum 9471.826 (CARC) vs 9469.593 (cloud), sum of squares
  23405.65 vs 23396.93, min -3.63373 vs -3.63218, max equal to 8e-14. The cloud's own AVX-512 on/off comparison
  puts the largest entry difference in this column at 4.4e-3 (17,329 rows), against a standard deviation of 0.51.

So the trees fitted on CARC's design and the linear rows of Table 1 (cloud design) see the same predictors except
for last-bit differences and this one column. The linear 2012 / 1,000-day rerun on CARC's design (pending the
author's approval) measures the effect on Table 1 directly.

## Linear check on CARC's design (job 13262220, branch insurance-selling-2026-09-19-trees-results 25f1a66)

The pooled linear 2012 / 1,000-day rerun on CARC's design
(`results/pooled_linear_rerun_2026-10-20_13_2012_w1000_carc/`), scored with `experiments/score_13bar_span.py` against
the cloud run that Table 1 prints (`results/pooled_linear_rerun_2026-10-20_13_2012_w1000/`), 2012-01-03 .. 2024-04-30,
39,975 bars:
- Every printed digit of Table 1 is unchanged: QLIKE (4 decimals), MSE (2), R2_OOS (1) and the DM statistics (2) of
  all 7 rows. Largest moves: lasso, all features, R2_OOS 43.3832 -> 43.3835 and DM -8.0677 -> -8.0711; elastic net,
  all features, DM -10.4463 -> -10.4453.
- Forecasts: the HAR-benchmark models agree to 2e-9 relative or better; on all features the largest relative
  difference of a forecast is 2.9e-5 (ridge), 3.8e-3 (elastic net) and 1.5e-2 (lasso, whose active set reacts to the
  one column that differs).
- Diebold-Mariano, CARC against cloud, daily average QLIKE: |DM| <= 1.74, p >= 0.08 for all 7 series.

Table 1's linear rows stay as they are; the trees fitted on CARC's design are compared with them on the same bars.
