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
