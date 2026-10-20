# Window check and 2012 rerun, scored as Table 1 (2026-10-20)

Scored with `experiments/score_13bar_span.py`, which reproduces Table 1's 13-bar numbers exactly
(`results/oos_rerun_2026-10-20`). Every bar of the 13-bar sample is scored: QLIKE and MSE on the bar's actual RV,
R2_OOS against the expanding mean of the same clock slot, Diebold-Mariano (HLN) tests on daily average losses.
Within a call, every run is scored on one common set of bars.

## window_check_2020

`--run w2000=results/pooled_linear_rerun_2026-10-19_13 --run w1000=results/pooled_linear_rerun_2026-10-20_13_w1000
--span 2020-01-03:2024-04-30 --compare w1000:w2000`. Penalties from the folds 2016-2019 in both runs.

## rerun_2012

`--run w2000=results/pooled_linear_rerun_2026-10-20_13_2012_w2000 --run w1000=results/pooled_linear_rerun_2026-10-20_13_2012_w1000
--span 2012-01-03:2019-12-31 --span 2020-01-03:2024-04-30 --span 2012-01-03:2024-04-30 --compare w1000:w2000`.
Penalties from the folds 2008-2011 in both runs.

## Headline numbers (QLIKE; HAR = the OLS on the HAR benchmark, AF = ridge / lasso / elastic net on all features)

| run | span | bars | HAR | AF | AF vs HAR, DM | HAR R2_OOS % | AF R2_OOS % |
|---|---|---|---|---|---|---|---|
| 2,000 days, folds 2016-19 (Table 1) | 2020-01-03 .. 2024-04-30 | 14,066 | 0.1926 | 0.1859-0.1866 | -2.35 .. -4.43 | 66.3 | 33.2-42.4 |
| 1,000 days, folds 2016-19 | 2020-01-03 .. 2024-04-30 | 14,066 | 0.1951 | 0.1869-0.1874 | -2.85 .. -4.71 | 66.4 | 38.0-45.7 |
| 2,000 days, folds 2008-11 | 2012-01-03 .. 2019-12-31 | 25,896 | 0.1841 | 0.1719-0.1720 | -7.34 .. -8.14 | 66.0 | 68.4-69.0 |
| 2,000 days, folds 2008-11 | 2020-01-03 .. 2024-04-30 | 14,066 | 0.1926 | 0.1861-0.1867 | -2.35 .. -2.53 | 66.3 | 28.7-35.0 |
| 2,000 days, folds 2008-11 | 2012-01-03 .. 2024-04-30 | 39,975 | 0.1871 | 0.1768-0.1771 | -6.95 .. -7.71 | 66.3 | 35.5-40.8 |
| 1,000 days, folds 2008-11 | 2012-01-03 .. 2024-04-30 | 39,975 | 0.1879 | 0.1773-0.1787 | -7.86 .. -10.45 | 66.4 | 43.4-49.7 |

1,000 against 2,000 days (DM on daily average QLIKE; positive = the 1,000-day window has the higher loss):
- 2020-2024, folds 2016-19: HAR benchmark +3.09 to +3.50 (p <= 0.002); all features +0.65 to +1.63 (p 0.10-0.51).
- 2012-2024, folds 2008-11: HAR benchmark +1.58 to +1.62 (p 0.11); all features +0.31 to +2.00 (p 0.045 for the
  elastic net, 0.34-0.76 otherwise).
