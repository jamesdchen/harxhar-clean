# Size of the strategy-return tests on synthetic data (writeup/hac_size_mc.py, 2026-10-21)

T = 614; LLSW bandwidth S = 33, b = 0.0537; simulated fixed-b two-sided 5 % critical value 2.105 (Kiefer-Vogelsang cubic 2.121). Nominal level 5 %; Monte Carlo standard error in column mc_se. Designs and tests: the module docstring.

| hypothesis          | design   | test            |   block |   resamples |   reps |   rejection_rate |   mc_se |
|:--------------------|:---------|:----------------|--------:|------------:|-------:|-----------------:|--------:|
| mean = 0            | iid_t5   | llsw_fixedb     |         |             |  10000 |           0.0459 |  0.0022 |
| mean = 0            | iid_t5   | nw_lag5_normal  |         |             |  10000 |           0.0502 |  0.0022 |
| mean = 0            | iid_t5   | nw_lag12_normal |         |             |  10000 |           0.0523 |  0.0022 |
| mean = 0            | iid_skew | llsw_fixedb     |         |             |  10000 |           0.0540 |  0.0022 |
| mean = 0            | iid_skew | nw_lag5_normal  |         |             |  10000 |           0.0598 |  0.0022 |
| mean = 0            | iid_skew | nw_lag12_normal |         |             |  10000 |           0.0613 |  0.0022 |
| mean = 0            | ar03_t5  | llsw_fixedb     |         |             |  10000 |           0.0504 |  0.0022 |
| mean = 0            | ar03_t5  | nw_lag5_normal  |         |             |  10000 |           0.0666 |  0.0022 |
| mean = 0            | ar03_t5  | nw_lag12_normal |         |             |  10000 |           0.0616 |  0.0022 |
| mean = 0            | ar05_t5  | llsw_fixedb     |         |             |  10000 |           0.0577 |  0.0022 |
| mean = 0            | ar05_t5  | nw_lag5_normal  |         |             |  10000 |           0.0894 |  0.0022 |
| mean = 0            | ar05_t5  | nw_lag12_normal |         |             |  10000 |           0.0737 |  0.0022 |
| equal Sharpe ratios | iid_t5   | ledoit_wolf     |       5 |         999 |   1000 |           0.0570 |  0.0069 |
| equal Sharpe ratios | iid_t5   | ledoit_wolf     |      10 |         999 |   1000 |           0.0460 |  0.0069 |
| equal Sharpe ratios | iid_t5   | ledoit_wolf     |      20 |         999 |   1000 |           0.0430 |  0.0069 |
| equal Sharpe ratios | iid_t5   | ledoit_wolf     |      10 |        4999 |    500 |           0.0340 |  0.0097 |
| equal Sharpe ratios | iid_skew | ledoit_wolf     |       5 |         999 |   1000 |           0.0550 |  0.0069 |
| equal Sharpe ratios | iid_skew | ledoit_wolf     |      10 |         999 |   1000 |           0.0440 |  0.0069 |
| equal Sharpe ratios | iid_skew | ledoit_wolf     |      20 |         999 |   1000 |           0.0600 |  0.0069 |
| equal Sharpe ratios | iid_skew | ledoit_wolf     |      10 |        4999 |    500 |           0.0600 |  0.0097 |
| equal Sharpe ratios | ar03_t5  | ledoit_wolf     |       5 |         999 |   1000 |           0.0580 |  0.0069 |
| equal Sharpe ratios | ar03_t5  | ledoit_wolf     |      10 |         999 |   1000 |           0.0680 |  0.0069 |
| equal Sharpe ratios | ar03_t5  | ledoit_wolf     |      20 |         999 |   1000 |           0.0470 |  0.0069 |
| equal Sharpe ratios | ar03_t5  | ledoit_wolf     |      10 |        4999 |    500 |           0.0760 |  0.0097 |
