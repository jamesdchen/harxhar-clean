# Shift (dating) test across the paper's model set (plain forecast f²·B, no recalibration)

Written by `experiments/close_shift_test_headline.py` (`main`, 2026-10-06: no headline model, user decision) from `shift_test_modelset.csv` (every forecast × k), `shift_test_by_k.csv`, `shift_test_not_shifted.csv` and `shift_test_gates.csv`. Model set: 94 forecasts (results/close_master_table/paper_model_set.csv). Each shifted row is placed on the traded bar's scale as ŷ_k²·B_16:00 (no smear, no recalibration); the deck's 15:30 sign(s) trade, mid and crossed fills; point estimates only.

- Shifted: 9 forecasts whose tables carry the 10:00–16:00 session rows (paper 9).
- Not shifted: 85 forecasts (per-bar trees 72, per-bar linear 7, pooled linear 6) — their tables hold only the 16:00 row (per-bar regressions of the 16:00 bar, and the pre-tuned pooled forecasts, exported at the 16:00 bar only), so there is no earlier or next-session row to read (`shift_test_not_shifted.csv`); nothing is fitted for this test.
- Days: the 865 deck days on which every shifted forecast has every shifted row. k = 0 on all 866 days reproduces the master table's Sharpe for every shifted forecast (gated to 1e-9).

| row | read | median Sharpe mid | range mid | median crossed | below k=0 | median Δ mid | median agree % |
|---|---|---|---|---|---|---|---|
| k=-12 | 10:00, same | 0.71 | 0.39 to 1.09 | 0.23 | 9 of 9 | -0.51 | 66 |
| k=-11 | 10:30, same | 0.81 | 0.55 to 1.02 | 0.33 | 7 of 9 | -0.57 | 66 |
| k=-10 | 11:00, same | 0.00 | -0.16 to 0.30 | -0.47 | 9 of 9 | -1.40 | 68 |
| k=-9 | 11:30, same | 0.53 | 0.31 to 0.86 | 0.05 | 9 of 9 | -0.65 | 68 |
| k=-8 | 12:00, same | 0.20 | -0.15 to 0.47 | -0.28 | 9 of 9 | -1.16 | 69 |
| k=-7 | 12:30, same | 0.67 | 0.32 to 0.96 | 0.19 | 8 of 9 | -0.54 | 71 |
| k=-6 | 13:00, same | 0.99 | 0.39 to 1.27 | 0.51 | 9 of 9 | -0.38 | 71 |
| k=-5 | 13:30, same | 1.16 | 0.94 to 1.33 | 0.67 | 7 of 9 | -0.12 | 72 |
| k=-4 | 14:00, same | 0.84 | 0.50 to 1.28 | 0.36 | 7 of 9 | -0.25 | 73 |
| k=-3 | 14:30, same | 1.40 | 0.91 to 1.55 | 0.91 | 3 of 9 | +0.09 | 73 |
| k=-2 | 15:00, same | 1.00 | 0.23 to 1.19 | 0.51 | 8 of 9 | -0.34 | 77 |
| k=-1 | 15:30, same | 0.53 | 0.31 to 0.74 | 0.04 | 8 of 9 | -0.94 | 81 |
| k=+0 | 16:00, same | 1.35 | 0.49 to 1.58 | 0.87 | 0 of 9 | +0.00 | 100 |
| k=+1 | 10:00, next | 0.41 | 0.29 to 0.68 | -0.06 | 8 of 9 | -0.94 | 67 |
| k=+2 | 10:30, next | 1.03 | 0.76 to 1.10 | 0.56 | 7 of 9 | -0.32 | 68 |
| k=+3 | 11:00, next | 1.24 | 1.05 to 1.38 | 0.76 | 6 of 9 | -0.09 | 68 |
| k=+4 | 11:30, next | 0.93 | 0.86 to 1.15 | 0.45 | 7 of 9 | -0.42 | 66 |
| k=+5 | 12:00, next | 1.48 | 1.19 to 1.57 | 1.00 | 4 of 9 | +0.12 | 67 |
| k=+6 | 12:30, next | 1.58 | 1.39 to 1.78 | 1.10 | 0 of 9 | +0.32 | 67 |
| realized (target) | 16:00 | 4.48 | | 4.02 | | | |
| realized (unclipped) | 16:00 | 4.50 | | 4.03 | | | |
| always short |  | 0.27 | | -0.22 | | | |

## Gates
- 16:00 baseline equals the target's B (a0): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (a0): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (blk2): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (blk2): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (blk2_inc): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (blk2_inc): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (lgbm): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (lgbm): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (xgb): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (xgb): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (lasso_t): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (lasso_t): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (lasso_f): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (lasso_f): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (enet): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (enet): 0 ≤ 1e-09 on n = 866 — PASS
- 16:00 baseline equals the target's B (b2lasso): 8.08e-13 ≤ 1e-09 on n = 866 — PASS
- bar+0 placed equals the master table's plain forecast (b2lasso): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (a0): 2.22e-16 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (blk2): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (blk2_inc): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (lgbm): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (xgb): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (lasso_t): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (lasso_f): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (enet): 0 ≤ 1e-09 on n = 866 — PASS
- k = 0 Sharpe mid on 866 days = the master table's (b2lasso): 0 ≤ 1e-09 on n = 866 — PASS
