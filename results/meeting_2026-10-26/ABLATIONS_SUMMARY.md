# Predictor-group ablations and augmented-HAR benchmarks (2026-10-26)

Source: branch insurance-selling-2026-09-19-e34-ablations (62fc566), results/ablations_2014start_w3000/ (RULE.txt with
addenda A-C, STATUS.md, scores/). Generated table tab:rh_ablation (writeup/make_rehaul_ablations.py).

**Pre-patch FOMC design.** The 18 Mar 2020 meeting is missing and the proximity ramps are winsorized; FOMC rows are
preliminary. The patch is material for the linear models (after it, linear full vs HAR is -6.0 / -9.0 / -6.7%;
trees -14.1 / -14.6%).

**Setup.** Table 1's design: same rows, 3,000-day window refit every bar, same scoring and DM test; only the columns
change; hyperparameters frozen at the full set's values. The forest runs the add arms and removing RM only. Cells are
% change in QLIKE with DM in parentheses; negative = the first-named model has the lower loss. Every bar = 33,527 bars
on 2,579 days, 2014-01-02 .. 2024-04-30.

## 1. Does the full set beat HAR + the futures' own realized moments (RM)?

| Full set vs ... (every bar) | Ridge | Lasso | Elastic net | LightGBM | XGBoost | Forest |
|---|---|---|---|---|---|---|
| HAR (Table 1) | -5.4 (-5.6) | -6.2 (-4.7) | -6.0 (-7.0) | -14.0 (-10.3) | -13.8 (-9.9) | -11.1 (-9.9) |
| its own HAR + RM | -2.8 (-3.8) | -4.2 (-4.1) | -3.4 (-5.1) | -11.3 (-7.7) | -10.8 (-7.0) | -8.2 (-6.5) |
| OLS SHAR + HARQ + LHAR | -2.7 (-3.0) | -3.6 (-2.8) | -3.4 (-4.3) | -12.4 (-8.2) | -11.6 (-7.8) | -8.4 (-7.0) |

Every bar: yes, in all six classes. The classical OLS extensions gain little over HAR: SHAR -0.8 (-3.5), HARQ 0.0,
LHAR -2.7 (-6.2), all three -2.7 (-6.5).

16:00 bar: no. Full set vs its own HAR + RM at the close:

| 16:00 | Ridge | Lasso | Elastic net | LightGBM | XGBoost | Forest |
|---|---|---|---|---|---|---|
| 2014-2024 | +4.0 (+2.5) | +6.2 (+3.2) | +3.2 (+2.1) | +0.6 (+0.9) | +1.1 (+1.7) | +2.2 (+2.7) |
| 2020-2024 | +9.0 (+3.0) | +10.6 (+3.3) | +7.5 (+2.6) | -0.1 | +0.5 | -0.2 |

The trees' full set beats plain HAR at 16:00 by 1.4-3.4%, but no more than HAR + RM does.

## 2. Which groups matter (every bar)

Remove = full set without the group vs full set. Add = HAR plus the group vs HAR.

| Group | Remove: EN | Remove: LGBM | Remove: XGB | Remove: RF | Add: EN | Add: LGBM | Add: XGB | Add: RF |
|---|---|---|---|---|---|---|---|---|
| Realized moments | +1.9 (4.4) | +1.5 (6.7) | +1.6 (6.4) | +1.5 (7.9) | -2.7 (-4.2) | -3.0 (-6.3) | -3.4 (-7.5) | -3.2 (-6.8) |
| Liquidity | +0.1 | +0.2 | +0.2 | -- | -0.6 | -2.9 (-6.8) | -3.5 (-7.7) | -2.4 (-5.8) |
| EW market | -0.1 | -0.0 | -0.1 | -- | -1.1 (-2.8) | -2.8 (-6.5) | -3.1 (-7.1) | -2.3 (-5.4) |
| VW market | -0.1 | +0.1 | +0.1 | -- | -1.1 (-2.0) | -3.1 (-7.1) | -3.3 (-7.6) | -2.5 (-5.7) |
| Sentiment | +0.2 | +0.2 (2.2) | +0.2 (1.8) | -- | +0.0 | -1.1 (-4.0) | -1.7 (-5.8) | -0.9 (-3.6) |
| Implied volatility | -0.0 | +0.2 (1.7) | +0.1 | -- | +0.1 | -1.3 (-3.9) | -1.6 (-4.8) | -0.2 |
| Volatility demand | -0.1 | -0.0 | +0.1 | -- | +0.4 (2.9) | -1.2 (-5.1) | -1.2 (-7.0) | -0.6 (-3.4) |
| FOMC (preliminary) | +2.3 (5.7) | +10.1 (7.0) | +9.3 (6.2) | -- | -2.4 (-5.9) | -9.0 (-7.0) | -8.8 (-6.8) | -7.5 (-6.6) |

- Trees: two indispensable groups, the FOMC calendar and the futures' realized moments. Liquidity and the
  constituent-market groups help alone (-2.3 to -3.5%) but are redundant given the rest.
- Linear: the same two groups (ridge / lasso: removing FOMC +2.3 / +3.6%, RM +1.7 / +1.3%); volatility demand hurts.
- At 16:00 only RM matters (trees +2.1 to +2.6% when removed); FOMC is about 0 there (its effect falls in 14:00-15:30).

## 3. Alternative data on the bars where it is live (subsamples)

| Series | Live from | Share of evaluation bars live |
|---|---|---|
| VIX | 2018-04-19 | 57% |
| VIX3M | 2021-07-06 | 25% |
| VVIX | 2024-02-12 | about 0% |
| StockTwits attention | 2021-02-11 | 30% |
| Other two StockTwits series | 2022-04-19 | 18% |
| Volatility demand | 2022-11-16 (feed stops 2023-08-31) | 7% |

| Window (days) | Add to HAR: trees (LGBM / XGB / RF) | Remove from full: trees | Add to HAR: linear |
|---|---|---|---|
| StockTwits all live (508) | +0.1 / -1.0 / -0.0 | about 0 | -1.0 to -1.3 (n.s.) |
| VIX live (1,506) | -0.9 / -1.2 (DM -2.5) / +0.1 | about 0 | -0.9 to 0.0 |
| VIX + VIX3M live (706) | -1.1 / -1.4 (DM -2.2) / -0.0 | about 0 | +1.1 to +1.4 (n.s.) |
| Volatility demand (196) | -0.4 / -0.6 / -1.4 (DM -2.7) | about 0 | +2.8 to +4.3 (DM +2.2 to +2.5) |

On their live windows the alternative-data groups add little: VIX about 1% for the trees, StockTwits nothing
measurable, volatility demand hurts the linear models (removing it gains 2-3% every bar, 14-17% at 16:00). The trees'
full-sample "add" gains for these groups come mostly from the availability shares, not the series' values.

Caveats: FOMC preliminary; linear penalties frozen at the full set's values (ridge 1000, lasso 1e-4, enet 1e-3,
r = 0.01 on the HAR block); the forest has no removal arms other than RM; SHAR/HARQ/LHAR use new series built from the
same E-mini data (HARQ = sqrt(RV x RQ^1/2) at k = 1, in the design's volatility units).
