# Close studies, 2026-10-03: why the 16:00-bar trees do not beat the linear models

Four studies of the 16:00-bar forecast (the 15:30-16:00 bar, issued at 15:30), each from one line of the
user's handwritten note. Every number below is copied from the study's own CSVs / SUMMARY.md. Common to all:
the cached design of `experiments/capture_design_close.py` (all_features unless stated), the master
table's research scorer (16:00-bar recalibration, QLIKE and the 15:30 sign(s) straddle on the 866 trade
days 2020-01-03 .. 2024-04-30), point estimates (the block bootstrap is off, commit b761b28), DM = Diebold-
Mariano on daily QLIKE, HAC t = Newey-West t of the daily mid-fill P&L difference. Tree arms ran locally
with LightGBM 4.7.0 / XGBoost 2.1.4 and are compared with a local control of the shipped configuration
(QLIKE 0.1007, Sharpe mid 1.99; the stored cluster run, LightGBM 4.6.0, has the same QLIKE and Sharpe 1.84).

| study | folder | question |
|---|---|---|
| shrinkage | `exog_penalty/` | leave the HAR + calendar backbone unpenalized in ridge / lasso / elastic net, penalize the exogenous columns |
| data size | `trees_datasize/` | is 2000 sessions too little for trees? learning curve, and both last-hour bars pooled |
| tuning | `trees_pretune/` | heavy pre-2020 pre-tune, then a light retune every 250 sessions |
| linear leaves | `trees_lineartree/` | LightGBM `linear_tree`: do step functions cost the trees? |
| bar count (2026-10-04) | `trees_morebars/` | train on the last N bars (N = 1 .. 13) ending 16:00, score the 16:00 forecast |

## Headline numbers (QLIKE / Sharpe mid)

| forecast | QLIKE | Sharpe mid | source |
|---|---|---|---|
| ridge, backbone unpenalized (grid widened) | 0.0940 | 1.52 | exog_penalty |
| ridge, backbone unpenalized | 0.0946 | 1.50 | exog_penalty |
| lasso, backbone unpenalized | 0.0957 | 1.52 | exog_penalty |
| lasso, expanding window (16:00 rows back to 2004) | 0.0968 | 1.42 | trees_datasize |
| lasso, HAR + calendar only (master table) | 0.0972 | 1.51 | master table |
| LightGBM, last 4 bars (8000 rows) | 0.0974 | 1.63 | trees_morebars |
| LightGBM, last 3 bars (6000 rows) | 0.0975 | 1.03 | trees_morebars |
| ridge, one penalty, last 13 bars (26000 rows) | 0.0975 | 1.77 | trees_morebars |
| LightGBM, expanding window | 0.0977 | 1.62 | trees_datasize |
| LightGBM, both last-hour bars (4000 rows) | 0.0982 | 1.95 | trees_datasize |
| LightGBM, pre-tune + light retune (switch on any gain) | 0.0984 | 2.00 | trees_pretune |
| LightGBM, `linear_tree`, linear_lambda 10 | 0.0994 | 1.43 | trees_lineartree |
| lasso, one penalty (master table) | 0.0998 | 1.45 | master table |
| ridge, one penalty (master table) | 0.1004 | 1.66 | master table |
| LightGBM, shipped configuration (local control) | 0.1007 | 1.99 | all tree studies |

## What each study found

- **Shrinkage.** One penalty for every column shrinks the HAR persistence (ridge: HAR coefficient sum 0.57
  of the HAR + calendar OLS) and lets the exogenous columns carry 36 % of the forecast variance. With the
  backbone unpenalized: ridge 0.1004 -> 0.0946 (DM -2.25, p 0.025), lasso 0.0998 -> 0.0957 (DM -1.80,
  p 0.072), elastic net 0.0994 -> 0.0982 (p 0.49); ridge and lasso then sit below HAR + calendar alone on
  QLIKE (not significant). No trade difference has HAC |t| above 1. Algorithms: the spec's own ridge and
  reclasso / reclasticnet homotopy, ported to C (`experiments/close_exogpen_kernel.c`).
- **Data size.** On the 16:00 rows, more sessions lower the QLIKE of trees and linear models alike; from
  2000 sessions on LightGBM ties the lasso (2000 to expanding: LightGBM minus lasso change +0.0009, DM
  0.40). Pooling both last-hour bars helps the trees (LightGBM 0.1007 -> 0.0982) and hurts the linear
  models, whose one set of coefficients must serve both bars (lasso 0.1049, ridge 0.1048).
- **Tuning.** 71 pre-tune trials (stopped early) lowered the 4-fold validation MSE 1.5 % below the
  shipped configuration; selected on 2 folds and scored on the other 2 it was 13 % worse, and two
  independent studies' picks differ by up to 47 % of an axis range. Frozen pre-tuned 0.1008 / 1.91 (DM
  +0.16); light retune switching on any gain 0.0984 / 2.00 (DM -1.91, p 0.056); with a significance
  margin it never switched.
- **Linear leaves.** Unstable at linear_lambda 0 (forecasts down to -176). With linear_lambda 1 / 10:
  0.0997 / 0.0994 vs 0.1007 (p 0.44 / 0.23), Sharpe lower (t -1.16 / -2.21). On HAR + calendar alone:
  0.1049 -> 0.1031 (DM -1.97, p 0.049), still behind the HAR + calendar lasso 0.0972 (DM +2.73).

- **Bar count.** LightGBM (2000 sessions, refit every 10) on the last N bars ending 16:00, scored on the
  16:00 forecast: N = 1 / 2 / 3 / 4 / 5 / 7 / 13 gives 0.1007 / 0.0982 / 0.0975 / 0.0974 / 0.0984 / 0.0996 /
  0.1019; no rung differs from the 1-bar or the 2-bar arm with |DM| >= 1.96 (4 vs 2 bars DM -0.57).
  Changing the seed (43 for 42) moves QLIKE by 0.0005-0.0007 at each rung, the size of the 2 vs 3-4 bar
  gap, and moves the 2-bar Sharpe from 1.95 to 1.49. With rows held at 4000, more bars means fewer sessions
  and worse QLIKE (13 bars over 311 sessions 0.1064, DM 2.98 vs 2 bars). XGBoost is lowest at 2 bars
  (0.0993). The one-penalty linear models pooled over 13 bars: ridge 0.0975, lasso 0.0987 (4 bars: 0.1041,
  0.1141).

## Reading across the studies

- The predictable part of the 16:00 variance is mostly HAR persistence. Linear models capture it best
  once the penalty leaves the backbone alone; no tree variant tried here reaches the backbone-unpenalized
  ridge / lasso on QLIKE.
- 2000 sessions is not what holds the trees back: more sessions help the linear models about as much.
  Pooling a few bars before 16:00 helps the trees a little (best 0.0974 at 4 bars), within the seed-to-seed
  spread of a single LightGBM run, and still above the backbone-unpenalized ridge / lasso.
- The Sharpe ratio moves by several tenths between forecasts whose QLIKE is nearly equal (e.g. the same
  LightGBM on two scalings of the design: 1.99 vs 1.56; ridge at 1000 sessions 1.92 vs expanding 0.70), so
  trade comparisons between close forecasts are weak evidence.

## Not run here (cluster packages written, not submitted; this container cannot reach Hoffman2)

- `cluster/README_close_pretune.md`: the full pre-tune (8 studies x 2000 trials, LightGBM and XGBoost) and
  the walk refit every 10 / every session; about 700 CPU-hours at local fit times (`TRIALS=` scales it).
- `cluster/submit_close_trees_datasize_h2.sh`: refits every session, random forest, the XGBoost curve
  (160 tasks).
