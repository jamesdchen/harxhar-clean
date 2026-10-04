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
| equal weights: ridge backbone unpenalized + LightGBM 4-bar pool | 0.0914 | 1.66 | trees_morebars/average.csv |
| equal weights: ridge backbone unpenalized + average of all LightGBM pools | 0.0916 | 1.49 | trees_morebars/average.csv |
| ridge, backbone unpenalized (grid widened) | 0.0940 | 1.52 | exog_penalty |
| ridge, backbone unpenalized | 0.0946 | 1.50 | exog_penalty |
| lasso, backbone unpenalized | 0.0957 | 1.52 | exog_penalty |
| lasso, expanding window (16:00 rows back to 2004) | 0.0968 | 1.42 | trees_datasize |
| lasso, HAR + calendar only (master table) | 0.0972 | 1.51 | master table |
| average of the LightGBM pools N = 1 .. 4 | 0.0965 | 1.56 | trees_morebars/average.csv |
| average of all seven LightGBM pools | 0.0968 | 1.55 | trees_morebars/average.csv |
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

- **Averaging the pooled forecasts** (`experiments/close_trees_morebars_average.py`, `trees_morebars/average.csv`;
  equal weights on the adjusted scale, then the same scorer). Average of the LightGBM pools N = 1 .. 4:
  0.0965 (DM -2.37 vs the 1-bar control, p 0.018; -0.90 vs the best single pool, 4 bars); all seven pools
  0.0968. Averaging two seeds of one pool does not do this (4 bars: 0.0974 -> 0.0976), so the gain comes
  from combining different pools. Equal weights of the backbone-unpenalized ridge and the average of all
  pools: 0.0916 (DM -1.63 vs the ridge alone, p 0.10); ridge + 4-bar pool 0.0914 (DM -1.68, p 0.09);
  Sharpe 1.49 / 1.66 vs the ridge's 1.50. The subset averages were chosen after seeing the ladder; the
  all-pools average and the equal-weight ridge + trees combination are the ones that could be fixed in
  advance.

- **Full sweep and tests (2026-10-04)** (`experiments/close_trees_kfull.py`, `experiments/close_kfull_tests.py`;
  `trees_kfull/`, `kfull_tests/SUMMARY.md`). LightGBM on the last k bars, k = 1 .. 13, with the column
  `bar_end_minute` for k > 1, seeds 42 / 43 / 44 averaged for each k. QLIKE k = 1 .. 13: 0.0996 / 0.0982 /
  0.0969 / 0.0980 / 0.0975 / 0.0982 / 0.0992 / 0.0994 / 0.1011 / 0.1004 / 0.1005 / 0.1009 / 0.1015; average of
  all pools 0.0975; ridge (HAR + calendar unpenalized) 0.0946.
  - Against the ridge: every DM negative (-0.65 .. -1.61, none significant at lags 0 .. 21 or with fixed-b
    critical values); SPA p 0.73, Reality Check p 0.91, Holm-adjusted p 1.00 for every k; the 90 % model
    confidence set keeps all 15 models. SPA / Reality Check / MCS use the test script's own stationary
    bootstrap (the repo's block-bootstrap helper is off, commit b761b28).
  - Encompassing: the pool chosen on 2019 rows (k* = 1) adds nothing to the ridge (lambda 0.04, t 0.33);
    the average of all pools does (lambda 0.75, t 5.93 against 0 and -1.93 against 1; QLIKE moment t
    -2.18), and the ridge adds to it (reverse moment t -4.36). Real-time combination (lambda from past
    trade days): QLIKE 0.0821 vs the ridge's 0.0842 on days 251 .. 866, DM 1.67 (one-sided p 0.048);
    equal weights on all 866 days 0.0916, DM 1.56 (p 0.060).
  - Seed sd 0.0001 .. 0.0012 for each k; one step along k has |DM| above 1.96 (k = 8 -> 9, -2.50). No
    fluctuation-test rejection. MZ slopes all below 1 (ridge 0.69, trees 0.57 .. 0.76). The trees split
    on `bar_end_minute` in 46 % (k = 2) to 100 % of refits but with a gain share of at most 0.08 % (median
    gain rank 202 .. 333 of ~380 columns; 40 at k = 13). Within-day correlation of the target 0.63 .. 0.79:
    13 bars (26000 rows) carry an effective sample of about 3046 sessions.

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
