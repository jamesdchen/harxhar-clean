# Audit of the pooled linear switch (2026-10-22)

**Question (James):** the switch of Section 7's pooled linear rows to
`results/pooled_linear_rerun_2026-10-20_13_2012_w1000` reversed the story (the full predictor set now lowers the
pooled linear Sharpe ratio). Is that a mistake?

**Answer: no code error was found in the new rows.** Four independent audits (scoring and trading, design and
leakage, tuning, attribution) each reproduce the published numbers from scratch. The only genuine error found is a
look-ahead in the **old** rows, and it was already fixed and documented in the 2026-10-15 data audit. The reversal
comes from the forecasts themselves. It appears at one design step (the rolling window from 2,000 to 1,000 days), is
small, is not significant at any step, and depends partly on how the Sharpe ratio is annualized.

**Nothing was fixed, rerun or changed in the paper because of this audit.** The 1,500-day window run you asked for
is running as a sensitivity run (section 6).

Scripts and outputs (read-only, outside the repo):
`C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad\audit_{A,B,C,D}\`.

## What is being compared

| | Old rows (`poolpt_*`, earlier Section 7) | New rows (`poolpt13_*`) |
|---|---|---|
| Source | `results/pool_linear_pretune_2026-10-06`, arm f (`experiments/stack_pool_pretune_yhat.py:54`), built 10-06, **before the 10-15 data fixes** | `results/pooled_linear_rerun_2026-10-20_13_2012_w1000`, built at eb109e3 (`src/` = HEAD before today's FOMC-flag change) |
| Sample | all 48 half-hours | 13 regular-hours bars (10:00–16:00) |
| Window | 96,000 bars (2,000 days) | 13,000 rows (1,000 days) |
| Penalties | pre-tuned on folds before 2020, then **re-tuned past a DM margin** every 250 solves (11–35 switches per arm) | frozen on 4 folds before 2012-01-03 |
| HAR-block penalty factor 0.01 on | a 34-column backbone (HAR, HAR × open/close, calendar, expiry) | 12 columns (HAR, DOW, hour) |
| First evaluation day | 2020-01-03 | 2012-01-03 |

## 1. Scoring and Sharpe code path (audit A): no bug

| Item | Old | New | Same? | Evidence | Verdict |
|---|---|---|---|---|---|
| Old `plain` through the new trade code | master q, ret_mid, ret_crossed | — | yes, bit for bit | q differs on 0/866 days; returns max abs diff 0; printed SR reproduced (0.7438/1.5626, 0.7398/1.4591, 0.7490/1.4364) | not a bug |
| New `plain` through the old trade rule (`master_table_close.trade_days`) | — | economics-path q and returns | yes | 0/866 days differ | not a bug |
| `make_rehaul_results.frame()` | reproduces master | equals the economics `poolpt13` frames | yes | | not a bug |
| Ties, NaN, duplicate days, dtype, time zone | none | none | yes | | not a bug |
| Truth on the 16:00 row | master `rv_unclipped` | `rv_actual` = core_stats, bar ending 16:00 | yes | 866/866 exact (linear and all 6 trees); the 15:30 and 16:30 bars match on 0/866 | not a bug |
| What the 16:00 forecast uses | weight on the 15:30 bar (+0.38 vs +0.02 on 16:00) | +0.46 vs +0.01; trees +0.41 vs +0.02 | yes | regression of the forecast on the day's bar RVs | no off-by-one |
| Back-transform | yhat² × baseline | pred_adj² × B (asserted to 1e-12) | yes | √ scale, no Duan correction in either | design identical |
| Diurnal scale B at 16:00 | baseline | B (= trees' B) | nearly | B_new/B_old median 1.000, mean 1.006; the old B was depressed on 94 trade days by an early-close artefact (fixed); swapping in the old B moves long days by +7 to +9 | not the cause |
| Forecast level | mean log(plain/rv) +0.035 to +0.063; long 24–27% | −0.025 to −0.055; long 18–19% | **no** | the forecast itself −0.075 to −0.12; B +0.006 | **the source of the change; a model difference, not a bug** |
| Alignment with the trees, warmup | — | same dates from 2012-01-03, same B, rv, true_adj; same 252-day warmup slice | yes | | not a bug |

## 2. Design matrix, scaling, leakage (audit B): no bug in the new rows; a known leak in the old

| Item | Old | New | Same? | Evidence | Verdict |
|---|---|---|---|---|---|
| Columns | 640 (6 HAR + 282 values + 282 availability + 42 activity averages + 16 calendar/expiry + 12 HAR × open/close); HAR benchmark 34 | 576 (6 + 282 + 282 + 6 calendar); HAR benchmark 12 | no | all 576 new names are in the old 640 | design difference (E6/E9/E15 flags) |
| HAR ladder | base 5 | base 5 | yes | `pool_linear_pretune.py:131` | (base 2 was only a parked 10-19 run) |
| Exogenous series | 47, FOMC included | the same 47 | yes | | same |
| FOMC / VIX data | FOMC flags winsorised (release averages constant); FOMC feed dead for the last 123 trade days; VIX family dead for the last 54 | fixed (10-15 data audit items 2, 3b, 3c) | no | | **old data defects** |
| New design = trees' design | — | names, order, y (4e-16), B (bit-identical) | yes | the trees' X is CARC's build: one column (`adj_voldemand_spx_open_only_ma_5`) differs materially (max entry diff 4.4e-3 vs s.d. 0.51), already documented; the linear rerun on CARC's design changed no printed digit | same design |
| Degenerate columns | same structure | 69 constant, 138 exact duplicates, DOW collinear with the intercept; about 369 distinct | — | handled by the identifiability mask (`specs/causal_tune_linear.py:349`) | not a bug |
| Feature scaling | rolling median/IQR on [t−W, t) | same, W = 13,000 | mechanism yes | `scaling.py:252-274` | not a bug |
| Scale-guard statistics (fill median, s.d. floor, mode-inflation test, point-mass guard, asinh routing, sign rule) | **full sample** | rows before t only | no | `executor.py`, `target.py` | **old look-ahead (data audit items 1, 4, 5a–5e)** |
| Target, B | √(RV/B), causal winsorisation; B = mean of 20 previous same-slot bars | same; early-close bars excluded | nearly | | no future data in either |
| Training window / embargo | fit [t−W, t), forecast t | same | yes | | not a bug |
| **Future-perturbation test** | randomising every raw input after T* moves 60 of 244 columns on rows before T* | moves **0 of 204** columns (y, B: 0 rows) | — | `perturb_results.csv`; turning on only the 7 causal fixes brings the old build to 0 of 256 | **new: causal. Old: genuine look-ahead, exactly the known items**; the data audit measured its effect as small (no significant QLIKE change; 23–40 flipped positions) |

## 3. Pre-sample folds, penalties, solver (audit C): no bug

The new run was rebuilt from its own commit (eb78186) and refit on pre-2012 rows only. This reproduces **all 96 fold
MSEs of `pretune_mse.csv` to 6 decimals**, and each model's first walk forecast exactly (ridge 0.70724, lasso
0.70541, enet 0.73805, OLS 0.783571).

| Item | Old | New | Same? | Evidence | Verdict |
|---|---|---|---|---|---|
| Folds | 4 × 12,000 bars (250 × 48), 1,200-bar embargo | 4 × 3,250 rows (250 × 13), 325-row embargo; last validation row 2011-12-30 16:00 | scheme same | fold loop unchanged b29774a → eb78186 | correct |
| Criterion | mean validation MSE, adjusted scale, all bars | same (13 bars) | yes | | not a bug |
| Grids | ridge 1e-2..1e3; lasso/enet 1e-6..1e-2; l1_ratio 0.5 | identical | yes | | not a bug |
| Full-set picks | ridge 1e3 (top), lasso 1e-3, enet 1e-3; then re-tuned | ridge 1e3 (top), lasso 1e-3, enet 1e-2 (top) | ridge, lasso | | design difference |
| Grid too narrow? | ridge at top | extended pre-2012: ridge 1e4 only 0.5% better, 1e5 worse; enet 1e-2 is the interior optimum (1e-1 zeros every exogenous coefficient) | — | `grid_ext.out` | not material |
| Fold-2 blow-up (MSE 25–38 at small penalties) | — | late-start series: `adj_vix3m_ma_*` (VIX3M from 2009-08-12) nearly constant in fit, then robust-scaled values of −3 to −15.5 in Sep–Oct 2010; without those 6 columns fold 2 is 0.075 | — | `fold2_repro.py`; the design's own feature-health check flags it FAIL | **data/scaling artefact, not a code bug, not the flash crash; no pick depends on fold 2** |
| One fold decides? | — | ridge 1e3 in every fold; lasso 1e-3 in every leave-one-out; only the elastic net is tipped by fold 0 (2008) | — | | not a bug |
| HAR-benchmark flat criterion | flat | flat (gap ≤ 3e-4%); forecasts equal the OLS (corr 1.0000) | yes | | harmless |
| Degeneracy (seed window) | — | nonzero: ridge 338 (df 115), lasso 85 (df 86), enet 34 (df 34.5); corr with HAR OLS 0.97–0.98, mean ratio ≈ 1; lower adjusted-scale MSE than the HAR OLS in every year 2012–2024 | — | `frozen_coefs.out` | real models |
| Solver, α convention, penalty factor | own ridge closed form; lasso/enet homotopy + KKT repair; ridge α in sum units | AST-identical code | yes | | not a bug; with a 7.4× shorter window, ridge 1e3 is 7.4× stronger per row (a consequence of the window, not an error) |

## 4. Where the reversal happens (audit D: attribution across the stored reruns)

Sharpe ratios: sign strategy, 614 post-warmup days, per-year annualized, written HAR / full / ΔSR [Ledoit–Wolf p].
"Pooled Δ" is the full − HAR difference of the plain 614-day Sharpe ratio (mean / sd × √252), the quantity the
Ledoit–Wolf test actually tests.

| Run | What changed vs the previous row | Elastic net | Lasso | Ridge | Pooled Δ (enet/lasso/ridge) |
|---|---|---|---|---|---|
| R0 old `poolpt` (arm f) | 48 intervals, 2,000 days, pre-fix data, margin re-tune | 0.74 / 1.56 / +0.82 [.037] | 0.74 / 1.46 / +0.72 [.071] | 0.75 / 1.44 / +0.69 [.062] | +1.12 / +0.80 / +0.94 |
| R1 `10-19_48` | 10-15 data fixes, column flags, FOMC calendar fix, frozen penalties | 0.90 / 1.42 / +0.52 [.114] | 1.01 / 1.48 / +0.47 [.175] | 1.01 / 1.10 / +0.09 [.885] | +0.79 / +0.64 / −0.10 |
| R2 `10-19_13` | 48 intervals → 13 bars | 1.07 / 1.63 / +0.55 [.049] | 1.07 / 1.23 / +0.16 [.335] | 1.07 / 1.49 / +0.42 [.145] | +0.92 / +0.42 / +0.65 |
| R3 `10-20_13_w1000` | **window 2,000 → 1,000 days** | 1.33 / 1.13 / −0.20 [.828] | 1.27 / 1.02 / −0.25 [.995] | 1.30 / 1.09 / −0.22 [.963] | +0.08 / 0.00 / −0.02 |
| **R5 new** `2012_w1000` | first day 2020 → 2012 (only the pre-tune folds move) | 1.30 / 1.13 / −0.17 [.754] | 1.28 / 1.02 / −0.26 [.996] | 1.30 / 1.09 / −0.22 [.963] | +0.11 / 0.00 / −0.02 |
| R4 `2012_w2000` (side branch) | first day 2012 at 2,000 days | 1.07 / 1.30 / +0.23 [.272] | 1.07 / 1.23 / +0.16 [.335] | 1.07 / 1.49 / +0.42 [.145] | +0.49 / +0.42 / +0.65 |

Closing-interval QLIKE (1,082 days), HAR / full: R0 enet .1142/.1078, lasso .1135/.1101, ridge .1143/.1086;
R5 .1205/.1194, .1203/.1229, .1205/.1236.

- **The 2012 start is not the cause:** at the same window, R5 and R3 (and R4 and R2) trade identically on all 614
  days wherever the penalty is the same.
- **The step that flips the sign is the window.** At 1,000 days the HAR rows gain (their forecasts drop about 3%,
  which turns 32–34 losing long days into short days), and the full-set rows lose (they drop winning long days and add
  losing ones).
- **No step's own change is significant** (Ledoit–Wolf p 0.11–0.96). On the old design the penalty rule alone (arms
  a, b, f) moves ΔSR anywhere from +0.32 to +0.99.
- **The result is fragile:** scaling the new elastic net forecasts by c = 0.95 / 1.00 / 1.10 gives full − HAR of
  +0.31 / −0.18 / +0.49; the new HAR Sharpe of 1.30 is a local spike (0.95 at c = 0.95, 1.15 at c = 1.05). The old
  full − HAR stays at +0.38 to +0.87 for c from 0.90 to 1.05.

## 5. Findings that are not code bugs but need a decision

1. **The printed ΔSR and its p are different statistics.** The printed ΔSR is the per-year annualized Sharpe ratio
   (each year's ratio scaled by √N_y, then N_y-weighted, so a year weighs N_y^1.5). The Ledoit–Wolf p tests the plain
   614-day Sharpe ratio. For the new rows they point in opposite directions: pooled Δ is +0.11 / 0.00 / −0.02 and the
   mean daily returns are nearly equal (enet full 0.098 vs HAR 0.090). The full set's edge is in the partial years
   2021 (64 days) and 2024 (83 days); HAR's in 2022. For the old rows both agreed in sign. **Decision:** report the
   pooled 614-day ΔSR beside the per-year one, or test the per-year statistic itself.
2. **Late-start series are scaled on nearly constant windows.** VIX3M (from 2009), VVIX (scaled values jump to 2–3 in
   2013Q2) and `adj_voldemand_all_open_only_ma_3125` (109.7 in 2012Q1) get extreme robust-scaled values while their
   windows are short of history. This explains the fold-2 blow-up and does not change any penalty pick; the frozen
   penalties damp it in 2012–13 (ridge-full / OLS forecast ratio 0.69–1.32). The 2020–2024 trade days use windows that
   start in 2016 or later, after these series have history, so the Section 7 numbers are probably unaffected (to be
   confirmed). A pre-registered guard (for example, mask a series until it has a full window of observations) would
   be a specification change and needs reruns of the linear models and the trees. **Your call.**
3. **The old rows were not a clean benchmark.** They come from before the 10-15 fixes: full-sample guard statistics
   (look-ahead), constant FOMC release averages, a dead FOMC feed for the last 123 trade days and a dead VIX feed for
   the last 54, a margin re-tune, and a depressed B on 94 trade days. The earlier story was produced with those.
4. **Window choice.** 1,000 days is the trees' window (one design for the 12 pooled forecasts). At 2,000 days the
   linear full-set rows keep a positive ΔSR (R2, R4). Choosing the window because of its Sharpe ratio would be a
   specification search; the 1,500-day run is a sensitivity run, to be reported with the other two.
5. **Cosmetic:** 31 walk rows (2011-12-28 14:00 to 12-30 16:00) precede the evaluation span; they are outside the
   option days and the accuracy span.

## 6. 1,500-day window (running)

At your request: the same configuration as R5, with only `--window-days 1500`, built with the code that produced R5
(fomc_release included), gated on reproducing R5's HAR-benchmark forecasts. Output
`results/pooled_linear_rerun_2026-10-22_13_2012_w1500/`. Results will be added here as a three-window table
(1,000 / 1,500 / 2,000 days).

## Bottom line

- **Bugs fixed:** none needed in the new pipeline. The old pipeline's look-ahead was already fixed on 10-15.
- **What the evidence says:** on corrected data and the trees' design, the full predictor set does not raise the
  pooled linear sign-strategy Sharpe ratio at the closing bar; the differences are small, fragile and not significant
  (they were not significant before either, at the 5% level, for two of the three models).
- **No rerun or paper change was made because of this audit.** Decisions for you: the Sharpe statistic (finding 1),
  the late-start scaling guard (2), and the window (4, after the 1,500-day run).
