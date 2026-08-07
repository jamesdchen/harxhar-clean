# Rolling-window hyperparameter tuning — meeting brief (2026-07-09)

## Question
Does re-tuning hyperparameters **causally** — every selection made only from the
trailing training window (forward fit → embargo → validation split), never from
the test period — beat the untouched machine default? Purpose: eliminate the
look-ahead bias embedded in any fixed config chosen after multiple full-sample
tests.

## Design (both experiments audited: 5-section notebook audit passed, human-signed diffs)
- **Grid**: 8 exog buckets × 100 chunks (full 2005–2025 series, emit space
  `[24000, 242934)` bars), each task replaying a 24,000-bar training-window halo;
  map-reduce over two clusters (linear → Hoffman2/SGE, trees → CARC/SLURM),
  canary-gated before fleet.
- **Linear tuned arm**: one causal selector over the pooled penalty grid —
  ridge (6 alphas, Sherman–Morrison rank-1), lasso (5 alphas) and elastic net
  (5 alphas) on the warm Garrigues `enet_online` homotopy — re-selected every
  250 bars, coefficients refit **every bar**. Baseline: sklearn's own Ridge
  default `alpha=1.0` (`DEFAULT_RIDGE_PARAMS`), per-bar refit, identical
  features and executor path.
- **Tree tuned arm**: (family ∈ {xgb, lgbm}) × depth {3,5,8} × lr {0.05,0.1},
  grid re-scored every 500 bars on the causal tail, model refit every bar.
  Baseline: the repo's fixed default LGBM (500 trees / depth 5 / lr 0.1).

## Linear results (QLIKE, stitched full series per bucket; `calculate_metrics`)

| bucket | chunks | tuned | default | delta |
|---|---|---|---|---|
| moments | 100/100 | 0.131459 | 0.130264 | +0.001194 |
| liquidity | 100/100 | 0.131566 | 0.130822 | +0.000744 |
| market_ew | 100/100 | 0.133233 | 0.132677 | +0.000556 |
| market_vw | 100/100 | 0.133015 | 0.132388 | +0.000627 |
| implied_vol | 100/100 | 0.132856 | 0.132091 | +0.000765 |
| sentiment* | 97/100 | 0.134395 | 0.133581 | +0.000814 |
| vol_demand* | 95/100 | 0.134480 | 0.133754 | +0.000726 |
| all_features | queued | — | — | — |

\* ex-pathology (see below). Raw stitches including the pathological chunks:
sentiment 1.900, vol_demand 5.003 (vs the same ~0.133 defaults).

**Read-out (preliminary, no significance tests yet):** the look-ahead-free
tuner trails the untouched default by +0.0006–0.0012 QLIKE **uniformly across
all seven landed buckets** — consistent with the earlier §7 finding that the
penalty/estimator is a non-lever; causal selection buys nothing and pays a
small variance tax. DM / MCS from `src/evaluation` run once the full panels
land.

## What does causal tuning actually choose? (selection shares, all tunings)

| bucket | ridge | lasso | enet |
|---|---|---|---|
| moments | 43% | 32% | 25% |
| liquidity | 41% | 35% | 24% |
| market_ew | 45% | 35% | 20% |
| market_vw | 45% | 35% | 20% |
| sentiment | 46% | 32% | 22% |
| implied_vol | 50% | 29% | 21% |
| vol_demand (partial) | 41% | 34% | 25% |

The selector never collapses to one estimator — all three stay live, ridge
plurality everywhere.

## Known pathology (localized, mechanistic, fix identified)
8 of 700 landed chunks exploded (chunk QLIKE 48–178), **all located at the
exog data-start boundary** (~bars 67k–96k ≈ 2010–2012, where stocktwits /
voldemand go live). Mechanism: under ffill+availability-indicators, columns
constant for years turn on mid-window; the Gram passes through
near-singularity, and the pure-**lasso** grid arm (`lam2=0`, no ridge floor)
is the one solver with no well-posedness guarantee there (the enet rows and
ridge carry a ridge term; the default arm was never affected). Fix: give the
lasso arm the same ridge floor (or drop `l1_ratio=1.0` from the grid) —
one-line grid change, re-audit, re-run the 8 chunks.

## Trees (CARC, in flight)
Fleet running (800 tasks, 24 h walltime headroom). Only the canary's first
chunk exists: tuned 0.176807 vs default 0.177566 on moments chunk 1 — sign
opposite to linear, but one chunk is noise; full panel after the meeting.
Canary's grid picks mixed xgb depths 3/8 and lgbm.

## Provenance
Audited sources `specs/rolling_linear_tune.py` / `specs/rolling_tree_tune.py`
(audit ids `rolling-linear-tune` / `rolling-tree-tune`, both `passed`,
sign-offs journaled). Runs: `rolling_linear_tune-80b268f1` (Hoffman2),
`rolling_tree_tune-60cdbb8d` (CARC). Known infra defects hit and logged:
driver watchdog-stamp `FileNotFoundError`, canary verifier `completed_unknown`
false negative, SGE wave submissions (env-var expansion + burst failures).
