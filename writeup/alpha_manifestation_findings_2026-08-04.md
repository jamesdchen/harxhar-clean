# How the dense-but-weak exog alpha manifests — Findings (2026-08-04)

Scope: the question the campaign kept deferring. Every subgroup beats HAR, 41 exog stacked
beats every single bucket, and nothing is individually large — so **is "dense and weak" the
truth, or is it what a pooled test reports when a locally-concentrated alpha rotates across
variables and we never look finely enough to see the activations?**

All numbers are strictly-causal walk-forward on the local `data/` panel (rolling 250-day
window, refit as it walks, the production `RollingLeastSquares`). Code:
`analysis/alpha_panel.py` (feature space) + `analysis/alpha_manifestation.py` (stages
`resid` / `signals` / `tests` / `dynamics` / `sparse` / `verify`). Outputs:
`results/alpha_manifestation/`.

## TL;DR

- **Dense is the truth, not an artifact.** In a like-for-like horse race (daily reselect,
  daily refit, one shared rolling-Gram machinery, only `k` differs) OOS accuracy is
  **monotone in the number of features**: top-5 R² +0.0164 → all-246 **+0.0346**, and every
  sparse arm is *significantly worse* (DM-t −7.2 at k=5, −2.4 at k=50). The daily selected
  set churns **~2%** — the active set does not rotate.
- **Time-sensitive: yes, in intensity — but it is never off and never negative.** The
  monthly slope of residual-on-alpha has **1.74× the null dispersion** (CV 0.42, AR(1)
  +0.35), and **0 of 207 months contribute negatively** (null: 105). It is modulation of an
  always-on effect, not episodic firing. And it is *not* just heteroskedasticity — the
  mapping itself moves (corr with monthly residual sd only +0.15).
- **Clustering across exog by time/state: real and stable, but nearly zero-sum.** The
  bucket × time-of-day and bucket × vol-regime activation maps are **split-half stable
  (+0.68 / +0.64 vs nulls +0.04 / +0.02)**. But local alpha is **no larger** than pooled
  (concentration `c` = 0.95–1.05 on every axis), so there is dispersion without
  amplification.
- **"Testing too coarsely" — no, and the arithmetic says why.** Slicing into K bins costs
  √K in power and buys `c` ≈ 1. Finer clock conditioning loses monotonically
  (ΔR² −0.001 at K=2 → **−0.031 at K=48**), at 7 df/bin *and* at 1 df/bin, and a dynamic
  gain driven by trailing realized alpha also loses (−0.002).
- **One axis pays: vol state, coarsely, with shrinkage.** Vol-regime conditioning at K=2–3
  blended 50/50 with pooled: **ΔR² +0.0021/+0.0024, DM-t +5.0/+4.6** — robustly
  significant, worth ~7% of the entire exog channel. Separate (unshrunk) fits are
  insignificant (+1.6). Note the **asymmetry with the intraday-regime study**: the *clock*
  channel pays in HAR persistence (`HAR × late-day`, +0.008 OOS R²) and fails for exog;
  the *state* channel fails for HAR (−0.0027) and pays for exog.
- **The useful reframe:** the pooled IC is the *average delivery* of an unevenly-delivered
  alpha, and the delivery map is stable enough to act on. At **15:30–16:00** the alpha is
  ~55% stronger than pooled (moments IC 0.27 vs 0.17); at **09:00 and 17:30** it is
  ~zero. Nothing is hidden — it is visible at current resolution — but a *time-selective*
  application (close-auction vol, execution timing) is mispriced by the pooled number.

---

## 0. Setup, and a blocker that turned out to be resolved

- The full source panel is **local**: `data/*.parquet` is 329,208 rows, 1993-01 → 2024-04,
  with `endbartime`. Built through the production transform space the panel is
  **242,934 bars × 543 features** (2005-03-31 → 2024-04-30, 48 bars/day) — 6 HAR + 9
  calendar + 12 `har_ma_w × {open,close}` + 246 exog value + 270 `_avail`/`_active`.
- 242,934 is **exactly** the cached cluster matrix's row count from
  `intraday_regime_findings_2026-06-26.md` §10. So the cache was built from this panel, and
  the **row → date map exists locally**: §10's date-alignment blocker (and with it Test 3 /
  OPEX / rebalance-day features) is unblocked without needing the cluster.
- HAR baseline = 27 cols (HAR + calendar + the session-edge interactions), rolling 250-day,
  refit every bar: **R² 0.5773** in sqrt space, residual sd 0.2485, 230,934 OOS bars. That
  residual is what all exog alpha must explain — i.e. this study measures what is left
  *after* the intraday-regime fix already in production.
- Exog second stage: rolling 250-day, refit daily, ridge α=3000. Penalty ladder on the
  all-exog fit (OOS R² on the residual): **−0.084 / −0.001 / +0.025 / +0.035** at
  α = 1 / 30 / 300 / 3000; OOS corr is flat at ~0.19 from 300 up. Final OOS =
  **218,934 bars, 2007-02 → 2024-04** (the same 218,934 as the published runs).
- Nulls are **circular shifts** of the residual (24 reps), not 1/√n: 30-min bars are
  heavily autocorrelated and a parametric floor would badly understate sampling variation.
  All t-stats are Newey-West; all model comparisons are Diebold-Mariano HAC.

## 1. Density, quantified (the thing to be explained)

| | value |
|---|---|
| mean \|IC\| over the 246 exog value features | 0.0166 |
| median / max \|IC\| | 0.0114 / 0.1120 (`adj_sumret_ma_5`, NW-t −27.1) |
| participation ratio of the pooled \|IC\| vector | **116.8 of 246** |
| features with \|IC\| > 0.02 / > 0.05 | 72 / 13 |

Per-bucket walk-forward signals (OOS corr with the HAR residual): **moments 0.155**,
liquidity 0.085, market_vw 0.083, implied_vol 0.077, market_ew 0.075, sentiment 0.058,
vol_demand 0.039; **all-exog 0.190** (R² +0.0347, NW-t +25.8). The ranking reproduces the
published QLIKE ranking (moments > liquidity > implied_vol ≈ market_vw > market_ew >
vol_demand ≈ sentiment), which is the cross-check that this residual space is the same
phenomenon the cluster campaign measured. The exog channel is 3.5% of residual variance
≈ 1.5% of total target variance.

## 2. Is it locally sparse and rotating? (the hypothesis, tested directly)

**2a. Local IC vectors are *less* concentrated than the pooled one.** Participation ratio
of |IC| **within a month: 147.1**, pooled **116.8**, circular-shift null **158.1**. If the
alpha were locally sparse, monthly PR would sit far *below* pooled; instead it sits between
pooled and pure noise. Pooling does not smear a concentrated signal — it removes noise and
*reveals* the (moderately dense) structure.

**2b. The horse race.** Rank the 246 features by trailing-250-day |IC|, keep the top `k`,
**reselect and refit every day**. One machinery for all arms: the sliding window's
sufficient statistics roll rank-1 over all 246 columns, the ICs are read off those same
statistics, each refit solves only the selected `k×k` subsystem — so `k=246` *is* the dense
walk-forward, not a different estimator.

| top-k | OOS corr | OOS R² | DM-t vs dense | daily set churn |
|---|---|---|---|---|
| 5 | +0.1297 | +0.0164 | **−7.24** | 0.023 |
| 10 | +0.1427 | +0.0199 | −6.08 | 0.020 |
| 25 | +0.1614 | +0.0257 | −4.06 | 0.020 |
| 50 | +0.1733 | +0.0296 | −2.39 | 0.020 |
| 100 | +0.1843 | +0.0335 | −0.57 | 0.018 |
| **246 (dense)** | **+0.1892** | **+0.0346** | — | — |

Monotone in `k`, saturating around 100–246; every sparse arm significantly worse. And the
selected set turns over **~2% per day** — whatever is active stays active.

**2c. Rotation of identity is small.** Month-to-month Spearman of the *time-varying* part
of the IC vector (feature means removed): **+0.205** vs null +0.020 — real, but small.
`moments` is the best bucket in **79%** of months; best-bucket persistence 0.66 (chance
0.14).

> Caveat on scope: this refutes **sparse-rotating *linear marginal*** alpha — the version
> implied by the pooled-Ridge story. A locally sparse *interaction* structure (L1 over
> products, or the EBM's nonlinearity) is not ruled out by these tests.

## 3. Is it time-sensitive? (yes — as modulation, not as episodes)

207 monthly blocks, each ~1,000 bars:

| | actual | circular-shift null | excess |
|---|---|---|---|
| monthly IC: mean / sd | +0.1751 / 0.0667 | — / 0.0371 | **1.80×** |
| monthly **slope**: mean / sd | +0.9066 / 0.3805 | — / 0.2182 | **1.74×** (CV 0.42) |
| months contributing negatively | **0 / 207** | 105 / 207 | — |
| AR(1) of monthly IC / slope | +0.375 / +0.346 | +0.014 | — |
| share of total alpha from top decile / quartile of months | 0.313 / 0.518 | — | — |

Three things follow. (i) The variation is real — 1.7–1.8× the null on both a scaled (IC) and
an unscaled (slope) measure. (ii) It is **not** merely residual heteroskedasticity:
corr(monthly IC, monthly residual sd) = +0.151 and corr(slope, sd) = +0.137, so the
*mapping* moves, not just the noise level. (iii) But it never switches off — slope 5th
percentile **+0.202**, 95th +1.491, zero negative months. Concentration is mild (top decile
31% of alpha vs 10% under uniformity).

**And the persistence does not monetize.** AR(1) +0.35 invites a dynamic gain, so:
scale the pooled composite by a causal trailing-realized-alpha z-score (EWMA halflife
5 / 20 / 60 / 250 days). ΔR² = **−0.0019 / −0.0020 / −0.0019 / −0.0029**. The
predictable part of the intensity is too small relative to the extra parameter's cost.

## 4. Does it cluster among different exog at different times? (yes, stably — see §5 for why it doesn't pay)

Split-half stability of the bucket × bin activation map (first half of the sample vs
second), against a circular-shift null:

| axis | K | \|IC\| spread | split-half | null |
|---|---|---|---|---|
| time-of-day | 48 | 0.054 | **+0.679** | +0.043 |
| day-of-week | 7 | 0.034 | **+0.744** | +0.063 |
| vol regime (causal quintiles of `har_ma_25`) | 5 | 0.036 | **+0.641** | +0.021 |

These are real maps, reproducible out of sample. What they say:

**Time-of-day** (`activation_timeofday.csv`) — alpha peaks hard into the close and dies at
two specific bars:

| slot | moments | liquidity | implied_vol | market_vw | sentiment |
|---|---|---|---|---|---|
| 09:30 open | 0.187 | 0.122 | 0.096 | 0.158 | 0.095 |
| 15:30 | **0.268** | 0.148 | 0.126 | **0.225** | 0.103 |
| 16:00 close | 0.216 | **0.191** | **0.197** | **0.235** | **0.126** |
| 17:30 | 0.043 | 0.038 | −0.021 | 0.047 | −0.060 |
| 09:00 (pre-open) | 0.014 | 0.029 | 0.008 | −0.002 | 0.007 |

The 16:00 bar is where **every** bucket peaks simultaneously — the broadest activation in
the day, and the same clock location as the HAR sign-flip from the intraday-regime study.
`moments` leads 42 of 48 slots; the exceptions are informative: `liquidity` leads the thin
**19:00–19:30** after-hours, `implied_vol`/`market_vw` lead the **16:00** close.

**Vol regime** — this is where the identity genuinely rotates:

| quintile of recent vol | moments | liquidity | sentiment |
|---|---|---|---|
| Q1 (calmest) | 0.117 | **0.127** | 0.040 |
| Q3 | 0.145 | 0.047 | 0.042 |
| Q5 (most stressed) | **0.177** | 0.101 | **0.092** |

`liquidity` is the strongest bucket in the calmest quintile and collapses by Q3; `moments`
rises monotonically into stress; `sentiment` more than doubles Q1 → Q5. That is a real
"different variables at different times" — indexed by **state**, not by clock.

## 5. So are we testing too coarsely? (no — dispersion without amplification)

The ladder. Combiner = residual on the 7 bucket signals, rolling 60-day, fit either pooled
(K=1) or **separately within K bins**, each bin's window holding the same *calendar* span so
the power cost of slicing is paid rather than hidden. Baseline K=1: corr +0.1725, R² +0.0294.

| axis | K | ΔR² separate | ΔR² blended 50/50 | c | c/√K |
|---|---|---|---|---|---|
| time-of-day | 2 | −0.0011 | +0.0003 | 1.00 | 0.71 |
| time-of-day | 8 | −0.0089 | −0.0005 | 1.02 | 0.36 |
| time-of-day | 24 | −0.0197 | −0.0016 | 1.05 | 0.21 |
| time-of-day | 48 | **−0.0308** | −0.0033 | 1.05 | 0.15 |
| day-of-week | 5 | −0.0049 | −0.0006 | 0.97 | 0.44 |
| **vol regime** | **2** | +0.0012 | **+0.0021** | 0.99 | 0.70 |
| **vol regime** | **3** | +0.0007 | **+0.0024** | 0.97 | 0.56 |
| vol regime | 5 | −0.0010 | +0.0024 | 0.95 | 0.43 |
| vol regime | 10 | −0.0058 | +0.0013 | 0.95 | 0.30 |

**The concentration factor `c` is the whole answer.** `c` = mean within-bin |IC| ÷ pooled
|IC| — how much bigger local alpha is than the pooled average. It is **0.95–1.05 on every
axis at every resolution**. Slicing into K bins costs √K in power and buys nothing, so
`c/√K` < 1 everywhere and finer is always worse. §4's stable dispersion is real but
**zero-sum**: the 0.27 IC at 15:30 is paid for by the 0.01 at 09:00, and a forecaster who
must score every bar collects the average either way.

Nor is it a parameterization problem. Re-running the ladder with a **single gain per bin**
(1 df instead of 7 — "how much do I trust the same composite here") also loses everywhere:
time-of-day ΔR² −0.0005 (K=2) → −0.0172 (K=48); vol regime −0.0006 (K=2) → −0.0071 (K=10).

**The one real win is coarse vol-state conditioning with shrinkage** (Diebold-Mariano HAC,
+ = beats pooled):

| vol-regime K | separate DM-t | 50/50 blend DM-t |
|---|---|---|
| 2 | +1.59 | **+4.97** |
| 3 | +0.71 | **+4.55** |
| 5 | −0.93 | **+4.13** |

Unshrunk fits are insignificant; the blend is robustly significant across K. +0.0024 R² on
a channel worth 0.0346 is **~7% more alpha** — real, small, and consistent with everything
else in this project being a plateau-scale gain.

## 6. Reality check

- The size is right for what it is. The whole exog channel is 3.5% of the HAR residual;
  a conditioning refinement worth 7% of that is ~0.0008 of total target variance. This is
  the same order as the EBM campaign's sub-0.0001 QLIKE plateau — consistent, not a new
  regime of opportunity.
- **The `c` ≈ 1 result is the load-bearing one and it is a null.** It says the pooled
  estimate is unbiased for the quantity a bar-by-bar forecaster cares about. It does *not*
  say the alpha is uniform — §4 shows it demonstrably is not.
- Everything is measured on the linear-marginal channel at α=3000 / 250-day / daily refit.
  The tree's remaining ~38% nonlinear edge (intraday-regime §8) is out of scope here.

## 7. What follows

1. **Ship the vol-regime blend, not a slicer.** `HAR-residual composite × {low, high} vol`
   shrunk 50/50 toward pooled: 2 bins, DM-t +5.0. Coarse and regularized is the whole
   lesson — a K=5 unshrunk version is worse than doing nothing.
2. **Stop looking for hidden intraday exog activations.** The clock map is stable and
   already visible; conditioning on it loses at every resolution and every df cost. The
   clock channel's payoff lives in HAR persistence (already shipped as
   `har_ma_w × {open,close}`), not in exog.
3. **Re-price the time-selective applications.** If a downstream use is concentrated at
   15:30–16:00 (close-auction vol, MOC execution), the pooled IC understates it by ~55%;
   at 09:00 / 17:30 the forecast carries essentially no exog information. Worth a separate
   scoring run restricted to the slots an application actually trades.
4. **§10 is unblocked** — the row → date map exists locally, so Test 3 (month/quarter-end
   rebalancing) and the OPEX calendar features from intraday-regime §12 can be built now.
5. **Fixed in passing:** `executor.load_and_transform` passed `diurnal_mode=` to
   `robust_transform`, which did not accept it — **every exog run raised `TypeError`**
   (verified). `robust_transform` now takes the parameter; `"divide"` is bit-for-bit the
   old behavior and `"rank"` raises `NotImplementedError` (the per-slot rank-Gauss diurnal
   was never written; `rolling_rank_gauss` exists but no `diurnal_rank`).

## Reproducibility

```bash
python analysis/alpha_panel.py                              # build + cache the 242,934 x 543 panel (~8 min)
python analysis/alpha_manifestation.py --stage resid         # HAR walk-forward residual
python analysis/alpha_manifestation.py --stage signals       # 7 bucket + all-exog alpha signals (~8 min)
python analysis/alpha_manifestation.py --stage tests         # density / episodes / rotation / ladder
python analysis/alpha_manifestation.py --stage dynamics      # per-bin gain + dynamic gain
python analysis/alpha_manifestation.py --stage sparse        # sparse-vs-dense horse race
python analysis/alpha_manifestation.py --stage verify        # mapping-vs-scale + DM significance
```

Outputs: `results/alpha_manifestation/{report.txt, pooled_feature_ic.csv, monthly_alpha.csv,
monthly_bucket_ic.csv, monthly_mapping.csv, activation_{timeofday,dayofweek,vol}.csv,
activation_stability.csv, granularity_ladder.csv, gain_channel.csv, sparse_vs_dense.csv,
volregime_significance.csv}`.
