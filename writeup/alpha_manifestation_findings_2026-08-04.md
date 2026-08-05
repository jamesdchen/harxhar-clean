# How the dense-but-weak exog alpha manifests — Findings (2026-08-04)

Scope: the question the campaign kept deferring. Every subgroup beats HAR, 41 exog stacked
beats every single bucket, and nothing is individually large — so **is "dense and weak" the
truth, or is it what a pooled test reports when a locally-concentrated alpha rotates across
variables and we never look finely enough to see the activations?**

All numbers are strictly-causal walk-forward on the local `data/` panel (rolling 250-day
window, refit as it walks, the production `RollingLeastSquares`). Code:
`analysis/alpha_panel.py` (feature space) + `analysis/alpha_manifestation.py` (stages
`resid` / `signals` / `tests` / `dynamics` / `sparse` / `verify`) + `analysis/nl_sparsity.py`
(§7, the interaction channel). Outputs: `results/alpha_manifestation/`.

## TL;DR

- **The linear channel is dense; the INTERACTION channel is sparse — and neither rotates.**
  ~100 pairwise products out of 8,911 candidates, **frozen once in 2006 and never reselected**,
  add **ΔR² +0.0067 → +35%, DM-t +2.71** over the linear base — with no clipping, `voldemand`
  dropped, and a separate (heavy) penalty on the product block. Reselecting monthly never
  beats freezing on any healthy specification. Sparse in identity, static in time.
  ⚠️ **Read §7.1 before quoting a number.** A first version of this result claimed +43% at
  DM-t +6.9; that figure came from a ±4 design clip and is inflated. The interaction gain is
  real but **strongly dependent on how the product block is regularized** (it is *negative*
  under some scalings), and the static-beats-dynamic margin reaches conventional significance
  only under the clip.
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

## 7. The interaction channel: sparse in identity, static in time

§2 refuted sparse-rotating alpha in the **linear marginal** channel only. The intraday-regime
study put ~38% of the tuned EBM's edge in genuinely nonlinear structure, so the same question
is re-asked over pairwise products (`analysis/nl_sparsity.py`). Candidate space: products
(including squares) of 133 base columns — 6 HAR + 41 exog × windows {1, 25, 625} + the clock
gates — giving **8,911 candidates**. All pairwise ICs at every refit come from three window
matmuls (`G = ZᵀZ`, `A = Zᵀdiag(e)Z`, `Q = (Z∘Z)ᵀ(Z∘Z)`), so only the selected `k` columns are
ever materialized. Monthly reselect + refit; 218 months.

> **A methods note, and a retraction of its first version.** The first run of this section
> scored the linear base arm at **R² −0.635**, and its pair ICs were outlier-driven, so the
> whole first result set was void. I diagnosed that as my own doing — re-standardizing an
> already-scaled panel by a training-window sd, i.e. the repo's Part-3 degenerate-divisor
> disease — and fixed it by clipping the standardized design to ±4. **That diagnosis was
> wrong**, and `--stage robust` is what caught it: a ``center``-only variant that performs *no
> division whatsoever* still collapses to R² −0.293, so the divisor was never the cause.
>
> The actual cause is in the panel: **`voldemand`, and only `voldemand`.** All twelve columns
> with |z| > 200 are `voldemand` MAs, and 3,065 of the 3,674 affected rows fall in **2023** —
> the 0DTE volume explosion is a genuine level break, and rolling-window standardization of a
> series that shifts by orders of magnitude correctly reports hundreds of sigmas. Panel-wide
> p99.9 of |z| is **7.6**; drop the 12 `voldemand` columns and max |z| falls from 2011 to 79.
> `_build_scale_guards` is *not* at fault — verified directly: its zero-inflation branch fires
> at all six windows for these columns (exact-zero fraction 0.31–0.73 vs the 0.20 threshold),
> so the active-value scale is being used as designed. The guard floors the *divisor*; it
> cannot bound a *numerator* that genuinely moved.
>
> And the clip was doing a second job I had not accounted for: standing in for **block-wise
> regularization**. Products of centered features have variance ~ the product of the parents',
> so one global ridge `alpha` cannot restrain the product block and fit the base block at the
> same time; the clip compressed the scale difference instead. §7.1 replaces it with two
> penalties, which is the legitimate version.

**Result 1 — interactions add, but the size depends on regularization (see §7.1).** Under the
±4 clip: base 133 cols R² +0.0215 → best interaction arm **+0.0305**, and against the *full*
516-column linear exog set, 100 frozen products add ΔR² +0.0081 → **+43%, DM-t +6.91**. Those
are the clipped numbers and they are **inflated**; §7.1 puts the robust figure at **+35%,
DM-t +2.71**. Either way it is not information the linear model already had.

**Result 2 — it IS sparse, pooled.** The ladder peaks at **k = 25–100 of 8,911** and degrades
by k=400. Contrast the linear channel, which improved monotonically all the way to 246/246.

**Result 3 — but it is NOT locally sparse. Static selection beats dynamic at every k.**
The decisive arm pair differs *only* in selection — coefficients are refit identically:

| k | dynamic R² | static R² | dyn−vs−stat DM-t | dynamic set churn / month |
|---|---|---|---|---|
| 10 | +0.0241 | +0.0257 | −1.65 | 0.308 |
| 25 | +0.0261 | +0.0266 | −0.42 | 0.291 |
| 50 | +0.0252 | +0.0277 | −1.93 | 0.274 |
| **100** | +0.0249 | **+0.0305** | **−3.21** | 0.260 |
| 200 | +0.0237 | +0.0276 | −2.10 | 0.256 |
| 400 | +0.0159 | +0.0234 | −3.16 | 0.242 |

Negative DM-t = dynamic is worse. Choosing 100 products once from the 2005–2006 window and
never touching them for 18 years **beats** reselecting every month, significantly. The
dynamic arm's 24–31% monthly churn is noise that costs real accuracy to chase.

Corroborated by the within-month diagnostic (top-1000 products by pooled |IC|): within-month
participation ratio **626.6** vs circular-shift null **622.3** — *at the noise floor*, i.e. no
detectable local concentration beyond chance. Month-to-month Spearman of the time-varying IC
part is **+0.089** (null −0.004): a real but small time-varying component, as in the linear
channel. (The pooled PR of 920 is not comparable here — preselecting the top 1000 by pooled
|IC| inflates it by construction. The within-month-vs-null comparison is the clean one.)

**Result 4 — what gets selected is interpretable, and it corrects an earlier note.** The most
persistently chosen products (of 218 months, top-100 arm):

| months | product | reading |
|---|---|---|
| 146 | `adj_sumret_ma_1²` | squared *net* bar return vs summed squared sub-returns — a variance-ratio / jump term, kin to `sumautocov` |
| 144 | `adj_sumret_ma_1 × adj_sumvolume_ma_1` | **signed return × volume — order-flow imbalance in all but name** |
| 134 | `har_ma_1 × adj_sumret_ma_1` | leverage effect *conditional on the vol level* |
| 132 | `adj_sumret_ma_1 × adj_sumabsret_ma_1` | signed × magnitude — a semivariance-flavoured asymmetry term |
| 121 | `adj_sumret_ma_1 × adj_sumret4_ma_1` | signed return × tail activity |
| 111 | `adj_sumret3_ma_1 × adj_sumvolume_ma_1` | skew × volume |

Nine of the top ten involve `sumret` — **signed return is the hub**, and its value is almost
entirely interactive rather than marginal. This **partially corrects** the 2026-06-23 note
("`r⁻ ≈ (sumret − sumabsret)/2`, so Ridge can already form it, nothing to build"): that is
true of the *linear combination*, but every term above is a **product**, which Ridge
structurally cannot form. That gap is the 43%.

Also note `sumret × sumvolume` arriving unprompted as the top-ranked genuine interaction —
that is the OFI probe listed as an open item in intraday-regime §9.3, and the selector found
it without being told to look.

### 7.1 Robustness: does any of this survive without the clip?

Two runs, `--stage robust` and `--stage grouppen`. First, the same three arms under four
scalings (all 133 base columns):

| scaling | base R² | stat100 − base | dyn vs stat DM-t |
|---|---|---|---|
| sd + clip ±4 | +0.0215 | +0.0090 (DM-t **+8.15**) | **−3.21** |
| sd, no clip | −0.635 | −0.302 (−1.02) | +1.28 |
| robust floored-IQR, no clip | −0.514 | −0.021 (−0.46) | −0.50 |
| center only, no division | −0.293 | +0.014 (+1.38) | −1.15 |

Only the clipped arm yields a working estimator, so the other three can neither confirm nor
refute anything. Dropping the 12 `voldemand` columns (max |z| 2011 → 79) makes two of them
healthy — and the picture changes:

| scaling, no `voldemand` | base R² | stat100 − base | dyn vs stat DM-t |
|---|---|---|---|
| center only, no division | **+0.0194** | +0.0047 (DM-t **+1.28**, n.s.) | −1.30 |
| robust floored-IQR, no clip | +0.0167 | **−0.049 (harmful)** | +0.69 |
| sd, no clip | −0.184 | — | — |
| sd + clip ±4 | +0.0214 | +0.0093 (+8.32) | −3.51 |

So the **+43% headline was an artifact of the clip**, and interactions can even hurt depending
on how the product block is scaled. The cause is a single global penalty across two blocks with
incommensurate scales, so the fix is two penalties. No clip, `voldemand` dropped, products on a
*floored* window sd, product block penalized separately:

| product penalty | dyn100 | stat100 | stat DM-t vs base | dyn vs stat |
|---|---|---|---|---|
| base only | — | +0.0194 | — | — |
| 300 | −0.0135 | −0.0429 | −1.12 | +0.54 |
| 3,000 | +0.0104 | +0.0138 | −0.72 | −0.41 |
| **30,000** | +0.0202 | **+0.0261** | **+2.71** | −1.74 |
| 300,000 | +0.0218 | +0.0238 | +2.77 | −0.93 |

**Conclusions that survive.** (i) The interaction channel is real: **ΔR² +0.0067 → +35%,
DM-t +2.71**, with no clipping — but only under *heavy* shrinkage of the product block, which
must be a separate hyper-parameter. (ii) Static selection is never beaten: dyn-vs-stat is
negative at every healthy specification (−3.51, −1.74, −1.30, −0.93, −0.41) and positive only
where the fit is over-fitting anyway (+0.54). Significant under the clip, directional
elsewhere — so "the active set does not rotate" is **supported and never contradicted, but not
established at conventional significance without the clip.** (iii) Pooled sparsity (peak at
k = 25–100 of 8,911) is unaffected by any of this. (iv) `voldemand` needs a **feature-level**
fix for its 2023 level break — a global clip is the wrong instrument, and its extremes were
distorting every arm of this section.

**Caveats.** (i) All §7 numbers are at monthly refit with the ±4 clip, so the base is +0.0188
rather than the §1 daily-refit +0.0347; both arms share the machinery, so the *relative* +43%
is clean, but the absolute daily-refit gain is untested. (ii) Products are drawn from 3 of 6
windows, so the space is a subset. (iii) Selection is by marginal |IC| on products; an L1 fit
over the full 8,911 may find a better set.

## 8. The `voldemand` fix (and a correction to the rule-drift claim)

§7.1 left `voldemand` as an open item: its extremes were distorting every arm of the
interaction study, and a global clip was the wrong instrument. Investigated and fixed
(`analysis/voldemand_fix.py`, fix in `src/features/transforms/target.py`).

### 8.1 The defect is three pipeline steps failing in sequence, not a data problem

`voldemand_spx_open_only` grows ~11x in scale across the sample (IQR 51k in 2011 -> 572k in
2023; median 0 -> 472k; p01 −181k -> **−8.15M**) — the 0DTE build-out. Tracing one column
stage by stage, max |value| by era:

| stage | 2011–16 | 2021–23 |
|---|---|---|
| raw (ffilled) | 1.05e7 | 1.34e7 |
| after `diurnal_adjust` (per-slot std divide) | 66.7 | **487.3** |
| after `apply_semantic_transform` (rule 5 = identity) | 66.7 | 487.3 |
| after `rolling_winsorize` | **6.2** | **487.3** |

1. **`diurnal_adjust`'s divisor is pinned at its floor on 73.8% of rows** (its 0.1%, 1% and
   *median* quantiles are all identically 14911.6). For an ffilled, 61%-zero series the per-slot
   rolling std is degenerate most of the time, so `DIURNAL_STD_FLOOR_FRAC` turns the
   "adjustment" into division by one global constant — which cannot track a drifting scale.
2. **Rule 5 returns identity**, so there is no variance stabilizer at all (flagged in the Part-3
   notes as "the deeper mis-design; asinh would fit").
3. **`rolling_winsorize` cannot bound it**: a trailing 5–95% band over 240 bars tracks a
   *trending* series, clipping only 6.9% of rows.

`_build_scale_guards` is **not** at fault — verified directly: its zero-inflation branch fires
at all six MA windows for these columns (exact-zero fraction 0.31–0.73 vs the 0.20 threshold).
It floors the *divisor*; it cannot bound a *numerator* that genuinely moved.

### 8.2 Correction: "rule 5 is the only group with elevated scale drift" was wrong

That claim came from comparing group *medians* of raw IQR drift (rule 5: 1.33 vs rule 6: 1.17)
across groups of n = 1–16, with no test. Tested properly, rule-5 pre-transform drift vs all
other rules is **Mann-Whitney p = 0.766** — no difference whatsoever.

The defensible claim is structural, and shows up *after* the transform: rule 5 is the only rule
that **passes drift through instead of compressing it**. Median attenuation (post-drift /
pre-drift) by rule: log **0.51**, 4th-root 0.83, sqrt 0.90, **identity 0.92**, cbrt n/a (its raw
drift is 0.11). Post-transform drift, rule 5 vs others: **p = 0.044** — marginal, and the honest
strength of the claim.

Also worth recording: **IQR drift was the wrong diagnostic for `voldemand` in the first place.**
Its post-transform IQR drift is unremarkable; its problem is the *tail*, which the IQR by
construction cannot see. The fix rests on the two direct measurements instead — the pinned-
divisor fraction (0.738) and max |z| (487) — not on the drift table.

### 8.3 The fix, and the gate that keeps it surgical

:func:`~src.features.transforms.target.asinh_stabilize` — `asinh(x / s_t)`, with `s_t` an EWMA
of |x| over **active** (observed, non-zero) rows, halflife 250 days, shifted one bar, floored at
1% of its own expanding median. `asinh` is the signed analogue of `log`: linear near zero (so
`asinh(0) = 0` leaves the zero mass alone, no clipping, no sign loss) and logarithmic in the
tails. `s_t` is causal, adaptive (tracks the 11x growth), and cannot go degenerate.

Applied through a **measured gate** in `robust_transform`, not a hardcoded column list: if a
signed feature's per-slot std divisor is pinned at its floor on ≥ `DIURNAL_PINNED_MAX` (0.5) of
rows *and* rules 1–4 give it no stabilizer of their own, the degenerate divide is replaced by
the stabilizer. Pinned fractions: `sumret3` 0.93, **voldemand ×4 ≈ 0.74**, sentiment 0.28,
`sumret3_vwstock` 0.27, `sumret3_ewstock` 0.20, `spread_vwstock` 0.09, `sumautocov` 0.02,
**`sumret` 0.001**. So the gate cleanly selects voldemand and leaves the strongest single
feature (`sumret`, |IC| 0.108) untouched; `has_name_stabilizer` keeps `sumret3` on its working
`cbrt` rather than swapping a transform on the best bucket with no evidence.

Blast radius, measured: **4 of 41 raw columns change; 37 are bit-for-bit identical.**

### 8.4 Evidence

Variant selection (each rebuilds only the voldemand block and splices it into the cached panel —
exact, because `rolling_robust_scale` and `_build_scale_guards` are per-column; the `status_quo`
splice reproduces the cached panel as an assertion):

| variant | panel max \|z\| | vol_demand R² | all-exog R² | unclipped base R² |
|---|---|---|---|---|
| status_quo | 2010.9 | **−0.00026** | +0.03471 | **−0.270** |
| asinh then diurnal | 109.6 | +0.00222 | +0.03721 | +0.0184 |
| **asinh only (shipped)** | **15.8** | **+0.00263** | +0.03766 | +0.0190 |
| asinh, 60d halflife | 16.3 | +0.00251 | +0.03770 | +0.0189 |
| rank-Gauss | 278.5 | +0.00209 | +0.03726 | +0.0150 |

Two negative results worth keeping: **asinh *then* diurnal does not work** (max 109.6) — keeping
the degenerate divide keeps the problem, which is itself confirmation of the diagnosis. And
**rank-Gauss is not immune** (278.5) despite bounding the adjusted series to ±3.9: a 61%-zero
series maps to a huge point mass whose rolling MA has near-zero dispersion in some windows, so
the degenerate divisor reappears at the *scaler*. Division-free at one stage does not mean
division-free downstream. The 250d/60d halflife tie (+0.00263 vs +0.00251) says the knob is not
tuned.

Full control — panel rebuilt with the production fix, all 8 buckets re-scored at the identical
spec (the HAR residual is exog-free, so the target is unchanged and the 4 columns are the only
difference):

| | corr | R² | DM-t |
|---|---|---|---|
| **panel max \|z\|** | | **2010.9 → 82.1** | |
| non-voldemand cells differing | | **0 (bitwise)** | |
| moments / liquidity / implied_vol / market_vw / market_ew / sentiment | unchanged | unchanged | 0.00 |
| **vol_demand** | +0.0390 → **+0.0523** | **−0.00026 → +0.00264** | +1.63 |
| **all exog** | +0.1897 → **+0.1955** | +0.03471 → **+0.03766** | +1.55 |

**What is unambiguous:** max |z| falls 24x; the panel stops poisoning any fit that does not
refit every bar (unclipped base R² **−0.270 → +0.0190**, i.e. the §7.1 pathology is gone with
the columns *in*); and `vol_demand` — the one bucket that failed to beat HAR in this residual
space — turns **negative R² positive**. **What is not:** the composite gain is +8.5% relative
but DM-t **+1.55**, directional and *not* significant. Do not quote it as a win.

### 8.5 A methodological catch that applies to every DM-t in this document

The control's first run reported DM-t **+5.00** for `moments` — a bucket whose columns are
*bitwise identical* between the two panels. Cause, measured: `walk_forward` is bit-deterministic
within a process (difference exactly 0) but differs by **9.4e-16** across processes through
non-deterministic BLAS reductions — 3.8e-15 of a residual sd — and DM, being a ratio of a mean
difference to its standard error, is 0/0 in that limit and explodes. `dm_test` now returns 0.0
when the two forecast series agree to 1e-10 of the target's scale.

The rule this implies, applied retroactively: **a DM-t must never be read without its ΔR².**
Every DM-t quoted in §§2–7 accompanies a ΔR² of 0.005–0.03 — 12+ orders of magnitude above this
noise floor — so those conclusions stand unchanged. But the statistic alone is not a sufficient
summary, and this document pairs them everywhere for that reason.

### 8.6 Still open

`stocktwits_sentiment` (post-transform drift **1.85**, attenuation 9.1x, max |adj| 70) and
`spread_vwstock` (1.37, 11.3x, max |adj| **389**) are the worst *post-transform* drifters, both
rule-5, and the pinned-divisor gate does **not** catch them (0.28 and 0.09) — their divisors are
healthy; their tails are not. After the fix the panel's largest |z| columns are no longer
voldemand at all but `adj_sumpret2_vwstock_ma_5` (82.1), `adj_numobs_ma_25` (78.8) and
`adj_sumbipow_ewstock_ma_1` (77.3) — sqrt-transformed positives, a different mechanism again.
A second pass wanting a tail diagnostic rather than a divisor diagnostic is the natural
follow-up; it was out of scope here and is not assumed to matter.

## 9. What follows

1. **Ship the ~100 frozen interaction products** (§7), with a **separate, heavy penalty on the
   product block** — that hyper-parameter is not optional, it is the difference between +35%
   and actively harmful. Robust size +35% (DM-t +2.71), and *cheaper* than what exists — a fixed feature list, no
   reselection machinery, no retuning. Start from the signed-return hub
   (`sumret × {volume, sumabsret, HAR level, sumret}`); build the explicit OFI
   `(buy−sell)/(buy+sell)` term alongside it, since the selector reached for its proxy
   unprompted. Reselect on a multi-year cadence at most — monthly is measurably harmful.
2. **Ship the vol-regime blend, not a slicer.** `HAR-residual composite × {low, high} vol`
   shrunk 50/50 toward pooled: 2 bins, DM-t +5.0. Coarse and regularized is the whole
   lesson — a K=5 unshrunk version is worse than doing nothing.
3. **Stop looking for hidden intraday exog activations.** The clock map is stable and
   already visible; conditioning on it loses at every resolution and every df cost. The
   clock channel's payoff lives in HAR persistence (already shipped as
   `har_ma_w × {open,close}`), not in exog.
4. **Re-price the time-selective applications.** If a downstream use is concentrated at
   15:30–16:00 (close-auction vol, MOC execution), the pooled IC understates it by ~55%;
   at 09:00 / 17:30 the forecast carries essentially no exog information. Worth a separate
   scoring run restricted to the slots an application actually trades.
5. **§10 is unblocked** — the row → date map exists locally, so Test 3 (month/quarter-end
   rebalancing) and the OPEX calendar features from intraday-regime §12 can be built now.
6. **`voldemand` is fixed** (§8) — `asinh_stabilize` behind a measured degenerate-divisor gate.
   Panel max |z| 2011 → 82, `vol_demand` R² negative → positive, 37 of 41 columns bit-identical.
   The **§7 interaction study should be re-run on the fixed panel**: it was conducted with
   voldemand dropped precisely because of this, and the ±4 clip may no longer be needed at all.
7. **Fixed in passing:** `executor.load_and_transform` passed `diurnal_mode=` to
   `robust_transform`, which did not accept it — **every exog run raised `TypeError`**
   (verified). `robust_transform` now takes the parameter; `"divide"` is bit-for-bit the
   old behavior and `"rank"` raises `NotImplementedError` (the per-slot rank-Gauss diurnal
   was never written; `rolling_rank_gauss` exists but no `diurnal_rank`).

## 10. Selection audit — which claims are exposed to multiple testing

Added after the fact, and it changes how two results above should be read.

By the end of this study **~100+ specifications had been scored against the same 218,934-bar OOS
residual**: a 4-point ridge ladder, 8 bucket signals, 13 granularity points x 3 axes, 13
gain-channel arms, a 6-point sparse ladder, 13 interaction arms, 4 scalings x 3 arms x 2, 4
product penalties, 5 voldemand variants. No unused data remains in this panel. And 219k
autocorrelated bars are only ~4,500 independent days, so the real noise floor on a ΔR² is well
above the naive one.

That does **not** contaminate everything equally. What matters is whether a claim is a *measured
quantity* or the *max of a search*:

| claim | form | exposure |
|---|---|---|
| `c` = 0.95–1.05 (concentration, §5) | measured, no selection | **safe** |
| split-half stability +0.68 / +0.64 vs nulls +0.04 / +0.02 (§4) | descriptive + explicit null | **safe** |
| 0 of 207 months negative; slope dispersion 1.74x null (§3) | descriptive + null | **safe** |
| dense monotone in k, ΔR² +0.018, DM-t −7.2 (§2) | a whole ordered curve, not a max | **safe** |
| 2% daily / 24–31% monthly selection churn (§2, §7) | measured | **safe** |
| vol-regime blend pays, ΔR² +0.0024, DM-t +5.0 (§5) | max over ~13 ladder points x 2 | **§11.1: survives**, SPA p 0.0000, RW adj. p 0.0000–0.0015 |
| interaction gain +35%, DM-t +2.71 (§7.1) | max over 4 product penalties | **§11.1: survives narrowly**, SPA p 0.024, RW adj. p 0.024 |
| voldemand composite gain +8.5%, DM-t +1.55 (§8.4) | max over 5 variants | already labelled n.s.; **treat as zero** |

The two bolded rows were reported as clean and should not have been. The right instrument for
them is a bootstrap over the **maximum** statistic across arms — Hansen's SPA or a step-down
Romano–Wolf, blocked for autocorrelation — not a per-arm DM-t. **Both have now been run (§11.1) and
both survive** — the conditioning claim comfortably, the interaction claim narrowly (p = 0.024) —
with the adjusted p-values being lower bounds, since each family is a subset of what was searched.
The structural conclusions in §§2–5 do not depend on either.

**Why §8's voldemand fix is not in this table.** It was justified by a **violated invariant with
a direct measurement** — divisor pinned on 74% of rows, 487-sigma tails, 51x era asymmetry, a
bucket sitting at negative R² — not by winning a bake-off. Its composite DM-t (+1.55) was
reported as insignificant at the time. That is the distinction worth generalising: **fix a
feature because an invariant is violated, not because an arm won.** An invariant check costs no
inferential budget, which is exactly why the diagnostic proposed below is the right next build
and another transform bake-off is not.

**Protocol for any future specification comparison here:** pre-register the candidate set and the
decision rule; hold out an era (search on ≤ 2020, one scored decision on 2021–2024 — this panel
no longer has clean holdout for that, a real cost of how this study ran); correct for the search
with Romano–Wolf or SPA; and gate on the invariant checks first, since a candidate that fails
scale-equivariance is out regardless of score. See `analysis/universal_transform.py` for the
transform-specific version of this argument, including why there is no single best *universal*
semantic transform (the 41 features do not share a distribution family, and the existing
sqrt/log/cbrt rules are Bartlett variance-stabilizing transforms that follow from each family's
mechanism — rule 5's defect was being a fall-through default, not being name-based).

## 11. Multiplicity correction, and the invariant diagnostic

### 11.1 Both flagged claims survive correction

`analysis/multiplicity.py`. Each arm family recomputed keeping per-arm loss differentials, then
Hansen SPA (H0: *no* arm beats the benchmark) and Romano–Wolf step-down adjusted p-values, both on
a circular block bootstrap with one-month blocks drawn jointly across arms (2,000 reps).

**Claim 1 — vol-state conditioning, family = the 26 granularity-ladder arms.** SPA
**p = 0.0000**.

| arm | ΔR² | DM-t | RW adj. p |
|---|---|---|---|
| **vol K=2 blend** | +0.00208 | 5.01 | **0.0000** |
| **vol K=3 blend** | +0.00242 | 4.56 | **0.0005** |
| **vol K=5 blend** | +0.00239 | 4.14 | **0.0015** |
| vol K=10 blend | +0.00132 | 1.89 | 0.166 |
| vol K=2 separate | +0.00127 | 1.65 | 0.240 |
| time-of-day K=2 blend | +0.00028 | 0.74 | 0.623 |

Survives comfortably. Note the corrected results reproduce the uncorrected story exactly: the
blends pay, the unshrunk fits do not, and no clock arm comes close.

**Claim 2 — the interaction gain, family = the 8 group-penalty arms.** SPA **p = 0.0240**.

| arm | ΔR² | DM-t | RW adj. p |
|---|---|---|---|
| **static, penalty 3e5** | +0.00443 | 2.77 | **0.0240** |
| **static, penalty 3e4** | +0.00673 | 2.71 | **0.0240** |
| dynamic, penalty 3e5 | +0.00243 | 2.14 | 0.028 |
| dynamic, penalty 3e4 | +0.00080 | 0.34 | 0.743 |
| static / dynamic, penalty 3e3 and 3e2 | negative | — | ≥0.97 |

Survives, with much less room — p = 0.024 against 0.05. The two significant arms are both
**static**, which is the §7 conclusion arriving independently through a corrected test.

**Two limits that remain.** The families here are subsets of what was actually searched (§5's 13
gain-channel arms and §7's 12 clipped-ladder arms are not re-run), and adding arms can only widen
the null of the maximum — so **every adjusted p above is a lower bound**. And no correction
recovers a held-out sample: these price in the search *within* a family, not the ~100
specifications the study scored overall. The §10 protocol is still the only clean route.

### 11.2 The invariant diagnostic, back-tested against its own incidents

`src/diagnostics.py` — five per-column checks (pinned-divisor fraction, tail profile, era
asymmetry, modal share, scale-guard bind rate) plus one run-level check (forecast scale over
target sd). They never look at forecast accuracy, so they cost no inferential budget.

`analysis/diagnostics_backtest.py` reconstructs seven incidents and asks whether the report flags
each. **6 of 7.** The exercise earned its keep by finding four defects in my own work:

| what the back-test found | fix |
|---|---|
| pinned-divisor fired on post-fix `voldemand`, whose divide is no longer in the pipeline | report 0 when `robust_transform` routes past the divide; add `divide_in_use` |
| modal share fired on a zero mass that the hurdle encoding already models | exempt hurdle-encoded columns |
| era asymmetry fired on a healthy column (1.9 → 3.2 across eras = "1.7x") | gate on absolute level (`max\|z\| >= 10`); a ratio of small maxima is meaningless |
| **`build_vd_block("status_quo")` had silently stopped reproducing the pre-fix state** — the asinh gate is now the production default, so I1 and I2 had collapsed onto one arm | pass `signed_stabilizer=False` explicitly |

That last one is the sharpest argument for the whole exercise: a committed script had quietly
stopped reproducing its own recorded numbers, and only a back-test that asserted a *known* outcome
could notice.

The one remaining miss is post-fix `voldemand` returning **warn** rather than clean — it is still
the panel's most drift-exposed family (max |z| 15.8, era asymmetry > 2.5), so a conservative flag
is defensible and the threshold was left alone rather than tuned until the test passed. I4
(imputing in log space) correctly does *not* fire: reconstructing it shows the Part-3 scale guards
already prevent the blow-up, and I5 — the same setup with guards off — does fire. Out of scope by
construction: the `diurnal_mode` `TypeError` (a code-path defect needing a smoke test), the
`dm_test` noise floor (a statistic bug, fixed at source), and the ±4 clip mis-diagnosis (no
invariant would have caught it; it took an adversarial variant).

**And it immediately found something new.** On the production panel the report FAILs **12+ columns
besides voldemand**, with max |z| of 68–82: `adj_sumpret2_vwstock_ma_5` (82.1),
`adj_numobs_ma_25` (78.8, and a **p99.9 of 46.4** against the panel's 7.6),
`adj_sumbipow_ewstock_ma_1` (77.3), `adj_stocktwits_sentiment_ma_1` (70.3, whose max *is* its
p99.99 — a single spike). These are sqrt-transformed positives and a bounded index, i.e. mechanisms
different from voldemand's. **The voldemand family was not special; it was just the worst.** Full
report in `results/alpha_manifestation/feature_health_production.csv`.

## 12. A heat equation on the interaction graph — tested, and it does not work

§7 established that the interaction alpha lives on a graph (nodes = features, edges = pairwise
products, weights = |IC| with the residual: a positive function over the interaction graph). Given a
graph, the natural prior is the heat equation on it — ``exp(-tL)`` smoothing, whose quadratic form
``lambda_2 * b' L b`` penalises *differences* between coefficients of adjacent terms rather than
their magnitudes. That is the textbook instrument for dense-but-weak effects with a known adjacency
(network-constrained regularisation), so it deserved a test. `analysis/heat_graph.py`.

**This was the first fresh hypothesis of the study, so it got the §10 protocol it deserved**: grid
pre-registered (``lambda_2/lambda_1`` in {0.1, 1, 10, 100} x two graph variants, nothing else),
graph frozen on the 2006 window, the choice made on rows **through 2020 only**, and the winner
scored **once** on a 2021-2024 holdout (176,574 / 42,360 rows). Benchmark = the §11.1 winner, the
same feature set with a plain diagonal product penalty, so the question is strictly *does graph
smoothing beat plain shrinkage of the same terms*.

**The graph's spectrum made an ex-ante prediction, before anything was scored:**

| graph | non-zero eigenvalues | participation ratio | prediction |
|---|---|---|---|
| feature (node Laplacian) | 48 | **19.7 of 133** — concentrated | smoothing degenerates toward a hub average; should not help |
| line (products adjacent when they share a parent) | 98 | **80.2 of 100** — rich | real multi-scale structure; the variant with a chance |

**The prediction was directionally right, and the answer is still no:**

| arm | search ΔR² | holdout ΔR² |
|---|---|---|
| node, ratio 0.1 → 100 | −0.00046 → **−0.01111** | −0.00098 → −0.01253 |
| edge, ratio 0.1 | +0.00014 | +0.00017 |
| **edge, ratio 1.0** (chosen on search) | +0.00047 (DM-t +0.62) | **+0.00022 (DM-t +0.15)** |
| edge, ratio 10 → 100 | −0.00134 → −0.00716 | −0.00203 → −0.00451 |

The node variant is monotonically harmful exactly as its concentrated spectrum implied. The edge
variant is the only one that ever helps and its help is nil.

**Why it fails — three reasons, and the first is conceptual.**

1. **Interaction is not similarity.** The Laplacian prior says "adjacent coefficients should be
   *close*". In genomics that is justified because pathway-adjacent genes have co-directed effects.
   Here adjacency means "these two features *interact*", which implies nothing about their
   coefficients being equal. The method's assumption and the graph's meaning do not match.
2. **It is redundant with shrinkage we already have.** §11.1 showed the product block needs a *heavy*
   diagonal penalty (3e4-3e5). Once every product coefficient is shrunk that hard they are all small
   and mutually close, so a differences-penalty has nothing left to do.
3. **The line graph is too dense to be informative.** 98 non-zero eigenvalues over 100 nodes with
   ``lambda_max`` only 2.25: most products share a parent with most others, so "smooth over the line
   graph" collapses to "shrink all product coefficients toward a common mean" — ridge again.

**The protocol paid for itself on its first use.** The search-period gain (+0.00047) halved out of
sample (+0.00022). Reported without the holdout, this would have been a ~2x overstated "small
positive"; with it, it is correctly a zero. That is the selection-bias haircut §10 exists to expose,
observed directly.

**Also worth recording: a discrepancy with §7.** The graph frozen on the 2006 window is centred on
``is_overnight`` (degree 21), then ``har_ma_1`` (13) and ``adj_numobs_ma_1`` (12), with mild
concentration (top node = 10.5% of edge ends). §7's hub was ``sumret`` (9 of the top 10 products).
The two are different estimators — §7 weighted by *persistence across 218 monthly reselections* on a
clipped, voldemand-free design; this is a single frozen draw on the fixed panel. §7's frozen-selection
result was about *fitting*, not about the graph being well estimated from one window, and this
suggests a single-window graph is a poor estimate. A persistence-weighted graph is the obvious
variant if anyone revisits this.

**What remains untested and is the natural next attempt**: the *similarity* graph — adjacency from
feature covariance rather than interaction strength — which is the adjacency the Laplacian prior
actually assumes. Cheap, and it is the version of "heat on the graph" whose assumption matches its
object.

### 12.1 The corrected test — the first one was a weak test, not evidence about nature

Four of stage ``fit``'s choices were bad, and reporting them as a finding about the graph was wrong:
the graph was estimated from **one** 2006 window (whose hub disagreed with §7's); the line graph was
made near-complete by construction (unweighted share-a-parent over 100 products from 50 nodes); the
λ grid *declined monotonically from its smallest value*, so it never bracketed the optimum; and the
literal heat equation — diffusing the **features**, ``Z <- Z (I + tL)^-1`` — was never tested at all,
only the coefficient penalty. `--stage fit2` fixes all four, adds the similarity graph, and adds a
weak baseline (plain ridge, no products) beside the strong one.

**The graph estimate was indeed the problem.** Averaged over 176 search-period windows the hubs are
``adj_sumret_ma_25`` (degree 27), ``har_ma_1`` (20), ``adj_sumabsret_ma_1`` (19) — §7's `sumret` hub,
reproduced. The single-window graph was simply high-variance. Worth noting alongside it: the chosen
edges sit in the top-100 in only **29%** of windows, so even "persistent" edges are only moderately
so. Spectra: interaction-node PR **14.0** of 133 (still concentrated), similarity PR **82.6** of 133,
weighted line graph PR **83.7** of 100 (the weighting did not decontaminate its density).

| arm | validation ΔR² | search ΔR² | holdout ΔR² |
|---|---|---|---|
| **heatcov_t0.3** (chosen on validation alone) | +0.00072 | +0.00028 | **+0.00039, DM-t +0.45** |
| nodeCov_r0.3 | +0.00054 | +0.00028 | +0.00035 |
| heatcov_t0.1 / nodeCov_r0.1 | +0.00041 / +0.00023 | +0.00023 / +0.00013 | +0.00028 / +0.00016 |
| every ``Int`` (interaction-adjacency) arm | negative | negative | negative |
| heatint_t3.0 (worst) | −0.01470 | −0.01163 | −0.01186 |
| plain ridge, no products | −0.02036 | −0.01051 | −0.00376 |

**Two things the corrected test establishes.** First, the conceptual diagnosis was right and is now
measured: **similarity adjacency helps, interaction adjacency hurts** — every ``Cov`` arm is positive
and every ``Int`` arm negative, with the sign consistent across three independent splits (validation,
full search, holdout). "Adjacent coefficients should be close" is a claim about *correlated*
features; it is false for *interacting* ones. Second, the magnitude is still nil: the licensed number
is holdout ΔR² **+0.00039, DM-t +0.45**.

So the honest state is not "the graph is useless" but "the graph's *similarity* structure points the
right way and delivers about four ten-thousandths of R², which this sample cannot distinguish from
zero." A better estimate of the graph moved the result from +0.00022 to +0.00039 — the right
direction, the same conclusion.

**Holdout accounting**: this is the **second** evaluation of the 2021-2024 block (the winner was
chosen on an internal 2017-2020 validation split so the holdout was touched once more, not
repeatedly). Two looks is not one look; discount accordingly.

## 13. Mining the intensity modulation with graph structure — stopped at its own gate

The second-order question, and the systematic version of the only thing that worked: **what predicts
*when* the dense alpha is strong?** Vol-state conditioning (§5, §11.1) is itself an intensity
modulator found by the crudest possible search, so the class is non-empty and unexplored — nobody had
looked at features, let alone products of features, as modulators. `analysis/intensity_graph.py`.

**Three graph roles, which §12's failure showed must not be conflated.** *Where to look* =
interaction adjacency (the hypothesis space). *How to share strength* = **similarity** adjacency,
per §12.1's measurement. *What to analyse* = the time-varying alpha field **as a graph signal**: the
alpha is not a graph but a function ``w_ij(t)`` on one, and with 8,911 edges against ~168 effective
monthly observations only a low-order **graph-Fourier projection** could have power.

**Pre-registered before the first run** (in the module docstring, so it cannot be revised into a
result): ~15-20% chance anything survives; magnitude if so, +0.001 to +0.003 R². Gate = split-half
stability of the modulator-IC map must exceed its circular-shift null. **Positive control** =
``har_ma_25``, the confirmed modulator (RW adjusted p 0.0005), must rank in the top quartile, because
a measurement that cannot find the known-good modulator cannot be trusted to find new ones.

**The control did its job and voided the first attempt.** At bar resolution the modulator ICs showed
a *7x excess over null* and every one of the top eight modulators was an ``_ma_625`` slow feature —
while ``har_ma_25`` ranked **98 of 133**. Diagnosis: the intensity series has a one-day
autocorrelation of **0.988**, so correlating it with other near-non-stationary series at bar
resolution measures *which features are also slow*, a textbook spurious regression. The 7x was an
artifact. (A secondary bug made the split-half nulls NaN, so that verdict was vacuous too.)

**Corrected to the honest resolution** — monthly blocks, 166 search observations, noise floor
1/sqrt(166) = 0.078; plus intensity *changes* as a second target, which a shared trend cannot inflate:

| target | excess over null (linear / products) | control rank | split-half stability (vs null) | gate |
|---|---|---|---|---|
| intensity **levels** | 1.14x / 1.04x | 36/133 (fail, threshold 33) | linear **−0.095** vs +0.177; products +0.013 vs −0.002 | **STOP** |
| intensity **changes** | 1.36x / 1.40x | 132/133 (fail) | linear +0.136 vs +0.010; products +0.065 vs +0.021 | **STOP** |

**So we stop, as pre-registered.** Three honest readings of that table. The 7x collapses to 1.04-1.40x
once resolution is honest. Split-half stability for levels is *below* its own null — the modulator map
does not replicate. And the control's marginal failure on levels (rank 36 of 133 = 27th percentile
against a 25th-percentile threshold) says the measurement is much improved over bar-level but still
imperfect; the threshold was **not** moved post hoc to rescue it. The changes-target control failure
(rank 132) is expected and is a flaw in the control rather than the measurement — vol *level* should
not predict intensity *changes* — which means the changes row has no valid control at all and its
modest +0.136 stability cannot be leaned on.

**One suggestive detail, recorded but not claimed.** The strongest modulators under both targets are
the **liquidity family at slow windows** — ``spread_vwstock/ewstock_ma_625``, ``sumvolume_ma_625``,
``numobs_ma_625``. That coheres with §4 (liquidity leads the calmest vol quintile), and the modulator
graph's hubs are the same family (``sumvolume_ma_625`` and ``numobs_ma_625``, degree 25 each). But it
does not replicate split-half, so it is a hypothesis for a future dataset, not a finding.

**What would license revisiting.** The binding constraint is the intensity estimator's signal share
(~67% at monthly aggregation, §3) against ~166 observations — not the graph machinery, which was
never reached. A materially better intensity estimator, a longer sample, or a cross-sectional panel
(many assets, so intensity is estimated across as well as through time) would each change the power
calculation. Absent one of those, this question is not answerable with this data.

### 13.1 A better intensity estimator — and the reframing it forced

§13 blamed its own failure on the intensity estimator, so that got fixed first. The fix worked, the
gate still failed, and the *reason* it failed turns out to be the most useful thing in this section.

**The incumbent estimator was worse than a constant.** Candidates and criteria were fixed in advance
(EWMA at 4 halflives vs an AR(1)-state heteroskedastic Kalman filter at 3; criteria = predictive
validity for next month's realised slope, and IC with it):

| estimator | predictive R² vs a constant | IC with realised slope |
|---|---|---|
| ewma_5d | **−0.436** | +0.367 |
| **ewma_21d** (what §13 used) | **−0.136** | +0.330 |
| ewma_63d / ewma_126d | −0.211 / −0.223 | +0.104 / −0.048 |
| kalman_hl0.5m | **+0.092** | +0.373 |
| kalman_hl1m | +0.069 | +0.385 |
| kalman_hl2m | +0.044 | **+0.388** |

Every EWMA has *negative* predictive R² — right direction (positive IC) but wrong magnitude, because
**an EWMA is exactly the Kalman filter for a random-walk state** and `--stage diffusion` had already
measured the slope to be mean-reverting (AR(1) ~0.52, signal variance *falling* with horizon). A
random-walk filter never shrinks toward the mean, so it over-reacts. Every AR(1)-state variant is
positive. That one misspecification explains both §13's failure **and** §3's dynamic-gain failure
(−0.002): both were driven by an intensity estimate worse than using its own average. (Rule ambiguity
recorded: my stated rule was "max signal share subject to beating the constant" and the code took max
IC among valid arms, which picks `kalman_hl2m`; `kalman_hl0.5m` has the best predictive R². The three
are within noise of each other.)

**With the better estimator the gate still fails, and the control fails *worse*:**

| target | excess over null (lin / prod) | control rank | split-half stability vs null | gate |
|---|---|---|---|---|
| levels | 1.21x / 1.10x | **57/133** (was 36) | +0.074 vs **+0.084** | STOP |
| changes | 1.52x / 1.36x | 66/133 | +0.085 vs +0.029 | STOP |

The pre-registration said: *if the control still fails, the problem was never the estimator*. It
still fails, so it wasn't.

**Why the control fails — and this is the finding.** The control assumed vol state should predict
*scalar* intensity, because vol-state conditioning is confirmed (RW adjusted p 0.0005). But look at
which vol-regime arm actually won: the **7-weight blend**, not the **1-df-per-bin gain**, which lost
at every K (§5's gain-channel row, −0.0006 at K=2). Vol state does **not** modulate the alpha's
overall magnitude. It re-allocates weight **across buckets** at roughly fixed total magnitude.

So the scalar-intensity framing is the wrong object, and that single correction explains the whole
pattern of results at once: ``c ~ 1.0`` (dispersion without amplification, §5); per-bin *gain* models
failing while per-bin *weight* models pay (§5); blends beating separate fits (§11.1); activation maps
being stable yet zero-sum (§4); and the dynamic gain failing (§3). None of those are separate facts —
they are all the same fact. **The dense-weak alpha's time variation is a rotation of composition, not
a modulation of magnitude.**

**Which reframes what to mine.** Not a scalar intensity against 8,911 candidates on 167 observations —
that is the underpowered question. The right object is the **composition vector**: the 7 bucket weights
as a function on the bucket graph, which is 7-dimensional rather than 8,911, is already known to be
split-half stable (+0.64 on the vol axis, §4), and is already known to pay when conditioned coarsely
and shrunk (§11.1). That is a well-posed question with ~24x fewer parameters than the one that just
failed, and it is where I would go next.

## 14. Is it a partition of unity? Testing the geometry — and a null result that walks back §13.1's wording

§13.1 characterised the state-dependence as "a rotation of composition, not a modulation of
magnitude", which suggests ``beta(s) = rho * u(s)`` with ``rho`` fixed and the unit direction rotating
— and, if the weights were also non-negative and summed to a constant, a literal **partition of
unity** (hence a mixture-of-experts softmax gate over buckets). That is a testable geometric claim, so
`analysis/composition.py` tests it: fit the 7 bucket weights within each causal vol-regime bin on the
**search period only**, then split the across-bin variation into a *radial* part (magnitude moving) and
a *tangential* part (direction rotating), against circular-shift nulls. Descriptive only — no forecast
is scored, so no inferential budget and no third look at the holdout.

| K | \|\|β\|\|₂ CV (null) | L1 CV | plain-sum CV | negative-weight share | tangential share (null) | max rotation |
|---|---|---|---|---|---|---|
| 2 | 0.204 (0.133) | 0.227 | 0.363 | 0.29 | 0.80 (**0.94**) | 22.0° |
| 3 | 0.255 (0.200) | 0.347 | 0.403 | 0.24 | 0.85 (**0.88**) | 33.4° |
| 5 | 0.324 (0.253) | 0.404 | 0.440 | 0.29 | 0.90 (**0.88**) | 48.4° |

**Three findings, two of them negative.**

1. **Not a partition of unity.** 24-29% of the fitted weights are **negative**, so the weight vector
   does not live on the simplex and no non-negative gate can represent it. The mixture-of-experts
   reading is out as a literal description.
2. **The radial/tangential decomposition is uninformative, and I should not have leaned on it.** The
   tangential share is 0.90 against a null of **0.88**. In 7 dimensions only 1 of 7 directions is
   radial, so a purely random perturbation is ~6/7 = 0.857 tangential *by construction* — the 9:1
   "rotation beats rescaling" ratio is a dimensional artifact, not evidence. The null exposes it.
3. **Magnitude is not conserved either.** ``||beta||_2`` has the smallest CV of the three candidate
   invariants (0.324 vs 0.404 for L1 and 0.440 for the plain sum), so if anything is conserved it is
   the L2 norm rather than a sum — but its CV *exceeds* its own null (0.324 vs 0.253), i.e. magnitude
   varies **more** than estimation noise alone would produce.

**So §13.1's wording was too strong and is walked back here.** What survives is the *behavioural*
evidence, which is forecast-based and multiplicity-corrected: a single state-dependent scalar gain on
the composite loses at every granularity (§5, −0.0006 at K=2), while state-dependent **re-weighting**
pays (§11.1, RW adjusted p 0.0005). That licenses "the parameterisation that works is per-bucket
weights, not an overall gain" — a statement about *which model form pays*. It does **not** license
"rotation at fixed magnitude" as a geometric description, and §13.1 should be read with that
correction.

**Why the two disagree, and what would settle it.** The per-bin weight vectors are *noisily estimated*
— which is the same fact that made unshrunk per-bin fits insignificant while only 50/50 blends paid
(§11.1). The geometry of estimated coefficients is therefore dominated by estimation error, exactly as
the nulls show. Settling the geometry needs the weights measured where they are actually reliable:
refit each bin with much heavier shrinkage (or take the geometry of the *blended* weights that
demonstrably pay) and redo this decomposition. Until then the geometric question is open, and the
honest summary of the state-dependence is the behavioural one. **§14.1 does exactly that and settles
it: the geometry is an ellipse of effective rank ~1-2, oriented orthogonally to the mean weight
vector, loading on a sentiment-vs-cross-section contrast.**

### 14.1 Redone with shrunk weights, then done properly — the geometry is an ELLIPSE

Two parts: shrinkage does not answer the question (and is *unstable*, which is worse than I said), and
subtracting the estimation-error covariance analytically does answer it.

**Shrinkage cannot settle it, and the shares are penalty-dependent.** §14's radial/tangential shares
are computed on deviations from the mean weight vector, so blending each bin 50/50 toward the pooled
vector — the thing §11.1 showed actually pays — scales every deviation by the same factor and leaves
the shares nearly untouched (0.900 → 0.850 at ridge 1; I said "exactly invariant" earlier, which is
slightly too strong since the mean *direction* is recomputed). Far worse, a heavier ridge **reverses
the story**:

| ridge penalty | tangential share (raw) | tangential (50/50 blend) | \|\|β\|\|₂ CV |
|---|---|---|---|
| 1 | 0.900 | 0.850 | 0.324 |
| 1e2 | 0.253 | 0.237 | 0.356 |
| 1e4 | **0.095** | 0.086 | 0.607 |
| 1e6 | 0.093 | 0.084 | 0.612 |

Under heavy shrinkage the variation becomes almost entirely *radial*, because ridge shrinks toward
**zero** and each bin shrinks by a different amount depending on its own signal strength. So the
radial/tangential split of *fitted* weights is not a stable descriptor of anything — it swings from
0.90 to 0.09 on a knob. That kills the §14 approach outright rather than merely weakening it.

**The right instrument: subtract the noise, which is known in closed form.** The observed across-bin
scatter is ``Sigma_obs = Sigma_signal + E[Cov(beta_hat)]``, and for ridge
``Cov(beta_hat) = sigma^2 A (Z'Z) A`` with ``A = (Z'Z + lam I)^-1``. So form
``Sigma_signal = Sigma_obs - mean(Cov(beta_hat))`` and read its eigen-geometry. Validation: under a
circular-shift null the corrected trace must collapse.

| K bins | trace observed | trace noise | **trace signal** | null signal trace | eig1 share | eig2 | effective rank | radial share (isotropic 0.14) |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.4635 | 0.0384 | **0.4251** | +0.0502 | 0.80 | 0.17 | **1.50** of 7 | 0.18 |
| 10 | 0.4961 | 0.0573 | **0.4388** | +0.0438 | 0.76 | 0.18 | **1.64** of 7 | 0.14 |
| 20 | 0.3896 | 0.0736 | **0.3161** | +0.0314 | 0.61 | 0.25 | **2.22** of 7 | 0.17 |

**Verdict: an ellipse, effective rank ~1.5–2.2 of 7.** Three things follow.

1. **The variation is real.** Signal trace is **8–10x** its circular-shift null at every K, so it
   survives the noise subtraction comfortably. That is what §14 could not establish and what the
   analytic correction delivers.
2. **It is one axis, not free rotation and not an isotropic blob.** The leading eigenvalue carries
   61–80% of the signal variance, effective rank 1.5–2.2. So the state does not move the weight vector
   in 7 directions — it moves it along essentially **one**.
3. **That axis is orthogonal to the mean direction.** Radial share 0.14–0.18 against an isotropic
   baseline of 1/7 = 0.14, i.e. *no excess* magnitude variation. This partially rehabilitates §13.1:
   the movement genuinely is direction-changing rather than rescaling — but as a **single specific
   axis**, not the free rotation §13.1's wording implied.

**What the axis is.** The leading eigenvector loads dominantly on **`sentiment`** (−0.74 to −0.84
across K), then `market_vw` (−0.33/−0.36), `vol_demand`, and `market_ew` with the opposite sign
(+0.23/+0.26). Eigenvector signs are arbitrary, so read it as a **sentiment-versus-cross-section
contrast** that moves monotonically with vol state — which is the same object §4 saw as sentiment's
IC doubling from the calmest to the most stressed quintile.

**Why this matters practically.** A rank-1 axis is worth **one parameter**: ``beta(s) = beta_0 + g(s) v``
with ``v`` the leading eigenvector and ``g`` a scalar gate. That is 1 extra df versus 7K for per-bin
fitting — the parameter economy that §11.1's noise problem demanded, arrived at from the geometry
rather than guessed.

**Caveats.** ``Sigma_obs`` has only K−1 degrees of freedom, so the subtraction leaves the matrix
indefinite (3 negative eigenvalues at K=5 and 10) and **only the leading one or two eigenvalues are
trustworthy** — the effective-rank figure should be read as "1 to 2", not as 1.64. Everything here is
descriptive on the search period. And the obvious next step — fit ``beta_0 + g(s) v`` and score it — is
a *forecast* test, so it needs the holdout, which has now been evaluated twice (§12, §12.1); a third
look should be paid for with a fresh split (search ≤ 2019, holdout 2020+) rather than reusing 2021-24.

## Reproducibility

```bash
python analysis/alpha_panel.py                              # build + cache the 242,934 x 543 panel (~8 min)
python analysis/alpha_manifestation.py --stage resid         # HAR walk-forward residual
python analysis/alpha_manifestation.py --stage signals       # 7 bucket + all-exog alpha signals (~8 min)
python analysis/alpha_manifestation.py --stage tests         # density / episodes / rotation / ladder
python analysis/alpha_manifestation.py --stage dynamics      # per-bin gain + dynamic gain
python analysis/alpha_manifestation.py --stage sparse        # sparse-vs-dense horse race
python analysis/alpha_manifestation.py --stage verify        # mapping-vs-scale + DM significance
python analysis/nl_sparsity.py --stage ladder                # interactions: add? sparse? rotating?
python analysis/nl_sparsity.py --stage local                 # within-month interaction PR vs null
python analysis/nl_sparsity.py --stage vsfull                # products vs the FULL linear model
python analysis/nl_sparsity.py --stage robust                # §7.1 four scalings (why the clip mattered)
python analysis/nl_sparsity.py --stage grouppen              # §7.1 no clip, separate product penalty
python analysis/voldemand_fix.py --stage diagnose            # §8 per-stage tails, all variants
python analysis/voldemand_fix.py --stage evaluate            # §8.4 variants vs the four bars
python analysis/voldemand_fix.py --stage control             # §8.4 full rebuild + all-bucket control
python analysis/multiplicity.py --stage conditioning         # §11.1 SPA + Romano-Wolf, claim 1
python analysis/multiplicity.py --stage interactions         # §11.1 SPA + Romano-Wolf, claim 2
python analysis/diagnostics_backtest.py                      # §11.2 five checks vs 7 incidents
python analysis/alpha_manifestation.py --stage diffusion      # is the coefficient a random walk?
python analysis/heat_graph.py --stage spectrum                # §12 interaction-graph structure
python analysis/heat_graph.py --stage fit                     # §12 pre-registered Laplacian + holdout
python analysis/heat_graph.py --stage fit2                    # §12.1 corrected: real graph + feature diffusion
python analysis/intensity_graph.py --stage gate               # §13 bar-level (voided by its control)
python analysis/intensity_graph.py --stage gate2              # §13 monthly resolution + changes target
python analysis/intensity_graph.py --stage estimator          # §13.1 EWMA vs AR(1) Kalman intensity
python analysis/intensity_graph.py --stage gate3              # §13.1 the gate on the better estimator
python analysis/composition.py --stage geometry                # §14 simplex / sphere / neither
python analysis/composition.py --stage geometry2               # §14.1 shrinkage ladder + noise-corrected
```

Outputs: `results/alpha_manifestation/{report.txt, pooled_feature_ic.csv, monthly_alpha.csv,
monthly_bucket_ic.csv, monthly_mapping.csv, activation_{timeofday,dayofweek,vol}.csv,
activation_stability.csv, granularity_ladder.csv, gain_channel.csv, sparse_vs_dense.csv,
volregime_significance.csv, nl_sparsity_ladder.csv, nl_top_pairs.csv, nl_local_sparsity.csv,
nl_pair_ic.csv, nl_vs_full_linear.csv, nl_scaling_robustness.csv,
nl_group_penalty.csv, voldemand_stage_tails.csv, voldemand_variants.csv,
voldemand_fix_control.csv, semantic_rule_map.csv, semantic_rule_drift.csv,
multiplicity_conditioning.csv, multiplicity_interactions.csv, diagnostics_backtest.csv,
feature_health_production.csv, slope_diffusion.csv, heat_graph_degrees.csv,
heat_graph_fit.csv, heat_graph_fit2.csv, intensity_gate.csv, intensity_modulators_linear.csv,
intensity_gate2_levels.csv, intensity_gate2_changes.csv, intensity_estimators.csv,
intensity_gate3.csv, composition_geometry.csv, composition_weights_K{2,3,5}.csv, composition_geometry2.csv,
composition_shrinkage_ladder.csv}`.
