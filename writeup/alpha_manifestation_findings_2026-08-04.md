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

> **Update (§§16–17): the §15.3 list is executed, and two of its conclusions are revised.**
> The two confirmed gains **do** add — combined ΔR² **+0.0088** (DM-t +8.1), of which **+0.0064**
> is structurally attributable (+0.0052 products, +0.0012 vol-regime, additive to −4%) and +0.0027
> is plain estimator diversification belonging to neither. My pre-registered prediction (t < 2) was
> wrong. `sentiment × vol-state` survives a genuinely fresh split (+0.00134, DM-t +2.35, selection
> redone on ≤2019). §7's interaction gain reproduces with `voldemand` in and no clip (+0.0069, t
> +2.79), so neither patch was load-bearing. **§14's "no good geometry to exploit" is narrowed**:
> there are **two additive state axes** — vol *and* intraday clock — worth +0.0014 at DM-t **+3.09**
> out of sample, both requiring heavy shrinkage; and the interaction channel, while spectrally flat
> and unidentified, has **no exploitable block geometry either**: the liquidity × liquidity
> concentration (53 of 100 pairs, z = +3.58) is where the *selector goes*, not where the alpha is —
> excluding the block keeps **113%** of the gain and restricting to it turns +0.0069 into **−0.0050**
> (§17.3, retracting §17.2's reading). Third time in this study that apparent concentration dissolved
> under a powered test, and §17.4 now gives the *mechanism* for why graph spectral methods keep
> failing rather than just recording it.


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
honest summary of the state-dependence is the behavioural one. **§14.1 attempts that; §14.2 audits it and
finds the variation real but the axis UNIDENTIFIED — the eig1 concentration is mostly a
degrees-of-freedom artifact and the leading direction does not replicate across halves.**

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

### 14.2 Audit of the ellipse — real variation, **unidentified axis**. §14.1's interpretation retracted.

§14.1's null was applied only to the *trace*, never to the eigenstructure — and with ``K`` bins
``Sigma_obs`` has ``K-1`` degrees of freedom in 7 dimensions, so its eigenvalues concentrate **by
construction**. That is the same class of error as §14's dimensional artifact. Its noise term also
assumed iid errors while the residual is heteroskedastic and autocorrelated, which shows up as a null
"signal" trace of +0.05 instead of 0. `--stage audit` fixes both (moving-block bootstrap for the noise
covariance) and adds the test that is immune to either: **does the leading axis replicate across halves
of the search period?**

| K | noise | trace signal (null) | eig1 share (**null**) | eff rank (null) | axis \|cos(v_h1,v_h2)\| (null) |
|---|---|---|---|---|---|
| 10 | analytic | +0.439 (+0.044) | 0.756 (**0.590**) | 1.64 (2.28) | — |
| 10 | **bootstrap** | +0.352 (+0.007) → **48x** | 0.780 (**0.668**) | 1.56 (1.93) | **0.309 (0.310)** |
| 20 | analytic | +0.316 (+0.031) | 0.615 (**0.555**) | 2.22 (2.50) | — |
| 20 | **bootstrap** | +0.285 (+0.030) → 9.5x | 0.618 (**0.529**) | 2.16 (2.62) | 0.549 (0.277) |

**What survives, and it is worth having.** The signal trace is 9.5-48x its null under the *honest*
bootstrap noise estimate — and the bootstrap makes it stronger, not weaker, because the analytic
formula was under-subtracting. So **the bucket weights genuinely do move with vol state, beyond
estimation noise.** That is now the third independent confirmation of §11.1's behavioural result, from
a completely different instrument.

**What does not survive: the axis.** Two failures.

1. **eig1 concentration is mostly the df artifact.** 0.780 against a null of **0.668** (K=10), 0.618
   against 0.529 (K=20). The excess is ~0.09-0.11, so "one dominant axis" is largely what 9 degrees of
   freedom in 7 dimensions produces on its own. §14.1 read that as a finding.
2. **The axis does not replicate.** At K=10 the split-half cosine is **0.309 against a null of 0.310** —
   exactly the null. K=20 does better (0.549 vs 0.277), but look at the loadings: first half is
   ``moments -0.79, vol_demand -0.31, market_ew +0.31``; second half is
   ``sentiment -0.73, market_vw -0.46, liquidity +0.33``. **The two halves name disjoint buckets.** A
   cosine of 0.55 between such vectors reflects shared structure (all bucket weights are positively
   correlated), not agreement about a direction.

**So §14.1's "the axis is a sentiment-versus-cross-section contrast" is retracted.** That was the
*second half's* draw; the first half says moments/implied_vol. I read one half of one specification as
the identity of a latent axis.

**Also recorded: my own verdict rule was too lenient and is fixed in the code.** It averaged the
split-half cosine across K, letting K=20's 0.549 mask K=10's 0.309, and printed "EXPLOITABLE". An axis
is identified only if it clears its null at *every* K and the loadings agree; the audit now requires
both and prints **"REAL VARIATION, UNIDENTIFIED AXIS"**.

**How to exploit it, then.** Not with a fitted rank-1 axis — the direction cannot be estimated from
this data, and a model that estimates ``v`` will fit whichever half it saw. Two routes remain:

* **What is already established** stays the answer: the coarse 2-3 bin vol-regime blend, ΔR² +0.0024 at
  RW adjusted p 0.0005 (§11.1). The geometry work did not improve on it. It confirmed the phenomenon a
  third time and showed why a finer parameterisation has not paid: the extra structure is real but not
  identifiable at this sample size.
* **Pre-specify the direction instead of estimating it.** A single *fixed* contrast — e.g. sentiment
  versus moments, which §4's activation map motivates independently (sentiment's IC doubles from the
  calmest to the most stressed quintile while moments rises monotonically) — costs 1 df and estimates
  no eigenvector, so it is far better powered than the rank-1 fit. That is the one version of this idea
  the data could still support, and it must be pre-registered with a fresh split (search <= 2019,
  holdout 2020+) since 2021-2024 has now been looked at twice.

### 14.3 The axis estimated at BAR resolution — and a correction to §14.2's diagnosis

§14.2's summary blamed the failure on "167 monthly observations". **That was wrong, and the error is
worth stating precisely** because it conflated two different objects:

* The **intensity** target (§13) really does have ~167 effective observations. It is a ~1-month EWMA
  with one-day autocorrelation 0.988, so its honest noise floor is 1/sqrt(167) = **0.077**, not
  1/sqrt(218,934) = 0.002 — a **36x** difference, and exactly why the bar-level 7x "excess" evaporated
  once resolution was honest. Slicing a smooth series finer manufactures no information.
* The **composition/axis** question has nothing to do with monthly anything. Each bin's weights were fit
  on ~17,657 bars (K=10 over 176,574 search rows). The binding constraint was **K**, because the
  across-bin covariance gets only ``K-1`` degrees of freedom in 7 dimensions. Binning was inherited
  from §5's granularity ladder and never re-examined.

So drop the bins entirely. Write the weights as a smooth function of a continuous causal state ``z``
and take the first-order term — ``e ~ sum_k beta_k s_k + sum_k gamma_k (s_k z)`` — where ``gamma`` **is**
the axis, estimated from all 176,574 bars with 7 parameters rather than read off an eigendecomposition
of 10 noisy bin estimates. HAC (Newey-West, 10 trading days) throughout.

| bucket | β | **γ** | HAC-t(γ) |
|---|---|---|---|
| moments | +0.749 | +0.043 | +0.82 |
| liquidity | +0.217 | −0.104 | −1.58 |
| implied_vol | +0.015 | −0.072 | −0.71 |
| market_vw | +0.193 | +0.233 | +1.56 |
| market_ew | −0.018 | −0.174 | −1.26 |
| vol_demand | +0.014 | +0.153 | +0.88 |
| **sentiment** | +0.162 | **+0.514** | **+2.97** |

**Joint HAC Wald test γ = 0: χ²(7) = 28.1, p = 2.1e-04.** ‖γ‖/‖β‖ = 0.762.

**Three conclusions, and they finally separate cleanly.**

1. **The state-dependence is real, now confirmed at full power.** A joint Wald test on all 176,574 bars
   rejects γ = 0 at p = 2e-4. That is the fourth independent confirmation (after §11.1's blend gain,
   §14.1's trace, §14.2's bootstrap-corrected trace) and the first at bar resolution with a proper test.
2. **The full 7-dimensional axis still does not replicate.** Split-half cos(γ_h1, γ_h2) = **+0.126**
   against a null of −0.193 with sd 0.404 — inside one standard deviation. Seven correlated coefficients
   each with |t| ~ 1 define a direction that is mostly noise, and no amount of bar-level precision fixes
   that: the problem is 7 parameters, not the sample size.
3. **But one component is consistent, and it is the one to pre-specify.** ``sentiment`` carries
   γ = +0.514 with HAC-t **+2.97** — the only individually significant loading — and it is positive in
   *both* halves (+0.143, +0.589). That is precisely the "pre-specify the direction instead of estimating
   it" route §14.2 recommended, with the data now nominating the specific contrast and attaching a
   t-statistic to it.

**Caveat on that nomination.** γ_sentiment is the largest of 7 tested loadings, so 7-way selection
applies: one |t| > 2 among 7 arises by chance roughly 30% of the time. t = 2.97 is stronger than that,
the joint Wald is independently significant, and the sign replicates across halves — but the honest
status is "well-motivated candidate", not "established effect". It was nominated on the search period
and must be scored on a **fresh** split (search <= 2019, holdout 2020+), since 2021-2024 has now been
looked at twice.

**Which makes the exploitation route concrete at last**: a single ``sentiment x vol-state`` interaction
term, 1 degree of freedom, added to the pooled combiner. Not a fitted 7-D axis, not per-bin weights, not
a rank-1 eigen-model — one pre-specified interaction whose sign and rough magnitude are already known.

## 15. Synthesis: what "dense but weak" turned out to mean, what the graph work contributed, what to do next

### 15.1 Did the graph-Laplacian work find the state-dependence? No.

Worth tracing honestly, because the answer is a clean no and the detour was expensive:

| step | what produced it | graph involved? |
|---|---|---|
| vol-state conditioning pays, clock does not (§5) | granularity ladder | no |
| the *magnitude* channel is flat — 1-df gain loses at every K (§5) | gain-channel arms | no |
| confirmed at SPA p 0.0000 / RW 0.0005 (§11.1) | bootstrap over the maximum | no |
| the composition reframing (§13.1) | **re-reading §5's own two arms against each other** | no |
| "ellipse" (§14.1) → retracted (§14.2) | 7x7 covariance eigendecomposition | no (linear algebra, not a Laplacian) |
| **sentiment x vol-state, γ = +0.514, HAC-t +2.97 (§14.3)** | **a 14-parameter interaction regression with HAC errors** | **no** |

The Laplacian sections (§12, §12.1) contributed **nothing** to the finding. Their outputs were a clean
negative (graph smoothing does not beat plain shrinkage: holdout ΔR² +0.00022, then +0.00039 with a
properly estimated graph), one reusable lesson (**similarity adjacency helps, interaction adjacency
hurts** — the prior's assumption is about correlation, not interaction), and a demonstration that the
pre-registered-holdout protocol catches inflation (a search gain of +0.00047 halved out of sample).

Worse: **the finding was already visible in §4** without any of it. §4's activation map showed
`sentiment`'s IC more than doubling from the calmest vol quintile (0.040) to the most stressed (0.092) —
which is exactly γ_sentiment > 0. Three sections of graph machinery arrived at a result that a plain
interaction regression on §4's own observation would have produced immediately. Recorded so the next
person does not repeat it.

### 15.2 How this is a dense-but-weak story — the three channels are the same fact

The findings look scattered until they are lined up by *where* the diffuseness lives:

| channel | what "weak and diffuse" means there | the measurement |
|---|---|---|
| **features** | many tiny contributions, no usable concentration | mean \|IC\| 0.017 over 246; dense beats top-5 at DM-t **−7.24**; ~100+ features needed to saturate |
| **time** | always on, never off, no local amplification | 0 of 207 months negative; concentration factor `c` = **0.95–1.05** on every axis; finer slicing loses monotonically to −0.031 |
| **state** | real re-weighting, but only 1 of 7 loadings resolvable | joint Wald p = 2e-4, yet split-half axis cos +0.126 vs null −0.193; only `sentiment` has \|t\| > 2 |

So "dense but weak" is not just about the cross-section of features — **it is the signature of the whole
problem in all three directions at once.** Each channel is genuinely non-zero and each is worth
~0.002–0.007 R² on the residual; none is concentrated enough to behave like a "signal" in the usual
sense.

That single characterisation predicts every result in this document, including the failures. **Everything
that assumed concentration failed**: sparse selection (−7.24), fine time-slicing (−0.031), dynamic gains
(−0.002), rank-1 axis fitting (unidentified), graph smoothing (+0.0004), a transformer (QLIKE 0.46 vs
0.14). **Everything that pooled and shrank worked**: all 246 features, coarse 2–3 state bins, 50/50
blends, heavy product penalties, a frozen-once selection. The correct posture is aggregation and
shrinkage everywhere, and the returns to structure-finding on this panel are approximately zero.

### 15.3 What to do next, in order

> **All seven executed — see §16 for results and §17 for the geometry follow-up this list did not
> anticipate. Two items came back differently than framed: item 1's gain is real but a third of it is
> not attributable to either lever, and item 4 is one mechanism rather than the "voldemand fix
> generalised" this list assumed.**


**Finish what is already confirmed** — small, well-defined, uses existing machinery:

1. **Test additivity of the two confirmed gains.** The vol-regime blend (+0.0024, RW p 0.0005) and the
   ~100 frozen interaction products (+0.0067, RW p 0.024) have never been run together. If additive that
   is +0.009 ≈ 26% of the entire exog channel — **the largest unclaimed number in this study**.
2. **Score `sentiment × vol-state` on a fresh split** (search ≤ 2019, holdout 2020+; 2021–24 has been
   used twice). 1 df, sign and magnitude pre-specified from §14.3, nothing left to tune.

**Ship the infrastructure that already exists**:

3. Wire `src/diagnostics.py` into the panel build so the five invariants run on every build.
4. Fix the **12+ columns** the diagnostic flagged at max \|z\| 68–82 (§11.2) — the voldemand fix
   generalised. `adj_numobs_ma_25` has a p99.9 of 46 against the panel's 7.6.
5. Re-run §7's interaction study on the §8-fixed panel; it was done with `voldemand` dropped and a ±4
   clip that may now be unnecessary.

**The one genuinely unexplored lever — new data, not new method**:

6. **Build the OPEX / rebalance / quad-witch calendar features.** §0 unblocked the row→date map, and the
   intraday-regime doc has wanted these since June. Deterministic, free, and *new information* rather
   than another re-analysis of the same 543 columns.
7. **Construct the explicit OFI term** `(buy−sell)/(buy+sell)`. §7's selector reached for its proxy
   (`sumret × sumvolume`, 144 of 218 months) unprompted, which is about as strong a prior as this data
   gives.

**Do not do** — the clearest output of the session: no more sparse selection, no finer time-slicing, no
dynamic gains, no graph regularisation, no rank-1 axis fitting, no sequence models.

**And the structural conclusion.** Every remaining lever *on this panel* is worth ~0.002–0.007 R². If the
goal is a materially better forecast rather than another 5%, the binding constraint is **data, not
method**: a cross-sectional panel (many assets, so state-dependence is estimated across assets as well as
through time — which is the one thing that fixes the identification failures in §13 and §14), or the
auction-imbalance / GEX / 0DTE feeds the intraday-regime study named. Everything else is polishing.

## 16. The §15.3 list, executed

All seven items. Code: `analysis/synthesis.py` (items 1, 2, 5), `analysis/tail_fix.py` (item 4),
`src/diagnostics.py` + both build sites (item 3), `src/features/extractors/expiry.py` and
`src/data/loading.add_derived_features` (items 6, 7). Everything measured on the **§8-fixed panel**,
whose own `prep` re-derives the HAR residual (R² 0.5773, sd 0.2485) and the seven bucket signals
(all-exog R² **+0.0377**, the exog channel every ΔR² below is a fraction of).

### 16.1 Item 1 — the two gains DO add, and my pre-registered prediction was wrong

Pre-registered before running: sub-additive, DM-t of the combined model against the better single arm
**positive but under 2**. Result: **+7.5**. The prediction was wrong by a wide margin.

Getting there took two architectures, and the first one has to be reported because rejecting it was a
judgement call. Attempt 1 put the interaction gain into the second-stage combiner as a *single* extra
signal (`aug − base`). It scored ΔR² **−0.0096** on the full sample while being additive and
significant on 2020+ — a pattern that indicts the wrapper, not the gain: collapsing 100 columns that
each carry their own ridge penalty into one coefficient destroys the selective shrinkage that
produced §7's +0.0069 in the first place. That is an *a priori* argument, not "it lost", but it is
still a second architecture scored on the same sample and §10's warning applies.

Attempt 2 leaves both estimators exactly as measured and asks what "do the gains add" actually means
— **does using both beat using either** — by 50/50 forecast combination (equal weight *pre-specified*,
because fitting the weight would be one more spec chosen on exhausted data):

| K | M0 pooled | f_A vol-regime blend | f_B linear+products | f_AB 50/50 | DM-t vs best single |
|---|---|---|---|---|---|
| 2 | +0.0297 | +0.0316 (+0.0018) | +0.0259 (−0.0039) | **+0.0383 (+0.0086)** | **+7.58** |
| 3 | +0.0298 | +0.0319 (+0.0022) | +0.0259 (−0.0039) | **+0.0385 (+0.0088)** | **+7.53** |
| 5 | +0.0298 | +0.0320 (+0.0021) | +0.0259 (−0.0039) | **+0.0387 (+0.0089)** | **+7.71** |

+0.0088 is the +0.009 §15.3 nominated. But `f_B` **alone loses to the pooled combiner**, so the
additivity ratio is negative and meaningless, and the gain is partly just two structurally different
estimators making independent errors. That has to be netted out before anything is credited, so the
2×2 (K=3, `--stage additivity3`):

| source | ΔR² | share | DM-t vs plain combination |
|---|---|---|---|
| plain diversification (M0 + linear direct model) | +0.00269 | 31% | — (t +2.33 vs M0) |
| **marginal from the ~100 frozen products** | **+0.00523** | **60%** | **+5.36** |
| **marginal from the vol-regime blend** | **+0.00119** | **14%** | **+5.32** |
| interaction / remainder | −0.00037 | −4% | |

**Both confirmed gains contribute on top of diversification, each at DM-t ≈ +5.3, and they are
additive to within −4%.** The structurally attributable total is **+0.0064 ≈ 17%** of the exog
channel; the *full* combined model is +0.0088 ≈ 23%, of which +0.0027 belongs to neither gain.
Anyone quoting +0.0088 as "the confirmed gain from the two levers" is over-claiming by a third.

Each marginal is smaller than its standalone measurement (+0.0052 vs +0.0069; +0.0012 vs +0.0022), so
against *standalone* the pair is mildly sub-additive — which is what was pre-registered, even though
the headline test was not.

### 16.2 Item 2 — `sentiment × vol-state` survives a fresh split

The §14.3 nomination was made on a sample that already contained the twice-scored 2021–24 block, so
the **selection** was redone on ≤2019 only. It independently returns the same answer:

| bucket | γ (≤2019) | HAC-t |
|---|---|---|
| moments | +0.019 | +0.34 |
| liquidity | −0.108 | −1.60 |
| implied_vol | −0.130 | −1.15 |
| market_vw | +0.264 | +1.69 |
| market_ew | −0.165 | −1.12 |
| vol_demand | +0.194 | +0.96 |
| **sentiment** | **+0.622** | **+3.28** |

Joint HAC Wald χ²(7) = 25.4, p = 6.5e-04. `sentiment` is again the only component past |t| = 2. On the
**2020+ holdout**, the single 1-df term scores ΔR² **+0.00134, DM-t +2.35** — clearing the
pre-registered one-sided 1.645 — and it is positive in-sample too (+0.00081, t +1.90). First
state-dependent term in this study to survive a genuinely fresher split.

### 16.3 Item 5 — neither of §7's two patches was load-bearing

§7.3 had to drop `voldemand` and §7.1's ±4 clip existed to survive exactly that class of column. On
the fixed panel, with `voldemand` **included** and **no clipping anywhere** (max |X| 78.8):

| α_prod | stat100 ΔR² | DM-t | dyn100 ΔR² | DM-t | dyn vs stat |
|---|---|---|---|---|---|
| 3e3 | −0.0044 | −0.64 | −0.0063 | −1.72 | −0.26 |
| **3e4** | **+0.0069** | **+2.79** | +0.0013 | +0.47 | −1.29 |
| 3e5 | +0.0044 | +2.78 | +0.0025 | +1.72 | −0.71 |
| 3e6 | +0.0007 | +2.76 | +0.0007 | +3.14 | +0.22 |

+0.0069 / +2.79 against §7.3's +0.0067 / +2.71 — reproduced to the third decimal. Static still beats
dynamic (monthly selection churn 0.257 and it buys nothing). Positive and significant at 3 of 4 grid
points, against 2 of 4 before, so the fix made it *less* penalty-sensitive as predicted.

### 16.4 Item 4 — one mechanism, not three, and one fix measured and rejected

Two of the three predicted mechanisms were wrong. The per-slot divisor is **not** pinned (0.000 for
the whole `sum*stock` family) and the winsoriser is **not** blind at those rows (0.001 of rows). What
the row-level trace shows is a single mechanism behind eleven of the thirteen columns: **the per-slot
diurnal baseline collapses within its own slot, on overnight bars.**

| column @ argmax | baseline | that slot's median baseline | ratio | → max\|z\| |
|---|---|---|---|---|
| `sumpret2_vwstock` @ 03:00 | 5.7e-08 | 1.1e-06 | **19× low** | 82.1 |
| `sumbipow_ewstock` @ 03:00 | 1.9e-08 | 4.4e-07 | **22× low** | 77.3 |
| `stocktwits_sentiment` @ 05:30 | 0.0142 | 0.143 | **10× low** | 70.3 |

And the winsoriser cannot catch it: at `sumpret2_vwstock`'s argmax the pooled upper bound **equals the
value**, 46.03 — twelve or more of the trailing 240 bars were already past it, because overnight bars
are ~62% of a pooled window and set the very quantile meant to bound them. `numobs` is the one genuine
exception: capped at 30 and *equal to* 30 on 78% of bars, so its IQR is a rounding artifact and a
holiday half-session is correctly ~79σ away.

Four fixes, each behind a switch so the ladder attributes change to a mechanism, and an assertion that
the `pre` arm still reproduces max|z| 82.1 (the check that caught §8's baseline silently drifting):

| arm | max\|z\| | cols ≥ 50 | cols ≥ 20 | rows ≥ 20 |
|---|---|---|---|---|
| `pre` | 82.1 | 25 | 37 / 156 | 4,383 |
| per-slot band 0.50 | 78.8 | **8** | 36 | 4,370 |
| **+ stem exclusion + mode hurdle + guard** | 78.8 | **4** | **29** | **3,349** |
| + per-slot winsorisation | **174.4** | 3 | 16 | 482 |
| per-slot winsorisation only | **218.7** | 6 | 17 | 287 |

- **The band** clips the per-slot baseline to `[f, 1/f]` of that slot's own **trailing-year** rolling
  median (not expanding — an expanding reference is dominated by the start of the sample and would
  clip the vol *regime*). It is two-sided because a transiently large baseline destroys signal as
  surely as a small one inflates it. `f` was fixed at 0.50 from the invariant ladder — 0.10 / 0.25 /
  0.50 leave 19 / 16 / 9 columns past 50, monotone — and choosing it that way spends no inferential
  budget because no forecast is ever scored.
- **The stem exclusion** is a latent bug: `DIURNAL_EXCLUDED = {... "vix", "sentiment"}` was tested by
  *exact membership*, so **no column named `sentiment` has ever existed** and `"vix"` matched one of
  three. `vvix` and `vix3m` were being diurnally adjusted against the constant's own stated intent,
  and `stocktwits_sentiment` — a ratio bounded in [−1, 1] — was dividing by a collapsing per-slot std.
  Matched as stems, `adj_stocktwits_sentiment_ma_*` goes **70.3 → 4.0–5.1**.
- **The mode hurdle** generalises `ZERO_INFLATED_FRAC` from exact zeros to the modal value. This is
  bit-identical for `voldemand` (whose mode *is* 0) and across all 43 raw exog catches exactly one new
  column, `numobs` at 0.78 (next highest: `stocktwits_sentiment`, 0.12). It adds the `numobs_active`
  ("this bar was not a full 30-observation bar") indicator — 156 columns → 162 — and it leaves
  `adj_numobs_ma_25`'s max at **78.8, unchanged**. That is not a failure, it is the point: the
  mode-inflation test inside `_build_scale_guards` reads the *moving-average* column, which no longer
  has an exact point mass once averaged, so the magnitude column's scale is untouched. And no scale
  estimator should shrink it — a 3-observation bar genuinely is nothing like a 30-observation bar. The
  extensive margin is what carries the information and the indicator now carries it explicitly; the
  magnitude column is handled by the **invariant** instead, which for a hurdle-encoded column judges
  the tail on p99.9 (46, so a `warn`) rather than on a max that a point mass makes uninformative.
- **The guard fix** erodes the availability mask by the MA window and gives the reference IQR a
  nonzero value while it is unmeasurable (it was **0.0** — "impose no floor" — on exactly the rows
  where the local window is all imputed constant). `adj_vix3m_ma_3125` goes **50.0 → 5.5**.

Per column, against the adopted arm:

| column | pre | post | ratio |
|---|---|---|---|
| `adj_stocktwits_sentiment_ma_125` | 56.0 | **2.9** | 0.05 |
| `adj_stocktwits_sentiment_ma_1` | 70.3 | **4.0** | 0.06 |
| `adj_stocktwits_sentiment_ma_{5,25}` | 70.3 | **4.2–4.9** | 0.06–0.07 |
| `adj_vix3m_ma_3125` | 50.0 | **5.5** | 0.11 |
| `adj_sumbipow_ewstock_ma_1` | 77.3 | 40.8 | 0.53 |
| `adj_sumpret2_vwstock_ma_{1,5}` | 78.1 / 82.1 | 45.8 / 48.8 | 0.59 |
| `adj_sumabsret_ewstock_ma_{1,5}` | 59.4 / 66.9 | **58.9 / 66.7** | 0.99 |
| `adj_numobs_ma_{5,25}` | 51.6 / 78.8 | **51.6 / 78.8** | 1.00 |

So the band **roughly halves** the `sum*stock` family rather than curing it, and the stem-exclusion and
guard fixes cure `sentiment` and `vix3m` outright. Four columns remain past 50 —
`adj_numobs_ma_{5,25}` (bimodal by construction, resolved by the invariant exemption above) and
`adj_sumabsret_ewstock_ma_{1,5}`, which the band barely touches because their argmax baseline is only
2.6× below its slot median while the *raw* print is 17× that slot's median. Those two are not a
transform defect: they are a real, extreme, thinly-traded overnight print in a cross-sectional stock
aggregate, and the honest treatment is to decide whether `sum*stock` columns are *defined* outside RTH
at all — the impute-and-indicate machinery already exists for "undefined here". That is a data-definition
change, out of scope for this item, and it is the recommendation.

**Per-slot winsorisation is implemented, measured, and rejected.** It is spectacular on the bulk —
extreme rows 4,383 → 287, a 15× reduction — and it makes three `_ma_1` columns far worse
(`adj_sumpret2_vwstock_ma_1` 78 → 219, `adj_sumabsret_ewstock_ma_1` 59 → 164). Self-inflicted:
clipping 10% of *every* slot into the body tightens the distribution enough that
`rolling_robust_scale`'s IQR collapses, so a value at its own trailing bound is a large number of very
small IQRs from the median. A 15× cut in extreme rows is not worth a new 219σ column. The obvious
variant — a per-slot **1/99** bound, capping the tail without crushing the body — is named as the next
step and deliberately **not** run: the quantile pair is the one free parameter and cycling it until an
arm clears is the pattern §10 exists to prevent, invariant or not.

Shipped: band 0.50 + stem exclusion + mode hurdle + guard fix, `WINSOR_BY_SLOT_DEFAULT = False`. 79 of
156 columns are bit-identical, so the fix is surgical. The band is applied to **exog only**; the
target keeps its chain exactly, because `baseline` is the multiplicative factor every raw-space
reconstruction and every published QLIKE in this repo is defined against. `RV` takes the same unsigned
mean branch and is a candidate for the same treatment — that needs its own re-baselining.

### 16.5 Items 3, 6, 7 — shipped

- **Item 3.** `src/diagnostics.check_and_report` runs in `executor._backtest_and_save` and
  `alpha_panel.build_panel`, writing `<output>_feature_health.csv` on every build. Advisory by
  default because 26 columns currently FAIL; `FEATURE_HEALTH_STRICT=1` makes it fatal. Two new
  false-alarm exemptions, both found by the checks firing on healthy columns: a diurnal-excluded
  column has no divisor to measure, and a hurdle-encoded column's tail is judged on p99.9 rather than
  the max, because a genuine point mass makes the max uninformative (`adj_numobs_ma_25` at p99.9 46
  correctly stays a **warn**, not a muted "ok").
- **Item 6.** `src/features/extractors/expiry.py`: `is_opex` (third Friday, snapped to the last
  trading day at or before it so a Good Friday expiry lands correctly), `is_opex_week`,
  `is_quad_witch`, `is_rebalance_close` (quad witch × the close gate — the index-tracker rebalance
  prints in *that* auction, so a day-level dummy would average it with 47 ordinary bars),
  `is_month_end`, `is_quarter_end` (a period's last observed day only counts if a later period is also
  observed, else the last row of the sample is mislabelled both), `days_to_opex` (signed **trading**-day
  ramp, ±10). Verified against 2024: Jan 19 / Feb 16 / Mar 15 / Apr 19, quad witch Mar 15.
- **Item 7.** `loading.add_derived_features`: `ofi_{ew,vw}stock = (buy−sell)/(buy+sell)`, NaN where
  the denominator is zero, built on the **unfilled** legs — `apply_overnight_fills` writes 1.0, and
  buy = sell = 1.0 would make the imbalance exactly 0, fabricating "perfectly balanced flow" on bars
  with no trading. Routed to the `liquidity` bucket explicitly, since the bare `ewstock` suffix would
  otherwise have put a flow-composition measure in `market_ew` with the return moments.

## 17. Revisiting the geometry — §14's conclusion was too broad

§14 ended on "no good geometry to exploit". That was over-stated, and its own numbers say why: the
joint HAC Wald test that the state-dependence direction is non-zero returned **χ²(7) = 28.1,
p = 2e-4** with `‖γ‖/‖β‖ = 0.762`, and then the *identification* test — a split-half cosine — returned
+0.126 against a null of −0.193 **with null sd 0.404**. A test whose null scatters over half the unit
interval cannot separate "no axis" from "an axis measured with 3× too little precision". §14 read the
first; §16.2 then proved the second for one coordinate. Code: `analysis/geometry.py`.

**A direction can be too noisy to report and still be good enough to use.** So the powered test:
freeze γ on ≤2019, score the forecast on 2020+, over a pre-specified shrinkage ladder.

### 17.1 There are TWO state axes, they are additive, and both need heavy shrinkage

ΔR² and DM-t on the 2020+ holdout, everything frozen on ≤2019, against the pooled combiner:

| shrink | vol only (7 df) | clock only (28 df) | **both (35 df)** |
|---|---|---|---|
| 0.15 | +0.00047 (t **+2.29**) | +0.00029 (t **+2.82**) | +0.00075 (t **+3.09**) |
| 0.30 | +0.00081 (t +2.02) | +0.00044 (t +2.15) | +0.00121 (t +2.58) |
| 0.50 | +0.00105 (t +1.63) | +0.00042 (t +1.24) | **+0.00140** (t +1.85) |
| 0.75 | +0.00101 (t +1.09) | +0.00005 (t +0.09) | +0.00094 (t +0.86) |
| 1.00 | +0.00060 (t +0.50) | −0.00072 (t −1.04) | −0.00030 (t −0.21) |

Three findings.

1. **The raw estimate is worthless and a shrunk one is not.** Every state shows an inverted U with its
   peak between 0.3 and 0.5 and the unshrunk axis at or below zero. That is the exact signature of a
   real direction estimated with too little precision — which is what §14 saw and misread as absence.
   The shape holds for all three states, which 18 arms of noise would not produce; the *pre-registered*
   expectation was that the optimum would sit well below 1, and it does.
2. **The intraday clock is a second, independent axis.** Never tried before: time-of-day had only ever
   entered as a bin split, which pays §5's `1/√K` tax, whereas two Fourier harmonics cost 2 df per
   harmonic per signal and are fit on every bar. It is *smaller* than the vol axis in ΔR² but *more
   reliably signed* (t +2.82 vs +2.29) — matching §4, where the bucket × time-of-day activation map was
   the more split-half-stable of the two (+0.68 vs +0.64). My pre-registered guess that it would be
   *larger* was wrong on magnitude and right on reliability.
3. **The two axes are almost exactly additive.** At shrink 0.15, 0.00047 + 0.00029 = 0.00076 against
   +0.00075 observed. `both` beats `vol` at every shrinkage. **DM-t +3.09 is the strongest
   state-dependence result in this study**, on a fresh holdout with the direction frozen.

One honest deduction: the 1-df `sentiment` term scores +0.00134 / t +2.35 when its coefficient is
**refit walk-forward** (§16.2) but only +0.00083 / t +0.89 when **frozen** on ≤2019. Freezing costs
about 40% of it. So the axis is real, and its *length* drifts. And under the freeze protocol 7 df
(+0.00105) barely beats 1 df (+0.00083; DM-t of the difference **+0.52**) — so the usable content is
about one direction's worth, which is consistent with §14's failure to resolve seven.

Caveat: 3 states × 6 shrinkages = 18 arms on the 2020+ block. The *shape* is the finding; the specific
numbers need a Romano-Wolf pass before being quoted as confirmed.

### 17.2 The interaction form has no spectral geometry — and real block geometry

§7's 100 frozen products are not a bag of features. They are a sparse symmetric bilinear form `B` with
`x'Bx` in the forecast, and nobody had ever looked at its spectrum. Fitted on the first window (what
§7's static arm actually freezes), against a null that re-fits the *same 100 positions* to a
circularly-shifted residual:

| | observed | null |
|---|---|---|
| participation ratio of \|λ\| | **20.3** of 133 | **19.4 (sd 6.1)** |
| top \|λ\| share | 0.092 | 0.107 (sd 0.038) |
| signature | 47 positive / 52 negative | |
| leading eigenvector split-half \|cos\| | **0.000** | 0.080 (sd 0.110) |

**Flat, indefinite, unidentified.** The effective dimensionality is exactly what fitting those 100
positions to noise gives, the top eigenvalue share is *below* null, and the leading eigenvector is
orthogonal across halves. So "project onto the top few eigenvectors instead of carrying 100 products"
is dead, and dense-weak now holds in a second channel. The near-balanced signature also kills the
"interactions only ever add volatility" reading: the channel encodes cancellation.

Its leading eigenvector had a loud pattern anyway — **all twelve largest loadings were `_ma_625`** —
and a spectrum is the wrong instrument for that, because it asks about directions in an unordered
133-dim space while the space carries two natural block labels: the HAR window, and the exog bucket.
Asked combinatorially, with the same circular-shift-selector null:

| HAR window block | selected | by chance | selector null | z |
|---|---|---|---|---|
| 625 × 625 | 37 | 10.1 | 49.6 (sd 23.2) | **−0.54** |
| 1 × 25 | 34 | 19.8 | 6.2 (sd 8.5) | **+3.27** |
| 1 × 1 | 26 | 10.1 | 2.2 (sd 3.3) | **+7.30** |

| exog bucket block | selected | by chance | selector null | z |
|---|---|---|---|---|
| **liquidity × liquidity** | **53** | 7.5 | 12.8 (sd 11.2) | **+3.58** |
| liquidity × market_ew | 12 | 7.3 | 8.8 (sd 7.5) | +0.42 |
| liquidity × market_vw | 9 | 7.3 | 7.9 (sd 5.7) | +0.19 |
| everything else | ≤ 6 | | | \|z\| ≤ 1.5 |

**The `_ma_625` pattern is a selector artifact and the null caught it.** Slow columns are smooth, so
their pair-ICs are inflated against *anything* — including pure noise, where the selector takes 49.6 of
them on average versus the 37 it takes for real. My eyeball read of the eigenvector was reading that
artifact. What survives the null is the opposite: the signal-bearing pairs are **fast**, `1×1` at
z = +7.3 and `1×25` at z = +3.3.

And the bucket answer looks sharp: **53 of 100 selected pairs are `liquidity × liquidity`**, 7× chance
and z = +3.58, with every other block inside 1.5σ.

### 17.3 The block concentration is real and NOT exploitable — §17.2's reading retracted

The obvious use of §17.2 is to restrict the candidate space to the concentrated block. Tested with the
control that decides it, `--stage block_exploit`: five arms differing **only** in which pairs the
selector may choose from, everything else (frozen-on-first-window selection, floored product scale,
α_prod, no clip) identical.

| candidate space | pairs | full ΔR² | DM-t | 2020+ ΔR² | DM-t | vs `all` |
|---|---|---|---|---|---|---|
| `all` | 8,911 | +0.00690 | +2.79 | +0.00415 | +1.77 | — |
| `fast` (windows 1, 25) | 3,570 | +0.00633 | +2.24 | +0.00536 | +3.86 | −0.56 |
| **`complement`** (liq × liq **excluded**) | 8,245 | **+0.00782** | **+3.26** | +0.00449 | +2.45 | +1.89 |
| `liq_x_liq` | 666 | **−0.00498** | −1.67 | −0.00406 | −0.88 | **−3.87** |
| `liq_x_liq_fast` | 300 | **−0.00684** | −1.56 | −0.01079 | −0.93 | **−3.35** |

**Excluding the block keeps 113% of the gain; restricting to it destroys the gain.** So §17.2's
concluding sentences — "the interaction channel *is* products of contemporaneous liquidity measures",
"a mechanism rather than a coincidence", "the one place in this study where a channel is genuinely
concentrated" — are **wrong and withdrawn.** The liquidity block is where the *selector goes*, not
where the *alpha is*.

The methodological error is worth naming exactly, because it is subtle and I built the null that was
supposed to prevent it. The circular-shift-selector null answers **"does the selector visit this block
more than it would on destroyed signal?"** — and z = +3.58 genuinely answers yes, so the block is not a
selector artifact in the way `_ma_625` was. But that is a question about *selection frequency*, and I
wrote a conclusion about *forecasting content*. Those come apart precisely when a block's in-sample
pair-ICs are inflated by the parent features being mutually correlated — which is exactly what
liquidity features are. The selector is drawn there by real in-sample covariance that does not
generalise. **A selection-frequency test and a forecast test are different tests, and only the second
one licenses "this is where the alpha lives."**

This is the third time in this study that apparent concentration dissolved under a properly powered
test (§13.1's intensity, §14.1's ellipse, now §17.2's block), and the pattern is itself the finding:
*every* structure that looked concentrated turned out not to be, in every channel examined.

**So the corrected verdict on geometry.** There is no *spectral* geometry, no identifiable
7-dimensional rotation axis, and no exploitable *block* geometry. What survives is exactly one thing:
a real, heavily shrinkage-dependent state-dependence with **two additive axes** — vol and intraday
clock — worth **+0.0014 at DM-t +3.09** on a fresh holdout. §14's "no good geometry to exploit" was
over-stated only in that one respect, and is otherwise confirmed and now confirmed in a second channel.

### 17.4 Why graph spectral methods are the wrong family here — with a mechanism, not just a record

§12 (heat kernel) and §13 (graph-structured intensity) both failed. §15.1 recorded that the graph work
contributed nothing. §17.2–17.3 now supply the *reason*, which is more useful than the record:

1. **There is no compression to buy.** A graph-Fourier or wavelet basis pays off when the signal is
   concentrated in few coefficients of that basis. The fitted interaction form's effective rank is
   **20.3 against a null of 19.4** — as spread out as fitting the same 100 positions to noise. Every
   orthonormal basis needs the same number of coefficients.
2. **There is nothing stable to align with.** A fixed basis helps only if it aligns with the signal's
   own directions. The leading eigenvector is **orthogonal across halves** (|cos| 0.000). No direction
   is estimable, so no basis can be validated as the right one.
3. **The structure that *is* measurable is a sharp partition, and low-pass filtering destroys sharp
   partitions.** A block indicator is maximally high-frequency on the feature graph; `exp(−tL)` damps
   exactly that. §12 did not fail from a mistuned `t` — it applied a smoothing operator to a
   discontinuity. (Running it backwards, `exp(+tL)`, is worse: condition number `exp(t·λ_max)`
   amplifies high-frequency **noise** preferentially, and §17.1 shows this problem needs *shrinkage*,
   the opposite of amplification. The stable analogue of "sharpen" is a diffusion model's **learned
   score**, where a prior supplies the missing high-frequency content — and the block indicator was
   the only candidate prior, which §17.3 just refuted.)
4. **And even the partition does not forecast** (§17.3), so a signed-graph clustering method — SPONGE
   and relatives, which are the correct tool for our **indefinite, 47+/52− signature** and a genuine
   criticism of §12's magnitude-only adjacency — would be well-suited machinery pointed at an object
   measured not to carry alpha. Its appeal was that it returns a *partition* (stable) rather than a
   *direction* (not), and §17.3 removes that appeal.

Where this family *would* apply is unchanged from §15.3: a **cross-sectional** panel, where an asset
graph has real community structure, coefficients are plausibly smooth within communities, and lead-lag
networks exist at all. That is a data problem, not a method problem.

## 18. The minimal model, built end to end — and what QLIKE keeps

Everything above is scored as OOS R² in the transformed sqrt space. The repo's production metric is
**QLIKE on raw RV**, reached through the Duan-smearing reconstruction — and no result in this study
had ever been pushed through it. `analysis/minimal_model.py` builds the study's concluding
recommendation as a runnable model and scores every ingredient on both scoreboards, with a machinery
control (QLIKE of the truth against itself = 0 exactly) and a negative control (a circularly-shifted
copy of the final signal, which must not improve anything).

The model: **Stage A** HAR ridge (27 columns, 250-day rolling, refit every bar) + **Stage B** one
ridge on the Stage-A residual — all 516 exog columns at α = 3000, with/without the 100
frozen-since-2006 products at their own α = 3e4. Final forecast = A + B. 218,934 OOS bars,
2007-02 → 2024-04, all on the §8-fixed panel.

| arm | resid R² | QLIKE | vs HAR | DM-t | 2020+ DM-t |
|---|---|---|---|---|---|
| HAR alone | — | 0.13275 | — | — | — |
| + dense ridge, monthly refit | +0.0204 | 0.12996 | −2.10% | +5.98 | +3.51 |
| + dense + products, monthly | +0.0274 | 0.12976 | −2.25% | +5.70 | +2.84 |
| **+ dense ridge, daily refit** | **+0.0377** | **0.12744** | **−3.99%** | **+11.36** | **+5.54** |
| + daily dense + product increment | +0.0372 | 0.12752 | −3.94% | +9.98 | +4.41 |
| noise control (shifted signal) | −0.0587 | 0.14218 | +7.11% | −8.67 | −4.95 |

Findings, in order of importance:

1. **The dense exog channel survives the reconstruction intact and emphatically.** −4.0% QLIKE at
   DM-t +11.4 (and +5.5 on 2020+ alone). The single daily-refit ridge on everything is worth twice
   the monthly-refit version — refit frequency matters more for the linear block than any
   architectural choice examined in this study.
2. **The product channel does NOT survive to QLIKE significance.** In sqrt space the products
   reproduce for the third time (+0.0070, DM-t +2.63 on the 516-column base — inside the
   pre-registered window). On QLIKE the same increment is worth −0.00020 at **DM-t +0.52**, and
   summed onto the daily-refit dense stage it is slightly *negative* (−0.21). The pre-registration
   explicitly declared this an open question with no prediction; the answer is that the gain lives
   in bars QLIKE down-weights. Squared-error in sqrt space weights quiet bars far more than QLIKE
   (which is variance-ratio-based) does, so a gain concentrated in ordinary bars dilutes through the
   reconstruction. It also has a named fragility: on **2018-02-06 (Volmageddon)** the product block
   forecasts +2.2..+3.4σ into a residual that spikes +1.8σ and reverses to −2.8σ — the composed
   daily+increment arm fails `prediction_health` on that single bar (12.6× vs the 10× gate).
3. **Both gate mis-specifications are recorded.** The noise gate first demanded "shifted signal ≈
   no-op"; a shifted signal is added variance with zero covariance and *must* hurt (it scored
   +7.1%, almost exactly the −var(s)/var(e) arithmetic). The fraud condition is noise *improving*,
   and it does not. `prediction_health` was first fed the level forecast against a mean-one target;
   it is calibrated for mean-zero residual signals and is applied to those.

**Interim deliverable (superseded by §18.1):** HAR + the daily-refit dense ridge, with the products
failing to transfer to QLIKE at monthly refit.

### 18.1 The products refit daily — the verdict reverses, and the cadence was the whole story

Point 2 above scored the product block at the **monthly** coefficient-refit cadence it inherited from
§7's selection studies — where monthly was the *selection* cadence under test, and "static beats
dynamic" was a claim about selection, never about coefficients. The untested cell was the frozen
products with **daily**-refit coefficients, as ordinary columns in the same rank-1 rolling solver as
the dense stage (block penalty imposed exactly by scaling the product columns by
`sqrt(α_lin/α_prod)`; per-window floored scale replaced by its causal whole-series analogue, a
trailing-250d floored sd). Pre-registered expectation: a *smaller* relative gain than the linear
block got from daily refit, since the product coefficients are shrunk 10× harder. Wrong again, in
the good direction:

| arm | resid R² | QLIKE | vs HAR | DM-t vs HAR | products' DM-t vs its linear stage |
|---|---|---|---|---|---|
| dense, daily refit | +0.0377 | 0.12744 | −3.99% | +11.36 | — |
| **+ products, daily refit** | **+0.0485** | **0.12651** | **−4.69%** | **+13.45** | **+3.56** |
| (+ products, monthly refit) | +0.0274 | 0.12976 | −2.25% | +5.70 | +0.52 |

Daily coefficient refit **grows** the product increment (+0.0108 vs +0.0070 in sqrt space, DM-t
+3.42) **and carries it through the reconstruction** (ΔQLIKE −0.00093, DM-t +3.56; 2020+ whole-model
DM-t +6.04). It also passes `prediction_health` at 4.7× — the causal rolling scale tames the
Volmageddon bar that the per-window scale let through at 9.4–12.6×. Two coherent mechanisms, both
consistent with earlier measurements rather than new stories: the selected products are
predominantly **fast** pairs (§17.2's `1×1` / `1×25` concentration), and fast features are exactly
where month-stale coefficients cost the most; and QLIKE's weight sits on high-vol bars, where fresh
coefficients matter most.

So §18's point 2 is corrected: **the products' QLIKE failure was an artifact of the inherited refit
cadence, not of the reconstruction.** The study's own "refit frequency is the biggest lever" finding
applied to its own product protocol.

**The deliverable.** Still two ridges: **HAR (27 cols, refit per bar) + one daily-refit ridge on 616
columns** — all 516 exog at α = 3000 plus the 100 frozen-since-2006 products at α = 3e4 via column
scaling. QLIKE **0.12651** vs HAR's 0.13275 (−4.7%, DM-t +13.5; 2020+ −5.0%, DM-t +6.0), residual
R² +0.0485 — the best model of the study on both scoreboards. The multiplicity caveat is the usual
one: this arm was prompted by a reviewer question and scored on the shared OOS span; its margin
(+3.56 full span, +6.04 on 2020+) is well clear of the thresholds that survived Romano-Wolf earlier,
but a formal pass over the §18 family is the remaining hygiene step.

### 18.2 The form's motion decomposed — the first low-dimensional object that survived

§18.1 established that the frozen form's *coefficients* move (daily vs monthly refit is worth +0.004
of residual R²). The last geometry question is whether that motion is itself low-dimensional:
``B_t ≈ g_t · B̄`` (one scalar dial on a frozen shape — rank-1 motion) or 100 idiosyncratic drifts
(dense-weak extended to the dynamics). `--stage gain` decomposes it with a causal 21-day rolling
scalar ``g_t`` fitted to the monthly-coefficient increment, 1 df, window fixed at the refit cadence,
not tuned. Pre-registered read: ≥ 2/3 gap closure = rank-1; ≤ 1/3 = unstructured; prediction was
unstructured, from the study's unbroken record of concentration claims dying.

| arm | resid R² | vs previous |
|---|---|---|
| A — monthly coefficients, no dial | +0.0372 | — |
| C0 — + constant gear (causal mean, 1.27) | +0.0314 | DM-t **−1.55** |
| C — + moving 21d dial (1 df) | +0.0431 | DM-t +1.70 vs C0 |
| B — full daily refit (616 coefficients) | +0.0485 | DM-t +1.48 vs C |

**Gap closure: 52% from one scalar — and its constant part is harmful.** The verdict is MIXED, the
first outcome in this study that is not a clean death for structure:

* The dial is a real process: mean 1.27, sd 0.92, daily AR(1) +0.961, and its correlation with the
  vol state is **+0.05**: this is *not* the vol axis wearing a costume. ⚠️ **§18.4 corrects the
  "slow and persistent" reading of that AR(1)**: a trailing 21-day window of pure white noise has
  daily AR(1) = 20/21 ≈ 0.952 mechanically, so 0.961 is barely above the overlap floor. At honest
  (non-overlapping) resolution the dial's month-over-month persistence is ~0.
* Its **level is destructive** (C0 loses to A at −1.55): gearing the frozen form up by its average
  1.27 statically over-amplifies exactly when it shouldn't. All of the dial's value is in its
  motion. This kills the mundane explanation ("the first-window ridge just under-fits the products")
  and is what makes the object genuinely dynamic.
* The other 48% of the freshness gain needs the full coefficient vector — dense-weak still holds for
  that half.

Honesty about strength: the *total* (B vs its linear stage) is the QLIKE-confirmed fact (+3.56); the
decomposition's internal splits sit at DM-t 1.5–1.8 individually and are directional, not decisive,
on a heavily reused OOS span. And the deliverable does not change — full daily refit *contains* the
dial implicitly and is simpler than dial machinery. What §18.2 buys is understanding: what daily
refit is paying for is, in roughly equal parts, one slow scalar intensity on the interaction channel
(the object §13 hunted on the wrong channel with an estimator that failed its own control) and
genuine high-dimensional coefficient freshness.

The geometry ledger, final form: static shape objects — spectrum, leading eigenvector, blocks,
7-dim rotation axis — all dead under powered tests. Surviving: the curvature itself (``x'Bx``,
QLIKE-confirmed), two small additive state axes (§17.1), and now one persistent scalar dial on the
form's motion, orthogonal to vol, worth about half the freshness gain at 1 df.

### 18.3 What drives the dial — mostly nothing nameable, with one coherent weak signature

`--stage driver`, hardened by §13's failure: ten pre-registered candidates with named mechanisms;
daily sampling; levels tested with 63-day Newey-West **and** non-overlapping 21-day changes (immune
to the smooth-vs-smooth spurious-regression trap); a circular-shift null per candidate; Bonferroni
over the ten; and a **power calibration** — a synthetic driver at true correlation 0.30 with matched
AR(1), which the machinery detects overwhelmingly (levels NW-t +7.1, changes r +0.36, p < 0.001). So
a moderate driver, had one existed, would have been unmissable.

| candidate | levels NW-t | z vs shift null | changes r (p) |
|---|---|---|---|
| **vrp** (implied − realized) | **−3.27** | **+4.3** | +0.03 (0.63) |
| **term_structure** (vix3m − vix) | **+3.24** | **+3.2** | +0.06 (0.32) |
| **vix_level** | **−2.75** | **+3.1** | +0.00 (0.95) |
| vvix | +2.05 | +2.2 | +0.10 (0.11) |
| mkt_ret_13d | +2.41 | +2.1 | +0.10 (0.12) |
| vol_level, resid_vol, liq_stress | \|t\| < 1.6 | < 1.4 | — |
| sentiment, opex_proximity | ~0 | — | — |

Joint R² of all ten: **0.046** against a shift-null of 0.016 (z +3.2).

Three conclusions, in decreasing confidence:

1. **The dial is ~95% its own latent process.** Every obvious state variable together explains under
   5% of its variance. The two hunches with the strongest priors both died cleanly: the expiry
   cycle (my read of the ~17-day half-life) at t −0.03, and sentiment at +0.20. Vol level is
   confirmed orthogonal (+0.05 raw).
2. **The nameable 5% is one coherent regime signature, not five separate facts:** the dial gears
   **up** in calm, contango, low-VRP, rallying markets and **down** into stress. The five
   levels-significant candidates are all facets of that one state, and the yearly path agrees —
   dial lows in 2008 (+0.70), 2018 (+0.95), 2020 (+1.01); highs in 2009 (+1.87), 2017 (+1.67),
   2024 (+1.97). Note the sign: the interaction form is *most* trustworthy in quiet regimes, which
   is the opposite of "interactions capture crisis nonlinearity".
3. **The signature is levels-only** — every changes test is null (best p_bonf = 1.0) against a
   power control that detects 0.30 easily. So calm-regime co-trending is established;
   month-to-month causation is not. It cannot presently be used as a *faster* dial estimator, and
   given the power control, the honest read is that no moderate-strength monthly driver exists
   among these ten.

The dial also never averages negative in any year (min +0.70) — the interaction channel, like the
linear one (§3's 0/207 months), is always-on with varying intensity. Dense-weak's signature
extends to the one low-dimensional object the study found: real, persistent, and driven by nothing
you can name — which is what a diffuse aggregate of many small causes looks like.

### 18.4 Is the dial forecastable? No beyond days — and half of it was never a forecast at all

`--stage memory`, at honest resolution: 217 **non-overlapping** 21-day block estimates, with a
split-half reliability correction (each block's two disjoint halves estimate the same dial value, so
their correlation across blocks measures signal share — the §6 attenuation move).

| quantity | value |
|---|---|
| split-half corr → full-block reliability | +0.157 → **0.272** |
| estimation-noise share of a monthly estimate | **≥ 73%** (lower bound: fast true motion is conflated into "noise") |
| observed ρ(1 month) | **+0.057 ± 0.068** — zero |
| reliability-corrected ρ_true(1 month) | +0.21 (wide) |
| ceiling for any history-based 1-month forecast | ~**4%** of true dial variance |
| block-change autocorrelation | **−0.472** (differenced-noise signature is −0.5) |

First, a correction to §18.2's characterization: the daily AR(1) of +0.961 that read as "slow and
persistent" is an **overlap artifact** — a trailing 21-day window of white noise is mechanically
AR(1) ≈ 0.952. At non-overlapping resolution the dial has essentially no month-over-month memory,
consistent with §18.3's null changes tests. So the dial is **not forecastable at monthly horizon or
beyond**: no persistence, no drivers, ceiling ~4%.

How can §18.2's trailing dial then be worth anything? Because most of its value never required
forecasting, and the constant-gear paradox is the tell. If the dial were noise around its mean, the
causal-mean gear (1.27) would have helped; it *hurt* (−1.55). Resolution, measured:

    corr(g_b, log var(inc)_b) = −0.375        corr(g_b, log var(e)_b) = −0.030
    pooled (variance-weighted) slope = +0.49   equal-weighted mean g_b = +1.32

**The dial is strongly anti-correlated with the form's own output variance** — and not with the
residual's, so this is a property of the signal, not target heteroskedasticity. When the frozen form
fires loudly, its per-unit reliability collapses (pooled slope 0.49); when it whispers, each unit is
worth ~1.3×. A constant 1.27 over-gears exactly the loud, squared-error-dominant periods — hence the
paradox — while the trailing estimate tracks the anti-correlation. This also re-reads §18.3: stress
is when vol-built products fire loudest, so the "calm-regime signature" in the VIX complex is
plausibly this one observable wearing regime clothing, and it re-reads the Volmageddon bar (§18.1)
as the concavity's worst case: the form's largest shout, trusted linearly.

So the dial decomposes into three parts with different epistemic status:

1. **An observable concavity** (~the anti-correlation above): the optimal response to the form's
   output is sublinear in its magnitude. Needs *no forecasting* — `var(inc)` is computable from
   features in real time. This is the one genuinely exploitable remainder on this panel: a
   pre-registered concave response (e.g. gearing ∝ a + b·log trailing-var) tested on a fresh
   protocol. Not run here — the response family is a free parameter and the OOS span is exhausted
   enough (§10) that it deserves its own registered test.
2. **A fast latent wiggle**, trackable at ~days by the adjacent trailing window (the rest of
   §18.2's 52%), decaying within a month.
3. **Nothing at monthly horizon+** — unforecastable and driverless.

Which closes the geometry arc the way this study keeps closing things: the one low-dimensional
object that survived turns out, on inspection, to be mostly *signal heteroskedasticity plus noise* —
the concavity of trust in a shouting model, not a hidden state of the market. Dense-but-weak, all
the way down.

## 19. Closing synthesis: the nonlinear channel vs the trees, and the story's final shape

**Is there an actual improvement on linear dense-but-weak? Yes — one, production-metric-confirmed.**
The frozen products at daily refit are the only thing in this study that beats the dense linear
ridge on QLIKE: 0.12744 → 0.12651 (increment DM-t +3.56; sqrt-space +0.0108 at +3.42; whole model
vs HAR +13.5, and +6.0 on 2020+ alone). Everything else that beat dense linear did so only in sqrt
space (vol blend, state axes) or failed a powered test. Relative size: the nonlinear channel is
~29% on top of the linear one (0.0108/0.0377), or ~22% of the total exog channel.

**Is it what the trees found? Structurally, almost certainly — and now with a mechanism per piece.**
The intraday-regime campaign's decomposition of the tuned EBM's edge (enet 0.12516 → +HAR×late-day
0.12457 → +6 regimes 0.12453 → EBM 0.12414) split it as ~58–62% the clock-regime sign-flip and
**~38% "genuinely nonlinear/higher-order, pending EBM-vs-XGB to say if pairwise"**. The
correspondence, term by term:

* the 58–62% clock-regime part is **already in our Stage A** (the `har_ma_w × open/close` columns
  were materialized upstream after that campaign);
* the ~38% remainder matches the **pairwise product channel** — an EBM is a GA2M, main effects plus
  pairwise interaction functions, and pairwise products are the parametric skeleton of exactly that;
* the products' need for **daily refit** matches the trees' walk-forward refits;
* and §18.4's **concavity** (per-unit trust collapsing when the form shouts) is precisely what a
  tree's bounded, piecewise-constant leaves provide *for free* — a leaf cannot extrapolate, so a
  tree is structurally immune to the Volmageddon +3.4σ shout that our linear products produced.

So the working identification: **the tree's unexplained 38% ≈ pairwise products + saturation**, and
we now hold both pieces in closed form. Two honesty notes: the QLIKE *levels* are not comparable
across the studies (tw1000 slim cell and a different diurnal protocol vs our tw250 fixed panel), so
this is a structural identification, not a numerical equality; and the decisive check — trees vs
`aug_daily` on the same fixed panel and span — is a single head-to-head that has never been run
(that campaign's own EBM-vs-XGB pairwise test was also left pending).

**What it means for dense-but-weak: the story bifurcates into two channels with mirror-image
geometry.**

| | linear channel | interaction channel |
|---|---|---|
| in features | **dense** (all 246; PR 117; top-5 loses at −7.2) | **sparse** (100 pairs, frozen since 2006) |
| in time | **stable** (0/207 negative months; selection churn 2%) | **fast** (daily refit +0.004; dial memory < 1 month) |
| per-unit strength | weak | weak (dial ≥73% noise, 95% driverless) |
| how it is earned | aggregate across features, shrink hard | aggregate across time: refit fast, saturate response |

"Dense but weak" was the correct answer to the campaign's original question — for the marginal,
linear structure. The nonlinear channel is its transpose: sparse-but-fast. What unifies them is the
*weak* part: in both channels the exploitable object is a diffuse aggregate of many small effects —
across features in one, across time in the other — with no nameable driver, no stable geometry, and
no concentration that survives a powered test. The playbook that earns both is the same three verbs,
which are exactly what a gradient-boosted tree does natively: **collect everything, refresh
constantly, saturate the response.** The two-ridge model is that playbook in closed form, minus the
saturation — which is the one pre-registered test this study leaves on the table.

### 19.1 The geometry, formalized: it is a metric — the information metric, and only partially applied

Asked directly whether the surviving structure is "a metric in the Riemannian sense", the answer
decomposes into one identity, one measured exponent, and one negative result.

**The identity.** QLIKE is exactly twice the KL divergence between mean-zero Gaussians differing in
variance, ``QLIKE(σ̂², σ²) = σ²/σ̂² − log(σ²/σ̂²) − 1``, whose second-order expansion is
``½(log σ² − log σ̂²)²`` — the squared geodesic distance in the **Fisher–Rao metric on the variance
manifold** (``ds = dσ/σ``). The production loss *is* an information-geometric squared distance;
log-vol is its geodesic normal coordinate; and the repo's transform stack — sqrt/log semantic
transforms, diurnal division, ratio-based smearing — is a change of charts that flattens this
metric. Every §8/§16.4 data defect was a **chart singularity**: a coordinate change (per-slot
divisor, reference IQR) degenerating to zero. This is established structure, not conjecture, and it
is already fully exploited — it is *why* the transform stack looks the way it does.

**The measured exponent.** If §18.4's dial were a *pure* metric rescaling — measure the form's
output in units of its own local noise — the elasticity of the dial to the form's variance would be
−1. Measured across the 199 positive-dial blocks: **d log g / d log var(inc) = −0.357 ± 0.046**.
One third of the way from raw units (0) to full SNR units (−1). Equivalently, the optimal
block-level response amplitude scales as (form amplitude)^0.28 — concave, between logarithm and cube
root — and gearing by the dial halves the cross-block amplitude dispersion (CV 1.27 → 0.68) without
equalizing it, as a pure metric would. The deviation from −1 is itself information: the form's
loudness is not pure noise — louder does mean more signal, just at less than half the raw rate.
This also pins down §18.4's "pre-registered concave response" test to a **zero-free-parameter**
form: gearing ∝ trailing-var^(−0.36), fixed here descriptively before any forecast is scored.

**The negative result, restated in metric language.** Feature space itself carries no exploitable
metric beyond diagonal whitening (the rolling robust scale — a time-varying, position-independent,
diagonal metric). Every attempt to give it richer geometry was an attempt to install an off-diagonal
or curved metric — a graph Laplacian *is* a metric choice on the feature graph, its heat kernel the
corresponding diffusion; the covariance-similarity and interaction-strength graphs were candidate
metrics; the spectral analysis of B asked for the metric's principal directions — and every one died
under a powered test (§§12, 13, 17.2–17.3).

So the final formal statement: **the signal has no geometry; the noise does, and the only structure
that ever paid — at every level of the stack — is the noise's metric pulled back onto everything
else.** Target: Fisher–Rao on the variance cone (QLIKE, the transforms). Clock: per-slot division.
Features: diagonal whitening. Form output: the dial, a partial (−0.36 of the full −1) inverse-noise
rescaling. The one genuinely *position-dependent* — curved — statement with empirical support is the
concavity itself: the effective metric on the form's output contracts in its tails. The geometry of
trust is curved; the geometry of the signal is flat. (If the panel ever goes cross-sectional, the
variance manifold becomes the SPD cone, whose affine-invariant metric is genuinely negatively curved
— the information geometry stops being scalar and this framing becomes load-bearing rather than
descriptive.)

### 19.2 The zero-free-parameter concave response test — gate FAILED; the deliverable already contains it

`--stage concave`. The exponent re-estimated on ≤2019 blocks only comes out **−0.351 ± 0.049**
against the full-span −0.357 — the descriptive relationship replicates almost exactly across the
split, so the concavity is real as a *description*. The forecast test, everything frozen before
scoring (gearing `w_t = (v_t/ref_t)^β`, `v_t` trailing-21d increment variance, `ref_t` causal
expanding median, no fitted constant anywhere):

| test | span | ΔR² (sqrt) | DM | ΔQLIKE | DM |
|---|---|---|---|---|---|
| **PRIMARY** geared vs ungeared monthly increment | **2020+** | +0.00208 | **+1.31** | −0.00065 | +1.32 |
| secondary: geared vs plain **daily-refit** increment | 2020+ | −0.00002 | −0.05 | −0.00012 | +1.05 |
| (primary, full span — descriptive) | full | +0.00779 | +1.77 | −0.00064 | +2.81 |

**The pre-registered gate (one-sided 1.645, sqrt, 2020+) fails at +1.31.** By this study's own
rules that is the end of the claim: no addition to the model. The reading is not "the concavity is
fake" — the exponent replicates, every span and metric is positive for the monthly arm, and the
full-span QLIKE sits at +2.81 — it is that **the concavity is not additive to the deliverable**.
The secondary arm says why: on top of daily refit the gearing is exactly zero (−0.05). Daily
coefficient refit already harvests the observable concavity at ~days lag, which is precisely what
§18.2's decomposition implied (the trailing dial and the daily solver capture the same observable
component). If a production system were ever stuck at monthly refit for compute reasons, the
gearing is a cheap +2.8-QLIKE-t substitute; with daily refit it is redundant.

So the panel's ledger closes with nothing left on the table: the two-ridge daily-refit model
contains the linear channel, the product channel, the trackable part of the dial, and the
observable concavity. What remains unharvested is, by measurement: the 48% of the freshness gain
that is idiosyncratic coefficient motion (irreducible without new information), a dial that is
unforecastable beyond days, and the cross-sectional data this document has now recommended three
times.

### 19.3 Cashing the principle at two more levels — one marginal pass, one fail

`analysis/noise_metric.py`, pre-registered gates at QLIKE DM > 2:

* **Conditional smearing** (the noise metric on the *reconstruction*): causal per-slot expanding
  error variance instead of one global constant. HAR arm +0.00014 QLIKE, DM +1.75 full-span /
  **+2.01 on 2020+**; deliverable +1.43 / **+2.24**. Direction right everywhere, magnitude tiny;
  passes the gate only on the 2020+ block. A real but very small calibration improvement —
  worth shipping since it is free and cannot lose, not worth celebrating.
* **QLIKE-aligned WLS** (the noise metric on the *estimation loss*): stage B refit with per-bar
  weights 1/HAR-level². The predicted signature appeared on the full span — sqrt-space R² worsens
  (−1.92) while QLIKE improves (+1.21) — but 2020+ flips negative (−1.13). **Gate fails.** The
  sqrt chart evidently absorbs most of the loss-metric misalignment already.

Consistent with §19.2: the two-ridge deliverable already sits close to the metric-optimal point of
this stack; the remaining direct applications of the principle at the observation level are
exhausted. The un-run items of the program — the parameter-space metric (per-coefficient Kalman
gains, the principled attack on the 48% idiosyncratic freshness) and target-precision weighting by
`numobs` — are recorded in `analysis/noise_metric.py`'s docstring lineage and §19.4.

### 19.4 Further readings of the information geometry — each one predicts a measured result

Frames, not new measurements; recorded because each makes at least one falsifiable statement that
already checks out against a number in this document.

1. **QLIKE is the Itakura–Saito divergence**, the Bregman divergence of φ = −log on the variance
   cone. Bregman ⇒ the optimal forecast is the conditional **mean of raw variance** — which is
   *why Duan smearing exists*: the model lives in the sqrt chart, the loss wants the m-chart mean,
   and the smear term is exactly the Jensen correction between them. The exponential-family
   structure gives three canonical charts — variance (m-flat, where means/mixtures live),
   precision 1/σ² (e-flat, where Bayesian information adds), log-variance (metric-flat, where
   errors are comparable). The repo's sqrt target is Box–Cox λ = ½, an intermediate chart; the
   dial's measured exponent −0.36 is another empirically-chosen intermediate representation. The
   data keeps electing charts partway along the α-family rather than at its named endpoints.
2. **Dual flatness ⇒ a Pythagorean theorem in model space**: divergence improvements from nested
   model extensions add exactly when the extensions are Fisher-orthogonal. The study measured this
   twice without naming it: §16.1's two gains additive to −4%, §17.1's two state axes additive to
   ~1%. "Stacking improvements works" is not luck — it is orthogonality, and the attribution
   tables were Pythagorean decompositions.
3. **Elicitability sorts the survivor list.** Every object that survived a powered test is an
   elicitable functional — conditional means, regression slopes, variances, the dial (a slope) —
   i.e. things with strictly consistent scoring rules. Every object that died is non-elicitable —
   eigenvectors, rotation axes, cluster memberships, argmax structure. The study's graveyard is
   the theorem's prediction.
4. **Identification failures are Cramér–Rao saturation.** The Fisher metric *is* the CR bound
   field; the "power arithmetic" (c/√K, null sd 0.404 on a 7-dim axis, the dial's 73% noise
   share) was CR accounting done empirically. The metric tells you in advance which objects the
   data's information volume can estimate; §§13–14's failures were computable before running.
5. **Item 4 of §19.3's program is "update in the e-chart"**: Kalman filtering in information form
   adds precisions — Bayesian updates are additive in natural parameters. Per-coefficient gains =
   doing parameter updates in the coordinates where information is additive.
6. **The tower of metrics has geometrically decaying information — which is why the stack
   terminates.** Vol is the noise metric of returns (R² 0.58); the exog structure is a correction
   to that metric (+0.05); the dial is the metric of that correction (forecastable ceiling ~4%,
   reliability 0.27); the next level is measurably nothing. Each derivative of the process loses
   an order of magnitude of predictability, so a two-ridge model is not a compromise — it is
   where the information runs out. Cross-sectionally the tower restarts with content: the variance
   cone becomes the SPD cone, genuinely negatively curved, where chart choices (arithmetic vs
   geodesic means, log-Euclidean vs affine-invariant) stop being conventions — the third
   independent argument that the next unit of progress is data, not method.

## 20. The cross-section was baked in — and its relational features pass the gate, with a dated edge

A user observation reframed §19's closing recommendation: the target is the MARKET's realized
variance and the exog set already carries the cross-section's *marginal* aggregates
(`sumret2_{ew,vw}stock`, bipower, semivariance families). What no linear span of those columns can
reach is the **relational** objects between market and stocks — ratios. The flagship is realized
average correlation, `RV / sumret2_vwstock`, which in §19.4's SPD language is the **top-eigenvalue
share of the return covariance matrix** — the first genuinely cross-sectional coordinate,
constructible from columns already in hand. Six ratio families × six HAR windows = 36 columns
(`analysis/cross_section.py`): realized correlation, ew/vw size tilt, market and average-stock jump
shares (bipower vs ret2), market and stock upside-semivariance shares. Mechanisms literature-named
before scoring; pre-registered gate: block increment over the daily-refit dense ridge, sqrt DM > 2.

| arm | ΔR² | DM sqrt | ΔQLIKE | DM QLIKE | 2020+ DM (sqrt/QL) |
|---|---|---|---|---|---|
| dense + xsec vs dense | +0.00175 | **+3.95** | −0.00021 | **+2.64** | **+0.24 / +0.26** |
| deliverable + xsec vs deliverable | +0.00168 | +3.94 | −0.00017 | +2.40 | +0.17 / +0.05 |

**Gate: PASS** — and the increment is nearly identical on top of the dense ridge and on top of the
full deliverable (+0.00175 vs +0.00168), i.e. orthogonal to the product channel: the Pythagorean
pattern a third time. Two honesty items, stated as loudly as the pass:

1. **The edge is dated.** On 2020+ alone the block is flat (DM ≈ +0.2). The pre-registered gate was
   the full-span statistic and it passes decisively, but the gain is concentrated pre-2020 — the
   same era-profile question §16.2's sentiment term was made to answer and this block, on present
   evidence, does not. Anyone adding these 36 columns should expect the historical average, not the
   recent one, and the honest forward-looking claim is "small or zero".
2. **One semantics caveat**: the market-level upside-share ratio sits in [0.35, 0.67] (consistent
   with the positive-semivariance reading of `pret2`); the stock-level one hits its clip at 2.0, so
   that reading is unverified for the `ewstock` aggregate. The feature is a well-defined bounded
   ratio either way.

Structurally this satisfies the study's own §15.3 closing advice ("new information, not new
method") from *inside* the panel, the same way OFI did: the information was present but not in the
model's span. And it connects to §19.4's tower argument the right way round — relational
cross-sectional coordinates widen **level 2** (more signal per bar), which is precisely what makes
higher floors of the tower estimable. A genuinely multi-asset panel remains the full version of
this move; these 36 columns are the largest fraction of it available for free.

### 19.5 Item 4's pilot: the gate fails, and the program closes

`analysis/noise_metric.py --stage kernel`. The 250-day flat window was the product block's last
untested inheritance (the same class as the refit cadence, whose correction produced §18.1), so the
pilot isolated kernel shape with everything else fixed: the 100 frozen products fit on the dense
stage's residual with a flat 250d rolling Gram vs an exponentially-forgotten Gram at halflife 21d —
pinned by §18.4's measured dial memory, not tuned — penalty normalized per effective observation.

| arm | resid R² | QLIKE | vs flat (sqrt DM) |
|---|---|---|---|
| two-stage, flat 250d | +0.04164 | 0.12595 | — |
| two-stage, **EWMA 21d** (primary) | +0.04287 | 0.12589 | **+0.79 — GATE FAILS** |
| two-stage, EWMA 63d (secondary observation) | +0.04426 | 0.12566 | +3.75 |
| deliverable (one-stage 616, for context) | **+0.04743** | **0.12562** | +2.60 |

Three readings, in order of epistemic weight:

1. **The pre-registered claim fails.** At the halflife the dial-memory measurement pinned, matched
   forgetting does not beat the flat window (+0.79). By this study's rules item 4 closes here.
2. **The unregistered observation**: 63d beats flat at +3.75 — kernel shape probably does matter,
   at a longer memory than the dial suggested. Recorded as what it is: an arm that won *after* the
   registered one lost, i.e. exactly the pattern §10 exists to stop anyone from claiming. Using it
   would need a fresh registered test on data this panel does not have left.
3. **Neither matters for the model**, because even the best kernel arm (+0.04426) loses to the
   shipped one-stage deliverable (+0.04743): the two-stage architecture costs more (−0.0032, DM
   +2.60 in the deliverable's favor) than the best kernel buys (+0.0026). A joint per-block
   forgetting implementation remains conceivable, but its plausible ceiling from these numbers is
   ~+0.001–0.002 over the deliverable — below the floor of what this exhausted span can still
   certify.

**The noise-metric program is now fully executed**: reconstruction (marginal pass, §19.3),
estimation loss (fail, §19.3), parameter space (fail-with-a-wrinkle, here), plus the two earlier
levels already in the deliverable. The two-ridge daily-refit model survives every principled attack
the framework generates, which is the strongest form of the §18 conclusion available: it is not
just good, it is the fixed point of its own design principle.

### 19.6 Design note: soft barriers for the boundary (a user proposal, adopted for next time)

The pipeline's guards (divisor floors, the §16.4 slot band, scale floors) are hard clips. The
principled continuous form, proposed in review: penalize ``r + 1/r = 2·cosh(log r)`` on the ratio
of an estimated divisor to its reference — the kernel of the **Generalized Inverse Gaussian**
prior, the natural conjugate boundary-repelling family on the positive cone. Nearly flat in the
healthy band, exponential wall at the boundary; the hard band is its degenerate limit. The clean
division of labor: **barriers for bounding parameters away from the boundary** (they leave healthy
estimates untouched), **the metric for shrinking noisy estimates toward a reference** (§17.1's
ladder — proportional by design); and never either one on raw data against a noisy proxy, where
``E[1/RV]`` is 3×-inflated at ``numobs = 3`` and infinite at 2. **Re-plumbed on review push-back, behind a flag** (`SOFT_GUARDS = False` default —
bit-identical off): every guard in `robust_transform`'s chain now has a smooth counterpart — a
saturating band in the log coordinate, a p-norm smooth max for the floors, a saturating winsoriser
— threaded through `soft_guards=`. The implementation's own equivalence audit earned its keep
immediately: the first cut used `tanh`, and the audit rejected it **by the barrier's stated
requirement** (leave healthy data untouched) — `tanh(0.5) = 0.462` distorts the mid-band by 7.6%
and dragged hard-vs-soft correlations on the guarded columns to 0.95. Replaced with the k-order
saturator `u(1+|u|^k)^{-1/k}` (identity to `O(u^{k+1})`, k = 6): correlations 0.990–0.9999, bulk
deviation p99 ≤ 0.46 sd, differences confined to the rows where the guards bind. The premise
deserves its own record: end-to-end gradient fitting **on this panel** is not justified — its
gains would sit below the ~0.002 floor this exhausted span can certify, and the tree record says
the flexible-model premium here was defect-robustness, not signal — so the flag ships OFF. It
exists so that the cross-sectional build, where capacity and information budget both grow, starts
differentiable instead of retrofitting it.

## 21. The two §19 leftovers, run — and the joint-fit question answered (2026-08-05)

Three pre-registered cells, run post-merge on a fresh rebuild of the fixed panel
(`analysis/refit_cadence.py`; outputs `results/alpha_manifestation/refit_cadence.csv`,
`saturation_insurance.csv`). Rebuild note: the local `data/` feeds now carry 546 exog columns
against the study's 516, so absolute levels shift slightly (the daily deliverable scores 0.12700
here vs 0.12651 in §18.1); the HAR baseline reproduces to four decimals (R² 0.5772) and the daily
product increment reproduces exactly (DM-t +3.42), and every §20 comparison is bar-for-bar within
this one panel.

### 21.1 Refit cadence below daily: the curve keeps paying, and the prediction was wrong again

The §18.1 deliverable (646 columns here), same rank-1 rolling solver at five cadences:

| refit_every | cadence | resid R² | QLIKE | vs daily | DM-t | 2020+ DM-t |
|---|---|---|---|---|---|---|
| 1008 | monthly | +0.0092 | 0.13075 | +0.00376 | −5.39 | −2.59 |
| 48 | daily | +0.0428 | 0.12700 | — | — | — |
| 24 | half-day | +0.0452 | 0.12660 | −0.00040 | **+4.94** | +3.80 |
| 8 | hourly | +0.0480 | 0.12635 | −0.00065 | **+5.61** | +2.31 |
| 1 | every bar | +0.0512 | 0.12619 | −0.00081 | **+4.90** | +1.02 |

**The pre-registered gate (refit-1 vs refit-48, QLIKE DM-t > 2) passes at +4.90**, against a
pre-registered expectation of diminishing returns — the study's "refit frequency is the biggest
lever" finding extends below the daily cadence it was measured at. The curve is monotone in both
metrics all the way to every-bar. Two qualifications, stated with the result: the marginal step
from hourly to every-bar is thin (−0.00016) and its 2020+ significance fades (+1.02, vs +2.31 at
hourly and +3.80 at half-day), so **hourly refit is the robust point of the curve**; and the gain
attributes almost entirely to the **linear channel** (−0.00072 of the −0.00081; the product
increment's cadence effect is −0.00009 at DM-t +0.61). The products need daily refit (§18.1) and
nothing faster; the dense linear block wants its coefficients as fresh as you can afford. Compute
is not a constraint: every-bar refit of 646 columns is one 646-column solve per 30-minute bar.

### 21.2 The joint group-ridge: cramming it into one solve is a wash

One solve on [backbone | exog | products] with the deliverable's exact per-block penalties
(1 / 3000 / 3e4, imposed by column scaling), daily refit — the fixed point the two-stage model's
single backfitting pass approximates:

| arm | resid R² | QLIKE | vs two-stage | DM-t |
|---|---|---|---|---|
| two-stage (aug_r48) | +0.0428 | 0.12700 | — | — |
| joint group-ridge (680 cols) | +0.0438 | 0.12711 | +0.00011 | −0.51 |

The two are statistically indistinguishable — the joint fit is a hair better in sqrt space and a
hair worse on QLIKE, both within noise. So sequential residualization costs nothing measurable in
accuracy, and the answer to "does one ridge work?" is: yes, provided the per-block penalties are
kept; the two-stage form earns its place on compute (the backbone can refit every bar at 34
columns instead of dragging 680 through per-bar solves), on the attribution discipline this study
runs on, and on the data-defect firewall — not on forecast skill. Notably the joint arm's backbone
refits only daily and still ties the two-stage whose backbone refits per bar, consistent with
§21.1's finding that freshness matters most for the exog block.


**Independent replication + the theorem (merged from the concurrent session).** The same joint-vs-
sequential question was run simultaneously in a sibling session on the original §18 fixed panel
(516 exog, `minimal_model.py --stage joint`, 643 columns, HAR block per-bar-refit reference):
deliverable +0.60634 / QLIKE 0.12651 vs joint +0.60649 / 0.12670 — sqrt DM **+0.16**, QLIKE −1.06.
Same verdict, two implementations, two panel builds. And the §19.5 asymmetry — why sequential cost
0.0058 *inside* stage B but nothing at the top level — has a theorem: **Frisch–Waugh–Lovell**.
Residual-on-second-block reproduces the joint fit when the first stage is OLS, and stage A's α = 1
shrinks by ~0.01%, i.e. it *is* effectively OLS, so the top-level split is FWL-protected. The
§19.5 pilot's first stage was shrunk at α = 3000, which breaks FWL and leaves mis-allocated shared
variance behind. One sentence: **sequential fitting is free after an unshrunk stage and expensive
after a shrunk one.** The deliverable sits, by inheritance rather than design, on the right side
of that line.

### 21.3 Saturation: REJECTED by its own gate — the tails are net-informative

The §19 "one verb the two-ridge lacks," tested as insurance: a causal tanh bound on the product
increment at k·(trailing 21d sd), k = 3 primary. Pre-registered rule: ship if the average QLIKE
cost is nil AND the worst-bar amplitude falls; reject if the average cost is significant.

| arm | QLIKE | dQL | DM-t | max\|inc\|/sd(e) | Volmageddon day dQL |
|---|---|---|---|---|---|
| unsaturated (aug_r1) | 0.12619 | — | — | 5.56 | — |
| tanh k=2 | 0.12641 | +0.00022 | −4.54 | 1.04 | −0.0294 |
| **tanh k=3** | 0.12633 | +0.00015 | **−4.00** | 1.44 | −0.0183 |
| tanh k=4 | 0.12629 | +0.00010 | −3.66 | 1.88 | −0.0118 |

**REJECT.** The amplitude condition passes emphatically (5.56σ → 1.44σ) and the named bar behaves
exactly as designed — the 2018-02-06 morning improves by −0.018 QLIKE under the clamp — but the
average premium is real: +0.00015 at DM-t −4.00, significant at every k and monotone in how hard
the bound bites. The product block's other shouts are, on net, informative; shaving them costs
more across nineteen years than the Volmageddon class of bar returns. This is §19.1's elasticity
made operational: the measured concavity of trust is −0.36 — sublinear, but far from the hard
bound a tanh (or a tree leaf) imposes. The right response to a loud form is partial discounting,
which every-bar refit already implements implicitly (§19.2); a hard cap overshoots. The tree
correspondence sharpens accordingly: bounded leaves are not free tail insurance on this channel —
at fast refit cadence they would cost measurable average skill, which further reduces what remains
of the trees' unexplained edge.

### 21.4 What §21 changes

The deliverable improves once more, by the same lever as §18.1: **HAR (34 cols, per bar) + one
ridge on 646 columns at hourly-or-faster refit** — QLIKE 0.12635–0.12619 vs 0.12700 at daily, the
increment linear-channel-driven and gate-passing. The two-stage-vs-joint question is settled as a
tie on accuracy. Saturation is off the table, rejected by a powered test rather than left as an
open verb; the playbook compresses to two verbs plus a penalty structure: **collect everything,
refresh constantly** — with saturation now measured as actively harmful where the coefficients are
fresh. The study's meta-pattern held one more time: both directional priors registered in this
section's docstring (diminishing cadence returns; joint slightly worse) were wrong in the details,
and the gates did the deciding.
## 22. The composed final model: one stage, everything, per bar — best numbers of the study

User-directed composition (`minimal_model.py --stage final`): ONE blockwise ridge, 679 columns in a
single rank-1 solver — 27-column backbone at effective α = 1, 516 exog at 3e3, the 36 §20
cross-section ratios at 3e3, the 100 frozen products at 3e4, all by column scaling — predicting y
directly, 250-day window, run at daily and per-bar cadence on the §18 fixed panel.

| arm | R² vs mean | QLIKE | vs deliverable (sqrt / QLIKE DM) |
|---|---|---|---|
| deliverable (2-stage, daily, no xsec) | +0.60634 | 0.12651 | — |
| one-stage daily (+ xsec) | +0.60696 | 0.12654 | small / flat |
| **one-stage per bar** | **+0.61176** | **0.12526** | **+7.37 / +5.64** |

**QLIKE 0.12526 vs HAR's 0.13275 (−5.6%) — the best model of the study on both scoreboards**, and
each margin decomposes exactly as the earlier sections predicted: the one-stage-vs-two-stage and
xsec increments are small (§21's tie, §20's dated edge — one-stage daily ≈ deliverable), and
essentially the entire improvement is **refit cadence**, replicating §21.1's sibling-panel finding
on this panel (per-bar vs daily: sqrt +5.48, QLIKE +5.17). The 2020+ caveat carries over too: the
sqrt gain holds late (+4.48) while the QLIKE cadence gain fades late (+1.38, matching the sibling's
+1.02 at every-bar), so hourly remains the robust cadence and per-bar the in-sample-best.

`prediction_health` on the residual-space signal: 5.0× for both one-stage arms — clean, and better
than the two-stage deliverable's 4.7× equivalent. (The stage's own printed "health FAIL" on all
three arms including the §18-passing deliverable was the §18.4 harness false alarm re-committed —
demeaned *level* forecasts fed to a check calibrated for residual signals; the correct object
passes. Recorded because making the same mistake twice, with the correction already documented, is
worth a line of shame.)

The final spec, one line: **y ~ ridge₍₂₅₀d, refit-per-bar₎ [backbone@1 | exog+xsec@3e3 |
products@3e4]** — collect everything, refresh constantly, grade the shrinkage by block. Nothing
else survived.

## 23. The manifold question: the state's linear frame is stable, its nonlinear shape is not — and the signal lives on the free coordinates

Prompted by the reading "there is a trajectory on a manifold here". The testable version: if the
state lives on a *curved* manifold rather than a linear 18-dim subspace, mid-spectrum PCs should be
partially quadratic functions of the top PCs (curvature coordinates). Measured — fit each PC 7–20
on the 21 quadratics of PCs 1–6 in the first half, evaluate in the second:

* curvature R² out-of-half: mostly **negative**, and violently so exactly where it matters —
  PC9 −3.15, PC17 −2.72, PC16 −1.86, PC11 −1.60 (max positive: +0.13). At n ≈ 109k per half and 21
  ridge-regularized coefficients, −3 is not overfitting noise (that would be −0.0002); it requires
  a STRONG in-half quadratic relation that then strongly reverses. The nonlinear relations among
  factors exist and are **unstable across halves**: the manifold's linear frame holds for 18 years,
  its curvature does not hold for nine.
* **corr(curvature stability, signal energy) = −0.53 (rank −0.59)**: the signal-carrying PCs
  (9, 11, 16) are precisely the ones whose relation to the dominant factors is least stable.

Reconciled with the §"map" result (the IC map on these same coordinates is split-half stable at
+0.62): the mid PCs' coupling **to vol** is stable while their nonlinear relation **to the top
factors** is not. So they are not curvature shadows of the big factors — they are **dynamically
autonomous coordinates**, and autonomy is exactly where the information is: a coordinate enslaved
to the top factors is redundant with the linear model; a free coordinate carries news. This also
finally *derives* the eigenvalue misalignment (variance ranks by common-factor loading; information
ranks by independence) and the γ = 1 failure.

So the trajectory-on-a-manifold picture, in its measured form: a fixed linear frame (stable),
carrying a state whose nonlinear self-geometry deforms (unstable), read by a response form whose
support is static and whose amplitudes move fast. The model was already the right instrument for
that object without knowing it: the HAR lags are a Takens delay-embedding of the vol coordinate,
per-bar refit is local linearization along the trajectory, and the dial is the loudness of the
path. The manifold's *shape* is part of the weather, not part of the map — which is the study's
one-line history of every static geometric object it buried.

### 23.1 Can the autonomous coordinates be exploited directly? No — the causal test inverts

The obvious exploit: measure autonomy causally (inside the first training window — fit the
factor-relations on its first half, evaluate on its second) and select the quadratics of the most
autonomous coordinates. Pre-registered with a falsification control: most-autonomous-7 ≥ 60% of the
IC gain, least-autonomous-7 ≤ 25%. Result, inverted on both:

| arm (28 cols each) | ΔR² | DM | % of IC-100 |
|---|---|---|---|
| most-autonomous 7 | −0.00236 | −0.60 | **−34%** |
| least-autonomous 7 | +0.00272 | **+3.38** | **+39%** |

The diagnosis is the study's oldest one — **timescale**. §23's signal-predicting autonomy is
*decadal* instability (relations that hold for years and shift across half-samples), measurable
only with a decade of hindsight. Autonomy measured on 6-month halves inside one window selects
coordinates whose factor-relations flip in *months* — i.e. noise-dominated coordinates, and their
quadratics hurt. The enslaved-7, being near-functions of the top factors, proxy top-factor
quadratics and land near the PC-6 arm (41%). So the free coordinates are: stable at the fitting
horizon, autonomous at the secular horizon — and no first-window statistic sees the second
property.

Verdict: autonomy is ex-post understanding, not an ex-ante selector. The channel stays banked by
the flat pool, and the meta-law of this study absorbs one more datapoint: IC selection ≈ flat
(n.s.), random < flat, eigenvalue-weighting ≪ flat, autonomy-selection < flat — every attempt to
be cleverer than *uniform over a generous span, shrunk hard* has now lost, including the one
derived from the study's own prettiest finding. (The cross-section is, once again, where this
would change: N assets give a contemporaneous estimate of coordinate autonomy without needing
decades of hindsight.)

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

# §16 — the §15.3 list. Run against the FIXED panel cache.
C=/path/to/scratchpad/fixed
ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage prep          # residual + bucket signals, fixed panel
ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage interactions  # item 5: no clip, voldemand in
ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage additivity    # item 1, attempt 1 (rejected wrapper)
ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage additivity2   # item 1: forecast combination
ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage additivity3   # item 1: the 2x2 attribution
ALPHA_PANEL_CACHE=$C python analysis/synthesis.py --stage sentiment     # item 2: fresh split, 1 df
python analysis/tail_fix.py --stage diagnose                 # item 4: raw/transform shape per column
python analysis/tail_fix.py --stage trace                    # item 4: one bar through the whole chain
python analysis/tail_fix.py --stage modes                    # item 4: modal share of every raw exog
python analysis/tail_fix.py --stage fix                      # item 4: the invariant ladder

# §17 — the geometry §14 stopped short of.
ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage axis    # frozen gamma, shrinkage ladder, 2020+
ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage clock   # intraday clock as a second axis
ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage form    # spectrum of the interaction form
ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage blocks  # window / bucket block concentration
ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage block_exploit  # is the block usable? (no)

# §18 — the minimal model, end to end, scored on QLIKE.
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage build   # stage-B walk-forwards (~12 min)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage daily   # products at DAILY refit (§18.1)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage verify  # QLIKE + controls + verdicts
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage gain    # rank-1 dial vs freshness (§18.2)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage driver  # what moves the dial (§18.3)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage memory  # is it forecastable (§18.4)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage concave # zero-free-param response test (§19.2)
ALPHA_PANEL_CACHE=$C python analysis/cross_section.py --stage build   # §20: relational cross-section features
ALPHA_PANEL_CACHE=$C python analysis/cross_section.py --stage score   # §20: gate test
ALPHA_PANEL_CACHE=$C python analysis/noise_metric.py --stage kernel   # §19.5: item-4 pilot (gate fails)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage joint    # §21: joint vs sequential (tie; FWL)
ALPHA_PANEL_CACHE=$C python analysis/minimal_model.py --stage final    # §22: one stage, everything, per bar
python analysis/pc_quadratics.py           # frame test: PC-quadratics, soft thresholds
python analysis/pc_quadratics.py --map     # where the signal lives in the factor grid
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
python analysis/composition.py --stage audit                   # §14.2 null the eigenstructure + replicate
python analysis/composition.py --stage axis_direct             # §14.3 bar-resolution axis, no bins
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
composition_shrinkage_ladder.csv, composition_audit.csv, axis_direct.csv,
axis_direct_summary.csv}`.
