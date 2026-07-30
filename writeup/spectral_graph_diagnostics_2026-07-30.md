# Spectral-kNN graph diagnostics — 2026-07-30

**Question.** The spectral_knn ablations showed plain kNN on raw lag views
(identity residualizer, no embedding) beating the full ridge-residualize →
spectral-embed → kNN pipeline. Hypothesis under test: the value/failure is
determined by the *graph construction* (representation + metric + edge
weights), not by the Laplacian eigenmaps filtering — and whether the
eigenmaps helps is equivalent to whether the forecast target is a smooth
function on the constructed graph.

**Method.** `diag_spectral_graph.py` — no walk-forward. One frozen 500-day
training window (2013-11-06..2015-09-24), views/targets built exactly as
`MultiStageBacktest` does at a refit step (W=960, stride 2 → 11,520 views),
data prep identical to `spectral_knn.run`. Grid: {resid, raw} view sources ×
{raw960 delay vector, recency-weighted (hl=240), multiscale means over
1/8/48/240/960 bars} representations × {binary, self-tuned heat} edge
weights, k_graph=10.

Three measurements per cell:

1. **Spectral energy profile** — fraction of view-target variance captured by
   the bottom-d eigenvectors of the normalized Laplacian, vs permutation null.
2. **Rayleigh quotient** f'Lf/f'f vs the same null (z-score).
3. **Neighbor-swap** — chronological 80/20 split; gaussian-weighted 25-NN
   forecast from (a) base-metric neighbors vs (b) the pipeline's actual
   mechanism (Laplacian eigenmap of train views + heat-kernel Nyström +
   25-NN in embedding space, d ∈ {8, 16}), same train pool.

## Results

### Residual views: no structure for any graph to find

| representation | d8 captured (null ≈ 0.08%) | RQ z | swap: const / base / embed_d8 |
|---|---|---|---|
| raw960     | 0.18–0.20% | +3.4/+7.2 | 0.0861 / **0.0870** / 0.0877–0.0881 |
| recw       | 0.11–0.26% | +3.3/+3.8 | 0.0861 / **0.0873** / 0.0872–0.0873 |
| multiscale | 0.81–0.94% | +3.9/+4.9 | 0.0861 / **0.0869** / 0.0883 |

Statistically nonzero smoothness (z ≈ +3–7 with 11.5k views), economically
nil. Base-metric kNN on residual views forecasts *worse than a constant*;
embeddings worse still. Ridge residualization removes the geometry the
neighbor search needs — no representation or weighting rescues it.

### Raw views: signal is large, and it lives in the representation

| representation | d8 captured | RQ (z) | swap: const / base / embed_d8 / embed_d16 |
|---|---|---|---|
| raw960     |  4.0–4.2% | 0.48 (+157) | 0.285 / **0.191** / 0.240 / 0.221 |
| recw       |  5.5–5.8% | 0.48 (+141) | 0.285 / **0.172** / 0.203–0.210 / 0.207 |
| multiscale | 48.4–50.0% | 0.40 (+105) | 0.285 / **0.119** / 0.126–0.129 / 0.124 |

* Representation ordering (multiscale ≫ recw > raw960) holds on both
  measurements, with large margins: multiscale cuts base-metric forecast MSE
  by 38% vs the raw delay vector and concentrates ~50% of target variance in
  the bottom 8 eigenvectors (vs 4% for raw960).
* **The embedding never beats the base metric** — not in one of the 12 cells.
  Best case (multiscale, d=16) it costs ~5% MSE; worst case (raw960, d=8)
  ~26%. The smoke-test exception (multiscale+heat+d16 edging base) did not
  survive full view density.
* Binary vs heat-kernel weighting: differences are noise-level throughout.
  Edge-magnitude loss is not the binding constraint; the representation is.

## Extended grid (same window/targets): W sweep + HAR-feature cells

Delay-window sweep and HAR-feature representations, identical target set
(all raw-target cells comparable to the table above; const = 0.2853):

| representation | dim | d8 captured | swap: base / embed_d8 / embed_d16 |
|---|---|---|---|
| raw960 (W=960)  | 960 |  4.0% | 0.1915 / 0.2395–0.2418 / 0.2212 |
| rawW240 (W=240) | 240 | 36.2% | **0.1677** / 0.1871–0.1878 / 0.1859–0.1877 |
| rawW48 (W=48)   |  48 | 46.2% | **0.1438** / 0.1539–0.1549 / 0.1575–0.1590 |
| recw (hl=240)   | 960 |  5.5% | 0.1716 / 0.2034–0.2105 / 0.2071 |
| multiscale      |   5 | 48–50% | **0.1189** / 0.1257–0.1293 / 0.1244–0.1258 |
| har (X rows)    |  27 |  2.6–3.2% | **0.1286** / 0.1631–0.1693 / 0.1514–0.1596 |

* **Smaller W monotonically improves the plain delay metric**
  (0.1915 → 0.1677 → 0.1438): the equal-weight 960-bar window mostly
  accumulates distance noise. A hard 1-day cutoff even beats the soft
  half-life-5-day weighting — but the best delay window still loses to
  multiscale by 17%, confirming that multi-horizon means (not the raw recent
  path) are the right coordinates.
* **kNN on the HAR features (0.1286) ≈ the multiscale result (0.1189)** —
  i.e. the winning diagnostic model is essentially the repo's existing plain
  kNN. The residual gap plausibly comes from calendar/interaction dims
  diluting the distance or the coarser HAR lag ladder (1/5/25/125/625/3125
  vs 1/8/48/240/960); a feature-subset ablation would settle it.
* **Spectrally embedding the HAR features is clearly negative**: 0.1286 →
  0.1631 (d=8) / 0.1514 (d=16). Compressing an already-low-dimensional,
  well-scaled space through a k=10 graph only loses information.
* **Residual targets stay dead in every new cell.** Best case
  (resid/rawW240) beats the constant by 0.2% — noise-level; the
  har→residual "stacking geometry" cell (does X-neighborhood structure
  predict what ridge misses?) is negative too (0.0882 vs const 0.0861).

## Scale-combination cells (same targets; can multiscale + small-W merge?)

| representation | dim | d8 captured | swap: base / embed_d8 / embed_d16 |
|---|---|---|---|
| multiscale (ref)   |  5 | 48–50% | **0.1189** / 0.1257–0.1293 / 0.1244–0.1258 |
| dyadic (11 levels) | 11 | 49–50% | 0.1222 / 0.1226–0.1235 / **0.1212–0.1213** |
| msdetail (level+trends) | 11 | 13.7–14.4% | 0.1671 / 0.1937–0.1944 / 0.1943–0.2002 |
| ms_plus_w48 (concat)    | 53 | 52.5% | 0.1230 / 0.1201–0.1246 / 0.1268–0.1281 |

* **The small-W signal is subsumed.** Neither a denser level ladder (dyadic),
  nor explicit recent-path coordinates (ms_plus_w48), nor scale-local trend
  coordinates (msdetail) improve on the 5-level multiscale. W=48's strong
  showing was just "levels at scales ≤ 48 without long-scale noise" — already
  inside the ladder. There is no additional shape/trajectory signal worth its
  coordinate noise.
* **msdetail is a useful negative**: informationally it equals dyadic up to a
  linear transform, but standardizing the trend coordinates re-weights the
  metric away from the level direction and costs 37% MSE. The metric's
  implicit weighting — not the information content — is what kNN sees.
* On these well-conditioned low-dim graphs the embedding finally goes
  ~neutral (dyadic d16 and ms_plus_w48 d8 marginally beat their own base
  metric), but never beats the multiscale base — consistent with "a good
  representation leaves the eigenmaps nothing to add."
* Residual-target versions of all three: still at/below the constant.

## kNN vs linear (diag_knn_vs_linear.py, same window/targets/split)

The reference the earlier grids never included (const = 0.2853):

| model | MSE | | model | MSE |
|---|---|---|---|---|
| **ridge_multiscale** | **0.0849** | | knn_multiscale_k25 | 0.1189 |
| ridge_har | 0.0915 | | knn_multiscale_k5 / k100 | 0.1183 / 0.1341 |
| loclin_multiscale_k25 | 0.1124 | | knn_prognostic_k25 | 0.1197 |
| loclin_har_k100 | 0.1158 | | knn_har_betaweighted_k25 | 0.1254 |

* **Plain ridge on the 5 multiscale coordinates beats every neighborhood
  method by ~25–30%** (and beats ridge on the full 27-dim HAR+calendar rows).
  The kNN family was never ahead of the linear baseline on this protocol —
  its apparent strength (R² 0.58 vs const) is a strict subset of ridge's
  (R² 0.70 on the same coordinates).
* **No exploitable nonlinearity found.** Prognostic-score kNN (free-form link
  on ridge's own index) loses to ridge; |beta|-weighted metrics don't help;
  this matches the dead har→residual cell. The relationship from multi-
  horizon levels to next-bar vol is, to the resolution of this window,
  linear — local averaging only adds variance.
* **Local-linear closes a third of the gap** (0.1124 vs 0.1189 at k=25) by
  fixing part of the locally-constant extrapolation bias, but a per-query
  ridge on 25 neighbors is just a noisier version of the global ridge.
* **The analogues are real, but analogy is the wrong estimator.** Selected
  neighbors are temporally spread (median |Δt| ≈ 11,190 bars ≈ 233 days —
  not persistence in disguise), yet matching-and-averaging is dominated by
  global regression when the signal is linear.
* **Shared failure mode, and the only structure left on the table:** both
  models compress toward the middle — overpredicting the bottom deciles
  (+0.15–0.19 bias) and badly underpredicting the top decile (bias −0.54
  kNN / −0.43 ridge; decile-9 MSE 0.74 / 0.48). kNN's inability to
  extrapolate makes it strictly worse exactly at spikes. Whatever improves
  this model class next lives in the top decile — jump/threshold features,
  regime interactions (cf. the open/close session-edge work), or asymmetric
  loss — not in neighborhood geometry.

## Alt-data blocks (diag_altdata.py): what exog adds beyond own-history

Nested-ridge increments over the multiscale base, train-only standardization,
causal features. Two windows; test tail of the first is Feb–Jun 2020 (COVID).

**COVID window** (base 0.1010, top-decile 0.5093, spike bias −0.442):

| block | ΔMSE | Δtop-decile | top bias |
|---|---|---|---|
| +vix_level | **−2.3%** | **−7.5%** | **−0.401** |
| +vix_struct (slope/vvix/VRP) | −1.8% | −0.9% | −0.442 |
| +jump / +flow / +sent / +breadth | ≈0 | ≈0 | — |
| +voldemand | **+1.0%** | **+3.3%** | −0.461 |
| +ALL | −0.6% | −4.5% | −0.428 |

**Calm window** (base 0.0796): vix_level −1.1%, vix_struct −1.0%, ALL −1.5%;
all top-decile deltas ≤ ±1%; everything else ≈ 0.

Findings:

1. **The implied-vol complex is the only alt block with real incremental
   signal**, worth ~1–2% MSE in both regimes — and in the crisis window the
   raw VIX *level* cuts top-decile (spike) error by 7.5% and shrinks the
   spike underprediction bias from −0.442 to −0.401. First thing in these
   diagnostics to move the spike bucket at all. Mechanism: implied vol is
   forward-looking and updates instantly on news; every own-history
   coordinate lags by construction.
2. **Structure beat level overall but not on spikes** — the VRP's slow
   22-day realized leg goes haywire exactly during a fast crash. A
   fast-realized-leg VRP or an inverted-term-structure *gate* (threshold
   encoding) is the indicated follow-up, not more linear columns.
3. **voldemand is actively harmful in a linear encoding** (+1.0% / +3.3%),
   independently confirming the EBM campaign's conclusion that its signal is
   interaction-only. The right test is gated/interaction encodings
   (voldemand × clock-regime), not inclusion as levels.
4. Jump fraction, flow imbalance, sentiment, EW/VW breadth: nulls in this
   protocol. ALL-blocks underperforms vix_level alone in the crisis window —
   noise blocks dilute; per-block L1 selection (the enet-survivor machinery)
   is the right integration path, consistent with campaign experience.

## all_features bucket embedding (diag_embedding_allfeat.py)

Laplacian eigenmap of the full 297-col exog matrix (41 raw series ×
HAR lags + target HAR + calendar, sweep_all_features construction), same
window/graph as the other embedding panels. d8 captured 4.9% (null 0.08%),
d16 11.2%, d64 39%. The geometry
(`writeup/figures/diag_embedding_allfeatures.png`) is thin closed loops with
sessions fully mixed: because most exog are daily-stepped (ffilled), bars
cluster by day and days chain temporally — **the embedding parametrizes
calendar time, not market state**. Its modest target smoothness is
autocorrelation-in-time (persistence rediscovered), not analogue structure;
signal accrues only slowly with d (harmonics of the time curve).

## Regime-discoverer variant (diag_regimes.py): gates close here too

Spectral clustering of an 11-dim slow exog state (train-window graph →
eigenvector k-means → multinomial-logit distillation → soft-membership
gates on the multiscale base), against the honest nulls. Nested-ridge MSE
(overall / top-decile):

| model | COVID window | calm window |
|---|---|---|
| M0 multiscale, no gates | **0.1010** / 0.509 | 0.0796 / 0.346 |
| + clock gates | 0.1030 / 0.526 | **0.0770** / 0.339 |
| + VIX-inversion hand gate | 0.1096 / 0.576 | 0.0773 / 0.350 |
| + GMM regime gates | 0.1051 / 0.526 | 0.0768 / 0.340 |
| + spectral regime gates | 0.1307 / 0.668 | 0.0781 / 0.351 |

Mechanics were healthy (median regime run ~16–19 bars, distill acc 0.96–0.98)
— the failure is in the economics:

1. **In the calm window** the validated clock gates reproduce (−3.3%) and
   GMM regimes merely match them; spectral regimes trail both. The graph
   layer adds nothing over k-means-style clustering, and discovered regimes
   add nothing over the free clock regime.
2. **In the COVID window every gate hurts, spectral worst** (+29% MSE, +31%
   top-decile). Gated models multiply features by state memberships; in an
   unprecedented state the memberships extrapolate (the spectral partition
   included a 233-sample micro-cluster whose gate exploded OOS) and the
   gated coefficients misfire exactly at the spikes. Regime gating is most
   dangerous precisely where the residual value is.
3. This replicates, with discovered regimes, the campaign's finding that
   vol-state-anchored regime interactions fail while clock-anchored ones
   hold: the clock never suffers distribution shift.

## Conclusions

1. **The residualizer is the first-order mistake.** Views of ridge residuals
   carry no usable neighbor structure (target smoothness at the null floor,
   kNN < constant). This closes the question of why spectral_knn < identity
   raw-view kNN in the run battery.
2. **The graph-construction hypothesis is half right.** The representation is
   decisive (it is where all signal enters), but the Laplacian filtering adds
   nothing even on the best graph — diffusion geometry is at best a lossy
   re-encoding of neighborhoods the base metric already gets right.
3. **[Superseded by the kNN-vs-linear section]** The multiscale-kNN model
   that won the representation search (OOS R² ≈ 0.58 vs const) is itself
   dominated by plain ridge on the same 5 coordinates (R² ≈ 0.70), with no
   detectable nonlinearity left for any neighborhood method to harvest.
   Verdict on the whole direction: close spectral-kNN; the surviving lead is
   ridge on multiscale coordinates (ridge_multiscale 0.0849 < ridge_har
   0.0915 on this protocol — worth a walk-forward check as a feature-set
   simplification), and the open problem is the shared top-decile spike
   underprediction, which no model in this family addresses.

Artifacts: `results/spectral_graph_diagnostics/` (summary.csv,
energy_profiles.csv/.png, neighbor_swap.csv, config.json). Reproduce:
`python diag_spectral_graph.py --stride 2 --n-perm 10 --anchors 0.5`.
Regime check on a late window: add `--anchors 0.5,0.95`.
