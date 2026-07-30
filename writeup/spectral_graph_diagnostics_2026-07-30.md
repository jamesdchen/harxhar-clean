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

## Conclusions

1. **The residualizer is the first-order mistake.** Views of ridge residuals
   carry no usable neighbor structure (target smoothness at the null floor,
   kNN < constant). This closes the question of why spectral_knn < identity
   raw-view kNN in the run battery.
2. **The graph-construction hypothesis is half right.** The representation is
   decisive (it is where all signal enters), but the Laplacian filtering adds
   nothing even on the best graph — diffusion geometry is at best a lossy
   re-encoding of neighborhoods the base metric already gets right.
3. **Actionable model:** identity residualizer + multiscale (HAR-style
   1/8/48/240/960-bar means, per-coordinate standardized) representation +
   gaussian-weighted 25-NN. On this window's view targets it reaches
   OOS R² ≈ 0.58 vs the constant predictor. Next experiment: proper
   walk-forward of that model against the ridge baseline — that comparison
   decides candidate vs curiosity.

Artifacts: `results/spectral_graph_diagnostics/` (summary.csv,
energy_profiles.csv/.png, neighbor_swap.csv, config.json). Reproduce:
`python diag_spectral_graph.py --stride 2 --n-perm 10 --anchors 0.5`.
Regime check on a late window: add `--anchors 0.5,0.95`.
