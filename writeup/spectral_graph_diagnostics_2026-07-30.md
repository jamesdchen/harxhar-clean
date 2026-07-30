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
