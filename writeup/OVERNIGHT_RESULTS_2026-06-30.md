# Overnight results — entity embeddings (#1) & long-history retrieval (#2), both gated

Both CTR/SFV leads tested through the OOS-robust harness (`src/evaluation/feature_cv.py`). **Both reject.** The
result sharpens the floor conclusion and is itself a clean demonstration of why the gate's placebo control matters.

## #1 — entity embeddings → pivoted to the information-block gate

**Critical finding: the cache has no entity/ticker IDs.** The cross-section is *pre-aggregated* into `ewstock`/
`vwstock` moments (378 features, single market-level series, `bucket=all_buckets`). True per-entity embeddings
would need the raw (time, stock) panel — which is *data-to-buy*, not in this cache. So #1 pivoted to the closest
feasible information-bearing test: does the **already-present information block** — microstructure (spread,
turnover, order-flow buy/sell, effspread), sentiment (stocktwits), cross-section (ew/vw stock moments), 196
features — add incremental OOS value to the HAR regime?

| candidate (vs HAR base) | target | gate |
|---|---|---|
| PCA-25(info) | r1 (beyond linear base) | **reject** (+0.00141, hurts) |
| PCA-25(info) | r2 (beyond HAR-d8) | **reject** (+0.00032, z=−2.1) |
| learned emb-8(info) | r1 (beyond linear base) | **reject** (−0.00091 but fails placebo z=+0.3, CI⊅0) |

The learned embedding *looks* like a small win (−0.00091, replicates 0.58/0.75) but **fails the placebo** —
indistinguishable from a shuffled-embedding null. **The microstructure + sentiment + cross-sectional information
is already fully exploited by the linear base (enetreg2); no nonlinear residual the regime can grab.**

## #2 — long-history regime retrieval (analog forecasting, CTR SIM/ETA)

For each close bar, retrieve K most-similar past regimes (causal kNN), attend to their realized residual.

| retrieval | gate |
|---|---|
| mean, HAR-state | **reject** (+0.00074, hurts) |
| mean, HAR+micro-state | **reject** (−0.00095, replicates 0.77/0.80, but **fails placebo** z=+0.7) |
| attention-weighted (SIM/ETA), HAR, K∈{30,100} | **reject** (−0.00010, fails placebo z=−1.1) |

The HAR+micro mean-retrieval is the sharpest near-miss: CI excludes 0 *and* replicates 4/5 folds — it would have
looked like a win — **but it does not beat a shuffled-retrieval null.** d8 already captures the regime analogs.

## What this means

1. **Both leads reject** across every variant (embedding/PCA, mean/attention, multiple states/K).
2. **The gate's placebo control was decisive** — two candidates (HAR+micro retrieval, emb-8) replicated with a
   negative CI and would have shipped without it. This is exactly the local-mirage failure mode the harness exists
   to stop, caught live.
3. **The floor conclusion sharpens:** not only is the *price-only* residual at the floor — the *already-present*
   microstructure/sentiment/cross-section information is fully exploited too (by the linear base). The remaining
   lever is **genuinely-new information** at finer granularity than the aggregated ew/vw moments (per-name auction
   imbalance / GEX / OFI / order-book) — which must be sourced; it is not latent in this cache.

## Also tonight — ω fully pinned

Fine grid around 0.8 (byte-identical `ggrid` re-average): `w80=w82=0.12022`, neighbors 0.12023 → a flat basin.
**Deploy fixed ω≈0.80–0.82 → 0.12022.** ω is frozen; sub-0.0001 is unresolvable (noise).

Scripts: `scratchpad/{retr,retr2,info,overnight}.py`. All gated through `feature_cv`. Honest bottom line: the
model + the data-in-hand are at the floor; the deliverable is the **gate** + the **data-to-buy spec**.

## Follow-up: longer HAR, term-structure differences, scale-attention — all reject

- **HAR construction VERIFIED:** `har_ma_W = rolling_mean(RV, window=W)` to machine precision (corr 1.0000,
  maxabsdiff ~1e-16 for W=5..3125; EWMA-span only 0.78–0.98). (First test mistakenly used `ewm` — confounded.)
- **Longer HAR, principled (rolling MA, matched):** `rollMA[15625]` rejects (+0.00230, z=−2.5); `[15625,78125]`
  rejects harder (+0.00951, z=−6.4); vs r2 (+0.00162, z=−4.2). `corr(rollMA15625, har_3125)=0.330` — genuinely
  distinct yet informationless. **Vol memory saturates ~3125; extending the ladder reaches noise, not signal.**
- **HAR term-structure differences** (`har_5−har_1`, ...): reject (−0.00058, fails placebo) — the diffs are
  *linear combos of the levels* a flexible model already forms → redundant. The levels are a basis, not a feature.
- **Attention over the scale dimension (values=diffs):** reject (placebo-level). **Longer-state retrieval:** reject.

The HAR basis (1→3125) plus the term structure it spans is **complete** for vol memory on this data. The gate
caught two more near-misses (HAR-diffs, scale-attn looked like −0.00058 helps, failed placebo). Confirms the floor.
