# Data-to-buy spec — the only lever not at the floor

## Why (the conclusion that forces this)

The model and the data-in-hand are at the **modeling floor** (~0.12022 full-OOS QLIKE). Every price-derived
avenue has been tested *through the gate* (purged walk-forward CV + bagging + circular-shift placebo + a
significance gate) and **rejects**: longer HAR memory (saturates ~scale 3125 → noise), alternative MA
aggregators (geometric is captured, the rest are subsumed by the sqrt-vol + dense rolling-HAR base), the entire
spike/extreme axis (distinct from HAR but signal-less as main effect *and* interaction — vol persistence is
regime-independent of spikiness), long-history retrieval, entity embeddings, and even the **already-present**
microstructure/sentiment/cross-section block (the aggregated ew/vw moments + stocktwits are fully exploited by
the linear base). The forecastable content of vol is its **level**; the residual edge is real but tiny and
already-arbitraged. **The lever is information** — new, finer-granularity data not latent in this cache.

## What the science named (the mechanism → the data)

The tuned tree's residual is an **intraday auction / session-transition regime**: HAR volatility-persistence
**sign-flips at the session edges** (sharp negative corr at the open 09:30 and the close/after-hours 16–19,
peak −0.21; clock-anchored, not vol-state-anchored). The d8 dissection: ~55% additive / 9% pairwise / **36%
higher-order and diffuse**, the interactions being **HAR × {stocktwits sentiment, attention, returns,
VIX-term}**; dealer-gamma is a weak (~20%) modulator. So the residual is the equilibrium footprint of
**auction / MOC liquidity + dealer hedging** — strongest exactly where trading costs are highest.

## The data-to-buy (ranked by mechanism strength)

| # | data | why (the named mechanism) | granularity vs what we have |
|---|---|---|---|
| 1 | **Auction imbalance (MOC/MOO order imbalance)** | the regime *is* the close/open auction footprint (the sign-flip at the session edges) — this is the direct cause | per-name, at the 15:50/09:28 imbalance prints — nothing like it in the cache |
| 2 | **Order-flow imbalance (OFI) / signed depth** | the buy/sell pressure that drives the session-edge move; the cache only has *aggregated* ew/vw turnover (gate-exhausted) | per-name, intraday, signed — finer than the aggregates |
| 3 | **Dealer gamma exposure (GEX)** | the ~20% gamma modulator of the regime; a cleaner per-name/index GEX is a real conditioner | per-name or index dealer-gamma, daily — we have none |
| 4 | **Richer / alternative sentiment & attention** | stocktwits is in the higher-order interactions but its *aggregate* is exhausted; finer or alternative sources may carry incremental | per-name, higher-frequency, multi-source |
| 5 | **Order-book depth / quoted-spread microstructure** | the cost surface the regime tracks; cache has only ew/vw spread aggregates | per-name, intraday book |

**Granularity principle:** the new data must carry information *not in the aggregated ew/vw moments* — the gate
proved those are fully exploited. Per-name, intraday, signed/directional is where the un-captured information is.

## How the harness gates it (de-risks the buy)

Do **not** buy a full feed on a hypothesis. The acquisition workflow is gate-first:

1. **Sample** — obtain a trial slice of the candidate feed (a few names / a short period).
2. **Feature** — build the candidate feature(s) on the sample, aligned to the close-window residual.
3. **Gate** — run `src/evaluation/feature_cv.score_feature` + `significance_gate` over the real base residual
   `r1 = y − ridge_oos`: it must clear `CI<0 ∧ replicates ∧ beats the circular-shift placebo`. (This is the
   instrument that has already killed ~half a dozen price-derived near-false-positives.)
4. **Go/no-go** — only a gate-passing feature justifies buying the full feed + a cluster deployment run.

The deliverable of the whole arc is therefore **mechanism + this map + the gate** — a tested shopping list, not
a 4th-decimal QLIKE. Auction imbalance (#1) is the highest-conviction first buy: it is the named cause of the
edge, and it is exactly the kind of finer-granularity, directional data the price-only cache cannot contain.
