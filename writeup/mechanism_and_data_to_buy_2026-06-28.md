# Intraday vol regime — mechanism + the data-to-buy case (2026-06-28)

Synthesis of the edge-features / legibility / decomposition arc. The deployed model is at the **modeling
floor**; the residual edge is an intraday **auction/session-transition regime**, and further QLIKE gains need
**direct microstructure data, not more model**. The single sharpest piece of evidence: one engineered
microstructure feature (`cumrv`) is **~27× more valuable per feature** than the 359 raw exogenous columns.

All QLIKE are full-OOS Duan-smeared on the slim `all_buckets` cell, `resid_subset` arm. Best deployed
stack = **0.12035** (enetreg2 base + within-day vol-path SHAPE + rolling-relative regime + XGB-d8 + EBM regime@h16–19).

---

## 1. Bottom line

| stage | QLIKE |
|---|---|
| plain enet | 0.12516 |
| `enetreg2` base (HAR×{open,close} + cumrv×close, fixed-α single-pass) | 0.12314 |
| + XGB-d8 global (Hero A) | 0.12129 |
| + untuned EBM regime@h16–19 (Hero B) | 0.12050 |
| + rolling-relative vol innovations | 0.12046 |
| **+ path-shape ⊥ regime stack (`shaperel`, best)** | **0.12035** |

The net gain from the later threads is **−0.00015** (the path-shape facet), on top of the −0.00264 base→Hero-B.
The durable outputs are the *understanding* (below) and the *data-to-buy case*, not the 4th-decimal QLIKE.

## 1b. The new-to-tree edge — and why one-off engineering is near its ceiling

Trees are invariant to monotone transforms of the current feature row, so the only features that beat a
fitted tree are **functionals of the sequence/history** the row doesn't contain. Mapped:
- **Rolling-relative vol regime** (today's vol vs its recent same-slot history) — a low-dim regime factor,
  reached identically via HAR or exog (they *overlap*, don't stack) ≈ −0.00007. Saturated.
- **Within-day path SHAPE** (when the day's vol happened, front/back-loaded; the time-channel low-order
  path-signature terms) — **orthogonal** to the regime factor; the two *stack* to the 0.12035 best.
- **Directional path** (cumret, overnight gap) — NULL: the close effect is a vol-regime/timing phenomenon,
  not a price-reversal one.

**kNN / relevance regression (Cartea et al., ssrn-4652980)**: naive K-mean over-adds variance and hurts; the
*proper* large-K local-linear ridge works (−0.00018 on the close leftover, shuffle-placebo-clean) — but is
**dominated by the EBM regime stage** (−0.00079), which is itself a stronger flexible close-leftover model.
So relevance regression is validated as a method but not deployable here. The **model class that would subsume
all of these** (path + relevance) is sequence-attention (PatchTST / Transformer: self-attention = learned kNN;
sequence input = path) or a signature-linear block — but both are data-hungry at ~195k rows; the hand-engineered
path features are the data-efficient, *legible* proxy, and they measure the residual sequence-signal as small
(5th-decimal after the tree). That smallness is itself the strongest case that the next lever is **data, not model**.

## 2. The mechanism — a session-edge vol regime

HAR (a pure vol-persistence model) **over-extrapolates at the session edges**, and the correction is
clock-anchored to the auction / session transitions (09:30 ET open; 16:00 close + after-hours, hours 16–19):

- **The close DAMPS short-horizon vol persistence.** Read directly off the legible *unpenalized-HAR* base
  (`harunpen`, undistorted by L1): `har_ma_5×close` coefficient = **−0.05** — high recent vol → the close
  forecast is pulled *below* HAR's extrapolation. This is the documented late-day sign-flip, now visible as a
  single coefficient. `har_ma_1×open` = **−0.045** (the open damps the 1-bar). The EBM regime shapes show the
  same monotone-down `har_ma_5×close` signature.
- **The intraday vol PATH matters at the close.** `cumrv×close` (abnormal realized vol accumulated through the
  day, gated to the close) is the single biggest engineered edge: base **0.12436 → 0.12314 (−0.00122)**, sign-
  stable. After a high-vol day the close/AH residual mean-reverts below HAR. Mechanistically isolated: the edge
  is the **sqrt (vol-scale) accumulation** — `cumsum(sqrt(adj RV))` recovers 97.5% of the gap vs raw-variance
  accumulation; it is the variance-stabilization (cap-mixture) that lets the cumsum measure vol *breadth*
  rather than be dominated by one extreme bar.
- **The open is a discrete auction/gap event, not a smooth path.** A symmetric overnight-cumrv-at-open feature
  is null (+0.00001) — the open regime does not linearize the way the close does.

## 3. Why we are at the floor (the decomposition)

**FWL block attribution** of the linear base (`fwl_attribution.py`; full fit reproduces 0.12314):

| block | unique contribution (Type-III, net of all others) |
|---|---|
| HAR (6) | **−0.02208** |
| EXOG (359) | −0.00449 |
| REGIME (HAR×{open,close}) | −0.00225 |
| CUMRV (1) | −0.00122 |

HAR alone explains **96% of the explainable variance** (R² 0.594 of 0.617 full). Everything engineered on top
is small — and most of it is **redundant with the tree**.

**The tree-subsumption law (the organizing principle).** A gradient-boosted tree is exactly invariant to
monotone/threshold transforms of an existing feature. So any linear-base gain that is monotone-shaped gets
absorbed by the d8 tree. Confirmed repeatedly:
- `har5rank` is ⅔ static curvature (a low-vol spline bend on `har_ma_5`) → redundant with the tree;
- implied-vol **magnitude** beats rank by −0.00013 *at the linear base* but is +0.00009 *through the tree*;
- the banded-HAR reparam is legible but penalty-entangled (not a model lever).

**The only thing that beat the tree was a feature it cannot reconstruct** — a *history-dependent* rolling-
relative vol innovation (today's vol vs its recent same-slot distribution), which is not a function of the
current feature row. That is the −0.00008 that produced the 0.12046 best.

⇒ The lever is no longer model capacity or feature re-encoding. It is **new information**.

## 4. The data-to-buy case

Two independent signals that the remaining edge is *data*, not *model*:

1. **Per-feature efficiency.** `cumrv` (one engineered intraday-path feature) = −0.00122 unique; the 359 raw
   exog combined = −0.00449. That is **~27× more value per feature** for a single, mechanism-targeted
   engineered feature than for piling on raw exogenous series. Direct measurements of the mechanism dominate.
2. **The mechanism is clock-anchored at the auction and the close/AH** — exactly where direct microstructure
   data lives, and exactly what the model is currently inferring *indirectly* from price-derived moments.

**Highest-value data to acquire (in priority order):**
- **Auction order imbalance** (opening & closing auction imbalance feeds) — the direct measurement of the
  09:30 / 16:00 transitions the regime is built on.
- **Dealer gamma / GEX** (gamma exposure) — drives end-of-day pinning / mean-reversion at the close, the exact
  `cumrv×close` damping signature.
- **Order-flow imbalance / signed volume (OFI)** at the bar level — the microstructure pressure the realized
  moments only proxy.

These are the direct observables of the auction/session-transition mechanism the model now reconstructs from
realized variance alone.

## 5. Exhausted levers (do not re-try on this data)

All tested and rejected (redundant with the tree, null, or penalty artifacts): online-λ, alternating backfit,
cumrv ratio, physical `cumrv_real`, OPEX/calendar, blind 200-trial Optuna (lost to the curated sweep), the
EBM-feature linear distill, `har²×close` convexity, overnight-cumrv-at-open, banded-HAR canon (+ scaled),
implied-vol magnitude through the tree. Model capacity and monotone feature re-encoding are done.

## 6. Open (small, no new data)

The one remaining *new-to-tree* direction without new data: **rolling-relative exog innovations** at a longer
window / different normalization (rolling-z, ratio, vol-of-vol) than the cache's short (~20-obs) `diurnal_rank`
— the exog analog of the har rolling-innovation. Expected small. (Encoding-axis sweep on har —
rank-Gauss vs robust-z vs ratio vs 48-slot — was in flight at writing.)

**Recommendation:** stop optimizing this OOS QLIKE (multiple-testing budget spent) and pursue the
auction/GEX/OFI data acquisition; the modeling stack (0.12046) is at its floor.
