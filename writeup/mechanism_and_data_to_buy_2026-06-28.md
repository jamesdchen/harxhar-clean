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

## 6. Post-floor rigor pass + the definitive path-lever-death test (2026-06-28, later)

After 0.12035 the work turned to (a) hardening the features against magic-number artifacts and (b) the
*systematic* test of whether the path lever has any life left.

**Rigor pass — no ad-hoc constants.** Removed: `±8` safety clips (one was *biting* — flattening the
extreme-demand weeks, i.e. the signal); `std>1e-9` constant-column drops → scale-free `min<max`; `+1e-9`
div-0 epsilons → `np.divide(where=denom>0)`. `sig_volskew`'s 3rd-standardized-moment (needed a 1-bar² floor)
→ **Bowley quartile skew** ∈[−1,1] by construction (no floor). The `robustz`/`ratio` rolling-innovation
rejection was *confounded*: a `min_periods=5` rolling-IQR **collapse** manufactured a |z|≈57 on `har_ma_125`
(6-obs window = divide-by-degenerate); fixed by matching the rank-Gauss warmup (`max(20,win//20)`), after
which robustz still loses to rank-Gauss on *genuine* heavy tails (rejection stands, now honest).
**Lesson: verify a number's source before concluding; an L1-zeroed feature ≠ no signal — check scale and
force it past L1 (`FORCE_COLS`) before calling it null.**

**Gate-window sweep — the 16-19 close window is empirically near-optimal**, not an un-tested guess:

| window | full-OOS QLIKE (enetreg2 + d8 + EBM) | what changes |
|---|---|---|
| **16-19** | **0.12050** | baseline |
| 15-19 | 0.12051 | +hour 15 (neutral) |
| 17-19 | 0.12064 | −hour 16 (close auction) |
| 16-20 | 0.12068 | +dead AH hour 20 |
| 16-18 | 0.12092 | −hour 19 (AH) — worst |

Both edges carry real signal — hour 19/AH most (3× the cost of dropping hour 16), hour 20 dead.

**The log-signature path-lever-death test (the decisive experiment).** A tree/EBM is order-blind: it sees the
current row, not the chronological *path*. Every `cum*` feature is one hand-picked path functional. The
systematic generalization is the **log-signature** (rough-path theory) — the *universal* basis whose
**antisymmetric** coordinates (Lévy areas at level 2, brackets at level 3) ARE the strict chronological-order
content. We ran the full truncated log-sig (channels τ / cumret / cumrv, level 3 = 14 coords + 3 QV terms),
**forced past the L1 gate** into the tree/EBM (`FORCE_COLS` — else the antisymmetric coords, which have no
linear main effect, get L1-zeroed = a fake null):

| rung | logsig (FORCE) | reference |
|---|---|---|
| +d8 | 0.12117 | (non-robust wiggle) |
| **+EBM** | **0.12059** | base+EBM 0.12050 / rel+EBM 0.12046 — **worse** |

**Null/harmful.** The complete universal antisymmetric path basis carries no robust OOS signal. ⇒ **the close
edge has NO strict chronological-order content — it is a pure state/level regime, fully captured.** The path
lever is **closed** (confirms the saturation of the hand-picked path-shape features and the null `sig_area_rv`
Lévy area). Hand-picked signature-timing (`sig`/Bowley) and the within-week Friday block both came back null
under the same scrutiny.

## 7. OFI proxy is dead — sharpening the data-to-buy case

The turnover-derived OFI (within-day net buy/sell imbalance, bounded [−1,1] by construction) is **null**:
base-alone 0.12313, and +EBM slightly *harmful* even when forced past L1 (the scale-invariant tree test).
The original "null" was a *fake* null — two artifacts (the resid_subset L1-survivor gating + a scale handicap:
raw OFI std 0.03 = smallest feature in the matrix, vs cumrv 13.3; L1 penalizes |β| scale-blind) — both fixed
and generalized into the reusable `FORCE_COLS` tool. But the underlying feature is genuinely dead: **cheap
turnover imbalance does not proxy the close-auction CROSS imbalance.** This sharpens §4: the data to buy is the
**actual auction-imbalance feed (NYSE Order Imbalances / Nasdaq NOII indicative imbalance, published pre-16:00)**;
a turnover-constructed OFI is not a substitute.

## 8. The best linear base + the DL handoff

For the pivot to deep-learning feature engineering, the linear base was locked at its best. The four
orthogonal, downstream-subsumed linear improvers (help the enet base; the tree/EBM absorb them) **stack**:

| component | base-alone | Δ vs enetreg2 |
|---|---|---|
| `enetreg2` | 0.12314 | — |
| + `har5rank` (rank-space close HAR-damping) | 0.12293 | −0.00021 |
| + `ivmag` (VIX magnitude, not rank) | 0.12301 | −0.00013 |
| + `sig` (vol-arrival path moments, Bowley) | 0.12302 | −0.00012 |
| + `exogrel` (long-window rolling-rel exog) | 0.12305 | −0.00009 |
| **`enetreg2_linbest` (all 4)** | **0.12266** | **−0.00048** (~87% of the orthogonal sum) |

**`linbest` = 0.12266 is the locked linear foundation; DL builds the residual `y − linbest`.** (The `rel`-family
is correctly excluded — it *hurts* the base, it's a downstream-only lever.)

**DL direction, constrained by the evidence:** the close edge is a **state/level regime with no order content**
(log-sig null). So DL capacity should NOT go to sequence / self-attention-over-the-path architectures — the
universal path basis is provably empty here. The value, if any, is in **richer state/level nonlinearities**
(high-order interactions the EBM's additive+pairwise form can't reach) or **new data** (the auction-imbalance
feed). A final pure-QLIKE dig pins the current-pipeline ceiling first: retuned Hero A (global XGB sweep on
`linbest`) + Hero B **pure power** (tuned XGB regime stage, dropping the EBM interpretability constraint, via
`REGIME_MODEL=xgb`). [result pending — in flight at writing.]

## 9. Exhausted levers (do not re-try on this data)

All tested and rejected: online-λ, alternating backfit, cumrv ratio, physical `cumrv_real`, OPEX/calendar,
blind 200-trial Optuna (lost to the curated sweep), the EBM-feature linear distill, `har²×close` convexity,
overnight-cumrv-at-open, banded-HAR canon (+scaled), implied-vol magnitude through the tree, **the higher-order
path-signature block (saturated), the within-week Friday block (null), the turnover-OFI proxy (null), and the
full level-3 log-signature antisymmetric basis (null — the path lever is closed).** Model capacity, monotone
feature re-encoding, and the path/order channel are done.

**Recommendation:** the modeling stack is at its floor (best 0.12035; linear floor `linbest` 0.12266). Stop
optimizing this OOS QLIKE — the levers are now **(1)** new data (auction-imbalance / GEX feed) and **(2)** DL on
the `linbest` residual targeting *state-space* nonlinearities, NOT path-order architectures.
