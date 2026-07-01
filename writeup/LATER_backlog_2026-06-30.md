# LATER backlog — research plan items deferred (for me, not the advisor update)

Bucketed out of the June meeting update (which covers NOW + NEXT). These are real
plan items but are either deprioritized, blocked on data, or exploratory. Kept
here so the advisor-facing writeup stays tight.

## Models
- **LSTM** — plan item ("might give some explainability; discuss with Chris"). Not
  implemented; the DL budget went to PatchTST + the differentiable regime-MoE.
  Revisit *if* the regime substrate gets new (auction/GEX) information to sequence.
- **Mamba** — not implemented. Same rationale; low priority until there is a
  sequence signal worth a state-space model (price-only sequence content is
  d8-captured, see DIN/BST prize-sizing: only −0.00037 survives d8).

## Sequence / attention deep-learning
- **Full multivariate-history attention** — the DIN/BST target-attention and the
  log-signature (level-3 antisymmetric) basis were tested on a few hand-picked
  channels and are null; the *full multivariate path* (all exog channels, longer
  horizon) is untested. Only worth it after new information is in the feature set.
- **Clustering → per-cluster local models** — beyond documenting the global-vs-local
  trade-off (that's NEXT), the deeper "cluster then fit a bespoke estimator per
  cluster" program.

## Estimator design (the "construct our own estimators" thread)
- Build estimators keyed to the *shape* of the discovered signal (e.g. the
  session-edge auction kernel) and analyse their statistical properties
  (bias/variance, consistency) — a methods-paper direction rather than a QLIKE lever.

## Data (blocked until acquired)
- **Order-book depth / quoted-spread microstructure** (per-name, intraday book) —
  the Cont–Kukanov–Stoikov depth-change OFI needs L2, which we don't have.
- **Richer sentiment/attention** beyond aggregated StockTwits (per-name,
  higher-frequency, multi-source).

## Interpretation / write-up
- **Financial-economics interpretation** of the auction/gamma mechanism "after the
  fact" (the plan's stated order: find something interesting first, interpret later).
- **Deeper options-market implications** (0DTE/OPEX gamma-dilution is a lead from the
  explainability audit; formalize the dealer-hedging story).

## Note
Everything here is downstream of the same conclusion: the price-only feature space is
at the floor, so these are worth pursuing mainly once the data-to-buy (NEXT) lands and
clears the gate. See `DATA_TO_BUY_SPEC.md` and `OVERNIGHT_RESULTS_2026-06-30.md`.
