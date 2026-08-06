# Paper2 unification campaign — spec sheet (draft for joint review)

Goal: every number quoted in a paper2 table comes from ONE campaign under ONE
frozen specification, so all comparisons are bar-for-bar under identical prep,
panel, horizon, and smear machinery. This retires the three-convention split
(old base-5 primary tables / AM-campaign no-Duan region / msweep tile line).

## Frozen specification (proposed — confirm before greenlight)

| Axis | Frozen value | Rationale |
|---|---|---|
| Panel | b2 convention, 242,934 rows | base-2 ladder beat base-5 (DM −10.6, monotone); newest line already lives here |
| Prep | fixmask / availability-honest fills | post-leak repair; indicator-before-fill discipline |
| Target | winsorize(sqrt(RV / rolling diurnal)), h=1 | the paper's deliverable horizon |
| Train window | 24,000 bars, per-bar refit (RollingLeastSquares / BlockRidge) | matches incumbent convention |
| Smear | Duan as implemented in src/evaluation/metrics.py (§3 contract) | what the code does; §3 freezes it before any comparison |
| Inference | per-bar DM; paper quotes at-scale pooled numbers only | tile-level results are discovery narrative, not table rows |
| Persistence | EVERY arm saves per-bar losses + raw preds (npz), cluster-side | standing rule; makes §3 sensitivity + future LSTM smear a re-score, not a recompute |

## Arms, by paper section

### §2 — benchmark (1 arm)
- A0: per-bar OLS on HAR(target), b2 ladder. THE reference number.

### §3 — smear sensitivity (0 new arms; re-scores of everything)
- Re-score the full campaign under: no smear / trailing-window smear /
  training-residual smear. Pure post-processing on persisted preds.
  Deliverable: does any pairwise ranking flip?

### §4 — bucket attribution (10 arms)
- A1..A8: OLS (literal α=0) per exogenous bucket added to the HAR backbone.
  NOTE: never run before — July battery was the causally-tuned-ridge
  surrogate on base-5.
- A9: joint all-buckets OLS.
- A10: no-exog negative control.

### §5 — dense-but-weak (7 arms)
- B1/B2/B3: ridge / elastic-net / lasso on the wide all_features basis
  (the head-to-head, center stage).
- B4: 2-block ridge — α=1 HAR(target) + α=100 HAR(all_features). ENDPOINT.
- B5–B7 (optional supporting): light-alpha tie check, oracle-penalty ceiling,
  selection-that-bites probe — confirm which of these the section actually
  quotes before spending arms.

### §6 — nonlinearity + product (5 arms)
- C1: XGBoost, C2: LightGBM on all_features (tree edge, defaults per the
  tuning-is-not-the-lever result).
- C3: trees on HAR-only (pure nonlinearity premium).
- C4: product block alone (diagnostic).
- C5: 3-block ridge — α=1 target-HAR + α=100 linear exog + α=1000 product.
  ENDPOINT. NOTE: product-block verdict is known to be regularization-
  sensitive; α=1000 is the frozen choice, sensitivity relegated to appendix.

### §7 — transmission (4 arms)
- D1: 20-factor construction on the frozen panel (deterministic; factor
  freshness/stability diagnostics ride along).
- D2: Cucuringu lead–lag features on the factor panel (antisymmetric part of
  lag-1 cross-correlation; battery per analysis/cucuringu.py, but only the
  arms §7 quotes).
- D3: transmission block alone (diagnostic).
- D4: 4-BLOCK RIDGE — target-HAR + linear exog + product + transmission.
  THE FINALE. Never scored as a single run under any convention.

## Arm count and shape
~27 scored arms + 3 campaign-wide re-scores. All arms share the identical
data build → natural fit for one hpc-agent campaign (frozen-list style,
per-arm specs, sidecar reducer for the pooled tables).

## Open decisions (blocking greenlight)
1. Confirm the frozen spec row-by-row (esp. base-2 and Duan-as-contract).
2. §5/§6 supporting arms: which appear as table rows vs prose citations?
3. Descriptive-analysis content for §2 — may add cheap diagnostic arms.
4. Cluster + budget: CARC vs Hoffman2, QOS caps.
5. Block alphas: the stated (1, 100, 1000) vs the documented composed-model
   precursor (1, 3e3, 3e4 on the interaction-study convention). The product
   block's sign is known to be regularization-sensitive — confirm whether
   (1, 100, 1000) is the intended frozen spec or whether the campaign should
   carry a small alpha grid on the product/transmission blocks and freeze by
   a pre-registered rule.
