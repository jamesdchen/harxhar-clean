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
| Smear | NONE cluster-side — arms emit raw per-bar (y, ŷ, B_t); QLIKE scored locally under the §3 contract | RESOLVED 2026-08-06: the smear must come FROM the smearing section's development; scoring is a local deterministic pass, so a §3 upgrade (incl. LSTM) re-scores without recompute |
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
- A1..A8: OLS (literal α=0) per bucket entry added to the HAR backbone, using
  the repo's canonical bucket enumeration AS-IS — the joint/all_features
  bucket is already one of its entries, so there is NO separate joint arm
  (dedupe 2026-08-06). NOTE: never run before — July battery was the
  causally-tuned-ridge surrogate on base-5.
- A10: no-exog negative control — collapses into A0 if pipeline-identical
  (alias noted in the executor registry rather than run twice).

### §5 — dense-but-weak (7 arms)
- B1/B2: ridge / lasso on the wide all_features basis (the head-to-head,
  center stage; enet CUT 2026-08-06 — user directive).
- B4: 2-block ridge — α=1 HAR(target) + α=100 HAR(all_features). ENDPOINT.
- B5–B7 (optional supporting): light-alpha tie check, oracle-penalty ceiling,
  selection-that-bites probe — confirm which of these the section actually
  quotes before spending arms.

### §6 — nonlinearity + product (5 arms)
- C1: XGBoost, C2: LightGBM on all_features (tree edge, defaults per the
  tuning-is-not-the-lever result).
- C3: trees on HAR-only (pure nonlinearity premium).
- C4a/C4b: product block alone (diagnostic), under BOTH conventions
  (user 1000@24k / doc 3e4@250d — confirmed 2026-08-06).
- C5: 3-block ridge — α=1 target-HAR + α=100 linear exog + α=1000 product.
  ENDPOINT. NOTE: product-block verdict is known to be regularization-
  sensitive; α=1000 is the frozen choice, sensitivity relegated to appendix.

### §7 — transmission (4 arms)
- D1: 20-factor construction on the frozen panel (deterministic; factor
  freshness/stability diagnostics ride along).
- D2: Cucuringu lead–lag features on the factor panel (antisymmetric part of
  lag-1 cross-correlation; battery per analysis/cucuringu.py, but only the
  arms §7 quotes).
- D3a/D3b: transmission block alone (diagnostic), under BOTH conventions.
- D4: 4-BLOCK RIDGE — target-HAR + linear exog + product + transmission.
  THE FINALE. Never scored as a single run under any convention.

## Arm count and shape
~27 scored arms + 3 campaign-wide re-scores. All arms share the identical
data build → natural fit for one hpc-agent campaign (frozen-list style,
per-arm specs, sidecar reducer for the pooled tables).

## Decisions (resolved 2026-08-06)
1. Frozen spec: b2 base-2 / fixmask / h=1 / 24k window; smear scored locally
   from §3 (see table). CONFIRMED with the smear amendment.
2. §5 supporting arms (light-alpha tie, oracle ceiling, selection-that-bites):
   SKIPPED — prose citations of existing findings only.
3. Block alphas: RUN BOTH — the stated (1,100,1000)@24k AND the documented
   (1,3e3,3e4)@250d, as parallel 2/3/4-block ladders (~6 extra arms). The
   convention question is settled empirically, no leakage.
4. Tree arms C1–C3: NOT in this campaign (user directive). Tree column may be
   servable from the existing covid_lgbm/xgb campaigns if convention-compatible.

5. Transmission-block alpha (_user convention): 1000 — author decision
   2026-08-06.
6. §5 head-to-head penalties: FIXED constants (repo's documented defaults),
   no tuning — author decision 2026-08-06.
7. Diagnostics under BOTH conventions (4 arms) — author decision 2026-08-06.
8. Ops: NO waves — all jobs submitted at once per cluster, scheduler QOS
   queues the overflow; detached overnight run, harvest in the morning.
9. Cluster split: Hoffman2 = OLS family; CARC = penalized + blocks +
   diagnostics + a0 float-parity canary. Nested comparison families never
   split across clusters.

## Still open
- Descriptive-analysis content for §2 — may add cheap diagnostic arms later
  (not part of this campaign).
- a10_noexog alias check: computational verification required before the arm
  count is final (19 vs 20 + doubled diagnostics → 21 or 22 total).
