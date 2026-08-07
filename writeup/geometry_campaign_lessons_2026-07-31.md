# Geometry campaign — lessons, evidence, iteration rules (2026-07-30/31)

Screening slice: chunk [24000, 26189), 2,189 OOS bars, all_features bucket.
Comparators: OLS-HAR incumbent 0.17856 · per-bar ridge-HAR base-5 0.16899 ·
base-2 ridge anchor 0.16425. Campaign best: anchored 0.16267 (p=0.0072 vs
matched control) · identity 0.16504 (tpolq, unproven). All local numbers are
one-chunk; the frozen 100-chunk list is the claims machine.

## The law

Weight of a direction in the metric =
**(predictive slope) x (variance/stability support) x (inside the memory horizon)**
= supervision x leash x hygiene. Each term has a measured ablation:

| term dropped | experiment | result |
|---|---|---|
| supervision | euclid/PCA/gcluster/har/tica/whiten geometries | 0.2246-0.45, all dead; NW-kNN dominated by own linear readout (plsreg 0.1707) |
| leash | ticaslp (slope on whitened dynamics modes) | 0.177 capped / 0.216 uncapped, p~1e-13; spectra show all weight at ts=2b = in-sample fishing |
| hygiene | uncapped basis under full supervised pipeline | 0.1732 vs 0.1655 capped (-0.008) |

## Lessons with evidence and iteration rules

**L1 — Supervision must shape the metric.**
Evidence: every y-free geometry dead (euclid 0.2334, PCA tie, gcluster 0.2246,
tica collapse 0.4515); first supervised compression (PLS-d3) 0.1915.
Rule: no arms on y-free geometry variants; every proposal must state how y enters.

**L2 — Supervision needs a variance leash.**
Evidence: cov-weighting double-counts the collinear level (0.1939 vs slp
0.1904); free slope on whitened modes = catastrophe (ticaslp); the balanced
blend PCovR beta=0.5 is the anchored champion coordinate system (0.16267).
Spectrum-steepness corollary: steep target spectrum -> hard truncation (d3);
anchored/flat residual spectrum -> more comps + soft filters (d8, soft > hard).
Rule: supervised weights only on variance-supported directions; beta is the
dial; distrust any ranking of isotropic candidates by in-sample correlation.

**L3 — Hygiene is irreducible and equals the persistence law's horizon.**
Evidence: cap sweep (break under 64, optimum ~256 = 5 days); capped view
recovers rungs 625/3125 only at R2 0.51/0.43 and STILL wins; anti-TICA fails
(lambda continuum 1.0018..0.997 — no gap; the level factor is slow AND
essential, so slowness cannot separate signal from drift); drift fools both
other terms (high variance + spuriously stable in-window slope).
Asymmetry: metrics need the cap; regressions self-clean (linear cap non-lever;
masked-OLS saga = hygiene at alpha=0; NEVER pre-compress a regression's basis
— anchored pol2 collapse 0.16928).
Rule: capped basis for metrics, full basis for anchors; never learn the
horizon from X or in-window y; it is factor-1 domain knowledge.

**L4 — The estimator must nest the linear model.**
Evidence: local-constant NW plateaus 0.188+; local-linear k-ladder ties
per-bar ridge-HAR exactly (k=4000, diff +1e-5, p=0.996) then dips below
(k~8000, alpha_loc=.003-.01, plateau 0.1678-0.1682); absolute local penalty
underfits reweighted coords (toy 0.75 -> 0.95 with scale-free alpha*tr(G)/d).
Rule: neighbor methods always carry the local plane; big k + Gaussian
weights; all penalties dimensionless.

**L5 — Memory representation is CLOSED: base-2 flat-window mean ladder to ~5d.**
Evidence: base knee (1.5: 0.16701 / 2: 0.16550 / 5-cap125: 0.16638, both
modes); flat kernel beats EWMA (+0.0028, p=0.017 — window exits ARE shock
clocks); bands hurt +0.003 in 3 architectures (per-column standardization
amplifies fine-scale noise); memory curve is log-linear (pol1 = 90%,
curvature = noise; rough-vol power law confirmed functionally); EIGHT
enrichments neutral-or-worse (shock ages, max-ladders, vol-of-vol, session
windows, clock rungs, detica m=4/16, EWMA, bands).
Rule: stop enriching memory features. Further gains must come from new
information (factor 2) or new state-maps (L6/tpolq class), not new memory.

**L6 — Dense-weak structure yields to shrinkage-over-products, never selection.**
Evidence: d8-dissection 36% diffuse; quad-ridge on 152 poly-2 PLS terms
matches ridge-HAR with no kNN (b2_q 0.16846, best identity oos_r2 of its
wave); identity spls/slp/exp statistically tied (pairwise p 0.17-0.40);
local paraboloid works anchored (0.16441), overfits identity (0.185);
roughness-state quadratics rv_level/tilt (+sq, x) = best identity number
0.16504 with oos_r2 0.10 -> 0.17 (DM p=0.50 — frozen-list candidate).
Rule: interactions enter as L2-shrunk products of compressed/named
coordinates; individual terms will never test significant — validate only
the aggregate through the gate.

**L7 — Mechanics: nested pairs + matched controls, or the number is void.**
Evidence: nested avail-MA contrast resolves +0.0005 at p=0.041 while a 2x
larger cross-family gap sits at p=0.66; the July battery lacked anchor-alone
controls (its "beats incumbent" partly = anchor); the base-2 control split
the champion's -0.006 into -0.0047 basis + -0.0016 correction (p=0.0072);
in-window probes lie under drift (tica won the probe, lost the backtest);
scoreboard-picked knobs dissolve (spls "win" p=0.40).
Rule: every arm ships with its matched control; local comparisons nested
only; knobs frozen before scale; probes must be forward.

## Iterate-further queue (in expected-value order)

1. **Cluster run of the frozen list** — the only instrument that converts
   0.001-0.002 into claims. Arms: b2 omega=0 control, b2a_pcovr50loc
   (champion), b2_cap256 + f1_tpolq (identity), b2_q (legible baseline),
   base-5 pair (continuity). All spec env knobs wired (HAR_BASE, RUNG_CAP,
   LOCAL_ALPHA, PLS_DECAY, GEOMETRY).
2. **lgbm anchor + local-linear correction** — the strongest base has never
   carried a correction; needs cadence-refit residualizer engineering.
3. **End-to-end metric (MLKR/ModernNCA direction)** — L1+L2 with gradient
   horsepower; the leash becomes early stopping/regularization; field SOTA
   says this is the ceiling-raiser for neighbor methods.
4. **Factor-1 state-map extension** — more terms of the (level, tilt) ->
   local-slope map beyond tpolq's quadratics, e.g. clock x tilt.
5. **Factor 2 when unparked** — GEX Fork A is shovel-ready (data + pipeline
   on disk; prior: gamma predicts variance).

## Artifacts

Driver `run_geometry_local.py` (prep caches, transforms bands/poly/shock/
session/vv/detica, arm syntax tag:geom:W:k:d:resid:decay:cap:loc:navma:
bands:omega); per-arm predictions in results/geo_preds/ (pooled per-bar DM);
battery CSV results/geometry_chunk_battery.csv; fast anchors
(RollingRidgeResidualizer, block-Woodbury FAST_ANCHOR); memory file
spectral-geometry-pls-2026-07-31.md holds the running record.
