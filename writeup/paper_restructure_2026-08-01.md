# Paper restructure — the Hawkes-first arc (2026-08-01)

The new spine: each section's CONCLUSION becomes the next section's
INSTRUMENT. Descriptive analysis concludes the backbone (power-law kernel +
clock); that backbone measures the buckets; the buckets reveal dense-weak;
dense-weak resolves into a low-rank operator; the operator yields the
paper's contribution (the five-question state + its stylized facts).

Multi-horizon / product material -> PAPER 2 (banked below).

---

## S1. Descriptive analysis -> the case for a kernel (Hawkes) backbone

Thesis: before any exogenous question can be asked, the target's own
structure fixes the instrument — and it fixes it as a truncated power-law
kernel with a clock organ, of which HAR is a 3-knot quadrature.

Content (existing -> new):
- RV panel description, distributional properties [existing descriptive_analysis]
- TARGET TRANSFORMS as measured choices, not conventions:
  sqrt = variance stabilization (threshold-mixture argument: the cap-mixture
  that refuses tail saturation); causal diurnal division under the
  slot-conditional multiplicative family (one-parameter sufficiency for
  positive series; signed series / slot std); **divide-vs-rank raced: rank
  +0.00995 WORSE in the active regime (p=.001), tie in calm — completeness
  of marginal adjustment is not a virtue when magnitude is signal**;
  causal winsorization.
- CLOCK VARIABLES: calendar organ (biggest single organ, -0.0023);
  session-transition regime preview (target kernel rotates: cos +0.83
  overnight -> -0.55 open) — the hook that pays off in S4.
- HAR AS KERNEL QUADRATURE: ladder = uniform knots in log-lag; power law =
  line in log-log; scale-free kernel <-> scale-free grid. MEASURED LADDER:
  base-2 beats base-5 (DM -10.6, monotone in base, cap non-lever);
  single fitted power law beta=1.36 ~ reproduces the ladder; exponential
  +0.011 worse; truncation law (U=2048 catastrophic, cap-256) — the kernel
  is power-law, truncated, and its quadrature refinement pays.
- CONCLUSION: the backbone = truncated power-law self-kernel (Hawkes-native
  reading; mixture-of-beta as the freshness-robust form) + clock organ.
  Fig: self-kernel vs u^-1.36 log-log (figs_op/fig3).

## S2. The instrument at work: kernel-backbone + OLS bucket attribution

Thesis: with the backbone fixed, exogenous information is measured by
adding buckets to it; attribution is least-squares so capacity never
confounds information.

Content: existing marginal_contribution (all-buckets-help, overlap = sum of
parts ~ 2x joint, FWL 96% to the backbone, MCS singleton on joint,
2^8 factorial) — RE-ANCHORED to the kernel backbone.
**COMPUTE GAP: the July bucket battery ran on base-5 OLS-HAR. Panel unity
demands a re-run of the bucket table on the b2/kernel backbone — this is
the main new computation the restructure requires. Design it into the
at-scale frozen-list run (per-bucket arms on the b2 convention).**

## S3. Dense but weak: shrink, never select

Thesis: the joint exogenous signal is spread across hundreds of columns,
none individually strong; the correct estimator response is uniform
shrinkage, and every selection mechanism loses.

Content: ridge-vs-lasso/enet on all_features [existing linear_vs_nonlinear
penalty material + TODAY: enet on the curated dictionary — light-alpha
ties (60-col free prune), selection that bites hurts +0.005 p=.002];
PCA/PCR fails, plateau spectrum ("sparse in basis, NOT low-rank in data
space" — worded carefully, sets up S4's resolution); oracle-penalty ceiling
0.00015; tuner-trails-default. DECISION (flagged): fold the tree/
nonlinearity evidence here as regime characterization — trees' modest edge
saturating at additive+pairwise on the wide basis [existing], AND today's
completion: on the kernel dictionary trees fail in both seats (pure +0.08
extrapolation clamp; residual +0.0096 vs the smooth local-linear reader's
-0.0016) — nonlinearity is real but smooth/diffuse, not tree-shaped.

## S4. The resolution: dense-weak is low-rank as an OPERATOR

Thesis: no low-dimensional subspace of feature space carries the signal,
but the coefficient field factorizes — 41 channels share 5 lag profiles.
Dense in coordinates because five questions are asked forty-one ways.

Content (all from this campaign; figs committed figs_op/):
- Construction: probe -> Km (41x12) -> causal EWMA pool -> SVD; gauge
  irrelevance proven bit-exact; pool memory ~10 refits = the estimation
  floor of subspace tracking; K=5 knee, K=12==full.
- THE FIVE QUESTIONS (named, with sv weights): slow vol-complex curvature /
  tails-vs-priced / quarter-vs-fortnight / day-vs-week band / participation
  surprise. Shapes are CONTRASTS because the backbone absorbs persistence:
  the exog dictionary is the orthogonal complement of the HAR approximation.
- ECONOMIC READING: (a) the INNOVATIONS-operator fact — every effective
  kernel = lag-1 spike minus a shape-specific reference past ("this bar
  vs which history"); (b) loadings built from four dualities
  (implied/realized, vw/ew, continuous/jump, buy/sell); (c) separability
  K(t) ~ sum a_k(t) u_k v_k' — static geometry, fast amplitudes (tshape:
  temporal modes add nothing beyond the power-law family; fig4 shows the
  freshness law raw: flat mode-1 amplitude, slowly drifting mode-2).
- PARSIMONY PRICING: 203p/32ch == 1077p exactly (prune exact; dead
  families named); the compression certificate.
- SESSION GEOMETRY: amplitudes swing 3-4x through the day; the daily loop;
  span coverage 57-76% (max midday, min at auctions — the geometry itself
  bends at transitions); target-kernel rotation completes the S1 hook.

## S5. The contribution: five clock-breathing questions as a portable state

Thesis (both "new feature" AND "stylized facts"):
- THE FEATURE: the five shape projections (+ their Fourier clock
  modulation at heavy shrinkage) = a 5-25 column portable state that
  carries the exogenous content of intraday RV. Evidence: legchamp+clock
  and prune32+clock+corr each TIE the 1,077-col champion (p=.95-.998) —
  the black box's performance is fully reproduced by nameable parts.
- STYLIZED FACTS (each with receipts):
  F1 Exogenous variables inform intraday RV through INNOVATIONS against
     channel-specific memory, not levels.
  F2 The response operator is rank-~5 with quasi-static geometry; all fast
     dynamics are amplitude motion (pool-structure/refresh-amplitudes).
  F3 The amplitudes breathe on the daily clock (continuous Fourier cycle,
     not session gates — gates lose on both tiles); harvesting the cycle
     requires heavy shrinkage and its value is regime-priced (penalty
     curve measured); the auctions bend the geometry itself.
  F4 Selection loses to shrinkage at every layer this was tested: columns
     (S3), cells (rawc regime-flip), marginal adjustment (rank), trees.
- Close with the correction/stack as the measurement instrument that
  DISCOVERED the missing state (autocov/participation loadings) — the
  algorithm_design material, repositioned as instrument-not-contribution;
  estimator engineering (rank-1, BlockRidge, homotopy) -> appendix.

---

## Structural decisions flagged for the author
1. linear_vs_nonlinear DISSOLVES into S3 (regime characterization), with
   the tree story compressed; algorithm_design DISSOLVES into S5 close +
   appendix. Neither survives as a standalone section.
2. PANEL UNIFICATION (the blocker): S1-S2's existing tables are on the old
   218,934-bar base-5 panel; S1's ladder/kernel results and all of S4-S5
   are on the 242,934-row b2 panel. The paper's one-panel commitment means
   the bucket table (S2) and the incumbent definition must be re-run on
   the b2 convention — fold into the at-scale frozen-list cluster run
   (bucket arms + frozen champion arms, one campaign).
3. Numbers quoted from tile-1 must ship with the tile-2 discipline story
   or wait for at-scale — recommend the paper quotes ONLY at-scale pooled
   numbers in S2-S5 tables, with tile-level results as the discovery
   narrative.

## Banked for PAPER 2 (multi-horizon / the product)
The leak autopsy + honest-label convention; capacity-vs-horizon shrinkage
law (alpha* ~ 1e4, stable across H, scales with capacity); two-block ridge
(nested incumbent) + announcement organ = -0.0095..-0.0101 at H=4/8/16
(p~0, honest, 4y); organ base-independence (3 bases); VIX hurts intraday /
helps daily; direct-per-horizon (propagation refuted, rho~0.999);
dictionary-vs-wide intraday gap ~0.003 (light-block minimalism); straddle
economics vs measured SPY friction (net t~6.5 daily); intraday implied-
side data gap. Handoff: SESSION_HANDOFF_2026-08-01B.md sections 1-2.
