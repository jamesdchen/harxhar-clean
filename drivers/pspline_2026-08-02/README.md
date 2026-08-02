# Smooth-β / penalized-fRRR campaign — 2026-08-02

Question: strip the Hawkes reading and build the lag structure under the
smooth-coefficient-function frame (β(log u) smooth; HAR = step-function
quadrature; the 5-shape operator = functional PCA of 41 coefficient curves).
Two predictions were pre-registered from the frame before running:
(1) a roughness-penalized spline at matched effective df should perform at
the b2-vs-b5 scale (~0.0008), NOT the power-law-vs-exponential scale (0.011)
— ties or small wins; (2) one-step penalized functional reduced-rank
regression should reproduce the two-step probe→SVD shapes.

Harness: both tiles (t1 [24000,26189), t2 [26189,28378)), W=24000, cadence
240, per-bar sliding ridge (α=1, free intercept), Duan smearing → per-bar
QLIKE → DM. Panels: `prep_cache_all_features_b2_r32000.npz` and the
base-√2 twin (`PREP_ROWS=32000` trims the raw build to the tiles; causal
transforms make trimmed values identical to the full build). Rebuild:

    HAR_BASE=2 PREP_ROWS=32000 python -c "import run_geometry_local as g; g.prepare_full('all_features')"
    HAR_BASE=1.414213562373095 PREP_ROWS=32000 python -c "..."   # same call

**Anchors reproduced** (harness credibility): fitted power-law β = 1.34–1.39
per segment (known 1.36); exponential-vs-ladder +0.0124 on t1 (known
+0.011); b5-vs-b2 sign; proj-vs-kbar pooling wash; K=8-vs-K=5 wash on b2.

## sk_race.py — self-kernel basis race

Same support (lags 1..2048; b5 to 3125), same calendar block, arms differ
only in the lag basis. QLIKE (vs b2 DM in parens):

| arm | df/edf | tile 1 | tile 2 |
|---|---|---|---|
| b2 ladder (comparator) | 22.0 | 0.177506 | 0.125033 |
| b5 ladder | 16.0 | +0.000454 (p=.42) | −0.000103 (p=.84) |
| arith12 (level-spaced) | 22.0 | **+0.018786 (p≈0)** | **+0.009540 (p≈0)** |
| b√2 ladder (22 rungs) | 32.0 | −0.001200 (p=.071) | −0.000499 (p=.095) |
| P-spline, matched edf | 22.0 | **−0.001251 (p=.007)** | **−0.000426 (p=.062)** |
| P-spline, λ=1 (best) | 25.1 | −0.001709 (p=.0006) | −0.000447 (p=.11) |
| power law U=256 | 11 | +0.000048 (p=.94) | −0.000322 (p=.20) |
| exponential | 11 | **+0.012425 (p≈0)** | +0.000791 (p=.27) |
| mix5 (fixed 5-β) | 14.4 | −0.000456 (p=.41) | −0.000361 (p=.08) |

Verdicts:
1. **The matched-df P-spline beats the b2 ladder with the SAME SIGN on both
   tiles** (t1 −0.00125 p=.007; t2 −0.00043 p=.062) — it survives the
   tile-2 contact that flipped every other sub-0.001 scalar knob in the
   08-01 campaign. Magnitude is regime-scaled and sits at the predicted
   b2-vs-b5 scale, not the 0.011 scale.
2. **λ is a non-lever across five decades** (0.01–1000 all within 0.0001 of
   each other) — penalty-insensitivity, the house law, holds within the
   spline family too. Only λ ≥ 1e5 (edf → 16, collapsing toward the D2 null
   space, which is LEVEL-linear in log-lag — not the power law) degrades.
3. **Smooth basis == fine ladder**: spl vs b√2 is a tie on both tiles
   (−0.0005 p=.30 / +0.0001 p=.82). The spline harvests the same
   refinement gain with 16 basis functions at edf 25 vs 22 rungs at edf 32
   — "refinement pays" and "β is smooth" are the same finding.
4. **Arithmetic spacing is catastrophic** (+0.019/+0.0095, p≈0 both tiles,
   same df as b2): the operative content of the prior is log-lag knot
   placement, exactly as the smooth-in-LOG-lag frame predicts (curvature
   concentrates at short lags; level spacing starves them).
5. Single fitted power law still ties the 12-rung ladder; exponential still
   fails on the active tile (calm tile: nearly a wash — the kernel-form
   penalty is itself regime-priced).
6. Footnote: the prep's rolling-robust-scaled ladder columns (`b2prep`)
   underperform the identical ladder with fixed first-window
   standardization on t1 (+0.0019 p=.0002; t2 wash). Rolling per-column
   rescaling of ladder features is itself a mild t1 lever — flagged, not
   concluded.

## pfrrr_operator.py — one-step penalized fRRR vs two-step probe→SVD

Static (windows ending at each tile start; ALS on the probe gram, gram-
contraction trick so each iteration is O(cols²) once the probe gram
exists):
- **Unpenalized one-step RRR ≈ two-step**: principal angles vs the SVD
  frame [1.9, 4.6, 13.5, 26.3, 67.2]° (t1 window; t2 similar) — the top-3
  directions are essentially identical; SSE 1432 vs 1451 (~1.3%). Only the
  weakest direction rotates. Caveat: at λ=0 ALS is init-sensitive in the
  deep directions (random init lands 79° away on the last shapes); with
  λ>0 it is init-independent (0.0°).
- Penalized shapes are 10–40× smoother (‖D2V‖² 0.9–2.7 vs 34.5) at ~1%
  SSE cost; spike-exempt vs full-D2 barely differ — the lag-1 spike is
  cheap to keep either way.

Walk-forward (identical amplitude model [1 | C(41×K) | self-ladder |
calendar]; arms differ only in shape estimation; DM vs ctrl):

| arm | tile 1 | tile 2 |
|---|---|---|
| ctrl = two-step, Kbar pool, K=5 | 0.164675 | 0.124369 |
| proj pool (gauge control) | +0.000283 (p=.25) | −0.000012 (p=.80) |
| rrr0 (one-step, λ=0) | +0.000676 (p=.58) | −0.000218 (p=.83) |
| rrrS (pen., spike-exempt) | +0.003573 (p=.11) | −0.001249 (p=.28) |
| rrrF (pen., full D2) | +0.004115 (p=.065) | −0.000757 (p=.50) |
| sq2 (√2 panel, K=5) | **+0.007731 (p=.0016)** | +0.001220 (p=.25) |
| ctrl K=8 (b2) | 0.164639 (wash) | 0.124302 (wash) |
| sq2 K=8 | **+0.008328 (p=.0008)** | +0.001191 (p=.31) |

Verdicts:
7. **Estimator-order irrelevance**: one-step RRR ties the two-step recipe
   in prediction on both tiles. Together with the static angles this
   CERTIFIES probe→SVD as the cheap equivalent of the "proper" one-step
   estimator — the two-step is not an approximation debt.
8. **Roughness-penalized shapes are a regime-priced knob**: hurt t1, help
   t2, significant nowhere — by the meta-law, not established; don't
   adopt. (Consistent with sk_race: smoothing pays on the SELF-kernel,
   where the curve is a raw transfer kernel, but the operator shapes are
   FWL contrasts — smoothing them fights the innovations structure.)
9. **Rank ceiling** (the "is 5-of-12 an artifact?" question): on the √2
   panel the ladder has 23 rungs and the spectrum flattens (top-5 = 81%
   of energy vs 94% on b2) — but the extra energy is NOT predictive:
   sq2 is worse at K=5 on both tiles, K=8 doesn't rescue it, while on b2
   K=8==K=5 exactly (wash). The finer operator resolution buys estimation
   noise, not rank. Self-kernel refinement pays (b√2, spline); OPERATOR
   refinement doesn't — quadrature refinement of the persistence backbone
   and lag-resolution of the exog operator are different economies.

## Files

- `sk_race.py`, `pfrrr_operator.py` — drivers (env knobs in docstrings).
- `verdicts/` — committed run logs.
- `results/pspline_race/` — per-bar preds (`preds_*`, `op_preds_*`),
  static shapes/spectra (`op_shapes_t*.npz`), `adj_full.npy` (pre-drop
  adjusted target, for exact full-history lag rebuilds).

Both tiles are screening chunks; the frozen-list discipline applies —
tile-1 magnitudes are comparators, only cross-tile-sign results are
candidate claims, and anything adopted goes through the at-scale run.
