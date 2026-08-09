"""unification — paper2 rerun-campaign executor: every table number from ONE frozen spec.

Contract (writeup/rerun_campaign_spec_2026-08-06.md): b2 (base-2) HAR-ladder panel
(~242,934 rows), fixmask/availability-honest prep, h=1 (the shift lives in feature
construction), rolling training window with PER-BAR refit, walk-forward over rows
[chunk_start, chunk_end). Arms apply NO Duan smearing and NO raw-space conversion —
each task emits raw per-bar arrays and QLIKE is scored locally later under the
paper's §3 smearing contract.

Framework contract: expose ``compute(args) -> None``. Flags are declared in
``.hpc/tasks.py`` FLAGS by the submit flow and arrive parsed on ``args``:

    arm          str   registry key (below); missing/empty prints the registry, exit 0
    chunk_start  int   first emitted OOS row (global b2-panel row index)
    chunk_end    int   one-past-last emitted row (-1 = end of panel)
    halo         int   history bars available before chunk_start (default 24000);
                       must be 0 (unchecked) or >= the arm's training window
    window       int   training window bars (default 24000 = 500 days x 48 bars/day)
    output_file  str   .npz written here

Per-task npz schema (all float64 unless stated; one aligned index, rows
[chunk_start, chunk_end)) — satisfies the persistence contract:

    row_id      int64          global b2-panel row index (panel row i = loader row
                               i + 3125: the fixed HAR burn-in drop; h=1 shift is a
                               no-op so no further offset)
    t           datetime64[ns] bar end timestamp (exact bar-for-bar joins + HAC order)
    y_fit       f64            target on the FIT scale: winsorized sqrt(RV/B_t)
                               (the panel's adj_RV) — what the model was trained on
    yhat        f64            forecast on the same fit scale (sqrt-space)
    rv_raw      f64            actual raw RV for the bar (unwinsorized; raw-space
                               truth is exact, never reconstructed from y_fit)
    baseline    f64            B_t, the rolling diurnal baseline reversing Stage-1
    valid_mask  bool           True where yhat is a real forecast (always all-True
                               for the arms here; persisted so arms declare coverage)

Downstream computability check: (a) per-bar QLIKE under ANY smear convention from
(rv_raw, yhat, baseline) + window residuals derivable from (y_fit, yhat); (b) per-bar
loss differentials + HAC DM from the above with ``t`` ordering; (c) OOS R^2 by joining
any two arms' (row_id, y_fit, yhat); (d) MZ calibration from (y_fit, yhat) or
(rv_raw, reconstructed raw preds). Nothing requires recompute.

Sufficient statistics ride INSIDE the npz (sqrt_mse, sqrt_sse, sum_y, sum_y2, n,
ols_dropped_cols) — the npz is the single source of truth; the hpc_agent metrics
sidecar (written only when $RESULT_DIR is set) duplicates the same scalars for
the on-cluster combiner and is otherwise a no-op.

Reproducibility: the executor is fully deterministic — no RNG anywhere (linear
algebra only), so HPC_TASK_ID seeding is not needed. Thread counts are pinned
before numpy import for serial-run parity with the cluster preamble.

Reused repo machinery (never re-derived):
  * panel build  — experiments/run_geometry_local.prepare_full("all_features",
    har_base=2): the executor's own prep path (load_and_transform, HAR+calendar,
    3125-row burn-in drop, horizon shift, scale guards, rolling robust scale +
    masked scaling), cached npz. Data path is the repo-wide literal "data"
    (the convention of specs/har_base_sweep.py / specs/causal_tune_linear.py);
    cache dir overridable via $UNIFY_CACHE_DIR (default "results").
  * per-bar solver — src/models/rolling_least_squares.RollingLeastSquares via
    analysis/wf.walk_forward (rank-1 sliding-window ridge; exact OLS at alpha=0).
  * per-block penalties — the column-scaling identity (ridge on col*c == penalty
    alpha/c^2 on the original coefficient): analysis/minimal_model.py:297-299 and
    955-960; drivers/msweep_2026-08-01/block_ridge.py holds the per-feature-penalty
    gram variant of the same math.
  * penalized §5 arms — FIXED penalties (no causal tuning, user directive
    2026-08-06). Ridge rides the plain rank-1 path at the untuned production
    default; the lasso reuses the RollingTunedLinear machinery of
    specs/causal_tune_linear.py:201-379 (copied below verbatim-with-adaptation:
    that spec is a notebook-style module whose import EXECUTES the full backtest,
    so it cannot be imported; grid made an instance parameter) with a
    SINGLE-POINT grid, so the periodic "tune" is only the cold-reseed drift
    re-anchor + identifiability-mask refresh. Deps (enet_coef, enet_online,
    forward_window_split) import cleanly from src/models/reclasso_har.
  * product block — analysis/nl_sparsity.py:79-113 (base_columns rule, _pair_ic,
    _upper, _products; small pure-numpy helpers copied here because importing
    analysis.nl_sparsity drags the alpha-study cache guards) + the causal floored
    rolling-sd scale of analysis/minimal_model.py:334-338. Selection: |IC| against
    the first-block OOS HAR residual (analysis/synthesis.py stage_prep protocol),
    frozen forever — no monthly reselect.
  * transmission block — frozen-frame factor scores per
    analysis/map_monitor._frame_and_scores (eigenvectors of the first-window
    correlation of the product-base columns, frame FROZEN) + the Cucuringu
    lead-lag features Ghat = G(t-1) @ D per analysis/trans_exploit._trans_block
    (D = antisymmetric part of the lag-1 cross-correlation, trailing 504 days,
    refreshed quarterly).

Arm-list dedupe vs the spec sheet: the repo's canonical bucket enumeration
(src/data/loading.SUBGROUPS) already contains ``all_features`` as a bucket row, so
the spec's A9 joint arm IS ``a_bucket_all_features`` (8 bucket arms total = A1..A8
with joint included); A10 no-exog is pipeline-identical to A0 (same design, same
estimator, same rows) and is registered as an alias of ``a0_ols_har``, not a
duplicate arm.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Pin BLAS threads before numpy import: serial-run parity with the cluster
# preamble (which exports the same), and chunk results independent of node core
# counts.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (deliberately after the thread-count pinning)
import pandas as pd  # noqa: E402

if TYPE_CHECKING:
    import argparse

# ── repo-root bootstrap (executor may be invoked from any cwd) ────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data.loading import SUBGROUPS  # noqa: E402
from src.features.extractors.har import resolve_har_lags  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402
from src.models.reclasso_har import enet_coef, enet_online, forward_window_split  # noqa: E402
from src.models.rolling_least_squares import RollingLeastSquares  # noqa: E402

# ── frozen-spec constants ─────────────────────────────────────────────────────
HAR_BASE = 2  # the crowned base-2 ladder (har_base_sweep, DM -10.6)
BURNIN_ROWS = resolve_har_lags()[-1]  # 3125 — fixed arm-invariant burn-in drop
DEFAULT_WINDOW_BARS = 500 * PERIODS_PER_DAY  # 24000 — the incumbent 500-day window
DOC_WINDOW_BARS = 250 * PERIODS_PER_DAY  # 12000 — the documented convention's 250-day
#   window (writeup/alpha_manifestation_findings_2026-08-04.md §22: "250-day window").
#   NOTE the panel PRESCALE window stays the frozen prep's 24000 bars for doc arms too:
#   prep is frozen campaign-wide; the doc convention's 250 days applies to the FIT window.

# OLS-family estimator (author directive 2026-08-07): MINIMUM-NORM least
# squares — the unique least-squares solution of minimal l2 norm, the
# alpha->0+ limit of ridge, defined at any design rank (see _walk_ols;
# eigh-based pinv of the centered gram at numpy.linalg.pinv's default rcond).
# Rank deficiency is handled natively, so the former loud-fail guard stack
# (QR entry drops, atomic ladder drops, sticky repair, per-bar verification)
# is retired from this path; KEPT: the deterministic panel-level bitwise
# dedup below (cheap hygiene — duplicate columns and panel-constants dropped,
# first in canonical order wins, recorded in the output metadata), the
# go-live/feed-dead participation masks (prep semantics), and the fit/raw
# alignment gate. a0's full-rank backbone makes min-norm == plain OLS there;
# it rides the same path for uniformity.

# Block-ridge alpha ladders. USER = the stated convention @ 24k window; DOC = the
# documented §22 convention @ 250d window (backbone 1 / exog 3e3 / products 3e4,
# transmission at the exog 3e3 per §37's 699 design riding the solver alpha).
USER_ALPHAS = {"backbone": 1.0, "exog": 100.0, "product": 1000.0, "trans": 1000.0}
# (transmission alpha 1000 under the user convention: fixed by author decision
# 2026-08-06 — same as the product block.)
DOC_ALPHAS = {"backbone": 1.0, "exog": 3e3, "product": 3e4, "trans": 3e3}

# §5 head-to-head penalties are FIXED, untuned (user directive 2026-08-06: the
# battery lore is that the untouched default beat the causal tuner).
FIXED_RIDGE_ALPHA = 1.0  # src/models/ridge.py:16 DEFAULT_RIDGE_PARAMS — the
#   production untuned default, the exact comparator the rolling tuner trailed
#   (specs/rolling_linear_tune.py "UNTUNED default (DEFAULT_RIDGE_PARAMS alpha=1.0)").
FIXED_LASSO_ALPHA = 1e-4  # sklearn Lasso(alpha=...) units at l1_ratio=1.0 —
#   pinned by final decision 2026-08-06 (center of the July battery's reclasso
#   grid, logspace(-6, -2)). sklearn's own Lasso default (alpha=1.0) zeroes
#   every coefficient on this standardized basis and was rejected as the pin.

# Product block (analysis/nl_sparsity.py + analysis/synthesis.py conventions).
N_PROD = 100  # synthesis.N_PROD — 100 frozen products
SELECTION_RIDGE_ALPHA = 1.0  # backbone ridge for the selection residual
#   (analysis/synthesis.py stage_prep: alpha=1.0, per-bar refit)
PRODUCT_EXOG_WINDOWS = (1, 32, 512)
# ACCEPTED (ruling 2026-08-06): nl_sparsity.EXOG_WINDOWS is (1, 25, 625) — the
# fast/intraday/slow rungs of the BASE-5 ladder. The b2 panel's exog rungs are
# powers of 2, so (1, 32, 512) is the nearest b2 tri preserving the documented
# fast (1 bar) / intraday (~half-day) / slow (~2-week) timescale roles.

# Transmission block (analysis/map_monitor.py + analysis/trans_exploit.py conventions).
TRANS_QPOOL = 20  # map_monitor.QPOOL — 20 frozen-frame factors
TRANS_TRAIL_DAYS = 504  # trans_exploit._trans_block trail_days
TRANS_REFRESH_DAYS = 63  # quarterly D refresh
TRANS_LAG_BARS = 1  # lag-1 (bars) cross-correlation
# Cadence-heterogeneous aggregate families (era dig 2026-08-07): the ew/vw
# feed-cadence homogenization lands Dec-2014, coinciding to the month with the
# flow component's sign-flip; sumret3_ewstock / sumret3_vwstock carry
# availability patterns differing from the other ten ew/vw columns on 6,029
# bars, ALL pre-2015 (last differing bar 2014-12-31 06:30). blk4_trailDropHet
# excludes these two families from the TRANSMISSION BASE ONLY (the test is
# about what feeds the operator/scores; the exog ridge block keeps them).
TRANS_HET_STEMS = ("sumret3_ewstock", "sumret3_vwstock")

# FULL LIVE SPECTRUM of the frozen transmission frame — the number of base
# columns with dispersion in the frame window, i.e. the maximum K the frame can
# supply. Documented from the transmission section's liveness count; asserted
# against _frame_live_rank at build time (see the trans_trailGFull block), so a
# drifted panel fails loudly instead of quietly redefining the arm.
TRANS_FULL_SPECTRUM = 115

# RE-AUDITED 2026-08-09 on the production panel: live rank is 115 of 145 (the

# panel drifted from the documented 106 as more base columns came live in the

# frame window [24000, 48000)); experiments/_audit_liveness.py reproduces it.

# Composition RULED 2026-08-06: _user arms carry [G (20 factor scores) | Ghat
# (20 lead-lag)] = 40 cols (the paper's own design); _doc arms mirror the
# documented construction EXACTLY — Ghat only (trans_exploit.py:64-66), because
# convention fidelity is the point of the _doc ladder.
# NOTE one causal deviation from map_monitor._frame_and_scores, disclosed not silent:
# factor scores are standardized with FROZEN frame-window stats here, where the study
# standardized G full-sample (descriptive code, mild leakage inadmissible in a scored arm).

# Lasso machinery cadence — constants of specs/causal_tune_linear.py:157-171. With
# the single-point FIXED grid below there is NO penalty selection: the periodic
# "tune" reduces to the cold-reseed float-drift re-anchor + the identifiability-
# mask refresh (the lam2=0 singularity fix), both still required.
TUNE_PER = 250  # reseed/mask cadence (solves)
VAL_TAIL = 125  # forward split geometry (unused for selection
EMBARGO = 25  #  with a 1-point grid; kept for the split call)
ESTIMATOR_GRIDS: dict[str, list[tuple[str, float, float]]] = {
    # single-point grid = fixed penalty (directive 2026-08-06; elastic net dropped:
    # §5 is ridge vs lasso)
    "lasso_fixed": [("lasso", FIXED_LASSO_ALPHA, 1.0)],
    # LASSO JIGGLE (author directive 2026-08-07). b2_lasso at alpha=1e-4 beats
    # every other single-estimator arm (0.22950, DM -8.39) — including tuned
    # ridge (0.23040, -4.45), tuned enet (0.23056, -2.58) and tuned lasso
    # (0.23134, -2.03) — but 1e-4 was HAND-PICKED with hindsight, so the number
    # is currently uninterpretable. Two explanations with opposite implications
    # must be separated: (a) 1e-4 is an oracle point and the result is an
    # artifact, or (b) the lasso family genuinely wins on this design and the
    # CAUSAL TUNER is what costs performance (independent evidence: the tuner's
    # selected l1_ratio is bimodal, flipping between 0.25 and 1.0 across ~1200
    # retunes — the signature of a 125-bar validation tail that cannot identify
    # the parameter). THE READOUT is QLIKE against log(alpha): a SHARP peak at
    # 1e-4 with much worse neighbours means luck; a BROAD flat optimum spanning
    # decades means the family wins and the tuner is the underperformer. Each
    # entry is a single-point grid, so there is no selection — identical
    # machinery to b2_lasso (warm Garrigues homotopy at l1_ratio=1.0, the
    # periodic "tune" reducing to the cold-reseed re-anchor + identifiability
    # mask), only alpha moves. 1e-4 is NOT rebuilt here: b2_lasso already IS
    # that point and its chunks stay untouched.
    **{
        f"lasso_fixed_a{tag}": [("lasso", a, 1.0)]
        for tag, a in (
            ("1em6", 1e-6),
            ("1em5", 1e-5),
            ("1em3", 1e-3),
            ("1em2", 1e-2),
        )
    },
    # causally-tuned CONTROL arms (directive 2026-08-06, second round): the July
    # battery's exact grids — specs/causal_tune_linear.py:167-171 ESTIMATOR_GRIDS.
    # With >1 grid point the TUNE_PER=250 reselection is a real causal re-tune:
    # forward_window_split(fit block, EMBARGO=25, VAL_TAIL=125) on the CURRENT
    # training window, argmin validation MSE adopted until the next boundary.
    # ridge alphas: logspace(-2, 3, 6) = 0.01, 0.1, 1, 10, 100, 1000
    "ridge_tuned": [("ridge", float(a), 0.0) for a in np.logspace(-2, 3, 6)],
    # lasso alphas: logspace(-6, -2, 5) = 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 (l1=1.0)
    "lasso_tuned": [("lasso", float(a), 1.0) for a in np.logspace(-6, -2, 5)],
    # elastic net (author directive 2026-08-06: the tuned ridge/lasso near-tie
    # needs the interpolating family) — the battery's documented reclasticnet
    # grid VERBATIM (specs/causal_tune_linear.py:170): alphas logspace(-6,-2,5)
    # at l1_ratio=0.5, sklearn ElasticNet units, warm Garrigues homotopy.
    "enet_tuned": [("enet", float(a), 0.5) for a in np.logspace(-6, -2, 5)],
    # FREE-l1 elastic net (merged-§4/§5 directive 2026-08-07): alphas
    # logspace(-6,-2,5) x l1_ratio {0.25, 0.5, 0.75, 1.0} = 20 combos per
    # retune. The l1_ratio=1.0 rows ARE lasso — the family spans selection
    # strength and the tuner's revealed (alpha, l1_ratio) trajectory (persisted
    # in meta.tuned_penalty) is the section's key exhibit.
    "enet_free": [
        ("enet", float(a), float(l1))
        for a in np.logspace(-6, -2, 5)
        for l1 in (0.25, 0.5, 0.75, 1.0)
    ],
}

# REACH-MATCHED elastic-net grid (author directive 2026-08-07). THIS IS A GRID
# DEFECT REPAIR, NOT A NEW HYPOTHESIS — the paper's "shrinkage beats selection"
# conclusion currently rests on a head-to-head between two families whose
# penalty grids do not span the same shrinkage.
#
# THE ARITHMETIC. reclasso's sklearn-compatible mapping (see
# src/models/reclasso_har, module docstring) over N = 24000 window rows is
#     mu   = N * alpha * l1_ratio        (L1)
#     lam2 = N * alpha * (1 - l1_ratio)  (L2, the ridge-equivalent)
# `enet_free` tops out at alpha = 1e-2, so its LARGEST expressible
# ridge-equivalent penalty is 24000 * 1e-2 * 0.75 = 180 at l1_ratio=0.25, and
# less at every heavier mixing. The tuned RIDGE grid reaches alpha = 1000 and
# SELECTS that top point in 41.3% of retunes. The elastic net therefore cannot
# reach the shrinkage level ridge picks: the two arms are not comparable at the
# margin where ridge actually operates.
#
# THE MEASURED SYMPTOM: the enet's pooled deficit vs tuned ridge is entirely
# concentrated at its own grid ceiling. On the 72.1% of bars where the tuner
# picked alpha <= 1e-3 the ENET WINS (d = -0.00039, DM -6.70, negative in 8/8
# designs); on the 27.9% where it picked alpha = 1e-2 it loses by +0.00250
# (DM +6.85, 8/8 designs, 18/24 years). Alpha-endpoint DiD +0.00289, stacked
# z = +7.78. Once alpha is controlled the l1_ratio heavy-vs-light effect halves
# and survives in only 1 of 8 designs, with a sign reversal at alpha=1e-4 — so
# MIXING IS A SYMPTOM, NOT THE CHANNEL. The losses are where it ran out of grid.
#
# THE FIX. The requirement is that EVERY mixing value be able to express the
# L2 weight ridge selects, since ridge pins alpha=1000 in 41.3% of retunes.
# Inverting lam2 = N * alpha * (1 - l1_ratio) at lam2 = 1000, N = 24000:
#     l1=0.25 -> alpha = 1000 / (24000 * 0.75) = 0.0556
#     l1=0.50 -> alpha = 1000 / (24000 * 0.50) = 0.0833
#     l1=0.75 -> alpha = 1000 / (24000 * 0.25) = 0.1667
#     l1=1.00 -> UNREACHABLE at any alpha (no L2 term at pure lasso)
# The l1=0.75 row is the binding one at 0.1667, so a top of 1e-1 would NOT
# suffice — it would leave the reach confound in place at exactly one mixing
# value, which is the sort of residual that later gets mistaken for a family
# effect. Top = 1e0, i.e. alphas = np.logspace(-6, 0, 7). Maximum expressible
# lam2 at that top (= 24000 * 1 * (1 - l1)):
#     l1=0.25 -> 18000   (18.0x ridge's selected 1000)
#     l1=0.50 -> 12000   (12.0x)
#     l1=0.75 ->  6000   ( 6.0x)
#     l1=1.00 ->     0   (structural, see below)
# Comfortably past ridge's reach at every mixing value where the comparison is
# defined, so an endpoint selection at 1e0 now MEANS something: that the enet
# genuinely wants more shrinkage than ridge's own grid offers, rather than that
# it ran out of room.
#
# STRUCTURAL LIMIT AT PURE LASSO, stated so no one later reads it as a second
# grid defect: at l1_ratio=1.0 the L2 term is identically zero for EVERY alpha,
# so pure lasso can never match ridge's shrinkage no matter how far the alpha
# axis extends. That is the definition of the family. This grid removes the
# reach confound at l1_ratio < 1 only; any residual pure-lasso deficit that
# survives is a genuine family limitation and may be reported as one.
#
# CONSTRUCTED AS A STRICT SUPERSET, by extension rather than by rebuilding the
# axis: the original 20 points are carried through as the SAME tuples, in the
# SAME order, so (a) set(enet_free) < set(enet_free_wide) holds bit-exactly and
# the wide-vs-narrow increment is attributable to the added points alone, and
# (b) the tuner's first-minimum-wins tie-break over the original points is
# unchanged. The resulting alpha axis equals np.logspace(-6, 0, 7); that is
# asserted in the synthetic rather than assumed.
#
# DEGENERATE CORNER, verified on a fixture (see the note in
# RollingTunedLinear._tune and section K of the verify script): alpha=1e0 at
# l1_ratio=1.0 gives mu = 24000 * 1 * 1 = 24000, far above mu_max = max|X'y|,
# so the homotopy returns an EMPTY active set. The forecast is still
# well-defined — the intercept is a locked, unpenalized augmented column, so
# the prediction degrades to the intercept-only limit (window mean of y) — and
# the frequency is disclosed in meta.tuned_penalty_summary.frac_intercept_only.
ENET_WIDE_ALPHA_ADDED: tuple[float, ...] = (1e-1, 1e0)
ESTIMATOR_GRIDS["enet_free_wide"] = list(ESTIMATOR_GRIDS["enet_free"]) + [
    ("enet", float(a), float(l1))
    for a in ENET_WIDE_ALPHA_ADDED
    for l1 in (0.25, 0.5, 0.75, 1.0)
]


def _halve_decades(grid: tuple[float, ...]) -> tuple[float, ...]:
    """Insert the GEOMETRIC MIDPOINT between consecutive grid points.

    Subset-by-construction: the original points are carried through as the
    SAME float64 objects, so ``set(coarse) < set(fine)`` holds bit-exactly and
    a fine-vs-coarse increment is attributable to the interstitial points
    alone. (Rebuilding the fine grid with ``np.logspace`` would usually — but
    is not GUARANTEED to — reproduce the coarse points bit-identically; that
    is not a property to leave to luck when it is the comparison's premise.)
    """
    out: list[float] = []
    for a, b in zip(grid[:-1], grid[1:]):
        out.append(float(a))
        out.append(float(np.sqrt(float(a) * float(b))))
    out.append(float(grid[-1]))
    return tuple(out)


# HALF-DECADE tuner grids (author directive 2026-08-07). THE ARITHMETIC: the
# fixed-ridge envelope moves ~0.0009 in QLIKE per DECADE of alpha (0.1: 0.23360,
# 0.3: 0.23312, 1: 0.23243, 3: 0.23169, 10: 0.23087), so a half-decade error in
# the selected penalty costs ~0.0004 — LARGER than increments this paper reports
# as significant (the transmission increment is 0.00028 at DM -2.06). The tuned
# grids are decade-spaced (ridge logspace(-2,3,6)) or 3-point (the block grids
# the BEST MODEL is tuned on). We are resolving the hyperparameter more coarsely
# than the effects we measure.
# COUNTER-CONSIDERATION, and why this is an empirical trade rather than an
# obvious win: a finer grid is also more opportunities to overfit the 125-bar
# validation tail, and there is direct evidence that tail is a noisy selector —
# the enet arms' chosen l1_ratio is BIMODAL, flipping between 0.25 and 1.0
# across ~1200 retunes. These arms must be able to show a LOSS, and the
# interstitial-usage diagnostic (meta.coarse_grids -> the scorer's
# fine_grid_usage.csv) is what makes either outcome interpretable:
#   * interstitial points rarely selected      -> resolution was never binding,
#                                                 the coarse grid is vindicated;
#   * selected constantly, no QLIKE gain       -> selection noise, i.e. evidence
#                                                 FOR the coarse grid as
#                                                 implicit regularization.
ESTIMATOR_GRIDS["ridge_tuned_fine"] = [
    ("ridge", float(a), 0.0)
    for a in _halve_decades(tuple(a for _, a, _ in ESTIMATOR_GRIDS["ridge_tuned"]))
]
# COARSE ancestry for the single-estimator fine arm (same role as
# FINE_GRID_PARENT for the block arms): persisted into meta.coarse_grids.
ESTIMATOR_GRID_PARENT: dict[str, str] = {"ridge_tuned_fine": "ridge_tuned"}

# Per-block alpha grids for the causally-tuned BLOCK arms (author directive
# 2026-08-06: the block ridges get the same fairness standard as the tuned
# head-to-head). RATIONALE: 3 log-spaced points per block, centered so each
# grid SPANS BOTH prior conventions — backbone: both ladders used 1 (grid
# brackets it); linear exog: user 100 vs doc 3e3 (grid 1e2..1e4 contains
# both); product: user 1e3 vs doc 3e4 (grid 1e3..1e5); transmission: user
# 1e3 vs doc 3e3 (grid 1e2..1e4 contains both). Selection is JOINT (full
# cartesian product per retune) so cross-block penalty trade-offs are seen.
# NOTE the value type is Any, not float: the SHAPED block grids (trans_shaped*,
# pc_ladder_tilt*, exog_tilt*) store descriptor TUPLES under the same mapping,
# e.g. (lambda0, gamma) or (lambda0, family, param[, group]). _pen_value
# normalizes whichever form a key holds; _fill_pen_span dispatches on it.
BLOCK_TUNE_GRIDS: dict[str, tuple[Any, ...]] = {
    "backbone": (0.1, 1.0, 10.0),
    "exog": (1e2, 1e3, 1e4),
    "product": (1e3, 1e4, 1e5),
    "trans": (1e2, 1e3, 1e4),
    "exog_timeaware": tuple(
        (float(lam0), "timescale", float(g))
        for lam0 in (1e2, 1e3, 1e4)
        for g in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    ),
}

# Penalty-allocation ladder (author directive 2026-08-07): 1 penalty (uniform)
# -> 4 (per-block, the current champion) -> PER-BUCKET (this rung) -> per-column
# (rejected: 526 penalties are unselectable on a 125-bar tail — that is
# estimation/ARD territory, and ARD is selection-like, which loses in this
# paper). Each canonical exogenous FAMILY gets its own penalty, drawn from the
# same grid the single exog block used, so the rung nests the champion exactly
# (all bucket penalties equal == the 4-block allocation).
CYCLIC_PASSES = 3

# Rank-SHAPED transmission penalty (author directive 2026-08-07): the K-ladder
# collapse at K=40 (increments vs blk3_user: K5 -17.7e-5, K10 -23.5e-5,
# K20 -30.9e-5, K40 -1.9e-5 n.s. and worse than K10 at t=2.62) may be a
# PENALTY-ALLOCATION artifact, not a signal boundary: the block carries ONE
# shared penalty, so widening the frame forces the tuner to over-shrink every
# direction. Shaped alternative, for factor i = 1..K in EIGEN-RANK order
# (1 = largest eigenvalue): lambda_i = lambda0 * i**gamma. gamma=0 reduces
# EXACTLY to the flat block penalty (i**0 == 1.0 bit-exactly), which is what
# makes the comparison honest. (lambda0, gamma) is ONE block's 12-point grid —
# selected by the same cyclic descent, never a cartesian explosion.
TRANS_SHAPE_GAMMAS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
BLOCK_TUNE_GRIDS["trans_shaped"] = tuple(
    (float(lam0), float(g))
    for lam0 in BLOCK_TUNE_GRIDS["trans"]
    for g in TRANS_SHAPE_GAMMAS
)

# ENDPOINT-RELIEF grid (author directive 2026-08-07). The shaped arm's tuner
# picks gamma=2 — the TOP of TRANS_SHAPE_GAMMAS — in ~45% of its 1092 retunes,
# and an optimum sitting at a grid endpoint is not a valid selection: it says
# the grid, not the data, chose the answer. This constant extends the exponent
# axis to 3 and 4 (18 (lambda0, gamma) points) so the same causal tuner can
# express a STEEPER tilt if it wants one. TRANS_SHAPE_GAMMAS itself is left
# untouched on purpose — blk4_trailGShaped's results are on disk and must stay
# reproducible byte-for-byte, so the wide grid lives under its own block key
# and its own arm. gamma=0 is retained, so the wide grid still nests the flat
# penalty exactly and STRICTLY CONTAINS the original 12 points (any wide-vs-
# narrow difference is attributable to the two added exponents alone).
#
# HOW TO READ A SELECTED GAMMA (derived + measured, same verify script,
# section D). The transmission scores are TRAILING-STANDARDIZED, i.e. score i
# is divided by its own rolling sd ~ sqrt(d_i) with d_i the frame eigenvalue.
# If c_i is the coefficient on the standardized column and b_i the coefficient
# on the RAW score, b_i = c_i / sqrt(d_i), so a penalty lambda_i on c IS a
# penalty lambda_i * d_i on b. The standardization is therefore ITSELF a
# spectral tilt. Writing the spectrum as d_i ~ c * i**-a:
#     gamma_eff (on the raw eigen-directions) = gamma - a
# so gamma=0 is ALREADY a tilt of -a (leading directions shrunk HARDEST),
# gamma=a is the genuinely FLAT point, and a grid that stops at gamma=2 can
# barely express a positive tilt at all. That is the mechanical reason the
# tuner pins BOTH ends of the old grid.
# `a` IS NOT A SINGLE NUMBER — do not quote one (correction 2026-08-07). It is
# truncation-dependent, and the log-log R^2 = 0.981 of the top-40 fit hides
# real curvature; refits on nested truncations of the frozen frame give
#     a = 1.111 (top-5), 0.983 (top-10), 1.018 (top-20), 1.176 (top-40)
# i.e. the tail STEEPENS. Anywhere this file quotes 1.176 it means the top-40
# frozen-frame fit specifically, which is the right scale only for the K=40
# blocks. On the causally REFRESHED frame — the only arm with genuinely
# distinct spectra, blk4_trailRefresh, 76 of them — a = 0.956 +/- 0.052.
# (The frozen arms' persisted trans_eigvals is ONE spectrum replicated across
# all 91 chunks, verified 1 unique row, so no stability claim may cite them.)
# The practical consequence: gamma_eff is a useful REFRAMING, not a calibrated
# offset, and the grid must bracket the flat point from both sides rather than
# assume where it sits — which is what the bipolar axis does.
#
# NUMERICAL NOTE — MEASURED, NOT ASSUMED (experiments/verify_unification_shapes
# .py section C). At the extreme corner lambda0=1e4, gamma=4, K=40 the largest
# block penalty is 1e4 * 40**4 = 2.56e10: EXACT in float64 (an integer power —
# no overflow, no rounding), and ~3 decades above the shipped arm's worst case
# (1e4 * 40**2 = 1.6e7). On a worst-case fit gram at the campaign's own scale
# (23,850 fit rows x 906 columns, exogenous block a power-law factor model,
# backbone deliberately made numerically singular so lambda_min is pinned at
# the 0.1 penalty FLOOR — the least favourable configuration the tuner can
# reach) the penalized gram's condition number rises from 1.6e8 at gamma=2 to
# 2.6e11 at gamma=4, and the fitted values from the normal-equation solve the
# code performs still agree with the numerically stable augmented-QR ridge
# solve to 1.2e-13 RELATIVE (4.9e-15 at gamma=2). SO NO CAP IS APPLIED: the
# large condition number is entirely due to the huge penalties themselves, and
# the directions carrying them are precisely the ones being shrunk to zero, so
# they contribute nothing to the fit that could be corrupted. (Individual
# coefficients of the near-duplicate backbone pair carry ~6e-10 relative
# error, but that is the collinear pair's own indeterminacy — it is identical
# at the FLAT penalty and is not caused by the shaping.) If a future grid goes
# steeper still, the documented fix is to cap lambda_i at a fixed multiple of
# the fit gram's trace rather than to drop grid points silently.
# HALF-DECADE block grids for blk4_trailGShaped_fine (author directive
# 2026-08-07): every block refined over the SAME range it already spans, so the
# arm differs from blk4_trailGShaped in RESOLUTION ONLY. Each coarse grid is a
# strict subset (see _halve_decades). Cyclic cost is 1 + passes * sum(|grid|-1)
# = 1 + 3 * (4+4+4+19) = 94 tail evaluations per retune, against the shipped
# arm's 52 — under 2x, and well inside the ~150 ceiling.
BLOCK_TUNE_GRIDS["backbone_fine"] = _halve_decades(BLOCK_TUNE_GRIDS["backbone"])
BLOCK_TUNE_GRIDS["exog_fine"] = _halve_decades(BLOCK_TUNE_GRIDS["exog"])
BLOCK_TUNE_GRIDS["product_fine"] = _halve_decades(BLOCK_TUNE_GRIDS["product"])
BLOCK_TUNE_GRIDS["trans_fine"] = _halve_decades(BLOCK_TUNE_GRIDS["trans"])
# Half-decade lambda0 axis for the shaped transmission block, gammas UNCHANGED
# (the resolution question is about the PENALTY LEVEL; changing both axes at
# once would confound it): 5 lambda0 x 4 gammas = 20 points, coarse 12 a strict
# subset.
BLOCK_TUNE_GRIDS["trans_shaped_fine"] = tuple(
    (float(lam0), float(g))
    for lam0 in BLOCK_TUNE_GRIDS["trans_fine"]
    for g in TRANS_SHAPE_GAMMAS
)
# COARSE ancestry, persisted into every chunk's meta so the interstitial-usage
# diagnostic needs no scorer-side hardcoding: a selected penalty is
# "interstitial" iff its level is NOT a point of the coarse grid this one
# refines.
FINE_GRID_PARENT: dict[str, str] = {
    "backbone_fine": "backbone",
    "exog_fine": "exog",
    "product_fine": "product",
    "trans_shaped_fine": "trans_shaped",
}

TRANS_SHAPE_GAMMAS_WIDE: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)

# BIPOLAR exponent axis — the shared relief for a TWO-SIDED endpoint pathology
# (author directive 2026-08-07, second round). MEASURED on blk4_trailGShaped's
# 1092 persisted retunes: 45% sit at the exponent grid FLOOR (gamma=0) and 41%
# at the CEILING (gamma=2). An upward-only extension relieves half the problem
# and leaves the larger half in place.
# NEGATIVE EXPONENTS ARE COHERENT HERE, which is why the floor is extended
# rather than merely documented. lambda_i = lambda0 * i**gamma stays strictly
# positive and monotone for gamma<0; it simply shrinks the LEADING directions
# hardest and lets the tail run. That is not an exotic prior on this design —
# it is what the trailing standardization already does implicitly (see the
# gamma_eff note below), so gamma=0 is an interior point of the mechanism, not
# a natural boundary, and a tuner pinned there in 45% of retunes is asking to
# go further in a direction the grid does not offer.
# gamma=0 is retained, so the bit-exact flat nesting survives and the axis is a
# strict superset of the original (0, 0.5, 1, 2).
TRANS_SHAPE_GAMMAS_BIPOLAR: tuple[float, ...] = (
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
)
# blk4_trailGShapedWide / blk4_trailGShapedFrozen carry this: 8 gammas x 3
# lambda0 = 24 points. Safe at this width — the block is K=40, so the steepest
# corner is lambda0*40**4 = 2.56e10 (cond 2.6e11, fitted agreement 1.2e-13,
# measured) and the shallowest is lambda0/40 (cond 1.3e6).
BLOCK_TUNE_GRIDS["trans_shaped_wide"] = tuple(
    (float(lam0), float(g))
    for lam0 in BLOCK_TUNE_GRIDS["trans"]
    for g in TRANS_SHAPE_GAMMAS_BIPOLAR
)
# PCR block grid (author directive 2026-08-07). DELIBERATELY WIDER than the
# `trans` grid: that one is calibrated for a factor block competing against
# 526 exogenous columns, where the tuner must shrink it hard to stop it
# duplicating their span. With the wide block REMOVED the factor block is the
# only exogenous information in the model, so the appropriate penalty scale is
# unknown a priori and plausibly orders of magnitude smaller. Grid scale is
# configuration-dependent; a grid whose optimum sits at an endpoint is not a
# valid selection, so this spans seven decades and the scorer flags endpoint
# pile-up (see meta.tuned_grids + the penalty summary's frac_at_grid_* cols).
# WIDENED levels grid for the CAPTURE-LADDER arms (2026-08-08): the `trans`
# grid (1e2..1e4) was calibrated for a factor block competing against the
# 526-column wide design. With the exogenous block ABSENT the levels are the
# only exogenous information, so the appropriate scale is plausibly lower.
# Same extension, same evidence: with the exogenous block absent the levels
# carry ALL the exogenous information, so if anything they want MORE reach than
# the rotated-exog block did. logspace(0,6,7); the previous 1e0..1e4 points are
# a strict subset.
BLOCK_TUNE_GRIDS["trans_sub"] = tuple(float(a) for a in np.logspace(0, 6, 7))
BLOCK_TUNE_GRIDS["pcr"] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
# Rotated-exogenous grid for the TUNED PCR twins (author directive 2026-08-08):
# logspace(-1, 4, 6) = 0.1 .. 1e4 — ONE DECADE ABOVE ridge's usual top,
# because the tuned-ridge family pins its modal alpha at 1000 in 8/8 bucket
# designs. If 1e4 pins too, the endpoint diagnostic says so and the grid moves
# again; that is the point of putting the extra decade there rather than
# assuming 1000 was the optimum.
# EXTENDED on canary evidence (2026-08-08): the first cut, logspace(-1,4,6),
# was PINNED at its 1e4 top in 67% of retunes — the endpoint gate did its job
# and the grid, not the data, was setting the penalty. Two more decades.
# NOTE the requested logspace(0,6,7) would have DROPPED the old 0.1 point,
# which contradicts keeping the previous grid as a subset; 0.1 is retained so
# the extension can only ADD reach. Any wide-vs-narrow read stays attributable
# to the added decades, and 0.1 costs three tail evaluations per retune.
BLOCK_TUNE_GRIDS["rotexog_wide"] = (0.1,) + tuple(
    float(a) for a in np.logspace(0, 6, 7)
)
# PC-ladder block: one rank-tilted penalty per PC rank, SHARED across that
# rank's ladder rungs (see _pc_ladder_design). Power family only, 12 points;
# the group size (rungs per rank) rides in the descriptor.
BLOCK_TUNE_GRIDS["pc_ladder_tilt"] = tuple(
    (float(lam0), "pcrank", float(g), len(PRODUCT_EXOG_WINDOWS))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS
)
# WIDE-exponent twin for the K-sweep PC-ladder arms (author directive
# 2026-08-07): at full rank the tail directions are the ones the tilt has to
# suppress, so the exponent axis must be able to go steeper than 2 — same
# extension, same rationale, as TRANS_SHAPE_GAMMAS_WIDE. 18 points.
BLOCK_TUNE_GRIDS["pc_ladder_tilt_wide"] = tuple(
    (float(lam0), "pcrank", float(g), len(PRODUCT_EXOG_WINDOWS))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS_WIDE
)

# Same BIPOLAR axis for the PC-ladder rung (author directive 2026-08-07, after
# the shape-endpoint diagnostic fired on real data). MEASURED PINNING, read off
# blk_pcladder_tuned's 1092 persisted retunes: the pcrank exponent is selected
# at gamma=0 in 49% of retunes and at gamma=2 in 32% — 81% at an ENDPOINT, with
# the LOW end the larger of the two, the same two-sided pattern as
# blk4_trailGShaped. The coherence argument for gamma<0 is identical and lives
# at TRANS_SHAPE_GAMMAS_BIPOLAR; these scores are trailing-standardized too.
# SAFE AT THIS WIDTH: the pcrank penalty is indexed by PC RANK, and this arm
# has K=20 ranks, so the steepest corner is lambda0 * 20**4 = 1.6e9 at
# lambda0=1e4 — measured cond 1.6e10, fitted agreement 3.4e-14 (section C3).
BLOCK_TUNE_GRIDS["pc_ladder_tilt_bipolar"] = tuple(
    (float(lam0), "pcrank", float(g), len(PRODUCT_EXOG_WINDOWS))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS_BIPOLAR
)

# The same (level, tilt) grid for the EXOGENOUS block — the generalized
# Tikhonov arm (2026-08-07): anisotropic ridge applied directly, with no
# duplicated columns, as the principled form of what the transmission block
# achieves by augmentation. gamma=0 is plain scalar ridge.
BLOCK_TUNE_GRIDS["exog_tilt"] = tuple(
    (float(lam0), float(g))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS
)

# HARD-vs-SOFT tilt, decided WITHIN one arm (author directive 2026-08-07).
# PCR is the hard-threshold limit of a rank-tilted penalty (weight alpha on the
# top K, effectively infinite beyond); the power law is a SMOOTH tilt, and with
# gamma in {0, .5, 1, 2} it cannot approximate a cutoff. Comparing the two
# ACROSS arms is confounded — they differ in design AND in which space the
# eigenbasis lives (transmission factors are PCs of the 133-column base;
# the Tikhonov rotation is of the ~526-column exog design). Putting both
# families in ONE block grid makes the causal tuner choose, per retune.
# STEP_MULTIPLIER is a FINITE stand-in for truncation: at 1e4 x alpha the
# beyond-K directions are effectively truncated relative to the signal scale,
# while the solve stays well-conditioned and the gamma=0 scalar-ridge nesting
# is preserved — literal infinity would forfeit both.
STEP_MULTIPLIER = 1e4
TIKHONOV_STEP_KS: tuple[int, ...] = (20, 40, 80)

# SHAPE ZOO for the K=40 transmission block (author directive 2026-08-07):
# three penalty FAMILIES in one block grid, so the causal tuner — not the
# author — picks the prior, per retune.
#   power       lambda_i = lambda0 * i**gamma          gamma in the wide grid
#   exponential lambda_i = lambda0 * exp(kappa*(i-1))  kappa in TRANS_SHAPE_KAPPAS
#   step        lambda0 up to rank K0, lambda0*M after K0 in TRANS_SHAPE_STEP_KS
# WHY THESE THREE AND NOT AN EIGENVALUE PROFILE: the frozen frame's spectrum is
# a power law over the top 40 (d_i ~ c*i^-a with a = 1.176 on THAT truncation
# — a is truncation-dependent, see TRANS_SHAPE_GAMMAS_WIDE), so
# lambda_i ∝ d_i^-theta IS the rank power law with gamma = a*theta. Running it
# would be a reparameterization, not an experiment. What a power-law spectrum does NOT already contain is a
# geometric tail (exponential) or a hard cutoff on this basis (step).
# 12 shape points x 3 lambda0 = 36 points, one block, cyclic selection.
# THE EXHIBIT is the selected FAMILY by retune and by year: a dominant family
# is this panel's revealed prior; a family that flips by era is a statement
# about regime-dependent effective dimension.
TRANS_SHAPE_KAPPAS: tuple[float, ...] = (0.02, 0.05, 0.10)
TRANS_SHAPE_STEP_KS: tuple[int, ...] = (10, 20, 30)
BLOCK_TUNE_GRIDS["trans_shaped_zoo"] = (
    tuple(
        (float(lam0), "power", float(g))
        for lam0 in BLOCK_TUNE_GRIDS["trans"]
        for g in TRANS_SHAPE_GAMMAS_WIDE
    )
    + tuple(
        (float(lam0), "exp", float(kap))
        for lam0 in BLOCK_TUNE_GRIDS["trans"]
        for kap in TRANS_SHAPE_KAPPAS
    )
    + tuple(
        (float(lam0), "step", float(k0))
        for lam0 in BLOCK_TUNE_GRIDS["trans"]
        for k0 in TRANS_SHAPE_STEP_KS
    )
)
BLOCK_TUNE_GRIDS["exog_tilt_step"] = tuple(
    (float(lam0), "power", float(g))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS
) + tuple(
    (float(lam0), "step", float(k))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for k in TIKHONOV_STEP_KS
)

# WIDE twin of the grid above (author directive 2026-08-07, after the
# shape-endpoint diagnostic fired). MEASURED PINNING over blk3_tikhonovStep_
# tuned's 1092 persisted retunes — the two families pin in OPPOSITE directions,
# which is why each is extended asymmetrically rather than both blindly widened:
#   power  gamma 0: 56%, 0.5: 19%, 1: 7%, 2: 18%   -> 74% at an endpoint,
#          dominated by the LOW end (gamma=0, the grid floor).
#   step   K0 20: 23%, 40: 29%, 80: 48%            -> 71% at an endpoint,
#          dominated by the HIGH end (K0=80, the grid ceiling).
# POWER, extended DOWN to -1: unlike the transmission block, the rotated
# exogenous design is NOT standardization-tilted — _exog_tilt_design applies an
# ORTHOGONAL rotation with no rescaling, so here gamma_eff == gamma exactly and
# gamma=0 is literally plain scalar ridge. A tuner pinned at 0 in 56% of
# retunes is asking for a tilt in the direction the grid does not offer:
# shrink the LEADING directions harder, not the tail.
# POWER, capped at 3 — a DELIBERATE, MEASURED deviation from the requested
# (0..4), documented rather than silently dropped. The power family here spans
# the WHOLE ~526-column rotated design, so its dynamic range is lambda0*526**g,
# not lambda0*40**g:
#     gamma=2 -> 2.77e9   (shipped)      cond 2.8e10   fitted agreement 1.1e-13
#     gamma=3 -> 1.46e12                 cond 1.5e13   fitted agreement 1.2e-12
#     gamma=4 -> 7.65e14                 cond 5.3e16   fitted agreement 3.4e-11
# cond 5.3e16 EXCEEDS 1/eps = 4.5e15: at gamma=4 the penalized gram is
# numerically singular to working precision. The fitted values happen to
# survive (the over-penalized directions are annihilated, so nothing corrupt
# reaches the score), but shipping a grid point whose normal-equation solve is
# formally meaningless is not something to do quietly. It also buys nothing
# scientifically: lambda_1/lambda_526 = 7.6e10 at gamma=4 is not "a tilt", it
# is hard truncation at rank ~10 — and hard truncation is ALREADY in this same
# block grid, explicitly and interpretably, as the step family. The exponent
# ceiling is a function of BLOCK WIDTH, which is why the 40-wide transmission
# block safely carries gamma=4 (corner 2.56e10, cond 2.6e11) and this one does
# not. No cap constant is introduced: capping lambda_i would be exactly the
# kind of clip that bites and destroys the signal-carrying extreme.
TIKHONOV_GAMMAS_WIDE: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
# STEP, extended BOTH ways and kept a STRICT SUPERSET of TIKHONOV_STEP_KS.
# NOTE the requested (5,10,20,30,60,100) drops 40 and 80 — the two points the
# shipped arm actually selects most (29% and 48%) — which would have made the
# wide-vs-narrow increment uninterpretable: a difference could then come from
# REMOVING the incumbent optimum rather than from adding reach. Same cardinality
# (6), same both-ended widening, superset restored.
TIKHONOV_STEP_KS_WIDE: tuple[int, ...] = (5, 10, 20, 40, 80, 100)
BLOCK_TUNE_GRIDS["exog_tilt_step_wide"] = tuple(
    (float(lam0), "power", float(g))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TIKHONOV_GAMMAS_WIDE
) + tuple(
    (float(lam0), "step", float(k))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for k in TIKHONOV_STEP_KS_WIDE
)


def _exog_tilt_design(p: _Panel, window: int) -> np.ndarray:
    """Exogenous design ROTATED into its frozen eigenframe (generalized
    Tikhonov arm, author directive 2026-08-07).

    THE EQUIVALENCE (documented because the arm's whole claim rests on it):
    the principled object is anisotropic ridge on the ORIGINAL columns,

        min ||y - X_e b||^2 + b' Gamma b,   Gamma = V diag(w) V',
        w_i = alpha * i**gamma  (i = eigen-rank, 1 = largest eigenvalue)

    with V the eigenvectors of the FROZEN first-window correlation of the
    exogenous columns. Because V is ORTHOGONAL, substituting b = V c gives
    X_e b = (X_e V) c and b' Gamma b = c' diag(w) c — i.e. rotating the
    columns ONCE and applying the DIAGONAL penalty diag(w) in the rotated
    coordinates is EXACTLY the dense-Gamma problem, at the cost of the
    existing diagonal-penalty solver (no new solver machinery, no per-bar
    dense add). Coefficients in original coordinates are V c if ever needed;
    fitted values and hence every scored quantity are identical.

    NO SCALING is applied in the rotation: V is orthogonal, so ||b||^2 is
    preserved and gamma=0 (w_i == alpha for all i) reproduces the plain
    scalar-ridge problem. Standardizing here would silently change the
    penalty and break that nesting.

    NESTING PRECISION (stated so nobody later reads the residue as a bug):
    at gamma=0 this arm is ALGEBRAICALLY EXACT against the scalar-penalty
    path of blk3_tuned, and agrees with it TO MACHINE PRECISION — measured
    ~3e-15 max absolute difference on fitted values — NOT bit-for-bit. An
    orthogonal reparameterization is exact in real arithmetic but not in
    float64; the penalty VECTOR is bit-identical (see _fill_pen_span at
    gamma=0), the ~1e-15 residue is the rotation itself. Recovering literal
    bit-identity would need a special-cased dense-Gamma branch that buys
    nothing.

    Frame convention MIRRORS ``_transmission_block._frame_of`` (same frame
    window [window, 2*window), the same ``_DEGENERATE_SD`` liveness test, the
    same eigh + descending-eigenvalue ordering). It is deliberately a
    SEPARATE code path rather than a refactor of that function: transmission
    arms are in flight, and any reordering of their float operations would
    silently perturb running results. Dead (zero-dispersion) columns carry no
    spectrum, so they pass through UNROTATED and are appended after the live
    rotated block — they occupy the highest eigen-ranks, i.e. the most
    heavily penalized positions under gamma>0, which is the correct treatment
    for directions with no variance to explain.
    """
    return _rotate_frozen(
        np.ascontiguousarray(p.X[:, _exog_all_cols(p.names)], dtype=np.float64),
        window,
        2 * window,
    )


def _rotate_frozen(z: np.ndarray, f0: int, f1: int) -> np.ndarray:
    """Orthogonally rotate a design into the frozen eigenframe of its own
    first-window CORRELATION matrix (shared by the Tikhonov-family arms).

    Same convention as ``_transmission_block._frame_of``: frame window rows
    [f0, f1), ``_DEGENERATE_SD`` liveness, eigh + descending-eigenvalue order.
    Live columns are rotated by the orthogonal V (RAW columns — no scaling, so
    ||b||^2 is preserved and gamma=0 nests scalar ridge exactly); dead columns
    pass through unrotated and are appended at the highest eigen-ranks.
    """
    zw = z[f0:f1]
    sd = zw.std(0)
    live = sd > _DEGENERATE_SD
    if not live.any():
        raise SystemExit(
            "tilt frame: no live columns in the frame window; refusing to "
            "build a degenerate rotation"
        )
    sdl = np.where(live, sd, 1.0)
    mu = zw.mean(0)
    lam, v_l = np.linalg.eigh(np.corrcoef(((zw - mu) / sdl)[:, live], rowvar=False))
    v_mat = v_l[:, np.argsort(lam)[::-1]]
    out = np.empty_like(z)
    n_live = int(live.sum())
    out[:, :n_live] = z[:, live] @ v_mat
    out[:, n_live:] = z[:, ~live]
    return out


def _frame_live_rank(p: _Panel, window: int) -> int:
    """FULL rank available to the frozen transmission/PC frame, read from the
    frame construction's own liveness rule — never hardcoded.

    ``_transmission_block._frame_of`` builds the frame from the correlation of
    the LIVE base columns of the frame window ``Z[window:2*window]``, where
    live means ``std > _DEGENERATE_SD``; the eigendecomposition therefore
    yields exactly ``live.sum()`` directions and ``_frame_of`` already fails
    LOUDLY when a requested K exceeds that. This function reproduces that count
    (same columns, same window, same threshold) so the full-rank PC-ladder arm
    can ASK for it instead of assuming a number that drifts with the panel.
    """
    bc = _product_base_cols(p.names)
    zw = np.ascontiguousarray(p.X[window : 2 * window, bc])
    n_live = int((zw.std(0) > _DEGENERATE_SD).sum())
    if n_live < 1:
        raise SystemExit(
            "PC frame: no live base columns in the frame window — refusing to "
            "build a degenerate full-rank design"
        )
    return n_live


def _pc_ladder_design(
    p: _Panel,
    window: int,
    qpool: int = TRANS_QPOOL,
    rungs: tuple[int, ...] = PRODUCT_EXOG_WINDOWS,
) -> np.ndarray:
    """Ladder-expanded principal components: columns {ma_j(G_i)}.

    CONSTRUCTION (author correction 2026-08-07): the moving-average ladder is
    applied TO THE EIGENVECTOR SERIES, not to the raw features. Factor scores
    G_i(t) are the projection of the BASE feature vector onto eigenvector i of
    the frozen first-window frame — the SAME frame ``_transmission_block``
    builds — and each score is then expanded through the ladder rungs.

    WHY THIS IS THE RIGHT ORGANIZATION: moving averages and linear projections
    commute, ma_j(V'x) = V' ma_j(x), so these columns span exactly the
    eigen-projection of the ladder-expanded design — the SAME subspace as
    rotating the expanded design, but organized so every column carries a
    two-part identity (PC rank x horizon). That is what makes a rank-tilted
    penalty meaningful: ONE penalty per PC rank, SHARED across that rank's
    ladder rungs. The tilt is a prior over DIRECTIONS, not over horizons.

    ORDER OF OPERATIONS (deliberate, asserted in the checks): the scores are
    TRAILING-STANDARDIZED FIRST — the same standardization the winning
    transmission arms use — and the ladder is applied to the standardized
    series. standardize-then-ladder and ladder-then-standardize are NOT the
    same operation: the former gives every rung a common, causally-rescaled
    input so a rank's rungs differ only in horizon, which is precisely the
    factorization the shared per-rank penalty assumes.

    COLUMN LAYOUT: grouped by PC rank, rungs contiguous within a rank —
    column (i-1)*n_rungs + j is ma_{rungs[j]}(G_i). ``_fill_pen_span``'s
    'pcrank' family relies on this grouping. Rungs are the base's own
    fast/intraday/slow tri (``PRODUCT_EXOG_WINDOWS``), so K=20 gives 60
    columns. Ladder convention matches ``generate_har_features``: causal
    rolling mean, ``min_periods=1``, shifted one bar.
    """
    g_scores = _transmission_block(
        p, window, parts="scores", standardization="trailing", qpool=qpool
    )
    out = np.empty((g_scores.shape[0], qpool * len(rungs)), dtype=np.float64)
    for i in range(qpool):
        s = pd.Series(g_scores[:, i])
        for j, w in enumerate(rungs):
            out[:, i * len(rungs) + j] = (
                s.rolling(window=int(w), min_periods=1).mean().shift(1).to_numpy()
            )
    out[~np.isfinite(out)] = 0.0  # the single shifted warm-up row
    return out


# Alignment diagnostic of the last per-rung build (mean |dot| of matched
# eigenvectors and the principal-angle cosines between the extreme rungs);
# read by the tests and printed at build time.
_LAST_PERRUNG_DIAG: dict[str, Any] = {}


def _pc_ladder_perrung_design(
    p: _Panel,
    window: int,
    qpool: int = TRANS_QPOOL,
    rungs: tuple[int, ...] = PRODUCT_EXOG_WINDOWS,
) -> np.ndarray:
    """PER-RUNG eigenbases: each horizon gets its OWN frozen frame.

    THE QUESTION (author directive 2026-08-07): every construction in this
    campaign so far uses ONE eigenbasis — that of the raw base features — and
    applies it at every horizon. But ma_j is a linear smoother, so ma_j(X) has
    its own cross-sectional correlation structure; the leading directions of
    the fast panel need not be the leading directions of the slow panel. If
    they coincide, this arm ties ``blk_pcladder_tuned`` and the alignment
    number below IS the explanation; if they do not, a per-horizon basis is
    strictly more expressive at the same column count.

    CONSTRUCTION, per rung j (smooth FIRST, rotate SECOND — the exact reverse
    of :func:`_pc_ladder_design`):
      1. Z_j = ma_j(Z) over the base features (causal rolling mean,
         min_periods=1, shifted one bar — the ladder convention of
         ``generate_har_features``).
      2. V_j = top-``qpool`` eigenvectors of the FROZEN first-window
         correlation of Z_j, same frame window ``Z_j[window:2*window]``, same
         ``_DEGENERATE_SD`` liveness rule, eigenvalues descending. LOUD failure
         if qpool exceeds that rung's live spectrum.
      3. Scores G_j = ((Z_j - mu_j) / sd_j) @ V_j, then TRAILING-standardized
         exactly as the winning transmission arms standardize theirs (rolling
         ``TRANS_TRAIL_DAYS`` mean/std, shifted one bar, warm-up rows zeroed).

    COLUMN LAYOUT is identical to ``_pc_ladder_design`` — grouped by PC rank,
    rungs contiguous within a rank, column (i-1)*n_rungs + j — because the
    penalty is the SAME ``pcrank`` family: one (lambda0, gamma) shared across
    all rungs of a rank AND across rungs. Sharing the penalty parameterization
    is deliberate: it leaves the BASIS as the only difference from
    ``blk_pcladder_tuned``, which is what makes the head-to-head clean.
    """
    bc = _product_base_cols(p.names)
    Z = np.ascontiguousarray(p.X[:, bc], dtype=np.float64)
    n = len(Z)
    k_pool = int(qpool)
    f0, f1 = window, 2 * window
    trail = TRANS_TRAIL_DAYS * PERIODS_PER_DAY
    n_rungs = len(rungs)
    out = np.empty((n, k_pool * n_rungs), dtype=np.float64)
    bases: list[np.ndarray] = []
    for j, w in enumerate(rungs):
        zj = (
            pd.DataFrame(Z)
            .rolling(window=int(w), min_periods=1)
            .mean()
            .shift(1)
            .to_numpy()
        )
        zj[~np.isfinite(zj)] = 0.0  # the single shifted warm-up row
        zw = zj[f0:f1]
        mu, sd = zw.mean(0), zw.std(0)
        live = sd > _DEGENERATE_SD
        n_avail = int(live.sum())
        if k_pool > n_avail:
            raise SystemExit(
                f"per-rung PC frame (rung {w}): K={k_pool} exceeds the "
                f"available spectrum ({n_avail} live base columns) — no "
                "silent cap"
            )
        sdl = np.where(live, sd, 1.0)
        lam, v_l = np.linalg.eigh(np.corrcoef(((zw - mu) / sdl)[:, live], rowvar=False))
        order = np.argsort(lam)[::-1]
        v_full = np.zeros((Z.shape[1], len(lam)))
        v_full[live] = v_l[:, order]
        v_j = v_full[:, :k_pool]
        bases.append(v_j)
        g = ((zj - mu) / sdl) @ v_j
        gm = pd.DataFrame(g).rolling(trail, min_periods=trail).mean().shift(1)
        gs = pd.DataFrame(g).rolling(trail, min_periods=trail).std().shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (g - gm.to_numpy()) / gs.to_numpy()
        g[~np.isfinite(g)] = 0.0  # warm-up rows (< trail bars of history)
        out[:, j::n_rungs] = g  # rank-major layout: col i*n_rungs + j
    # SUBSPACE-ALIGNMENT DIAGNOSTIC (the arm's own explanation if it ties):
    # principal angles between the fastest and slowest rung's top-K subspaces
    # (singular values of V_fast' V_slow, all 1.0 <=> identical subspace) and
    # the mean |dot| of RANK-MATCHED eigenvectors (sign-invariant).
    v_a, v_b = bases[0], bases[-1]
    cos_pa = np.linalg.svd(v_a.T @ v_b, compute_uv=False)
    matched = np.abs(np.sum(v_a * v_b, axis=0))
    _LAST_PERRUNG_DIAG.clear()
    _LAST_PERRUNG_DIAG.update(
        {
            "rungs": [int(w) for w in rungs],
            "qpool": k_pool,
            "principal_angle_cos": cos_pa.tolist(),
            "mean_principal_angle_cos": float(np.mean(cos_pa)),
            "min_principal_angle_cos": float(np.min(cos_pa)),
            "mean_matched_abs_dot": float(np.mean(matched)),
        }
    )
    print(
        f"[pc_ladder_perrung] K={k_pool} rungs={list(rungs)}: "
        f"fast-vs-slow subspace mean cos(principal angle) "
        f"{float(np.mean(cos_pa)):.4f} (min {float(np.min(cos_pa)):.4f}), "
        f"mean |dot| of rank-matched eigenvectors "
        f"{float(np.mean(matched)):.4f}"
    )
    return out


def _walk_blocks_ew(
    F: np.ndarray,
    segments: list[tuple[int, int, str]],
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    half_lives: tuple[float, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-bar-refit block ridge on a DISCOUNTED gram, with the half-life
    selected causally ALONGSIDE the per-block penalties.

    Identical to ``_walk_blocks_tuned`` in every respect except that the
    sufficient statistics carry an exponential weight: same TUNE_PER=250
    cadence, same forward split (fit block, EMBARGO=25 gap, VAL_TAIL=125 tail),
    same per-block grids, same argmin-validation-MSE rule, same first-minimum
    tie-break. The search is the CARTESIAN product of the block grids TIMES the
    half-life grid, so the tuner sees the forgetting-vs-shrinking tradeoff
    jointly rather than in sequence — which is the whole point: if it picks a
    finite H with LIGHTER penalties than the flat arm selects, the tradeoff is
    real and measured.

    A single-element ``half_lives`` makes this a FIXED-H arm at no extra cost:
    the H axis degenerates and only the penalties are selected.

    COST: one weighted fit-gram build per half-life per boundary (the flat arm
    builds one), then the same n_combos solves off each. The per-bar path is
    one rank-one EW update plus one diag-penalty solve — the same O(p^3) the
    incumbent pays.
    """
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    p_ = Xs.shape[1]
    keys = [k for _, _, k in segments]
    grids = [BLOCK_TUNE_GRIDS[k] for k in keys]

    def pen_vec(alphas: dict[str, Any]) -> np.ndarray:
        pen = np.empty(p_)
        for s0, s1, k in segments:
            _fill_pen_span(pen, s0, s1, alphas[k])
        return pen

    out = np.empty(hi - lo, dtype=np.float64)
    traj: list[dict[str, Any]] = []
    stats: _EwStats | None = None
    pen: np.ndarray | None = None
    sel_h = float(half_lives[0])
    for i in range(hi - lo):
        t = window + i
        if i % TUNE_PER == 0:
            Xw, yw = Xs[i:t], ys[i:t]
            fl, fh, vl, vh = forward_window_split(window, window, VAL_TAIL, EMBARGO)
            Xf, yf = Xw[fl:fh], yw[fl:fh]
            Xv, yv = Xw[vl:vh], yw[vl:vh]
            best: tuple[float, float, dict[str, Any], np.ndarray] | None = None
            n_evals = 0
            for hl in half_lives:
                # weights anchored at the MOST RECENT fit row (weight 1.0)
                wts = ew_weights(hl, fh - fl)
                gf, cf, sw_f, sx_f, sy_f = _ew_weighted_stats(Xf, yf, wts)
                muf = sx_f / sw_f
                myf = sy_f / sw_f
                Xvc = Xv - muf
                for combo in itertools.product(*grids):
                    alphas = dict(zip(keys, (_pen_value(a) for a in combo)))
                    pv = pen_vec(alphas)
                    g = gf.copy()
                    g[np.diag_indices_from(g)] += pv
                    b = np.linalg.solve(g, cf)
                    mse = float(np.mean((Xvc @ b + myf - yv) ** 2))
                    n_evals += 1
                    if best is None or mse < best[0]:
                        best = (mse, float(hl), alphas, pv)
            assert best is not None  # the grids are non-empty by construction
            _, sel_h, sel, pen = best
            # exact rebuild at the selected half-life (also bounds update drift)
            stats = _EwStats(Xw, yw, sel_h)
            traj.append(
                {
                    "row": int(lo + i),
                    "half_life": sel_h,
                    "decay": ew_decay(sel_h),
                    "n_eff": ew_n_eff(sel_h, window),
                    "alphas": sel,
                    "n_tail_evals": int(n_evals),
                }
            )
        # i == 0 is a boundary, so both are set on the first pass
        assert stats is not None and pen is not None
        gram, rhs = stats.centered()
        gram[np.diag_indices_from(gram)] += pen
        coef = np.linalg.solve(gram, rhs)
        out[i] = float(Xs[t] @ coef + stats.intercept(coef))
        stats.roll(Xs[t], ys[t], Xs[t - window], ys[t - window])
    return out, traj


# ── PROPER PCR: rotate the LADDER TENSOR, never the scores ────────────────────
# TENSOR STRUCTURE, verified against the real panel name list
# (results/panel_columns.json, 1144 columns) rather than assumed:
#     value      43 base quantities x 12 MA windows (1,2,4,...,2048) = 516
#     indicator  48 prefixes        x 12 MA windows                  = 576
#     extras     12 har + 24 regime + 16 calendar                    =  52
#                                                          total      1144  (exact)
# Every value stem is present at EVERY window and every indicator prefix is
# present at every window — no ragged columns, so the ladder IS a clean uniform
# tensor and "rotate each window's slab by a shared V" is well posed.
# THE ARITHMETIC DIFFERS FROM THE BRIEF, and it changes the arm ladder: the
# rotatable set is the VALUE tensor, which is 12 x 43, NOT 12 x 92. The 48
# indicator prefixes exceed the 43 value stems because five quantities carry
# BOTH an _avail_ and an _active_ column (numobs and the four voldemand
# families). So the maximum meaningful K is 43, and K = 92/80/60 do not exist.
# The A-family ladder is therefore FULL/40/30/20 and the B-family 30/20/10/5,
# which also yields matched-K pairs at BOTH 30 and 20 for the ordering test.
#
# WHY V IS BUILT ON THE ma_1 SLAB: under the commuting identity
# ma_w(V'x) = V' ma_w(x), the object being rotated is the BASE QUANTITY vector
# x, and `generate_har_features` builds ma_1 as rolling(1).mean().shift(1) —
# i.e. the ma_1 slab IS x, lagged one bar. It is the only slab that is the base
# quantity rather than a smoothing of it.
#
# WHAT IS STANDARDIZED, AND WHAT IS NOT (the author's central instruction): the
# INPUTS are standardized — each slab is centered and scaled by its own
# FRAME-WINDOW statistics, which is what makes this a correlation-matrix PCA
# and keeps the twelve slabs commensurate under one shared shrinkage. The
# SCORES ARE NEVER STANDARDIZED. Per-score standardization is a time-varying
# per-column rescaling; it is not orthogonal, it silently broke the PC-ladder's
# equivalence (measured at 0.796 relative fitted-value difference), and it is
# why that family sat ~1.6pp below the panel at every K. Nothing here touches a
# score after the rotation.
# (Per-slab standardization keeps the design an ORTHOGONAL reparameterization
# of the standardized unrotated design, which is what the K=full gate needs;
# standardizing every slab by the ma_1 statistics instead would also satisfy
# the gate but would leave the smoothed slabs at shrinking scale.)
_ROT_ORDERINGS = ("variance", "predictive")


def _rot_prod_design(p: _Panel, window: int) -> np.ndarray:
    """PRODUCTION-SCALED full-rank rotation: [rotated VALUE columns | dead
    value columns | UNROTATED indicators], as ONE penalized block.

    WHY THIS EXISTS (measured 2026-08-08). Uniform ridge is invariant under an
    orthogonal rotation of a block, so a FULL-RANK rotated exogenous design
    must tie the raw one. The rotation family did not tie — it lost to
    blk2_gated_tuned by ~2.2e-3 even at full rank (DM +7.28). The rotation was
    never the problem: the two paths standardize their INPUTS differently. The
    rotation family (_rot_value_frame) rebuilds each slab from FROZEN
    frame-window statistics, while the production/gated path uses the panel's
    own ROLLING scaler (run_geometry_local.prepare_full applies
    rolling_robust_scale to X before any arm sees it). The full-rank gap is
    therefore the measured cost of frozen-vs-rolling input standardization —
    the same lesson the transmission revival already paid for once, when
    trailing standardization beat frozen.

    THE CORRECTION: take the value columns STRAIGHT FROM p.X, which are already
    rolling-scaled by the prep, and apply NO further standardization. The frame
    is the eigenvectors of those columns' correlation over the frame window
    (rows [window, 2*window), the same rows and the same ``_DEGENERATE_SD``
    liveness rule every other frozen frame uses). Dead columns pass through
    unrotated so the block's SPAN is preserved exactly.

    ORTHOGONALITY, which is the whole point: V is the eigenvector matrix of a
    symmetric matrix, hence orthogonal, and the applied map is
    blkdiag(V, I_dead, I_indicators) — orthogonal. Under the single uniform
    penalty this block carries, fitted values are therefore IDENTICAL to the
    unrotated exog_all block at full rank. That identity is asserted offline to
    machine precision in the verify script, so the arm's gate does not depend
    on reading a canary. This mirrors _exog_tilt_design, which already rotates
    the production exogenous columns with no rescaling for exactly this reason.

    CAUTION, stated rather than glossed: the rolling scaler is TIME-VARYING, so
    "V computed on rolling-scaled frame inputs" is V of one SNAPSHOT, and
    applying it to later rolling-scaled columns is a FROZEN linear map of
    time-varying-scaled inputs. That is not a new convention — it is precisely
    what the production trailing-standardized transmission levels already do
    (``_transmission_block`` with standardization="trailing": a frozen frame
    applied to rolling-rescaled inputs), and it is the configuration that won
    the transmission revival. Cited, not invented.
    """
    val = _cols(p.names, {"value"})
    ind = _cols(p.names, {"indicator"})
    z = np.ascontiguousarray(p.X[:, val], dtype=np.float64)
    zw = z[window : 2 * window]
    sd = zw.std(0)
    live = sd > _DEGENERATE_SD
    n_live = int(live.sum())
    if n_live < 2:
        raise SystemExit(
            "production-scaled rotation: fewer than 2 live value columns in "
            "the frame window; refusing to build a degenerate rotation"
        )
    mu = zw[:, live].mean(0)
    lam, vec = np.linalg.eigh(
        np.corrcoef(((zw[:, live] - mu) / sd[live]), rowvar=False)
    )
    v_mat = vec[:, np.argsort(lam)[::-1]]  # descending eigenvalue order
    out = np.empty((len(p.y), z.shape[1] + ind.size), dtype=np.float64)
    out[:, :n_live] = z[:, live] @ v_mat
    out[:, n_live : z.shape[1]] = z[:, ~live]
    out[:, z.shape[1] :] = p.X[:, ind]
    return out


def _rot_prod_timeaware_design(p: _Panel, window: int, arm: str) -> np.ndarray:
    """Production-scaled full-rank rotation with time-scale penalty profile."""
    val = _cols(p.names, {"value"})
    ind = _cols(p.names, {"indicator"})
    z = np.ascontiguousarray(p.X[:, val], dtype=np.float64)
    zw = z[window : 2 * window]
    sd = zw.std(0)
    live = sd > _DEGENERATE_SD
    n_live = int(live.sum())
    if n_live < 2:
        raise SystemExit("timeaware rotation: fewer than 2 live value columns")
    mu = zw[:, live].mean(0)
    lam, vec = np.linalg.eigh(
        np.corrcoef(((zw[:, live] - mu) / sd[live]), rowvar=False)
    )
    order = np.argsort(lam)[::-1]
    v_mat = vec[:, order]

    cols_by_win = _value_slab_cols(p.names)
    panel_to_live = {int(idx): i for i, idx in enumerate(val[live])}
    log2w = np.empty(n_live, dtype=np.float64)
    pos = 0
    for w in sorted(cols_by_win):
        for c in cols_by_win[w]:
            if c in panel_to_live:
                log2w[panel_to_live[c]] = np.log2(w)
                pos += 1
    if pos != n_live:
        raise SystemExit(f"timeaware: window count mismatch {pos} != {n_live}")

    profile = np.sum(v_mat**2 * log2w[np.newaxis, :], axis=1)
    profile = profile - profile.mean()
    block_profile = np.zeros(z.shape[1] + ind.size, dtype=np.float64)
    block_profile[:n_live] = profile
    key = f"{arm}_{window}"
    _TIMEAWARE_PROFILES[key] = block_profile

    out = np.empty((len(p.y), z.shape[1] + ind.size), dtype=np.float64)
    out[:, :n_live] = z[:, live] @ v_mat
    out[:, n_live:z.shape[1]] = z[:, ~live]
    out[:, z.shape[1]:] = p.X[:, ind]
    return out


def _wobble_eps_from_arm(arm: str) -> float:
    """Fixed wobble radius encoded in the arm name: ...Eps1em2 -> 1e-2.

    Kept OUT of the causal tuner on purpose: epsilon is a design perturbation,
    not a penalty; tuning it on the same 125-bar tail would confound the test.
    """
    m = re.search(r"Eps(\d+)em(\d+)", arm)
    if not m:
        return 0.0
    return float(int(m.group(1)) * (10.0 ** (-int(m.group(2)))))


def _causal_wobble_q(n_live: int, arm: str, window: int, eps: float) -> np.ndarray:
    """Deterministic orthogonal perturbation Q = exp(eps * A), A^T = -A.

    Causal by construction: the seed depends only on (arm, window, n_live),
    never on current-chunk outcomes. A is normalized to unit spectral norm so
    eps is interpretable as the maximum rotation angle in radians.
    """
    if eps == 0.0:
        return np.eye(n_live, dtype=np.float64)
    import hashlib
    from scipy.linalg import expm

    seed = int.from_bytes(
        hashlib.sha256(f"{arm}|{window}|{n_live}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_live, n_live))
    a = 0.5 * (a - a.T)
    norm = float(np.linalg.norm(a, 2))
    if norm > 0:
        a /= norm
    q = np.asarray(expm(eps * a), dtype=np.float64)
    # Numerical hygiene: expm of an antisymmetric matrix is orthogonal to roundoff.
    return q


def _rot_prod_wobble_design(
    p: _Panel, window: int, arm: str, timeaware: bool
) -> np.ndarray:
    """Full-rank production rotation plus a causal orthogonal wobble.

    Full-rank information is preserved exactly; only the orientation relative
    to the penalty ellipsoid changes. With timeaware=True the timescale profile
    is computed AFTER the effective rotation VQ so penalty direction i and
    output column i remain the same object.
    """
    val = _cols(p.names, {"value"})
    ind = _cols(p.names, {"indicator"})
    z = np.ascontiguousarray(p.X[:, val], dtype=np.float64)
    zw = z[window : 2 * window]
    sd = zw.std(0)
    live = sd > _DEGENERATE_SD
    n_live = int(live.sum())
    if n_live < 2:
        raise SystemExit("wobble rotation: fewer than 2 live value columns")
    mu = zw[:, live].mean(0)
    lam, vec = np.linalg.eigh(
        np.corrcoef(((zw[:, live] - mu) / sd[live]), rowvar=False)
    )
    v_mat = vec[:, np.argsort(lam)[::-1]]
    q = _causal_wobble_q(n_live, arm, window, _wobble_eps_from_arm(arm))
    v_eff = np.ascontiguousarray(v_mat @ q, dtype=np.float64)

    if timeaware:
        cols_by_win = _value_slab_cols(p.names)
        panel_to_live = {int(idx): i for i, idx in enumerate(val[live])}
        log2w = np.empty(n_live, dtype=np.float64)
        pos = 0
        for w in sorted(cols_by_win):
            for c in cols_by_win[w]:
                if c in panel_to_live:
                    log2w[panel_to_live[c]] = np.log2(w)
                    pos += 1
        if pos != n_live:
            raise SystemExit(f"timeaware wobble: window count mismatch {pos} != {n_live}")
        profile = np.sum(v_eff**2 * log2w[np.newaxis, :], axis=1)
        profile = profile - profile.mean()
        block_profile = np.zeros(z.shape[1] + ind.size, dtype=np.float64)
        block_profile[:n_live] = profile
        _TIMEAWARE_PROFILES[f"{arm}_{window}"] = block_profile

    out = np.empty((len(p.y), z.shape[1] + ind.size), dtype=np.float64)
    out[:, :n_live] = z[:, live] @ v_eff
    out[:, n_live:z.shape[1]] = z[:, ~live]
    out[:, z.shape[1]:] = p.X[:, ind]
    return out

def _rot_value_frame(p: _Panel, window: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(V, live mask over the 43 base quantities, live rank) from the FRAME
    WINDOW only — rows [window, 2*window), the same window and the same
    ``_DEGENERATE_SD`` liveness rule the transmission frame uses.

    V is the eigenvector matrix of the CORRELATION of the ma_1 slab, columns
    ordered by descending eigenvalue. Frozen: computed once, applied to every
    bar and every MA window.
    """
    cols = _value_slab_cols(p.names)
    z1 = np.ascontiguousarray(p.X[window : 2 * window, cols[1]])
    sd = z1.std(0)
    live = sd > _DEGENERATE_SD
    n_live = int(live.sum())
    if n_live < 2:
        raise SystemExit(
            "proper PCR frame: fewer than 2 live base quantities in the frame "
            "window — refusing to build a degenerate rotation"
        )
    mu = z1[:, live].mean(0)
    zs = (z1[:, live] - mu) / sd[live]
    lam, vec = np.linalg.eigh(np.corrcoef(zs, rowvar=False))
    return vec[:, np.argsort(lam)[::-1]], live, n_live


# Weight-matrix provenance of the last PLS build (hash + shape), persisted so
# the frozen supervised frame is auditable chunk to chunk.
_LAST_PLS_DIAG: dict[str, Any] = {}

# TIME-AWARE POST-PCA PENALTY PROFILES (author directive 2026-08-09)
_TIMEAWARE_PROFILES: dict[str, np.ndarray] = {}


def _pls_frame(p: _Panel, window: int, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(W, live mask, live rank) — SUPERVISED basis for the value tensor.

    The PCA frame maximizes VARIANCE among the base quantities and never looks
    at the target; PLS maximizes COVARIANCE WITH THE TARGET, so at equal K it
    should carry more forecasting content. That is the whole point of the
    supervised rung of the capture ladder.

    CAUSALITY, identical in status to the frozen predictive ORDERING already
    shipped: the weights are fit on the FRAME WINDOW ONLY — rows
    [window, 2*window), which precede every scored bar (the first legal OOS row
    is 2*window) — and then FROZEN permanently, exactly as V is. No post-frame
    row can touch them; asserted in the verify script by perturbing every row
    beyond the frame window and requiring bit-identical weights.

    ESTIMATOR: sklearn's PLSRegression (NIPALS with deflation). Deterministic —
    NIPALS has no random initialization and the deflation order is fixed — so
    repeated builds give bit-identical weights, which the synthetic checks.
    Inputs are the ma_1 slab (the base-quantity vector) standardized by its own
    frame-window statistics, matching the PCA path exactly; ``scale=False``
    because the standardization is already applied and letting PLS rescale
    again would silently change the basis.
    """
    from sklearn.cross_decomposition import PLSRegression

    cols = _value_slab_cols(p.names)
    z1 = np.ascontiguousarray(p.X[window : 2 * window, cols[1]])
    sd = z1.std(0)
    live = sd > _DEGENERATE_SD
    n_live = int(live.sum())
    if n_live < 2:
        raise SystemExit("PLS frame: fewer than 2 live base quantities")
    if k > n_live:
        raise SystemExit(f"PLS frame: K={k} exceeds live rank {n_live}")
    zs = (z1[:, live] - z1[:, live].mean(0)) / sd[live]
    yv = p.y[window : 2 * window]
    model = PLSRegression(n_components=int(k), scale=False)
    model.fit(zs, yv - yv.mean())
    w = np.ascontiguousarray(model.x_weights_, dtype=np.float64)
    _LAST_PLS_DIAG.clear()
    _LAST_PLS_DIAG.update(
        {
            "k": int(k),
            "n_live": n_live,
            "weights_sha256": hashlib.sha256(w.tobytes()).hexdigest(),
            "weights_shape": list(w.shape),
            "frame_rows": [int(window), int(2 * window)],
        }
    )
    return w, live, n_live


def _value_slab_cols(names: list[str]) -> dict[int, np.ndarray]:
    """{MA window -> column indices of that window's VALUE slab}, base
    quantities in a single canonical (sorted-stem) order shared by every
    window. LOUD failure if the tensor is ragged — a missing stem at one
    window would silently misalign the rotation."""
    by_w: dict[int, dict[str, int]] = {}
    for j, nm in enumerate(names):
        kind, stem, w = _classify(nm)
        if kind == "value":
            by_w.setdefault(w, {})[stem] = j
    if not by_w:
        raise SystemExit("proper PCR: no value columns found in the panel")
    stems = sorted(next(iter(by_w.values())))
    out: dict[int, np.ndarray] = {}
    for w, d in sorted(by_w.items()):
        if sorted(d) != stems:
            raise SystemExit(
                f"proper PCR: value tensor is RAGGED at MA window {w} "
                f"({len(d)} stems vs {len(stems)}) — the shared rotation would "
                "misalign; refusing to build"
            )
        out[w] = np.asarray([d[s] for s in stems], dtype=np.int64)
    return out


def _rot_predictive_order(
    p: _Panel, window: int, rot: np.ndarray, n_dir: int, n_w: int
) -> np.ndarray:
    """Direction ranking by PREDICTIVE content, estimated on the FRAME WINDOW
    ONLY — rows [window, 2*window), which precede every scored bar (the first
    legal OOS row is 2*window), so the ordering cannot see a scored outcome.
    Frozen permanently, exactly as V is.

    THE STATISTIC, stated because it is a choice: for each direction i, the
    MULTIPLE R^2 of regressing the frame-window target on that direction's
    OWN 12 ladder columns (12 numerator df), computed with an intercept. A
    direction is a 12-column object under this design, so a literal
    per-column univariate R^2 would force an arbitrary choice of which MA
    window speaks for the direction; the block R^2 is the natural
    "how much does this direction explain" statistic and treats every
    direction identically. Ties break by direction index (deterministic).
    """
    y = p.y[window : 2 * window]
    yc = y - y.mean()
    sst = float(yc @ yc)
    r2 = np.zeros(n_dir)
    if sst > 0.0:
        for i in range(n_dir):
            blk = rot[window : 2 * window, i * n_w : (i + 1) * n_w]
            bc = blk - blk.mean(0)
            g = bc.T @ bc
            c = bc.T @ yc
            try:
                coef = np.linalg.solve(g, c)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(g) @ c
            r2[i] = float(coef @ c) / sst
    return np.argsort(-r2, kind="stable")


def _rot_value_design(
    p: _Panel,
    window: int,
    qpool: int | None,
    ordering: str,
    basis: str = "pca",
    rungs: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Rotated VALUE tensor, top-``qpool`` directions by ``ordering``.

    Columns are grouped by DIRECTION, the direction's 12 MA-window columns
    contiguous within it — so a K-truncation is a contiguous prefix and the
    column count is exactly K x 12.
    """
    if ordering not in _ROT_ORDERINGS:
        raise SystemExit(f"unknown PCR ordering '{ordering}'")
    cols = _value_slab_cols(p.names)
    if basis == "pls":
        k_req = n_live_guess = None
        v_mat, live, n_live = _pls_frame(
            p, window, int(qpool) if qpool is not None else 0
        )
        del k_req, n_live_guess
    else:
        v_mat, live, n_live = _rot_value_frame(p, window)
    wins = sorted(cols) if rungs is None else [w for w in sorted(cols) if w in rungs]
    n_w = len(wins)
    n = len(p.y)
    n_dir = v_mat.shape[1]
    rot = np.empty((n, n_dir * n_w), dtype=np.float64)
    for j, w in enumerate(wins):
        slab = np.ascontiguousarray(p.X[:, cols[w][live]])
        fw = slab[window : 2 * window]
        mu, sd = fw.mean(0), fw.std(0)
        sd = np.where(sd > _DEGENERATE_SD, sd, 1.0)
        scores = ((slab - mu) / sd) @ v_mat  # INPUTS standardized, scores NOT
        rot[:, j::n_w] = scores  # direction-major: col i*n_w + j
    order = (
        np.arange(n_dir)
        if ordering == "variance"
        else _rot_predictive_order(p, window, rot, n_live, n_w)
    )
    if basis == "pls":
        # the weight matrix already HAS k columns and its order is the PLS
        # component order; there is nothing to re-rank or truncate
        return np.ascontiguousarray(rot)
    k = n_live if qpool is None else int(qpool)
    if k > n_live:
        raise SystemExit(
            f"proper PCR: K={k} exceeds the live base-quantity rank "
            f"({n_live} of {len(cols[wins[0]])}) — no silent cap"
        )
    keep = order[:k]
    take = np.concatenate([np.arange(i * n_w, (i + 1) * n_w) for i in keep])
    return np.ascontiguousarray(rot[:, take])


def _pen_value(a: Any) -> Any:
    """Normalize a grid point so tuner equality comparisons are exact.

    scalar                        -> float (flat penalty)
    (lambda0, gamma)              -> ('power', lambda0, gamma), the existing
                                     2-tuple shaped grids
    (lambda0, family, par)        -> (family, lambda0, par)
    (lambda0, family, par, group) -> (family, lambda0, par, group) for the
                                     grouped 'pcrank' family
    """
    if isinstance(a, (tuple, list)):
        if len(a) == 4:
            return (str(a[1]), float(a[0]), float(a[2]), int(a[3]))
        if len(a) == 3:
            return (str(a[1]), float(a[0]), float(a[2]))
        return ("power", float(a[0]), float(a[1]))
    return float(a)


def _fill_pen_span(pen: np.ndarray, s0: int, s1: int, value: Any) -> None:
    """Write one block's penalties into the diag-penalty vector.

    Scalar -> flat span. (lambda0, gamma) -> RANK-SHAPED span:
    lambda_i = lambda0 * i**gamma for i = 1..K, columns in eigen-rank order
    (the transmission builder returns factor scores in descending-eigenvalue
    order, so column j IS factor rank j+1). At gamma=0 this is lambda0 * 1.0
    for every column — bit-identical to the flat fill.
    """
    if isinstance(value, tuple) and value[0] == "pcrank":
        # grouped rank tilt: columns are blocked by PC rank with `group`
        # contiguous ladder rungs each, so every rung of rank i shares the
        # penalty alpha * i**gamma (the tilt is over DIRECTIONS, not horizons)
        _fam, lam0, gamma, group = value
        idx = np.arange(s1 - s0)
        ranks = (idx // int(group) + 1).astype(np.float64)
        pen[s0:s1] = lam0 * ranks**gamma
    elif isinstance(value, tuple):
        family, lam0, par = value
        k_span = s1 - s0
        if family == "power":
            ranks = np.arange(1, k_span + 1, dtype=np.float64)
            pen[s0:s1] = lam0 * ranks**par
        elif family == "exp":
            # EXPONENTIAL profile: lambda_i = lam0 * exp(kappa * (i-1)).
            # NOT redundant with the power family (author directive
            # 2026-08-07): the frozen frame's eigenvalue spectrum is itself a
            # power law over the top 40 (d_i ~ c*i^-a, a = 1.176 on that
            # truncation; a is truncation-dependent — see
            # TRANS_SHAPE_GAMMAS_WIDE), so an EIGENVALUE-based profile
            # lambda_i ∝ d_i^-theta is the SAME family as the rank power law
            # under gamma = a*theta — a reparameterization, deliberately
            # not run. An exponential has genuinely different TAIL behaviour
            # (geometric, not polynomial) and is therefore a real alternative
            # prior. kappa=0 would nest flat, but the flat nesting is already
            # carried EXACTLY by power/gamma=0, so the grids do not duplicate
            # it (see the duplicate-free assertion in the shape-zoo test).
            ranks0 = np.arange(k_span, dtype=np.float64)
            pen[s0:s1] = lam0 * np.exp(par * ranks0)
        elif family == "step":
            # HARD-threshold family: flat at lam0 up to rank K, elevated by
            # STEP_MULTIPLIER beyond it — a finite stand-in for truncation
            # (see STEP_MULTIPLIER). Degenerates to flat when K >= span width.
            pen[s0:s1] = lam0
            k_cut = int(par)
            if k_cut < k_span:
                pen[s0 + k_cut : s1] = lam0 * STEP_MULTIPLIER
        elif family == "timescale":
            key = next(
                (k for k in _TIMEAWARE_PROFILES if _TIMEAWARE_PROFILES[k].shape[0] == s1 - s0),
                None
            )
            if key is None:
                raise KeyError(f"timescale penalty: no profile for span [{s0},{s1})")
            prof = _TIMEAWARE_PROFILES[key]
            pen[s0:s1] = lam0 * np.exp(par * prof)
        else:
            raise KeyError(f"unknown penalty shape family '{family}'")
    else:
        pen[s0:s1] = value


_BUCKET_PEN_KEY = "bkt_"  # alpha-key prefix; one key per SUBGROUP family
# Canonical FAMILY enumeration = SUBGROUPS minus `baseline` (empty) and
# `all_features` (the joint design, not a family). NOTE this is SEVEN families,
# not eight — the same count correction as the A1..A8 bucket arms.
_BUCKET_FAMILIES = [b for b in SUBGROUPS if b not in ("baseline", "all_features")]
for _b in _BUCKET_FAMILIES:
    BLOCK_TUNE_GRIDS[f"{_BUCKET_PEN_KEY}{_b}"] = BLOCK_TUNE_GRIDS["exog"]


def assert_bucket_partition(names: list[str]) -> dict[str, int]:
    """LOUD guard: the canonical families must partition the exogenous columns
    EXACTLY — every value/indicator column assigned to one family, none to two.

    Silent misassignment would invalidate the per-bucket-penalty arm (a column
    shrunk under the wrong family's penalty, or dropped from the design), so
    this runs before the arm builds. Returns {family: n_cols}.
    """
    exog = set(_exog_all_cols(names).tolist())
    counts: dict[str, int] = {}
    seen: dict[int, str] = {}
    for fam in _BUCKET_FAMILIES:
        cols = _bucket_cols(names, fam).tolist()
        counts[fam] = len(cols)
        for j in cols:
            if j in seen:
                raise SystemExit(
                    f"bucket partition broken: column '{names[j]}' assigned to "
                    f"BOTH '{seen[j]}' and '{fam}' — the per-bucket penalty arm "
                    "refuses to run on an ambiguous taxonomy"
                )
            seen[j] = fam
    unassigned = sorted(exog - set(seen))
    if unassigned:
        raise SystemExit(
            f"bucket partition broken: {len(unassigned)} exogenous column(s) "
            f"belong to NO canonical family (e.g. {[names[j] for j in unassigned[:5]]}) "
            "— the per-bucket penalty arm refuses to silently drop them"
        )
    stray = sorted(set(seen) - exog)
    if stray:
        raise SystemExit(
            f"bucket partition broken: {len(stray)} family column(s) are not in "
            f"the exogenous set (e.g. {[names[j] for j in stray[:5]]})"
        )
    return counts


_DEGENERATE_SD = 1e-8  # "no dispersion in window" guard (frame/live cols)
_SD_EPS = 1e-12  # standardization epsilon (study convention)

CACHE_DIR_ENV = "UNIFY_CACHE_DIR"  # prep-cache dir override (default "results")

# Tree expert bank (author directive 2026-08-07, mixed-family expansion): 20
# frozen tree configs — LightGBM AND XGBoost, tagged per entry — emitted by
# the dev menu study (experiments/tree_menu_dev.py --freeze). The arms READ
# the menu — no tuning inside an arm; causal, family-agnostic selection over
# the experts happens scorer-side. DESIGN (author correction 2026-08-07): the
# tree design matrix is IDENTICAL to the tuned penalized linear arms' wide
# all_features design (backbone + all exog through the same ladder/indicator
# expansion) — same information set, richer hypothesis class; NO product block,
# no tree-specific columns.
N_TREE_EXPERTS = 20
TREE_MENU_PATH = os.path.join(_ROOT, "experiments", "tree_menu.json")


def _load_tree_menu() -> list[dict[str, Any]]:
    """The frozen expert menu; LOUD failure when absent — an expert arm must
    never run against an implicit or stale configuration."""
    if not os.path.exists(TREE_MENU_PATH):
        raise SystemExit(
            f"tree expert menu absent: {TREE_MENU_PATH} — run the dev menu "
            "study and freeze it first (experiments/tree_menu_dev.py "
            "--freeze); tree_expert_* arms refuse to run without it."
        )
    with open(TREE_MENU_PATH, encoding="utf-8") as fh:
        menu = json.load(fh)
    if not isinstance(menu, list) or not menu:
        raise SystemExit(f"tree expert menu malformed/empty: {TREE_MENU_PATH}")
    return menu


# ── panel ─────────────────────────────────────────────────────────────────────


@dataclass
class _Panel:
    X: np.ndarray  # (n, p) prescaled features
    y: np.ndarray  # (n,) adj_RV — winsorized sqrt(RV / B_t), the FIT target
    baseline: np.ndarray  # (n,) B_t diurnal baseline
    rv_raw: np.ndarray  # (n,) raw RV (unwinsorized truth)
    t: np.ndarray  # (n,) datetime64[ns]
    names: list[str]
    # per-stem OBSERVED availability, causally shifted one bar (row r reflects
    # source data at r-1, matching the shift(1) in the feature build): the
    # notna-before-ffill indicator of the honest-indicator prep, straight from
    # the loader. Feeds the full-support mask of the OLS arms.
    avail: np.ndarray  # (n, n_stems) bool
    stem_index: dict[str, int]


_PANEL: _Panel | None = None


def _assert_fit_raw_alignment(
    y_fit: np.ndarray, rv_raw: np.ndarray, baseline: np.ndarray, where: str
) -> None:
    """Loud-fail unless the fit-side target and the re-read raw side describe the
    SAME rows (the v2 trail-arm incident, ruling 2026-08-07).

    Identity used: the panel's ``y`` is winsorize(sqrt(RV / B_t)) with B_t the
    stored baseline — so on every NON-winsorized row, ``sqrt(rv_raw/baseline)``
    recomputed here is BIT-IDENTICAL to ``y_fit`` (same float64 division and
    sqrt on the same values). Winsorization clips only the rolling 1/99 tails
    (a few % of rows), therefore the MEDIAN absolute difference over any honest
    row set is exactly 0.0 — threshold-free. A stale prep cache read against a
    loader whose grid convention has since moved (the trail-arm root cause: the
    cache key does not encode the grid convention, and the ``[:n]`` truncation
    of the raw re-read masks the length mismatch) shifts the raw side by
    thousands of rows and the median goes far from zero — caught here instead
    of producing plausible-looking garbage QLIKE.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y_check = np.sqrt(rv_raw / baseline)
    d = np.abs(y_fit - y_check)
    med = float(np.nanmedian(d))
    if not (med == 0.0):
        raise RuntimeError(
            f"fit/raw row misalignment detected ({where}): median "
            f"|y_fit - sqrt(rv_raw/baseline)| = {med:.6g} (must be exactly 0.0 "
            "on aligned rows — winsorization touches only the 1/99 tails). "
            "The prep cache and the current loader grid convention disagree "
            "(stale cache vs dead-session/start-date change). Rebuild the prep "
            "cache under the current loader before rerunning; refusing to emit."
        )


def _load_panel() -> _Panel:
    """The frozen b2 panel via the repo's own builder, plus timestamps / raw RV.

    ``prepare_full`` caches the (X, y, baselines, names) build; timestamps and raw
    RV are re-read from the pinned loader with the same row discipline the msweep
    line verified by execution (drivers/msweep_2026-08-01/product_tune.py:36-41):
    panel row i = load_raw_data(...).dropna(RV) row i + BURNIN_ROWS (h=1 makes
    apply_horizon_shift a no-op).
    """
    global _PANEL
    if _PANEL is not None:
        return _PANEL
    import run_geometry_local as R  # experiments/ module (path bootstrapped above)

    R.CACHE_DIR = os.environ.get(CACHE_DIR_ENV, "results")
    os.makedirs(R.CACHE_DIR, exist_ok=True)
    X, y, baselines, names = R.prepare_full("all_features", har_base=HAR_BASE)
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    baselines = np.asarray(baselines, dtype=np.float64)
    n = len(y)

    from src.data.loading import load_raw_data

    raw = (
        load_raw_data("data", allow_missing=True)
        .dropna(subset=["RV"])
        .reset_index(drop=True)
    )
    rv_raw = raw["RV"].to_numpy(dtype=np.float64)[BURNIN_ROWS:][:n]
    t = pd.to_datetime(raw["t"]).to_numpy()[BURNIN_ROWS:][:n]
    if len(rv_raw) != n or len(t) != n:
        raise RuntimeError(
            f"panel/loader row misalignment: panel n={n}, loader rows after burn-in "
            f"{len(rv_raw)} — the b2 prep and the raw loader disagree; refusing to emit"
        )
    # VALUE-level alignment gate (the length check above is insufficient: a
    # LONGER raw series under a moved grid convention truncates to the right
    # length while describing entirely different bars — the v2 trail incident).
    _assert_fit_raw_alignment(y, rv_raw, baselines, "panel load, whole series")

    # Observed-availability bitmap per exog stem: notna BEFORE any fill (the
    # honest-indicator source, executor.load_and_transform's obs-before-ffill),
    # shifted one bar for causal parity with the feature build's shift(1).
    from src.data.loading import ALL_FEATURES

    stems = [c for c in ALL_FEATURES if c in raw.columns]
    avail = np.zeros((n, len(stems)), dtype=bool)
    for k, c in enumerate(stems):
        av = raw[c].notna().shift(1, fill_value=False).to_numpy()
        avail[:, k] = av[BURNIN_ROWS:][:n]
    _PANEL = _Panel(
        X=X,
        y=y,
        baseline=baselines,
        rv_raw=rv_raw,
        t=t,
        names=list(names),
        avail=avail,
        stem_index={c: k for k, c in enumerate(stems)},
    )
    return _PANEL


# ── column taxonomy (name-driven; single source: executor naming conventions) ─


def _classify(name: str) -> tuple[str, str, int]:
    """(kind, stem, window) — kinds: har | regime | value | indicator | calendar."""
    m = re.match(r"^har_ma_(\d+)_x_(open|close)$", name)
    if m:
        return "regime", "", int(m.group(1))
    m = re.match(r"^har_ma_(\d+)$", name)
    if m:
        return "har", "", int(m.group(1))
    m = re.match(r"^(.+)_(avail|active)_ma_(\d+)$", name)
    if m:
        return "indicator", m.group(1), int(m.group(3))
    m = re.match(r"^adj_(.+)_ma_(\d+)$", name)
    if m:
        return "value", m.group(1), int(m.group(2))
    return "calendar", "", 0


def _cols(
    names: list[str],
    kinds: set[str],
    stems: set[str] | None = None,
    windows: set[int] | None = None,
) -> np.ndarray:
    idx = []
    for j, nm in enumerate(names):
        kind, stem, w = _classify(nm)
        if kind not in kinds:
            continue
        if stems is not None and kind in ("value", "indicator") and stem not in stems:
            continue
        if windows is not None and w not in windows:
            continue
        idx.append(j)
    return np.asarray(idx, dtype=np.int64)


def _backbone_cols(names: list[str]) -> np.ndarray:
    """HAR(target) ladder + session-edge interactions + calendar/expiry — the
    audited OLS-HAR incumbent design (har + regime + calendar kinds)."""
    return _cols(names, {"har", "regime", "calendar"})


def _exog_all_cols(names: list[str]) -> np.ndarray:
    """The wide exog basis: every value + indicator MA column."""
    return _cols(names, {"value", "indicator"})


def _bucket_cols(names: list[str], bucket: str) -> np.ndarray:
    """One canonical bucket's value + indicator columns (stems from SUBGROUPS)."""
    return _cols(names, {"value", "indicator"}, stems=set(SUBGROUPS[bucket]))


def _product_base_cols(names: list[str]) -> np.ndarray:
    """The product-candidate base — analysis/nl_sparsity.base_columns translated to
    the b2 panel: HAR(target) rungs + exog VALUE cols at the fast/intraday/slow
    windows + the four session calendar columns."""
    base = list(_cols(names, {"har"}))
    base += list(_cols(names, {"value"}, windows=set(PRODUCT_EXOG_WINDOWS)))
    for nm in ("is_open", "is_close", "is_overnight", "hour"):
        base.append(names.index(nm))
    return np.asarray(base, dtype=np.int64)


# ── product block (frozen selection, no reselect) ─────────────────────────────
# _pair_ic / _upper / _products copied from analysis/nl_sparsity.py:92-113 (pure
# numpy; importing that module drags the alpha-study cache-dir guard chain).


def _pair_ic(Z: np.ndarray, e: np.ndarray) -> np.ndarray:
    n = len(Z)
    G = Z.T @ Z
    A = Z.T @ (Z * e[:, None])
    Z2 = Z * Z
    Q = Z2.T @ Z2
    mean_p = G / n
    var_p = np.maximum(Q / n - mean_p**2, _SD_EPS)
    cov = A / n - mean_p * e.mean()
    sd_e = e.std()
    return cov / (np.sqrt(var_p) * (sd_e if sd_e > 0 else 1.0))


def _upper(pb: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(pb)


def _selection_residual(p: _Panel, window: int) -> np.ndarray:
    """OOS backbone-ridge residual on rows [window, 2*window) — the synthesis
    stage_prep selection target (alpha=1 backbone, per-bar refit), computed only
    over the first two windows so every task freezes the identical selection."""
    from analysis.wf import walk_forward

    bb = _backbone_cols(p.names)
    Xb = np.ascontiguousarray(p.X[: 2 * window, bb])
    pred = walk_forward(
        Xb, p.y[: 2 * window], window, alpha=SELECTION_RIDGE_ALPHA, refit_every=1
    )
    return p.y[window : 2 * window] - pred


def _causal_floored_scale(P: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling-sd scale, floored at 10% of the cross-column median —
    analysis/minimal_model.py:334-338 verbatim, window = the arm's fit window."""
    sd = pd.DataFrame(P).rolling(window, min_periods=1000).std().shift(1).to_numpy()
    # cross-column median per row; warm-up rows are all-NaN — computed only where
    # any column is finite (avoids numpy's all-NaN-slice warning), NaN elsewhere
    med = np.full((len(sd), 1), np.nan)
    ok = np.isfinite(sd).any(axis=1)
    if ok.any():
        med[ok, 0] = np.nanmedian(sd[ok], axis=1)
    sd = np.maximum(sd, 0.1 * np.where(np.isfinite(med), med, 1.0))
    sd = pd.DataFrame(sd).bfill().to_numpy()  # warm-up rows only; inside first window
    with np.errstate(divide="ignore", invalid="ignore"):
        out = P / sd
    # A column with literally zero window dispersion (e.g. a product of session
    # dummies inactive over the whole trailing window) yields 0/0 here; it carries
    # no signal, so it is zeroed — the identifiability-mask philosophy, not a clip.
    out[~np.isfinite(out)] = 0.0
    return out


def _frozen_products(p: _Panel, window: int) -> np.ndarray:
    """(n, N_PROD) frozen product columns: pairs of the product base ranked by
    |IC| against the first-block OOS residual, selected ONCE, causally scaled."""
    bc = _product_base_cols(p.names)
    Z = np.ascontiguousarray(p.X[:, bc])
    e = _selection_residual(p, window)
    ii, jj = _upper(len(bc))
    Zw = Z[window : 2 * window]
    ic = np.abs(np.nan_to_num(_pair_ic(Zw - Zw.mean(0), e)[ii, jj]))
    frozen = np.argsort(-ic)[:N_PROD]
    P = Z[:, ii[frozen]] * Z[:, jj[frozen]]
    return _causal_floored_scale(P, window)


# ── transmission block (frozen frame + Cucuringu lead-lag) ────────────────────


# Diagnostics stash for the LAST transmission construction in this process
# (transmission dig, 2026-08-07): per-refresh operator matrices, frame
# eigenvalues (top-K per estimation), and refresh row indices — persisted as
# npz arrays by compute() (arrays belong in the npz; JSON meta stays light).
_LAST_TRANS_DIAG: dict[str, Any] | None = None


def _lagged_xcorr_op(g_hist: np.ndarray, lag: int, operator: str) -> np.ndarray:
    """The refresh-window operator from the lag-``lag`` cross-correlation C of
    the (already standardized) trailing factor scores. ``operator``:
      * "antisym" — D = (C - C')/2, the Cucuringu lead-lag flow (diagonal is
        STRUCTURALLY zero: (C_ii - C_ii)/2 = 0).
      * "sym"     — S = (C + C')/2 with the diagonal zeroed EXPLICITLY to
        mirror D's structurally-zero diagonal (the lead-lag falsification
        control must differ only in the antisymmetric/symmetric split, not in
        self-persistence terms).
      * "full"    — the undecomposed C itself (diagonal kept; S+D == C).
    """
    a, b = g_hist[:-lag], g_hist[lag:]
    az = (a - a.mean(0)) / (a.std(0) + _SD_EPS)
    bz = (b - b.mean(0)) / (b.std(0) + _SD_EPS)
    c_mat = (az.T @ bz) / len(az)
    if operator == "antisym":
        return (c_mat - c_mat.T) / 2.0
    if operator == "sym":
        s_mat = (c_mat + c_mat.T) / 2.0
        np.fill_diagonal(s_mat, 0.0)
        return s_mat
    if operator == "full":
        return c_mat
    raise KeyError(f"unknown transmission operator '{operator}'")


def _align_frame(v_prev: np.ndarray, v_new: np.ndarray) -> np.ndarray:
    """Order+sign alignment of a refreshed eigenframe against its predecessor
    (blk4_trailRefresh). RULE: greedy maximum-|dot| assignment — enumerate all
    (prev_col, new_col) pairs by |v_prev_i . v_new_j| descending, assign each
    prev slot the best unused new eigenvector; then flip each assigned
    vector's sign so its dot with the predecessor is positive. Eigen-solvers
    return vectors up to sign and eigenvalue-crossings reorder them; without
    this, coefficient continuity across refreshes is destroyed by pure gauge
    flips. Greedy on |dot| is the standard frame-tracking heuristic (near-
    optimal when the frame is span-stable, which the gauge-degeneracy result
    established)."""
    k = v_prev.shape[1]
    dots = np.abs(v_prev.T @ v_new)
    pairs = sorted(
        ((dots[i, j], i, j) for i in range(k) for j in range(k)), reverse=True
    )
    order = np.full(k, -1)
    used: set[int] = set()
    for _, i, j in pairs:
        if order[i] == -1 and j not in used:
            order[i] = j
            used.add(j)
    v_out = v_new[:, order]
    signs = np.sign(np.sum(v_prev * v_out, axis=0))
    signs[signs == 0] = 1.0
    return v_out * signs


def _transmission_block(
    p: _Panel,
    window: int,
    parts: str,
    standardization: str = "frozen",
    operator: str = "antisym",
    frame: str = "frozen",
    qpool: int | None = None,
    lag: int | None = None,
    exclude_stems: tuple[str, ...] = (),
) -> np.ndarray:
    """Transmission block; ``parts`` selects the composition: "both" =
    [factor scores G | lead-lag Ghat] (2K cols, the _user convention),
    "scores" = G only, "flow" = Ghat only (the _doc documented construction).

    ``standardization``: "frozen" (frame-window stats + floored-sd Ghat scale
    — the original campaign construction), "trailing" (the revival: causal
    trailing demeaning AND division by the trailing sd + the standard rolling
    robust scaler; see the 2026-08-06 ruling notes at the TRANS constants),
    or the decomposition modes "trail_mean" (causal trailing demeaning only)
    and "trail_sd" (division by the causal trailing sd only).

    TRANSMISSION DIG knobs (author directive 2026-08-07; every new arm keeps
    the trailing standardization and the fixed user penalties):
      * ``operator`` — "antisym" (D, the construction), "sym" (S, the
        lead-lag falsification control; diagonal zeroed to mirror D), "full"
        (undecomposed C). See :func:`_lagged_xcorr_op`.
      * ``frame`` — "frozen" (top-K eigenvectors of the first-window
        correlation, the span-stability convention) or "refresh" (top-K
        eigenvectors of the TRAILING 504d correlation, re-estimated quarterly
        IN LOCKSTEP with the operator, never seeing the future; consecutive
        frames order+sign-aligned per :func:`_align_frame`; scores are
        standardized by each refresh's own estimation-window stats — a causal
        quarterly standardization, so the extra rolling G-standardization of
        the frozen-frame trailing path is redundant here and skipped).
      * ``qpool`` — frame width K (default TRANS_QPOOL=20; LOUD failure when
        K exceeds the available spectrum).
      * ``lag`` — cross-correlation lag in bars (default 1); Ghat_t =
        Op·G(t-lag) with Op estimated from the lag-``lag`` cross-correlation.
      * ``exclude_stems`` — raw-feature families removed from the transmission
        BASE Z before the frame/operator (blk4_trailDropHet: the
        cadence-heterogeneous aggregates, TRANS_HET_STEMS — the surgical test
        of the Dec-2014 cadence-homogenization hypothesis; the exog ridge
        block keeps these columns, only what feeds the operator changes).
        Loud failure if the filter removes nothing (a misspelled stem must
        not silently degenerate to blk4_trail).

    Diagnostics: per-refresh operator matrices, top-K frame eigenvalues, and
    refresh row indices are stashed in module-level ``_LAST_TRANS_DIAG`` and
    persisted by compute() into the chunk npz (mechanism analysis without
    refits).
    """
    global _LAST_TRANS_DIAG
    k_pool = TRANS_QPOOL if qpool is None else int(qpool)
    lag = TRANS_LAG_BARS if lag is None else int(lag)
    bc = _product_base_cols(p.names)
    excluded_names: list[str] = []
    if exclude_stems:
        kept = []
        for j in bc:
            _kind, stem, _w = _classify(p.names[j])
            if stem in exclude_stems:
                excluded_names.append(p.names[j])
            else:
                kept.append(j)
        if not excluded_names:
            raise SystemExit(
                f"transmission exclude_stems {exclude_stems} removed NO base "
                "columns — misspelled stem or drifted naming; refusing to run "
                "an arm identical to its parent"
            )
        bc = np.asarray(kept, dtype=np.int64)
    Z = np.ascontiguousarray(p.X[:, bc])
    f0, f1 = window, 2 * window
    n = len(Z)
    trail = TRANS_TRAIL_DAYS * PERIODS_PER_DAY
    refresh = TRANS_REFRESH_DAYS * PERIODS_PER_DAY
    ops_rec: list[np.ndarray] = []
    eig_rec: list[np.ndarray] = []
    rows_rec: list[int] = []

    def _frame_of(zw: np.ndarray):
        """(unit eigvec matrix V (n_base, K), per-col scale, mean, top-K eigvals)
        of the correlation of one estimation window; LOUD failure when K
        exceeds the available (live-column) spectrum."""
        mu, sd = zw.mean(0), zw.std(0)
        live = sd > _DEGENERATE_SD
        sdl = np.where(live, sd, 1.0)
        lam, v_l = np.linalg.eigh(np.corrcoef(((zw - mu) / sdl)[:, live], rowvar=False))
        order = np.argsort(lam)[::-1]
        n_avail = int(live.sum())
        if k_pool > n_avail:
            raise SystemExit(
                f"transmission qpool K={k_pool} exceeds the available spectrum "
                f"({n_avail} live base columns) — no silent cap"
            )
        v_full = np.zeros((Z.shape[1], len(lam)))
        v_full[live] = v_l[:, order]
        return v_full[:, :k_pool], sdl, mu, lam[order][:k_pool]

    if frame == "frozen":
        v0, sd0, mu0, eig0 = _frame_of(Z[f0:f1])
        eig_rec.append(eig0)
        w_mat = v0 / sd0[:, None]
        G = (Z - mu0) @ w_mat
        if standardization in ("trailing", "trail_mean", "trail_sd"):
            gm = pd.DataFrame(G).rolling(trail, min_periods=trail).mean().shift(1)
            gs = pd.DataFrame(G).rolling(trail, min_periods=trail).std().shift(1)
            gm = gm.to_numpy()
            gs = gs.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                if standardization == "trailing":
                    G = (G - gm) / gs
                elif standardization == "trail_mean":
                    G = G - gm
                else:  # trail_sd
                    G = G / gs
            G[~np.isfinite(G)] = 0.0  # warm-up rows (< trail bars of history)
        else:
            g_mu, g_sd = G[f0:f1].mean(0), G[f0:f1].std(0) + _SD_EPS
            G = (G - g_mu) / g_sd
        Ghat = np.zeros_like(G)
        for start in range(f1 + trail, n, refresh):
            op = _lagged_xcorr_op(G[start - trail : start], lag, operator)
            end = min(start + refresh, n)
            Ghat[start:end] = G[start - lag : end - lag] @ op
            ops_rec.append(op)
            rows_rec.append(start)
    elif frame == "refresh":
        # Causal trailing frame, quarterly, order+sign-aligned; G and Ghat both
        # zero before the first refresh boundary (frame needs 504d of history
        # and starts in lockstep with the operator loop).
        G = np.zeros((n, k_pool))
        Ghat = np.zeros((n, k_pool))
        v_prev: np.ndarray | None = None
        for start in range(f1 + trail, n, refresh):
            zw = Z[start - trail : start]
            v_new, sdl, mu, eig = _frame_of(zw)
            if v_prev is not None:
                v_new = _align_frame(v_prev, v_new)
            v_prev = v_new
            w_mat = v_new / sdl[:, None]
            g_hist = (zw - mu) @ w_mat
            g_mu, g_sd = g_hist.mean(0), g_hist.std(0) + _SD_EPS
            g_hist = (g_hist - g_mu) / g_sd
            end = min(start + refresh, n)
            G[start:end] = ((Z[start:end] - mu) @ w_mat - g_mu) / g_sd
            op = _lagged_xcorr_op(g_hist, lag, operator)
            g_lagged = ((Z[start - lag : end - lag] - mu) @ w_mat - g_mu) / g_sd
            Ghat[start:end] = g_lagged @ op
            ops_rec.append(op)
            eig_rec.append(eig)
            rows_rec.append(start)
    else:
        raise KeyError(f"unknown transmission frame mode '{frame}'")

    _LAST_TRANS_DIAG = {
        "operator": operator,
        "frame": frame,
        "qpool": k_pool,
        "lag": lag,
        "excluded_base_cols": excluded_names,
        "ops": np.stack(ops_rec) if ops_rec else np.zeros((0, k_pool, k_pool)),
        "eigvals": np.stack(eig_rec) if eig_rec else np.zeros((0, k_pool)),
        "refresh_rows": np.asarray(rows_rec, dtype=np.int64),
    }
    if standardization in ("trailing", "trail_mean", "trail_sd"):
        from src.features.transforms.scaling import rolling_robust_scale

        def _scale_from_activation(M: np.ndarray) -> np.ndarray:
            """Robust-scale a sub-block from its ACTIVATION row on (v2 c027 fix:
            feeding the all-zero warm-up region into the scaler let per-row
            trailing windows straddle the zero->live transition, collapsing the
            IQR to its floor and detonating the first live values — yhat 629.7
            in the chunk one window past Ghat activation). Scaling from the
            first nonzero row gives the scaler's initial-window convention real
            data; the warm-up stays exactly zero. G and Ghat are scaled
            separately because their activation rows differ (trail vs
            frame+trail); within each sub-block all columns share one
            activation row by construction."""
            nz = np.flatnonzero(np.abs(M).sum(axis=1) > 0)
            out = np.zeros_like(M)
            if len(nz):
                a0 = int(nz[0])
                out[a0:] = rolling_robust_scale(np.ascontiguousarray(M[a0:]), window)
            out[~np.isfinite(out)] = 0.0
            return out

        G = _scale_from_activation(G)
        Ghat = _scale_from_activation(Ghat)
        if parts == "scores":
            return G
        if parts == "flow":
            return Ghat
        return np.hstack([G, Ghat])
    Ghat = _causal_floored_scale(Ghat, window)
    Ghat[~np.isfinite(Ghat)] = 0.0
    # Composition per the 2026-08-06 ruling: _doc arms mirror the documented
    # construction EXACTLY — Ghat only, the study's design (trans_exploit.py:64-66);
    # _user arms carry [G | Ghat] (40 cols), the paper's own design.
    if parts == "scores":
        return G
    if parts == "flow":
        return Ghat
    return np.hstack([G, Ghat])


# ── walk-forward drivers (per-bar refit; chunk seam) ──────────────────────────


def _slice(lo: int, hi: int, window: int, n: int) -> tuple[int, int]:
    if lo - window < 0:
        raise ValueError(
            f"chunk_start={lo} leaves no {window}-bar training window (panel starts at 0)"
        )
    if hi > n:
        raise ValueError(f"chunk_end={hi} beyond panel end {n}")
    return lo, hi


def _walk_ridge(
    F: np.ndarray, y: np.ndarray, window: int, lo: int, hi: int, alpha: float
) -> np.ndarray:
    """Per-bar-refit rolling ridge for rows [lo, hi) — analysis/wf.walk_forward on
    the [lo-window, hi) slice (RollingLeastSquares rank-1, unpenalized intercept)."""
    from analysis.wf import walk_forward

    return walk_forward(
        F[lo - window : hi], y[lo - window : hi], window, alpha=alpha, refit_every=1
    )


def _walk_blocks_tuned(
    F: np.ndarray,
    segments: list[tuple[int, int, str]],
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    selection: str = "cartesian",
    sweep_order: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-bar-refit block ridge with CAUSALLY re-selected per-block alphas.

    Every TUNE_PER=250 solves, the current training window is split forward
    (forward_window_split: fit block, EMBARGO=25 gap, VAL_TAIL=125 tail) and the
    per-block alpha vector minimizing validation MSE is selected JOINTLY over
    the cartesian product of BLOCK_TUNE_GRIDS. Each combo is one diag-penalty
    solve off the SHARED fit-window gram — the column-scaling identity in gram
    space (scaling block j by sqrt(a_ref/a_j) under global ridge a_ref has gram
    D·G·D + a_ref·I, identically (G + diag(pen)) after undoing D; minimal_model
    .py:297-299) — so no design rebuild per combo.

    PER-BOUNDARY COST: one centered fit-gram build O(W_fit·p^2) plus
    n_combos × O(p^3) solves (blk2/c4/d3-class: 9 combos; blk3: 27; blk4: 81 —
    at p≈1200 roughly 1–3 min single-thread per boundary, amortized over 250
    bars). Between retunes the selected alpha vector is FIXED and the existing
    per-bar exact path runs unchanged: rank-1 sufficient-statistic rolls
    (RollingLeastSquares) + one diag-penalty solve per bar (same O(p^3) as the
    scalar-ridge path; intercept unpenalized via centering).

    ``selection="cyclic"`` (the per-bucket penalty rung, 2026-08-07) replaces
    the cartesian sweep with deterministic CYCLIC COORDINATE DESCENT: all
    penalties start at their grid's MIDPOINT, then for ``CYCLIC_PASSES``
    passes the blocks are swept in the FIXED order ``sweep_order``, each
    block's penalty re-selected from its own grid with all others held — same
    fit gram, same embargoed tail, same criterion. Cost is
    passes x n_blocks x |grid| tail evaluations instead of |grid|^n_blocks
    (10 blocks x 3 points: 90 vs 59,049), which is what makes the per-bucket
    rung feasible at all. No randomness anywhere: the initial point, the
    sweep order, and the argmin tie-break (first minimum wins) are fixed.

    Returns (yhat, trajectory) — trajectory is the paper-relevant selection
    record: [{row: global retune row, alphas: {block_key: alpha},
    n_tail_evals: int}, ...].
    """
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    p_ = Xs.shape[1]
    keys = [k for _, _, k in segments]
    grids = [BLOCK_TUNE_GRIDS[k] for k in keys]

    def pen_vec(alphas: dict[str, Any]) -> np.ndarray:
        pen = np.empty(p_)
        for s0, s1, k in segments:
            _fill_pen_span(pen, s0, s1, alphas[k])  # scalar or (lambda0, gamma)
        return pen

    solver = RollingLeastSquares(alpha=0.0, fit_intercept=True)  # stats carrier
    solver.init_window(Xs[:window], ys[:window])
    out = np.empty(hi - lo, dtype=np.float64)
    traj: list[dict[str, Any]] = []
    pen: np.ndarray | None = None
    n = float(window)
    for i in range(hi - lo):
        t = window + i
        if i % TUNE_PER == 0:
            Xw, yw = Xs[i:t], ys[i:t]
            fl, fh, vl, vh = forward_window_split(window, window, VAL_TAIL, EMBARGO)
            Xf, yf = Xw[fl:fh], yw[fl:fh]
            Xv, yv = Xw[vl:vh], yw[vl:vh]
            muf, myf = Xf.mean(0), float(yf.mean())
            Xfc = Xf - muf
            Gf = Xfc.T @ Xfc  # shared fit gram — built ONCE per boundary
            cf = Xfc.T @ (yf - myf)
            Xvc = Xv - muf

            def _tail_mse(alphas: dict[str, float]) -> tuple[float, np.ndarray]:
                pv = pen_vec(alphas)
                G = Gf.copy()
                G[np.diag_indices_from(G)] += pv
                b = np.linalg.solve(G, cf)
                return float(np.mean((Xvc @ b + myf - yv) ** 2)), pv

            n_evals = 0
            if selection == "cyclic":
                # deterministic start: every block at its grid's midpoint
                sel = {
                    k: _pen_value(BLOCK_TUNE_GRIDS[k][len(BLOCK_TUNE_GRIDS[k]) // 2])
                    for k in keys
                }
                cur, pen = _tail_mse(sel)
                n_evals += 1
                for _ in range(CYCLIC_PASSES):
                    for k in sweep_order or keys:
                        for a in BLOCK_TUNE_GRIDS[k]:
                            if _pen_value(a) == sel[k]:
                                continue
                            trial = dict(sel)
                            trial[k] = _pen_value(a)
                            mse, pv = _tail_mse(trial)
                            n_evals += 1
                            if mse < cur:  # strict: first minimum wins ties
                                cur, sel, pen = mse, trial, pv
            else:
                best: tuple[float, dict[str, float], np.ndarray] | None = None
                for combo in itertools.product(*grids):
                    alphas = dict(zip(keys, (_pen_value(a) for a in combo)))
                    mse, pv = _tail_mse(alphas)
                    n_evals += 1
                    if best is None or mse < best[0]:
                        best = (mse, alphas, pv)
                _, sel, pen = best
            traj.append(
                {"row": int(lo + i), "alphas": sel, "n_tail_evals": int(n_evals)}
            )
        gram = solver._Sxx - np.outer(solver._sx, solver._sx) / n
        rhs = solver._Sxy - solver._sx * (solver._sy / n)
        gram[np.diag_indices_from(gram)] += pen  # gram is a fresh array per bar
        coef = np.linalg.solve(gram, rhs)
        intercept = solver._sy / n - float(solver._sx @ coef) / n
        out[i] = float(Xs[t] @ coef + intercept)
        solver.roll(Xs[t], ys[t], Xs[t - window], ys[t - window])
    return out, traj


# ── GRID-FREE SHRINKAGE (author directive 2026-08-07) ─────────────────────────
# THE PRINCIPLE. Every estimator in this campaign allocates shrinkage by a
# hand-designed structure and then SELECTS its level on a 125-bar validation
# tail. The diagnostics say that machinery is failing on its own terms: tuned
# ridge pins its modal alpha at the grid top in 8/8 bucket designs, the elastic
# net pins 43%, the PCR block 74% bimodally, backbone blocks 53%. These arms
# replace SELECTION with ESTIMATION. Each fitted coefficient carries its own
# standard error — beta_hat ~ N(beta, sigma^2 (X'X)^-1) — so the risk-minimizing
# shrinkage is computable from the TRAINING WINDOW ALONE. No grid, no tail, no
# tuned hyperparameter of any kind.
#
# THE NEGATIVE RESULT WE ARE NOT REDISCOVERING: empirical Bayes with a SINGLE
# prior variance tau^2 gives shrinkage d_i tau^2 / (d_i tau^2 + sigma^2), which
# is EXACTLY ridge at lambda = sigma^2/tau^2. A common prior IS the overarching
# shrinkage we already have. The content is in a NON-CONSTANT prior, ESTIMATED
# rather than parameterized as a shape — the shape family was already tested
# (power/exponential/step are interchangeable, and the tilt loses where it is
# actually applied: DiD z = -4.15).
#
# WHY IT MATTERS FOR THE PAPER: if a zero-hyperparameter estimator matches or
# beats the tuned arms, the whole tuning apparatus — and the endpoint pinning
# that contaminates it — was unnecessary, and the paper's methodological
# conclusion turns from cautionary into constructive.
_SHRINK_PROFILE_DECILES = np.arange(1, 10) / 10.0


def roll_rank_rcond(n_updates: int) -> float:
    """Relative eigenvalue-retention tolerance for a ROLLED gram, DERIVED from
    the update path rather than adopted from a library default.

    WHY A DERIVED TOLERANCE, AND WHAT IT IS *NOT* FOR (measured on the real
    panel, 2026-08-08). An earlier version of this docstring claimed that
    ``PINV_RCOND = 1e-15`` sits below the rolled gram's noise floor and so
    retains amplified junk. That claim came from a synthetic fixture with true
    rank 25 and a CONTINUOUS spectrum, and it does NOT hold on this design.
    Measured on the real b2 panel (242,934 x 1,077, driven through
    ``RollingLeastSquares`` with ``_walk_ols``'s exact call pattern over
    2,800-3,000 rolls, two eras):
        rolled-vs-exact gram, relative      1.7e-15 .. 6.7e-15
        same gram, summation order reversed 7.4e-15   (i.e. rolling costs
                                            nothing beyond forming X'X at all)
        null cluster top                    2.9e-15
        smallest GENUINE eigenvalue         7.4e-05   (an ~11-decade gap)
        retained rank, rcond 1e-15 .. 1e-12 IDENTICAL (597 / 585)
        forecasts across those four decades BIT-IDENTICAL (rms delta 0.0)
        max diag(G^+)                       6.5e+03, not 7.8e+11
    The design's spectrum is bimodal because pre-go-live indicator columns give
    bit-exact ZERO gram rows, and a rank-one update of a zero row stays zero
    (median smallest positive eigenvalue 1.2e-31). Any cutoff inside that gap
    gives the same answer, and ``PINV_RCOND`` is inside it. So ``_walk_ols`` is
    sound as it stands and Section 4 needs no re-run.

    The tolerance below is therefore a MARGIN, not a correction. It is derived
    rather than picked so that the one thin spot closes: the largest measured
    eigenvalue perturbation (1.12e-8) sat only 1.2x below the production cutoff
    (1.32e-8) — never enough to tip a direction over 2,800 rolls, and the gap
    above is six decades wide, but 1.2x is not a margin worth defending in
    print. On a fixture with a continuous spectrum the retained junk would
    the shrinkage is computed from — which is why this cannot be left at the
    library default here.

    THE DERIVATION. The gram is not formed once; it is accumulated. Building it
    costs ``n`` fused products per entry and each subsequent bar applies two
    rank-one updates (add the entering row, subtract the leaving one). Floating
    point contributes O(eps) relative error per operation and these accumulate
    linearly in the worst case, so after ``n_updates`` operations the entries
    carry O(n_updates * eps) relative error and eigenvalues below that are
    indistinguishable from zero. Measured on the fixture: the true relative
    discrepancy between the rolled and exactly-formed grams was 2.635e-14
    against a bound of n*eps = 8.882e-14 — the bound is ~3.4x conservative,
    which is the right side to err on, and it still sits ~13 orders of
    magnitude below the smallest genuine direction (2.51e-01).

    PER WINDOW, NOT FIXED: the floor is a property of the UPDATE PATH and the
    window length, both of which the caller knows exactly, so it is computed
    from the running update count rather than frozen as a literal. Nothing here
    is tuned — change the window and the tolerance follows the same derivation.

    ``_walk_ols`` IS DELIBERATELY UNTOUCHED, and now for a measured reason
    rather than an assumed one: on this design its cutoff sits in the middle of
    an 11-decade spectral gap, retains zero below-floor directions, and yields
    bit-identical forecasts across four decades of rcond (see above). Its
    estimator is also a projection rather than an inverse. There is nothing to
    fix there. The converse also holds and is measured: applying THIS tolerance
    to ``_walk_ols`` would be over-aggressive — on ``a_bucket_all_features``
    chunk 0, W*eps = 5.33e-12 cuts an eigenvalue (6.17e-6 relative) that is
    present at the same magnitude in the exactly-formed gram, i.e. genuine
    data 1,637x above the roll perturbation, moving mean QLIKE by -1.485e-4
    and per-bar loss by up to 0.178 (independent audit, 2026-08-08). The two
    paths need different tolerances for measured reasons; do not unify them.
    """
    return max(int(n_updates), 1) * float(np.finfo(np.float64).eps)


def _integrated_act(resid: np.ndarray, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time tau of a residual series, by GEYER's
    INITIAL POSITIVE SEQUENCE truncation. Deterministic; nothing is tuned.

    tau = 1 + 2 * sum_{k>=1} rho_k is the factor by which serial correlation
    inflates the variance of a sample mean: the effective sample is n/tau, not
    n. The infinite sum is not estimable, so it is truncated by Geyer's rule —
    pair the autocorrelations, Gamma_j = rho_{2j} + rho_{2j+1}, and stop at the
    first j with Gamma_j <= 0. For a reversible/stationary process the Gamma_j
    are positive and decreasing in population, so the first non-positive pair
    sum is noise, not signal. The rule has NO bandwidth to choose, which is why
    it is used here in preference to a Newey-West window (whose bandwidth
    would be exactly the kind of tuned quantity these arms exist to remove).

    The ACF is computed by FFT, O(n log n), so this is negligible against the
    O(p^3) solve it accompanies. Normalization is the biased (divide-by-n)
    estimator, the standard choice for ACT: it damps high-lag noise and keeps
    the sequence positive-definite.

    CLAMPED AT 1.0. A tau below 1 means net NEGATIVE serial correlation, which
    would license SHRINKING LESS than the independence assumption does. This
    arm exists to test whether the independence assumption under-shrinks;
    letting the correction run the other way would be claiming a precision
    bonus from a noisy ACF tail. Clamping is the conservative direction.
    """
    x = np.asarray(resid, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 8:
        return 1.0
    ss = float(x @ x)
    if not np.isfinite(ss) or ss <= 0.0:
        return 1.0
    m = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, m)
    acf = np.fft.irfft(fx * np.conjugate(fx), m)[:n].real / ss
    lag_cap = n // 4 if max_lag is None else int(max_lag)
    rho = acf[: max(lag_cap, 2)]
    n_pair = (rho.size - 1) // 2
    if n_pair < 1:
        return 1.0
    gam = rho[0 : 2 * n_pair : 2] + rho[1 : 2 * n_pair + 1 : 2]
    nonpos = np.flatnonzero(gam <= 0.0)
    j_stop = int(nonpos[0]) if nonpos.size else int(gam.size)
    if j_stop < 1:
        return 1.0
    tau = 2.0 * float(gam[:j_stop].sum()) - 1.0
    return float(max(tau, 1.0)) if np.isfinite(tau) else 1.0


def _psd_pinv_parts(
    mat: np.ndarray, rcond: float | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """(retained eigenvectors, retained eigenvalues, rank) of a symmetric PSD
    matrix, retaining ``lam > rcond * lam_max``.

    ``rcond=None`` falls back to ``PINV_RCOND`` (the campaign's library-default
    convention, kept so a caller that wants exact parity with ``_walk_ols`` can
    ask for it). The shrinkage path passes the DERIVED tolerance from
    :func:`roll_rank_rcond`. Also discards FP-negative modes.
    """
    lam, vec = np.linalg.eigh(mat)
    lam_max = float(lam[-1]) if lam.size else 0.0
    if lam_max <= 0.0:
        return vec[:, :0], lam[:0], 0
    keep = lam > (PINV_RCOND if rcond is None else float(rcond)) * lam_max
    return vec[:, keep], lam[keep], int(keep.sum())


class _BlockFit:
    """Min-norm least-squares fit of a partitioned design, rank-aware.

    WHY THIS EXISTS (defect found 2026-08-07, after the first grid-free run):
    James-Stein shrinkage of the OLS estimator presumes OLS EXISTS. This
    design's gram is ALWAYS rank-deficient — the 576 availability indicators
    contain columns that are dead or constant inside any given 24000-bar
    window, and the go-live rule guarantees it. The first implementation tried
    a Cholesky, caught the LinAlgError, fell back to min-norm, and returned
    ``None`` for the factorization, at which point the JS factor degraded to
    1.0. That fallback fired on 100% of bars, so the family scored UNSHRUNK
    min-norm least squares on a wide design (QLIKE 0.2785-0.2813 against a
    0.2335 benchmark) and never ran its own estimator. The specification was
    wrong, not the code: rank deficiency here is the NORMAL case and the
    estimator has to be defined on it.

    THE FIX. Work in the design's actual row space:
      * beta_hat is the MIN-NORM least-squares solution, i.e. the projection
        onto the row space, with sampling covariance sigma^2 G^+ (Moore-
        Penrose) supported on an r-dimensional subspace, r = rank(G) << p.
      * The backbone is partialled out exactly as before (Frank-Wolfe-Lovell),
        so the object carrying the shrinkage is the BACKBONE-RESIDUALIZED
        gram G_S|B = G_SS - G_SB G_BB^+ G_BS. In the full-rank case that is
        precisely the Schur complement the shipped code used, so this is the
        strict generalization of the previous estimator, not a different one.
      * ONE eigendecomposition of G_S|B yields everything the estimators need:
        the min-norm beta_S, the retained rank r_S, the Wald quantity, and
        diag(G_S|B^+) for the per-coefficient standard errors.
      * The backbone coefficient is recovered by the FWL identity from the
        UNSHRUNK beta_S, which reproduces the joint least-squares backbone
        exactly — so "backbone unshrunk" still means bit-identical.
    """

    def __init__(
        self,
        gram: np.ndarray,
        rhs: np.ndarray,
        idx_b: np.ndarray,
        idx_s: np.ndarray,
        rcond: float | None = None,
    ) -> None:
        g_ss = gram[np.ix_(idx_s, idx_s)]
        c_s = rhs[idx_s]
        if idx_b.size:
            g_bb = gram[np.ix_(idx_b, idx_b)]
            g_bs = gram[np.ix_(idx_b, idx_s)]
            vb, lb, self.rank_b = _psd_pinv_parts(g_bb, rcond)
            # G_BB^+ applied without forming the dense pseudo-inverse
            pinv_bs = vb @ ((vb.T @ g_bs) / lb[:, None]) if self.rank_b else 0.0 * g_bs
            pinv_cb = (
                vb @ ((vb.T @ rhs[idx_b]) / lb) if self.rank_b else np.zeros(idx_b.size)
            )
            self.g_sb_res = g_ss - g_bs.T @ pinv_bs
            self.c_res = c_s - g_bs.T @ pinv_cb
            self._g_bb_parts = (vb, lb)
            self._g_bs = g_bs
            self._c_b = rhs[idx_b]
        else:
            self.rank_b = 0
            self.g_sb_res, self.c_res = g_ss, c_s
            self._g_bb_parts = (np.empty((0, 0)), np.empty(0))
            self._g_bs = np.empty((0, idx_s.size))
            self._c_b = np.empty(0)
        self.vec, self.lam, self.rank_s = _psd_pinv_parts(self.g_sb_res, rcond)
        self.rcond = rcond
        if self.rank_s:
            proj = self.vec.T @ self.c_res  # coordinates in the retained basis
            self.beta_s = self.vec @ (proj / self.lam)
            # ||gamma||^2 with gamma = Lam^{1/2} V' beta_s: spherical by
            # construction, since Cov(beta_s) = sigma^2 (G_S|B)^+ restricted
            # to the retained subspace. Equals c_res' (G_S|B)^+ c_res.
            self.wald = float(np.sum(proj * proj / self.lam))
        else:
            self.beta_s = np.zeros(idx_s.size)
            self.wald = 0.0

    def backbone(self, beta_s: np.ndarray) -> np.ndarray:
        """FWL recovery of the backbone coefficient — fed the UNSHRUNK beta_s,
        this reproduces the joint least-squares backbone bit-identically."""
        vb, lb = self._g_bb_parts
        if not self.rank_b:
            return np.zeros(self._c_b.size)
        resid = self._c_b - self._g_bs @ beta_s
        return vb @ ((vb.T @ resid) / lb)

    def diag_cov(self, sigma2: float) -> np.ndarray:
        """diag(sigma^2 (G_S|B)^+) — per-coefficient sampling variances. Exactly
        zero on columns the row space does not reach (dead/constant indicators),
        which is how the NPEB path identifies and excludes them."""
        if not self.rank_s:
            return np.zeros(self.g_sb_res.shape[0])
        return sigma2 * ((self.vec * self.vec) / self.lam).sum(1)


def _sigma2_hat(syy_c: float, beta: np.ndarray, rhs: np.ndarray, n: int, rank: int):
    """(sigma^2, residual dof) from the window's own residuals.

    RSS = y_c'y_c - beta'X_c'y_c, exact for ANY least-squares solution
    including the min-norm one (it satisfies the normal equations on the row
    space, so the cross term collapses), so no residual vector is ever formed —
    it is read straight off the rolling sufficient statistics.

    DEGREES OF FREEDOM: n - r - 1, where r is the RETAINED RANK of the design
    (backbone rank + residualized-block rank) and the +1 is the intercept
    absorbed by centering. NOT n - p: on a rank-deficient design the null-space
    directions cost no degrees of freedom, and charging for them would inflate
    sigma^2 and therefore over-shrink. It is deliberately NOT reduced further
    for the shrinkage itself — the factor is estimated FROM this sigma^2, so a
    shrinkage-aware dof would be circular.
    """
    dof = max(int(n) - int(rank) - 1, 1)
    rss = float(syy_c) - float(beta @ rhs)
    return max(rss, 0.0) / dof, dof


def _js_factor_rank(wald: float, rank_s: int, sigma2: float, dof: int) -> float:
    """POSITIVE-PART JAMES-STEIN factor on a RANK-DEFICIENT design.

    Identical to the full-rank estimator except that the dimension entering the
    JS constant is the RETAINED RANK r of the subspace the estimator actually
    lives in, NOT the column count p:

        f = (1 - [(r-2)/(m+2)] * m * sigma2_hat / ||gamma||^2)_+

    with ||gamma||^2 = beta_S' (G_S|B)^+ ^+ beta_S = c_res' (G_S|B)^+ c_res the
    Wald quantity over the retained directions only, and m the residual dof.
    Using p here instead of r would be a straightforward error: the null-space
    directions carry no estimation risk, so counting them would over-shrink by
    the ratio p/r.

    Defined only for r > 2 — with r <= 2 there is no Stein effect to exploit
    and the estimator returns 1.0. That case is counted as genuinely
    pathological (see ``_walk_shrink``'s n_singular), unlike ordinary rank
    deficiency, which is now the NORMAL path rather than a fallback.
    """
    if rank_s <= 2 or not np.isfinite(wald) or wald <= 0.0:
        return 1.0
    m = float(dof)
    return max(0.0, 1.0 - ((rank_s - 2) / (m + 2)) * m * sigma2 / wald)


def _tweedie_shrink(z: np.ndarray) -> np.ndarray:
    """NONPARAMETRIC EMPIRICAL BAYES posterior means via Tweedie's formula.

    For z_i | mu_i ~ N(mu_i, 1) with marginal density f, Tweedie gives the
    posterior mean in closed form, WITHOUT ever parameterizing the prior:
        E[mu | z] = z + d/dz log f(z).
    Standardizing each coefficient by its own standard error is what makes the
    unit-variance assumption hold BY CONSTRUCTION, so the estimator is applied
    in exactly the setting it is derived for.

    WHY THIS AND NOT A PARAMETRIC MIXTURE: the whole point is a prior the DATA
    chooses. A two-group or spike-and-slab prior would reintroduce exactly the
    hand-designed structure these arms exist to remove.

    IMPLEMENTATION, chosen for numerical robustness at this dimension (~1200
    coefficients per window):
      * f and f' come from a Gaussian KDE evaluated ANALYTICALLY, never by
        finite differences. Both are smooth sums over the sample, so
        f'/f is a weighted average of -(z - z_j)/h^2 and is therefore BOUNDED
        by max_j |z - z_j| / h^2 — no tail blow-up of the kind that makes naive
        density-ratio estimates unusable.
      * f is evaluated AT THE SAMPLE POINTS, so every f(z_i) >= phi(0)/(n h) is
        strictly positive by construction and the ratio can never divide by
        zero. This is the reason to evaluate on-sample rather than on a grid.
      * bandwidth by Silverman's rule, h = 0.9 min(sd, IQR/1.34) n^(-1/5). This
        is a DETERMINISTIC function of the data, not a tuned hyperparameter:
        nothing is selected, nothing is validated. Degenerate spread (h <= 0)
        returns the identity, i.e. no shrinkage.

    DOCUMENTED APPROXIMATION: Tweedie treats the z_i as a sample from one
    marginal, i.e. as independent. The coefficients are NOT independent — the
    rolling gram is not diagonal — so the estimated marginal is the empirical
    distribution of correlated draws. This is the standard NPEB compromise at
    this dimension and it is why the PC-basis twin exists: rotating toward the
    frame's eigenbasis is what makes the independence assumption least wrong.

    NO CLIPPING. Tweedie can return |E[mu|z]| > |z| where the estimated
    marginal is multimodal; that is a legitimate posterior mean under a
    multimodal prior, not an error, and clipping it would be exactly the kind
    of ad-hoc bound this campaign refuses. The full factor distribution is
    persisted per boundary so any pathology is visible rather than hidden.

    Returns the posterior means (same shape as z).
    """
    n = z.size
    if n < 2:
        return z.copy()
    sd = float(np.std(z, ddof=1))
    q75, q25 = np.percentile(z, [75, 25])
    spread = min(sd, float(q75 - q25) / 1.34) if q75 > q25 else sd
    h = 0.9 * spread * n ** (-0.2)
    if not np.isfinite(h) or h <= 0.0:
        return z.copy()
    u = (z[:, None] - z[None, :]) / h  # (n, n) standardized pair distances
    k = np.exp(-0.5 * u * u)
    dens = k.sum(1)  # proportional to f(z_i); constants cancel in f'/f
    dscore = -(k * u).sum(1) / h  # proportional to f'(z_i), same constant
    return z + dscore / dens


def _shrink_profile(factors: np.ndarray, row: int, extra: dict) -> dict:
    """One persisted row of the shrinkage-factor exhibit — the paper's picture
    of the shrinkage the DATA asks for, against the shapes we imposed."""
    f = np.asarray(factors, dtype=np.float64)
    rec = {
        "row": int(row),
        "n_coef": int(f.size),
        "mean": float(np.mean(f)),
        "median": float(np.median(f)),
        "frac_below_0p1": float(np.mean(f < 0.1)),
        "frac_above_0p9": float(np.mean(f > 0.9)),
        "frac_negative": float(np.mean(f < 0.0)),
        "deciles": [float(v) for v in np.quantile(f, _SHRINK_PROFILE_DECILES)],
    }
    rec.update(extra)
    return rec


def _walk_shrink(
    F: np.ndarray,
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    segments: list[tuple[int, int, str]],
    estimator: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-bar-refit walk-forward with GRID-FREE, ESTIMATED shrinkage.

    No grid, no validation tail, no tuned hyperparameter — the shrinkage is a
    function of the training window's own sufficient statistics. The backbone
    block (HAR ladder + session interactions + calendar) is left UNSHRUNK and
    its coefficients are bit-identical to the plain least-squares solve: JS's
    risk guarantee is a statement about the shrunk block, and the backbone is
    the incumbent's own audited basis, not something this estimator is entitled
    to touch.

    ``estimator``:
      "js"      positive-part James-Stein, EXACT block covariance via the
                Schur complement (see _js_shrink_factor).
      "js_diag" the same, but whitening by the DIAGONAL of the block covariance
                only — i.e. pretending the coefficients decouple. Exact iff the
                gram is diagonal; the gap to "js" measures what the decoupling
                costs in a given basis.
      "npeb"    per-coefficient Tweedie posterior means (see _tweedie_shrink).

    CAUSALITY: every quantity — gram, rhs, y'y, sigma^2, the factor — is built
    from rows strictly inside [t-window, t). Row t enters only through the
    prediction, and only after the coefficients are fixed. Asserted end-to-end
    in the verify script by perturbing post-window rows and requiring
    bit-identical coefficients.
    """
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    p_ = Xs.shape[1]
    # the backbone block is the one carrying the "backbone" alpha key; every
    # other column belongs to the shrunk set
    shrink_mask = np.ones(p_, dtype=bool)
    for s0, s1, key in segments:
        if key == "backbone":
            shrink_mask[s0:s1] = False
    idx_s = np.where(shrink_mask)[0]
    idx_b = np.where(~shrink_mask)[0]
    if idx_s.size == 0:
        raise SystemExit("grid-free shrinkage: empty shrink block")

    # "<est>_neff" selects the serial-correlation-corrected noise scale
    want_neff = estimator.endswith("_neff")
    if want_neff:
        estimator = estimator[: -len("_neff")]
    # 1.0 == "no serial-correlation correction", which is exactly what the
    # uncorrected arms apply: sigma2 * 1.0 is bit-identical to sigma2, so
    # blk3_js_tuned's numbers stay reproducible while every arm records the
    # key. Only the *_neff arms ever re-estimate it.
    tau_resid: float = 1.0
    solver = RollingLeastSquares(alpha=0.0, fit_intercept=True)
    solver.init_window(Xs[:window], ys[:window])
    syy = float(ys[:window] @ ys[:window])  # rolled alongside; RLS omits it
    out = np.empty(hi - lo, dtype=np.float64)
    profile: list[dict[str, Any]] = []
    n_singular = 0
    n = float(window)
    for i in range(hi - lo):
        t = window + i
        gram = solver._Sxx - np.outer(solver._sx, solver._sx) / n
        rhs = solver._Sxy - solver._sx * (solver._sy / n)
        syy_c = syy - (solver._sy * solver._sy) / n
        # DERIVED retention tolerance: window build + 2 rank-one updates per
        # bar rolled so far (see roll_rank_rcond).
        rcond = roll_rank_rcond(int(n) + 2 * i)
        fit = _BlockFit(gram, rhs, idx_b, idx_s, rcond)
        if want_neff and i % TUNE_PER == 0:
            # RESIDUAL SERIAL CORRELATION, re-estimated on the retune cadence
            # and held between boundaries (tau is a slowly varying property of
            # the panel, and forming the residual vector is O(W p) per call).
            # Strictly causal: the residuals are the CURRENT window's own.
            b_tmp = np.zeros(p_)
            b_tmp[idx_s] = fit.beta_s
            b_tmp[idx_b] = fit.backbone(fit.beta_s)
            xw_, yw_ = Xs[i:t], ys[i:t]
            fitted = xw_ @ b_tmp
            tau_resid = _integrated_act(yw_ - fitted - float(np.mean(yw_ - fitted)))
        # r <= 2 is GENUINELY pathological (no Stein effect exists); ordinary
        # rank deficiency is the normal path here, not a fallback.
        if fit.rank_s <= 2:
            n_singular += 1
        beta = np.zeros(p_)
        beta_s = fit.beta_s
        beta[idx_s] = beta_s
        beta[idx_b] = fit.backbone(beta_s)  # UNSHRUNK beta_s => joint-LS backbone
        rank_total = fit.rank_b + fit.rank_s
        sigma2, dof = _sigma2_hat(syy_c, beta, rhs, int(n), rank_total)
        # SERIAL-CORRELATION CORRECTION. With correlated residuals the
        # effective sample is n/tau, so the coefficient covariance
        # sigma^2 (X'X)^+ is understated by ~tau and the Wald form is
        # overstated by ~tau. Inflating the noise scale by tau is exactly
        # equivalent to dividing the Wald quantity by it.
        # DOF IS DELIBERATELY NOT ALSO DIVIDED: n/tau at tau~175 is ~137,
        # far below the retained rank, so an "effective dof" of n/tau - r - 1
        # would go NEGATIVE. dof normalizes the RSS, which is a sum over
        # actual rows and is correct as it stands; tau belongs to the
        # SAMPLING variance of the coefficients, which is where it is applied.
        sigma2_js = sigma2 * tau_resid if want_neff else sigma2

        if estimator == "npeb":
            var = fit.diag_cov(sigma2_js)
            se = np.sqrt(np.maximum(var, 0.0))
            # EXCLUDE the null space entirely. Under min-norm those
            # coefficients are exactly zero and their sampling variance is
            # exactly zero; feeding them to Tweedie would flood the marginal
            # with a spike at z=0 and corrupt the density estimate that IS the
            # estimated prior.
            # ROW-SPACE membership, on the SAME derived relative scale as
            # the rank retention. A bare ``> 0`` test is not enough: a
            # null-space column's variance is ~1e-30 of the largest, not
            # exactly 0, so it would survive and flood the marginal with a
            # spike at z ~ 0/0 — precisely the corruption this exclusion
            # exists to prevent (unguarded, the fixture's mean factor came
            # back at -2.5e8).
            good = np.isfinite(var) & (var > rcond * float(np.max(var)))
            new_s = beta_s.copy()
            fac = np.ones(idx_s.size)
            if good.any():
                z = beta_s[good] / se[good]
                mu = _tweedie_shrink(z)
                new_s[good] = mu * se[good]
                with np.errstate(divide="ignore", invalid="ignore"):
                    f_all = np.where(np.abs(z) > 0.0, mu / z, 1.0)
                fac[good] = np.where(np.isfinite(f_all), f_all, 1.0)
            beta[idx_s] = new_s
            beta[idx_b] = fit.backbone(beta_s)
            factors = fac[good] if good.any() else fac
            extra = {
                "sigma2": sigma2,
                "tau_resid": tau_resid,
                "n_eff_resid": float(n) / tau_resid,
                "rank_s": fit.rank_s,
                "rank_b": fit.rank_b,
                "n_in_rowspace": int(good.sum()),
                "frac_in_rowspace": float(good.mean()),
            }
        elif estimator == "js_diag":
            # DECOUPLED whitening: treat the retained coefficients as
            # independent, i.e. use only diag((G_S|B)^+). Exact iff the
            # residualized gram is diagonal; the gap to "js" is what the
            # decoupling costs in a given basis. The effective dimension is the
            # count of coefficients the row space actually reaches.
            cdiag = fit.diag_cov(1.0)
            good = np.isfinite(cdiag) & (cdiag > rcond * float(np.max(cdiag)))
            k_diag = int(good.sum())
            wald_d = float(np.sum(beta_s[good] ** 2 / cdiag[good])) if k_diag else 0.0
            f = _js_factor_rank(wald_d, k_diag, sigma2_js, dof)
            beta[idx_s] = f * beta_s
            factors = np.full(1, f)
            extra = {
                "sigma2": sigma2,
                "tau_resid": tau_resid,
                "n_eff_resid": float(n) / tau_resid,
                "js_factor": float(f),
                "rank_s": fit.rank_s,
                "rank_b": fit.rank_b,
                "n_in_rowspace": k_diag,
                "frac_in_rowspace": float(k_diag) / max(idx_s.size, 1),
            }
        else:
            f = _js_factor_rank(fit.wald, fit.rank_s, sigma2_js, dof)
            beta[idx_s] = f * beta_s
            factors = np.full(1, f)
            extra = {
                "sigma2": sigma2,
                "tau_resid": tau_resid,
                "n_eff_resid": float(n) / tau_resid,
                "js_factor": float(f),
                "rank_s": fit.rank_s,
                "rank_b": fit.rank_b,
                "frac_in_rowspace": float(fit.rank_s) / max(idx_s.size, 1),
            }

        intercept = solver._sy / n - float(solver._sx @ beta) / n
        out[i] = float(Xs[t] @ beta + intercept)
        if i % TUNE_PER == 0:
            # same cadence as the tuned arms' retune boundaries, so the exhibit
            # is directly comparable to their trajectories
            profile.append(_shrink_profile(factors, lo + i, extra))
        y_in, y_out = ys[t], ys[t - window]
        syy += float(y_in) * float(y_in) - float(y_out) * float(y_out)
        solver.roll(Xs[t], y_in, Xs[t - window], y_out)
    if n_singular:
        print(
            f"[grid-free shrinkage] DEGENERATE rank (r<=2) on {n_singular} of "
            f"{hi - lo} bars — no Stein effect exists there; this counts "
            "genuine pathology, NOT ordinary rank deficiency"
        )
    for rec in profile:
        rec["n_singular_bars"] = n_singular
    return out, profile


# ── DISCOUNTED-GRAM (exponentially weighted) FAMILY ───────────────────────────
# MOTIVATION, from the JS verdict (2026-08-08). blk3_js_tuned scored 0.24631
# (DM +26.6 vs blk3_tuned) with its estimator VERIFIED firing — mean shrinkage
# factor 0.77, stable, zero singular bars. So risk-optimal IN-SAMPLE shrinkage
# (23%) is orders of magnitude below forecast-optimal shrinkage. The reading:
# beta DRIFTS, and in-sample precision is a bad guide to forecast risk when the
# coefficients being estimated are not constant over the window. Heavy ridge on
# a flat 24000-bar window is a crude patch for that — it distrusts ALL evidence
# equally when the real problem is that OLD evidence is STALE. Window length is
# the one hyperparameter this campaign never varied.
#
# THE OBJECT: exponentially weighted sufficient statistics
#     G_t = sum_s w^(t-s) x_s x_s',   c_t = sum_s w^(t-s) x_s y_s
# with forgetting factor w, parameterized by HALF-LIFE H (bars):
#     w = 2^(-1/H)   equivalently   H = ln(2) / ln(1/w).
#
# TRUNCATED, NOT INFINITE-MEMORY — a deliberate deviation from the brief, for
# two reasons that both matter more than the saved flop:
#   (1) NESTING. The brief asks that H=infinity be the flat window EXACTLY. An
#       infinite-memory recursion (G_t = w G_{t-1} + x_t x_t') at w=1 is a
#       CUMULATIVE sum over all history, not the 24000-bar window, so it would
#       NOT nest blk3_tuned. Summing over the same trailing W bars does:
#           G_t = sum_{s=t-W+1..t} w^(t-s) x_s x_s'
#       and at w=1 this is the sliding-window gram term for term.
#   (2) COMPARABILITY. Holding the data span fixed at W=24000 means these arms
#       vary the WEIGHTING and nothing else — same rows, same legality, same
#       first legal OOS row as every other frozen-construction arm.
# The cost is that the recurrence keeps a subtraction:
#     G_{t+1} = w G_t + x_{t+1} x_{t+1}' - w^W x_{t-W+1} x_{t-W+1}'
# which at w=1 is exactly the existing sliding-window update. So it is the same
# price as the incumbent, not cheaper — the brief's "no subtraction" saving is
# available only to the non-nesting infinite-memory variant.
#
# NO POWERS OF w ARE EVER ACCUMULATED: the recurrence is applied per bar and
# the single boundary weight w^W is computed as exp(-ln2 * W / H) in closed
# form. For the shipped grid the smallest is H=1000 -> w^W = 2^-24 = 6.0e-8,
# nowhere near underflow; the closed form keeps that true for any H.
EW_HALF_LIVES: tuple[float, ...] = (1000.0, 3000.0, 6000.0, 12000.0, 24000.0, np.inf)


def ew_decay(half_life: float) -> float:
    """Forgetting factor w from a half-life in bars. H=inf -> w=1.0 EXACTLY
    (float equality, not approximately), which is what makes the flat-window
    nesting bit-exact rather than merely close."""
    if not np.isfinite(half_life):
        return 1.0
    if half_life <= 0.0:
        raise SystemExit(f"ew: half-life must be positive, got {half_life}")
    return float(np.exp(-np.log(2.0) / float(half_life)))


def ew_weights(half_life: float, w_len: int) -> np.ndarray:
    """Weights w^(t-s) for the trailing ``w_len`` rows, OLDEST FIRST, so the
    most recent row carries weight 1.0. At H=inf this is exactly ones(w_len)."""
    w = ew_decay(half_life)
    if w == 1.0:
        return np.ones(int(w_len))
    age = np.arange(int(w_len) - 1, -1, -1, dtype=np.float64)
    return np.exp(-np.log(2.0) * age / float(half_life))


def ew_n_eff(half_life: float, w_len: int) -> float:
    """KISH effective sample size of the truncated EW window:
        n_eff = (sum_k w^k)^2 / sum_k w^(2k),  k = 0 .. W-1.
    At w=1 this is exactly W (the flat window's own count). As W -> infinity it
    converges to the (1+w)/(1-w) form quoted in the brief; the FINITE sum is
    used because the window IS finite, and for H comparable to W the two differ
    by a factor of three (H=24000, W=24000: finite 23083 vs
    infinite-horizon 69249) — quoting the infinite form on a truncated
    window would overstate the evidence by that factor.
    This is the dof any sigma^2 on this path must use — the arms themselves
    tune penalties by validation MSE and need no sigma^2, so it is persisted as
    a diagnostic rather than consumed."""
    w = ew_decay(half_life)
    n = int(w_len)
    if w == 1.0:
        return float(n)
    s1 = (1.0 - w**n) / (1.0 - w)
    s2 = (1.0 - w ** (2 * n)) / (1.0 - w * w)
    return float(s1 * s1 / s2)


def _ew_weighted_stats(x: np.ndarray, y: np.ndarray, wts: np.ndarray):
    """(centered weighted gram, rhs, sum of weights) for one block of rows.

    Weighted centering is the generalization of the incumbent's centering: the
    unpenalized intercept rides via the WEIGHTED mean, so at w=1 every
    expression below collapses term-for-term to the flat-window formulas
    ``_walk_blocks_tuned`` uses.
    """
    sw = float(wts.sum())
    xw = x * wts[:, None]
    sx = xw.sum(0)
    sy = float(wts @ y)
    gram = x.T @ xw - np.outer(sx, sx) / sw
    rhs = xw.T @ y - sx * (sy / sw)
    return gram, rhs, sw, sx, sy


class _EwStats:
    """Rolling EW sufficient statistics over a FIXED-LENGTH trailing window.

    Maintains Sxx/Sxy/sx/sy under
        S_{t+1} = w S_t + (entering) - w^W (leaving)
    which at w=1.0 is bit-identical to ``RollingLeastSquares``'s update. The
    total weight ``sw`` is time-invariant (a geometric sum over the fixed
    window length) so it is computed once in closed form.

    Drift: with w < 1 old contributions decay, so update error is
    self-correcting; at w = 1 the behaviour is the incumbent's. The caller
    additionally rebuilds exactly at every retune boundary (every TUNE_PER
    bars), which bounds accumulation regardless.
    """

    def __init__(self, x_win: np.ndarray, y_win: np.ndarray, half_life: float) -> None:
        self.w = ew_decay(half_life)
        self.n = int(len(y_win))
        self.w_pow = 1.0 if self.w == 1.0 else float(self.w**self.n)
        wts = ew_weights(half_life, self.n)
        xw = x_win * wts[:, None]
        self._Sxx = x_win.T @ xw
        self._Sxy = xw.T @ y_win
        self._sx = xw.sum(0)
        self._sy = float(wts @ y_win)
        self.sw = float(wts.sum())

    def centered(self):
        gram = self._Sxx - np.outer(self._sx, self._sx) / self.sw
        rhs = self._Sxy - self._sx * (self._sy / self.sw)
        return gram, rhs

    def intercept(self, beta: np.ndarray) -> float:
        return self._sy / self.sw - float(self._sx @ beta) / self.sw

    def roll(self, x_in, y_in, x_out, y_out) -> None:
        w, wp = self.w, self.w_pow
        self._Sxx *= w
        self._Sxx += np.outer(x_in, x_in) - wp * np.outer(x_out, x_out)
        self._Sxy *= w
        self._Sxy += x_in * y_in - wp * x_out * y_out
        self._sx *= w
        self._sx += x_in - wp * x_out
        self._sy = w * self._sy + float(y_in) - wp * float(y_out)


def _dedup_ols_design(
    F: np.ndarray, col_names: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Deterministic exact-duplicate dedup for OLS designs (no shrinkage, ever).

    Over the FULL panel (so every chunk of an arm sees the identical design),
    drop (a) columns bitwise-equal to an earlier column — first in canonical
    order wins — and (b) panel-constant columns, which are exact duplicates of
    the unpenalized intercept. Returns (F_deduped, kept_names, dropped_names).
    """
    keep: list[int] = []
    dropped: list[str] = []
    seen: dict[bytes, int] = {}
    for j in range(F.shape[1]):
        col = np.ascontiguousarray(F[:, j])
        if col.max() == col.min():  # panel-constant ≡ intercept duplicate
            dropped.append(col_names[j])
            continue
        key = col.tobytes()  # bitwise equality, not tolerance-based
        if key in seen:
            dropped.append(col_names[j])
            continue
        seen[key] = j
        keep.append(j)
    kept_names = [col_names[j] for j in keep]
    return np.ascontiguousarray(F[:, keep]), kept_names, dropped


def _eliminate_exact_collinear(
    F: np.ndarray, col_names: list[str], window: int, lo: int
) -> tuple[np.ndarray, list[str], list[str], dict[str, list[int]]]:
    """RETAINED-UNUSED (2026-08-07): the OLS family moved to minimum-norm least
    squares (see _walk_ols), which handles rank deficiency natively, so this
    guard is no longer wired into any arm. It stays implemented — the guard
    stack remains in force for any future arm using a plain normal-equations
    solve, and the c080 ladder analysis below documents a real failure class.

    Deterministic exact-collinearity elimination for OLS designs (ruling
    2026-08-06, cluster loud-fails): drop columns that are EXACT linear
    combinations of retained earlier columns — accounting identities like
    total turnover = buy-initiated + sell-initiated, replicated by the MA
    ladder at every rung. Stays inside the no-shrinkage discipline: columns are
    removed, never penalized.

    Method: one unpivoted Householder QR of the CENTERED chunk-first-window
    design (rows [lo-window, lo)). Unpivoted QR processes columns in canonical
    order, so |R_jj| is the residual norm of column j after projection on the
    RETAINED span of columns 0..j-1 (an already-dependent predecessor adds no
    span, to working precision) — first-in-canonical-order wins by
    construction. Centering makes dependence-up-to-a-constant collapse onto the
    unpenalized intercept.

    "Exact" criterion (machine-precision-derived, NOT a tuned epsilon): the
    numpy.linalg.matrix_rank / LAPACK rank convention (Golub & Van Loan)
    applied column-wise —

        drop j  iff  |R_jj| <= max(n_rows, n_cols) * eps_float64 * ||col_j||

    with ||col_j|| the centered column norm. At n=24000 this is ~5e-12
    relative: generous against float64 accumulation in a true identity
    (~sqrt(n)*eps), far below any economically distinct near-collinearity.
    Householder QR is backward-stable, so the diagnostic survives
    ill-conditioning that a gram-based check would not.

    CADENCE (stated per the ruling): the rank repair runs ONCE on the chunk's
    first training window; an exact identity holds at every row, so it holds in
    every subsequent window. Re-verification is the per-bar solve itself —
    _walk_ols retains the loud singularity/finiteness guard, so a dependency
    arising only mid-chunk still fails loudly rather than silently.

    Columns with ZERO dispersion in the first window are exempt (they belong to
    the per-bar identifiability mask and may go live mid-chunk). Returns
    (F_reduced, kept_names, collinear_dropped_names, ladder_drops) — the last
    being the atomic availability-ladder drops {ladder_prefix: rung_windows}
    (see the c080 block below).
    """
    Xw = np.asarray(F[lo - window : lo], dtype=np.float64)
    Xc = Xw - Xw.mean(0)
    norms = np.linalg.norm(Xc, axis=0)
    R = np.linalg.qr(Xc, mode="r")
    diag = np.abs(np.diag(R))
    eps = np.finfo(np.float64).eps
    tol = max(Xc.shape) * eps
    collinear = (norms > 0) & (diag <= tol * norms)

    # ATOMIC availability-ladder dropping (c080 root cause, ruling 2026-08-06):
    # if ANY rung of a stem's availability/occurrence-indicator MA ladder is
    # exactly dependent on retained columns at chunk entry, drop that stem's
    # ENTIRE indicator ladder for the chunk. RATIONALE: a partial ladder drop is
    # itself a DISCLOSED PROOF that the retained sibling rungs are
    # near-duplicates whose separating variation lies entirely before the
    # window (the short rungs went exactly dependent precisely because the
    # ladder's distinguishing history predates the window; the long rungs then
    # sit at ~1/rung separation — below any rank tolerance's radar but exactly
    # the canceling-coefficient configuration that amplified a one-bar
    # availability desync into yhat=+77.8 in chunk c080). Removing the whole
    # ladder is rank hygiene derived from that exact disclosed fact — not
    # shrinkage, and NO new tolerance is introduced (the trigger is the
    # existing exact-dependence criterion above).
    ladder_cols: dict[str, list[int]] = {}
    for j, nm in enumerate(col_names):
        m = re.match(r"^(.+_(?:avail|active))_ma_(\d+)$", nm)
        if m:
            ladder_cols.setdefault(m.group(1), []).append(j)
    ladder_drops: dict[str, list[int]] = {}
    for ladder, cols_ in ladder_cols.items():
        if any(collinear[j] for j in cols_):
            ladder_drops[ladder] = sorted(
                int(re.match(r"^.+_ma_(\d+)$", col_names[j]).group(1)) for j in cols_
            )
            for j in cols_:
                collinear[j] = True
    keep = np.flatnonzero(~collinear)
    dropped = [col_names[j] for j in np.flatnonzero(collinear)]
    kept_names = [col_names[j] for j in keep]
    return np.ascontiguousarray(F[:, keep]), kept_names, dropped, ladder_drops


def _support_masks(
    p: _Panel, col_names: list[str], window: int, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Go-live/go-dead participation masks (corrected ruling 2026-08-06 — the
    earlier every-bar full-support rule wrongly masked session-bound feeds
    that structurally print on a fraction of bars; designs collapsed).

    RULE (deterministic, threshold-free; window = [s, s+W)): a VALUE column
    participates iff BOTH
      (a) the stem printed at least once BEFORE the window starts
          (cnt[s] >= 1 — the window sees only the live regime), and
      (b) the stem printed at least once WITHIN the window
          (cnt[s+W] - cnt[s] >= 1 — the feed is not dead).
    Session-bound in-window ffill is legitimate under the fill discipline; only
    genuine pre-go-live and post-death regimes are masked. An availability-
    INDICATOR column applies (a) only: before go-live it is the pure toxic
    step; after go-live it varies at session cadence and carries the
    transition information ((b) is irrelevant — a dead feed's indicator decays
    to constant and falls to the zero-dispersion mask).

    Returns (pre_go_live, feed_dead) bool matrices of shape (hi-lo, p_design);
    O(rows x stems) once via cumulative counts, per-window flags by lookup.
    """
    m = hi - lo
    pre = np.zeros((m, len(col_names)), dtype=bool)
    dead = np.zeros((m, len(col_names)), dtype=bool)
    starts = np.arange(m) + (lo - window)  # global window starts s
    by_stem: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for j, nm in enumerate(col_names):
        kind, stem, _w = _classify(nm)
        if kind not in ("value", "indicator") or stem not in p.stem_index:
            continue
        flags = by_stem.get(stem)
        if flags is None:
            cnt = np.concatenate(
                [[0], np.cumsum(p.avail[:, p.stem_index[stem]].astype(np.int64))]
            )
            live_before = cnt[starts] >= 1
            printed_in = (cnt[starts + window] - cnt[starts]) >= 1
            flags = (live_before, printed_in)
            by_stem[stem] = flags
        live_before, printed_in = flags
        pre[:, j] = ~live_before
        if kind == "value":
            dead[:, j] = live_before & ~printed_in
    return pre, dead


# numpy.linalg.pinv's default rcond — the standard machine-precision cutoff
# convention (eigenvalues below PINV_RCOND * lambda_max are treated as zero).
# Cited, not invented: this is the documented numpy default, no new threshold.
PINV_RCOND = 1e-15


def _walk_ols(
    F: np.ndarray,
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    col_names: list[str],
    support_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """MINIMUM-NORM least-squares per-bar walk-forward (estimator change,
    author directive 2026-08-07: §4's story needs min-norm OLS).

    ESTIMATOR: at each bar, coefficients are the unique least-squares solution
    of minimal l2 norm on the centered training window — the alpha->0+ limit
    of ridge, defined at ANY design rank: coef = pinv(gram) @ rhs via
    eigendecomposition of the (symmetric PSD) centered gram, with eigenvalues
    below ``PINV_RCOND * lambda_max`` treated as zero (numpy.linalg.pinv's
    default rcond convention; for a symmetric PSD matrix eigh-based pinv is
    the SVD-based pinv). The unpenalized intercept rides via centering; a
    column with zero in-window dispersion centers to an exact null direction
    and receives coefficient 0 natively.

    IMPLEMENTATION CHOICE (simplest exact approach): one eigendecomposition
    per bar, recomputed from the rank-1-rolled sufficient statistics — no
    incremental eigen-update machinery, nothing approximate. COST: O(p^3)
    eigh per bar; ~1 s/bar at p~650 (all_features bucket), ~2-3 s/bar at the
    ~1,151-col worst case -> roughly 1-2.5 h per 2,763-bar v3 chunk for the
    widest arms (inside the 6 h spec); a0's ~50-col design is negligible.

    GUARD STACK (simplified per the same directive): pinv handles rank
    deficiency natively, so the exact-collinearity QR entry drops, atomic
    ladder drops, sticky lazy repair, and per-bar solve verification are
    REDUNDANT here and removed from this path — there is nothing left to
    loud-fail numerically. KEPT: panel-level bitwise dedup (cheap hygiene),
    the go-live/feed-dead participation masks (prep semantics, not numerics),
    and the fit/raw alignment gate (orthogonal concern). The retired guards
    remain implemented in :func:`_eliminate_exact_collinear` for any future
    arm using a plain normal-equations solve (none currently).

    NOTE: a0's backbone design is full-rank, so min-norm == plain OLS there;
    it is routed through this same path for uniformity.

    Returns (yhat, mask report: name -> {bars, first, last, reasons} with
    global row indices; reasons are pre_go_live / feed_dead only).
    """
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    p = Xs.shape[1]
    solver = RollingLeastSquares(alpha=0.0, fit_intercept=True)  # stats carrier
    solver.init_window(Xs[:window], ys[:window])
    out = np.empty(hi - lo, dtype=np.float64)
    mask_rows = np.zeros((hi - lo, p), dtype=bool)
    any_pre = np.zeros(p, dtype=bool)
    any_dead = np.zeros(p, dtype=bool)
    n = float(window)
    for i in range(hi - lo):
        t = window + i
        mask = np.zeros(p, dtype=bool)
        if support_masks is not None:
            pre_i, dead_i = support_masks[0][i], support_masks[1][i]
            mask = pre_i | dead_i
            any_pre |= pre_i
            any_dead |= dead_i
        mask_rows[i] = mask
        free = ~mask
        gram = solver._Sxx - np.outer(solver._sx, solver._sx) / n
        rhs = solver._Sxy - solver._sx * (solver._sy / n)
        coef = np.zeros(p)
        if free.any():
            sub = gram[np.ix_(free, free)]
            lam, V = np.linalg.eigh(sub)
            lam_max = float(lam[-1])
            if lam_max > 0.0:
                keep = lam > PINV_RCOND * lam_max  # also discards FP-negative modes
                if keep.any():
                    Vk = V[:, keep]
                    coef[free] = Vk @ ((Vk.T @ rhs[free]) / lam[keep])
        intercept = solver._sy / n - float(solver._sx @ coef) / n
        out[i] = float(Xs[t] @ coef + intercept)
        solver.roll(Xs[t], ys[t], Xs[t - window], ys[t - window])

    masked: dict[str, dict[str, Any]] = {}
    for j in np.flatnonzero(mask_rows.any(axis=0)):
        bars = np.flatnonzero(mask_rows[:, j])
        reasons = []
        if any_pre[j]:
            reasons.append("pre_go_live")
        if any_dead[j]:
            reasons.append("feed_dead")
        masked[col_names[j]] = {
            "bars": int(len(bars)),
            "first": int(lo + bars[0]),
            "last": int(lo + bars[-1]),
            "reasons": reasons,
        }
    return out, masked


# ── causally-tuned penalized arms ─────────────────────────────────────────────
# _batch_theta + RollingTunedLinear copied from specs/causal_tune_linear.py:201-379
# (verified rolling_linear_tune / winablate_r1 machinery; the spec module executes
# its backtest on import and cannot be imported). Adaptation: grid is an instance
# parameter; class-level traces dropped (the sidecar reports scalars only).


def _batch_theta(Xa, yr, locked, a, l1):
    """Cold seed: FWL on the locked (intercept) block + enet_coef on the rest."""
    exog = ~locked
    Hh, Ee = Xa[:, locked], Xa[:, exog]
    Hp = np.linalg.pinv(Hh)
    bHy, bHE = Hp @ yr, Hp @ Ee
    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
    bE = enet_coef(Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1)
    th = np.zeros(Xa.shape[1])
    th[locked] = bHy - bHE @ bE
    th[exog] = bE
    return th


class RollingTunedLinear:
    """Rolling linear model, ONE estimator kind: periodic causal alpha re-selection
    with WARM rank-1 per-bar updates between tunings. See the spec source cited
    above for the full derivation notes (identifiability mask included)."""

    def __init__(self, grid: list[tuple[str, float, float]]) -> None:
        self.grid = list(grid)
        # selection trajectory: (solve index at retune, alpha, l1_ratio,
        # n_active) — persisted by the caller as meta.tuned_penalty. n_active
        # is the number of NON-INTERCEPT coefficients left nonzero at the
        # selected penalty; 0 means the forecast fell back to the
        # intercept-only limit (see _tune).
        self.trace: list[tuple[int, float, float, int]] = []

    def init_window(self, X_win, y_win):
        X = np.asarray(X_win, dtype=np.float64)
        self._X = np.hstack([X, np.ones((X.shape[0], 1))])  # augmented intercept
        self._y = np.asarray(y_win, dtype=np.float64).copy()
        self._ptr = 0
        self._n_solve = 0
        self._locked = np.zeros(self._X.shape[1], dtype=bool)
        self._locked[-1] = True  # intercept: unpenalized, never exits the active set
        self._maskout = np.zeros(self._X.shape[1], dtype=bool)
        self._tune()
        return self

    def _window(self):
        k = self._ptr
        return (
            np.vstack([self._X[k:], self._X[:k]]),
            np.concatenate([self._y[k:], self._y[:k]]),
        )

    def _recompute_mask(self, Xraw):
        """Mask non-locked columns constant in the window or exact duplicates of an
        earlier kept column — threshold-free, scale-free (the lam2=0 singularity fix)."""
        ncol = Xraw.shape[1]
        maskout = np.zeros(ncol, dtype=bool)
        seen: dict = {}
        for j in range(ncol):
            if self._locked[j]:
                continue
            col = Xraw[:, j]
            if col.max() == col.min():
                maskout[j] = True
                continue
            key = col.tobytes()
            if key in seen:
                maskout[j] = True
            else:
                seen[key] = j
        self._maskout = maskout

    def _masked_window(self):
        Xa, yr = self._window()  # fresh arrays — safe to zero
        Xa[:, self._maskout] = 0.0
        return Xa, yr

    def _seed(self):
        """Cold seed (also the exact re-anchor) of the warm state on the current window."""
        Xa, yr = self._masked_window()
        tw = len(yr)
        kind, a, l1 = self.kind_, self.alpha_, self.l1_
        if kind == "ridge":
            D = np.where(self._locked, 0.0, a)  # sklearn Ridge units, free intercept
            G = Xa.T @ Xa
            G[np.diag_indices_from(G)] += D
            self._K = np.linalg.inv(G)
            self._c = Xa.T @ yr
            self._th = self._K @ self._c
        else:
            mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)  # sklearn ElasticNet units
            self._mu_vec = np.where(self._locked, 0.0, mu)
            Gr = Xa.T @ Xa
            ei = np.where(~self._locked)[0]
            Gr[ei, ei] += lam2
            self._Gr = Gr
            self._c = Xa.T @ yr
            th = _batch_theta(Xa, yr, self._locked, a, l1)
            self._A = list(np.where((np.abs(th) > 1e-9) | self._locked)[0])
            self._s = [
                0.0 if self._locked[j] else float(np.sign(th[j])) for j in self._A
            ]
            self._th = th

    def roll(self, x_in, y_in, x_out, y_out):
        ua = np.append(np.asarray(x_in, dtype=np.float64), 1.0)
        ur = np.append(np.asarray(x_out, dtype=np.float64), 1.0)
        self._X[self._ptr] = ua  # ring stores RAW rows; the mask re-derives from raw
        self._y[self._ptr] = float(y_in)
        self._ptr = (self._ptr + 1) % len(self._y)
        if self._maskout.any():  # masked columns stay zero in the update rows
            ua = ua.copy()
            ur = ur.copy()
            ua[self._maskout] = 0.0
            ur[self._maskout] = 0.0
        if self.kind_ == "ridge":  # Sherman-Morrison: add entering, drop leaving
            Ku = self._K @ ua
            self._K -= np.outer(Ku, Ku) / (1.0 + ua @ Ku)
            self._c += ua * float(y_in)
            Kv = self._K @ ur
            self._K += np.outer(Kv, Kv) / (1.0 - ur @ Kv)
            self._c -= ur * float(y_out)
        else:  # warm Garrigues homotopy: one rank-1 data change per call
            self._th, self._A, self._s = enet_online(
                self._Gr,
                self._c,
                self._mu_vec,
                ua,
                float(y_in),
                +1.0,
                self._th,
                self._A,
                self._s,
                self._locked,
            )
            self._th, self._A, self._s = enet_online(
                self._Gr,
                self._c,
                self._mu_vec,
                ur,
                float(y_out),
                -1.0,
                self._th,
                self._A,
                self._s,
                self._locked,
            )

    def _tune(self):
        Xraw, _ = self._window()
        self._recompute_mask(Xraw)
        Xa, yr = self._masked_window()
        n = len(yr)
        fit_lo, fit_hi, val_lo, val_hi = forward_window_split(n, n, VAL_TAIL, EMBARGO)
        Xf, yf = Xa[fit_lo:fit_hi], yr[fit_lo:fit_hi]
        Xv, yv = Xa[val_lo:val_hi], yr[val_lo:val_hi]
        best = None
        for kind, a, l1 in self.grid:
            if kind == "ridge":
                D = np.where(self._locked, 0.0, a)
                G = Xf.T @ Xf
                G[np.diag_indices_from(G)] += D
                th = np.linalg.solve(G, Xf.T @ yf)
            else:
                th = _batch_theta(Xf, yf, self._locked, a, l1)
            mse = float(np.mean((Xv @ th - yv) ** 2))
            if best is None or mse < best[0]:
                best = (mse, kind, a, l1)
        _, self.kind_, self.alpha_, self.l1_ = best
        self._seed()
        # ACTIVE-SET SIZE at the selected penalty (author directive 2026-08-07,
        # asked of the reach-matched enet grid). At a large alpha with
        # l1_ratio=1.0 the L1 penalty can exceed mu_max = max|X'y| and the
        # homotopy returns an EMPTY active set — a legitimate selection
        # outcome, not an error. It is safe here BY CONSTRUCTION rather than by
        # luck: the intercept is an augmented column carried in ``_locked``, so
        # it takes neither the L1 penalty (``_mu_vec`` is 0 there) nor the L2
        # ridge (added only at ``~_locked``), never leaves the active set, and
        # ``_batch_theta``'s FWL step then returns th[intercept] = mean(y).
        # The forecast therefore degenerates to INTERCEPT-ONLY, which is the
        # correct limit of the family, never to an undefined or all-zero
        # prediction. Recorded per retune so the meta can disclose how often
        # it happens instead of leaving it to be assumed.
        n_active = int(np.sum(np.abs(self._th[~self._locked]) > 0.0))
        self.trace.append(
            (int(self._n_solve), float(self.alpha_), float(self.l1_), n_active)
        )

    def solve(self):
        if self._n_solve > 0 and self._n_solve % TUNE_PER == 0:
            self._tune()
        elif self.kind_ == "ridge":
            self._th = self._K @ self._c
        self._n_solve += 1

    def predict_one(self, x):
        return float(np.append(np.asarray(x, dtype=np.float64), 1.0) @ self._th)


def _walk_tuned(
    F: np.ndarray,
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    grid: list[tuple[str, float, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Per-bar walk-forward driving the RollingTunedLinear incremental protocol
    (init_window / solve / predict_one / roll), chunk seam as in _walk_ridge.
    Returns (yhat, tuned_penalty trajectory: [{row, alpha, l1_ratio}] per
    retune boundary — the revealed shrink-vs-select preference)."""
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    solver = RollingTunedLinear(grid).init_window(Xs[:window], ys[:window])
    out = np.empty(hi - lo, dtype=np.float64)
    for i in range(hi - lo):
        t = window + i
        solver.solve()
        out[i] = solver.predict_one(Xs[t])
        solver.roll(Xs[t], ys[t], Xs[t - window], ys[t - window])
    traj = [
        {"row": int(lo + s), "alpha": a, "l1_ratio": l1, "n_active": k}
        for s, a, l1, k in solver.trace
    ]
    return out, traj


def _walk_tree(
    F: np.ndarray,
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    params: dict[str, Any],
    family: str,
) -> np.ndarray:
    """Per-bar-refit tree walk-forward (mixed-family expert bank, 2026-08-07).

    Dispatches on the menu entry's ``family`` tag: "lgbm" (LightGBM,
    num_threads) or "xgb" (XGBoost, tree_method='hist' via the frozen params,
    n_jobs). One full refit per bar on the trailing ``window`` rows, one
    prediction — NO tuning inside the arm (the config is a frozen menu entry;
    causal expert selection happens scorer-side from the persisted banks).
    Threads from $SLURM_CPUS_PER_TASK (1 outside SLURM; the sbatch exports
    OMP_NUM_THREADS to match); fixed seed for determinism at a given thread
    count. Tree libraries are imported here, not at module top, so the
    executor stays importable on envs without them.
    """
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if family == "lgbm":
        import lightgbm as lgb

        def _model():
            return lgb.LGBMRegressor(
                **params, num_threads=n_threads, random_state=42, verbosity=-1
            )
    elif family == "xgb":
        import xgboost as xgb

        def _model():
            return xgb.XGBRegressor(
                **params, n_jobs=n_threads, random_state=42, verbosity=0
            )
    else:
        raise SystemExit(f"unknown tree family '{family}' in the frozen menu")
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    out = np.empty(hi - lo, dtype=np.float64)
    for i in range(hi - lo):
        t = window + i
        model = _model()
        model.fit(Xs[i:t], ys[i:t])
        out[i] = float(model.predict(Xs[t : t + 1])[0])
    return out


# ── arm registry ──────────────────────────────────────────────────────────────


@dataclass
class ArmSpec:
    """One campaign arm: design builder + estimator config under the frozen spec."""

    describe: str
    kind: str  # "ols" | "blocks" | "tuned"
    blocks: list[tuple[str, str]] = field(default_factory=list)
    #   (block name, alpha key) for kind="blocks"; block names:
    #   backbone | exog_all | bucket:<name> | product | trans
    alphas: dict[str, float] = field(default_factory=dict)
    grid: str = ""  # ESTIMATOR_GRIDS key for kind="tuned"
    window_bars: int | None = None  # None = args.window; doc arms pin 12000
    oos_mult: int = 1  # first legal OOS row = oos_mult * window
    aliases: tuple[str, ...] = ()

    @property
    def window(self) -> int:
        """Training-window bars for this arm (runner interface): the pinned
        doc-convention window, else the campaign default 24000."""
        return int(self.window_bars or DEFAULT_WINDOW_BARS)


def _bucket_arm(bucket: str) -> ArmSpec:
    return ArmSpec(
        describe=f"min-norm LS: HAR backbone + '{bucket}' bucket "
        "(pinv; the alpha->0+ ridge limit)",
        kind="ols",
        blocks=[("backbone", ""), (f"bucket:{bucket}", "")],
    )


def _blk(
    arms: list[tuple[str, str]],
    alphas: dict[str, float],
    window: int | None,
    label: str,
) -> ArmSpec:
    needs_frozen = any(b == "product" or b.startswith("trans") for b, _ in arms)
    return ArmSpec(
        describe=label,
        kind="blocks",
        blocks=arms,
        alphas=alphas,
        window_bars=window,
        oos_mult=2 if needs_frozen else 1,
    )


# Canonical bucket enumeration = SUBGROUPS minus the empty `baseline` entry (that IS
# a0). `all_features` is a bucket row (the spec sheet's A9 joint arm) — 8 bucket arms.
_BUCKETS = [b for b in SUBGROUPS if b != "baseline"]

ARMS: dict[str, ArmSpec] = {
    "a0_ols_har": ArmSpec(
        describe="min-norm LS on the base-2 HAR ladder of the target (backbone: "
        "HAR + session-edge interactions + calendar; full-rank, so "
        "identical to plain OLS — routed through the min-norm path for "
        "uniformity). THE reference arm. "
        "Alias: a10_noexog — collapse VERIFIED computationally "
        "2026-08-06, not assumed: on the real 242,934-row panel the "
        "empty-bucket joint design == the plain backbone design "
        "bit-identically (np.array_equal True) and 300-bar walk-forward "
        "predictions match with max|diff| = 0.0; row sets identical by "
        "construction (one shared panel, no per-arm row filtering).",
        kind="ols",
        blocks=[("backbone", "")],
        aliases=("a10_noexog",),
    ),
    **{f"a_bucket_{b}": _bucket_arm(b) for b in _BUCKETS},
    # Merged-§4/§5 bucket grid (author directive 2026-08-07): the attribution
    # story rerun under competent causal estimators — per bucket design
    # (backbone + bucket cols, same matrices as a_bucket_*), a causally tuned
    # ridge and a causally tuned FREE-l1 elastic net. NOTE the directive's
    # count of 9 designs resolves to the canonical SUBGROUPS enumeration's 8
    # (7 single buckets + all_features = the joint design), as with A1..A8.
    # No legality exclusion: no frozen constructions involved.
    **{
        f"br_tuned_{b}": ArmSpec(
            describe=f"bucket grid: '{b}' design, causally TUNED ridge "
            "(battery grid logspace(-2,3,6), TUNE_PER=250)",
            kind="tuned",
            grid="ridge_tuned",
            blocks=[("backbone", ""), (f"bucket:{b}", "")],
        )
        for b in _BUCKETS
    },
    **{
        f"be_tuned_{b}": ArmSpec(
            describe=f"bucket grid: '{b}' design, causally TUNED elastic net "
            "with FREE l1_ratio (20-combo grid; l1=1 rows are lasso; "
            "trajectory in meta.tuned_penalty)",
            kind="tuned",
            grid="enet_free",
            blocks=[("backbone", ""), (f"bucket:{b}", "")],
        )
        for b in _BUCKETS
    },
    # REACH-MATCHED twin of the arm above (author directive 2026-08-07).
    # Identical in every respect — same design, free l1_ratio, warm
    # enet_online homotopy, identifiability mask, window, TUNE_PER=250,
    # 25-embargo/125-tail, per-bar refit, same (alpha, l1_ratio) persistence —
    # EXCEPT that the alpha axis extends one decade to 1e-1 so the family can
    # express the shrinkage the tuned ridge actually selects. See
    # ESTIMATOR_GRIDS["enet_free_wide"] for the lam2 arithmetic and for the
    # measured evidence that the enet's whole reported deficit sits at its own
    # grid ceiling. The vs-br_tuned pair is the CORRECTED head-to-head; the
    # vs-be_tuned pair measures how much of the original deficit was grid reach.
    **{
        f"be_tunedWide_{b}": ArmSpec(
            describe=f"bucket grid: '{b}' design, causally TUNED elastic net "
            "with FREE l1_ratio on the REACH-MATCHED alpha grid "
            "(logspace(-6,0,7) x 4 mixings = 28 combos; nests be_tuned's "
            "20-point choice set exactly)",
            kind="tuned",
            grid="enet_free_wide",
            blocks=[("backbone", ""), (f"bucket:{b}", "")],
        )
        for b in _BUCKETS
    },
    # Tree expert bank (2026-08-07): 20 frozen LightGBM configs from
    # experiments/tree_menu.json (LOUD failure at compute time if the menu is
    # absent). Design = the wide all_features basis, IDENTICAL to the tuned
    # linear arms (author correction); per-bar refit; no in-arm tuning.
    **{
        f"tree_expert_{k:02d}": ArmSpec(
            describe=f"tree expert #{k} from experiments/tree_menu.json "
            "(frozen mixed-family menu, lgbm|xgb per entry tag; per-bar "
            "refit on the wide linear design; causal selection scorer-side)",
            kind="tree",
            grid=str(k),
        )
        for k in range(N_TREE_EXPERTS)
    },
    "b1_ridge": ArmSpec(
        describe="FIXED-penalty ridge on the wide all_features basis, alpha=1.0 "
        "(DEFAULT_RIDGE_PARAMS — untuned production default; no tuning "
        "by user directive 2026-08-06)",
        kind="blocks",
        blocks=[("wide", "wide")],
        alphas={"wide": FIXED_RIDGE_ALPHA},
    ),
    # Fixed-ridge JIGGLE (user directive 2026-08-06): penalty-level sensitivity
    # of the head-to-head, log-symmetric around b1_ridge's alpha=1.0. Same wide
    # design, window, and persistence contract — only alpha moves.
    **{
        f"b1_ridge_a{tag}": ArmSpec(
            describe=f"fixed-ridge jiggle: wide all_features basis, alpha={a:g} "
            "(penalty-level sensitivity around b1_ridge alpha=1.0)",
            kind="blocks",
            blocks=[("wide", "wide")],
            alphas={"wide": a},
        )
        for tag, a in (("0p1", 0.1), ("0p3", 0.3), ("3", 3.0), ("10", 10.0))
    },
    # RIDGE GRID EXTENSION (author directive 2026-08-07). The jiggle grid above
    # is MONOTONE across its whole range — 0.1: 0.23360, 0.3: 0.23312,
    # 1: 0.23243, 3: 0.23169, 10: 0.23087 — i.e. still improving at its top
    # endpoint, so it never located its own optimum, and b1_ridge_tuned (whose
    # grid runs to 1e3) beats every point on it. Without these three the
    # FIXED-RIDGE ENVELOPE is unmeasured and the fixed-vs-tuned comparison is
    # not a comparison. Same wide design, same window, same persistence — only
    # alpha moves.
    **{
        f"b1_ridge_a{tag}": ArmSpec(
            describe=f"fixed-ridge grid extension: wide all_features basis, "
            f"alpha={a:g} (the jiggle grid was monotone to its endpoint; these "
            "bracket the fixed-ridge envelope)",
            kind="blocks",
            blocks=[("wide", "wide")],
            alphas={"wide": a},
        )
        for tag, a in (("30", 30.0), ("100", 100.0), ("300", 300.0))
    },
    "b2_lasso": ArmSpec(
        describe="FIXED-penalty lasso on the wide basis, alpha=1e-4 l1_ratio=1.0 "
        "(pinned 2026-08-06; warm Garrigues homotopy, per-bar refit; "
        "enet arm dropped: §5 is ridge vs lasso)",
        kind="tuned",
        grid="lasso_fixed",
    ),
    # LASSO JIGGLE arms (author directive 2026-08-07): the fixed-penalty
    # envelope of the lasso family, four points bracketing b2_lasso's pinned
    # alpha=1e-4 by two decades each way. See ESTIMATOR_GRIDS["lasso_fixed_a*"]
    # for why this is the experiment that makes 0.22950 interpretable. b2_lasso
    # itself is the 1e-4 rung and is NOT rebuilt.
    **{
        f"b2_lasso_a{tag}": ArmSpec(
            describe=f"fixed-lasso jiggle: wide all_features basis, alpha={a:g} "
            "l1_ratio=1.0 (fixed-penalty envelope around b2_lasso's pinned "
            "alpha=1e-4; same warm Garrigues homotopy, no selection)",
            kind="tuned",
            grid=f"lasso_fixed_a{tag}",
        )
        for tag, a in (
            ("1em6", 1e-6),
            ("1em5", 1e-5),
            ("1em3", 1e-3),
            ("1em2", 1e-2),
        )
    },
    "b1_ridge_tuned": ArmSpec(
        describe="causally-TUNED ridge control on the wide basis: battery grid "
        "logspace(-2,3,6), re-selected every 250 solves on the "
        "fit/embargo-25/val-tail-125 forward split of the current window",
        kind="tuned",
        grid="ridge_tuned",
    ),
    # GRID-RESOLUTION test, single-estimator rung (author directive
    # 2026-08-07): identical to b1_ridge_tuned except the alpha grid is
    # HALF-DECADE (11 points over the same 1e-2..1e3 range, the coarse 6 a
    # strict subset). See ESTIMATOR_GRIDS["ridge_tuned_fine"] for the arithmetic
    # that motivates it AND the selection-variance counter-consideration —
    # this arm is allowed to LOSE, and the interstitial-usage diagnostic makes
    # either outcome readable.
    "b1_ridge_tuned_fine": ArmSpec(
        describe="causally-tuned ridge at HALF-DECADE grid resolution: wide "
        "basis, logspace(-2,3,11), otherwise identical to b1_ridge_tuned "
        "(same 250-solve cadence, same embargo-25/tail-125 forward split)",
        kind="tuned",
        grid="ridge_tuned_fine",
    ),
    "b2_lasso_tuned": ArmSpec(
        describe="causally-TUNED lasso control on the wide basis: battery grid "
        "logspace(-6,-2,5) l1=1.0, 250-solve reselection cadence, warm "
        "Garrigues homotopy between reselections, identifiability mask",
        kind="tuned",
        grid="lasso_tuned",
    ),
    "b3_enet_tuned": ArmSpec(
        describe="causally-TUNED elastic net on the wide basis (the "
        "ridge/lasso-interpolating family): battery reclasticnet grid "
        "verbatim — logspace(-6,-2,5) at l1_ratio=0.5 "
        "(specs/causal_tune_linear.py:170), 250-solve reselection, warm "
        "enet_online homotopy, identifiability mask",
        kind="tuned",
        grid="enet_tuned",
    ),
    "blk2_user": _blk(
        [("backbone", "backbone"), ("exog_all", "exog")],
        USER_ALPHAS,
        None,
        "2-block ridge: backbone@1 + exog@100, args.window",
    ),
    "blk3_user": _blk(
        [("backbone", "backbone"), ("exog_all", "exog"), ("product", "product")],
        USER_ALPHAS,
        None,
        "3-block ridge: + frozen products@1000",
    ),
    "blk4_user": _blk(
        [
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_user", "trans"),
        ],
        USER_ALPHAS,
        None,
        "4-block ridge: + transmission [G|Ghat]@1000 (alpha + composition per "
        "the 2026-08-06 rulings)",
    ),
    "blk2_doc": _blk(
        [("backbone", "backbone"), ("exog_all", "exog")],
        DOC_ALPHAS,
        DOC_WINDOW_BARS,
        "2-block ridge, documented convention: backbone@1 + exog@3e3, 250-day window",
    ),
    "blk3_doc": _blk(
        [("backbone", "backbone"), ("exog_all", "exog"), ("product", "product")],
        DOC_ALPHAS,
        DOC_WINDOW_BARS,
        "3-block ridge, documented: + products@3e4, 250-day window",
    ),
    "blk4_doc": _blk(
        [
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_doc", "trans"),
        ],
        DOC_ALPHAS,
        DOC_WINDOW_BARS,
        "4-block ridge, documented: + transmission Ghat-only@3e3, 250-day window "
        "(documented construction verbatim)",
    ),
    # Isolation diagnostics run under BOTH block conventions (user directive
    # 2026-08-06): _user = stated alphas @ 24k window, _doc = documented §22
    # alphas @ 250-day window. Convention explicit in the arm name.
    "c4_product_alone_user": _blk(
        [("backbone", "backbone"), ("product", "product")],
        USER_ALPHAS,
        None,
        "diagnostic: backbone@1 + frozen products@1000, args.window",
    ),
    "c4_product_alone_doc": _blk(
        [("backbone", "backbone"), ("product", "product")],
        DOC_ALPHAS,
        DOC_WINDOW_BARS,
        "diagnostic: backbone@1 + frozen products@3e4, 250-day window",
    ),
    "d3_transmission_alone_user": _blk(
        [("backbone", "backbone"), ("trans_user", "trans")],
        USER_ALPHAS,
        None,
        "diagnostic: backbone@1 + transmission [G|Ghat]@1000, "
        "args.window (per the 2026-08-06 rulings)",
    ),
    "d3_transmission_alone_doc": _blk(
        [("backbone", "backbone"), ("trans_doc", "trans")],
        DOC_ALPHAS,
        DOC_WINDOW_BARS,
        "diagnostic: backbone@1 + transmission Ghat-only@3e3, 250-day window",
    ),
    # Transmission-REVIVAL arms (author directive 2026-08-06): the original
    # transmission promise is attributed to the full-sample-standardization
    # look-ahead; these arms provide that service causally via TRAILING
    # standardization + the standard rolling robust scaler (see
    # _transmission_block "trailing"). Same alphas/window as their _user twins.
    "d3_transmission_alone_trail": _blk(
        [("backbone", "backbone"), ("trans_trail", "trans")],
        USER_ALPHAS,
        None,
        "revival diagnostic: backbone@1 + TRAILING-standardized transmission "
        "[G|Ghat]@1000, args.window",
    ),
    "blk4_trail": _blk(
        [
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trail", "trans"),
        ],
        USER_ALPHAS,
        None,
        "revival 4-block ridge: 1/100/1000/1000 with TRAILING-standardized "
        "transmission",
    ),
    # Ablation triple (author directive 2026-08-07): decompose and maximize the
    # revived transmission block. G-only isolates well-scaled factor LEVELS,
    # Ghat-only isolates the lead-lag FLOW mechanism, and the tuned variant is
    # the best-model contender (per-block causal grids, transmission {1e2,1e3,1e4}).
    "blk4_trailG": _blk(
        [
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG", "trans"),
        ],
        USER_ALPHAS,
        None,
        "ablation 4-block ridge: trailing transmission, factor scores G ONLY "
        "(20 cols), 1/100/1000/1000",
    ),
    "blk4_trailGhat": _blk(
        [
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailGhat", "trans"),
        ],
        USER_ALPHAS,
        None,
        "ablation 4-block ridge: trailing transmission, lead-lag flow Ghat "
        "ONLY (20 cols), 1/100/1000/1000",
    ),
    "blk4_trail_tuned": ArmSpec(
        describe="best-model contender: [G|Ghat] trailing transmission with "
        "PER-BLOCK CAUSAL TUNING (BLOCK_TUNE_GRIDS, 81 combos per retune)",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trail", "trans"),
        ],
        oos_mult=2,
    ),
    # Ladder-expanded principal components (author correction 2026-08-07):
    # the ladder is applied TO THE EIGENVECTOR SERIES, giving every column a
    # (PC rank x horizon) identity and letting ONE rank-tilted penalty be
    # SHARED across a rank's rungs. See _pc_ladder_design for the commuting
    # identity that justifies the construction and for the deliberate
    # standardize-then-ladder ordering. Availability indicators are binary and
    # are NOT projected — they ride in their own scalar-penalty block.
    "blk_pcladder_tuned": ArmSpec(
        describe="ridge on ladder-expanded principal components: backbone + "
        "{ma_j(G_i)} at K=20 x 3 rungs = 60 cols with a per-rank tilted "
        "penalty shared across rungs + availability indicators + product; "
        "cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("pc_ladder", "pc_ladder_tilt"),
            ("avail_ind", "exog"),
            ("product", "product"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # K-SWEEP of the PC-ladder construction (author directive 2026-08-07).
    # THE STRUCTURAL POINT: moving averages commute with a linear projection,
    # so {ma_j(G_i)} spans the eigen-projection of the ladder-expanded base
    # design. At K=20 that projection THROWS AWAY most of the space and the arm
    # measures compression. At FULL RANK nothing is thrown away: the two
    # designs differ by a block-diagonal orthogonal rotation, a flat-penalty
    # ridge on either is identical, and the ONLY thing that can move the score
    # is the penalty PROFILE. So gamma=0 at full rank should reproduce the flat
    # result and any gain is attributable to the tilt alone.
    # CAVEAT — MEASURED, and it matters (experiments/verify_unification_shapes
    # .py section D). The exact-rotation identity holds for the FROZEN-frame
    # scores: at full rank, flat-penalty ridge fitted values on the ladder-of-
    # PCs and on the ladder-expanded standardized base agree to 1.6e-14
    # RELATIVE. But the arm's TRAILING standardization of the scores is a
    # time-varying PER-COLUMN rescaling and is NOT orthogonal, and with it the
    # same comparison differs by 0.80 relative — not a rounding effect, a
    # different estimator. Reason: dividing score i by sqrt(d_i) turns a flat
    # penalty into an eigenvalue-weighted one (see the gamma_eff note at
    # TRANS_SHAPE_GAMMAS_WIDE). So "gamma=0 at full rank reproduces the flat
    # exog result" is TRUE of the rotation and FALSE of the arm as built; what
    # the arm tests is the tilt ON TOP OF the standardization's implicit
    # NEGATIVE tilt. Read the K-sweep as a reparameterization of the SPAN, not
    # of the estimator.
    # All three arms are identical to blk_pcladder_tuned except K and the WIDE
    # exponent grid (steeper tilts are exactly what the tail directions of a
    # full-rank frame may need).
    **{
        name: ArmSpec(
            describe=(
                "ladder-expanded principal components at "
                f"{k_desc} x {len(PRODUCT_EXOG_WINDOWS)} rungs with a per-rank "
                "tilted penalty (WIDE exponent grid) shared across rungs + "
                "backbone + availability indicators + product; NO raw exog "
                "block; cyclic causal tuning"
            ),
            kind="blocks_tuned",
            blocks=[
                ("backbone", "backbone"),
                (blk, "pc_ladder_tilt_wide"),
                ("avail_ind", "exog"),
                ("product", "product"),
            ],
            grid="cyclic",
            oos_mult=2,
        )
        for name, blk, k_desc in (
            ("blk_pcladder_fortyK_tuned", "pc_ladder40", "K=40"),
            ("blk_pcladder_eightyK_tuned", "pc_ladder80", "K=80"),
            (
                "blk_pcladder_fullK_tuned",
                "pc_ladder_full",
                "K=FULL live frame rank",
            ),
        )
    },
    # PER-RUNG EIGENBASES (author directive 2026-08-07). Everything above uses
    # ONE eigenbasis — that of the raw base features — at every horizon. But
    # ma_j is a linear smoother, so ma_j(X) has its own correlation structure
    # and the leading cross-sectional directions of the fast and slow panels
    # need not coincide. Here each rung is SMOOTHED FIRST and rotated into its
    # OWN frozen frame second (see _pc_ladder_perrung_design). The penalty
    # parameterization is deliberately UNCHANGED from blk_pcladder_tuned (same
    # pcrank family, same 12-point grid, one (lambda0, gamma) shared across
    # rungs) so the BASIS is the only difference and the head-to-head is clean.
    # The build log prints the fast-vs-slow subspace alignment: near-identical
    # bases predict a tie, and the number is then the explanation.
    "blk_pcladderPerRung_tuned": ArmSpec(
        describe="per-RUNG eigenbases: each ladder horizon is smoothed first "
        "and rotated into its OWN frozen first-window frame (K=20 x 3 rungs), "
        "with the rank-tilted penalty shared across rungs; + backbone + "
        "availability indicators + product, NO raw exog block; cyclic causal "
        "tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("pc_ladder_perrung", "pc_ladder_tilt"),
            ("avail_ind", "exog"),
            ("product", "product"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # Hard-vs-soft tilt as a WITHIN-ARM selection (author directive
    # 2026-08-07): identical to blk3_tikhonov_tuned except the exogenous tilt
    # grid is the UNION of the power family (smooth) and the step family
    # (PCR's hard threshold, finite-M stand-in) — 3 alphas x (4 power + 3
    # step) = 21 points on one block. THE EXHIBIT is which SHAPE FAMILY the
    # causal tuner picks per retune and whether it is stable across eras:
    # consistent step at some K vindicates truncation (PCR) on the
    # estimator's own terms; consistent power says smooth tilting wins; a
    # flip era to era is a finding about regime-dependent effective
    # dimension. power/gamma=0 remains the exact scalar-ridge nesting.
    "blk3_tikhonovStep_tuned": ArmSpec(
        describe="generalized Tikhonov with BOTH tilt families: exogenous "
        "penalty selected causally from power (alpha*i**gamma) UNION step "
        "(alpha, then alpha*1e4 beyond rank K in {20,40,80}) — 21-point "
        "block grid, cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_rot", "exog_tilt_step"),
            ("product", "product"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # Genuine PCR (author directive 2026-08-07): the principal components
    # REPLACE the wide exogenous design instead of augmenting it. Two blocks
    # only — HAR ladder + calendar, plus the trailing-standardized frozen-frame
    # factor scores. No exog_all, no product, no operator columns.
    # SCIENTIFIC QUESTION: can a 20-column PCA summary REPLACE the 526-column
    # exogenous panel? Reference points on the same rows: blk3_user
    # (HAR+exog+product) OOS R^2 3.12% vs benchmark, HAR+transmission-alone
    # 2.57%, best model blk4_trailG_tuned QLIKE 0.21909. The K=40 twin asks a
    # second question: does the K=40 collapse reproduce when the factor block
    # is NOT sharing a penalty with a wide block alongside it? If K=40 does
    # fine here but collapsed there, the collapse was contamination, not noise.
    # Penalty grid: BLOCK_TUNE_GRIDS["pcr"], deliberately wide — see there.
    "blk2_pcr_tuned": ArmSpec(
        describe="genuine PCR: HAR backbone + K=20 trailing-standardized "
        "frozen-frame factor scores, NOTHING else (no exog, no product, no "
        "Ghat); per-block causal tuning over 3 x 7 = 21 combos",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("trans_trailG", "pcr")],
        oos_mult=2,
    ),
    "blk2_pcrForty_tuned": ArmSpec(
        describe="genuine PCR at K=40: HAR backbone + 40 trailing-standardized "
        "factor scores, NOTHING else — the K question without a wide block "
        "sharing the penalty",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("trans_trailG40", "pcr")],
        oos_mult=2,
    ),
    # Generalized-Tikhonov arm (author directive 2026-08-07): the clean
    # formulation of what the transmission block achieves by AUGMENTATION.
    # Diagnosis: the transmission columns lie in the span of the existing
    # design, so they add no information — they help only because their own
    # weaker block penalty reduces effective shrinkage along those
    # directions, i.e. anisotropic ridge implemented by duplication. Here the
    # anisotropy is applied DIRECTLY: three blocks, no duplicated columns,
    # exogenous penalty Gamma = V diag(alpha * i**gamma) V' realized exactly
    # in rotated coordinates (see _exog_tilt_design). gamma=0 == blk3_tuned.
    "blk3_tikhonov_tuned": ArmSpec(
        describe="generalized Tikhonov: 3 blocks, NO transmission columns, "
        "exogenous penalty spectrum-tilted as alpha*i**gamma in the frozen "
        "eigenframe (12-point (alpha, gamma) grid), cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_rot", "exog_tilt"),
            ("product", "product"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # Spectral analogue of the per-bucket rung (author directive 2026-08-07):
    # is the K=40 collapse a penalty-allocation artifact rather than a signal
    # boundary? Levels-only (per the parsimony finding), frozen frame at K=40,
    # transmission penalty RANK-SHAPED as lambda_i = lambda0 * i**gamma with
    # (lambda0, gamma) causally selected as ONE 12-point block grid. gamma=0
    # nests the flat penalty exactly, so the comparison is honest.
    "blk4_trailGShaped": ArmSpec(
        describe="spectral rung: trailing factor LEVELS at K=40 with a "
        "RANK-SHAPED transmission penalty (lambda_i = lambda0 * i**gamma, "
        "12-point (lambda0, gamma) grid), cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG40", "trans_shaped"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # ENDPOINT-RELIEF twin of the arm above (author directive 2026-08-07).
    # blk4_trailGShaped's tuner selects gamma=2 — the TOP of its exponent grid
    # — in ~45% of 1092 retunes, which by this campaign's own >20% rule is an
    # invalid selection: the grid boundary, not the validation tail, is setting
    # the tilt. This arm is BYTE-FOR-BYTE identical except that the exponent
    # axis runs to 4 (TRANS_SHAPE_GAMMAS_WIDE, 18 (lambda0, gamma) points, a
    # STRICT superset of the original 12), so the increment against
    # blk4_trailGShaped isolates exactly one thing: whether a steeper-than-
    # quadratic rank tilt buys anything. If the wide tuner still piles up at
    # the new top, the tilt is a proxy for truncation and the step family (see
    # BLOCK_TUNE_GRIDS["exog_tilt_step"]) is the honest next rung.
    "blk4_trailGShapedWide": ArmSpec(
        describe="endpoint relief for the spectral rung: trailing factor "
        "LEVELS at K=40 with a RANK-SHAPED transmission penalty over the "
        "EXTENDED exponent grid (lambda_i = lambda0 * i**gamma, gamma up to "
        "4, 18-point (lambda0, gamma) grid), cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG40", "trans_shaped_wide"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # TRAILING-STANDARDIZATION DECOMPOSITION (author directive 2026-08-09):
    # the parent's causal z-score has two moving parts — the trailing mean and
    # the trailing sd. These twins keep the frame, truncation, production
    # scaler, block grids, and tuner byte-identical to blk4_trailGShapedWide,
    # changing only which half of the z-score is active. Read against the
    # parent and the frozen twin: mean-vs-parent isolates the moving divisor;
    # sd-vs-parent isolates the moving location.
    "blk4_trailGShapedTrailMean": ArmSpec(
        describe="trailing-standardization decomposition: K=40 factor LEVELS, "
        "causal trailing DEMEANING ONLY (no trailing-sd division), same "
        "rank-shaped wide transmission grid, cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG40_trailmean", "trans_shaped_wide"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    "blk4_trailGShapedTrailSd": ArmSpec(
        describe="trailing-standardization decomposition: K=40 factor LEVELS, "
        "division by the causal trailing SD ONLY (no trailing demeaning), "
        "same rank-shaped wide transmission grid, cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG40_trailsd", "trans_shaped_wide"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # GRID-RESOLUTION test on the BEST MODEL (author directive 2026-08-07):
    # identical to blk4_trailGShaped except EVERY block grid is refined to
    # half-decade spacing over the SAME range it already spans, so the arm
    # differs in RESOLUTION ONLY and each coarse grid is a strict subset. The
    # paper's best model is currently tuned on 3-point-per-block grids while
    # the fixed-penalty envelope moves ~0.0009/decade — i.e. coarser than the
    # effects reported as significant. Cost: 94 tail evaluations per retune vs
    # the shipped arm's 52. Allowed to lose (see the counter-consideration at
    # ESTIMATOR_GRIDS["ridge_tuned_fine"]).
    "blk4_trailGShaped_fine": ArmSpec(
        describe="grid-resolution test on the best model: blk4_trailGShaped "
        "with HALF-DECADE per-block grids (5 points each; the shaped "
        "transmission block's lambda0 refined to 5, gammas unchanged, 20 "
        "points), cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone_fine"),
            ("exog_all", "exog_fine"),
            ("product", "product_fine"),
            ("trans_trailG40", "trans_shaped_fine"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # GRID-FREE SHRINKAGE (author directive 2026-08-07). Three estimators with
    # ZERO tuned hyperparameters, on blk3_tuned's exact design, so the bare-stem
    # increment against blk3_tuned asks the campaign's sharpest methodological
    # question: does shrinkage ESTIMATED from the training window beat shrinkage
    # SELECTED on a 125-bar tail? See the block comment above _causal_ols for
    # the principle, the single-prior negative result these deliberately avoid,
    # and why a positive answer makes the paper's conclusion constructive.
    # No grid => no tuning machinery at all; meta.tuned_alphas / tuned_grids
    # stay EMPTY for these arms by construction (the shape summary skips arms
    # with no descriptors, so nothing downstream sees a malformed entry), and
    # the exhibit rides in meta.shrink_profile instead.
    "blk3_js_tuned": ArmSpec(
        describe="grid-free shrinkage: three-block design with POSITIVE-PART "
        "JAMES-STEIN shrinkage of the exog+product coefficients (exact block "
        "covariance via the Schur complement, backbone unshrunk), NO tuned "
        "hyperparameters",
        kind="shrink",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
        ],
        grid="js",
        oos_mult=2,
    ),
    "blk3_npeb_tuned": ArmSpec(
        describe="grid-free shrinkage: three-block design with NONPARAMETRIC "
        "EMPIRICAL BAYES per-coefficient shrinkage (Tweedie's formula on a "
        "Gaussian-KDE marginal of the standardized coefficients, backbone "
        "unshrunk), NO tuned hyperparameters",
        kind="shrink",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
        ],
        grid="npeb",
        oos_mult=2,
    ),
    # PC-BASIS twin. IMPORTANT ALGEBRAIC FINDING, recorded here because it
    # changes what this arm can possibly measure: the EXACT James-Stein factor
    # is INVARIANT under any orthogonal rotation of the shrunk block. It
    # depends on the data only through beta_S' C^-1 beta_S, and under
    # beta -> A beta, C -> A C A' with A orthogonal that quadratic form is
    # unchanged. So "blk3_js in the PC basis" with the exact covariance would
    # be a BIT-IDENTICAL DUPLICATE of blk3_js_tuned — 100 chunks of cluster
    # time for a provable no-op. (The synthetic asserts the invariance rather
    # than asserting a difference that cannot exist.)
    # What DOES depend on the basis is the DECOUPLING APPROXIMATION: whitening
    # by the diagonal of the block covariance alone is exact only when the gram
    # is diagonal, and rotating into the frozen frame is precisely what makes
    # it nearly so. So this arm is diagonal-whitened JS in the PC basis, and
    # blk3_jsDiag_tuned below is its raw-basis twin. The pair isolates the
    # basis; each against blk3_js_tuned measures what the decoupling costs.
    "blk3_js_pcbasis_tuned": ArmSpec(
        describe="grid-free shrinkage in the PC basis: exogenous block rotated "
        "into the frozen base-feature eigenbasis, then DIAGONAL-whitened "
        "positive-part James-Stein (the decoupling is closest to exact there), "
        "NO tuned hyperparameters",
        kind="shrink",
        blocks=[
            ("backbone", "backbone"),
            ("exog_rot", "exog"),
            ("product", "product"),
        ],
        grid="js_diag",
        oos_mult=2,
    ),
    "blk3_jsDiag_tuned": ArmSpec(
        describe="raw-basis control for the PC-basis shrinkage arm: identical "
        "diagonal-whitened positive-part James-Stein WITHOUT the rotation, so "
        "the pair isolates the basis; NO tuned hyperparameters",
        kind="shrink",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
        ],
        grid="js_diag",
        oos_mult=2,
    ),
    # ── PROPER PCR (author-specified design, 2026-08-07) ──────────────────────
    # Supersedes the blk3_rotK_tuned sketch. The earlier PC families could
    # never answer the replacement question: blk_pcladder_* standardized each
    # SCORE (a non-orthogonal, time-varying rescale that broke the equivalence
    # and capped the family ~1.6pp below the panel at every K), and blk2_pcr_*
    # dropped the product block AND the indicators, so they differed from the
    # comparator in more than the representation. This family fixes both.
    #
    # SHARED DESIGN, only ORDERING and K vary:
    #   backbone            UNROTATED, UNSHRUNK — the benchmark's own basis,
    #                       not something a rotation is entitled to mix into.
    #   rotated VALUE       one frozen V from the frame window applied to every
    #                       MA window's slab (12 x 43 tensor, verified uniform
    #                       against the real name list; see _value_slab_cols).
    #                       INPUTS standardized, SCORES NEVER.
    #   availability        UNROTATED, own block — a linear combination of
    #                       binary indicators indicates nothing — but SHRUNK
    #                       together with the rotated block as "the exogenous
    #                       set".
    #   NO PRODUCT BLOCK    the comparison is against the two-block model.
    #
    # ESTIMATOR: the exact positive-part James-Stein of blk3_js_tuned, no tuned
    # penalty anywhere. THIS CHOICE IS LOAD-BEARING: the exact JS factor is
    # INVARIANT under an orthogonal rotation of the shrunk block (proved and
    # asserted in the verify script), so at FULL RANK the rotation is provably
    # a no-op and the ONLY thing this family varies is what the truncation
    # DISCARDS. That is what makes it a clean isolation of truncation and
    # ordering rather than a confound of representation with penalty.
    #
    # K LADDER, corrected against the panel: the rotatable set is 12 x 43, not
    # 12 x 92 (the 48 indicator prefixes exceed the 43 value stems because five
    # quantities carry both _avail_ and _active_). Maximum K is 43, so the
    # requested 92/80/60 do not exist; the ladder is FULL/40/30/20 for the
    # variance family and 30/20/10/5 for the predictive one, which also gives
    # matched-K pairs at BOTH 30 and 20.
    **{
        f"blk2_rotVar{tag}_js": ArmSpec(
            describe=(
                "proper PCR, VARIANCE ordering, "
                f"K={'full live rank' if kt == 'full' else kt} directions x 12 "
                "MA windows: backbone (unrotated, unshrunk) + frozen-frame "
                "rotated VALUE tensor + unrotated availability indicators; "
                "inputs standardized, scores never; exact positive-part "
                "James-Stein on the exogenous set, NO tuned penalty"
            ),
            kind="shrink",
            blocks=[
                ("backbone", "backbone"),
                (f"rotval:variance:{kt}", "exog"),
                ("avail_ind", "exog"),
            ],
            grid="js",
            oos_mult=2,
        )
        for tag, kt in (
            ("FullK", "full"),
            ("FortyK", "40"),
            ("ThirtyK", "30"),
            ("TwentyK", "20"),
        )
    },
    **{
        f"blk2_rotPred{tag}_js": ArmSpec(
            describe=(
                f"proper PCR, PREDICTIVE ordering, K={kt} directions x 12 MA "
                "windows: directions ranked by frame-window-only block R^2 of "
                "the target on each direction's own 12 ladder columns, frozen "
                "exactly as V is; otherwise identical to the variance family"
            ),
            kind="shrink",
            blocks=[
                ("backbone", "backbone"),
                (f"rotval:predictive:{kt}", "exog"),
                ("avail_ind", "exog"),
            ],
            grid="js",
            oos_mult=2,
        )
        for tag, kt in (
            ("ThirtyK", "30"),
            ("TwentyK", "20"),
            ("TenK", "10"),
            ("FiveK", "5"),
        )
    },
    # FRAME-GATED comparator (author directive 2026-08-07). blk2_tuned runs at
    # oos_mult=1 and is therefore scored on 273,554 rows, while every arm with
    # a frozen construction is scored on 248,686. Comparing their QLIKE is a
    # row-set confound that has already bitten this campaign once. This arm is
    # blk2_tuned's design at oos_mult=2 so it lands on the SAME rows as the
    # PCR family. DELIBERATELY NOT grid-free: it is the incumbent two-block
    # ridge, tuned on the ordinary per-block grids, and it is what the PCR arms
    # must beat — putting it on the grid-free estimator would change what is
    # being compared.
    "blk2_gated_tuned": ArmSpec(
        describe="frame-gated two-block ridge: backbone + full UNROTATED "
        "exogenous set, per-block causal tuning on the ordinary grids, "
        "oos_mult=2 so it is scored on the same rows as the proper-PCR family",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("exog_all", "exog")],
        oos_mult=2,
    ),
    # TUNED-RIDGE twins of the proper-PCR ladder (author directive 2026-08-08).
    # WHY THE RERUN: the _js ladder came back CONFOUNDED. James-Stein
    # under-shrinks on this design (factor ~0.77-0.83 everywhere, consistent
    # with blk3_js_tuned's 0.24631 verdict), so its K-curve is a
    # REGULARIZATION curve, not an information curve — FullK (which discards
    # nothing) was the family's WORST arm at 0.2407 and performance improved
    # monotonically as K fell, in BOTH orderings, with every arm losing to
    # blk2_gated_tuned. Under-shrinkage makes a wide design look bad and a
    # narrow one look good regardless of what the discarded directions carry.
    # The one clean readout was ordering at matched K: predictive beat variance
    # at K=30 (-0.00107, DM -2.94) and was null at K=20 (+1.63).
    # WHAT THESE SETTLE: with shrinkage restored to the level the design
    # actually demands, the K-curve becomes interpretable. If truncation still
    # improves or holds under proper tuning, compression is REAL; if the curve
    # flattens toward blk2_gated_tuned as K grows, the panel needs its full
    # span and the earlier "compression" was regularization in disguise.
    # Designs are BIT-IDENTICAL to the _js twins (same frozen V on the ma_1
    # slab, same per-slab input standardization, scores never standardized,
    # same frozen predictive ordering, indicators unrotated); the ONLY change
    # is the estimator: two blocks, backbone on its usual grid and the whole
    # exogenous set on rotexog_wide.
    **{
        f"blk2_rot{fam}{tag}_tuned": ArmSpec(
            describe=(
                f"proper PCR under TUNED RIDGE, {fam_label} ordering, "
                f"K={'full live rank' if kt == 'full' else kt} directions x 12 "
                "MA windows: backbone + (rotated VALUE tensor | unrotated "
                "availability indicators) as one penalized block on the wide "
                "0.1..1e4 grid; the estimator-corrected twin of the _js ladder"
            ),
            kind="blocks_tuned",
            blocks=[
                ("backbone", "backbone"),
                (f"rotexog:{ordering}:{kt}", "rotexog_wide"),
            ],
            oos_mult=2,
        )
        for fam, fam_label, ordering, tag, kt in (
            ("Var", "VARIANCE", "variance", "FullK", "full"),
            ("Var", "VARIANCE", "variance", "ThirtyK", "30"),
            ("Var", "VARIANCE", "variance", "TwentyK", "20"),
            ("Pred", "PREDICTIVE", "predictive", "ThirtyK", "30"),
            ("Pred", "PREDICTIVE", "predictive", "TwentyK", "20"),
            ("Pred", "PREDICTIVE", "predictive", "TenK", "10"),
            ("Pred", "PREDICTIVE", "predictive", "FiveK", "5"),
        )
    },
    # DISCOUNTED-GRAM family (author directive 2026-08-08). Same blk3 design
    # and same per-block penalty grids as blk3_tuned; the ONLY change is that
    # the sufficient statistics are exponentially weighted. See the block
    # comment at EW_HALF_LIVES for the JS-verdict motivation, the truncation
    # decision, and why H=infinity nests the flat window exactly.
    # THE EXHIBIT is meta.tuned_alphas' half_life trajectory: a finite H chosen
    # WITH lighter penalties than the flat arm selects is direct evidence that
    # the campaign has been paying for drift with shrinkage.
    "blk3_ew_tuned": ArmSpec(
        describe="discounted gram: blk3 design with EXPONENTIALLY WEIGHTED "
        "sufficient statistics, half-life selected causally from "
        "{1000,3000,6000,12000,24000,inf} bars JOINTLY with the per-block "
        "penalties (inf = the flat 24000-bar window exactly)",
        kind="blocks_ew",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
        ],
        grid="ew_grid",
        oos_mult=2,
    ),
    # FIXED-H twins: they bound the selection-noise question. The H grid adds a
    # dial the 125-bar validation tail must resolve, and this campaign has
    # repeatedly found that tail to be a noisy selector; these show what a
    # half-life is worth WITHOUT paying for its selection.
    **{
        f"blk3_ewFixed_H{int(hl)}": ArmSpec(
            describe=f"discounted gram at FIXED half-life {int(hl)} bars "
            "(blk3 design, per-block penalties tuned as usual) — the "
            "selection-free counterpart of blk3_ew_tuned",
            kind="blocks_ew",
            blocks=[
                ("backbone", "backbone"),
                ("exog_all", "exog"),
                ("product", "product"),
            ],
            grid=f"ew_fixed:{int(hl)}",
            oos_mult=2,
        )
        for hl in (3000.0, 12000.0)
    },
    # CAN THE TRAILING-STANDARDIZED PCA LEVELS REPLICATE THE EXOGENOUS BLOCK
    # BY THEMSELVES? (author directive 2026-08-08, product block removed on
    # correction — the comparison must be pure, two blocks each side with
    # nothing shared but the backbone.)
    #
    # WHAT IS ALREADY KNOWN, from arms on disk at matched rows (248,686) and
    # matched tuning:
    #     blk2_gated_tuned            backbone + FULL exog      0.21997
    #     blk2_pcrForty_tuned         backbone + K=40 levels    0.22134
    #     d3_transmission_alone_tuned backbone + K=20 levels    0.22249
    #     blk2_pcr_tuned              backbone + K=20 levels    0.22287
    # So the K=40 case IS ALREADY RUN — blk2_pcrForty_tuned is exactly
    # "backbone + trans_trailG40, nothing else, tuned", and it LOSES to the
    # exogenous block by 1.37e-3. Widening K from 20 to 40 closed roughly 45%
    # of the K=20 gap (2.5e-3 -> 1.37e-3) but did not close it. A separate
    # K=40 arm on a NARROWER penalty grid would be a strict re-run of that
    # result and could only do worse, so it is deliberately not built.
    #
    # WHAT IS GENUINELY UNTESTED is the FULL SPECTRUM: if the residual 1.37e-3
    # is truncation, taking every live direction should close it; if it is
    # content the construction cannot see, it will not.
    #
    # GRID CHOICE, and why it is `pcr` rather than a fresh one: the K=40 arm's
    # penalty selections are already on disk under `pcr`, and the K=40-vs-full
    # comparison is only clean if both arms search the SAME grid — otherwise
    # spectrum width is confounded with grid reach. `pcr` also spans 1e-2..1e4,
    # and blk2_pcrForty_tuned selects the 1e-2 end in 32% of retunes, so a grid
    # starting at 1e0 would amputate the region a third of its selections
    # actually use.
    #
    # INTERPRETATION, fixed in advance:
    #   full ~= blk2_gated_tuned -> the levels DO replicate the exogenous block
    #                               once the spectrum is not truncated;
    #                               truncation at 40 was the binding limit.
    #   full still < gated       -> the exogenous block carries content
    #                               structurally OUTSIDE this construction.
    #                               Candidates in order of prior: the nine
    #                               ladder windows the frame's base omits (it
    #                               takes value columns at only 1/32/512 of
    #                               twelve), the 576 availability indicators
    #                               (absent from the base entirely), and
    #                               per-column idiosyncratic content a
    #                               correlation frame cannot represent.
    "blk2_gfull_tuned": ArmSpec(
        describe="capture test at the FULL frame spectrum: HAR backbone + "
        "every live trailing-standardized frozen-frame factor level (K read "
        "from the frame's liveness rule at build time), NOTHING else — the "
        "spectrum-width twin of blk2_pcrForty_tuned on the same penalty grid",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("trans_trailGFull", "pcr")],
        oos_mult=2,
    ),
    # CORRECTED ROTATION BASE (author directive 2026-08-08). The PCR-tuned
    # ladder's K-curve came back FLAT under working shrinkage (FullK 0.22215 /
    # K30 0.22221 / K20 0.22182 / K5 0.22270) with ordering null at matched K,
    # confirming the JS family's monotonicity was a regularization artifact.
    # But the whole family lost to blk2_gated_tuned by ~2.2e-3 INCLUDING AT
    # FULL RANK (+2.18e-3, DM +7.28), which is impossible if the only
    # difference is an orthogonal rotation — uniform ridge cannot see one.
    # The residual is the INPUT STANDARDIZATION: the rotation family rebuilds
    # its slabs from frozen frame-window statistics, the production path uses
    # the panel's rolling scaler. This arm removes that difference and nothing
    # else, so it is the corrected base every capture-ladder rung should be
    # rebuilt on once the tie is confirmed.
    #
    # THE GATE is not a tolerance, it is an identity: at full rank this design
    # is an ORTHOGONAL reparameterization of blk2_gated_tuned's exogenous
    # block under the single uniform penalty both carry, so the two arms must
    # agree to machine precision. That is asserted offline in the verify script
    # rather than inferred from a canary; the canary then only has to confirm
    # the production path reproduces it (expected |dQLIKE| at float noise,
    # ~1e-12, against the +2.18e-3 it is explaining — three orders of magnitude
    # of headroom, so the test cannot be passed by accident).
    "blk2_rotFullProd_timeaware_tuned": ArmSpec(
        describe="time-aware post-PCA penalization: production-scaled full-rank "
        "rotation with per-PC penalty scaling by effective MA-window horizon",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_timeaware", "exog_timeaware")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_tuned": ArmSpec(
        describe="corrected rotation base: HAR backbone + full-rank rotation "
        "of the PRODUCTION rolling-scaled exogenous VALUE columns, indicators "
        "unrotated, as one penalized block — isolates frozen-vs-rolling input "
        "standardization from the rotation itself",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_full", "exog")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_wobbleEps1em3_tuned": ArmSpec(
        describe="full-rank production rotation + fixed causal orthogonal wobble "
        "(eps=1e-3); full-rank span preserved, uniform exog penalty",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_wobble", "exog")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_wobbleEps1em2_tuned": ArmSpec(
        describe="full-rank production rotation + fixed causal orthogonal wobble "
        "(eps=1e-2); full-rank span preserved, uniform exog penalty",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_wobble", "exog")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_wobbleEps3em2_tuned": ArmSpec(
        describe="full-rank production rotation + fixed causal orthogonal wobble "
        "(eps=3e-2); full-rank span preserved, uniform exog penalty",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_wobble", "exog")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_wobbleEps1em1_tuned": ArmSpec(
        describe="full-rank production rotation + fixed causal orthogonal wobble "
        "(eps=1e-1); full-rank span preserved, uniform exog penalty",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_wobble", "exog")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_timeaware_wobbleEps1em2_tuned": ArmSpec(
        describe="time-aware post-PCA penalties with the eigenframe wobbled "
        "(eps=1e-2) BEFORE the horizon profile is computed",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rot_prod_timeaware_wobble", "exog_timeaware")],
        oos_mult=2,
    ),
    "blk2_rotFullProd_wobbleBagEps1em2_tuned": ArmSpec(
        describe="bagged causal ellipsoid wobble (8 seeded members, eps=1e-2) "
        "under the uniform exog block penalty",
        kind="blocks_tuned_bag",
        blocks=[("backbone", "backbone"), ("rot_prod_wobble", "exog")],
        oos_mult=2,
    ),
    # CUMULATIVE REPAIR LADDER (author directive 2026-08-08). The goal is no
    # longer to test whether the levels capture the exogenous block but to MAKE
    # them, discarding one deficit at a time so the increments DECOMPOSE the
    # gap by cause. Replication is guaranteed at the limit: full rank + all
    # rungs + indicators + NO score standardization is an exact orthogonal
    # reparameterization of the exogenous design (proven end-to-end at
    # 3.99e-15 in the verify script's GATE), so every rung of this ladder is a
    # step along a path whose destination is known.
    #
    # THE DEFICITS, and which arm discards each:
    #   0 baseline   blk2_pcrForty_tuned (on disk, 0.22134) is backbone + K=40
    #                levels, nothing else, and loses to blk2_gated_tuned
    #                (0.21997) by 1.37e-3. blk2_gfull_tuned takes K to the full
    #                live spectrum, isolating pure truncation.
    #   1 RUNGS      the frame's base takes value columns at only 1/32/512 of
    #                the exogenous block's TWELVE windows, so nine rungs of
    #                information are simply absent. blk2_gFortyRungs_tuned
    #                expands each retained direction through all twelve via
    #                ma_j(V'x) = V' ma_j(x) — the commuting identity, which is
    #                what makes the ladder-of-scores the eigen-projection of
    #                the full ladder rather than an approximation of it.
    #   2 INDICATORS the 576 availability columns are absent from the frame's
    #                base entirely, and a correlation frame over continuous
    #                values cannot represent them. They enter UNROTATED as
    #                their own block (a linear combination of binary indicators
    #                indicates nothing).
    #   3 SUPERVISION the PCA frame maximizes variance among the regressors and
    #                never looks at the target. The PLS arms replace it with a
    #                frame that maximizes covariance WITH the target, at HALF
    #                and a QUARTER of the directions.
    #   4 REMAINDER  whatever gap blk2_plsTwentyRungsInd_tuned still leaves
    #                against blk2_gated_tuned is the measured IRREDUCIBLE part:
    #                per-column idiosyncratic content no low-rank frame over
    #                these base quantities can carry. That number is the
    #                deliverable, not a failure.
    #
    # Column counts on the real panel (backbone 52): K=33 live directions
    # (audited 2026-08-09; 10 of 43 base quantities pre-go-live dead in the
    # frame window) x 12 rungs = 396; plus 576 indicators = 972.
    # K RE-AUDIT NOTE (2026-08-09): the rotval arms requested K=40 but the
    # value frame's live rank is 33 of 43 — ten base quantities (3 stocktwits,
    # vix/vix3m/vvix, 4 voldemand) are pre-go-live DEAD (sd=0) in the frozen
    # frame window [24000, 48000), reproduced by experiments/_audit_liveness.py.
    # K=33 is the FULL live spectrum of this frame; the go-live signal is
    # carried by the 576 availability indicators (the Ind twin's own block),
    # so no information is lost. Registry keys keep the "Forty" label because
    # the increments pairs are already registered under it.
    "blk2_gFortyRungs_tuned": ArmSpec(
        describe="capture ladder, rungs repaired: backbone + K=33 (full live "
        "spectrum of the value frame, audited 2026-08-09) variance-ordered "
        "frozen-frame directions expanded through ALL TWELVE ladder rungs "
        "(396 columns), inputs standardized, scores NEVER standardized; no indicators",
        kind="blocks_tuned",
        blocks=[("backbone", "backbone"), ("rotval:variance:33", "trans_sub")],
        oos_mult=2,
    ),
    "blk2_gFortyRungsInd_tuned": ArmSpec(
        describe="capture ladder, rungs + indicators: the arm above plus the "
        "availability indicators UNROTATED in their own scalar-penalty block",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("rotval:variance:33", "trans_sub"),
            ("avail_ind", "exog"),
        ],
        oos_mult=2,
    ),
    **{
        f"blk2_pls{tag}RungsInd_tuned": ArmSpec(
            describe=f"capture ladder, SUPERVISED basis at K={k}: PLS weight "
            "vectors fit on the FRAME WINDOW ONLY and frozen permanently, "
            "expanded through all twelve rungs, plus the unrotated "
            "availability indicators — covariance-with-target in place of "
            "variance-among-regressors, at a fraction of the directions",
            kind="blocks_tuned",
            blocks=[
                ("backbone", "backbone"),
                (f"plsval:{k}", "trans_sub"),
                ("avail_ind", "exog"),
            ],
            oos_mult=2,
        )
        for tag, k in (("Twenty", 20), ("Ten", 10))
    },
    # SERIAL-CORRELATION-CORRECTED grid-free shrinkage (author directive
    # 2026-08-08). THE LEADING REMAINING EXPLANATION of the JS gap, now that
    # drift has been measured and found insufficient: the drift audit put the
    # optimal-shrinkage increase from exog coefficient drift at only 1.4-3x
    # (17% -> 23-35% next-bar), with the backbone stable 94-97% over 22 years
    # and the drift itself a one-time ~40% drop on a 62-250 day timescale
    # followed by a 20-year plateau. That is nowhere near the observed gap:
    # blk3_js_tuned shrank 23% and scored 0.24631 (DM +26.6 vs blk3_tuned).
    #
    # THE SUSPECT. blk3_js_tuned's sigma^2 and coefficient covariance assume
    # INDEPENDENT residuals. They are not: this panel's bars carry substantial
    # serial correlation. With integrated autocorrelation time tau the
    # effective sample is n/tau, so sigma^2 (X'X)^+ UNDERSTATES the coefficient
    # covariance by ~tau, the Wald form ||gamma||^2 OVERSTATES the signal by
    # ~tau, and the positive-part factor 1 - (r-2) sigma^2 / ||gamma||^2 lands
    # far too close to 1. That is precisely the direction and roughly the
    # magnitude of the observed under-shrinkage.
    #
    # THE CORRECTION: estimate tau from the CURRENT WINDOW's own residuals by
    # Geyer's initial-positive-sequence rule (deterministic, no bandwidth to
    # choose — see _integrated_act) and inflate the noise scale entering the
    # factor by it. Everything else is byte-identical to blk3_js_tuned.
    # NOTE tau is measured on the RESIDUALS rather than reusing the ~175-bar
    # figure from the transmission section's factor-score spectrum: the
    # residuals are a different series and are what the estimator's
    # distributional assumption is actually about.
    #
    # A THIRD OUTCOME TO WATCH FOR, found in the synthetic and not in the
    # brief's two branches: the correction can OVERSHOOT into the positive
    # part's floor. On an AR(1) fixture with tau ~12 the factor went 0.769 ->
    # 0.000, i.e. the whole exog+product block is shrunk away and the model
    # reduces to the BACKBONE ALONE. With the panel's tau plausibly 50-300
    # that is a live possibility here. It is a legitimate output of the
    # estimator, not a failure — but it is a THIRD verdict, distinct from both
    # "closes the gap" and "still under-shrinks", and it would mean the
    # correction is right in direction and wrong in magnitude (tau applied to
    # the whole coefficient vector when only part of the covariance is
    # inflated by it). Read the factor in the canary before reading the QLIKE.
    #
    # OUTCOME (measured 2026-08-08, recorded because a negative that cost a
    # canary is worth keeping): THE HYPOTHESIS IS REFUTED, and by measurement
    # rather than by bug. The canary returned tau_resid = 1.0, and tau measured
    # directly on blk3_tuned's persisted OUT-OF-SAMPLE residuals gives Geyer
    # IPS tau = 1.88 with first-lag ACF 0.029. Per-bar refitting WHITENS the
    # one-step errors: each forecast is made from a window ending at t-1, so
    # consecutive residuals share almost no estimation error. The correction is
    # therefore INERT AND CORRECTLY SO — a tau near 1 multiplies sigma^2 by
    # near 1 — and this arm stands as a documented negative, not a fleet
    # candidate. The standing explanation for the JS gap reverts to the third
    # hypothesis below: the OBJECTIVE MISMATCH between QLIKE's asymmetry (and
    # the retransform from sqrt space) and the squared-error loss this
    # estimator is risk-optimal for.
    #
    # PREDICTION, recorded so the outcome is falsifiable either way. If serial
    # correlation IS the mechanism, the corrected factor should land at
    # shrinkage comparable to what the tuned ridge effectively applies, and the
    # arm should score near blk3_tuned (0.2194) rather than near the benchmark
    # (0.2335). If it STILL under-shrinks, the gap has a third cause, and the
    # standing next hypothesis is the OBJECTIVE MISMATCH: QLIKE is asymmetric
    # in the forecast, while this estimator minimizes squared error in the
    # sqrt-transformed space — a risk-optimal rule for the wrong loss is not
    # optimal for the reported one.
    "blk3_jsNeff_tuned": ArmSpec(
        describe="grid-free shrinkage with a SERIAL-CORRELATION-CORRECTED "
        "noise scale: positive-part James-Stein on the exog+product block "
        "with sigma^2 inflated by the residuals' integrated autocorrelation "
        "time (Geyer initial-positive-sequence, causal, no tuned quantity); "
        "otherwise identical to blk3_js_tuned",
        kind="shrink",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
        ],
        grid="js_neff",
        oos_mult=2,
    ),
    "blk3_npebNeff_tuned": ArmSpec(
        describe="the same serial-correlation correction applied to the "
        "nonparametric empirical-Bayes arm (standard errors scaled by "
        "sqrt(tau) before Tweedie); otherwise identical to blk3_npeb_tuned",
        kind="shrink",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
        ],
        grid="npeb_neff",
        oos_mult=2,
    ),
    # DE-CONFOUNDING CONTROL (author directive 2026-08-07). THIS ARM EXISTS TO
    # FIX A PUBLISHED COMPARISON, not to win.
    #
    # THE PROBLEM. The paper reports blk4_trailGShaped beating blk4_trailG_tuned
    # (-1.836e-4, DM -2.019) and attributes the gain to the RANK-SHAPED penalty.
    # The persisted trajectories do not support that attribution, twice over:
    #  (1) gamma=0 is BIT-EXACTLY the flat penalty (_fill_pen_span writes
    #      lambda0 * i**0 == lambda0) and is selected in 45% of retunes. The
    #      entire aggregate edge lives in the gamma<=0.5 subset — mean diff
    #      -4.697e-4, DM -3.604, better in 19 of 21 years — while on the windows
    #      where a tilt is actually applied (gamma>=2) the shaped arm LOSES:
    #      +2.079e-4, DM +2.113. Difference-in-differences z = -4.15, p = 6.7e-5.
    #      So the edge is concentrated exactly where the "shaped" arm is not
    #      shaped.
    #  (2) The two arms also differ in FRAME WIDTH: blk4_trailGShaped uses
    #      trans_trailG40 (K=40), blk4_trailG_tuned uses trans_trailG (K=20,
    #      TRANS_QPOOL). Every other block and grid is identical. The pair
    #      therefore confounds TILT with K, and the honest reading is that the
    #      gain is the K=40-vs-K=20 effect measured on the windows where the
    #      penalty happened to be flat.
    #
    # THE CONTROL. K=40 factor levels with the ORDINARY FLAT `trans` grid
    # (1e2, 1e3, 1e4) — no shape axis at all. That splits the confounded
    # comparison into two clean ones:
    #     vs blk4_trailG_tuned  -> the pure K effect at flat penalty (40 vs 20)
    #     vs blk4_trailGShaped  -> the pure SHAPE effect at matched K=40
    # The second is the comparison that decides whether rank-shaping earns
    # anything, and it is the one the campaign never ran.
    "blk4_trailG40_tuned": ArmSpec(
        describe="de-confounding control: K=40 trailing factor LEVELS with the "
        "ORDINARY FLAT transmission penalty (no shape axis) — isolates the K "
        "effect from the tilt effect in the blk4_trailGShaped vs "
        "blk4_trailG_tuned comparison",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG40", "trans"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # ADAPTIVE vs FIXED spectral tilt (author directive 2026-08-07).
    #
    # THIS ARM IS NOT ABOUT STANDARDIZATION CONVENTION. It exists to separate
    # two mechanisms the campaign currently conflates. Standardizing factor
    # score i divides it by ~sqrt(d_i), so a penalty lambda_i on the
    # standardized coefficient is lambda_i * d_i on the raw eigen-direction:
    # STANDARDIZATION IS ITSELF A SPECTRAL TILT (the gamma_eff identity at
    # TRANS_SHAPE_GAMMAS_WIDE). The two conventions differ in whether that
    # tilt ADAPTS:
    #   * TRAILING (blk4_trailGShapedWide) divides by a ROLLING sd, which
    #     tracks the CURRENT spectrum -> a TIME-VARYING tilt. Any single
    #     exponent quoted for it is only an average under the frozen power
    #     law, and even that is truncation-dependent (a = 0.98..1.18 across
    #     top-10..top-40 refits) — see TRANS_SHAPE_GAMMAS_WIDE.
    #   * FROZEN (this arm) divides by a CONSTANT frame-window sd -> a FIXED
    #     tilt of the same average magnitude.
    # The paper currently attributes trailing-beats-frozen to "causal scale
    # stability". If that is the whole story the two arms should differ by
    # little once both can reach the same gamma_eff. If the trailing arm still
    # wins, the claim is a stronger and more interesting one: an ADAPTIVE
    # spectral prior beats a fixed one, i.e. the useful content of the
    # transmission block is that its shrinkage tracks a moving spectrum.
    #
    # THE WIDE GAMMA GRID IS ESSENTIAL, not incidental: the two standardizations
    # sit at different points of the SAME tilt family, so a narrow exponent grid
    # would confound the mechanism question with grid REACH. Both arms carry
    # trans_shaped_wide, so both span the same gamma_eff range and the increment
    # isolates adaptivity.
    "blk4_trailGShapedFrozen": ArmSpec(
        describe="adaptive-vs-fixed tilt control: K=40 factor LEVELS with "
        "FROZEN-window standardization (the fixed-tilt twin of "
        "blk4_trailGShapedWide) and the same RANK-SHAPED transmission penalty "
        "over the wide exponent grid, cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_frozenG40", "trans_shaped_wide"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # ENDPOINT RELIEF for the two arms the shape-endpoint diagnostic flagged
    # (author directive 2026-08-07). Both change ONLY their shape grid; the
    # originals are untouched and stay byte-reproducible. Extension DIRECTION
    # was chosen from the measured pinning in each arm's own persisted
    # trajectory, not assumed — see TRANS_SHAPE_GAMMAS_BIPOLAR and
    # TIKHONOV_GAMMAS_WIDE / TIKHONOV_STEP_KS_WIDE for the percentages and for
    # why the pcrank and power families needed relief at the LOW end (the
    # larger pile-up in both cases) while the step family needed it at the top.
    "blk_pcladderWide_tuned": ArmSpec(
        describe="endpoint relief for the PC-ladder rung: blk_pcladder_tuned "
        "with the pcrank exponent grid made BIPOLAR (-1..4, 24 points) — the "
        "tuner pinned at BOTH ends of the old grid, 49% at gamma=0 and 32% at "
        "gamma=2; cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("pc_ladder", "pc_ladder_tilt_bipolar"),
            ("avail_ind", "exog"),
            ("product", "product"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    "blk3_tikhonovStepWide_tuned": ArmSpec(
        describe="endpoint relief for the hard-vs-soft rung: "
        "blk3_tikhonovStep_tuned with BOTH families widened — power to "
        "(-1..3) where it pinned at the gamma=0 floor in 56% of retunes, and "
        "step K0 to (5..100) where it pinned at the K0=80 ceiling in 48%; "
        "39-point block grid, cyclic causal tuning",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_rot", "exog_tilt_step_wide"),
            ("product", "product"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # SHAPE ZOO (author directive 2026-08-07): the same K=40 transmission
    # block, but the tuner chooses the penalty FAMILY as well as its parameter
    # — power (polynomial tail), exponential (geometric tail), or step (hard
    # cutoff on this basis), 36 points on one block. An eigenvalue-based
    # profile is deliberately ABSENT: the frame's spectrum is a power law
    # (d_i ~ c*i^-a, a = 1.176 on the top-40 truncation), so
    # lambda_i ∝ d_i^-theta is the rank power law at gamma = a*theta and
    # would be a reparameterization, not an experiment. See
    # BLOCK_TUNE_GRIDS["trans_shaped_zoo"]. THE EXHIBIT is the selected family
    # per retune and by year (penalty_shape_summary.csv shape_family /
    # frac_of_year), not the QLIKE alone.
    "blk4_trailGZoo": ArmSpec(
        describe="shape zoo: trailing factor LEVELS at K=40 with the "
        "transmission penalty selected from THREE families — power, "
        "exponential and step (36-point block grid) — by the same cyclic "
        "causal tuner; power/gamma=0 still nests the flat penalty exactly",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG40", "trans_shaped_zoo"),
        ],
        grid="cyclic",
        oos_mult=2,
    ),
    # Penalty-allocation ladder's per-bucket rung (author directive
    # 2026-08-07): champion structure, but the ONE exogenous penalty is
    # replaced by one penalty per canonical family, selected by deterministic
    # cyclic coordinate descent (see _walk_blocks_tuned selection="cyclic").
    # Nests blk4_trail_tuned exactly when all family penalties coincide.
    "blk_bucketpen_tuned": ArmSpec(
        describe="penalty-allocation rung: per-BUCKET exogenous penalties "
        f"({len(_BUCKET_FAMILIES)} families, one each) + backbone/product/"
        "trailing-transmission, cyclic causal tuning (3 passes, fixed order)",
        kind="blocks_tuned",
        blocks=(
            [("backbone", "backbone")]
            + [(f"bucket:{b}", f"{_BUCKET_PEN_KEY}{b}") for b in _BUCKET_FAMILIES]
            + [("product", "product"), ("trans_trail", "trans")]
        ),
        grid="cyclic",  # selection mode marker (see compute dispatch)
        oos_mult=2,
    ),
    # Parsimony counterpart of the champion (author directive 2026-08-07): the
    # falsification result put the transmission block's value in the trailing
    # factor LEVELS G, not the operator columns Ghat (levels-only vs full:
    # paired DM t=1.00, p=0.32; stronger vs the 3-block ridge, -5.95 vs -4.14;
    # half the columns; no operator machinery at all). The headline model
    # cannot claim parsimony without this arm.
    "blk4_trailG_tuned": ArmSpec(
        describe="parsimony contender: trailing factor SCORES ONLY (20 cols, "
        "no Ghat, no operator machinery) with PER-BLOCK CAUSAL TUNING "
        "— blk4_trail_tuned's levels-only twin",
        kind="blocks_tuned",
        blocks=[
            ("backbone", "backbone"),
            ("exog_all", "exog"),
            ("product", "product"),
            ("trans_trailG", "trans"),
        ],
        oos_mult=2,
    ),
    # Transmission dig (author directive 2026-08-07): 7 variants of the
    # trailing construction, fixed user penalties, same window/legality as
    # blk4_trail. Mechanism diagnostics (per-refresh operators, frame
    # eigenvalues, refresh rows) persist per chunk as npz arrays.
    **{
        name: _blk(
            [
                ("backbone", "backbone"),
                ("exog_all", "exog"),
                ("product", "product"),
                (trans_key, "trans"),
            ],
            USER_ALPHAS,
            None,
            desc,
        )
        for name, trans_key, desc in (
            (
                "blk4_trailSym",
                "trans_trailSym",
                "dig: SYMMETRIC part S of the lagged cross-correlation "
                "(lead-lag falsification control; diag zeroed to mirror D)",
            ),
            (
                "blk4_trailFullC",
                "trans_trailFullC",
                "dig: undecomposed lagged cross-correlation C as the operator",
            ),
            (
                "blk4_trailRefresh",
                "trans_trailRefresh",
                "dig: frame re-estimated causally (trailing 504d corr, "
                "quarterly, order+sign-aligned across refreshes)",
            ),
            (
                "blk4_trailKFive",
                "trans_trailK5",
                "dig: frozen frame at K=5 eigenvectors",
            ),
            (
                "blk4_trailKTen",
                "trans_trailK10",
                "dig: frozen frame at K=10 eigenvectors",
            ),
            (
                "blk4_trailKForty",
                "trans_trailK40",
                "dig: frozen frame at K=40 eigenvectors (loud fail if the "
                "spectrum is narrower)",
            ),
            (
                "blk4_trailLagTwo",
                "trans_trailLag2",
                "dig: lag-2-bar cross-correlation operator, Ghat = D2.G(t-2)",
            ),
            (
                "blk4_trailDropHet",
                "trans_trailDropHet",
                "dig: transmission base excludes the cadence-heterogeneous "
                "sumret3_{ew,vw}stock families (Dec-2014 homogenization "
                "hypothesis; exog ridge block keeps them)",
            ),
        )
    },
    # Causally-TUNED block ladder (author directive 2026-08-06): _user block
    # structures, per-block alphas re-selected jointly every TUNE_PER=250
    # solves over BLOCK_TUNE_GRIDS (see _walk_blocks_tuned; selection
    # trajectory persisted in meta.tuned_alphas).
    **{
        name: ArmSpec(
            describe=desc,
            kind="blocks_tuned",
            blocks=blocks,
            oos_mult=2
            if any(b == "product" or b.startswith("trans") for b, _ in blocks)
            else 1,
        )
        for name, blocks, desc in (
            (
                "blk2_tuned",
                [("backbone", "backbone"), ("exog_all", "exog")],
                "tuned 2-block ridge: joint 3x3 alpha grid per retune",
            ),
            (
                "blk3_tuned",
                [
                    ("backbone", "backbone"),
                    ("exog_all", "exog"),
                    ("product", "product"),
                ],
                "tuned 3-block ridge: joint 27-combo alpha grid per retune",
            ),
            (
                "blk4_tuned",
                [
                    ("backbone", "backbone"),
                    ("exog_all", "exog"),
                    ("product", "product"),
                    ("trans_user", "trans"),
                ],
                "tuned 4-block ridge: joint 81-combo alpha grid per retune",
            ),
            (
                "c4_product_alone_tuned",
                [("backbone", "backbone"), ("product", "product")],
                "tuned diagnostic: backbone + products, joint 9-combo grid",
            ),
            (
                "d3_transmission_alone_tuned",
                [("backbone", "backbone"), ("trans_user", "trans")],
                "tuned diagnostic: backbone + transmission [G|Ghat], joint "
                "9-combo grid",
            ),
        )
    },
}
_ALIASES = {alias: name for name, spec in ARMS.items() for alias in spec.aliases}


def _build_block(p: _Panel, block: str, window: int, arm: str = "") -> np.ndarray:
    if block == "wide":
        return p.X  # the full all_features basis (backbone + every exog MA)
    if block == "backbone":
        return p.X[:, _backbone_cols(p.names)]
    if block == "exog_all":
        return p.X[:, _exog_all_cols(p.names)]
    if block == "exog_rot":  # frozen-eigenframe rotation (generalized Tikhonov)
        return _exog_tilt_design(p, window)
    if block == "rot_prod_timeaware":  # time-aware post-PCA (author directive 2026-08-09)
        return _rot_prod_timeaware_design(p, window, arm)
    if block == "rot_prod_wobble":  # deterministic causal ellipsoid wobble, uniform exog penalty
        return _rot_prod_wobble_design(p, window, arm, timeaware=False)
    if block == "rot_prod_timeaware_wobble":  # wobble aligned to the timescale penalty profile
        return _rot_prod_wobble_design(p, window, arm, timeaware=True)
    if block == "pc_ladder":  # {ma_j(G_i)} — ladder applied to the PC series
        return _pc_ladder_design(p, window)
    # K-sweep of the same construction (2026-08-07): K=20 discards most of the
    # space, so the arm is a COMPRESSION experiment; at full rank the ladder-of-
    # PCs and the ladder-expanded base span the same columns and it becomes a
    # REPARAMETERIZATION experiment where only the penalty profile can differ.
    if block == "pc_ladder40":
        return _pc_ladder_design(p, window, qpool=40)
    if block == "pc_ladder80":
        return _pc_ladder_design(p, window, qpool=80)
    if block == "pc_ladder_full":  # K read from the frame's own liveness rule
        return _pc_ladder_design(p, window, qpool=_frame_live_rank(p, window))
    if block == "pc_ladder_perrung":  # one frozen eigenbasis PER ladder rung
        return _pc_ladder_perrung_design(p, window)
    if block.startswith("rotval:"):  # proper PCR: rotated VALUE tensor
        _, ordering, ktag = block.split(":")
        return _rot_value_design(
            p, window, None if ktag == "full" else int(ktag), ordering
        )
    if block == "rot_prod_full":  # production-scaled full-rank rotation
        return _rot_prod_design(p, window)
    if block.startswith("plsval:"):
        # SUPERVISED basis, all 12 ladder rungs, indicators handled separately
        return _rot_value_design(
            p, window, int(block.split(":")[1]), "variance", basis="pls"
        )
    if block.startswith("rotexog:"):
        # ROTATED VALUES + UNROTATED INDICATORS as ONE penalized block. The
        # _js twins carry these as two segments sharing the shrinkage; the
        # tuned twins must give them ONE alpha, and the block tuner keys its
        # search on segment alpha-keys, so two segments with the same key
        # would collapse in the combo dict and silently waste the axis.
        # Concatenated in the SAME column order as the _js arms
        # (rotated values first, then indicators), which is asserted
        # bit-identical in the verify script.
        _, ordering, ktag = block.split(":")
        rot = _rot_value_design(
            p, window, None if ktag == "full" else int(ktag), ordering
        )
        return np.hstack([rot, p.X[:, _cols(p.names, {"indicator"})]])
    if block == "value_all":  # the unrotated VALUE tensor (gate comparator)
        cols = _value_slab_cols(p.names)
        return np.ascontiguousarray(
            p.X[:, np.concatenate([cols[w] for w in sorted(cols)])]
        )
    if block == "avail_ind":  # availability indicators (binary; never projected)
        return p.X[:, _cols(p.names, {"indicator"})]
    if block.startswith("bucket:"):
        return p.X[:, _bucket_cols(p.names, block.split(":", 1)[1])]
    if block == "product":
        return _frozen_products(p, window)
    if block == "trans_user":  # [G | Ghat] — the paper's own design (ruling 2026-08-06)
        return _transmission_block(p, window, parts="both")
    if block == "trans_doc":  # Ghat only — the documented construction verbatim
        return _transmission_block(p, window, parts="flow")
    if block == "trans_trail":  # [G | Ghat], TRAILING standardization (revival)
        return _transmission_block(p, window, parts="both", standardization="trailing")
    if block == "trans_trailG":  # ablation: trailing factor LEVELS only
        return _transmission_block(
            p, window, parts="scores", standardization="trailing"
        )
    if block == "trans_trailG40":  # levels only, WIDE frame (rank-shaped arm)
        return _transmission_block(
            p, window, parts="scores", standardization="trailing", qpool=40
        )
    if block == "trans_trailG40_trailmean":
        # Trailing-standardization decomposition: causal trailing DEMEANING
        # only, then the same production rolling robust scaler as the parent.
        return _transmission_block(
            p, window, parts="scores", standardization="trail_mean", qpool=40
        )
    if block == "trans_trailG40_trailsd":
        # Trailing-standardization decomposition: division by the causal
        # trailing SD only, then the same production rolling robust scaler.
        return _transmission_block(
            p, window, parts="scores", standardization="trail_sd", qpool=40
        )
    if block == "trans_trailGFull":
        # FULL LIVE SPECTRUM of the frozen frame, read from the frame's own
        # liveness rule at build time (never hardcoded). The documented count
        # is TRANS_FULL_SPECTRUM; a panel whose live rank has drifted changes
        # what this arm MEANS, so the mismatch is loud rather than silent.
        k_live = _frame_live_rank(p, window)
        if k_live != TRANS_FULL_SPECTRUM:
            raise SystemExit(
                f"trans_trailGFull: frame live rank is {k_live}, not the "
                f"documented {TRANS_FULL_SPECTRUM} (of "
                f"{len(_product_base_cols(p.names))} base columns). The arm's "
                "identity is tied to that count — re-audit the liveness before "
                "running, or update TRANS_FULL_SPECTRUM deliberately."
            )
        return _transmission_block(
            p, window, parts="scores", standardization="trailing", qpool=k_live
        )
    if block == "trans_frozenG40":
        # ADAPTIVE-vs-FIXED tilt control (see blk4_trailGShapedFrozen): the
        # SAME K=40 factor levels, standardized with FROZEN frame-window stats
        # instead of trailing ones. Identical columns, identical frame; only
        # the divisor's time-dependence changes.
        return _transmission_block(
            p, window, parts="scores", standardization="frozen", qpool=40
        )
    if block == "trans_trailGhat":  # ablation: trailing lead-lag FLOW only
        return _transmission_block(p, window, parts="flow", standardization="trailing")
    # Transmission dig (2026-08-07): variants of the trailing construction.
    if block == "trans_trailSym":  # lead-lag falsification: symmetric part S
        return _transmission_block(
            p, window, parts="both", standardization="trailing", operator="sym"
        )
    if block == "trans_trailFullC":  # undecomposed lagged cross-correlation C
        return _transmission_block(
            p, window, parts="both", standardization="trailing", operator="full"
        )
    if block == "trans_trailRefresh":  # causal quarterly-refreshed frame
        return _transmission_block(
            p, window, parts="both", standardization="trailing", frame="refresh"
        )
    if block == "trans_trailK5":
        return _transmission_block(
            p, window, parts="both", standardization="trailing", qpool=5
        )
    if block == "trans_trailK10":
        return _transmission_block(
            p, window, parts="both", standardization="trailing", qpool=10
        )
    if block == "trans_trailK40":
        return _transmission_block(
            p, window, parts="both", standardization="trailing", qpool=40
        )
    if block == "trans_trailLag2":  # lag-2-bar cross-correlation; Ghat = Op.G(t-2)
        return _transmission_block(
            p, window, parts="both", standardization="trailing", lag=2
        )
    if block == "trans_trailDropHet":  # cadence-homogenization surgical test
        return _transmission_block(
            p,
            window,
            parts="both",
            standardization="trailing",
            exclude_stems=TRANS_HET_STEMS,
        )
    raise KeyError(f"unknown block '{block}'")


def _build_design(p: _Panel, spec: ArmSpec, window: int) -> tuple[np.ndarray, float]:
    """(design, solver_alpha). Per-block penalties imposed by column scaling:
    scaling block j by sqrt(a_ref / a_j) under global ridge a_ref is EXACTLY the
    per-block penalty (minimal_model.py:297-299); intercept stays unpenalized."""
    if spec.kind == "tuned":
        if spec.blocks:  # per-bucket tuned family: backbone + bucket design
            idx, _ = _design_cols(p, spec)
            return np.ascontiguousarray(p.X[:, idx]), 0.0
        return p.X, 0.0  # head-to-head tuned arms: the full wide basis
    a_ref = (
        spec.alphas[spec.blocks[0][1]]
        if len(spec.blocks) == 1
        else (spec.alphas.get("exog", spec.alphas[spec.blocks[0][1]]))
    )
    parts = []
    for b, akey in spec.blocks:
        blk = np.ascontiguousarray(_build_block(p, b, window, arm=arm), dtype=np.float64)
        a_j = spec.alphas[akey]
        if a_j != a_ref:
            blk = blk * np.sqrt(a_ref / a_j)
        parts.append(blk)
    return np.hstack(parts), a_ref


def panel_length() -> int:
    """Total b2-panel row count after prep — the runner's chunk-grid sizing call.

    Cheap when either the in-process panel or the ``prepare_full`` cache exists
    (reads ONE member of the npz); falls back to a full panel build otherwise.
    """
    if _PANEL is not None:
        return len(_PANEL.y)
    cache = os.path.join(
        os.environ.get(CACHE_DIR_ENV, "results"), "prep_cache_all_features_b2.npz"
    )  # prepare_full's key for
    #   bucket=all_features, har_base=2, kernel=mean, diurnal=divide, no PREP_ROWS
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=True) as z:
            return int(z["y"].shape[0])
    return len(_load_panel().y)


def _tuned_blocks_design(
    p: _Panel, spec: ArmSpec, window: int, arm: str = ""
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """UNSCALED block design + column segments for the causally-tuned block
    arms: (F, [(col_start, col_end, alpha_key), ...]). Penalties are applied at
    solve time as diag vectors (see _walk_blocks_tuned), never by pre-scaling."""
    parts: list[np.ndarray] = []
    segments: list[tuple[int, int, str]] = []
    start = 0
    for b, akey in spec.blocks:
        blk = np.ascontiguousarray(_build_block(p, b, window, arm=arm), dtype=np.float64)
        segments.append((start, start + blk.shape[1], akey))
        start += blk.shape[1]
        parts.append(blk)
    return np.hstack(parts), segments


def _design_cols(p: _Panel, spec: ArmSpec) -> tuple[np.ndarray, list[str]]:
    """Column indices + names for backbone/bucket-composed designs (the OLS
    family and the per-bucket tuned family share this assembly)."""
    cols: list[int] = []
    for b, _ in spec.blocks:
        if b == "backbone":
            cols += list(_backbone_cols(p.names))
        elif b.startswith("bucket:"):
            cols += list(_bucket_cols(p.names, b.split(":", 1)[1]))
        else:
            raise KeyError(
                f"only backbone/bucket blocks compose this design, got '{b}'"
            )
    idx = np.asarray(cols, dtype=np.int64)
    return idx, [p.names[j] for j in idx]


def _ols_design(p: _Panel, spec: ArmSpec) -> tuple[np.ndarray, list[str], list[str]]:
    """OLS design by named column selection + deterministic dedup (see
    :func:`_dedup_ols_design`). Returns (F, kept_names, dropped_names)."""
    idx, names_design = _design_cols(p, spec)
    return _dedup_ols_design(p.X[:, idx], names_design)


def _registry_text() -> str:
    lines = ["unification arm registry (all arms REAL — no stubs):"]
    for name, spec in ARMS.items():
        lines.append(
            f"  {name:24s} [{spec.kind:6s}] w={spec.window:5d}  {spec.describe}"
        )
    for alias, target in _ALIASES.items():
        lines.append(f"  {alias:24s} -> alias of {target}")
    lines.append(
        "required flags: --arm --chunk-start --chunk-end [--window 24000] "
        "[--halo 24000] --output-file"
    )
    return "\n".join(lines)


# ── compute ───────────────────────────────────────────────────────────────────


def compute(args: argparse.Namespace) -> None:
    """Run one arm x chunk task and write the raw per-bar npz (schema: module
    docstring). No smearing, no raw-space conversion — scoring is a local pass."""
    arm = getattr(args, "arm", "") or ""
    if not arm:
        # No-arm invocation (e.g. smoke-test-executor defaults): print the
        # registry and exit cleanly — the import + dispatch path is the test.
        print(_registry_text())
        return
    arm = _ALIASES.get(arm, arm)
    if arm not in ARMS:
        raise SystemExit(f"unknown arm '{arm}'.\n{_registry_text()}")
    spec = ARMS[arm]
    global _LAST_TRANS_DIAG
    _LAST_TRANS_DIAG = None  # populated by any transmission-block build below
    # Same discipline for the PLS frame: CLEARED here so a diag left behind by
    # a previous arm in the same process can never be mistaken for this arm's
    # provenance, and so the loud check below tests THIS build.
    _LAST_PLS_DIAG.clear()

    missing = [
        f
        for f in ("chunk_start", "chunk_end", "output_file")
        if getattr(args, f, None) is None
    ]
    if missing:
        raise SystemExit(
            f"arm '{arm}' needs flags: {', '.join('--' + m.replace('_', '-') for m in missing)}\n"
            + _registry_text()
        )
    window = int(spec.window_bars or getattr(args, "window", 0) or DEFAULT_WINDOW_BARS)
    halo = int(getattr(args, "halo", 0) or 0)
    if halo and halo < window:
        raise SystemExit(
            f"halo={halo} < training window {window}: a mid-series chunk cannot warm up"
        )

    p = _load_panel()
    n = len(p.y)
    lo = int(args.chunk_start)
    hi = n if int(args.chunk_end) < 0 else min(int(args.chunk_end), n)
    first_oos = spec.oos_mult * window
    if lo < first_oos:
        raise SystemExit(
            f"arm '{arm}': chunk_start={lo} < first legal OOS row {first_oos} "
            f"({'selection/frame block excluded' if spec.oos_mult == 2 else 'training window'})"
        )
    if hi <= lo:
        raise SystemExit(f"empty chunk [{lo}, {hi})")
    _slice(lo, hi, window, n)

    dropped_cols: list[str] = []
    collinear_cols: list[str] = []
    ladder_drops: dict[str, list[int]] = {}
    masked_cols: dict[str, dict[str, Any]] = {}
    tuned_alphas: list[dict[str, Any]] = []
    tuned_grids: dict[str, list[Any]] = {}
    coarse_grids: dict[str, list[Any]] = {}
    tuned_penalty: list[dict[str, Any]] = []
    shrink_profile: list[dict[str, Any]] = []
    tree_cfg: dict[str, Any] = {}
    if spec.kind == "ols":
        # Min-norm path (2026-08-07): panel bitwise dedup stays (hygiene); the
        # QR entry drops / atomic ladder drops / sticky repair are retired —
        # pinv handles rank deficiency natively (collinear/ladder meta keys
        # stay in the schema, permanently empty for min-norm runs).
        F, kept_names, dropped_cols = _ols_design(p, spec)
        support = _support_masks(p, kept_names, window, lo, hi)
        yhat, masked_cols = _walk_ols(
            F, p.y, window, lo, hi, kept_names, support_masks=support
        )
    elif spec.kind == "blocks":
        F, a_ref = _build_design(p, spec, window, arm=arm)
        yhat = _walk_ridge(F, p.y, window, lo, hi, alpha=a_ref)
    elif spec.kind == "blocks_tuned":
        arm_keys = [k for _, k in spec.blocks]
        tuned_grids = {k: list(BLOCK_TUNE_GRIDS[k]) for k in arm_keys}
        # Fine-grid arms carry their COARSE ancestor so the interstitial-usage
        # diagnostic reads out of the chunk itself (no scorer-side arm list).
        coarse_grids = {
            k: list(BLOCK_TUNE_GRIDS[FINE_GRID_PARENT[k]])
            for k in arm_keys
            if k in FINE_GRID_PARENT
        }
        if spec.grid == "cyclic" and any(
            k.startswith(_BUCKET_PEN_KEY) for k in arm_keys
        ):
            # per-bucket rung: verify the family taxonomy partitions the
            # exogenous columns BEFORE building anything (loud on failure)
            assert_bucket_partition(p.names)
        F, segments = _tuned_blocks_design(p, spec, window, arm=arm)
        # Sweep order (fixed, deterministic): bucket families in canonical
        # order first (when present, per the per-bucket directive), then the
        # remaining block keys in their declared order.
        sweep: tuple[str, ...] = ()
        if spec.grid == "cyclic":
            bucket_keys = [k for k in arm_keys if k.startswith(_BUCKET_PEN_KEY)]
            sweep = tuple(bucket_keys) + tuple(
                k for k in arm_keys if k not in bucket_keys
            )
        yhat, tuned_alphas = _walk_blocks_tuned(
            F,
            segments,
            p.y,
            window,
            lo,
            hi,
            selection="cyclic" if spec.grid == "cyclic" else "cartesian",
            sweep_order=sweep,
        )
    elif spec.kind == "blocks_tuned_bag":
        # ROTATION BAGGING: average per-bar predictions over 8 seeded causal
        # orthogonal perturbations. Designs are built one member at a time and
        # released immediately — eight full panels would not fit in memory.
        arm_keys = [k for _, k in spec.blocks]
        tuned_grids = {k: list(BLOCK_TUNE_GRIDS[k]) for k in arm_keys}
        yhat = None
        tuned_alphas = []
        for m in range(8):
            F, segments = _tuned_blocks_design(p, spec, window, arm=f"{arm}__bag{m}")
            y_m, ta_m = _walk_blocks_tuned(
                F,
                segments,
                p.y,
                window,
                lo,
                hi,
                selection="cartesian",
                sweep_order=(),
            )
            for d in ta_m:
                d["bag_member"] = m
            tuned_alphas.extend(ta_m)
            yhat = y_m if yhat is None else yhat + y_m
            if m < 7:
                del F
        assert yhat is not None
        yhat = yhat / 8.0
    elif spec.kind == "blocks_ew":
        # DISCOUNTED GRAM: same block grids as the flat tuned arms, plus the
        # half-life axis. "ew_grid" searches EW_HALF_LIVES; "ew_fixed:<H>"
        # pins a single half-life so only the penalties are selected.
        arm_keys = [k for _, k in spec.blocks]
        tuned_grids = {k: list(BLOCK_TUNE_GRIDS[k]) for k in arm_keys}
        if spec.grid == "ew_grid":
            hls: tuple[float, ...] = EW_HALF_LIVES
        elif spec.grid.startswith("ew_fixed:"):
            hls = (float(spec.grid.split(":", 1)[1]),)
        else:
            raise SystemExit(f"arm '{arm}': unknown EW grid tag '{spec.grid}'")
        tuned_grids["half_life"] = [float(h) for h in hls]
        F, segments = _tuned_blocks_design(p, spec, window, arm=arm)
        yhat, tuned_alphas = _walk_blocks_ew(F, segments, p.y, window, lo, hi, hls)
    elif spec.kind == "shrink":
        # GRID-FREE: no BLOCK_TUNE_GRIDS lookup, no tuner, no validation tail.
        # The block keys are carried only to mark which columns are the
        # backbone (left unshrunk); no penalty is ever read from them.
        F, segments = _tuned_blocks_design(p, spec, window, arm=arm)
        yhat, shrink_profile = _walk_shrink(
            F, p.y, window, lo, hi, segments, estimator=spec.grid
        )
    elif spec.kind == "tree":
        menu = _load_tree_menu()  # LOUD failure when the frozen menu is absent
        k = int(spec.grid)
        if k >= len(menu):
            raise SystemExit(
                f"arm '{arm}': expert index {k} beyond menu length {len(menu)} "
                f"({TREE_MENU_PATH})"
            )
        tree_cfg = menu[k]
        if "family" not in tree_cfg:
            raise SystemExit(
                f"arm '{arm}': menu entry {k} lacks a 'family' tag "
                f"({TREE_MENU_PATH} predates the mixed-family expansion?)"
            )
        # author correction 2026-08-07: design IDENTICAL to the tuned
        # penalized linear arms — the full wide all_features basis, unchanged
        F = p.X
        yhat = _walk_tree(
            F, p.y, window, lo, hi, tree_cfg["params"], tree_cfg["family"]
        )
    else:
        F, _ = _build_design(p, spec, window)
        # Persist the estimator grid the tuner actually searched, so the scorer
        # can flag alpha/l1 endpoint pile-up for the SINGLE-ESTIMATOR arms the
        # same way it already does for the block arms — without importing this
        # module or hardcoding any grid. (The enet reach defect was invisible
        # for exactly this reason: nothing downstream knew where the ceiling
        # was.)
        tuned_grids = {
            "__estimator__": [
                [str(k), float(a), float(l1)] for k, a, l1 in ESTIMATOR_GRIDS[spec.grid]
            ]
        }
        if spec.grid in ESTIMATOR_GRID_PARENT:  # fine-grid twin: carry the ancestor
            coarse_grids = {
                "__estimator_alpha__": [
                    float(a)
                    for _, a, _ in ESTIMATOR_GRIDS[ESTIMATOR_GRID_PARENT[spec.grid]]
                ]
            }
        yhat, tuned_penalty = _walk_tuned(
            F, p.y, window, lo, hi, ESTIMATOR_GRIDS[spec.grid]
        )

    y_fit = p.y[lo:hi]
    # Persistence-path alignment gate for EVERY arm (v2 trail incident): the
    # fit-side rows and the raw-side rows written to the npz must describe the
    # same bars, verified by value, not by length.
    _assert_fit_raw_alignment(
        y_fit, p.rv_raw[lo:hi], p.baseline[lo:hi], f"persistence, chunk [{lo},{hi})"
    )
    resid = y_fit - yhat
    result: dict[str, Any] = {
        "sqrt_mse": float(np.mean(resid**2)),
        "sqrt_sse": float(np.sum(resid**2)),
        "sum_y": float(np.sum(y_fit)),
        "sum_y2": float(np.sum(y_fit**2)),
        "n": int(hi - lo),
        "ols_dropped_cols": len(dropped_cols),
        "ols_collinear_cols": len(collinear_cols),
        "ols_masked_cols": len(masked_cols),
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    # transmission mechanism diagnostics (dig 2026-08-07): arrays ride in the
    # npz (meta carries only the summary — a 90x40x40 float stack does not
    # belong in JSON), unlocking operator/spectrum analysis without refits.
    # PLS frame provenance. THE GAP THIS CLOSES (2026-08-08): the weights were
    # recorded into a module global that the synthetic read, but nothing ever
    # merged it into the meta dict the runner serializes — so the arm passed
    # every local check and shipped chunks with no provenance at all. The
    # assertion below makes that class of gap impossible to repeat: an arm that
    # BUILDS a PLS block must be able to SHOW the frame it used, or it refuses
    # to write. Checked against what is about to be persisted, not against the
    # builder's intent.
    pls_diag = dict(_LAST_PLS_DIAG)
    _uses_pls = any(b.startswith("plsval:") for b, _ in spec.blocks)
    if _uses_pls and not pls_diag:
        raise SystemExit(
            f"arm '{arm}' declares a PLS block "
            f"({[b for b, _ in spec.blocks if b.startswith('plsval:')]}) but no "
            "pls_diag was produced by the build — the frozen supervised frame "
            "would be unauditable in the persisted chunk. Refusing to write."
        )
    if _uses_pls:
        missing_keys = {"weights_sha256", "weights_shape", "frame_rows"} - set(pls_diag)
        if missing_keys:
            raise SystemExit(
                f"arm '{arm}': pls_diag is missing {sorted(missing_keys)} — the "
                "frame's provenance must be complete or it is not provenance. "
                "Refusing to write."
            )
    if pls_diag and not _uses_pls:
        # a stale diag from another arm would silently mislabel this chunk
        raise SystemExit(
            f"arm '{arm}' produced a pls_diag but declares no PLS block — "
            "stale module state; refusing to write a mislabelled chunk."
        )
    td = _LAST_TRANS_DIAG
    trans_arrays: dict[str, Any] = (
        {
            "trans_ops": td["ops"],
            "trans_eigvals": td["eigvals"],
            "trans_refresh_rows": td["refresh_rows"],
        }
        if td is not None
        else {}
    )
    np.savez_compressed(
        args.output_file,
        **trans_arrays,
        row_id=np.arange(lo, hi, dtype=np.int64),
        t=p.t[lo:hi],
        y_fit=np.asarray(y_fit, dtype=np.float64),
        yhat=np.asarray(yhat, dtype=np.float64),
        rv_raw=np.asarray(p.rv_raw[lo:hi], dtype=np.float64),
        baseline=np.asarray(p.baseline[lo:hi], dtype=np.float64),
        valid_mask=np.ones(hi - lo, dtype=bool),
        # pooled-R^2 sufficient statistics as scalar arrays — the npz is the
        # single source of truth (the metrics sidecar below is optional)
        sqrt_sse=np.float64(result["sqrt_sse"]),
        sqrt_mse=np.float64(result["sqrt_mse"]),
        sum_y=np.float64(result["sum_y"]),
        sum_y2=np.float64(result["sum_y2"]),
        n=np.int64(result["n"]),
        ols_dropped_cols=np.int64(len(dropped_cols)),
        ols_collinear_cols=np.int64(len(collinear_cols)),
        ols_masked_cols=np.int64(len(masked_cols)),
        meta=json.dumps(
            {
                "arm": arm,
                "kind": spec.kind,
                "window_bars": window,
                "chunk_start": lo,
                "chunk_end": hi,
                "first_oos": first_oos,
                "har_base": HAR_BASE,
                "alphas": spec.alphas,
                "grid": spec.grid,
                "n_design_cols": int(F.shape[1]),
                # deterministic exact-duplicate/constant drops (OLS arms; see
                # _dedup_ols_design) — recorded so the design is auditable per task
                "ols_dropped_cols": dropped_cols,
                # exact-collinearity drops on the chunk's first training window
                # (accounting identities; see _eliminate_exact_collinear)
                "ols_collinear_cols": collinear_cols,
                # ATOMIC availability-ladder drops (c080 guard): ladder prefix
                # -> rung windows dropped whole when any rung went exact-
                # collinear at chunk entry
                "ols_ladder_drops": ladder_drops,
                # window-level identifiability mask report (OLS arms; ruling
                # 2026-08-06): name -> {bars, first, last} in GLOBAL row indices
                "ols_masked_cols": masked_cols,
                # causal per-block alpha selection trajectory (blocks_tuned
                # arms): [{row, alphas}] per retune boundary — paper-relevant
                "tuned_alphas": tuned_alphas,
                # the grids the tuner actually searched, per block key — lets
                # the scorer flag endpoint pile-up without importing the
                # executor (a grid whose optimum sits at an endpoint is not a
                # valid selection)
                "tuned_grids": tuned_grids,
                # GRID-FREE arms only: the per-boundary distribution of the
                # ESTIMATED shrinkage factors. Empty for every other arm. This
                # is the exhibit — the shrinkage the DATA asks for, against the
                # shapes the tuned arms impose.
                "shrink_profile": shrink_profile,
                # HALF-DECADE arms only: the COARSE grid each fine grid
                # refines, so the scorer can report what fraction of retunes
                # land on an INTERSTITIAL point — the number that separates
                # "resolution was never binding" from "selection noise"
                # (fine_grid_usage.csv). Empty for every other arm.
                "coarse_grids": coarse_grids,
                # tree expert arms: the frozen menu entry (name, params, sha)
                "tree_config": tree_cfg,
                # transmission dig: construction summary (arrays in the npz
                # under trans_ops / trans_eigvals / trans_refresh_rows)
                # PLS arms only: sha256 + shape of the FROZEN supervised
                # weight matrix and the frame rows it was fit on. Constant
                # across every chunk of an arm by construction (the frame is
                # frozen) — a varying hash means the frame is being refit and
                # the arm is not what it claims. Empty for every other arm.
                "pls_diag": pls_diag,
                "transmission_diag": (
                    {
                        "operator": td["operator"],
                        "frame": td["frame"],
                        "qpool": td["qpool"],
                        "lag": td["lag"],
                        "n_refresh": int(len(td["refresh_rows"])),
                        "excluded_base_cols": td.get("excluded_base_cols", []),
                    }
                    if td is not None
                    else {}
                ),
                # single-estimator tuned arms: selected (alpha, l1_ratio) per
                # retune boundary + per-chunk means (the shrink-vs-select
                # exhibit; l1_ratio=0 for pure-ridge grids)
                "tuned_penalty": tuned_penalty,
                "tuned_penalty_summary": (
                    {
                        "mean_alpha": float(
                            np.mean([e["alpha"] for e in tuned_penalty])
                        ),
                        "mean_l1_ratio": float(
                            np.mean([e["l1_ratio"] for e in tuned_penalty])
                        ),
                        "n_retunes": len(tuned_penalty),
                        # DEGENERATE-ACTIVE-SET DISCLOSURE (2026-08-07): how
                        # often the selected penalty zeroes every non-intercept
                        # coefficient, i.e. the forecast falls back to the
                        # intercept-only limit (mean of the window target).
                        # A legitimate selection outcome — see the note in
                        # RollingTunedLinear._tune — but it must be VISIBLE,
                        # not inferred, now that the alpha grid reaches a
                        # decade higher.
                        "min_n_active": int(
                            min(e.get("n_active", -1) for e in tuned_penalty)
                        ),
                        "mean_n_active": float(
                            np.mean([e.get("n_active", 0) for e in tuned_penalty])
                        ),
                        "frac_intercept_only": float(
                            np.mean([e.get("n_active", -1) == 0 for e in tuned_penalty])
                        ),
                    }
                    if tuned_penalty
                    else {}
                ),
                "smear": "none (scored locally, §3 contract)",
            }
        ),
    )
    print(
        f"[{arm}] rows [{lo}, {hi}) window={window} cols={F.shape[1]} "
        f"sqrt_mse={result['sqrt_mse']:.6f} dropped={len(dropped_cols)} "
        f"collinear={len(collinear_cols)} masked={len(masked_cols)} -> {args.output_file}"
    )

    if os.environ.get("RESULT_DIR"):
        from hpc_agent.execution.mapreduce.metrics_io import write_metrics

        write_metrics(result)
