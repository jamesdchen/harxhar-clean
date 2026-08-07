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

# Per-block alpha grids for the causally-tuned BLOCK arms (author directive
# 2026-08-06: the block ridges get the same fairness standard as the tuned
# head-to-head). RATIONALE: 3 log-spaced points per block, centered so each
# grid SPANS BOTH prior conventions — backbone: both ladders used 1 (grid
# brackets it); linear exog: user 100 vs doc 3e3 (grid 1e2..1e4 contains
# both); product: user 1e3 vs doc 3e4 (grid 1e3..1e5); transmission: user
# 1e3 vs doc 3e3 (grid 1e2..1e4 contains both). Selection is JOINT (full
# cartesian product per retune) so cross-block penalty trade-offs are seen.
BLOCK_TUNE_GRIDS: dict[str, tuple[float, ...]] = {
    "backbone": (0.1, 1.0, 10.0),
    "exog": (1e2, 1e3, 1e4),
    "product": (1e3, 1e4, 1e5),
    "trans": (1e2, 1e3, 1e4),
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
BLOCK_TUNE_GRIDS["trans_shaped"] = tuple(  # type: ignore[assignment]
    (float(lam0), float(g))
    for lam0 in BLOCK_TUNE_GRIDS["trans"]
    for g in TRANS_SHAPE_GAMMAS
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
BLOCK_TUNE_GRIDS["pcr"] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
# PC-ladder block: one rank-tilted penalty per PC rank, SHARED across that
# rank's ladder rungs (see _pc_ladder_design). Power family only, 12 points;
# the group size (rungs per rank) rides in the descriptor.
BLOCK_TUNE_GRIDS["pc_ladder_tilt"] = tuple(  # type: ignore[assignment]
    (float(lam0), "pcrank", float(g), len(PRODUCT_EXOG_WINDOWS))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS
)

# The same (level, tilt) grid for the EXOGENOUS block — the generalized
# Tikhonov arm (2026-08-07): anisotropic ridge applied directly, with no
# duplicated columns, as the principled form of what the transmission block
# achieves by augmentation. gamma=0 is plain scalar ridge.
BLOCK_TUNE_GRIDS["exog_tilt"] = tuple(  # type: ignore[assignment]
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
BLOCK_TUNE_GRIDS["exog_tilt_step"] = tuple(  # type: ignore[assignment]
    (float(lam0), "power", float(g))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for g in TRANS_SHAPE_GAMMAS
) + tuple(
    (float(lam0), "step", float(k))
    for lam0 in BLOCK_TUNE_GRIDS["exog"]
    for k in TIKHONOV_STEP_KS
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
        elif family == "step":
            # HARD-threshold family: flat at lam0 up to rank K, elevated by
            # STEP_MULTIPLIER beyond it — a finite stand-in for truncation
            # (see STEP_MULTIPLIER). Degenerates to flat when K >= span width.
            pen[s0:s1] = lam0
            k_cut = int(par)
            if k_cut < k_span:
                pen[s0 + k_cut : s1] = lam0 * STEP_MULTIPLIER
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
    — the original campaign construction) or "trailing" (the revival: causal
    trailing standardization + the standard rolling robust scaler; see the
    2026-08-06 ruling notes at the TRANS constants).

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
        if standardization == "trailing":
            gm = pd.DataFrame(G).rolling(trail, min_periods=trail).mean().shift(1)
            gs = pd.DataFrame(G).rolling(trail, min_periods=trail).std().shift(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                G = (G - gm.to_numpy()) / gs.to_numpy()
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
    if standardization == "trailing":
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
        # selection trajectory: (solve index at retune, alpha, l1_ratio) —
        # persisted by the caller as meta.tuned_penalty (2026-08-07 directive)
        self.trace: list[tuple[int, float, float]] = []

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
        self.trace.append((int(self._n_solve), float(self.alpha_), float(self.l1_)))
        self._seed()

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
        {"row": int(lo + s), "alpha": a, "l1_ratio": l1} for s, a, l1 in solver.trace
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
    "b2_lasso": ArmSpec(
        describe="FIXED-penalty lasso on the wide basis, alpha=1e-4 l1_ratio=1.0 "
        "(pinned 2026-08-06; warm Garrigues homotopy, per-bar refit; "
        "enet arm dropped: §5 is ridge vs lasso)",
        kind="tuned",
        grid="lasso_fixed",
    ),
    "b1_ridge_tuned": ArmSpec(
        describe="causally-TUNED ridge control on the wide basis: battery grid "
        "logspace(-2,3,6), re-selected every 250 solves on the "
        "fit/embargo-25/val-tail-125 forward split of the current window",
        kind="tuned",
        grid="ridge_tuned",
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


def _build_block(p: _Panel, block: str, window: int) -> np.ndarray:
    if block == "wide":
        return p.X  # the full all_features basis (backbone + every exog MA)
    if block == "backbone":
        return p.X[:, _backbone_cols(p.names)]
    if block == "exog_all":
        return p.X[:, _exog_all_cols(p.names)]
    if block == "exog_rot":  # frozen-eigenframe rotation (generalized Tikhonov)
        return _exog_tilt_design(p, window)
    if block == "pc_ladder":  # {ma_j(G_i)} — ladder applied to the PC series
        return _pc_ladder_design(p, window)
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
        blk = np.ascontiguousarray(_build_block(p, b, window), dtype=np.float64)
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
    p: _Panel, spec: ArmSpec, window: int
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """UNSCALED block design + column segments for the causally-tuned block
    arms: (F, [(col_start, col_end, alpha_key), ...]). Penalties are applied at
    solve time as diag vectors (see _walk_blocks_tuned), never by pre-scaling."""
    parts: list[np.ndarray] = []
    segments: list[tuple[int, int, str]] = []
    start = 0
    for b, akey in spec.blocks:
        blk = np.ascontiguousarray(_build_block(p, b, window), dtype=np.float64)
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
    tuned_penalty: list[dict[str, Any]] = []
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
        F, a_ref = _build_design(p, spec, window)
        yhat = _walk_ridge(F, p.y, window, lo, hi, alpha=a_ref)
    elif spec.kind == "blocks_tuned":
        arm_keys = [k for _, k in spec.blocks]
        tuned_grids = {k: list(BLOCK_TUNE_GRIDS[k]) for k in arm_keys}
        if spec.grid == "cyclic" and any(
            k.startswith(_BUCKET_PEN_KEY) for k in arm_keys
        ):
            # per-bucket rung: verify the family taxonomy partitions the
            # exogenous columns BEFORE building anything (loud on failure)
            assert_bucket_partition(p.names)
        F, segments = _tuned_blocks_design(p, spec, window)
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
                # tree expert arms: the frozen menu entry (name, params, sha)
                "tree_config": tree_cfg,
                # transmission dig: construction summary (arrays in the npz
                # under trans_ops / trans_eigvals / trans_refresh_rows)
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
