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

# OLS arms run at literal alpha=0 (RollingLeastSquares rank-1 is EXACT OLS at zero
# penalty — the audited incumbent estimator). Bucket designs can contain
# exactly-duplicate availability-indicator columns and panel-constant columns
# (indicator of an always-available source ≡ the intercept): both are dropped
# DETERMINISTICALLY before the walk — bitwise equality over the FULL panel,
# first column in canonical order wins, drops recorded in the output metadata —
# so every chunk of an arm sees the identical design. Remaining WINDOW-level
# degeneracies (availability steps constant within a training window) are
# handled by the window-level identifiability mask in _walk_ols (ruling
# 2026-08-06; the repo's lam2=0 mask precedent): zero-dispersion-in-window
# columns get coefficient 0 and re-enter when the feed varies, masking
# disclosed per chunk. No shrinkage fallback anywhere: a gram still singular
# AFTER dedup + masking fails loudly.

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

_DEGENERATE_SD = 1e-8  # "no dispersion in window" guard (frame/live cols)
_SD_EPS = 1e-12  # standardization epsilon (study convention)

CACHE_DIR_ENV = "UNIFY_CACHE_DIR"  # prep-cache dir override (default "results")


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


def _transmission_block(p: _Panel, window: int, include_scores: bool) -> np.ndarray:
    """(n, 2*TRANS_QPOOL) = [factor scores G | lead-lag Ghat] when
    ``include_scores`` (the _user convention), else (n, TRANS_QPOOL) = Ghat only
    (the _doc convention — the documented construction verbatim).

    Frame: eigenvectors of the correlation of the product-base columns over rows
    [window, 2*window), FROZEN (analysis/map_monitor._frame_and_scores). Scores
    standardized by frozen frame-window stats (causal deviation disclosed above).
    Ghat: analysis/trans_exploit._trans_block — D = antisym(lag-1 cross-corr) of
    the trailing 504 days of G, refreshed quarterly, Ghat_t = G_{t-1} @ D; zero
    until the first D exists (the study zero-fills identically and masks activity
    downstream via the persisted row indices). Ghat causally floored-sd scaled.
    """
    bc = _product_base_cols(p.names)
    Z = np.ascontiguousarray(p.X[:, bc])
    f0, f1 = window, 2 * window
    mu0, sd0 = Z[f0:f1].mean(0), Z[f0:f1].std(0)
    live = sd0 > _DEGENERATE_SD
    sd0 = np.where(live, sd0, 1.0)
    lam, V_l = np.linalg.eigh(
        np.corrcoef(((Z[f0:f1] - mu0) / sd0)[:, live], rowvar=False)
    )
    order = np.argsort(lam)[::-1]
    V = np.zeros((Z.shape[1], len(lam)))
    V[live] = V_l[:, order]
    W = V[:, :TRANS_QPOOL] / sd0[:, None]
    G = (Z - mu0) @ W
    g_mu, g_sd = G[f0:f1].mean(0), G[f0:f1].std(0) + _SD_EPS
    G = (G - g_mu) / g_sd

    n = len(G)
    trail = TRANS_TRAIL_DAYS * PERIODS_PER_DAY
    refresh = TRANS_REFRESH_DAYS * PERIODS_PER_DAY
    lag = TRANS_LAG_BARS
    Ghat = np.zeros_like(G)
    for start in range(f1 + trail, n, refresh):
        a, b = G[start - trail : start - lag], G[start - trail + lag : start]
        az = (a - a.mean(0)) / (a.std(0) + _SD_EPS)
        bz = (b - b.mean(0)) / (b.std(0) + _SD_EPS)
        C = (az.T @ bz) / len(az)
        D = (C - C.T) / 2.0
        end = min(start + refresh, n)
        Ghat[start:end] = G[start - lag : end - lag] @ D
    Ghat = _causal_floored_scale(Ghat, window)
    Ghat[~np.isfinite(Ghat)] = 0.0
    # Composition per the 2026-08-06 ruling: _doc arms mirror the documented
    # construction EXACTLY — Ghat only, the study's design (trans_exploit.py:64-66);
    # _user arms carry [G | Ghat] (40 cols), the paper's own design.
    return np.hstack([G, Ghat]) if include_scores else Ghat


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

    Returns (yhat, trajectory) — trajectory is the paper-relevant selection
    record: [{row: global retune row, alphas: {block_key: alpha}}, ...].
    """
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    p_ = Xs.shape[1]
    keys = [k for _, _, k in segments]
    grids = [BLOCK_TUNE_GRIDS[k] for k in keys]

    def pen_vec(alphas: dict[str, float]) -> np.ndarray:
        pen = np.empty(p_)
        for s0, s1, k in segments:
            pen[s0:s1] = alphas[k]
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
            best: tuple[float, dict[str, float], np.ndarray] | None = None
            for combo in itertools.product(*grids):
                alphas = dict(zip(keys, (float(a) for a in combo)))
                pv = pen_vec(alphas)
                G = Gf.copy()
                G[np.diag_indices_from(G)] += pv
                b = np.linalg.solve(G, cf)
                mse = float(np.mean((Xvc @ b + myf - yv) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, alphas, pv)
            _, sel, pen = best
            traj.append({"row": int(lo + i), "alphas": sel})
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
    """Deterministic exact-collinearity elimination for OLS designs (ruling
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


def _walk_ols(
    F: np.ndarray,
    y: np.ndarray,
    window: int,
    lo: int,
    hi: int,
    col_names: list[str],
    support_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Exact OLS (alpha=0) per-bar walk-forward with WINDOW-LEVEL identifiability
    masking (ruling 2026-08-06; the repo's own lam2=0 identifiability-mask
    precedent, specs/causal_tune_linear.py, applied to the OLS path).

    Per training window, columns with ZERO dispersion in that window (an
    availability step that has not yet gone live, or has exited the window) are
    masked: coefficient 0, excluded from the solve, re-entering the moment the
    feed varies inside the window. ``support_masks`` layers the go-live /
    feed-dead participation rule on top (see :func:`_support_masks` — the
    go-live blowup fix, corrected form). Threshold-free and exact — a column is
    masked iff its last value-change index precedes the window start, computed
    from the RAW slice, never from drift-prone rolled statistics. Exact OLS on
    the surviving columns (the unpenalized intercept absorbs in-window
    constants). No shrinkage anywhere; a gram still singular AFTER masking
    fails loudly. Returns (yhat, mask report: name -> {bars, first, last}
    with global row indices).
    """
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    ns, p = Xs.shape
    # last-change index per (row, col): largest r' <= r with Xs[r'] != Xs[r'-1]
    # (0 if the column never changed up to r). Column j is constant over window
    # rows [w0, t) iff lastchg[t-1, j] <= w0.
    lastchg = np.zeros((ns, p), dtype=np.int32)
    lastchg[1:] = np.where(
        Xs[1:] != Xs[:-1], np.arange(1, ns, dtype=np.int32)[:, None], 0
    )
    np.maximum.accumulate(lastchg, axis=0, out=lastchg)

    solver = RollingLeastSquares(alpha=0.0, fit_intercept=True)
    solver.init_window(Xs[:window], ys[:window])
    out = np.empty(hi - lo, dtype=np.float64)
    mask_rows = np.zeros((hi - lo, p), dtype=bool)
    # Sticky window-collinearity mask, populated LAZILY: only when a solve fails
    # is the CURRENT window rank-diagnosed (one QR), and any column found exactly
    # dependent there stays masked for the rest of the chunk. Lazy because the
    # chunk-first-window QR in _eliminate_exact_collinear already proved the
    # design full-rank at entry; a later failure means the degeneracy formed
    # mid-chunk (or on this machine's float path — the repo's documented CARC
    # precedent), so it must be diagnosed at the failing window itself.
    forced = np.zeros(p, dtype=bool)
    n = float(window)

    def _window_rank_repair(i: int, t: int, free: np.ndarray) -> int:
        """Rank-diagnose the CURRENT window (raw rows, not rolled stats) and
        extend the sticky mask with exactly-dependent columns. Criterion: the
        normal-equations rank tolerance — solving via the gram SQUARES the
        design's conditioning (Golub & Van Loan), so a column is numerically
        dependent for the gram path iff its QR residual ratio is below
        sqrt(max(n_rows, n_cols) * eps_float64) (~7e-6 at n=24000). Machine-
        epsilon-derived, not tuned. Returns the number of newly masked cols."""
        Xw = Xs[i:t][:, free]
        Xc = Xw - Xw.mean(0)
        norms = np.linalg.norm(Xc, axis=0)
        diag = np.abs(np.diag(np.linalg.qr(Xc, mode="r")))
        ne_tol = np.sqrt(max(Xc.shape) * np.finfo(np.float64).eps)
        bad = (norms > 0) & (diag <= ne_tol * norms)
        forced[np.flatnonzero(free)[bad]] = True
        return int(bad.sum())

    any_zero = np.zeros(p, dtype=bool)
    any_pre = np.zeros(p, dtype=bool)
    any_dead = np.zeros(p, dtype=bool)
    # Normal-equations verification tolerance — the SAME machine-derived formula
    # as _window_rank_repair (sqrt(max(n_rows, n_cols) * eps): the gram squares
    # the design's conditioning). Not tuned.
    ne_tol = np.sqrt(max(window, p) * np.finfo(np.float64).eps)
    for i in range(hi - lo):
        t = window + i
        # zero dispersion within window rows [i, t), plus the go-live/feed-dead
        # participation rule; the sticky repair mask joins below (it can grow
        # inside this bar's repair loop)
        zero = lastchg[t - 1] <= i
        base = zero.copy()
        if support_masks is not None:
            pre_i, dead_i = support_masks[0][i], support_masks[1][i]
            base |= pre_i | dead_i
            any_pre |= pre_i
            any_dead |= dead_i
        any_zero |= zero
        gram = solver._Sxx - np.outer(solver._sx, solver._sx) / n
        rhs = solver._Sxy - solver._sx * (solver._sy / n)
        while True:  # solve -> verify -> (repair, retry); bounded: forced only grows
            mask = base | forced
            free = ~mask
            coef = np.zeros(p)
            if not free.any():
                break
            sub = gram[np.ix_(free, free)]
            r = rhs[free]
            failure = ""
            try:
                cf = np.linalg.solve(sub, r)
                # VERIFY the solution against the normal equations (one O(p^2)
                # matvec per bar — the cost of closing the near-singular
                # approach path, not just its exact-singularity endpoint where
                # LU raises). SCALE CHOICE: the residual is compared to
                # ||rhs||_inf, NOT to ||gram||*||coef|| — LU is backward-stable,
                # so the latter ratio stays ~eps even when the FORWARD error has
                # contaminated every coefficient; relative to the right-hand
                # side's own scale, a contaminated solve (residual ~
                # eps*||gram||*||coef|| with ||coef|| exploded by cond*eps)
                # stands out exactly when cond exceeds 1/ne_tol. Same
                # machine-derived tolerance as the rank repair; no new
                # thresholds.
                scale = float(np.max(np.abs(r)))
                if not np.all(np.isfinite(cf)):
                    failure = "non-finite coefficients"
                elif scale > 0.0 and (
                    float(np.max(np.abs(sub @ cf - r))) > ne_tol * scale
                ):
                    failure = "normal-equations residual check failed"
                else:
                    coef[free] = cf
                    break
            except np.linalg.LinAlgError:
                failure = "singular gram"
            if _window_rank_repair(i, t, free) == 0:
                raise RuntimeError(
                    f"OLS solve at global row {lo + i}: {failure}, and the "
                    "window rank diagnosis finds NO near-collinear column to "
                    "mask — a pure float-path artifact or a panel-build "
                    "divergence; no shrinkage fallback by design. Compare the "
                    "panel builds before rerunning."
                )
        mask_rows[i] = mask
        intercept = solver._sy / n - float(solver._sx @ coef) / n
        out[i] = float(Xs[t] @ coef + intercept)
        solver.roll(Xs[t], ys[t], Xs[t - window], ys[t - window])

    masked: dict[str, dict[str, Any]] = {}
    for j in np.flatnonzero(mask_rows.any(axis=0)):
        bars = np.flatnonzero(mask_rows[:, j])
        reasons = []
        if any_zero[j]:
            reasons.append("zero_dispersion")
        if any_pre[j]:
            reasons.append("pre_go_live")
        if any_dead[j]:
            reasons.append("feed_dead")
        if forced[j]:
            reasons.append("sticky_collinear")
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
) -> np.ndarray:
    """Per-bar walk-forward driving the RollingTunedLinear incremental protocol
    (init_window / solve / predict_one / roll), chunk seam as in _walk_ridge."""
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    solver = RollingTunedLinear(grid).init_window(Xs[:window], ys[:window])
    out = np.empty(hi - lo, dtype=np.float64)
    for i in range(hi - lo):
        t = window + i
        solver.solve()
        out[i] = solver.predict_one(Xs[t])
        solver.roll(Xs[t], ys[t], Xs[t - window], ys[t - window])
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
        describe=f"OLS: HAR backbone + '{bucket}' bucket (literal alpha=0)",
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
        describe="OLS on the base-2 HAR ladder of the target (backbone: HAR + "
        "session-edge interactions + calendar). THE reference arm. "
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
    if block.startswith("bucket:"):
        return p.X[:, _bucket_cols(p.names, block.split(":", 1)[1])]
    if block == "product":
        return _frozen_products(p, window)
    if block == "trans_user":  # [G | Ghat] — the paper's own design (ruling 2026-08-06)
        return _transmission_block(p, window, include_scores=True)
    if block == "trans_doc":  # Ghat only — the documented construction verbatim
        return _transmission_block(p, window, include_scores=False)
    raise KeyError(f"unknown block '{block}'")


def _build_design(p: _Panel, spec: ArmSpec, window: int) -> tuple[np.ndarray, float]:
    """(design, solver_alpha). Per-block penalties imposed by column scaling:
    scaling block j by sqrt(a_ref / a_j) under global ridge a_ref is EXACTLY the
    per-block penalty (minimal_model.py:297-299); intercept stays unpenalized."""
    if spec.kind == "tuned":
        return p.X, 0.0
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


def _ols_design(p: _Panel, spec: ArmSpec) -> tuple[np.ndarray, list[str], list[str]]:
    """OLS design by named column selection + deterministic dedup (see
    :func:`_dedup_ols_design`). Returns (F, kept_names, dropped_names)."""
    cols: list[int] = []
    for b, _ in spec.blocks:
        if b == "backbone":
            cols += list(_backbone_cols(p.names))
        elif b.startswith("bucket:"):
            cols += list(_bucket_cols(p.names, b.split(":", 1)[1]))
        else:
            raise KeyError(f"OLS arms only compose backbone/bucket blocks, got '{b}'")
    names_design = [p.names[j] for j in cols]
    return _dedup_ols_design(p.X[:, np.asarray(cols, dtype=np.int64)], names_design)


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
    if spec.kind == "ols":
        F, kept_names, dropped_cols = _ols_design(p, spec)
        F, kept_names, collinear_cols, ladder_drops = _eliminate_exact_collinear(
            F, kept_names, window, lo
        )
        support = _support_masks(p, kept_names, window, lo, hi)
        yhat, masked_cols = _walk_ols(
            F, p.y, window, lo, hi, kept_names, support_masks=support
        )
    elif spec.kind == "blocks":
        F, a_ref = _build_design(p, spec, window)
        yhat = _walk_ridge(F, p.y, window, lo, hi, alpha=a_ref)
    elif spec.kind == "blocks_tuned":
        F, segments = _tuned_blocks_design(p, spec, window)
        yhat, tuned_alphas = _walk_blocks_tuned(F, segments, p.y, window, lo, hi)
    else:
        F, _ = _build_design(p, spec, window)
        yhat = _walk_tuned(F, p.y, window, lo, hi, ESTIMATOR_GRIDS[spec.grid])

    y_fit = p.y[lo:hi]
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
    np.savez_compressed(
        args.output_file,
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
