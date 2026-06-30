"""Amortized full-OOS residualized (Ridge->tree) tuning scorer.

Naive ``MultiStage(Residualizer(Ridge) + tree)`` refits sklearn Ridge EVERY bar
(231k x / trial; ~825s/eval even for the small implied_vol/250d cell) -> infeasible
to tune at scale. But the Ridge base is CONFIG-INDEPENDENT within a cell, so we
precompute it ONCE and each trial then fits only trees on ``y - ridge``.

Exact-amortization (reproduces MultiStage bit-for-bit, see ``selfcheck``):
  * every-bar Ridge OOS preds  -> incremental RollingLeastSquares (fast + exact)
  * Ridge coef/intercept at each tree-refit cadence point t_r (sklearn Ridge on the
    window ending at t_r == the every-bar Ridge AT t_r) -> used to form the in-sample
    residual ``y_train - X_train@beta_tr`` the tree trains on at that refit.
  final[t] = ridge_oos[t] + tree_{t_r}.predict(X[t])     (tree from the block's refit)

modes:
  prep      MODEL BUCKET TWD ALPHA TREE_REFIT        -> cache to results/resid_prep/<cell>/
  trial     CELL ARM [OUTNAME]   (TREE_CFG env json) -> load cache, score one config
  selfcheck MODEL BUCKET TWD ALPHA TREE_REFIT NSEG   -> assert amortized == naive MultiStage
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, os.getcwd())
from src.evaluation.metrics import apply_duan_smearing
from src.features.transforms.scaling import rolling_robust_scale
from src.features.transforms.target import PERIODS_PER_DAY
from src.models.ridge import fit_predict_ridge

CACHE_ROOT = "results/resid_prep"
PCA_REFIT = 2000  # cadence for the rolling-PCA basis (stable -> coarse is fine + cheap)

# Close/after-hours regime window [CLOSE_LO, CLOSE_HI] in clock hours. Default 16-19 = the -0.21
# HAR-persistence reversal lobe (peak 17:00) from the mechanism analysis. Overridable via env so the
# gate window can be SENSITIVITY-SWEPT (is the edge robust to 16-19 vs 16-20 / 15-19 / 17-19?) -- the
# one piece of un-tested structural judgment in the pipeline. ALL close gates route through _close_mask
# so a swept window shifts the feature gates AND the regime-EBM train/predict gate together.
CLOSE_LO = int(os.environ.get("CLOSE_LO", "16"))
CLOSE_HI = int(os.environ.get("CLOSE_HI", "19"))


def _close_mask(hour):
    """Boolean close/AH gate: CLOSE_LO <= hour <= CLOSE_HI (default 16-19, env-overridable)."""
    return (hour >= CLOSE_LO) & (hour <= CLOSE_HI)


def cell_id(model, bucket, twd, alpha, refit, pipe="slim", base_kind="ridge"):
    tag = (
        f"a{alpha:g}" if base_kind == "ridge" else base_kind
    )  # ridge: "a100" (back-compat); enet: "enet"
    # Non-default close window -> distinct cell (the gate-window sweep), so swept windows do not
    # collide with / clobber the deployed 16-19 cells (default -> empty suffix, names unchanged).
    cw = "" if (CLOSE_LO, CLOSE_HI) == (16, 19) else f"_cw{CLOSE_LO}-{CLOSE_HI}"
    return f"{model}_{bucket}_tw{twd}_{tag}_rf{refit}_{pipe}{cw}"


def _split_cols(feats):
    """Partition feature columns into (HAR, exog, indicator) index lists. PCA compresses only
    the dense-weak EXOG block; HAR (strong linear core) and the availability indicators pass raw."""
    har = [i for i, f in enumerate(feats) if "har_ma" in f]
    ind = [i for i, f in enumerate(feats) if "_avail" in f or "_active" in f]
    skip = set(har) | set(ind)
    exog = [i for i in range(len(feats)) if i not in skip]
    return har, exog, ind


# Clock-anchored session-transition gates for the OPEN/CLOSE x HAR regime interaction (base_kind=
# "enetreg"). The auction / session-transition microstructure localizes the HAR vol-persistence
# sign-flip at the 09:30 ET OPEN (hour==9) and the 16:00 CLOSE + after-hours, hours 16-19 (capped
# below 20 -- the -0.21 reversal lobe peaks at 17:00, dead overnight bars excluded) -- per tests123.py.
# These are ACTUAL clock hours, NOT hour-quantile buckets (bucket-0-of-6 is overnight, not the open).
# Crossing them with the 6 HAR columns lets the linear enet base ABSORB the regime the residualized
# EBM otherwise has to rediscover, so the downstream EBM digs into the structure that remains.
_HAR_COLS = (
    "har_ma_1",
    "har_ma_5",
    "har_ma_25",
    "har_ma_125",
    "har_ma_625",
    "har_ma_3125",
)


def _regime_interactions(Xs, feats, which=("open", "close")):
    """HAR x {open, close} clock-anchored interaction columns. Returns (INT[N,k], names[k]).
    open = (hour==9), the 09:30 ET opening bar; close = (16<=hour<=19), the 16:00 close auction +
    after-hours (capped below 20 to exclude dead overnight bars; AH trading ends ~20:00 ET).
    Constant products (std<=1e-9, e.g. a gate empty in this cache) drop out.
    ``which`` selects which gates to emit (preserving open-before-close order); the default emits
    BOTH (bit-identical to the original enetreg/enetreg2 output). which=("open",) drops the close
    half -- the redundancy probe vs the cumrv x close column."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    all_gates = (
        ("open", (hour == 9).astype(np.float64)),
        ("close", (_close_mask(hour)).astype(np.float64)),
    )
    gates = tuple((gn, g) for gn, g in all_gates if gn in which)
    cols, names = [], []
    for hn in _HAR_COLS:
        if hn not in feats:
            continue
        h = Xs[:, feats.index(hn)].astype(np.float64)
        for gname, g in gates:
            col = h * g
            if col.min() < col.max():
                cols.append(col)
                names.append(f"{hn}_x_{gname}")
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _cumrv_close(Xs, feats):
    """Intraday volatility accumulated TODAY (cumsum of the 1-bar HAR within the trading day, days
    segmented where the hour drops) gated to the close + after-hours (hour 16-19). The strongest
    single linear edge feature (-0.00122 OOS QLIKE, sign-stable): after a high-vol day the close/AH
    residual vol mean-reverts BELOW the HAR base. Causal -- har_ma_1 is itself shift(1)."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    cumrv = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    close = (_close_mask(hour)).astype(np.float64)
    return (cumrv * close)[:, None], ["cumrv_x_close"]


def _cumrv_open(Xs, feats):
    """TRUE cross-boundary OVERNIGHT accumulated vol feeding the OPEN -- the open-side analog of
    _cumrv_close. The overnight window is the contiguous bar-run from AFTER the close/AH (hour>=20)
    THROUGH the midnight wrap to the next open (hour<=9): hours {20,21,22,23, 0,1,...,9} = AH-end +
    overnight + pre-market (ET: AH closes 20:00, pre-market 04:00, open 09:30). A new night SEGMENT
    starts on each ENTRY into that window (hour 19->20), so it deliberately crosses the midnight
    day-wrap (NOT the midnight-anchored day_id used by _cumrv_close). cumsum of the 1-bar HAR within
    each night-run (non-night bars masked to 0 so the daytime gap doesn't accumulate), gated to the
    OPEN (hour==9, the run's last bars): at 09:30 the feature = sum of har_ma_1 over the prior evening's
    after-hours + overnight + pre-market = the overnight vol PATH (diurnally-adjusted -> overnight
    ABNORMALITY, the proxy form that beat physical cumrv_real at the close). Disjoint support from
    cumrv_x_close (hour 16-19) -> no collinearity. Tests whether the session-edge mean-reversion the
    close shows also fires at the open -- the intraday-regime sign-flip peaks at BOTH 09:30 and 16-19.
    Causal -- har_ma_1 is itself shift(1)."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    night = (hour >= 20) | (
        hour <= 9
    )  # overnight window feeding the open (AH-end -> pre-market)
    entry = np.concatenate(
        [[night[0]], night[1:] & ~night[:-1]]
    )  # new night-run at each 19->20 entry
    night_id = np.cumsum(entry.astype(int))
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    cumrv = pd.Series(np.where(night, har1, 0.0)).groupby(night_id).cumsum().to_numpy()
    open_ = (hour == 9).astype(np.float64)
    return (cumrv * open_)[:, None], ["cumrv_x_open"]


# Canonical banded HAR basis (base_kind enetreg2_canon): differenced multi-resolution decomposition of
# the 6 nested sqrt-magnitude HAR MAs into ~ORTHOGONAL timescale bands. band_k = har_ma_k - har_ma_{k-1}
# (band_1 = har_ma_1). A LINEAR, span-preserving reparameterization (OLS fit identical; enet ~identical,
# penalty geometry aside) whose POINT is legibility: the nested MAs are heavily collinear (nested windows)
# so their coefficients/signs are uninterpretable; the band coefficients are each timescale's MARGINAL
# vol-persistence contribution. Magnitude scale, NO rank (per the cumrv result: vol path keeps magnitude).
_BAND_DEFS = (  # (band_name, this_ma, prev_ma_or_None)
    ("band_1", "har_ma_1", None),
    ("band_5m1", "har_ma_5", "har_ma_1"),
    ("band_25m5", "har_ma_25", "har_ma_5"),
    ("band_125m25", "har_ma_125", "har_ma_25"),
    ("band_625m125", "har_ma_625", "har_ma_125"),
    ("band_3125m625", "har_ma_3125", "har_ma_625"),
)


def _har_bands(Xs, feats):
    """Differenced HAR bands band_k = har_ma_k - har_ma_{k-1} (band_1 = har_ma_1). Returns (BANDS[N,k],
    names[k]); a band whose MA is absent from this cache is skipped."""
    cols, names = [], []
    for nm, this_ma, prev_ma in _BAND_DEFS:
        if this_ma not in feats:
            continue
        v = Xs[:, feats.index(this_ma)].astype(np.float64)
        if prev_ma is not None and prev_ma in feats:
            v = v - Xs[:, feats.index(prev_ma)].astype(np.float64)
        cols.append(v)
        names.append(nm)
    return np.ascontiguousarray(np.column_stack(cols)), names


def _band_regime_interactions(bands, band_names, Xs, feats):
    """band_k x {open(hour==9), close(16<=hour<=19)} -- the canonical (banded) _regime_interactions.
    Constant products (a gate empty in this cache) drop out. Preserves the span of HAR x {open,close}."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    gates = (
        ("open", (hour == 9).astype(np.float64)),
        ("close", (_close_mask(hour)).astype(np.float64)),
    )
    cols, names = [], []
    for j, bn in enumerate(band_names):
        for gname, g in gates:
            col = bands[:, j] * g
            if col.min() < col.max():
                cols.append(col)
                names.append(f"{bn}_x_{gname}")
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _har5_spline_close(Xs, feats, train_win):
    """Clean monotone SPLINE on the sqrt-MAGNITUDE har_ma_5, gated close: relu(har_ma_5 - tau) x close at
    two knots tau = q40/q75 of the CAUSAL reference window [0, train_win). The legible, magnitude-scale
    version of what har5rank captured opaquely (rank-nonlinearity stacked on the sqrt-MA): explicit
    piecewise-linear curvature, no rank. Tests whether the har5rank -0.00021 reappears as clean spline
    curvature on the magnitude feature. Knot LOCATION from the reference window is causal; the linear
    har_ma_5 slope is already spanned by the bands. Off-hours are exact zeros (gate AFTER the relu)."""
    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    close = (_close_mask(hour)).astype(np.float64)
    taus = np.percentile(h5[:train_win], [40, 75])
    cols, names = [], []
    for q, tau in zip((40, 75), taus):
        col = np.maximum(h5 - float(tau), 0.0) * close
        if col.min() < col.max():
            cols.append(col)
            names.append("har5_relu_q%d_x_close" % q)
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


_HAR5_SPLINE_QS = (10, 20, 30, 40, 50, 60, 70, 80)


def _har5_spline_dense_close(Xs, feats, train_win):
    """DENSE (decile-knot) relu spline on the sqrt-MAGNITUDE har_ma_5, gated close: relu(har_ma_5 - tau) x
    close for tau at q10..q80 of the CAUSAL reference window [0, train_win). The high-resolution STATIC
    curvature basis for the direct test of whether har5rank's -0.00021 is reproducible by a static magnitude
    transform (vs har5rank's ROLLING per-slot relative rank). enetreg2_har5spl (isolated, on enetreg2) and
    enetreg2_har5rank_spl (this + har5rank, the STACKING test): if the dense spline ~= -0.00021 and har5rank
    does NOT stack on top -> the gain is static curvature; if the spline falls short and har5rank stacks ->
    the rolling/relative standardization carries the unique signal."""
    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    close = (_close_mask(hour)).astype(np.float64)
    taus = np.percentile(h5[:train_win], list(_HAR5_SPLINE_QS))
    cols, names = [], []
    for q, tau in zip(_HAR5_SPLINE_QS, taus):
        col = np.maximum(h5 - float(tau), 0.0) * close
        if col.min() < col.max():
            cols.append(col)
            names.append("har5_relu_q%d_x_close" % q)
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


# The Hero-B regime EBM (trained on the h16-19 close/AH residual LEFTOVER) flagged three MICROSTRUCTURE
# main effects as its top features: adj_sumvolume_ma_1 (down), adj_numobs_ma_5 (up),
# adj_voldemand_all_open_only_ma_125 (down). HAR x close is ALREADY in enetreg2; these three are NOT.
# Crossing each with the SAME close gate (16<=hour<=19) the EBM was trained on lets the LINEAR enet base
# absorb the microstructure-x-close effect the EBM otherwise has to rediscover -- the distillation test:
# if base-alone improves toward the regime gain, the EBM's edge is LINEARLY distillable; if it doesn't
# move, the regime contribution is genuinely nonlinear/interaction-based.
_DISTILL_COLS = (
    "adj_sumvolume_ma_1",
    "adj_numobs_ma_5",
    "adj_voldemand_all_open_only_ma_125",
)


def _distill_interactions(Xs, feats):
    """close-gated (1[16<=hour<=19]) interactions for the 3 EBM-flagged microstructure cache columns
    (_DISTILL_COLS): col * close for each. Columns are read from the slim cache via feats.index; a
    column absent from this cache is skipped + logged (not every bucket carries every series). Constant
    products (std<=1e-9, e.g. a column dead off the close hours) drop out. Returns (INT[N,k], names[k])
    with names like ``adj_sumvolume_ma_1_x_close``."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    close = (_close_mask(hour)).astype(np.float64)
    cols, names = [], []
    for cn in _DISTILL_COLS:
        if cn not in feats:
            print("DISTILL skip missing col %s" % cn, flush=True)
            continue
        col = Xs[:, feats.index(cn)].astype(np.float64) * close
        if col.min() < col.max():
            cols.append(col)
            names.append("%s_x_close" % cn)
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _har_sq_close(Xs, feats):
    """NONLINEAR aggressive-slope close term: SIGN-PRESERVING ``har_ma_5 * |har_ma_5| * close``. The
    Hero-B EBM showed har_ma_5 as a CONVEX, accelerating close-damping (the damp steepens with vol
    level); a plain har_ma_5 x close column (already in enetreg2) is linear in har, so it cannot capture
    that curvature. The signed square (x*|x|, NOT x**2 -- a constant scale would be a no-op direction-wise
    in the enet) gives the enet a quadratic basis for the close-damping. Gate to close (16<=hour<=19)
    AFTER the square so off-hours are exact zeros. Name ``har_ma_5_sq_x_close``."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    close = (_close_mask(hour)).astype(np.float64)
    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    return ((h5 * np.abs(h5)) * close)[:, None], ["har_ma_5_sq_x_close"]


def _cumrv_ratio_close(Xs, feats):
    """Self-anchored intraday-vol RATIO gated to the close/AH: (cumrv_today / (har_ma_25 + 1e-6)) x close.
    cumrv_today is the SAME proxy cumsum as _cumrv_close (cumsum of rank-gauss har_ma_1 within the trading
    day, days segmented where the hour drops); har_ma_25 is the regime-level HAR column. ONE self-normalized
    column replacing the additive cumrv x close pair -- tests whether a clean ratio recovers the pair."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    cumrv = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    har25 = Xs[:, feats.index("har_ma_25")].astype(np.float64)
    close = (_close_mask(hour)).astype(np.float64)
    ratio = np.divide(cumrv, har25, out=np.zeros_like(cumrv), where=har25 > 0)
    return (ratio * close)[:, None], ["cumrv_ratio_x_close"]


def _floored_robust_scale(cols, train_win):
    """Causal robust-scale of PHYSICAL column(s) to an ~N(0,1)-comparable scale so the enet L1/L2
    penalty sees them on a par with the rank-gauss slim columns: (x - rolling_median) /
    max(rolling_IQR, ref_iqr) over train_win (src.features.transforms.scaling.rolling_robust_scale).
    The IQR floor ref_iqr is the IQR of the INITIAL [0, train_win) reference window -- a CAUSAL,
    real-valued scale, NOT a constant 1.0/1e-12 (a constant floor divides an incoming -- different
    regime -- point by a degenerate collapsed scale; that was a real divide-by-degenerate bug).
    cols is (N,) or (N,k); returns the same shape. A constant column (ref==0) falls through to the
    internal degenerate handling -> 0, and is then dropped by the do_prep constant-drop."""
    X = np.asarray(cols, dtype=np.float64)
    one_d = X.ndim == 1
    if one_d:
        X = X[:, None]
    ref = np.percentile(X[:train_win], 75, axis=0) - np.percentile(
        X[:train_win], 25, axis=0
    )
    ref_iqr = np.ascontiguousarray(
        np.broadcast_to(np.asarray(ref, dtype=np.float64), X.shape)
    )
    out = rolling_robust_scale(X, train_win, ref_iqr=ref_iqr)
    return out[:, 0] if one_d else out


def _cumrv_real_close(Xs, feats, train_win):
    """PHYSICAL post-diurnal intraday cumulative RV (results/intraday_feats/cumrv_real.npy: RV
    diurnally adjusted by a per-slot rolling mean, cumsum within the true calendar day, causal shift)
    robust-scaled then gated to the close/AH (hour 16-19) -- the physical replacement for the
    rank-space proxy _cumrv_close. cumrv_real is cache-aligned to Xs by construction (same 242934-row
    ordering). The UNGATED series is scaled FIRST (its median/IQR are well-defined); scaling the
    zero-inflated GATED column would corrupt the robust stats. THEN gate."""
    cumrv = np.asarray(
        np.load("results/intraday_feats/cumrv_real.npy"), dtype=np.float64
    )
    assert len(cumrv) == len(Xs), "cumrv_real len %d != Xs len %d" % (
        len(cumrv),
        len(Xs),
    )
    scaled = _floored_robust_scale(cumrv, train_win)
    hour = Xs[:, feats.index("hour")]
    close = (_close_mask(hour)).astype(np.float64)
    return (scaled * close)[:, None], ["cumrv_real_x_close"]


def _cumrv_realsqrt_close(Xs, feats, train_win):
    """ISOLATES THE SQRT. Identical to _cumrv_real_close EXCEPT the physical per-bar diurnal-adj RV is
    SQRT'd BEFORE the within-day cumsum (results/intraday_feats/cumrv_realsqrt.npy: cumsum of
    sqrt(diurnal-adj RV) on the VOL scale, same calendar-day seg + causal shift as cumrv_real), then
    the SAME floored robust-scale + close gate. The missing control for the cumrv edge: enetreg2_real
    (cumsum of raw VARIANCE) = 0.12436; the rank-space PROXY (cumsum of sqrt-vol har_ma_1) = 0.12314.
    If this lands near 0.12314 -> the edge IS the sqrt variance-stabilization (cap-mixture refusing to
    let the tail saturate the accumulated signal); if it stays ~0.124 -> the edge is the slim har_ma_1
    construction (diurnal-rank space), not sqrt per se."""
    cumrv = np.asarray(
        np.load("results/intraday_feats/cumrv_realsqrt.npy"), dtype=np.float64
    )
    assert len(cumrv) == len(Xs), "cumrv_realsqrt len %d != Xs len %d" % (
        len(cumrv),
        len(Xs),
    )
    scaled = _floored_robust_scale(cumrv, train_win)
    hour = Xs[:, feats.index("hour")]
    close = (_close_mask(hour)).astype(np.float64)
    return (scaled * close)[:, None], ["cumrv_realsqrt_x_close"]


def _calfeats(Xs, train_win):
    """The 7 OPEX/calendar features (results/date_reconstruct/calfeats.npy, names in names.json):
    binaries is_opex/is_quad_witch/is_eom/is_eoq and the normalized day-of-month dom_norm pass RAW
    (already O(1) scale); the two COUNT columns days_to_opex/days_since_opex get the same floored
    robust-scaling as the physical cumrv so the enet penalty sees them on an N(0,1)-comparable scale.
    Cache-aligned to Xs (same 242934 rows). Names prefixed cal_ for ebm_interpret labeling."""
    cal = np.asarray(np.load("results/date_reconstruct/calfeats.npy"), dtype=np.float64)
    names = json.load(open("results/date_reconstruct/names.json"))
    assert len(cal) == len(Xs), "calfeats len %d != Xs len %d" % (len(cal), len(Xs))
    assert cal.shape[1] == len(names), "calfeats width %d != names %d" % (
        cal.shape[1],
        len(names),
    )
    count_set = {"days_to_opex", "days_since_opex"}
    cols, out_names = [], []
    for j, nm in enumerate(names):
        c = (
            _floored_robust_scale(cal[:, j], train_win)
            if nm in count_set
            else cal[:, j]
        )
        cols.append(np.asarray(c, dtype=np.float64))
        out_names.append("cal_%s" % nm)
    return np.ascontiguousarray(np.column_stack(cols)), out_names


def _causal_orth_resid(target, P, train_win, refit):
    """Causal rolling-OLS residual: target minus its projection on predictors P. Per cadence block
    [t_r, t_{r+1}) the OLS beta is fit on the TRAILING window [t_r-train_win, t_r) (np.linalg.lstsq,
    SVD min-norm -> rank-deficient-safe), so the block's residual uses only past data -- the same
    cadence as the enet refits (pass the prep's refit). The warmup [0, train_win) (never scored OOS,
    but the first enet refit trains on it) is residualized with the FIRST block's beta -- an in-sample
    transform of that window computed only from that window (mirrors rolling_robust_scale's warmup)."""
    target = np.asarray(target, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    n = len(target)
    out = np.array(target, copy=True)
    if P.shape[1] == 0:
        return out
    starts = list(range(train_win, n, refit))
    for i, t_r in enumerate(starts):
        beta = np.linalg.lstsq(
            P[t_r - train_win : t_r], target[t_r - train_win : t_r], rcond=None
        )[0]
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        out[t_r:t_end] = target[t_r:t_end] - P[t_r:t_end] @ beta
        if i == 0:
            out[:train_win] = target[:train_win] - P[:train_win] @ beta
    return out


def _cumrv_realorth_close(Xs, feats, train_win, refit):
    """Like _cumrv_real_close but the (scaled, UNGATED) physical cumrv is first ORTHOGONALIZED
    against the 6 HAR x close interaction columns via a CAUSAL rolling OLS (_causal_orth_resid) before
    gating -- the residual cumrv is the part HAR x close cannot explain (its pure marginal signal).
    The HAR x close columns are zero off the close hours, so the OLS beta is driven purely by close-
    hour rows; gating the residual to close then yields the marginal column. refit matches the prep
    cadence so the orthogonalization windows align with the enet refits."""
    cumrv = np.asarray(
        np.load("results/intraday_feats/cumrv_real.npy"), dtype=np.float64
    )
    assert len(cumrv) == len(Xs), "cumrv_real len %d != Xs len %d" % (
        len(cumrv),
        len(Xs),
    )
    scaled = _floored_robust_scale(cumrv, train_win)
    HARclose, _ = _regime_interactions(
        Xs, feats, which=("close",)
    )  # (N, k<=6) HAR x close
    resid = _causal_orth_resid(scaled, HARclose, train_win, refit)
    hour = Xs[:, feats.index("hour")]
    close = (_close_mask(hour)).astype(np.float64)
    return (resid * close)[:, None], ["cumrv_realorth_x_close"]


def _cumrv_real_rank_close(Xs, feats, train_win):
    """PHYSICAL post-diurnal intraday cumrv (results/intraday_feats/cumrv_real.npy) PER-SLOT causal
    rolling rank-Gauss normalized (NOT robust-scaled) then gated to the close/AH (hour 16-19) -- the
    rank-space sibling of _cumrv_real_close. The robust-scaled physical column (enetreg2_real) lost the
    rank-space PROXY's edge (enetreg2); this tests whether the cumrv signal lives in RANK-GAUSS space
    (the slim features AND the proxy's har_ma_1 are all PER-SLOT rank-Gauss) and robust-scaling put
    cumrv_real in an incompatible space. cumrv_real accumulates THROUGH the day (a strong, near-monotone
    time-of-day level), so a GLOBAL rank would just encode clock position; we rank PER TIME-OF-DAY SLOT
    (today's value vs other days' SAME-slot values = the abnormality), the spirit of the slim per-slot
    pipe (target.diurnal_rank). Slot resolution = HOUR: the cache exposes only the 'hour' column (24
    slots) and results/date_reconstruct/dates.npy is date-only (its time component is zeroed -- it carries
    the calendar date for the OPEX/calfeats, not the 30-min bar), so the finer 48-slot time-of-day is not
    recoverable from the available cache artifacts. Hour-resolution still strips the DOMINANT across-hour
    accumulation level (the residual within-hour :00/:30 offset is second order). Per-slot rolling
    rank-Gauss look-back = train_win//PERIODS_PER_DAY observations (causal + division-free). Gate to
    close AFTER the rank so off-hours are exact zeros."""
    from src.features.transforms.rank_gauss import rolling_rank_gauss

    cumrv = np.asarray(
        np.load("results/intraday_feats/cumrv_real.npy"), dtype=np.float64
    )
    assert len(cumrv) == len(Xs), "cumrv_real len %d != Xs len %d" % (
        len(cumrv),
        len(Xs),
    )
    hour = Xs[:, feats.index("hour")]
    slot = np.rint(hour).astype(
        np.int64
    )  # per-time-of-day-HOUR slot (the cache's time resolution)
    win_slot = max(2, train_win // PERIODS_PER_DAY)  # per-slot rank look-back (obs)
    ranked = np.zeros(len(cumrv), dtype=np.float64)
    for s in np.unique(slot):
        idx = np.where(slot == s)[
            0
        ]  # cache rows are chronological -> this slot's series in time order
        ranked[idx] = rolling_rank_gauss(cumrv[idx][:, None], win_slot)[:, 0]
    close = (_close_mask(hour)).astype(np.float64)
    return (ranked * close)[:, None], ["cumrv_realrank_x_close"]


def _har5rank_close(Xs, feats, train_win):
    """RANK-SPACE encoding of the close HAR-damping shape: causal PER-SLOT (hour) rolling rank-Gauss of
    har_ma_5, gated to close (hour 16-19). The Hero-B EBM's har_ma_5_x_close main effect is monotone-down
    (Spearman -0.99) but CURVED in raw sqrt-vol space (saturates at high vol). Rank-Gauss is the natural
    monotone flattening for a positive right-skewed vol variable -- the empirical CDF = a uniform mixture
    of step-thresholds = the knot-free threshold (vs the existing sqrt-space har_ma_5_x_close in enetreg2,
    which is one fixed monotone basis). Added ALONGSIDE the base's sqrt-space column, so this tests the
    INCREMENT from re-encoding the same shape in rank space. Per-slot (today vs same-HOUR history) and
    look-back train_win//PERIODS_PER_DAY mirror _cumrv_real_rank_close; causal; gate AFTER the rank so
    off-hours are exact zeros. CAVEAT: Spearman -0.99 = monotone, NOT linear-in-rank (the shape still
    curves vs percentile), so a single rank column captures the trend, not the residual curvature."""
    from src.features.transforms.rank_gauss import rolling_rank_gauss

    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    hour = Xs[:, feats.index("hour")]
    slot = np.rint(hour).astype(
        np.int64
    )  # per-time-of-day-HOUR slot (the cache's time resolution)
    win_slot = max(2, train_win // PERIODS_PER_DAY)  # per-slot rank look-back (obs)
    ranked = np.zeros(len(h5), dtype=np.float64)
    for s in np.unique(slot):
        idx = np.where(slot == s)[
            0
        ]  # chronological -> this slot's series in time order
        ranked[idx] = rolling_rank_gauss(h5[idx][:, None], win_slot)[:, 0]
    close = (_close_mask(hour)).astype(np.float64)
    return (ranked * close)[:, None], ["har5rank_x_close"]


def _rel_slot(Xs, feats, fine):
    """Per-time-of-day slot id. fine=False -> 24 HOUR slots; fine=True -> 48 slots (hour split into the
    :00 vs :30 bar by within-(day,hour) order, day = hour-wrap segmentation). The 48-slot id IS recoverable
    from the cache's row order + hour column (each hour appears exactly twice per day, in time order)."""
    import pandas as pd

    hour = np.rint(Xs[:, feats.index("hour")]).astype(np.int64)
    if not fine:
        return hour
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    sub = (
        pd.DataFrame({"d": day_id, "h": hour}).groupby(["d", "h"]).cumcount().to_numpy()
    )
    return hour * 2 + np.clip(sub, 0, 1)


def _har_rel_innov(Xs, feats, train_win, mode="rankgauss", fine=False):
    """ROLLING per-slot vol innovations for k in {1,5,25,125}, UNGATED -- har5rank's rolling/relative component
    (today's vol vs its recent same-slot history), the one signal a TREE cannot reconstruct from level features.
    ``mode`` selects the per-slot rolling normalization (all causal, shift(1) to exclude the current obs):
      rankgauss -- rolling rank-Gauss (bounded ~N(0,1); the deployed `enetreg2_rel` form, 0.12046 best)
      robustz   -- (v - rolling_median)/rolling_IQR, CLIPPED to +-5 (robust standardized surprise)
      ratio     -- v / rolling_median, CLIPPED to [0,10] (interpretable 'x times normal vol')
    ``fine`` -> 48-slot (vs 24 HOUR). Window = train_win//PERIODS_PER_DAY obs. har_ma_k is itself shift(1)."""
    from src.features.transforms.rank_gauss import rolling_rank_gauss
    import pandas as pd

    slot = _rel_slot(Xs, feats, fine)
    win = max(2, train_win // PERIODS_PER_DAY)
    # robustz/ratio min-obs = rolling_rank_gauss's OWN warmup policy (max(20, win//20)), NOT 5:
    # a rolling IQR/median over ~6 obs COLLAPSES (divide-by-degenerate -> z~57 on smooth har_ma_125,
    # the documented scaling bug). Matching rank-gauss's warmup makes the robustz-vs-rankgauss
    # comparison fair AND removes the collapse; warmup rows (< mp obs) emit 0 (no innovation yet).
    mp = max(20, win // 20)
    tag = {"rankgauss": "zr", "robustz": "rz", "ratio": "rr"}[mode] + (
        "f" if fine else ""
    )
    cols, names = [], []
    for hn in ("har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125"):
        if hn not in feats:
            continue
        v = Xs[:, feats.index(hn)].astype(np.float64)
        z = np.zeros(len(v), dtype=np.float64)
        for s in np.unique(slot):
            idx = np.where(slot == s)[0]
            if mode == "rankgauss":
                z[idx] = rolling_rank_gauss(v[idx][:, None], win)[:, 0]
            else:
                ser = pd.Series(v[idx])
                med = ser.rolling(win, min_periods=mp).median().shift(1)
                if mode == "robustz":
                    q = ser.rolling(win, min_periods=mp).quantile
                    iqr = (q(0.75) - q(0.25)).shift(1)
                    zz = (ser - med) / iqr.replace(0.0, np.nan)
                    z[idx] = np.nan_to_num(
                        zz.to_numpy(), nan=0.0
                    )  # NO clip (was +-5): the clip
                    # confounded the rejection of robustz vs rank-gauss -- test it unclipped
                else:  # ratio
                    rr = ser / med.replace(0.0, np.nan)
                    z[idx] = np.nan_to_num(
                        rr.to_numpy(), nan=1.0
                    )  # NO clip (was [0,10])
        cols.append(z)
        names.append("%s_%s" % (tag, hn))
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


# EBM-top microstructure exog + implied vol, for the LONG-window rolling-relative EXOG innovations
# (har5rank-x-close analog for exog). RAW levels read from the covid_imp cache (diurnal-adj magnitude,
# pre-rank); the slim cache's diurnal_rank is only a SHORT (~20-obs) rolling rank, so a long-window rank
# is a different history horizon (regime vs short surprise) -> new-to-tree.
_EXOG_REL_COLS = (
    "adj_sumvolume_ma_1",
    "adj_numobs_ma_5",
    "adj_voldemand_all_open_only_ma_125",
    "adj_effspread_vwstock_ma_5",
    "adj_sumautocov_ma_1",
    "adj_vix_ma_25",
)


def _exog_rel_innov(Xs, feats, train_win, bucket):
    """LONG-window rolling per-slot rank-Gauss of key RAW exog (covid_imp levels), gated to close -- the
    har5rank-x-close analog for EXOG. Tests whether a regime-relative (long-horizon) view of the EBM's top
    microstructure features + implied vol carries signal the tree can't reconstruct (the cache only has a
    short ~20-obs exog rank). Window = train_win//PERIODS_PER_DAY; causal; rank-Gauss (the tree-tied encoding
    from the har sweep). Gate to close (16-19) AFTER the rank, off-hours exact zeros."""
    from src.features.transforms.rank_gauss import rolling_rank_gauss

    raw = np.load(f"results/covid_imp/{bucket}/X_imp.npy", mmap_mode="r")
    hour = Xs[:, feats.index("hour")]
    slot = np.rint(hour).astype(np.int64)
    close = (_close_mask(hour)).astype(np.float64)
    win = max(2, train_win // PERIODS_PER_DAY)
    cols, names = [], []
    for cn in _EXOG_REL_COLS:
        if cn not in feats:
            continue
        v = np.asarray(raw[:, feats.index(cn)], dtype=np.float64)
        z = np.zeros(len(v), dtype=np.float64)
        for s in np.unique(slot):
            idx = np.where(slot == s)[0]
            z[idx] = rolling_rank_gauss(v[idx][:, None], win)[:, 0]
        cols.append(z * close)
        names.append("xrel_%s" % cn)
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _cumret_close(Xs, feats, train_win):
    """Cumulative intraday SIGNED-return PATH gated to close -- the DIRECTIONAL sibling of cumrv (cumrv is the
    |magnitude| path). cumret.npy = cumsum of per-bar sumret within the calendar day, causal (build_intraday_
    cumret.py); robust-scaled then gated to close. Tests close mean-reversion of the day's directional move
    (leverage / intraday-reversal) -- a within-day open-anchored path HAR's day-agnostic windows cannot see."""
    cum = np.asarray(np.load("results/intraday_feats/cumret.npy"), dtype=np.float64)
    assert len(cum) == len(Xs), "cumret len %d != Xs len %d" % (len(cum), len(Xs))
    scaled = _floored_robust_scale(cum, train_win)
    hour = Xs[:, feats.index("hour")]
    close = (_close_mask(hour)).astype(np.float64)
    return (scaled * close)[:, None], ["cumret_x_close"]


def _cumrv_shape_close(Xs, feats):
    """Within-day vol-path SHAPE (#2): afternoon vol SHARE = (cumrv_now - cumrv_midday)/(cumrv_now+eps),
    gated close. cumrv = cumsum(har_ma_1) within the hour-wrap day; cumrv_midday = its value at the last
    hour==12 bar of the day, broadcast. At the close this is the fraction of the day's vol realized in the
    afternoon -- front- vs back-loaded. A within-day path SHAPE that needs the midday snapshot (not in the
    feature row), so the tree cannot reconstruct it. Causal (har_ma_1 is shift(1); midday is past bars)."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    cum = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    df = pd.DataFrame({"d": day_id, "h": np.rint(hour).astype(int), "c": cum})
    mid = df[df.h == 12].groupby("d")["c"].last()
    midbroad = df["d"].map(mid).fillna(0.0).to_numpy()
    share = np.divide(
        cum - midbroad, cum, out=np.zeros_like(cum), where=cum > 0
    )  # cum==0 (no vol yet, day-start) -> 0; no +eps. No-op at the close gate (cum>0 there)
    close = (_close_mask(hour)).astype(np.float64)
    return (share * close)[:, None], ["cumrv_aftshare_x_close"]


# ── REGIME-HIERARCHICAL PERSISTENCE features (the persist_* family) ───────────────────────────────
# The residual edge is an intraday vol-PERSISTENCE regime: HAR over-extrapolates at the session edges,
# and the only signal that ever beat the d8 tree was a HISTORY-dependent rolling-relative vol innovation
# (today's vol vs its recent same-slot distribution -- a functional the feature row does not contain).
# The log-signature path-ORDER basis was NULL (the close edge has no chronological-order content). So
# these features make PERSISTENCE explicitly REGIME-CONDITIONAL and HIERARCHICAL while staying STATE/level-
# based (not order-based): each is causal, bounded-by-construction (rank-Gauss ~N(0,1) / np.divide where>0,
# NO ad-hoc clips/floors/eps), and new-to-tree via a rolling history factor.


def _rank_gauss_by_group(v, group, win):
    """Causal rolling rank-Gauss of v WITHIN each group level (group = a per-row slot/phase/DOW id). Bounded
    ~N(0,1); reuses rolling_rank_gauss per group (its own shift(1) + warmup -> warmup rows emit 0). The shared
    primitive behind every persist_* feature: 'v relative to its own recent history within this regime tier'."""
    from src.features.transforms.rank_gauss import rolling_rank_gauss

    z = np.zeros(len(v), dtype=np.float64)
    for s in np.unique(group):
        idx = np.where(group == s)[0]
        z[idx] = rolling_rank_gauss(v[idx][:, None], win)[:, 0]
    return z


def _persist_clockstate(Xs, feats, train_win):
    """F1 -- NESTED clock x state persistence. Generalizes the base HAR x {open,close} to HAR_k gated to a
    finer clock hierarchy {open(h9), mid(10-15), close(16-19)} AND modulated by the LOCAL vol-STATE z_k (the
    rank-Gauss rolling same-slot innovation of har_k = the new-to-tree history factor). The persistence
    coefficient is thus free to vary across the nested (timescale -> clock-phase -> vol-state) regime. Emits
    ONLY the parts NOT already in the enetreg2 base (which has har_k x {open,close} for all 6 windows): the
    mid-phase clock bucket, and the state-MODULATED persistence har_k x phase x z_k. Causal; z_k ~N(0,1)."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    win = max(2, train_win // PERIODS_PER_DAY)
    slot = _rel_slot(Xs, feats, fine=False)
    phases = (
        ("open", (hour == 9).astype(np.float64)),
        ("mid", ((hour >= 10) & (hour <= 15)).astype(np.float64)),
        ("close", _close_mask(hour).astype(np.float64)),
    )
    cols, names = [], []
    for hn in ("har_ma_1", "har_ma_5", "har_ma_25"):
        if hn not in feats:
            continue
        h = Xs[:, feats.index(hn)].astype(np.float64)
        z = _rank_gauss_by_group(h, slot, win)  # local vol-state at this timescale
        # mid-phase clock bucket (new vs the base's open/close); state-modulated persistence per phase
        mid_col = h * phases[1][1]
        if mid_col.min() < mid_col.max():
            cols.append(mid_col)
            names.append(f"{hn}_x_mid")
        for pn, g in phases:
            sc = h * g * z
            if sc.min() < sc.max():
                cols.append(sc)
                names.append(f"{hn}_x_{pn}_x_state")
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _persist_cascade(Xs, feats, train_win):
    """F2 -- cross-TIMESCALE persistence cascade. Rank-Gauss rolling same-slot innovation of short/long HAR
    RATIOS (har_ma_1/har_ma_25, har_ma_5/har_ma_125), gated close: 'is the FAST scale diverging from the SLOW
    regime, vs its own history?'. A functional ACROSS the HAR timescale hierarchy; the rolling-rank of the
    ratio is history-dependent (new-to-tree). Causal; bounded ~N(0,1); ratio = np.divide where denom>0 (no eps)."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    win = max(2, train_win // PERIODS_PER_DAY)
    slot = _rel_slot(Xs, feats, fine=False)
    close = _close_mask(hour).astype(np.float64)
    cols, names = [], []
    for a, b in (("har_ma_1", "har_ma_25"), ("har_ma_5", "har_ma_125")):
        if a not in feats or b not in feats:
            continue
        va = Xs[:, feats.index(a)].astype(np.float64)
        vb = Xs[:, feats.index(b)].astype(np.float64)
        r = np.divide(va, vb, out=np.zeros_like(va), where=vb > 0)
        col = _rank_gauss_by_group(r, slot, win) * close
        if col.min() < col.max():
            cols.append(col)
            names.append(f"casc_{a}_{b}_x_close")
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _persist_dwell(Xs, feats, train_win):
    """F3 -- regime DWELL / run-length. Vol-state sign s = sign(z_5), z_5 = rank-Gauss rolling same-slot
    innovation of har_ma_5 (>0 = high local-vol regime). runlen = consecutive same-sign bars up to t-1 (causal
    shift; the LITERAL 'how long has the vol regime persisted' -- a state-DWELL path functional the log-sig
    ORDER basis did not cover). Bounded by a rank-Gauss of runlen over its own same-slot history (no magic
    scale/threshold). Gated close. The boldest/novel bet: it partially tensions with the path-order-death
    result, so a clean null here further seals 'state-only, no path'; a hit = dwell is the live path channel."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    win = max(2, train_win // PERIODS_PER_DAY)
    slot = _rel_slot(Xs, feats, fine=False)
    close = _close_mask(hour).astype(np.float64)
    if "har_ma_5" not in feats:
        return np.empty((len(Xs), 0), dtype=np.float64), []
    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    s = np.sign(_rank_gauss_by_group(h5, slot, win))
    runlen = np.zeros(len(s), dtype=np.float64)
    c, prev = 0, 0.0
    for t in range(len(s)):
        c = c + 1 if (s[t] == prev and s[t] != 0.0) else 1
        runlen[t] = c
        prev = s[t]
    runlen = np.concatenate([[0.0], runlen[:-1]])  # shift(1): run length as of t-1 (causal)
    col = _rank_gauss_by_group(runlen, slot, win) * close
    if col.min() < col.max():
        return col[:, None], ["dwell_har5_x_close"]
    return np.empty((len(Xs), 0), dtype=np.float64), []


def _persist_nested(Xs, feats, train_win):
    """F5 -- nested calendar-seasonality decomposition. The deployed `rel` measures har_ma_5 vs its recent
    same-HOUR history (one tier). Here the SAME vol's rank-Gauss rolling innovation is computed against a
    HIERARCHY of progressively coarser reference regimes: 48-slot -> hour -> session-phase -> day-of-week.
    Each tier standardizes the same vol against a different history horizon (an ANOVA-style nesting); the
    tier differences are the variance attributable to each regime level. Causal; bounded ~N(0,1); new-to-tree
    (history-dependent at each horizon). Ungated, matching the deployed rel (persistence-vs-history is
    regime-agnostic). DOW reconstructed from the one-hot dummies (weakest tier -- the Friday block was null)."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    win = max(2, train_win // PERIODS_PER_DAY)
    if "har_ma_5" not in feats:
        return np.empty((len(Xs), 0), dtype=np.float64), []
    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    hr = np.rint(hour).astype(np.int64)
    phase = np.where(hr == 9, 0, np.where(hr <= 15, 1, np.where(hr <= 19, 2, 3)))
    levels = [
        ("slot48", _rel_slot(Xs, feats, fine=True)),
        ("hour", hr),
        ("phase", phase),
    ]
    dow_cols = [f for f in feats if f.startswith("DOW_")]
    if dow_cols:
        D = np.column_stack([Xs[:, feats.index(c)] for c in dow_cols])
        dow = np.where(D.sum(axis=1) > 0, D.argmax(axis=1) + 1, 0)  # baseline=0, DOW_k=k (grouping id only)
        levels.append(("dow", dow))
    cols, names = [], []
    for ln, g in levels:
        z = _rank_gauss_by_group(h5, g, win)
        if z.min() < z.max():
            cols.append(z)
            names.append(f"nest_har5_{ln}")
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _day_tier_level(v, day_id, tier, win, mp):
    """Causal rolling mean of v at a regime TIER on a coherent win-DAY horizon: collapse to one value per
    (tier, day) = that day's tier-mean, roll over `win` DAYS WITHIN the tier (shift(1) = causal, excludes
    today), broadcast back to bars. tier=constant -> the GLOBAL daily level. The shared primitive for F4 so
    every tier shares the SAME ~win-day lookback (no cross-tier horizon mismatch); warmup (<mp days) -> 0."""
    import pandas as pd

    daily = (
        pd.DataFrame({"t": tier, "d": day_id, "v": v}).groupby(["t", "d"], sort=True)["v"].mean().reset_index()
    )
    daily["lvl"] = daily.groupby("t")["v"].transform(
        lambda s: s.rolling(win, min_periods=mp).mean().shift(1)
    )
    lut = {(t, d): lv for t, d, lv in zip(daily["t"], daily["d"], daily["lvl"])}
    out = np.array([lut.get((t, d), np.nan) for t, d in zip(tier, day_id)], dtype=np.float64)
    return np.nan_to_num(out, nan=0.0)


def _persist_pool(Xs, feats, train_win):
    """F4 -- HIERARCHICAL partial-pooling deviations (the multilevel-model idea as INTERPRETABLE features).
    Decompose har_ma_5's recent level into nested regime-tier departures from the grand level, each on a
    coherent win-DAY horizon, gated close: pool_dev_phase = (close-PHASE level - GLOBAL level), pool_dev_slot
    = (close hour-SLOT level - phase level), pool_dev_resid = (current har_5 - slot level). These ARE the
    random-effect (partial-pooling) estimates a hierarchical model shrinks toward the parent; each is signed,
    in vol units, so an enet coef / EBM shape reads off directly ('the close loads + on the phase anomaly,
    - on the slot anomaly'). Causal (shift(1) day; har_5 itself shift(1)); new-to-tree (rolling tier levels
    are not in the feature row). The interpretable raw-units sibling of F5's rank-Gauss tier innovations."""
    hour = Xs[:, feats.index("hour")].astype(np.float64)
    win = max(2, train_win // PERIODS_PER_DAY)
    mp = max(20, win // 20)
    if "har_ma_5" not in feats:
        return np.empty((len(Xs), 0), dtype=np.float64), []
    h5 = Xs[:, feats.index("har_ma_5")].astype(np.float64)
    hr = np.rint(hour).astype(np.int64)
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hr) < 0).astype(int)]))
    phase = np.where(hr == 9, 0, np.where(hr <= 15, 1, np.where(hr <= 19, 2, 3)))
    L_global = _day_tier_level(h5, day_id, np.zeros_like(day_id), win, mp)
    L_phase = _day_tier_level(h5, day_id, phase, win, mp)
    L_slot = _day_tier_level(h5, day_id, hr, win, mp)
    close = _close_mask(hour).astype(np.float64)
    candidates = [
        ((L_phase - L_global) * close, "pool_dev_phase_x_close"),
        ((L_slot - L_phase) * close, "pool_dev_slot_x_close"),
        ((h5 - L_slot) * close, "pool_dev_resid_x_close"),
    ]
    cols, names = [], []
    for c, nm in candidates:
        if c.min() < c.max():
            cols.append(c)
            names.append(nm)
    INT = (
        np.ascontiguousarray(np.column_stack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


_PERSIST_BUILDERS = {
    "cs": _persist_clockstate,
    "casc": _persist_cascade,
    "dwell": _persist_dwell,
    "nest": _persist_nested,
    "pool": _persist_pool,
}


def _build_persist(Xs, feats, train_win, families):
    """Concatenate the named persist families (cs/casc/dwell/nest/pool) into one [N,k] block + names."""
    cols, names = [], []
    for fam in families:
        c, nm = _PERSIST_BUILDERS[fam](Xs, feats, train_win)
        if c.shape[1]:
            cols.append(c)
            names += nm
    INT = (
        np.ascontiguousarray(np.hstack(cols))
        if cols
        else np.empty((len(Xs), 0), dtype=np.float64)
    )
    return INT, names


def _gap_open(Xs, feats, train_win):
    """Overnight directional GAP (#4): cumulative SIGNED return over the cross-midnight night run (20->9),
    gated to the OPEN (hour==9) -- the directional sibling of the (null) overnight-VOL cumrv_x_open.
    overnight_gap.npy = cumsum of per-bar sumret over each night run, causal (build_overnight_gap.py);
    robust-scaled, gated open. Tests open continuation/reversal of the overnight move -- an overnight path
    the tree cannot reconstruct (the cache has no overnight-anchored signed accumulation)."""
    cum = np.asarray(
        np.load("results/intraday_feats/overnight_gap.npy"), dtype=np.float64
    )
    assert len(cum) == len(Xs), "overnight_gap len %d != Xs len %d" % (
        len(cum),
        len(Xs),
    )
    scaled = _floored_robust_scale(cum, train_win)
    hour = Xs[:, feats.index("hour")]
    open_ = (hour == 9).astype(np.float64)
    return (scaled * open_)[:, None], ["gap_x_open"]


def _path_shape_block(Xs, feats):
    """RICHER within-day vol-path SHAPE block -- digging the cumrvshape facet (the best new feature, d8
    -0.00020). (1) vol TIME-CENTROID = cumsum(hour*har_ma_1)/cumsum(har_ma_1) within the day = the
    center-of-mass of WHEN the day's vol happened (front- vs back-loaded); (2) accumulation PROFILE = cumrv
    at the last hour-{11,13,15} bar of the day (broadcast) / cumrv_now = fraction of the day's vol realised
    by each anchor. All gated to close, causal (har_ma_1 is shift(1); anchors are earlier-in-day bars). These
    are the hand-computable time-channel low-order path-signature terms (the orthogonal facet to the saturated
    cross-day regime factor)."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    h = np.rint(hour).astype(int)
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    df = pd.DataFrame({"d": day_id, "h": h, "v": har1})
    cum = df.groupby("d")["v"].cumsum().to_numpy()
    cumh = (df["v"] * df["h"]).groupby(df["d"]).cumsum().to_numpy()
    close = (_close_mask(hour)).astype(np.float64)
    nz = (
        cum > 0
    )  # cum==0 (no vol yet) -> centroid/share undefined -> 0; no +eps (no-op at the gate)
    cols = [np.divide(cumh, cum, out=np.zeros_like(cum), where=nz)]
    names = ["vol_centroid"]
    cser = pd.Series(cum)
    for ha in (11, 13, 15):
        anc = (
            cser[h == ha].groupby(df["d"][h == ha]).last()
        )  # cumrv at the last hour==ha bar of each day
        br = df["d"].map(anc).fillna(0.0).to_numpy()
        cols.append(np.divide(br, cum, out=np.zeros_like(cum), where=nz))
        names.append("share_h%d" % ha)
    block = np.column_stack([c * close for c in cols])
    return np.ascontiguousarray(block), names


def _bowley_arrival_skew(dV, day_id, n, cumV):
    """Bowley (quartile) skewness of the within-day vol-ARRIVAL-TIME distribution: positions = bar
    index n (1..L), weights = dV (per-bar sqrt-vol). For each bar t the a-quantile POSITION is the bar
    where the cumulative vol-mass first reaches a*cumV_t; since a*cumV_t <= cumV_t and cumV is
    increasing, np.searchsorted(cumV_day, a*cumV_t) returns an index <= t -> CAUSAL (only bars <= t).
    Bowley = (Q1 + Q3 - 2*Q2)/(Q3 - Q1) is bounded [-1, 1] BY CONSTRUCTION -- no floor, no clip (it
    replaces the 3rd-standardized-moment skew that needed the 1-bar^2 resolution floor). Q3==Q1 (a
    degenerate single-mass day) -> 0. Vectorized per contiguous day-slice (O(N) total)."""
    bounds = np.flatnonzero(np.diff(day_id)) + 1
    q = {0.25: [], 0.5: [], 0.75: []}
    for C, P in zip(np.split(cumV, bounds), np.split(n, bounds)):
        for a in (0.25, 0.5, 0.75):
            j = np.searchsorted(C, a * C, side="left")
            np.clip(
                j, 0, len(C) - 1, out=j
            )  # defensive index bound (a*C<=C so j<=t already)
            q[a].append(P[j])
    q1, q2, q3 = (np.concatenate(q[a]) for a in (0.25, 0.5, 0.75))
    denom = q3 - q1
    return np.divide(
        q3 + q1 - 2.0 * q2, denom, out=np.zeros_like(denom), where=denom > 0
    )


def _path_sig_block(Xs, feats, train_win):
    """HIGHER-ORDER within-day path-SIGNATURE block (levels 2-3), gated close -- the systematic
    generalization of the hand-picked path-shape features (vol_centroid / afternoon-share are the
    LEVEL-1 time-moment of this hierarchy). Truncated signature of the causal within-day path with
    channels (tau = elapsed-bar time, V = cumsum(har_ma_1) the vol path, R = cumret the signed-return
    path); each coordinate accumulates over bars <= t (causal), robust-scaled to ~N(0,1) (the cumret/
    cumrv_real convention), then gated to close (16-19):
      sig_volcen   tau-V lvl2  Sum(n*dV)/Sum(dV)                = WHEN the day's vol happened (centroid)
      sig_voldisp  tau-V lvl2  std of the vol-arrival-time dist = vol in one burst vs spread all day
      sig_volskew  tau-V lvl3  skew of that dist                = early-burst vs late-burst asymmetry
      sig_area_rv  R-V   lvl2  Levy area Sum(R_<*dV - V_<*dR)   = intraday leverage (vol leads/lags price)
      sig_retcen   tau-R lvl2  Sum(n*dR)/(Sum|dR|+eps)          = WHEN the directional move happened
    The tau-V moments strictly generalize the deployed cumrv_aftshare / vol_centroid (their level-1);
    the R-V Levy area + tau-R centroid are the un-tested cross-channel terms (directional path enters
    ONLY through its lead-lag with vol / its timing -- cumret level-1 was null). Causal: har_ma_1 and
    cumret are themselves shift(1) within-day cumulatives, moments use only bars <= t. The R-channel
    terms need results/intraday_feats/cumret.npy; absent -> skipped+logged (the tau-V vol-timing block
    still builds from the in-cache har_ma_1 alone, so the cell preps with or without the return feed)."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    dV = Xs[:, feats.index("har_ma_1")].astype(np.float64)  # per-bar sqrt-vol increment
    di = pd.Series(day_id)
    n = (
        di.groupby(di).cumcount().to_numpy().astype(np.float64) + 1.0
    )  # within-day elapsed bars (1..)
    cumV = pd.Series(dV).groupby(day_id).cumsum().to_numpy()
    # cumV==0 (a day's first bars, no vol accumulated yet) -> the time-moment is undefined -> 0
    # (no signal), not a +eps quotient. cumV is a vol-mass denominator; zero mass = no observation.
    nz = cumV > 0
    s1 = pd.Series(n * dV).groupby(day_id).cumsum().to_numpy()
    s2 = pd.Series(n * n * dV).groupby(day_id).cumsum().to_numpy()
    m1 = np.divide(
        s1, cumV, out=np.zeros_like(cumV), where=nz
    )  # centroid (1st time-moment)
    m2 = np.divide(s2, cumV, out=np.zeros_like(cumV), where=nz)
    disp = np.sqrt(
        np.maximum(m2 - m1 * m1, 0.0)
    )  # dispersion (2nd central moment, FP-clamped)
    # Skew = early- vs late-vol-burst asymmetry, via BOWLEY (quartile) skewness: bounded [-1,1] by
    # construction, so no resolution floor and no clip (replaces the 3rd-standardized-moment form).
    skew = _bowley_arrival_skew(dV, day_id, n, cumV)
    cols = [m1, disp, skew]
    names = ["sig_volcen", "sig_voldisp", "sig_volskew"]
    try:
        cumret = np.asarray(
            np.load("results/intraday_feats/cumret.npy"), dtype=np.float64
        )
        assert len(cumret) == len(Xs), "cumret len %d != Xs len %d" % (
            len(cumret),
            len(Xs),
        )
        rser = pd.Series(cumret).groupby(day_id)
        Rprev = (
            rser.shift(1).fillna(0.0).to_numpy()
        )  # R at the prior bar (0 at day start)
        dR = cumret - Rprev  # per-bar signed return
        Vprev = pd.Series(cumV).groupby(day_id).shift(1).fillna(0.0).to_numpy()
        area = (
            pd.Series(Rprev * dV - Vprev * dR).groupby(day_id).cumsum().to_numpy()
        )  # discrete Levy area
        cumabsR = pd.Series(np.abs(dR)).groupby(day_id).cumsum().to_numpy()
        retnum = pd.Series(n * dR).groupby(day_id).cumsum().to_numpy()
        # cumabsR==0 (no return movement yet) -> return-timing undefined -> 0, not a +eps quotient
        retcen = np.divide(
            retnum, cumabsR, out=np.zeros_like(cumabsR), where=cumabsR > 0
        )
        cols += [area, retcen]
        names += ["sig_area_rv", "sig_retcen"]
    except FileNotFoundError:
        print("SIG skip R-channel (cumret.npy missing)", flush=True)
    block = _floored_robust_scale(
        np.column_stack(cols), train_win
    )  # scale UNGATED, then gate
    close = (_close_mask(hour)).astype(np.float64)
    return np.ascontiguousarray(block * close[:, None]), names


def _friday_week_block(Xs, feats, train_win):
    """FRIDAY-CLOSE features a tree CANNOT represent: WITHIN-WEEK cumulative paths that RESET each
    week (Mon->Fri) and are read at the Friday (DOW_4) close (h16-19). The regime-EBM shape probe
    flagged `voldemand_*_active x DOW_4` as its top close interaction -- an AVAILABILITY flag x
    Friday = the coverage-artifact signature. This replaces that flag with the weekly accumulation
    of the voldemand VALUE (+ a weekly realized-vol path + the pre-Friday demand setup), so a
    NON-null result is a REAL weekly dealer-demand -> Friday-close mechanism (dealer gamma / OPEX
    unwind into the Friday close), and a null says the Friday term was just the availability artifact.
    A tree sees only the current row's LEVEL -- it cannot reconstruct a week-resetting cumulative
    (no cache MA resets weekly; har_ma_625 ~13d is a rolling window, not a Mon-anchored sum).
    Week id is derived from the cache DOW one-hots (a DROP in day-index = a new week; robust to
    holiday-shortened weeks, no date-file dtype dependence). Causal: har_ma_1 / voldemand are
    shift(1); the week path and the pre-Friday setup are past bars. Robust-scaled, gated Friday&close."""
    import pandas as pd

    dow_num = (
        1 * Xs[:, feats.index("DOW_1")]
        + 2 * Xs[:, feats.index("DOW_2")]
        + 3 * Xs[:, feats.index("DOW_3")]
        + 4 * Xs[:, feats.index("DOW_4")]
    ).astype(np.int64)  # 0..4 (Mon..Fri); exactly one DOW one-hot is set per row
    new_week = np.concatenate(
        [[0], (dow_num[1:] < dow_num[:-1]).astype(int)]
    )  # day-index drop (Fri->Mon, or any holiday gap) = new week
    week_id = np.cumsum(new_week)
    hour = Xs[:, feats.index("hour")]
    close = _close_mask(hour)
    friday = Xs[:, feats.index("DOW_4")] > 0.5
    gate = (friday & close).astype(np.float64)
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    cumrv_w = (
        pd.Series(har1).groupby(week_id).cumsum().to_numpy()
    )  # week realized-vol path
    # cumrv is a POSITIVE accumulation -> causal robust-scale (the cumrv_real / cumret convention).
    cols = [_floored_robust_scale(cumrv_w[:, None], train_win)[:, 0]]
    names = ["friw_cumrv"]
    vdname = next(
        (
            c
            for c in (
                "adj_voldemand_all_open_only_ma_1",
                "adj_voldemand_spx_open_only_ma_1",
            )
            if c in feats
        ),
        None,
    )
    if vdname is not None:
        # voldemand is SIGNED rank-Gauss, so a within-week cumSUM is a random walk (drifts to +-90).
        # Standardize the partial sum by 1/sqrt(count) -> the demand "t-stat": how many sigma the
        # week's accumulated dealer demand sits from zero. This is ~unit by the CLT (no rescale, no
        # clip) AND still week-RESET, so a tree cannot reconstruct it from a fixed-window MA.
        #   friw_demand = standardized Mon->now demand (the week demand REGIME at the Friday close)
        #   friw_predem = standardized Mon-Thu demand (the demand SETUP into Friday; excludes Friday)
        vd = Xs[:, feats.index(vdname)].astype(np.float64)
        n_week = pd.Series(np.ones(len(week_id))).groupby(week_id).cumsum().to_numpy()
        friw_demand = pd.Series(vd).groupby(week_id).cumsum().to_numpy() / np.sqrt(
            n_week
        )
        nf = (~friday).astype(np.float64)
        d = pd.DataFrame({"w": week_id, "nf": nf, "vd": vd * nf})
        presum = (
            d.groupby("w")["vd"].transform("sum").to_numpy()
        )  # week Mon-Thu demand SUM
        precnt = (
            d.groupby("w")["nf"].transform("sum").to_numpy()
        )  # Mon-Thu bar COUNT (the causal expanding standardizer, not a fixed floor)
        # precnt==0 is a (real-data-impossible) Friday-only holiday week = NO pre-Friday data, so the
        # setup is genuinely undefined -> emit 0 (no signal), NOT a floored quotient (count==0 is
        # missing-data, not a collapsed scale; a scale would get a causal floor, a count does not).
        friw_predem = np.divide(
            presum, np.sqrt(precnt), out=np.zeros_like(presum), where=precnt > 0
        )
        cols += [friw_demand, friw_predem]
        names += ["friw_demand", "friw_predem"]
    else:
        print("FRI skip voldemand (col missing)", flush=True)
    return np.ascontiguousarray(np.column_stack(cols) * gate[:, None]), names


def _ofi_block(Xs, feats, bucket, train_win):
    """ORDER-FLOW IMBALANCE -- a NEW microstructure DATA channel (not another vol-path encoding), the
    closest buildable proxy to the auction-imbalance mechanism behind the close reversal. Loads the
    PHYSICAL value-weighted buy/sell turnover from covid_imp (verified NON-NEGATIVE: frac_neg=0), so
    per-bar OFI = (buy-sell)/(buy+sell) is bounded [-1,1] BY CONSTRUCTION. Accumulated within the
    trading day (reset daily, causal) -> at the close = the day's NET order-flow imbalance feeding the
    close auction:
      ofi_net    = cumsum(buy-sell)/cumsum(buy+sell)  signed net imbalance (directional) in [-1,1]
      ofi_absnet = |ofi_net|                          imbalance MAGNITUDE in [0,1] -- the vol-relevant
                   facet (a lopsided day -> larger close-auction price impact -> the reversal/vol the
                   close regime feeds on; the close edge is a vol-regime, not a directional one)
    Bounded -> appended RAW (no clip, no robust-scale, no eps: np.divide(where=cumsum>0)). A tree can't
    reconstruct a within-day cumulative; |.| is added for the LINEAR base (the tree subsumes it).
    Source = results/covid_imp (the _exog_rel_innov physical cache; same 529-feat layout as the slim
    cache, so feats.index aligns). Gated close."""
    import pandas as pd

    raw = np.load(f"results/covid_imp/{bucket}/X_imp.npy", mmap_mode="r")
    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    buy = np.asarray(
        raw[:, feats.index("adj_buyturnover_vwstock_ma_1")], dtype=np.float64
    )
    sell = np.asarray(
        raw[:, feats.index("adj_sellturnover_vwstock_ma_1")], dtype=np.float64
    )
    cumd = pd.Series(buy - sell).groupby(day_id).cumsum().to_numpy()
    cums = pd.Series(buy + sell).groupby(day_id).cumsum().to_numpy()
    ofi = np.divide(
        cumd, cums, out=np.zeros_like(cums), where=cums > 0
    )  # net day OFI in [-1,1]
    # Robust-scale the UNGATED OFI to ~unit so the L1 penalty sees it on a par with the other enet
    # features: raw OFI std ~0.03 = the SMALLEST in the matrix (median 0.42, cumrv_x_close 13.3), and
    # L1 penalizes |beta| scale-blind -> a tiny-scale feature is zeroed even WITH signal (it was, 0/407
    # blocks). Scale ungated (defined all rows -> no zero-inflation) THEN gate. The tree/EBM are
    # scale-invariant so this is neutral downstream; it only un-handicaps the linear base.
    block = _floored_robust_scale(np.column_stack([ofi, np.abs(ofi)]), train_win)
    close = _close_mask(hour).astype(np.float64)
    return np.ascontiguousarray(block * close[:, None]), ["ofi_net", "ofi_absnet"]


def _logsig_block(Xs, feats, train_win):
    """SYSTEMATIC truncated LOG-SIGNATURE of the causal within-day path (channels tau=elapsed bar,
    R=cumret, V=cumrv=cumsum(sqrt-vol)) to LEVEL 3 -- the universal nonparametric basis of path
    functionals, whose ANTISYMMETRIC coords (Levy areas at lvl2, brackets at lvl3) are the strict
    CHRONOLOGICAL-ORDER content a tree cannot reconstruct. The DEFINITIVE path-lever-death test: if
    the full antisymmetric basis (+ the QV realized-(co)variance terms) is null OOS even when FORCED
    into the tree/EBM, the close edge has NO order content = a pure state/level regime, fully captured,
    and we stop. 14 logsig coords (iisignature 3ch lvl3 = 3 increments + 3 Levy areas + 8 lvl-3
    brackets) + 3 QV terms (qv_rr=realized var of returns, qv_vv=vol-of-vol, qv_rv=intraday-leverage
    covariation -- the lead-lag-QV essence without the channel-doubling blowup). Computed at ALL
    within-day bars (t>=2) so the robust-scale per coord has no gated zero-inflation, THEN gated close.
    Needs results/intraday_feats/cumret.npy. MUST be dug with FORCE_COLS=<all coord names> -- the
    antisymmetric coords have ~no linear main effect, so L1 would zero them = a fake null (the OFI
    lesson). [[no-magic-numbers-feature-rigor]]."""
    import iisignature
    import pandas as pd

    s = iisignature.prepare(3, 3)
    D = int(iisignature.logsiglength(3, 3))
    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    di = pd.Series(day_id)
    n = di.groupby(di).cumcount().to_numpy().astype(np.float64) + 1.0
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)  # per-bar sqrt-vol = dV
    V = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    cumret = np.asarray(np.load("results/intraday_feats/cumret.npy"), dtype=np.float64)
    R = cumret  # already within-day cumulative signed return
    tau = n / PERIODS_PER_DAY  # elapsed time channel (~[0,1] within the day)
    dR = R - pd.Series(R).groupby(day_id).shift(1).fillna(0.0).to_numpy()
    out = np.zeros((len(Xs), D), dtype=np.float64)
    qv = np.zeros((len(Xs), 3), dtype=np.float64)
    bounds = np.flatnonzero(np.diff(day_id)) + 1
    for ta, RR, VV, dr, dv, ix in zip(
        np.split(tau, bounds),
        np.split(R, bounds),
        np.split(V, bounds),
        np.split(dR, bounds),
        np.split(har1, bounds),
        np.split(np.arange(len(Xs)), bounds),
    ):
        path = np.ascontiguousarray(np.column_stack([ta, RR, VV]))
        crr, cvv, crv = np.cumsum(dr * dr), np.cumsum(dv * dv), np.cumsum(dr * dv)
        for t in range(
            2, len(path)
        ):  # need >=3 points for a non-degenerate lvl-3 logsig
            out[ix[t]] = iisignature.logsig(path[: t + 1], s)
            qv[ix[t]] = (crr[t], cvv[t], crv[t])
    block = _floored_robust_scale(np.column_stack([out, qv]), train_win)
    close = _close_mask(hour).astype(np.float64)
    names = ["logsig_%d" % i for i in range(D)] + ["qv_rr", "qv_vv", "qv_rv"]
    return np.ascontiguousarray(block * close[:, None]), names


def _cumrv_rankcum_close(Xs, feats):
    """Order-matched diagnostic: cumsum-of-per-slot-RANK over TRUE calendar days (the proxy's
    rank-THEN-sum 'breadth' construction, but on cumrv_real's calendar-day segmentation), gated
    close. Built by build_rankcum.py -> results/intraday_feats/cumrv_realrankcum.npy. Pairs with
    enetreg2_real (cumsum-of-MAGNITUDE, same calendar seg -> isolates rank-vs-magnitude) and the
    proxy enetreg2 (rank-then-sum, hour-wrap seg -> isolates segmentation). Appended RAW: a cumsum
    of per-slot N(0,1) ranks is already on the slim feature scale (the proxy appends raw too)."""
    cum = np.asarray(
        np.load("results/intraday_feats/cumrv_realrankcum.npy"), dtype=np.float64
    )
    assert len(cum) == len(Xs), "rankcum len %d != Xs len %d" % (len(cum), len(Xs))
    hour = Xs[:, feats.index("hour")]
    close = (_close_mask(hour)).astype(np.float64)
    return (cum * close)[:, None], ["cumrv_rankcum_x_close"]


def _cumrv_v2rankcum_close(Xs, feats):
    """Order/input pivot for PART A's settled result. PART A proved the cache's har_ma_1 (the proxy's
    per-bar summand) is the UNRANKED HAR feature shift(1) adj_RV (Pearson=Spearman=1.000000 vs H_cache;
    strictly >=0, sqrt-RV magnitude) -- NOT a per-slot rank-Gauss. So the proxy (enetreg2, 0.12314) is
    cumsum-of-MAGNITUDE and rankcum (enetreg2_rankcum, 0.12429) is cumsum-of-RANK; that magnitude-vs-rank
    gap -- not order/input/window -- is the edge. This arm cumsums v2 = per-slot rank-Gauss( shift(1)
    adj_RV ) (shift-THEN-rank, adj_RV pipeline), the construction the task hypothesized matched the cache.
    Built by build_v2rankcum.py -> results/intraday_feats/cumrv_v2rankcum.npy, calendar-day segmented
    (== hour-wrap, ruled out). vs enetreg2_rankcum (cumsum v1 = rank-THEN-shift, RAW RV): isolates the
    rank ORDER+INPUT within rank space. vs enetreg2 proxy (0.12314): if ~0.1243 like rankcum, ANY rank
    construction caps below the magnitude proxy -> the edge is rank-vs-magnitude, full stop."""
    cum = np.asarray(
        np.load("results/intraday_feats/cumrv_v2rankcum.npy"), dtype=np.float64
    )
    assert len(cum) == len(Xs), "v2rankcum len %d != Xs len %d" % (len(cum), len(Xs))
    hour = Xs[:, feats.index("hour")]
    close = (_close_mask(hour)).astype(np.float64)
    return (cum * close)[:, None], ["cumrv_v2rankcum_x_close"]


def _cumrv_lag_close(Xs, feats, lag=PERIODS_PER_DAY):
    """Causality PLACEBO: the proxy cumrv (cumsum of the cache's har_ma_1 within the hour-wrap trading
    day -- IDENTICAL series to _cumrv_close BEFORE its gate) LAGGED by one full session (PERIODS_PER_DAY
    =48 bars) and THEN gated close. Row t sees the intraday-vol path as it stood 48 bars (one session)
    earlier, so there is NO same-session dependence. If base-alone ~= the proxy's 0.12314, the edge is a
    slow CROSS-session vol regime (yesterday's vol path predicts today's close); if it degrades toward
    0.12436 the signal is genuinely INTRADAY (today's own accumulating path matters). Causal either way
    (har_ma_1 is itself shift(1); the extra 48-bar lag only makes it MORE lagged)."""
    import pandas as pd

    hour = Xs[:, feats.index("hour")]
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    har1 = Xs[:, feats.index("har_ma_1")].astype(np.float64)
    cumrv = pd.Series(har1).groupby(day_id).cumsum().to_numpy()
    lagged = np.concatenate(
        [np.zeros(int(lag), dtype=np.float64), cumrv[: len(cumrv) - int(lag)]]
    )
    close = (_close_mask(hour)).astype(np.float64)
    return (lagged * close)[:, None], ["cumrv_lag_x_close"]


def _load_matrix(bucket, train_win, pipe="slim"):
    """pipe="slim": raw per-slot-rank matrix, NO robust-scale (ablation C_norob_all winner:
    drop semantic + drop robust-scale -> per-slot-rank -> indicators -> Ridge).
    pipe="pca<K>": slim, but the EXOG block compressed to K causal rolling-PCA factors
    (HAR + indicators kept raw) -> [HAR, K factors, indicators]. Tests the dense-weak
    low-rank hypothesis (what colsample->0.1 bagging gropes toward).
    pipe="std": production divide+semantic matrix WITH robust-scale (the old baseline)."""
    if pipe.startswith("pca"):
        from src.features.transforms.rolling_pca import rolling_pca

        k = int(pipe[3:])
        b = f"results/covid_imp_rank/{bucket}"
        X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
        y = np.load(f"{b}/y.npy")
        base = np.load(f"{b}/base.npy")
        har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
        factors = rolling_pca(X[:, exog], train_win, k, PCA_REFIT)
        Xs = np.ascontiguousarray(np.hstack([X[:, har], factors, X[:, ind]]))
        return Xs, y, base
    if pipe == "slim":
        b = f"results/covid_imp_rank/{bucket}"
        X = np.load(f"{b}/X_imp.npy")
        y = np.load(f"{b}/y.npy")
        base = np.load(f"{b}/base.npy")
        Xs = np.ascontiguousarray(
            np.asarray(X, dtype=np.float64)
        )  # raw rank features; no scaling
        return Xs, y, base
    b = f"results/covid_imp/{bucket}"
    X = np.load(f"{b}/X_imp.npy")
    y = np.load(f"{b}/y.npy")
    base = np.load(f"{b}/base.npy")
    ref_iqr = np.load(f"{b}/ref_iqr.npy")
    meta = json.load(open(f"{b}/meta.json"))
    Xs = np.ascontiguousarray(
        rolling_robust_scale(
            np.asarray(X, dtype=np.float64),
            train_win,
            ref_iqr=ref_iqr,
            fixed_cols=meta["fixed_cols"],
        )
    )
    return Xs, y, base


def _cadence_ridge(Xs, y, train_win, alpha, refit):
    """sklearn Ridge coef/intercept at each refit point t_r (the in-sample basis for the tree)."""
    n = len(Xs)
    starts = list(range(train_win, n, refit))
    coefs = np.empty((len(starts), Xs.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(starts), dtype=np.float64)
    for i, t_r in enumerate(starts):
        rg = Ridge(alpha=alpha).fit(Xs[t_r - train_win : t_r], y[t_r - train_win : t_r])
        coefs[i] = rg.coef_
        intercepts[i] = rg.intercept_
    return np.asarray(starts, dtype=np.int64), coefs, intercepts


def _cadence_predict(Xs, y, train_win, make_est, refit):
    """Cadence-refit OOS preds for ANY sklearn estimator (Ridge/ElasticNet/PLS): fit on the
    trailing window at each refit point, predict the block. Generalizes _cadence_ridge to
    estimators without a clean coef/intercept (PLS)."""
    n = len(Xs)
    oos = np.empty(n - train_win, dtype=np.float64)
    starts = list(range(train_win, n, refit))
    for i, t_r in enumerate(starts):
        est = make_est()
        est.fit(Xs[t_r - train_win : t_r], y[t_r - train_win : t_r])
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - train_win : t_end - train_win] = np.asarray(
            est.predict(Xs[t_r:t_end])
        ).ravel()
    return oos


def _dr_configs():
    """Linear/DR bases to compare against Ridge-a100 (0.12565): elastic net (sparse supervised
    selection) + PLS (supervised components) — the SUPERVISED DR that variance-PCA isn't."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet, Ridge

    cfgs = [("ridge_a100", lambda: Ridge(alpha=100))]
    for a in (1e-3, 1e-2, 1e-1):
        for l1 in (0.2, 0.5, 0.8):
            cfgs.append(
                (
                    f"enet_a{a:g}_l1{l1:g}",
                    lambda a=a, l1=l1: ElasticNet(
                        alpha=a, l1_ratio=l1, max_iter=1000, tol=1e-3
                    ),
                )
            )
    for nc in (10, 20, 40, 80):
        cfgs.append((f"pls_nc{nc}", lambda nc=nc: PLSRegression(n_components=nc)))
    return cfgs


def do_dimreduce_cell(model, bucket, idx, refit=8000):
    """One config of the supervised-DR comparison (SLURM-array task). Full slim inputs (same as
    raw-slim Ridge), cadence-refit, fixed OOS. Caveat: enet/Ridge L1/L2 see UNstandardized slim
    (HAR raw, exog ~N(0,1), ind 0/1) for apples-to-apples with the Ridge baseline."""
    name, make_est = _dr_configs()[idx]
    b = f"results/covid_imp_rank/{bucket}"
    Xs = np.ascontiguousarray(np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64))
    y = np.load(f"{b}/y.npy")
    base = np.load(f"{b}/base.npy")
    tw = 1000 * PERIODS_PER_DAY
    oos_start = tw
    t0 = time.time()
    preds = _cadence_predict(Xs, y, tw, make_est, refit)
    pr, tr = apply_duan_smearing(
        preds[oos_start - tw :], y[oos_start:], base[oos_start:]
    )
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    print(
        "DR idx=%-2d %-18s qlike=%.5f %.0fs"
        % (idx, name, float(np.mean(r - np.log(r) - 1)), time.time() - t0),
        flush=True,
    )


def _cadence_enet(Xs, y, train_win, refit, alpha=0.001, l1=0.2):
    """Cadence-refit ELASTIC-NET base coef/intercept (the winning linear base, 0.12530). Warm-started
    coordinate descent chains the rolling refits (no exact rank-1 for L1; warm-start is the cheap route)."""
    from sklearn.linear_model import ElasticNet

    n = len(Xs)
    starts = list(range(train_win, n, refit))
    en = ElasticNet(alpha=alpha, l1_ratio=l1, warm_start=True, max_iter=1000, tol=1e-3)
    coefs = np.empty((len(starts), Xs.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(starts), dtype=np.float64)
    for i, t_r in enumerate(starts):
        en.fit(Xs[t_r - train_win : t_r], y[t_r - train_win : t_r])
        coefs[i] = en.coef_
        intercepts[i] = float(en.intercept_)
    return np.asarray(starts, dtype=np.int64), coefs, intercepts


def _cadence_enet_har_unpen(Xs, y, train_win, refit, har_mask, alpha=0.001, l1=0.2):
    """Cadence-refit enet with the HAR block (har_mask) UNPENALIZED (OLS), the rest L1+L2 -- the
    structured-penalty realization. Unpenalized (not just L2-only) because an OLS block is invariant to ANY
    linear reparam, so the HAR basis becomes a free legibility choice; L2-only is invariant only under
    ORTHONORMAL maps and so would NOT neutralize the non-orthonormal differenced bands. EXACT via FWL:
    profile the HAR block out by OLS (M_H = I - H H^+ is an idempotent projection), fit a standard sklearn
    enet on the HAR-residualized penalized features (E_res, y_res), recover HAR coefs by OLS. With an empty
    mask this reduces EXACTLY to _cadence_enet. Same (starts, coefs, intercepts) contract."""
    from sklearn.linear_model import ElasticNet

    n, p = Xs.shape
    starts = list(range(train_win, n, refit))
    hm = np.asarray(har_mask, dtype=bool)
    em = ~hm
    coefs = np.zeros((len(starts), p), dtype=np.float64)
    intercepts = np.empty(len(starts), dtype=np.float64)
    en = ElasticNet(
        alpha=alpha,
        l1_ratio=l1,
        fit_intercept=False,
        warm_start=True,
        max_iter=5000,
        tol=1e-4,
    )
    for i, t_r in enumerate(starts):
        X = Xs[t_r - train_win : t_r]
        yt = y[t_r - train_win : t_r]
        xbar = X.mean(axis=0)
        ybar = float(yt.mean())
        Xc = X - xbar
        yc = yt - ybar
        H = np.ascontiguousarray(Xc[:, hm])
        E = np.ascontiguousarray(Xc[:, em])
        Hpinv = np.linalg.pinv(
            H
        )  # (k, m) -- rank-deficient-safe OLS for the collinear HAR block
        bH_y = Hpinv @ yc  # H-projection of y
        bH_E = Hpinv @ E  # (k, p_E) H-projection of each penalized col
        y_res = yc - H @ bH_y
        E_res = E - H @ bH_E
        en.fit(E_res, y_res)
        bE = en.coef_
        bH = bH_y - bH_E @ bE  # FWL recovery of the (unpenalized) HAR coefs
        beta = np.zeros(p, dtype=np.float64)
        beta[hm] = bH
        beta[em] = bE
        coefs[i] = beta
        intercepts[i] = ybar - xbar @ beta
    return np.asarray(starts, dtype=np.int64), coefs, intercepts


def _cadence_enet_online(Xs, y, train_win, refit, alpha=0.001, l1=0.2, mu_factors=None):
    """Cadence-refit elastic-net base with the L1 penalty mu picked ONLINE per refit block from a
    leakage-clean forward split (vs _cadence_enet's FIXED alpha). The L2 ridge lam2 is HELD at the
    fixed-alpha baseline level (train_win*alpha*(1-l1)); only mu is data-driven. Per block over the
    window W=[t_r-train_win, t_r): build a forward fit/val split inside W (embargo>=3125 kills HAR
    rolling-window bleed -- max HAR lag is 3125), select mu* over a 0.1x..10x log-grid centered on the
    baseline mu0 via reclasso_har's homotopy forward selector, then REFIT on the full window W at
    (lam2, mu*). Same (starts, coefs, intercepts) contract as _cadence_enet. mu_factors overrides the
    default log-grid -- olamcheck passes [1.0] to collapse to the fixed-alpha base for a cross-check."""
    from reclasso_har import GramState, forward_window_split, select_l1_forward

    n = len(Xs)
    starts = list(range(train_win, n, refit))
    lam2 = train_win * alpha * (1.0 - l1)  # fixed L2 (matches the fixed-alpha baseline)
    mu0 = train_win * alpha * l1  # baseline L1 center
    factors = (
        np.logspace(-1, 1, 9)
        if mu_factors is None
        else np.asarray(mu_factors, dtype=float)
    )
    mu_grid = mu0 * factors
    val_len = 20 * PERIODS_PER_DAY
    embargo = 3125  # >= max HAR lag -> no rolling-window bleed
    fit_lo, fit_hi, val_lo, val_hi = forward_window_split(
        train_win, train_win, val_len, embargo
    )
    coefs = np.empty((len(starts), Xs.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(starts), dtype=np.float64)
    for i, t_r in enumerate(starts):
        W = Xs[t_r - train_win : t_r]
        yW = y[t_r - train_win : t_r]
        g_fit, c_fit, xbar, ybar = GramState.from_block(
            W[fit_lo:fit_hi], yW[fit_lo:fit_hi]
        ).centered_ridged(lam2)
        mu_star, _ = select_l1_forward(
            g_fit, c_fit, W[val_lo:val_hi], yW[val_lo:val_hi], mu_grid, xbar, ybar
        )
        coef, intercept = GramState.from_block(W, yW).coef_intercept_at_mu(
            lam2, mu_star
        )
        coefs[i] = coef
        intercepts[i] = intercept
    return np.asarray(starts, dtype=np.int64), coefs, intercepts


def _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts):
    """Cadence-refit Ridge OOS preds: within block [t_r, t_{r+1}) use that block's coef
    (a matvec) — free from the cadence coefs, no slow every-bar incremental. Ridge and the
    tree then refit on the SAME cadence (internally consistent baseline)."""
    n = len(Xs)
    oos = np.empty(n - train_win, dtype=np.float64)
    for i, t_r in enumerate(starts):
        t_r = int(t_r)
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - train_win : t_end - train_win] = (
            Xs[t_r:t_end] @ coefs[i] + intercepts[i]
        )
    return oos


def _tree_factory(model, cfg):
    if model in ("lgbm", "lightgbm"):
        from lightgbm import LGBMRegressor

        kw = dict(cfg)
        kw.setdefault("n_jobs", 4)
        kw.setdefault("verbose", -1)
        kw.setdefault("random_state", 42)
        return lambda: LGBMRegressor(**kw)
    if (
        model == "ebm"
    ):  # Microsoft EBM: bagged boosted GAM (additive + pairwise), interpretable
        from interpret.glassbox import ExplainableBoostingRegressor

        kw = dict(cfg)
        kw.setdefault("n_jobs", 4)
        kw.setdefault("random_state", 42)
        return lambda: ExplainableBoostingRegressor(**kw)
    if (
        model == "moe"
    ):  # interpretable regime mixture-of-experts (learned soft-tree gate + NAM experts)
        from src.models.regime_moe import RegimeMoE

        return lambda: RegimeMoE(**dict(cfg))
    from xgboost import XGBRegressor

    kw = dict(cfg)
    kw.setdefault("n_jobs", 4)
    kw.setdefault("tree_method", "hist")
    kw.setdefault("random_state", 42)
    return lambda: XGBRegressor(**kw)


def _residualized_preds(
    Xs, y, train_win, ridge_oos, starts, coefs, intercepts, make_tree
):
    """final[t] = ridge_oos[t] + tree_{t_r}.predict(X[t]); tree fits y_train - X_train@beta_tr."""
    n = len(Xs)
    preds = np.array(
        ridge_oos, dtype=np.float64, copy=True
    )  # OOS preds indexed by k = t - train_win
    for i, t_r in enumerate(starts):
        Xtr = Xs[t_r - train_win : t_r]
        r_train = y[t_r - train_win : t_r] - (Xtr @ coefs[i] + intercepts[i])
        tree = make_tree()
        tree.fit(Xtr, r_train)
        t_end = min(t_r + (starts[i + 1] - t_r if i + 1 < len(starts) else n - t_r), n)
        preds[t_r - train_win : t_end - train_win] += tree.predict(Xs[t_r:t_end])
    return preds


def _qlike(preds, y, base, train_win):
    yo, bo = y[train_win:], base[train_win:]
    pr, tr = apply_duan_smearing(preds, yo, bo)
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def do_prep(model, bucket, twd, alpha, refit, pipe="slim", base_kind="ridge"):
    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    feats_aug = None
    if base_kind in (
        "enetreg2_canon",
        "enetreg2_canon_spline",
        "enetreg2_canon_scaled",
    ):
        # CANONICAL banded HAR base: the 6 nested sqrt-magnitude HAR MAs REPLACED by their ~orthogonal
        # differenced bands (band_k = har_ma_k - har_ma_{k-1}); regime interactions built from the bands;
        # cumrv x close kept (computed from har_ma_1 = band_1 before the native HAR removal). Span-preserving
        # in OLS, but NOT under the fixed-alpha enet (enetreg2_canon = 0.12423, +0.00109): the bands are
        # smaller-scale (differences of correlated MAs) than the nested MAs the alpha was tuned for, so the
        # penalty over-shrinks them. enetreg2_canon_scaled robust-scales the bands -> should recover ~0.12314
        # (isolates the loss as SCALE-mismatch, not the rotation). The band coefficients (printed below) are
        # interpretable either way. enetreg2_canon_spline adds a 2-knot relu spline on magnitude har_ma_5.
        feats0 = json.load(open(f"results/covid_imp_rank/{bucket}/meta.json"))["feats"]
        bands, band_names = _har_bands(Xs, feats0)
        if (
            base_kind == "enetreg2_canon_scaled"
        ):  # robust-scale bands to the ~N(0,1) scale the penalty expects
            bands = _floored_robust_scale(bands, train_win)
        cv, cv_names = _cumrv_close(
            Xs, feats0
        )  # from har_ma_1 (= band_1), BEFORE native HAR removal
        bint, bint_names = _band_regime_interactions(bands, band_names, Xs, feats0)
        add_cols = [bands, bint, cv]
        add_names = list(band_names) + list(bint_names) + list(cv_names)
        if base_kind == "enetreg2_canon_spline":
            sp, sp_names = _har5_spline_close(Xs, feats0, train_win)
            add_cols.append(sp)
            add_names += list(sp_names)
        har_idx = [
            feats0.index(h) for h in _HAR_COLS if h in feats0
        ]  # the 6 nested MAs the bands replace
        keep_native = np.ones(Xs.shape[1], dtype=bool)
        keep_native[har_idx] = False
        Xs = np.ascontiguousarray(np.hstack([Xs[:, keep_native]] + add_cols))
        feats_aug = [f for i, f in enumerate(feats0) if keep_native[i]] + add_names
    elif base_kind in (
        "enetreg",
        "enetreg2",
        "enetcum",
        "enetreg2_noclose",
        "enetreg2_olam",
        "enetreg2_ratio",
        "enetreg2_real",
        "enetreg2_opex",
        "enetreg2_realorth",
        "enetreg2_realrank",
        "enetreg2_rankcum",
        "enetreg2_rankcum2",
        "enetreg2_lag",
        "enetreg2_distill",
        "enetreg2_distill_harsq",
        "enetreg2_open",
        "enetreg2_realsqrt",
        "enetreg2_har5rank",
        "enetreg2_har5spl",
        "enetreg2_har5rank_spl",
        "enetreg2_harunpen",
        "enetreg2_rel",
        "enetreg2_relz",
        "enetreg2_relratio",
        "enetreg2_relfine",
        "enetreg2_ivmag",
        "enetreg2_linbest",
        "enetreg2_exogrel",
        "enetreg2_exogorder",
        "enetreg2_cumret",
        "enetreg2_cumrvshape",
        "enetreg2_gapopen",
        "enetreg2_shape",
        "enetreg2_shaperel",
        "enetreg2_sig",
        "enetreg2_sigrel",
        "enetreg2_fri",
        "enetreg2_frirel",
        "enetreg2_ofi",
        "enetreg2_ofirel",
        "enetreg2_logsig",
        "enetreg2_persist_cs",
        "enetreg2_persist_casc",
        "enetreg2_persist_dwell",
        "enetreg2_persist_nest",
        "enetreg2_persist_pool",
    ):
        # Fold the distilled OPEN/CLOSE x HAR regime interaction and/or the intraday vol-path
        # cumrv x close into the linear base. Appended BEFORE the constant-drop so they flow
        # through enet/masks/tree exactly like native columns. Variants (cumrv-presence ablation):
        #   enetreg          -> interactions (open+close), no cumrv
        #   enetreg2         -> interactions (open+close) + cumrv x close (rank-space PROXY)
        #   enetcum          -> cumrv x close ONLY (isolates cumrv's value vs plain enet)
        #   enetreg2_noclose -> interactions (OPEN only) + cumrv x close (is the close-half the
        #                       redundant part? cumrv x close already covers hours 16-19)
        #   enetreg2_olam    -> SAME features as enetreg2 (open+close + cumrv x close) but the cadence
        #                       base picks mu online per refit (_cadence_enet_online), not fixed alpha
        #   enetreg2_ratio   -> interactions (open+close) + self-anchored cumrv RATIO x close (one clean
        #                       normalized column replacing the additive cumrv x close pair)
        #   enetreg2_real    -> interactions (open+close) + PHYSICAL post-diurnal cumrv x close
        #                       (cumrv_real.npy, robust-scaled) -- replaces the rank-space proxy. PRIMARY.
        #   enetreg2_opex    -> enetreg2 (PROXY cumrv) + the 7 OPEX/calendar features (calfeats.npy):
        #                       marginal value of OPEX/quad-witch/EOM/EOQ/day-of-month structure
        #   enetreg2_realorth-> like enetreg2_real but the physical cumrv is causally orthogonalized
        #                       against the HAR x close columns before gating (pure marginal cumrv)
        #   enetreg2_realrank-> like enetreg2_real but the physical cumrv is PER-SLOT rank-Gauss
        #                       normalized (NOT robust-scaled) before gating -- tests whether the cumrv
        #                       signal lives in RANK space (vs robust-scaled enetreg2_real / proxy enetreg2)
        #   enetreg2_distill -> enetreg2 (open+close + PROXY cumrv x close) + the 3 Hero-B-EBM-flagged
        #                       microstructure cols x close (_distill_interactions): does the regime EBM's
        #                       top main effects distill LINEARLY as close-gated interactions?
        #   enetreg2_distill_harsq-> enetreg2_distill + a NONLINEAR signed har_ma_5^2 x close
        #                       (_har_sq_close): does a quadratic close-damping term add beyond linear HAR x close?
        feats0 = json.load(open(f"results/covid_imp_rank/{bucket}/meta.json"))["feats"]
        if base_kind in ("enetreg2_ivmag", "enetreg2_linbest"):
            # PER-BUCKET space test: swap the 18 implied-vol LEVEL columns (adj_{vix,vvix,vix3m}_ma_*) from
            # the slim rank-Gauss space to robust-scale MAGNITUDE (raw source = the covid_imp std-pipe cache,
            # identical 529-feat layout), holding the diurnal-adjust + MA structure constant. Isolates rank-vs-
            # magnitude for implied vol (VIX^2 ~ variance forecast, commensurate with RV -> rank may discard
            # the level). All other exog stay rank; availability indicators (vix_avail_*) untouched.
            # Scale via the STD-pipe's SAVED ref_iqr + fixed_cols (NOT _floored_robust_scale's self-computed
            # ref -- that hit a degenerate IQR on the early-missing vix -> million-scale blowup). This matches
            # _load_matrix(pipe='std') exactly, so these are the production magnitude features for vix.
            ivb = f"results/covid_imp/{bucket}"
            ivraw = np.asarray(np.load(f"{ivb}/X_imp.npy"), dtype=np.float64)
            iv_ref = np.load(f"{ivb}/ref_iqr.npy")
            iv_fixed = json.load(open(f"{ivb}/meta.json"))["fixed_cols"]
            Xmag = rolling_robust_scale(
                ivraw, train_win, ref_iqr=iv_ref, fixed_cols=iv_fixed
            )
            ividx = [
                i
                for i, f in enumerate(feats0)
                if f.startswith(("adj_vix_", "adj_vvix_", "adj_vix3m_"))
            ]
            Xs = np.ascontiguousarray(Xs)
            Xs[:, ividx] = Xmag[:, ividx]
        if base_kind == "enetreg2_exogorder":
            # CLEAN MA/rank ORDER swap for the EBM-top exog. Cache = MA_k(rank_w20(exog)); this REPLACES those
            # cols with rank_w20(MA_k(exog)) -- the SAME diurnal_rank (window=DIURNAL_WINDOW) at 48-slot, just
            # AFTER the MA instead of before. Holds window + slot + ungated constant -> isolates the rank/MA
            # non-commutativity alone (vs exogrel, which also changed window 20->1000 and gated to close).
            import pandas as pd

            from src.features.transforms.target import (
                DIURNAL_MIN_PERIODS,
                DIURNAL_WINDOW,
                diurnal_rank,
            )

            ma_mag = np.load(
                f"results/covid_imp/{bucket}/X_imp.npy", mmap_mode="r"
            )  # MA_k(magnitude), unranked
            tod = pd.Series(
                _rel_slot(Xs, feats0, fine=True)
            )  # 48-slot, matches the cache's resolution
            Xs = np.ascontiguousarray(Xs)
            for cn in _EXOG_REL_COLS:
                if cn not in feats0:
                    continue
                j = feats0.index(cn)
                ranked, _ = diurnal_rank(
                    pd.Series(np.asarray(ma_mag[:, j], dtype=np.float64)),
                    tod,
                    window=DIURNAL_WINDOW,
                    min_periods=DIURNAL_MIN_PERIODS,
                    gaussianize=True,
                )
                Xs[:, j] = np.nan_to_num(ranked.to_numpy(dtype=np.float64), nan=0.0)
        add_cols, add_names = [], []
        if base_kind in (
            "enetreg",
            "enetreg2",
            "enetreg2_olam",
            "enetreg2_ratio",
            "enetreg2_real",
            "enetreg2_opex",
            "enetreg2_realorth",
            "enetreg2_realrank",
            "enetreg2_rankcum",
            "enetreg2_rankcum2",
            "enetreg2_lag",
            "enetreg2_distill",
            "enetreg2_distill_harsq",
            "enetreg2_open",
            "enetreg2_realsqrt",
            "enetreg2_har5rank",
            "enetreg2_har5spl",
            "enetreg2_har5rank_spl",
            "enetreg2_harunpen",
            "enetreg2_rel",
            "enetreg2_relz",
            "enetreg2_relratio",
            "enetreg2_relfine",
            "enetreg2_ivmag",
            "enetreg2_linbest",
            "enetreg2_exogrel",
            "enetreg2_exogorder",
            "enetreg2_cumret",
            "enetreg2_cumrvshape",
            "enetreg2_gapopen",
            "enetreg2_shape",
            "enetreg2_shaperel",
            "enetreg2_sig",
            "enetreg2_sigrel",
            "enetreg2_fri",
            "enetreg2_frirel",
            "enetreg2_ofi",
            "enetreg2_ofirel",
            "enetreg2_logsig",
            "enetreg2_persist_cs",
            "enetreg2_persist_casc",
            "enetreg2_persist_dwell",
            "enetreg2_persist_nest",
            "enetreg2_persist_pool",
        ):
            INT, int_names = _regime_interactions(Xs, feats0)
            add_cols.append(INT)
            add_names += list(int_names)
        elif base_kind == "enetreg2_noclose":
            INT, int_names = _regime_interactions(Xs, feats0, which=("open",))
            add_cols.append(INT)
            add_names += list(int_names)
        if base_kind in (
            "enetreg2",
            "enetcum",
            "enetreg2_noclose",
            "enetreg2_olam",
            "enetreg2_opex",
            "enetreg2_distill",
            "enetreg2_distill_harsq",
            "enetreg2_open",
            "enetreg2_har5rank",
            "enetreg2_har5spl",
            "enetreg2_har5rank_spl",
            "enetreg2_harunpen",
            "enetreg2_rel",
            "enetreg2_relz",
            "enetreg2_relratio",
            "enetreg2_relfine",
            "enetreg2_ivmag",
            "enetreg2_linbest",
            "enetreg2_exogrel",
            "enetreg2_exogorder",
            "enetreg2_cumret",
            "enetreg2_cumrvshape",
            "enetreg2_gapopen",
            "enetreg2_shape",
            "enetreg2_shaperel",
            "enetreg2_sig",
            "enetreg2_sigrel",
            "enetreg2_fri",
            "enetreg2_frirel",
            "enetreg2_ofi",
            "enetreg2_ofirel",
            "enetreg2_logsig",
            "enetreg2_persist_cs",
            "enetreg2_persist_casc",
            "enetreg2_persist_dwell",
            "enetreg2_persist_nest",
            "enetreg2_persist_pool",
        ):
            cv, cv_names = _cumrv_close(
                Xs, feats0
            )  # rank-space proxy (enetreg2_opex/distill reuse it)
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_ratio":
            cv, cv_names = _cumrv_ratio_close(Xs, feats0)
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_real":
            cv, cv_names = _cumrv_real_close(
                Xs, feats0, train_win
            )  # physical post-diurnal cumrv
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_realsqrt":
            cv, cv_names = _cumrv_realsqrt_close(
                Xs, feats0, train_win
            )  # physical cumrv, SQRT scale (isolates sqrt)
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_realorth":
            cv, cv_names = _cumrv_realorth_close(
                Xs, feats0, train_win, refit
            )  # HAR-orthogonalized physical cumrv
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_realrank":
            cv, cv_names = _cumrv_real_rank_close(
                Xs, feats0, train_win
            )  # physical cumrv, PER-SLOT rank-Gauss
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_rankcum":
            cv, cv_names = _cumrv_rankcum_close(
                Xs, feats0
            )  # cumsum-of-RANK (breadth), calendar-day seg
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_rankcum2":
            cv, cv_names = _cumrv_v2rankcum_close(
                Xs, feats0
            )  # cumsum-of-RANK of v2 (shift-then-rank, adj_RV)
            add_cols.append(cv)
            add_names += cv_names
        elif base_kind == "enetreg2_lag":
            cv, cv_names = _cumrv_lag_close(
                Xs, feats0
            )  # proxy cumrv lagged one full session (placebo)
            add_cols.append(cv)
            add_names += cv_names
        if (
            base_kind == "enetreg2_opex"
        ):  # + the 7 OPEX/calendar columns (count cols robust-scaled)
            cf, cf_names = _calfeats(Xs, train_win)
            add_cols.append(cf)
            add_names += cf_names
        if base_kind in (
            "enetreg2_distill",
            "enetreg2_distill_harsq",
        ):  # + EBM-flagged microstructure x close
            di, di_names = _distill_interactions(Xs, feats0)
            add_cols.append(di)
            add_names += di_names
        if (
            base_kind == "enetreg2_distill_harsq"
        ):  # + nonlinear convex har^2 x close (close-damping curvature)
            hs, hs_names = _har_sq_close(Xs, feats0)
            add_cols.append(hs)
            add_names += hs_names
        if base_kind in (
            "enetreg2_har5rank",
            "enetreg2_har5rank_spl",
            "enetreg2_linbest",
        ):  # + RANK-SPACE har_ma_5 x close
            hr, hr_names = _har5rank_close(Xs, feats0, train_win)
            add_cols.append(hr)
            add_names += hr_names
        if base_kind in (
            "enetreg2_har5spl",
            "enetreg2_har5rank_spl",
        ):  # + DENSE static magnitude spline x close
            sd, sd_names = _har5_spline_dense_close(Xs, feats0, train_win)
            add_cols.append(sd)
            add_names += sd_names
        if (
            base_kind
            in (
                "enetreg2_rel",
                "enetreg2_relz",
                "enetreg2_relratio",
                "enetreg2_relfine",
            )
        ):  # + ROLLING per-slot vol innovations (new-to-tree signal); encoding/slot vary by base_kind
            _rmode = {
                "enetreg2_rel": ("rankgauss", False),
                "enetreg2_relz": ("robustz", False),
                "enetreg2_relratio": ("ratio", False),
                "enetreg2_relfine": ("rankgauss", True),
            }[base_kind]
            rz, rz_names = _har_rel_innov(
                Xs, feats0, train_win, mode=_rmode[0], fine=_rmode[1]
            )
            add_cols.append(rz)
            add_names += rz_names
        if base_kind in (
            "enetreg2_exogrel",
            "enetreg2_linbest",
        ):  # + LONG-window rolling-relative rank of key exog x close
            xr, xr_names = _exog_rel_innov(Xs, feats0, train_win, bucket)
            add_cols.append(xr)
            add_names += xr_names
        if (
            base_kind == "enetreg2_cumret"
        ):  # + cumulative intraday SIGNED-return path x close (cumrv sibling)
            cr, cr_names = _cumret_close(Xs, feats0, train_win)
            add_cols.append(cr)
            add_names += cr_names
        if (
            base_kind == "enetreg2_cumrvshape"
        ):  # #2 within-day vol-path SHAPE (afternoon share) x close
            cs, cs_names = _cumrv_shape_close(Xs, feats0)
            add_cols.append(cs)
            add_names += cs_names
        if (
            base_kind == "enetreg2_gapopen"
        ):  # #4 overnight directional GAP (cumulative signed return) x open
            go, go_names = _gap_open(Xs, feats0, train_win)
            add_cols.append(go)
            add_names += go_names
        if (
            base_kind == "enetreg2_shape"
        ):  # RICHER path-shape block (vol time-centroid + accumulation profile)
            ps, ps_names = _path_shape_block(Xs, feats0)
            add_cols.append(ps)
            add_names += ps_names
        if (
            base_kind == "enetreg2_shaperel"
        ):  # afternoon-share (shape) + rolling-relative regime: ORTHOG STACK
            cs, cs_names = _cumrv_shape_close(Xs, feats0)
            add_cols.append(cs)
            add_names += cs_names
            rz, rz_names = _har_rel_innov(
                Xs, feats0, train_win, mode="rankgauss", fine=False
            )
            add_cols.append(rz)
            add_names += rz_names
        if (
            base_kind
            in (
                "enetreg2_sig",
                "enetreg2_sigrel",
                "enetreg2_linbest",
            )
        ):  # HIGHER-ORDER within-day path-signature block (lvl 2-3 vol-timing + R-V leverage area)
            sg, sg_names = _path_sig_block(Xs, feats0, train_win)
            add_cols.append(sg)
            add_names += sg_names
        if (
            base_kind == "enetreg2_sigrel"
        ):  # + rolling-relative regime (does the signature STACK on the proven rel factor past 0.12035?)
            rz, rz_names = _har_rel_innov(
                Xs, feats0, train_win, mode="rankgauss", fine=False
            )
            add_cols.append(rz)
            add_names += rz_names
        if (
            base_kind
            in (
                "enetreg2_fri",
                "enetreg2_frirel",
            )
        ):  # WITHIN-WEEK Friday-close path block (value-based replacement for the DOW_4 x voldemand_active artifact)
            fb, fb_names = _friday_week_block(Xs, feats0, train_win)
            add_cols.append(fb)
            add_names += fb_names
        if (
            base_kind == "enetreg2_frirel"
        ):  # + rolling-relative regime (does the Friday-week signal STACK on the proven rel factor?)
            rz, rz_names = _har_rel_innov(
                Xs, feats0, train_win, mode="rankgauss", fine=False
            )
            add_cols.append(rz)
            add_names += rz_names
        if (
            base_kind
            in (
                "enetreg2_ofi",
                "enetreg2_ofirel",
            )
        ):  # NEW DATA CHANNEL: within-day order-flow imbalance (auction-microstructure mechanism)
            ob, ob_names = _ofi_block(Xs, feats0, bucket, train_win)
            add_cols.append(ob)
            add_names += ob_names
        if (
            base_kind == "enetreg2_logsig"
        ):  # SYSTEMATIC log-signature (lvl-3) path basis -- the path-lever-death test (dig w/ FORCE_COLS)
            lg, lg_names = _logsig_block(Xs, feats0, train_win)
            add_cols.append(lg)
            add_names += lg_names
        if (
            base_kind == "enetreg2_ofirel"
        ):  # + rolling-relative regime (does OFI STACK on the proven rel factor?)
            rz, rz_names = _har_rel_innov(
                Xs, feats0, train_win, mode="rankgauss", fine=False
            )
            add_cols.append(rz)
            add_names += rz_names
        if (
            base_kind == "enetreg2_open"
        ):  # + OVERNIGHT cumrv gated to the OPEN (hour==9), the open-side analog
            co, co_names = _cumrv_open(Xs, feats0)
            add_cols.append(co)
            add_names += co_names
        if base_kind == "enetreg2_persist_cs":  # F1 nested clock x state persistence
            pc, pc_names = _persist_clockstate(Xs, feats0, train_win)
            add_cols.append(pc)
            add_names += pc_names
        if base_kind == "enetreg2_persist_casc":  # F2 cross-timescale persistence cascade
            pc, pc_names = _persist_cascade(Xs, feats0, train_win)
            add_cols.append(pc)
            add_names += pc_names
        if base_kind == "enetreg2_persist_dwell":  # F3 regime dwell / run-length
            pc, pc_names = _persist_dwell(Xs, feats0, train_win)
            add_cols.append(pc)
            add_names += pc_names
        if base_kind == "enetreg2_persist_nest":  # F5 nested calendar decomposition
            pc, pc_names = _persist_nested(Xs, feats0, train_win)
            add_cols.append(pc)
            add_names += pc_names
        if base_kind == "enetreg2_persist_pool":  # F4 hierarchical partial-pooling deviations
            pc, pc_names = _persist_pool(Xs, feats0, train_win)
            add_cols.append(pc)
            add_names += pc_names
        Xs = np.ascontiguousarray(np.hstack([Xs] + add_cols))
        feats_aug = list(feats0) + add_names
    # Drop dead (zero-variance) columns before fitting/caching. ~31% of the all_buckets
    # slim matrix is constant always-on availability indicators (_avail/_active flags for
    # series that are never missing -> pinned at 1). They are inert to enet (collinear with
    # the intercept -> zero/absorbed) and to trees (no split on a constant), so dropping
    # them leaves QLIKE unchanged but shrinks the cached matrix, makes the effective
    # feature count honest, and removes the divide-by-zero landmine for the robust-scale
    # pipe. Done post-assembly so the _split_cols/fixed_cols column-index machinery is
    # untouched. (No effect on an already-cached cell; takes effect on the next prep.)
    keep = Xs.min(axis=0) < Xs.max(
        axis=0
    )  # informative iff not constant (scale-free; no 1e-9 floor)
    n_dropped = int((~keep).sum())
    Xs = np.ascontiguousarray(Xs[:, keep])
    if feats_aug is not None:
        feats_aug = [f for f, k in zip(feats_aug, keep) if k]
    t0 = time.time()
    if base_kind == "enetreg2_olam":
        starts, coefs, intercepts = _cadence_enet_online(Xs, y, train_win, refit)
    elif base_kind == "enetreg2_harunpen":
        har_mask = [
            f in set(_HAR_COLS) for f in feats_aug
        ]  # the 6 HAR mains -> UNPENALIZED (OLS)
        starts, coefs, intercepts = _cadence_enet_har_unpen(
            Xs, y, train_win, refit, har_mask
        )
    elif base_kind in (
        "enet",
        "enetreg",
        "enetreg2",
        "enetcum",
        "enetreg2_noclose",
        "enetreg2_ratio",
        "enetreg2_real",
        "enetreg2_opex",
        "enetreg2_realorth",
        "enetreg2_realrank",
        "enetreg2_rankcum",
        "enetreg2_rankcum2",
        "enetreg2_lag",
        "enetreg2_distill",
        "enetreg2_distill_harsq",
        "enetreg2_open",
        "enetreg2_realsqrt",
        "enetreg2_har5rank",
        "enetreg2_canon",
        "enetreg2_canon_spline",
        "enetreg2_canon_scaled",
        "enetreg2_har5spl",
        "enetreg2_har5rank_spl",
        "enetreg2_rel",
        "enetreg2_relz",
        "enetreg2_relratio",
        "enetreg2_relfine",
        "enetreg2_ivmag",
        "enetreg2_linbest",
        "enetreg2_exogrel",
        "enetreg2_exogorder",
        "enetreg2_cumret",
        "enetreg2_cumrvshape",
        "enetreg2_gapopen",
        "enetreg2_shape",
        "enetreg2_shaperel",
        "enetreg2_sig",
        "enetreg2_sigrel",
        "enetreg2_fri",
        "enetreg2_frirel",
        "enetreg2_ofi",
        "enetreg2_ofirel",
        "enetreg2_logsig",
    ):
        starts, coefs, intercepts = _cadence_enet(Xs, y, train_win, refit)
    else:
        starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, alpha, refit)
    ridge_oos = _cadence_ridge_oos(
        Xs, train_win, starts, coefs, intercepts
    )  # base OOS preds (base-agnostic matvec)
    cid = cell_id(model, bucket, twd, alpha, refit, pipe, base_kind)
    d = f"{CACHE_ROOT}/{cid}"
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/Xs.npy", Xs)
    np.save(f"{d}/y.npy", y)
    np.save(f"{d}/base.npy", base)
    np.save(f"{d}/ridge_oos.npy", ridge_oos)
    np.savez(f"{d}/cadence.npz", starts=starts, coefs=coefs, intercepts=intercepts)
    n_int = (
        0
        if feats_aug is None
        else sum(1 for f in feats_aug if "_x_open" in f or "_x_close" in f)
    )
    if (
        feats_aug is not None
    ):  # augmented column names (enetreg) so ebm_interpret can label the interactions
        json.dump(feats_aug, open(f"{d}/feats.json", "w"))
    json.dump(
        {
            "model": model,
            "bucket": bucket,
            "twd": twd,
            "alpha": alpha,
            "refit": refit,
            "pipe": pipe,
            "base_kind": base_kind,
            "train_win": train_win,
            "n": int(len(Xs)),
            "n_refits": int(len(starts)),
            "p": int(Xs.shape[1]),
            "n_dropped_const": n_dropped,
            "n_regime_int": n_int,
        },
        open(f"{d}/cell.json", "w"),
        indent=2,
    )
    print(
        "PREP %s n=%d n_refits=%d base=%s base_alone_qlike=%.5f %.0fs"
        % (
            cid,
            len(Xs),
            len(starts),
            base_kind,
            _qlike(ridge_oos, y, base, train_win),
            time.time() - t0,
        ),
        flush=True,
    )
    if (
        base_kind
        in (
            "enetreg2_canon",
            "enetreg2_canon_spline",
            "enetreg2_canon_scaled",
            "enetreg2_har5spl",
            "enetreg2_har5rank_spl",
            "enetreg2_harunpen",
        )
        and feats_aug is not None
    ):
        mc = np.asarray(coefs, dtype=np.float64).mean(
            axis=0
        )  # mean enet coef across refits (post-drop order)
        print(
            "CANON_COEFS %s (mean across %d refits)" % (base_kind, len(starts)),
            flush=True,
        )
        for nm, c in zip(feats_aug, mc):
            if (
                nm.startswith("band_")
                or "cumrv" in nm
                or "har5_relu" in nm
                or "har5rank" in nm
                or nm in set(_HAR_COLS)
                or "_x_close" in nm
            ):
                print("  COEF %-30s %+.5f" % (nm, c), flush=True)


def do_regime_extra(cid, families_str):
    """Build the regime-persistence features for an EXISTING cell and save them as a SEPARATE
    regime_extra_<families>.npy (NOT folded into the base matrix). They BYPASS the global enet base + the
    global d8 tree and are injected ONLY into the resid_regime EBM (env REGIME_EXTRA=<families>) -- the
    proper home for regime-specific features the global fixed-alpha enet mis-penalizes. families_str =
    comma-joined family tags from {cs, casc, dwell, nest, pool}. Row-aligned 1:1 with the cell's cached Xs."""
    d = f"{CACHE_ROOT}/{cid}"
    cell = json.load(open(f"{d}/cell.json"))
    bucket, train_win = cell["bucket"], int(cell["train_win"])
    pipe = cell.get("pipe", "slim")
    Xs, _y, _base = _load_matrix(bucket, train_win, pipe)
    feats0 = json.load(open(f"results/covid_imp_rank/{bucket}/meta.json"))["feats"]
    families = (
        list(_PERSIST_BUILDERS) if families_str == "all" else [f for f in families_str.split(",") if f]
    )
    bad = [f for f in families if f not in _PERSIST_BUILDERS]
    if bad:
        raise ValueError(f"unknown persist families {bad}; valid = {sorted(_PERSIST_BUILDERS)}")
    re, names = _build_persist(Xs, feats0, train_win, families)
    np.save(f"{d}/regime_extra_{families_str}.npy", re)
    json.dump(names, open(f"{d}/regime_extra_{families_str}_feats.json", "w"))
    print(
        f"REGIME_EXTRA {cid} fams={families_str} cols={re.shape[1]} names={names}",
        flush=True,
    )


def load_cache(cid):
    """Load the per-cell amortized cache ONCE (worker reuses across trials)."""
    d = f"{CACHE_ROOT}/{cid}"
    cad = np.load(f"{d}/cadence.npz")
    out = {
        "cell": json.load(open(f"{d}/cell.json")),
        "Xs": np.load(f"{d}/Xs.npy"),
        "y": np.load(f"{d}/y.npy"),
        "base": np.load(f"{d}/base.npy"),
        "ridge_oos": np.load(f"{d}/ridge_oos.npy"),
        "starts": cad["starts"],
        "coefs": cad["coefs"],
        "intercepts": cad["intercepts"],
    }
    if os.path.exists(
        f"{d}/masks.npy"
    ):  # per-cadence-block enet survivor masks (for arm=resid_subset)
        out["masks"] = np.load(f"{d}/masks.npy")
    if os.path.exists(
        f"{d}/prunable.npy"
    ):  # safe-prune (224-signalless) mask (for arm=resid_pruned)
        out["prunable"] = np.load(f"{d}/prunable.npy")
    # Live availability-indicator mask (arm=resid_subset_ind): _avail/_active columns that VARY.
    # enet-survivor selection (resid_subset) filters these out (~5% survival) because their
    # signal is interaction-only (zero linear main effect), so the residual tree never sees the
    # event channel. This mask lets resid_subset_ind union them back in to TEST that channel.
    out["live_ind"] = None
    out["cov_mask"] = None
    out["feats"] = None  # aligned column names (for arm=resid_regime's hour-gate index)
    out["force_mask"] = (
        None  # FORCE_COLS env: named columns unioned into the tree/EBM masks
    )
    # REGIME_EXTRA=<tag>: load the SEPARATE regime-persistence array regime_extra_<tag>.npy (built by
    # `regime_extra`), injected into the resid_regime EBM only -- bypasses the global base + global tree.
    out["regime_extra"] = None
    _re = os.environ.get("REGIME_EXTRA", "")
    if _re:
        _rep = f"{d}/regime_extra_{_re}.npy"
        if os.path.exists(_rep):
            out["regime_extra"] = np.ascontiguousarray(np.load(_rep), dtype=np.float64)
        else:
            raise FileNotFoundError(f"REGIME_EXTRA={_re} but {_rep} missing (run `regime_extra {cid} {_re}`)")
    # REGARDLESS of L1 survival -- tests a purely-nonlinear feature that the enet zeros (0 linear main
    # effect -> 0/407 mask survival -> tree/EBM never see it -> byte-identical to base = a fake null).
    try:
        # feats.json (augmented, aligned with the cached Xs) is written for enetreg cells; the
        # raw covid_imp_rank meta.json only aligns for non-augmented cells.
        cell_feats = f"{d}/feats.json"
        feats = (
            json.load(open(cell_feats))
            if os.path.exists(cell_feats)
            else json.load(
                open(f"results/covid_imp_rank/{out['cell']['bucket']}/meta.json")
            )["feats"]
        )
        if len(feats) == out["Xs"].shape[1]:
            out["feats"] = feats
            # split on comma/colon/space -- a comma value can't pass sbatch --export (it splits it),
            # so callers use colon-separated FORCE_COLS=ofi_net:ofi_absnet through --export.
            _fc = (
                os.environ.get("FORCE_COLS", "")
                .replace(",", " ")
                .replace(":", " ")
                .split()
            )
            if _fc:
                out["force_mask"] = np.array([f in set(_fc) for f in feats])
            xs = out["Xs"]
            isind = np.array([("_avail" in f or "_active" in f) for f in feats])
            varies = xs.min(axis=0) < xs.max(
                axis=0
            )  # non-constant (scale-free; no 1e-9 floor)
            out["live_ind"] = isind & varies
            # coverage-artifact indicators: availability flags that are ~all-zero in the first decile
            # then turn on later = a data-AVAILABILITY step (e.g. voldemand started being recorded
            # mid-sample), not signal. Drop candidates for arm=resid_subset_nocov.
            n0 = max(1, len(xs) // 10)
            early_const = xs[:n0].std(axis=0) < 1e-9
            out["cov_mask"] = isind & varies & early_const
    except Exception:
        pass
    return out


def _preds(cache, arm, cfg):
    """OOS preds for one config. arm in {residualized, raw_tree, ridge_alone}."""
    c = cache
    tw = c["cell"]["train_win"]
    model = c["cell"]["model"]
    if arm == "ridge_alone":
        return c["ridge_oos"]
    mk = _tree_factory(model, cfg)
    if arm == "raw_tree":  # tree on y directly (control); ridge base zeroed
        return _residualized_preds(
            c["Xs"],
            c["y"],
            tw,
            np.zeros_like(c["ridge_oos"]),
            c["starts"],
            np.zeros_like(c["coefs"]),
            np.zeros_like(c["intercepts"]),
            mk,
        )
    return _residualized_preds(
        c["Xs"],
        c["y"],
        tw,
        c["ridge_oos"],
        c["starts"],
        c["coefs"],
        c["intercepts"],
        mk,
    )


def score(cache, arm, cfg):
    """QLIKE for one config against a preloaded cache."""
    c = cache
    return _qlike(_preds(cache, arm, cfg), c["y"], c["base"], c["cell"]["train_win"])


def run_to_csv(cid, arm, cfg, out_csv):
    """Score one config and write a results.csv (true_raw/pred_raw) for the campaign tell."""
    import pandas as pd

    c = load_cache(cid)
    tw = c["cell"]["train_win"]
    yo, bo = c["y"][tw:], c["base"][tw:]
    pr, tr = apply_duan_smearing(_preds(c, arm, cfg), yo, bo)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame({"true_raw": tr, "pred_raw": pr}).to_csv(out_csv, index=False)
    return out_csv


def preds_chunk(cache, arm, cfg, blk0, blk1):
    """Residualized OOS preds for the rows covered by cadence blocks [blk0, blk1).

    Chunks partition the cadence blocks, so each tree-refit point is fit in exactly ONE
    chunk (no repeated work) and the trial's full preds are the ordered concat of chunks.
    Returns (k0, k1, preds) with k = t - train_win (OOS index)."""
    c = cache
    tw = c["cell"]["train_win"]
    model = c["cell"]["model"]
    n = len(c["Xs"])
    starts = c["starts"]
    k0 = int(starts[blk0]) - tw
    k1 = (int(starts[blk1]) if blk1 < len(starts) else n) - tw
    if arm == "ridge_alone":
        return k0, k1, np.array(c["ridge_oos"][k0:k1], copy=True)
    if arm == "resid_regime":
        # Hero B regime cascade, per cadence block i:
        #   out = base_oos + GLOBAL_tree(survivors) + gated REGIME_tree(leftover, h16-19 only)
        # GLOBAL stage = Hero-A winning XGB (env GLOBAL_CFG) on the resid_subset survivors masks[i],
        # fit on r1 = y_train - base_train (the residualized target). REGIME stage = EBM (env
        # REGIME_CFG) on the LEFTOVER r2 = r1 - global(Xtr), trained on h16-19 TRAIN rows only and
        # predicting ZERO outside h16-19. With REGIME_CFG empty ({} / unset) the regime stage is
        # skipped, so this arm reproduces the single-pass resid_subset number (SANITY invariant --
        # given the resid_subset run uses the SAME xgb cfg as GLOBAL_CFG on an xgb cell).
        if "masks" not in c:
            raise KeyError(
                "resid_regime needs per-block enet survivor masks (run enet_masks first)"
            )
        if c.get("feats") is None:
            raise KeyError(
                "resid_regime needs aligned feats (cell feats.json) for the hour gate"
            )
        out = np.array(c["ridge_oos"][k0:k1], copy=True)
        masks = c["masks"]
        fm = c.get(
            "force_mask"
        )  # FORCE_COLS: union into survivors so the tree/EBM see them
        hr = c["Xs"][:, c["feats"].index("hour")]
        re = c.get(
            "regime_extra"
        )  # REGIME_EXTRA: regime-persistence features injected into the EBM ONLY -- they BYPASS the
        # global enet base (unchanged ridge_oos/coefs) and the global d8 tree (g fits survivors only),
        # entering at the regime stage where there is no global alpha to mis-penalize them. None -> off.
        gcfg = json.loads(os.environ.get("GLOBAL_CFG", "{}"))
        rcfg = json.loads(os.environ.get("REGIME_CFG", "{}"))
        # regime-stage learner: default EBM (interpretable); REGIME_MODEL=xgb -> tuned XGB for PURE
        # POWER (drops the EBM additive/pairwise constraint = the QLIKE ceiling, no interpretability).
        rmodel = os.environ.get("REGIME_MODEL", "ebm")
        regime_full = os.environ.get("REGIME_FULL", "0") not in ("", "0")  # FREE the gate (#1): fit/predict
        # on ALL hours so the learned MoE gate discovers the regime, vs the hand h16-19 pre-gate (default).
        # REGIME_INVERT (#3, structural-starvation test): swap the cascade order to REGIME-then-global, so
        # the soft-routing MoE gets FIRST crack at the un-starved r1 = y - base and d8 only soaks up its
        # leftover. Default (off) = global-then-regime (d8 eats the X-routable structure -> gate starved).
        invert = os.environ.get("REGIME_INVERT", "0") not in ("", "0")
        # REGIME_NOGLOBAL: drop the d8 global stage entirely -> base + regime ONLY (the purest un-starved
        # cascade: additive base + gate-as-sole-regime, no hard router competing). Only honored with invert.
        skip_global = os.environ.get("REGIME_NOGLOBAL", "0") not in ("", "0")
        hr_idx = c["feats"].index("hour")
        mk_g = _tree_factory("xgb", gcfg)
        for i in range(blk0, blk1):
            t_r = int(starts[i])
            cols = masks[i] if fm is None else (masks[i] | fm)
            Xtr = c["Xs"][t_r - tw : t_r]
            t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
            r1 = c["y"][t_r - tw : t_r] - (Xtr @ c["coefs"][i] + c["intercepts"][i])
            if invert:  # REGIME FIRST on the un-starved r1 (gate's first crack), then d8 mops up the leftover
                Xblk = c["Xs"][t_r:t_end][:, cols]
                m_tr = _close_mask(hr[t_r - tw : t_r])
                pe_tr = np.zeros_like(r1)
                if rcfg and (regime_full or int(m_tr.sum()) >= 50):
                    cols_e = cols
                    if regime_full:
                        cols_e = cols.copy()
                        cols_e[hr_idx] = True
                    Xtr_e = Xtr[:, cols_e]
                    Xblk_e = c["Xs"][t_r:t_end][:, cols_e]
                    if re is not None:
                        Xtr_e = np.hstack([Xtr_e, re[t_r - tw : t_r]])
                        Xblk_e = np.hstack([Xblk_e, re[t_r:t_end]])
                    e = _tree_factory(rmodel, rcfg)()
                    if regime_full:
                        e.fit(Xtr_e, r1)
                        out[t_r - tw - k0 : t_end - tw - k0] += e.predict(Xblk_e).ravel()
                        pe_tr = e.predict(Xtr_e).ravel()
                    else:
                        e.fit(Xtr_e[m_tr], r1[m_tr])
                        pe = e.predict(Xblk_e).ravel()
                        pe[~_close_mask(hr[t_r:t_end])] = 0.0
                        out[t_r - tw - k0 : t_end - tw - k0] += pe
                        pe_tr = e.predict(Xtr_e).ravel()
                        pe_tr[~m_tr] = 0.0
                if not skip_global:  # d8 soaks up the regime leftover (off -> base + regime only)
                    g = mk_g()
                    g.fit(Xtr[:, cols], r1 - pe_tr)
                    out[t_r - tw - k0 : t_end - tw - k0] += g.predict(Xblk).ravel()
                continue
            g = mk_g()
            g.fit(Xtr[:, cols], r1)
            Xblk = c["Xs"][t_r:t_end][:, cols]
            out[t_r - tw - k0 : t_end - tw - k0] += g.predict(Xblk).ravel()
            if (
                not rcfg
            ):  # regime stage disabled -> single-pass resid_subset (sanity invariant)
                continue
            m_tr = _close_mask(hr[t_r - tw : t_r])
            if (
                not regime_full and int(m_tr.sum()) < 50
            ):  # too few close/AH train rows -> skip regime this block (predict 0)
                continue
            r2 = r1 - g.predict(Xtr[:, cols]).ravel()
            cols_e = cols
            if regime_full:  # add `hour` so the freed gate can route on the clock itself
                cols_e = cols.copy()
                cols_e[hr_idx] = True
            Xtr_e = Xtr[:, cols_e]
            Xblk_e = c["Xs"][t_r:t_end][:, cols_e]
            if re is not None:  # regime model also sees the persistence extras (global g unaffected above)
                Xtr_e = np.hstack([Xtr_e, re[t_r - tw : t_r]])
                Xblk_e = np.hstack([Xblk_e, re[t_r:t_end]])
            if rmodel == "ebm_mtfm":
                # EBM (+) multi-task-FM ENSEMBLE: the binned-bagged EBM on r2 plus an un-starved multi-task
                # FM (aux head predicts r1, the pre-d8 residual). They are DIVERSE (pre-check corr ~0.34) so
                # the weighted average can beat the EBM alone. ENS_W = weight on the EBM; MT_* = the FM knobs.
                from src.models.regime_moe import MultiTaskFM

                ens_w = float(os.environ.get("ENS_W", "0.4"))
                # un-starve the ANTICIPATION: MT_GATE=ghat -> per-row aux weight gated by |ghat|=|r1-r2|
                # (d8's bite, the taken structure made explicit) — spend the aux where d8 took a big bite.
                gate_kind = os.environ.get("MT_GATE", "uniform")
                aux_hi = float(os.environ.get("MT_AUXHI", "0.6"))
                ghat = r1 - r2  # = g.predict(Xtr[:, cols]); causal, available at train

                def _auxw(sel):
                    if gate_kind != "ghat":
                        return None
                    b = np.abs(ghat[sel])
                    return np.where(b > np.median(b), aux_hi, 0.0).astype(np.float32)

                ebm = _tree_factory("ebm", json.loads(os.environ.get("EBM_CFG", "{}")))()
                mt = MultiTaskFM(
                    rank=int(os.environ.get("MT_RANK", "4")),
                    n_bags=int(os.environ.get("MT_NBAGS", "8")),
                    epochs=int(os.environ.get("MT_EPOCHS", "250")),
                    aux_weight=float(os.environ.get("MT_AUXW", "0.3")),
                    weight_decay=float(os.environ.get("MT_WD", "0.1")),
                )
                if regime_full:
                    ebm.fit(Xtr_e, r2)
                    mt.fit(Xtr_e, r2, r1, aux_w=_auxw(slice(None)))
                    pe = (ens_w * ebm.predict(Xblk_e) + (1 - ens_w) * mt.predict(Xblk_e)).ravel()
                else:
                    ebm.fit(Xtr_e[m_tr], r2[m_tr])
                    mt.fit(Xtr_e[m_tr], r2[m_tr], r1[m_tr], aux_w=_auxw(m_tr))
                    pe = (ens_w * ebm.predict(Xblk_e) + (1 - ens_w) * mt.predict(Xblk_e)).ravel()
                    pe[~_close_mask(hr[t_r:t_end])] = 0.0
            else:
                e = _tree_factory(rmodel, rcfg)()
                if regime_full:  # fit/predict on ALL hours; the learned gate discovers WHERE the regime is
                    e.fit(Xtr_e, r2)
                    pe = e.predict(Xblk_e).ravel()
                else:  # hand pre-gate: fit on h16-19 train rows, predict gated to h16-19
                    e.fit(Xtr_e[m_tr], r2[m_tr])
                    pe = e.predict(Xblk_e).ravel()
                    pe[~_close_mask(hr[t_r:t_end])] = 0.0
            out[t_r - tw - k0 : t_end - tw - k0] += pe
        return k0, k1, out
    raw = arm == "raw_tree"
    out = np.zeros(k1 - k0) if raw else np.array(c["ridge_oos"][k0:k1], copy=True)
    if (
        arm == "resid_subset"
    ):  # tree sees only the block's rolling enet survivors (~120)
        fm = c.get("force_mask")  # FORCE_COLS: union forced cols into the survivor set

        def colsel(i):
            return c["masks"][i] if fm is None else (c["masks"][i] | fm)
    elif (
        arm == "resid_subset_ind"
    ):  # survivors UNION live availability indicators (event channel)
        li = c["live_ind"]

        def colsel(i):
            return c["masks"][i] if li is None else (c["masks"][i] | li)
    elif (
        arm == "resid_subset_nocov"
    ):  # survivors MINUS coverage-artifact indicators (voldemand-type
        cov = c.get(
            "cov_mask"
        )  # availability steps) -> tests if the EBM leans on a data artifact

        def colsel(i):
            return c["masks"][i] if cov is None else (c["masks"][i] & ~cov)
    elif (
        arm == "resid_pruned"
    ):  # tree sees all-but-the-signalless (the 224-indicator prune)
        keep = ~c["prunable"]

        def colsel(i):
            return keep
    else:  # residualized / raw_tree: tree sees all features

        def colsel(i):
            return slice(None)

    mk = _tree_factory(model, cfg)
    for i in range(blk0, blk1):
        t_r = int(starts[i])
        Xtr = c["Xs"][t_r - tw : t_r]
        cols = colsel(i)
        r_train = (
            c["y"][t_r - tw : t_r]
            if raw
            else c["y"][t_r - tw : t_r] - (Xtr @ c["coefs"][i] + c["intercepts"][i])
        )
        tree = mk()
        tree.fit(Xtr[:, cols], r_train)
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        out[t_r - tw - k0 : t_end - tw - k0] += tree.predict(
            c["Xs"][t_r:t_end][:, cols]
        ).ravel()
    return k0, k1, out


def run_chunk_to_csv(cid, arm, cfg, blk0, blk1, out_csv):
    """Score one cadence-block chunk; write its ADJ-space slice (k, pred_adj, y_true, base).
    Smearing is DEFERRED to collect-time (global) so the concatenated QLIKE equals the whole-
    backtest QLIKE — per-chunk smearing would use a per-chunk retransformation factor."""
    import pandas as pd

    c = load_cache(cid)
    tw = c["cell"]["train_win"]
    k0, k1, preds = preds_chunk(c, arm, cfg, blk0, blk1)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(
        {
            "k": np.arange(k0, k1),
            "pred_adj": preds,
            "y_true": c["y"][tw + k0 : tw + k1],
            "base": c["base"][tw + k0 : tw + k1],
        }
    ).to_csv(out_csv, index=False)
    return out_csv


def do_enet_masks(cid, alpha=0.001, l1=0.2):
    """Augment a built cache with (1) per-cadence-block rolling enet survivor masks (arm=resid_subset)
    and (2) the safe-prune mask (dead to BOTH enet-all-windows AND a deep residual tree). Warm-started
    coordinate descent chains the cadence enet fits so the rolling refits are cheap."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import ElasticNet, Ridge

    d = f"{CACHE_ROOT}/{cid}"
    c = load_cache(cid)
    Xs, y, starts = c["Xs"], c["y"], c["starts"]
    tw = c["cell"]["train_win"]
    p = Xs.shape[1]
    en = ElasticNet(alpha=alpha, l1_ratio=l1, warm_start=True, max_iter=1000, tol=1e-3)
    masks = np.zeros((len(starts), p), dtype=bool)
    t0 = time.time()
    for i, t_r in enumerate(starts):
        t_r = int(t_r)
        en.fit(Xs[t_r - tw : t_r], y[t_r - tw : t_r])
        masks[i] = en.coef_ != 0.0
    np.save(f"{d}/masks.npy", masks)
    rg = Ridge(alpha=float(c["cell"]["alpha"])).fit(Xs, y)
    lg = LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        num_leaves=127,
        learning_rate=0.05,
        min_child_samples=20,
        n_jobs=8,
        verbose=-1,
        importance_type="split",
    ).fit(Xs, y - rg.predict(Xs))
    prunable = (masks.sum(axis=0) == 0) & (lg.feature_importances_ == 0)
    np.save(f"{d}/prunable.npy", prunable)
    print(
        "ENETMASKS %s avg_survivors=%.0f/%d prunable=%d %.0fs"
        % (cid, masks.sum(axis=1).mean(), p, int(prunable.sum()), time.time() - t0),
        flush=True,
    )


def do_chunk_task(cid, arm, idx, nchunks, label=""):
    """One chunk of a FIXED-config chunked eval (SLURM/SGE array task). Tree cfg via env TREE_CFG.
    Optional label suffixes the output dir as {arm}_{label} so several fixed configs on the SAME
    cell+arm (e.g. xgb depth 2 vs 6) don't overwrite each other; colsel still keys off bare arm."""
    import math

    nb = int(json.load(open(f"{CACHE_ROOT}/{cid}/cell.json"))["n_refits"])
    sz = max(1, math.ceil(nb / nchunks))
    ranges = [(b, min(b + sz, nb)) for b in range(0, nb, sz)]
    if idx >= len(ranges):
        print(
            "CHUNK idx=%d >= effective_chunks=%d (skip)" % (idx, len(ranges)),
            flush=True,
        )
        return
    blk0, blk1 = ranges[idx]
    cfg = json.loads(os.environ.get("TREE_CFG", "{}"))
    sub = f"{arm}_{label}" if label else arm
    out = run_chunk_to_csv(
        cid,
        arm,
        cfg,
        blk0,
        blk1,
        f"results/resid_ab/{cid}/{sub}/chunk_{blk0}_{blk1}.csv",
    )
    print(
        "CHUNK %s %s idx=%d blk[%d:%d] -> %s" % (cid, sub, idx, blk0, blk1, out),
        flush=True,
    )


def preds_chunk_ggrid(cache, blk0, blk1):
    """EBM (+) MTFM ensemble GRID with the EBM-INVARIANT parts CACHED. omega (ensemble weight) and the aux
    variant do NOT change the EBM, so per cadence block we fit base / d8 / EBM and the MTFM aux-variants
    ONCE, then emit the full (aux x omega) grid by cheap re-averaging -- instead of refitting the (expensive)
    EBM once per config. resid_regime, h16-19 pre-gate. Returns (k0, k1, {grid_key: preds})."""
    from src.models.regime_moe import MultiTaskFM

    c = cache
    tw = c["cell"]["train_win"]
    n = len(c["Xs"])
    starts = c["starts"]
    k0 = int(starts[blk0]) - tw
    k1 = (int(starts[blk1]) if blk1 < len(starts) else n) - tw
    if "masks" not in c or c.get("feats") is None:
        raise KeyError("ggrid needs enet survivor masks + aligned feats")
    masks = c["masks"]
    fm = c.get("force_mask")
    hr = c["Xs"][:, c["feats"].index("hour")]
    gcfg = json.loads(os.environ.get("GLOBAL_CFG", "{}"))
    ebm_cfg = json.loads(os.environ.get("EBM_CFG", "{}"))
    mk_g = _tree_factory("xgb", gcfg)
    aux_grid = os.environ.get("MT_AUX_GRID", "uniform,ghat").split(",")
    w_grid = [float(x) for x in os.environ.get("MT_W_GRID", "0.2,0.3,0.4").split(",")]
    aux_hi = float(os.environ.get("MT_AUXHI", "0.6"))
    mt_kw = dict(
        rank=int(os.environ.get("MT_RANK", "4")),
        n_bags=int(os.environ.get("MT_NBAGS", "8")),
        epochs=int(os.environ.get("MT_EPOCHS", "250")),
        aux_weight=float(os.environ.get("MT_AUXW", "0.3")),
        weight_decay=float(os.environ.get("MT_WD", "0.1")),
    )
    keys = [f"a{a}_w{int(round(w * 100)):02d}" for a in aux_grid for w in w_grid]
    base = np.array(c["ridge_oos"][k0:k1], copy=True)
    grid = {kk: np.array(base, copy=True) for kk in keys}
    for i in range(blk0, blk1):
        t_r = int(starts[i])
        cols = masks[i] if fm is None else (masks[i] | fm)
        Xtr = c["Xs"][t_r - tw : t_r]
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        r1 = c["y"][t_r - tw : t_r] - (Xtr @ c["coefs"][i] + c["intercepts"][i])
        g = mk_g()
        g.fit(Xtr[:, cols], r1)  # d8 fit ONCE
        gblk = g.predict(c["Xs"][t_r:t_end][:, cols]).ravel()
        for kk in keys:
            grid[kk][t_r - tw - k0 : t_end - tw - k0] += gblk
        m_tr = _close_mask(hr[t_r - tw : t_r])
        if int(m_tr.sum()) < 50:
            continue
        r2 = r1 - g.predict(Xtr[:, cols]).ravel()
        Xtr_e = Xtr[:, cols]
        Xblk_e = c["Xs"][t_r:t_end][:, cols]
        ghat = r1 - r2  # d8's bite (for the |ghat|-gate)
        closeb = _close_mask(hr[t_r:t_end])
        ebm = _tree_factory("ebm", ebm_cfg)()  # FIT ONCE — the expensive, omega/aux-invariant part
        ebm.fit(Xtr_e[m_tr], r2[m_tr])
        pe_ebm = ebm.predict(Xblk_e).ravel()
        pe_mt = {}
        for a in aux_grid:  # one MTFM per aux variant (cheap vs the EBM), cached across omega
            aw = None
            if a == "multi":  # SFV multi-objective aux stack: r1 (un-starve) + |r2| (magnitude) + y (raw vol)
                yaux = np.stack([r1[m_tr], np.abs(r2[m_tr]), c["y"][t_r - tw : t_r][m_tr]], 1).astype(np.float32)
            else:
                yaux = r1[m_tr]
                if a == "ghat":
                    b = np.abs(ghat[m_tr])
                    aw = np.where(b > np.median(b), aux_hi, 0.0).astype(np.float32)
            pe_mt[a] = MultiTaskFM(**mt_kw).fit(Xtr_e[m_tr], r2[m_tr], yaux, aux_w=aw).predict(Xblk_e).ravel()
        for a in aux_grid:  # cheap re-average for every omega
            for w in w_grid:
                pe = w * pe_ebm + (1 - w) * pe_mt[a]
                pe[~closeb] = 0.0
                grid[f"a{a}_w{int(round(w * 100)):02d}"][t_r - tw - k0 : t_end - tw - k0] += pe
    return k0, k1, grid


def do_chunk_task_ggrid(cid, idx, nchunks, base_label):
    """Grid array task: ONE task emits ALL (aux x omega) outputs with the EBM fit ONCE per block. Writes
    one CSV per grid point into resid_regime_{base_label}_{key}; collect each with chunk_collect."""
    import math

    import pandas as pd

    c = load_cache(cid)
    tw = c["cell"]["train_win"]
    nb = int(json.load(open(f"{CACHE_ROOT}/{cid}/cell.json"))["n_refits"])
    sz = max(1, math.ceil(nb / nchunks))
    ranges = [(b, min(b + sz, nb)) for b in range(0, nb, sz)]
    if idx >= len(ranges):
        print("CHUNK idx=%d >= %d (skip)" % (idx, len(ranges)), flush=True)
        return
    blk0, blk1 = ranges[idx]
    k0, k1, grid = preds_chunk_ggrid(c, blk0, blk1)
    for kk, preds in grid.items():
        sub = f"resid_regime_{base_label}_{kk}"
        out_csv = f"results/resid_ab/{cid}/{sub}/chunk_{blk0}_{blk1}.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(
            {
                "k": np.arange(k0, k1),
                "pred_adj": preds,
                "y_true": c["y"][tw + k0 : tw + k1],
                "base": c["base"][tw + k0 : tw + k1],
            }
        ).to_csv(out_csv, index=False)
        print("CHUNK %s %s idx=%d blk[%d:%d]" % (cid, sub, idx, blk0, blk1), flush=True)


def preds_chunk_adaptive(cache, blk0, blk1):
    """Per-cadence-block ADAPTIVE omega: at each block, inner-split the close train rows, fit EBM/MTFM on the
    inner-fit, SELECT omega on the PURGED inner-val (AW_MODE=scalar via tune_hparam, or =context via the
    context-attention context_omega), then refit EBM/MTFM on the full train and blend the OOS block at the
    selected omega. omega is thus tuned per step on OOS val -- not fixed, not in-sample (which collapses).
    AW_MODE = fixed | scalar | context. resid_regime, h16-19. Returns (k0, k1, preds)."""
    from sklearn.cluster import KMeans

    from src.evaluation.feature_cv import context_omega, inner_split, tune_hparam
    from src.evaluation.metrics import apply_duan_smearing as _smear
    from src.models.regime_moe import MultiTaskFM

    c = cache
    tw = c["cell"]["train_win"]
    n = len(c["Xs"])
    starts = c["starts"]
    k0 = int(starts[blk0]) - tw
    k1 = (int(starts[blk1]) if blk1 < len(starts) else n) - tw
    masks = c["masks"]
    fm = c.get("force_mask")
    hr = c["Xs"][:, c["feats"].index("hour")]
    mk_g = _tree_factory("xgb", json.loads(os.environ.get("GLOBAL_CFG", "{}")))
    mk_ebm = _tree_factory("ebm", json.loads(os.environ.get("EBM_CFG", "{}")))
    mode = os.environ.get("AW_MODE", "scalar")
    fixedw = float(os.environ.get("AW_FIXEDW", "0.8"))
    kanch = int(os.environ.get("AW_K", "4"))
    oms = np.round(np.linspace(0.5, 1.0, 11), 2)
    mt_kw = dict(
        rank=int(os.environ.get("MT_RANK", "4")),
        n_bags=int(os.environ.get("MT_NBAGS", "8")),
        epochs=int(os.environ.get("MT_EPOCHS", "250")),
        aux_weight=float(os.environ.get("MT_AUXW", "0.3")),
        weight_decay=float(os.environ.get("MT_WD", "0.1")),
    )
    aux_hi = float(os.environ.get("MT_AUXHI", "0.6"))
    # AW_CV_CFG=1 also CV-selects the MTFM config (aux_weight/gate/rank/wd) per step on the inner-val; the
    # cheap MTFM-side HPs help (validated); base-alpha / d8-depth / EBM-cfg do NOT (high-variance selection).
    cfg_grid: list = (
        [
            dict(aux_weight=0.3, gate="u", rank=4, weight_decay=0.1),
            dict(aux_weight=0.1, gate="u", rank=4, weight_decay=0.1),
            dict(aux_weight=0.6, gate="u", rank=4, weight_decay=0.1),
            dict(aux_weight=0.3, gate="g", rank=4, weight_decay=0.1),
            dict(aux_weight=0.3, gate="u", rank=8, weight_decay=0.3),
            dict(aux_weight=0.3, gate="u", rank=2, weight_decay=0.5),
        ]
        if os.environ.get("AW_CV_CFG", "0") == "1"
        else [None]
    )

    def _fit_mt(cfg, X, t2, t1, ght):
        kw = dict(mt_kw)
        aw = None
        if cfg is not None:
            kw.update(rank=cfg["rank"], aux_weight=cfg["aux_weight"], weight_decay=cfg["weight_decay"])
            if cfg["gate"] == "g" and ght is not None:
                aw = np.where(ght > np.median(ght), aux_hi, 0.0).astype(np.float32)
        return MultiTaskFM(**kw).fit(X, t2, t1, aux_w=aw)

    out = np.array(c["ridge_oos"][k0:k1], copy=True)
    for i in range(blk0, blk1):
        t_r = int(starts[i])
        cols = masks[i] if fm is None else (masks[i] | fm)
        Xtr = c["Xs"][t_r - tw : t_r]
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        r1 = c["y"][t_r - tw : t_r] - (Xtr @ c["coefs"][i] + c["intercepts"][i])
        g = mk_g()
        g.fit(Xtr[:, cols], r1)
        gblk = g.predict(c["Xs"][t_r:t_end][:, cols]).ravel()
        out[t_r - tw - k0 : t_end - tw - k0] += gblk  # base + d8 (the non-regime part)
        m_tr = _close_mask(hr[t_r - tw : t_r])
        if int(m_tr.sum()) < 200:
            continue
        r2 = r1 - g.predict(Xtr[:, cols]).ravel()
        Xtr_e, Xblk_e = Xtr[:, cols], c["Xs"][t_r:t_end][:, cols]
        closeb = _close_mask(hr[t_r:t_end])
        ebm_f = mk_ebm()
        ebm_f.fit(Xtr_e[m_tr], r2[m_tr])  # full-train EBM -> the OOS block prediction
        pe_blk = ebm_f.predict(Xblk_e).ravel()
        ghf = np.abs(g.predict(Xtr_e[m_tr]).ravel())
        if mode == "fixed":
            mt_f = _fit_mt(None, Xtr_e[m_tr], r2[m_tr], r1[m_tr], None)
            pm_blk = mt_f.predict(Xblk_e).ravel()
            w_blk: object = fixedw
        else:
            cidx = np.where(m_tr)[0]
            fi, vi = inner_split(len(cidx), val_frac=0.25, embargo=0.01)
            ifit, ival = cidx[fi], cidx[vi]
            ebm_if = mk_ebm()
            ebm_if.fit(Xtr_e[ifit], r2[ifit])  # inner-fit -> per-step CV on the purged inner-val
            pe_v = ebm_if.predict(Xtr_e[ival]).ravel()
            ytr, btr = c["y"][t_r - tw : t_r], c["base"][t_r - tw : t_r]
            y_v, base_v = ytr[ival], btr[ival]
            off_v = y_v - r2[ival]  # base + d8 on inner-val (= y - r2)
            ghi = np.abs(g.predict(Xtr_e[ifit]).ravel())
            best = None  # CV the MTFM config (cfg_grid) -- each scored at its own best omega on inner-val
            for cfg in cfg_grid:
                pmv = _fit_mt(cfg, Xtr_e[ifit], r2[ifit], r1[ifit], ghi).predict(Xtr_e[ival]).ravel()
                wc, sc = tune_hparam(
                    lambda w, pe=pe_v, pm=pmv: w * pe + (1.0 - w) * pm,
                    list(oms), r2[ival], off_v, y_v, base_v, _smear,
                    n_boot=100, shrink_to=fixedw, shrink_lambda=0.15,
                )
                s = min(sc.values())
                if best is None or s < best[0]:
                    best = (s, cfg, float(wc), pmv)
            _, cfg_star, wg, pm_v = best
            mt_f = _fit_mt(cfg_star, Xtr_e[m_tr], r2[m_tr], r1[m_tr], ghf)  # refit the winning config on full train
            pm_blk = mt_f.predict(Xblk_e).ravel()
            if mode == "scalar":
                w_blk = float(wg)
            else:  # context-attention omega(x): anchors on (|ghat|, hour), CV-fit per anchor, shrunk
                cf = np.column_stack([ghi, hr[t_r - tw : t_r][ifit]])
                cv = np.column_stack([np.abs(g.predict(Xtr_e[ival]).ravel()), hr[t_r - tw : t_r][ival]])
                cb = np.column_stack([np.abs(gblk), hr[t_r:t_end]])
                mu, sd = cf.mean(0), cf.std(0) + 1e-9
                czf, czv, czb = (cf - mu) / sd, (cv - mu) / sd, (cb - mu) / sd
                anch = KMeans(n_clusters=kanch, n_init=4, random_state=0).fit(czf).cluster_centers_
                tau = float(np.median([((czv - a) ** 2).sum(1).mean() for a in anch])) + 1e-9
                w_blk, _ = context_omega(czv, pe_v, pm_v, off_v, y_v, base_v, _smear, czb, anch, oms, tau, float(wg))
        pe = np.asarray(w_blk * pe_blk + (1.0 - np.asarray(w_blk)) * pm_blk).ravel()
        pe[~closeb] = 0.0
        out[t_r - tw - k0 : t_end - tw - k0] += pe
    return k0, k1, out


def do_chunk_task_adaptive(cid, idx, nchunks, base_label):
    """Array task for the adaptive-omega regime: one CSV into resid_regime_{base_label}; collect with chunk_collect."""
    import math

    import pandas as pd

    c = load_cache(cid)
    tw = c["cell"]["train_win"]
    nb = int(json.load(open(f"{CACHE_ROOT}/{cid}/cell.json"))["n_refits"])
    sz = max(1, math.ceil(nb / nchunks))
    ranges = [(b, min(b + sz, nb)) for b in range(0, nb, sz)]
    if idx >= len(ranges):
        print("CHUNK idx=%d >= %d (skip)" % (idx, len(ranges)), flush=True)
        return
    blk0, blk1 = ranges[idx]
    k0, k1, preds = preds_chunk_adaptive(c, blk0, blk1)
    sub = f"resid_regime_{base_label}"
    out_csv = f"results/resid_ab/{cid}/{sub}/chunk_{blk0}_{blk1}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(
        {
            "k": np.arange(k0, k1),
            "pred_adj": preds,
            "y_true": c["y"][tw + k0 : tw + k1],
            "base": c["base"][tw + k0 : tw + k1],
        }
    ).to_csv(out_csv, index=False)
    print("CHUNK %s %s idx=%d blk[%d:%d]" % (cid, sub, idx, blk0, blk1), flush=True)


def do_chunk_collect(cid, arm, label=""):
    """Concat a chunked eval's ADJ-space CSVs, apply GLOBAL Duan smearing, report full-OOS QLIKE.
    label selects the {arm}_{label} output dir written by do_chunk_task (e.g. xgb d2 vs d6)."""
    import glob

    import pandas as pd

    cell = json.load(open(f"{CACHE_ROOT}/{cid}/cell.json"))
    n_oos = int(cell["n"] - cell["train_win"])
    sub = f"{arm}_{label}" if label else arm
    files = sorted(glob.glob(f"results/resid_ab/{cid}/{sub}/chunk_*.csv"))
    df = (
        pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("k").sort_values("k")
    )
    if len(df) != n_oos:
        print(
            "COLLECT %s %s INCOMPLETE %d/%d (%d files)"
            % (cid, sub, len(df), n_oos, len(files)),
            flush=True,
        )
        return
    pr, tr = apply_duan_smearing(
        df["pred_adj"].to_numpy(), df["y_true"].to_numpy(), df["base"].to_numpy()
    )
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    print(
        "COLLECT %s %-16s qlike=%.5f n=%d chunks=%d"
        % (cid, sub, float(np.mean(r - np.log(r) - 1)), len(df), len(files)),
        flush=True,
    )
    if (
        arm == "resid_regime"
    ):  # regime-lift: sub-population QLIKE on the h16-19 close/AH rows (SAME global smearing)
        dcache = f"{CACHE_ROOT}/{cid}"
        feats = (
            json.load(open(f"{dcache}/feats.json"))
            if os.path.exists(f"{dcache}/feats.json")
            else json.load(open(f"results/covid_imp_rank/{cell['bucket']}/meta.json"))[
                "feats"
            ]
        )
        hi = feats.index("hour")
        rows = int(cell["train_win"]) + df["k"].to_numpy()
        hour = np.asarray(np.load(f"{dcache}/Xs.npy", mmap_mode="r")[rows, hi])
        g = _close_mask(hour) & (tr > 0) & (pr > 0)
        rg = tr[g] / pr[g]
        print(
            "COLLECT %s %-16s qlike_h%d-%d=%.5f n=%d"
            % (
                cid,
                sub,
                CLOSE_LO,
                CLOSE_HI,
                float(np.mean(rg - np.log(rg) - 1)),
                int(g.sum()),
            ),
            flush=True,
        )


def do_trial(cid, arm, outname=None):
    cache = load_cache(cid)
    cfg = json.loads(os.environ.get("TREE_CFG", "{}"))
    t0 = time.time()
    q = score(cache, arm, cfg)
    dt = time.time() - t0
    print("TRIAL %s %-12s qlike=%.5f %.0fs" % (cid, arm, q, dt), flush=True)
    if outname:
        os.makedirs("results/resid_tree", exist_ok=True)
        json.dump(
            {"cid": cid, "arm": arm, "qlike": q, "secs": dt, "cfg": cfg},
            open(f"results/resid_tree/{outname}.json", "w"),
            indent=2,
        )
    return q


def do_selfcheck(model, bucket, twd, alpha, refit, nseg, pipe="slim"):
    """Assert amortized == naive MultiStage(Residualizer(Ridge)+tree) on the first NSEG OOS bars."""
    from src.backtest.multi_stage import MultiStageBacktest
    from src.features.transforms.residualizer import Residualizer

    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    seg = train_win + nseg
    Xs, y, base = Xs[:seg], y[:seg], base[:seg]
    cfg = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
    }
    make_tree = _tree_factory(model, cfg)
    # amortized
    ridge_oos = fit_predict_ridge(
        Xs, y, train_win, {"alpha": alpha, "_refit_frequency": 1, "_incremental": True}
    )
    starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, alpha, refit)
    amort = _residualized_preds(
        Xs, y, train_win, ridge_oos, starts, coefs, intercepts, make_tree
    )
    # naive MultiStage
    bt = MultiStageBacktest(
        Residualizer(lambda: Ridge(alpha=alpha)), make_tree, refit_frequency=refit
    )
    naive = bt.run(np.ascontiguousarray(Xs), y, train_win, desc="naive")
    d = float(np.max(np.abs(amort - naive)))
    qa = _qlike(amort, y, base, train_win)
    qn = _qlike(naive, y, base, train_win)
    print(
        "SELFCHECK %s/%s rf%d nseg%d: max|amort-naive|=%.3e  qlike amort=%.6f naive=%.6f  %s"
        % (model, bucket, refit, nseg, d, qa, qn, "OK" if d < 1e-6 else "MISMATCH"),
        flush=True,
    )
    return d


def do_alphascan(model, bucket, twd, pipe, alphas, refit=2000):
    """Cadence-refit ridge_alone QLIKE over a grid of alpha (coarse refit -> fast) to pick alpha*."""
    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    for a in alphas:
        starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, a, refit)
        ro = _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts)
        print(
            "ALPHASCAN %s %s tw%d %s a%g ridge_alone_qlike=%.5f"
            % (model, bucket, twd, pipe, a, _qlike(ro, y, base, train_win)),
            flush=True,
        )


def do_chunkcheck(model, bucket, twd, alpha, refit, pipe, nchunks):
    """Verify chunked == whole: concat of preds_chunk over a CHUNKS-partition equals _preds."""
    import math

    train_win = twd * PERIODS_PER_DAY
    Xs, y, base = _load_matrix(bucket, train_win, pipe)
    starts, coefs, intercepts = _cadence_ridge(Xs, y, train_win, alpha, refit)
    ridge_oos = _cadence_ridge_oos(Xs, train_win, starts, coefs, intercepts)
    cache = {
        "cell": {"train_win": train_win, "model": model},
        "Xs": Xs,
        "y": y,
        "base": base,
        "ridge_oos": ridge_oos,
        "starts": starts,
        "coefs": coefs,
        "intercepts": intercepts,
    }
    cfg = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
    }
    whole = _preds(cache, "residualized", cfg)
    nb = len(starts)
    sz = max(1, math.ceil(nb / nchunks))
    parts = []
    for b0 in range(0, nb, sz):
        k0, _k1, p = preds_chunk(cache, "residualized", cfg, b0, min(b0 + sz, nb))
        parts.append((k0, p))
    cat = np.concatenate([p for _, p in sorted(parts)])
    d = float(np.max(np.abs(whole - cat)))
    print(
        "CHUNKCHECK %s/%s rf%d nchunks%d: max|whole-chunked|=%.3e %s"
        % (model, bucket, refit, nchunks, d, "OK" if d < 1e-9 else "MISMATCH"),
        flush=True,
    )


def do_pcatest(model, bucket, twd, k, nrows):
    """Quick PCA validation on a REAL slice (first NROWS) of the all_buckets walk-forward:
    1. causality  — perturb the LAST exog row, recompute; earlier OOS factors must be identical;
    2. var_expl   — fraction of exog variance the top-K PCs capture on the last window;
    3. QLIKE      — cadence-Ridge (alpha=1) on full slim [HAR,exog,ind] vs PCA-compressed
                    [HAR,K factors,ind]. qp approx qf at K<<n_exog => the cross-section is low-rank."""
    from src.features.transforms.rolling_pca import rolling_pca

    train_win = twd * PERIODS_PER_DAY
    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy")[:nrows], dtype=np.float64)
    y = np.load(f"{b}/y.npy")[:nrows]
    base = np.load(f"{b}/base.npy")[:nrows]
    har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
    Xe = X[:, exog]
    f = rolling_pca(Xe, train_win, k, PCA_REFIT)
    xp = Xe.copy()
    xp[-1] += 1e3
    fp = rolling_pca(xp, train_win, k, PCA_REFIT)
    leak = float(np.max(np.abs(f[train_win:-PCA_REFIT] - fp[train_win:-PCA_REFIT])))
    w = Xe[nrows - train_win : nrows]
    sv = np.linalg.svd(w - w.mean(0), compute_uv=False)
    var_expl = float((sv[:k] ** 2).sum() / (sv**2).sum())

    def ql(xs):
        st, co, ic = _cadence_ridge(
            np.ascontiguousarray(xs), y, train_win, 1.0, PCA_REFIT
        )
        return _qlike(
            _cadence_ridge_oos(np.ascontiguousarray(xs), train_win, st, co, ic),
            y,
            base,
            train_win,
        )

    qf = ql(np.hstack([X[:, har], Xe, X[:, ind]]))
    qp = ql(np.hstack([X[:, har], f, X[:, ind]]))
    print(
        "PCATEST %s/%s tw%d K%d rows%d n_exog=%d: leak=%.3e (%s) var_expl=%.3f | ridge_qlike full=%.5f pca=%.5f (d=%+.5f)"
        % (
            model,
            bucket,
            twd,
            k,
            nrows,
            len(exog),
            leak,
            "CAUSAL" if leak < 1e-9 else "LEAK!",
            var_expl,
            qf,
            qp,
            qp - qf,
        ),
        flush=True,
    )


def do_pcawindowsweep(model, bucket, k_list, pca_window_days, pca_refit=2000):
    """Decouple test: LONG (pca_window_days) PCA basis + SHORT model window. Build factors ONCE
    at max(k_list) (PCs are ordered -> factors[:, :K] is the top-K set), then sweep K x model
    window on the low-dim set. ALL windows scored on a FIXED OOS region (1000d start) for
    fairness (Part-6 FIXED_OOS). Controls: raw slim (no PCA) at {250,1000}d. Ridge base alpha=1."""
    from src.features.transforms.rolling_pca import rolling_pca

    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.load(f"{b}/y.npy")
    base = np.load(f"{b}/base.npy")
    har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
    oos_start = 1000 * PERIODS_PER_DAY  # fixed OOS [oos_start, n) for every window
    Xraw = np.ascontiguousarray(np.hstack([X[:, har], X[:, exog], X[:, ind]]))
    t0 = time.time()
    maxk = max(k_list)
    factors = rolling_pca(
        X[:, exog], pca_window_days * PERIODS_PER_DAY, maxk, pca_refit
    )  # built once
    print(
        "SWEEP built pca-maxK%d (pca_window=%dd, refit=%d) %.0fs; n_exog=%d"
        % (maxk, pca_window_days, pca_refit, time.time() - t0, len(exog)),
        flush=True,
    )

    def ql(Xs, tw_days):
        tw = tw_days * PERIODS_PER_DAY
        st, co, ic = _cadence_ridge(Xs, y, tw, 1.0, 480)
        preds = _cadence_ridge_oos(Xs, tw, st, co, ic)  # indexed from tw
        pr, tr = apply_duan_smearing(
            preds[oos_start - tw :], y[oos_start:], base[oos_start:]
        )
        m = (tr > 0) & (pr > 0)
        r = tr[m] / pr[m]
        return float(np.mean(r - np.log(r) - 1))

    for tw_days in (250, 1000):
        print(
            "SWEEP raw_slim       model_tw%-5dd qlike=%.5f"
            % (tw_days, ql(Xraw, tw_days)),
            flush=True,
        )
    for k in k_list:
        Xpca = np.ascontiguousarray(np.hstack([X[:, har], factors[:, :k], X[:, ind]]))
        for tw_days in (60, 125, 250, 500, 1000):
            print(
                "SWEEP pca%-2d_w%dd model_tw%-5dd qlike=%.5f"
                % (k, pca_window_days, tw_days, ql(Xpca, tw_days)),
                flush=True,
            )


def do_pca_alpha_compare(
    model, bucket, pca_window_days=1000, pca_refit=2000, refit=4000
):
    """Each feature set at its OWN alpha*: raw-slim-1000d vs pcaK-500d (K=10/20/40), alpha-scanned,
    fair fixed-OOS. Settles 'does PCA beat raw' at PROPER shrinkage (the alpha=1 sweep was unfair:
    raw wants ~100). refit coarse (rank-robust) for speed; absolute level shifts, ordering holds."""
    from src.features.transforms.rolling_pca import rolling_pca

    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.load(f"{b}/y.npy")
    base = np.load(f"{b}/base.npy")
    har, exog, ind = _split_cols(json.load(open(f"{b}/meta.json"))["feats"])
    oos_start = 1000 * PERIODS_PER_DAY
    factors = rolling_pca(X[:, exog], pca_window_days * PERIODS_PER_DAY, 40, pca_refit)

    def ql(Xs, tw_days, alpha):
        tw = tw_days * PERIODS_PER_DAY
        Xs = np.ascontiguousarray(Xs)
        st, co, ic = _cadence_ridge(Xs, y, tw, alpha, refit)
        preds = _cadence_ridge_oos(Xs, tw, st, co, ic)
        pr, tr = apply_duan_smearing(
            preds[oos_start - tw :], y[oos_start:], base[oos_start:]
        )
        m = (tr > 0) & (pr > 0)
        r = tr[m] / pr[m]
        return float(np.mean(r - np.log(r) - 1))

    Xraw = np.hstack([X[:, har], X[:, exog], X[:, ind]])
    for a in (10, 100, 1000):
        print(
            "ACMP raw_slim tw1000 a%-5g qlike=%.5f" % (a, ql(Xraw, 1000, a)), flush=True
        )
    for k in (10, 20, 40):
        Xp = np.hstack([X[:, har], factors[:, :k], X[:, ind]])
        for a in (1, 10, 100):
            print(
                "ACMP pca%-2d   tw500  a%-5g qlike=%.5f" % (k, a, ql(Xp, 500, a)),
                flush=True,
            )


def do_signalless_scan(bucket, alpha=0.001, l1=0.2, nwin=12):
    """ABSOLUTELY signalless exog = enet zeros it in EVERY rolling window (linearly dead) AND a
    deep lgbm NEVER splits on it in the residual (nonlinearly dead). Reports raw-exog families
    that are FULLY dead (all derived features prunable) — those raw exog can be dropped."""
    import re

    from lightgbm import LGBMRegressor
    from sklearn.linear_model import ElasticNet, Ridge

    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.load(f"{b}/y.npy")
    feats = json.load(open(f"{b}/meta.json"))["feats"]
    p = X.shape[1]
    tw = 1000 * PERIODS_PER_DAY
    n = len(X)

    def fam(name):
        return re.sub(r"_(avail|active)$", "", re.sub(r"_ma_\d+$", "", name))

    nz = np.zeros(p, dtype=int)  # per-feature count of rolling windows with coef != 0
    for t in np.linspace(tw, n, nwin, dtype=int):
        en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=2000, tol=1e-3).fit(
            X[t - tw : t], y[t - tw : t]
        )
        nz += en.coef_ != 0.0
    dead_enet = nz == 0

    rg = Ridge(alpha=100).fit(
        X, y
    )  # full-sample residual (in-sample; this is a usefulness probe, not a forecast)
    lg = LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        num_leaves=127,
        learning_rate=0.05,
        min_child_samples=20,
        n_jobs=8,
        verbose=-1,
        importance_type="split",
    ).fit(X, y - rg.predict(X))
    dead_tree = lg.feature_importances_ == 0
    prunable = dead_enet & dead_tree

    print(
        "SIGNALLESS p=%d enet_always_dead=%d tree_never_split=%d PRUNABLE(both)=%d"
        % (p, int(dead_enet.sum()), int(dead_tree.sum()), int(prunable.sum())),
        flush=True,
    )
    by_fam = {}
    for i, f in enumerate(feats):
        by_fam.setdefault(fam(f), []).append(i)
    full_dead = sorted(k for k, idx in by_fam.items() if all(prunable[j] for j in idx))
    live = sorted(k for k, idx in by_fam.items() if any(not prunable[j] for j in idx))
    print(
        "SIGNALLESS FULLY-DEAD families (%d, prune these): %s"
        % (len(full_dead), ", ".join(full_dead)),
        flush=True,
    )
    print(
        "SIGNALLESS SURVIVING families (%d): %s" % (len(live), ", ".join(live)),
        flush=True,
    )


def do_enet_sparsity(bucket, alpha=0.001, l1=0.2):
    """How many features does the winning enet actually kill? Fit on early/mid/late windows of
    the slim matrix, count zeroed coefficients (the realized sparsity behind the -0.00035 gain)."""
    from sklearn.linear_model import ElasticNet

    b = f"results/covid_imp_rank/{bucket}"
    X = np.asarray(np.load(f"{b}/X_imp.npy"), dtype=np.float64)
    y = np.load(f"{b}/y.npy")
    tw = 1000 * PERIODS_PER_DAY
    p = X.shape[1]
    for label, t in (("early", tw), ("mid", len(X) // 2), ("late", len(X))):
        en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=3000, tol=1e-4).fit(
            X[t - tw : t], y[t - tw : t]
        )
        nnz = int(np.sum(en.coef_ != 0.0))
        print(
            "ENET_SPARSITY %-5s a%g l1%g: nonzero=%d/%d  killed=%d (%.0f%%)"
            % (label, alpha, l1, nnz, p, p - nnz, 100.0 * (p - nnz) / p),
            flush=True,
        )


def do_ebmsmoke(bucket, nblocks=3):
    """Verify EBM runs in the residual pipeline + time one fit (EBM is slow -> per-fit time
    decides chunked-campaign feasibility). Reuses the built lgbm cache (model-independent),
    forces the EBM factory, runs the first nblocks cadence blocks of the residualized backtest."""
    cid = cell_id("lgbm", bucket, 1000, 100.0, 480, "slim")
    cache = load_cache(cid)
    cache["cell"]["model"] = "ebm"
    cfg = {
        "learning_rate": 0.02,
        "max_leaves": 3,
        "interactions": 10,
        "max_bins": 256,
        "max_rounds": 500,
        "outer_bags": 4,
    }
    t0 = time.time()
    k0, k1, preds = preds_chunk(cache, "residualized", cfg, 0, nblocks)
    dt = time.time() - t0
    print(
        "EBMSMOKE blocks[0:%d] rows[%d:%d] finite=%s total=%.0fs per_fit~%.1fs (407 blocks -> ~%.1f min/trial serial)"
        % (
            nblocks,
            k0,
            k1,
            bool(np.all(np.isfinite(preds))),
            dt,
            dt / nblocks,
            dt / nblocks * 407 / 60,
        ),
        flush=True,
    )


def do_olamcheck(seed=0):
    """Sanity: _cadence_enet_online with a single-point mu_grid (mu_factors=[1.0]) collapses to the
    fixed-alpha base (the online mu-selection has no choice -> picks the baseline mu0; the full-window
    refit is then enet(alpha,l1) on the window). The homotopy refit is EXACT, so we validate it against
    an EXACT (tight-tol) sklearn ElasticNet reference (the production _cadence_enet uses tol=1e-3, so
    its CD-tolerance gap ~1e-4 is reported separately -- that gap is sklearn's, not the path's)."""
    from sklearn.linear_model import ElasticNet

    rng = np.random.default_rng(seed)
    train_win = (
        100 * PERIODS_PER_DAY
    )  # > val_len(20d) + embargo(3125) so the forward fit block is non-empty
    n = train_win + 3 * 480
    p = 12
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) * (rng.random(p) < 0.5)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    starts, c1, i1 = _cadence_enet_online(
        X, y, train_win, 480, mu_factors=np.array([1.0])
    )
    cex = np.empty_like(c1)
    iex = np.empty(len(starts))  # exact (tight-tol) fixed-alpha reference
    for k, t in enumerate(int(s) for s in starts):
        en = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=500000, tol=1e-12).fit(
            X[t - train_win : t], y[t - train_win : t]
        )
        cex[k] = en.coef_
        iex[k] = float(en.intercept_)
    dc = float(np.max(np.abs(cex - c1)))
    di = float(np.max(np.abs(iex - i1)))
    _s0, c0, i0 = _cadence_enet(
        X, y, train_win, 480
    )  # production base (sklearn CD tol=1e-3)
    dc_prod = float(np.max(np.abs(c0 - c1)))
    di_prod = float(np.max(np.abs(i0 - i1)))
    print(
        "OLAMCHECK n=%d p=%d vs_exact max|dcoef|=%.3e max|dint|=%.3e %s | vs_prod_tol1e-3 dcoef=%.3e dint=%.3e"
        % (n, p, dc, di, "OK" if max(dc, di) < 1e-6 else "MISMATCH", dc_prod, di_prod),
        flush=True,
    )
    return max(dc, di)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "olamcheck":
        do_olamcheck(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif mode == "enet_masks":
        do_enet_masks(sys.argv[2])
    elif mode == "regime_extra":  # build regime_extra_<families>.npy for the EBM-only injection
        do_regime_extra(sys.argv[2], sys.argv[3])
    elif mode == "chunk_task":
        do_chunk_task(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            int(sys.argv[5]),
            sys.argv[6] if len(sys.argv) > 6 else "",
        )
    elif mode == "chunk_task_ggrid":  # EBM (+) MTFM ensemble grid, EBM fit once/block, all omega x aux
        do_chunk_task_ggrid(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif mode == "chunk_task_adaptive":  # per-block omega tuned on purged inner-val (AW_MODE fixed|scalar|context)
        do_chunk_task_adaptive(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif mode == "chunk_collect":
        do_chunk_collect(
            sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else ""
        )
    elif mode == "signalless_scan":
        do_signalless_scan(sys.argv[2])
    elif mode == "enet_sparsity":
        do_enet_sparsity(sys.argv[2])
    elif mode == "ebmsmoke":
        do_ebmsmoke(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif mode == "dimreduce_cell":
        do_dimreduce_cell(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif mode == "pca_alpha_compare":
        do_pca_alpha_compare(sys.argv[2], sys.argv[3])
    elif mode == "pcawindowsweep":
        do_pcawindowsweep(
            sys.argv[2],
            sys.argv[3],
            [int(x) for x in sys.argv[4].split(",")],
            int(sys.argv[5]),
            int(sys.argv[6]) if len(sys.argv) > 6 else 2000,
        )
    elif mode == "pcatest":
        do_pcatest(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
        )
    elif mode == "alphascan":
        rf = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
        do_alphascan(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            sys.argv[5],
            [float(a) for a in sys.argv[6].split(",")],
            rf,
        )
    elif mode == "chunkcheck":
        do_chunkcheck(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            float(sys.argv[5]),
            int(sys.argv[6]),
            sys.argv[7],
            int(sys.argv[8]),
        )
    elif mode == "prep":
        pipe = sys.argv[7] if len(sys.argv) > 7 else "slim"
        base_kind = sys.argv[8] if len(sys.argv) > 8 else "ridge"
        do_prep(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            float(sys.argv[5]),
            int(sys.argv[6]),
            pipe,
            base_kind,
        )
    elif mode == "trial":
        do_trial(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == "selfcheck":
        pipe = sys.argv[8] if len(sys.argv) > 8 else "slim"
        do_selfcheck(
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            float(sys.argv[5]),
            int(sys.argv[6]),
            int(sys.argv[7]),
            pipe,
        )
    else:
        raise SystemExit(f"unknown mode {mode}")
