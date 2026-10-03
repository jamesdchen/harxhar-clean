"""Leave the HAR + calendar backbone (nearly) unpenalized, penalize the exogenous block.

Question (user, 2026-10-03): the 16:00-bar linear models (specs/causal_tune_linear.py)
put ONE penalty on every column, and with all features they lose to the same model on
HAR + calendar alone.  The paper's forecast of record (the 2-block ridge of
src/unification.py) penalizes the backbone at 1 and the exogenous block at 100.  Does
the all-features ridge / lasso / elastic net recover when the backbone is left alone?

Design: the shared 16:00-bar cache of experiments/capture_design_close.py
(results/close_design/_work/design_bar1600_<bucket>.npz: prescaled X, target y, W = 2000
training sessions, 1469 forecast sessions).  Backbone = the target's HAR ladder
``har_ma_*`` + the calendar / expiry columns, i.e. exactly the 22 columns of the
``baseline`` design (asserted).  Everything else (606 columns) is the exogenous block.

Protocol (the spec's, unchanged): window = the 2000 sessions before the forecast, refit
every session, intercept unpenalized, the identifiability mask (columns constant in the
window or byte-copies of an earlier kept column get coefficient 0) recomputed at every
penalty re-choice, the penalty re-chosen every 250 sessions on the last 125 sessions of
the window after a 25-session embargo (fit on the first 1850, mean squared error on the
125), the spec's grids (ridge 1e-2 .. 1e3, lasso / elastic net 1e-6 .. 1e-2, elastic net
l1_ratio 0.5), sklearn units with n = rows of the fit.

Penalty factors (glmnet's penalty.factor convention): the penalty on column j is
alpha * pf_j * (l1 |b_j| + (1 - l1) / 2 b_j^2); pf = 1 on the exogenous columns and
pf = r on the backbone.
  single   r = 1 (the spec's models; GATE against the stored forecasts).
  bb0      r = 0: the backbone is unpenalized.  Exact by profiling: the penalized
           problem is solved on the exogenous columns after projecting the backbone
           (and the intercept) out of y and the exogenous columns on the window, then
           the backbone coefficients are the least-squares fit of the remaining
           residual (minimum-norm solution; the five weekday dummies sum to the
           intercept, so the unpenalized block has an exact linear dependency).
  bbr      r tuned jointly with alpha on the same validation tail, r in R_GRID =
           {0, 0.01, 0.1, 1}: r = 1 nests the single-penalty model, r = 0 nests bb0,
           and 0.01 is the paper's ratio (backbone penalty 1, exogenous 100, in ridge
           units).
  bbfix    r fixed at the paper's 1/100 (alpha tuned as usual).
  mpr      ridge only: the backbone unpenalized and one penalty for each exogenous
           feature group (src.data.loading.SUBGROUPS: moments, liquidity, market_ew,
           market_vw, sentiment, implied_vol, vol_demand, fomc; an availability /
           activity indicator goes with its source's group) -- multi-penalty ridge
           (van de Wiel, van Nee & Rauschenberger 2021, "Fast cross-validation for
           multi-penalty high-dimensional ridge regression"), also called
           group-regularized ridge (van de Wiel et al. 2016).  The 8 penalties are
           chosen on the same validation tail by cyclic coordinate search over the
           ridge grid (start: every group at the bb0 choice; MPR_PASSES passes).
Solvers (every fit is the exact optimum of its window, as the spec's): ridge = Cholesky
on the profiled Gram; lasso / elastic net = at a re-choice, the spec's exact homotopy
(src.models.reclasso_har.lasso_homotopy, penalty factors by column scaling) for every
candidate; at a refit, the KKT system on the previous session's support with a full KKT
check (support repaired, else the homotopy from scratch), compared with the homotopy every
CHECK_EVERY sessions (_work/active_vs_homotopy.csv).  The spec itself slides the lasso /
elastic net by a warm homotopy in the data (enet_online), exact up to float drift until
the next re-choice re-anchors it.

Stages (``python experiments/close_exogpen.py <stage>``):
  gate     the spec's own RollingTunedLinear (experiments/dense_vs_sparse_1530.py
           run_linear_blocks, the spec's estimator section executed read-only) on the
           cached all_features design for ridge / lasso / elastic net; compared with
           the stored 16:00 forecasts results/spxw_pnl/yhat_sub_<est>_all_features and,
           through the research scorer, with the master table's QLIKE / Sharpe.
  run      every arm with the engine above (also the engine's single-penalty arms,
           compared with the gate's, and the HAR + calendar OLS on the baseline design,
           compared with the master table's sub_ols_baseline).
  analyze  scoring (16:00-bar recalibration, QLIKE on the 866 trade days, the 15:30
           sign(s) straddle Sharpe mid / crossed), Diebold-Mariano on daily QLIKE and a
           HAC t on the paired daily P&L difference against the single-penalty arm and
           the baseline arm, the penalty path, backbone shrinkage, variance shares,
           figures, SUMMARY.md.

The circular block bootstrap is off in this branch (commit b761b28): point estimates
only; differences are tested with Diebold-Mariano (src.evaluation.diebold_mariano) and
the Newey-West t of notebooks/atm_straddle_lib.newey_west_t, as master_table_close.py.

Compute: one process, single-threaded BLAS (set below before numpy loads).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "close_studies_2026-10-03" / "exog_penalty"
WORK = OUT / "_work"
DESIGN = REPO / "results" / "close_design" / "_work"
SPXW = REPO / "results" / "spxw_pnl"
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
SEG = "bar1600"
TW = 2000
TUNE_PER = 250
VAL_TAIL = 125
EMBARGO = 25
EST = ("ridge", "reclasso", "reclasticnet")
EST_SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
LABEL = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "elastic net"}
# backbone : exogenous penalty-factor ratios searched jointly with alpha (bbr arms):
# 1 = the single-penalty model, 0 = backbone unpenalized, 0.01 = the paper's 1 : 100
R_GRID = (0.0, 0.01, 0.1, 1.0)
R_PAPER = 0.01
MPR_PASSES = 2  # cyclic coordinate-search passes over the 8 group penalties
CHECK_EVERY = 25  # sessions between checks of the active-set solve against the homotopy


# ============================================================================ data
def load_design(bucket: str) -> dict:
    z = np.load(DESIGN / f"design_{SEG}_{bucket}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["W"] = int(d["W"])
    d["names"] = [str(v) for v in d["names"]]
    return d


def forecast_dz(d: dict) -> dict:
    """The research scorer's input for the forecast rows (date, true_adj, true_raw)."""
    W = d["W"]
    return dict(date=d["date"][W:], true_adj=d["y"][W:], true_raw=d["true_raw"])


def stored_1600(key: str) -> pd.Series:
    """The stored forecast table's 16:00 ET rows (pred_adj), indexed by naive-ET stamp."""
    t = pd.read_parquet(SPXW / f"yhat_{key}.parquet")
    et = pd.to_datetime(t["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    return pd.Series(t.loc[m, "yhat"].to_numpy(float), index=pd.DatetimeIndex(et[m]))


def block_starts(n_test: int) -> list[tuple[int, int]]:
    return [(i0, min(i0 + TUNE_PER, n_test)) for i0 in range(0, n_test, TUNE_PER)]


# ============================================================================ gate
def gate() -> None:
    """The spec's own class on the cached design == the stored 16:00 forecasts."""
    import dense_vs_sparse_1530 as dvs

    d = load_design("all_features")
    X, y, W = d["X"], d["y"], d["W"]
    n_test = len(X) - W
    blocks = [np.ascontiguousarray(X[i0 : W + i1]) for i0, i1 in block_starts(n_test)]
    idx = pd.DatetimeIndex(pd.to_datetime(d["date"][W:]))
    rows = []
    for est in EST:
        f = WORK / f"gate_spec_{est}.npz"
        if f.is_file():
            r = dict(np.load(f))
            sec = float(r["sec"])
        else:
            t0 = time.process_time()
            r = dvs.run_linear_blocks(est, blocks, W, y, want_coef=False)
            sec = time.process_time() - t0
            np.savez_compressed(f, pred=r["pred"], alpha=r["alpha"], sec=sec,
                                n_reseed=r["n_reseed"])
        st = stored_1600(f"sub_{EST_SHORT[est]}_all_features").reindex(idx)
        diff = np.abs(r["pred"] - st.to_numpy())
        rows.append(dict(est=est, n=len(st), n_missing=int(st.isna().sum()),
                         max_abs_diff=float(np.nanmax(diff)),
                         max_rel_diff=float(np.nanmax(diff / np.abs(st.to_numpy()))),
                         cpu_sec=sec,
                         alphas=json.dumps(sorted(set(np.round(r["alpha"], 12).tolist())))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "gate_spec_vs_stored.csv", index=False)


# ============================================================================ numpy reference
# The Python reference the C kernel is checked against on a short slice: the profiled
# problem solved directly (ridge: Cholesky; lasso / elastic net: the spec's exact
# homotopy, penalty factors by column scaling).
def identifiable(Xw: np.ndarray) -> np.ndarray:
    """The spec's identifiability rule (RollingTunedLinear._recompute_mask): False for
    columns constant in the window or byte-copies of an earlier kept column."""
    keep = np.ones(Xw.shape[1], dtype=bool)
    seen: set[bytes] = set()
    for j in range(Xw.shape[1]):
        col = np.ascontiguousarray(Xw[:, j])
        if col.max() == col.min():
            keep[j] = False
            continue
        key = col.tobytes()
        if key in seen:
            keep[j] = False
        else:
            seen.add(key)
    return keep


def enet_exact(G, c, mu: float, lam2: float, pf: np.ndarray) -> np.ndarray:
    """The spec's exact homotopy (src.models.reclasso_har.lasso_homotopy) with glmnet
    penalty factors by column scaling: b = g / pf solves the problem on
    D^-1 G D^-1 + lam2 diag(1 / pf), D^-1 c, unit L1 weights (exact for pf > 0)."""
    from src.models.reclasso_har import lasso_homotopy

    inv = 1.0 / pf
    Gs = G * np.outer(inv, inv)
    Gs[np.diag_indices_from(Gs)] += lam2 * inv
    g, _, _ = lasso_homotopy(Gs, c * inv, mu)
    return g * inv


class Window:
    """Centered statistics of one window; the projection onto the centered unpenalized
    columns (minimum-norm, numpy.linalg.matrix_rank's rank rule)."""

    def __init__(self, Xw: np.ndarray, yw: np.ndarray):
        self.n = len(yw)
        self.xbar = Xw.mean(axis=0)
        self.ybar = float(yw.mean())
        self.Xc = Xw - self.xbar
        self.yc = yw - self.ybar
        self.G = self.Xc.T @ self.Xc
        self.c = self.Xc.T @ self.yc

    def profile(self, B: np.ndarray, P: np.ndarray):
        GPP, cP = self.G[np.ix_(P, P)], self.c[P]
        if len(B) == 0:
            return GPP, cP, lambda bP: np.zeros(0)
        U, s, Vt = np.linalg.svd(self.Xc[:, B], full_matrices=False)
        k = int((s > s.max() * max(self.Xc.shape[0], len(B)) * np.finfo(float).eps).sum())
        Q = U[:, :k]
        A = Q.T @ self.Xc[:, P]
        bq = Q.T @ self.yc
        Gt = GPP - A.T @ A
        Vk = Vt[:k].T / s[:k]
        return 0.5 * (Gt + Gt.T), cP - A.T @ bq, lambda bP: Vk @ (bq - A @ bP)


def solve_ref(win: Window, est: str, B, P, pen):
    """Coefficients (length p) and intercept; pen = alpha x pf on P (pf max 1)."""
    p = win.G.shape[0]
    Gt, ct, back = win.profile(B, P)
    if est == "ridge":
        A = Gt.copy()
        A[np.diag_indices_from(A)] += pen
        bP = np.linalg.solve(A, ct)
    else:
        l1 = 1.0 if est == "reclasso" else 0.5
        a = float(pen.max())
        bP = enet_exact(Gt, ct, win.n * l1 * a, win.n * (1.0 - l1) * a, pen / a)
    th = np.zeros(p)
    th[P] = bP
    if len(B):
        th[B] = back(bP)
    return th, win.ybar - float(win.xbar @ th)


def ols_backbone(win: Window, B: np.ndarray):
    """HAR + calendar OLS (the master table's sub_ols_baseline: production
    RollingLeastSquares at alpha 0 = centered window, minimum-norm least squares on a
    rank-deficient window)."""
    th = np.zeros(win.G.shape[0])
    th[B] = np.linalg.lstsq(win.Xc[:, B], win.yc, rcond=None)[0]
    return th, win.ybar - float(win.xbar @ th)


# ============================================================================ arms
GROUPS = (
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
)
MODES = ("single", "bb0", "bbr", "bbfix")
ARMS = [(e, m) for e in EST for m in MODES] + [("ridge", "mpr")]
R_OF = {"single": (1.0,), "bb0": (0.0,), "bbr": R_GRID, "bbfix": (R_PAPER,), "mpr": (0.0,)}
CD_REL = 1e-8  # coordinate descent stops when max |step| sqrt(G_jj) < CD_REL |y_c|
KKT_REL = 1e-9  # KKT slack off the support, relative to max |X'y_c|
MAX_ROUNDS = 6  # descent rounds (tolerance / 100 each) before giving up the exact solve
MAX_SWEEPS = 200000
STATUS = ("ridge", "polished", "polished_after_tighter_descent", "descent_only")


def arm_key(est: str, mode: str) -> str:
    return f"{EST_SHORT[est]}_{mode}"


def exog_groups(names: list[str], bb: np.ndarray) -> np.ndarray:
    """Group index (into GROUPS) of every exogenous column, -1 on the backbone.  A design
    column is <adj_>raw<_ma_k>, <raw>_avail<_ma_k> or <raw>_active<_ma_k>."""
    import re

    from src.data.loading import SUBGROUPS

    of: dict[str, int] = {}
    for gi, g in enumerate(GROUPS):
        for f in SUBGROUPS[g]:
            assert f not in of, f
            of[f] = gi
    assert set(of) == set(SUBGROUPS["all_features"]), "groups must cover all_features"
    out = np.full(len(names), -1, dtype=np.int32)
    for j, c in enumerate(names):
        if bb[j]:
            continue
        raw = re.sub(r"_ma_\d+$", "", c)
        raw = re.sub(r"^adj_", "", raw)
        raw = re.sub(r"_(avail|active)$", "", raw)
        out[j] = of[raw]
    return out


def spec_grids() -> dict:
    """The spec's grids and constants (its estimator section, executed read-only)."""
    import dense_vs_sparse_1530 as dvs

    ns = dvs.spec_ns("linear")
    assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"], ns["TRAIN_WIN"]) == (
        TUNE_PER,
        VAL_TAIL,
        EMBARGO,
        TW,
    )
    return {e: [float(a) for _, a, _ in ns["ESTIMATOR_GRIDS"][e]] for e in EST}


class Setup:
    """Everything the kernel needs that does not change with the arm."""

    def __init__(self):
        from scipy.linalg import qr

        d = load_design("all_features")
        db = load_design("baseline")
        self.d = d
        self.X = np.ascontiguousarray(d["X"], dtype=np.float64)
        self.y = np.ascontiguousarray(d["y"], dtype=np.float64)
        self.W = d["W"]
        self.names = d["names"]
        self.p = self.X.shape[1]
        self.n_test = len(self.X) - self.W
        self.bb = np.isin(self.names, db["names"])
        self.bbi = np.flatnonzero(self.bb)
        assert [self.names[j] for j in self.bbi] == db["names"]
        assert np.array_equal(self.X[:, self.bbi], db["X"])
        assert np.array_equal(self.y, db["y"])
        self.grp = exog_groups(self.names, self.bb)
        self.blocks = block_starts(self.n_test)
        self.bstart = np.array([b[0] for b in self.blocks] + [self.n_test], dtype=np.int32)
        self.keep, self.drop = [], []
        for i0, _ in self.blocks:
            Xw = self.X[i0 : i0 + self.W]
            keep = identifiable(Xw)
            # when the backbone is unpenalized: drop the columns beyond the numerical rank
            # of the centered kept backbone (pivoted QR); the weekday dummies sum to one,
            # so one of them goes.  Fit and forecasts do not change (the dependency holds
            # on every row).
            B = np.flatnonzero(self.bb & keep)
            Z = Xw[:, B] - Xw[:, B].mean(axis=0)
            _, Rm, piv = qr(Z, mode="economic", pivoting=True)
            dg = np.abs(np.diag(Rm))
            k = int((dg > dg[0] * max(Z.shape) * np.finfo(float).eps).sum())
            self.keep.append(keep)
            self.drop.append(np.sort(B[piv[k:]]))

    def pf(self, rs) -> np.ndarray:
        out = np.full((len(self.blocks), len(rs), self.p), -1.0)
        for b in range(len(self.blocks)):
            keep = self.keep[b]
            for i, r in enumerate(rs):
                out[b, i, ~self.bb & keep] = 1.0
                out[b, i, self.bb & keep] = r
                if r == 0.0:
                    out[b, i, self.drop[b]] = -1.0
        return out


_LIB = None


def kernel():
    """Compile experiments/close_exogpen_kernel.c when the .c is newer than the .so."""
    global _LIB
    if _LIB is not None:
        return _LIB
    import ctypes
    import subprocess

    from numpy.ctypeslib import ndpointer

    src = REPO / "experiments" / "close_exogpen_kernel.c"
    so = WORK / "close_exogpen_kernel.so"
    if not so.is_file() or so.stat().st_mtime < src.stat().st_mtime:
        WORK.mkdir(parents=True, exist_ok=True)
        # the system ships only liblapack.so.3 / libblas.so.3 (no unversioned link)
        cmd = ["gcc", "-O3", "-march=native", "-fPIC", "-shared", "-o", str(so), str(src),
               "-l:liblapack.so.3", "-l:libblas.so.3", "-lm"]
        subprocess.run(cmd, check=True)
        print("compiled", " ".join(cmd), flush=True)
    lib = ctypes.CDLL(str(so))
    f8 = ndpointer(np.float64, flags="C_CONTIGUOUS")
    i4 = ndpointer(np.int32, flags="C_CONTIGUOUS")
    i8 = ndpointer(np.int64, flags="C_CONTIGUOUS")
    ci, cd, cl = ctypes.c_int, ctypes.c_double, ctypes.c_long
    lib.exogpen_run.restype = ci
    lib.exogpen_run.argtypes = [
        ci, ci, f8, f8, ci,  # N, p, X, y, W
        ci, ci,  # est, mode
        ci, i4,  # n_blocks, bstart
        ci, f8,  # n_r, pf
        ci, f8, ci, ci,  # n_alpha, alphas, val_tail, embargo
        i4, ci, f8, ci,  # group, n_groups, mpr_start, mpr_passes
        i4, cd, cd, ci, cl,  # is_bb, cd_rel, kkt_rel, max_rounds, max_sweeps
        f8, f8, f8, f8, f8, f8,  # pred, theta, b0, vBB, vEE, vBE
        i4, i8, f8, i4, f8, f8,  # status, sweeps, val_mse, choice, pen_g, mpr_mse
    ]
    _LIB = lib
    return lib


def c_run(S: Setup, est: str, mode: str, grid: list[float], mpr_start=None,
          n_blocks: int | None = None, X=None) -> dict:
    """One arm through the C walk-forward (optionally only the first n_blocks blocks)."""
    lib = kernel()
    rs = R_OF[mode]
    nb = len(S.blocks) if n_blocks is None else n_blocks
    pf = np.ascontiguousarray(S.pf(rs)[:nb])
    bstart = np.ascontiguousarray(S.bstart[: nb + 1])
    n_out = int(bstart[-1])
    X = S.X if X is None else X
    p = S.p
    out = dict(
        pred=np.zeros(n_out), theta=np.zeros((n_out, p)), b0=np.zeros(n_out),
        vBB=np.zeros(n_out), vEE=np.zeros(n_out), vBE=np.zeros(n_out),
        status=np.zeros(n_out, np.int32), sweeps=np.zeros(n_out, np.int64),
        val_mse=np.full((nb, len(rs), len(grid)), np.nan),
        choice=np.zeros((nb, 2), np.int32), pen_g=np.full((nb, len(GROUPS)), np.nan),
        mpr_mse=np.full(nb, np.nan),
    )
    est_i = {"ridge": 0, "reclasso": 1, "reclasticnet": 2}[est]
    ms = np.ascontiguousarray(
        np.zeros(nb) if mpr_start is None else np.asarray(mpr_start, float)[:nb]
    )
    t0 = time.process_time()
    rc = lib.exogpen_run(
        len(X), p, X, S.y, S.W, est_i, 1 if mode == "mpr" else 0, nb, bstart,
        len(rs), pf, len(grid), np.ascontiguousarray(grid, dtype=np.float64), VAL_TAIL,
        EMBARGO, np.ascontiguousarray(np.maximum(S.grp, 0), dtype=np.int32), len(GROUPS),
        ms, MPR_PASSES, np.ascontiguousarray(S.bb, dtype=np.int32), CD_REL, KKT_REL,
        MAX_ROUNDS, MAX_SWEEPS, out["pred"], out["theta"], out["b0"], out["vBB"],
        out["vEE"], out["vBE"], out["status"], out["sweeps"], out["val_mse"],
        out["choice"], out["pen_g"], out["mpr_mse"],
    )
    out["cpu_sec"] = time.process_time() - t0
    if rc != 0:
        raise RuntimeError(f"kernel returned {rc} for {est} {mode}")
    ch = out["choice"]
    if mode == "mpr":
        out["alpha_blk"] = np.full(nb, np.nan)
        out["r_blk"] = np.zeros(nb)
    else:
        out["alpha_blk"] = np.array([grid[i] for i in ch[:, 1]])
        out["r_blk"] = np.array([rs[i] for i in ch[:, 0]])
    return out


def ref_slice(S: Setup, est: str, mode: str, res: dict, n_sess: int) -> dict:
    """The numpy reference on the first n_sess sessions of block 0, at the penalties the
    kernel chose there; also the reference's tail MSE at those penalties."""
    keep = S.keep[0]
    r = float(res["r_blk"][0])
    if mode == "mpr":
        B = np.flatnonzero(S.bb & keep)
        P = np.flatnonzero(~S.bb & keep)
        pen = res["pen_g"][0][S.grp[P]]
    elif r == 0.0:
        B = np.flatnonzero(S.bb & keep)
        P = np.flatnonzero(~S.bb & keep)
        pen = np.full(len(P), res["alpha_blk"][0])
    else:
        B = np.zeros(0, dtype=int)
        P = np.flatnonzero(keep)
        pen = res["alpha_blk"][0] * np.where(S.bb[P], r, 1.0)
    W = S.W
    fit_hi = W - VAL_TAIL - EMBARGO
    wf = Window(S.X[:fit_hi], S.y[:fit_hi])
    thf, b0f = solve_ref(wf, est, B, P, pen)
    Xv, yv = S.X[W - VAL_TAIL : W], S.y[W - VAL_TAIL : W]
    mse = float(np.mean((b0f + Xv @ thf - yv) ** 2))
    pred = np.empty(n_sess)
    for j in range(n_sess):
        win = Window(S.X[j : W + j], S.y[j : W + j])
        th, b0 = solve_ref(win, est, B, P, pen)
        pred[j] = b0 + float(S.X[W + j] @ th)
    return dict(pred=pred, val_mse=mse)


# ============================================================================ check
def check() -> None:
    """C kernel vs the numpy reference (short slice) and vs the gated spec forecasts."""
    grids = spec_grids()
    S = Setup()
    rows = []
    n_sess = 20
    for est, mode in ARMS:
        if mode == "mpr":
            continue  # its start value comes from the ridge bb0 arm, checked below
        res = c_run(S, est, mode, grids[est], n_blocks=1)
        t0 = time.process_time()
        ref = ref_slice(S, est, mode, res, n_sess)
        sec = time.process_time() - t0
        ir, ia = res["choice"][0]
        rows.append(dict(
            arm=arm_key(est, mode), alpha=res["alpha_blk"][0], r=res["r_blk"][0],
            max_abs_pred_gap=float(np.max(np.abs(res["pred"][:n_sess] - ref["pred"]))),
            max_rel_pred_gap=float(np.max(np.abs(res["pred"][:n_sess] / ref["pred"] - 1))),
            val_mse_c=float(res["val_mse"][0, ir, ia]), val_mse_ref=ref["val_mse"],
            c_cpu_sec_block0=res["cpu_sec"], ref_cpu_sec_slice=sec,
            status=json.dumps(np.bincount(res["status"], minlength=4).tolist())))
        print(rows[-1], flush=True)
        if est == "ridge" and mode == "bb0":
            mres = c_run(S, "ridge", "mpr", grids["ridge"], mpr_start=res["alpha_blk"],
                         n_blocks=1)
            mref = ref_slice(S, "ridge", "mpr", mres, n_sess)
            rows.append(dict(
                arm="ridge_mpr", alpha=np.nan, r=0.0,
                max_abs_pred_gap=float(np.max(np.abs(mres["pred"][:n_sess] - mref["pred"]))),
                max_rel_pred_gap=float(np.max(np.abs(mres["pred"][:n_sess] / mref["pred"] - 1))),
                val_mse_c=float(mres["mpr_mse"][0]), val_mse_ref=mref["val_mse"],
                c_cpu_sec_block0=mres["cpu_sec"], ref_cpu_sec_slice=np.nan,
                status=json.dumps(mres["pen_g"][0].tolist())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "check_c_vs_numpy_slice.csv", index=False)
    # the full single-penalty arms vs the gated spec forecasts and the stored tables
    idx = pd.DatetimeIndex(pd.to_datetime(S.d["date"][S.W :]))
    grows = []
    for est in EST:
        res = c_run(S, est, "single", grids[est])
        np.savez_compressed(WORK / f"check_c_single_{est}.npz", pred=res["pred"])
        spec = np.load(WORK / f"gate_spec_{est}.npz")
        st = stored_1600(f"sub_{EST_SHORT[est]}_all_features").reindex(idx).to_numpy()
        spec_alpha = spec["alpha"][S.bstart[:-1]]
        grows.append(dict(
            est=est,
            c_cpu_sec_full_arm=res["cpu_sec"],
            spec_python_cpu_sec_full_arm=float(spec["sec"]),
            alphas_c=json.dumps(res["alpha_blk"].tolist()),
            alphas_spec=json.dumps(spec_alpha.tolist()),
            same_alphas=bool(np.array_equal(res["alpha_blk"], spec_alpha)),
            max_rel_gap_vs_spec=float(np.max(np.abs(res["pred"] / spec["pred"] - 1))),
            max_rel_gap_vs_stored=float(np.max(np.abs(res["pred"] / st - 1))),
            spec_max_rel_gap_vs_stored=float(np.max(np.abs(spec["pred"] / st - 1))),
            n_sessions_rel_gap_vs_stored_above_1e9=int(np.sum(np.abs(res["pred"] / st - 1) > 1e-9)),
            status=json.dumps(np.bincount(res["status"], minlength=4).tolist()),
            median_sweeps=float(np.median(res["sweeps"])),
        ))
        print(grows[-1], flush=True)
    pd.DataFrame(grows).to_csv(WORK / "check_c_vs_spec_gate.csv", index=False)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "gate":
        gate()
    elif stage == "check":
        check()
    else:
        raise SystemExit(__doc__)
