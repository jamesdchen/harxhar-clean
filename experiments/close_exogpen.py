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


# ============================================================================ engine
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


KKT_TOL = 1e-9  # relative slack in the KKT check of the active-set solve
ACTIVE_ITERS = 25  # support repairs before falling back to the exact homotopy
STATS = {"active_ok": 0, "active_repaired": 0, "homotopy_fallback": 0}
SOLVE_PATH = ("ridge", *STATS)


def enet_active(G, c, mu_s: float, lam2_s: float, pf, warm):
    """Exact elastic-net / lasso solution, warm-started from the previous session's
    support: solve the KKT system on the support with its signs, verify every KKT
    condition (signs on the support, |c_j - G_j b| <= mu_j off it), repair the support
    (drop sign violators, add the worst violator) and fall back to the exact homotopy
    (enet_exact) when the repairs do not settle.  The answer is the unique optimum
    (to KKT_TOL) whichever path finds it."""
    p = len(c)
    mu, lam2 = mu_s * pf, lam2_s * pf
    act = np.flatnonzero(warm != 0.0)
    sgn = np.sign(warm[act])
    scale = float(np.max(mu)) if np.max(mu) > 0 else 1.0
    for it in range(ACTIVE_ITERS):
        b = np.zeros(p)
        if len(act):
            M = G[np.ix_(act, act)].copy()
            M[np.diag_indices_from(M)] += lam2[act]
            try:
                bA = np.linalg.solve(M, c[act] - mu[act] * sgn)
            except np.linalg.LinAlgError:
                break
            bad = np.sign(bA) != sgn
            if bad.any():  # a coefficient crossed zero: drop it and re-solve
                act, sgn = act[~bad], sgn[~bad]
                continue
            b[act] = bA
        g = c - G @ b
        viol = np.abs(g) - mu
        viol[act] = -np.inf
        jmax = int(np.argmax(viol))
        if viol[jmax] <= KKT_TOL * scale:
            STATS["active_ok" if it == 0 else "active_repaired"] += 1
            return b
        act = np.append(act, jmax)
        sgn = np.append(sgn, np.sign(g[jmax]))
    STATS["homotopy_fallback"] += 1
    return enet_exact(G, c, mu_s, lam2_s, pf)


class Window:
    """Centered sufficient statistics of one window (rows of X / y), and the projection
    onto the centered backbone block, cached for each backbone set."""

    def __init__(self, Xw: np.ndarray, yw: np.ndarray):
        self.n = len(yw)
        self.xbar = Xw.mean(axis=0)
        self.ybar = float(yw.mean())
        self.Xc = Xw - self.xbar
        self.yc = yw - self.ybar
        self.G = self.Xc.T @ self.Xc
        self.c = self.Xc.T @ self.yc
        self.ynorm = float(np.sqrt(self.yc @ self.yc))
        self._proj: dict = {}

    def profile(self, B: np.ndarray, P: np.ndarray):
        """Profiled Gram / moment of the penalized columns P after projecting the
        (centered) unpenalized columns B out; returns (Gt, ct, back) with back(bP) -> bB
        the minimum-norm least-squares backbone coefficients given the penalized ones."""
        key = (B.tobytes(), P.tobytes())
        if key in self._proj:
            return self._proj[key]
        GPP = self.G[np.ix_(P, P)]
        cP = self.c[P]
        if len(B) == 0:
            out = (GPP, cP, lambda bP: np.zeros(0))
        else:
            U, s, Vt = np.linalg.svd(self.Xc[:, B], full_matrices=False)
            # numerical rank by numpy.linalg.matrix_rank's rule (the five weekday dummies
            # sum to one, so after centering they are exactly dependent)
            k = int((s > s.max() * max(self.Xc.shape[0], len(B)) * np.finfo(float).eps).sum())
            Q = U[:, :k]
            A = Q.T @ self.Xc[:, P]
            bq = Q.T @ self.yc
            Gt = GPP - A.T @ A
            Gt = 0.5 * (Gt + Gt.T)
            ct = cP - A.T @ bq
            Vk = Vt[:k].T / s[:k]

            def back(bP, Vk=Vk, A=A, bq=bq):
                return Vk @ (bq - A @ bP)

            out = (Gt, ct, back)
        self._proj[key] = out
        return out


def solve_arm(win: Window, est: str, B, P, pen, warm=None, exact: bool = False):
    """Coefficients (full length p) and intercept of one fit on one window; pen = the
    penalty of each column of P (alpha x penalty factor, sklearn units)."""
    from scipy.linalg import cho_factor, cho_solve

    p = win.G.shape[0]
    Gt, ct, back = win.profile(B, P)
    info: dict = {}
    if est == "ridge":
        A = Gt.copy()
        A[np.diag_indices_from(A)] += pen
        bP = cho_solve(cho_factor(A), ct)
    else:
        l1 = 1.0 if est == "reclasso" else 0.5
        a = float(pen.max())  # pen = alpha x pf with max pf = 1 (the exogenous columns)
        pf = pen / a
        mu_s, lam2_s = win.n * l1 * a, win.n * (1.0 - l1) * a
        if exact:  # the spec's homotopy from scratch
            bP = enet_exact(Gt, ct, mu_s, lam2_s, pf)
        else:
            b0 = np.zeros(len(P)) if warm is None else warm[P]
            n0 = dict(STATS)
            bP = enet_active(Gt, ct, mu_s, lam2_s, pf, b0)
            info["path"] = next(k for k in STATS if STATS[k] != n0[k])
    th = np.zeros(p)
    th[P] = bP
    if len(B):
        th[B] = back(bP)
    b0 = win.ybar - float(win.xbar @ th)
    return th, b0, info


def ols_backbone(win: Window, B: np.ndarray):
    """HAR + calendar OLS (the master table's sub_ols_baseline: production
    RollingLeastSquares at alpha 0 = centered Gram, lstsq on a rank-deficient window)."""
    p = win.G.shape[0]
    th = np.zeros(p)
    sol = np.linalg.lstsq(win.Xc[:, B], win.yc, rcond=None)[0]
    th[B] = sol
    return th, win.ybar - float(win.xbar @ th)


# ============================================================================ run
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
    out = np.full(len(names), -1, dtype=int)
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


def sets_for(mode: str, r: float, keep, bb, grp, pen_g=None, alpha: float = 1.0):
    """(B, P, pen) of one arm: B unpenalized (profiled), P penalized with pen."""
    if mode in ("bb0", "mpr") or (mode in ("bbr", "bbfix") and r == 0.0):
        B = np.flatnonzero(bb & keep)
        P = np.flatnonzero(~bb & keep)
        if mode == "mpr":
            pen = np.asarray(pen_g, float)[grp[P]]
        else:
            pen = np.full(len(P), alpha)
    else:
        B = np.zeros(0, dtype=int)
        P = np.flatnonzero(keep)
        pf = np.where(bb[P], r if mode in ("bbr", "bbfix") else 1.0, 1.0)
        pen = alpha * pf
    return B, P, pen


def tune_arm(est, mode, wf, Xv, yv, keep, bb, grp, grid, bb0_alpha=None):
    """The spec's re-choice on the block-start window's fit / embargo / tail split.
    Returns the choice and the validation MSE of every candidate."""

    def mse(th, b0):
        return float(np.mean((b0 + Xv @ th - yv) ** 2))

    if mode == "mpr":
        pen_g = [float(bb0_alpha)] * len(GROUPS)
        B, P, pen = sets_for("mpr", 0.0, keep, bb, grp, pen_g)
        th, b0, _ = solve_arm(wf, "ridge", B, P, pen)
        best = mse(th, b0)
        cand = [dict(pass_=0, group="start", value=float(bb0_alpha), mse=best)]
        present = set(grp[P].tolist())
        for ps in range(1, MPR_PASSES + 1):
            for gi in range(len(GROUPS)):
                if gi not in present:
                    continue
                for a in grid:
                    if a == pen_g[gi]:
                        continue
                    trial = list(pen_g)
                    trial[gi] = a
                    B, P, pen = sets_for("mpr", 0.0, keep, bb, grp, trial)
                    th, b0, _ = solve_arm(wf, "ridge", B, P, pen)
                    m = mse(th, b0)
                    cand.append(dict(pass_=ps, group=GROUPS[gi], value=a, mse=m))
                    if m < best:
                        best, pen_g = m, trial
        return dict(alpha=np.nan, r=0.0, pen_g=pen_g, mse=best), cand
    rs = {"single": (1.0,), "bb0": (0.0,), "bbr": R_GRID, "bbfix": (R_PAPER,)}[mode]
    res: dict = {}
    for r in rs:
        for a in grid:  # lasso / elastic net: the spec's exact homotopy at every candidate
            B, P, pen = sets_for(mode, r, keep, bb, grp, alpha=a)
            th, b0, _ = solve_arm(wf, est, B, P, pen, exact=True)
            res[(r, a)] = (mse(th, b0), th, b0)
    best = None
    for r in rs:  # selection in grid order, strict improvement (the spec's argmin)
        for a in grid:
            m = res[(r, a)][0]
            if best is None or m < best[0]:
                best = (m, r, a)
    assert best is not None
    m, r, a = best
    out = dict(alpha=a, r=r, pen_g=None, mse=m)
    cand = [dict(r=r_, alpha=a_, mse=v[0]) for (r_, a_), v in res.items()]
    return out, cand


def run() -> None:
    t_cpu0 = time.process_time()
    t_wall0 = time.time()
    grids = spec_grids()
    d = load_design("all_features")
    db = load_design("baseline")
    X, y, W, names = d["X"], d["y"], d["W"], d["names"]
    bb = np.isin(names, db["names"])
    bbi = np.flatnonzero(bb)
    assert [names[j] for j in bbi] == db["names"]
    assert np.array_equal(X[:, bbi], db["X"]) and np.array_equal(y, db["y"])
    grp = exog_groups(names, bb)
    n_test, p = len(X) - W, X.shape[1]
    exog = ~bb
    rec: dict = {}
    for est, mode in ARMS:
        rec[arm_key(est, mode)] = dict(
            pred=np.empty(n_test),
            th=np.empty((n_test, p), np.float32),
            b0=np.empty(n_test),
            alpha=np.empty(n_test),
            r=np.empty(n_test),
            vBB=np.empty(n_test),
            vEE=np.empty(n_test),
            vBE=np.empty(n_test),
            phiB=np.empty(n_test),
            phiE=np.empty(n_test),
            nnzE=np.empty(n_test),
            solve_path=np.zeros(n_test),
        )
    rec["ols_baseline"] = dict(
        pred=np.empty(n_test),
        th=np.empty((n_test, len(bbi))),
        b0=np.empty(n_test),
        vBB=np.empty(n_test),
        phiB=np.empty(n_test),
    )
    tune_rows, cand_rows, mpr_rows, check_rows = [], [], [], []
    prev: dict = {k: None for k in rec}
    for b, (i0, i1) in enumerate(block_starts(n_test)):
        tb = time.time()
        keep = identifiable(X[i0 : i0 + W])
        fit_hi = i0 + W - VAL_TAIL - EMBARGO
        wf = Window(X[i0:fit_hi], y[i0:fit_hi])
        Xv, yv = X[i0 + W - VAL_TAIL : i0 + W], y[i0 + W - VAL_TAIL : i0 + W]
        choice: dict = {}
        for est, mode in ARMS:
            k = arm_key(est, mode)
            ch, cand = tune_arm(
                est, mode, wf, Xv, yv, keep, bb, grp, grids[est],
                bb0_alpha=choice.get("ridge_bb0", {}).get("alpha"),
            )
            choice[k] = ch
            grid = grids[est]
            tune_rows.append(
                dict(
                    block=b,
                    first_forecast=str(d["date"][W + i0])[:10],
                    arm=k,
                    alpha=ch["alpha"],
                    r=ch["r"],
                    val_mse=ch["mse"],
                    alpha_at_low_edge=bool(ch["alpha"] == min(grid)),
                    alpha_at_high_edge=bool(ch["alpha"] == max(grid)),
                    n_masked=int((~keep).sum()),
                    n_backbone_masked=int((bb & ~keep).sum()),
                )
            )
            for c in cand:
                cand_rows.append(dict(block=b, arm=k, **c))
            if mode == "mpr":
                for gi, g in enumerate(GROUPS):
                    a = ch["pen_g"][gi]
                    mpr_rows.append(
                        dict(block=b, group=g, penalty=a,
                             n_cols=int((exog & keep & (grp == gi)).sum()),
                             at_low_edge=bool(a == min(grid)),
                             at_high_edge=bool(a == max(grid)))
                    )
        print(f"block {b}: tuned in {time.time() - tb:.1f}s  "
              + "  ".join(f"{k}={v['alpha']:g}/r{v['r']:g}" for k, v in choice.items()
                          if k != "ridge_mpr")
              + f"  mpr={choice['ridge_mpr']['pen_g']}", flush=True)
        tb = time.time()
        for j in range(i0, i1):
            t = W + j
            win = Window(X[j:t], y[j:t])
            xt = X[t]
            dx = xt - win.xbar
            GBB = win.G[np.ix_(bb, bb)]
            GEE = win.G[np.ix_(exog, exog)]
            GBE = win.G[np.ix_(bb, exog)]
            for est, mode in ARMS:
                k = arm_key(est, mode)
                ch = choice[k]
                B, P, pen = sets_for(mode, ch["r"], keep, bb, grp, ch["pen_g"],
                                     alpha=ch["alpha"])
                th, b0, info = solve_arm(win, est, B, P, pen, warm=prev[k])
                prev[k] = th
                if est != "ridge" and (j - i0) % CHECK_EVERY == 0:
                    thx, b0x, _ = solve_arm(win, est, B, P, pen, exact=True)
                    check_rows.append(dict(
                        session=j, arm=k,
                        max_coef_gap=float(np.max(np.abs(th - thx))),
                        pred_gap=float(abs((b0 + xt @ th) - (b0x + xt @ thx))),
                        same_support=bool(np.array_equal(th != 0, thx != 0))))
                R = rec[k]
                R["pred"][j] = b0 + float(xt @ th)
                R["th"][j] = th
                R["b0"][j] = b0
                R["alpha"][j] = ch["alpha"]
                R["r"][j] = ch["r"]
                tB, tE = th[bb], th[exog]
                R["vBB"][j] = float(tB @ GBB @ tB) / win.n
                R["vEE"][j] = float(tE @ GEE @ tE) / win.n
                R["vBE"][j] = float(tB @ GBE @ tE) / win.n
                R["phiB"][j] = float(tB @ dx[bb])
                R["phiE"][j] = float(tE @ dx[exog])
                R["nnzE"][j] = int(np.count_nonzero(tE))
                R["solve_path"][j] = SOLVE_PATH.index(info.get("path", "ridge"))
            th, b0 = ols_backbone(win, bbi)
            R = rec["ols_baseline"]
            R["pred"][j] = b0 + float(xt @ th)
            R["th"][j] = th[bbi]
            R["b0"][j] = b0
            R["vBB"][j] = float(th[bbi] @ win.G[np.ix_(bbi, bbi)] @ th[bbi]) / win.n
            R["phiB"][j] = float(th[bbi] @ dx[bbi])
        print(f"block {b}: {i1 - i0} refits in {time.time() - tb:.1f}s", flush=True)
    runs = WORK / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for k, R in rec.items():
        np.savez_compressed(runs / f"{k}.npz", **R)
    pd.DataFrame(tune_rows).to_csv(WORK / "tune_choices.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(WORK / "tune_candidates.csv", index=False)
    pd.DataFrame(mpr_rows).to_csv(WORK / "mpr_group_penalties.csv", index=False)
    pd.DataFrame(check_rows).to_csv(WORK / "active_vs_homotopy.csv", index=False)
    meta = dict(
        names=names,
        backbone=[names[j] for j in bbi],
        groups=list(GROUPS),
        group_of=grp.tolist(),
        cpu_sec=time.process_time() - t_cpu0,
        wall_sec=time.time() - t_wall0,
        solve_stats=dict(STATS),
        kkt_tol=KKT_TOL,
        r_grid=list(R_GRID),
        grids=grids,
    )
    (WORK / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"run done: cpu {meta['cpu_sec']:.0f}s wall {meta['wall_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "gate":
        gate()
    elif stage == "run":
        run()
    else:
        raise SystemExit(__doc__)
