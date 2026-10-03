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


def spec_ns() -> dict:
    """The spec's estimator section (executed read-only), constants asserted."""
    import dense_vs_sparse_1530 as dvs

    ns = dvs.spec_ns("linear")
    assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"], ns["TRAIN_WIN"]) == (
        TUNE_PER,
        VAL_TAIL,
        EMBARGO,
        TW,
    )
    return ns


def spec_grids() -> dict:
    ns = spec_ns()
    return {e: [float(a) for _, a, _ in ns["ESTIMATOR_GRIDS"][e]] for e in EST}


def structure(Xraw: np.ndarray, bb_aug: np.ndarray, r: float):
    """(locked, maskout, pf) of one candidate on a raw augmented window (intercept last).
    r > 0: the spec's rule (intercept locked; a non-locked column constant in the window
    or a byte-copy of an earlier kept non-locked column is masked), backbone penalty
    factor r.  r = 0: the backbone is locked (never leaves the active set, no penalty),
    except backbone columns constant in the window, byte-copies of an earlier backbone
    column, or beyond the numerical rank of the centered backbone (pivoted QR; the five
    weekday dummies sum to the intercept, so one of them), which are masked; the spec's
    rule for the other columns."""
    from scipy.linalg import qr

    m = Xraw.shape[1]
    locked = np.zeros(m, dtype=bool)
    locked[-1] = True
    maskout = np.zeros(m, dtype=bool)
    if r == 0.0:
        kept, seen = [], set()
        for j in np.flatnonzero(bb_aug):
            col = np.ascontiguousarray(Xraw[:, j])
            if col.max() == col.min() or col.tobytes() in seen:
                maskout[j] = True
                continue
            seen.add(col.tobytes())
            kept.append(j)
        kept = np.array(kept)
        Z = Xraw[:, kept] - Xraw[:, kept].mean(axis=0)
        _, Rm, piv = qr(Z, mode="economic", pivoting=True)
        dg = np.abs(np.diag(Rm))
        k = int((dg > dg[0] * max(Z.shape) * np.finfo(float).eps).sum())
        maskout[kept[piv[k:]]] = True
        locked[kept[piv[:k]]] = True
    seen2: set[bytes] = set()
    for j in range(m):
        if locked[j] or (r == 0.0 and bb_aug[j]):
            continue
        col = np.ascontiguousarray(Xraw[:, j])
        if col.max() == col.min():
            maskout[j] = True
            continue
        key = col.tobytes()
        if key in seen2:
            maskout[j] = True
        else:
            seen2.add(key)
    pf = np.ones(m)
    pf[bb_aug & ~locked] = r if r > 0.0 else 1.0
    pf[locked] = 0.0
    return locked, maskout, pf


class Setup:
    """The design (intercept appended as the last column), blocks, groups, structures."""

    def __init__(self):
        d = load_design("all_features")
        db = load_design("baseline")
        self.d = d
        X = np.asarray(d["X"], dtype=np.float64)
        self.Xaug = np.ascontiguousarray(np.hstack([X, np.ones((len(X), 1))]))
        self.y = np.ascontiguousarray(d["y"], dtype=np.float64)
        self.W = d["W"]
        self.names = d["names"]
        self.p = X.shape[1]
        self.m = self.p + 1
        self.n_test = len(X) - self.W
        self.bb = np.isin(self.names, db["names"])
        self.bbi = np.flatnonzero(self.bb)
        assert [self.names[j] for j in self.bbi] == db["names"]
        assert np.array_equal(X[:, self.bbi], db["X"]) and np.array_equal(self.y, db["y"])
        self.bb_aug = np.append(self.bb, False)
        self.grp = np.append(exog_groups(self.names, self.bb), -1).astype(np.int32)
        self.blocks = block_starts(self.n_test)
        self.bstart = np.array([b[0] for b in self.blocks] + [self.n_test], dtype=np.int32)
        self._st: dict = {}

    def structures(self, rs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nb, m = len(self.blocks), self.m
        lk = np.zeros((nb, len(rs), m), np.uint8)
        mk = np.zeros((nb, len(rs), m), np.uint8)
        pf = np.ones((nb, len(rs), m))
        for b, (i0, _) in enumerate(self.blocks):
            for i, r in enumerate(rs):
                key = (b, r)
                if key not in self._st:
                    self._st[key] = structure(self.Xaug[i0 : i0 + self.W], self.bb_aug, r)
                lk[b, i], mk[b, i], pf[b, i] = self._st[key]
        return lk, mk, pf


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
        print("compiled:", " ".join(cmd), flush=True)
    lib = ctypes.CDLL(str(so))
    f8 = ndpointer(np.float64, flags="C_CONTIGUOUS")
    i4 = ndpointer(np.int32, flags="C_CONTIGUOUS")
    u1 = ndpointer(np.uint8, flags="C_CONTIGUOUS")
    i8 = ndpointer(np.int64, flags="C_CONTIGUOUS")
    ci = ctypes.c_int
    lib.exogpen_run.restype = ci
    lib.exogpen_run.argtypes = [
        ci, ci, f8, f8, ci,  # N, m, X, y, W
        ci, ci, ci,  # est, mode, track
        ci, i4,  # n_blocks, bstart
        ci, u1, u1, f8,  # n_s, locked, maskout, pf
        ci, f8, ci, ci,  # n_alpha, alphas, val_tail, embargo
        i4, ci, f8, ci,  # group, n_groups, mpr_start, mpr_passes
        f8, f8, i4, f8, i4, f8, f8, i4, i8,  # outputs
    ]
    _LIB = lib
    return lib


def c_run(S: Setup, est: str, mode: str, grid: list[float], mpr_start=None,
          n_blocks: int | None = None) -> dict:
    """One arm through the C port (optionally only the first n_blocks blocks)."""
    lib = kernel()
    rs = R_OF[mode]
    nb = len(S.blocks) if n_blocks is None else n_blocks
    lk, mk, pf = (np.ascontiguousarray(a[:nb]) for a in S.structures(rs))
    bstart = np.ascontiguousarray(S.bstart[: nb + 1])
    n_out, m = int(bstart[-1]), S.m
    out = dict(
        pred=np.zeros(n_out), theta=np.zeros((n_out, m)), events=np.zeros(n_out, np.int32),
        val_mse=np.full((nb, len(rs), len(grid)), np.nan), choice=np.zeros((nb, 2), np.int32),
        pen_g=np.full((nb, len(GROUPS)), np.nan), mpr_mse=np.full(nb, np.nan),
        n_reseed=np.zeros(nb, np.int32), n_singular=np.zeros(1, np.int64),
    )
    est_i = {"ridge": 0, "reclasso": 1, "reclasticnet": 2}[est]
    ms = np.ascontiguousarray(
        np.zeros(nb) if mpr_start is None else np.asarray(mpr_start, float)[:nb]
    )
    t0 = time.process_time()
    rc = lib.exogpen_run(
        len(S.Xaug), m, S.Xaug, S.y, S.W, est_i, 1 if mode == "mpr" else 0,
        1 if est == "reclasso" else 0,  # the spec tracks degenerate columns for lasso only
        nb, bstart, len(rs), lk, mk, pf, len(grid),
        np.ascontiguousarray(grid, dtype=np.float64), VAL_TAIL, EMBARGO, S.grp, len(GROUPS),
        ms, MPR_PASSES, out["pred"], out["theta"], out["events"], out["val_mse"],
        out["choice"], out["pen_g"], out["mpr_mse"], out["n_reseed"], out["n_singular"],
    )
    out["cpu_sec"] = time.process_time() - t0
    if rc != 0:
        raise RuntimeError(f"kernel returned {rc} for {est} {mode}")
    ch = out["choice"]
    out["r_blk"] = np.array([rs[i] for i in ch[:, 0]])
    out["alpha_blk"] = (
        np.full(nb, np.nan) if mode == "mpr" else np.array([grid[i] for i in ch[:, 1]])
    )
    return out


def py_locked_slice(S: Setup, est: str, n_sess: int, b: int = 0) -> dict:
    """The Python reference for the backbone-locked arm: the spec's own RollingTunedLinear
    with the locked set and mask of structure(r = 0) at every tune (the block loop of
    dense_vs_sparse_1530.run_linear_blocks), on the first n_sess sessions of block b."""
    ns = spec_ns()
    Base = ns["RollingTunedLinear"]
    bb_aug = S.bb_aug

    class Locked(Base):  # type: ignore[misc, valid-type]
        def _recompute_mask(self, Xraw):
            self._locked, self._maskout, _ = structure(Xraw, bb_aug, 0.0)

    Locked.grid = ns["ESTIMATOR_GRIDS"][est]
    Locked.trace, Locked.mask_trace, Locked.reseed_trace = [], [], []
    X, y, W = S.Xaug[:, :-1], S.y, S.W
    i0 = int(S.bstart[b])
    reg = Locked()
    preds = []
    t0 = time.process_time()
    for j in range(i0, i0 + n_sess):
        t = W + j
        if j == i0:
            reg.init_window(X[i0:t], y[i0:t])
        else:
            reg.roll(X[t - 1], y[t - 1], X[j - 1], y[j - 1])
        reg.solve()
        preds.append(reg.predict_one(X[t]))
    return dict(pred=np.array(preds), alpha=float(reg.alpha_), sec=time.process_time() - t0,
                n_reseed=len(Locked.reseed_trace))


# ============================================================================ check
def check() -> None:
    """Gate (1): the C port's single-penalty arms == gate_spec_*.npz (the spec's class on
    the cached design) and the stored tables.  Gate (2): on a short slice, the C
    backbone-locked arm == the spec's class with the same locked set."""
    grids = spec_grids()
    S = Setup()
    idx = pd.DatetimeIndex(pd.to_datetime(S.d["date"][S.W :]))
    grows = []
    for est in EST:
        res = c_run(S, est, "single", grids[est])
        np.savez_compressed(WORK / f"check_c_single_{est}.npz", pred=res["pred"],
                            alpha=res["alpha_blk"])
        spec = np.load(WORK / f"gate_spec_{est}.npz")
        st = stored_1600(f"sub_{EST_SHORT[est]}_all_features").reindex(idx).to_numpy()
        spec_alpha = spec["alpha"][S.bstart[:-1]]
        rel_spec = np.abs(res["pred"] / spec["pred"] - 1)
        rel_st = np.abs(res["pred"] / st - 1)
        grows.append(dict(
            est=est,
            c_cpu_sec_full_arm=round(res["cpu_sec"], 2),
            python_spec_cpu_sec_full_arm=round(float(spec["sec"]), 2),
            alphas_c=json.dumps(res["alpha_blk"].tolist()),
            same_alphas_as_spec=bool(np.array_equal(res["alpha_blk"], spec_alpha)),
            max_rel_gap_vs_spec=float(rel_spec.max()),
            n_sessions_rel_gap_vs_spec_above_1e9=int((rel_spec > 1e-9).sum()),
            max_rel_gap_vs_stored=float(rel_st.max()),
            n_sessions_rel_gap_vs_stored_above_1e9=int((rel_st > 1e-9).sum()),
            spec_max_rel_gap_vs_stored=float(np.max(np.abs(spec["pred"] / st - 1))),
            n_reseed=int(res["n_reseed"].sum()),
            n_singular=int(res["n_singular"][0]),
            median_events=float(np.median(res["events"])),
        ))
        print(grows[-1], flush=True)
    pd.DataFrame(grows).to_csv(WORK / "check_c_vs_spec_gate.csv", index=False)
    rows = []
    n_sess = int(os.environ.get("EXOGPEN_SLICE", "40"))
    for est in EST:
        res = c_run(S, est, "bb0", grids[est], n_blocks=1)
        ref = py_locked_slice(S, est, n_sess)
        rel = np.abs(res["pred"][:n_sess] / ref["pred"] - 1)
        rows.append(dict(
            est=est, sessions=n_sess, alpha_c=float(res["alpha_blk"][0]),
            alpha_python=ref["alpha"], max_rel_gap=float(rel.max()),
            c_cpu_sec_block0_250_sessions=round(res["cpu_sec"], 2),
            python_cpu_sec_slice=round(ref["sec"], 2),
            n_reseed_c_block0=int(res["n_reseed"][0]), n_reseed_python_slice=ref["n_reseed"],
        ))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "check_c_vs_python_locked_slice.csv", index=False)


# ============================================================================ run
def ols_baseline(S: Setup) -> dict:
    """HAR + calendar OLS (the master table's sub_ols_baseline: production
    RollingLeastSquares at alpha 0 = centered window, least squares, minimum norm on a
    rank-deficient window), refit every session."""
    X, y, W, bbi = S.Xaug[:, :-1], S.y, S.W, S.bbi
    pred = np.empty(S.n_test)
    th = np.zeros((S.n_test, len(bbi)))
    t0 = time.process_time()
    for j in range(S.n_test):
        Xw, yw = X[j : W + j][:, bbi], y[j : W + j]
        xb, yb = Xw.mean(axis=0), yw.mean()
        b = np.linalg.lstsq(Xw - xb, yw - yb, rcond=None)[0]
        th[j] = b
        pred[j] = yb + float((X[W + j, bbi] - xb) @ b)
    return dict(pred=pred, theta_bb=th, cpu_sec=time.process_time() - t0)


def run(ests: list[str]) -> None:
    grids = spec_grids()
    S = Setup()
    runs = WORK / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if "ridge" in ests:
        r = ols_baseline(S)
        np.savez_compressed(runs / "ols_baseline.npz", **r)
        print(f"ols_baseline: {r['cpu_sec']:.1f}s", flush=True)
    bb0_alpha = None
    for est, mode in ARMS:
        if est not in ests:
            continue
        k = arm_key(est, mode)
        res = c_run(S, est, mode, grids[est],
                    mpr_start=bb0_alpha if mode == "mpr" else None)
        if est == "ridge" and mode == "bb0":
            bb0_alpha = res["alpha_blk"]
        lk, mk, _ = S.structures(R_OF[mode])
        ch = res["choice"][:, 0]
        nb = len(S.blocks)
        np.savez_compressed(
            runs / f"{k}.npz",
            pred=res["pred"], theta=res["theta"].astype(np.float32),
            alpha_blk=res["alpha_blk"], r_blk=res["r_blk"], val_mse=res["val_mse"],
            pen_g=res["pen_g"], mpr_mse=res["mpr_mse"], events=res["events"],
            n_reseed=res["n_reseed"], n_singular=res["n_singular"], cpu_sec=res["cpu_sec"],
            locked=np.array([lk[b, ch[b]] for b in range(nb)]),
            maskout=np.array([mk[b, ch[b]] for b in range(nb)]),
        )
        print(f"{k}: {res['cpu_sec']:.1f}s alpha {res['alpha_blk'].tolist()} "
              f"r {res['r_blk'].tolist()} reseeds {int(res['n_reseed'].sum())}"
              + (f" groups {np.round(res['pen_g'], 4).tolist()}" if mode == "mpr" else ""),
              flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "gate":
        gate()
    elif stage == "check":
        check()
    elif stage == "run":
        run(sys.argv[2:] or list(EST))
    else:
        raise SystemExit(__doc__)
