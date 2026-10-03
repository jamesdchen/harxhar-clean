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

Algorithm: the spec's own, specs/causal_tune_linear.py::RollingTunedLinear and the
parts of src/models/reclasso_har.py it calls, ported line by line to C
(experiments/close_exogpen_kernel.c, compiled on demand: gcc -O3 -march=native, no
-ffast-math, reference LAPACK / BLAS; called through ctypes).  Ridge: Sherman-Morrison
rank-one add / drop on the ridged inverse, theta = K c every session.  Lasso / elastic
net: the Garrigues-El Ghaoui online homotopy, two enet_online updates a session, warm
(theta, active set, signs), locked coordinates never leave the active set.  Every 250
sessions: the identifiability mask, the forward split (fit 1850 / embargo 25 / tail
125), the spec's grid scored by the batch solution (ridge solve; _batch_theta = FWL on
the locked block + batch homotopy), argmin, cold seed.  Lasso only: the between-re-choice
mask additions (_degenerate_live -> reseed).  Window 2000 sessions, refit every session,
intercept unpenalized, spec grids (ridge 1e-2 .. 1e3, lasso / elastic net 1e-6 .. 1e-2,
elastic net l1_ratio 0.5), sklearn units.

Penalty factors (glmnet's penalty.factor convention): the penalty on column j is
alpha * pf_j * (l1 |b_j| + (1 - l1) / 2 b_j^2); pf = 1 on the exogenous columns.
  single   pf = 1 everywhere: the spec's models (GATE: the stored forecasts).
  bb0      the backbone is LOCKED (as the spec's intercept: in the active set always, no
           penalty).  Backbone columns constant in the window, byte-copies of an earlier
           backbone column, or beyond the numerical rank of the centered backbone (the
           five weekday dummies sum to the intercept) are masked instead.
  bbr      backbone : exogenous ratio r tuned jointly with alpha on the same validation
           tail, r in R_GRID = {0, 0.01, 0.1, 1} (r = 0: locked as bb0; r > 0: the
           backbone's mu_vec, lam2 and ridge D scaled by r); r = 1 nests single, r = 0
           nests bb0, 0.01 is the paper's 1 : 100 in ridge units.
  bbfix    r fixed at the paper's 1/100.
  mpr      ridge, backbone locked, one penalty for each exogenous feature group
           (src.data.loading.SUBGROUPS: moments, liquidity, market_ew, market_vw,
           sentiment, implied_vol, vol_demand, fomc; an availability / activity indicator
           goes with its source's group): multi-penalty ridge (van de Wiel, van Nee &
           Rauschenberger 2021), the 8 penalties chosen on the validation tail by cyclic
           coordinate search over the ridge grid (start = the bb0 ridge's alpha;
           MPR_PASSES passes).
  singlew / bb0w (supplement)  single and bb0 with the grid widened upward (WIDE_GRIDS),
           because the spec-grid arms choose its top edge at most re-choices.
With pf != 1 the batch elastic net is solved exactly by column scaling.

Stages (``python experiments/close_exogpen.py <stage>``):
  gate            the spec's own class (dense_vs_sparse_1530.run_linear_blocks, the
                  spec's estimator section executed read-only) on the cached design vs
                  the stored 16:00 forecasts results/spxw_pnl/yhat_sub_<est>_all_features.
  check           the C port: single-penalty arms vs the gate and the stored tables (and
                  Python-vs-C seconds); on a short slice, the backbone-locked C arm vs the
                  spec's class with the same locked set.
  run [ests]      the arms of record (and the HAR + calendar OLS).
  run_supplement  the widened-grid arms.
  crosscheck      an independent solver (experiments/close_exogpen_cdcheck.c: centered
                  Gram, Cholesky ridge, coordinate descent + exact KKT solve) at a few dates.
  analyze         scoring (16:00-bar recalibration, QLIKE on the 866 trade days, the 15:30
                  sign(s) straddle Sharpe mid / crossed), Diebold-Mariano on daily QLIKE and
                  a HAC t on the paired daily P&L difference against the single-penalty arm
                  and the HAR + calendar arm, the penalty path, backbone shrinkage, variance
                  shares, figures, SUMMARY.md, then the DONE flag.

The circular block bootstrap is off in this branch (commit b761b28): point estimates
only; differences are tested with Diebold-Mariano (src.evaluation.diebold_mariano) and
the Newey-West t of notebooks/atm_straddle_lib.newey_west_t, as master_table_close.py.

Compute: single-threaded BLAS (set below before numpy loads); at most two processes.
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
CHECK_EVERY = 25  # sessions between exact batch solutions (warm-path drift check)


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
R_OF = {"single": (1.0,), "bb0": (0.0,), "bbr": R_GRID, "bbfix": (R_PAPER,), "mpr": (0.0,),
        "singlew": (1.0,), "bb0w": (0.0,)}
# supplement: the spec's grids widened upward (decades), because the one-penalty and the
# backbone-unpenalized arms choose the top edge of the spec's grid at most re-choices
WIDE_GRIDS = {
    "ridge": [float(a) for a in np.logspace(-2, 6, 9)],
    "reclasso": [float(a) for a in np.logspace(-6, 0, 7)],
    "reclasticnet": [float(a) for a in np.logspace(-6, 0, 7)],
}
SUPP_ARMS = [(e, m) for e in EST for m in ("singlew", "bb0w")]


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
        ci,  # check_every
        f8, f8, i4, f8, i4, f8, f8, i4, i8, f8,  # outputs
    ]
    _LIB = lib
    return lib


def c_run(S: Setup, est: str, mode: str, grid: list[float], mpr_start=None,
          n_blocks: int | None = None, check_every: int = 0) -> dict:
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
        n_reseed=np.zeros(nb, np.int32), n_singular=np.zeros(3, np.int64),  # LU fallbacks, pf != 1 batch repairs, uncertified
        exact_pred=np.full(n_out, np.nan),
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
        ms, MPR_PASSES, check_every, out["pred"], out["theta"], out["events"],
        out["val_mse"], out["choice"], out["pen_g"], out["mpr_mse"], out["n_reseed"],
        out["n_singular"], out["exact_pred"],
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

    # the spec's _tune reads the grid off the base class by name
    Base.grid = Locked.grid = ns["ESTIMATOR_GRIDS"][est]
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
    for est in EST if os.environ.get("EXOGPEN_SKIP_FULL") != "1" else ():
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
    if grows:
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


def run(ests: list[str], supplement: bool = False) -> None:
    grids = spec_grids()
    if supplement:
        for e in EST:  # the widened grid contains the spec's
            assert set(grids[e]) <= set(WIDE_GRIDS[e]), e
    S = Setup()
    runs = WORK / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if "ridge" in ests and not supplement:
        r = ols_baseline(S)
        np.savez_compressed(runs / "ols_baseline.npz", **r)
        print(f"ols_baseline: {r['cpu_sec']:.1f}s", flush=True)
    bb0_alpha = None
    only = [m for m in os.environ.get("EXOGPEN_MODES", "").split(",") if m]
    for est, mode in SUPP_ARMS if supplement else ARMS:
        if est not in ests or (only and mode not in only):
            continue
        k = arm_key(est, mode)
        res = c_run(S, est, mode, WIDE_GRIDS[est] if mode.endswith("w") else grids[est],
                    mpr_start=bb0_alpha if mode == "mpr" else None,
                    check_every=CHECK_EVERY if est != "ridge" else 0)
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
            exact_pred=res["exact_pred"],
            locked=np.array([lk[b, ch[b]] for b in range(nb)]),
            maskout=np.array([mk[b, ch[b]] for b in range(nb)]),
        )
        print(f"{k}: {res['cpu_sec']:.1f}s alpha {res['alpha_blk'].tolist()} "
              f"r {res['r_blk'].tolist()} reseeds {int(res['n_reseed'].sum())} "
              f"[LU fallbacks, batch repairs, uncertified] {res['n_singular'].tolist()}"
              + (f" groups {np.round(res['pen_g'], 4).tolist()}" if mode == "mpr" else ""),
              flush=True)


# ============================================================================ cross-check
def crosscheck(n_sess: int = 10) -> None:
    """An independent solver at a few dates: experiments/close_exogpen_cdcheck.c (centered
    Gram, Cholesky ridge, covariance-form coordinate descent + exact KKT solve on the
    support for lasso / elastic net) on the first n_sess sessions of block 0 of the
    backbone-unpenalized arms, at the alpha the C port chose there."""
    import ctypes
    import subprocess

    from numpy.ctypeslib import ndpointer

    src = REPO / "experiments" / "close_exogpen_cdcheck.c"
    so = WORK / "close_exogpen_cdcheck.so"
    if not so.is_file() or so.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["gcc", "-O3", "-march=native", "-fPIC", "-shared", "-o", str(so), str(src),
                        "-l:liblapack.so.3", "-l:libblas.so.3", "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    f8 = ndpointer(np.float64, flags="C_CONTIGUOUS")
    i4 = ndpointer(np.int32, flags="C_CONTIGUOUS")
    i8 = ndpointer(np.int64, flags="C_CONTIGUOUS")
    ci, cd, cl = ctypes.c_int, ctypes.c_double, ctypes.c_long
    lib.exogpen_cd_run.restype = ci
    lib.exogpen_cd_run.argtypes = [ci, ci, f8, f8, ci, ci, ci, ci, i4, ci, f8, ci, f8, ci, ci,
                                   i4, ci, f8, ci, i4, cd, cd, ci, cl,
                                   f8, f8, f8, f8, f8, f8, i4, i8, f8, i4, f8, f8]
    S = Setup()
    X = np.ascontiguousarray(S.Xaug[:, :-1])
    p = S.p
    lk, mk, _ = S.structures((0.0,))
    pf = np.where(mk[0, 0].astype(bool), -1.0, np.where(lk[0, 0].astype(bool), 0.0, 1.0))[:-1]
    pf = np.ascontiguousarray(pf.reshape(1, 1, p))
    rows = []
    for est in EST:
        z = np.load(WORK / "runs" / f"{arm_key(est, 'bb0')}.npz")
        a = np.array([float(z["alpha_blk"][0])])
        o = {k: np.zeros(n_sess) for k in ("pred", "b0", "vBB", "vEE", "vBE")}
        th = np.zeros((n_sess, p))
        st, sw = np.zeros(n_sess, np.int32), np.zeros(n_sess, np.int64)
        t0 = time.process_time()
        rc = lib.exogpen_cd_run(
            len(X), p, X, S.y, S.W, {"ridge": 0, "reclasso": 1, "reclasticnet": 2}[est], 0, 1,
            np.array([0, n_sess], np.int32), 1, pf, 1, a, VAL_TAIL, EMBARGO,
            np.zeros(p, np.int32), len(GROUPS), np.zeros(1), MPR_PASSES,
            np.ascontiguousarray(S.bb, dtype=np.int32), 1e-8, 1e-9, 6, 200000,
            o["pred"], th, o["b0"], o["vBB"], o["vEE"], o["vBE"], st, sw,
            np.zeros(1), np.zeros(2, np.int32), np.zeros(len(GROUPS)), np.zeros(1))
        assert rc == 0, rc
        gap = np.abs(o["pred"] / z["pred"][:n_sess] - 1.0)
        rows.append(dict(est=est, alpha=float(a[0]), sessions=n_sess, max_rel_gap=float(gap.max()),
                         cd_status=json.dumps(np.bincount(st, minlength=4).tolist()),
                         cpu_sec=round(time.process_time() - t0, 2)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "check_cd_crosscheck.csv", index=False)


# ============================================================================ analyze
MODE_LABEL = {
    "singlew": "one penalty, grid widened (supplement)",
    "bb0w": "backbone unpenalized, grid widened (supplement)",
    "single": "one penalty (spec)",
    "bb0": "backbone unpenalized",
    "bbr": "backbone ratio r tuned",
    "bbfix": "backbone ratio 1/100",
    "mpr": "backbone unpenalized, one penalty for each group",
}
COLOR = {"ridge": "#2a78d6", "reclasso": "#4a3aa7", "reclasticnet": "#eb6834"}
MARKER = {"baseline": "s", "single": "o", "bb0": "D", "bbr": "^", "bbfix": "v", "mpr": "P",
          "singlew": "o", "bb0w": "D"}
ALL_ARMS = ARMS + SUPP_ARMS


def _fmt(v, nd=4):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def analyze() -> None:  # noqa: C901 - one linear report
    import atm_straddle_lib as asl
    import dense_vs_sparse_1530 as dvs

    from src.evaluation.diebold_mariano import dm_test

    S = Setup()
    d = S.d
    W = S.W
    runs = WORK / "runs"
    dz = forecast_dz(d)
    idx = pd.DatetimeIndex(pd.to_datetime(dz["date"]))
    master = pd.read_csv(MASTER).set_index("key")
    # ---- every forecast scored: the study's arms, the stored references, the OLS
    R: dict[str, dict] = {}
    for est, mode in ALL_ARMS:
        f = runs / f"{arm_key(est, mode)}.npz"
        if not f.is_file():
            raise SystemExit(f"missing run {f}")
        z = np.load(f)
        R[arm_key(est, mode)] = dict(est=est, mode=mode, pred=z["pred"], z=z)
    zo = np.load(runs / "ols_baseline.npz")
    R["ols_baseline"] = dict(est="ols", mode="baseline", pred=zo["pred"], z=zo)
    for est in EST:
        for b in ("baseline", "all_features"):
            k = f"sub_{EST_SHORT[est]}_{b}"
            R[f"stored_{k}"] = dict(
                est=est, mode="baseline" if b == "baseline" else "stored_single",
                pred=stored_1600(k).reindex(idx).to_numpy(), master_key=k,
            )
    keys = list(R)
    frames = [dvs.research_frame(dz, R[k]["pred"]) for k in keys]
    dk = dvs.deck_frame()
    P = dvs.deck_panel(frames, dk)
    pt = dvs.point(P)
    col = {k: i for i, k in enumerate(keys)}
    # ---- scorer gate: the stored forecasts and the OLS against the master table
    gate_rows = []
    for k in keys:
        mk = R[k].get("master_key", "sub_ols_baseline" if k == "ols_baseline" else None)
        if mk is None:
            continue
        i = col[k]
        gate_rows.append(dict(
            forecast=k, master_key=mk,
            qlike=pt["ql"][i], qlike_master=master.loc[mk, "qlike_recal"],
            sharpe_mid=pt["sh"][i], sharpe_mid_master=master.loc[mk, "Sharpe_mid"],
            sharpe_crossed=pt["shx"][i], sharpe_crossed_master=master.loc[mk, "Sharpe_crossed"],
        ))
    gate_df = pd.DataFrame(gate_rows)
    gate_df["abs_diff_qlike"] = (gate_df["qlike"] - gate_df["qlike_master"]).abs()
    gate_df["abs_diff_sharpe_mid"] = (gate_df["sharpe_mid"] - gate_df["sharpe_mid_master"]).abs()
    gate_df.to_csv(OUT / "scorer_gate.csv", index=False)

    # ---- headline table: point estimates, DM on daily QLIKE, HAC t on daily P&L difference
    def contrast(a: str, b: str) -> dict:
        ia, ib = col[a], col[b]
        dm = dm_test(P["ql"][:, ia], P["ql"][:, ib])
        t, lag = asl.newey_west_t(P["pnl"][:, ia] - P["pnl"][:, ib])
        return dict(dq_pct=100.0 * (pt["ql"][ia] / pt["ql"][ib] - 1.0), dm=dm["dm"],
                    dm_p=dm["p"], dm_lag=dm["hac_lag"], dsh=pt["sh"][ia] - pt["sh"][ib],
                    t_hac=t, hac_lag=lag,
                    same_position=float(np.mean((P["pnl"][:, ia] > 0) == (P["pnl"][:, ib] > 0))))

    rows = []
    for k in keys:
        est, mode = R[k]["est"], R[k]["mode"]
        i = col[k]
        row = dict(key=k, estimator=LABEL.get(est, "OLS"), mode=mode,
                   label={**MODE_LABEL, "baseline": "HAR + calendar",
                          "stored_single": "one penalty (stored table)"}[mode],
                   qlike=pt["ql"][i], sharpe_mid=pt["sh"][i], sharpe_crossed=pt["shx"][i],
                   pct_buy=100.0 * P["buy"][i], n_days=P["ql"].shape[0])
        if k in R and "z" in R[k] and mode not in ("baseline",):
            single = arm_key(est, "single")
            base = f"stored_sub_{EST_SHORT[est]}_baseline"
            if k != single:
                c = contrast(k, single)
                row.update({f"{n}_vs_single": v for n, v in c.items()})
            c = contrast(k, base)
            row.update({f"{n}_vs_baseline": v for n, v in c.items()})
            c = contrast(k, "ols_baseline")
            row.update({f"{n}_vs_ols": v for n, v in c.items()})
        rows.append(row)
    head = pd.DataFrame(rows)
    head.to_csv(OUT / "headline.csv", index=False)

    # ---- penalty path at every re-choice (grid edges), the group penalties, drift check
    grids = spec_grids()
    path_rows, grp_rows, drift_rows, cpu_rows = [], [], [], []
    for est, mode in ALL_ARMS:
        k = arm_key(est, mode)
        z = R[k]["z"]
        g = WIDE_GRIDS[est] if mode.endswith("w") else grids[est]
        for b, (i0, _) in enumerate(S.blocks):
            a = float(z["alpha_blk"][b])
            vm = z["val_mse"][b]
            path_rows.append(dict(
                arm=k, block=b, first_forecast=str(dz["date"][i0])[:10], alpha=a,
                r=float(z["r_blk"][b]),
                alpha_at_low_edge=bool(a == min(g)), alpha_at_high_edge=bool(a == max(g)),
                val_mse=float(np.nanmin(vm)) if mode != "mpr" else float(z["mpr_mse"][b]),
                n_locked=int(z["locked"][b].sum()) - 1, n_masked=int(z["maskout"][b].sum()),
                n_reseed=int(z["n_reseed"][b]),
            ))
            if mode == "mpr":
                for gi, gname in enumerate(GROUPS):
                    pv = float(z["pen_g"][b, gi])
                    grp_rows.append(dict(block=b, first_forecast=str(dz["date"][i0])[:10],
                                         group=gname, penalty=pv,
                                         n_cols=int(((S.grp == gi) & ~z["maskout"][b].astype(bool)).sum()),
                                         at_low_edge=bool(pv == min(g)), at_high_edge=bool(pv == max(g))))
        ex = z["exact_pred"]
        ok = np.isfinite(ex)
        if ok.any():
            gap = np.abs(z["pred"][ok] / ex[ok] - 1.0)
            drift_rows.append(dict(arm=k, n_checked=int(ok.sum()), max_rel_gap=float(gap.max()),
                                   median_rel_gap=float(np.median(gap)),
                                   n_checked_rel_gap_above_1e6=int((gap > 1e-6).sum())))
        cpu_rows.append(dict(arm=k, cpu_sec=float(z["cpu_sec"])))
    pd.DataFrame(path_rows).to_csv(OUT / "penalty_path.csv", index=False)
    pd.DataFrame(grp_rows).to_csv(OUT / "group_penalties.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUT / "warm_path_vs_exact.csv", index=False)
    pd.DataFrame(cpu_rows).to_csv(OUT / "cpu_seconds.csv", index=False)

    # ---- backbone shrinkage vs the HAR + calendar OLS, and where the forecast variance sits
    X = S.Xaug[:, :-1]
    bb, ex_ = S.bb, ~S.bb
    har = np.array([n.startswith("har_ma_") for n in S.names])
    har_in_bb = har[S.bbi]
    arms = [arm_key(e, m) for e, m in ALL_ARMS]
    TH = {k: np.asarray(R[k]["z"]["theta"], dtype=np.float64)[:, :-1] for k in arms}
    th_ols = np.asarray(zo["theta_bb"])
    n_test = S.n_test
    stats = {k: {n: np.empty(n_test) for n in ("vBB", "vEE", "vBE", "vOLS", "phiB", "phiE")}
             for k in arms}
    phi_ols = np.empty(n_test)
    cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    for j in range(n_test):
        xbar = (cs[W + j] - cs[j]) / W
        Xc = X[j : W + j] - xbar
        cols = []
        for k in arms:
            th = TH[k][j]
            cols += [np.where(bb, th, 0.0), np.where(ex_, th, 0.0)]
        F = Xc @ np.column_stack(cols)
        fo = Xc[:, S.bbi] @ th_ols[j]
        vo = float(fo @ fo) / W
        dx = X[W + j] - xbar
        phi_ols[j] = float(dx[S.bbi] @ th_ols[j])
        for ai, k in enumerate(arms):
            fb, fe = F[:, 2 * ai], F[:, 2 * ai + 1]
            st = stats[k]
            st["vBB"][j] = float(fb @ fb) / W
            st["vEE"][j] = float(fe @ fe) / W
            st["vBE"][j] = float(fb @ fe) / W
            st["vOLS"][j] = vo
            th = TH[k][j]
            st["phiB"][j] = float(dx[bb] @ th[bb])
            st["phiE"][j] = float(dx[ex_] @ th[ex_])
    shr_rows, har_rows = [], []
    ols_har_sum = th_ols[:, har_in_bb].sum(axis=1)
    for k in arms:
        st = stats[k]
        th = TH[k]
        tot = st["vBB"] + st["vEE"] + 2.0 * st["vBE"]
        pb, pe = st["phiB"], st["phiE"]
        vf = np.var(pb + pe)
        shr_rows.append(dict(
            arm=k,
            har_sum_ratio_median=float(np.median(th[:, har].sum(axis=1) / ols_har_sum)),
            backbone_var_ratio_median=float(np.median(st["vBB"] / st["vOLS"])),
            share_backbone_window_median=float(np.median(st["vBB"] / tot)),
            share_exog_window_median=float(np.median(st["vEE"] / tot)),
            share_cross_window_median=float(np.median(2.0 * st["vBE"] / tot)),
            share_backbone_forecasts=float(np.var(pb) / vf),
            share_exog_forecasts=float(np.var(pe) / vf),
            share_cross_forecasts=float(2.0 * np.cov(pb, pe, ddof=0)[0, 1] / vf),
            corr_backbone_part_with_ols=float(np.corrcoef(pb, phi_ols)[0, 1]),
            n_exog_nonzero_median=float(np.median((th[:, ex_] != 0).sum(axis=1))),
        ))
        for jj, name in enumerate(np.array(S.names)[har]):
            har_rows.append(dict(arm=k, column=name,
                                 coef_median=float(np.median(th[:, har][:, jj])),
                                 ols_coef_median=float(np.median(th_ols[:, har_in_bb][:, jj]))))
    shr = pd.DataFrame(shr_rows)
    shr.to_csv(OUT / "backbone_shrinkage_and_variance_shares.csv", index=False)
    pd.DataFrame(har_rows).to_csv(OUT / "har_coefficients.csv", index=False)
    figures(head, pd.DataFrame(path_rows), shr, grids)
    write_summary()
    (OUT / "DONE").touch()
    print("analyze done", flush=True)


def figures(head: pd.DataFrame, path: pd.DataFrame, shr: pd.DataFrame, grids: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # 1. QLIKE and Sharpe (two panels, one scale each)
    order = []
    for est in EST:
        order += [f"stored_sub_{EST_SHORT[est]}_baseline"] + [arm_key(est, m) for m in MODES]
        if est == "ridge":
            order.append("ridge_mpr")
        order += [arm_key(est, "singlew"), arm_key(est, "bb0w")]
    order.append("ols_baseline")
    h = head.set_index("key").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.0), sharey=True)
    ypos = np.arange(len(order))[::-1]
    for ax, colname, xl in ((axes[0], "qlike", "QLIKE (866 trade days, lower is better)"),
                            (axes[1], "sharpe_mid", "sign(s) straddle Sharpe, mid fill")):
        for y_, k in zip(ypos, order):
            r = h.loc[k]
            est = {"ridge": "ridge", "lasso": "reclasso", "elastic net": "reclasticnet"}.get(r["estimator"])
            c = COLOR.get(est, "#7a7a7a")
            ax.plot(r[colname], y_, MARKER.get(r["mode"], "o"), color=c, ms=7,
                    mfc="white" if r["mode"] == "baseline" else c, mew=1.6)
        ax.set_xlabel(xl)
        ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    lab = []
    for k in order:
        r = h.loc[k]
        lab.append(f"{r['estimator']}: {r['label']}")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(lab)
    fig.suptitle("16:00-bar linear models on all_features: penalty structure vs HAR + calendar "
                 "(hollow = HAR + calendar arm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_qlike_sharpe.png", dpi=150)
    plt.close(fig)
    # 2. penalty chosen at each re-choice
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, est in zip(axes, EST):
        for mode in MODES + ("bb0w",):
            k = arm_key(est, mode)
            pp = path[path["arm"] == k]
            ax.plot(pp["block"], pp["alpha"], marker=MARKER[mode], color=COLOR[est],
                    lw=1.5, ms=6, alpha=0.9, label=MODE_LABEL[mode],
                    ls={"single": "-", "bb0": "--", "bbr": ":", "bbfix": "-.", "bb0w": (0, (5, 1, 1, 1))}[mode])
        for a in (min(grids[est]), max(grids[est]), max(WIDE_GRIDS[est])):
            ax.axhline(a, color="#9a9a9a", lw=0.8, ls=(0, (2, 2)))
        ax.set_yscale("log")
        ax.set_title(LABEL[est])
        ax.set_xlabel("re-choice (every 250 sessions)")
    axes[0].set_ylabel("alpha chosen (dotted = spec grid edges, widened top)")
    axes[2].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_penalty_path.png", dpi=150)
    plt.close(fig)
    # 3. share of the forecasts' variance on the backbone and on the exogenous columns
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    arms = [arm_key(e, m) for e, m in ALL_ARMS]
    s2 = shr.set_index("arm").loc[arms]
    yp = np.arange(len(arms))[::-1]
    for y_, k in zip(yp, arms):
        est = R_EST(k)
        ax.plot(s2.loc[k, "share_backbone_forecasts"], y_, "o", color=COLOR[est], ms=7)
        ax.plot(s2.loc[k, "share_exog_forecasts"], y_, "o", color=COLOR[est], ms=7, mfc="white", mew=1.6)
    ax.set_yticks(yp)
    ax.set_yticklabels([f"{LABEL[R_EST(k)]}: {MODE_LABEL[k.split('_', 1)[1]]}" for k in arms])
    ax.set_xlabel("share of the forecasts' variance (filled = backbone part, hollow = exogenous part)")
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_variance_shares.png", dpi=150)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_summary() -> None:  # noqa: C901 - one linear report
    """SUMMARY.md from this folder's CSVs (and the gate CSVs in _work)."""
    head = pd.read_csv(OUT / "headline.csv").set_index("key")
    sg = pd.read_csv(OUT / "scorer_gate.csv")
    path = pd.read_csv(OUT / "penalty_path.csv")
    grp = pd.read_csv(OUT / "group_penalties.csv")
    drift = pd.read_csv(OUT / "warm_path_vs_exact.csv")
    cpu = pd.read_csv(OUT / "cpu_seconds.csv")
    shr = pd.read_csv(OUT / "backbone_shrinkage_and_variance_shares.csv").set_index("arm")
    harc = pd.read_csv(OUT / "har_coefficients.csv")
    g0 = pd.read_csv(WORK / "gate_spec_vs_stored.csv")
    g1 = pd.read_csv(WORK / "check_c_vs_spec_gate.csv")
    g2 = pd.read_csv(WORK / "check_c_vs_python_locked_slice.csv")
    L: list[str] = []
    say = L.append
    say("# The 16:00-bar linear models with the HAR + calendar backbone left (nearly) unpenalized")
    say("")
    say(f"Written by `experiments/close_exogpen.py analyze` on {time.strftime('%Y-%m-%d %H:%M')}; "
        "every number below is read from the CSVs in this folder (and the gate CSVs in `_work/`).")
    say("")
    say("## Question and design")
    say("")
    say("The user: the other shrinkage models should also leave the base HAR features alone and mainly "
        "penalize the exogenous features, as the paper's 2-block ridge does (backbone penalty 1, "
        "exogenous block 100). The 16:00-bar linear models (`specs/causal_tune_linear.py`) put one "
        "penalty on every column. Here the same models are refit on the `all_features` 16:00-bar "
        "design (628 columns; cache `results/close_design/_work/design_bar1600_all_features.npz`) "
        "with a penalty factor on the backbone (the 22 columns of the `baseline` design: the target's "
        "HAR ladder `har_ma_1 .. har_ma_3125` and the calendar / expiry columns), glmnet's "
        "penalty.factor convention: column j carries alpha x pf_j x (l1 |b_j| + (1 - l1) / 2 b_j^2), "
        "pf = 1 on the 606 exogenous columns.")
    say("")
    say("| arm | backbone | alpha | ratio r = backbone : exogenous penalty |")
    say("|---|---|---|---|")
    say("| one penalty (spec) | penalized like every column | spec grid, re-chosen every 250 sessions | 1 |")
    say("| backbone unpenalized | locked: in the active set always, no penalty (as the spec's intercept) | spec grid | 0 |")
    say("| backbone ratio r tuned | r chosen jointly with alpha on the same validation tail | spec grid | {0, 0.01, 0.1, 1} |")
    say("| backbone ratio 1/100 | penalized at r times the exogenous penalty | spec grid | 0.01 (the paper's 1 : 100 in ridge units) |")
    say("| ridge, one penalty for each group | locked | one ridge penalty for each of the 8 exogenous feature groups (multi-penalty ridge; van de Wiel, van Nee & Rauschenberger 2021), chosen on the validation tail by cyclic coordinate search over the ridge grid (2 passes, start = the backbone-unpenalized ridge's alpha) | 0 |")
    say("")
    say("Why r is tuned rather than fixed: the paper's 1/100 was set for ridge on the pooled 48-bar design; "
        "for an L1 penalty the same ratio is a different amount of shrinkage, so the ratio is chosen "
        "causally with alpha (r = 1 nests the spec's model, r = 0 nests the unpenalized backbone), "
        "and the fixed 1/100 arm is recorded beside it.")
    say("")
    say("Protocol (the spec's, unchanged): window = the 2000 sessions before the forecast, refit every "
        "session, intercept unpenalized, the identifiability mask at every re-choice, penalty re-chosen "
        "every 250 sessions on the last 125 sessions of the window after a 25-session embargo, the "
        "spec's grids (ridge 1e-2 .. 1e3; lasso / elastic net 1e-6 .. 1e-2; elastic net l1_ratio 0.5). "
        "When the backbone is locked, backbone columns constant in the window, byte-copies of an "
        "earlier backbone column, or beyond the numerical rank of the centered backbone are masked "
        "(the five weekday dummies sum to the intercept, so one of them is left out; fit and "
        "forecasts are unchanged by that choice).")
    say("")
    say("Algorithm: the spec's own (`RollingTunedLinear` and `src/models/reclasso_har.py`), ported "
        "to C (`experiments/close_exogpen_kernel.c`, gcc -O3 -march=native, no -ffast-math, reference "
        "LAPACK / BLAS): ridge = Sherman-Morrison rank-one add / drop on the ridged inverse; lasso / "
        "elastic net = the Garrigues-El Ghaoui online homotopy (two `enet_online` updates a session), "
        "cold seed `_batch_theta` (FWL on the locked block + batch homotopy) at every re-choice, and "
        "for the lasso the between-re-choice mask additions (`_degenerate_live`). With pf != 1 the "
        "batch elastic net is solved exactly by column scaling.")
    say("")
    say("## Gates")
    say("")
    say("1. The spec's own class (executed read-only on the cached design) against the stored 16:00 "
        "forecasts `results/spxw_pnl/yhat_sub_<est>_all_features.parquet`:")
    say("")
    t = g0[["est", "n", "max_rel_diff", "cpu_sec", "alphas"]].copy()
    t["max_rel_diff"] = t["max_rel_diff"].map(lambda v: f"{v:.1e}")
    t["cpu_sec"] = t["cpu_sec"].map(lambda v: f"{v:.1f}")
    say(_md_table(t))
    say("")
    say("2. The C port, one penalty, all 1469 sessions, against (1) and against the stored tables "
        "(relative gap of the 16:00 forecast; CPU seconds for one full arm, Python spec class vs C):")
    say("")
    t = g1[["est", "same_alphas_as_spec", "max_rel_gap_vs_spec", "n_sessions_rel_gap_vs_spec_above_1e9",
            "max_rel_gap_vs_stored", "n_sessions_rel_gap_vs_stored_above_1e9",
            "python_spec_cpu_sec_full_arm", "c_cpu_sec_full_arm"]].copy()
    for c in ("max_rel_gap_vs_spec", "max_rel_gap_vs_stored"):
        t[c] = t[c].map(lambda v: f"{v:.1e}")
    say(_md_table(t))
    say("")
    en = g1.set_index("est").loc["reclasticnet"]
    say(f"   The elastic net is the exception: the C port differs from the spec class on "
        f"{int(en['n_sessions_rel_gap_vs_spec_above_1e9'])} sessions and from the stored table on "
        f"{int(en['n_sessions_rel_gap_vs_stored_above_1e9'])} (largest {en['max_rel_gap_vs_stored']:.2%}), "
        f"and the spec class itself differs from the stored table by up to {en['spec_max_rel_gap_vs_stored']:.2%}. "
        "All three are the same warm homotopy; at alpha 1e-3 (blocks 3 and 4) it leaves the exact "
        "path at different sessions on different floating-point paths (local Python spec from 2021-11-15 "
        "in block 3, the stored run from block 4, the C port late in block 4) and returns to it at the "
        "next re-choice (cold reseed). Checked against the exact batch solution (`_batch_theta` on the "
        "same window) in `warm_path_vs_exact.csv`.")
    say("")
    say("3. On a short slice (the first sessions of block 0), the C backbone-locked arm against the "
        "spec's own class with the same locked set (a subclass whose mask step locks the backbone):")
    say("")
    t = g2[["est", "sessions", "alpha_c", "alpha_python", "max_rel_gap"]].copy()
    t["max_rel_gap"] = t["max_rel_gap"].map(lambda v: f"{v:.1e}")
    say(_md_table(t))
    say("")
    say("4. The scorer (16:00-bar recalibration (f^2 + s) x B, QLIKE on the 866 trade days, the 15:30 "
        "sign(s) straddle; `experiments/dense_vs_sparse_1530.py` helpers) against the master table:")
    say("")
    t = sg[["forecast", "qlike", "qlike_master", "sharpe_mid", "sharpe_mid_master"]].copy()
    for c in ("qlike", "qlike_master"):
        t[c] = t[c].map(lambda v: f"{v:.4f}")
    for c in ("sharpe_mid", "sharpe_mid_master"):
        t[c] = t[c].map(lambda v: f"{v:.2f}")
    say(_md_table(t))
    say("")
    say("5. Warm path vs the exact batch solution, every 25th session (lasso / elastic net arms):")
    say("")
    t = drift.copy()
    t["max_rel_gap"] = t["max_rel_gap"].map(lambda v: f"{v:.1e}")
    t["median_rel_gap"] = t["median_rel_gap"].map(lambda v: f"{v:.1e}")
    say(_md_table(t))
    say("")
    cdx = WORK / "check_cd_crosscheck.csv"
    if cdx.is_file():
        g3 = pd.read_csv(cdx)
        say("6. An independent solver at a few dates (`experiments/close_exogpen_cdcheck.c`: centered "
            "Gram, Cholesky ridge, covariance-form coordinate descent + exact KKT solve on the support) "
            "against the C port's backbone-unpenalized arms, first sessions of block 0:")
        say("")
        t = g3[["est", "alpha", "sessions", "max_rel_gap"]].copy()
        t["max_rel_gap"] = t["max_rel_gap"].map(lambda v: f"{v:.1e}")
        say(_md_table(t))
        say("")
    say("## QLIKE and the trade (point estimates; the block bootstrap is off, commit b761b28)")
    say("")
    say("DM = Diebold-Mariano statistic on the daily QLIKE difference (negative: the row forecasts "
        "better); t = Newey-West HAC t of the daily mid-fill P&L difference (positive: the row trades "
        "better). Reference 'one penalty' = the C port's spec model of the same estimator; reference "
        "'HAR + calendar' = the stored 16:00-bar model of the same estimator on the `baseline` design "
        "(the master table's `sub_<est>_baseline`); the multi-penalty ridge is set against the ridge rows.")
    say("")
    rows = []
    order = []
    for est in EST:
        order += [f"stored_sub_{EST_SHORT[est]}_baseline"] + [arm_key(est, m) for m in MODES]
        if est == "ridge":
            order.append("ridge_mpr")
        order += [arm_key(est, "singlew"), arm_key(est, "bb0w")]
    order.append("ols_baseline")
    for k in order:
        r = head.loc[k]

        def g(c, nd=2):
            v = r.get(c, np.nan)
            return "" if pd.isna(v) else f"{v:+.{nd}f}"

        def pv(c):
            v = r.get(c, np.nan)
            return "" if pd.isna(v) else f"{v:.3f}"

        rows.append({
            "model": f"{r['estimator']}: {r['label']}",
            "QLIKE": f"{r['qlike']:.4f}",
            "Sharpe mid / crossed": f"{r['sharpe_mid']:.2f} / {r['sharpe_crossed']:.2f}",
            "vs one penalty: dQLIKE %, DM (p)": "" if pd.isna(r.get("dm_vs_single", np.nan)) else
                f"{g('dq_pct_vs_single', 1)}, {g('dm_vs_single')} ({pv('dm_p_vs_single')})",
            "vs one penalty: dSharpe, t": "" if pd.isna(r.get("t_hac_vs_single", np.nan)) else
                f"{g('dsh_vs_single')}, {g('t_hac_vs_single')}",
            "vs HAR + calendar: dQLIKE %, DM (p)": "" if pd.isna(r.get("dm_vs_baseline", np.nan)) else
                f"{g('dq_pct_vs_baseline', 1)}, {g('dm_vs_baseline')} ({pv('dm_p_vs_baseline')})",
            "vs HAR + calendar: dSharpe, t": "" if pd.isna(r.get("t_hac_vs_baseline", np.nan)) else
                f"{g('dsh_vs_baseline')}, {g('t_hac_vs_baseline')}",
        })
    say(_md_table(pd.DataFrame(rows)))
    say("")
    hk = head.loc["ols_baseline"]
    say(f"HAR + calendar OLS (master table `sub_ols_baseline`): QLIKE {hk['qlike']:.4f}, Sharpe "
        f"{hk['sharpe_mid']:.2f} mid / {hk['sharpe_crossed']:.2f} crossed. Each new arm against it: "
        "columns `*_vs_ols` of `headline.csv`.")
    say("")
    say("## The penalty chosen at each re-choice")
    say("")
    pv_ = path.copy()
    pv_["choice"] = [
        (f"{a:g}" if np.isfinite(a) else "groups") + (f" (r {r_:g})" if k.endswith("bbr") else "")
        + (" *" if (lo or hi) else "")
        for a, r_, k, lo, hi in zip(pv_["alpha"], pv_["r"], pv_["arm"], pv_["alpha_at_low_edge"],
                                    pv_["alpha_at_high_edge"])
    ]
    tab = pv_.pivot(index="arm", columns="first_forecast", values="choice")
    tab = tab.loc[[arm_key(e, m) for e, m in ALL_ARMS]].reset_index()
    say("`*` = at an edge of the spec's grid.")
    say("")
    say(_md_table(tab))
    say("")
    n_edge = int((path["alpha_at_low_edge"] | path["alpha_at_high_edge"]).sum())
    n_all = int(np.isfinite(path["alpha"]).sum())
    say(f"{n_edge} of {n_all} alpha choices sit at an edge of their grid (spec grids: ridge top 1e3, "
        "lasso / elastic net top 1e-2; widened grids of the supplement: ridge top 1e6, lasso / elastic net top 1).")
    say("")
    gt = grp.pivot(index="group", columns="first_forecast", values="penalty").reindex(list(GROUPS))
    gt = gt.map(lambda v: f"{v:g}").reset_index()
    say("Multi-penalty ridge, the penalty of each exogenous group at each re-choice:")
    say("")
    say(_md_table(gt))
    say("")
    say("## Why one penalty loses: backbone shrinkage and where the forecast variance sits")
    say("")
    say("har-sum ratio = sum of the six HAR coefficients over the HAR + calendar OLS's (median over "
        "sessions; 1 = no shrinkage of the persistence); backbone variance ratio = in-window variance of "
        "the backbone part of the fit over that of the OLS fit; shares = the forecasts' variance over the "
        "1469 sessions split into the backbone part, the exogenous part and twice their covariance "
        "(each part = coefficients x (row - window mean)).")
    say("")
    t = shr.loc[[arm_key(e, m) for e, m in ALL_ARMS],
                ["har_sum_ratio_median", "backbone_var_ratio_median", "share_backbone_forecasts",
                 "share_exog_forecasts", "share_cross_forecasts", "corr_backbone_part_with_ols",
                 "n_exog_nonzero_median"]].copy()
    for c in t.columns:
        t[c] = t[c].map(lambda v: f"{v:.2f}" if c != "n_exog_nonzero_median" else f"{v:.0f}")
    say(_md_table(t.reset_index()))
    say("")
    hc = harc.pivot(index="arm", columns="column", values="coef_median")
    oc = harc.groupby("column")["ols_coef_median"].first()
    hcols = [c for c in ["har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"] if c in hc.columns]
    hc = hc.loc[[arm_key(e, m) for e, m in ALL_ARMS], hcols]
    hc.loc["HAR + calendar OLS"] = oc[hcols]
    say("Median HAR coefficients (prescaled design):")
    say("")
    say(_md_table(hc.map(lambda v: f"{v:+.3f}").reset_index()))
    say("")
    say("## Answer (recorded comparisons)")
    say("")
    for est in EST:
        base = head.loc[f"stored_sub_{EST_SHORT[est]}_baseline"]
        single = head.loc[arm_key(est, "single")]
        say(f"- {LABEL[est]}: HAR + calendar QLIKE {base['qlike']:.4f} / Sharpe {base['sharpe_mid']:.2f}; "
            f"all features with one penalty {single['qlike']:.4f} / {single['sharpe_mid']:.2f}.")
        for m in MODES[1:] + (("mpr",) if est == "ridge" else ()) + ("singlew", "bb0w"):
            r = head.loc[arm_key(est, m)]
            say(f"  - {MODE_LABEL[m]}: {r['qlike']:.4f} / {r['sharpe_mid']:.2f}; vs one penalty DM "
                f"{r['dm_vs_single']:+.2f} (p {r['dm_p_vs_single']:.3f}), HAC t {r['t_hac_vs_single']:+.2f}; "
                f"vs HAR + calendar DM {r['dm_vs_baseline']:+.2f} (p {r['dm_p_vs_baseline']:.3f}), "
                f"HAC t {r['t_hac_vs_baseline']:+.2f}.")
    say("")
    say("## Caveats")
    say("")
    say("- Point estimates with 866 trade days; the Sharpe ratios of these models differ by amounts "
        "of the order of their sampling error (the HAC t column).")
    say("- The ratio r and the group penalties are chosen on 125-session tails; with the spec's "
        "grids, several choices sit at a grid edge (table above), as in the spec's own one-penalty models.")
    say("- The lasso / elastic net warm path is the spec's algorithm; where it leaves the exact path "
        "(gate 5) the forecast differs from the exact window optimum until the next re-choice.")
    say("- The HAR + calendar references are the stored master-table forecasts (scorer gate 4).")
    say("")
    say("## CPU")
    say("")
    tot = float(cpu["cpu_sec"].sum())
    say(f"C walk-forward, all {len(cpu)} arms: {tot:.0f} CPU seconds (`cpu_seconds.csv`); one process "
        "for ridge + lasso and one for the elastic net, single-threaded BLAS.")
    say("")
    say("## Files")
    say("")
    say("- `experiments/close_exogpen.py` (stages gate, check, run, analyze), "
        "`experiments/close_exogpen_kernel.c` (the C port), `experiments/close_exogpen_cdcheck.c` "
        "(an independent coordinate-descent / eigendecomposition solver kept as a cross-check, not "
        "the arms of record).")
    say("- `headline.csv`, `scorer_gate.csv`, `penalty_path.csv`, `group_penalties.csv`, "
        "`warm_path_vs_exact.csv`, `backbone_shrinkage_and_variance_shares.csv`, `har_coefficients.csv`, "
        "`cpu_seconds.csv`; figures `fig_qlike_sharpe.png`, `fig_penalty_path.png`, "
        "`fig_variance_shares.png`; forecasts and coefficients `_work/runs/*.npz`; gates `_work/gate_*.csv`, "
        "`_work/check_*.csv`.")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def R_EST(k: str) -> str:
    return {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}[k.split("_", 1)[0]]


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "gate":
        gate()
    elif stage == "check":
        check()
    elif stage == "run":
        run(sys.argv[2:] or list(EST))
    elif stage == "run_supplement":
        run(sys.argv[2:] or list(EST), supplement=True)
    elif stage == "crosscheck":
        crosscheck()
    elif stage == "analyze":
        analyze()
    else:
        raise SystemExit(__doc__)
