"""STANDARDIZE_PENALTY checks (b) and (c) on the real w1000 full-set design (UNOBSERVED_NEUTRAL on, unobs/d_on).

(b) one origin each for ridge, lasso, elastic net (full set, r = 0.01, the flag-on frozen penalties): the walk's own cold
    seed at that origin (PFTuned.init_window -> _tune -> mask -> _seed_std, KKT-certified batch) against an explicit fit
    on the masked window's columns divided by their window s.d. with the existing (unweighted) code -- ridge
    (Z'Z + diag(a pf)) g = Z'y, lasso / enet batch_theta_pf on Z (KKT-certified) -- back-transformed theta = g / s.
    Reported: max |d theta| / max |theta|, per-coefficient relative on the support, the next-row forecast, the KKT
    violation of each fit in its own problem (walk_kkt, relative to max(max|c|, 1)).
(c) causality: a 600-solve walk from that origin (halo W) on the design and on a copy whose rows >= T = origin + 300
    are perturbed (X and y); every solve < T must give byte-identical coefficients, seed weights and forecasts; the
    solves >= T must differ (control).  Ridge also: every bar's walk coefficients against the batch fit_std of that
    bar's window (the running moments vs np.std).

usage: python check_bc.py <frozen_penalties.json of the flag-on select> <out json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
DZ = Path(
    r"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor"
    r"\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad\unobs\d_on\_work\design_all_features"
)
sys.path.insert(0, str(WT / "experiments"))
sys.path.insert(0, str(WT))
import pool_linear_pretune as P  # noqa: E402

fr = json.loads(Path(sys.argv[1]).read_text())
OUTJ = Path(sys.argv[2])
P.SAMPLE, P.TW_DAYS, P.PERIODS_PER_DAY = "rth13", 1000, 13
P.W = 13000
P.HAR_BASE = "5"
P.EX_FLAGS = dict(
    DROP_ACTIVITY_MA=True,
    CALENDAR_THROUGH_OVERNIGHT=True,
    DROP_OVERNIGHT_INDICATOR=True,
    LEAN_GLOBAL_LAGS=True,
    LATE_START_WARMUP=True,
)
ns = P.spec_section("ridge", "all_features", 0, -1, 0)
import src.models.reclasso_har as RH  # noqa: E402

meta = json.loads((DZ / "design.json").read_text())
P.check_meta(meta, "all_features")
names = meta["names"]
bb = P.backbone_mask(names)
X = np.load(DZ / "X.npy")
y = np.load(DZ / "y.npy")
W = P.W
p = X.shape[1]
locked = np.zeros(p + 1, dtype=bool)
locked[-1] = True
pf = np.ones(p + 1)
pf[:-1][bb] = 0.01
EST = dict(ridge=("ridge", "ridge", 0.0), lasso=("reclasso", "lasso", 1.0), enet=("reclasticnet", "enet", 0.5))
S0 = 44250  # the walk start (the re-choice boundary at or before the first 2012 row)
ORIGIN = dict(ridge=S0 + 250 * 20, lasso=S0 + 250 * 60, enet=S0 + 250 * 120)
rep: dict = {"design": str(DZ), "origins": ORIGIN, "penalties": {m: fr[m]["all_features"] for m in EST}}


def make(m: str, g0: int):
    est, kind, l1 = EST[m]
    a = float(fr[m]["all_features"])
    Base = ns["RollingTunedLinear"]
    Base.grid = [(kind if kind != "lasso" else "lasso", a, l1)]
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    Cls = P.make_class(ns, 0.01, P.ConstSched(a), g0, [])
    Cls.bb = bb
    Cls.kkt_rows, Cls.kkt_reseeds, Cls.kkt_sec = [], [], [0.0, 0.0]
    Cls.std_seeds, Cls.std_ridge, Cls.std_blockend = [], [], []
    return Cls, a, kind, l1


def kkt_rel(Gr, c, mu_vec, th):
    ok, v, sc = RH.walk_kkt(Gr, c, mu_vec, th)
    return bool(ok), float(v / sc)


# ---------------------------------------------------------------- (b)
P.STANDARDIZE_PENALTY = True
rb: dict = {}
for m in EST:
    o = ORIGIN[m]
    Cls, a, kind, l1 = make(m, o - W)
    t0 = time.time()
    rg = Cls()
    rg.init_window(X[o - W : o], y[o - W : o])  # the walk's cold seed at a tune boundary (a chunk starting here)
    th = rg._th.copy()
    t_w = time.time() - t0
    Xa, yr = rg._masked_window()
    n = len(yr)
    s = np.std(Xa, axis=0)
    out = ~locked & (s < P.STD_MIN)
    sc = np.where(out | locked, 1.0, s)
    Z = Xa / sc
    Z[:, out] = 0.0
    rep0 = dict(P.REPAIR)
    t0 = time.time()
    if kind == "ridge":
        k = np.flatnonzero(~out)
        G = (Z.T @ Z)[np.ix_(k, k)]
        G[np.diag_indices_from(G)] += np.where(locked[k], 0.0, a * pf[k])
        g = np.zeros(p + 1)
        g[k] = np.linalg.solve(G, (Z.T @ yr)[k])
        kk_w = kk_e = None
    else:
        g = P.batch_theta_pf(ns, Z, yr, locked, a, l1, pf)
        mu, lam2 = n * a * l1, n * a * (1.0 - l1)
        # the weighted problem (the walk's own state) and the explicit problem on Z
        kk_w = kkt_rel(rg._Gr, rg._c, rg._mu_vec, th)
        Gz = Z.T @ Z
        ei = np.flatnonzero(~locked)
        Gz[ei, ei] += lam2 * pf[ei]
        kk_e = kkt_rel(Gz, Z.T @ yr, np.where(locked, 0.0, mu * pf), g)
    t_e = time.time() - t0
    beta = g / sc
    sup = (np.abs(beta) > 0) | (np.abs(th) > 0)
    xn = np.append(X[o], 1.0)
    xn[rg._maskout] = 0.0
    rb[m] = dict(
        origin=o,
        alpha=a,
        masked=int(rg._maskout.sum()),
        left_out_s_lt_1e12=int((out & ~rg._maskout).sum()),
        nnz_weighted=int((th != 0).sum()),
        nnz_explicit=int((beta != 0).sum()),
        support_equal=bool(np.array_equal(th != 0, beta != 0)),
        max_abs_dtheta_over_max_abs_theta=float(np.abs(th - beta).max() / np.abs(beta).max()),
        max_rel_dtheta_on_support=float(
            (np.abs(th - beta)[sup] / np.maximum(np.abs(beta)[sup], 1e-300)).max()
        ),
        forecast_next_row_weighted=float(xn @ th),
        forecast_next_row_explicit=float(xn @ beta),
        forecast_rel_diff=float(abs(xn @ th - xn @ beta) / abs(xn @ beta)),
        kkt_weighted_ok_rel=kk_w,
        kkt_explicit_ok_rel=kk_e,
        repair_explicit={k2: P.REPAIR[k2] - rep0[k2] for k2 in P.REPAIR},
        s_quantiles_5_50_95=[float(v) for v in np.quantile(s[~out & ~locked], [0.05, 0.5, 0.95])],
        sec_walk_seed=t_w,
        sec_explicit=t_e,
    )
    print(m, json.dumps(rb[m]), flush=True)
rep["b"] = rb


# ---------------------------------------------------------------- (c)
def short_walk(m: str, Xs, ys, o: int, nsolve: int):
    Cls, a, kind, l1 = make(m, o - W)
    ths: list = []

    class Rec(Cls):
        def solve(self):
            super().solve()
            ths.append(self._th.copy())

    bt = ns["MultiStageBacktest"](
        residualizer=ns["IdentityResidualizer"](), regressor_factory=Rec, refit_frequency=1
    )
    pred = bt.run(Xs[o - W : o + nsolve], ys[o - W : o + nsolve], W, desc="c")
    return pred, ths, [(s_, v.copy()) for s_, v, _ in Cls.std_seeds], Cls


rc: dict = {}
rng = np.random.default_rng(20261022)
NS, OFF = 600, 300
for m in EST:
    o = ORIGIN[m]
    T = o + OFF
    t0 = time.time()
    pr0, th0, sd0, C0 = short_walk(m, X, y, o, NS)
    Xp = X.copy(order="K")
    yp = y.copy()
    Xp[T:] += rng.standard_normal((len(X) - T, p)) * 0.5
    yp[T:] += rng.standard_normal(len(y) - T) * 0.5
    pr1, th1, sd1, C1 = short_walk(m, Xp, yp, o, NS)
    del Xp
    k = OFF  # solves 0 .. OFF-1 forecast rows o .. T-1 from windows ending <= T-1
    same_pred = bool(np.array_equal(pr0[:k], pr1[:k]))
    same_th = all(np.array_equal(a_, b_) for a_, b_ in zip(th0[:k], th1[:k]))
    seeds_pre = [(s_, v) for s_, v in sd0 if s_ - (o - W) < k]
    seeds_pre1 = [(s_, v) for s_, v in sd1 if s_ - (o - W) < k]
    same_sd = len(seeds_pre) == len(seeds_pre1) and all(
        a_[0] == b_[0] and np.array_equal(a_[1], b_[1], equal_nan=True) for a_, b_ in zip(seeds_pre, seeds_pre1)
    )
    moved = int((pr0[k:] != pr1[k:]).sum())
    r = dict(
        origin=o,
        T=T,
        solves=NS,
        seeds=[int(s_ - (o - W)) for s_, _ in sd0],
        identical_forecasts_before_T=same_pred,
        identical_coefficients_before_T=same_th,
        identical_seed_weights_before_T=same_sd,
        n_seeds_before_T=len(seeds_pre),
        forecasts_moved_at_or_after_T=moved,
        of=NS - k,
        kkt_all_ok=all(x[2] for x in C0.kkt_rows) if C0.kkt_rows else None,
        kkt_reseeds=len(C0.kkt_reseeds),
        sec=time.time() - t0,
    )
    if m == "ridge":  # every bar's walk coefficients against the batch fit_std of that bar's window
        worst, worst_f = 0.0, 0.0
        for i in range(0, NS, 7):
            t = o + i
            Xw = np.hstack([X[t - W : t], np.ones((W, 1))])
            # the mask in force at solve i: the tune at the block start (recomputed from the raw window there)
            blk0 = o + (i // 250) * 250
            pr_ = ns["RollingTunedLinear"]()
            pr_._locked = locked
            pr_._recompute_mask(np.hstack([X[blk0 - W : blk0], np.ones((W, 1))]))
            Xw[:, pr_._maskout] = 0.0
            tb = P.fit_std(Xw, y[t - W : t], locked, "ridge", float(fr["ridge"]["all_features"]), 0.0, pf)[0]
            worst = max(worst, float(np.abs(th0[i] - tb).max() / np.abs(tb).max()))
            xn = np.append(X[t], 1.0)
            worst_f = max(worst_f, float(abs(xn @ th0[i] - xn @ tb) / abs(xn @ tb)))
        r |= dict(
            ridge_walk_vs_batch_every_7th_bar_max_rel_dtheta=worst,
            ridge_walk_vs_batch_max_rel_dforecast=worst_f,
            ridge_within_block_drift=C0.std_ridge,
        )
    rc[m] = r
    print(m, json.dumps(r), flush=True)
rep["c"] = rc
OUTJ.write_text(json.dumps(rep, indent=1))
print("written", OUTJ)
