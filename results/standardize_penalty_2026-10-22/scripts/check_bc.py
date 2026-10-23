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
    Cls.ev_ref = P.evid_ref(names, bb)
    Cls.av = P.avail_k(names, bb)
    Cls.std_binom, Cls.std_binom_reseeds = [], []
    Cls.kkt_rows, Cls.kkt_reseeds, Cls.kkt_sec = [], [], [0.0, 0.0]
    Cls.std_seeds, Cls.std_ridge, Cls.std_blockend = [], [], []
    Cls.std_evid_reseeds = []
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
    # the explicit side: the identifiability mask of the window (the spec's, recomputed here), then s and the evidence
    # (rows off the modal value, np.unique counts) computed independently of the runner's std_weights / col_mode
    Xa = np.hstack([np.asarray(X[o - W : o]), np.ones((W, 1))])
    yr = y[o - W : o]
    probe = ns["RollingTunedLinear"]()
    probe._locked = locked
    probe._recompute_mask(Xa)
    Xa[:, probe._maskout] = 0.0
    n = len(yr)
    s = np.std(Xa, axis=0)
    own = np.array([len(c) - np.unique(c, return_counts=True)[1].max() for c in Xa.T])
    evid = np.full(p + 1, 10**9)  # addendum 2, independently: the variable's k = 1 column of the same kind
    for j, nm in enumerate(names):
        if bb[j]:
            continue
        if nm.startswith("adj_"):
            k1 = nm.rsplit("_ma_", 1)[0] + "_ma_1"
        elif "_avail_ma_" in nm:
            k1 = nm.split("_avail_ma_")[0] + "_avail_ma_1"
        else:
            k1 = nm
        evid[j] = own[names.index(k1)] if k1 in names else 0
    out = ~locked & ((s < P.STD_MIN) | (evid < P.STD_MIN_EVIDENCE))
    # addendum 3, independently: the binomial availability rule (np.mean / np.std, greedy in design order)
    binom = []
    kept: dict = {}
    for j, nmj in enumerate(names):
        if bb[j] or out[j] or "_avail_ma_" not in nmj or nmj.endswith("_avail_ma_1"):
            continue
        kj = int(nmj.rsplit("_", 1)[1])
        x = Xa[:, j]
        pj = float(x.mean())
        if float(x.std()) <= np.sqrt(pj * (1 - pj) / kj):
            out[j] = True
            binom.append((nmj, "near-constant", None))
            continue
        hit = None
        for i in kept.get(kj, []):
            pb = 0.5 * (pj + float(Xa[:, i].mean()))
            if float((x - Xa[:, i]).std()) <= np.sqrt(2) * np.sqrt(pb * (1 - pb) / kj):
                hit = names[i]
                break
        if hit is not None:
            out[j] = True
            binom.append((nmj, "near-duplicate", hit))
        else:
            kept.setdefault(kj, []).append(j)
    walk_out = np.isnan(rg._sd_seed) & ~locked
    sc = np.where(out | locked, 1.0, s)
    mc = np.where(locked, 0.0, Xa.mean(axis=0))  # explicit: centre and scale (window mean and s.d.)
    Z = (Xa - mc) / sc
    Z[:, out] = 0.0  # (the intercept column stays 1)
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
        # the weighted problem (the walk's own centred state) and the explicit problem on Z
        kk_w = kkt_rel(rg._Gr, rg._c, rg._mu_vec, th)
        Gz = Z.T @ Z
        ei = np.flatnonzero(~locked)
        Gz[ei, ei] += lam2 * pf[ei]
        kk_e = kkt_rel(Gz, Z.T @ yr, np.where(locked, 0.0, mu * pf), g)
    t_e = time.time() - t0
    beta = g / sc  # back-transform: slopes gamma / s, intercept gamma_0 - sum_j m_j beta_j
    beta[-1] = g[-1] - mc @ beta
    th_w = rg.coef_original()  # the walk's scaled coordinates -> the original ones
    # the uncentred solution of the same problem, same estimator (raw masked columns; the mask applied)
    Xu = Xa.copy()
    Xu[:, out] = 0.0
    if kind == "ridge":
        th_u = P.ridge_keep(Xu.T @ Xu, Xu.T @ yr, np.where(locked, 0.0, a * pf * sc * sc), ~out)
    else:
        th_u = P.batch_theta_w(Xu, yr, locked, a, l1, pf * sc, pf * sc * sc)
    xn = np.append(X[o], 1.0)

    def cmp(u, v):
        sup = (np.abs(u) > 0) | (np.abs(v) > 0)
        sl = ~locked
        return dict(
            normwise_slopes=float(np.abs(u - v)[sl].max() / np.abs(v[sl]).max()),
            max_rel_on_support=float((np.abs(u - v)[sup] / np.maximum(np.abs(v)[sup], 1e-300)).max()),
            intercept_rel=float(abs(u[-1] - v[-1]) / abs(v[-1])),
            forecast_rel=float(abs(xn @ u - xn @ v) / abs(xn @ v)),
            support_equal=bool(np.array_equal(u[sl] != 0, v[sl] != 0)),
        )

    rb[m] = dict(
        origin=o,
        alpha=a,
        masked_identifiability=int(probe._maskout.sum()),
        evidence_out_beyond_mask_explicit=int((out & ~probe._maskout).sum()),
        evidence_out_sets_equal_walk_vs_explicit=bool(np.array_equal(walk_out, out)),
        evidence_out_columns=[names[j] for j in np.flatnonzero(out & ~probe._maskout)][:60],
        binomial_rule_explicit=binom,
        binomial_rule_walk=[(names[c], kd, names[q] if q >= 0 else None) for c, kd, q in Cls.std_binom[-1][1]],
        nnz_slopes=int((th_w[:-1] != 0).sum()),
        walk_vs_explicit_standardized=cmp(th_w, beta),
        walk_vs_weighted_uncentred_same_problem=cmp(th_w, th_u),
        kkt_walk_centred_ok_rel=kk_w,
        kkt_explicit_ok_rel=kk_e,
        repair_explicit={k2: P.REPAIR[k2] - rep0[k2] for k2 in P.REPAIR},
        s_quantiles_5_50_95=[float(v) for v in np.quantile(s[~out & ~locked], [0.05, 0.5, 0.95])],
        sec_walk_seed=t_w,
        sec_explicit=t_e,
    )
    print(m, json.dumps(rb[m]), flush=True)
rep["b"] = rb
cn: dict = {}
for m in ("lasso", "enet"):
    for o in (44250 + 250 * 60, 44250 + 250 * 120):
        Cls, a, kind, l1 = make(m, o - W)
        rg = Cls()
        rg.init_window(X[o - W : o], y[o - W : o])
        A = np.flatnonzero((rg._th != 0) | locked)
        cn[f"{m}/{o}"] = dict(n_active=int(len(A) - 1), cond_solver_z=float(np.linalg.cond(rg._Gr[np.ix_(A, A)])))
        print(m, o, cn[f"{m}/{o}"], flush=True)
rep["cond_solver_coordinates"] = cn


# ---------------------------------------------------------------- (c)
def short_walk(m: str, Xs, ys, o: int, nsolve: int):
    Cls, a, kind, l1 = make(m, o - W)
    ths: list = []

    class Rec(Cls):
        def solve(self):
            super().solve()
            ths.append(self.coef_original())  # the scaled state (addendum 3) -> the original coordinates

    bt = ns["MultiStageBacktest"](
        residualizer=ns["IdentityResidualizer"](), regressor_factory=Rec, refit_frequency=1
    )
    pred = bt.run(Xs[o - W : o + nsolve], ys[o - W : o + nsolve], W, desc="c")
    return pred, ths, [(x[0], x[1].copy()) for x in Cls.std_seeds], Cls


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
        evidence_reseeds=len(C0.std_evid_reseeds),
        evidence_out_per_seed=[int(x[2]) for x in C0.std_seeds],
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
            tb = P.fit_std(
                Xw,
                y[t - W : t],
                locked,
                "ridge",
                float(fr["ridge"]["all_features"]),
                0.0,
                pf,
                P.evid_ref(names, bb),
                P.avail_k(names, bb),
            )[0]
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
