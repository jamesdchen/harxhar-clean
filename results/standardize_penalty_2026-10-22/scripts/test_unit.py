"""STANDARDIZE_PENALTY unit checks on synthetic data (the runner's code, imported from the std worktree).

1. fit_std == an explicit fit on x_j / s_j with the existing penalty factors, back-transformed (ridge, lasso, enet).
2. The walk (PFTuned through MultiStageBacktest, const schedule): ridge -- every bar's coefficients equal the batch
   fit_std on that bar's window; lasso / enet -- at every seed equal to fit_std, between seeds KKT-clean on the held
   weights.  Includes a column that becomes constant mid-block and one masked at the tune.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
sys.path.insert(0, str(WT / "experiments"))
sys.path.insert(0, str(WT))
import pool_linear_pretune as P  # noqa: E402

P.SAMPLE, P.TW_DAYS, P.PERIODS_PER_DAY = "48", 2000, 48
ns = P.spec_section("ridge", "baseline", 0, -1, 0)
import src.models.reclasso_har as RH  # noqa: E402

rng = np.random.default_rng(7)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def evidence(Xa):
    """rows off the modal value, by np.unique counts (independent of P.col_mode)"""
    return np.array([len(c) - np.unique(c, return_counts=True)[1].max() for c in Xa.T])


def explicit(Xa, yr, locked, kind, a, l1, pf):
    s = np.std(Xa, axis=0)
    out = ~locked & ((s < P.STD_MIN) | (evidence(Xa) < P.STD_MIN_EVIDENCE))
    sc = np.where(out | locked, 1.0, s)
    Z = Xa / sc
    Z[:, out] = 0.0
    if kind == "ridge":
        keep = ~out
        k = np.flatnonzero(keep)
        G = (Z.T @ Z)[np.ix_(k, k)]
        G[np.diag_indices_from(G)] += np.where(locked[k], 0.0, a * pf[k])
        g = np.zeros(Z.shape[1])
        g[k] = np.linalg.solve(G, (Z.T @ yr)[k])
    else:
        pf2 = pf.copy()
        g = P.batch_theta_pf(ns, Z, yr, locked, a, l1, pf2)
    return g / sc


# ---- 1. batch equivalence
n, p = 3000, 25
X = rng.standard_normal((n, p)) * rng.uniform(0.2, 3.0, p) + rng.uniform(-2, 2, p)
X[:, 3] = 0.0  # a masked column (zero)
X[:, 4] = 1.7  # constant (s < STD_MIN)
X[:, 7] = np.where(rng.uniform(size=n) < 0.03, rng.standard_normal(n), 0.0)  # ~90 rows off the mode: evidence-masked
X[:, 8] = np.where(np.arange(n) < 400, rng.standard_normal(n), 0.5)  # 400 rows off the mode: kept
beta = rng.standard_normal(p) * (rng.uniform(size=p) < 0.4)
y = X @ beta + rng.standard_normal(n)
Xa = np.hstack([X, np.ones((n, 1))])
locked = np.zeros(p + 1, dtype=bool)
locked[-1] = True
pf = np.ones(p + 1)
pf[:6] = 0.01
for kind, a, l1 in [("ridge", 30.0, 0.0), ("lasso", 3e-3, 1.0), ("enet", 3e-3, 0.5)]:
    th, s, out = P.fit_std(Xa, y, locked, kind, a, l1, pf)
    th2 = explicit(Xa, y, locked, kind, a, l1, pf)
    print(f"1. {kind}: rel diff {rel(th, th2):.2e}, nnz {int((th != 0).sum())}, out {int(out.sum())}, REPAIR {P.REPAIR}")
    assert rel(th, th2) < 1e-8
    assert th[3] == 0.0 and th[4] == 0.0 and th[7] == 0.0 and th[8] != 0.0, th[[3, 4, 7, 8]]

# ---- 2. the walk
Wn = 1200
N = Wn + 760
X = rng.standard_normal((N, p)) * rng.uniform(0.2, 3.0, p) + rng.uniform(-2, 2, p)
X[: Wn + 200, 5] = 0.0  # constant 0 until row Wn+200: masked at the first tune, live later
X[:, 6] = np.where(np.arange(N) < 30, rng.standard_normal(N), 2.0)  # last change at row 29: constant from bar 30 on
X[:, 7] = np.where((np.arange(N) >= 100) & (np.arange(N) < 460), rng.standard_normal(N), 0.0)  # 360 off-mode rows;
# the window [t-Wn, t) keeps >= 250 of them until t = 1200 + 210: evidence falls below 250 mid-block 0
y = X @ beta + rng.standard_normal(N)
names = [f"har_ma_{k}" for k in range(4)] + [f"adj_x{j}_ma_1" for j in range(4, p)]


def run_walk(est, alpha, flag):
    P.STANDARDIZE_PENALTY = flag
    Base = ns["RollingTunedLinear"]
    Base.grid = ns["ESTIMATOR_GRIDS"][est]
    Base.grid = [(Base.grid[0][0], alpha, Base.grid[0][2])]
    rec: list = []
    Cls = P.make_class(ns, 0.01, P.ConstSched(alpha), 0, rec)
    Cls.bb = np.array([n.startswith("har_ma_") for n in names])
    Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
    Cls.kkt_rows, Cls.kkt_reseeds, Cls.kkt_sec = [], [], [0.0, 0.0]
    Cls.std_seeds, Cls.std_ridge, Cls.std_blockend = [], [], []
    Cls.std_evid_reseeds = []
    ths: list = []

    class Rec(Cls):
        def solve(self):
            super().solve()
            Xa, yr = self._masked_window()
            ths.append((self._th.copy(), self._maskout.copy()))

    bt = ns["MultiStageBacktest"](
        residualizer=ns["IdentityResidualizer"](), regressor_factory=Rec, refit_frequency=1
    )
    pred = bt.run(X, y, Wn, desc="t")
    return pred, ths, Cls


for est, alpha in [("ridge", 30.0), ("reclasso", 3e-3), ("reclasticnet", 3e-3)]:
    kind = "ridge" if est == "ridge" else ("lasso" if est == "reclasso" else "enet")
    l1 = {"ridge": 0.0, "lasso": 1.0, "enet": 0.5}[kind]
    pred, ths, Cls = run_walk(est, alpha, True)
    worst, worst_seed = 0.0, 0.0
    seeds = {x[0] for x in Cls.std_seeds}
    for i, (th, mk) in enumerate(ths):
        t = Wn + i
        Xw = np.hstack([X[t - Wn : t], np.ones((Wn, 1))])
        Xw[:, mk] = 0.0
        pfv = np.ones(p + 1)
        pfv[:4] = 0.01
        if kind == "ridge" or i in seeds:
            tb = P.fit_std(Xw, y[t - Wn : t], locked, kind, alpha, l1, pfv)[0]
            d = rel(th, tb)
            if kind == "ridge":
                worst = max(worst, d)
            else:
                worst_seed = max(worst_seed, d)
    print(
        f"2. {est}: {len(ths)} solves, seeds {sorted(seeds)}, ridge per-bar max rel {worst:.2e}, "
        f"seed max rel {worst_seed:.2e}, kkt rows {len(Cls.kkt_rows)} all ok "
        f"{all(r[2] for r in Cls.kkt_rows)}, kkt reseeds {len(Cls.kkt_reseeds)}, std_ridge {Cls.std_ridge}, "
        f"evidence reseeds {Cls.std_evid_reseeds}, out per seed {[x[2] for x in Cls.std_seeds]}"
    )
    assert worst < 1e-8 and worst_seed < 1e-8
    pred0, _, _ = run_walk(est, alpha, False)
    print(f"   flag off vs on: max |d pred| {np.abs(pred - pred0).max():.3e}")
P.STANDARDIZE_PENALTY = True
print("all unit checks passed")
