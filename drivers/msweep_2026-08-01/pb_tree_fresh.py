# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Tree freshness on the Hawkes basis: refit cadence ladder down to per-bar.

env CADENCE: tree refit interval in bars (1 = per-bar). Shapes/cores refresh
at 240 as always (the structural layer); only the TREE refit cadence moves.
Repo-standard LGBM (500 trees, d5, lr 0.1). DM vs pbe ridge_cad control and
vs the champion hybrid_dp.
"""

import os
import time
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from lightgbm import LGBMRegressor
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar

CADENCE = int(os.environ.get("CADENCE", "1"))
NJOBS = int(os.environ.get("NJOBS", "6"))
MIX = bool(os.environ.get("PB_MIX"))
READER = os.environ.get("READER", "lgbm")
TAG = f"{READER}_c{CADENCE}" + ("_mix" if MIX else "")
D = "results/b2_mmap"
START, END, TRAIN_WIN = 24000, 26189, 24000
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[: END + 300])
y = np.load(f"{D}/y.npy")
b = np.load(f"{D}/b.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n0, n1, U = START, END, 256

chan = {}
for j, nm in enumerate(names):
    m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
    if m:
        chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
for c in chan:
    chan[c].sort()
labels = list(chan)
ladder_cols = [j for c in chan for _, j in chan[c]]
other_cols = [j for j in range(len(names)) if j not in set(ladder_cols)]
L = np.stack([X[:, [j for _, j in chan[c]]] for c in chan], axis=0)
O = X[:, other_cols]
cal_cols = [
    names.index(nm)
    for nm in [
        "DOW_0",
        "DOW_1",
        "DOW_2",
        "DOW_3",
        "DOW_4",
        "hour",
        "is_overnight",
        "is_open",
        "is_close",
    ]
]
bar1 = {c: X[:, chan[c][0][1]] for c in labels}


def ridge_fit(A, yy):
    G = A.T @ A + np.eye(A.shape[1])
    G[0, 0] -= 1.0
    return np.linalg.solve(G, A.T @ yy)


def kfeat_series(s, beta):
    u = np.arange(1, U + 1, dtype=np.float64)
    w = u ** (-beta)
    w /= w.sum()
    K = np.zeros(n1 + 8)
    sw = sliding_window_view(s[: n1 + 8], U)
    K[U:] = sw[:-1] @ w[::-1]
    return K


def prof_series(s, t):
    if s[t - TRAIN_WIN : t].std() <= 0:
        return 1.0

    def loss(be):
        K = kfeat_series(s, be)
        A = np.stack([np.ones(TRAIN_WIN), K[t - TRAIN_WIN : t]], axis=1)
        cf, *_ = np.linalg.lstsq(A, y[t - TRAIN_WIN : t], rcond=None)
        r = y[t - TRAIN_WIN : t] - A @ cf
        return float(r @ r)

    return minimize_scalar(loss, bounds=(0.05, 3.0), method="bounded").x


P = np.empty(n1 - n0)
Kbar = None
t0 = time.time()
nfit = 0
if MIX:
    KM_ALL = np.column_stack(
        [kfeat_series(y, be) for be in (0.3, 0.6, 1.0, 1.5, 2.2)]
        + [kfeat_series(bar1[c], be) for c in labels for be in (0.6, 1.5)]
    )
for seg in range(0, n1 - n0, 240):
    t = n0 + seg
    full = slice(t - TRAIN_WIN, t)
    A0 = np.hstack([np.ones((TRAIN_WIN, 1)), O[full], X[full][:, ladder_cols]])
    cff = ridge_fit(A0, y[full])
    Km = cff[1 + O.shape[1] :].reshape(len(chan), -1)
    Kbar = Km if Kbar is None else 0.9 * Kbar + 0.1 * Km
    _, sv, Vt = np.linalg.svd(Kbar, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    S = (Vt * sgn[:, None])[:5]
    lo, hi = t - TRAIN_WIN, min(t + 240, n1)
    C = np.einsum("cnr,kr->nck", L[:, lo:hi], S).reshape(hi - lo, -1)
    if MIX:
        F = np.hstack([KM_ALL[lo:hi], C, X[lo:hi][:, cal_cols]])
    else:
        cores = [kfeat_series(y, prof_series(y, t))]
        cores += [kfeat_series(bar1[c], prof_series(bar1[c], t)) for c in labels]
        F = np.hstack([Kc[lo:hi, None] for Kc in cores] + [C, X[lo:hi][:, cal_cols]])
    mdl = None
    for i in range(hi - t):
        if mdl is None or i % CADENCE == 0:
            if READER == "ebm":
                from interpret.glassbox import ExplainableBoostingRegressor

                mdl = ExplainableBoostingRegressor(n_jobs=NJOBS)
            else:
                mdl = LGBMRegressor(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.1,
                    verbose=-1,
                    n_jobs=NJOBS,
                )
            mdl.fit(F[i : TRAIN_WIN + i], y[lo + i : lo + TRAIN_WIN + i])
            nfit += 1
        P[seg + i] = mdl.predict(F[TRAIN_WIN + i : TRAIN_WIN + i + 1])[0]
    print(
        f"[{TAG}] seg {seg // 240 + 1}/10 fits={nfit} ({time.time() - t0:.0f}s)",
        flush=True,
    )
pr, tr = apply_duan_smearing(P, y[n0:n1], b[n0:n1])
mm = forecast_metrics(pr, tr)
np.savez(f"results/geo_preds/pbt_{TAG}.npz", pred_raw=pr, true_raw=tr)
for ref in ("pbe_ridge_cad", "hybrid257_dp"):
    z = np.load(f"results/geo_preds/{ref}.npz")
    r = dm_test(qlike_per_bar(pr, tr), qlike_per_bar(z["pred_raw"], z["true_raw"]), h=1)
    print(
        f"[{TAG}] qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} vs {ref}: "
        f"diff={r['mean_diff']:+.5f} p={r['p']:.4f}",
        flush=True,
    )
print(f"[{TAG}] DONE ({time.time() - t0:.0f}s)", flush=True)
