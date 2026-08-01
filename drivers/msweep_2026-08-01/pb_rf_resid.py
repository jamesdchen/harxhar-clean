# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""RANDOM FOREST in the residual seat (bagging vs boosting): per-bar ridge base + EBM correction + stack.

Completes the distillation triad vs hybrid_dp (nested):
  reader=none  -> hybrid_dp 0.165145
  reader=EBM   -> additive+pairwise share of the tree edge (BAKEABLE)
  reader=LGBM  -> full tree share (lgbm_resid arm, running)
LGBM - EBM = the diffuse, unbakeable remainder.
"""

import time
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestRegressor
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.features.transforms.residualizer import RollingRidgeResidualizer

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
stk_c, stk_r = [], []
Kbar = None
t0 = time.time()
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
    cores = [kfeat_series(y, prof_series(y, t))]
    cores += [kfeat_series(bar1[c], prof_series(bar1[c], t)) for c in labels]
    lo, hi = t - TRAIN_WIN, min(t + 240, n1)
    C = np.einsum("cnr,kr->nck", L[:, lo:hi], S).reshape(hi - lo, -1)
    F = np.hstack([Kc[lo:hi, None] for Kc in cores] + [C, X[lo:hi][:, cal_cols]])
    rr = RollingRidgeResidualizer(1.0, fast_inverse=True)
    rr.fit(F[:TRAIN_WIN], y[lo : lo + TRAIN_WIN])
    resid_win = y[lo : lo + TRAIN_WIN] - rr.predict(F[:TRAIN_WIN])
    mdl = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=50,
        max_features=0.33,
        n_jobs=3,
        random_state=seg,
    )
    mdl.fit(F[:TRAIN_WIN], resid_win)
    corr_seg = mdl.predict(F[TRAIN_WIN:])
    for i in range(hi - t):
        tt_ = TRAIN_WIN + i
        base = rr.predict(F[tt_ : tt_ + 1])[0]
        corr = float(corr_seg[i])
        if len(stk_c) >= 1000:
            A2 = np.stack([np.ones(1000), np.array(stk_c[-1000:])], axis=1)
            wf, *_ = np.linalg.lstsq(A2, np.array(stk_r[-1000:]), rcond=None)
            P[seg + i] = base + wf[0] + wf[1] * corr
        else:
            P[seg + i] = base + corr
        stk_c.append(corr)
        stk_r.append(float(y[t + i]) - base)
        if i < hi - t - 1:
            rr.roll(F[tt_], y[lo + tt_], F[i], y[lo + i])
    print(f"[rf_resid] seg {seg // 240 + 1}/10 ({time.time() - t0:.0f}s)", flush=True)
pr, tr = apply_duan_smearing(P, y[n0:n1], b[n0:n1])
mm = forecast_metrics(pr, tr)
np.savez("results/geo_preds/pbt_rf_resid.npz", pred_raw=pr, true_raw=tr)
z = np.load("results/geo_preds/hybrid257_dp.npz")
r = dm_test(qlike_per_bar(pr, tr), qlike_per_bar(z["pred_raw"], z["true_raw"]), h=1)
print(
    f"[rf_resid] qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} "
    f"vs hybrid_dp: diff={r['mean_diff']:+.5f} p={r['p']:.4f}",
    flush=True,
)
print("[rf_resid] DONE", flush=True)
