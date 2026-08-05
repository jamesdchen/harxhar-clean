# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Enet-on-Hawkes-native probe, freshness-matched.

Enet has no rank-1 rolling path -> fits at segment cadence (240). The
control is ridge at the SAME cadence (not the per-bar champion), isolating
the penalty family. Features standardized by trailing-window std (causal).
Reports QLIKE, DM vs the cadence-ridge control, and the surviving-column
count (the legibility payoff if the null holds).
"""

import time
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from sklearn.linear_model import ElasticNet
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar

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


def run(tag, enet_alpha=None, l1_ratio=0.5):
    P = np.empty(n1 - n0)
    Kbar = None
    nz = []
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
        Ftr, ytr = F[:TRAIN_WIN], y[lo : lo + TRAIN_WIN]
        mu, sd = Ftr.mean(0), Ftr.std(0)
        sd[sd == 0] = 1.0
        if enet_alpha is None:
            cf = ridge_fit(np.hstack([np.ones((TRAIN_WIN, 1)), (Ftr - mu) / sd]), ytr)
            b0, w = cf[0], cf[1:]
        else:
            mdl = ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio, max_iter=5000)
            mdl.fit((Ftr - mu) / sd, ytr)
            b0, w = mdl.intercept_, mdl.coef_
            nz.append(int((w != 0).sum()))
        P[seg : seg + (hi - t)] = ((F[TRAIN_WIN:] - mu) / sd) @ w + b0
    pr, tr = apply_duan_smearing(P, y[n0:n1], b[n0:n1])
    mm = forecast_metrics(pr, tr)
    np.savez(f"results/geo_preds/pbe_{tag}.npz", pred_raw=pr, true_raw=tr)
    nzs = f" nonzero={np.mean(nz):.0f}/{F.shape[1]}" if nz else ""
    print(
        f"[pbe] {tag:<12} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f}{nzs} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return pr, tr


pc, tr = run("ridge_cad")
for a in (3e-5, 3e-4, 3e-3):
    pe, _ = run(f"enet_{a:g}", enet_alpha=a)
    r = dm_test(qlike_per_bar(pe, tr), qlike_per_bar(pc, tr), h=1)
    print(
        f"[pbe]   enet_{a:g} vs ridge_cad: diff={r['mean_diff']:+.5f} p={r['p']:.4f}",
        flush=True,
    )
print("[pbe] DONE", flush=True)
