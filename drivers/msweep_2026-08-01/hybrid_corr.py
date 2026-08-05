# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""hybrid_dp + PCovR local-linear correction + trailing-1000 stack.

The legchamp composition applied to the PARSIMONY champion base
(hybrid257_dp: fitted-beta target core + private channel kernels +
EWMA-10 discounted shape pool). Control = the in-run anchor replica
(exact nested pair). Tests whether the correction increment (-0.0042 on
the mixture-HN base, p=0.0023) composes with the parsimonious base.
"""

import time
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.features.transforms.residualizer import RollingRidgeResidualizer
from src.models.spectral_knn import PCOVRKNN

D = "results/b2_mmap"
START, END, TRAIN_WIN = 24000, 26189, 24000
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[: END + 300])
y = np.load(f"{D}/y.npy")
b = np.load(f"{D}/b.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n0, n1, U, W = START, END, 256, 12

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
state_cols = [i for i, nm in enumerate(names) if not nm.startswith("har_ma_")]
state_cols = [
    i
    for i in state_cols
    if (mm := re.search(r"_ma_(\d+)$", names[i])) is None or int(mm.group(1)) <= 128
]


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


P_anchor = np.empty(n1 - n0)
P_final = np.empty(n1 - n0)
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
    views = np.stack([resid_win[j - W : j] for j in range(W, TRAIN_WIN)])
    vt = resid_win[W:]
    exog_rows = X[lo + W : lo + TRAIN_WIN][:, state_cols]
    scale = np.sqrt(views.shape[1] + exog_rows.shape[1])
    Vfull = np.hstack([views, exog_rows]) / scale
    m_corr = PCOVRKNN(16, 8000, beta=0.5, local_alpha=0.01).fit(Vfull, vt)
    resid_roll = list(resid_win[-W:])
    for i in range(hi - t):
        tt = TRAIN_WIN + i
        base = rr.predict(F[tt : tt + 1])[0]
        v = np.array(resid_roll[-W:])
        q = np.concatenate([v, X[t + i][state_cols]]) / scale
        corr = float(m_corr.predict(q[None, :])[0])
        if len(stk_c) >= 1000:
            A2 = np.stack([np.ones(1000), np.array(stk_c[-1000:])], axis=1)
            wf, *_ = np.linalg.lstsq(A2, np.array(stk_r[-1000:]), rcond=None)
            P_final[seg + i] = base + wf[0] + wf[1] * corr
        else:
            P_final[seg + i] = base + corr
        P_anchor[seg + i] = base
        stk_c.append(corr)
        stk_r.append(float(y[t + i]) - base)
        resid_roll.append(float(y[t + i]) - base)
        if i < hi - t - 1:
            rr.roll(F[tt], y[lo + tt], F[i], y[lo + i])
    print(f"[hc] seg {seg // 240 + 1}/10 ({time.time() - t0:.0f}s)", flush=True)

for tag, P in (
    ("hybrid_dp replica (anchor)", P_anchor),
    ("HYBRID+CORR+STACK", P_final),
):
    pr, tr = apply_duan_smearing(P, y[n0:n1], b[n0:n1])
    mm = forecast_metrics(pr, tr)
    np.savez(
        f"results/geo_preds/hc_{tag.split()[0].lower().replace('+', '_')}.npz",
        pred_raw=pr,
        true_raw=tr,
    )
    print(f"[hc] {tag:<28} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f}", flush=True)
pa, tr = apply_duan_smearing(P_anchor, y[n0:n1], b[n0:n1])
pf, _ = apply_duan_smearing(P_final, y[n0:n1], b[n0:n1])
r = dm_test(qlike_per_bar(pf, tr), qlike_per_bar(pa, tr), h=1)
print(f"[hc] corr+stack vs hybrid_dp anchor: diff={r['mean_diff']:+.5f} p={r['p']:.4f}")
print("[hc] DONE", flush=True)
