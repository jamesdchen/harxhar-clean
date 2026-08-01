# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""LEGCHAMP + CLOCK ORGAN: the legibility champion with the Fourier-modulated
shape block (s=0.3) added to its anchor. Everything else byte-identical to
the original legchamp recipe (mixture-HN anchor, expanding Kbar pool, PCovR
correction, trailing-1000 stack). Tiles via PB_START/PB_END.
"""

import os
import time
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.features.transforms.residualizer import RollingRidgeResidualizer
from src.models.spectral_knn import PCOVRKNN

D = "results/b2_mmap"
START = int(os.environ.get("PB_START", "24000"))
END = int(os.environ.get("PB_END", "26189"))
TAG = os.environ.get("TAG", f"lcc_{START}")
TRAIN_WIN, U, W = 24000, 256, 12
SMOD = 0.3
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[: END + 300])
y = np.load(f"{D}/y.npy")
b = np.load(f"{D}/b.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n0, n1 = START, END

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
state_cols = [i for i, nm in enumerate(names) if not nm.startswith("har_ma_")]
state_cols = [
    i
    for i in state_cols
    if (mm := re.search(r"_ma_(\d+)$", names[i])) is None or int(mm.group(1)) <= 128
]
hour_col = names.index("hour")


def ridge_fit(A, yy):
    G = A.T @ A + np.eye(A.shape[1])
    G[0, 0] -= 1.0
    return np.linalg.solve(G, A.T @ yy)


def kfeat(beta):
    u = np.arange(1, U + 1, dtype=np.float64)
    w = u ** (-beta)
    w /= w.sum()
    K = np.zeros(len(y))
    sw = sliding_window_view(y, U)
    K[U:] = sw[:-1] @ w[::-1]
    return K


KMIX = np.stack([kfeat(be) for be in (0.3, 0.6, 1.0, 1.5, 2.2)], axis=1)

P_anchor = np.empty(n1 - n0)
P_final = np.empty(n1 - n0)
stk_c, stk_r = [], []
Kbar, nseen = None, 0
t0 = time.time()
for seg in range(0, n1 - n0, 240):
    t = n0 + seg
    full = slice(t - TRAIN_WIN, t)
    A0 = np.hstack([np.ones((TRAIN_WIN, 1)), O[full], X[full][:, ladder_cols]])
    cff = ridge_fit(A0, y[full])
    Km = cff[1 + O.shape[1] :].reshape(len(chan), -1)
    Kbar = Km if Kbar is None else (nseen * Kbar + Km) / (nseen + 1)
    nseen += 1
    _, sv, Vt = np.linalg.svd(Kbar, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    S = (Vt * sgn[:, None])[:5]
    lo, hi = t - TRAIN_WIN, min(t + 240, n1)
    C = np.einsum("cnr,kr->nck", L[:, lo:hi], S).reshape(hi - lo, -1)
    ph = 2 * np.pi * X[lo:hi][:, hour_col] / 24.0
    mods = [np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)]
    F = np.hstack(
        [KMIX[lo:hi], C, X[lo:hi][:, cal_cols]]
        + [C * m_[:, None] * SMOD for m_ in mods]
    )
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
    print(f"[{TAG}] seg {seg // 240 + 1}/10 ({time.time() - t0:.0f}s)", flush=True)

pa, tr = apply_duan_smearing(P_anchor, y[n0:n1], b[n0:n1])
pf, _ = apply_duan_smearing(P_final, y[n0:n1], b[n0:n1])
np.savez(f"results/geo_preds/{TAG}_final.npz", pred_raw=pf, true_raw=tr)
np.savez(f"results/geo_preds/{TAG}_anchor.npz", pred_raw=pa, true_raw=tr)
ma = forecast_metrics(pa, tr)
mf = forecast_metrics(pf, tr)
r = dm_test(qlike_per_bar(pf, tr), qlike_per_bar(pa, tr), h=1)
print(f"[{TAG}] anchor(+clock)  qlike={ma['qlike']:.6f} mz={ma['mz_beta']:.3f}")
print(
    f"[{TAG}] FINAL(+corr)    qlike={mf['qlike']:.6f} mz={mf['mz_beta']:.3f} "
    f"| corr increment diff={r['mean_diff']:+.5f} p={r['p']:.4f}"
)
for ref, lbl in (
    ("legibility_legchamp", "original legchamp"),
    ("s_stack1000", "CHAMPION"),
):
    try:
        z = np.load(f"results/geo_preds/{ref}.npz")
        if len(z["pred_raw"]) == len(pf):
            r2 = dm_test(
                qlike_per_bar(pf, tr), qlike_per_bar(z["pred_raw"], z["true_raw"]), h=1
            )
            print(f"[{TAG}] vs {lbl}: diff={r2['mean_diff']:+.5f} p={r2['p']:.4f}")
    except FileNotFoundError:
        pass
print(f"[{TAG}] DONE", flush=True)
