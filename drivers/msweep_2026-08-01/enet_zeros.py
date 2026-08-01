# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from sklearn.linear_model import ElasticNet

D = "results/b2_mmap"
START, END, TRAIN_WIN = 24000, 26189, 24000
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[: END + 300])
y = np.load(f"{D}/y.npy")
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
cal_names = [
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
cal_cols = [names.index(nm) for nm in cal_names]
bar1 = {c: X[:, chan[c][0][1]] for c in labels}


def ridge_fit(A, yy):
    G = A.T @ A + np.eye(A.shape[1])
    G[0, 0] -= 1.0
    return np.linalg.solve(G, A.T @ yy)


def kfeat(s, beta):
    u = np.arange(1, U + 1, dtype=np.float64)
    w = u ** (-beta)
    w /= w.sum()
    K = np.zeros(n1 + 8)
    sw = sliding_window_view(s[: n1 + 8], U)
    K[U:] = sw[:-1] @ w[::-1]
    return K


def prof(s, t):
    if s[t - TRAIN_WIN : t].std() <= 0:
        return 1.0

    def loss(be):
        K = kfeat(s, be)
        A = np.stack([np.ones(TRAIN_WIN), K[t - TRAIN_WIN : t]], axis=1)
        cf, *_ = np.linalg.lstsq(A, y[t - TRAIN_WIN : t], rcond=None)
        r = y[t - TRAIN_WIN : t] - A @ cf
        return float(r @ r)

    return minimize_scalar(loss, bounds=(0.05, 3.0), method="bounded").x


feat_names = (
    ["core_target"]
    + [f"core_{c}" for c in labels]
    + [f"C_{c}_s{k + 1}" for c in labels for k in range(5)]
    + cal_names
)
Kbar = None
zero_sets = []
for seg in (0, 1200, 2160):
    t = n0 + seg
    full = slice(t - TRAIN_WIN, t)
    A0 = np.hstack([np.ones((TRAIN_WIN, 1)), O[full], X[full][:, ladder_cols]])
    cff = ridge_fit(A0, y[full])
    Km = cff[1 + O.shape[1] :].reshape(len(chan), -1)
    Kbar = Km if Kbar is None else 0.9 * Kbar + 0.1 * Km
    _, sv, Vt = np.linalg.svd(Kbar, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    S = (Vt * sgn[:, None])[:5]
    cores = [kfeat(y, prof(y, t))] + [kfeat(bar1[c], prof(bar1[c], t)) for c in labels]
    lo = t - TRAIN_WIN
    C = np.einsum("cnr,kr->nck", L[:, lo:t], S).reshape(TRAIN_WIN, -1)
    F = np.hstack([Kc[lo:t, None] for Kc in cores] + [C, X[lo:t][:, cal_cols]])
    mu, sd = F.mean(0), F.std(0)
    sd[sd == 0] = 1.0
    mdl = ElasticNet(alpha=3e-5, l1_ratio=0.5, max_iter=5000)
    mdl.fit((F - mu) / sd, y[lo:t])
    z = set(np.flatnonzero(mdl.coef_ == 0))
    zero_sets.append(z)
    print(f"seg@{seg}: {len(z)} zeros", flush=True)
common = set.intersection(*zero_sets)
print(f"\ncommon zeros across 3 segments: {len(common)}")
byorg = {"cores": [], "shapes": [], "cal": []}
for i in sorted(common):
    nm = feat_names[i]
    byorg[
        "cores" if nm.startswith("core") else ("cal" if nm in cal_names else "shapes")
    ].append(nm)
print(f"cores zeroed ({len(byorg['cores'])}):", byorg["cores"])
print(f"cal zeroed ({len(byorg['cal'])}):", byorg["cal"])
sh = byorg["shapes"]
from collections import Counter

by_shape = Counter(nm.rsplit("_s", 1)[1] for nm in sh)
by_chan = Counter(nm[2:].rsplit("_s", 1)[0] for nm in sh)
print(f"shape cells zeroed ({len(sh)}): by shape rank {dict(sorted(by_shape.items()))}")
print(
    "channels losing >=3 shape cells:",
    {c: k for c, k in by_chan.most_common() if k >= 3},
)
print("full shape-cell list:", sh)
