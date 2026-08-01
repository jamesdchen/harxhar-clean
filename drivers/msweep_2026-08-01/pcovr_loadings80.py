# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from src.features.transforms.residualizer import RollingRidgeResidualizer
from src.models.spectral_knn import PCOVRKNN

D = "results/b2_mmap"
START, TRAIN_WIN, U, W = 24000, 24000, 256, 12
n1 = 26189
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[: n1 + 300])
y = np.load(f"{D}/y.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
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


t = START + 1200
full = slice(t - TRAIN_WIN, t)
A0 = np.hstack([np.ones((TRAIN_WIN, 1)), O[full], X[full][:, ladder_cols]])
cff = ridge_fit(A0, y[full])
Km = cff[1 + O.shape[1] :].reshape(len(chan), -1)
_, sv, Vt = np.linalg.svd(Km, full_matrices=False)
sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
S = (Vt * sgn[:, None])[:5]
cores = [kfeat(y, prof(y, t))] + [kfeat(bar1[c], prof(bar1[c], t)) for c in labels]
lo = t - TRAIN_WIN
C = np.einsum("cnr,kr->nck", L[:, lo:t], S).reshape(TRAIN_WIN, -1)
F = np.hstack([Kc[lo:t, None] for Kc in cores] + [C, X[lo:t][:, cal_cols]])
rr = RollingRidgeResidualizer(1.0, fast_inverse=True)
rr.fit(F, y[lo:t])
resid = y[lo:t] - rr.predict(F)
views = np.stack([resid[j - W : j] for j in range(W, TRAIN_WIN)])
state_cols = [i for i, nm in enumerate(names) if not nm.startswith("har_ma_")]
state_cols = [
    i
    for i in state_cols
    if (mm := re.search(r"_ma_(\d+)$", names[i])) is None or int(mm.group(1)) <= 128
]
exog = X[lo + W : t][:, state_cols]
scale = np.sqrt(views.shape[1] + exog.shape[1])
V = np.hstack([views, exog]) / scale
m = PCOVRKNN(16, 8000, beta=0.5, local_alpha=0.01).fit(V, resid[W:])
Wrot = np.asarray(m._R)
print("rotation shape:", Wrot.shape)
if Wrot.shape[0] != V.shape[1]:
    Wrot = Wrot.T
agg = np.abs(Wrot).sum(axis=1)
v_share = agg[:W].sum() / agg.sum()
print(f"residual-view share of total loading: {v_share:.3f}")
raw_agg = agg[W:]
order = np.argsort(raw_agg)[::-1]
print("top 80 raw-state features by aggregate |loading|:")
for i in order[:80]:
    print(f"  {names[state_cols[i]]:<45} {raw_agg[i]:.4f}")
np.save(
    "results/rawc_ranking.npy",
    np.array([names[state_cols[i]] for i in order], dtype=object),
)
fam = {}
for i, a in enumerate(raw_agg):
    base = re.sub(r"_ma_\d+$", "", names[state_cols[i]])
    fam[base] = fam.get(base, 0.0) + a
print("\nfamily totals (top 15):")
for k, v_ in sorted(fam.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {k:<40} {v_:.4f}")
