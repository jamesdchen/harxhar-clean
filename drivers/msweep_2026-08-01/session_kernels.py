# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Session-conditional kernels projected on the pooled shape frame.

Descriptive (full-sample, not walk-forward): per session-bin probe ridge ->
per-bin target kernel (12-vec) + exog operator (41x12); amplitudes
a_k(bin) = u_k' Km_bin v_k on the UNCONDITIONAL pooled SVD frame.
Tests: intraday regime == clock-periodic amplitude motion on fixed shapes.
"""

import numpy as np
import re
import pandas as pd
from src.data.loading import load_raw_data

D = "results/b2_mmap"
X = np.load(f"{D}/X.npy", mmap_mode="r")
y = np.load(f"{D}/y.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n = X.shape[0]
raw = (
    load_raw_data("data", allow_missing=True)
    .dropna(subset=["RV"])
    .reset_index(drop=True)
)
tt = pd.to_datetime(raw["t"]).iloc[3125 : 3125 + n].reset_index(drop=True)
del raw
hours = tt.dt.hour.to_numpy() + tt.dt.minute.to_numpy() / 60.0

chan = {}
for j, nm in enumerate(names):
    m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
    if m:
        chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
for c in chan:
    chan[c].sort()
labels = list(chan)
ladder_cols = [j for c in chan for _, j in chan[c]]
har_cols = [j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)]
other_cols = [j for j in range(len(names)) if j not in set(ladder_cols)]
har_in_other = [oi for oi, j in enumerate(other_cols) if j in set(har_cols)]

BINS = [
    ("overnight", 20.0, 28.0),
    ("early", 4.0, 9.5),
    ("open", 9.5, 11.0),
    ("midday", 11.0, 14.5),
    ("close", 14.5, 16.0),
    ("after", 16.0, 20.0),
]


def binmask(lo, hi):
    h = hours % 24
    if hi <= 24:
        return (h >= lo) & (h < hi)
    return (h >= lo) | (h < hi - 24)


def probe(mask):
    idx = np.flatnonzero(mask)
    p = 1 + len(other_cols) + len(ladder_cols)
    G = np.eye(p)
    G[0, 0] = 0.0
    c = np.zeros(p)
    for s in range(0, len(idx), 20000):
        ii = idx[s : s + 20000]
        Xi = np.asarray(X[ii])
        A = np.hstack([np.ones((len(ii), 1)), Xi[:, other_cols], Xi[:, ladder_cols]])
        G += A.T @ A
        c += A.T @ y[ii]
        del Xi, A
    cf = np.linalg.solve(G, c)
    ktar = cf[1:][har_in_other]
    Km = cf[1 + len(other_cols) :].reshape(len(labels), -1)
    return ktar, Km, len(idx)


# unconditional frame
ktar_all, Km_all, n_all = probe(np.ones(n, bool))
U_, sv, Vt = np.linalg.svd(Km_all, full_matrices=False)
sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
V5 = (Vt * sgn[:, None])[:5]
U5 = (U_ * sgn[None, : len(U_)])[:, :5]
ktar_dir = ktar_all / np.linalg.norm(ktar_all)
print(f"unconditional: n={n_all}  sv={np.round(sv[:5], 3)}")
print(f"target kernel (base-2 rungs): {np.round(ktar_all, 3).tolist()}\n")

print(
    f"{'bin':<10} {'n':>7} | target: ||k||, cos(k, k_all) | exog shape amplitudes a1..a5"
)
for nm_, lo, hi in BINS:
    ktar_b, Km_b, nb = probe(binmask(lo, hi))
    cos = float(ktar_b @ ktar_dir / (np.linalg.norm(ktar_b) + 1e-300))
    amps = [float(U5[:, k] @ Km_b @ V5[k]) for k in range(5)]
    proj = Km_b @ V5.T @ V5  # projection onto the 5-dim LAG subspace
    covg = float(np.linalg.norm(proj) ** 2 / np.linalg.norm(Km_b) ** 2)
    np.save(f"results/sesskern_{nm_}.npy", Km_b)
    print(
        f"{nm_:<10} {nb:>7} | {np.linalg.norm(ktar_b):6.3f}  {cos:+.3f}  span={covg:.2f} | "
        + " ".join(f"{a:+.3f}" for a in amps)
    )
print("\nper-bin target kernels (rows = bins, cols = rungs 1..2048):")
for nm_, lo, hi in BINS:
    ktar_b, _, _ = probe(binmask(lo, hi))
    print(f"{nm_:<10} {np.round(ktar_b, 2).tolist()}")
