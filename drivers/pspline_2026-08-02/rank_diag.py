# ruff: noqa: E402, E741, E731  (experiment driver, verified by execution)
"""Rank-ceiling diagnostic: is the sqrt2 panel's flatter spectrum signal or noise?

A rank-5 smooth operator sampled at 23 rungs instead of 12 is still rank 5 —
function-space dimension does not grow with grid resolution. So the observed
flattening (top-5 energy 94% -> 81%) is either (a) real structure the 12-rung
grid quantized away, or (b) estimation noise: neighboring sqrt2-spaced MAs are
severely collinear, their coefficients noisier, and coefficient noise is
FULL-rank, padding the tail. Two direct discriminators, no QLIKE needed:

1. SPLIT-HALF STABILITY: estimate Km separately on the two disjoint halves of
   the training window. Real directions replicate across halves; noise does
   not. Report principal angles between the halves' top-K subspaces (lag/row
   space) for K=1..10, per panel.

2. CROSS-GRID SHAPES: interpolate the b2 top-5 shapes (functions of log2-lag)
   onto the sqrt2 rung grid and compare with the sqrt2 top-5 subspace. Small
   angles => the predictive geometry is grid-invariant; the sqrt2 panel found
   the SAME five questions plus a noisier tail.

3. NOISE-FLOOR SCALE: trailing singular values per panel (the noise shelf),
   and per-rung coefficient-collinearity (min eigenvalue of each channel's
   normalized ladder gram) as the mechanism.

env: PREP_B2, PREP_SQ2, T (window end, default 24000)
"""

import os
import re

import numpy as np


def load_panel(path):
    z = np.load(path, allow_pickle=True)
    X, y = z["X"], np.asarray(z["y"], float)
    names = [str(s) for s in z["names"]]
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = list(chan)
    nr = len(chan[labels[0]])
    ladder = [j for c in labels for _, j in chan[c]]
    other = [j for j in range(len(names)) if j not in set(ladder)]
    return dict(X=np.ascontiguousarray(X), y=y, names=names, labels=labels,
                nr=nr, ladder=ladder, other=other)


def svd_shapes(Km, k):
    _, sv, Vt = np.linalg.svd(Km, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    return sv, (Vt * sgn[:, None])[:k]

PREP_B2 = os.environ.get("PREP_B2", "results/prep_cache_all_features_b2_r32000.npz")
PREP_SQ2 = os.environ.get(
    "PREP_SQ2", "results/prep_cache_all_features_b1.41421_r32000.npz")
T = int(os.environ.get("T", "24000"))
W = 24000


def probe_km_win(P, lo, hi):
    A = np.hstack([np.ones((hi - lo, 1)), P["X"][lo:hi][:, P["other"]],
                   P["X"][lo:hi][:, P["ladder"]]])
    G = A.T @ A + np.eye(A.shape[1])
    G[0, 0] -= 1.0
    cf = np.linalg.solve(G, A.T @ P["y"][lo:hi])
    return cf[1 + len(P["other"]) :].reshape(len(P["labels"]), P["nr"])


def angles(V1, V2):
    Q1, _ = np.linalg.qr(V1.T)
    Q2, _ = np.linalg.qr(V2.T)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def rungs_of(P):
    import re

    ch0 = P["labels"][0]
    out = []
    for nm in P["names"]:
        m = re.match(rf"^adj_{re.escape(ch0)}_ma_(\d+)$", nm)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


PB = load_panel(PREP_B2)
PS = load_panel(PREP_SQ2)
RB, RS = rungs_of(PB), rungs_of(PS)
print(f"b2 rungs {RB}\nsq2 rungs {RS}\n")

for tag, P in [("b2 ", PB), ("sq2", PS)]:
    KmA = probe_km_win(P, T - W, T - W // 2)
    KmB = probe_km_win(P, T - W // 2, T)
    KmF = probe_km_win(P, T - W, T)
    svA, _ = svd_shapes(KmA, P["nr"])
    svF = np.linalg.svd(KmF, compute_uv=False)
    _, sv_, VtA = np.linalg.svd(KmA, full_matrices=False)
    _, sv2, VtB = np.linalg.svd(KmB, full_matrices=False)
    print(f"[{tag}] full-window sv: {np.round(svF[:10], 3)}")
    print(f"[{tag}] half-A sv: {np.round(sv_[:10], 3)}")
    print(f"[{tag}] split-half max principal angle of top-K subspace (deg):")
    for k in range(1, 11):
        if k > P["nr"]:
            break
        a = angles(VtA[:k], VtB[:k])
        print(f"    K={k:<2} max={a.max():5.1f}  all={np.round(a, 0)}")
    # collinearity mechanism: per-channel normalized ladder gram min-eig
    lows = []
    for ci, c in enumerate(P["labels"]):
        cols = P["ladder"][ci * P["nr"] : (ci + 1) * P["nr"]]
        Lc = P["X"][T - W : T][:, cols]
        keep = Lc.std(axis=0) > 1e-12
        if keep.sum() < 2:
            continue
        Gc = np.corrcoef(Lc[:, keep], rowvar=False)
        if not np.all(np.isfinite(Gc)):
            continue
        lows.append(np.linalg.eigvalsh(Gc)[0])
    print(f"[{tag}] per-channel ladder-gram min eigenvalue: "
          f"median={np.median(lows):.2e} min={np.min(lows):.2e}\n")

# cross-grid: b2 top-5 shapes resampled to the sq2 grid vs sq2 top-5
Km_b2 = probe_km_win(PB, T - W, T)
Km_s2 = probe_km_win(PS, T - W, T)
_, V5b = svd_shapes(Km_b2, 5)
_, V5s = svd_shapes(Km_s2, 5)
gb, gs = np.log2(RB), np.log2(RS)
V5b_on_s = np.vstack([np.interp(gs, gb, V5b[k]) for k in range(5)])
a = angles(V5b_on_s, V5s)
print(f"cross-grid: angles(interp(b2 top-5) , sq2 top-5) = {np.round(a, 1)}")
for k in range(1, 9):
    _, Vkb = svd_shapes(Km_b2, min(k, PB["nr"]))
    _, Vks = svd_shapes(Km_s2, k)
    Vb_on_s = np.vstack([np.interp(gs, gb, Vkb[j]) for j in range(len(Vkb))])
    a = angles(Vb_on_s, Vks)
    print(f"  K={k}: max angle={a.max():5.1f}  all={np.round(a, 0)}")
