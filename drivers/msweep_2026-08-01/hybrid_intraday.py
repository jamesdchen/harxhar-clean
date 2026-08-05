# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Hybrid Hawkes-native model as the DAILY (H=48) straddle forecaster.

Full-sample port of the parsimony champion to the product horizon:
  - mixture cores (5 target betas + 2 per channel, fixed) — privmix was a
    wash on the chunk and removes all per-segment scalar fits
  - shapes: probe ridge (full wide stack, h=1 target, BlockRidge slide at
    240-bar cadence) -> Km -> EWMA-10 pool -> SVD top-5 (champion recipe)
  - amplitudes: ~302 features, honest-lag daily target, BlockRidge daily
    solves at alpha in {1, 0.1}
Compared against the incumbent (same harness) on decision-day QLIKE +
incremental-to-implied + strategy, implied side = SPX chain ATM.

env: H (48), DTE_LO, DTE_HI, DTE_TARGET, TAG
"""

import os
import time
import numpy as np
import pandas as pd
import re

H = int(os.environ.get("H", "48"))
DTE_LO = int(os.environ.get("DTE_LO", "1"))
DTE_HI = int(os.environ.get("DTE_HI", "2"))
DTE_TGT = float(os.environ.get("DTE_TARGET", "1"))
TAG = os.environ.get("TAG", f"hd{H}")
W = 24000
U = 256
CAD = 240
ALPHAS = [
    float(a)
    for a in __import__("os").environ.get("HI_ALPHAS", "100,1000,10000").split(",")
]
D = "results/b2_mmap"

from src.data.loading import load_raw_data
from block_ridge import BlockRidge

X = np.load(f"{D}/X.npy", mmap_mode="r")
y = np.load(f"{D}/y.npy")
b = np.load(f"{D}/b.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n = X.shape[0]
raw = (
    load_raw_data("data", allow_missing=True)
    .dropna(subset=["RV"])
    .reset_index(drop=True)
)
RVraw = raw["RV"].to_numpy()[3125:][:n]
tt = pd.to_datetime(raw["t"]).iloc[3125 : 3125 + n].reset_index(drop=True)
del raw

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
nch, nr = len(labels), 12
L32 = np.empty((nch, n, nr), dtype=np.float32)
for ci, c in enumerate(labels):
    L32[ci] = X[:, [j for _, j in chan[c]]]
probe_cols = other_cols + ladder_cols
npc = len(probe_cols)


def kfeat_full(s, beta):
    from numpy.lib.stride_tricks import sliding_window_view

    u = np.arange(1, U + 1, dtype=np.float64)
    w = u ** (-beta)
    w /= w.sum()
    K = np.zeros(len(s))
    sw = sliding_window_view(s, U)
    K[U:] = sw[:-1] @ w[::-1]
    return K.astype(np.float32)


t0 = time.time()
MIX_T = [0.3, 0.6, 1.0, 1.5, 2.2]
MIX_C = [0.6, 1.5]
KM = np.column_stack(
    [kfeat_full(y, be) for be in MIX_T]
    + [
        kfeat_full(np.ascontiguousarray(X[:, chan[c][0][1]], dtype=np.float64), be)
        for c in labels
        for be in MIX_C
    ]
)
print(f"[{TAG}] cores built {KM.shape} ({time.time() - t0:.0f}s)", flush=True)

rel = pd.read_parquet("data/releases.parquet")
rel["endbartime"] = pd.to_datetime(rel["endbartime"])
_r2 = pd.DataFrame({"t": tt})
mr = _r2.merge(rel, left_on="t", right_on="endbartime", how="left").fillna(0.0)
flags = mr[[c for c in rel.columns if c != "endbartime"]].to_numpy()[:n]
types = [c for c in rel.columns if c != "endbartime"]
ar_ = np.arange(n)
ANN = np.zeros((n, 4 * len(types)))
for i_ in range(len(types)):
    f_ = flags[:, i_]
    idx_ = np.flatnonzero(f_)
    last = np.full(n, -float(n))
    nxt = np.full(n, float(n))
    pos_ = np.searchsorted(idx_, ar_, side="right")
    hp_ = pos_ > 0
    last[hp_] = idx_[np.clip(pos_ - 1, 0, None)][hp_]
    hn_ = pos_ < len(idx_)
    nxt[hn_] = idx_[np.clip(pos_, None, len(idx_) - 1)][hn_]
    cum_ = np.concatenate([[0], np.cumsum(f_)])
    cnt_ = np.zeros(n)
    cnt_[: n - H] = cum_[H + 1 : n + 1] - cum_[1 : n - H + 1]
    ANN[:, 4 * i_] = np.log1p(ar_ - last)
    ANN[:, 4 * i_ + 1] = np.log1p(nxt - ar_)
    ANN[:, 4 * i_ + 2] = cnt_
    ANN[:, 4 * i_ + 3] = np.log1p(np.minimum(nxt - ar_, H + 1.0))
USE_ANN = bool(int(__import__("os").environ.get("USE_ANN", "0")))
USE_TEX = float(__import__("os").environ.get("USE_TEX", "0"))


def fwd(arr, HH):
    c = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.full(len(arr), np.nan)
    out[: len(arr) - HH + 1] = c[HH:] - c[:-HH]
    return out


RVf, Bf = fwd(RVraw, H), fwd(b, H)
yH = np.sqrt(RVf / Bf)
okrow = ~np.isnan(yH)
yHz = np.where(okrow, yH, 0.0)
N0, N1 = 24048, 72048
dec = np.arange(N0, N1, 48)  # daily refits; predictions for every bar
assert okrow[N0 - W - H : N1].all()
print(f"[{TAG}] eval bars: {N1 - N0}", flush=True)


def build_F(lo, hi, S):
    C = np.einsum("cnr,kr->nck", L32[:, lo:hi].astype(np.float64), S).reshape(
        hi - lo, -1
    )
    parts = [KM[lo:hi].astype(np.float64), C, np.asarray(X[lo:hi][:, cal_cols])]
    if USE_ANN:
        parts.append(ANN[lo:hi])
    if USE_TEX > 0:
        # wide overlay: the FULL stack as a heavy block via column scaling
        # (penalty ratio = 1/USE_TEX^2; s=0.1 -> overlay at 100x light alpha)
        parts.append(np.asarray(X[lo:hi]) * USE_TEX)
    return np.hstack(parts)


# --- probe engine for shapes (h=1 target, no lag needed) ---
probe = BlockRidge(1.0, rebuild_every=40)
Kbar = None
P_out = {a: np.empty(N1 - N0) for a in ALPHAS}
amp = None
prev_probe = None
S = None
last_refresh = -(10**9)
t0 = time.time()
for i, d in enumerate(dec):
    if d - last_refresh >= CAD:
        plo, phi = d - W, d
        Xp = np.asarray(X[plo:phi])[:, probe_cols]
        if prev_probe is None or probe.needs_rebuild():
            probe.build(Xp, y[plo:phi])
        else:
            olo, ohi = prev_probe
            probe.slide(
                np.asarray(X[ohi:phi])[:, probe_cols],
                y[ohi:phi],
                np.asarray(X[olo:plo])[:, probe_cols],
                y[olo:plo],
            )
        prev_probe = (plo, phi)
        cff = probe.coef()
        Km = cff[1 + len(other_cols) :].reshape(nch, nr)
        Kbar = Km if Kbar is None else 0.9 * Kbar + 0.1 * Km
        _, sv, Vt = np.linalg.svd(Kbar, full_matrices=False)
        sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
        S = (Vt * sgn[:, None])[:5]
        # amplitude gram rebuilt from scratch (features change with S)
        lo, hi = d - H - W, d - H
        amp = BlockRidge(1.0, rebuild_every=10**9)
        amp.build(build_F(lo, hi, S), yHz[lo:hi])
        prev_amp = (lo, hi)
        last_refresh = d
    else:
        lo, hi = d - H - W, d - H
        olo, ohi = prev_amp
        amp.slide(build_F(ohi, hi, S), yHz[ohi:hi], build_F(olo, lo, S), yHz[olo:lo])
        prev_amp = (lo, hi)
    blk = build_F(d, d + 48, S)
    for a in ALPHAS:
        beta = amp.coef(a)
        P_out[a][d - N0 : d - N0 + 48] = blk @ beta[1:] + beta[0]
    if i % 200 == 199:
        print(f"  [{TAG}] day {i + 1}/{len(dec)} ({time.time() - t0:.0f}s)", flush=True)
print(f"[{TAG}] hybrid walkforward done ({time.time() - t0:.0f}s)", flush=True)

from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from block_ridge import BlockRidge as _BR

inc_cols = [
    j
    for j, nm in enumerate(names)
    if nm.startswith("har_ma_")
    or nm
    in (
        "DOW_0",
        "DOW_1",
        "DOW_2",
        "DOW_3",
        "DOW_4",
        "hour",
        "is_overnight",
        "is_open",
        "is_close",
    )
]
Xi = np.ascontiguousarray(X[:N1, inc_cols])
Pi = np.empty(N1 - N0)
bri = _BR(1.0)
prev = None
for t_ in range(N0, N1, 48):
    lo_, hi_ = t_ - H - W, t_ - H
    if prev is None or bri.needs_rebuild():
        bri.build(Xi[lo_:hi_], yHz[lo_:hi_])
    else:
        bri.slide(
            Xi[prev[1] : hi_], yHz[prev[1] : hi_], Xi[prev[0] : lo_], yHz[prev[0] : lo_]
        )
    prev = (lo_, hi_)
    beta = bri.coef()
    Pi[t_ - N0 : t_ - N0 + 48] = Xi[t_ : t_ + 48] @ beta[1:] + beta[0]

pri, tr = apply_duan_smearing(Pi, yHz[N0:N1], Bf[N0:N1])
mi = forecast_metrics(pri, tr)
print(
    f"[{TAG}] incumbent    qlike={mi['qlike']:.6f} mz={mi['mz_beta']:.3f}", flush=True
)
for a in ALPHAS:
    pr, _ = apply_duan_smearing(P_out[a], yHz[N0:N1], Bf[N0:N1])
    mm = forecast_metrics(pr, tr)
    r = dm_test(qlike_per_bar(pr, tr), qlike_per_bar(pri, tr), h=H)
    np.savez(f"results/geo_preds/{TAG}_a{a:g}.npz", pred_raw=pr, true_raw=tr)
    print(
        f"[{TAG}] hybrid a={a:<7g} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} "
        f"vs inc diff={r['mean_diff']:+.5f} p={r['p']:.4f}",
        flush=True,
    )
print(f"[{TAG}] DONE", flush=True)
