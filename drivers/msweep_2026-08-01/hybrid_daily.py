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
ALPHAS = [1.0, 0.1]
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


def fwd(arr, HH):
    c = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.full(len(arr), np.nan)
    out[: len(arr) - HH + 1] = c[HH:] - c[:-HH]
    return out


RVf, Bf = fwd(RVraw, H), fwd(b, H)
yH = np.sqrt(RVf / Bf)
okrow = ~np.isnan(yH)
yHz = np.where(okrow, yH, 0.0)
close_t = pd.to_datetime("16:00").time()
dec = np.flatnonzero((tt.dt.time == close_t).to_numpy())
dec = dec[(dec >= W + H + U) & (dec + H < n)]
dec = dec[np.array([okrow[d - H - W : d - H].all() for d in dec])]
print(f"[{TAG}] decision days: {len(dec)}", flush=True)


def build_F(lo, hi, S):
    C = np.einsum("cnr,kr->nck", L32[:, lo:hi].astype(np.float64), S).reshape(
        hi - lo, -1
    )
    return np.hstack(
        [KM[lo:hi].astype(np.float64), C, np.asarray(X[lo:hi][:, cal_cols])]
    )


# --- probe engine for shapes (h=1 target, no lag needed) ---
probe = BlockRidge(1.0, rebuild_every=40)
Kbar = None
P_out = {a: np.empty(len(dec)) for a in ALPHAS}
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
    row = build_F(d, d + 1, S)[0]
    for a in ALPHAS:
        beta = amp.coef(a)
        P_out[a][i] = float(row @ beta[1:] + beta[0])
    if i % 500 == 499:
        print(f"  [{TAG}] {i + 1}/{len(dec)} ({time.time() - t0:.0f}s)", flush=True)
print(f"[{TAG}] hybrid walkforward done ({time.time() - t0:.0f}s)", flush=True)

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
Xi = np.ascontiguousarray(X[:, inc_cols])
Pi = np.empty(len(dec))
br = BlockRidge(1.0)
prev = None
for i, d in enumerate(dec):
    lo, hi = d - H - W, d - H
    if prev is None or br.needs_rebuild():
        br.build(Xi[lo:hi], yHz[lo:hi])
    else:
        br.slide(
            Xi[prev[1] : hi], yHz[prev[1] : hi], Xi[prev[0] : lo], yHz[prev[0] : lo]
        )
    prev = (lo, hi)
    beta = br.coef()
    Pi[i] = float(Xi[d] @ beta[1:] + beta[0])
print(f"[{TAG}] incumbent done", flush=True)

dates = pd.to_datetime(tt.iloc[dec].dt.date.to_numpy())
Rvar = RVf[dec]
ch = pd.read_parquet(
    "data/optionm_spx_chain.parquet",
    columns=["date", "exdate", "cp_flag", "strike", "impl_volatility"],
)
ch["dte"] = (ch["exdate"] - ch["date"]).dt.days
ch = ch[(ch.dte >= DTE_LO) & (ch.dte <= DTE_HI) & ch.impl_volatility.notna()]
sp = pd.read_parquet("data/optionm_spx_spot.parquet")
ch = ch.merge(sp, on="date", how="inner")
ch["dist"] = (ch.strike - ch.close).abs()
best = ch.loc[
    ch.groupby(["date", "exdate"])["dist"].idxmin(), ["date", "exdate", "strike"]
]
atm = ch.merge(best, on=["date", "exdate", "strike"])
iv = (
    atm.groupby(["date", "exdate", "dte"])
    .agg(
        iv=("impl_volatility", "mean"), dist=("dist", "first"), close=("close", "first")
    )
    .reset_index()
)
iv = iv[iv.dist / iv.close < 0.02]
iv["tgt"] = (iv.dte - DTE_TGT).abs()
iv = iv.loc[iv.groupby("date")["tgt"].idxmin()]
imp = pd.DataFrame({"date": dates}).merge(iv[["date", "iv"]], on="date", how="left")
IVvar = (imp.iv.to_numpy() ** 2) * (H / 48) / 252.0
ok = np.isfinite(IVvar) & (Rvar > 0)
print(f"[{TAG}] matched days: {ok.sum()}/{len(dec)}", flush=True)

ql = lambda p, t: np.log(p) + t / p
LAG = max(1, (H // 48) - 1)
lR, lI = np.log(Rvar[ok]), np.log(IVvar[ok])


def causal_smear(Pm):
    res2 = (
        pd.Series((yH[dec] - Pm) ** 2)
        .rolling(250, min_periods=60)
        .mean()
        .shift(1)
        .to_numpy()
    )
    fill = np.nanmean(res2[np.isfinite(res2)])
    return (Pm**2 + np.where(np.isfinite(res2), res2, fill)) * Bf[dec]


def nw_ols(A, yy, lag):
    A = np.column_stack([np.ones(len(yy))] + list(A))
    G = np.linalg.inv(A.T @ A)
    be = G @ A.T @ yy
    u = yy - A @ be
    Au = A * u[:, None]
    Sm = Au.T @ Au
    for Lg in range(1, lag + 1):
        Gm = Au[Lg:].T @ Au[:-Lg]
        Sm += (1 - Lg / (lag + 1)) * (Gm + Gm.T)
    V = G @ Sm @ G
    return (
        be,
        be / np.sqrt(np.diag(V)),
        1 - u @ u / ((yy - yy.mean()) @ (yy - yy.mean())),
    )


def nw_mean_t(x, lag):
    g0 = x - x.mean()
    var = g0 @ g0 / len(x)
    for Lg in range(1, lag + 1):
        var += 2 * (1 - Lg / (lag + 1)) * (g0[Lg:] @ g0[:-Lg]) / len(x)
    return x.mean() / np.sqrt(var / len(x))


_, _, r2_i = nw_ols([lI], lR, LAG)
print(f"[{TAG}] implied-only R2={r2_i:.4f}", flush=True)
models = {f"hybrid_a{a:g}": P_out[a] for a in ALPHAS}
models["incumbent"] = Pi
save = {"dates": dates.astype("int64"), "Rvar": Rvar, "IVvar": IVvar, "ok": ok}
for k, Pm in models.items():
    Fv = causal_smear(Pm)
    save[f"P_{k}"] = Pm
    q = ql(Fv[ok], Rvar[ok]).mean()
    _, ts, r2 = nw_ols([lI, np.log(Fv[ok])], lR, LAG)
    s_raw = np.log(Fv[ok]) - lI
    mu = pd.Series(s_raw).expanding(60).mean().shift(1).to_numpy()
    v = np.isfinite(mu)
    pos = np.sign(s_raw[v] - mu[v])
    ret = pos * (Rvar[ok][v] / IVvar[ok][v] - 1.0)
    print(
        f"[{TAG}] {k:<12} qlike={q:+.4f} dR2={r2 - r2_i:+.4f} t_incr={ts[2]:+.2f} "
        f"strat_t={nw_mean_t(ret, LAG):+.2f} hit={np.mean(ret > 0):.3f}",
        flush=True,
    )
np.savez(f"results/hybrid_daily_{TAG}.npz", **save)
print(f"[{TAG}] DONE", flush=True)
