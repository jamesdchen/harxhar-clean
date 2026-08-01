# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Straddle test, take 2: alpha-ladder for the full stack + causal smear.

Same honest-lag walkforward as straddle_test.py, but each day's gram is
solved at EVERY alpha in the ladder (near-free with BlockRidge), and all
product metrics use the causal Duan smear (trailing-250-day residual
variance) — the floor the first pass lacked. Answers: does breadth survive
honest labels at daily+ horizons once shrinkage is set for the horizon?

env: H, DTE_LO, DTE_HI, DTE_TARGET, TAG, ALPHAS (comma list)
"""

import os
import time
import numpy as np
import pandas as pd

H = int(os.environ["H"])
DTE_LO = int(os.environ.get("DTE_LO", "4"))
DTE_HI = int(os.environ.get("DTE_HI", "9"))
DTE_TGT = float(os.environ.get("DTE_TARGET", "7"))
TAG = os.environ.get("TAG", f"ha{H}")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "1,10,100,1000,10000").split(",")]
W = 24000
D = "results/b2_mmap"

from src.data.loading import load_raw_data
from block_ridge import BlockRidge

X = np.load(f"{D}/X.npy", mmap_mode="r")
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
dec = dec[(dec >= W + H) & (dec + H < n)]
dec = dec[np.array([okrow[d - H - W : d - H].all() for d in dec])]
print(f"[{TAG}] decision days: {len(dec)}", flush=True)

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


def walkforward(F, tag, alphas):
    t0c = time.time()
    P = {a: np.empty(len(dec)) for a in alphas}
    br = BlockRidge(alphas[0])
    prev = None
    for i, d in enumerate(dec):
        lo, hi = d - H - W, d - H
        if prev is None or br.needs_rebuild():
            br.build(F[lo:hi], yHz[lo:hi])
        else:
            br.slide(
                F[prev[1] : hi], yHz[prev[1] : hi], F[prev[0] : lo], yHz[prev[0] : lo]
            )
        prev = (lo, hi)
        row = np.asarray(F[d])
        for a in alphas:
            beta = br.coef(a)
            P[a][i] = float(row @ beta[1:] + beta[0])
        if i % 1000 == 999:
            print(
                f"  [{TAG}] {tag} {i + 1}/{len(dec)} ({time.time() - t0c:.0f}s)",
                flush=True,
            )
    print(f"[{TAG}] {tag} done ({time.time() - t0c:.0f}s)", flush=True)
    return P


P_full = walkforward(X, "full", ALPHAS)
P_inc = walkforward(Xi, "incumbent", [1.0])[1.0]

dates = pd.to_datetime(tt.iloc[dec].dt.date.to_numpy())

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
Rvar = RVf[dec]
ok = np.isfinite(IVvar) & (Rvar > 0)
LAG = max(1, (H // 48) - 1)
ql = lambda p, t: np.log(p) + t / p


def causal_smear(P):
    res2 = (
        pd.Series((yH[dec] - P) ** 2)
        .rolling(250, min_periods=60)
        .mean()
        .shift(1)
        .to_numpy()
    )
    return (
        P**2 + np.where(np.isfinite(res2), res2, np.nanmean(res2[np.isfinite(res2)]))
    ) * Bf[dec]


def nw_ols(A, yy, lag):
    A = np.column_stack([np.ones(len(yy))] + list(A))
    G = np.linalg.inv(A.T @ A)
    be = G @ A.T @ yy
    u = yy - A @ be
    Au = A * u[:, None]
    S = Au.T @ Au
    for Lg in range(1, lag + 1):
        Gm = Au[Lg:].T @ Au[:-Lg]
        S += (1 - Lg / (lag + 1)) * (Gm + Gm.T)
    V = G @ S @ G
    r2 = 1 - u @ u / ((yy - yy.mean()) @ (yy - yy.mean()))
    return be, be / np.sqrt(np.diag(V)), r2


def strat(lFc, lI, Rv, IVv, lag):
    s_raw = lFc - lI
    mu = pd.Series(s_raw).expanding(60).mean().shift(1).to_numpy()
    v = np.isfinite(mu)
    pos = np.sign(s_raw[v] - mu[v])
    ret = pos * (Rv[v] / IVv[v] - 1.0)
    g0 = ret - ret.mean()
    var = g0 @ g0 / len(ret)
    for Lg in range(1, lag + 1):
        var += 2 * (1 - Lg / (lag + 1)) * (g0[Lg:] @ g0[:-Lg]) / len(ret)
    t_ = ret.mean() / np.sqrt(var / len(ret))
    stride = H // 48
    nov = ret[::stride]
    return t_, nov.mean() / nov.std() * np.sqrt(252 / stride)


lR, lI = np.log(Rvar[ok]), np.log(IVvar[ok])
Fi_sm = causal_smear(P_inc)
_, _, r2_i = nw_ols([lI], lR, LAG)
qi = ql(Fi_sm[ok], Rvar[ok]).mean()
_, tsi, r2_ic = nw_ols([lI, np.log(Fi_sm[ok])], lR, LAG)
ti_, shi = strat(np.log(Fi_sm[ok]), lI, Rvar[ok], IVvar[ok], LAG)
print(
    f"\n[{TAG}] implied-only R2={r2_i:.4f} | incumbent(a1): qlike={qi:+.4f} dR2={r2_ic - r2_i:+.4f} "
    f"t={tsi[2]:+.2f} strat_t={ti_:+.2f} sharpe={shi:+.2f}",
    flush=True,
)

save = {
    "dates": dates.astype("int64"),
    "Rvar": Rvar,
    "IVvar": IVvar,
    "ok": ok,
    "P_inc": P_inc,
}
for a in ALPHAS:
    Fv = causal_smear(P_full[a])
    save[f"P_full_a{a:g}"] = P_full[a]
    q = ql(Fv[ok], Rvar[ok]).mean()
    _, ts, r2 = nw_ols([lI, np.log(Fv[ok])], lR, LAG)
    t_, sh = strat(np.log(Fv[ok]), lI, Rvar[ok], IVvar[ok], LAG)
    win = np.mean(ql(Fv[ok], Rvar[ok]) < ql(Fi_sm[ok], Rvar[ok])) * 100
    print(
        f"[{TAG}] full a={a:<7g} qlike={q:+.4f} dR2={r2 - r2_i:+.4f} t={ts[2]:+.2f} "
        f"strat_t={t_:+.2f} sharpe={sh:+.2f} win%={win:.1f}",
        flush=True,
    )
np.savez(f"results/straddle_alpha_{TAG}.npz", **save)
print(f"[{TAG}] DONE", flush=True)
