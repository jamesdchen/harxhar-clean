# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Straddle signal pilot: forecast-vs-implied, full-sample walk-forward.

Daily decision at the 16:00 bar. Direct-H ridge stack (full X, b2 basis)
with HONEST label lag: training window [t-H-W, t-H) so every label's
forward RV window is realized by decision time. Incumbent-class (HAR +
calendar) run identically as the breadth control.

Implied side: OptionMetrics SPX chain, ATM straddle IV (mean of C/P at the
strike nearest spot) at the expiry nearest DTE_TARGET calendar days.
Implied variance over the forecast horizon = IV^2 * (H/48)/252.

Tests: VRP known-answer ratio, log-space incremental-to-implied regression
(NW HAC), sign strategy on trailing-demeaned forecast/implied spread with
variance-swap-style P&L proxy.

env: H (48|240), DTE_LO, DTE_HI, DTE_TARGET, TAG
"""

import os
import time
import numpy as np
import pandas as pd

H = int(os.environ["H"])
DTE_LO = int(os.environ.get("DTE_LO", "4"))
DTE_HI = int(os.environ.get("DTE_HI", "9"))
DTE_TGT = float(os.environ.get("DTE_TARGET", "7"))
TAG = os.environ.get("TAG", f"h{H}")
W = 24000
D = "results/b2_mmap"

from src.data.loading import load_raw_data

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
assert len(RVraw) == n


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
print(
    f"[{TAG}] decision days: {len(dec)}  {tt.iloc[dec[0]].date()} -> {tt.iloc[dec[-1]].date()}",
    flush=True,
)

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

from block_ridge import BlockRidge


def walkforward(F, tag):
    t0c = time.time()
    P = np.empty(len(dec))
    br = BlockRidge(1.0)
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
        beta = br.coef()
        P[i] = float(np.asarray(F[d]) @ beta[1:] + beta[0])
        if i % 500 == 499:
            print(
                f"  [{TAG}] {tag} {i + 1}/{len(dec)} ({time.time() - t0c:.0f}s)",
                flush=True,
            )
    print(f"[{TAG}] {tag} done ({time.time() - t0c:.0f}s)", flush=True)
    return P


P_full = walkforward(X, "full")
P_inc = walkforward(Xi, "incumbent")

dates = pd.to_datetime(tt.iloc[dec].dt.date.to_numpy())
Rvar = RVf[dec]
Fvar_full = P_full**2 * Bf[dec]
Fvar_inc = P_inc**2 * Bf[dec]

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
        iv=("impl_volatility", "mean"),
        ncp=("cp_flag", "nunique"),
        dist=("dist", "first"),
        close=("close", "first"),
    )
    .reset_index()
)
iv = iv[iv.dist / iv.close < 0.02]
iv["tgt"] = (iv.dte - DTE_TGT).abs()
iv = iv.loc[iv.groupby("date")["tgt"].idxmin()]
imp = pd.DataFrame({"date": dates}).merge(
    iv[["date", "iv", "dte", "ncp"]], on="date", how="left"
)
IVvar = (imp.iv.to_numpy() ** 2) * (H / 48) / 252.0

# P<=0 would be pathological ridge output on a positive target: drop, count
bad = (P_full <= 0) | (P_inc <= 0)
ok = np.isfinite(IVvar) & (Rvar > 0) & ~bad
print(
    f"[{TAG}] matched days: {ok.sum()}/{len(dec)}  nonpos preds dropped: {bad.sum()}  "
    f"(both C&P at ATM: {(imp.ncp[ok] == 2).mean():.2f})",
    flush=True,
)
np.savez(
    f"results/straddle_{TAG}.npz",
    dates=dates.astype("int64"),
    Rvar=Rvar,
    Fvar_full=Fvar_full,
    Fvar_inc=Fvar_inc,
    IVvar=IVvar,
    P_full=P_full,
    P_inc=P_inc,
    Bf=Bf[dec],
    ok=ok,
)

lR, lI = np.log(Rvar[ok]), np.log(IVvar[ok])
lF, lC = np.log(Fvar_full[ok]), np.log(Fvar_inc[ok])
LAG = max(1, (H // 48) - 1)


def nw_ols(A, yy, lag):
    A = np.column_stack([np.ones(len(yy))] + list(A))
    G = np.linalg.inv(A.T @ A)
    be = G @ A.T @ yy
    u = yy - A @ be
    Au = A * u[:, None]
    S = Au.T @ Au
    for L in range(1, lag + 1):
        Gm = Au[L:].T @ Au[:-L]
        S += (1 - L / (lag + 1)) * (Gm + Gm.T)
    V = G @ S @ G
    r2 = 1 - u @ u / ((yy - yy.mean()) @ (yy - yy.mean()))
    return be, be / np.sqrt(np.diag(V)), r2


print(f"\n[{TAG}] === known-answer ===")
print(
    f"median implied/realized ratio: {np.median(IVvar[ok] / Rvar[ok]):.3f} (VRP: expect ~1.1-1.6)"
)
print(
    f"corr(logR,logIV)={np.corrcoef(lR, lI)[0, 1]:.3f}  corr(logR,logF)={np.corrcoef(lR, lF)[0, 1]:.3f}  "
    f"corr(logR,logInc)={np.corrcoef(lR, lC)[0, 1]:.3f}"
)

_, _, r2_i = nw_ols([lI], lR, LAG)
be, ts, r2_if = nw_ols([lI, lF], lR, LAG)
_, tsc, r2_ic = nw_ols([lI, lC], lR, LAG)
print(f"\n[{TAG}] === incremental-to-implied (log space, NW lag {LAG}) ===")
print(f"implied-only R2: {r2_i:.4f}")
print(
    f"+full forecast : R2 {r2_if:.4f} (d={r2_if - r2_i:+.4f})  beta={be[2]:+.3f} t={ts[2]:+.2f}"
)
print(f"+incumbent     : R2 {r2_ic:.4f} (d={r2_ic - r2_i:+.4f})  t={tsc[2]:+.2f}")


def strategy(lFc, lbl):
    s_raw = lFc - lI
    mu = pd.Series(s_raw).expanding(60).mean().shift(1).to_numpy()
    v = np.isfinite(mu)
    pos = np.sign(s_raw[v] - mu[v])
    ret = pos * (Rvar[ok][v] / IVvar[ok][v] - 1.0)
    g0 = ret - ret.mean()
    var = g0 @ g0 / len(ret)
    for L in range(1, LAG + 1):
        var += 2 * (1 - L / (LAG + 1)) * (g0[L:] @ g0[:-L]) / len(ret)
    tstat = ret.mean() / np.sqrt(var / len(ret))
    stride = H // 48
    nov = ret[::stride]
    sharpe = nov.mean() / nov.std() * np.sqrt(252 / stride)
    print(
        f"strategy[{lbl}]: n={len(ret)} mean={ret.mean():+.4f} t(NW)={tstat:+.2f} "
        f"annSharpe(non-overlap)={sharpe:+.2f} hit={np.mean(ret > 0):.3f}"
    )


print(f"\n[{TAG}] === sign strategy (variance-swap proxy, trailing-demeaned) ===")
strategy(lF, "full")
strategy(lC, "incumbent")
print(f"\n[{TAG}] DONE", flush=True)
