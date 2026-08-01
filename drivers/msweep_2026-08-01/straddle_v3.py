# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Straddle v3: arm the WINNER (incumbent) + real friction pricing.

Models (all honest-lag, daily block solves, causal smear):
  inc          — HAR + calendar (the honest-label winner, control)
  inc_vix      — + log vix / vvix / vix3m + term slope (bar-keyed, ex-ante)
  inc_ann      — + window-aware announcement organ (count-in-window,
                 bars-until-first; the H=8/16 winner, now at product H)
  inc_vix_ann  — both
Friction: ATM straddle half-spread in VOL units (half-spread / straddle
vega) from om_friction dte<=10 quotes, secid 109820 (SPX). Reported vs the
edge distribution + net-of-cost strategy on |edge| > half-spread days.

env: H, DTE_LO, DTE_HI, DTE_TARGET, TAG
"""

import os
import time
import numpy as np
import pandas as pd

H = int(os.environ["H"])
DTE_LO = int(os.environ.get("DTE_LO", "4"))
DTE_HI = int(os.environ.get("DTE_HI", "9"))
DTE_TGT = float(os.environ.get("DTE_TARGET", "7"))
TAG = os.environ.get("TAG", f"v3h{H}")
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

# --- bar-keyed VIX block (values at bar close = known at decision) ---
vx = pd.read_parquet("data/vix_and_voldemand.parquet")
vx["endbartime"] = pd.to_datetime(vx["endbartime"])
raw["t"] = pd.to_datetime(raw["t"])
mv = raw[["t"]].merge(vx, left_on="t", right_on="endbartime", how="left")
VIXB = mv[["vix", "vvix", "vix3m"]].ffill().bfill().to_numpy()[3125:][:n]
VIX = np.column_stack([np.log(VIXB), np.log(VIXB[:, 0] / VIXB[:, 2])])

# --- announcement organ (v1 + window-aware shape at this H) ---
rel = pd.read_parquet("data/releases.parquet")
rel["endbartime"] = pd.to_datetime(rel["endbartime"])
mr = raw[["t"]].merge(rel, left_on="t", right_on="endbartime", how="left").fillna(0.0)
flags = mr[[c for c in rel.columns if c != "endbartime"]].to_numpy()[3125:][:n]
types = [c for c in rel.columns if c != "endbartime"]
del raw, mr, mv
ar = np.arange(n)
ANN = np.zeros((n, 4 * len(types)))
for i in range(len(types)):
    f = flags[:, i]
    idx = np.flatnonzero(f)
    last = np.full(n, -float(n))
    nxt = np.full(n, float(n))
    pos = np.searchsorted(idx, ar, side="right")
    hp = pos > 0
    last[hp] = idx[np.clip(pos - 1, 0, None)][hp]
    hn = pos < len(idx)
    nxt[hn] = idx[np.clip(pos, None, len(idx) - 1)][hn]
    cum = np.concatenate([[0], np.cumsum(f)])
    cnt = np.zeros(n)
    cnt[: n - H] = cum[H + 1 : n + 1] - cum[1 : n - H + 1]
    ANN[:, 4 * i] = np.log1p(ar - last)
    ANN[:, 4 * i + 1] = np.log1p(nxt - ar)
    ANN[:, 4 * i + 2] = cnt
    ANN[:, 4 * i + 3] = np.log1p(np.minimum(nxt - ar, H + 1.0))


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
    print(f"[{TAG}] {tag} done ({time.time() - t0c:.0f}s)", flush=True)
    return P


arms = {
    "inc": Xi,
    "inc_vix": np.hstack([Xi, VIX]),
    "inc_ann": np.hstack([Xi, ANN]),
    "inc_vix_ann": np.hstack([Xi, VIX, ANN]),
}
P = {k: walkforward(F, k) for k, F in arms.items()}

dates = pd.to_datetime(tt.iloc[dec].dt.date.to_numpy())
Rvar = RVf[dec]

# --- implied + friction from the dte<=10 quoted chain (SPX secid 109820) ---
fr = pd.read_parquet(
    "data/om_friction/chain_109820_dte0_10.parquet",
    columns=[
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "impl_volatility",
        "vega",
    ],
)
fr["dte"] = (fr["exdate"] - fr["date"]).dt.days
fr = fr[
    (fr.dte >= DTE_LO)
    & (fr.dte <= DTE_HI)
    & fr.impl_volatility.notna()
    & (fr.best_bid > 0)
    & (fr.vega > 0)
]
sp = pd.read_parquet("data/optionm_spx_spot.parquet")
fr = fr.merge(sp, on="date", how="inner")
# secid 109820 = SPY: underlying ~ SPX/10 (the July SPY+XSP friction study)
fr["spot"] = fr.close / 10.0
fr["dist"] = (fr.strike_price / 1000.0 - fr.spot).abs()
best = fr.loc[
    fr.groupby(["date", "exdate"])["dist"].idxmin(), ["date", "exdate", "strike_price"]
]
atm = fr.merge(best, on=["date", "exdate", "strike_price"])
g = (
    atm.groupby(["date", "exdate", "dte"])
    .agg(
        iv=("impl_volatility", "mean"),
        spr=("best_offer", "sum"),
        bid=("best_bid", "sum"),
        vega=("vega", "sum"),
        ncp=("cp_flag", "nunique"),
        dist=("dist", "first"),
        spot=("spot", "first"),
    )
    .reset_index()
)
g = g[(g.ncp == 2) & (g.dist / g.spot < 0.02)]
g["half_spread_vol"] = (
    (g.spr - g.bid) / 2.0 / g.vega
)  # OM vega is per unit sigma -> decimal vol
g["tgt"] = (g.dte - DTE_TGT).abs()
g = g.loc[g.groupby("date")["tgt"].idxmin()]
imp = pd.DataFrame({"date": dates}).merge(
    g[["date", "iv", "dte", "half_spread_vol"]], on="date", how="left"
)
IVvar = (imp.iv.to_numpy() ** 2) * (H / 48) / 252.0
HSV = imp.half_spread_vol.to_numpy()
ok = np.isfinite(IVvar) & (Rvar > 0) & np.isfinite(HSV)
print(f"[{TAG}] matched days (quoted, ncp=2): {ok.sum()}/{len(dec)}", flush=True)
assert ok.sum() > 100, "degenerate option match — check strike/spot scale"
print(
    f"[{TAG}] half-spread (vol pts): median={np.nanmedian(HSV) * 100:.2f} "
    f"p25={np.nanpercentile(HSV, 25) * 100:.2f} p75={np.nanpercentile(HSV, 75) * 100:.2f}",
    flush=True,
)

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
    S = Au.T @ Au
    for Lg in range(1, lag + 1):
        Gm = Au[Lg:].T @ Au[:-Lg]
        S += (1 - Lg / (lag + 1)) * (Gm + Gm.T)
    V = G @ S @ G
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
save = {
    "dates": dates.astype("int64"),
    "Rvar": Rvar,
    "IVvar": IVvar,
    "HSV": HSV,
    "ok": ok,
}
sigI = np.sqrt(IVvar[ok] * 252 / (H / 48))
for k in arms:
    Fv = causal_smear(P[k])
    save[f"P_{k}"] = P[k]
    q = ql(Fv[ok], Rvar[ok]).mean()
    _, ts, r2 = nw_ols([lI, np.log(Fv[ok])], lR, LAG)
    s_raw = np.log(Fv[ok]) - lI
    mu = pd.Series(s_raw).expanding(60).mean().shift(1).to_numpy()
    v = np.isfinite(mu)
    edge = s_raw[v] - mu[v]
    pos = np.sign(edge)
    ret = pos * (Rvar[ok][v] / IVvar[ok][v] - 1.0)
    t_g = nw_mean_t(ret, LAG)
    # net of friction: trade only when |edge| clears the round-trip in the
    # same units; edge(log-var) ~ 2*(sig_f-sig_i)/sig_i, cost = 2*HSV/sig_i
    hsv = HSV[ok][v]
    sI = sigI[v]
    cost = 2.0 * 2.0 * hsv / sI  # round trip (enter+exit), log-var units
    tr = np.abs(edge) > cost
    net = pos[tr] * (Rvar[ok][v][tr] / IVvar[ok][v][tr] - 1.0) - cost[tr]
    t_n = nw_mean_t(net, LAG) if tr.sum() > 30 else float("nan")
    print(
        f"[{TAG}] {k:<12} qlike={q:+.4f} dR2={r2 - r2_i:+.4f} t_incr={ts[2]:+.2f} | "
        f"gross_t={t_g:+.2f} | traded={tr.mean() * 100:.0f}% net_mean={net.mean() if tr.sum() > 0 else float('nan'):+.4f} net_t={t_n:+.2f}",
        flush=True,
    )
np.savez(f"results/straddle_v3_{TAG}.npz", **save)
print(f"[{TAG}] DONE", flush=True)
