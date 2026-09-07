"""Proposal 17 (one cell, pre-declared): Gaussian expected payoff of the 15:30 package under the variance forecast,
with the forecast sd scaled by a causal, bin-wise terminal-move correction (session-trend tercile x pin-proximity bin),
normalized to mean one across bins each day. q = sign(E_adj[R]). Compared with sign(s) and always short."""

import glob
import sys
from math import erf, exp, pi, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/james/CC Allowed/harxhar-0dte-professor/notebooks")
import atm_straddle_lib as asl  # noqa: E402

R = "C:/Users/james/CC Allowed/harxhar-0dte-professor/results/atm_straddle_0dte_1530/"
d = pd.read_parquet(R + "daily_blk2.parquet")
cache = sorted(
    glob.glob(
        "C:/Users/james/CC Allowed/harxhar-0dte-professor/results/atm_straddle_intraday/cache/trade_*.parquet"
    )
)[-1]
w = pd.read_parquet(cache)
w = w[w["is_last"].notna()].copy()
w["et"] = pd.to_datetime(w["timestamp"], utc=True).dt.tz_convert("America/New_York")
w["hhmm"] = w["et"].dt.strftime("%H:%M")
w["date"] = w["et"].dt.normalize().dt.tz_localize(None)
S10 = w[w["hhmm"] == "10:00"].set_index("date")["S"]
pan = asl.load_yhat_panel_mz(asl.yhat_paths(asl.find_repo())["blk2"])
pan["date"] = pd.to_datetime(pan["date"])
rv_sofar = (
    pan[(pan["mins"] >= 630) & (pan["mins"] <= 930)].groupby("date")["rv_raw"].sum()
)
x = (
    d.join(S10.rename("S10"))
    .join(rv_sofar.rename("rv_sofar"))
    .dropna(subset=["S10", "rv_sofar"])
    .copy()
)
n = len(x)
x["move"] = x["S_close"] - x["S"]
x["sd"] = np.sqrt(x["rv_hat"]) * x["S"]  # forecast sd of the 30-min move in points
x["trend"] = np.log(x["S"] / x["S10"]).abs() / np.sqrt(x["rv_sofar"])
gap = (x["K_c"] - x["K_p"]).replace(0, np.nan)
x["pinprox"] = (np.minimum(x["S"] - x["K_p"], x["K_c"] - x["S"]) / gap).fillna(0.0)
x["pinbin"] = pd.cut(x["pinprox"], [-0.01, 0.1, 0.3, 0.51], labels=[0, 1, 2]).astype(
    int
)
x["m2"] = x["move"] ** 2

# --- causal bins and corrections (prior days only; min 63 days for the tercile cuts; min 30 obs per cell, else 1)
trend_bin = np.full(n, -1)
k_rel = np.ones(n)
tr = x["trend"].to_numpy()
m2, sd2 = x["m2"].to_numpy(), x["sd"].to_numpy() ** 2
pb = x["pinbin"].to_numpy()
for i in range(n):
    if i < 63:
        continue
    q1, q2 = np.quantile(tr[:i], [1 / 3, 2 / 3])
    tb_hist = np.where(tr[:i] <= q1, 0, np.where(tr[:i] <= q2, 1, 2))
    tb_i = 0 if tr[i] <= q1 else (1 if tr[i] <= q2 else 2)
    trend_bin[i] = tb_i
    ks, ws = {}, {}
    for a in range(3):
        for b in range(3):
            sel = (tb_hist == a) & (pb[:i] == b)
            if sel.sum() >= 30:
                ks[(a, b)] = m2[:i][sel].mean() / sd2[:i][sel].mean()
                ws[(a, b)] = sel.sum()
    if (tb_i, pb[i]) in ks and len(ks) >= 2:
        kbar = sum(ks[c] * ws[c] for c in ks) / sum(
            ws.values()
        )  # observation-weighted mean over cells
        k_rel[i] = ks[(tb_i, pb[i])] / kbar
x["trend_bin"], x["k_rel"] = trend_bin, k_rel
print(
    f"days {n}; days with a bin correction {int((trend_bin >= 0).sum())}; k_rel mean {k_rel[trend_bin >= 0].mean():.3f}, sd {k_rel[trend_bin >= 0].std():.3f}, min {k_rel.min():.3f}, max {k_rel.max():.3f}"
)


def phi(z):
    return exp(-0.5 * z * z) / sqrt(2 * pi)


def Phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def e_payout(S, Kc, Kp, sd):
    a, b = Kc - S, S - Kp
    ec = sd * phi(a / sd) - a * (1 - Phi(a / sd))
    ep = sd * phi(b / sd) - b * (1 - Phi(b / sd))
    return ec + ep


S, Kc, Kp, entry = (
    x["S"].to_numpy(),
    x["K_c"].to_numpy(),
    x["K_p"].to_numpy(),
    x["entry"].to_numpy(),
)
sd = x["sd"].to_numpy()
ER_g = np.array([e_payout(S[i], Kc[i], Kp[i], sd[i]) / entry[i] - 1 for i in range(n)])
ER_adj = np.array(
    [
        e_payout(S[i], Kc[i], Kp[i], sd[i] * sqrt(k_rel[i])) / entry[i] - 1
        for i in range(n)
    ]
)
Rr = x["R"].to_numpy()
s = x["signal"].to_numpy()
q0 = np.where(s > 0, 1.0, -1.0)
qg = np.where(ER_g > 0, 1.0, -1.0)
qa = np.where(ER_adj > 0, 1.0, -1.0)
ex = x["exit"].to_numpy()
bid = (x["bid_c"] + x["bid_p"]).to_numpy()
ask = (x["ask_c"] + x["ask_p"]).to_numpy()
A = np.sqrt(asl.PERIODS_PER_YEAR)


def sh(v):
    v = np.asarray(v, float)
    return float(v.mean() / v.std(ddof=1) * A)


def tt(v):
    v = np.asarray(v, float)
    return float(np.sqrt(len(v)) * v.mean() / v.std(ddof=1))


def crossed(q):
    return np.where(q > 0, ex / ask - 1, 1 - ex / bid)


def hac_t(v, L):
    v = np.asarray(v, float)
    m = v.mean()
    e = v - m
    s2 = e @ e / len(v)
    for k in range(1, L + 1):
        s2 += 2 * (1 - k / (L + 1)) * (e[:-k] @ e[k:]) / len(v)
    return float(m / np.sqrt(s2 / len(v)))


rows = []
AS = -Rr
for name, q in (
    ("always short", -np.ones(n)),
    ("sign(s) [T0]", q0),
    ("Gaussian payoff > mid, no correction [T2b]", qg),
    ("Gaussian payoff, bin-corrected sd [cell]", qa),
):
    m = q * Rr
    c = crossed(q)
    buy = q > 0
    act = m - AS
    L = asl.newey_west_lag(n)
    rows.append(
        {
            "rule": name,
            "buy share": 100 * buy.mean(),
            "mean": m.mean(),
            "t": tt(m),
            "Sharpe mid": sh(m),
            "Sharpe crossed": sh(c),
            "E[R|buy]": Rr[buy].mean() if buy.any() else np.nan,
            "E[-R|sell]": -Rr[~buy].mean() if (~buy).any() else np.nan,
            "contrib buy": buy.mean() * (Rr[buy].mean() if buy.any() else 0),
            "contrib sell": (~buy).mean() * (-Rr[~buy].mean() if (~buy).any() else 0),
            "IR vs AS": float(act.mean() / act.std(ddof=1) * A)
            if act.std() > 0
            else np.nan,
            "t_active HAC": hac_t(act, L) if act.std() > 0 else np.nan,
            "worst": m.min(),
            "flips vs T0": int((q != q0).sum()),
        }
    )
t = pd.DataFrame(rows).set_index("rule")
pd.set_option("display.width", 250)
print(t.round(3).to_string())
# paired vs T0 with the block bootstrap
rng = np.random.default_rng(0)
idx = asl.circular_block_bootstrap_idx(rng, n, 21, 2000)
for name, q in (("T2b", qg), ("cell", qa)):
    m0, m1 = q0 * Rr, q * Rr
    c0, c1 = crossed(q0), crossed(q)
    dm = np.array([sh(m1[i]) - sh(m0[i]) for i in idx])
    dc = np.array([sh(c1[i]) - sh(c0[i]) for i in idx])
    print(
        f"{name} minus T0: dSharpe mid {sh(m1) - sh(m0):+.3f} [{np.percentile(dm, 2.5):+.2f},{np.percentile(dm, 97.5):+.2f}]  crossed {sh(c1) - sh(c0):+.3f} [{np.percentile(dc, 2.5):+.2f},{np.percentile(dc, 97.5):+.2f}]"
    )
# what the flipped days look like: the estimand on the days the cell changed
fl = qa != q0
print(
    f"\nflipped days {int(fl.sum())}: T0->cell short->long {int((fl & (qa > 0)).sum())}, long->short {int((fl & (qa < 0)).sum())}; E[R] on flipped-to-long days {Rr[fl & (qa > 0)].mean() if (fl & (qa > 0)).any() else float('nan'):+.3f}; E[R] on flipped-to-short days {Rr[fl & (qa < 0)].mean() if (fl & (qa < 0)).any() else float('nan'):+.3f}"
)
# did the decile step grow a slope? mean R by decile of ER_adj vs of s
for name, v in (("s", s), ("E_adj[R]", ER_adj)):
    dec = pd.qcut(pd.Series(v), 10, labels=False, duplicates="drop")
    print(
        f"mean R by decile of {name}:",
        np.round(pd.Series(Rr).groupby(dec).mean().to_numpy(), 2).tolist(),
    )
