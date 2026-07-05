"""Stage-2 overnight-expression test: modeled close->next-open straddle P&L.

The close->close straddle test was NULL because t+1's RTH dilutes the AH/overnight window the
signal speaks to. This models the UNDILUTED trade: enter the real quoted ATM straddle at close t
(OptionMetrics mids/IVs), exit at next open, where the exit value is Black-Scholes repriced at
  S_open  = S_close * exp(gap)          (gap = realized night sumret from the panel's 30-min bars)
  tau'    = tau - 17.5h                  (calendar decay 16:00 -> 9:30)
  IV'     = IV + beta * dVIX/100         (vega leg; realized overnight dVIX; beta in {1.0, 0.5})
No OM greek unit conventions are trusted -- entry is a real price, exit is a full reprice.
Costs: entry crosses frac*spread (real quoted spread); exit crosses frac*spread*OPEN_MULT
(open-auction spreads wider than close). Position: side=-sign(signal), w=min(|s|,3).

Both legs of this trade were independently validated (HAR-resid overnight RV: NW-t +4.47;
overnight dVIX: NW-t +6.59); this combines them through real entry prices.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.data.loading as _loading

_real_listdir = os.listdir
_loading.os.listdir = lambda p: [  # type: ignore[assignment]
    f for f in _real_listdir(p) if not f.startswith("optionm_")
]

DTE_LO, DTE_HI = 1, 5
CAP = 3.0
COMMISSION = 0.65  # $/contract/side
TIERS = {"optimistic": 0.0, "base": 0.25, "conservative": 0.5}
OPEN_MULT = 1.5  # open-auction spread multiple of close spread
OVERNIGHT_YEARS = 17.5 / 24.0 / 365.0
BETAS = {"beta1.0": 1.0, "beta0.5": 0.5}


def bs_value(S, K, tau, iv, cp):
    tau = max(tau, 1e-6)
    if iv <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    st = iv * np.sqrt(tau)
    d1 = (np.log(S / K)) / st + 0.5 * st
    d2 = d1 - st
    if cp == "C":
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def night_quantities():
    """Per anchor-date: realized night gap return (16:30->10:00) and overnight dVIX."""
    df = _loading.load_raw_data("data", allow_missing=True)
    t = df["t"]
    hour = t.dt.hour.to_numpy()
    anchor = (t - pd.Timedelta(hours=16, minutes=1)).dt.normalize()
    night = (hour >= 17) | (hour <= 10)
    gap = (
        pd.DataFrame({"a": anchor[night], "r": df.loc[night, "sumret"].to_numpy(float)})
        .groupby("a")["r"]
        .sum()
        .rename("gap")
    )
    v = pd.read_parquet("data/vix_and_voldemand.parquet")[
        ["endbartime", "vix"]
    ].dropna()
    v["t"] = pd.to_datetime(v["endbartime"])
    v["date"] = v["t"].dt.normalize()
    tod = v["t"].dt.time
    day = v[
        (tod >= pd.Timestamp("1900-01-01 10:00").time())
        & (tod <= pd.Timestamp("1900-01-01 16:00").time())
    ]
    close = day.groupby("date")["vix"].last()
    openv = day.groupby("date")["vix"].first()
    dvix = (openv.shift(-1) - close).rename("dvix")
    return pd.concat([gap, dvix], axis=1)


def main(chain_path: str, secid: int, out_tag: str) -> None:
    chain = pd.read_parquet(chain_path)
    chain["date"] = pd.to_datetime(chain["date"])
    chain["exdate"] = pd.to_datetime(chain["exdate"])
    chain["strike"] = chain["strike_price"] / 1000.0
    chain["cp_flag"] = chain["cp_flag"].str.upper().str[0]
    chain["mid"] = (chain["best_bid"] + chain["best_offer"]) / 2.0
    chain["spread"] = chain["best_offer"] - chain["best_bid"]
    chain["dte"] = (chain["exdate"] - chain["date"]).dt.days
    chain = chain[
        chain["dte"].between(DTE_LO, DTE_HI)
        & (chain["best_bid"] > 0)
        & (chain["impl_volatility"] > 0)
    ]

    spot = pd.read_csv("data/om_friction/nci0id5guwsw4w7t.csv.gz")
    spot = spot[spot["secid"] == secid]
    spot["date"] = pd.to_datetime(spot["date"])
    spot = spot.set_index("date")["close"]

    sig = pd.read_parquet("eval_sim/data/cumrv_signal_dated.parquet")
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.set_index("date")["signal_1600"]

    nq = night_quantities()

    by_day = dict(tuple(chain.groupby("date")))
    dates = sorted(
        set(by_day) & set(sig.dropna().index) & set(spot.index) & set(nq.dropna().index)
    )
    rows = []
    for d in dates:
        s = float(sig.loc[d])
        if not np.isfinite(s) or s == 0.0:
            continue
        S0 = float(spot.loc[d])
        day = by_day[d]
        exd = day.loc[day["dte"].idxmin(), "exdate"]
        ch = day[day["exdate"] == exd]
        both = ch.groupby("strike")["cp_flag"].nunique()
        ks = both[both == 2].index.to_numpy()
        if ks.size == 0:
            continue
        K = ks[np.argmin(np.abs(ks - S0))]
        legs = ch[ch["strike"] == K]
        gap, dvix = float(nq.loc[d, "gap"]), float(nq.loc[d, "dvix"])
        S1 = S0 * np.exp(gap)
        side = -np.sign(s)
        w = min(abs(s), CAP)
        entry_mid = float(legs["mid"].sum())
        sp = float(legs["spread"].sum())
        tau0 = float(legs["dte"].iloc[0]) / 365.0
        for bname, beta in BETAS.items():
            v1 = sum(
                bs_value(
                    S1,
                    K,
                    tau0 - OVERNIGHT_YEARS,
                    max(float(r.impl_volatility) + beta * dvix / 100.0, 0.01),
                    r.cp_flag,
                )
                for r in legs.itertuples()
            )
            for tier, frac in TIERS.items():
                entry = entry_mid + side * frac * sp
                exitp = v1 - side * frac * sp * OPEN_MULT
                pnl = side * w * (exitp - entry) * 100.0 - w * 4 * COMMISSION
                rows.append(
                    {"date": d, "beta": bname, "tier": tier, "pnl": pnl, "side": side}
                )

    df = pd.DataFrame(rows)
    print(f"[{out_tag}] {df['date'].nunique():,} nights")
    for bname in BETAS:
        for tier in TIERS:
            p = df[(df["beta"] == bname) & (df["tier"] == tier)].set_index("date")[
                "pnl"
            ]
            sh = p.mean() / p.std()
            print(
                f"  {bname} {tier:>12}: mean=${p.mean():+8.2f}  Sharpe/day={sh:+.3f}  ann={sh * np.sqrt(252):+.2f}"
            )
    df.to_parquet(
        f"eval_sim/data/overnight_model_{out_tag}_perday.parquet", index=False
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
