"""Tier-1 amplification: condition breakouts on the LIVE, strike-resolved net dealer gamma evaluated at the
breakout's ACTUAL intraday price -- not the stale daily EOD sign.

Mechanism sharpening: the forced-hedging flow depends on where spot is in the gamma PROFILE right now. A
breakout moving through a region of dealer short-gamma forces chasing (continuation); the stale daily sign
misses this. For each breakout we take the intraday spot, map SPY->SPX (daily ratio), and evaluate
G(S) = sum_i sign_i * gammaBS(S; K_i, sigma_i, tau_i) * OI_i from the PRIOR session's chain (causal). Arms:
live_shortG (G(S)<0 at the break), live_longG, live_deepshortG (bottom-third G = strongest short-gamma
force), vs eod_shortG/longG (the stale daily-sign baseline, structure_gex). Same gate. The amplification
PASSES if the live/deep arms beat the stale EOD arms (higher avgR / better placebo). Runs on 1h SPY over the
GEX-overlap (~500 sessions); drops into 1-min ES. Usage: python structure_gex_live.py --data <1h_spy.parquet>
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

from gex import bs_gamma
from gex_regime_test import lag_regime
from jj_backtest import atr, gate, load_ohlc

W, COMP, MIN_BREAK = 3, 0.8, 0.1
RTH0, RTH1 = time(9, 30), time(16, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = load_ohlc(a.data)
    df["a"] = atr(df)
    tod = np.array([t.time() for t in df.index])
    day = df.index.tz_localize(None).normalize()
    o, h, low, c, av = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    inr = (tod >= RTH0) & (tod < RTH1)

    # SPY->SPX daily ratio (to place the intraday spot on the SPX strike grid)
    spy_daily = pd.Series(c, index=pd.DatetimeIndex(day)).groupby(level=0).last()
    spx = pd.read_parquet("data/optionm_spx_spot.parquet")
    spx_daily = pd.Series(spx["close"].to_numpy(), index=pd.to_datetime(spx["date"]))
    ratio = spx_daily.reindex(spy_daily.index).ffill() / spy_daily

    # prior-session chain per date: (strike, iv, tau, w=sign*OI)
    start = pd.Timestamp(spy_daily.index.min())
    ch = pd.read_parquet(
        "data/optionm_spx_chain.parquet", filters=[("date", ">=", start)]
    )
    ch["date"] = pd.to_datetime(ch["date"])
    ch["tau"] = (pd.to_datetime(ch["exdate"]) - ch["date"]).dt.days / 365.25
    sgn = np.where(ch["cp_flag"].astype(str).str.upper().str[0] == "C", 1.0, -1.0)
    ch["w"] = sgn * ch["open_interest"].fillna(0).to_numpy()
    chain = {
        d: (
            g["strike"].to_numpy(float),
            g["impl_volatility"].to_numpy(float),
            g["tau"].to_numpy(float),
            g["w"].to_numpy(float),
        )
        for d, g in ch.groupby("date")
    }
    cdates = np.array(sorted(chain.keys()))

    def net_gamma(S: float, d: pd.Timestamp) -> float:
        j = np.searchsorted(cdates, d, side="left") - 1  # PRIOR session's chain
        if j < 0:
            return float("nan")
        K, sig, tau, w = chain[cdates[j]]
        return float((w * bs_gamma(S, K, sig, tau)).sum())

    reg = lag_regime(
        pd.read_parquet("results/gex/gex_daily.parquet"),
        pd.DatetimeIndex(pd.unique(day)),
    )
    eod_sign = pd.Series(
        reg["regime_sign"].to_numpy(), index=pd.DatetimeIndex(pd.unique(day))
    )

    chi = pd.Series(h).rolling(W).max().shift(1).to_numpy()
    clo = pd.Series(low).rolling(W).min().shift(1).to_numpy()
    raw = []
    for d in pd.unique(day):
        idxs = np.where(inr & (day == d))[0]
        rt = ratio.get(pd.Timestamp(d), np.nan)
        if len(idxs) < W + 1 or not np.isfinite(rt):
            continue
        j_end = idxs[-1]
        for i in idxs:
            if av[i] <= 0 or not np.isfinite(chi[i]):
                continue
            rng = chi[i] - clo[i]
            if not (0 < rng < COMP * av[i] * np.sqrt(W)):
                continue
            if c[i] > chi[i] + MIN_BREAK * av[i]:
                brk = 1.0
            elif c[i] < clo[i] - MIN_BREAK * av[i]:
                brk = -1.0
            else:
                continue
            G = net_gamma(c[i] * rt, pd.Timestamp(d))
            raw.append((d, int(i), int(j_end), brk, float(av[i]), G))

    Gs = np.array([r[5] for r in raw if np.isfinite(r[5])])
    q33 = float(np.quantile(Gs, 1 / 3))  # bottom third = deepest short-gamma
    entries = []
    for d, i, j_end, brk, sd, G in raw:
        if not np.isfinite(G):
            continue
        es = eod_sign.get(pd.Timestamp(d), np.nan)
        entries.append((d, "cont_all", i, j_end, brk, sd))
        entries.append((d, "live_shortG" if G < 0 else "live_longG", i, j_end, brk, sd))
        if G < q33:
            entries.append((d, "live_deepshortG", i, j_end, brk, sd))
        if np.isfinite(es):
            entries.append(
                (d, "eod_shortG" if es < 0 else "eod_longG", i, j_end, brk, sd)
            )
    ohlc = (o, h, low, c)
    print(
        f"1h SPY breakouts w/ live SPX gamma: {len(Gs)} | short-gamma-at-break: {(Gs < 0).mean():.0%}"
    )
    for ph in (
        "cont_all",
        "eod_shortG",
        "live_shortG",
        "live_deepshortG",
        "eod_longG",
        "live_longG",
    ):
        g = gate(entries, ohlc, ph)
        if g.get("insufficient"):
            print(f"[{ph:16s}] n_days={g['n_days']} (too few)")
            continue
        print(
            f"[{ph:16s}] GATE={'PASS' if g['gate_pass'] else 'fail'} | n={g['n_trades']:4d} "
            f"days={g['n_days']:3d} win={g['win_rate']:.2f} avgR={g['avg_R']:+.3f} | "
            f"t_dayR={g['t_dayR']:+.2f} foldpos={g['fold_pos_frac']:.2f} placebo_p={g['placebo_p']:.3f}"
        )


if __name__ == "__main__":
    main()
