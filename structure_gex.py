"""GEX-conditioned breakout selector: does the dealer-gamma regime SELECT which breakouts continue?

Mechanism (Baltussen hedging-demand): SHORT-gamma -> dealers amplify moves -> a break off a compressed
base should CONTINUE harder; LONG-gamma -> dealers suppress -> the break should fade. So condition the
one placebo-beating signal (structure_setup: breakout-continuation) on the lagged daily GEX regime.

Data reality: our GEX ends Aug-2025 and free 5m is recent-only (no overlap), so this runs on 1h SPY (matches
our SPX GEX; ~2yr / ~500 sessions overlap). 1h is COARSE for breakout structure -- this tests the
CONDITIONING hypothesis, not JJ's 1-min execution; it's a bridge until 1-min ES over the GEX period. Same
gate as jj_backtest (per-day R + purged folds + direction-shuffle placebo). Arms: cont_all, cont_shortG,
cont_longG -- the hypothesis PASSES if short-gamma continuation beats long-gamma (and the placebo).
Usage: python structure_gex.py --data <1h_spy.parquet>
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

from gex_regime_test import lag_regime
from jj_backtest import atr, gate, load_ohlc

W = 3  # consolidation lookback in bars (~3h at 1h resolution)
COMP = 0.8  # compression: prior-W range < COMP*ATR*sqrt(W)
MIN_BREAK = 0.1  # decisive break beyond the base by >= MIN_BREAK*ATR
RTH0, RTH1 = time(9, 30), time(16, 0)


def collect(df: pd.DataFrame, regime: pd.Series):
    df = df.copy()
    df["a"] = atr(df)
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    dday = df.index.tz_localize(None).normalize()
    inr = (tod >= RTH0) & (tod < RTH1)
    chi = pd.Series(h).rolling(W).max().shift(1).to_numpy()
    clo = pd.Series(low).rolling(W).min().shift(1).to_numpy()
    entries = []
    for day in pd.unique(dday):
        reg = regime.get(pd.Timestamp(day), np.nan)
        if not np.isfinite(reg):
            continue
        tag = "shortG" if reg < 0 else "longG"
        idxs = np.where(inr & (dday == day))[0]
        if len(idxs) < W + 1:
            continue
        j_end = idxs[-1]
        for i in idxs:
            if a[i] <= 0 or not np.isfinite(chi[i]):
                continue
            rng = chi[i] - clo[i]
            if not (0 < rng < COMP * a[i] * np.sqrt(W)):
                continue
            if c[i] > chi[i] + MIN_BREAK * a[i]:
                brk = 1.0
            elif c[i] < clo[i] - MIN_BREAK * a[i]:
                brk = -1.0
            else:
                continue
            sd = float(a[i])
            entries.append(
                (day, f"cont_{tag}", int(i), int(j_end), brk, sd)
            )  # regime-split
            entries.append((day, "cont_all", int(i), int(j_end), brk, sd))  # pooled
    return entries, (o, h, low, c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = load_ohlc(a.data)
    days = pd.DatetimeIndex(pd.unique(df.index.tz_localize(None).normalize()))
    reg = lag_regime(pd.read_parquet("results/gex/gex_daily.parquet"), days)
    regime = pd.Series(reg["regime_sign"].to_numpy(), index=days)
    entries, ohlc = collect(df, regime)
    n_break = sum(1 for e in entries if e[1] == "cont_all")
    print(f"1h SPY breakouts off compressed bases (GEX-overlap): {n_break}")
    for ph in ("cont_all", "cont_shortG", "cont_longG"):
        g = gate(entries, ohlc, ph)
        if g.get("insufficient"):
            print(f"[{ph:12s}] n_days={g['n_days']} (too few)")
            continue
        print(
            f"[{ph:12s}] GATE={'PASS' if g['gate_pass'] else 'fail'} | n={g['n_trades']} days={g['n_days']} "
            f"win={g['win_rate']:.2f} avgR={g['avg_R']:+.3f} | t_dayR={g['t_dayR']:+.2f} "
            f"foldpos={g['fold_pos_frac']:.2f} placebo_p={g['placebo_p']:.3f}"
        )


if __name__ == "__main__":
    main()
