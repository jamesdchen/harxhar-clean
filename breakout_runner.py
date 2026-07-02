"""Rebuild the breakout test with REAL trade mechanics -- the fixed-1.5R/window-close exit was wrong.

Breakout-continuation's edge is POSITIVE SKEW: most breaks fail small, a few RUN for hours. Capping winners
at 1.5R and cutting them at the window boundary destroys exactly the fat right tail that is the edge, and an
ATR stop is not the STRUCTURAL invalidation a real trader uses. So:
  entry = close of the break bar (broke beyond a compressed consolidation base),
  stop  = the consolidation MIDPOINT (structural invalidation; risk R = |entry - mid|),
  exit  = RUN to the RTH session close (16:00) -- winners uncapped, losers cut at the structural stop (-1R).
R per trade = direction*(close_at_session_end - entry)/R, or -1 if the structural stop is hit first. The mean
R now reflects the true skewed distribution. Same gate (per-day R + purged folds + direction-shuffle placebo).
Usage: python breakout_runner.py --data <ohlc.parquet>
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

from jj_backtest import atr, load_ohlc

W, COMP, MIN_BREAK = 6, 0.8, 0.1
RTH0, RTH1 = time(9, 30), time(16, 0)


def simulate_runner(h, low, c, i, j_end, direction, stop) -> float:
    entry = c[i]
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    for j in range(i + 1, j_end + 1):
        if (direction > 0 and low[j] <= stop) or (direction < 0 and h[j] >= stop):
            return -1.0  # structural stop hit
    return float(
        direction * (c[j_end] - entry) / risk
    )  # runner: held to the session close


def collect(df: pd.DataFrame):
    df = df.copy()
    df["a"] = atr(df)
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    day = df.index.tz_localize(None).normalize()
    inr = (tod >= RTH0) & (tod < RTH1)
    bar_min = int(
        np.median(np.diff(df.index.values).astype("timedelta64[m]").astype(int)) or 5
    )
    w = (
        3 if bar_min >= 60 else W
    )  # coarse bars (1h) need a shorter base or there are no setups
    chi = pd.Series(h).rolling(w).max().shift(1).to_numpy()
    clo = pd.Series(low).rolling(w).min().shift(1).to_numpy()
    entries = []
    for d in pd.unique(day):
        idxs = np.where(inr & (day == d))[0]
        if len(idxs) < w + 2:
            continue
        j_end = idxs[-1]  # RTH session close -- winners run to here
        for i in idxs:
            if a[i] <= 0 or not np.isfinite(chi[i]):
                continue
            rng = chi[i] - clo[i]
            if not (0 < rng < COMP * a[i] * np.sqrt(w)):
                continue
            mid = 0.5 * (chi[i] + clo[i])  # structural stop level
            if c[i] > chi[i] + MIN_BREAK * a[i]:
                brk = 1.0
            elif c[i] < clo[i] - MIN_BREAK * a[i]:
                brk = -1.0
            else:
                continue
            entries.append((d, "breakout", int(i), int(j_end), brk, float(mid)))
    return entries, (o, h, low, c)


def gate_runner(entries, ohlc, phase, n_placebo=300, seed=0) -> dict:
    """Gate using the runner P&L (structural stop + run-to-close), aggregated per day."""
    o, h, low, c = ohlc
    from jj_backtest import purged_walk_forward  # reuse the fold splitter

    def day_R(directions=None):
        byday, r = {}, []
        for k, (d, ph, i, j_end, brk, stop) in enumerate(entries):
            if ph != phase:
                continue
            dd = brk if directions is None else directions[k]
            R = simulate_runner(h, low, c, i, j_end, dd, stop)
            byday[d] = byday.get(d, 0.0) + R
            r.append(R)
        return pd.Series(byday).sort_index(), np.array(r)

    dR, r = day_R()
    if len(dR) < 20:
        return {"phase": phase, "n_days": int(len(dR)), "insufficient": True}
    v = dR.to_numpy()
    n = len(v)
    rng = np.random.default_rng(seed)
    se = float(np.array([v[rng.integers(0, n, n)].mean() for _ in range(2000)]).std())
    mean = float(v.mean())
    folds = purged_walk_forward(n, n_folds=5, embargo=0.01)
    foldpos = (
        float(np.mean([v[te].mean() > 0 for _, te in folds])) if folds else float("nan")
    )
    idxp = [k for k, e in enumerate(entries) if e[1] == phase]
    plac = np.empty(n_placebo)
    for b in range(n_placebo):
        dirs = [0.0] * len(entries)
        for k in idxp:
            dirs[k] = float(rng.choice([-1.0, 1.0]))
        plac[b] = day_R(dirs)[0].mean()
    p = float((plac >= mean).mean())
    return {
        "phase": phase,
        "n": int(len(r)),
        "n_days": n,
        "win": float((r > 0).mean()),
        "avgR": float(r.mean()),
        "medR": float(np.median(r)),
        "maxR": float(r.max()),
        "mean_dayR": mean,
        "t": mean / se if se > 0 else float("nan"),
        "foldpos": foldpos,
        "placebo_p": p,
        "pass": bool(mean - 2 * se > 0 and foldpos >= 0.7 and p < 0.05),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--gex",
        action="store_true",
        help="split by lagged GEX regime (needs GEX overlap)",
    )
    a = ap.parse_args()
    df = load_ohlc(a.data)
    entries, ohlc = collect(df)
    phases = ["breakout"]
    if a.gex:
        from gex_regime_test import lag_regime

        days = pd.DatetimeIndex(pd.unique(df.index.tz_localize(None).normalize()))
        reg = lag_regime(pd.read_parquet("results/gex/gex_daily.parquet"), days)
        rs = pd.Series(reg["regime_sign"].to_numpy(), index=days)
        tagged = []
        for d, _ph, i, j, brk, stop in entries:
            s = rs.get(pd.Timestamp(d), np.nan)
            if np.isfinite(s):
                tagged.append(
                    (d, "bo_shortG" if s < 0 else "bo_longG", i, j, brk, stop)
                )
        entries = entries + tagged
        phases = ["breakout", "bo_shortG", "bo_longG"]
    for ph in phases:
        g = gate_runner(entries, ohlc, ph)
        if g.get("insufficient"):
            print(f"[{ph:10s}] n_days={g['n_days']} (too few)")
            continue
        print(
            f"[{ph:10s}] GATE={'PASS' if g['pass'] else 'fail'} | n={g['n']:4d} days={g['n_days']:3d} "
            f"win={g['win']:.2f} avgR={g['avgR']:+.3f} medR={g['medR']:+.1f} maxR={g['maxR']:+.1f} | "
            f"mean_dayR={g['mean_dayR']:+.3f} t={g['t']:+.2f} foldpos={g['foldpos']:.2f} placebo_p={g['placebo_p']:.3f}"
        )


if __name__ == "__main__":
    main()
