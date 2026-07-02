"""Maker (break-and-retest) execution of the breakout-continuation, + P(pass).

Taker = chase the break at the bar close, pay the spread (~1bp), take EVERY break.
Maker = post a resting limit at the BROKEN level (retest); fill only if price pulls back to it -> EARN the
spread (better entry, ~0 cost) but MISS the runaways that never retest. The fat-tail runners ARE the edge,
so whether maker wins depends on: does earning-the-spread + better-entry on the retests beat losing the
runaways? Then run the maker daily-R through eval_sim.simulate_topstep for P(pass) (Topstep 50k rules,
block-bootstrap paths, leverage sweep). Runs on QQQ 5m (60d proxy -- ILLUSTRATIVE, not a verdict).
Usage: python breakout_maker.py --data <ohlc.parquet>
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

from eval_sim.adp import simulate_topstep
from eval_sim.config import EvalConfig
from jj_backtest import atr, load_ohlc

W, COMP, MIN_BREAK = 6, 0.8, 0.1
RTH0, RTH1 = time(9, 30), time(16, 0)
CFG = EvalConfig(initial=50_000.0, profit_target=3_000.0, max_drawdown=2_000.0)


def run(df, mode, cost):
    df = df.copy()
    df["a"] = atr(df)
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    day = df.index.tz_localize(None).normalize()
    inr = (tod >= RTH0) & (tod < RTH1)
    bar_min = int(
        np.median(np.diff(df.index.values).astype("timedelta64[m]").astype(int)) or 5
    )
    w = 3 if bar_min >= 60 else W
    chi = pd.Series(h).rolling(w).max().shift(1).to_numpy()
    clo = pd.Series(low).rolling(w).min().shift(1).to_numpy()
    byday, rs, filled, missed = {}, [], 0, 0
    for d in pd.unique(day):
        idx = np.where(inr & (day == d))[0]
        if len(idx) < w + 2:
            continue
        je = idx[-1]
        for i in idx:
            if a[i] <= 0 or not np.isfinite(chi[i]):
                continue
            rng = chi[i] - clo[i]
            if not (0 < rng < COMP * a[i] * np.sqrt(w)):
                continue
            mid = 0.5 * (chi[i] + clo[i])
            if c[i] > chi[i] + MIN_BREAK * a[i]:
                brk, lvl = 1.0, chi[i]
            elif c[i] < clo[i] - MIN_BREAK * a[i]:
                brk, lvl = -1.0, clo[i]
            else:
                continue
            if mode == "taker":
                entry, fj = c[i], i  # chase at the close
            else:  # maker: limit at the broken level, fill only on a retest
                fj = next(
                    (
                        j
                        for j in range(i + 1, je + 1)
                        if (brk > 0 and low[j] <= lvl) or (brk < 0 and h[j] >= lvl)
                    ),
                    None,
                )
                if fj is None:
                    missed += 1
                    continue
                entry = lvl
                filled += 1
            risk = abs(entry - mid)
            if risk <= 0:
                continue
            R = next(
                (
                    -1.0
                    for j in range(fj + 1, je + 1)
                    if (brk > 0 and low[j] <= mid) or (brk < 0 and h[j] >= mid)
                ),
                None,
            )
            if R is None:
                R = brk * (c[je] - entry) / risk
            R -= cost * entry / risk
            byday[d] = byday.get(d, 0.0) + R
            rs.append(R)
    return pd.Series(byday).sort_index(), np.array(rs), filled, missed


def p_pass(dailyR, n_paths=20_000):
    sd = dailyR.std()
    if not (sd > 0):
        return {}
    unit = (dailyR / sd).to_numpy()
    n = len(unit)
    rng = np.random.default_rng(0)
    T, blk = 250, 10
    nb = int(np.ceil(T / blk))
    paths = np.empty((n_paths, T))
    for i in range(n_paths):
        s = rng.integers(0, n - blk, nb)
        paths[i] = np.concatenate([unit[x : x + blk] for x in s])[:T]
    out = {}
    for wlev in (0.005, 0.01, 0.02, 0.03, 0.05):
        oc, _, _ = simulate_topstep(paths, CFG, float(wlev), daily_loss_limit=1_000.0)
        out[str(wlev)] = float((oc == 1).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--maker-cost",
        type=float,
        default=0.00003,
        help="maker round-trip cost frac (entry free, stop is taker)",
    )
    ap.add_argument("--taker-cost", type=float, default=0.0001)
    a = ap.parse_args()
    df = load_ohlc(a.data)
    for mode, cost in (("taker", a.taker_cost), ("maker", a.maker_cost)):
        dR, r, filled, missed = run(df, mode, cost)
        fillrate = filled / (filled + missed) if (filled + missed) else float("nan")
        sh = float(dR.mean() / dR.std()) if dR.std() > 0 else float("nan")
        tag = f"fill={fillrate:.0%}" if mode == "maker" else "chase-all"
        print(
            f"[{mode:5s} cost={cost * 1e4:.1f}bp {tag}] n={len(r)} win={(r > 0).mean():.2f} "
            f"avgR={r.mean():+.3f} medR={np.median(r):+.1f} maxR={r.max():+.1f} | "
            f"dayR_mean={dR.mean():+.3f} dayR_Sharpe={sh:+.3f}"
        )
        if dR.mean() > 0:
            pp = p_pass(dR)
            print(
                "        P(pass) by daily-risk w: "
                + "  ".join(f"{k}:{v:.2f}" for k, v in pp.items())
            )
        else:
            print("        P(pass): N/A (net-negative daily expectancy)")


if __name__ == "__main__":
    main()
