"""Better objectification of JJ's edge: the SKILL is SELECTING real consolidation + real break-of-structure,
not a continuous forecast. Test that selection properly (vs my crude body/range hack in jj_backtest):

  consolidation = the prior W bars form a COMPRESSED range (high-low < COMP * ATR*sqrt(W)) -- a real
                  low-volatility base, the "fair value" zone; midpoint = fair price.
  break         = the current bar CLOSES beyond that consolidation's high/low (Donchian break of structure).
  reversion arm = fade the break back toward the consolidation MID (his "revert to fair value").
  continuation  = ride the break.

All causal (consolidation from bars strictly before the entry). Same exits (ATR stop + 1.5R) and the SAME
gate as jj_backtest (per-day R + purged folds + DIRECTION-SHUFFLE placebo) -- so we test whether the
selection RULE carries directional edge, or whether random direction on the same setups does as well.
On 60 free QQQ-5m sessions this can't reach significance; the point is (a) a faithful selector, (b) whether
it moves placebo_p vs the crude version, (c) turnkey for 1-min. Usage: python structure_setup.py --data <ohlc>
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from jj_backtest import WINDOWS, atr, gate, load_ohlc

W = 6  # consolidation lookback (bars)
COMP = 0.8  # compression: prior-W range must be < COMP * ATR * sqrt(W) to count as consolidation
MIN_BREAK = (
    0.1  # break beyond the range by >= MIN_BREAK*ATR (a decisive break, not a graze)
)


def collect_structure_entries(df: pd.DataFrame):
    df = df.copy()
    df["a"] = atr(df)
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    dday = df.index.date
    cons_hi = (
        pd.Series(h).rolling(W).max().shift(1).to_numpy()
    )  # consolidation range (STRICTLY prior W bars)
    cons_lo = pd.Series(low).rolling(W).min().shift(1).to_numpy()
    entries = []
    for w0, w1 in WINDOWS:
        inwin = (tod >= w0) & (tod < w1)
        for day in np.unique(dday):
            idxs = np.where(inwin & (dday == day))[0]
            if len(idxs) < 3:
                continue
            j_end = idxs[-1]
            for i in idxs:
                if a[i] <= 0 or not np.isfinite(cons_hi[i]):
                    continue
                rng = cons_hi[i] - cons_lo[i]
                if not (
                    0 < rng < COMP * a[i] * np.sqrt(W)
                ):  # require a COMPRESSED base
                    continue
                mid = 0.5 * (cons_hi[i] + cons_lo[i])
                if c[i] > cons_hi[i] + MIN_BREAK * a[i]:
                    brk = 1.0
                elif c[i] < cons_lo[i] - MIN_BREAK * a[i]:
                    brk = -1.0
                else:
                    continue
                sd = float(a[i])
                entries.append(
                    (day, "reversion", int(i), int(j_end), -brk, sd)
                )  # fade to fair value
                entries.append(
                    (day, "continuation", int(i), int(j_end), brk, sd)
                )  # ride the break
    return entries, (o, h, low, c), {"n_bars": len(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    entries, ohlc, meta = collect_structure_entries(load_ohlc(a.data))
    print(
        f"structure setups: {len(entries) // 2} breaks off compressed bases | n_bars={meta['n_bars']}"
    )
    for ph in ("continuation", "reversion"):
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
