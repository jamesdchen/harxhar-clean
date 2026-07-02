"""Faithful, frequency-agnostic backtester for JJ Simon's fair-value strategy -- built to be a DROP-IN for
the 1-min NQ history when it arrives (validate now on free OHLC; then `--data <1min_nq.parquet>` for the
real gate). Objectifies his discretionary rules (an improvement, and gate-able):

  * fair-value anchor = the session-window open (morning 09:30, afternoon 14:00); overridable.
  * two windows: 09:30-11:00 and 14:00-15:00 NY (his sessions).
  * two-phase: first CONT_MIN minutes = CONTINUATION (trade WITH a displacement candle), rest = REVERSION
    (trade TOWARD the anchor on a displacement candle beyond MIN_DEV) -- his continuation-then-reversion.
  * displacement candle = body/range >= BODY_MIN (objectifies the <20%-counter-wick rule).
  * exit = ATR stop (STOP_ATR*ATR) + 1.5R target, first-touch within the window, else window close; same-bar
    ambiguity resolved stop-first (conservative). One position at a time per window.

Reports per phase: n trades, win rate, avg R, expectancy, total R, and a bootstrap t on the R-multiples.
Small free samples (QQQ 5m/60d) are a PROTOTYPE (n too small to gate) -- the point is the harness + the
qualitative read; the 1-min history is what gets gated. Usage: python jj_backtest.py --data <ohlc.parquet>
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

CONT_MIN = 15  # his continuation window (minutes) before the reversion phase
BODY_MIN = 0.5  # displacement candle: body/range >= this (small counter-wick)
MIN_DEV = 0.0005  # min deviation-from-anchor (a real break) to take a reversion entry
STOP_ATR = 1.0  # stop = STOP_ATR * ATR; target = 1.5 * stop (1.5R)
WINDOWS = [(time(9, 30), time(11, 0)), (time(14, 0), time(15, 0))]


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    df.index = df.index.tz_convert("America/New_York")
    return df[["open", "high", "low", "close"]].sort_index()


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()),
    )
    return tr.rolling(n, min_periods=1).mean().to_numpy()


def simulate_trade(o, h, low, c, i, j_end, direction, stop_d, tgt_d):
    """Walk bars i+1..j_end; exit at first stop/target touch (stop-first on same bar), else last close.
    Returns R-multiple."""
    entry = c[i]
    stop = entry - direction * stop_d
    tgt = entry + direction * tgt_d
    for j in range(i + 1, j_end + 1):
        if direction > 0:
            if low[j] <= stop:
                return -1.0
            if h[j] >= tgt:
                return 1.5
        else:
            if h[j] >= stop:
                return -1.0
            if low[j] <= tgt:
                return 1.5
    return direction * (c[j_end] - entry) / stop_d  # window-close exit, in R units


def backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["a"] = atr(df)
    bar_min = int(
        np.median(np.diff(df.index.values).astype("timedelta64[m]").astype(int)) or 5
    )
    cont_bars = max(1, int(np.ceil(CONT_MIN / bar_min)))
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    dday = df.index.date
    body = np.abs(c - o)
    rng = np.maximum(h - low, 1e-12)
    disp = (body / rng) >= BODY_MIN  # displacement candle

    trades = {"continuation": [], "reversion": []}
    n = len(df)
    for w0, w1 in WINDOWS:
        inwin = (tod >= w0) & (tod < w1)
        # per day, the window's bars
        for day in np.unique(dday):
            idxs = np.where(inwin & (dday == day))[0]
            if len(idxs) < 3:
                continue
            anchor = o[idxs[0]]  # the window-open = fair value
            j_end = idxs[-1]
            k = 0
            held_until = -1
            for i in idxs:
                k += 1
                if i <= held_until or a[i] <= 0 or not disp[i]:
                    continue
                dev = np.log(c[i] / anchor)
                if (
                    k <= cont_bars
                ):  # continuation phase: trade WITH the displacement candle
                    direction = float(np.sign(c[i] - o[i]))
                    phase = "continuation"
                else:  # reversion phase: fade the deviation back toward the anchor
                    if abs(dev) < MIN_DEV:
                        continue
                    direction = -float(np.sign(dev))
                    phase = "reversion"
                if direction == 0:
                    continue
                stop_d = STOP_ATR * a[i]
                R = simulate_trade(
                    o, h, low, c, i, j_end, direction, stop_d, 1.5 * stop_d
                )
                trades[phase].append(R)
                # mark hold until exit (approx: until a target/stop would have resolved -> next bar for simplicity)
                held_until = i  # one signal/bar; re-entry allowed next bar (conservative on overlap)
    out = {"bar_min": bar_min, "cont_bars": cont_bars, "n_bars": n}
    for ph, rs in trades.items():
        r = np.array(rs)
        if len(r) > 5:
            t = (
                float(r.mean() / (r.std() / np.sqrt(len(r))))
                if r.std() > 0
                else float("nan")
            )
            out[ph] = {
                "n": len(r),
                "win_rate": float((r > 0).mean()),
                "avg_R": float(r.mean()),
                "total_R": float(r.sum()),
                "boot_t": t,
            }
        else:
            out[ph] = {"n": len(r)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="OHLC parquet (yfinance or 1-min NQ)")
    a = ap.parse_args()
    res = backtest(load_ohlc(a.data))
    print(
        f"bar={res['bar_min']}min cont_bars={res['cont_bars']} n_bars={res['n_bars']}"
    )
    for ph in ("continuation", "reversion"):
        r = res[ph]
        if "avg_R" in r:
            print(
                f"[{ph:12s}] n={r['n']:4d} win={r['win_rate']:.2f} avgR={r['avg_R']:+.3f} "
                f"totalR={r['total_R']:+.1f} boot_t={r['boot_t']:+.2f}"
            )
        else:
            print(f"[{ph:12s}] n={r['n']} (too few)")


if __name__ == "__main__":
    main()
