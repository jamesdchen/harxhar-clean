"""Two levers on the maker break-retest continuation:
  (1) BOOST the signal with exog regime filters (take only breakouts in a trend-favorable vol regime) --
      does filtering raise win rate / avgR without killing the tail?
  (2) OPTIMIZE for P(pass) with a capped target -- trade the fat-tail expectancy for a higher win rate /
      smoother equity that the tight trailing DD rewards. Quantifies the eval-vs-funded tension.

Exog we can align to the recent QQQ-5m sample: daily VIX + VIX3M (yfinance). high_vix (vol above its
trailing median) and backwardation (VIX>VIX3M) are classic TREND regimes -> breakouts should run;
low-vol/contango = range -> breakouts fade. (For the eventual NQ strategy use VXN; VIX is the market
proxy. GEX doesn't reach 2026, so it's not usable on this sample.) All exog LAGGED (causal).
Variants: maker_all, +high_vix, +backwardation, and target2R (P(pass)-optimized cap). Reports win/avgR/
daily-Sharpe/P(pass). QQQ 5m 60d = ILLUSTRATIVE. Usage: python breakout_boost.py --data <qqq_5m.parquet>
"""

from __future__ import annotations

import argparse
import warnings
from datetime import time

import numpy as np
import pandas as pd
import yfinance as yf

from breakout_maker import p_pass
from jj_backtest import atr, load_ohlc

W, COMP, MIN_BREAK = 6, 0.8, 0.1
RTH0, RTH1 = time(9, 30), time(16, 0)


def pull_vix(days: pd.DatetimeIndex) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    out = {}
    for tk, nm in (("^VIX", "vix"), ("^VIX3M", "vix3m")):
        s = yf.download(
            tk, period="6mo", interval="1d", auto_adjust=True, progress=False
        )["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out[nm] = pd.Series(s.to_numpy().ravel(), index=s.index).reindex(days).ffill()
    v = pd.DataFrame(out)
    med = v["vix"].rolling(20, min_periods=5).median().shift(1)
    v["high_vix"] = (
        v["vix"].shift(1) > med
    )  # lagged: yesterday's VIX above its trailing median
    v["backward"] = v["vix"].shift(1) > v["vix3m"].shift(
        1
    )  # VIX>VIX3M = stress/trend regime
    return v


def sim(h, low, c, fj, je, brk, entry, mid, cost, target=None) -> float:
    risk = abs(entry - mid)
    if risk <= 0:
        return 0.0
    cr = cost * entry / risk
    tgt = entry + brk * target * risk if target else None
    for j in range(fj + 1, je + 1):
        if (brk > 0 and low[j] <= mid) or (brk < 0 and h[j] >= mid):
            return -1.0 - cr
        if target is not None and (
            (brk > 0 and h[j] >= tgt) or (brk < 0 and low[j] <= tgt)
        ):
            return float(target) - cr
    return brk * (c[je] - entry) / risk - cr


def run_variant(df, vix, filt=None, target=None, cost=0.00003):
    df = df.copy()
    df["a"] = atr(df)
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    day = df.index.tz_localize(None).normalize()
    inr = (tod >= RTH0) & (tod < RTH1)
    chi = pd.Series(h).rolling(W).max().shift(1).to_numpy()
    clo = pd.Series(low).rolling(W).min().shift(1).to_numpy()
    byday, rs = {}, []
    for d in pd.unique(day):
        if filt is not None and not bool(vix[filt].get(pd.Timestamp(d), False)):
            continue
        idx = np.where(inr & (day == d))[0]
        if len(idx) < W + 2:
            continue
        je = idx[-1]
        for i in idx:
            if a[i] <= 0 or not np.isfinite(chi[i]):
                continue
            rng = chi[i] - clo[i]
            if not (0 < rng < COMP * a[i] * np.sqrt(W)):
                continue
            mid = 0.5 * (chi[i] + clo[i])
            if c[i] > chi[i] + MIN_BREAK * a[i]:
                brk, lvl = 1.0, chi[i]
            elif c[i] < clo[i] - MIN_BREAK * a[i]:
                brk, lvl = -1.0, clo[i]
            else:
                continue
            fj = next(
                (
                    j
                    for j in range(i + 1, je + 1)
                    if (brk > 0 and low[j] <= lvl) or (brk < 0 and h[j] >= lvl)
                ),
                None,
            )
            if fj is None:
                continue  # maker: no retest -> no fill
            R = sim(h, low, c, fj, je, brk, lvl, mid, cost, target)
            byday[d] = byday.get(d, 0.0) + R
            rs.append(R)
    return pd.Series(byday).sort_index(), np.array(rs)


def report(name, dR, r):
    if len(r) < 5:
        print(f"[{name:20s}] n={len(r)} (too few)")
        return
    sh = float(dR.mean() / dR.std()) if dR.std() > 0 else float("nan")
    line = (
        f"[{name:20s}] n={len(r):4d} win={(r > 0).mean():.2f} avgR={r.mean():+.3f} "
        f"maxR={r.max():+.1f} | dayR_Sh={sh:+.3f}"
    )
    if dR.mean() > 0:
        pp = p_pass(dR)
        best = max(pp.values()) if pp else float("nan")
        line += f" | P(pass) peak={best:.2f}"
    else:
        line += " | P(pass) N/A (net-neg)"
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = load_ohlc(a.data)
    days = pd.DatetimeIndex(pd.unique(df.index.tz_localize(None).normalize()))
    vix = pull_vix(days)
    print("--- (1) BOOST: exog vol-regime filters ---")
    for name, filt in (
        ("maker_all", None),
        ("+high_vix", "high_vix"),
        ("+backwardation", "backward"),
    ):
        dR, r = run_variant(df, vix, filt=filt)
        report(name, dR, r)
    print("--- (2) OPTIMIZE for P(pass): capped target (trades tail for win rate) ---")
    for tgt in (None, 3.0, 2.0, 1.0):
        dR, r = run_variant(df, vix, target=tgt)
        report(f"target={tgt}", dR, r)


if __name__ == "__main__":
    main()
