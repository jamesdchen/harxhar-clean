"""Keep digging: (A) WALK-FORWARD the target (is 'target=3R optimal' overfit?), (B) more exog / tweaks.

(A) The static best target on all 60 days is an in-sample overfit. Walk it forward: each day, pick the
    target that maximised trailing daily Sharpe on PRIOR days, apply it OOS, accumulate. Compare the
    adaptive-OOS result to (i) the static overfit-best, (ii) a robust fixed 2R. If OOS << overfit, the
    3R was curve-fit.
(B) New exog/tweaks, each a lagged/causal FILTER on the maker break-retest (target=2R baseline):
    volume  = break bar volume > its trailing median (confirmation -- fake breaks are low-volume),
    trend   = break aligned with a longer intraday trend EMA (with-trend breaks run),
    vxn     = high VXN day (Nasdaq vol regime, matched to QQQ). Report win/avgR/Sharpe/P(pass).
60d QQQ-5m proxy = ILLUSTRATIVE; the point is the METHOD + which levers help, not the numbers.
Usage: python breakout_dig.py --data <qqq_5m.parquet>
"""

from __future__ import annotations

import argparse
import warnings
from datetime import time

import numpy as np
import pandas as pd
import yfinance as yf

from breakout_maker import p_pass
from jj_backtest import atr

W, COMP, MIN_BREAK = 6, 0.8, 0.1
RTH0, RTH1 = time(9, 30), time(16, 0)
TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, None]  # None = run to close


def load_ohlcv(path):
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(x).lower() for x in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    df.index = df.index.tz_convert("America/New_York")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def sim(h, low, c, fj, je, brk, entry, mid, cost, target):
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


def detect(df, vol_filt=False, trend_filt=False, day_ok=None):
    df = df.copy()
    df["a"] = atr(df)
    o, h, low, c, a, vol = (
        df[x].to_numpy(float) for x in ("open", "high", "low", "close", "a", "volume")
    )
    tod = np.array([t.time() for t in df.index])
    day = df.index.tz_localize(None).normalize()
    inr = (tod >= RTH0) & (tod < RTH1)
    chi = pd.Series(h).rolling(W).max().shift(1).to_numpy()
    clo = pd.Series(low).rolling(W).min().shift(1).to_numpy()
    vmed = pd.Series(vol).rolling(20, min_periods=5).median().shift(1).to_numpy()
    ema = pd.Series(c).ewm(span=30).mean().shift(1).to_numpy()
    ents = []
    for d in pd.unique(day):
        if day_ok is not None and not bool(day_ok.get(pd.Timestamp(d), False)):
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
            if vol_filt and not (vol[i] > vmed[i]):
                continue
            if trend_filt and not (
                (brk > 0 and c[i] > ema[i]) or (brk < 0 and c[i] < ema[i])
            ):
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
                continue
            ents.append((d, int(fj), int(je), brk, float(lvl), float(mid)))
    return ents, (o, h, low, c)


def daily_R(ents, ohlc, target, cost=0.00003):
    o, h, low, c = ohlc
    byday, r = {}, []
    for d, fj, je, brk, entry, mid in ents:
        R = sim(h, low, c, fj, je, brk, entry, mid, cost, target)
        byday[d] = byday.get(d, 0.0) + R
        r.append(R)
    return pd.Series(byday).sort_index(), np.array(r)


def report(name, dR, r):
    if len(r) < 5:
        print(f"[{name:22s}] n={len(r)} (too few)")
        return
    sh = float(dR.mean() / dR.std()) if dR.std() > 0 else float("nan")
    pp = max(p_pass(dR).values()) if dR.mean() > 0 else float("nan")
    print(
        f"[{name:22s}] n={len(r):4d} win={(r > 0).mean():.2f} avgR={r.mean():+.3f} "
        f"dayR_Sh={sh:+.3f} P(pass)={pp:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = load_ohlcv(a.data)
    ents, ohlc = detect(df)
    days = sorted({e[0] for e in ents})
    perT = {t: daily_R(ents, ohlc, t)[0] for t in TARGETS}

    print("--- (A) WALK-FORWARD the target (overfit check) ---")
    # static: each fixed target on the full sample (the in-sample/overfit view)
    for t in (2.0, 3.0):
        dR, r = daily_R(ents, ohlc, t)
        report(f"static target={t}", dR, r)
    # adaptive: pick trailing-best-Sharpe target each day, apply OOS
    warm = 25
    oos = {}
    for k in range(warm, len(days)):
        tr = days[:k]

        def tsh(t):
            v = perT[t].reindex(tr).fillna(0.0).to_numpy()
            return v.mean() / v.std() if v.std() > 0 else -9

        best = max(TARGETS, key=tsh)
        oos[days[k]] = perT[best].get(days[k], 0.0)
    oR = pd.Series(oos)
    rr = oR.to_numpy()
    sh = float(oR.mean() / oR.std()) if oR.std() > 0 else float("nan")
    pp = max(p_pass(oR).values()) if oR.mean() > 0 else float("nan")
    # overfit ceiling = best fixed target ON the same OOS window
    ceil_t = max(TARGETS, key=lambda t: perT[t].reindex(days[warm:]).fillna(0).mean())
    print(
        f"[walk-fwd adaptive OOS ] n_days={len(oR)} avgDayR={oR.mean():+.3f} Sh={sh:+.3f} P(pass)={pp:.2f}"
    )
    print(f"    (OOS overfit ceiling = static target={ceil_t} on the same window)")

    print("--- (B) MORE EXOG / TWEAKS (target=2R baseline) ---")
    report("baseline", *daily_R(detect(df)[0], ohlc, 2.0))
    report("+volume", *daily_R(detect(df, vol_filt=True)[0], ohlc, 2.0))
    report("+trend", *daily_R(detect(df, trend_filt=True)[0], ohlc, 2.0))
    report(
        "+volume+trend",
        *daily_R(detect(df, vol_filt=True, trend_filt=True)[0], ohlc, 2.0),
    )
    warnings.filterwarnings("ignore")
    vxn = yf.download(
        "^VXN", period="6mo", interval="1d", auto_adjust=True, progress=False
    )["Close"]
    vxn.index = pd.to_datetime(vxn.index).tz_localize(None).normalize()
    vs = pd.Series(vxn.to_numpy().ravel(), index=vxn.index)
    hi = vs.shift(1) > vs.rolling(20, min_periods=5).median().shift(1)
    report(
        "+high_VXN",
        *daily_R(
            detect(
                df,
                day_ok=hi.reindex(
                    pd.DatetimeIndex(pd.unique(df.index.tz_localize(None).normalize()))
                ).fillna(False),
            )[0],
            ohlc,
            2.0,
        ),
    )


if __name__ == "__main__":
    main()
