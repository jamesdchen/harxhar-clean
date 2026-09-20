"""26 — one DH short to close, causal entry clock, flatten if s_15:30>0,
optional extra 15:30 sign(s) on a new K.

Entry clock is NOT 10:00 by fiat. Each day, pick the daytime clock
(10:00-15:00) with the best trailing expanding Sharpe of always-short
DH t→T on prior days (min 63, shift 1). Warmup: flat.

Flatten: at 15:30, if s = rv_hat - IV^2/2 > 0, buy back the short at
the Black-76 mark (same K, S and IV at 15:30) and drop the last stock
step. Else cash-settle.

The extra 15:30 sign(s) is a second lot (new nearest-OTM).

Run: python writeup/intraday_proposals/26_causal_entry_flatten.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "26"
CLOSE = "15:30"
DAYTIME = None  # set after clocks
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
MIN_DAYS = 63


def _sh(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * ANN)


def cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def pkg_price(s, F, Kc, Kp):
    s, F, Kc, Kp = map(lambda x: np.asarray(x, float), (s, F, Kc, Kp))
    out = np.full(np.broadcast_shapes(s.shape, F.shape), np.nan)
    pos = (s > 0) & np.isfinite(s) & (F > 0)
    if pos.any():
        ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
        d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
        d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
        out[pos] = (
            FF * cdf(d1c) - kc * cdf(d1c - ss) + kp * cdf(-(d1p - ss)) - FF * cdf(-d1p)
        )
    z = (s <= 0) & np.isfinite(F)
    out[z] = np.maximum(F[z] - Kc[z], 0) + np.maximum(Kp[z] - F[z], 0)
    return out


def pkg_delta(s, F, Kc, Kp):
    s = np.asarray(s, float)
    F = np.asarray(F, float)
    shape = np.broadcast_shapes(s.shape, F.shape)
    Kc = np.broadcast_to(np.asarray(Kc, float), shape)
    Kp = np.broadcast_to(np.asarray(Kp, float), shape)
    F = np.broadcast_to(F, shape)
    s = np.broadcast_to(s, shape)
    out = np.zeros(shape)
    pos = (s > 0) & np.isfinite(s) & (F > 0)
    ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
    d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
    d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
    out[pos] = cdf(d1c) + cdf(d1p) - 1.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks = sorted(pkg["hhmm"].unique())
    daytime = [c for c in clocks if c != CLOSE]
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    pkg["h_rem"] = pkg["hhmm"].map(n_rem).astype(float) * 0.5
    pm = panel.set_index("t")[["rv_hat"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    work = work.sort_values(["date", "hhmm"]).reset_index(drop=True)

    w15 = work.loc[work["hhmm"] == CLOSE, ["date", "rv_hat", "iv_hourly", "R"]].copy()
    w15["s"] = w15["rv_hat"] - w15["iv_hourly"].astype(float) ** 2 * 0.5
    s15 = w15.set_index("date")["s"]
    r15_long = w15.set_index("date")["R"]

    S_grid = work.pivot_table(index="date", columns="hhmm", values="S", aggfunc="first")
    S_grid = S_grid.reindex(columns=clocks)
    IV_grid = work.pivot_table(
        index="date", columns="hhmm", values="iv_hourly", aggfunc="first"
    ).reindex(columns=clocks)
    h_row = np.array([n_rem[c] * 0.5 for c in clocks])
    j15 = clocks.index(CLOSE)

    dates_all = pd.DatetimeIndex(sorted(work["date"].unique()))
    n_d, m = len(dates_all), len(clocks)
    Sg = S_grid.reindex(dates_all).to_numpy(float)
    IVg = IV_grid.reindex(dates_all).to_numpy(float)
    # S_close / K / entry / exit per (date, clock) from work
    meta = work.set_index(["date", "hhmm"])

    r_hold = pd.DataFrame(np.nan, index=dates_all, columns=clocks)
    r_stop = pd.DataFrame(np.nan, index=dates_all, columns=clocks)
    r_f_by_clock = {}
    tot15 = np.where(IVg[:, j15] > 0, IVg[:, j15] * np.sqrt(h_row[j15]), np.nan)

    for j, c in enumerate(clocks):
        if c == CLOSE:
            continue
        sl = meta.xs(c, level="hhmm").reindex(dates_all)
        Kc = sl["K_c"].to_numpy(float)
        Kp = sl["K_p"].to_numpy(float)
        entry = sl["entry"].to_numpy(float)
        ex = sl["exit"].to_numpy(float)
        ST = sl["S_close"].to_numpy(float)
        tot = np.where(IVg > 0, IVg * np.sqrt(h_row[None, :]), np.nan)
        tot[:, :j] = np.nan
        dlt = pkg_delta(tot, Sg, Kc[:, None], Kp[:, None])
        nxt = np.full_like(Sg, np.nan)
        nxt[:, :-1] = Sg[:, 1:]
        nxt[:, -1] = ST
        dS = np.where(np.isfinite(Sg) & np.isfinite(nxt), nxt - Sg, 0.0)
        dS[:, :j] = 0.0
        hedge = (dlt * dS).sum(axis=1)
        r_h = (-(ex - entry) + hedge) / entry
        mark = pkg_price(tot15, Sg[:, j15], Kc, Kp)
        dS_f = dS.copy()
        dS_f[:, j15] = 0.0
        hedge_f = (dlt * dS_f).sum(axis=1)
        r_f = (-(mark - entry) + hedge_f) / entry
        s_cat = s15.reindex(dates_all).to_numpy(float)
        flatten = np.isfinite(s_cat) & (s_cat > 0)
        r_s = np.where(flatten, r_f, r_h)
        r_hold[c] = r_h
        r_stop[c] = r_s
        r_f_by_clock[c] = pd.Series(r_f, index=dates_all)

    q15 = np.sign(s15.reindex(dates_all).to_numpy(float))
    q15 = np.where(q15 == 0, -1.0, q15)
    r15 = q15 * r15_long.reindex(dates_all).to_numpy(float)

    # in-sample clock table (labeled)
    rows = []
    for c in daytime:
        rows.append(
            {
                "clock": c,
                "n": int(np.isfinite(r_hold[c]).sum()),
                "Sharpe hold": _sh(r_hold[c]),
                "Sharpe flatten": _sh(r_stop[c]),
                "Sharpe flatten+15:30": _sh(r_stop[c] + r15),
                "mean flatten": float(np.nanmean(r_stop[c])),
            }
        )
    clk = pd.DataFrame(rows).set_index("clock")
    print("IN-SAMPLE by entry clock (always-short DH t→T)")
    print(clk.to_string(float_format=lambda x: f"{x:+.3f}"))
    clk.to_csv(OUT / "26_clock_is.csv")

    # causal: trailing Sharpe of HOLD-THROUGH (no flatten in the picker)
    trail = pd.DataFrame(index=dates_all, columns=daytime, dtype=float)
    for c in daytime:
        s = r_hold[c]
        mu = s.expanding(min_periods=MIN_DAYS).mean().shift(1)
        sd = s.expanding(min_periods=MIN_DAYS).std(ddof=1).shift(1)
        trail[c] = mu / sd * ANN
    pick = trail.idxmax(axis=1)
    pick[trail.isna().all(axis=1)] = pd.NA
    r_c_hold = pd.Series(np.nan, index=dates_all)
    r_c_stop = pd.Series(np.nan, index=dates_all)
    r_c_mark = pd.Series(np.nan, index=dates_all)
    for c in daytime:
        m = pick == c
        r_c_hold[m] = r_hold.loc[m, c]
        r_c_stop[m] = r_stop.loc[m, c]
        r_c_mark[m] = r_f_by_clock[c].loc[m]
    print("\ncausal pick counts (trailing Sharpe of hold-through DH AS):")
    print(pick.value_counts(dropna=False).to_string())

    def row(name, s):
        v = np.asarray(s, float)
        v = v[np.isfinite(v)]
        return {
            "book": name,
            "Sharpe": _sh(v),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "p01": float(np.quantile(v, 0.01)),
            "min": float(v.min()),
            "n": int(len(v)),
        }

    tab = pd.DataFrame(
        [
            row("IS 10:00 flatten", r_stop["10:00"]),
            row("IS 10:00 flatten + 15:30 sign(s)", r_stop["10:00"] + r15),
            row("causal clock, hold through", r_c_hold),
            row("causal clock, flatten if s>0", r_c_stop),
            row("causal flatten + 15:30 sign(s)", r_c_stop + r15),
            row("15:30 sign(s) alone", r15),
        ]
    )
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    tab.to_csv(OUT / "26_rules.csv", index=False)
    daily = pd.DataFrame(
        {
            "r_causal_hold": r_c_hold,
            "r_causal_exit": r_c_stop,
            "r_causal_mark": r_c_mark,
            "r_15_sign": r15,
            "r_causal_exit_plus_15": r_c_stop + r15,
            "pick": pick,
        }
    )
    daily.to_csv(OUT / "26_daily.csv")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
