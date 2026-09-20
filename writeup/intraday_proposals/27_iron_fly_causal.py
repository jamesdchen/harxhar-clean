"""27 — causal DH short as an iron fly (short body, long wings).

Same entry clock as 26 (trailing Sharpe, prior days). Same 15:30 exit:
buy back if s>0, else cash-settle. Per-body-premium return so the wing
cost reads against the naked short (Sharpe 4.06).

Widths 25 and 50 (the lab). pick_wings at the entry stamp.

Run: python writeup/intraday_proposals/27_iron_fly_causal.py
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
OUT = HOLD / "proposals" / "27"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
CLOSE = "15:30"
WIDTHS = (25.0, 50.0)


def _sh(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * ANN)


def cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def call_put_delta(s, F, Kc, Kp):
    s = np.asarray(s, float)
    F = np.asarray(F, float)
    Kc = np.broadcast_to(np.asarray(Kc, float), s.shape)
    Kp = np.broadcast_to(np.asarray(Kp, float), s.shape)
    dc = np.zeros(s.shape)
    dp = np.zeros(s.shape)
    pos = (s > 0) & np.isfinite(s) & (F > 0)
    ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
    d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
    d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
    dc[pos] = cdf(d1c)
    dp[pos] = cdf(d1p) - 1.0
    return dc, dp


def bs_call_put(s, F, Kc, Kp):
    s, F, Kc, Kp = (np.asarray(x, float) for x in (s, F, Kc, Kp))
    call = np.maximum(F - Kc, 0.0)
    put = np.maximum(Kp - F, 0.0)
    pos = (s > 0) & np.isfinite(s) & (F > 0)
    ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
    d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
    d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
    call[pos] = FF * cdf(d1c) - kc * cdf(d1c - ss)
    put[pos] = kp * cdf(-(d1p - ss)) - FF * cdf(-d1p)
    return call, put


def pick_wings_at_stamp(live, body, width):
    """body: expiration, timestamp, K_c, K_p, entry. live: chain at those stamps."""
    c = live[live["cp"] == "C"]
    p = live[live["cp"] == "P"]
    key = ["expiration", "timestamp"]
    want_c = body[key + ["K_c"]].copy()
    want_p = body[key + ["K_p"]].copy()
    c = c.merge(want_c, on=key, how="inner")
    p = p.merge(want_p, on=key, how="inner")
    c = c[c["strike"].astype(float) >= (c["K_c"] + width)]
    p = p[p["strike"].astype(float) <= (p["K_p"] - width)]
    c["k_gap"] = c["strike"].astype(float) - c["K_c"]
    p["k_gap"] = p["K_p"] - p["strike"].astype(float)
    c_w = c.sort_values(key + ["k_gap", "strike"]).groupby(key, as_index=False).first()
    p_w = p.sort_values(key + ["k_gap", "strike"]).groupby(key, as_index=False).first()
    w = c_w.merge(p_w, on=key, suffixes=("_cw", "_pw"))
    out = body.merge(
        w[key + ["strike_cw", "strike_pw", "bid_cw", "ask_cw", "bid_pw", "ask_pw"]],
        on=key,
        how="left",
    )
    out["K_c_wing"] = out["strike_cw"].astype(float)
    out["K_p_wing"] = out["strike_pw"].astype(float)
    out["mid_c_wing"] = asl.quote_mid(out["bid_cw"], out["ask_cw"]).to_numpy()
    out["mid_p_wing"] = asl.quote_mid(out["bid_pw"], out["ask_pw"]).to_numpy()
    out["entry_wings"] = out["mid_c_wing"] + out["mid_p_wing"]
    out["credit"] = out["entry"].astype(float) - out["entry_wings"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(
        SRC_DAILY := HOLD / "proposals" / "26" / "26_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    pick = daily["pick"]
    r_naked = daily["r_causal_exit"]

    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks = sorted(pkg["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    pm = panel.set_index("t")[["rv_hat"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["date", "hhmm"])

    w15 = work.loc[work["hhmm"] == CLOSE, ["date", "rv_hat", "iv_hourly"]].copy()
    w15["s"] = w15["rv_hat"] - w15["iv_hourly"].astype(float) ** 2 * 0.5
    s15 = w15.set_index("date")["s"]

    # body rows at the causal pick
    scored = pick.dropna()
    body_rows = []
    for dt, c in scored.items():
        sl = work[(work["date"] == dt) & (work["hhmm"] == c)]
        if len(sl) != 1:
            continue
        body_rows.append(sl.iloc[0])
    body = pd.DataFrame(body_rows)
    print(f"causal entries with a pick: {len(body)}")

    chain = pd.read_parquet(sorted((HOLD / "cache").glob("chain_*.parquet"))[-1])
    ts = pd.to_datetime(body["timestamp"].unique())
    live = chain[pd.to_datetime(chain["timestamp"]).isin(ts)].copy()
    print(f"chain rows at entry stamps: {len(live):,}")

    S_grid = work.pivot_table(index="date", columns="hhmm", values="S", aggfunc="first")
    S_grid = S_grid.reindex(columns=clocks)
    IV_grid = work.pivot_table(
        index="date", columns="hhmm", values="iv_hourly", aggfunc="first"
    )
    IV_grid = IV_grid.reindex(columns=clocks)
    h_row = np.array([n_rem[c] * 0.5 for c in clocks])
    j15 = clocks.index(CLOSE)

    rows = [
        {
            "book": "naked short, exit if s>0",
            "width": np.nan,
            "n": int(r_naked.notna().sum()),
            "Sharpe": _sh(r_naked),
            "mean": float(np.nanmean(r_naked)),
            "min": float(np.nanmin(r_naked)),
            "p01": float(np.nanquantile(r_naked.dropna(), 0.01)),
        }
    ]

    for width in WIDTHS:
        fl = pick_wings_at_stamp(live, body, width)
        ok = (
            np.isfinite(fl["credit"]) & (fl["credit"] > 0) & np.isfinite(fl["K_c_wing"])
        )
        fl = fl.loc[ok].copy()
        print(f"width {width:g}: priced {len(fl)} / {len(body)}")
        dates = pd.DatetimeIndex(pd.to_datetime(fl["date"]))
        t_idx = pd.Index(clocks).get_indexer(fl["hhmm"].to_numpy())
        Sg = S_grid.loc[dates].to_numpy(float)
        IVg = IV_grid.loc[dates].to_numpy(float)
        n, m = Sg.shape
        col = np.arange(m)[None, :]
        active = col >= t_idx[:, None]
        tot = np.where((IVg > 0) & active, IVg * np.sqrt(h_row[None, :]), np.nan)
        Kc = fl["K_c"].to_numpy(float)[:, None]
        Kp = fl["K_p"].to_numpy(float)[:, None]
        Kcw = fl["K_c_wing"].to_numpy(float)[:, None]
        Kpw = fl["K_p_wing"].to_numpy(float)[:, None]
        dc, dp = call_put_delta(tot, Sg, Kc, Kp)
        dcw, dpw = call_put_delta(tot, Sg, Kcw, Kpw)
        # short body, long wings: delta_fly = -(dc+dp) + dcw + dpw
        # hedge n = -delta_fly = (dc+dp) - dcw - dpw
        n_stk = np.where(active, (dc + dp) - dcw - dpw, 0.0)
        ST = fl["S_close"].to_numpy(float)
        nxt = np.full_like(Sg, np.nan)
        nxt[:, :-1] = Sg[:, 1:]
        last = np.where(active, col, -1).max(axis=1)
        for j in range(m):
            nxt[last == j, j] = ST[last == j]
        dS = np.where(active & np.isfinite(Sg) & np.isfinite(nxt), nxt - Sg, 0.0)
        hedge = (n_stk * dS).sum(axis=1)
        pay_c = np.maximum(ST - fl["K_c"].to_numpy(float), 0.0)
        pay_p = np.maximum(fl["K_p"].to_numpy(float) - ST, 0.0)
        pay_cw = np.maximum(ST - fl["K_c_wing"].to_numpy(float), 0.0)
        pay_pw = np.maximum(fl["K_p_wing"].to_numpy(float) - ST, 0.0)
        exit_ic = (pay_c + pay_p) - (pay_cw + pay_pw)
        credit = fl["credit"].to_numpy(float)
        body_px = fl["entry"].to_numpy(float)
        opt_hold = credit - exit_ic
        r_hold = (opt_hold + hedge) / body_px

        tot15 = tot[:, j15]
        S15 = Sg[:, j15]
        bc, bp = bs_call_put(tot15, S15, fl["K_c"].to_numpy(), fl["K_p"].to_numpy())
        wc, wp = bs_call_put(
            tot15, S15, fl["K_c_wing"].to_numpy(), fl["K_p_wing"].to_numpy()
        )
        mark = (bc + bp) - (wc + wp)
        dS_f = dS.copy()
        dS_f[:, j15] = 0.0
        hedge_f = (n_stk * dS_f).sum(axis=1)
        opt_exit = credit - mark
        r_ex = (opt_exit + hedge_f) / body_px
        s_cat = s15.reindex(dates).to_numpy(float)
        flatten = np.isfinite(s_cat) & (s_cat > 0)
        r = np.where(flatten, r_ex, r_hold)
        r = pd.Series(r, index=dates)
        beyond = (ST >= fl["K_c_wing"].to_numpy()) | (ST <= fl["K_p_wing"].to_numpy())
        rows.append(
            {
                "book": f"iron fly w={width:g}, exit if s>0",
                "width": width,
                "n": int(np.isfinite(r).sum()),
                "Sharpe": _sh(r),
                "mean": float(np.nanmean(r)),
                "min": float(np.nanmin(r)),
                "p01": float(np.nanquantile(r.dropna(), 0.01)),
                "pct beyond wing": 100.0 * float(np.mean(beyond)),
                "median credit": float(np.median(credit)),
                "median wing prem": float(np.median(fl["entry_wings"])),
            }
        )
        print(
            f"w={width:g}  n={len(r)}  Sharpe {_sh(r):+.3f}  mean {float(np.nanmean(r)):+.4f}  "
            f"min {float(np.nanmin(r)):+.3f}  beyond {100 * np.mean(beyond):.1f}%"
        )

    tab = pd.DataFrame(rows)
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tab.to_csv(OUT / "27_iron_fly.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
