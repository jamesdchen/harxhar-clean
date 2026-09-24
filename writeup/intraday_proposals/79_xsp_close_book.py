"""Study 79 -- the XSP 0DTE book at 15:30 (Databento OPRA cbbo-1m) behind the card.

The card prices the long leg as ``ask <= P*`` on XSP, having only ever seen
SPXW quotes.  From the pulled ``data/archive/xsp_opra/`` files, per session
day: the nearest-OTM XSP pair on the LISTED grid at the BBO standing at
15:30:00, its bid / ask / mid / touch sizes, the relative spread, the
Black-76 package variance implied by the mid and by the ask (the deck's
``iv_var`` construction), the 16:00 book, and the settlement payoff at the
SPX close (XSP settles to SPX / 10).  Joined to the deck
(``daily_sub_live_ridge``) on the 273 overlapping days:

* parity: XSP mid-implied variance / SPXW iv_var (same day, own strikes);
* the long leg re-priced: the deck's long days (rv_hat > iv_var) held to
  settlement at the XSP ASK, vs the deck's R at the SPXW mid; and the
  card's own rule -- pass iff the XSP ask <= P*(rv_hat) -- with its R;
* depth: touch size at the ask vs the straddles the budget implies;
* the strike grid actually listed (1 or 0.5 points) by day.

Gate first: the deck's own iv_var is re-derived from its SPXW mid through
``live.ibkr.pricing.invert_total_vol`` before any XSP number is read --
measured 2026-09-23: within 2.8 % on the last 200 deck days (the deck's
inversion carries the vendor's hours-to-expiration and forward conventions
this spot-based one does not), so XSP / SPXW parity below is read to that
tolerance, not finer.

    python writeup/intraday_proposals/79_xsp_close_book.py [--capital 70000]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.ibkr.pricing import invert_total_vol, package_price  # noqa: E402

ET = "America/New_York"
XSP_SCALE = 0.1
OPRA_DIR = REPO / "data" / "archive" / "xsp_opra"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
SPOT = REPO / "data" / "spxw_spot.parquet"
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "79"
DECISION = "15:30:00"
SETTLE_BOOK = "16:00:00"
LONG_FRACTION = 0.033
MONTH_END_FRACTION = 0.15
MULT = 100.0


def load_quotes(path: Path) -> pd.DataFrame:
    q = pd.read_parquet(path)
    q["t"] = (
        pd.to_datetime(q["ts_recv"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    )
    sym = q["symbol"].astype(str)
    q["strike"] = sym.str.slice(13).astype(float) / 1000.0
    q["cp"] = sym.str.slice(12, 13)
    return q


def book_at(q: pd.DataFrame, day: pd.Timestamp, hhmmss: str) -> pd.DataFrame:
    t = pd.Timestamp(f"{day.date()} {hhmmss}")
    b = q[q["t"] == t]
    if b.empty:  # the last sample at or before the stamp
        b = q[q["t"] <= t]
        b = b[b["t"] == b["t"].max()]
    return b.set_index(["strike", "cp"]).sort_index()


def nearest_otm(strikes: np.ndarray, spot: float) -> tuple[float, float]:
    above = strikes[strikes >= spot]
    below = strikes[strikes <= spot]
    if len(above) == 0 or len(below) == 0:
        return float("nan"), float("nan")
    return float(above.min()), float(below.max())


def quote(b: pd.DataFrame, k: float, cp: str) -> tuple[float, float, float, float]:
    """bid, ask, bid size, ask size of one contract; NaN when absent or unquoted."""
    try:
        r = b.loc[(k, cp)]
    except KeyError:
        return (float("nan"),) * 4
    if isinstance(r, pd.DataFrame):
        r = r.iloc[-1]
    bid, ask = float(r["bid_px_00"]), float(r["ask_px_00"])
    if not (np.isfinite(bid) and np.isfinite(ask)) or ask <= 0:
        return (float("nan"),) * 4
    return bid, ask, float(r["bid_sz_00"]), float(r["ask_sz_00"])


def spot_table() -> pd.Series:
    """SPX at 15:30 and 16:00 by day from the chain's spot file (naive ET)."""
    s = pd.read_parquet(SPOT)
    t = pd.to_datetime(s["timestamp"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    s = s.assign(t=t, day=t.dt.normalize(), clk=t.dt.strftime("%H:%M"))
    return s.set_index(["day", "clk"])["spot"].astype(float)


def one_day(
    day: pd.Timestamp,
    q: pd.DataFrame,
    defs: pd.DataFrame,
    spot_1530: float,
    spot_close: float,
) -> dict[str, float]:
    strikes = np.sort(defs["strike_price"].astype(float).unique())
    xs = spot_1530 * XSP_SCALE
    kc, kp = nearest_otm(strikes, xs)
    near = strikes[(strikes > xs * 0.99) & (strikes < xs * 1.01)]
    step = float(np.min(np.diff(near))) if len(near) > 2 else float("nan")
    b = book_at(q, day, DECISION)
    bc, ac, bsc, asc = quote(b, kc, "C")
    bp, ap, bsp, asp = quote(b, kp, "P")
    bid, ask = bc + bp, ac + ap
    mid = (bid + ask) / 2
    vol_mid = invert_total_vol(xs, kc, kp, mid)
    vol_ask = invert_total_vol(xs, kc, kp, ask)
    b16 = book_at(q, day, SETTLE_BOOK)
    bc16, ac16, _, _ = quote(b16, kc, "C")
    bp16, ap16, _, _ = quote(b16, kp, "P")
    xc = spot_close * XSP_SCALE
    payoff = max(xc - kc, 0.0) + max(kp - xc, 0.0)
    return {
        "spot_xsp": xs,
        "kc": kc,
        "kp": kp,
        "grid_step": step,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "rel_spread": (ask - bid) / mid if mid > 0 else float("nan"),
        "ask_size": min(asc, asp),
        "bid_size": min(bsc, bsp),
        "ask_c": ac,
        "ask_p": ap,
        "iv_var_xsp_mid": vol_mid**2 if np.isfinite(vol_mid) else float("nan"),
        "iv_var_xsp_ask": vol_ask**2 if np.isfinite(vol_ask) else float("nan"),
        "mid_1600": ((bc16 + ac16) / 2 + (bp16 + ap16) / 2),
        "payoff": payoff,
        "R_ask": payoff / ask - 1 if ask > 0 else float("nan"),
        "R_mid": payoff / mid - 1 if mid > 0 else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=70000.0)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    deck = pd.read_parquet(DECK)
    deck.index = pd.DatetimeIndex(deck.index)
    # gate: the deck's iv_var is the Black-76 package variance of its SPXW mid
    chk = deck.dropna(subset=["S", "K_c", "K_p", "entry", "iv_var"]).tail(200)
    rep = np.array(
        [invert_total_vol(r.S, r.K_c, r.K_p, r.entry) ** 2 for r in chk.itertuples()]
    )
    ratio = rep / chk["iv_var"].to_numpy()
    ok = np.isfinite(ratio)
    print(
        f"GATE deck iv_var reproduced from the SPXW mid on {int(ok.sum())} days: "
        f"max |ratio-1| = {np.abs(ratio[ok] - 1).max():.2e}"
    )
    spots = spot_table()
    rows = []
    for f in sorted(glob.glob(str(OPRA_DIR / "cbbo1m_*.parquet"))):
        day = pd.Timestamp(Path(f).stem.split("_")[1])
        dfile = OPRA_DIR / f"defs_{day:%Y-%m-%d}.parquet"
        if not dfile.exists():
            continue
        defs = pd.read_parquet(dfile)
        if defs.empty:
            continue
        if day in deck.index:
            s1530, sclose = float(deck.at[day, "S"]), float(deck.at[day, "S_close"])
        else:
            try:
                s1530 = float(spots.loc[(day, "15:30")])
                sclose = float(spots.loc[(day, "16:00")])
            except KeyError:
                continue
        r = one_day(day, load_quotes(Path(f)), defs, s1530, sclose)
        r["day"] = day
        r["in_deck"] = day in deck.index
        rows.append(r)
    x = pd.DataFrame(rows).set_index("day").sort_index()
    if x.empty:
        print("no XSP days on disk yet")
        return 1
    j = x.join(
        deck[["S", "K_c", "K_p", "entry", "R", "pos", "rv_hat", "iv_var", "R_p"]],
        how="left",
    )
    j["month_end"] = (
        ~j.index.to_series().dt.to_period("M").duplicated(keep="last").to_numpy()
    )
    # the card's rule on the XSP book: buy iff ask <= P*(rv_hat) on the XSP pair
    j["p_star_xsp"] = [
        package_price(np.sqrt(r.rv_hat), r.spot_xsp, r.kc, r.kp)
        if np.isfinite(r.rv_hat)
        else np.nan
        for r in j.itertuples()
    ]
    j["card_long"] = j["ask"] <= j["p_star_xsp"]
    j["n_general"] = np.floor(a.capital * LONG_FRACTION / (j["ask"] * MULT))
    j["n_month_end"] = np.floor(a.capital * MONTH_END_FRACTION / (j["ask"] * MULT))
    j.to_csv(OUT / "xsp_close_book_daily.csv", float_format="%.6g")

    lines = []
    n = len(j)
    lines.append(
        f"days with an XSP book: {n} ({j.index.min().date()} .. {j.index.max().date()}); in the deck: {int(j.in_deck.sum())}"
    )
    g = j["grid_step"].value_counts().sort_index()
    lines.append(
        "listed strike step near ATM: "
        + ", ".join(f"{k:g} pt on {v} days" for k, v in g.items())
    )
    lines.append(
        f"straddle at 15:30 (nearest-OTM on the listed grid): mid median {j.mid.median():.3f}, "
        f"relative spread median {j.rel_spread.median():.1%} (p90 {j.rel_spread.quantile(0.9):.1%}); "
        f"touch size at the ask median {j.ask_size.median():.0f} straddles (p10 {j.ask_size.quantile(0.1):.0f})"
    )
    lines.append(
        f"budget fits the displayed ask size: general {LONG_FRACTION:.1%} on {(j.n_general <= j.ask_size).mean():.0%} of days "
        f"(N median {j.n_general.median():.0f}); month-end {MONTH_END_FRACTION:.0%} on {(j.n_month_end <= j.ask_size).mean():.0%} "
        f"(N median {j.n_month_end.median():.0f})"
    )
    d = j[j.in_deck & np.isfinite(j.iv_var)]
    if len(d):
        rm = np.log(d.iv_var_xsp_mid / d.iv_var)
        ra = np.log(d.iv_var_xsp_ask / d.iv_var)
        lines.append(
            f"parity on {len(d)} deck days, log(XSP implied var / SPXW iv_var): at the XSP mid median {rm.median():+.3f} "
            f"(IQR {rm.quantile(0.25):+.3f}..{rm.quantile(0.75):+.3f}); at the XSP ask median {ra.median():+.3f}"
        )
        lg = d[d.pos > 0]
        lines.append(
            f"deck long days {len(lg)}: R at SPXW mid mean {lg.R.mean():+.3f} (hit {(lg.R > 0).mean():.0%}); "
            f"same days at the XSP ASK mean {lg.R_ask.mean():+.3f} (hit {(lg.R_ask > 0).mean():.0%}), at the XSP mid {lg.R_mid.mean():+.3f}"
        )
        cl = d[d.card_long]
        t = (
            cl.R_ask.mean() / (cl.R_ask.std(ddof=1) / np.sqrt(len(cl)))
            if len(cl) > 2
            else float("nan")
        )
        lines.append(
            f"card rule (XSP ask <= P*): passes on {len(cl)} of {len(d)} days ({len(cl[cl.pos > 0])} of the deck's longs); "
            f"R at the ask mean {cl.R_ask.mean():+.3f}, t {t:.2f}, hit {(cl.R_ask > 0).mean():.0%}"
        )
        me = d[d.month_end]
        if len(me):
            lines.append(
                f"month-ends in the overlap {len(me)}: long at the XSP ask mean R {me.R_ask.mean():+.3f} "
                f"(hit {(me.R_ask > 0).mean():.0%}); deck R at SPXW mid {me.R.mean():+.3f}"
            )
    post = j[~j.in_deck]
    if len(post):
        me = post[post.month_end]
        lines.append(
            f"post-deck days {len(post)} (no rv_hat yet): unconditional long at the XSP ask mean R {post.R_ask.mean():+.3f} "
            f"(hit {(post.R_ask > 0).mean():.0%}); month-ends {len(me)} mean R {me.R_ask.mean():+.3f}"
        )
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
