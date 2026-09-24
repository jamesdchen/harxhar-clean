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

from live.ibkr.calendar_guard import is_last_session_of_month  # noqa: E402
from live.ibkr.pricing import invert_total_vol, package_price  # noqa: E402

ET = "America/New_York"
XSP_SCALE = 0.1
OPRA_DIR = REPO / "data" / "archive" / "xsp_opra"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
SPOT = REPO / "data" / "spxw_spot.parquet"
CHAIN = REPO / "data" / "spxw_chain.parquet"
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
    """SPX at 15:30 and 16:00 by day from the chain's spot file.

    The file is stamped in TRUE UTC (first bar 14:35 in January, 13:35 in
    July = 09:35 ET), so it is converted to ET -- not the panel's naive-ET
    convention.  Dropping the zone instead read the 11:30 ET spot as 15:30.
    """
    s = pd.read_parquet(SPOT)
    t = pd.to_datetime(s["timestamp"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(ET).dt.tz_localize(None)
    s = s.assign(t=t, day=t.dt.normalize(), clk=t.dt.strftime("%H:%M"))
    return s.set_index(["day", "clk"])["spot"].astype(float)


def parity_spot(b: pd.DataFrame) -> float:
    """XSP spot from the 15:30 book: K + C_mid - P_mid at the 3 strikes nearest C = P.

    Same-day expiry, so the carry term is below a cent of XSP; used where the
    chain has no spot (2026) and gated against the chain spot where it does.
    """
    m = (b["bid_px_00"] + b["ask_px_00"]) / 2
    ok = (b["ask_px_00"] > 0) & np.isfinite(m)
    m = m[ok].unstack("cp")
    if not {"C", "P"} <= set(m.columns):
        return float("nan")
    d = (m["C"] - m["P"]).dropna()
    if d.empty:
        return float("nan")
    near = d.abs().nsmallest(3).index
    return float(np.median(near.to_numpy() + d.loc[near].to_numpy()))


def spx_closes(days: list[pd.Timestamp]) -> pd.Series:
    """Official SPX closes (^GSPC daily, yfinance) for days the chain does not cover, cached."""
    cache = OUT / "spx_close_yf.csv"
    have = (
        pd.read_csv(cache, index_col=0, parse_dates=True)["close"]
        if cache.exists()
        else pd.Series(dtype=float)
    )
    need = [d for d in days if d not in have.index]
    if need:
        import yfinance as yf

        h = yf.Ticker("^GSPC").history(
            start=min(need).strftime("%Y-%m-%d"),
            end=(max(need) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        got = pd.Series(
            h["Close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(h.index.tz_localize(None).normalize()),
        )
        have = pd.concat([have, got[~got.index.isin(have.index)]]).sort_index()
        have.rename("close").to_frame().to_csv(cache)
    return have


def spxw_book(days: pd.DatetimeIndex) -> pd.DataFrame:
    """The SPXW nearest-OTM 0DTE straddle at 15:30 ET from the chain (true UTC stamps).

    No sizes in the chain: this is the SPX line's price and spread, not its depth.
    bid == ask == 0 is no quote (a known chain defect) and is dropped.
    """
    c = pd.read_parquet(
        CHAIN,
        columns=[
            "expiration",
            "strike",
            "cp",
            "timestamp",
            "bid",
            "ask",
            "underlying_price",
        ],
        filters=[("expiration", ">=", days.min()), ("expiration", "<=", days.max())],
    )
    t = pd.to_datetime(c["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None)
    c = c[(t.dt.strftime("%H:%M") == "15:30") & (t.dt.normalize() == c["expiration"])]
    c = c[c["ask"] > 0]
    rows = []
    for day, g in c.groupby("expiration"):
        s = float(g["underlying_price"].median())
        g = g.set_index(["strike", "cp"]).sort_index()
        ks = np.sort(g.index.get_level_values(0).unique().to_numpy(dtype=float))
        kc, kp = nearest_otm(ks, s)
        try:
            ask = float(g.at[(kc, "C"), "ask"] + g.at[(kp, "P"), "ask"])
            bid = float(g.at[(kc, "C"), "bid"] + g.at[(kp, "P"), "bid"])
        except KeyError:
            continue
        rows.append(
            {
                "day": day,
                "spx_S": s,
                "spx_kc": kc,
                "spx_kp": kp,
                "spx_bid": bid,
                "spx_ask": ask,
            }
        )
    return pd.DataFrame(rows).set_index("day")


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
    files = sorted(glob.glob(str(OPRA_DIR / "cbbo1m_*.parquet")))
    all_days = [pd.Timestamp(Path(f).stem.split("_")[1]) for f in files]
    chain_days = set(spots.index.get_level_values(0))
    closes = spx_closes([d for d in all_days if d not in chain_days])
    rows, skipped, gate = [], [], []
    for f, day in zip(files, all_days):
        dfile = OPRA_DIR / f"defs_{day:%Y-%m-%d}.parquet"
        if not dfile.exists():
            skipped.append((day, "no defs"))
            continue
        defs = pd.read_parquet(dfile)
        if defs.empty:
            skipped.append((day, "empty defs"))
            continue
        q = load_quotes(Path(f))
        ps = parity_spot(book_at(q, day, DECISION)) / XSP_SCALE
        if day in deck.index:
            s1530, sclose = float(deck.at[day, "S"]), float(deck.at[day, "S_close"])
            src = "deck"
        elif (day, "15:30") in spots.index and (day, "16:00") in spots.index:
            s1530 = float(spots.loc[(day, "15:30")])
            sclose = float(spots.loc[(day, "16:00")])
            src = "chain"
        elif np.isfinite(ps) and day in closes.index:
            s1530, sclose, src = ps, float(closes.loc[day]), "parity+yf"
        else:
            skipped.append((day, "no spot or no ^GSPC close"))
            continue
        if src != "parity+yf" and np.isfinite(ps):
            gate.append(ps / s1530 - 1)
        r: dict[str, object] = dict(one_day(day, q, defs, s1530, sclose))
        r["day"] = day
        r["in_deck"] = day in deck.index
        r["spot_source"] = src
        rows.append(r)
    x = pd.DataFrame(rows).set_index("day").sort_index()
    if x.empty:
        print("no XSP days on disk yet")
        return 1
    g_ = np.abs(np.array(gate))
    print(
        f"GATE parity spot vs the chain/deck 15:30 spot on {len(g_)} days: median |rel err| "
        f"{np.median(g_):.1e}, p99 {np.quantile(g_, 0.99):.1e}, max {g_.max():.1e}"
    )
    if skipped:
        print("skipped:", ", ".join(f"{d.date()} ({why})" for d, why in skipped))
    j = x.join(
        deck[["S", "K_c", "K_p", "entry", "R", "pos", "rv_hat", "iv_var", "R_p"]],
        how="left",
    )
    # the session calendar, not the last pulled day (2026-09-22 is not a month-end)
    j["month_end"] = [is_last_session_of_month(d.date()) for d in j.index]
    sx = spxw_book(j.index[j.spot_source != "parity+yf"])
    j = j.join(sx, how="left")
    j["spx_rel_spread"] = (j.spx_ask - j.spx_bid) / ((j.spx_ask + j.spx_bid) / 2)
    # the SPX close is the XSP payoff's own settlement, x 10 (one_day's xc)
    s_close = pd.Series(
        [
            deck.at[d, "S_close"] if src == "deck" else spots.loc[(d, "16:00")]
            for d, src in zip(j.index, j.spot_source)
            if src != "parity+yf"
        ],
        index=j.index[j.spot_source != "parity+yf"],
        dtype=float,
    ).reindex(j.index)
    j["spx_payoff"] = np.maximum(s_close - j.spx_kc, 0) + np.maximum(
        j.spx_kp - s_close, 0
    )
    j["spx_R_ask"] = j.spx_payoff / j.spx_ask - 1
    # the card's rule on the XSP book: buy iff ask <= P*(rv_hat) on the XSP pair
    j["p_star_xsp"] = [
        package_price(np.sqrt(r.rv_hat), r.spot_xsp, r.kc, r.kp)
        if np.isfinite(r.rv_hat)
        else np.nan
        for r in j.itertuples()
    ]
    j["card_long"] = j["ask"] <= j["p_star_xsp"]
    # the same rule on the SPX line: buy iff the SPXW ask <= P*(rv_hat) on its own pair
    j["p_star_spx"] = [
        package_price(np.sqrt(r.rv_hat), r.spx_S, r.spx_kc, r.spx_kp)
        if np.isfinite(r.rv_hat) and np.isfinite(r.spx_S)
        else np.nan
        for r in j.itertuples()
    ]
    j["card_long_spx"] = j["spx_ask"] <= j["p_star_spx"]
    j["n_general"] = np.floor(a.capital * LONG_FRACTION / (j["ask"] * MULT))
    j["n_month_end"] = np.floor(a.capital * MONTH_END_FRACTION / (j["ask"] * MULT))

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
        cs = d[d.card_long_spx & d.spx_R_ask.notna()]
        ts = (
            cs.spx_R_ask.mean() / (cs.spx_R_ask.std(ddof=1) / np.sqrt(len(cs)))
            if len(cs) > 2
            else float("nan")
        )
        lines.append(
            f"card rule on the SPX line (SPXW ask <= P*): passes on {len(cs)} of {len(d)} days "
            f"({len(cs[cs.pos > 0])} of the deck's longs); R at the SPXW ask mean {cs.spx_R_ask.mean():+.3f}, "
            f"t {ts:.2f}, hit {(cs.spx_R_ask > 0).mean():.0%}"
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
    # venue: the XSP book vs the SPX line (same days, own nearest-OTM strikes)
    lines.append(
        "spot source: "
        + ", ".join(f"{k} {v}" for k, v in j.spot_source.value_counts().items())
    )
    for yr, g in j.groupby(j.index.year):
        spx = (
            f"; SPXW rel spread {g.spx_rel_spread.median():.1%} on {int(g.spx_ask.notna().sum())} days"
            if g.spx_ask.notna().any()
            else "; SPXW: no chain"
        )
        lines.append(
            f"  {yr}: XSP rel spread {g.rel_spread.median():.1%}, touch at the ask median "
            f"{g.ask_size.median():.0f} (p10 {g.ask_size.quantile(0.1):.0f}), month-end N fits on "
            f"{(g.n_month_end <= g.ask_size).mean():.0%}, general N on {(g.n_general <= g.ask_size).mean():.0%}{spx}"
        )
    both = j[j.spx_ask.notna() & j.ask.notna()]
    me = both[both.month_end]
    lines.append(
        f"same-day venue check on {len(both)} days: rel spread XSP {both.rel_spread.median():.1%} vs SPXW "
        f"{both.spx_rel_spread.median():.1%}; unconditional long at the ask mean R XSP {both.R_ask.mean():+.3f} "
        f"vs SPXW {both.spx_R_ask.mean():+.3f}"
    )
    if len(me):
        lines.append(
            f"month-ends with both books {len(me)}: long at the ask mean R XSP {me.R_ask.mean():+.3f} (hit "
            f"{(me.R_ask > 0).mean():.0%}) vs SPXW {me.spx_R_ask.mean():+.3f} (hit {(me.spx_R_ask > 0).mean():.0%}); "
            f"XSP touch median {me.ask_size.median():.0f} vs month-end N median {me.n_month_end.median():.0f}; "
            f"SPX N at the same budget median {np.floor(a.capital * MONTH_END_FRACTION / (me.spx_ask * MULT)).median():.0f}"
        )
    me_all = j[j.month_end]
    lines.append(
        f"all month-ends with an XSP book {len(me_all)}: XSP long at the ask mean R {me_all.R_ask.mean():+.3f}; "
        f"month-end N fits the XSP touch on {int((me_all.n_month_end <= me_all.ask_size).sum())}"
    )
    j.to_csv(OUT / "xsp_close_book_daily.csv", float_format="%.6g")
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
