"""Proposal 29 -- does the 11:00 delta-hedged short survive a defined-risk retail venue?

The book: sell the nearest-OTM SPX 0DTE straddle at 11:00 ET, Black-76
delta-hedge every 30 minutes on vendor spot with the hourly implied
volatility re-inverted on the censored nodes, then either hold through cash
settlement or buy the straddle back at 15:30.  Returns are per unit of the
midpoint straddle entry premium, one unit a day.

A retail venue (Robinhood) will not write a naked index straddle: the only
short-premium structure it permits is defined-risk, i.e. the straddle plus a
long call above and a long put below (an iron butterfly).  It also has no
API, so the hedge is placed by hand in MES futures.  Three questions:

  Q1  wings -- what does buying the protection at 11:00 cost, per width on a
      grid stated as a fraction of spot, and what is the hard loss cap?
  Q2  cadence -- what survives when the delta is rebalanced by hand at 60
      minutes, at three fixed clocks, once, or never?
  Q3  fees -- the published per-contract schedule against the daily P&L.

Gate first: the 11:00 book is rebuilt from the same primitives and asserted
against writeup/make_dh_causal_standalone_tex.py::hold_mark_1100 before a
single new number is computed.

Run:  python writeup/intraday_proposals/29_defined_risk_retail.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "writeup"))
import atm_straddle_lib as asl  # noqa: E402

HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "29"
CHAIN = ROOT / "data" / "spxw_chain.parquet"

ENTRY = "11:00"
CLOSE = "15:30"

# The daily-0DTE era: SPXW expirations every session from this date on.
ERA0 = pd.Timestamp("2022-05-16")

# Wing distance grid, stated as a fraction of spot at the 11:00 entry.
WIDTH_FRACS = (0.005, 0.010, 0.015, 0.020, 0.030)

SPX_MULT = 100.0  # dollars per index point, one SPX option contract
MES_MULT = 5.0  # dollars per index point, one Micro E-mini S&P contract

# Robinhood's published schedule, per contract per side.
FEE_OPT_SIDE = 0.50 + 0.04  # index-option commission + regulatory
FEE_MES_BROKER_SIDE = 0.50
# Exchange + regulatory on MES is not published by the broker; the stated
# retail band is $0.35-0.50 per side, so the midpoint of that band is used.
FEE_MES_EXCH_SIDE = 0.5 * (0.35 + 0.50)
FEE_MES_SIDE = FEE_MES_BROKER_SIDE + FEE_MES_EXCH_SIDE

# Pass bars set by the study.
BAR_SHARPE = 2.5  # daily-era crossed Sharpe a width must hold
BAR_CAP_UNITS = 5.0  # hard loss cap, in units of straddle premium
BAR_CADENCE_KEEP = 0.80  # share of the 30-minute Sharpe the 60-minute must keep

# Hand-hedging cadences. None means every 30-minute stamp from the entry on.
# A cadence that starts after 11:00 leaves the straddle unhedged until its
# first clock; the position is flat, not carried, over that stretch.
CADENCES: dict[str, tuple[str, ...] | None] = {
    "30min": None,
    "60min_top_of_hour": ("11:00", "12:00", "13:00", "14:00", "15:00"),
    "3clocks_1200_1330_1500": ("12:00", "13:30", "15:00"),
    "once_1300": ("13:00",),
    "never": (),
}


# ------------------------------------------------------------------ utils --
def _maxdd(x) -> float:
    """Worst peak-to-trough of the cumulative *sum* path, peak seeded at zero.

    Same definition as writeup/make_dh_causal_standalone_tex.py::_maxdd: one
    unit a day, summed (not compounded), in expiration-date order, so a
    first-day loss already counts against the running peak.
    """
    v = pd.Series(x).dropna().sort_index().to_numpy(float)
    if v.size < 1:
        return float("nan")
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def _stats(r) -> dict:
    s = pd.Series(r).dropna().sort_index()
    n = int(s.size)
    if n < 2:
        return {
            "n": n,
            "mean": np.nan,
            "sd": np.nan,
            "Sharpe_ann": np.nan,
            "t": np.nan,
            "MaxDD": np.nan,
            "worst": np.nan,
        }
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    return {
        "n": n,
        "mean": mu,
        "sd": sd,
        "Sharpe_ann": mu / sd * np.sqrt(asl.PERIODS_PER_YEAR) if sd > 0 else np.nan,
        "t": mu / sd * np.sqrt(n) if sd > 0 else np.nan,
        "MaxDD": _maxdd(s),
        "worst": float(s.min()),
    }


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _p26():
    return _load(
        ROOT / "writeup" / "intraday_proposals" / "26_causal_entry_flatten.py",
        "p26_r29",
    )


def _rule_mod():
    if "--dh-holdclose" not in sys.argv:
        sys.argv.append("--dh-holdclose")
    import make_rule_by_strategy_intraday_tex as m  # noqa: E402

    return m


def _standalone():
    return _load(ROOT / "writeup" / "make_dh_causal_standalone_tex.py", "mdc_r29")


# ------------------------------------------------------------- book frame --
def book_frame(pkg: pd.DataFrame) -> dict:
    """The 11:00 entry slice and the hedge grids, exactly as in the book.

    The grid construction below is the hedge-sum block of
    make_dh_causal_standalone_tex.py::hold_mark_1100, copied (not edited) so
    the hedge sum can be taken over a coarser rebalance set in Q2.
    """
    p26 = _p26()
    m = _rule_mod()
    deck = pd.read_parquet(m.DECK / "daily_blk2.parquet")
    days = pd.to_datetime(deck.index)
    p = pkg.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"].isin(days)].copy()
    clocks = sorted(p["hhmm"].unique())
    if ENTRY not in clocks or CLOSE not in clocks:
        raise SystemExit(f"need {ENTRY} and {CLOSE} in {clocks}")
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    j = clocks.index(ENTRY)
    j15 = clocks.index(CLOSE)
    p["h_rem"] = p["hhmm"].map(n_rem).astype(float) * 0.5
    p["iv_hourly_used"] = m.iv_hourly_reinverted(p).to_numpy()
    dh = m.attach_long_dh(p)
    sl = (
        dh.loc[dh["hhmm"] == ENTRY]
        .dropna(subset=["entry", "exit", "K_c", "K_p", "S_close", "R"])
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    sl = sl.loc[sl["entry"].astype(float) > 0]
    dates = sl.index
    S_grid = p.pivot_table(
        index="date", columns="hhmm", values="S", aggfunc="first"
    ).reindex(index=dates, columns=clocks)
    IV_grid = p.pivot_table(
        index="date", columns="hhmm", values="iv_hourly_used", aggfunc="first"
    ).reindex(index=dates, columns=clocks)
    Kc = sl["K_c"].to_numpy(float)
    Kp = sl["K_p"].to_numpy(float)
    entry = sl["entry"].to_numpy(float)
    ex = sl["exit"].to_numpy(float)
    ST = sl["S_close"].to_numpy(float)
    Sg = S_grid.to_numpy(float)
    IVg = IV_grid.to_numpy(float)
    h_row = np.array([n_rem[c] * 0.5 for c in clocks])
    tot = np.where(IVg > 0, IVg * np.sqrt(h_row[None, :]), np.nan)
    tot[:, :j] = np.nan
    dlt = p26.pkg_delta(tot, Sg, Kc[:, None], Kp[:, None])
    nxt = np.full_like(Sg, np.nan)
    nxt[:, :-1] = Sg[:, 1:]
    nxt[:, -1] = ST
    dS = np.where(np.isfinite(Sg) & np.isfinite(nxt), nxt - Sg, 0.0)
    dS[:, :j] = 0.0
    tot15 = np.where(IVg[:, j15] > 0, IVg[:, j15] * np.sqrt(h_row[j15]), np.nan)
    mark = p26.pkg_price(tot15, Sg[:, j15], Kc, Kp)
    dS_f = dS.copy()
    dS_f[:, j15] = 0.0
    bid = (
        sl["bid_entry"].to_numpy(float)
        if "bid_entry" in sl.columns
        else sl["bid_c"].to_numpy(float) + sl["bid_p"].to_numpy(float)
    )
    return {
        "sl": sl,
        "dates": dates,
        "clocks": clocks,
        "j": j,
        "j15": j15,
        "Sg": Sg,
        "IVg": IVg,
        "dlt": dlt,
        "dS": dS,
        "dS_f": dS_f,
        "Kc": Kc,
        "Kp": Kp,
        "entry": entry,
        "ex": ex,
        "ST": ST,
        "mark": mark,
        "bid": bid,
        "ok_bid": np.isfinite(bid) & (bid > 0),
        "hedge_long": sl["hedge_long"].to_numpy(float),
        "S_entry": sl["S"].to_numpy(float),
    }


def hedge_sum(dlt: np.ndarray, dS: np.ndarray, rebal) -> np.ndarray:
    """Hedge P&L of the SHORT straddle, delta held between rebalances.

    Delta is read at each rebalance stamp and applied unchanged to every
    later spot change until the next rebalance; before the first rebalance
    the hedge is flat.  With every stamp a rebalance this is the book's
    (dlt * dS).sum(axis=1) term exactly.
    """
    used = np.zeros_like(dlt)
    rb = set(int(k) for k in rebal)
    last = -1
    for k in range(dlt.shape[1]):
        if k in rb:
            last = k
        if last >= 0:
            used[:, k] = dlt[:, last]
    return (used * dS).sum(axis=1)


# ------------------------------------------------------------------ chain --
def chain_at(stamps: pd.DatetimeIndex, stamp_to_date: pd.Series) -> pd.DataFrame:
    """0DTE bid/ask at the requested stamps, keyed (date, cp, strike).

    Only the 11:00 and 15:30 stamps are read -- never the 16:00 row.  A row
    with bid == ask == 0 is the vendor's no-quote sentinel and is flagged
    dead, not dropped, so the untradeable days can be counted.
    """
    tbl = pq.read_table(
        CHAIN,
        columns=[
            "expiration",
            "strike",
            "cp",
            "timestamp",
            "bid",
            "ask",
            "hours_to_expiration",
        ],
        filters=[("timestamp", "in", list(stamps))],
    )
    ch = tbl.to_pandas()
    ch["date"] = ch["timestamp"].map(stamp_to_date)
    ch = ch.dropna(subset=["date"])
    ch = ch[ch["expiration"] == ch["date"]]  # 0DTE only
    ch = ch[ch["hours_to_expiration"] > 0]  # frozen half-sessions out
    ch["strike"] = ch["strike"].astype(float)
    ch["bid"] = ch["bid"].astype(float)
    ch["ask"] = ch["ask"].astype(float)
    ch["mid"] = 0.5 * (ch["bid"] + ch["ask"])
    ch["live"] = ~((ch["bid"] == 0.0) & (ch["ask"] == 0.0))
    ch = ch.drop_duplicates(["date", "cp", "strike"])
    return ch.set_index(["date", "cp", "strike"])[
        ["bid", "ask", "mid", "live"]
    ].sort_index()


def quote(q: pd.DataFrame, dates, cp: str, strikes, field: str) -> np.ndarray:
    key = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates), np.full(len(dates), cp), np.asarray(strikes, float)]
    )
    return q[field].reindex(key).to_numpy(float)


def outward_strike(listed: dict, dates, target: np.ndarray, up: bool) -> np.ndarray:
    """Round each target outward (away from the money) to a listed strike."""
    out = np.full(len(target), np.nan)
    for i, d in enumerate(dates):
        a = listed.get(d)
        if a is None or not np.isfinite(target[i]):
            continue
        if up:
            k = int(np.searchsorted(a, target[i], side="left"))
            if k < a.size:
                out[i] = a[k]
        else:
            k = int(np.searchsorted(a, target[i], side="right")) - 1
            if k >= 0:
                out[i] = a[k]
    return out


def wing_legs(
    bk: dict,
    q11: pd.DataFrame,
    q15: pd.DataFrame,
    listed_c: dict,
    listed_p: dict,
    wf: float,
) -> dict:
    """Long call above / long put below at wf of spot, and their quotes."""
    dates, Kc, Kp, S, ST = bk["dates"], bk["Kc"], bk["Kp"], bk["S_entry"], bk["ST"]
    kw_c = outward_strike(listed_c, dates, Kc + wf * S, up=True)
    kw_p = outward_strike(listed_p, dates, Kp - wf * S, up=False)
    g = {"K_wing_c": kw_c, "K_wing_p": kw_p}
    g["no_listed"] = ~(np.isfinite(kw_c) & np.isfinite(kw_p))
    c11a = quote(q11, dates, "C", kw_c, "ask")
    p11a = quote(q11, dates, "P", kw_p, "ask")
    g["cost_mid"] = quote(q11, dates, "C", kw_c, "mid") + quote(
        q11, dates, "P", kw_p, "mid"
    )
    g["cost_ask"] = c11a + p11a
    g["sale_mid"] = quote(q15, dates, "C", kw_c, "mid") + quote(
        q15, dates, "P", kw_p, "mid"
    )
    g["sale_bid"] = quote(q15, dates, "C", kw_c, "bid") + quote(
        q15, dates, "P", kw_p, "bid"
    )
    live11 = (quote(q11, dates, "C", kw_c, "live") == 1.0) & (
        quote(q11, dates, "P", kw_p, "live") == 1.0
    )
    g["live11"] = live11 & ~g["no_listed"] & np.isfinite(c11a) & np.isfinite(p11a)
    g["live15"] = (quote(q15, dates, "C", kw_c, "live") == 1.0) & (
        quote(q15, dates, "P", kw_p, "live") == 1.0
    )
    g["intr"] = np.maximum(ST - kw_c, 0.0) + np.maximum(kw_p - ST, 0.0)
    g["w_up"] = kw_c - Kc
    g["w_dn"] = Kp - kw_p
    g["width"] = np.maximum(g["w_up"], g["w_dn"])
    return g


# --------------------------------------------------------------------- Q1 --
def q1_wings(
    bk: dict,
    q11: pd.DataFrame,
    q15: pd.DataFrame,
    listed_c: dict,
    listed_p: dict,
    legs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = bk["dates"]
    entry, bid, ex, S = bk["entry"], bk["bid"], bk["ex"], bk["S_entry"]
    allk = list(range(bk["j"], len(bk["clocks"])))
    hedge_h = hedge_sum(bk["dlt"], bk["dS"], allk)
    hedge_f = hedge_sum(bk["dlt"], bk["dS_f"], allk)
    era = np.asarray(pd.DatetimeIndex(dates) >= ERA0)

    bb_mid = legs["c15_mid"] + legs["p15_mid"]
    bb_ask = legs["c15_ask"] + legs["p15_ask"]
    bb_live = legs["c15_live"] & legs["p15_live"]

    med_cols = [
        "wing_cost_pct_prem",
        "net_credit_units",
        "w_up_pts",
        "w_dn_pts",
        "width_pts",
        "width_pct_spot",
        "cap_units",
        "cap_dollars",
        "regt_margin_proxy_dollars",
    ]
    rows = []

    def emit(label, w_frac, sample, mask, exit_mode, basis, r, extra):
        sel = np.asarray(mask) & (era if sample == "era" else np.ones(len(dates), bool))
        s = pd.Series(np.where(sel, r, np.nan), index=dates)
        d = {
            "width": label,
            "w_frac": w_frac,
            "sample": sample,
            "exit_mode": exit_mode,
            "basis": basis,
        }
        d.update(_stats(s))
        for k in med_cols:
            v = extra.get(k)
            d[k] = (
                float(np.nanmedian(v[sel])) if (v is not None and sel.sum()) else np.nan
            )
        rows.append(d)

    base_ok = np.isfinite(entry) & (entry > 0)
    okb = base_ok & bk["ok_bid"]

    # --- naked straddle reference -------------------------------------------
    r_h_mid = (-(ex - entry) + hedge_h) / entry
    r_h_x = (-(ex - bid) + hedge_h) / entry
    r_f_mid_q = (-(bb_mid - entry) + hedge_f) / entry
    r_f_x_q = (-(bb_ask - bid) + hedge_f) / entry
    r_f_mid_m = (-(bk["mark"] - entry) + hedge_f) / entry
    r_f_x_m = (-(bk["mark"] - bid) + hedge_f) / entry
    nak: dict = {}
    for sample in ("all", "era"):
        emit("naked straddle", np.nan, sample, base_ok, "hold", "mid", r_h_mid, nak)
        emit("naked straddle", np.nan, sample, okb, "hold", "crossed", r_h_x, nak)
        emit(
            "naked straddle",
            np.nan,
            sample,
            base_ok & bb_live,
            "exit1530",
            "mid",
            r_f_mid_q,
            nak,
        )
        emit(
            "naked straddle",
            np.nan,
            sample,
            okb & bb_live,
            "exit1530",
            "crossed",
            r_f_x_q,
            nak,
        )
        emit(
            "naked straddle (model mark)",
            np.nan,
            sample,
            base_ok,
            "exit1530",
            "mid",
            r_f_mid_m,
            nak,
        )
        emit(
            "naked straddle (model mark)",
            np.nan,
            sample,
            okb,
            "exit1530",
            "crossed",
            r_f_x_m,
            nak,
        )

    # --- iron butterflies ----------------------------------------------------
    diag = []
    for wf in WIDTH_FRACS:
        label = f"{wf * 100:.1f}% of spot"
        g = wing_legs(bk, q11, q15, listed_c, listed_p, wf)
        nc_mid = entry - g["cost_mid"]
        nc_x = bid - g["cost_ask"]
        cap_mid = g["width"] - nc_mid
        cap_x = g["width"] - nc_x

        ok11 = base_ok & g["live11"]
        ok11x = okb & g["live11"]
        diag.append(
            {
                "width": label,
                "w_frac": wf,
                "n_scored": int(base_ok.sum()),
                "n_no_listed_strike": int((base_ok & g["no_listed"]).sum()),
                "n_dead_wing_quote_1100": int(
                    (base_ok & ~g["no_listed"] & ~g["live11"]).sum()
                ),
                "n_tradeable_1100": int(ok11.sum()),
                "n_dead_wing_quote_1530": int((ok11 & ~g["live15"]).sum()),
                "n_dead_straddle_quote_1530": int((ok11 & ~bb_live).sum()),
                "n_tradeable_exit1530": int((ok11 & g["live15"] & bb_live).sum()),
            }
        )

        ex_mid = {
            "wing_cost_pct_prem": 100.0 * g["cost_mid"] / entry,
            "net_credit_units": nc_mid / entry,
            "w_up_pts": g["w_up"],
            "w_dn_pts": g["w_dn"],
            "width_pts": g["width"],
            "width_pct_spot": 100.0 * g["width"] / S,
            "cap_units": cap_mid / entry,
            "cap_dollars": cap_mid * SPX_MULT,
            "regt_margin_proxy_dollars": (g["width"] - nc_mid) * SPX_MULT,
        }
        ex_x = {
            "wing_cost_pct_prem": 100.0 * g["cost_ask"] / entry,
            "net_credit_units": nc_x / entry,
            "w_up_pts": g["w_up"],
            "w_dn_pts": g["w_dn"],
            "width_pts": g["width"],
            "width_pct_spot": 100.0 * g["width"] / S,
            "cap_units": cap_x / entry,
            "cap_dollars": cap_x * SPX_MULT,
            "regt_margin_proxy_dollars": (g["width"] - nc_x) * SPX_MULT,
        }

        r_h_mid_f = (-(ex - entry) + hedge_h + (g["intr"] - g["cost_mid"])) / entry
        r_h_x_f = (-(ex - bid) + hedge_h + (g["intr"] - g["cost_ask"])) / entry
        r_f_mid_f = (
            -(bb_mid - entry) + hedge_f + (g["sale_mid"] - g["cost_mid"])
        ) / entry
        r_f_x_f = (-(bb_ask - bid) + hedge_f + (g["sale_bid"] - g["cost_ask"])) / entry

        okx = ok11 & g["live15"] & bb_live
        okxx = ok11x & g["live15"] & bb_live
        for sample in ("all", "era"):
            emit(label, wf, sample, ok11, "hold", "mid", r_h_mid_f, ex_mid)
            emit(label, wf, sample, ok11x, "hold", "crossed", r_h_x_f, ex_x)
            emit(label, wf, sample, okx, "exit1530", "mid", r_f_mid_f, ex_mid)
            emit(label, wf, sample, okxx, "exit1530", "crossed", r_f_x_f, ex_x)

    return pd.DataFrame(rows), pd.DataFrame(diag)


# --------------------------------------------------------------------- Q2 --
def q2_cadence(bk: dict, legs: dict, wings: dict, best_label: str) -> pd.DataFrame:
    dates = bk["dates"]
    clocks, j, j15 = bk["clocks"], bk["j"], bk["j15"]
    entry, bid, ex = bk["entry"], bk["bid"], bk["ex"]
    era = np.asarray(pd.DatetimeIndex(dates) >= ERA0)
    bb_mid = legs["c15_mid"] + legs["p15_mid"]
    bb_ask = legs["c15_ask"] + legs["p15_ask"]
    bb_live = legs["c15_live"] & legs["p15_live"]
    base_ok = np.isfinite(entry) & (entry > 0)
    okb = base_ok & bk["ok_bid"]
    zero = np.zeros(len(dates))
    on = np.ones(len(dates), bool)

    rows = []
    for cname, cl in CADENCES.items():
        rebal = (
            list(range(j, len(clocks)))
            if cl is None
            else [clocks.index(c) for c in cl if c in clocks]
        )
        hh = hedge_sum(bk["dlt"], bk["dS"], rebal)
        hf = hedge_sum(bk["dlt"], bk["dS_f"], [k for k in rebal if k < j15])
        for struct in ("naked straddle", best_label):
            if struct == "naked straddle":
                cm = ca = sm = sb = it = zero
                l11 = l15 = on
            else:
                cm, ca = wings["cost_mid"], wings["cost_ask"]
                sm, sb = wings["sale_mid"], wings["sale_bid"]
                it, l11, l15 = wings["intr"], wings["live11"], wings["live15"]
            variants = {
                ("hold", "mid"): (
                    (-(ex - entry) + hh + (it - cm)) / entry,
                    base_ok & l11,
                ),
                ("hold", "crossed"): (
                    (-(ex - bid) + hh + (it - ca)) / entry,
                    okb & l11,
                ),
                ("exit1530", "mid"): (
                    (-(bb_mid - entry) + hf + (sm - cm)) / entry,
                    base_ok & l11 & l15 & bb_live,
                ),
                ("exit1530", "crossed"): (
                    (-(bb_ask - bid) + hf + (sb - ca)) / entry,
                    okb & l11 & l15 & bb_live,
                ),
            }
            for (em, basis), (rr, mask) in variants.items():
                for sample in ("all", "era"):
                    sel = np.asarray(mask) & (era if sample == "era" else on)
                    s = pd.Series(np.where(sel, rr, np.nan), index=dates)
                    d = {
                        "structure": struct,
                        "cadence": cname,
                        "n_rebalances": len(rebal),
                        "exit_mode": em,
                        "basis": basis,
                        "sample": sample,
                    }
                    d.update(_stats(s))
                    rows.append(d)
    df = pd.DataFrame(rows)
    ref = df[df["cadence"] == "30min"].set_index(
        ["structure", "exit_mode", "basis", "sample"]
    )["Sharpe_ann"]
    key = pd.MultiIndex.from_frame(df[["structure", "exit_mode", "basis", "sample"]])
    df["Sharpe_frac_of_30min"] = (
        df["Sharpe_ann"].to_numpy() / ref.reindex(key).to_numpy()
    )
    return df


# --------------------------------------------------------------------- Q3 --
def q3_fees(bk: dict, q1: pd.DataFrame, best_label: str) -> tuple[pd.DataFrame, list]:
    dates = bk["dates"]
    clocks, j, j15 = bk["clocks"], bk["j"], bk["j15"]
    entry = bk["entry"]
    era = np.asarray(pd.DatetimeIndex(dates) >= ERA0)
    dlt = bk["dlt"]

    notes = []
    rows = []
    for cname in ("30min", "60min_top_of_hour"):
        cl = CADENCES[cname]
        rebal = (
            list(range(j, len(clocks)))
            if cl is None
            else [clocks.index(c) for c in cl if c in clocks]
        )
        for em in ("hold", "exit1530"):
            # exit at 15:30 flattens there, so no 15:30 rebalance
            rb = rebal if em == "hold" else [k for k in rebal if k < j15]
            # whole-MES target position after each rebalance, then flattened
            tgt = (
                np.column_stack([np.round(dlt[:, k] * SPX_MULT / MES_MULT) for k in rb])
                if rb
                else np.zeros((len(dates), 0))
            )
            seq = np.column_stack([np.zeros(len(dates)), tgt, np.zeros(len(dates))])
            traded = np.abs(np.diff(seq, axis=1))
            mes_per_day = traded.sum(axis=1)
            mes_fee = mes_per_day * FEE_MES_SIDE
            resid = (
                np.abs(tgt * MES_MULT / SPX_MULT - dlt[:, rb])
                if rb
                else np.zeros((len(dates), 0))
            )
            notes.append(
                {
                    "cadence": cname,
                    "exit_mode": em,
                    "n_rebalances": len(rb),
                    "n_mes_trade_events": int(traded.shape[1]),
                    "med_mes_contracts_per_rebalance": float(np.median(traded))
                    if traded.size
                    else 0.0,
                    "med_mes_contracts_per_day": float(np.median(mes_per_day)),
                    "med_abs_residual_delta_units": float(np.median(resid))
                    if resid.size
                    else 0.0,
                    "pct_rebalances_with_residual": (
                        100.0 * float(np.mean(resid > 0)) if resid.size else 0.0
                    ),
                }
            )
            for struct, ncon in (("naked straddle", 2), (best_label, 4)):
                sides = 1 if em == "hold" else 2
                opt_fee = ncon * FEE_OPT_SIDE * sides
                tot = opt_fee + mes_fee
                for sample in ("all", "era"):
                    sel = era if sample == "era" else np.ones(len(dates), bool)
                    sub = q1[
                        (q1["width"] == struct)
                        & (q1["exit_mode"] == em)
                        & (q1["basis"] == "crossed")
                        & (q1["sample"] == sample)
                    ]
                    mean_units = float(sub["mean"].iloc[0]) if len(sub) else np.nan
                    med_prem = float(np.median(entry[sel]))
                    pnl_med = mean_units * med_prem * SPX_MULT
                    pnl_mean_prem = mean_units * float(np.mean(entry[sel])) * SPX_MULT
                    fee_mean = float(np.mean(tot[sel]))
                    rows.append(
                        {
                            "structure": struct,
                            "cadence": cname,
                            "exit_mode": em,
                            "sample": sample,
                            "n_option_contracts": ncon,
                            "option_sides": sides,
                            "option_fee_per_day_usd": opt_fee,
                            "mes_contracts_per_day_med": float(
                                np.median(mes_per_day[sel])
                            ),
                            "mes_contracts_per_day_mean": float(
                                np.mean(mes_per_day[sel])
                            ),
                            "mes_fee_per_day_mean_usd": float(np.mean(mes_fee[sel])),
                            "total_fee_per_day_med_usd": float(np.median(tot[sel])),
                            "total_fee_per_day_mean_usd": fee_mean,
                            "mean_r_units_crossed": mean_units,
                            "median_premium_pts": med_prem,
                            "pnl_per_day_usd_at_median_premium": pnl_med,
                            "pnl_per_day_usd_at_mean_premium": pnl_mean_prem,
                            "fee_pct_of_pnl": (
                                100.0 * fee_mean / pnl_med
                                if np.isfinite(pnl_med) and pnl_med != 0
                                else np.nan
                            ),
                            "pnl_net_of_fees_usd": pnl_med - fee_mean,
                        }
                    )
    return pd.DataFrame(rows), notes


# ------------------------------------------------------------------- main --
def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 80)
    OUT.mkdir(parents=True, exist_ok=True)
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])

    # ---------------------------------------------------------------- gate --
    mdc = _standalone()
    r_hold, r_mark, r_hold_x, r_mark_x = mdc.hold_mark_1100(pkg)
    bk = book_frame(pkg)
    allk = list(range(bk["j"], len(bk["clocks"])))
    hh = hedge_sum(bk["dlt"], bk["dS"], allk)
    hf = hedge_sum(bk["dlt"], bk["dS_f"], allk)
    assert np.allclose(hh, -bk["hedge_long"], atol=1e-9), (
        "hedge sum differs from the book"
    )
    e, b, ex, mk = bk["entry"], bk["bid"], bk["ex"], bk["mark"]
    mine = {
        "hold_mid": pd.Series((-(ex - e) + hh) / e, index=bk["dates"]),
        "hold_crossed": pd.Series((-(ex - b) + hh) / e, index=bk["dates"]).where(
            bk["ok_bid"]
        ),
        "mark_mid": pd.Series((-(mk - e) + hf) / e, index=bk["dates"]),
        "mark_crossed": pd.Series((-(mk - b) + hf) / e, index=bk["dates"]).where(
            bk["ok_bid"]
        ),
    }
    book = {
        "hold_mid": r_hold,
        "hold_crossed": r_hold_x,
        "mark_mid": r_mark,
        "mark_crossed": r_mark_x,
    }
    for k in mine:
        a = mine[k].dropna()
        c = book[k].dropna()
        assert a.index.equals(c.index), (k, len(a), len(c))
        assert np.allclose(a.to_numpy(), c.to_numpy(), atol=1e-12), k
    g = {k: _stats(v) for k, v in mine.items()}
    print(
        "GATE  n",
        g["hold_mid"]["n"],
        "| hold mid",
        f"{g['hold_mid']['Sharpe_ann']:.6f}",
        "| hold crossed",
        f"{g['hold_crossed']['Sharpe_ann']:.6f}",
        "| exit-15:30 mid",
        f"{g['mark_mid']['Sharpe_ann']:.6f}",
        f"(mean {g['mark_mid']['mean']:.6f}, sd {g['mark_mid']['sd']:.6f})",
    )
    assert g["hold_mid"]["n"] == 865
    assert abs(g["hold_mid"]["Sharpe_ann"] - 4.200674) < 1e-6
    assert abs(g["hold_crossed"]["Sharpe_ann"] - 3.582391) < 1e-6
    assert abs(g["mark_mid"]["Sharpe_ann"] - 4.695674) < 1e-6
    assert abs(g["mark_mid"]["mean"] - 0.093772) < 1e-6
    assert abs(g["mark_mid"]["sd"] - 0.317013) < 1e-6
    era = np.asarray(pd.DatetimeIndex(bk["dates"]) >= ERA0)
    print(
        f"GATE  reproduced to 1e-6; daily-0DTE era (>= {ERA0.date()}) n = {int(era.sum())}"
    )
    assert int(era.sum()) == 489

    # --------------------------------------------------------------- chain --
    dates = bk["dates"]
    sub = pkg[pkg["hhmm"].isin([ENTRY, CLOSE])][["date", "timestamp", "hhmm"]].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub[sub["date"].isin(dates)].drop_duplicates()
    s2d = sub.set_index("timestamp")["date"]
    s2d = s2d[~s2d.index.duplicated()]
    q11 = chain_at(
        pd.DatetimeIndex(sub.loc[sub["hhmm"] == ENTRY, "timestamp"].unique()), s2d
    )
    q15 = chain_at(
        pd.DatetimeIndex(sub.loc[sub["hhmm"] == CLOSE, "timestamp"].unique()), s2d
    )
    print(
        f"chain: {len(q11)} 0DTE quotes at {ENTRY} over "
        f"{q11.index.get_level_values(0).nunique()} sessions; "
        f"{len(q15)} at {CLOSE} over {q15.index.get_level_values(0).nunique()}"
    )
    cc = q11[q11.index.get_level_values(1) == "C"]
    pp = q11[q11.index.get_level_values(1) == "P"]
    listed_c = {
        d: np.sort(v.to_numpy(float))
        for d, v in pd.Series(
            cc.index.get_level_values(2), index=cc.index.get_level_values(0)
        ).groupby(level=0)
    }
    listed_p = {
        d: np.sort(v.to_numpy(float))
        for d, v in pd.Series(
            pp.index.get_level_values(2), index=pp.index.get_level_values(0)
        ).groupby(level=0)
    }
    legs = {
        "c15_mid": quote(q15, dates, "C", bk["Kc"], "mid"),
        "p15_mid": quote(q15, dates, "P", bk["Kp"], "mid"),
        "c15_ask": quote(q15, dates, "C", bk["Kc"], "ask"),
        "p15_ask": quote(q15, dates, "P", bk["Kp"], "ask"),
        "c15_live": quote(q15, dates, "C", bk["Kc"], "live") == 1.0,
        "p15_live": quote(q15, dates, "P", bk["Kp"], "live") == 1.0,
    }
    print(
        f"15:30 quoted straddle at the 11:00 strikes: live on "
        f"{int((legs['c15_live'] & legs['p15_live']).sum())} of {len(dates)} days"
    )

    # ------------------------------------------------------------------ Q1 --
    q1, diag = q1_wings(bk, q11, q15, listed_c, listed_p, legs)
    q1 = q1.merge(diag.drop(columns=["w_frac"]), on="width", how="left")
    q1.to_csv(OUT / "q1_wings.csv", index=False)
    print("\n=== Q1  iron butterfly wings bought at 11:00 (tradeability) ===")
    print(diag.to_string(index=False))
    print("\n=== Q1  q1_wings.csv ===")
    print(q1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cand = q1[
        (q1["sample"] == "era")
        & (q1["basis"] == "crossed")
        & (q1["exit_mode"] == "hold")
        & q1["w_frac"].notna()
    ].copy()
    ok = cand[(cand["cap_units"] < BAR_CAP_UNITS) & (cand["Sharpe_ann"] > BAR_SHARPE)]
    pick = (ok if len(ok) else cand).sort_values("Sharpe_ann", ascending=False).iloc[0]
    best_label = str(pick["width"])
    best_wf = float(pick["w_frac"])
    print(
        f"\nQ1 pass bar (daily era, crossed, hold): Sharpe > {BAR_SHARPE} and "
        f"loss cap < {BAR_CAP_UNITS} units of premium -> {len(ok)} of "
        f"{len(cand)} widths pass; carried into Q2/Q3: {best_label}"
    )

    # ------------------------------------------------------------------ Q2 --
    wings = wing_legs(bk, q11, q15, listed_c, listed_p, best_wf)
    q2 = q2_cadence(bk, legs, wings, best_label)
    q2.to_csv(OUT / "q2_cadence.csv", index=False)
    print(
        "\n=== Q2  q2_cadence.csv  (wings unhedged; the delta is the straddle's only) ==="
    )
    print(q2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    keep = q2[(q2["cadence"] == "60min_top_of_hour")]["Sharpe_frac_of_30min"]
    print(
        f"Q2 pass bar: 60-minute keeps >= {BAR_CADENCE_KEEP:.0%} of the 30-minute "
        f"Sharpe on {int((keep >= BAR_CADENCE_KEEP).sum())} of {len(keep)} cells"
    )

    # ------------------------------------------------------------------ Q3 --
    q3, notes = q3_fees(bk, q1, best_label)
    q3 = q3.merge(pd.DataFrame(notes), on=["cadence", "exit_mode"], how="left")
    q3.to_csv(OUT / "q3_fees.csv", index=False)
    print("\n=== Q3  fee line ===")
    print(
        f"assumption: index option ${FEE_OPT_SIDE:.2f} per contract per side "
        f"($0.50 + $0.04); MES ${FEE_MES_BROKER_SIDE:.2f} broker + "
        f"${FEE_MES_EXCH_SIDE:.3f} exchange/regulatory (midpoint of the stated "
        f"$0.35-0.50 band) = ${FEE_MES_SIDE:.3f} per contract per side"
    )
    print(pd.DataFrame(notes).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Q3  q3_fees.csv ===")
    print(q3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nwrote", OUT / "q1_wings.csv")
    print("wrote", OUT / "q2_cadence.csv")
    print("wrote", OUT / "q3_fees.csv")


if __name__ == "__main__":
    main()
