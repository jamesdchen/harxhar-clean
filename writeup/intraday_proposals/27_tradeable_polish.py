"""27 - tradeable polish on the 11:00 delta-hedged short straddle.

The book under test is the one the standalone PDF scores: every session
sells the nearest-OTM SPX 0DTE straddle at 11:00 ET, delta-hedges on the
vendor spot every 30 minutes with the hourly implied volatility
re-inverted on the censored vendor nodes, and either cash-settles or buys
the straddle back at 15:30.  Three questions, each with its own table:

  Q1  exit at 15:30 marked to the QUOTED bid/ask of the two 11:00 strikes
      instead of the Black-76 mark the standalone uses;
  Q2  hedge realism - the per-rebalance stock cost in basis points of
      S x |n|, and a one-bar execution lag on every rebalance;
  Q3  a calendar filter known before the 11:00 entry: q = 0 on FOMC
      statement days, on month-end sessions, and on both.

Plus the sizing note: median entry premium, dollars per contract, and the
Reg-T margin proxy.

The script begins by reproducing, and asserting, the published reference
numbers for the hold-to-settle and 15:30-model-mark books; nothing new is
computed until that gate passes.

Run:  python writeup/intraday_proposals/27_tradeable_polish.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "27"
CHAIN = REPO / "data" / "spxw_chain.parquet"

ENTRY = "11:00"
CLOSE = "15:30"
# the repo's daily-0DTE era split (experiments/spxw_delta_hedged_legs.py:57)
DAILY_0DTE = pd.Timestamp("2022-05-16")
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))

# Sensitivity axis for the hedge's stock cost, in basis points of S x |n|
# per rebalance.  0.5 bp is the repo's charged rate (UNDERLYING_COST_BP in
# writeup/intraday_proposals/25_dh_holdclose.py:32 and
# experiments/spxw_delta_hedged_legs.py:56); 0.0 is what the hedge tape
# behind the published Sharpes actually charges (attach_long_dh in
# writeup/make_rule_by_strategy_intraday_tex.py charges nothing).
COST_BP_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)

# Reg-T short-option initial requirement, as the regulation states it.
REG_T_PREMIUM_SHARE = 1.00  # 100% of the option premium
REG_T_INDEX_SHARE = 0.15  # 15% of the index value, less the out-of-the-money amount
CONTRACT_MULTIPLIER = 100.0  # dollars per index point, one SPX option contract

# The published reference numbers this script must reproduce before it
# computes anything new: (label, value, decimals as published).
REFERENCE = (
    ("n whole sample", 865.0, 0),
    ("hold Sharpe_ann mid", 4.20, 2),
    ("hold Sharpe_ann crossed", 3.58, 2),
    ("exit-15:30 model mark Sharpe_ann mid", 4.70, 2),
    ("exit-15:30 model mark Sharpe_ann crossed", 3.94, 2),
    ("exit-15:30 model mark mean mid", 0.094, 3),
    ("exit-15:30 model mark sd mid", 0.317, 3),
    ("n daily-0DTE era", 489.0, 0),
    ("era hold Sharpe_ann mid", 3.74, 2),
    ("era hold Sharpe_ann crossed", 3.32, 2),
    ("era exit-15:30 model mark Sharpe_ann mid", 4.23, 2),
    ("era exit-15:30 model mark Sharpe_ann crossed", 3.72, 2),
)


# --------------------------------------------------------------- stats ----
def _standalone():
    """The module that defines the published book (never edited, only read)."""
    path = REPO / "writeup" / "make_dh_causal_standalone_tex.py"
    spec = importlib.util.spec_from_file_location("p27_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def maxdd(x) -> float:
    """Worst peak-to-trough of the cumulative SUM path, peak seeded at 0."""
    v = pd.Series(x).dropna().sort_index().to_numpy(float)
    if v.size < 1:
        return float("nan")
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def stats(s: pd.Series, label: str, sample: str, **extra) -> dict:
    """n / mean / sd / Sharpe_ann / t / MaxDD / worst day of a daily series."""
    d = pd.Series(s).dropna().sort_index()
    n = int(d.size)
    mean = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n >= 2 else float("nan")
    sharpe = mean / sd * ANN if (sd and np.isfinite(sd) and sd > 0) else float("nan")
    t = (
        float(np.sqrt(n)) * mean / sd
        if (sd and np.isfinite(sd) and sd > 0)
        else float("nan")
    )
    row = {
        "book": label,
        "sample": sample,
        "n": n,
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": sharpe,
        "t": t,
        "MaxDD": maxdd(d),
        "worst_day": float(d.min()) if n else float("nan"),
        "worst_date": str(d.idxmin().date()) if n else "",
    }
    row.update(extra)
    return row


def both_samples(s: pd.Series, label: str, **extra) -> list[dict]:
    d = pd.Series(s).dropna()
    era = d[d.index >= DAILY_0DTE]
    return [stats(d, label, "whole", **extra), stats(era, label, "daily_era", **extra)]


def show(df: pd.DataFrame, cols: list[str], title: str) -> None:
    print(f"\n{title}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].to_string(index=False))


# ----------------------------------------------------------- the tape ----
def build_tape(pkg: pd.DataFrame) -> dict:
    """The 11:00 short's hedge tape, rebuilt from the standalone's own steps.

    Re-implements the arithmetic of hold_mark_1100 (the module is read, never
    edited) so the delta grid, the spot grid and the half-hour spot changes
    are available to re-cost and to lag.  Every object here is asserted
    against the module's own output before it is used.
    """
    std = _standalone()
    p26 = std._p26()
    m = std._rule_mod()
    deck = pd.read_parquet(m.DECK / "daily_blk2.parquet")
    days = pd.to_datetime(deck.index)
    p = pkg.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"].isin(days)].copy()
    clocks = sorted(p["hhmm"].unique())
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
    Sg = S_grid.to_numpy(float)
    IVg = IV_grid.to_numpy(float)
    ST = sl["S_close"].to_numpy(float)
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
    return {
        "std": std,
        "sl": sl,
        "dates": dates,
        "clocks": clocks,
        "j": j,
        "j15": j15,
        "Sg": Sg,
        "dlt": dlt,
        "dS": dS,
        "dS_f": dS_f,
        "mark": pd.Series(mark, index=dates),
        "entry": sl["entry"].astype(float),
        "bid_entry": sl["bid_entry"].astype(float)
        if "bid_entry" in sl.columns
        else (sl["bid_c"].astype(float) + sl["bid_p"].astype(float)),
    }


def hedge_and_cost(
    npos: np.ndarray, dS_used: np.ndarray, Sg: np.ndarray, cost_bp: float
):
    """Stock P&L of holding npos index units over each half hour, and its cost.

    npos[:, k] is the index position held from stamp k to stamp k+1 (the
    short straddle holds plus-delta).  The trade that establishes it happens
    at stamp k, at that stamp's spot, so the turnover charged at k is
    |npos_k - npos_{k-1}| x S_k, with npos_{-1} = 0 (flat before entry) and
    the final column's step to zero being the unwind.
    """
    hedge = (npos * dS_used).sum(axis=1)
    prev = np.zeros_like(npos)
    prev[:, 1:] = npos[:, :-1]
    S = np.where(np.isfinite(Sg), Sg, 0.0)
    cost = (np.abs(npos - prev) * S * cost_bp * 1e-4).sum(axis=1)
    return hedge, cost


# ------------------------------------------------------- 15:30 quotes ----
def quotes_1530(chain: Path) -> pd.DataFrame:
    """Every 0DTE bid/ask at the 15:30 ET stamp, keyed (expiration, strike, cp).

    The chain's timestamp is UTC; the ET clock and the ET date come from it.
    A 0DTE row is one whose expiration equals the ET date of its stamp.
    Only the columns needed are read, one row group at a time.
    """
    t0 = time.time()
    f = pq.ParquetFile(chain)
    cols = ["expiration", "strike", "cp", "bid", "ask", "timestamp", "underlying_price"]
    parts = []
    for i in range(f.num_row_groups):
        b = f.read_row_group(i, columns=cols).to_pandas()
        et = b["timestamp"].dt.tz_convert("America/New_York")
        # integer hour/minute, not strftime: the same mask, two orders of
        # magnitude faster over the 8.2M-row chain
        hh, mm = CLOSE.split(":")
        keep = (
            (et.dt.hour == int(hh))
            & (et.dt.minute == int(mm))
            & (b["expiration"] == et.dt.normalize().dt.tz_localize(None))
        )
        parts.append(
            b.loc[
                keep, ["expiration", "strike", "cp", "bid", "ask", "underlying_price"]
            ]
        )
    q = pd.concat(parts, ignore_index=True)
    assert not q.duplicated(["expiration", "strike", "cp"]).any(), (
        "duplicate 15:30 chain rows"
    )
    print(
        f"15:30 0DTE chain rows {len(q):,} on {q['expiration'].nunique():,} expirations "
        f"({time.time() - t0:.0f}s)"
    )
    return q


def leg_quotes(q: pd.DataFrame, expirations, strikes, cp: str) -> pd.DataFrame:
    """bid/ask of one leg, aligned to the (expiration, strike) asked for."""
    side = q.loc[q["cp"] == cp].set_index(["expiration", "strike"])
    key = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(expirations), np.asarray(strikes, float)]
    )
    return pd.DataFrame(
        {
            "bid": side["bid"].reindex(key).to_numpy(),
            "ask": side["ask"].reindex(key).to_numpy(),
        }
    )


# -------------------------------------------------------------- gate ----
def gate(std, pkg: pd.DataFrame):
    """Reproduce the published numbers; nothing new runs until this passes."""
    r_hold, r_mark, r_hold_x, r_mark_x = std.hold_mark_1100(pkg)
    h, mk = r_hold.dropna(), r_mark.dropna()
    hx, mkx = r_hold_x.dropna(), r_mark_x.dropna()

    def sh(d):
        return float(d.mean() / d.std(ddof=1) * ANN)

    era = lambda d: d[d.index >= DAILY_0DTE]  # noqa: E731
    got = {
        "n whole sample": float(h.size),
        "hold Sharpe_ann mid": sh(h),
        "hold Sharpe_ann crossed": sh(hx),
        "exit-15:30 model mark Sharpe_ann mid": sh(mk),
        "exit-15:30 model mark Sharpe_ann crossed": sh(mkx),
        "exit-15:30 model mark mean mid": float(mk.mean()),
        "exit-15:30 model mark sd mid": float(mk.std(ddof=1)),
        "n daily-0DTE era": float(era(h).size),
        "era hold Sharpe_ann mid": sh(era(h)),
        "era hold Sharpe_ann crossed": sh(era(hx)),
        "era exit-15:30 model mark Sharpe_ann mid": sh(era(mk)),
        "era exit-15:30 model mark Sharpe_ann crossed": sh(era(mkx)),
    }
    print("\nGATE - published reference numbers")
    for label, want, dec in REFERENCE:
        have = got[label]
        # the reference is published to `dec` decimals, so the gate is that
        # the rounded value matches it exactly (to 1e-6 on the rounded value)
        assert abs(round(have, dec) - want) < 1e-6, (label, have, want)
        print(f"  {label:<45s} {have:>12.6f}  reference {want:.{dec}f}  OK")
    # the hold book is also asserted against its persisted table, unrounded
    ref = pd.read_csv(
        HOLD / "rule_by_strategy_dh" / "1100" / "rule_by_strategy_always_short.csv",
        index_col=0,
    ).loc["all models"]
    assert int(h.size) == int(ref["n"]), (int(h.size), int(ref["n"]))
    assert abs(sh(h) - float(ref["Sharpe_ann"])) < 1e-6, (
        sh(h),
        float(ref["Sharpe_ann"]),
    )
    assert abs(sh(hx) - float(ref["Sharpe_crossed"])) < 1e-6, (
        sh(hx),
        float(ref["Sharpe_crossed"]),
    )
    print("  rule_by_strategy_dh_holdclose/1100 always-short table matched to 1e-6")
    print("GATE PASSED\n")
    return r_hold, r_mark, r_hold_x, r_mark_x


# -------------------------------------------------------------- main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    std = _standalone()
    r_hold, r_mark, r_hold_x, r_mark_x = gate(std, pkg)

    tape = build_tape(pkg)
    idx = r_hold.dropna().index
    sl = tape["sl"].reindex(idx)
    entry = tape["entry"].reindex(idx).to_numpy(float)
    bid_entry = tape["bid_entry"].reindex(idx).to_numpy(float)
    mark = tape["mark"].reindex(idx).to_numpy(float)
    rows = tape["dates"].get_indexer(idx)
    Sg, dlt, dS, dS_f = (tape[k][rows] for k in ("Sg", "dlt", "dS", "dS_f"))
    j15 = tape["j15"]

    # the exit-at-15:30 hedge: plus-delta of the index from 11:00, flat at
    # 15:30 (no rebalance at 15:30, the last stock step is dropped)
    npos_f = dlt.copy()
    npos_f[:, j15:] = 0.0
    hedge_f, _ = hedge_and_cost(npos_f, dS_f, Sg, 0.0)
    r_mark_re = pd.Series((-(mark - entry) + hedge_f) / entry, index=idx)
    assert np.nanmax(np.abs(r_mark_re - r_mark.reindex(idx))) < 1e-12, (
        "hedge tape mismatch"
    )
    hedge_h, _ = hedge_and_cost(dlt, dS, Sg, 0.0)
    r_hold_re = pd.Series(
        (-(sl["exit"].to_numpy(float) - entry) + hedge_h) / entry, index=idx
    )
    assert np.nanmax(np.abs(r_hold_re - r_hold.reindex(idx))) < 1e-12, (
        "hold tape mismatch"
    )
    print(
        "tape rebuilt: exit-15:30 and hold-to-settle hedges reproduce the module to 1e-12"
    )

    # ------------------------------------------------------------ Q1 ----
    q = quotes_1530(CHAIN)
    exp = pd.DatetimeIndex(sl["expiration"])
    assert (exp == pd.DatetimeIndex(idx)).all(), (
        "the 11:00 row's expiration is not its own date"
    )
    qc = leg_quotes(q, exp, sl["K_c"].to_numpy(float), "C")
    qp = leg_quotes(q, exp, sl["K_p"].to_numpy(float), "P")
    no_row = qc["bid"].isna().to_numpy() | qp["bid"].isna().to_numpy()
    # the vendor's no-quote sentinel is bid == ask == 0 (asl.quote_mid)
    sent_c = (qc["bid"].to_numpy() == 0) & (qc["ask"].to_numpy() == 0)
    sent_p = (qp["bid"].to_numpy() == 0) & (qp["ask"].to_numpy() == 0)
    dead = no_row | sent_c | sent_p
    live = ~dead
    print(
        f"\n15:30 quotes on the two 11:00 strikes: {int(live.sum())} of {len(idx)} days live; "
        f"{int(no_row.sum())} days with no chain row, "
        f"{int((sent_c & ~no_row).sum())} call / {int((sent_p & ~no_row).sum())} put "
        f"no-quote sentinels (bid == ask == 0)"
    )

    mid_q = (
        asl.quote_mid(qc["bid"], qc["ask"]).to_numpy()
        + asl.quote_mid(qp["bid"], qp["ask"]).to_numpy()
    )
    ask_q = qc["ask"].to_numpy() + qp["ask"].to_numpy()
    mid_q = np.where(live, mid_q, np.nan)
    ask_q = np.where(live, ask_q, np.nan)

    r_exit_mid_q = pd.Series(
        np.where(live, (-(mid_q - entry) + hedge_f) / entry, np.nan), index=idx
    )
    r_exit_x_q = pd.Series(
        np.where(live, (-(ask_q - bid_entry) + hedge_f) / entry, np.nan), index=idx
    )
    # days without a live 15:30 quote fall back to holding through settlement
    r_exit_mid_q_fb = r_exit_mid_q.where(
        pd.Series(live, index=idx), r_hold.reindex(idx)
    )
    r_exit_x_q_fb = r_exit_x_q.where(pd.Series(live, index=idx), r_hold_x.reindex(idx))

    q1 = []
    for lab, ser in (
        ("hold through cash-settle, mid", r_hold.reindex(idx)),
        ("hold through cash-settle, crossed", r_hold_x.reindex(idx)),
        ("exit 15:30 at the model mark, mid", r_mark.reindex(idx)),
        ("exit 15:30 at the model mark, crossed", r_mark_x.reindex(idx)),
        ("exit 15:30 at the quoted mid", r_exit_mid_q),
        ("exit 15:30 at the quoted ask, entry at the bid (crossed)", r_exit_x_q),
        ("exit 15:30 at the quoted mid, hold when no quote", r_exit_mid_q_fb),
        ("exit 15:30 crossed, hold when no quote", r_exit_x_q_fb),
    ):
        q1 += [dict(block="book", **r) for r in both_samples(ser, lab)]
    q1 = pd.DataFrame(q1)

    # the model mark against the quoted mid
    gap_pts = pd.Series(np.where(live, mark - mid_q, np.nan), index=idx).dropna()
    gap_prem = pd.Series(
        np.where(live, (mark - mid_q) / entry, np.nan), index=idx
    ).dropna()
    gap_rows = []
    for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", DAILY_0DTE)):
        gp = gap_pts[gap_pts.index >= cut]
        gr = gap_prem[gap_prem.index >= cut]
        gap_rows.append(
            {
                "block": "gap",
                "book": "model mark minus quoted mid",
                "sample": sample,
                "n": int(gp.size),
                "median_points": float(gp.median()),
                "p95_points": float(gp.quantile(0.95)),
                "median_premium_units": float(gr.median()),
                "p95_premium_units": float(gr.quantile(0.95)),
                "mean_points": float(gp.mean()),
                "min_points": float(gp.min()),
                "max_points": float(gp.max()),
            }
        )
    # what the crossed book pays: the 15:30 half-spread it buys back over, and
    # the 11:00 half-spread it sells into
    half_15 = pd.Series(np.where(live, ask_q - mid_q, np.nan), index=idx).dropna()
    half_11 = pd.Series(entry - bid_entry, index=idx)
    for lab, s in (
        ("15:30 ask minus quoted mid", half_15),
        ("11:00 mid minus bid", half_11),
    ):
        for sample, cut in (("whole", pd.Timestamp.min), ("daily_era", DAILY_0DTE)):
            d = s[s.index >= cut]
            dp = (s / pd.Series(entry, index=idx))[s.index >= cut]
            gap_rows.append(
                {
                    "block": "spread",
                    "book": lab,
                    "sample": sample,
                    "n": int(d.size),
                    "median_points": float(d.median()),
                    "p95_points": float(d.quantile(0.95)),
                    "median_premium_units": float(dp.median()),
                    "p95_premium_units": float(dp.quantile(0.95)),
                    "mean_points": float(d.mean()),
                    "min_points": float(d.min()),
                    "max_points": float(d.max()),
                }
            )
    q1 = pd.concat([q1, pd.DataFrame(gap_rows)], ignore_index=True)
    q1.to_csv(OUT / "q1_exit_quote.csv", index=False)
    show(
        q1[q1["block"] == "book"],
        [
            "book",
            "sample",
            "n",
            "mean",
            "sd",
            "Sharpe_ann",
            "t",
            "MaxDD",
            "worst_day",
            "worst_date",
        ],
        "Q1  exit at 15:30, model mark vs quoted",
    )
    show(
        q1[q1["block"].isin(["gap", "spread"])],
        [
            "book",
            "sample",
            "n",
            "median_points",
            "p95_points",
            "median_premium_units",
            "p95_premium_units",
            "mean_points",
            "min_points",
            "max_points",
        ],
        "Q1  the 15:30 gap and the crossed half-spreads (index points; premium units)",
    )
    era_x = r_exit_x_q.dropna()
    era_x = era_x[era_x.index >= DAILY_0DTE]
    sh_q1 = float(era_x.mean() / era_x.std(ddof=1) * ANN)
    print(
        f"\nQ1 verdict: crossed quoted exit, daily era Sharpe_ann {sh_q1:.3f} "
        f"(bar > 3.0) -> {'PASS' if sh_q1 > 3.0 else 'FAIL'}"
    )

    # ------------------------------------------------------------ Q2 ----
    # one-bar execution lag: the delta computed at stamp k is traded at k+1,
    # so the position held over [k, k+1) is the delta computed at k-1
    npos_lag = np.zeros_like(npos_f)
    npos_lag[:, 1:] = npos_f[:, :-1]
    npos_lag[:, j15:] = 0.0
    # the same lag with the OPENING hedge still done at 11:00 on the entry
    # ticket, so only the later rebalances are late: this separates the cost
    # of a stale delta from the cost of an unhedged first half hour
    npos_lag_e = npos_lag.copy()
    npos_lag_e[:, tape["j"]] = npos_f[:, tape["j"]]
    q2 = []
    for lag, npos in (
        ("none", npos_f),
        ("one bar", npos_lag),
        ("one bar, 11:00 hedge on time", npos_lag_e),
    ):
        for bp in COST_BP_GRID:
            hedge, cost = hedge_and_cost(npos, dS_f, Sg, bp)
            r = pd.Series(
                np.where(live, (-(ask_q - bid_entry) + hedge - cost) / entry, np.nan),
                index=idx,
            )
            lab = f"crossed quoted exit, {bp:.1f} bp, lag {lag}"
            for row in both_samples(r, lab):
                row["cost_bp"] = bp
                row["lag"] = lag
                row["mean_cost_points"] = float(np.nanmean(cost))
                q2.append(row)
    q2 = pd.DataFrame(q2)
    q2.to_csv(OUT / "q2_hedge_grid.csv", index=False)
    show(
        q2,
        [
            "lag",
            "cost_bp",
            "sample",
            "n",
            "mean",
            "sd",
            "Sharpe_ann",
            "t",
            "MaxDD",
            "worst_day",
            "mean_cost_points",
        ],
        "Q2  hedge realism: stock cost in bp of S x |n| per rebalance, and a one-bar lag",
    )
    pick = lambda lag, bp, smp: float(  # noqa: E731
        q2[(q2["lag"] == lag) & (q2["cost_bp"] == bp) & (q2["sample"] == smp)][
            "Sharpe_ann"
        ].iloc[0]
    )
    s_2bp = pick("none", 2.0, "daily_era")
    s_lag = pick("one bar", 0.5, "daily_era")
    print(
        f"\nQ2 verdict: daily era, 2 bp no lag Sharpe_ann {s_2bp:.3f} (bar > 3.0) -> "
        f"{'PASS' if s_2bp > 3.0 else 'FAIL'}; one-bar lag at 0.5 bp {s_lag:.3f} (bar > 2.5) -> "
        f"{'PASS' if s_lag > 2.5 else 'FAIL'}"
    )

    # ------------------------------------------------------------ Q3 ----
    # the deck's own flags: FOMC statement days and month-end sessions, from
    # the exchange calendar alone, so the position is known before 11:00.
    # sessions = the deck's full trading-day list, so a month end is not
    # slid earlier by the days this book does not trade.
    m = std._rule_mod()
    sessions = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(m.DECK / "daily_blk2.parquet").index)
    )
    flags = asl.fomc_and_monthend(pd.DatetimeIndex(idx), REPO, sessions=sessions)
    known_until = flags.attrs["fomc_known_until"]
    is_fomc = flags["is_fomc"].fillna(False).to_numpy(bool)
    is_me = flags["is_me"].to_numpy(bool)
    print(
        f"\ncalendar flags: {int(is_fomc.sum())} FOMC statement days, {int(is_me.sum())} month-end "
        f"sessions, {int((is_fomc | is_me).sum())} either, on {len(idx)} scored days; "
        f"FOMC knowledge horizon {pd.Timestamp(known_until).date()} "
        f"(all {int(flags['fomc_known'].sum())} scored days are inside it)"
    )

    q3 = []
    for base_lab, base in (
        ("crossed quoted exit 15:30", r_exit_x_q),
        ("hold through cash-settle, crossed", r_hold_x.reindex(idx)),
    ):
        for filt_lab, flat in (
            ("no filter", np.zeros(len(idx), bool)),
            ("q = 0 on FOMC statement days", is_fomc),
            ("q = 0 on month-end sessions", is_me),
            ("q = 0 on both", is_fomc | is_me),
        ):
            # flat days stay in the series as zeros, so every filter is
            # scored on the same days (the deck's convention)
            r = base.copy()
            r[pd.Series(flat, index=idx) & base.notna()] = 0.0
            traded = base.notna().to_numpy() & ~flat
            for row in both_samples(r, f"{base_lab} | {filt_lab}"):
                smp = pd.Series(traded, index=idx)
                if row["sample"] == "daily_era":
                    smp = smp[smp.index >= DAILY_0DTE]
                row["n_traded"] = int(smp.sum())
                row["base"] = base_lab
                row["filter"] = filt_lab
                q3.append(row)
    q3 = pd.DataFrame(q3)
    q3.to_csv(OUT / "q3_calendar.csv", index=False)
    show(
        q3,
        [
            "base",
            "filter",
            "sample",
            "n",
            "n_traded",
            "mean",
            "Sharpe_ann",
            "t",
            "MaxDD",
            "worst_day",
            "worst_date",
        ],
        "Q3  calendar filter known before the 11:00 entry",
    )

    print("\nQ3  the removed days themselves, crossed quoted exit 15:30")
    for lab, mask in (
        ("FOMC statement days", is_fomc),
        ("month-end sessions", is_me),
        ("either", is_fomc | is_me),
        ("all other days", ~(is_fomc | is_me)),
    ):
        d = r_exit_x_q[pd.Series(mask, index=idx)].dropna()
        de = d[d.index >= DAILY_0DTE]
        print(
            f"  {lab:<22s} whole n {d.size:>4d} mean {d.mean():+.4f} worst {d.min():+.3f} "
            f"({d.idxmin().date()}) | era n {de.size:>4d} mean {de.mean():+.4f} "
            f"worst {de.min():+.3f}"
        )
    base_e = q3[
        (q3["base"] == "crossed quoted exit 15:30") & (q3["sample"] == "daily_era")
    ]
    b0 = base_e[base_e["filter"] == "no filter"].iloc[0]
    verdicts = []
    for f_lab in (
        "q = 0 on FOMC statement days",
        "q = 0 on month-end sessions",
        "q = 0 on both",
    ):
        r = base_e[base_e["filter"] == f_lab].iloc[0]
        ok = (r["Sharpe_ann"] > b0["Sharpe_ann"]) and (r["worst_day"] > b0["worst_day"])
        verdicts.append((f_lab, r["Sharpe_ann"], r["worst_day"], ok))
    print(
        f"\nQ3 verdict (daily era; bar = raises Sharpe AND lifts the worst day; "
        f"unfiltered Sharpe {b0['Sharpe_ann']:.3f}, worst {b0['worst_day']:.3f}):"
    )
    for f_lab, s, w, ok in verdicts:
        print(
            f"  {f_lab:<32s} Sharpe {s:.3f} worst {w:+.3f} -> {'PASS' if ok else 'FAIL'}"
        )
    base_h = q3[
        (q3["base"] == "hold through cash-settle, crossed")
        & (q3["sample"] == "daily_era")
    ]
    h0 = base_h[base_h["filter"] == "no filter"].iloc[0]
    print(
        f"  (hold-to-settle reference, daily era; unfiltered Sharpe "
        f"{h0['Sharpe_ann']:.3f}, worst {h0['worst_day']:.3f}):"
    )
    for f_lab in (
        "q = 0 on FOMC statement days",
        "q = 0 on month-end sessions",
        "q = 0 on both",
    ):
        r = base_h[base_h["filter"] == f_lab].iloc[0]
        ok = (r["Sharpe_ann"] > h0["Sharpe_ann"]) and (r["worst_day"] > h0["worst_day"])
        print(
            f"    {f_lab:<32s} Sharpe {r['Sharpe_ann']:.3f} worst {r['worst_day']:+.3f} -> "
            f"{'both up' if ok else 'not both up'}"
        )

    # -------------------------------------------------------- sizing ----
    prem = pd.Series(entry, index=idx)
    S11 = sl["S"].astype(float)
    Kc = sl["K_c"].astype(float)
    Kp = sl["K_p"].astype(float)
    prem_c = asl.quote_mid(sl["bid_c"], sl["ask_c"])
    prem_p = asl.quote_mid(sl["bid_p"], sl["ask_p"])
    # Reg-T proxy, one short straddle: the larger leg's requirement
    #   premium + 15% of the index value - the out-of-the-money amount
    # plus the other leg's premium, all times the $100 multiplier.
    otm_c = (Kc - S11).clip(lower=0.0)
    otm_p = (S11 - Kp).clip(lower=0.0)
    req_c = REG_T_PREMIUM_SHARE * prem_c + REG_T_INDEX_SHARE * S11 - otm_c
    req_p = REG_T_PREMIUM_SHARE * prem_p + REG_T_INDEX_SHARE * S11 - otm_p
    larger_is_call = req_c >= req_p
    margin_pts = np.where(larger_is_call, req_c + prem_p, req_p + prem_c)
    margin_usd = pd.Series(margin_pts * CONTRACT_MULTIPLIER, index=idx)

    size_rows = []
    for lab, ser in (
        ("hold through cash-settle, crossed", r_hold_x.reindex(idx)),
        ("exit 15:30 at the quoted mid", r_exit_mid_q),
        ("exit 15:30 at the quoted ask, entry at the bid (crossed)", r_exit_x_q),
    ):
        dollars = (ser * prem * CONTRACT_MULTIPLIER).dropna()
        for smp, d in (
            ("whole", dollars),
            ("daily_era", dollars[dollars.index >= DAILY_0DTE]),
        ):
            p = prem.reindex(d.index)
            mg = margin_usd.reindex(d.index)
            size_rows.append(
                {
                    "book": lab,
                    "sample": smp,
                    "n": int(d.size),
                    "median_entry_premium_points": float(p.median()),
                    "median_entry_premium_usd": float(p.median() * CONTRACT_MULTIPLIER),
                    "median_spot_points": float(S11.reindex(d.index).median()),
                    "mean_pnl_usd_per_contract": float(d.mean()),
                    "sd_pnl_usd_per_contract": float(d.std(ddof=1)),
                    "maxdd_usd_per_contract": maxdd(d),
                    "worst_day_usd_per_contract": float(d.min()),
                    "median_regt_margin_usd": float(mg.median()),
                    "mean_pnl_over_median_margin": float(d.mean() / mg.median()),
                }
            )
    sizing = pd.DataFrame(size_rows)
    sizing.to_csv(OUT / "sizing.csv", index=False)
    show(
        sizing,
        [
            "book",
            "sample",
            "n",
            "median_entry_premium_points",
            "median_entry_premium_usd",
            "median_spot_points",
            "mean_pnl_usd_per_contract",
            "maxdd_usd_per_contract",
            "worst_day_usd_per_contract",
            "median_regt_margin_usd",
            "mean_pnl_over_median_margin",
        ],
        "Sizing, one contract (SPX multiplier $100 per index point)",
    )
    print(
        "\nReg-T margin proxy (not broker margin): for the leg with the larger "
        "requirement, 100% of that leg's premium + 15% of the index value - that "
        "leg's out-of-the-money amount, plus the other leg's premium; x $100."
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
