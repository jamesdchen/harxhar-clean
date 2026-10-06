"""Study 91 -- quarter-ends (and quarterly expiration Fridays) instead of month-ends.

The month-end long (buy the SPXW 0DTE nearest-OTM strangle at 15:30 at the
ask, hold to the close) faded after 2024.  Quarter-ends carry bigger
rebalancing (quarterly pension and fund rebalancing, window dressing) and
the third Friday of Mar/Jun/Sep/Dec carries the S&P quarterly index
rebalance and quarterly expiration -- the largest closing auctions of the
quarter.  Here, three day types against ordinary days:

  QE   last session of Mar/Jun/Sep/Dec
  ME   other month-ends
  QX   third Friday of Mar/Jun/Sep/Dec (quarterly expiration, index rebalance)

A. the strangle at 15:30 at the ask, held to the close: month-ends from study
   83 (SPXW chain, 2020-2025) plus 2026 from the SPXW OPRA files (2026-09-30
   from the operator's 15:37 quote, 6.80 on 7690P/7695C, until its file is
   pulled); QX and ordinary days from study 88's per-day table;
B. the ES 15:30-16:00 bar since 1998 (the vendor panel + the store): median
   variance and median |return| against ordinary days, by era.

    python writeup/intraday_proposals/91_quarter_end.py
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.ibkr.calendar_guard import is_last_session_of_month  # noqa: E402

P = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals"
OUT = P / "91"


def third_friday(d: date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def kind(d: date) -> str:
    if is_last_session_of_month(d):
        return "QE" if d.month in (3, 6, 9, 12) else "ME"
    if d.month in (3, 6, 9, 12) and third_friday(d):
        return "QX"
    return "ordinary"


def strangle_month_ends() -> pd.DataFrame:
    me = pd.read_csv(P / "83" / "month_end_by_clock.csv", parse_dates=["day"])
    m = me[me["clock"] == "15:30"].set_index("day")[["ask", "payoff"]].copy()
    spec = importlib.util.spec_from_file_location(
        "s79", REPO / "writeup" / "intraday_proposals" / "79_xsp_close_book.py"
    )
    assert spec is not None and spec.loader is not None
    s79 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s79)
    close = pd.read_csv(P / "79" / "spx_close_yf.csv", index_col=0, parse_dates=True)[
        "close"
    ]
    for f in sorted(
        glob.glob(str(REPO / "data/archive/spxw_opra/cbbo1m_2026-*.parquet"))
    ):
        day = pd.Timestamp(Path(f).stem.split("_")[1])
        if not is_last_session_of_month(day.date()) or day not in close.index:
            continue
        q = s79.load_quotes(Path(f))
        b = s79.book_at(q, day, "15:30:00")
        s = s79.parity_spot(b)
        ks = np.sort(b.index.get_level_values(0).unique().to_numpy(float))
        kc, kp = s79.nearest_otm(ks, s)
        ask = s79.quote(b, kc, "C")[1] + s79.quote(b, kp, "P")[1]
        sc = float(close.loc[day])
        m.loc[day] = [ask, max(sc - kc, 0.0) + max(kp - sc, 0.0)]
    sep30 = pd.Timestamp("2026-09-30")
    if sep30 not in m.index:  # the operator's quote until the file is pulled
        m.loc[sep30] = [6.80, max(7667.93 - 7695.0, 0.0) + max(7690.0 - 7667.93, 0.0)]
    m["R"] = m["payoff"] / m["ask"] - 1.0
    return m.sort_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    # ---- A. the strangle
    m = strangle_month_ends()
    m["kind"] = [kind(d.date()) for d in m.index]
    d = pd.read_parquet(P / "88" / "days.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")
    d = d[np.isfinite(d["R"]) & (d["ask"] > 0)]
    d["kind"] = [kind(x.date()) for x in d.index]
    allr = pd.concat([m[["R", "kind"]], d.loc[d["kind"] != "ME", ["R", "kind"]]])
    allr = allr[~allr.index.duplicated(keep="first")]
    lines.append(
        "A. the 15:30 SPXW strangle at the ask, held to the close (R = return on the premium)"
    )
    lines.append("kind      period     n    mean R    t      hit   share >= 3x")
    for k in ("QE", "ME", "QX", "ordinary"):
        for lab, a, b in (
            ("2020-2024", "2020", "2024"),
            ("2025-2026", "2025", "2026"),
            ("all", "2020", "2026"),
        ):
            x = allr[allr["kind"] == k]
            yr = x.index.year
            g = x.loc[(yr >= int(a)) & (yr <= int(b)), "R"]
            if len(g) < 2:
                continue
            t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
            lines.append(
                f"{k:9s} {lab:9s} {len(g):4d}  {g.mean():+.3f}  {t:+5.2f}  {(g > 0).mean():4.0%}  {(g >= 2).mean():5.1%}"
            )
    qe = m[m["kind"] == "QE"]
    lines.append(
        "\nquarter-ends one by one: "
        + ", ".join(f"{x.date()} {r:+.2f}" for x, r in qe["R"].items())
    )
    # ---- B. the ES last half hour since 1998
    v = pd.read_parquet(
        REPO / "data" / "core_stats.parquet",
        columns=["endbartime", "sumret2", "sumret"],
    )
    s = pd.read_parquet(
        REPO / "live" / "close_signal" / "state" / "panel_free.parquet",
        columns=["endbartime", "sumret2", "sumret", "source"],
    )
    s = s[s["source"].isin(["databento_es", "yahoo_es"])].drop(columns="source")
    p = pd.concat([v, s])
    p["t"] = pd.to_datetime(p["endbartime"])
    p = p.drop_duplicates("t", keep="last").set_index("t").sort_index()
    last = p[p.index.strftime("%H:%M") == "16:00"].copy()
    last.index = last.index.normalize()
    last = last[last.index.year >= 1998]
    last["kind"] = [kind(x.date()) for x in last.index]
    lines.append("\nB. the ES 15:30-16:00 bar vs ordinary days (medians)")
    lines.append("period      kind   n    variance ratio   |return| ratio")
    for a, b in (
        ("1998", "2009"),
        ("2010", "2019"),
        ("2020", "2024"),
        ("2025", "2026"),
    ):
        g = last[(last.index.year >= int(a)) & (last.index.year <= int(b))]
        o = g[g["kind"] == "ordinary"]
        for k in ("QE", "ME", "QX"):
            x = g[g["kind"] == k]
            if len(x) < 2:
                continue
            lines.append(
                f"{a}-{b}   {k:4s} {len(x):4d}   {x.sumret2.median() / o.sumret2.median():6.2f}x        "
                f"{x.sumret.abs().median() / o.sumret.abs().median():6.2f}x"
            )
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
