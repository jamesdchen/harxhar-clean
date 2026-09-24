"""Study 83 -- the month-end long: is it more profitable bought in the morning than at 15:30?

The month-end leg buys the SPXW 0DTE nearest-OTM strangle (call at/above the
spot, put at/below) at 15:30 on the last session of each month, whatever the
price, and holds it to the 16:00 cash settlement.  It does not use the
forecast, so it could be placed at any time of day.  Here, per entry clock
from 09:35 to 15:30 (the SPXW chain's 30-minute snapshots, 2020-01 .. 2025-12;
its stamps are TRUE UTC), on every month-end with a 16:00 close:

* the strangle nearest the spot AT THAT CLOCK, bought at its ask, settled at
  the 16:00 SPX close;  R = payoff / ask - 1 is the return on the premium,
  which is also the return on the month-end budget (the leg is sized as a
  fraction of capital, so a costlier morning strangle buys fewer pairs);
* against 15:30 on the same month-ends: the paired difference in R, its t,
  and a Holm correction over the 12 earlier clocks (the clock is chosen
  after looking, so an uncorrected best clock is not evidence).

Study 69 asked this for 13:30-15:30 in points per contract (14:00 best,
multiplicity-corrected p 0.17, not adopted); this extends it to the morning
and to the budget-sized return the card actually earns.

    python writeup/intraday_proposals/83_month_end_entry_clock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.close_signal.schedule import is_early_close  # noqa: E402
from live.ibkr.calendar_guard import is_last_session_of_month  # noqa: E402

ET = "America/New_York"
CHAIN = REPO / "data" / "spxw_chain.parquet"
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "83"
CLOCKS = [
    "09:35", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
    "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
]  # fmt: skip
BASE = "15:30"


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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
    )
    days = sorted(
        d
        for d in c["expiration"].unique()
        if is_last_session_of_month(pd.Timestamp(d).date())
        and not is_early_close(pd.Timestamp(d).date())
    )
    c = c[c["expiration"].isin(days)]
    t = pd.to_datetime(c["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None)
    c = c.assign(clk=t.dt.strftime("%H:%M"), d=t.dt.normalize())
    c = c[c["d"] == c["expiration"]]  # same-day quotes of the 0DTE contracts
    rows = []
    for day, g in c.groupby("expiration"):
        close = g.loc[g["clk"] == "16:00", "underlying_price"]
        if close.empty:
            continue
        s_close = float(close.median())
        for clk in CLOCKS:
            b = g[(g["clk"] == clk) & (g["ask"] > 0)]
            if b.empty:
                continue
            s = float(b["underlying_price"].median())
            b = b.set_index(["strike", "cp"]).sort_index()
            ks = np.sort(b.index.get_level_values(0).unique().to_numpy(dtype=float))
            above, below = ks[ks >= s], ks[ks <= s]
            if not len(above) or not len(below):
                continue
            kc, kp = float(above.min()), float(below.max())
            try:
                ask = float(b.at[(kc, "C"), "ask"] + b.at[(kp, "P"), "ask"])
                bid = float(b.at[(kc, "C"), "bid"] + b.at[(kp, "P"), "bid"])
            except KeyError:
                continue
            payoff = max(s_close - kc, 0.0) + max(kp - s_close, 0.0)
            rows.append(
                {
                    "day": day,
                    "clock": clk,
                    "spot": s,
                    "kc": kc,
                    "kp": kp,
                    "ask": ask,
                    "rel_spread": (ask - bid) / ((ask + bid) / 2),
                    "s_close": s_close,
                    "payoff": payoff,
                    "R": payoff / ask - 1.0,
                    "pnl_pts": payoff - ask,
                }
            )
    x = pd.DataFrame(rows)
    x.to_csv(OUT / "month_end_by_clock.csv", index=False, float_format="%.6g")
    R = x.pivot(index="day", columns="clock", values="R")
    R = R.dropna()  # month-ends quoted at every clock: a paired comparison
    base = R[BASE]
    lines = [
        f"month-ends with a 16:00 close and a quote at every clock: {len(R)} "
        f"({R.index.min().date()} .. {R.index.max().date()})",
        "clock  ask_med  spread_med  mean_R    t_R   hit   diff_vs_1530  t_diff  p_holm",
    ]
    stats = []
    for clk in CLOCKS:
        r = R[clk]
        d = r - base
        t_r = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        t_d = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if clk != BASE else np.nan
        stats.append((clk, r.mean(), t_r, (r > 0).mean(), d.mean(), t_d))
    from scipy import stats as st  # noqa: PLC0415

    earlier = [s for s in stats if s[0] != BASE]
    p_raw = np.array([2 * st.t.sf(abs(s[5]), df=len(R) - 1) for s in earlier])
    p_adj = dict(zip([s[0] for s in earlier], holm(p_raw)))
    sub = x[x["day"].isin(R.index)]
    for clk, m, tr, hit, dm, td in stats:
        g = sub[sub["clock"] == clk]
        lines.append(
            f"{clk}  {g.ask.median():7.2f}  {g.rel_spread.median():9.1%}  {m:+7.3f}  {tr:5.2f}  {hit:4.0%}  "
            + (
                f"{dm:+12.3f}  {td:6.2f}  {p_adj[clk]:6.3f}"
                if clk != BASE
                else f"{'(base)':>12}"
            )
        )
    # stability: first half vs second half of the sample
    half = len(R) // 2
    for name, part in (("first half", R.iloc[:half]), ("second half", R.iloc[half:])):
        lines.append(
            f"{name} ({len(part)} month-ends, {part.index.min().date()} .. {part.index.max().date()}): "
            + ", ".join(
                f"{k} {part[k].mean():+.3f}"
                for k in ("09:35", "11:00", "13:00", "14:00", BASE)
            )
        )
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
