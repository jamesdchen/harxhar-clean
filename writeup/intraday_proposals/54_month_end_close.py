"""54 - the last half hour on the last trading day of the month.

How this was found (stated because it matters for how far to trust it): the
15:30 sign(s) straddle trade's P&L was decomposed day by day.  20 of its 866
days carry 89% of the total, the ridge was LONG on only 11 of those 20, and
three of the nine it was short on were the last trading day of a month.  The
calendar rule below was therefore suggested by the deck sample.  Everything
after Part A is the attempt to break it on data that played no part in that.

The mechanism is not new: month-end portfolio rebalancing is executed at the
closing auction, so the last half hour of the last trading day of a month
carries an unusually large NET move.

  A  the decomposition on the deck days (reproduces the deck's 1.338 / 0.870):
     concentration of the P&L, which of the top days each forecast was long,
     and where the eight forecasts disagree.
  B  the underlying, 1998-2024, no options: the last bar's realized variance
     and squared net move relative to the bar before it, month-end against
     other days, split at 2020 so the pre-2020 half never touched an option.
  C  forecast-free, every chain session 2020-01-03 .. 2025-12-31: long the
     15:30 nearest-OTM straddle on month-end days, at the mid and at the quoted
     ask, in premium units and index points per contract; the 413 sessions from
     2024-05-01 are the holdout (the forecasts, the deck and Part A stop on
     2024-04-30).
  D  what it does to the books: sign(s) with a month-end override (long, or
     flat), and the short afternoon hold books of proposal 43 on month-end days.

GATE  the chain-built 15:30 straddle return equals the deck's R on every deck
      day, and the ridge's sign(s) Sharpe is the deck's 1.338322 / 0.869588.

Run:  python writeup/intraday_proposals/54_month_end_close.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = DECK / "proposals" / "54"
P43_DAILY = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "43"
    / "a_daily_by_clock.csv"
)
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
DECK_END, OOS_START, PRE_OPTIONS_END = "2024-04-30", "2024-05-01", "2019-12-31"
TOP_DAYS = 20
GATE_SHARPE = (1.338322, 0.869588)
GATE_TOL = 1e-6
GATE_R_TOL = 1e-6  # the chain stores quotes as float32


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def tstat(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def month_end_flags(days: pd.DatetimeIndex) -> pd.Series:
    """True on the last session of each calendar month in ``days``."""
    s = pd.Series(days, index=days)
    return s.groupby([days.year, days.month]).transform("max") == s


def part_a() -> pd.DataFrame:
    books = {
        t: pd.read_parquet(DECK / f"daily_{t}.parquet").sort_index()
        for t in asl.MODEL_ORDER
    }
    d = books["blk2"]
    r = d["R"].to_numpy(float)
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    ex = d["exit"].to_numpy(float)
    signs = {
        t: np.where(
            b["rv_hat"].to_numpy(float) > d["iv_var"].to_numpy(float), 1.0, -1.0
        )
        for t, b in books.items()
    }

    def crossed(q: np.ndarray) -> np.ndarray:
        return np.where(q > 0, q * (ex / ask - 1.0), q * (ex / bid - 1.0))

    q0 = signs["blk2"]
    got = (sharpe(q0 * r), sharpe(crossed(q0)))
    assert abs(got[0] - GATE_SHARPE[0]) < GATE_TOL, got
    assert abs(got[1] - GATE_SHARPE[1]) < GATE_TOL, got
    print(
        f"GATE  ridge sign(s) on the deck days: {got[0]:.6f} mid / {got[1]:.6f} crossed"
    )
    top = np.argsort(r)[::-1][:TOP_DAYS]
    pnl = q0 * r
    print(
        f"A  {len(r)} days; total mid P&L {pnl.sum():.1f} premium units, of which the "
        f"{TOP_DAYS} largest days {np.sort(pnl)[::-1][:TOP_DAYS].sum():.1f}; the long "
        f"straddle is positive on {float((r > 0).mean()):.0%} of days, median "
        f"{np.median(r):+.2f}, and averages {r[top].mean():.1f}x premium on its "
        f"{TOP_DAYS} best days"
    )
    stack = np.array(list(signs.values()))
    agree = np.abs(stack.sum(axis=0)) == len(stack)
    rows = [
        {
            "rule": f"{t} sign(s)",
            "buy_share": float((s > 0).mean()),
            f"long_on_top{TOP_DAYS}": int((s[top] > 0).sum()),
            "Sharpe_mid": sharpe(s * r),
            "Sharpe_crossed": sharpe(crossed(s)),
        }
        for t, s in signs.items()
    ]
    for name, q in (
        ("average of the eight signs", stack.mean(axis=0)),
        ("ridge, only when all eight agree", np.where(agree, q0, 0.0)),
        ("ridge, only when they disagree", np.where(agree, 0.0, q0)),
    ):
        rows.append(
            {
                "rule": name,
                "buy_share": float((q > 0).mean()),
                f"long_on_top{TOP_DAYS}": int((q[top] > 0).sum()),
                "Sharpe_mid": sharpe(q * r),
                "Sharpe_crossed": sharpe(crossed(q)),
            }
        )
    me = month_end_flags(pd.DatetimeIndex(d.index)).to_numpy()
    print(
        f"   of the {TOP_DAYS} best long-straddle days {int(me[top].sum())} are month-ends "
        f"({int(me.sum())} month-ends in all); the ridge is long on "
        f"{float((q0[me] > 0).mean()):.0%} of month-ends against {float((q0 > 0).mean()):.0%} "
        f"of all days"
    )
    return pd.DataFrame(rows)


def part_b() -> pd.DataFrame:
    c = pd.read_parquet(ROOT / "data" / "core_stats.parquet")
    c["t"] = pd.to_datetime(c["endbartime"])  # naive ET, bar-END labelled
    c = c[c["t"] >= "1998-01-05"]
    c["date"] = c["t"].dt.normalize()
    c["hhmm"] = c["t"].dt.strftime("%H:%M")
    p = c[c["hhmm"].isin(["15:30", "16:00"])].pivot_table(
        index="date", columns="hhmm", values=["sumret", "sumret2"], aggfunc="first"
    )
    p = p[(p[("sumret2", "15:30")] > 0) & (p[("sumret2", "16:00")] > 0)].dropna()
    days = pd.DatetimeIndex(p.index)
    me = month_end_flags(days).to_numpy()
    log_rv = np.log(p[("sumret2", "16:00")] / p[("sumret2", "15:30")]).to_numpy()
    net2 = (p[("sumret", "16:00")] ** 2 / p[("sumret2", "15:30")]).to_numpy()
    rows = []
    for name, m in (
        ("1998-2019 (before any option in the study)", days <= PRE_OPTIONS_END),
        ("2020-2024", days > PRE_OPTIONS_END),
    ):
        a, b = log_rv[m & me], log_rv[m & ~me]
        t = (a.mean() - b.mean()) / np.sqrt(
            a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
        )
        rows.append(
            {
                "sample": name,
                "month_ends": int((m & me).sum()),
                "other_days": int((m & ~me).sum()),
                "last_bar_RV_over_prior_bar_month_end": float(np.exp(a.mean())),
                "last_bar_RV_over_prior_bar_other": float(np.exp(b.mean())),
                "t_log_ratio": float(t),
                "median_net_move_sq_over_prior_RV_month_end": float(
                    np.median(net2[m & me])
                ),
                "median_net_move_sq_over_prior_RV_other": float(
                    np.median(net2[m & ~me])
                ),
            }
        )
    return pd.DataFrame(rows)


def part_c() -> tuple[pd.DataFrame, pd.DataFrame]:
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    k = p43.CLOCKS.index("15:30")
    mid, bid = ch["entry"][:, k], ch["bid"][:, k]
    ask = 2.0 * mid - bid
    sc = ch["S_close"]
    settle = np.maximum(sc - ch["K_c"][:, k], 0.0) + np.maximum(
        ch["K_p"][:, k] - sc, 0.0
    )
    ok = np.isfinite(mid) & (mid > 0) & (bid > 0) & np.isfinite(settle)
    df = pd.DataFrame(
        {
            "R_mid": settle / mid - 1.0,
            "R_ask": settle / ask - 1.0,
            "pts_ask": settle - ask,
        },
        index=ch["dates"],
    )[ok]
    deck = pd.read_parquet(DECK / "daily_blk2.parquet")["R"]
    j = df.join(deck.rename("R_deck"), how="inner")
    dev = float((j["R_mid"] - j["R_deck"]).abs().max())
    assert len(j) == len(deck) and dev < GATE_R_TOL, (len(j), dev)
    print(
        f"GATE  chain-built 15:30 straddle return vs the deck's on {len(j)} days: {dev:.1e}"
    )
    # every weekday session is listed from 2022-05; before that the panel's own
    # regular-hours calendar names the month's last session
    c = pd.read_parquet(ROOT / "data" / "core_stats.parquet", columns=["endbartime"])
    t = pd.to_datetime(c["endbartime"])
    rth = pd.DatetimeIndex(
        sorted(t[t.dt.strftime("%H:%M") == "16:00"].dt.normalize().unique())
    )
    cal = rth.union(pd.DatetimeIndex(stamp.index))
    df["month_end"] = month_end_flags(cal).reindex(df.index).to_numpy(bool)
    rows = []
    for name, m in (
        ("deck period (to 2024-04-30)", df.index <= DECK_END),
        ("HOLDOUT 2024-05-01 .. 2025-12-31", df.index >= OOS_START),
        ("all sessions", np.ones(len(df), bool)),
    ):
        g = df[m]
        a, b = g[g["month_end"]], g[~g["month_end"]]
        rows.append(
            {
                "sample": name,
                "sessions": len(g),
                "month_ends": len(a),
                "long_mid_mean": float(a["R_mid"].mean()),
                "long_mid_t": tstat(a["R_mid"]),
                "long_ask_mean": float(a["R_ask"].mean()),
                "long_ask_t": tstat(a["R_ask"]),
                "hit": float((a["R_mid"] > 0).mean()),
                "pts_per_contract_at_ask": float(a["pts_ask"].mean()),
                "other_days_long_mid_mean": float(b["R_mid"].mean()),
            }
        )
    return pd.DataFrame(rows), df


def part_d(days_c: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_parquet(DECK / "daily_blk2.parquet").sort_index()
    r = d["R"].to_numpy(float)
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    ex = d["exit"].to_numpy(float)
    q = np.where(d["signal"].to_numpy(float) > 0, 1.0, -1.0)
    me = days_c["month_end"].reindex(pd.DatetimeIndex(d.index)).to_numpy(bool)
    rows = []
    for name, qq in (
        ("sign(s)", q),
        ("sign(s), long on month-ends", np.where(me, 1.0, q)),
        ("sign(s), flat on month-ends", np.where(me, 0.0, q)),
    ):
        cr = np.where(qq > 0, qq * (ex / ask - 1.0), qq * (ex / bid - 1.0))
        rows.append(
            {
                "book": f"15:30 {name}",
                "unit": "premium",
                "n": len(r),
                "mean": float((qq * r).mean()),
                "Sharpe_mid": sharpe(qq * r),
                "Sharpe_crossed": sharpe(cr),
            }
        )
    a = pd.read_csv(P43_DAILY, index_col=0, parse_dates=True)
    mm = days_c["month_end"].reindex(a.index).fillna(False).to_numpy(bool)
    for clock in ("11:00", "13:30", "14:30"):
        for kind in ("hold", "flatten"):
            pts = a[f"{clock}|{kind}"] * a[f"{clock}|entry"]
            x, y = pts[mm].dropna(), pts[~mm].dropna()
            rows.append(
                {
                    "book": f"short straddle sold {clock}, {kind}",
                    "unit": "index points per contract",
                    "n": len(x),
                    "mean": float(x.mean()),
                    "t_month_end": tstat(x),
                    "mean_other_days": float(y.mean()),
                    "t_other_days": tstat(y),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    a = part_a()
    a.to_csv(OUT / "a_decomposition.csv", index=False)
    print(a.round(3).to_string(index=False))
    b = part_b()
    b.to_csv(OUT / "b_underlying.csv", index=False)
    print("\nB  the underlying's last half hour, month-end against other days")
    print(b.round(3).T.to_string())
    c, days_c = part_c()
    c.to_csv(OUT / "c_long_straddle_month_end.csv", index=False)
    days_c.to_csv(OUT / "c_daily.csv")
    print("\nC  long the 15:30 straddle on the last trading day of the month")
    print(c.round(3).T.to_string())
    d = part_d(days_c)
    d.to_csv(OUT / "d_books.csv", index=False)
    print("\nD  what it does to the books")
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
