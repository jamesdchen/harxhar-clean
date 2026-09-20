"""28 — crash stress for the 11:00 delta-hedged short.

The book is the one in writeup/make_dh_causal_standalone_tex.py: sell the
nearest-OTM SPX 0DTE straddle at 11:00, Black-76 delta rebalanced every 30
minutes on the vendor spot (hourly implied re-inverted on the censored
nodes), then either hold through cash settlement or buy the straddle back at
the 15:30 model mark.  Returns are per unit of midpoint entry premium.

This script does not change the book.  It reprices one day's tape under
index jumps that did not happen (A), under 11:00-to-close paths that did
happen on other sessions (B), counts how often such moves occur in the
30-minute panel (C), and expresses the resulting losses against the book's
own mean, worst day and drawdown (D).

Gate first: the reproduced 11:00 always-short tape must match the deck
(n, Sharpe at mid and crossed, worst day) before anything new is computed.

Run:  python writeup/intraday_proposals/28_crash_stress.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "28"
PANEL = REPO / "data" / "core_stats.parquet"

ENTRY = "11:00"
CLOSE = "15:30"
MIDDAY_BAR_END = "13:30"  # the jump is placed inside the 13:00 -> 13:30 bar
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
CONTRACT_MULTIPLIER = 100.0

# Stated grids.
JUMPS = (0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05)
BAR_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05)
LAST_BAR_THRESHOLDS = (0.005, 0.01, 0.02, 0.03)
SESSION_THRESHOLDS = (0.02, 0.03, 0.05, 0.07, 0.10)
REPLAY_SESSIONS = (
    "2008-10-28",
    "2008-10-16",
    "2008-11-20",
    "2020-03-13",
    "2020-03-20",
    "2018-02-05",
    "2010-05-06",
)
# Panel bar-END stamps that span 11:00 -> official close, ten 30-minute steps.
WINDOW_BARS = (
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
)
LAST_BAR = "16:00"

# The representative day is the sample median of entry premium / spot; the
# high-premium day is the nearest-rank 95th percentile of the same ratio.
HIGH_PREMIUM_QUANTILE = 0.95
# Proposal D: the capital per unit of premium at which a 3% mid-day jump
# costs this fraction of capital.
CAPITAL_LOSS_BUDGET = 0.05
SIZING_JUMPS = (0.02, 0.03)
# Reg-T proxy: 15% of the index, less the out-of-the-money amount, plus the
# leg premium; the other leg's premium is added on top.
REGT_INDEX_FRACTION = 0.15


def _standalone():
    """The module of record for the book (hold_mark_1100, p26, rule helpers)."""
    path = REPO / "writeup" / "make_dh_causal_standalone_tex.py"
    spec = importlib.util.spec_from_file_location("mk_dh_causal", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_grids(pkg: pd.DataFrame, mk) -> dict:
    """Rebuild exactly the arrays hold_mark_1100 prices the 11:00 book from."""
    p26 = mk._p26()
    m = mk._rule_mod()
    deck = pd.read_parquet(m.DECK / "daily_blk2.parquet")
    days = pd.to_datetime(deck.index)
    p = pkg.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"].isin(days)].copy()
    clocks = sorted(p["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
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
    return {
        "p26": p26,
        "clocks": clocks,
        "dates": dates,
        "sl": sl,
        "Sg": S_grid.to_numpy(float),
        "IVg": IV_grid.to_numpy(float),
        "h_row": np.array([n_rem[c] * 0.5 for c in clocks], float),
        "j": clocks.index(ENTRY),
        "j15": clocks.index(CLOSE),
    }


def reprice(g: dict, i: int, S_row: np.ndarray, ST: float) -> dict:
    """Price the short book on one day given a spot path and a settlement.

    Same machinery as hold_mark_1100: delta from the stamp's implied and the
    hours remaining, hedge P&L = delta times the spot change over the next
    half hour, settlement intrinsic against ST.  Nothing is re-fitted; only
    the spot path and the settlement print move.
    """
    p26 = g["p26"]
    j, j15 = g["j"], g["j15"]
    Kc = float(g["sl"]["K_c"].to_numpy(float)[i])
    Kp = float(g["sl"]["K_p"].to_numpy(float)[i])
    entry = float(g["sl"]["entry"].to_numpy(float)[i])
    IV_row = g["IVg"][i]
    h_row = g["h_row"]

    tot = np.where(IV_row > 0, IV_row * np.sqrt(h_row), np.nan)
    tot[:j] = np.nan
    dlt = p26.pkg_delta(tot, S_row, np.full_like(S_row, Kc), np.full_like(S_row, Kp))
    nxt = np.full_like(S_row, np.nan)
    nxt[:-1] = S_row[1:]
    nxt[-1] = ST
    dS = np.where(np.isfinite(S_row) & np.isfinite(nxt), nxt - S_row, 0.0)
    dS[:j] = 0.0

    settle = max(ST - Kc, 0.0) + max(Kp - ST, 0.0)
    hedge_hold = float((dlt * dS).sum())
    r_hold = (-(settle - entry) + hedge_hold) / entry

    tot15 = np.array([IV_row[j15] * np.sqrt(h_row[j15])])
    mark = float(
        p26.pkg_price(
            np.where(IV_row[j15] > 0, tot15, np.nan),
            np.array([S_row[j15]]),
            np.array([Kc]),
            np.array([Kp]),
        )[0]
    )
    dS_f = dS.copy()
    dS_f[j15] = 0.0
    hedge_mark = float((dlt * dS_f).sum())
    r_mark = (-(mark - entry) + hedge_mark) / entry
    return {
        "r_hold": float(r_hold),
        "r_mark": float(r_mark),
        "settle": float(settle),
        "mark": mark,
        "hedge_hold": hedge_hold,
        "hedge_mark": hedge_mark,
        "entry": entry,
        "K_c": Kc,
        "K_p": Kp,
        "S_1100": float(S_row[j]),
        "S_1530": float(S_row[j15]),
        "S_close": float(ST),
    }


def jumped_path(g: dict, i: int, x: float, placement: str) -> tuple[np.ndarray, float]:
    """The day's spot path with one 30-minute bar carrying an extra jump x."""
    S_row = g["Sg"][i].copy()
    ST = float(g["sl"]["S_close"].to_numpy(float)[i])
    if placement == "last bar":
        # The 15:30 hedge is frozen: only the settlement print moves.
        return S_row, float(S_row[g["j15"]]) * (1.0 + x)
    if placement == "mid-day bar":
        k = g["clocks"].index(MIDDAY_BAR_END)
        f = np.ones_like(S_row)
        f[k:] = 1.0 + x
        return S_row * f, ST * (1.0 + x)
    raise ValueError(placement)


def gate(mk, g: dict) -> dict:
    """Reproduce the deck's 11:00 always-short row, then the per-day rebuild."""
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    r_hold, r_mark, r_hold_x, r_mark_x = mk.hold_mark_1100(pkg)
    h = r_hold.dropna()
    hx = r_hold_x.dropna()
    sh = float(h.mean() / h.std(ddof=1) * ANN)
    shx = float(hx.mean() / hx.std(ddof=1) * ANN)
    ref = pd.read_csv(
        HOLD / "rule_by_strategy_dh" / "1100" / "rule_by_strategy_always_short.csv",
        index_col=0,
    ).loc["all models"]
    assert int(h.size) == int(ref["n"]), (int(h.size), int(ref["n"]))
    assert abs(sh - float(ref["Sharpe_ann"])) < 1e-6, (sh, float(ref["Sharpe_ann"]))
    assert abs(shx - float(ref["Sharpe_crossed"])) < 1e-6, (shx, ref["Sharpe_crossed"])
    worst_day = h.idxmin()
    print(
        f"GATE  n={h.size}  Sharpe_ann={sh:.6f}  Sharpe_crossed={shx:.6f}  "
        f"worst={float(h.min()):.6f} on {worst_day.date()}  MaxDD={mk._maxdd(h):.6f}"
    )
    # The per-day rebuild must reproduce the same tape from the same grids.
    rebuilt = np.array(
        [
            reprice(g, i, g["Sg"][i], float(g["sl"]["S_close"].to_numpy(float)[i]))[
                "r_hold"
            ]
            for i in range(len(g["dates"]))
        ]
    )
    dev_h = float(
        np.nanmax(np.abs(rebuilt - r_hold.reindex(g["dates"]).to_numpy(float)))
    )
    rebuilt_m = np.array(
        [
            reprice(g, i, g["Sg"][i], float(g["sl"]["S_close"].to_numpy(float)[i]))[
                "r_mark"
            ]
            for i in range(len(g["dates"]))
        ]
    )
    dev_m = float(
        np.nanmax(np.abs(rebuilt_m - r_mark.reindex(g["dates"]).to_numpy(float)))
    )
    assert dev_h < 1e-12 and dev_m < 1e-12, (dev_h, dev_m)
    print(f"GATE  per-day rebuild max |dev|: hold {dev_h:.3e}  15:30 mark {dev_m:.3e}")
    return {
        "r_hold": r_hold.dropna(),
        "r_mark": r_mark.dropna(),
        "Sharpe_ann": sh,
        "Sharpe_crossed": shx,
        "worst": float(h.min()),
        "worst_day": worst_day,
        "maxdd": float(mk._maxdd(h)),
    }


def pick_days(g: dict) -> dict:
    """Median and 95th-percentile premium/spot days among the scored days."""
    entry = g["sl"]["entry"].to_numpy(float)
    S11 = g["Sg"][:, g["j"]]
    ratio = pd.Series(entry / S11, index=g["dates"])
    srt = ratio.sort_values()
    i_med = len(srt) // 2  # n is odd: the exact middle order statistic
    k95 = int(np.ceil(HIGH_PREMIUM_QUANTILE * len(srt)))
    med_day, hi_day = srt.index[i_med], srt.index[k95 - 1]
    print(
        f"premium/spot on the {len(srt)} scored days: median {100 * srt.iloc[i_med]:.4f}% "
        f"({entry[g['dates'].get_loc(med_day)]:.2f} pts, {med_day.date()}); "
        f"5th {100 * ratio.quantile(0.05):.4f}%  95th (nearest rank) "
        f"{100 * srt.iloc[k95 - 1]:.4f}% ({hi_day.date()})"
    )
    return {"ratio": ratio, "median_day": med_day, "high_day": hi_day}


def day_facts(g: dict, day: pd.Timestamp) -> dict:
    i = g["dates"].get_loc(day)
    base = reprice(g, i, g["Sg"][i], float(g["sl"]["S_close"].to_numpy(float)[i]))
    base["i"] = i
    base["date"] = day
    base["iv_1100"] = float(g["IVg"][i, g["j"]])
    base["iv_1530"] = float(g["IVg"][i, g["j15"]])
    return base


def jump_table(g: dict, days: dict[str, pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for role, day in days.items():
        f = day_facts(g, day)
        i = f["i"]
        for placement in ("last bar", "mid-day bar"):
            for x in JUMPS:
                S_row, ST = jumped_path(g, i, x, placement)
                r = reprice(g, i, S_row, ST)
                rows.append(
                    {
                        "day_role": role,
                        "date": day.date().isoformat(),
                        "placement": placement,
                        "jump_pct": 100.0 * x,
                        "K_c": f["K_c"],
                        "K_p": f["K_p"],
                        "S_1100": f["S_1100"],
                        "iv_hourly_1100": f["iv_1100"],
                        "entry_premium_pts": f["entry"],
                        "premium_over_spot_pct": 100.0 * f["entry"] / f["S_1100"],
                        "S_settle_stressed": r["S_close"],
                        "r_hold_units": r["r_hold"],
                        "r_mark_units": r["r_mark"],
                        "r_hold_base_units": f["r_hold"],
                        "r_mark_base_units": f["r_mark"],
                        "d_hold_units": r["r_hold"] - f["r_hold"],
                        "d_mark_units": r["r_mark"] - f["r_mark"],
                        "hold_dollars": r["r_hold"] * f["entry"] * CONTRACT_MULTIPLIER,
                        "mark_dollars": r["r_mark"] * f["entry"] * CONTRACT_MULTIPLIER,
                    }
                )
    return pd.DataFrame(rows)


def validate_1130(g: dict, book: dict) -> pd.DataFrame:
    """Replay the real 2023-11-30 last bar through the stress machinery."""
    day = book["worst_day"]
    i = g["dates"].get_loc(day)
    S15 = float(g["Sg"][i, g["j15"]])
    ST = float(g["sl"]["S_close"].to_numpy(float)[i])
    actual = float(book["r_hold"].loc[day])
    x_exact = ST / S15 - 1.0
    rows = []
    for name, x in (("exact vendor ratio", x_exact), ("printed +0.51%", 0.0051)):
        S_row, ST_x = jumped_path(g, i, x, "last bar")
        r = reprice(g, i, S_row, ST_x)
        rows.append(
            {
                "date": day.date().isoformat(),
                "X_source": name,
                "jump_pct": 100.0 * x,
                "S_1530": S15,
                "S_settle_actual": ST,
                "S_settle_stressed": r["S_close"],
                "r_hold_actual": actual,
                "r_hold_stressed": r["r_hold"],
                "residual": r["r_hold"] - actual,
            }
        )
    return pd.DataFrame(rows)


def panel_window() -> pd.DataFrame:
    """Weekday 11:00-to-close 30-minute log returns, bar-END stamped.

    The frame carries two session counts: every weekday session that shows
    any of the ten stamps (the panel's full 1993-2024 span) and the sessions
    that carry all ten finite returns (sumret is empty before 1998-01-05).
    """
    d = pd.read_parquet(PANEL, columns=["endbartime", "sumret"])
    t = pd.to_datetime(d["endbartime"])
    d = d.assign(t=t, date=t.dt.normalize(), hhmm=t.dt.strftime("%H:%M"))
    w = d[d["hhmm"].isin(WINDOW_BARS) & (d["t"].dt.weekday < 5)]
    grid = w.pivot_table(
        index="date", columns="hhmm", values="sumret", aggfunc="first"
    ).reindex(columns=list(WINDOW_BARS))
    grid.attrs["n_sessions_stamped"] = int(w["date"].nunique())
    grid.attrs["first_stamped"] = w["date"].min()
    grid.attrs["last_stamped"] = w["date"].max()
    return grid


def frequency_table(grid: pd.DataFrame) -> pd.DataFrame:
    complete = grid.dropna(how="any")
    n_sessions_stamped = int(grid.attrs["n_sessions_stamped"])
    n_sessions_complete = int(len(complete))
    yrs_stamped = n_sessions_stamped / asl.PERIODS_PER_YEAR
    yrs_complete = n_sessions_complete / asl.PERIODS_PER_YEAR
    bars = complete.to_numpy(float)
    last = complete[LAST_BAR].to_numpy(float)
    tot = complete.sum(axis=1).to_numpy(float)
    rows = []

    def add(kind, th, n):
        rows.append(
            {
                "kind": kind,
                "threshold_pct": 100.0 * th,
                "count": int(n),
                "per_year_sessions_with_returns": n / yrs_complete,
                "per_year_sessions_1993_2024": n / yrs_stamped,
            }
        )

    for th in BAR_THRESHOLDS:
        add("any single 30-min bar", th, (np.abs(bars) > th).sum())
    for th in LAST_BAR_THRESHOLDS:
        add("last bar (16:00 stamp)", th, (np.abs(last) > th).sum())
    for th in SESSION_THRESHOLDS:
        add("11:00 -> close", th, (np.abs(tot) > th).sum())
    out = pd.DataFrame(rows)
    out.attrs["n_sessions_complete"] = n_sessions_complete
    out.attrs["n_sessions_stamped"] = n_sessions_stamped
    out.attrs["first_complete"] = complete.index.min()
    out.attrs["last_complete"] = complete.index.max()
    out.attrs["yrs_complete"] = yrs_complete
    out.attrs["yrs_stamped"] = yrs_stamped
    out.attrs["first_stamped"] = grid.attrs["first_stamped"]
    out.attrs["last_stamped"] = grid.attrs["last_stamped"]
    return out


def replay_table(g: dict, grid: pd.DataFrame, rep_day: pd.Timestamp) -> pd.DataFrame:
    complete = grid.dropna(how="any")
    last = complete[LAST_BAR]
    extra = last.abs().idxmax()
    # The largest last bar of the opposite sign, so the replay carries both
    # tails of the settlement-print exposure.
    opposite = (last * -np.sign(last.loc[extra])).idxmax()
    sessions = [pd.Timestamp(s) for s in REPLAY_SESSIONS]
    for d in (extra, opposite):
        if d not in sessions:
            sessions.append(d)
    f = day_facts(g, rep_day)
    i = f["i"]
    j, j15 = g["j"], g["j15"]
    rows = []
    for s in sessions:
        if s not in complete.index:
            rows.append({"session": s.date().isoformat(), "in_panel": False})
            continue
        ret = complete.loc[s].to_numpy(float)
        S_row = g["Sg"][i].copy()
        S0 = float(S_row[j])
        path = S0 * np.exp(np.cumsum(ret))
        S_row[j + 1 : j15 + 1] = path[: j15 - j]
        ST = float(path[-1])
        r = reprice(g, i, S_row, ST)
        rows.append(
            {
                "session": s.date().isoformat(),
                "in_panel": True,
                "largest_last_bar": bool(s == extra),
                "largest_opposite_last_bar": bool(s == opposite),
                "path_1100_to_close_logpct": 100.0 * float(ret.sum()),
                "path_1100_to_close_pct": 100.0 * float(np.expm1(ret.sum())),
                "path_last_bar_logpct": 100.0 * float(ret[-1]),
                "path_max_abs_bar_logpct": 100.0 * float(np.abs(ret).max()),
                "S_1100": S0,
                "S_1530_replayed": float(S_row[j15]),
                "S_settle_replayed": ST,
                "r_hold_units": r["r_hold"],
                "r_mark_units": r["r_mark"],
                "hold_dollars": r["r_hold"] * f["entry"] * CONTRACT_MULTIPLIER,
                "mark_dollars": r["r_mark"] * f["entry"] * CONTRACT_MULTIPLIER,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["largest_last_bar_day"] = extra
    out.attrs["opposite_last_bar_day"] = opposite
    return out


def sizing_table(book: dict, jumps: pd.DataFrame, rep: dict) -> pd.DataFrame:
    mean_hold = float(book["r_hold"].mean())
    mean_mark = float(book["r_mark"].mean())
    worst = abs(book["worst"])
    maxdd = abs(book["maxdd"])
    sel = jumps[
        (jumps["day_role"] == "representative (median premium)")
        & (jumps["placement"] == "mid-day bar")
    ]
    rows = []
    for x in SIZING_JUMPS:
        for sign in (1.0, -1.0):
            r = sel[np.isclose(sel["jump_pct"], 100.0 * sign * x)]
            if r.empty:
                continue
            r = r.iloc[0]
            for book_name, col, mean_r in (
                ("hold through cash-settle", "r_hold_units", mean_hold),
                ("exit at 15:30", "r_mark_units", mean_mark),
            ):
                loss = -float(r[col])  # positive = a loss, in premium units
                capital = loss / CAPITAL_LOSS_BUDGET
                rows.append(
                    {
                        "book": book_name,
                        "jump_pct": 100.0 * sign * x,
                        "placement": "mid-day bar",
                        "pnl_units": float(r[col]),
                        "loss_units": loss,
                        "mean_daily_units": mean_r,
                        "x_mean_daily": loss / mean_r,
                        "x_worst_day": loss / worst,
                        "x_maxdd": loss / maxdd,
                        "capital_per_unit_premium": capital,
                        "premium_fraction_of_capital": 1.0 / capital,
                        "daily_mean_on_capital": mean_r / capital,
                        "annual_mean_on_capital": asl.PERIODS_PER_YEAR
                        * mean_r
                        / capital,
                    }
                )
    out = pd.DataFrame(rows)
    out.attrs["mean_hold"] = mean_hold
    out.attrs["mean_mark"] = mean_mark
    out.attrs["worst"] = worst
    out.attrs["maxdd"] = maxdd
    out.attrs["regt"] = regt_proxy(rep)
    return out


def regt_proxy(rep: dict) -> dict:
    """Reg-T initial margin PROXY for one short straddle (not a broker quote)."""
    S = rep["S_1100"]
    mid_c, mid_p = rep["mid_c"], rep["mid_p"]
    otm_c = max(rep["K_c"] - S, 0.0)
    otm_p = max(S - rep["K_p"], 0.0)
    req_c = mid_c + REGT_INDEX_FRACTION * S - otm_c
    req_p = mid_p + REGT_INDEX_FRACTION * S - otm_p
    larger_is_call = req_c >= req_p
    req = max(req_c, req_p) + (mid_p if larger_is_call else mid_c)
    return {
        "S": S,
        "call_leg_pts": req_c,
        "put_leg_pts": req_p,
        "larger_leg": "call" if larger_is_call else "put",
        "margin_pts": req,
        "margin_dollars": req * CONTRACT_MULTIPLIER,
        "premium_dollars": rep["entry"] * CONTRACT_MULTIPLIER,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mk = _standalone()
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    g = build_grids(pkg, mk)
    book = gate(mk, g)

    picks = pick_days(g)
    rep_day, hi_day = picks["median_day"], picks["high_day"]
    rep = day_facts(g, rep_day)
    raw = pkg.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    e = (
        raw[(raw["hhmm"] == ENTRY) & (raw["date"] == rep_day)]
        .drop_duplicates("date")
        .iloc[0]
    )
    rep["mid_c"], rep["mid_p"] = float(e["mid_c"]), float(e["mid_p"])
    print(
        f"\nrepresentative day {rep_day.date()}: S(11:00) {rep['S_1100']:.2f}  "
        f"K_c {rep['K_c']:.0f}  K_p {rep['K_p']:.0f}  entry {rep['entry']:.2f} pts "
        f"({100 * rep['entry'] / rep['S_1100']:.4f}% of spot)  "
        f"hourly IV 11:00 {rep['iv_1100']:.6f}  15:30 {rep['iv_1530']:.6f}  "
        f"settle {rep['S_close']:.2f}  r_hold {rep['r_hold']:+.4f}  "
        f"r_mark {rep['r_mark']:+.4f}"
    )
    hi = day_facts(g, hi_day)
    print(
        f"high-premium day {hi_day.date()}: S(11:00) {hi['S_1100']:.2f}  "
        f"K_c {hi['K_c']:.0f}  K_p {hi['K_p']:.0f}  entry {hi['entry']:.2f} pts "
        f"({100 * hi['entry'] / hi['S_1100']:.4f}% of spot)  "
        f"hourly IV 11:00 {hi['iv_1100']:.6f}  r_hold {hi['r_hold']:+.4f}"
    )

    days = {
        "representative (median premium)": rep_day,
        "high premium (95th pct)": hi_day,
    }
    jumps = jump_table(g, days)
    val = validate_1130(g, book)
    jumps = pd.concat(
        [jumps, val.assign(day_role="validation", placement="last bar")],
        ignore_index=True,
        sort=False,
    )
    jumps.to_csv(OUT / "jump_stress.csv", index=False)
    print(
        "\n--- A. jump stress (P&L in units of entry premium; $ = units x premium x 100)"
    )
    print(
        jumps[jumps["day_role"] != "validation"][
            [
                "day_role",
                "date",
                "placement",
                "jump_pct",
                "S_settle_stressed",
                "r_hold_units",
                "hold_dollars",
                "r_mark_units",
                "mark_dollars",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )
    print("\n--- A. validation on the real 2023-11-30 last bar")
    print(val.to_string(index=False, float_format=lambda v: f"{v:,.8f}"))

    grid = panel_window()
    freq = frequency_table(grid)
    freq.to_csv(OUT / "frequency.csv", index=False)
    print(
        f"\n--- C. frequency; sessions carrying all ten returns: "
        f"{freq.attrs['n_sessions_complete']} "
        f"({freq.attrs['first_complete'].date()}..{freq.attrs['last_complete'].date()}, "
        f"{freq.attrs['yrs_complete']:.3f} yr at 252/yr); weekday sessions carrying any "
        f"of the ten stamps: {freq.attrs['n_sessions_stamped']} "
        f"({freq.attrs['first_stamped'].date()}..{freq.attrs['last_stamped'].date()}, "
        f"{freq.attrs['yrs_stamped']:.3f} yr) -- sumret is empty before "
        f"{freq.attrs['first_complete'].date()}"
    )
    print(freq.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    rep_table = replay_table(g, grid, rep_day)
    rep_table.to_csv(OUT / "replay.csv", index=False)
    print(
        f"\n--- B. historical path replay on {rep_day.date()} "
        f"(IV held at that day's level; largest last bar in the panel: "
        f"{rep_table.attrs['largest_last_bar_day'].date()}, largest of the "
        f"opposite sign: {rep_table.attrs['opposite_last_bar_day'].date()})"
    )
    print(rep_table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    siz = sizing_table(book, jumps, rep)
    siz.to_csv(OUT / "sizing.csv", index=False)
    print(
        f"\n--- D. sizing; mean daily hold {siz.attrs['mean_hold']:.6f}, "
        f"exit-at-15:30 {siz.attrs['mean_mark']:.6f}, worst day {siz.attrs['worst']:.6f}, "
        f"MaxDD {siz.attrs['maxdd']:.6f} (units of premium)"
    )
    print(siz.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    rt = siz.attrs["regt"]
    print(
        f"Reg-T PROXY, one short straddle at {rep_day.date()} "
        f"(S {rt['S']:.2f}): call leg {rt['call_leg_pts']:.2f} pts, "
        f"put leg {rt['put_leg_pts']:.2f} pts, larger = {rt['larger_leg']}; "
        f"margin {rt['margin_pts']:.2f} pts = ${rt['margin_dollars']:,.0f} "
        f"against ${rt['premium_dollars']:,.0f} of premium"
    )
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
