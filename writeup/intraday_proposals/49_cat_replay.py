"""49 - the insurer's catastrophe model: every session in the panel, today's book.

Proposal 28 stressed the 11:00 delta-hedged short straddle with eight named
paths.  Proposal 45 stressed the 13:30 one with a sixteen-cell jump grid.  This
script replaces the grid with the record: EVERY session in the 30-minute index
panel (data/core_stats.parquet, bar-END stamps, naive ET) is replayed through a
REPRESENTATIVE RECENT DAY'S priced book, and the result is a loss distribution
rather than a stress table.

The convention is the "unanticipated" one, stated once and carried everywhere:

  the strikes, the entry fill, the unit denominator and the whole implied
  volatility tape are the representative day's and are HELD FIXED; only the
  index path from the entry stamp to the settlement print is replaced by the
  historical session's own 30-minute bar returns.

Nothing about the historical session's option market enters.  A 1998 session
replayed through a 2023 book is what a 1998 index path would have cost the 2023
book if the 2023 book had been priced as it was and the path had been a
surprise.  That is the insurer's question, and it is not a backtest.

The book replayed is proposal 43's always-short delta-hedged straddle:

  ENTRY   at 13:30 (and at 11:00 for comparison) the nearest-OTM SPX 0DTE
          straddle is SOLD AT THE QUOTED BID (proposal 43's own fill); the unit
          denominator is the quoted MIDPOINT of the same two legs;
  HEDGE   +delta index units per short straddle, rebalanced at every 30-minute
          stamp, the delta being Black-76 on the stamp's own mid-inverted total
          volatility (proposal 43's pkg_delta_vec, the branch
          live.ibkr.pricing.package_delta takes);
  HOLD    the two legs pay their intrinsic into the 16:00 cash settlement and
          the hedge is carried into that print (the primary terminal);
  FLATTEN the two legs are bought back at 15:30 and the hedge is unwound there
          (the secondary terminal).

Two volatility conventions, both replayed, both labelled at the point of use:

  frozen IV        the representative day's total volatility at every stamp,
                   unchanged.  APPROXIMATION: on a large path the delta of a
                   frozen-IV package becomes a step function - the hedge jumps
                   to +-1 the moment the path leaves the strikes - so the hedge
                   leg of a crash session is priced with a volatility that no
                   crash session ever carried.  This is proposal 28's caveat,
                   restated.
  regime-scaled IV implied_var_replay = implied_var_rep x max(RV21_hist(d) /
                   RV21_rep, 1), where RV21 is the mean of the panel's own
                   session realized variance (the sum of sumret2 over the 13
                   regular-hours bars) over the 21 sessions STRICTLY BEFORE the
                   replayed date - causal, session-level, floored at the
                   representative day's own implied so the scaling can only
                   raise the volatility.  APPROXIMATION: a session-level
                   realized/implied ratio is not an implied volatility; it has
                   no smile, no term structure, and it does not move inside the
                   session.  Sessions whose 21-session trailing window is
                   incomplete take the floor (factor 1.0) and are counted.

Gates, asserted before a single new number is computed:
  GATE 28  proposal 28's 11:00 representative day (2021-06-16, premium/spot
           median of its 865 scored days) replayed on the 2020-03-20 and
           2008-11-20 paths reproduces its published -4.097283 and -13.421349
           premium units, through THIS script's pricer, to 1e-12
  GATE 45  proposal 45's 13:30 representative day (2020-11-23) under its own
           +5% last-bar jump reproduces its published naked four-option-leg
           -19.174050 units / -$18,023.61 and hedge +$15,237.08, to 1e-9

Outputs, all under results/atm_straddle_intraday_holdclose/proposals/49/:
  a_distribution.csv  one row per replayed session x book
  a_summary.csv       moments, the 1/0.5/0.1st percentiles, ES, worst-1% anatomy
  a_worst.csv         the ten worst sessions of every book, with path shape
  a_by_era.csv        1998-2007 / 2008-2009 / 2010-2019 / 2020-2024
  b_sizing.csv        capital from the 1-in-200 loss, and the ruin statistic
  c_blindspots.csv    what the 30-minute panel cannot see

Per-contract units everywhere: index points, dollars at $100 an index point,
and units of the representative day's entry midpoint premium.

No randomness is used anywhere in this script, so there is no seed to state.
Parallelism: the replay itself is one vectorised numpy expression over
(n_sessions x n_stamps) arrays, and the 16 books are farmed to a
ProcessPoolExecutor.

Run:  python writeup/intraday_proposals/49_cat_replay.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
import atm_straddle_lib as asl  # noqa: E402

HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "49"
PANEL = ROOT / "data" / "core_stats.parquet"
P43_PATH = ROOT / "writeup" / "intraday_proposals" / "43_causal_entry_over_time.py"
P28_PATH = ROOT / "writeup" / "intraday_proposals" / "28_crash_stress.py"
REF28 = HOLD / "proposals" / "28" / "replay.csv"
REF45 = HOLD / "proposals" / "45" / "e_stress.csv"

SPX_MULT = 100.0  # dollars per index point, one SPX option contract
HEDGE_COST_BP = 0.5e-4  # 0.5 bp of S on every index unit traded (43's charge)
DECK_END = "2024-04-30"  # proposal 43's last deck-scored session
PANEL_FIRST_RETURN = "1998-01-05"  # sumret is empty before this session

#: the 13 regular-hours bar-END stamps of the 30-minute panel (09:30 -> 16:00).
RTH_BARS: tuple[str, ...] = (
    "10:00",
    "10:30",
    "11:00",
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
SETTLE_BAR = "16:00"
N_RTH_BARS = len(RTH_BARS)

#: the entry clocks replayed, and the panel bars that span entry -> settlement.
ENTRY_CLOCKS: tuple[str, ...] = ("13:30", "11:00")
MAIN_ENTRY = "13:30"
TERMINALS: tuple[str, ...] = ("hold to cash settlement", "flatten at 15:30")
IV_MODES: tuple[str, ...] = ("frozen IV", "regime-scaled IV")

#: the three representative days, by nearest-rank quantile of the 13:30 entry
#: premium as a fraction of spot over proposal 43's deck-scored sessions.
REP_QUANTILES: tuple[float, ...] = (0.25, 0.50, 0.75)
REP_LABEL: dict[float, str] = {
    0.25: "p25 premium",
    0.50: "median premium",
    0.75: "p75 premium",
}
MEDIAN_ROLE = REP_LABEL[0.50]

RV_WINDOW = 21  # sessions in the trailing realized-variance window
ERAS: tuple[tuple[str, str, str], ...] = (
    ("1998-2007", "1998-01-01", "2007-12-31"),
    ("2008-2009", "2008-01-01", "2009-12-31"),
    ("2010-2019", "2010-01-01", "2019-12-31"),
    ("2020-2024", "2020-01-01", "2024-12-31"),
)
TAIL_PCTS: tuple[float, ...] = (1.0, 0.5, 0.1)  # 1-in-100, 1-in-200, 1-in-1000
ES_PCT = 1.0
WORST_N = 10
WORST_FRAC = 0.01  # the worst-1% set the anatomy is taken over
TREND_BAR = 0.5  # |net move| / sum|bar moves| at or above this = a trend
RUIN_WINDOW = 20  # sessions in the ruin statistic's cumulative loss window
CAPITAL_BUDGETS: tuple[float, ...] = (0.05, 0.10)
SIZING_TAIL_PCT = 0.5  # the 1-in-200 session loss sizes the capital
N_VOL_DECILES = 10

# GATE targets: the published prints of proposals 28 and 45.
GATE28_REP_DAY = "2021-06-16"
GATE28_SESSIONS: tuple[str, ...] = ("2020-03-20", "2008-11-20")
GATE28_PRINTED: dict[str, float] = {"2020-03-20": -4.10, "2008-11-20": -13.42}
GATE28_TOL = 1e-12
GATE28_PRINT_TOL = 5e-3
GATE45_REP_DAY = "2020-11-23"
GATE45_JUMP = 0.05
GATE45_FOUR_LEG_UNITS = -19.174050
GATE45_FOUR_LEG_DOLLARS = -18023.607163
GATE45_HEDGE_DOLLARS = 15237.077743
GATE45_TOL = 1e-9
GATE45_PRINT_TOL = 5e-6


# ------------------------------------------------------------------ module --
def load_by_path(name: str, path: Path) -> ModuleType:
    """Import a proposal script by path; it is read, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- the pricer --
def price_book(
    p43: ModuleType,
    tot: np.ndarray,
    s_path: np.ndarray,
    s_settle: np.ndarray,
    k_c: float,
    k_p: float,
    short_fill: float,
    entry_mid: float,
    j: int,
) -> dict[str, np.ndarray]:
    """The short delta-hedged straddle on many spot paths at once.

    ``tot`` (n x K) is the total volatility held at each stamp, ``s_path``
    (n x K) the index level at each stamp and ``s_settle`` (n,) the settlement
    print.  Stamps before the entry index ``j`` carry no position.  This is
    proposal 43's ``_clock_book`` and proposal 28's ``reprice``, vectorised over
    paths instead of over sessions, with the fill and the unit denominator
    passed in so the same code prices both proposals' books.

    The FLATTEN terminal's buyback is the Black-76 model mark at the 15:30
    stamp on the same held volatility.  APPROXIMATION, labelled here: a
    counterfactual path has no quoted 15:30 ask, so the flatten leg is a model
    mark and not a fill, and it carries none of the half-spread that proposal
    43's quoted-ask flatten book pays.
    """
    n, n_k = s_path.shape
    tot = np.array(tot, float, copy=True)
    tot[:, :j] = np.nan
    dlt = p43.pkg_delta_vec(tot, s_path, k_c, k_p)
    dlt[:, :j] = 0.0
    nxt = np.full_like(s_path, np.nan)
    nxt[:, :-1] = s_path[:, 1:]
    nxt[:, -1] = s_settle
    d_s = np.where(np.isfinite(s_path) & np.isfinite(nxt), nxt - s_path, 0.0)
    d_s[:, :j] = 0.0
    d_s_flat = d_s.copy()
    d_s_flat[:, -1] = 0.0
    hedge_hold = (dlt * d_s).sum(axis=1)
    hedge_flat = (dlt * d_s_flat).sum(axis=1)

    settle = np.maximum(s_settle - k_c, 0.0) + np.maximum(k_p - s_settle, 0.0)
    tot15 = tot[:, -1]
    s15 = s_path[:, -1]
    intrinsic15 = np.maximum(s15 - k_c, 0.0) + np.maximum(k_p - s15, 0.0)
    good15 = np.isfinite(tot15) & (tot15 > 0.0) & np.isfinite(s15) & (s15 > 0.0)
    mark15 = np.where(
        good15,
        p43.pkg_price_vec(
            np.where(good15, tot15, 1.0),
            np.where(good15, s15, 1.0),
            np.full(n, k_c),
            np.full(n, k_p),
        ),
        intrinsic15,
    )

    # Hedge turnover, index units traded x the level they are traded at.
    pos_flat = dlt.copy()
    pos_flat[:, -1] = 0.0
    prev_h = np.zeros(n)
    prev_f = np.zeros(n)
    turn_h = np.zeros(n)
    turn_f = np.zeros(n)
    for k in range(j, n_k):
        turn_h += np.abs(dlt[:, k] - prev_h) * s_path[:, k]
        turn_f += np.abs(pos_flat[:, k] - prev_f) * s_path[:, k]
        prev_h = dlt[:, k]
        prev_f = pos_flat[:, k]
    turn_h += np.abs(prev_h) * s_settle
    turn_f += np.abs(prev_f) * s_path[:, -1]

    option_hold = -(settle - short_fill)
    option_flat = -(mark15 - short_fill)
    return {
        "option_hold_pts": option_hold,
        "option_flatten_pts": option_flat,
        "hedge_hold_pts": hedge_hold,
        "hedge_flatten_pts": hedge_flat,
        "pnl_hold_pts": option_hold + hedge_hold,
        "pnl_flatten_pts": option_flat + hedge_flat,
        "settle_intrinsic_pts": settle,
        "mark_1530_pts": mark15,
        "turnover_hold_pts": turn_h,
        "turnover_flatten_pts": turn_f,
        "cost_hold_pts": HEDGE_COST_BP * turn_h,
        "cost_flatten_pts": HEDGE_COST_BP * turn_f,
        "delta_entry": dlt[:, j],
        "delta_1530": dlt[:, -1],
        "entry_mid_pts": np.full(n, entry_mid),
    }


def spot_from_path(
    s_entry: float, j: int, n_k: int, cum: np.ndarray, s_rep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spot at every stamp, and the settlement print, from cumulative returns.

    ``cum`` (n x B) carries the cumulative log return from the entry stamp to
    the end of each of the B panel bars that span entry -> 16:00; the last
    column is the settlement print.  Stamps strictly before the entry keep the
    representative day's own levels (nothing is held there, so they never enter
    a P&L; they are carried so the array shape matches the book's).
    """
    n = cum.shape[0]
    s_path = np.repeat(s_rep[None, :], n, axis=0)
    n_after = n_k - j - 1
    s_path[:, j] = s_entry
    s_path[:, j + 1 :] = s_entry * np.exp(cum[:, :n_after])
    s_settle = s_entry * np.exp(cum[:, -1])
    return s_path, s_settle


# ------------------------------------------------------------- the panel ---
def panel_bars() -> dict[str, Any]:
    """Regular-hours 30-minute log returns and realized variance, by session.

    Returns the (session x 13) grids of ``sumret`` (the bar's log return),
    ``sumret2`` (the sum of squared one-minute log returns inside the bar) and
    ``numobs``, plus the per-session realized variance and its causal trailing
    21-session mean.  Weekday sessions only; the panel is 24-hour, so only the
    13 regular-hours stamps are kept.
    """
    d = pd.read_parquet(PANEL, columns=["endbartime", "sumret", "sumret2", "numobs"])
    t = pd.to_datetime(d["endbartime"])
    d = d.assign(t=t, date=t.dt.normalize(), hhmm=t.dt.strftime("%H:%M"))
    w = d[d["hhmm"].isin(RTH_BARS) & (d["t"].dt.weekday < 5)]

    def piv(col: str) -> pd.DataFrame:
        return w.pivot_table(
            index="date", columns="hhmm", values=col, aggfunc="first"
        ).reindex(columns=list(RTH_BARS))

    ret = piv("sumret")
    rv = piv("sumret2")
    nobs = piv("numobs")
    n_bars = int(np.isfinite(ret.to_numpy(float)).sum(axis=1).max())
    bar_count = pd.Series(
        np.isfinite(ret.to_numpy(float)).sum(axis=1), index=ret.index, name="n_bars"
    )
    full = bar_count == N_RTH_BARS
    rv_day = pd.Series(
        np.where(full.to_numpy(), rv.to_numpy(float).sum(axis=1), np.nan),
        index=ret.index,
    )
    # The trailing window is taken over the sessions that HAVE a full-day
    # realized variance, so that a market holiday or a half session the panel
    # still stamps does not blank the next 21 windows.  shift(1) on that
    # compacted series makes the mean strictly causal: it never contains the
    # replayed session's own variance.
    rv_ok = rv_day.dropna()
    trail = (
        rv_ok.rolling(RV_WINDOW, min_periods=RV_WINDOW)
        .mean()
        .shift(1)
        .reindex(rv_day.index, method="ffill")
    )
    return {
        "ret": ret,
        "rv": rv,
        "numobs": nobs,
        "bar_count": bar_count,
        "rv_day": rv_day,
        "rv_trail": trail,
        "max_bars": n_bars,
    }


def window_bars(entry: str) -> list[str]:
    """The panel bars that span the entry stamp through the settlement print."""
    return list(RTH_BARS[RTH_BARS.index(entry) + 1 :])


def window_set(pan: dict[str, Any], entry: str) -> dict[str, Any]:
    """Sessions with a complete bar path from ``entry`` to 16:00, and the skips."""
    bars = window_bars(entry)
    sub = pan["ret"][bars]
    finite = np.isfinite(sub.to_numpy(float))
    ok = finite.all(axis=1)
    idx = pd.DatetimeIndex(sub.index)
    ret = sub.to_numpy(float)[ok]
    dates = idx[ok]
    skipped = idx[~ok]
    pre98 = skipped[skipped < pd.Timestamp(PANEL_FIRST_RETURN)]
    post98 = skipped[skipped >= pd.Timestamp(PANEL_FIRST_RETURN)]
    return {
        "bars": bars,
        "dates": dates,
        "ret": ret,
        "cum": np.cumsum(ret, axis=1),
        "n_stamped": int(len(idx)),
        "n_replayed": int(ok.sum()),
        "n_skipped": int((~ok).sum()),
        "skipped_pre_1998": int(len(pre98)),
        "skipped_1998_on": int(len(post98)),
        "skipped_1998_on_dates": post98,
    }


# --------------------------------------------------------------- the books --
_P43_CACHE: list[ModuleType] = []


def _worker_p43() -> ModuleType:
    """Proposal 43, imported once per worker process."""
    if not _P43_CACHE:
        _P43_CACHE.append(load_by_path("p43_for_49_worker", P43_PATH))
    return _P43_CACHE[0]


def _replay_job(job: dict[str, Any]) -> pd.DataFrame:
    """One book: one representative day x entry clock x IV convention.

    Both terminals are priced in the same pass (they share the delta tape), so
    a job returns two blocks of rows.  Run in a worker process; the work is a
    handful of (n_sessions x 12) numpy expressions and nothing is shared.
    """
    p43 = _worker_p43()
    j = int(job["j"])
    n_k = int(job["n_k"])
    cum = job["cum"]
    s_path, s_settle = spot_from_path(float(job["s_entry"]), j, n_k, cum, job["s_rep"])
    tot = np.repeat(job["tot_rep"][None, :], cum.shape[0], axis=0)
    tot = tot * np.sqrt(job["var_scale"])[:, None]
    res = price_book(
        p43,
        tot,
        s_path,
        s_settle,
        float(job["k_c"]),
        float(job["k_p"]),
        float(job["short_fill"]),
        float(job["entry_mid"]),
        j,
    )
    ret = job["ret"]
    net = ret.sum(axis=1)
    absum = np.abs(ret).sum(axis=1)
    big = np.abs(ret).max(axis=1)
    big_k = np.abs(ret).argmax(axis=1)
    bars = list(job["bars"])
    entry_mid = float(job["entry_mid"])
    frames = []
    for terminal, o_key, h_key, p_key, c_key in (
        (
            "hold to cash settlement",
            "option_hold_pts",
            "hedge_hold_pts",
            "pnl_hold_pts",
            "cost_hold_pts",
        ),
        (
            "flatten at 15:30",
            "option_flatten_pts",
            "hedge_flatten_pts",
            "pnl_flatten_pts",
            "cost_flatten_pts",
        ),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "book": job["book"],
                    "day_role": job["day_role"],
                    "rep_date": job["rep_date"],
                    "entry_clock": job["entry_clock"],
                    "terminal": terminal,
                    "iv_mode": job["iv_mode"],
                    "session": pd.DatetimeIndex(job["dates"]).strftime("%Y-%m-%d"),
                    "era": job["era"],
                    "var_scale": job["var_scale"],
                    "var_scale_at_floor": job["var_scale_at_floor"],
                    "option_pts": res[o_key],
                    "hedge_pts": res[h_key],
                    "pnl_pts": res[p_key],
                    "option_units": res[o_key] / entry_mid,
                    "hedge_units": res[h_key] / entry_mid,
                    "pnl_units": res[p_key] / entry_mid,
                    "option_dollars": res[o_key] * SPX_MULT,
                    "hedge_dollars": res[h_key] * SPX_MULT,
                    "pnl_dollars": res[p_key] * SPX_MULT,
                    "hedge_cost_pts": res[c_key],
                    "S_settle_replayed": s_settle,
                    "settle_intrinsic_pts": res["settle_intrinsic_pts"],
                    "path_net_logpct": 100.0 * net,
                    "path_sum_abs_bar_logpct": 100.0 * absum,
                    "path_trend_ratio": np.where(
                        absum > 0, np.abs(net) / absum, np.nan
                    ),
                    "path_max_abs_bar_logpct": 100.0 * big,
                    "path_max_abs_bar_stamp": [bars[k] for k in big_k],
                    "path_last_bar_logpct": 100.0 * ret[:, -1],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def era_of(dates: pd.DatetimeIndex) -> np.ndarray:
    out = np.full(len(dates), "other", dtype=object)
    for name, lo, hi in ERAS:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        out[np.asarray(m)] = name
    return out


# ------------------------------------------------------------- statistics --
def tail_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Moments and the three loss percentiles of one leg, in one unit."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    out: dict[str, float] = {
        f"{prefix}mean": float(v.mean()),
        f"{prefix}median": float(np.median(v)),
        f"{prefix}sd": float(v.std(ddof=1)),
        f"{prefix}min": float(v.min()),
        f"{prefix}max": float(v.max()),
    }
    for q in TAIL_PCTS:
        out[f"{prefix}p{q:g}"] = float(np.percentile(v, q))
    thr = float(np.percentile(v, ES_PCT))
    sel = v[v <= thr]
    out[f"{prefix}ES_below_p{ES_PCT:g}"] = (
        float(sel.mean()) if sel.size else float("nan")
    )
    out[f"{prefix}n_below_p{ES_PCT:g}"] = float(sel.size)
    return out


def worst_anatomy(d: pd.DataFrame) -> dict[str, float]:
    """The worst-1% set, split by a stated mechanical rule.

    last bar          the largest |bar| of the window IS the 16:00 bar
    earlier bar       the largest |bar| is earlier AND |net| / sum|bar| >= 0.5
    whipsaw           the largest |bar| is earlier AND |net| / sum|bar| <  0.5
    """
    k = max(1, int(round(WORST_FRAC * len(d))))
    w = d.nsmallest(k, "pnl_pts")
    last = w["path_max_abs_bar_stamp"] == SETTLE_BAR
    trend = w["path_trend_ratio"] >= TREND_BAR
    cls = np.where(last, "last bar", np.where(trend, "earlier bar", "whipsaw"))
    tot = float(w["pnl_pts"].sum())
    out: dict[str, float] = {"w1_n": float(k), "w1_total_pts": tot}
    for name in ("last bar", "earlier bar", "whipsaw"):
        m = cls == name
        key = name.replace(" ", "_")
        out[f"w1_n_{key}"] = float(m.sum())
        out[f"w1_share_n_{key}"] = float(m.sum()) / k
        s = float(w.loc[m, "pnl_pts"].sum())
        out[f"w1_pts_{key}"] = s
        out[f"w1_share_loss_{key}"] = s / tot if tot != 0.0 else float("nan")
        out[f"w1_mean_pts_{key}"] = (
            float(w.loc[m, "pnl_pts"].mean()) if m.sum() else float("nan")
        )
    return out


def summarise(dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (book, role, rep_date, clock, term, mode), d in dist.groupby(
        ["book", "day_role", "rep_date", "entry_clock", "terminal", "iv_mode"],
        sort=False,
    ):
        row: dict[str, Any] = {
            "book": book,
            "day_role": role,
            "rep_date": rep_date,
            "entry_clock": clock,
            "terminal": term,
            "iv_mode": mode,
            "n_sessions": int(len(d)),
            "n_var_scale_at_floor": int(d["var_scale_at_floor"].sum()),
            "var_scale_max": float(d["var_scale"].max()),
            "mean_hedge_cost_pts": float(d["hedge_cost_pts"].mean()),
        }
        for leg, col in (
            ("total_", "pnl"),
            ("option_", "option"),
            ("hedge_", "hedge"),
        ):
            for unit in ("pts", "units", "dollars"):
                row.update(
                    tail_stats(d[f"{col}_{unit}"].to_numpy(float), f"{leg}{unit}_")
                )
        row.update(worst_anatomy(d))
        rows.append(row)
    return pd.DataFrame(rows)


def worst_table(dist: pd.DataFrame) -> pd.DataFrame:
    keys = ["book", "day_role", "rep_date", "entry_clock", "terminal", "iv_mode"]
    cols = keys + [
        "session",
        "era",
        "pnl_pts",
        "pnl_units",
        "pnl_dollars",
        "option_pts",
        "hedge_pts",
        "var_scale",
        "path_net_logpct",
        "path_sum_abs_bar_logpct",
        "path_trend_ratio",
        "path_max_abs_bar_logpct",
        "path_max_abs_bar_stamp",
        "path_last_bar_logpct",
        "S_settle_replayed",
    ]
    out = []
    for _, d in dist.groupby(keys, sort=False):
        w = d.nsmallest(WORST_N, "pnl_pts").copy()
        w["rank"] = np.arange(1, len(w) + 1)
        out.append(w[cols + ["rank"]])
    return pd.concat(out, ignore_index=True)


def era_table(dist: pd.DataFrame) -> pd.DataFrame:
    keys = ["book", "day_role", "rep_date", "entry_clock", "terminal", "iv_mode"]
    rows = []
    for k, d in dist.groupby(keys, sort=False):
        for era_name, _lo, _hi in ERAS:
            sel = d[d["era"] == era_name]
            row = dict(zip(keys, k, strict=True))
            row["era"] = era_name
            row["n_sessions"] = int(len(sel))
            if len(sel) == 0:
                rows.append(row)
                continue
            for leg, col in (
                ("total_", "pnl"),
                ("option_", "option"),
                ("hedge_", "hedge"),
            ):
                v = sel[f"{col}_pts"].to_numpy(float)
                row[f"{leg}mean_pts"] = float(v.mean())
                row[f"{leg}sd_pts"] = float(v.std(ddof=1))
                row[f"{leg}p1_pts"] = float(np.percentile(v, 1.0))
                row[f"{leg}min_pts"] = float(v.min())
                row[f"{leg}mean_dollars"] = float(v.mean()) * SPX_MULT
                row[f"{leg}p1_dollars"] = float(np.percentile(v, 1.0)) * SPX_MULT
            u = sel["pnl_units"].to_numpy(float)
            row["total_mean_units"] = float(u.mean())
            row["total_p1_units"] = float(np.percentile(u, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def rolling_worst(x: np.ndarray, window: int) -> float:
    """The worst ``window``-session cumulative sum of a chronological series."""
    v = np.asarray(x, float)
    if v.size < window:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(v)])
    return float((c[window:] - c[:-window]).min())


# -------------------------------------------------------------------- main --
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.max_rows", 900)
    t_start = time.time()

    p43 = load_by_path("p43_for_49", P43_PATH)
    clocks: tuple[str, ...] = tuple(p43.CLOCKS)
    n_k = len(clocks)

    # ------------------------------------------------------ the panel ------
    pan = panel_bars()
    ret_all = pan["ret"]
    print(
        f"panel data/core_stats.parquet: {len(ret_all)} weekday sessions carrying at "
        f"least one of the {N_RTH_BARS} regular-hours 30-minute stamps, "
        f"{pd.DatetimeIndex(ret_all.index).min().date()}.."
        f"{pd.DatetimeIndex(ret_all.index).max().date()}; sumret is empty before "
        f"{PANEL_FIRST_RETURN}"
    )
    bc = pan["bar_count"].value_counts().sort_index()
    print(
        "regular-hours bar count per session: "
        + ", ".join(f"{int(k)} bars x {int(v)} sessions" for k, v in bc.items())
    )

    windows = {c: window_set(pan, c) for c in ENTRY_CLOCKS}
    for c in ENTRY_CLOCKS:
        w = windows[c]
        print(
            f"entry {c}: window bars {', '.join(w['bars'])} ({len(w['bars'])} bars); "
            f"sessions replayed {w['n_replayed']}, skipped {w['n_skipped']} "
            f"(missing a bar) of which {w['skipped_pre_1998']} are before "
            f"{PANEL_FIRST_RETURN} (no return data) and {w['skipped_1998_on']} are "
            f"1998 or later; replay span "
            f"{w['dates'].min().date()}..{w['dates'].max().date()}"
        )

    # ------------------------------------------------- proposal 43's tape --
    stamp, _px = p43.session_stamps()
    half = p43.half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    print(
        f"\nchain sessions {len(stamp)}; half sessions dropped "
        f"(hours_to_expiration <= 0 at 15:30, {len(half)}); sessions scored "
        f"{len(sessions)}"
    )
    ch = p43.build_chain(stamp, sessions)
    idx = pd.DatetimeIndex(ch["dates"])

    # ---------------------------------------------------------- GATE 28 ----
    p28 = load_by_path("p28_for_49", P28_PATH)
    mk = p28._standalone()
    pkg28 = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    g28 = p28.build_grids(pkg28, mk)
    e28 = g28["sl"]["entry"].to_numpy(float)
    ratio28 = pd.Series(e28 / g28["Sg"][:, g28["j"]], index=g28["dates"]).sort_values()
    rep28 = pd.Timestamp(ratio28.index[len(ratio28) // 2])
    assert rep28 == pd.Timestamp(GATE28_REP_DAY), rep28
    i28 = g28["dates"].get_loc(rep28)
    j28 = int(g28["j"])
    tot28 = np.where(
        g28["IVg"][i28] > 0, g28["IVg"][i28] * np.sqrt(g28["h_row"]), np.nan
    )
    w28 = windows["11:00"]
    pos28 = {d.strftime("%Y-%m-%d"): k for k, d in enumerate(w28["dates"])}
    rows28 = np.array([pos28[s] for s in GATE28_SESSIONS])
    s_path28, s_settle28 = spot_from_path(
        float(g28["Sg"][i28, j28]), j28, n_k, w28["cum"][rows28], g28["Sg"][i28]
    )
    r28 = price_book(
        p43,
        np.repeat(tot28[None, :], len(rows28), axis=0),
        s_path28,
        s_settle28,
        float(g28["sl"]["K_c"].to_numpy(float)[i28]),
        float(g28["sl"]["K_p"].to_numpy(float)[i28]),
        float(e28[i28]),  # proposal 28's book is sold at the MIDPOINT
        float(e28[i28]),
        j28,
    )
    mine28 = r28["pnl_hold_pts"] / float(e28[i28])
    ref28 = pd.read_csv(REF28).set_index("session")["r_hold_units"]
    for k, s in enumerate(GATE28_SESSIONS):
        dev = abs(float(mine28[k]) - float(ref28.loc[s]))
        dev_p = abs(float(mine28[k]) - GATE28_PRINTED[s])
        assert dev < GATE28_TOL, (s, dev)
        assert dev_p < GATE28_PRINT_TOL, (s, dev_p)
        print(
            f"GATE 28  {rep28.date()} book, {s} path: r_hold {float(mine28[k]):+.12f} "
            f"units vs proposal 28's {float(ref28.loc[s]):+.12f} (|dev| {dev:.1e}, "
            f"tolerance {GATE28_TOL:.0e}); its printed {GATE28_PRINTED[s]:+.2f} "
            f"(|dev| {dev_p:.1e})"
        )

    # ---------------------------------------------------------- GATE 45 ----
    j13 = clocks.index(MAIN_ENTRY)
    ok13 = np.isfinite(ch["entry"][:, j13]) & (ch["entry"][:, j13] > 0.0)
    ratio45 = (
        pd.Series(
            np.where(ok13, ch["entry"][:, j13] / ch["S"][:, j13], np.nan), index=idx
        )
        .dropna()
        .sort_values()
    )
    rep45 = pd.Timestamp(ratio45.index[len(ratio45) // 2])
    assert rep45 == pd.Timestamp(GATE45_REP_DAY), rep45
    i45 = int(idx.get_loc(rep45))
    s_row45 = ch["S"][i45].copy()
    s_settle45 = np.array([float(s_row45[-1]) * (1.0 + GATE45_JUMP)])
    r45 = price_book(
        p43,
        ch["tot"][i45][None, :].copy(),
        s_row45[None, :].copy(),
        s_settle45,
        float(ch["K_c"][i45, j13]),
        float(ch["K_p"][i45, j13]),
        float(ch["bid"][i45, j13]),  # proposal 43/45's book is sold at the BID
        float(ch["entry"][i45, j13]),
        j13,
    )
    fl_units = float(r45["option_hold_pts"][0]) / float(ch["entry"][i45, j13])
    fl_dollars = float(r45["option_hold_pts"][0]) * SPX_MULT
    hg_dollars = float(r45["hedge_hold_pts"][0]) * SPX_MULT
    ref45 = pd.read_csv(REF45)
    sel45 = ref45[
        ref45["structure"].str.startswith("naked")
        & (ref45["placement"] == "last bar")
        & np.isclose(ref45["jump_pct"], 100.0 * GATE45_JUMP)
    ].iloc[0]
    for name, mine, ref, printed in (
        (
            "four option legs, units",
            fl_units,
            float(sel45["four_leg_units"]),
            GATE45_FOUR_LEG_UNITS,
        ),
        (
            "four option legs, dollars",
            fl_dollars,
            float(sel45["four_leg_dollars"]),
            GATE45_FOUR_LEG_DOLLARS,
        ),
        (
            "hedge, dollars",
            hg_dollars,
            float(sel45["hedge_dollars"]),
            GATE45_HEDGE_DOLLARS,
        ),
    ):
        dev = abs(mine - ref)
        dev_p = abs(mine - printed) / max(1.0, abs(printed))
        assert dev < GATE45_TOL, (name, dev)
        assert dev_p < GATE45_PRINT_TOL, (name, dev_p)
        print(
            f"GATE 45  {rep45.date()} book, +{100 * GATE45_JUMP:.0f}% last bar, "
            f"{name}: {mine:+.6f} vs proposal 45's {ref:+.6f} (|dev| {dev:.1e}, "
            f"tolerance {GATE45_TOL:.0e}); its stated {printed:+.6f}"
        )

    # ------------------------------------------- the representative days ---
    deck = np.asarray(idx <= pd.Timestamp(DECK_END))
    reps: list[dict[str, Any]] = []
    for clock in ENTRY_CLOCKS:
        j = clocks.index(clock)
        ok = np.isfinite(ch["entry"][:, j]) & (ch["entry"][:, j] > 0.0) & deck
        r = (
            pd.Series(
                np.where(ok, ch["entry"][:, j] / ch["S"][:, j], np.nan), index=idx
            )
            .dropna()
            .sort_values()
        )
        qs = REP_QUANTILES if clock == MAIN_ENTRY else (0.50,)
        print(
            f"\nrepresentative days at {clock}: nearest-rank quantiles of the "
            f"{len(r)} deck-scored sessions' entry premium / spot "
            f"(deck = chain sessions through {DECK_END})"
        )
        for q in qs:
            k = int(np.ceil(q * len(r)))
            day = pd.Timestamp(r.index[k - 1])
            i = int(idx.get_loc(day))
            reps.append(
                {
                    "day_role": REP_LABEL[q],
                    "entry_clock": clock,
                    "quantile": q,
                    "rank": k,
                    "rep_date": day.strftime("%Y-%m-%d"),
                    "i": i,
                    "j": j,
                    "S_entry": float(ch["S"][i, j]),
                    "K_c": float(ch["K_c"][i, j]),
                    "K_p": float(ch["K_p"][i, j]),
                    "entry_mid_pts": float(ch["entry"][i, j]),
                    "entry_bid_pts": float(ch["bid"][i, j]),
                    "premium_over_spot_pct": 100.0 * float(r.iloc[k - 1]),
                    "total_vol_entry": float(ch["tot"][i, j]),
                    "total_vol_1530": float(ch["tot"][i, -1]),
                    "S_1530": float(ch["S"][i, -1]),
                    "S_close_actual": float(ch["S_close"][i]),
                }
            )
            rr = reps[-1]
            print(
                f"  {REP_LABEL[q]:>16s} (q={q:.2f}, rank {k}/{len(r)}): "
                f"{rr['rep_date']}  S({clock}) {rr['S_entry']:.2f}  K_c "
                f"{rr['K_c']:.0f}  K_p {rr['K_p']:.0f}  entry mid "
                f"{rr['entry_mid_pts']:.2f} pts (bid {rr['entry_bid_pts']:.2f}) = "
                f"{rr['premium_over_spot_pct']:.4f}% of spot; total vol at entry "
                f"{rr['total_vol_entry']:.6f}, at 15:30 {rr['total_vol_1530']:.6f}; "
                f"actual settle {rr['S_close_actual']:.2f}"
            )
    rep_tab = pd.DataFrame(reps)
    print(
        f"\n(proposal 45's own representative day is {rep45.date()}, the median of "
        f"all {len(ratio45)} chain sessions rather than of the deck-scored ones; "
        f"it is the GATE 45 day and is not one of the {len(rep_tab)} books' days)"
    )

    # ------------------------------------------- the regime-scaling factor --
    trail = pan["rv_trail"]
    jobs: list[dict[str, Any]] = []
    for rr in reps:
        clock = str(rr["entry_clock"])
        w = windows[clock]
        rep_day = pd.Timestamp(str(rr["rep_date"]))
        rv_rep = float(trail.reindex([rep_day]).to_numpy(float)[0])
        assert np.isfinite(rv_rep), (rep_day, rv_rep)
        rv_hist = trail.reindex(w["dates"]).to_numpy(float)
        raw = rv_hist / rv_rep
        at_floor = ~(np.isfinite(raw) & (raw > 1.0))
        scale = np.where(at_floor, 1.0, raw)
        eras = era_of(w["dates"])
        for mode in IV_MODES:
            v = np.ones(len(w["dates"])) if mode == IV_MODES[0] else scale
            fl = np.zeros(len(w["dates"]), bool) if mode == IV_MODES[0] else at_floor
            jobs.append(
                {
                    "book": f"{rr['day_role']} | {clock} | {mode}",
                    "day_role": rr["day_role"],
                    "rep_date": rr["rep_date"],
                    "entry_clock": clock,
                    "iv_mode": mode,
                    "j": int(rr["j"]),
                    "n_k": n_k,
                    "cum": w["cum"],
                    "ret": w["ret"],
                    "bars": w["bars"],
                    "dates": w["dates"],
                    "era": eras,
                    "s_entry": rr["S_entry"],
                    "s_rep": ch["S"][int(rr["i"])].copy(),
                    "tot_rep": ch["tot"][int(rr["i"])].copy(),
                    "k_c": rr["K_c"],
                    "k_p": rr["K_p"],
                    "short_fill": rr["entry_bid_pts"],
                    "entry_mid": rr["entry_mid_pts"],
                    "var_scale": v,
                    "var_scale_at_floor": fl,
                    "rv_trail_rep": rv_rep,
                }
            )
        print(
            f"regime scaling for {rr['day_role']} @ {clock} ({rr['rep_date']}): "
            f"the representative day's trailing {RV_WINDOW}-session realized "
            f"variance {rv_rep:.3e}; over the {len(w['dates'])} replayed sessions "
            f"the factor max(RV21/RV21_rep, 1) has median {np.median(scale):.3f}, "
            f"mean {scale.mean():.3f}, max {scale.max():.3f}, and sits at its floor "
            f"on {int(at_floor.sum())} sessions ({int(np.isnan(rv_hist).sum())} of "
            f"them for want of a complete trailing window)"
        )

    # ------------------------------------------------------- the replay ----
    t0 = time.time()
    n_workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        parts = list(pool.map(_replay_job, jobs))
    dist = pd.concat(parts, ignore_index=True)
    print(
        f"\nreplay: {len(jobs)} books x {len(TERMINALS)} terminals on "
        f"{n_workers} worker processes, {len(dist):,} session rows, "
        f"{time.time() - t0:.1f} s"
    )

    # -------------------------------------------------------- A. tables ----
    summ = summarise(dist)
    worst = worst_table(dist)
    eras_tab = era_table(dist)

    print(
        "\n--- A1. the loss distribution, per contract.  Units are of the "
        "representative day's entry MIDPOINT premium; the short is filled at "
        "that day's BID; dollars are index points x $100.  Percentiles are "
        "numpy linear interpolation on the replayed sessions."
    )
    show = [
        "day_role",
        "rep_date",
        "entry_clock",
        "terminal",
        "iv_mode",
        "n_sessions",
        "total_pts_mean",
        "total_pts_median",
        "total_pts_sd",
        "total_pts_p1",
        "total_pts_p0.5",
        "total_pts_p0.1",
        "total_pts_ES_below_p1",
        "total_pts_min",
    ]
    print(summ[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n--- A1b. the same, in dollars per contract")
    show_d = [
        "day_role",
        "entry_clock",
        "terminal",
        "iv_mode",
        "total_dollars_mean",
        "total_dollars_p1",
        "total_dollars_p0.5",
        "total_dollars_p0.1",
        "total_dollars_ES_below_p1",
        "total_dollars_min",
    ]
    print(summ[show_d].to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print("\n--- A1c. the same, in units of the representative day's entry premium")
    show_u = [
        "day_role",
        "entry_clock",
        "terminal",
        "iv_mode",
        "total_units_mean",
        "total_units_p1",
        "total_units_p0.5",
        "total_units_p0.1",
        "total_units_ES_below_p1",
        "total_units_min",
    ]
    print(summ[show_u].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print(
        "\n--- A2. the two legs apart: the four option legs are bounded below by "
        "-(settlement intrinsic) + fill; the FUTURES HEDGE IS UNBOUNDED."
    )
    show_l = [
        "day_role",
        "entry_clock",
        "terminal",
        "iv_mode",
        "option_pts_mean",
        "option_pts_p1",
        "option_pts_p0.5",
        "option_pts_p0.1",
        "option_pts_min",
        "hedge_pts_mean",
        "hedge_pts_p1",
        "hedge_pts_p0.5",
        "hedge_pts_p0.1",
        "hedge_pts_min",
        "hedge_pts_max",
    ]
    print(summ[show_l].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print(
        f"\n--- A3. the worst {WORST_N} sessions of every book, with the path "
        "shape.  trend ratio = |net move| / sum |bar moves|."
    )
    print(
        worst[
            [
                "day_role",
                "entry_clock",
                "terminal",
                "iv_mode",
                "rank",
                "session",
                "pnl_pts",
                "pnl_units",
                "pnl_dollars",
                "option_pts",
                "hedge_pts",
                "path_net_logpct",
                "path_sum_abs_bar_logpct",
                "path_trend_ratio",
                "path_max_abs_bar_logpct",
                "path_max_abs_bar_stamp",
                "path_last_bar_logpct",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )

    print(
        f"\n--- A4. the worst-1% set, split by a mechanical rule: 'last bar' = the "
        f"largest |bar| of the window is the {SETTLE_BAR} bar; 'earlier bar' = the "
        f"largest |bar| is earlier and the trend ratio is >= {TREND_BAR}; "
        f"'whipsaw' = the largest |bar| is earlier and the trend ratio is "
        f"< {TREND_BAR}."
    )
    show_w = [
        "day_role",
        "entry_clock",
        "terminal",
        "iv_mode",
        "w1_n",
        "w1_n_last_bar",
        "w1_n_earlier_bar",
        "w1_n_whipsaw",
        "w1_share_loss_last_bar",
        "w1_share_loss_earlier_bar",
        "w1_share_loss_whipsaw",
        "w1_mean_pts_last_bar",
        "w1_mean_pts_earlier_bar",
        "w1_mean_pts_whipsaw",
    ]
    print(summ[show_w].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n--- A5. per era, index points per contract")
    print(
        eras_tab[
            [
                "day_role",
                "entry_clock",
                "terminal",
                "iv_mode",
                "era",
                "n_sessions",
                "total_mean_pts",
                "total_p1_pts",
                "total_min_pts",
                "option_mean_pts",
                "option_p1_pts",
                "hedge_mean_pts",
                "hedge_p1_pts",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )

    # ------------------------ A6. the replay's 2020-2024 slice vs the book --
    j_main = clocks.index(MAIN_ENTRY)
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    b13 = {
        k: pd.Series(v, index=idx) for k, v in p43._clock_book((j_main, arrays)).items()
    }
    realised_units = b13["hold"].dropna()
    realised_pts = (b13["hold"] * b13["entry"]).dropna()
    era2020 = dist[(dist["era"] == "2020-2024")]["session"].unique()
    common = pd.DatetimeIndex(
        sorted(set(pd.DatetimeIndex(era2020)) & set(realised_pts.index))
    )
    rp = realised_pts.reindex(common)
    ru = realised_units.reindex(common)
    cmp_rows = [
        {
            "series": "actual 13:30 hold book (proposal 43, quoted bid fill, own premium)",
            "iv_mode": "actual market",
            "n_sessions": int(len(common)),
            "mean_pts": float(rp.mean()),
            "median_pts": float(rp.median()),
            "sd_pts": float(rp.std(ddof=1)),
            "p1_pts": float(np.percentile(rp.to_numpy(float), 1.0)),
            "min_pts": float(rp.min()),
            "mean_units": float(ru.mean()),
            "p1_units": float(np.percentile(ru.to_numpy(float), 1.0)),
            "min_units": float(ru.min()),
            "mean_entry_premium_pts": float(b13["entry"].reindex(common).mean()),
        }
    ]
    for mode in IV_MODES:
        d = dist[
            (dist["day_role"] == MEDIAN_ROLE)
            & (dist["entry_clock"] == MAIN_ENTRY)
            & (dist["terminal"] == TERMINALS[0])
            & (dist["iv_mode"] == mode)
        ]
        d = d[d["session"].isin(common.strftime("%Y-%m-%d"))]
        cmp_rows.append(
            {
                "series": f"replay, {MEDIAN_ROLE} book, same sessions",
                "iv_mode": mode,
                "n_sessions": int(len(d)),
                "mean_pts": float(d["pnl_pts"].mean()),
                "median_pts": float(d["pnl_pts"].median()),
                "sd_pts": float(d["pnl_pts"].std(ddof=1)),
                "p1_pts": float(np.percentile(d["pnl_pts"].to_numpy(float), 1.0)),
                "min_pts": float(d["pnl_pts"].min()),
                "mean_units": float(d["pnl_units"].mean()),
                "p1_units": float(np.percentile(d["pnl_units"].to_numpy(float), 1.0)),
                "min_units": float(d["pnl_units"].min()),
                "mean_entry_premium_pts": float(
                    rep_tab.loc[
                        (rep_tab["day_role"] == MEDIAN_ROLE)
                        & (rep_tab["entry_clock"] == MAIN_ENTRY),
                        "entry_mid_pts",
                    ].iloc[0]
                ),
            }
        )
    cmp_tab = pd.DataFrame(cmp_rows)
    print(
        "\n--- A6. the 2020-2024 replay slice against the book's ACTUAL realised "
        "13:30 hold tape on the same sessions.  The actual book's premium "
        "co-moved with volatility; the replay's premium is frozen at the "
        "representative day's, so the two series are not the same experiment."
    )
    print(cmp_tab.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ------------------------------------------------------ B. sizing ------
    med = rep_tab[
        (rep_tab["day_role"] == MEDIAN_ROLE) & (rep_tab["entry_clock"] == MAIN_ENTRY)
    ].iloc[0]
    entry_mid_med = float(med["entry_mid_pts"])
    series: dict[str, np.ndarray] = {}
    means: dict[str, float] = {}
    worst20: dict[str, float] = {}
    for mode in IV_MODES:
        d = dist[
            (dist["day_role"] == MEDIAN_ROLE)
            & (dist["entry_clock"] == MAIN_ENTRY)
            & (dist["terminal"] == TERMINALS[0])
            & (dist["iv_mode"] == mode)
        ].sort_values("session")
        v = d["pnl_pts"].to_numpy(float)
        series[mode] = v
        means[mode] = float(v.mean())
        worst20[mode] = rolling_worst(v, RUIN_WINDOW)

    loss_sources: list[tuple[str, str, float]] = []
    for mode in IV_MODES:
        loss_sources.append(
            (
                f"replay {mode}, {SIZING_TAIL_PCT:g}th percentile session (1-in-200)",
                f"{MEDIAN_ROLE} {med['rep_date']}",
                float(np.percentile(series[mode], SIZING_TAIL_PCT)),
            )
        )
    naked45 = ref45[ref45["structure"].str.startswith("naked")]
    for _, r in naked45[
        np.isclose(np.abs(naked45["jump_pct"]), 100.0 * GATE45_JUMP)
    ].iterrows():
        loss_sources.append(
            (
                f"proposal 45 stress, {r['placement']} {r['jump_pct']:+.0f}%",
                f"proposal 45 representative day {rep45.date()}",
                float(r["pnl_pts"]),
            )
        )
    realised_mean_pts = float(rp.mean())
    b_rows = []
    for name, loss_book, loss_pts in loss_sources:
        for budget in CAPITAL_BUDGETS:
            cap_pts = abs(loss_pts) / budget
            row: dict[str, Any] = {
                "loss_source": name,
                "loss_book": loss_book,
                "loss_pts": loss_pts,
                "loss_dollars": loss_pts * SPX_MULT,
                "loss_units_of_median_rep_book": loss_pts / entry_mid_med,
                "budget_frac_of_capital": budget,
                "capital_pts": cap_pts,
                "capital_dollars": cap_pts * SPX_MULT,
                "capital_over_entry_premium": cap_pts / entry_mid_med,
            }
            for mode in IV_MODES:
                tag = "frozen" if mode == IV_MODES[0] else "regime"
                row[f"mean_{tag}_pts"] = means[mode]
                row[f"ann_mean_{tag}_dollars"] = (
                    means[mode] * asl.PERIODS_PER_YEAR * SPX_MULT
                )
                row[f"ann_return_on_capital_{tag}"] = (
                    means[mode] * asl.PERIODS_PER_YEAR / cap_pts
                )
                row[f"worst{RUIN_WINDOW}_{tag}_pts"] = worst20[mode]
                row[f"worst{RUIN_WINDOW}_{tag}_frac_capital"] = worst20[mode] / cap_pts
            row["mean_realised_2020_2024_pts"] = realised_mean_pts
            row["ann_return_on_capital_realised"] = (
                realised_mean_pts * asl.PERIODS_PER_YEAR / cap_pts
            )
            b_rows.append(row)
    b_siz = pd.DataFrame(b_rows)
    print(
        f"\n--- B. sizing from the distribution, {MEDIAN_ROLE} book "
        f"({med['rep_date']}), {MAIN_ENTRY} entry, hold to cash settlement, one "
        f"contract.  Capital = |loss| / budget.  The annual mean is the replay's "
        f"own mean x {asl.PERIODS_PER_YEAR:.0f}; the realised column instead uses "
        f"the actual 2020-2024 13:30 hold mean ({realised_mean_pts:+.4f} index "
        f"points a day on {len(common)} sessions).  worst{RUIN_WINDOW} is the "
        f"minimum {RUIN_WINDOW}-session rolling sum of the replay series in "
        f"chronological order."
    )
    print(b_siz.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ------------------------------------------------- C. the blind spots --
    c_rows: list[dict[str, Any]] = []

    def cadd(block: str, item: str, n: Any, value: Any, unit: str, note: str) -> None:
        c_rows.append(
            {
                "block": block,
                "item": item,
                "n": n,
                "value": value,
                "unit": unit,
                "note": note,
            }
        )

    # C1: intra-bar path.  sumret2 (sum of squared 1-minute returns inside the
    # bar) against sumret^2 (the squared bar return).
    w_main = windows[MAIN_ENTRY]
    bars_main = w_main["bars"]
    med_book = dist[
        (dist["day_role"] == MEDIAN_ROLE)
        & (dist["entry_clock"] == MAIN_ENTRY)
        & (dist["terminal"] == TERMINALS[0])
        & (dist["iv_mode"] == IV_MODES[0])
    ]
    k_w1 = max(1, int(round(WORST_FRAC * len(med_book))))
    worst_sessions = pd.DatetimeIndex(
        med_book.nsmallest(k_w1, "pnl_pts")["session"].to_numpy()
    )
    worst10_sessions = pd.DatetimeIndex(
        med_book.nsmallest(WORST_N, "pnl_pts")["session"].to_numpy()
    )
    r_all = pan["ret"][bars_main].reindex(w_main["dates"]).to_numpy(float)
    q_all = pan["rv"][bars_main].reindex(w_main["dates"]).to_numpy(float)
    ratio_all = np.where(r_all**2 > 0.0, q_all / r_all**2, np.nan)
    bench = "a bar whose 30 one-minute returns were all equal has ratio 1/30 = 0.0333"
    for label, sel in (
        ("all replayed sessions", pd.DatetimeIndex(w_main["dates"])),
        (f"worst-1% sessions ({k_w1})", worst_sessions),
        (f"worst-{WORST_N} sessions", worst10_sessions),
    ):
        m = np.isin(pd.DatetimeIndex(w_main["dates"]).to_numpy(), sel.to_numpy())
        sub_ratio = ratio_all[m]
        v = sub_ratio[np.isfinite(sub_ratio)]
        cadd(
            "C1 intra-bar path",
            f"sumret2 / sumret^2 over the {len(bars_main)} window bars, {label}",
            int(v.size),
            float(np.median(v)),
            "ratio (median)",
            f"sumret2 is the sum of the squared one-minute returns inside the "
            f"bar, sumret^2 the square of the bar's net move; {bench}, and the "
            f"excess over that is path the 30-minute replay never sees",
        )
        q_sel = q_all[m]
        r2_sel = r_all[m] ** 2
        both = np.isfinite(q_sel) & np.isfinite(r2_sel)
        cadd(
            "C1 intra-bar path",
            f"sumret2 / sumret^2 over the {len(bars_main)} window bars, {label}",
            int(both.sum()),
            float(q_sel[both].sum() / r2_sel[both].sum()),
            "ratio (pooled sums)",
            "the sum of sumret2 over the selected bars divided by the sum of "
            "sumret^2; immune to the near-zero denominators that make a per-bar "
            "mean meaningless",
        )
        big = np.abs(r_all[m]).argmax(axis=1)
        rows_m = np.arange(int(m.sum()))
        rb = ratio_all[m][rows_m, big]
        rb = rb[np.isfinite(rb)]
        cadd(
            "C1 intra-bar path",
            f"the same ratio on each session's LARGEST window bar, {label}",
            int(rb.size),
            float(np.median(rb)),
            "ratio (median)",
            f"the bar that drives the loss; {bench}",
        )
        qb = q_all[m][rows_m, big]
        r2b = r_all[m][rows_m, big] ** 2
        okb = np.isfinite(qb) & np.isfinite(r2b)
        cadd(
            "C1 intra-bar path",
            f"the same ratio on each session's LARGEST window bar, {label}",
            int(okb.sum()),
            float(qb[okb].sum() / r2b[okb].sum()),
            "ratio (pooled sums)",
            "the bar that drives the loss",
        )
    nob = pan["numobs"][bars_main].reindex(w_main["dates"]).to_numpy(float)
    cadd(
        "C1 intra-bar path",
        "window bars carrying fewer than 30 one-minute observations",
        int(np.isfinite(nob).sum()),
        int((np.isfinite(nob) & (nob < 30.0)).sum()),
        "bars",
        "a short bar hides proportionally more of its own path",
    )

    # C2: the premium's co-movement with volatility, 2020-2024 overlap.
    prem_act = (ch["entry"][:, j_main] / ch["S"][:, j_main]) * 100.0
    prem_s = pd.Series(prem_act, index=idx).reindex(common)
    rv_s = pan["rv_trail"].reindex(common)
    dec = pd.qcut(rv_s, N_VOL_DECILES, labels=False, duplicates="drop")
    frozen_pct = float(med["premium_over_spot_pct"])
    for dv in sorted(pd.Series(dec).dropna().unique()):
        m = (dec == dv).to_numpy()
        cadd(
            "C2 premium vs volatility",
            f"decile {int(dv) + 1} of the trailing {RV_WINDOW}-session realized "
            f"variance, 2020-2024 overlap",
            int(m.sum()),
            float(prem_s[m].mean()),
            "actual 13:30 premium / spot, %",
            f"the replay holds it at {frozen_pct:.4f}% (ratio "
            f"{float(prem_s[m].mean()) / frozen_pct:.3f}); RV21 mean "
            f"{float(rv_s[m].mean()):.3e}",
        )
    cadd(
        "C2 premium vs volatility",
        "whole 2020-2024 overlap",
        int(len(common)),
        float(prem_s.mean()),
        "actual 13:30 premium / spot, %",
        f"the replay holds it at {frozen_pct:.4f}% (ratio "
        f"{float(prem_s.mean()) / frozen_pct:.3f})",
    )

    # C3: the settlement print.
    last_bar = pan["ret"][SETTLE_BAR].reindex(w_main["dates"]).to_numpy(float)
    cadd(
        "C3 settlement print",
        f"the panel's {SETTLE_BAR} bar, |log return|, replayed sessions",
        int(np.isfinite(last_bar).sum()),
        100.0 * float(np.nanmean(np.abs(last_bar))),
        "mean |move|, %",
        "the panel has no separate settlement print: the replay settles on the "
        f"close of the {SETTLE_BAR} bar",
    )
    cadd(
        "C3 settlement print",
        f"the panel's {SETTLE_BAR} bar, |log return|, replayed sessions",
        int(np.isfinite(last_bar).sum()),
        100.0 * float(np.nanpercentile(np.abs(last_bar), 99.0)),
        "99th percentile |move|, %",
        "same",
    )
    gap = ch["S_close"] / ch["S"][:, -1] - 1.0
    cadd(
        "C3 settlement print",
        "the live book's own 15:30 spot -> settlement move, chain sessions",
        int(np.isfinite(gap).sum()),
        100.0 * float(np.nanmean(np.abs(gap))),
        "mean |move|, %",
        "proposal 43's settlement is the GSPC close; the replay's is the panel's "
        f"{SETTLE_BAR} bar close, a different series",
    )

    # C4: half sessions.
    for n_b, cnt in pan["bar_count"].value_counts().sort_index().items():
        cadd(
            "C4 half sessions",
            f"weekday sessions with {int(n_b)} of the {N_RTH_BARS} regular-hours bars",
            int(cnt),
            int(n_b),
            "bars",
            "market holidays the panel still stamps, and half sessions; short "
            "of the full bar count they never carry a complete window and are "
            "skipped by the window filter, never filled",
        )
    cadd(
        "C4 half sessions",
        f"sessions skipped by the {MAIN_ENTRY} window filter",
        int(w_main["n_skipped"]),
        int(w_main["skipped_1998_on"]),
        "of them dated 1998 or later",
        f"{w_main['skipped_pre_1998']} are before {PANEL_FIRST_RETURN}, where "
        "sumret is empty",
    )

    # C5: the product.
    d_main = pd.DatetimeIndex(w_main["dates"])
    for label, lo, hi in (
        ("1998-2019", "1998-01-01", "2019-12-31"),
        ("2020-2024", "2020-01-01", "2024-12-31"),
    ):
        m = (d_main >= pd.Timestamp(lo)) & (d_main <= pd.Timestamp(hi))
        cadd(
            "C5 the product",
            f"replayed sessions dated {label}",
            int(m.sum()),
            int(m.sum()),
            "sessions",
            "the replay assumes an SPX 0DTE straddle was listed and quoted on "
            "every one of them; the chain the book is priced from begins "
            f"{idx.min().date()}",
        )
    c_blind = pd.DataFrame(c_rows)
    print(
        "\n--- C. what the replay cannot see (numbers only; every row is a "
        "measurement, not a correction)"
    )
    print(c_blind.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # ------------------------------------------------------------ writes ---
    rep_tab.to_csv(OUT / "a_representative_days.csv", index=False)
    drop = ["book", "option_dollars", "hedge_dollars", "settle_intrinsic_pts"]
    dist.drop(columns=drop).to_csv(
        OUT / "a_distribution.csv", index=False, float_format="%.6g"
    )
    summ.to_csv(OUT / "a_summary.csv", index=False)
    worst.to_csv(OUT / "a_worst.csv", index=False)
    eras_tab.to_csv(OUT / "a_by_era.csv", index=False)
    cmp_tab.to_csv(OUT / "a_vs_realised_2020_2024.csv", index=False)
    b_siz.to_csv(OUT / "b_sizing.csv", index=False)
    c_blind.to_csv(OUT / "c_blindspots.csv", index=False)
    print(
        f"\nwrote {OUT}: a_representative_days.csv ({len(rep_tab)} rows), "
        f"a_distribution.csv ({len(dist):,} rows x "
        f"{len(dist.columns) - len(drop)} columns; option_dollars and "
        f"hedge_dollars are the pts columns x {SPX_MULT:.0f} and are not "
        f"stored), a_summary.csv ({len(summ)}), "
        f"a_worst.csv ({len(worst)}), a_by_era.csv ({len(eras_tab)}), "
        f"a_vs_realised_2020_2024.csv ({len(cmp_tab)}), b_sizing.csv "
        f"({len(b_siz)}), c_blindspots.csv ({len(c_blind)}); total "
        f"{time.time() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
