"""56 - the close-calendar family, tail capture, and a causal calendar adjustment.

Study 54 found, BY INSPECTING the deck sample, that the last half hour of the
last trading day of the month is underpriced by the 15:30 straddle and that the
forecasts sell into it.  One inspected day type is an anecdote.  This study
fixes, before looking, the family of scheduled "large close" day types, the
rule by which a member counts as real, and the way a forecast is judged for a
trade whose P&L sits in a handful of days.  Nothing here is chosen on the
holdout.

PART A - the family, forecast-free.  Every chain session 2020-01-03 ..
2025-12-31 (proposal 43's ``build_chain``): long the 15:30 nearest-OTM straddle
to cash settlement, at the midpoint and at the quoted ask, in premium units and
in index points per contract.  The DECK PERIOD (to 2024-04-30) is the selection
sample; the HOLDOUT (2024-05-01 .. 2025-12-31, 413 sessions no forecast or
earlier study of this trade has touched, except study 54's month-end row) is
printed beside it and plays no part in any choice.  Day types, each knowable in
advance and defined by rule on the session calendar:

   1  month-end            last session of the calendar month (study 54's;
                           FOUND BY INSPECTION, so its deck-period leg is not
                           a test and it is carried as the reference row)
   2  quarter-end          month-end of March, June, September, December
   3  third Friday         standard monthly option expiration (the preceding
                           session when that Friday is a market holiday)
   4  quarterly expiration third Friday of Mar/Jun/Sep/Dec, which is also the
                           S&P quarterly index rebalance
   5  Russell recon.       the annual reconstitution, effective at the close of
                           the fourth Friday of June; the rule is gated to the
                           six published dates 2020-06-26, 2021-06-25,
                           2022-06-24, 2023-06-23, 2024-06-28, 2025-06-27 (its
                           use before 2020 extrapolates the rule, not a checked
                           schedule)
   6  MSCI review close    last session of Feb/May/Aug/Nov, when the MSCI index
                           reviews take effect: a SUBSET of month-ends
   7  month-end ex MSCI    month-ends that are not row 6
   8  first session        first session of the calendar month
   9  pre-holiday          the session before a weekday market closure (the
                           next weekday is not a session)
  10  CPI day              data/releases.parquet ``cpi release``
  11  employment day       data/releases.parquet ``employment release``
  Rows 10-11 use the repo's own feed and are NOT extended by hand: the script
  prints the feed's coverage, and a day type with no holdout observations
  cannot survive the rule below.

  The underlying alone, no options, 1998-2019 against 2020-2024 (the 30-minute
  panel ends 2024-04-30): the 15:30-16:00 bar's realized variance and squared
  net move, relative to the bar before it and relative to the same clock's own
  median over the previous 63 sessions (the session's standing warm-up length).

  The rule below concerns the LONG side, which is what study 54 proposed.  The
  table also prints the short straddle at the quoted bid for every day type;
  those two columns are descriptive and belong to no decision.

  DECISION RULE, fixed here: a day type SURVIVES only if (i) its at-the-ask mean
  is positive with t > 2 in the deck period, AND (ii) its at-the-ask mean has
  the same sign in the holdout, AND (iii) in 1998-2019, before any option in
  this project existed, the log of the last bar's variance over its own
  trailing-63-session median is higher on those days with Welch t > 2.
  MULTIPLE TESTING: 11 rows, of which rows 6-7 are splits of row 1 and row 4 is
  a subset of row 3, so about 8 distinct questions.  If the underlying effect is
  real for a type but the option market prices it, the chance of passing (i) and
  (ii) is about 0.023 x 0.5 = 0.011 a row, or about 0.1 false survivors over the
  family; under a global null it is smaller still.  Row 1's leg (i) was seen
  before this rule was written, so row 1 is reported, not tested, on that leg.

PART B - tail capture.  On the deck's 866 days, for the eight forecasts: of the
k best long-straddle days (k = 10, 20, 50), on how many was the forecast long,
against its buy share (hypergeometric upper-tail p); the mean return on its buy
days and on its sell days; mid and crossed Sharpe; the mid Sharpe with those k
days removed; and the forecast's QLIKE for the 15:30-16:00 bar beside them.

PART C - a causal close-calendar adjustment of the production forecasts.  For
each forecast the long panel is read through ``load_yhat_panel_mz`` (the deck's
own back-transform), restricted to the row stamped 16:00 on session dates.  By
EXPANDING-window OLS on strictly prior sessions (at least 252) the log forecast
error log(rv_raw / rv_hat) is regressed on an intercept and six dummies fixed
before looking: month-end, quarter-end, third Friday, quarterly expiration,
FOMC statement day (the repo's flag, which ends 2023-11-01, extended with the
Federal Reserve's published schedule), first session of the month.  The
adjusted forecast is rv_hat x exp(gamma'D) x (smear on flagged days / smear on
unflagged days), the smearing terms being the means of exp(residual) over the
same prior sessions.  An unflagged day is left exactly as it was: this is not a
recalibration.  No free parameter.  Reported: the final gammas with Newey-West
t, QLIKE at that bar before and after (2003-2024 and the deck period, day-block
bootstrap interval), then on the 866 deck days the sign(s) trade with the
adjusted forecast against the unadjusted one: signs changed, buy share, tail
capture, mid and crossed Sharpe with a paired bootstrap interval, by year, and
how many of the eight forecasts improve.  The gammas are causal throughout, but
the TRADE evaluation cannot reach the holdout: every forecast ends 2024-04-30.

GATES  ridge sign(s) on the deck days 1.338322 mid / 0.869588 crossed; the
       chain-built 15:30 straddle return equals the deck's R on all 866 days;
       study 54's month-end row reproduces (52 / 18 / 70 days, the same means);
       the Russell rule returns the six published dates; the panel's 16:00-row
       rv_hat equals the deck's rv_hat for the ridge (found: identical).

Run:  python writeup/intraday_proposals/56_close_calendar_family.py
"""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = DECK / "proposals" / "56"
REF54 = DECK / "proposals" / "54" / "c_long_straddle_month_end.csv"
CORE = ROOT / "data" / "core_stats.parquet"
RELEASES = ROOT / "data" / "releases.parquet"

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
PANEL_START = "1998-01-05"  # returns are empty before this date
DECK_END, OOS_START, PRE_OPTIONS_END = "2024-04-30", "2024-05-01", "2019-12-31"
TRAIL = 63  # the session's standing warm-up length
MIN_HISTORY = 252  # sessions before the calendar regression is used
QLIKE_START = "2003-01-01"
TOP_K = (10, 20, 50)
T_BAR = 2.0
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
GATE_SHARPE = (1.338322, 0.869588)
GATE_TOL = 1e-6
GATE_R_TOL = 1e-6  # the chain stores quotes as float32
GATE_REF_TOL = 1e-9
CLOSE_MIN = 16 * 60

RUSSELL_PUBLISHED = (
    "2020-06-26",
    "2021-06-25",
    "2022-06-24",
    "2023-06-23",
    "2024-06-28",
    "2025-06-27",
)
#: Federal Reserve statement days after the repo's feed ends (2023-11-01)
FOMC_EXTENSION = (
    "2023-12-13",
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
)
FAMILY = (
    "month-end",
    "quarter-end",
    "third Friday",
    "quarterly expiration",
    "Russell reconstitution",
    "MSCI review close",
    "month-end ex MSCI",
    "first session",
    "pre-holiday",
    "CPI day",
    "employment day",
)
INSPECTED = ("month-end", "MSCI review close", "month-end ex MSCI")
ADJ_DUMMIES = (
    "month-end",
    "quarter-end",
    "third Friday",
    "quarterly expiration",
    "FOMC statement day",
    "first session",
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ helpers --
def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * ANN) if sd > 0 else float("nan")


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 3:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(
        (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    )


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    if n not in _IDX:
        _IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
        )
    return _IDX[n]


def boot_mean_ci(d: np.ndarray) -> tuple[float, float]:
    m = np.asarray(d, float)[boot_idx(len(d))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def boot_sharpe_diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ii = boot_idx(len(a))
    xa, xb = a[ii], b[ii]
    d = ANN * (
        xa.mean(axis=1) / xa.std(axis=1, ddof=1)
        - xb.mean(axis=1) / xb.std(axis=1, ddof=1)
    )
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def hac_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    u = y - x @ beta
    xu = x * u[:, None]
    lag = asl.newey_west_lag(len(y))
    s = xu.T @ xu
    for j in range(1, lag + 1):
        g = xu[j:].T @ xu[:-j]
        s += (1.0 - j / (lag + 1.0)) * (g + g.T)
    xtx_inv = np.linalg.pinv(x.T @ x)
    cov = xtx_inv @ s @ xtx_inv
    return beta, beta / np.sqrt(np.diag(cov))


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


# ------------------------------------------------------------- the calendar --
def session_calendar(chain_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sessions: days whose 30-minute panel reaches the 16:00 bar, 1998 on, joined
    with the chain's own session list past the panel's end (2024-04-30)."""
    c = pd.read_parquet(CORE, columns=["endbartime"])
    t = pd.to_datetime(c["endbartime"])
    t = t[t >= PANEL_START]
    rth = pd.DatetimeIndex(
        sorted(t[t.dt.strftime("%H:%M") == "16:00"].dt.normalize().unique())
    )
    return rth.union(chain_days)


def day_types(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Every rule of the family (and the adjustment's dummies) on the calendar."""
    s = pd.Series(cal, index=cal)
    ym = [cal.year, cal.month]
    me = (s.groupby(ym).transform("max") == s).to_numpy()
    fs = (s.groupby(ym).transform("min") == s).to_numpy()
    fs &= ~((cal.year == cal[0].year) & (cal.month == cal[0].month))  # partial month
    months = pd.period_range(cal[0], cal[-1], freq="M").to_timestamp()

    def nth_friday(n: int) -> pd.DatetimeIndex:
        off = (4 - months.dayofweek) % 7 + 7 * (n - 1)
        return pd.DatetimeIndex(months + pd.to_timedelta(off, unit="D"))

    def session_on_or_before(days: pd.DatetimeIndex) -> np.ndarray:
        pos = cal.searchsorted(days, side="right") - 1
        ok = pos >= 0
        got = cal[pos[ok]]
        same = (got.year == days[ok].year) & (got.month == days[ok].month)
        out = np.zeros(len(cal), bool)
        out[pos[ok][same]] = True
        return out

    tf = session_on_or_before(nth_friday(3))
    f4 = session_on_or_before(nth_friday(4)) & (cal.month == 6)
    got = [str(d.date()) for d in cal[f4] if d.year >= 2020]
    assert tuple(got) == RUSSELL_PUBLISHED, got

    nxt = cal + pd.offsets.BDay(1)
    in_cal = np.isin(nxt, cal)
    beyond = nxt > cal[-1]
    pre = np.where(beyond, (nxt.month == 1) & (nxt.day == 1), ~in_cal)

    rel = pd.read_parquet(RELEASES)
    rel_day = pd.to_datetime(rel["endbartime"]).dt.normalize()
    fomc = pd.DatetimeIndex(rel_day[rel["fomc release"] != 0].unique()).union(
        pd.DatetimeIndex(FOMC_EXTENSION)
    )
    q = np.isin(cal.month, (3, 6, 9, 12))
    msci = me & np.isin(cal.month, (2, 5, 8, 11))
    return pd.DataFrame(
        {
            "month-end": me,
            "quarter-end": me & q,
            "third Friday": tf,
            "quarterly expiration": tf & q,
            "Russell reconstitution": f4,
            "MSCI review close": msci,
            "month-end ex MSCI": me & ~msci,
            "first session": fs,
            "pre-holiday": pre,
            "CPI day": np.isin(cal, rel_day[rel["cpi release"] != 0].unique()),
            "employment day": np.isin(
                cal, rel_day[rel["employment release"] != 0].unique()
            ),
            "FOMC statement day": np.isin(cal, fomc),
        },
        index=cal,
    )


# ------------------------------------------------------------------ part A ---
def chain_straddle() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    k = p43.CLOCKS.index("15:30")
    mid, bid, sc = ch["entry"][:, k], ch["bid"][:, k], ch["S_close"]
    ask = 2.0 * mid - bid
    settle = np.maximum(sc - ch["K_c"][:, k], 0.0) + np.maximum(
        ch["K_p"][:, k] - sc, 0.0
    )
    ok = np.isfinite(mid) & (mid > 0) & (bid > 0) & np.isfinite(settle)
    df = pd.DataFrame(
        {
            "R_mid": settle / mid - 1.0,
            "R_ask": settle / ask - 1.0,
            "pts_ask": settle - ask,
            "S_bid": -(settle / bid - 1.0),
        },
        index=ch["dates"],
    )[ok]
    deck = pd.read_parquet(DECK / "daily_blk2.parquet")["R"]
    j = df.join(deck.rename("R_deck"), how="inner")
    dev = float((j["R_mid"] - j["R_deck"]).abs().max())
    assert len(j) == len(deck) and dev < GATE_R_TOL, (len(j), dev)
    print(
        f"GATE  chain-built 15:30 straddle return vs the deck's R on {len(j)} days: {dev:.1e}"
    )
    return df, pd.DatetimeIndex(stamp.index)


def part_a_options(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "deck period": np.asarray(df.index <= DECK_END),
        "holdout": np.asarray(df.index >= OOS_START),
        "all sessions": np.ones(len(df), bool),
    }
    f = flags.reindex(df.index)
    assert not f.isna().any().any()
    rows = []
    for typ in FAMILY:
        on = f[typ].to_numpy(bool)
        for pname, m in periods.items():
            a, b = df[m & on], df[m & ~on]
            rows.append(
                {
                    "day_type": typ,
                    "period": pname,
                    "n": len(a),
                    "mid_mean": float(a["R_mid"].mean()) if len(a) else np.nan,
                    "mid_t": tstat(a["R_mid"].to_numpy()),
                    "ask_mean": float(a["R_ask"].mean()) if len(a) else np.nan,
                    "ask_t": tstat(a["R_ask"].to_numpy()),
                    "hit": float((a["R_mid"] > 0).mean()) if len(a) else np.nan,
                    "pts_per_contract_at_ask": float(a["pts_ask"].mean())
                    if len(a)
                    else np.nan,
                    "short_at_bid_mean": float(a["S_bid"].mean()) if len(a) else np.nan,
                    "short_at_bid_t": tstat(a["S_bid"].to_numpy()),
                    "other_days_mid_mean": float(b["R_mid"].mean()),
                }
            )
    tab = pd.DataFrame(rows)
    ref = pd.read_csv(REF54)
    mine = tab[tab["day_type"] == "month-end"].reset_index(drop=True)
    assert list(mine["n"]) == list(ref["month_ends"]), (
        list(mine["n"]),
        list(ref["month_ends"]),
    )
    dev = float(
        np.max(np.abs(mine["ask_mean"].to_numpy() - ref["long_ask_mean"].to_numpy()))
    )
    assert dev < GATE_REF_TOL, dev
    print(
        f"GATE  study 54's month-end row reproduces: days {list(mine['n'])}, at-the-ask means to {dev:.1e}"
    )
    return tab


def part_a_underlying(flags: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret", "sumret2"])
    c["t"] = pd.to_datetime(c["endbartime"])  # naive ET, bar-END labelled
    c = c[c["t"] >= PANEL_START]
    c["date"] = c["t"].dt.normalize()
    c["hhmm"] = c["t"].dt.strftime("%H:%M")
    p = c[c["hhmm"].isin(["15:30", "16:00"])].pivot_table(
        index="date", columns="hhmm", values=["sumret", "sumret2"], aggfunc="first"
    )
    p = p[(p[("sumret2", "15:30")] > 0) & (p[("sumret2", "16:00")] > 0)].dropna()
    rv16, rv1530 = p[("sumret2", "16:00")], p[("sumret2", "15:30")]
    net2 = p[("sumret", "16:00")] ** 2
    x = pd.DataFrame(
        {
            "log_rv_vs_prior_bar": np.log(rv16 / rv1530),
            "log_rv_vs_own_median": np.log(
                rv16 / rv16.rolling(TRAIL).median().shift(1)
            ),
            "net2_vs_prior_bar_rv": net2 / rv1530,
            "net2_vs_own_median": net2 / net2.rolling(TRAIL).median().shift(1),
        }
    ).dropna()
    f = flags.reindex(x.index).fillna(False)
    days = pd.DatetimeIndex(x.index)
    rows = []
    for typ in FAMILY:
        on = f[typ].to_numpy(bool)
        for sname, m in (
            ("1998-2019", np.asarray(days <= PRE_OPTIONS_END)),
            ("2020-2024", np.asarray(days > PRE_OPTIONS_END)),
        ):
            a, b = x[m & on], x[m & ~on]
            rec: dict[str, Any] = {"day_type": typ, "sample": sname, "n": len(a)}
            for col in ("log_rv_vs_prior_bar", "log_rv_vs_own_median"):
                rec[f"{col}_flagged"] = float(a[col].mean()) if len(a) else np.nan
                rec[f"{col}_other"] = float(b[col].mean())
                rec[f"{col}_welch_t"] = welch(a[col].to_numpy(), b[col].to_numpy())
            for col in ("net2_vs_prior_bar_rv", "net2_vs_own_median"):
                rec[f"{col}_median_flagged"] = (
                    float(a[col].median()) if len(a) else np.nan
                )
                rec[f"{col}_median_other"] = float(b[col].median())
                rec[f"{col}_ranksum_p"] = (
                    float(
                        stats.mannwhitneyu(a[col], b[col], alternative="greater").pvalue
                    )
                    if len(a) >= 3
                    else np.nan
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def survival(opt: pd.DataFrame, und: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for typ in FAMILY:
        o = opt[opt["day_type"] == typ].set_index("period")
        u = und[(und["day_type"] == typ) & (und["sample"] == "1998-2019")].iloc[0]
        leg1 = bool(
            o.loc["deck period", "ask_mean"] > 0
            and o.loc["deck period", "ask_t"] > T_BAR
        )
        n_hold = int(o.loc["holdout", "n"])
        leg2 = bool(n_hold > 0 and o.loc["holdout", "ask_mean"] > 0)
        leg3 = bool(u["log_rv_vs_own_median_welch_t"] > T_BAR)
        rows.append(
            {
                "day_type": typ,
                "deck_n": int(o.loc["deck period", "n"]),
                "deck_ask_mean": float(o.loc["deck period", "ask_mean"]),
                "deck_ask_t": float(o.loc["deck period", "ask_t"]),
                "holdout_n": n_hold,
                "holdout_ask_mean": float(o.loc["holdout", "ask_mean"]),
                "holdout_ask_t": float(o.loc["holdout", "ask_t"]),
                "underlying_1998_2019_n": int(u["n"]),
                "underlying_1998_2019_welch_t": float(
                    u["log_rv_vs_own_median_welch_t"]
                ),
                "leg_i_deck": leg1,
                "leg_ii_holdout": leg2,
                "leg_iii_underlying": leg3,
                "SURVIVES": leg1 and leg2 and leg3,
                "note": "leg (i) seen before the rule was written"
                if typ in INSPECTED
                else "",
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ part B ---
def deck_books() -> dict[str, pd.DataFrame]:
    return {
        t: pd.read_parquet(DECK / f"daily_{t}.parquet").sort_index()
        for t in asl.MODEL_ORDER
    }


def trade_metrics(
    f: np.ndarray, d: pd.DataFrame, rv_close: np.ndarray, top: dict[int, np.ndarray]
) -> dict[str, Any]:
    r = d["R"].to_numpy(float)
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    ex = d["exit"].to_numpy(float)
    q = np.where(f > d["iv_var"].to_numpy(float), 1.0, -1.0)
    mid = q * r
    crossed = np.where(q > 0, ex / ask - 1.0, -(ex / bid - 1.0))
    n, n_buy = len(r), int((q > 0).sum())
    rec: dict[str, Any] = {
        "buy_share": n_buy / n,
        "mean_R_on_buy_days": float(r[q > 0].mean()),
        "mean_short_R_on_sell_days": float((-r[q < 0]).mean()),
        "Sharpe_mid": sharpe(mid),
        "Sharpe_crossed": sharpe(crossed),
        "QLIKE_close_bar": float(np.nanmean(qlike(rv_close, f))),
    }
    for k, idx in top.items():
        x = int((q[idx] > 0).sum())
        rec[f"long_on_top{k}"] = x
        rec[f"top{k}_hypergeom_p"] = float(stats.hypergeom.sf(x - 1, n, n_buy, k))
        keep = np.ones(n, bool)
        keep[idx] = False
        rec[f"Sharpe_mid_without_top{k}"] = sharpe(mid[keep])
    rec["_mid"], rec["_crossed"], rec["_q"] = mid, crossed, q
    return rec


def part_b(
    books: dict[str, pd.DataFrame], rv_close: np.ndarray
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    d0 = books["blk2"]
    r = d0["R"].to_numpy(float)
    top = {k: np.argsort(r)[::-1][:k] for k in TOP_K}
    rows = []
    for tag, d in books.items():
        assert d.index.equals(d0.index), tag
        rec = trade_metrics(d["rv_hat"].to_numpy(float), d, rv_close, top)
        if tag == "blk2":
            got = (rec["Sharpe_mid"], rec["Sharpe_crossed"])
            assert abs(got[0] - GATE_SHARPE[0]) < GATE_TOL, got
            assert abs(got[1] - GATE_SHARPE[1]) < GATE_TOL, got
            print(
                f"GATE  ridge sign(s) on the deck days: {got[0]:.6f} mid / {got[1]:.6f} crossed"
            )
        rows.append(
            {"forecast": tag} | {k: v for k, v in rec.items() if not k.startswith("_")}
        )
    tab = pd.DataFrame(rows)
    for col, asc in (
        ("QLIKE_close_bar", True),
        ("long_on_top20", False),
        ("Sharpe_mid", False),
    ):
        tab[f"rank_{col}"] = tab[col].rank(ascending=asc, method="min").astype(int)
    return tab, top


# ------------------------------------------------------------------ part C ---
def adjust_one(job: dict[str, Any]) -> dict[str, Any]:
    """One forecast's causal calendar adjustment, in its own process."""
    tag = job["tag"]
    flags: pd.DataFrame = job["flags"]
    panel = asl.load_yhat_panel_mz(Path(job["path"]))
    rows = panel[
        (panel["mins"] == CLOSE_MIN)
        & panel["in_fit"]
        & np.isfinite(panel["rv_hat"])
        & (panel["rv_raw"] > 0)
    ]
    days = pd.DatetimeIndex(pd.to_datetime(rows["date"]))
    rv, f = rows["rv_raw"].to_numpy(float), rows["rv_hat"].to_numpy(float)
    y = np.log(rv / f)
    d = flags.reindex(days)[list(ADJ_DUMMIES)].to_numpy(float)
    assert np.isfinite(d).all(), tag
    x = np.column_stack([np.ones(len(y)), d])
    any_flag = d.sum(axis=1) > 0
    factor = np.ones(len(y))
    for t in np.flatnonzero(any_flag):
        if t < MIN_HISTORY:
            continue
        xs, ys, fl = x[:t], y[:t], any_flag[:t]
        beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
        e = np.exp(ys - xs @ beta)
        factor[t] = float(np.exp(d[t] @ beta[1:]) * e[fl].mean() / e[~fl].mean())
    beta, tval = hac_ols(x, y)
    return {
        "tag": tag,
        "days": days,
        "rv": rv,
        "f": f,
        "factor": factor,
        "any_flag": any_flag,
        "gamma": beta,
        "gamma_t": tval,
        "n_by_dummy": d.sum(axis=0),
    }


def part_c(
    books: dict[str, pd.DataFrame],
    flags: pd.DataFrame,
    rv_close: np.ndarray,
    top: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = asl.yhat_paths(ROOT)
    jobs = [{"tag": t, "path": str(paths[t]), "flags": flags} for t in asl.MODEL_ORDER]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        res = {r["tag"]: r for r in pool.map(adjust_one, jobs)}

    gam, ql, tr, yr = [], [], [], []
    deck_days = pd.DatetimeIndex(books["blk2"].index)
    for tag in asl.MODEL_ORDER:
        r = res[tag]
        for j, name in enumerate(("intercept", *ADJ_DUMMIES)):
            gam.append(
                {
                    "forecast": tag,
                    "term": name,
                    "n_days": len(r["days"]) if j == 0 else int(r["n_by_dummy"][j - 1]),
                    "gamma_full_sample": float(r["gamma"][j]),
                    "newey_west_t": float(r["gamma_t"][j]),
                    "implied_variance_multiple": float(np.exp(r["gamma"][j])),
                }
            )
        l0, l1 = qlike(r["rv"], r["f"]), qlike(r["rv"], r["f"] * r["factor"])
        for sname, m in (
            ("2003-2024", np.asarray(r["days"] >= QLIKE_START)),
            ("deck period", np.asarray(r["days"] >= str(deck_days[0].date()))),
        ):
            dd = (l1 - l0)[m]
            lo, hi = boot_mean_ci(dd)
            fl = m & r["any_flag"]
            ql.append(
                {
                    "forecast": tag,
                    "sample": sname,
                    "n_days": int(m.sum()),
                    "n_flagged": int(fl.sum()),
                    "QLIKE_before": float(l0[m].mean()),
                    "QLIKE_after": float(l1[m].mean()),
                    "diff": float(dd.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "QLIKE_before_flagged_days": float(l0[fl].mean()),
                    "QLIKE_after_flagged_days": float(l1[fl].mean()),
                    "improves": bool(hi < 0.0),
                }
            )
        d = books[tag]
        fac = pd.Series(r["factor"], index=r["days"]).reindex(deck_days)
        if tag == "blk2":
            pan = pd.Series(r["f"], index=r["days"]).reindex(deck_days)
            dev = float((pan / d["rv_hat"].to_numpy(float) - 1.0).abs().max())
            assert pan.notna().all() and dev < GATE_TOL, dev
            print(
                f"GATE  panel 16:00-row rv_hat vs the deck's rv_hat, ridge: max relative difference {dev:.1e}"
            )
        assert fac.notna().all(), tag
        f0 = d["rv_hat"].to_numpy(float)
        base = trade_metrics(f0, d, rv_close, top)
        adj = trade_metrics(f0 * fac.to_numpy(float), d, rv_close, top)
        rec: dict[str, Any] = {
            "forecast": tag,
            "days_adjusted": int((fac != 1.0).sum()),
            "signs_changed": int((base["_q"] != adj["_q"]).sum()),
            "to_long": int(((base["_q"] < 0) & (adj["_q"] > 0)).sum()),
            "to_short": int(((base["_q"] > 0) & (adj["_q"] < 0)).sum()),
        }
        for k in (
            "buy_share",
            "Sharpe_mid",
            "Sharpe_crossed",
            "QLIKE_close_bar",
            *[f"long_on_top{n}" for n in TOP_K],
        ):
            rec[f"{k}_before"], rec[f"{k}_after"] = base[k], adj[k]
        for fill in ("mid", "crossed"):
            lo, hi = boot_sharpe_diff_ci(adj[f"_{fill}"], base[f"_{fill}"])
            rec[f"dSharpe_{fill}"] = adj[f"Sharpe_{fill}"] - base[f"Sharpe_{fill}"]
            rec[f"dSharpe_{fill}_ci_lo"], rec[f"dSharpe_{fill}_ci_hi"] = lo, hi
        tr.append(rec)
        for year in sorted(set(deck_days.year)):
            m = np.asarray(deck_days.year == year)
            yr.append(
                {
                    "forecast": tag,
                    "year": year,
                    "n": int(m.sum()),
                    "crossed_before": sharpe(base["_crossed"][m]),
                    "crossed_after": sharpe(adj["_crossed"][m]),
                    "mid_before": sharpe(base["_mid"][m]),
                    "mid_after": sharpe(adj["_mid"][m]),
                }
            )
    return pd.DataFrame(gam), pd.DataFrame(ql), pd.DataFrame(tr), pd.DataFrame(yr)


# -------------------------------------------------------------------- main ---
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 200)

    straddle, chain_days = chain_straddle()
    cal = session_calendar(chain_days)
    flags = day_types(cal)
    flags.astype(int).to_csv(OUT / "a_calendar_flags.csv")
    rel = pd.read_parquet(RELEASES)
    rel_day = pd.to_datetime(rel["endbartime"]).dt.normalize()
    for col in ("cpi release", "employment release", "fomc release"):
        dd = rel_day[rel[col] != 0]
        print(
            f"feed coverage  {col}: {dd.nunique()} days, {dd.min().date()} .. {dd.max().date()}"
        )
    whole = flags[(flags.index.year >= 2000) & (flags.index.year <= 2023)]
    per_year = whole.groupby(whole.index.year).sum()
    print(
        "flags per year, 2000-2023 mean: "
        + ", ".join(f"{c} {per_year[c].mean():.1f}" for c in flags.columns)
    )

    opt = part_a_options(straddle, flags)
    und = part_a_underlying(flags)
    surv = survival(opt, und)
    write(opt, "a_family_options.csv")
    write(und, "a_family_underlying.csv")
    write(surv, "a_survival.csv")
    print(
        "\nA1  long the 15:30 straddle by day type (premium units; pts = index points per contract at the ask)"
    )
    print(opt.round(3).to_string(index=False))
    ucols = [
        "day_type",
        "sample",
        "n",
        "log_rv_vs_own_median_flagged",
        "log_rv_vs_own_median_other",
        "log_rv_vs_own_median_welch_t",
        "log_rv_vs_prior_bar_welch_t",
        "net2_vs_own_median_median_flagged",
        "net2_vs_own_median_median_other",
        "net2_vs_own_median_ranksum_p",
    ]
    print("\nA2  the underlying's last half hour by day type")
    print(und[ucols].round(3).to_string(index=False))
    print("\nA3  the pre-set survival rule")
    print(surv.round(3).to_string(index=False))

    books = deck_books()
    deck_days = pd.DatetimeIndex(books["blk2"].index)
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    t = pd.to_datetime(c["endbartime"])
    last = c[t.dt.strftime("%H:%M") == "16:00"].set_index(
        t[t.dt.strftime("%H:%M") == "16:00"].dt.normalize()
    )
    rv_close = last["sumret2"].reindex(deck_days).to_numpy(float)
    assert np.isfinite(rv_close).all()
    b_tab, top = part_b(books, rv_close)
    write(b_tab, "b_tail_capture.csv")
    print("\nB  tail capture on the deck's 866 days")
    print(b_tab.round(3).to_string(index=False))

    gam, ql, tr, yr = part_c(books, flags, rv_close, top)
    write(gam, "c_gammas.csv")
    write(ql, "c_qlike.csv")
    write(tr, "c_trade.csv")
    write(yr, "c_trade_by_year.csv")
    print(
        "\nC1  full-sample calendar gammas of log(rv_raw / rv_hat) at the 16:00 bar, ridge"
    )
    print(gam[gam["forecast"] == "blk2"].round(3).to_string(index=False))
    print("\nC2  QLIKE at the 16:00 bar, before and after the causal adjustment")
    print(ql.round(5).to_string(index=False))
    print("\nC3  sign(s) on the deck days with the adjusted forecast")
    print(tr.round(3).T.to_string())
    print("\nC4  crossed Sharpe by year, ridge")
    print(yr[yr["forecast"] == "blk2"].round(2).to_string(index=False))
    up_mid = int((tr["dSharpe_mid"] > 0).sum())
    up_cr = int((tr["dSharpe_crossed"] > 0).sum())
    sig = int((tr["dSharpe_crossed_ci_lo"] > 0).sum())
    print(
        f"\nforecasts whose Sharpe rises under the adjustment: {up_mid} of {len(tr)} at the mid, "
        f"{up_cr} crossed; with a paired interval above zero: {sig}"
    )


if __name__ == "__main__":
    main()
