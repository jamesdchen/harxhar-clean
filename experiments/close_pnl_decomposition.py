"""B1: decomposition of the daily return sequence of the last-30-min straddle trade.

The trade (rv_iv notebook §4-§9): at 15:30 ET buy or sell one straddle (nearest
out-of-the-money call + nearest out-of-the-money put, same-day expiry, one
position), hold it to the cash settlement at the official close. The sign(s) rule
buys when the forecast of the 15:30-16:00 realized variance exceeds the implied
variance quoted at 15:30 and sells otherwise; always short sells every day and
uses no forecast. One unit of premium per day, returns summed, never compounded.

Series (the same 866 days, the rows of the notebook's per-day tables
results/atm_straddle_0dte_1530/daily_<tag>.parquet):
  sign(s) with the per-bar ridge, live-feasible set   (tag sub_live_ridge)
  sign(s) with the per-bar ridge, all features        (tag sub_ridge)
  sign(s) with the block-diagonal ridge, the paper's headline forecast (tag blk2)
  sign(s) with the HAR + calendar OLS, the paper's baseline (tag a0)
  always short
Fills: mid (R' = q (X/P - 1)) and crossed (buy at the ask, sell at the bid,
asl.crossed_premium_return), the notebook's §13 definitions.

SCORER. The primary set is the notebook scorer (the rv_iv notebook's
recalibration on all 13 session bars), because the per-day tables come from
it and they carry all four forecasts. The research scorer (16:00-bar
recalibration, score_linear_subsection_causal.causal_forecasts +
score_linear_subsection.trade_1530) is run as a separate set for the two
per-bar ridges (the paper's forecasts have no research-scorer rows yet) and
written to results/close_pnl_decomp/research_scorer/ -- never mixed with the
primary set in one table.

Blocks (one CSV each, plus supporting CSVs):
  1 tails        tail_concentration.csv, tail_days.csv
  2 calendar     calendar_cells.csv
  3 regimes      regime_cells.csv
  4 hit/payoff   hit_payoff.csv
  5 path         path_summary.csv, path_drawdowns.csv, path_rolling_sharpe.csv,
                 path_autocorr.csv
  6 difference   diff_attribution.csv  (sign(s) minus always short, paired)
Figures: fig_cum_pnl_tails.png, fig_regime_bars.png, fig_drawdown.png.
Summary: SUMMARY.md (every number read from the tables above).

Intervals: circular moving-block bootstrap (asl.circular_block_bootstrap_idx),
block BOOT_BLOCK sessions, BOOT_B draws, one seed, the SAME resampled days for
every series (so a difference is paired). A cell statistic (month-ends, a VIX
tercile, ...) is recomputed on the resampled days that fall in the cell.

Usage:  python experiments/close_pnl_decomposition.py [--workers 4] [--boot 2000]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

DAILY = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = ROOT / "results" / "close_pnl_decomp"
MARKET_CACHE = OUT / "market_daily_closes.csv"
RESEARCH_ARMS = ROOT / "results" / "linear_subsection" / "arms_hoffman2"

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))  # per-trade-day annualization, as the deck
MAX_WORKERS = 4  # the overnight brief's local cap on worker processes per agent

# bootstrap: the research scorer's day-block settings (score_linear_subsection.py)
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0  # 21 sessions = one trading month
CI_LEVEL = 95.0

TOP_K = (5, 10, 20, 50)  # tail sizes the task names
SHARE_LEVELS = (0.5, 0.8, 1.0)  # "days needed for 50 / 80 / 100 % of the P&L"
ROLL_WINDOWS = (63, 126, 252)  # a quarter, half a year, a year of sessions
LB_LAGS = (5, 10, 21)  # a week, two weeks, a month of trade days
ACF_LAGS = 10
VR_HORIZONS = (2, 5, 10, 21)
N_DD_EPISODES = 5  # deepest drawdown episodes listed per series
N_TAIL_LIST = 20  # days listed per tail in tail_days.csv

# regimes: known before the 15:30 entry -- the previous session's closes only
MARKET_HISTORY_START = "1990-01-02"  # first ^VIX close yfinance serves
MARKET_FETCH_END = "2024-06-29"  # a month past the frame, so April 2024's T+1 exists
RV_WINDOW = 21  # trailing close-to-close realized vol, one trading month
TERCILE_NAMES = ("low", "mid", "high")

# FOMC statement days. data/releases.parquet carries the flags through
# 2023-11-01 only, and dates the September 2013 statement 2013-09-19 (the
# statement was 2013-09-18) -- fixed below (outside this 2020-2024 frame, fixed
# anyway). The feed is extended with the scheduled statement days (the second
# day of each two-day meeting) from the Federal Reserve Board's meeting
# calendar, https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# (2023-12 .. 2026-09). The two unscheduled 2020 actions follow the deck's own
# convention (atm_straddle_lib.FOMC_STATEMENT_DAYS): 2020-03-03 (a Tuesday
# statement) and 2020-03-16 (first session after the Sunday 2020-03-15 one).
RELEASES_FOMC_DATE_FIX = {"2013-09-19": "2013-09-18"}
FOMC_UNSCHEDULED_2020 = ("2020-03-03", "2020-03-16")
FOMC_STATEMENT_DAYS_2023_12_ON = (
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
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
)

# the series of the primary (notebook-scorer) set
SIGN_TAGS = {
    "sub_live_ridge": "sign(s): per-bar ridge (live-feasible)",
    "sub_ridge": "sign(s): per-bar ridge (all features)",
    "blk2": "sign(s): block-diagonal ridge (paper headline)",
    "a0": "sign(s): HAR + calendar OLS (paper baseline)",
}
SHORT = "always short"
HEADLINE = SIGN_TAGS["sub_live_ridge"]
# research-scorer arms: (label, bucket) of the per-bar ridge, 2000-session window
RESEARCH_ARMS_SPEC = {
    SIGN_TAGS["sub_live_ridge"]: "live_feasible",
    SIGN_TAGS["sub_ridge"]: "all_features",
}
FILLS = ("mid", "crossed")
# figure colours: the dataviz reference palette's first three categorical slots
# (validated all-pairs) for the sign(s) series, a neutral for the control
COLOURS = {
    SIGN_TAGS["sub_live_ridge"]: "#2a78d6",
    SIGN_TAGS["sub_ridge"]: "#eb6834",
    SIGN_TAGS["blk2"]: "#1baf7a",
    SHORT: "#7d7c78",
}
FIG_SERIES = (
    SIGN_TAGS["sub_live_ridge"],
    SIGN_TAGS["sub_ridge"],
    SIGN_TAGS["blk2"],
    SHORT,
)


# ---------------------------------------------------------------- inputs
def load_frame() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """The per-day frame (entry, exit, bid/ask sums, R) and the sign(s) positions."""
    books = {
        t: pd.read_parquet(DAILY / f"daily_{t}.parquet").sort_index() for t in SIGN_TAGS
    }
    px = books["blk2"].copy()
    for t, b in books.items():
        # the instrument is the same on every table; only the forecast differs
        assert b.index.equals(px.index), t
        assert np.allclose(b["R"], px["R"]) and np.allclose(
            b["iv_var"], px["iv_var"]
        ), t
    px.index = pd.DatetimeIndex(px.index).normalize()
    px["bid_entry"] = px["bid_c"].astype(float) + px["bid_p"].astype(float)
    px["ask_entry"] = px["ask_c"].astype(float) + px["ask_p"].astype(float)
    pos = {}
    for t, lab in SIGN_TAGS.items():
        q = asl.rule_sizes(books[t], ROOT)["sign(s)"]
        assert (q.to_numpy() == books[t]["pos"].to_numpy()).all(), t
        pos[lab] = pd.Series(q.to_numpy(float), index=px.index)
    pos[SHORT] = pd.Series(-1.0, index=px.index)
    return px, pos


def research_positions(px: pd.DataFrame) -> dict[str, pd.Series]:
    """sign(s) under the research scorer (16:00-bar recalibration), per-bar ridges."""
    import score_linear_subsection as base
    import score_linear_subsection_causal as slc

    out: dict[str, pd.Series] = {}
    for lab, bucket in RESEARCH_ARMS_SPEC.items():
        path = RESEARCH_ARMS / bucket / "ridge" / "tw2000" / "results_bar1600.csv"
        r = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)].copy()
        r["baseline"] = r["true_raw"] / r["true_adj"] ** 2
        r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
        r["hhmm"] = r.index.strftime("%H:%M")
        r["day"] = r.index.normalize()
        f = slc.causal_forecasts(r)["pred_clock"]
        ref = base.trade_1530(
            f
        )  # the scorer's own Sharpe, to check the rebuild against
        f = f[f.index.strftime("%H:%M") == "16:00"]
        f.index = f.index.normalize()
        f = f.reindex(px.index)
        assert f.notna().all(), (
            f"{lab}: research forecast missing on {int(f.isna().sum())} days"
        )
        q = pd.Series(
            np.where(f.to_numpy(float) > px["iv_var"].to_numpy(float), 1.0, -1.0),
            index=px.index,
        )
        mine = q * px["R"]
        sh = float(mine.mean() / mine.std(ddof=1) * ANN)
        assert abs(sh - ref["Sharpe_mid"]) < 1e-9, (lab, sh, ref["Sharpe_mid"])
        ratio = f / px["iv_var"]
        print(
            f"research scorer {lab}: Sharpe mid {sh:.4f} (= trade_1530's), buys {int((q > 0).sum())}, "
            f"forecast / implied variance max {float(ratio.max()):.3f}, min {float(ratio.min()):.3f}"
        )
        out[lab] = q
    out[SHORT] = pd.Series(-1.0, index=px.index)
    return out


def daily_pnl(px: pd.DataFrame, q: pd.Series) -> dict[str, pd.Series]:
    mid = (q * px["R"]).astype(float)
    crossed = asl.crossed_premium_return(
        q, px["exit"], px["bid_entry"], px["ask_entry"]
    )
    assert asl.crossed_untradeable_count(q, px["bid_entry"], px["ask_entry"]) == 0
    return {"mid": mid, "crossed": crossed.astype(float)}


def load_market() -> pd.DataFrame:
    """^GSPC and ^VIX daily closes since MARKET_HISTORY_START (cached as CSV)."""
    if MARKET_CACHE.exists():
        mk = pd.read_csv(MARKET_CACHE, parse_dates=["date"]).set_index("date")
        print(
            f"market closes: cache {MARKET_CACHE.name}, {len(mk)} sessions {mk.index.min().date()} .. {mk.index.max().date()}"
        )
        return mk
    import yfinance as yf

    raw = yf.download(
        ["^GSPC", "^VIX"],
        start=MARKET_HISTORY_START,
        end=MARKET_FETCH_END,
        auto_adjust=True,
        progress=False,
    )
    c = raw["Close"].rename(columns={"^GSPC": "gspc", "^VIX": "vix"})[["gspc", "vix"]]
    ci = pd.DatetimeIndex(c.index)
    c.index = (ci.tz_localize(None) if ci.tz is not None else ci).normalize()
    c.index.name = "date"
    c = c.dropna(how="all")
    OUT.mkdir(parents=True, exist_ok=True)
    c.to_csv(MARKET_CACHE, float_format="%.6f")
    print(
        f"market closes: fetched {len(c)} sessions {c.index.min().date()} .. {c.index.max().date()} -> {MARKET_CACHE}"
    )
    return c


def fomc_days() -> set[pd.Timestamp]:
    r = pd.read_parquet(
        ROOT / "data" / "releases.parquet", columns=["endbartime", "fomc release"]
    )
    flag = pd.to_numeric(r["fomc release"], errors="coerce").fillna(0.0) > 0
    d = pd.to_datetime(r.loc[flag, "endbartime"]).dt.normalize()
    fix = {pd.Timestamp(k): pd.Timestamp(v) for k, v in RELEASES_FOMC_DATE_FIX.items()}
    d = d.map(lambda x: fix.get(x, x))
    feed_last = d.max()
    ext = [pd.Timestamp(s) for s in FOMC_STATEMENT_DAYS_2023_12_ON]
    assert feed_last < min(ext), (feed_last, min(ext))
    # cross-check the extension against the deck's own list on the overlap
    lib = {
        pd.Timestamp(s) for s in asl.FOMC_STATEMENT_DAYS if pd.Timestamp(s) >= min(ext)
    }
    mine = {x for x in ext if x <= max(lib)}
    assert lib == mine, sorted(lib ^ mine)
    days = set(d) | {pd.Timestamp(s) for s in FOMC_UNSCHEDULED_2020} | set(ext)
    print(
        f"FOMC: releases.parquet flags through {feed_last.date()} (+{len(fix)} date fix), "
        f"+{len(ext)} scheduled statement days {ext[0].date()} .. {ext[-1].date()}, +{len(FOMC_UNSCHEDULED_2020)} unscheduled 2020"
    )
    return days


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7 + 14)


def causal_tercile(
    value: pd.Series, history: pd.Series, days: pd.DatetimeIndex
) -> tuple[pd.Series, pd.DataFrame]:
    """Tercile of value[t] against the cutoffs of history over sessions <= t.

    value and history are indexed by session; both hold only what was known at
    the previous close (the caller shifts them), so nothing from day t enters.
    """
    h = history.dropna()
    hv = h.to_numpy(float)
    hi = h.index
    lab, cuts = [], []
    for t in days:
        past = hv[hi <= t]
        c1, c2 = np.quantile(past, [1.0 / 3.0, 2.0 / 3.0])
        v = float(value.loc[t])
        lab.append(
            TERCILE_NAMES[0]
            if v < c1
            else TERCILE_NAMES[1]
            if v < c2
            else TERCILE_NAMES[2]
        )
        cuts.append((c1, c2, len(past)))
    return pd.Series(lab, index=days), pd.DataFrame(
        cuts, index=days, columns=["cut1", "cut2", "n_hist"]
    )


def build_calendar(days: pd.DatetimeIndex, mk: pd.DataFrame) -> pd.DataFrame:
    sess = pd.DatetimeIndex(mk.index)
    assert days.isin(sess).all(), "a trade day is not an S&P session"
    s = pd.Series(sess, index=sess)
    last = s.groupby([sess.year, sess.month]).transform("max")
    me = pd.DatetimeIndex(sess[(s == last).to_numpy()])
    pos_me = sess.get_indexer(me)
    me_m1 = pd.DatetimeIndex(sess[pos_me - 1])
    me_p1 = pd.DatetimeIndex(sess[pos_me[pos_me + 1 < len(sess)] + 1])
    fomc = fomc_days()
    q_opex: list[pd.Timestamp] = []
    m_opex: list[pd.Timestamp] = []
    for y in range(days.min().year, days.max().year + 1):
        for m in range(1, 13):
            tf = third_friday(y, m)
            eff = sess[
                sess <= tf
            ].max()  # a holiday Friday moves expiry to the session before
            (q_opex if m in (3, 6, 9, 12) else m_opex).append(eff)
    cal = pd.DataFrame(index=days)
    cal["month_end"] = days.isin(me)
    cal["me_minus1"] = days.isin(me_m1)
    cal["me_plus1"] = days.isin(me_p1)
    cal["fomc"] = days.isin(pd.DatetimeIndex(sorted(fomc)))
    cal["opex_quarterly"] = days.isin(pd.DatetimeIndex(q_opex))
    cal["opex_monthly"] = days.isin(pd.DatetimeIndex(m_opex))
    cal["dow"] = days.day_name().str[:3]
    cal["year"] = days.year.astype(str)
    # SPXW listing calendar: Mon/Wed/Fri (+ month-end, + holiday-shifted) series
    # until Tuesday and Thursday expirations were added in 2022. The split is
    # read off the frame: the first trade day after the last Tuesday or Thursday
    # session of the span that the frame does not trade (a single traded Tue/Thu
    # is not enough -- a Friday holiday moves that week's expiry to Thursday)
    span = sess[(sess >= days.min()) & (sess <= days.max())]
    miss = span.difference(days)
    miss_tt = miss[miss.dayofweek.isin([1, 3])]
    split = days[days > miss_tt.max()].min()
    cal["listing_era"] = np.where(
        days < split, "Mon/Wed/Fri + month-end listings", "every-session listings"
    )
    print(
        f"listing era split (first trade day after the last untraded Tue/Thu session {miss_tt.max().date()}): "
        f"{split.date()}; untraded sessions after it: {[str(x.date()) for x in miss[miss > split]]}"
    )
    # regimes from the previous session's closes
    vix_prev = mk["vix"].shift(1)
    lr = np.log(mk["gspc"]).diff()
    rv = np.sqrt(
        asl.PERIODS_PER_YEAR * (lr**2).rolling(RV_WINDOW, min_periods=RV_WINDOW).mean()
    )
    rv_prev = rv.shift(1)
    cal["vix_prev"] = vix_prev.reindex(days).to_numpy()
    cal["rv21_prev"] = rv_prev.reindex(days).to_numpy()
    cal["vix_tercile"], vcut = causal_tercile(vix_prev, vix_prev, days)
    cal["rv_tercile"], rcut = causal_tercile(rv_prev, rv_prev, days)
    for nm, cut in (("VIX", vcut), ("RV21", rcut)):
        print(
            f"{nm} tercile cutoffs over the frame: low/mid {cut['cut1'].min():.2f} .. {cut['cut1'].max():.2f}, "
            f"mid/high {cut['cut2'].min():.2f} .. {cut['cut2'].max():.2f} (history {int(cut['n_hist'].min())} .. {int(cut['n_hist'].max())} sessions)"
        )
    cal["vix_cut1"], cal["vix_cut2"] = vcut["cut1"], vcut["cut2"]
    cal["rv_cut1"], cal["rv_cut2"] = rcut["cut1"], rcut["cut2"]
    return cal


def cell_masks(cal: pd.DataFrame, family: str) -> dict[str, np.ndarray]:
    """Named boolean masks over the frame for one family of cells."""
    m: dict[str, np.ndarray] = {}
    if family == "calendar":
        me, m1, p1 = (
            cal[c].to_numpy(bool) for c in ("month_end", "me_minus1", "me_plus1")
        )
        m["month-end (last session)"] = me
        m["month-end T-1"] = m1
        m["month-end T+1"] = p1
        m["not within one session of a month-end"] = ~(me | m1 | p1)
        m["not month-end"] = ~me
        f = cal["fomc"].to_numpy(bool)
        m["FOMC statement day"] = f
        m["not FOMC"] = ~f
        qo, mo = (
            cal["opex_quarterly"].to_numpy(bool),
            cal["opex_monthly"].to_numpy(bool),
        )
        m["quarterly opex (3rd Fri Mar/Jun/Sep/Dec)"] = qo
        m["monthly opex (other 3rd Fridays)"] = mo
        m["not opex"] = ~(qo | mo)
        m["neither month-end nor FOMC"] = ~(me | f)
        for d in ("Mon", "Tue", "Wed", "Thu", "Fri"):
            m[f"day of week: {d}"] = (cal["dow"] == d).to_numpy()
        for y in sorted(cal["year"].unique()):
            m[f"year {y}"] = (cal["year"] == y).to_numpy()
        for e in ("Mon/Wed/Fri + month-end listings", "every-session listings"):
            m[f"era: {e}"] = (cal["listing_era"] == e).to_numpy()
    elif family == "regime":
        for t in TERCILE_NAMES:
            m[f"VIX tercile: {t}"] = (cal["vix_tercile"] == t).to_numpy()
        for t in TERCILE_NAMES:
            m[f"S&P 21-day realized vol tercile: {t}"] = (
                cal["rv_tercile"] == t
            ).to_numpy()
        for y in sorted(cal["year"].unique()):
            m[f"year {y}"] = (cal["year"] == y).to_numpy()
    else:
        raise ValueError(family)
    return m


# ---------------------------------------------------------------- bootstrap
def boot_idx(n: int, b: int) -> np.ndarray:
    return asl.circular_block_bootstrap_idx(
        np.random.default_rng(BOOT_SEED), n, BOOT_BLOCK, b
    )


def _ci(v: np.ndarray) -> tuple[float, float]:
    a = (100.0 - CI_LEVEL) / 2.0
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return float(lo), float(hi)


def _cell_stats(x: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    """Cell statistics of x on mask m; with ax=1 x and m are (B, n) resamples."""
    x = np.asarray(x, float)
    m = np.asarray(m, bool)
    n_all = x.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        xm = np.where(m, x, 0.0)
        cnt = m.sum(axis=-1).astype(float)
        s = xm.sum(axis=-1)
        tot = x.sum(axis=-1)
        mean = s / cnt
        var = ((xm**2).sum(axis=-1) - cnt * mean**2) / (cnt - 1.0)
        sd = np.sqrt(np.where(var > 0, var, np.nan))
        win = m & (x > 0)
        loss = m & (x < 0)
        nw, nl = win.sum(axis=-1), loss.sum(axis=-1)
        gain = np.where(win, x, 0.0).sum(axis=-1) / nw
        lss = np.where(loss, x, 0.0).sum(axis=-1) / nl
        out = {
            "n": cnt,
            "sum": s,
            "share_of_total_pct": 100.0 * s / tot,
            "mean": mean,
            "Sharpe_ann": mean / sd * ANN,
            "hit_rate": nw / cnt,
            "mean_gain": gain,
            "mean_loss": lss,
            "payoff_ratio": gain / np.abs(lss),
            "total_ex_cell": tot - s,
            "mean_in_minus_out": mean - (tot - s) / (n_all - cnt),
        }
    return out


# statistics that get an interval; a cell's share of the total gets none -- when the
# total's own interval comes near zero the ratio is unstable, so shares are point values
BOOT_KEYS = (
    "sum",
    "mean",
    "Sharpe_ann",
    "hit_rate",
    "payoff_ratio",
    "total_ex_cell",
    "mean_in_minus_out",
)


def boot_cells(
    job: tuple[str, str, str, np.ndarray, np.ndarray, dict[str, np.ndarray], int],
) -> list[dict]:
    """Worker: point estimates + block-bootstrap intervals for every cell of one series."""
    series, fill, family, x, q, masks, b = job
    idx = boot_idx(len(x), b)
    xb = x[idx]
    rows = []
    for name, m in masks.items():
        pt = _cell_stats(x, m)
        bs = _cell_stats(xb, m[idx])
        row = {
            "series": series,
            "fill": fill,
            "family": family,
            "cell": name,
            "n": int(pt["n"]),
            "n_buy": int((m & (q > 0)).sum()),
            "pct_of_days": 100.0 * float(m.mean()),
        }
        for k in (
            "sum",
            "share_of_total_pct",
            "mean",
            "Sharpe_ann",
            "hit_rate",
            "mean_gain",
            "mean_loss",
            "payoff_ratio",
            "total_ex_cell",
            "mean_in_minus_out",
        ):
            row[k] = float(pt[k])
            if k in BOOT_KEYS:
                row[f"{k}_lo"], row[f"{k}_hi"] = _ci(bs[k])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- blocks
def days_for_share(x: np.ndarray, level: float) -> float:
    tot = float(x.sum())
    if not tot > 0:
        return float("nan")
    c = np.cumsum(np.sort(x)[::-1])
    hit = np.nonzero(c >= level * tot - 1e-12)[0]
    return float(hit[0] + 1) if hit.size else float("nan")


def tail_block(
    pnl: dict, pos: dict, px: pd.DataFrame, cal: pd.DataFrame
) -> pd.DataFrame:
    from scipy.stats import hypergeom

    R = px["R"].to_numpy(float)
    order_r = np.argsort(-R, kind="stable")
    rows = []
    for series, fills in pnl.items():
        q = pos[series].to_numpy(float)
        n_buy = int((q > 0).sum())
        for fill, s in fills.items():
            x = s.to_numpy(float)
            n = len(x)
            o = np.argsort(-x, kind="stable")
            tot = float(x.sum())
            for k in TOP_K:
                top, bot = o[:k], o[::-1][:k]
                rest = np.setdiff1d(np.arange(n), top)
                rest_b = np.setdiff1d(np.arange(n), bot)
                xr, xb = x[rest], x[rest_b]
                mkt_top = order_r[
                    :k
                ]  # the k days the straddle paid most (largest moves)
                n_bought = int((q[mkt_top] > 0).sum())
                rows.append(
                    {
                        "series": series,
                        "fill": fill,
                        "k": k,
                        "n": n,
                        "total": tot,
                        "top_k_sum": float(x[top].sum()),
                        "top_k_share_pct": 100.0 * float(x[top].sum()) / tot
                        if tot > 0
                        else float("nan"),
                        "bottom_k_sum": float(x[bot].sum()),
                        "bottom_k_share_pct": 100.0 * float(x[bot].sum()) / tot
                        if tot > 0
                        else float("nan"),
                        "total_ex_top_k": float(xr.sum()),
                        "mean_ex_top_k": float(xr.mean()),
                        "t_ex_top_k": float(
                            xr.mean() / xr.std(ddof=1) * np.sqrt(len(xr))
                        ),
                        "Sharpe_ex_top_k": float(xr.mean() / xr.std(ddof=1) * ANN),
                        "total_ex_bottom_k": float(xb.sum()),
                        "Sharpe_ex_bottom_k": float(xb.mean() / xb.std(ddof=1) * ANN),
                        "top_k_n_buy": int((q[top] > 0).sum()),
                        "top_k_n_month_end": int(
                            cal["month_end"].to_numpy()[top].sum()
                        ),
                        "top_k_n_fomc": int(cal["fomc"].to_numpy()[top].sum()),
                        "bottom_k_n_buy": int((q[bot] > 0).sum()),
                        "bottom_k_n_month_end": int(
                            cal["month_end"].to_numpy()[bot].sum()
                        ),
                        # is the tail the trade's or the market's: of the k days the straddle
                        # paid most, how many did the rule own, against its buy rate
                        "mkt_top_k_bought": n_bought,
                        "mkt_top_k_bought_expected": k * n_buy / n,
                        "mkt_top_k_bought_p_hypergeom": float(
                            hypergeom.sf(n_bought - 1, n, n_buy, k)
                        )
                        if n_buy
                        else float("nan"),
                        "top_k_overlap_mkt_top_k": int(np.isin(top, mkt_top).sum()),
                        # ties at the k-th value make the top-k set arbitrary (always short
                        # earns exactly +1 on every day the straddle expires worthless)
                        "kth_value": float(x[o[k - 1]]),
                        "n_days_equal_kth_value": int(
                            np.isclose(x, x[o[k - 1]], rtol=0, atol=1e-12).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def tail_days(
    pnl: dict, pos: dict, px: pd.DataFrame, cal: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    move = (px["S_close"] / px["S"] - 1.0) * 100.0
    for series, fills in pnl.items():
        x = fills["mid"]
        o = np.argsort(-x.to_numpy(float), kind="stable")
        for side, ix in (("top", o[:N_TAIL_LIST]), ("bottom", o[::-1][:N_TAIL_LIST])):
            for rank, i in enumerate(ix, 1):
                d = x.index[i]
                rows.append(
                    {
                        "series": series,
                        "side": side,
                        "rank": rank,
                        "date": d.date().isoformat(),
                        "pnl_mid": float(x.iloc[i]),
                        "pnl_crossed": float(fills["crossed"].iloc[i]),
                        "position": float(pos[series].iloc[i]),
                        "R": float(px["R"].iloc[i]),
                        "entry_premium_pts": float(px["entry"].iloc[i]),
                        "spx_move_1530_close_pct": float(move.iloc[i]),
                        "month_end": bool(cal["month_end"].iloc[i]),
                        "fomc": bool(cal["fomc"].iloc[i]),
                        "dow": cal["dow"].iloc[i],
                        "vix_prev": float(cal["vix_prev"].iloc[i]),
                        "vix_tercile": cal["vix_tercile"].iloc[i],
                        "rv_tercile": cal["rv_tercile"].iloc[i],
                    }
                )
    return pd.DataFrame(rows)


def lo_mackinlay(x: np.ndarray, q: int) -> tuple[float, float]:
    """Variance ratio VR(q) of a daily series and its heteroskedasticity-robust z*.

    Lo and MacKinlay (1988): overlapping q-day sums, the unbiased variance
    estimators, and the robust asymptotic variance theta(q).
    """
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    e = x - mu
    s_a = (e @ e) / (n - 1)
    c = np.convolve(x, np.ones(q), mode="valid") - q * mu
    m = q * (n - q + 1) * (1.0 - q / n)
    s_c = (c @ c) / m
    vr = s_c / s_a
    e2 = e**2
    den = (e2.sum()) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * float(e2[j:] @ e2[:-j]) / den
        theta += (2.0 * (q - j) / q) ** 2 * delta
    return float(vr), float(np.sqrt(n) * (vr - 1.0) / np.sqrt(theta))


def path_block(
    pnl: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    summ, eps, roll, ac = [], [], {}, []
    for series, fills in pnl.items():
        for fill, s in fills.items():
            x = s.to_numpy(float)
            idx = s.index
            n = len(x)
            cum = np.cumsum(x)
            peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
            dd = cum - peak
            # episodes: maximal runs with dd < 0
            under = dd < -1e-12
            ep = []
            i = 0
            while i < n:
                if under[i]:
                    j = i
                    while j < n and under[j]:
                        j += 1
                    tr = i + int(np.argmin(dd[i:j]))
                    ep.append(
                        {
                            "peak_date": idx[i - 1].date().isoformat()
                            if i > 0
                            else "start",
                            "trough_date": idx[tr].date().isoformat(),
                            "recovery_date": idx[j].date().isoformat()
                            if j < n
                            else "not recovered",
                            "depth": float(dd[tr]),
                            "sessions_peak_to_trough": tr - i + 1,
                            "sessions_peak_to_recovery": (j - i + 1)
                            if j < n
                            else np.nan,
                            "sessions_under_water": j - i,
                            "calendar_days_peak_to_recovery": (idx[j] - idx[i - 1]).days
                            if (j < n and i > 0)
                            else np.nan,
                        }
                    )
                    i = j
                else:
                    i += 1
            ep_df = pd.DataFrame(ep).sort_values("depth").reset_index(drop=True)
            for r_, e in enumerate(ep_df.head(N_DD_EPISODES).to_dict("records"), 1):
                eps.append({"series": series, "fill": fill, "rank": r_, **e})
            longest = ep_df.sort_values("sessions_under_water", ascending=False).iloc[0]
            mdd = ep_df.iloc[0]
            row = {
                "series": series,
                "fill": fill,
                "n": n,
                "total": float(cum[-1]),
                "mean": float(x.mean()),
                "Sharpe_ann": float(x.mean() / x.std(ddof=1) * ANN),
                "t_newey_west": asl.newey_west_t(x)[0],
                "max_drawdown": float(mdd["depth"]),
                "max_dd_peak": mdd["peak_date"],
                "max_dd_trough": mdd["trough_date"],
                "max_dd_recovery": mdd["recovery_date"],
                "max_dd_sessions_to_recovery": mdd["sessions_peak_to_recovery"],
                "max_dd_in_daily_sd": float(mdd["depth"] / x.std(ddof=1)),
                "annual_mean_over_max_dd": float(
                    x.mean() * asl.PERIODS_PER_YEAR / abs(mdd["depth"])
                ),
                "longest_dd_sessions": int(longest["sessions_under_water"]),
                "longest_dd_peak": longest["peak_date"],
                "longest_dd_recovery": longest["recovery_date"],
                "pct_days_under_water": 100.0 * float(under.mean()),
                "n_dd_episodes": len(ep_df),
            }
            for w in ROLL_WINDOWS:
                r = s.rolling(w, min_periods=w)
                rs = (r.mean() / r.std(ddof=1) * ANN).dropna()
                roll[(series, fill, w)] = rs
                row[f"roll{w}_min"] = float(rs.min())
                row[f"roll{w}_min_end_date"] = rs.idxmin().date().isoformat()
                row[f"roll{w}_median"] = float(rs.median())
                row[f"roll{w}_max"] = float(rs.max())
                row[f"roll{w}_pct_windows_positive"] = 100.0 * float((rs > 0).mean())
            lb = acorr_ljungbox(x, lags=list(LB_LAGS), return_df=True)
            for L in LB_LAGS:
                row[f"LB_Q{L}"] = float(lb.loc[L, "lb_stat"])
                row[f"LB_p{L}"] = float(lb.loc[L, "lb_pvalue"])
            for h in VR_HORIZONS:
                vr, z = lo_mackinlay(x, h)
                row[f"VR{h}"], row[f"VR{h}_z_robust"] = vr, z
            summ.append(row)
            a = acf(x, nlags=ACF_LAGS, fft=False)
            band = 1.96 / np.sqrt(n)
            for L in range(1, ACF_LAGS + 1):
                ac.append(
                    {
                        "series": series,
                        "fill": fill,
                        "stat": "acf",
                        "lag": L,
                        "value": float(a[L]),
                        "band_95": band,
                        "outside_band": bool(abs(a[L]) > band),
                    }
                )
            for L in LB_LAGS:
                ac.append(
                    {
                        "series": series,
                        "fill": fill,
                        "stat": "ljung_box_p",
                        "lag": L,
                        "value": row[f"LB_p{L}"],
                        "band_95": np.nan,
                        "outside_band": bool(row[f"LB_p{L}"] < 0.05),
                    }
                )
            for h in VR_HORIZONS:
                ac.append(
                    {
                        "series": series,
                        "fill": fill,
                        "stat": "variance_ratio",
                        "lag": h,
                        "value": row[f"VR{h}"],
                        "band_95": np.nan,
                        "outside_band": bool(abs(row[f"VR{h}_z_robust"]) > 1.96),
                    }
                )
                ac.append(
                    {
                        "series": series,
                        "fill": fill,
                        "stat": "variance_ratio_z_robust",
                        "lag": h,
                        "value": row[f"VR{h}_z_robust"],
                        "band_95": 1.96,
                        "outside_band": bool(abs(row[f"VR{h}_z_robust"]) > 1.96),
                    }
                )
    roll_df = pd.DataFrame({f"{s} | {f} | {w}d": v for (s, f, w), v in roll.items()})
    roll_df.index.name = "date"
    return pd.DataFrame(summ), pd.DataFrame(eps), roll_df, pd.DataFrame(ac)


def hit_payoff_block(cell_rows: list[dict], pnl: dict) -> pd.DataFrame:
    """Hit rate / payoff rows (sides all, buy, sell) plus the few-big-days counts."""
    df = pd.DataFrame(cell_rows)
    extra = []
    for series, fills in pnl.items():
        for fill, s in fills.items():
            x = s.to_numpy(float)
            e = {"series": series, "fill": fill, "cell": "all days"}
            for lv in SHARE_LEVELS:
                d = days_for_share(x, lv)
                e[f"days_for_{int(lv * 100)}pct_of_pnl"] = d
                e[f"pct_days_for_{int(lv * 100)}pct_of_pnl"] = 100.0 * d / len(x)
            e["breakeven_hit_rate"] = np.nan
            extra.append(e)
    df = df.merge(pd.DataFrame(extra), on=["series", "fill", "cell"], how="left")
    df["breakeven_hit_rate"] = 1.0 / (1.0 + df["payoff_ratio"])
    return df.drop(columns=["family"])


def side_masks(q: np.ndarray) -> dict[str, np.ndarray]:
    return {"all days": np.ones(len(q), bool), "buy days": q > 0, "sell days": q < 0}


def diff_masks(
    cal: pd.DataFrame, d: np.ndarray, q: np.ndarray
) -> dict[str, np.ndarray]:
    m = {
        "all days": np.ones(len(d), bool),
        "buy days": q > 0,
        "sell days (difference is zero)": q < 0,
    }
    o = np.argsort(-d, kind="stable")
    for k in TOP_K:
        mk = np.zeros(len(d), bool)
        mk[o[:k]] = True
        m[f"top {k} days of the difference"] = mk
    for k in TOP_K:
        mk = np.zeros(len(d), bool)
        mk[o[::-1][:k]] = True
        m[f"bottom {k} days of the difference"] = mk
    m.update(cell_masks(cal, "calendar"))
    for nm, v in cell_masks(cal, "regime").items():
        if not nm.startswith("year"):
            m[nm] = v
    return m


# ---------------------------------------------------------------- figures
def figures(
    pnl: dict,
    pos: dict,
    cal: pd.DataFrame,
    regime: pd.DataFrame,
    path_summ: pd.DataFrame,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )
    ntail = 20
    # 1. cumulative P&L with the headline's tail days marked
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for ax, fill in zip(axes, FILLS):
        for s in FIG_SERIES:
            x = pnl[s][fill]
            ax.plot(
                x.index,
                x.cumsum(),
                lw=2.0 if s == HEADLINE else 1.4,
                color=COLOURS[s],
                label=f"{s}  (total {x.sum():+.1f})",
            )
        x = pnl[HEADLINE][fill]
        cum = x.cumsum()
        o = np.argsort(-x.to_numpy(float), kind="stable")
        top, bot = o[:ntail], o[::-1][:ntail]
        ax.scatter(
            x.index[top],
            cum.iloc[top],
            marker="^",
            s=40,
            color=COLOURS[HEADLINE],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=f"live-feasible: its {ntail} best days",
        )
        ax.scatter(
            x.index[bot],
            cum.iloc[bot],
            marker="v",
            s=40,
            color="#0b0b0b",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=f"live-feasible: its {ntail} worst days",
        )
        keep = np.ones(len(x), bool)
        keep[top] = False
        ex = pd.Series(np.where(keep, x.to_numpy(float), 0.0), index=x.index).cumsum()
        ax.plot(
            x.index,
            ex,
            lw=1.2,
            ls=":",
            color=COLOURS[HEADLINE],
            label=f"live-feasible without its {ntail} best days (total {ex.iloc[-1]:+.1f})",
        )
        ax.axhline(0.0, color="#52514e", lw=0.6)
        ax.set_ylabel(
            "cumulative P&L, premium units\n(one unit of premium a day, summed)"
        )
        ax.set_title(
            f"{'Midpoint' if fill == 'mid' else 'Crossed spread (buy at the ask, sell at the bid)'} fill",
            loc="left",
            fontsize=9.5,
        )
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=False)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Last-30-min straddle trade: cumulative P&L of sign(s) and always short, same 866 days",
        x=0.01,
        ha="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_cum_pnl_tails.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. per-regime bars: mean P&L per day with 95 % block-bootstrap intervals (mid)
    fams = [
        ("VIX tercile", "VIX tercile: "),
        ("S&P 21-day realized vol tercile", "S&P 21-day realized vol tercile: "),
        ("calendar year", "year "),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [3, 3, 5]}, sharey=True
    )
    reg = regime[regime["fill"] == "mid"]
    nser = len(FIG_SERIES)
    wbar = 0.8 / nser
    for ax, (title, pre) in zip(axes, fams):
        cells = [c for c in reg["cell"].unique() if c.startswith(pre)]
        for j, s in enumerate(FIG_SERIES):
            r = reg[(reg["series"] == s)].set_index("cell").loc[cells]
            xpos = np.arange(len(cells)) + (j - (nser - 1) / 2) * wbar
            ax.bar(xpos, r["mean"], width=wbar * 0.9, color=COLOURS[s], label=s)
            ax.errorbar(
                xpos,
                r["mean"],
                yerr=[r["mean"] - r["mean_lo"], r["mean_hi"] - r["mean"]],
                fmt="none",
                ecolor="#52514e",
                elinewidth=0.9,
                capsize=2,
            )
        ns = reg[(reg["series"] == HEADLINE)].set_index("cell").loc[cells, "n"]
        ax.set_xticks(np.arange(len(cells)))
        ax.set_xticklabels(
            [f"{c[len(pre) :]}\n(n={int(v)})" for c, v in zip(cells, ns)], fontsize=8
        )
        ax.axhline(0.0, color="#52514e", lw=0.6)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean P&L per day, premium units (midpoint)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="lower center",
        ncol=len(FIG_SERIES),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
    fig.suptitle(
        "P&L per day by regime (regimes from the previous close only; 95 % block-bootstrap intervals)",
        x=0.01,
        ha="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_regime_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. drawdown curves: one panel per series (small multiples, shared scale), both fills
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.0), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), FIG_SERIES):
        txt = []
        for fill, lw, alpha in (("mid", 1.6, 1.0), ("crossed", 1.0, 0.55)):
            x = pnl[s][fill]
            cum = x.cumsum().to_numpy()
            dd = cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]
            ax.plot(
                x.index,
                dd,
                lw=lw,
                alpha=alpha,
                color=COLOURS[s],
                label=f"{'midpoint' if fill == 'mid' else 'crossed spread'} (max {dd.min():.1f})",
            )
            txt.append(dd.min())
        ax.axhline(0.0, color="#52514e", lw=0.6)
        ax.set_title(s, loc="left", fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    for ax in axes[:, 0]:
        ax.set_ylabel("drawdown from the running peak,\npremium units")
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Drawdowns of the summed P&L (dark: midpoint fill; light: crossed spread), same 866 days",
        x=0.01,
        ha="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figures:", ", ".join(p.name for p in sorted(out.glob("fig_*.png"))))


# ---------------------------------------------------------------- driver
def run_set(
    px: pd.DataFrame,
    pos: dict[str, pd.Series],
    cal: pd.DataFrame,
    out: Path,
    workers: int,
    b: int,
    make_figures: bool,
) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    pnl = {s: daily_pnl(px, q) for s, q in pos.items()}
    series_order = list(pos)

    # bootstrap jobs: calendar, regime, side (hit/payoff) and difference cells
    jobs = []
    cal_m, reg_m = cell_masks(cal, "calendar"), cell_masks(cal, "regime")
    for s in series_order:
        q = pos[s].to_numpy(float)
        for fill in FILLS:
            x = pnl[s][fill].to_numpy(float)
            jobs.append((s, fill, "calendar", x, q, cal_m, b))
            jobs.append((s, fill, "regime", x, q, reg_m, b))
            jobs.append((s, fill, "side", x, q, side_masks(q), b))
            if s != SHORT:
                d = x - pnl[SHORT][fill].to_numpy(float)
                jobs.append((s, fill, "difference", d, q, diff_masks(cal, d, q), b))
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        res = list(ex.map(boot_cells, jobs))
    by: dict[str, list[dict]] = {
        "calendar": [],
        "regime": [],
        "side": [],
        "difference": [],
    }
    for job, rows in zip(jobs, res):
        by[job[2]].extend(rows)

    tabs: dict[str, pd.DataFrame] = {}
    tabs["tail_concentration"] = tail_block(pnl, pos, px, cal)
    tabs["tail_days"] = tail_days(pnl, pos, px, cal)
    tabs["calendar_cells"] = pd.DataFrame(by["calendar"]).drop(columns=["family"])
    tabs["regime_cells"] = pd.DataFrame(by["regime"]).drop(columns=["family"])
    tabs["hit_payoff"] = hit_payoff_block(by["side"], pnl)
    ps, pe, pr, pa = path_block(pnl)
    tabs["path_summary"], tabs["path_drawdowns"], tabs["path_autocorr"] = ps, pe, pa
    diff = pd.DataFrame(by["difference"]).drop(columns=["family"])
    diff = diff.rename(columns={"series": "sign_s_series"})
    diff.insert(1, "minus", SHORT)
    tabs["diff_attribution"] = diff
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
    pr.to_csv(out / "path_rolling_sharpe.csv", float_format="%.4f")
    # the day-level table every block is computed from, for anyone re-cutting it
    day = cal.copy()
    day.insert(0, "R", px["R"])
    for s in series_order:
        day[f"q | {s}"] = pos[s]
        for fill in FILLS:
            day[f"pnl {fill} | {s}"] = pnl[s][fill]
    day.index.name = "date"
    day.to_csv(out / "daily_pnl_and_flags.csv", float_format="%.6g")
    print(f"wrote {len(tabs) + 2} CSVs to {out}")
    if make_figures:
        figures(pnl, pos, cal, tabs["regime_cells"], ps, out)
    tabs["_pnl"] = pd.DataFrame({(s, f): pnl[s][f] for s in pnl for f in FILLS})
    return tabs


# ---------------------------------------------------------------- summary
def _pick(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    r = df[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def _f(v: float, nd: int = 2, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _fci(r: pd.Series, k: str, nd: int = 2, sign: bool = False) -> str:
    return f"{_f(r[k], nd, sign)} [{_f(r[k + '_lo'], nd, sign)}, {_f(r[k + '_hi'], nd, sign)}]"


def _short(s: str) -> str:
    return s.replace("sign(s): ", "")


def write_summary(
    out: Path,
    t: dict,
    rt: dict | None,
    cal: pd.DataFrame,
    px: pd.DataFrame,
    agree: dict[str, float],
    b: int,
) -> None:
    """SUMMARY.md: professor-facing; every number is read from the tables just written."""
    H, AF, BK, A0 = (
        SIGN_TAGS[k] for k in ("sub_live_ridge", "sub_ridge", "blk2", "a0")
    )
    MAIN = (H, AF, BK)
    hp, tc, cc, rc = (
        t["hit_payoff"],
        t["tail_concentration"],
        t["calendar_cells"],
        t["regime_cells"],
    )
    ps, da = t["path_summary"], t["diff_attribution"]
    n = len(px)
    L: list[str] = []
    w = L.append

    # claims the prose makes; a re-run that breaks one fails here instead of printing a wrong sentence
    for s in MAIN:
        assert _pick(tc, series=s, fill="mid", k=20)["top_k_n_buy"] == 20, s
        assert _pick(tc, series=s, fill="mid", k=20)["bottom_k_n_buy"] == 0, s
    assert _pick(tc, series=SHORT, fill="mid", k=20)["n_days_equal_kth_value"] > 20
    dows = ("Mon", "Tue", "Wed", "Thu", "Fri")
    for s in MAIN:
        neg = [
            d
            for d in dows
            if _pick(cc, series=s, fill="mid", cell=f"day of week: {d}")["sum"] < 0
        ]
        assert neg == ["Thu"], (s, neg)
    assert (
        max(
            dows,
            key=lambda d: _pick(cc, series=H, fill="mid", cell=f"day of week: {d}")[
                "sum"
            ],
        )
        == "Mon"
    )
    for s in MAIN:
        assert (
            min(_pick(ps, series=s, fill="mid")[f"LB_p{L}"] for L in LB_LAGS) > 0.05
        ), s
        assert all(
            abs(_pick(ps, series=s, fill="mid")[f"VR{h}_z_robust"]) <= 1.96
            for h in VR_HORIZONS
        ), s
    for s in MAIN:
        assert _pick(tc, series=s, fill="mid", k=20)["total"] > 0, s

    w("# Where the P&L of the last-30-min straddle trade comes from")
    w("")
    w(
        "Checklist item B1 (2026-09-29). Script: `experiments/close_pnl_decomposition.py`; every number below is "
        "read by that script from the CSVs in this directory."
    )
    w("")
    w("## Set-up")
    w("")
    w(
        "- **The trade.** At 15:30 ET one **straddle** — the nearest out-of-the-money call plus the nearest "
        "out-of-the-money put, same-day expiry, one position — held to the cash settlement at the official close. "
        "The **sign(s)** rule buys it when the forecast of the 15:30–16:00 realized variance exceeds the variance "
        "implied by the 15:30 quotes, and sells it otherwise. **Always short** sells it every day and uses no forecast."
    )
    w(
        f"- **Days.** The same {n} days for every series, {px.index.min().date()} to {px.index.max().date()} "
        f"(the rv_iv notebook's frame, {asl.trades_per_year(px.index):.0f} trade days a year)."
    )
    w(
        "- **Units.** Premium units: one unit of premium staked every day, daily returns summed, never compounded "
        "(the notebook's §13 convention). A day's P&L is q(X/P − 1) with q = ±1, X the settlement value and P the "
        "entry price. Sharpe ratios are annualized by √252 per trade day, as in the deck."
    )
    w(
        "- **Fills.** Midpoint, and crossed spread (buy at the ask, sell at the bid — the whole quoted spread once; "
        "a bound, not an estimate)."
    )
    w(
        "- **Forecasts.** Per-bar ridge on the live-feasible set (the 16 columns a 15:30 forecaster can rebuild; "
        "the lead series below), per-bar ridge on all features, and the block-diagonal ridge (the paper's headline "
        "forecast). The paper's HAR + calendar OLS baseline is in every CSV as a fourth sign(s) series."
    )
    w(
        "- **Scorer.** The rv_iv notebook's (recalibration on all 13 session bars), because the per-day tables come "
        "from it. The research scorer (recalibration on the 16:00 bar alone) is run separately for the two per-bar "
        "ridges in `research_scorer/` and summarized in §7; the two are never mixed in one table."
    )
    w(
        f"- **Intervals.** 95 % circular block bootstrap, {BOOT_BLOCK}-session blocks, {b:,} draws, one seed, the "
        "same resampled days for every series (so differences are paired). A cell statistic (month-ends, a VIX "
        "tercile, …) is recomputed on the resampled days that fall in the cell. Shares of a total are point values."
    )
    w("")
    w(
        "| series | P&L, mid | Sharpe, mid | P&L, crossed | Sharpe, crossed | buy days | max drawdown, mid |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    for s in (*MAIN, A0, SHORT):
        rm, rx = (
            _pick(hp, series=s, fill="mid", cell="all days"),
            _pick(hp, series=s, fill="crossed", cell="all days"),
        )
        nb = int(_pick(hp, series=s, fill="mid", cell="buy days")["n"])
        w(
            f"| {s} | {_f(rm['sum'], 1, True)} | {_fci(rm, 'Sharpe_ann')} | {_f(rx['sum'], 1, True)} | "
            f"{_fci(rx, 'Sharpe_ann')} | {nb} | {_f(_pick(ps, series=s, fill='mid')['max_drawdown'], 1)} |"
        )
    w("")

    # 1. tails
    r10, r20 = (
        _pick(tc, series=H, fill="mid", k=10),
        _pick(tc, series=H, fill="mid", k=20),
    )
    r20x = _pick(tc, series=H, fill="crossed", k=20)
    r50 = _pick(tc, series=H, fill="mid", k=50)
    hh = _pick(hp, series=H, fill="mid", cell="all days")
    s20 = _pick(tc, series=SHORT, fill="mid", k=20)
    w(
        "## 1. Tail days: a few days carry the P&L — the market's tail, the rule's selection"
    )
    w("")
    w(
        f"- The live-feasible sign(s) makes {_f(r10['total'], 1)} premium units at the midpoint. Its 10 best days "
        f"carry {_f(r10['top_k_share_pct'], 0)} % of that and its 20 best {_f(r20['top_k_share_pct'], 0)} %; without "
        f"the 20 best days the other {n - 20} days make {_f(r20['total_ex_top_k'], 1, True)} (Sharpe "
        f"{_f(r20['Sharpe_ex_top_k'])}, t {_f(r20['t_ex_top_k'])}). At the crossed spread the 20 best days carry "
        f"{_f(r20x['top_k_share_pct'], 0)} % and the other days make {_f(r20x['total_ex_top_k'], 1, True)}."
    )
    w(
        f"- {int(hh['days_for_50pct_of_pnl'])} days make half of the total, {int(hh['days_for_80pct_of_pnl'])} make "
        f"80 %, and {int(hh['days_for_100pct_of_pnl'])} days ({_f(hh['pct_days_for_100pct_of_pnl'], 1)} % of "
        f"{n}) add up to the whole of it: the remaining days net to zero."
    )
    w(
        f"- Every one of the 20 best days of each of the three sign(s) series is a **buy** day, and every one of the "
        f"20 worst is a **sell** day. Of the live-feasible's 20 best days {int(r20['top_k_n_month_end'])} are "
        f"month-ends and {int(r20['top_k_n_fomc'])} FOMC days (`tail_days.csv` lists them)."
    )
    w(
        f"- **Whose tail is it?** Always short shows the market's side of it: it earns at most +1 a day (the premium, "
        f"kept whole on {int(s20['n_days_equal_kth_value'])} days) and its 20 worst days make "
        f"{_f(s20['bottom_k_sum'], 1)} against a total of {_f(s20['total'], 1, True)}; without them it would make "
        f"{_f(s20['total_ex_bottom_k'], 1, True)}. The concentration comes from the straddle's payoff — a few large "
        f"moves in the last half hour — not from the rule. What the rule adds is *which* of those days it owns: of the "
        f"20 days the straddle paid most, the live-feasible sign(s) bought {int(r20['mkt_top_k_bought'])} against "
        f"{_f(r20['mkt_top_k_bought_expected'], 1)} expected at its buy rate (hypergeometric p = "
        f"{_f(r20['mkt_top_k_bought_p_hypergeom'], 3)}); of the 50, {int(r50['mkt_top_k_bought'])} against "
        f"{_f(r50['mkt_top_k_bought_expected'], 1)} (p = {_f(r50['mkt_top_k_bought_p_hypergeom'], 3)})."
    )
    w("")
    w(
        "| series (mid) | P&L | best 10: share | best 20: share | P&L without best 20 | Sharpe without best 20 | "
        "worst 20 | days for 100 % | biggest-20 payoff days bought (expected; p) | biggest-50 (expected; p) |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in (*MAIN, A0, SHORT):
        a10, a20, a50 = (_pick(tc, series=s, fill="mid", k=k) for k in (10, 20, 50))
        hs = _pick(hp, series=s, fill="mid", cell="all days")
        if s == SHORT:
            mk20 = mk50 = "0 (never buys)"
        else:
            mk20 = (
                f"{int(a20['mkt_top_k_bought'])} ({_f(a20['mkt_top_k_bought_expected'], 1)}; "
                f"{_f(a20['mkt_top_k_bought_p_hypergeom'], 3)})"
            )
            mk50 = (
                f"{int(a50['mkt_top_k_bought'])} ({_f(a50['mkt_top_k_bought_expected'], 1)}; "
                f"{_f(a50['mkt_top_k_bought_p_hypergeom'], 3)})"
            )
        sh10 = "ties at +1" if s == SHORT else f"{_f(a10['top_k_share_pct'], 0)} %"
        sh20 = "ties at +1" if s == SHORT else f"{_f(a20['top_k_share_pct'], 0)} %"
        w(
            f"| {_short(s)} | {_f(a20['total'], 1, True)} | {sh10} | {sh20} | {_f(a20['total_ex_top_k'], 1, True)} | "
            f"{_f(a20['Sharpe_ex_top_k'])} | {_f(a20['bottom_k_sum'], 1)} | {_f(hs['days_for_100pct_of_pnl'], 0)} | "
            f"{mk20} | {mk50} |"
        )
    w("")

    # 2. calendar
    def cal_row(s: str, cell: str, fill: str = "mid") -> pd.Series:
        return _pick(cc, series=s, fill=fill, cell=cell)

    me_h, me_s = (
        cal_row(H, "month-end (last session)"),
        cal_row(SHORT, "month-end (last session)"),
    )
    away = cal_row(H, "not within one session of a month-end")
    w("## 2. Calendar: the P&L comes from ordinary days, not from month-ends")
    w("")
    w(
        f"- **Month-ends** ({int(me_h['n'])} last sessions of a month in the frame): the straddle's long side pays on "
        f"them on average — always short makes {_fci(me_s, 'sum', 1, True)} there — and the rule sells most of them "
        f"(the live-feasible sign(s) buys {int(me_h['n_buy'])}), so it makes {_fci(me_h, 'sum', 1, True)} on "
        f"month-ends. Month-ends are not the source of the sign(s) P&L: on the {int(away['n'])} days more than one "
        f"session away from a month-end it makes {_fci(away, 'sum', 1, True)}, "
        f"{_f(away['share_of_total_pct'], 0)} % of its total."
    )
    f_h = cal_row(H, "FOMC statement day")
    w(
        f"- **FOMC statement days** ({int(f_h['n'])} in the frame; the release file's flags through 2023-11-01 plus "
        f"the Federal Reserve's published schedule after it): {_fci(f_h, 'sum', 1, True)} for the live-feasible "
        f"sign(s), {_fci(cal_row(SHORT, 'FOMC statement day'), 'sum', 1, True)} for always short — nothing measurable."
    )
    th = [cal_row(s, "day of week: Thu") for s in MAIN]
    w(
        "- **Day of week.** Thursday is the one weekday on which all three sign(s) series lose: "
        + "; ".join(f"{_short(s)} {_fci(r, 'sum', 1, True)}" for s, r in zip(MAIN, th))
        + f" over {int(th[0]['n'])} Thursdays (five weekdays times three series were looked at, so read it as a "
        f"flag, not a finding). Mondays pay most for the live-feasible series "
        f"({_fci(cal_row(H, 'day of week: Mon'), 'sum', 1, True)}; always short "
        f"{_fci(cal_row(SHORT, 'day of week: Mon'), 'sum', 1, True)})."
    )
    yrs = sorted(cal["year"].unique())
    w(
        "- **Years.** Live-feasible sign(s), P&L per day at the midpoint: "
        + "; ".join(
            f"{y} {_fci(cal_row(H, f'year {y}'), 'mean', 3, True)}" for y in yrs
        )
        + ". Always short: "
        + "; ".join(
            f"{y} {_fci(cal_row(SHORT, f'year {y}'), 'mean', 3, True)}" for y in yrs
        )
        + "."
    )
    split = cal.index[cal["listing_era"] == "every-session listings"].min().date()
    eras = ("era: Mon/Wed/Fri + month-end listings", "era: every-session listings")
    w(
        f"- **Listing era** (every-session expirations from {split} in this frame): "
        + "; ".join(f"{e[5:]} {_fci(cal_row(H, e), 'Sharpe_ann')} Sharpe" for e in eras)
        + "."
    )
    w("")
    w(
        "| cell (mid) | days | live-feasible: buys | live-feasible: P&L | all features: P&L | "
        "block-diagonal ridge: P&L | always short: P&L |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    for cell in (
        "month-end (last session)",
        "month-end T-1",
        "month-end T+1",
        "not within one session of a month-end",
        "FOMC statement day",
        "quarterly opex (3rd Fri Mar/Jun/Sep/Dec)",
        "monthly opex (other 3rd Fridays)",
        "day of week: Mon",
        "day of week: Thu",
    ):
        rh = cal_row(H, cell)
        w(
            f"| {cell} | {int(rh['n'])} | {int(rh['n_buy'])} | {_fci(rh, 'sum', 1, True)} | "
            f"{_fci(cal_row(AF, cell), 'sum', 1, True)} | {_fci(cal_row(BK, cell), 'sum', 1, True)} | "
            f"{_fci(cal_row(SHORT, cell), 'sum', 1, True)} |"
        )
    w("")

    # 3. regimes
    def reg_row(s: str, cell: str, fill: str = "mid") -> pd.Series:
        return _pick(rc, series=s, fill=fill, cell=cell)

    v_cells = [f"VIX tercile: {x}" for x in TERCILE_NAMES]
    r_cells = [f"S&P 21-day realized vol tercile: {x}" for x in TERCILE_NAMES]
    w("## 3. Regimes (known at the previous close)")
    w("")
    w(
        f"- **VIX tercile**: the previous session's VIX close against the terciles of every VIX close since "
        f"{MARKET_HISTORY_START[:4]} up to that session (cut-offs {_f(cal['vix_cut1'].min())}–"
        f"{_f(cal['vix_cut1'].max())} and {_f(cal['vix_cut2'].min())}–{_f(cal['vix_cut2'].max())} over the frame). "
        f"**Realized-vol tercile**: the S&P's {RV_WINDOW}-session close-to-close realized volatility up to the previous "
        f"close, against its own terciles since {MARKET_HISTORY_START[:4]} (cut-offs "
        f"{_f(100 * cal['rv_cut1'].min(), 1)}–{_f(100 * cal['rv_cut1'].max(), 1)} % and "
        f"{_f(100 * cal['rv_cut2'].min(), 1)}–{_f(100 * cal['rv_cut2'].max(), 1)} % a year). The terciles are levels "
        f"against a long history, so the 2020–2024 frame is unbalanced across them "
        f"({', '.join(str(int(reg_row(H, c)['n'])) for c in v_cells)} days by VIX tercile)."
    )
    excl = []
    for s in MAIN:
        for c in v_cells + r_cells:
            rr = reg_row(s, c)
            if rr["mean_lo"] > 0:
                excl.append(
                    f"{_short(s)} in {c.replace('S&P 21-day realized vol', 'realized-vol')}"
                )
    w(
        "- Cells whose P&L-per-day interval excludes zero (midpoint): "
        + "; ".join(excl)
        + "."
    )
    vh, vhs = reg_row(H, "VIX tercile: high"), reg_row(SHORT, "VIX tercile: high")
    dvh = _pick(da, sign_s_series=H, fill="mid", cell="VIX tercile: high")
    w(
        f"- In the **high-VIX tercile** ({int(vh['n'])} days) the live-feasible sign(s) makes "
        f"{_fci(vh, 'mean', 3, True)} a day and always short {_fci(vhs, 'mean', 3, True)}; the paired difference over "
        f"those days is {_fci(dvh, 'sum', 1, True)}. The low- and mid-VIX terciles carry "
        f"{_f(_pick(da, sign_s_series=H, fill='mid', cell='VIX tercile: low')['share_of_total_pct'], 0)} % and "
        f"{_f(_pick(da, sign_s_series=H, fill='mid', cell='VIX tercile: mid')['share_of_total_pct'], 0)} % of the "
        f"rule's advantage over always short; in stressed markets the two make the same."
    )
    w("")
    w(
        "| regime (mid) | days | live-feasible: per day | live-feasible: Sharpe | block-diagonal ridge: Sharpe | "
        "always short: per day | always short: Sharpe |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|")
    for c in v_cells + r_cells:
        rh, rb, rs = reg_row(H, c), reg_row(BK, c), reg_row(SHORT, c)
        w(
            f"| {c} | {int(rh['n'])} | {_fci(rh, 'mean', 3, True)} | {_fci(rh, 'Sharpe_ann')} | "
            f"{_fci(rb, 'Sharpe_ann')} | {_fci(rs, 'mean', 3, True)} | {_fci(rs, 'Sharpe_ann')} |"
        )
    w("")

    # 4. hit rate vs payoff
    ha, hb, hs_ = (
        _pick(hp, series=H, fill="mid", cell=c)
        for c in ("all days", "buy days", "sell days")
    )
    sa = _pick(hp, series=SHORT, fill="mid", cell="all days")
    w("## 4. Hit rate against payoff")
    w("")
    w(
        f"- The live-feasible sign(s) is right on {_fci(ha, 'hit_rate', 3)} of days — less often than always short "
        f"({_fci(sa, 'hit_rate', 3)}), which wins every day the straddle settles below its premium. The difference is "
        f"the payoff: sign(s) makes {_f(ha['mean_gain'], 3)} on a winning day and loses {_f(abs(ha['mean_loss']), 3)} on a "
        f"losing day (payoff ratio {_fci(ha, 'payoff_ratio', 3)}, so it breaks even at a {_f(ha['breakeven_hit_rate'], 3)} "
        f"hit rate); always short makes {_f(sa['mean_gain'], 3)} and loses {_f(abs(sa['mean_loss']), 3)} "
        f"(payoff {_fci(sa, 'payoff_ratio', 3)}, break-even {_f(sa['breakeven_hit_rate'], 3)})."
    )
    w(
        f"- **Buy days** ({int(hb['n'])}): hit rate {_f(hb['hit_rate'], 3)}, payoff ratio {_fci(hb, 'payoff_ratio')}, "
        f"P&L {_fci(hb, 'sum', 1, True)} — a long-shot profile. **Sell days** ({int(hs_['n'])}): hit rate "
        f"{_f(hs_['hit_rate'], 3)}, payoff ratio {_fci(hs_, 'payoff_ratio')}, P&L {_fci(hs_, 'sum', 1, True)}. Both legs "
        f"earn. On the days the rule sells, it and always short hold the same position and make "
        f"{_f(hs_['mean'], 3, True)} a day; on the days the rule buys, always short makes {_f(-hb['mean'], 3, True)} a "
        f"day — the days it moves to the long side are the ones that hurt the seller."
    )
    w("")
    w(
        "| series (mid) | hit rate | payoff ratio | break-even hit | buy days: P&L per day | buy days: hit | "
        "sell days: P&L per day | sell days: hit |"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in (*MAIN, A0, SHORT):
        a_, b_, c_ = (
            _pick(hp, series=s, fill="mid", cell=c)
            for c in ("all days", "buy days", "sell days")
        )
        buy = "—" if int(b_["n"]) == 0 else _fci(b_, "mean", 3, True)
        bhit = "—" if int(b_["n"]) == 0 else _f(b_["hit_rate"], 3)
        w(
            f"| {_short(s)} | {_fci(a_, 'hit_rate', 3)} | {_fci(a_, 'payoff_ratio')} | {_f(a_['breakeven_hit_rate'], 3)} | "
            f"{buy} | {bhit} | {_fci(c_, 'mean', 3, True)} | {_f(c_['hit_rate'], 3)} |"
        )
    w("")

    # 5. path
    ph = _pick(ps, series=H, fill="mid")
    w("## 5. Path: drawdowns, stability, serial dependence")
    w("")
    w(
        f"- Live-feasible sign(s), midpoint: deepest drawdown {_f(ph['max_drawdown'], 1)} premium units (peak "
        f"{ph['max_dd_peak']}, trough {ph['max_dd_trough']}, recovered {ph['max_dd_recovery']}, "
        f"{int(ph['max_dd_sessions_to_recovery'])} sessions), {_f(abs(ph['max_dd_in_daily_sd']), 1)} daily standard "
        f"deviations; longest time under water {int(ph['longest_dd_sessions'])} sessions ({ph['longest_dd_peak']} to "
        f"{ph['longest_dd_recovery']}). Always short's deepest drawdown is "
        f"{_f(_pick(ps, series=SHORT, fill='mid')['max_drawdown'], 1)} and its longest spell under water "
        f"{int(_pick(ps, series=SHORT, fill='mid')['longest_dd_sessions'])} sessions "
        f"({_pick(ps, series=SHORT, fill='mid')['longest_dd_peak']} to "
        f"{_pick(ps, series=SHORT, fill='mid')['longest_dd_recovery']})."
    )
    w(
        f"- Rolling Sharpe (trailing windows, midpoint): the 252-session Sharpe of the live-feasible sign(s) is positive "
        f"in {_f(ph['roll252_pct_windows_positive'], 1)} % of windows (minimum {_f(ph['roll252_min'])}, window ending "
        f"{ph['roll252_min_end_date']}); the 126-session one in {_f(ph['roll126_pct_windows_positive'], 1)} %; the "
        f"63-session one in {_f(ph['roll63_pct_windows_positive'], 1)} % (minimum {_f(ph['roll63_min'])}, ending "
        f"{ph['roll63_min_end_date']}). Always short: {_f(_pick(ps, series=SHORT, fill='mid')['roll252_pct_windows_positive'], 1)} % "
        f"of 252-session windows positive."
    )
    vr_flags = []
    for s in (*MAIN, A0, SHORT):
        r = _pick(ps, series=s, fill="mid")
        bad = [
            f"VR({h}) {_f(r[f'VR{h}'])} (z {_f(r[f'VR{h}_z_robust'])})"
            for h in VR_HORIZONS
            if abs(r[f"VR{h}_z_robust"]) > 1.96
        ]
        if bad:
            vr_flags.append(f"{_short(s)}: " + ", ".join(bad))
    lbp = {
        s: min(_pick(ps, series=s, fill="mid")[f"LB_p{L}"] for L in LB_LAGS)
        for s in MAIN
    }
    w(
        f"- Serial dependence: the three sign(s) series show none — smallest Ljung–Box p over lags "
        f"{', '.join(map(str, LB_LAGS))}: "
        + ", ".join(f"{_short(s)} {_f(v, 3)}" for s, v in lbp.items())
        + "; variance ratios (Lo–MacKinlay, heteroskedasticity-robust z) with |z| > 1.96: "
        + ("; ".join(vr_flags) if vr_flags else "none")
        + ". A variance ratio below one means multi-day P&L varies "
        "less than the daily P&L implies: losing days tend to be followed by recovering ones. Autocorrelations "
        f"outside ±1.96/√{n} at lags 1–{ACF_LAGS}: "
        + ", ".join(
            f"{_short(s)} {len(o)}"
            + (f" (lag {', '.join(str(int(x)) for x in o['lag'])})" if len(o) else "")
            for s in MAIN
            for o in [
                t["path_autocorr"].query(
                    "series == @s and fill == 'mid' and stat == 'acf' and outside_band"
                )
            ]
        )
        + f" of {ACF_LAGS} each — about what one in twenty by chance gives."
    )
    w("")
    w(
        "| series | fill | max drawdown | peak → trough → recovery | sessions to recover | 252-session Sharpe > 0 | "
        "Ljung–Box p (5/10/21) | VR(21), z |"
    )
    w("|---|---|---:|---|---:|---:|---|---:|")
    for s in (*MAIN, SHORT):
        for fill in FILLS:
            r = _pick(ps, series=s, fill=fill)
            rec = r["max_dd_sessions_to_recovery"]
            w(
                f"| {_short(s)} | {fill} | {_f(r['max_drawdown'], 1)} | {r['max_dd_peak']} → {r['max_dd_trough']} → "
                f"{r['max_dd_recovery']} | {_f(rec, 0)} | {_f(r['roll252_pct_windows_positive'], 1)} % | "
                f"{_f(r['LB_p5'], 2)} / {_f(r['LB_p10'], 2)} / {_f(r['LB_p21'], 2)} | "
                f"{_f(r['VR21'])}, {_f(r['VR21_z_robust'])} |"
            )
    w("")

    # 6. difference
    def d_row(s: str, cell: str, fill: str = "mid") -> pd.Series:
        return _pick(da, sign_s_series=s, fill=fill, cell=cell)

    dall, db = d_row(H, "all days"), d_row(H, "buy days")
    w("## 6. The difference sign(s) − always short, day by day")
    w("")
    w(
        "The two portfolios hold the same short straddle on every day sign(s) sells, so the paired difference is zero "
        "on those days and 2R on the days sign(s) buys (at the midpoint; at the crossed spread it is the ask-bought "
        "long minus the bid-sold short). The whole difference is therefore a statement about the buy days."
    )
    w("")
    w(
        f"- Live-feasible sign(s) minus always short: {_fci(dall, 'sum', 1, True)} premium units over {n} days "
        f"({_fci(dall, 'mean', 3, True)} a day), all of it on the {int(db['n'])} buy days."
    )
    for k in (10, 20):
        rk = d_row(H, f"top {k} days of the difference")
        w(
            f"- Without its {k} best days the difference is {_fci(rk, 'total_ex_cell', 1, True)}; those {k} days add "
            f"{_f(rk['sum'], 1, True)} ({_f(rk['share_of_total_pct'], 0)} % of the total)."
        )
    me_d, aw_d = (
        d_row(H, "month-end (last session)"),
        d_row(H, "not within one session of a month-end"),
    )
    w(
        f"- Month-ends add {_fci(me_d, 'sum', 1, True)} ({_f(me_d['share_of_total_pct'], 0)} %; {int(me_d['n_buy'])} buys "
        f"on {int(me_d['n'])} month-ends); days more than one session from a month-end add {_fci(aw_d, 'sum', 1, True)} "
        f"({_f(aw_d['share_of_total_pct'], 0)} %)."
    )
    w("")
    w(
        "| cell (live-feasible, mid) | days | buys | difference | share | difference without the cell |"
    )
    w("|---|---:|---:|---:|---:|---:|")
    cells6 = (
        [
            "all days",
            "top 10 days of the difference",
            "top 20 days of the difference",
            "bottom 20 days of the difference",
            "month-end (last session)",
            "month-end T-1",
            "month-end T+1",
            "not within one session of a month-end",
            "FOMC statement day",
            "day of week: Mon",
            "day of week: Thu",
        ]
        + [f"year {y}" for y in yrs]
        + v_cells
        + r_cells
    )
    for c in cells6:
        r = d_row(H, c)
        ex = "—" if c == "all days" else _fci(r, "total_ex_cell", 1, True)
        w(
            f"| {c} | {int(r['n'])} | {int(r['n_buy'])} | {_fci(r, 'sum', 1, True)} | "
            f"{_f(r['share_of_total_pct'], 0)} % | {ex} |"
        )
    w("")
    w(
        "| sign(s) series | fill | difference vs always short | per day | without its 10 best days | "
        "month-ends | high-VIX tercile |"
    )
    w("|---|---|---:|---:|---:|---:|---:|")
    for s in (*MAIN, A0):
        for fill in FILLS:
            r = d_row(s, "all days", fill)
            w(
                f"| {_short(s)} | {fill} | {_fci(r, 'sum', 1, True)} | {_fci(r, 'mean', 3, True)} | "
                f"{_fci(d_row(s, 'top 10 days of the difference', fill), 'total_ex_cell', 1, True)} | "
                f"{_fci(d_row(s, 'month-end (last session)', fill), 'sum', 1, True)} | "
                f"{_fci(d_row(s, 'VIX tercile: high', fill), 'sum', 1, True)} |"
            )
    w("")

    # 7. research scorer
    if rt is not None:
        w("## 7. The same decomposition under the research scorer")
        w("")
        w(
            "The research scorer recalibrates the forecast on the 16:00 bar alone (the scorer of the per-bar QLIKE "
            "tables and the closing-strategy master table). Only the two per-bar ridges have research-scorer rows; "
            "full tables in `research_scorer/`."
        )
        w("")
        w(
            "| series | same position as the notebook scorer | Sharpe, mid | Sharpe, crossed | best-20 share, mid | "
            "days for 100 % | biggest-20 payoff days bought | difference vs always short, mid | without its 10 best days |"
        )
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        rhp, rtc, rda = (
            rt["hit_payoff"],
            rt["tail_concentration"],
            rt["diff_attribution"],
        )
        for s in RESEARCH_ARMS_SPEC:
            a_ = _pick(rhp, series=s, fill="mid", cell="all days")
            x_ = _pick(rhp, series=s, fill="crossed", cell="all days")
            t20 = _pick(rtc, series=s, fill="mid", k=20)
            dd_ = _pick(rda, sign_s_series=s, fill="mid", cell="all days")
            d10 = _pick(
                rda, sign_s_series=s, fill="mid", cell="top 10 days of the difference"
            )
            w(
                f"| {_short(s)} | {_f(100 * agree[s], 1)} % | {_fci(a_, 'Sharpe_ann')} | {_fci(x_, 'Sharpe_ann')} | "
                f"{_f(t20['top_k_share_pct'], 0)} % | {_f(a_['days_for_100pct_of_pnl'], 0)} | "
                f"{int(t20['mkt_top_k_bought'])} ({_f(t20['mkt_top_k_bought_expected'], 1)}; "
                f"{_f(t20['mkt_top_k_bought_p_hypergeom'], 3)}) | {_fci(dd_, 'sum', 1, True)} | "
                f"{_fci(d10, 'total_ex_cell', 1, True)} |"
            )
        w("")

    w("## What this does not cover")
    w("")
    w(
        "- The frame ends 2024-04-30. Studies 85–90 (other session; `writeup/AUDIT_2026-09-18.md`) found no daily "
        "edge after the research span; nothing here speaks to those years."
    )
    w(
        "- The rule holds one unit whatever the size of the forecast, so a rare extreme forecast can only turn a day "
        "into a buy; over these days the recalibrated forecast is at most "
        + ", ".join(
            f"{_f(float((pd.read_parquet(DAILY / f'daily_{tag}.parquet')['rv_hat'] / pd.read_parquet(DAILY / f'daily_{tag}.parquet')['iv_var']).max()), 2)}× ({_short(lab)})"
            for tag, lab in SIGN_TAGS.items()
        )
        + " the implied variance — no extreme values enter."
    )
    w(
        "- Intervals on the best- and worst-day cells are conditional on the days the sample picked; read them as "
        "'how much rides on these particular days', not as a test."
    )
    w(
        "- Premium units weight every day alike. The dollar and index-point versions are in the notebook's §13."
    )
    w("")
    w("## Files")
    w("")
    for nm, d in (
        (
            "tail_concentration.csv",
            "block 1: shares of the best/worst k days, P&L without them, "
            "the market's biggest-payoff days the rule bought",
        ),
        (
            "tail_days.csv",
            "the 20 best and 20 worst days of each series with their flags",
        ),
        (
            "calendar_cells.csv",
            "block 2: month-end (T, T−1, T+1), FOMC, opex, weekday, year, listing era",
        ),
        ("regime_cells.csv", "block 3: VIX and realized-vol terciles, years"),
        (
            "hit_payoff.csv",
            "block 4: hit rate, payoff ratio, buy/sell split, days for 50/80/100 %",
        ),
        (
            "path_summary.csv, path_drawdowns.csv, path_rolling_sharpe.csv, path_autocorr.csv",
            "block 5: drawdowns, rolling Sharpe, autocorrelation, Ljung–Box, variance ratios",
        ),
        (
            "diff_attribution.csv",
            "block 6: sign(s) − always short by cell, paired intervals",
        ),
        ("daily_pnl_and_flags.csv", "the day-level table every block is cut from"),
        (
            "market_daily_closes.csv",
            f"^GSPC and ^VIX closes since {MARKET_HISTORY_START[:4]} "
            "(session calendar and regimes)",
        ),
        ("fig_cum_pnl_tails.png, fig_regime_bars.png, fig_drawdown.png", "figures"),
        (
            "research_scorer/",
            "the same CSVs under the research scorer (two per-bar ridges)",
        ),
    ):
        w(f"- `{nm}` — {d}")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "SUMMARY.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--boot", type=int, default=BOOT_B)
    ap.add_argument(
        "--no-research", action="store_true", help="skip the research-scorer set"
    )
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    px, pos = load_frame()
    print(
        f"frame: {len(px)} days {px.index.min().date()} .. {px.index.max().date()}; "
        f"trades per year {asl.trades_per_year(px.index):.1f}"
    )
    mk = load_market()
    # the ^GSPC closes double as the session calendar; check them against the settlement closes
    gap = (mk["gspc"].reindex(px.index) - px["S_close"]).abs()
    print(
        f"^GSPC close vs the frame's settlement close: max |gap| {float(gap.max()):.4f}"
    )
    cal = build_calendar(px.index, mk)
    print(
        "calendar counts on the frame:",
        {
            c: int(cal[c].sum())
            for c in (
                "month_end",
                "me_minus1",
                "me_plus1",
                "fomc",
                "opex_quarterly",
                "opex_monthly",
            )
        },
    )
    print(
        "forecast / implied variance (notebook scorer), max over the frame:",
        {
            t: round(
                float(
                    (
                        pd.read_parquet(DAILY / f"daily_{t}.parquet")["rv_hat"]
                        / pd.read_parquet(DAILY / f"daily_{t}.parquet")["iv_var"]
                    ).max()
                ),
                3,
            )
            for t in SIGN_TAGS
        },
    )
    tabs = run_set(px, pos, cal, OUT, a.workers, a.boot, make_figures=True)
    rtabs, agree = None, {}
    if not a.no_research:
        rpos = research_positions(px)
        for lab in RESEARCH_ARMS_SPEC:
            agree[lab] = float((rpos[lab] == pos[lab]).mean())
            print(
                f"research vs notebook scorer, {lab}: same position on {100 * agree[lab]:.1f} % of days"
            )
        rtabs = run_set(
            px,
            rpos,
            cal,
            OUT / "research_scorer",
            a.workers,
            a.boot,
            make_figures=False,
        )
    write_summary(OUT, tabs, rtabs, cal, px, agree, a.boot)


if __name__ == "__main__":
    main()
