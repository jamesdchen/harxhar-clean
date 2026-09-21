"""55 - calendar overrides on the 15:30 sign(s) straddle trade: month-end and FOMC days.

Where the two ideas came from, because it decides how far to trust each.  The
MONTH-END override was found by inspecting the deck sample (proposal 54: the
ridge is short into 83% of month-end closes and the long 15:30 straddle earns
+0.43 of premium on those days), so on the deck days it is a hypothesis the
sample suggested; only the 2024-05..2025-12 holdout and the pre-2020 underlying
are clean for it.  The FOMC override -- "force the trade long on FOMC days" --
was proposed a priori, before any FOMC-day number of this trade had been
looked at, so for it the deck days are a fair test and the holdout a second one.

The trade.  At 15:30 ET buy (q = +1) or sell (q = -1) the nearest-OTM SPX 0DTE
straddle according to sign(s), s = rv_hat - implied slice, and carry it to the
16:00 cash settlement.  Returns are per unit of the 15:30 midpoint premium;
"crossed" fills buy at the quoted ask and sell at the quoted bid.

Calendars.
  month-end   the last trading session of the calendar month (proposal 54's
              definition and flags, reproduced exactly).
  FOMC day    the day of a SCHEDULED statement (14:00 ET).  The repository's own
              flag (``data/releases.parquet``, column ``fomc release``, one 14:30
              bar per scheduled statement) is used where it exists; the feed
              ends on 2023-11-01, so the Federal Reserve's published schedule
              extends it through 2025-12-31 (the dates are listed below and the
              overlap 2020-2023 is cross-checked date by date).  The two March
              2020 emergency actions were unscheduled and are not override days.
  SECONDARY, reported and labelled as such, never part of a pass: the last
  trading day of the quarter, and the sessions immediately BEFORE and AFTER an
  FOMC statement (placebo-style neighbours: if the day before "works" as well
  as the day itself, the day itself is not the story).

PART A  forecast-free, all chain sessions 2020-01-03 .. 2025-12-31; the 413
        sessions from 2024-05-01 are the holdout.  Long the 15:30 straddle on
        FOMC days, on month-ends and on their union: at the mid and at the
        quoted ask, in premium units and in index points per contract; mean, t,
        hit; by period and by year.  Plus the underlying alone (no options),
        1998-2019 against 2020-2024, for FOMC days: the last bar's realized
        variance and squared net move relative to the bar before it -- and,
        because on FOMC days the 14:00-15:00 bars are themselves abnormal, also
        relative to the last bar's OWN trailing median over the prior 63
        sessions.
PART B  the deck's 866 days, all eight forecasts: sign(s); sign(s) with
        month-ends forced long / forced flat; with FOMC days forced long /
        forced flat; with both forced long.  Mean, Sharpe (mid, crossed), MaxDD,
        buy share, crossed Sharpe by year.  Each override against plain sign(s):
        a paired circular-block bootstrap interval of the Sharpe difference
        (block 21, B = 2000, seed 0) and two placebos -- (i) force the same
        NUMBER of randomly chosen days (B = 2000, seed 0), (ii) keep the
        calendar's spacing and shift it by every possible number of sessions
        (all n - 1 circular shifts, enumerated) -- with
        p = (1 + #{placebo gain >= observed}) / (1 + draws).
        A cell PASSES only if the interval of the CROSSED Sharpe difference
        excludes zero on the positive side AND both placebo p-values are below
        0.05.  8 forecasts x 5 overrides = 40 cells; the bootstrap leg alone
        passes about 0.025 x 40 = 1 by chance.
PART C  the short afternoon hold books of proposal 43 (sold 11:00 / 13:30 /
        14:30; held to settlement or bought back at 15:30; index points per
        contract) on FOMC days and on month-ends against other days, by period.

GATES, in order
  G1  the ridge's sign(s) on the deck days is 1.338322 mid / 0.869588 crossed.
  G2  the chain-built 15:30 straddle return equals the deck's R on all 866 days.
  G3  this script's daily tape and month-end flags equal proposal 54's
      ``c_daily.csv`` row for row; month-end counts 52 deck / 18 holdout / 70
      all; the all-sessions at-the-ask mean is +0.430.

Run:  python writeup/intraday_proposals/55_calendar_overrides_1530.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = DECK / "proposals" / "55"
P54_DAILY = DECK / "proposals" / "54" / "c_daily.csv"
P43_DAILY = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "43"
    / "a_daily_by_clock.csv"
)
RELEASES = ROOT / "data" / "releases.parquet"
CORE = ROOT / "data" / "core_stats.parquet"

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
DECK_END, OOS_START, PRE_OPTIONS_END = "2024-04-30", "2024-05-01", "2019-12-31"
PANEL_START = "1998-01-05"
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
PLACEBO_B, PLACEBO_SEED, PLACEBO_ALPHA = 2000, 0, 0.05
TRAIL_SESSIONS = 63

GATE_SHARPE = (1.338322, 0.869588)
GATE_TOL = 1e-6
GATE_R_TOL = 1e-6  # the chain stores quotes as float32
GATE_CSV_TOL = 1e-9
GATE_COUNTS = {"deck": 52, "holdout": 18, "all": 70}
GATE_ASK_MEAN, GATE_ASK_TOL = 0.430, 5e-4

#: Scheduled FOMC statement days, the Federal Reserve's published calendar.
FOMC_PUBLISHED = (
    "2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 "
    "2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 "
    "2021-12-15 2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 "
    "2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 "
    "2023-09-20 2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 "
    "2024-07-31 2024-09-18 2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 "
    "2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10"
).split()
PUBLISHED_FROM = "2020-01-01"

OVERRIDES: tuple[tuple[str, str, float], ...] = (
    ("month-ends forced long", "month_end", 1.0),
    ("month-ends forced flat", "month_end", 0.0),
    ("FOMC days forced long", "fomc", 1.0),
    ("FOMC days forced flat", "fomc", 0.0),
    ("month-ends and FOMC days forced long", "union", 1.0),
)


# ------------------------------------------------------------------ helpers --
def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(x, float))
    sd = a.std(axis=1, ddof=1)
    return np.where(sd > 0, a.mean(axis=1) / np.where(sd > 0, sd, 1.0) * ANN, np.nan)


def sharpe(x: np.ndarray) -> float:
    return float(sharpe_rows(x)[0])


def maxdd(x: np.ndarray) -> float:
    path = np.cumsum(np.asarray(x, float))
    peak = np.maximum(np.maximum.accumulate(path), 0.0)
    return float((path - peak).min())


def tstat(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan


def month_end_flags(days: pd.DatetimeIndex) -> pd.Series:
    """True on the last session of each calendar month in ``days`` (proposal 54)."""
    s = pd.Series(days, index=days)
    return s.groupby([days.year, days.month]).transform("max") == s


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"wrote {OUT / name}  ({len(df)} rows)")


# ---------------------------------------------------------------- calendars --
def trading_calendar(chain_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Proposal 54's calendar: the panel's 16:00-bar days, then the chain's."""
    c = pd.read_parquet(CORE, columns=["endbartime"])
    t = pd.to_datetime(c["endbartime"])
    rth = pd.DatetimeIndex(
        sorted(t[t.dt.strftime("%H:%M") == "16:00"].dt.normalize().unique())
    )
    return rth.union(chain_days)


def fomc_days() -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Repo flag through its last print, the published schedule after it."""
    r = pd.read_parquet(RELEASES, columns=["endbartime", "fomc release"])
    flagged = pd.to_datetime(r.loc[r["fomc release"] > 0, "endbartime"]).dt.normalize()
    repo = pd.DatetimeIndex(sorted(flagged.unique()))
    pub = pd.DatetimeIndex(pd.to_datetime(FOMC_PUBLISHED))
    last_live = repo.max()
    both_lo = pd.Timestamp(PUBLISHED_FROM)
    repo_ov = repo[(repo >= both_lo) & (repo <= last_live)]
    pub_ov = pub[(pub >= both_lo) & (pub <= last_live)]
    rows = [
        {
            "date": str(d.date()),
            "in_repo_flag": d in repo_ov,
            "in_published": d in pub_ov,
        }
        for d in repo_ov.union(pub_ov)
    ]
    check = pd.DataFrame(rows)
    check["agree"] = check["in_repo_flag"] == check["in_published"]
    days = repo.union(pub[pub > last_live])
    print(
        f"FOMC calendar: repo flag {len(repo)} scheduled statement days "
        f"{repo.min().date()} .. {last_live.date()}; published extension "
        f"{int((pub > last_live).sum())} days to {pub.max().date()}; overlap "
        f"{both_lo.date()} .. {last_live.date()}: {len(check)} days, "
        f"{int((~check['agree']).sum())} disagreements"
    )
    return days, check


def neighbours(
    cal: pd.DatetimeIndex, days: pd.DatetimeIndex, step: int
) -> pd.DatetimeIndex:
    pos = cal.get_indexer(days)
    pos = pos[pos >= 0] + step
    return cal[pos[(pos >= 0) & (pos < len(cal))]]


# --------------------------------------------------------------- the 15:30 tape
def build_tape() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
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
            "pts_mid": settle - mid,
            "pts_ask": settle - ask,
        },
        index=ch["dates"],
    )[ok]
    cal = trading_calendar(pd.DatetimeIndex(stamp.index))
    df["month_end"] = month_end_flags(cal).reindex(df.index).to_numpy(bool)
    return df, cal


def gates(df: pd.DataFrame) -> None:
    deck = pd.read_parquet(DECK / "daily_blk2.parquet").sort_index()
    r = deck["R"].to_numpy(float)
    q = np.where(deck["signal"].to_numpy(float) > 0, 1.0, -1.0)
    ask = (deck["ask_c"] + deck["ask_p"]).to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    ex = deck["exit"].to_numpy(float)
    got = (sharpe(q * r), sharpe(np.where(q > 0, ex / ask - 1.0, -(ex / bid - 1.0))))
    assert abs(got[0] - GATE_SHARPE[0]) < GATE_TOL, got
    assert abs(got[1] - GATE_SHARPE[1]) < GATE_TOL, got
    print(
        f"G1  ridge sign(s) on the deck days: {got[0]:.6f} mid / {got[1]:.6f} crossed"
    )
    j = df.join(deck["R"].rename("R_deck"), how="inner")
    dev = float((j["R_mid"] - j["R_deck"]).abs().max())
    assert len(j) == len(deck) and dev < GATE_R_TOL, (len(j), dev)
    print(
        f"G2  chain-built 15:30 straddle return vs the deck's R on {len(j)} days: {dev:.1e}"
    )
    ref = pd.read_csv(P54_DAILY, index_col=0, parse_dates=True)
    same_days = np.array_equal(
        ref.index.to_numpy("datetime64[ns]"), df.index.to_numpy("datetime64[ns]")
    )
    assert same_days, "day axis differs from proposal 54's"
    for col in ("R_mid", "R_ask", "pts_ask"):
        d54 = float(np.abs(ref[col].to_numpy(float) - df[col].to_numpy(float)).max())
        assert d54 < GATE_CSV_TOL, (col, d54)
    assert (ref["month_end"].to_numpy(bool) == df["month_end"].to_numpy(bool)).all()
    me = df[df["month_end"]]
    counts = {
        "deck": int((me.index <= DECK_END).sum()),
        "holdout": int((me.index >= OOS_START).sum()),
        "all": len(me),
    }
    assert counts == GATE_COUNTS, counts
    ask_mean = float(me["R_ask"].mean())
    assert abs(ask_mean - GATE_ASK_MEAN) < GATE_ASK_TOL, ask_mean
    print(
        f"G3  tape and month-end flags equal proposal 54's row for row; month-ends "
        f"{counts}; all-sessions long straddle at the ask {ask_mean:+.4f}"
    )


# ------------------------------------------------------------------- part A --
def period_masks(idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "deck period (to 2024-04-30)": np.asarray(idx <= DECK_END),
        "HOLDOUT 2024-05-01 .. 2025-12-31": np.asarray(idx >= OOS_START),
        "all sessions": np.ones(len(idx), bool),
    }


def long_row(g: pd.DataFrame, flag: np.ndarray) -> dict[str, Any]:
    a, b = g[flag], g[~flag]
    return {
        "sessions": len(g),
        "flagged_days": len(a),
        "long_mid_mean": float(a["R_mid"].mean()) if len(a) else np.nan,
        "long_mid_t": tstat(a["R_mid"]),
        "long_ask_mean": float(a["R_ask"].mean()) if len(a) else np.nan,
        "long_ask_t": tstat(a["R_ask"]),
        "hit": float((a["R_mid"] > 0).mean()) if len(a) else np.nan,
        "pts_per_contract_mid": float(a["pts_mid"].mean()) if len(a) else np.nan,
        "pts_per_contract_ask": float(a["pts_ask"].mean()) if len(a) else np.nan,
        "other_days_long_mid_mean": float(b["R_mid"].mean()),
        "other_days_short_mid_t": tstat(-b["R_mid"]),
    }


def part_a(df: pd.DataFrame, flags: dict[str, tuple[str, np.ndarray]]) -> None:
    idx = pd.DatetimeIndex(df.index)
    rows, yrows = [], []
    for name, (family, f) in flags.items():
        for pname, m in period_masks(idx).items():
            rows.append(
                {"calendar": name, "family": family, "sample": pname}
                | long_row(df[m], f[m])
            )
        for y in sorted(set(idx.year)):
            m = np.asarray(idx.year == y)
            yrows.append(
                {"calendar": name, "family": family, "year": y} | long_row(df[m], f[m])
            )
    tab, ytab = pd.DataFrame(rows), pd.DataFrame(yrows)
    write(tab, "a_long_straddle_by_period.csv")
    write(ytab, "a_long_straddle_by_year.csv")
    cols = [
        "calendar",
        "family",
        "sample",
        "flagged_days",
        "long_mid_mean",
        "long_mid_t",
        "long_ask_mean",
        "long_ask_t",
        "hit",
        "pts_per_contract_ask",
        "other_days_long_mid_mean",
    ]
    print("\n--- A1. long the 15:30 straddle on the flagged days (forecast-free)")
    print(tab[cols].round(3).to_string(index=False))
    ycols = [
        "calendar",
        "year",
        "flagged_days",
        "long_ask_mean",
        "hit",
        "pts_per_contract_ask",
    ]
    prim = ytab[ytab["family"] == "primary"]
    print("\n--- A2. by year, primary calendars, at the quoted ask")
    print(prim[ycols].round(3).to_string(index=False))


def part_a_underlying(fomc: pd.DatetimeIndex) -> pd.DataFrame:
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret", "sumret2"])
    c["t"] = pd.to_datetime(c["endbartime"])  # naive ET, bar-END labelled
    c = c[c["t"] >= PANEL_START]
    c["date"] = c["t"].dt.normalize()
    c["hhmm"] = c["t"].dt.strftime("%H:%M")
    p = c[c["hhmm"].isin(["15:30", "16:00"])].pivot_table(
        index="date", columns="hhmm", values=["sumret", "sumret2"], aggfunc="first"
    )
    p = p[(p[("sumret2", "15:30")] > 0) & (p[("sumret2", "16:00")] > 0)].dropna()
    days = pd.DatetimeIndex(p.index)
    rv_last = p[("sumret2", "16:00")]
    trail = rv_last.rolling(TRAIL_SESSIONS).median().shift(1)
    log_prior = np.log(rv_last / p[("sumret2", "15:30")]).to_numpy()
    log_trail = np.log(rv_last / trail).to_numpy()
    net_prior = (p[("sumret", "16:00")] ** 2 / p[("sumret2", "15:30")]).to_numpy()
    net_trail = (p[("sumret", "16:00")] ** 2 / trail).to_numpy()
    f = np.asarray(days.isin(fomc))
    have = np.isfinite(log_trail)
    rows = []
    for name, m in (
        (
            "1998-2019 (before any option in the study)",
            np.asarray(days <= PRE_OPTIONS_END),
        ),
        ("2020-2024", np.asarray(days > PRE_OPTIONS_END)),
    ):
        mm = m & have
        rec: dict[str, Any] = {
            "sample": name,
            "fomc_days": int((mm & f).sum()),
            "other_days": int((mm & ~f).sum()),
        }
        for tag, x in (
            ("vs_prior_bar", log_prior),
            ("vs_own_trailing_median", log_trail),
        ):
            a, b = x[mm & f], x[mm & ~f]
            t = (a.mean() - b.mean()) / np.sqrt(
                a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
            )
            rec[f"last_bar_RV_{tag}_fomc"] = float(np.exp(a.mean()))
            rec[f"last_bar_RV_{tag}_other"] = float(np.exp(b.mean()))
            rec[f"t_{tag}"] = float(t)
        for tag, x in (
            ("vs_prior_bar", net_prior),
            ("vs_own_trailing_median", net_trail),
        ):
            rec[f"median_net_move_sq_{tag}_fomc"] = float(np.median(x[mm & f]))
            rec[f"median_net_move_sq_{tag}_other"] = float(np.median(x[mm & ~f]))
        rows.append(rec)
    tab = pd.DataFrame(rows)
    write(tab, "a_underlying_fomc.csv")
    print("\n--- A3. the underlying's last half hour on FOMC days (no options)")
    print(tab.round(3).T.to_string())
    return tab


# ------------------------------------------------------------------- part B --
def part_b(flags_all: dict[str, pd.Series]) -> pd.DataFrame:
    books = {
        t: pd.read_parquet(DECK / f"daily_{t}.parquet").sort_index()
        for t in asl.MODEL_ORDER
    }
    d = books["blk2"]
    days = pd.DatetimeIndex(pd.to_datetime(d.index))
    n = len(d)
    r = d["R"].to_numpy(float)
    ask = (d["ask_c"] + d["ask_p"]).to_numpy(float)
    bid = (d["bid_c"] + d["bid_p"]).to_numpy(float)
    ex = d["exit"].to_numpy(float)
    cl, cs = ex / ask - 1.0, ex / bid - 1.0
    iv = d["iv_var"].to_numpy(float)
    flags = {
        k: v.reindex(days).fillna(False).to_numpy(bool) for k, v in flags_all.items()
    }
    print(
        "\ndeck days flagged: "
        + ", ".join(f"{k} {int(v.sum())}" for k, v in flags.items())
        + f" of {n}"
    )

    def pnl(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, float)
        return q * r, np.where(q > 0, q * cl, q * cs)

    boot = asl.circular_block_bootstrap_idx(
        np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
    )
    rng = np.random.default_rng([PLACEBO_SEED, n])
    rank = rng.random((PLACEBO_B, n)).argsort(
        axis=1
    )  # one draw set, reused by every cell
    shifts = np.arange(1, n)

    rows, yrows = [], []
    for tag, b in books.items():
        assert pd.DatetimeIndex(pd.to_datetime(b.index)).equals(days), tag
        q0 = np.where(b["rv_hat"].to_numpy(float) > iv, 1.0, -1.0)
        m0, c0 = pnl(q0)
        base = {
            "forecast": tag,
            "rule": "sign(s)",
            "days_overridden": 0,
            "positions_changed": 0,
            "buy_share": float((q0 > 0).mean()),
            "mean_mid": float(m0.mean()),
            "Sharpe_mid": sharpe(m0),
            "Sharpe_crossed": sharpe(c0),
            "MaxDD_mid": maxdd(m0),
            "MaxDD_crossed": maxdd(c0),
        }
        rows.append(base)
        by_year = {"forecast": tag, "rule": "sign(s)"} | {
            str(y): sharpe(c0[days.year == y]) for y in sorted(set(days.year))
        }
        yrows.append(by_year)
        for name, key, action in OVERRIDES:
            f = flags[key]
            q = np.where(f, action, q0)
            m1, c1 = pnl(q)
            rec: dict[str, Any] = {
                "forecast": tag,
                "rule": name,
                "days_overridden": int(f.sum()),
                "positions_changed": int((q != q0).sum()),
                "buy_share": float((q > 0).mean()),
                "mean_mid": float(m1.mean()),
                "Sharpe_mid": sharpe(m1),
                "Sharpe_crossed": sharpe(c1),
                "MaxDD_mid": maxdd(m1),
                "MaxDD_crossed": maxdd(c1),
            }
            for fill, x1, x0 in (("mid", m1, m0), ("crossed", c1, c0)):
                obs = sharpe(x1) - sharpe(x0)
                bd = sharpe_rows(x1[boot]) - sharpe_rows(x0[boot])
                lo, hi = np.percentile(bd, [2.5, 97.5])
                rec[f"dSharpe_{fill}"] = obs
                rec[f"dSharpe_{fill}_ci_lo"] = float(lo)
                rec[f"dSharpe_{fill}_ci_hi"] = float(hi)
                # placebo (i): the same NUMBER of days, chosen at random
                k = int(f.sum())
                masks = rank < k
                qp = np.where(masks, action, q0[None, :])
                xp = (
                    qp * r[None, :]
                    if fill == "mid"
                    else np.where(qp > 0, qp * cl[None, :], qp * cs[None, :])
                )
                gain = sharpe_rows(xp) - sharpe(x0)
                rec[f"p_random_days_{fill}"] = float(
                    (1 + int((gain >= obs).sum())) / (1 + PLACEBO_B)
                )
                # placebo (ii): the calendar's own spacing, shifted by every offset
                fs = np.stack([np.roll(f, int(s)) for s in shifts])
                qs = np.where(fs, action, q0[None, :])
                xs = (
                    qs * r[None, :]
                    if fill == "mid"
                    else np.where(qs > 0, qs * cl[None, :], qs * cs[None, :])
                )
                gain_s = sharpe_rows(xs) - sharpe(x0)
                rec[f"p_shifted_calendar_{fill}"] = float(
                    (1 + int((gain_s >= obs).sum())) / (1 + len(shifts))
                )
            rec["passes"] = bool(
                rec["dSharpe_crossed_ci_lo"] > 0.0
                and rec["p_random_days_crossed"] < PLACEBO_ALPHA
                and rec["p_shifted_calendar_crossed"] < PLACEBO_ALPHA
            )
            rows.append(rec)
            yrows.append(
                {"forecast": tag, "rule": name}
                | {str(y): sharpe(c1[days.year == y]) for y in sorted(set(days.year))}
            )
    tab, ytab = pd.DataFrame(rows), pd.DataFrame(yrows)
    write(tab, "b_overrides.csv")
    write(ytab, "b_crossed_sharpe_by_year.csv")
    ridge = tab[tab["forecast"] == "blk2"]
    print("\n--- B1. the ridge, every rule")
    print(
        ridge[
            [
                "rule",
                "days_overridden",
                "positions_changed",
                "buy_share",
                "mean_mid",
                "Sharpe_mid",
                "Sharpe_crossed",
                "MaxDD_mid",
                "MaxDD_crossed",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\n--- B2. the ridge, each override against plain sign(s)")
    print(
        ridge[ridge["rule"] != "sign(s)"][
            [
                "rule",
                "dSharpe_mid",
                "dSharpe_mid_ci_lo",
                "dSharpe_mid_ci_hi",
                "dSharpe_crossed",
                "dSharpe_crossed_ci_lo",
                "dSharpe_crossed_ci_hi",
                "p_random_days_crossed",
                "p_shifted_calendar_crossed",
                "passes",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\n--- B3. the ridge, crossed Sharpe by year")
    print(
        ytab[ytab["forecast"] == "blk2"]
        .drop(columns="forecast")
        .round(2)
        .to_string(index=False)
    )
    cells = tab[tab["rule"] != "sign(s)"]
    count = (
        cells.groupby("rule")
        .agg(
            forecasts=("passes", "size"),
            passes=("passes", "sum"),
            dSharpe_crossed_positive=("dSharpe_crossed", lambda v: int((v > 0).sum())),
            interval_excludes_zero=(
                "dSharpe_crossed_ci_lo",
                lambda v: int((v > 0).sum()),
            ),
            median_dSharpe_crossed=("dSharpe_crossed", "median"),
            median_p_random=("p_random_days_crossed", "median"),
            median_p_shifted=("p_shifted_calendar_crossed", "median"),
        )
        .reset_index()
    )
    write(count, "b_pass_counts.csv")
    print("\n--- B4. over the eight forecasts")
    print(count.round(3).to_string(index=False))
    print(
        f"cells {len(cells)}; passes {int(cells['passes'].sum())}; the bootstrap leg "
        f"alone passes about {0.025 * len(cells):.1f} by chance"
    )
    return tab


# ------------------------------------------------------------------- part C --
def part_c(flags_all: dict[str, pd.Series]) -> pd.DataFrame:
    a = pd.read_csv(P43_DAILY, index_col=0, parse_dates=True)
    idx = pd.DatetimeIndex(a.index)
    rows = []
    for key in ("fomc", "month_end"):
        f = flags_all[key].reindex(idx).fillna(False).to_numpy(bool)
        for pname, m in period_masks(idx).items():
            for clock in ("11:00", "13:30", "14:30"):
                for kind in ("hold", "flatten"):
                    pts = a[f"{clock}|{kind}"] * a[f"{clock}|entry"]
                    x, y = pts[m & f].dropna(), pts[m & ~f].dropna()
                    rows.append(
                        {
                            "calendar": key,
                            "sample": pname,
                            "book": f"short straddle sold {clock}, "
                            + (
                                "held to settlement"
                                if kind == "hold"
                                else "bought back 15:30"
                            ),
                            "flagged_days": len(x),
                            "mean_pts_flagged": float(x.mean()) if len(x) else np.nan,
                            "t_flagged": tstat(x),
                            "mean_pts_other": float(y.mean()),
                            "t_other": tstat(y),
                        }
                    )
    tab = pd.DataFrame(rows)
    write(tab, "c_hold_books.csv")
    print(
        "\n--- C. the short afternoon hold books on FOMC days (index points per contract)"
    )
    print(
        tab[tab["calendar"] == "fomc"]
        .drop(columns="calendar")
        .round(3)
        .to_string(index=False)
    )
    return tab


# ------------------------------------------------------------------- main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)

    df, cal = build_tape()
    gates(df)
    fomc, check = fomc_days()
    write(check, "a_fomc_calendar_crosscheck.csv")
    bad = check[~check["agree"]]
    print(
        "every disagreement in the overlap: "
        + ("none" if bad.empty else bad.to_string(index=False))
    )
    missing = [
        str(x.date())
        for x in fomc[(fomc >= df.index.min()) & (fomc <= df.index.max())]
        if x not in df.index
    ]
    print(
        f"FOMC days in the chain window absent from the scored tape: {missing or 'none'}"
    )

    idx = pd.DatetimeIndex(df.index)
    me = df["month_end"].to_numpy(bool)
    fo = np.asarray(idx.isin(fomc))
    qe = me & np.asarray(idx.month.isin([3, 6, 9, 12]))
    before = np.asarray(idx.isin(neighbours(cal, fomc, -1)))
    after = np.asarray(idx.isin(neighbours(cal, fomc, +1)))
    flags = {
        "FOMC day": ("primary", fo),
        "month-end": ("primary", me),
        "FOMC day or month-end": ("primary", fo | me),
        "quarter-end": ("SECONDARY", qe),
        "session before FOMC": ("SECONDARY", before & ~fo),
        "session after FOMC": ("SECONDARY", after & ~fo),
    }
    df_out = df.assign(fomc=fo, quarter_end=qe, before_fomc=before, after_fomc=after)
    df_out.to_csv(OUT / "a_daily.csv")
    print(
        f"sessions {len(df)}; FOMC days {int(fo.sum())}, month-ends {int(me.sum())}, "
        f"both {int((fo & me).sum())}"
    )
    part_a(df, flags)
    part_a_underlying(fomc)

    series = {
        "month_end": pd.Series(me, index=idx),
        "fomc": pd.Series(fo, index=idx),
        "union": pd.Series(fo | me, index=idx),
    }
    part_b(series)
    part_c(series)


if __name__ == "__main__":
    main()
