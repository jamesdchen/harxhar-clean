"""50 - the catastrophe replay, REPRICED: the book sold at the regime's premium.

Proposal 49 replayed every session of the 30-minute index panel through a
representative recent day's priced book with the whole option market FROZEN:
the strikes, the fill, the unit denominator and the implied volatility tape
were the representative day's and never moved.  That is an upper bound on the
damage, because a 2008 path was sold a 2023 premium.

This script keeps 49's paths, its gates and its pricer, and moves ONE thing:
on each historical session d the book is sold at the premium the market would
have charged IN THAT REGIME, and hedged with the matching implied.  The
repricing is calibrated once, on the 2020-2024 chain overlap, causally and by
regime:

  CALIBRATION   on the deck sessions that carry a straddle at the entry clock
                (866 at 13:30, 865 at 11:00), ln(entry midpoint / spot) is
                regressed on ln(RV21), the trailing 21-SESSION realized
                variance of the index lagged one session (proposal 49's own
                holiday-safe ``rv_trail``); the re-inverted implied REMAINING
                variance at the same stamp, ln(tot^2), is regressed on the same
                regressor.  Two OLS fits, intercept and slope, no selection.
                This is a calibration of the market's premium-vs-regime curve,
                NOT a forecast: RV21(d) is known before d opens, and the fit is
                a description of the 2020-2024 cross-section, used out of its
                own sample on 1998-2019.
  APPLICATION   for every replayed session d, premium_d / spot = exp(fitted) at
                that session's RV21, clipped to [5th percentile, 99th
                percentile] of the OVERLAP's own premium / spot; implied_d
                likewise, clipped to the overlap's own implied remaining
                variance percentiles.  Both clip levels are printed and the
                sessions on which they bind are counted.
  SCALING       premium_d is carried into points as (premium_d / spot) x the
                REPRESENTATIVE DAY'S entry spot, because the replayed path
                starts at that spot; the settlement intrinsic that pays against
                it is the path's own, on the same spot.  The strikes are the
                representative day's strike gap as a fraction of spot applied
                to the entry spot of the replayed path - which, the path
                starting at the representative day's spot, reproduces the
                representative day's strikes exactly (asserted, max deviation
                0).
  HEDGE         the delta tape is the representative day's total-volatility
                term structure scaled so that its ENTRY stamp carries exactly
                implied_d: tot = tot_rep x sqrt(implied_d / tot_rep[entry]^2).

Three conventions, all replayed on the same paths:

  (i)   repriced premium AND repriced implied            PRIMARY
  (ii)  repriced premium, frozen implied for the hedge
  (iii) proposal 49's frozen book, REPRODUCED             the bound

and a fourth COLUMN rather than a fourth book: the inverse-premium sizing
convention the research uses - one unit of premium sold per day, contracts
proportional to 1 / premium_d - which is the ``pnl_units`` column, the
per-unit distribution the deck's Sharpes live in.

APPROXIMATIONS, every one labelled again where it is made in the code:
  A1  the credit is the fitted MIDPOINT premium.  Proposal 49 and proposal 43
      sell at the quoted BID; convention (iii) therefore carries a half-spread
      that (i) and (ii) do not.  The summary carries the representative day's
      own half-spread fraction applied to premium_d, so the reader can subtract
      it; it is not subtracted anywhere.
  A2  the implied is a single number per session - a remaining-session total
      variance.  It has no smile, no term structure of its own, and does not
      move inside the session: only the LEVEL of the representative day's
      intraday decay shape is repriced.
  A3  the premium curve is fitted on 2020-2024 and applied to 1998-2019.  Part
      C states the extrapolation range in RV21 and counts the sessions outside
      the overlap's support and the sessions on which the clips bind.
  A4  21 replayed sessions have no complete trailing 21-session window; they
      take the calibration evaluated at the OVERLAP'S MEDIAN RV21 and are
      counted.
  A5  the FLATTEN terminal's buyback is proposal 49's Black-76 model mark at
      15:30 on the held volatility, not a quoted ask (49's own caveat).
  A6  the settlement is the panel's 16:00 bar close, not a settlement print.
  A7  exp(fitted) is the conditional GEOMETRIC mean of the premium, not its
      arithmetic mean: a log-linear fit with residual standard deviation sigma
      sits a factor exp(sigma^2 / 2) below the conditional arithmetic mean, and
      NO smearing correction is applied anywhere.  The factor is printed with
      the regressions and the resulting shortfall is visible, decile by decile,
      in table A7.

Gates, asserted before a single new number is computed:
  GATE 28   proposal 28's 11:00 representative day (2021-06-16) on the
            2020-03-20 and 2008-11-20 paths, through THIS script's pricer, to
            1e-12 of proposal 28's replay.csv - as proposal 49 did it
  GATE 45   proposal 45's 13:30 representative day (2020-11-23) under its own
            +5% last-bar jump, to 1e-9 of proposal 45's e_stress.csv - as
            proposal 49 did it
  GATE 49   proposal 49's frozen headline, reproduced to 1e-9 by CALLING 49's
            own functions (never its main): the median-premium 13:30 hold book,
            mean / p1 / p0.5 / p0.1 / min / worst-20, and the 2020-2024 overlap
            means of both the replay and the realised book
  GATE DEC  the additive decomposition this script uses to carry a PER-SESSION
            credit through 49's fixed-credit pricer (price the book at a zero
            credit, add the credit afterwards) reproduces 49's direct call
            bit-for-bit on the frozen convention

Outputs, all under results/atm_straddle_intraday_holdclose/proposals/50/:
  a_distribution.csv.gz  one row per replayed session x book x convention x
                         terminal (gzipped: the plain CSV is ~36 MB)
  a_summary.csv          moments, 1 / 0.5 / 0.1st percentiles, ES, anatomy
  a_worst.csv            the ten worst sessions of every book, with premium_d
  a_by_era.csv           1998-2007 / 2008-2009 / 2010-2019 / 2020-2024
  a_calibration.csv      the two regressions, the clips, the decile residual
  a_vs_realised.csv      the 2020-2024 overlap against the realised book
  b_sizing.csv           capital from the 1-in-200 session, and ruin
  b_delever.csv          the trailing-variance deleveraging rule
  c_blindspots.csv       what the replay still cannot see

Per-contract units everywhere: index points, dollars at $100 an index point,
and units of premium_d (the inverse-premium sizing column).

No randomness is used anywhere in this script, so there is no seed to state.
Parallelism: the replay is vectorised numpy over (n_sessions x n_stamps)
arrays and the 12 books are farmed to a ProcessPoolExecutor.

Run:  python writeup/intraday_proposals/50_cat_replay_repriced.py
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


def load_by_path(name: str, path: Path) -> ModuleType:
    """Import a proposal script by path; it is read, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P49_PATH = ROOT / "writeup" / "intraday_proposals" / "49_cat_replay.py"
#: proposal 49 itself: its pricer, its panel, its windows, its statistics.
#: Importing the module runs its body only; ``main`` is never called.
P49 = load_by_path("p49_for_50", P49_PATH)

HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "50"
REF49 = HOLD / "proposals" / "49"
SEED_LEDGER = ROOT / "results" / "live_seed" / "premium_ledger.parquet"

SPX_MULT = P49.SPX_MULT  # dollars per index point, one SPX option contract
DECK_END = P49.DECK_END
RV_WINDOW = P49.RV_WINDOW
ERAS = P49.ERAS
TAIL_PCTS = P49.TAIL_PCTS
ES_PCT = P49.ES_PCT
WORST_N = P49.WORST_N
WORST_FRAC = P49.WORST_FRAC
SETTLE_BAR = P49.SETTLE_BAR
MAIN_ENTRY = P49.MAIN_ENTRY
ENTRY_CLOCKS = P49.ENTRY_CLOCKS
TERMINALS = P49.TERMINALS
REP_QUANTILES = P49.REP_QUANTILES
REP_LABEL = P49.REP_LABEL
MEDIAN_ROLE = P49.MEDIAN_ROLE
N_RTH_BARS = P49.N_RTH_BARS

#: the three repricing conventions.
C_BOTH = "repriced premium + repriced implied"
C_PREM = "repriced premium, frozen implied"
C_FROZ = "49 frozen (reproduced)"
CONVENTIONS: tuple[str, ...] = (C_BOTH, C_PREM, C_FROZ)
PRIMARY = C_BOTH

#: the round-trip tolerance on "the representative day's strike gap as a
#: fraction of spot, applied to the entry spot of the replayed path": the path
#: starts at the representative day's own spot, so the round trip returns that
#: day's strikes up to one double-precision rounding of the division.
STRIKE_TOL = 1e-9  # index points
CLIP_LO_PCT = 5.0  # the overlap percentile the fitted level is floored at
CLIP_HI_PCT = 99.0  # the overlap percentile it is capped at
N_VOL_DECILES = 10

#: part B.
SIZING_TAIL_PCT = 0.5  # the 1-in-200 session loss sizes the capital
CAPITAL_BUDGETS: tuple[float, ...] = (0.05, 0.10)
RUIN_WINDOWS: tuple[int, ...] = (20, 60)
P45_STRESS_PTS = -305.731206  # proposal 45's last-bar -5% naked straddle
P45_STRESS_LABEL = "proposal 45 stress, last bar -5% (its $30.6k cell)"
DELEVER_HALF_PCT = 90.0
DELEVER_ZERO_PCT = 97.5
DELEVER_MIN_OBS = 252  # sessions before the expanding percentile is trusted

#: part C: the listing milestones the replay's product assumption leans on,
#: stated as ASSUMPTIONS and used only to count sessions.
LISTING_DAILY_FROM = "2022-05-11"
LISTING_MWF_FROM = "2016-01-01"
MWF = (0, 2, 4)  # Monday, Wednesday, Friday

#: GATE 49 targets, read from proposal 49's own written tables.
GATE49_BOOK = (MEDIAN_ROLE, MAIN_ENTRY, TERMINALS[0], "frozen IV")
GATE49_STATS: tuple[str, ...] = (
    "total_pts_mean",
    "total_pts_median",
    "total_pts_sd",
    "total_pts_p1",
    "total_pts_p0.5",
    "total_pts_p0.1",
    "total_pts_min",
)
GATE49_PRINTED: dict[str, float] = {
    "total_pts_mean": -1.9786,
    "total_pts_p1": -63.42,
    "total_pts_p0.5": -81.29,
    "total_pts_p0.1": -163.96,
    "total_pts_min": -431.99,
}
GATE49_WORST20_PTS = -1971.897096
GATE49_OVERLAP_REPLAY_MEAN = -2.414563806565796
GATE49_OVERLAP_REALISED_MEAN = 0.7983451428262665
GATE49_OVERLAP_REALISED_SD = 7.5481790630095436
GATE49_OVERLAP_REALISED_P1 = -25.5789426814204
GATE49_OVERLAP_REALISED_PREM = 13.984209001133127
GATE49_TOL = 1e-9
GATE49_PRINT_TOL = 5e-3
GATE49_WORST20_TOL = 5e-6


# ------------------------------------------------------------- the worker --
_P43_CACHE: list[ModuleType] = []


def _worker_p43() -> ModuleType:
    """Proposal 43, imported once per worker process (proposal 49's pattern)."""
    if not _P43_CACHE:
        _P43_CACHE.append(load_by_path("p43_for_50_worker", P49.P43_PATH))
    return _P43_CACHE[0]


def _reprice_job(job: dict[str, Any]) -> pd.DataFrame:
    """One book: one representative day x entry clock x repricing convention.

    Proposal 49's ``price_book`` takes a SCALAR credit, because 49's book is
    sold once at the representative day's bid.  This script needs a per-session
    credit, so the book is priced at a ZERO credit and premium_d is added
    afterwards.  The decomposition is exact: 49's option leg is
    ``-(settle - fill)``, which is ``-settle + fill``, and the hedge leg, the
    15:30 mark and the turnover do not see the credit at all.  GATE DEC asserts
    it bit-for-bit on the frozen convention.
    """
    p43 = _worker_p43()
    j = int(job["j"])
    n_k = int(job["n_k"])
    cum = job["cum"]
    n = int(cum.shape[0])
    s_path, s_settle = P49.spot_from_path(
        float(job["s_entry"]), j, n_k, cum, job["s_rep"]
    )
    tot = np.repeat(job["tot_rep"][None, :], n, axis=0)
    tot = tot * np.sqrt(job["iv_scale"])[:, None]
    res = P49.price_book(
        p43,
        tot,
        s_path,
        s_settle,
        float(job["k_c"]),
        float(job["k_p"]),
        0.0,  # zero credit; premium_d is added below
        1.0,  # the unit denominator is carried per session, not here
        j,
    )
    credit = np.asarray(job["credit_pts"], float)
    denom = np.asarray(job["denom_pts"], float)
    ret = job["ret"]
    net = ret.sum(axis=1)
    absum = np.abs(ret).sum(axis=1)
    big = np.abs(ret).max(axis=1)
    big_k = np.abs(ret).argmax(axis=1)
    bars = list(job["bars"])
    frames = []
    for terminal, o_key, h_key, c_key in (
        (TERMINALS[0], "option_hold_pts", "hedge_hold_pts", "cost_hold_pts"),
        (TERMINALS[1], "option_flatten_pts", "hedge_flatten_pts", "cost_flatten_pts"),
    ):
        opt = res[o_key] + credit
        hed = res[h_key]
        pnl = opt + hed
        frames.append(
            pd.DataFrame(
                {
                    "book": job["book"],
                    "day_role": job["day_role"],
                    "rep_date": job["rep_date"],
                    "entry_clock": job["entry_clock"],
                    "terminal": terminal,
                    "convention": job["convention"],
                    "session": pd.DatetimeIndex(job["dates"]).strftime("%Y-%m-%d"),
                    "era": job["era"],
                    "weekday": pd.DatetimeIndex(job["dates"]).weekday,
                    "rv21_trail": job["rv_used"],
                    "rv21_is_fallback": job["rv_fallback"],
                    "premium_over_spot": credit / float(job["s_entry"]),
                    "premium_pts": credit,
                    "premium_clipped": job["prem_clip"],
                    "implied_rem_var": job["implied_used"],
                    "implied_clipped": job["impl_clip"],
                    "iv_scale": job["iv_scale"],
                    "option_pts": opt,
                    "hedge_pts": hed,
                    "pnl_pts": pnl,
                    "option_units": opt / denom,
                    "hedge_units": hed / denom,
                    "pnl_units": pnl / denom,
                    "pnl_dollars": pnl * SPX_MULT,
                    "hedge_cost_pts": res[c_key],
                    "halfspread_pts": credit * float(job["halfspread_frac"]),
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


# -------------------------------------------------------- the calibration --
def ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """One univariate OLS fit with an intercept, and its fit statistics."""
    xv = np.asarray(x, float)
    yv = np.asarray(y, float)
    des = np.column_stack([np.ones(xv.size), xv])
    beta = np.linalg.lstsq(des, yv, rcond=None)[0]
    resid = yv - des @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {
        "n": float(xv.size),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": 1.0 - ss_res / ss_tot,
        "resid_sd": float(np.sqrt(ss_res / (xv.size - 2))),
        "resid_mean_abs": float(np.abs(resid).mean()),
    }


def calibrate(
    ch: dict[str, Any],
    idx: pd.DatetimeIndex,
    clocks: tuple[str, ...],
    clock: str,
    rv_trail: "pd.Series[float]",
    common: pd.DatetimeIndex,
) -> dict[str, Any]:
    """The premium-vs-regime and implied-vs-regime curves at one entry clock."""
    j = clocks.index(clock)
    prem = pd.Series(ch["entry"][:, j] / ch["S"][:, j], index=idx).reindex(common)
    implied = pd.Series(ch["tot"][:, j] ** 2, index=idx).reindex(common)
    rv = rv_trail.reindex(common)
    ok = (
        np.isfinite(prem)
        & (prem > 0.0)
        & np.isfinite(implied)
        & (implied > 0.0)
        & np.isfinite(rv)
        & (rv > 0.0)
    ).to_numpy()
    x = np.log(rv.to_numpy(float)[ok])
    fit_p = ols(x, np.log(prem.to_numpy(float)[ok]))
    fit_i = ols(x, np.log(implied.to_numpy(float)[ok]))
    pv = prem.to_numpy(float)[ok]
    iv = implied.to_numpy(float)[ok]
    return {
        "clock": clock,
        "j": j,
        "n": int(ok.sum()),
        "dates": common[ok],
        "premium_over_spot": pv,
        "implied_rem_var": iv,
        "rv21": rv.to_numpy(float)[ok],
        "fit_premium": fit_p,
        "fit_implied": fit_i,
        "prem_lo": float(np.percentile(pv, CLIP_LO_PCT)),
        "prem_hi": float(np.percentile(pv, CLIP_HI_PCT)),
        "impl_lo": float(np.percentile(iv, CLIP_LO_PCT)),
        "impl_hi": float(np.percentile(iv, CLIP_HI_PCT)),
        "rv_median": float(np.median(rv.to_numpy(float)[ok])),
        "rv_min": float(rv.to_numpy(float)[ok].min()),
        "rv_max": float(rv.to_numpy(float)[ok].max()),
    }


def apply_curve(cal: dict[str, Any], rv: np.ndarray) -> dict[str, np.ndarray]:
    """The fitted, clipped premium / spot and implied remaining variance."""
    rv_used = np.where(np.isfinite(rv) & (rv > 0.0), rv, cal["rv_median"])
    fallback = ~(np.isfinite(rv) & (rv > 0.0))
    lx = np.log(rv_used)
    prem_raw = np.exp(
        cal["fit_premium"]["intercept"] + cal["fit_premium"]["slope"] * lx
    )
    impl_raw = np.exp(
        cal["fit_implied"]["intercept"] + cal["fit_implied"]["slope"] * lx
    )
    prem = np.clip(prem_raw, cal["prem_lo"], cal["prem_hi"])
    impl = np.clip(impl_raw, cal["impl_lo"], cal["impl_hi"])
    return {
        "rv_used": rv_used,
        "rv_fallback": fallback,
        "prem_raw": prem_raw,
        "prem": prem,
        "prem_clip": np.where(
            prem_raw < cal["prem_lo"], -1, np.where(prem_raw > cal["prem_hi"], 1, 0)
        ),
        "impl_raw": impl_raw,
        "impl": impl,
        "impl_clip": np.where(
            impl_raw < cal["impl_lo"], -1, np.where(impl_raw > cal["impl_hi"], 1, 0)
        ),
    }


# -------------------------------------------------------------- the tables --
KEYS: list[str] = [
    "day_role",
    "rep_date",
    "entry_clock",
    "terminal",
    "convention",
]


def summarise(dist: pd.DataFrame) -> pd.DataFrame:
    """Moments, tail percentiles, ES and worst-1% anatomy, per book."""
    rows = []
    for key, d in dist.groupby(KEYS, sort=False):
        row: dict[str, Any] = dict(zip(KEYS, key, strict=True))
        row["n_sessions"] = int(len(d))
        row["n_rv_fallback"] = int(d["rv21_is_fallback"].sum())
        row["n_premium_clipped_low"] = int((d["premium_clipped"] < 0).sum())
        row["n_premium_clipped_high"] = int((d["premium_clipped"] > 0).sum())
        row["n_implied_clipped_low"] = int((d["implied_clipped"] < 0).sum())
        row["n_implied_clipped_high"] = int((d["implied_clipped"] > 0).sum())
        row["mean_premium_pts"] = float(d["premium_pts"].mean())
        row["median_premium_pts"] = float(d["premium_pts"].median())
        row["min_premium_pts"] = float(d["premium_pts"].min())
        row["max_premium_pts"] = float(d["premium_pts"].max())
        row["mean_hedge_cost_pts"] = float(d["hedge_cost_pts"].mean())
        row["mean_halfspread_pts"] = float(d["halfspread_pts"].mean())
        row["mean_pts_net_of_rep_halfspread"] = float(d["pnl_pts"].mean()) - float(
            d["halfspread_pts"].mean()
        )
        for leg, col in (("total_", "pnl"), ("option_", "option"), ("hedge_", "hedge")):
            for unit in ("pts", "units"):
                row.update(
                    P49.tail_stats(d[f"{col}_{unit}"].to_numpy(float), f"{leg}{unit}_")
                )
        row.update(P49.tail_stats(d["pnl_dollars"].to_numpy(float), "total_dollars_"))
        row.update(P49.worst_anatomy(d))
        rows.append(row)
    return pd.DataFrame(rows)


def worst_table(dist: pd.DataFrame) -> pd.DataFrame:
    cols = KEYS + [
        "session",
        "era",
        "pnl_pts",
        "pnl_units",
        "pnl_dollars",
        "option_pts",
        "hedge_pts",
        "premium_pts",
        "premium_over_spot",
        "implied_rem_var",
        "rv21_trail",
        "path_net_logpct",
        "path_sum_abs_bar_logpct",
        "path_trend_ratio",
        "path_max_abs_bar_logpct",
        "path_max_abs_bar_stamp",
        "path_last_bar_logpct",
        "S_settle_replayed",
    ]
    out = []
    for _, d in dist.groupby(KEYS, sort=False):
        w = d.nsmallest(WORST_N, "pnl_pts").copy()
        w["rank"] = np.arange(1, len(w) + 1)
        out.append(w[cols + ["rank"]])
    return pd.concat(out, ignore_index=True)


def era_table(dist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, d in dist.groupby(KEYS, sort=False):
        for era_name, _lo, _hi in ERAS:
            sel = d[d["era"] == era_name]
            row: dict[str, Any] = dict(zip(KEYS, key, strict=True))
            row["era"] = era_name
            row["n_sessions"] = int(len(sel))
            if len(sel) == 0:
                rows.append(row)
                continue
            row["mean_premium_pts"] = float(sel["premium_pts"].mean())
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


def expanding_pct(v: np.ndarray, pct: float, min_obs: int) -> np.ndarray:
    """The percentile of every STRICTLY EARLIER element, element by element."""
    s = pd.Series(np.asarray(v, float))
    q = s.expanding(min_periods=min_obs).quantile(pct / 100.0).shift(1)
    return q.to_numpy(float)


# -------------------------------------------------------------------- main --
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 140)
    pd.set_option("display.max_rows", 900)
    t_start = time.time()

    p43 = load_by_path("p43_for_50", P49.P43_PATH)
    clocks: tuple[str, ...] = tuple(p43.CLOCKS)
    n_k = len(clocks)

    # ------------------------------------------------------ the panel ------
    pan = P49.panel_bars()
    ret_all = pan["ret"]
    print(
        f"panel data/core_stats.parquet: {len(ret_all)} weekday sessions carrying at "
        f"least one of the {N_RTH_BARS} regular-hours 30-minute stamps, "
        f"{pd.DatetimeIndex(ret_all.index).min().date()}.."
        f"{pd.DatetimeIndex(ret_all.index).max().date()} (proposal 49's panel_bars, "
        "called, not copied)"
    )
    windows = {c: P49.window_set(pan, c) for c in ENTRY_CLOCKS}
    for c in ENTRY_CLOCKS:
        w = windows[c]
        print(
            f"entry {c}: window bars {', '.join(w['bars'])}; sessions replayed "
            f"{w['n_replayed']}, skipped {w['n_skipped']}; span "
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
    p28 = load_by_path("p28_for_50", P49.P28_PATH)
    mk = p28._standalone()
    pkg28 = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    g28 = p28.build_grids(pkg28, mk)
    e28 = g28["sl"]["entry"].to_numpy(float)
    ratio28 = pd.Series(e28 / g28["Sg"][:, g28["j"]], index=g28["dates"]).sort_values()
    rep28 = pd.Timestamp(ratio28.index[len(ratio28) // 2])
    assert rep28 == pd.Timestamp(P49.GATE28_REP_DAY), rep28
    i28 = g28["dates"].get_loc(rep28)
    j28 = int(g28["j"])
    tot28 = np.where(
        g28["IVg"][i28] > 0, g28["IVg"][i28] * np.sqrt(g28["h_row"]), np.nan
    )
    w28 = windows["11:00"]
    pos28 = {d.strftime("%Y-%m-%d"): k for k, d in enumerate(w28["dates"])}
    rows28 = np.array([pos28[s] for s in P49.GATE28_SESSIONS])
    s_path28, s_settle28 = P49.spot_from_path(
        float(g28["Sg"][i28, j28]), j28, n_k, w28["cum"][rows28], g28["Sg"][i28]
    )
    r28 = P49.price_book(
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
    ref28 = pd.read_csv(P49.REF28).set_index("session")["r_hold_units"]
    for k, s in enumerate(P49.GATE28_SESSIONS):
        dev = abs(float(mine28[k]) - float(ref28.loc[s]))
        dev_p = abs(float(mine28[k]) - P49.GATE28_PRINTED[s])
        assert dev < P49.GATE28_TOL, (s, dev)
        assert dev_p < P49.GATE28_PRINT_TOL, (s, dev_p)
        print(
            f"GATE 28  {rep28.date()} book, {s} path: r_hold {float(mine28[k]):+.12f} "
            f"units vs proposal 28's {float(ref28.loc[s]):+.12f} (|dev| {dev:.1e}, "
            f"tolerance {P49.GATE28_TOL:.0e})"
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
    assert rep45 == pd.Timestamp(P49.GATE45_REP_DAY), rep45
    i45 = int(idx.get_loc(rep45))
    s_row45 = ch["S"][i45].copy()
    s_settle45 = np.array([float(s_row45[-1]) * (1.0 + P49.GATE45_JUMP)])
    r45 = P49.price_book(
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
    ref45 = pd.read_csv(P49.REF45)
    sel45 = ref45[
        ref45["structure"].str.startswith("naked")
        & (ref45["placement"] == "last bar")
        & np.isclose(ref45["jump_pct"], 100.0 * P49.GATE45_JUMP)
    ].iloc[0]
    for name, mine, ref in (
        (
            "four option legs, units",
            float(r45["option_hold_pts"][0]) / float(ch["entry"][i45, j13]),
            float(sel45["four_leg_units"]),
        ),
        (
            "four option legs, dollars",
            float(r45["option_hold_pts"][0]) * SPX_MULT,
            float(sel45["four_leg_dollars"]),
        ),
        (
            "hedge, dollars",
            float(r45["hedge_hold_pts"][0]) * SPX_MULT,
            float(sel45["hedge_dollars"]),
        ),
    ):
        dev = abs(mine - ref)
        assert dev < P49.GATE45_TOL, (name, dev)
        print(
            f"GATE 45  {rep45.date()} book, +{100 * P49.GATE45_JUMP:.0f}% last bar, "
            f"{name}: {mine:+.6f} vs proposal 45's {ref:+.6f} (|dev| {dev:.1e}, "
            f"tolerance {P49.GATE45_TOL:.0e})"
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
        for q in qs:
            k = int(np.ceil(q * len(r)))
            day = pd.Timestamp(r.index[k - 1])
            i = int(idx.get_loc(day))
            reps.append(
                {
                    "day_role": REP_LABEL[q],
                    "entry_clock": clock,
                    "quantile": q,
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
                    "implied_rem_var_entry": float(ch["tot"][i, j]) ** 2,
                }
            )
    rep_tab = pd.DataFrame(reps)
    ref_rep = pd.read_csv(REF49 / "a_representative_days.csv")
    for _, rr in rep_tab.iterrows():
        m = (ref_rep["day_role"] == rr["day_role"]) & (
            ref_rep["entry_clock"] == rr["entry_clock"]
        )
        assert str(ref_rep.loc[m, "rep_date"].iloc[0]) == rr["rep_date"], rr
    print(
        "\nrepresentative days reproduced from proposal 49: "
        + ", ".join(
            f"{r['day_role']} @ {r['entry_clock']} = {r['rep_date']}"
            for _, r in rep_tab.iterrows()
        )
    )

    # ----------------------------------------- the calibration's overlap ---
    rv_trail = pan["rv_trail"]
    cals: dict[str, dict[str, Any]] = {}
    commons: dict[str, pd.DatetimeIndex] = {}
    realised_pts: "pd.Series[float]" = pd.Series(dtype=float)
    realised_u: "pd.Series[float]" = pd.Series(dtype=float)
    realised_prem: "pd.Series[float]" = pd.Series(dtype=float)
    for clock in ENTRY_CLOCKS:
        j = clocks.index(clock)
        arrays = {
            k: ch[k]
            for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
        }
        arrays["S_close"] = ch["S_close"]
        raw = p43._clock_book((j, arrays))
        book = {k: pd.Series(v, index=idx) for k, v in raw.items()}
        realised_units = book["hold"].dropna()
        w = windows[clock]
        era2020 = pd.DatetimeIndex(
            [
                d
                for d in w["dates"]
                if pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2024-12-31")
            ]
        )
        common = pd.DatetimeIndex(sorted(set(era2020) & set(realised_units.index)))
        commons[clock] = common
        cals[clock] = calibrate(ch, idx, clocks, clock, rv_trail, common)
        if clock == MAIN_ENTRY:
            realised_pts = (book["hold"] * book["entry"]).dropna().reindex(common)
            realised_u = realised_units.reindex(common)
            realised_prem = book["entry"].reindex(common)
    cal_main = cals[MAIN_ENTRY]

    # cross-check against the live seed ledger, which carries the same stamps
    led = pd.read_parquet(SEED_LEDGER)
    led13 = led[led["clock"] == MAIN_ENTRY].set_index("session")
    led_prem = led13["premium_mid"].reindex(cal_main["dates"]).to_numpy(float)
    led_iv = led13["implied_rem_var"].reindex(cal_main["dates"]).to_numpy(float)
    pos_cal = idx.get_indexer(cal_main["dates"])
    mine_prem = cal_main["premium_over_spot"] * ch["S"][:, j13][pos_cal]
    ok_led = np.isfinite(led_prem) & np.isfinite(led_iv)
    dev_led_prem = float(np.max(np.abs(led_prem[ok_led] - mine_prem[ok_led])))
    dev_led_iv = float(
        np.max(np.abs(led_iv[ok_led] - cal_main["implied_rem_var"][ok_led]))
    )
    print(
        f"\nseed ledger cross-check ({SEED_LEDGER.name}, clock {MAIN_ENTRY}): "
        f"{int(ok_led.sum())} of the {cal_main['n']} calibration sessions matched; "
        f"max |premium_mid - 43's entry| {dev_led_prem:.3e} pts, "
        f"max |implied_rem_var - tot^2| {dev_led_iv:.3e}"
    )

    print(
        "\n--- CALIBRATION.  One OLS fit per entry clock on the 2020-2024 chain "
        "overlap: ln(y) on ln(RV21), where RV21 is the trailing 21-session "
        "realized variance of the index lagged one session.  A DESCRIPTION of "
        "the market's premium-vs-regime curve, not a forecast.  APPROXIMATION "
        "A7: exp(fitted) is the conditional GEOMETRIC mean; "
        "retransform_factor_exp_half_sigma2 is the factor by which that sits "
        "below the conditional arithmetic mean under lognormal residuals, and "
        "no smearing correction is applied anywhere in this script."
    )
    cal_rows: list[dict[str, Any]] = []
    for clock in ENTRY_CLOCKS:
        cal = cals[clock]
        for target, fit, lo, hi in (
            ("ln(entry midpoint / spot)", cal["fit_premium"], "prem_lo", "prem_hi"),
            (
                "ln(implied remaining variance)",
                cal["fit_implied"],
                "impl_lo",
                "impl_hi",
            ),
        ):
            cal_rows.append(
                {
                    "block": "regression",
                    "entry_clock": clock,
                    "target": target,
                    "n": int(fit["n"]),
                    "intercept": fit["intercept"],
                    "slope": fit["slope"],
                    "r2": fit["r2"],
                    "resid_sd_ln": fit["resid_sd"],
                    "resid_mean_abs_ln": fit["resid_mean_abs"],
                    f"clip_lo_p{CLIP_LO_PCT:g}": cal[lo],
                    f"clip_hi_p{CLIP_HI_PCT:g}": cal[hi],
                    "retransform_factor_exp_half_sigma2": float(
                        np.exp(0.5 * fit["resid_sd"] ** 2)
                    ),
                    "rv21_overlap_min": cal["rv_min"],
                    "rv21_overlap_median": cal["rv_median"],
                    "rv21_overlap_max": cal["rv_max"],
                }
            )
    cal_tab = pd.DataFrame(cal_rows)
    print(cal_tab.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))

    # --------------------------------------------------------- the books ---
    jobs: list[dict[str, Any]] = []
    for rr in reps:
        clock = str(rr["entry_clock"])
        cal = cals[clock]
        w = windows[clock]
        rv = rv_trail.reindex(w["dates"]).to_numpy(float)
        cur = apply_curve(cal, rv)
        s_entry = float(rr["S_entry"])
        # the strikes: the representative day's gap as a fraction of spot,
        # applied to the entry spot of the replayed path.  The path starts at
        # the representative day's own spot, so this returns its strikes.
        k_c = s_entry * (float(rr["K_c"]) / s_entry)
        k_p = s_entry * (float(rr["K_p"]) / s_entry)
        dev_k = max(abs(k_c - float(rr["K_c"])), abs(k_p - float(rr["K_p"])))
        assert dev_k < STRIKE_TOL, (rr["rep_date"], dev_k)
        # the round trip is the identity up to one double-precision rounding;
        # the representative day's own strike is used so that the reproduced
        # frozen convention matches proposal 49 bit-for-bit.
        k_c, k_p = float(rr["K_c"]), float(rr["K_p"])
        prem_pts = cur["prem"] * s_entry  # SCALING: fitted ratio x rep-day spot
        impl_rep = float(rr["implied_rem_var_entry"])
        n_sess = len(w["dates"])
        eras = P49.era_of(w["dates"])
        halfspread_frac = 1.0 - float(rr["entry_bid_pts"]) / float(rr["entry_mid_pts"])
        credit: np.ndarray
        denom: np.ndarray
        iv_scale: np.ndarray
        impl_used: np.ndarray
        prem_clip: np.ndarray
        impl_clip: np.ndarray
        for conv in CONVENTIONS:
            if conv == C_FROZ:
                credit = np.full(n_sess, float(rr["entry_bid_pts"]))
                denom = np.full(n_sess, float(rr["entry_mid_pts"]))
                iv_scale = np.ones(n_sess)
                impl_used = np.full(n_sess, impl_rep)
                prem_clip = np.zeros(n_sess, int)
                impl_clip = np.zeros(n_sess, int)
                hs_frac = halfspread_frac
            else:
                credit = prem_pts
                denom = prem_pts
                prem_clip = cur["prem_clip"]
                if conv == C_BOTH:
                    iv_scale = cur["impl"] / impl_rep
                    impl_used = cur["impl"]
                    impl_clip = cur["impl_clip"]
                else:
                    iv_scale = np.ones(n_sess)
                    impl_used = np.full(n_sess, impl_rep)
                    impl_clip = np.zeros(n_sess, int)
                hs_frac = halfspread_frac
            jobs.append(
                {
                    "book": f"{rr['day_role']} | {clock} | {conv}",
                    "day_role": rr["day_role"],
                    "rep_date": rr["rep_date"],
                    "entry_clock": clock,
                    "convention": conv,
                    "j": int(rr["j"]),
                    "n_k": n_k,
                    "cum": w["cum"],
                    "ret": w["ret"],
                    "bars": w["bars"],
                    "dates": w["dates"],
                    "era": eras,
                    "s_entry": s_entry,
                    "s_rep": ch["S"][int(rr["i"])].copy(),
                    "tot_rep": ch["tot"][int(rr["i"])].copy(),
                    "k_c": k_c,
                    "k_p": k_p,
                    "credit_pts": credit,
                    "denom_pts": denom,
                    "iv_scale": iv_scale,
                    "implied_used": impl_used,
                    "prem_clip": prem_clip,
                    "impl_clip": impl_clip,
                    "rv_used": cur["rv_used"],
                    "rv_fallback": cur["rv_fallback"],
                    "halfspread_frac": hs_frac,
                }
            )
        print(
            f"repricing {rr['day_role']} @ {clock} ({rr['rep_date']}): entry spot "
            f"{s_entry:.2f}, strikes {k_c:.0f}/{k_p:.0f} (the representative "
            f"day's gap as a fraction of spot, round-tripped to within "
            f"{dev_k:.1e} index points of its own strikes and then snapped to "
            f"them), actual entry mid "
            f"{float(rr['entry_mid_pts']):.2f} pts; the fitted premium over the "
            f"{n_sess} replayed sessions has median {np.median(prem_pts):.2f}, mean "
            f"{prem_pts.mean():.2f}, min {prem_pts.min():.2f}, max "
            f"{prem_pts.max():.2f} pts; the low clip binds on "
            f"{int((cur['prem_clip'] < 0).sum())} sessions, the high clip on "
            f"{int((cur['prem_clip'] > 0).sum())}; implied clips "
            f"{int((cur['impl_clip'] < 0).sum())} / "
            f"{int((cur['impl_clip'] > 0).sum())}; "
            f"{int(cur['rv_fallback'].sum())} sessions take the median-RV21 "
            f"fallback for want of a complete trailing window"
        )

    # ------------------------------------------------------- the replay ----
    t0 = time.time()
    n_workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        parts = list(pool.map(_reprice_job, jobs))
    dist = pd.concat(parts, ignore_index=True)
    print(
        f"\nreplay: {len(jobs)} books x {len(TERMINALS)} terminals on {n_workers} "
        f"worker processes, {len(dist):,} session rows, {time.time() - t0:.1f} s"
    )

    # ---------------------------------------------------------- GATE 49 ----
    med = rep_tab[
        (rep_tab["day_role"] == MEDIAN_ROLE) & (rep_tab["entry_clock"] == MAIN_ENTRY)
    ].iloc[0]
    w_main = windows[MAIN_ENTRY]
    froz = dist[
        (dist["day_role"] == MEDIAN_ROLE)
        & (dist["entry_clock"] == MAIN_ENTRY)
        & (dist["terminal"] == TERMINALS[0])
        & (dist["convention"] == C_FROZ)
    ].sort_values("session")
    ref_summ = pd.read_csv(REF49 / "a_summary.csv")
    m49 = (
        (ref_summ["day_role"] == GATE49_BOOK[0])
        & (ref_summ["entry_clock"] == GATE49_BOOK[1])
        & (ref_summ["terminal"] == GATE49_BOOK[2])
        & (ref_summ["iv_mode"] == GATE49_BOOK[3])
    )
    r49 = ref_summ[m49].iloc[0]
    mine49 = P49.tail_stats(froz["pnl_pts"].to_numpy(float), "total_pts_")
    assert int(len(froz)) == int(r49["n_sessions"]), (len(froz), r49["n_sessions"])
    for stat in GATE49_STATS:
        dev = abs(float(mine49[stat]) - float(r49[stat]))
        assert dev < GATE49_TOL, (stat, dev)
        extra = ""
        if stat in GATE49_PRINTED:
            dp = abs(float(mine49[stat]) - GATE49_PRINTED[stat])
            assert dp < GATE49_PRINT_TOL, (stat, dp)
            extra = f"; its stated {GATE49_PRINTED[stat]:+.4f} (|dev| {dp:.1e})"
        print(
            f"GATE 49  frozen {GATE49_BOOK[0]} {med['rep_date']} {GATE49_BOOK[1]} "
            f"{GATE49_BOOK[2]}, {stat}: {float(mine49[stat]):+.9f} vs proposal 49's "
            f"{float(r49[stat]):+.9f} (|dev| {dev:.1e}, tolerance "
            f"{GATE49_TOL:.0e}){extra}"
        )
    w20 = P49.rolling_worst(froz["pnl_pts"].to_numpy(float), 20)
    dev20 = abs(w20 - GATE49_WORST20_PTS)
    assert dev20 < GATE49_WORST20_TOL, dev20
    print(
        f"GATE 49  worst 20-session cumulative loss {w20:+.6f} pts vs proposal 49's "
        f"stated {GATE49_WORST20_PTS:+.6f} (|dev| {dev20:.1e})"
    )
    common = commons[MAIN_ENTRY]
    common_s = set(common.strftime("%Y-%m-%d"))
    ov_froz = froz[froz["session"].isin(common_s)]
    dev_ov = abs(float(ov_froz["pnl_pts"].mean()) - GATE49_OVERLAP_REPLAY_MEAN)
    assert dev_ov < GATE49_TOL, dev_ov
    dev_rl = abs(float(realised_pts.mean()) - GATE49_OVERLAP_REALISED_MEAN)
    assert dev_rl < GATE49_TOL, dev_rl
    print(
        f"GATE 49  2020-2024 overlap ({len(common)} sessions): frozen replay mean "
        f"{float(ov_froz['pnl_pts'].mean()):+.12f} vs 49's "
        f"{GATE49_OVERLAP_REPLAY_MEAN:+.12f} (|dev| {dev_ov:.1e}); realised 13:30 "
        f"hold book mean {float(realised_pts.mean()):+.12f} vs 49's "
        f"{GATE49_OVERLAP_REALISED_MEAN:+.12f} (|dev| {dev_rl:.1e})"
    )

    # --------------------------------------------------------- GATE DEC ----
    j_med = int(med["j"])
    s_path_d, s_settle_d = P49.spot_from_path(
        float(med["S_entry"]), j_med, n_k, w_main["cum"], ch["S"][int(med["i"])]
    )
    tot_d = np.repeat(ch["tot"][int(med["i"])][None, :], len(w_main["dates"]), axis=0)
    direct = P49.price_book(
        p43,
        tot_d,
        s_path_d,
        s_settle_d,
        float(med["K_c"]),
        float(med["K_p"]),
        float(med["entry_bid_pts"]),
        float(med["entry_mid_pts"]),
        j_med,
    )
    order = np.argsort(pd.DatetimeIndex(w_main["dates"]).strftime("%Y-%m-%d"))
    dev_dec = float(
        np.max(np.abs(direct["pnl_hold_pts"][order] - froz["pnl_pts"].to_numpy(float)))
    )
    assert dev_dec == 0.0, dev_dec
    print(
        f"GATE DEC zero-credit + premium_d decomposition vs proposal 49's direct "
        f"fixed-credit call on all {len(froz)} sessions: max abs difference "
        f"{dev_dec:.1e}"
    )

    # -------------------------------------------------------- A. tables ----
    summ = summarise(dist)
    worst = worst_table(dist)
    eras_tab = era_table(dist)
    print(
        "\n--- A1. the loss distribution per contract, index points.  Credit = the "
        "fitted MIDPOINT premium for the two repriced conventions, the "
        "representative day's quoted BID for the reproduced frozen one "
        "(APPROXIMATION A1: the frozen column carries a half-spread the repriced "
        "ones do not; mean_halfspread_pts is the size of that gap)."
    )
    show = KEYS + [
        "n_sessions",
        "mean_premium_pts",
        "total_pts_mean",
        "total_pts_median",
        "total_pts_sd",
        "total_pts_p1",
        "total_pts_p0.5",
        "total_pts_p0.1",
        "total_pts_ES_below_p1",
        "total_pts_min",
        "mean_halfspread_pts",
        "mean_pts_net_of_rep_halfspread",
    ]
    print(summ[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\n--- A1b. the same, in dollars per contract at $100 an index point")
    show_d = KEYS + [
        "total_dollars_mean",
        "total_dollars_median",
        "total_dollars_sd",
        "total_dollars_p1",
        "total_dollars_p0.5",
        "total_dollars_p0.1",
        "total_dollars_ES_below_p1",
        "total_dollars_min",
    ]
    print(summ[show_d].to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    print(
        "\n--- A1c. the FOURTH COLUMN: the inverse-premium sizing convention, one "
        "unit of premium sold per day (contracts proportional to 1 / premium_d).  "
        "This is the per-unit distribution the deck's Sharpes live in."
    )
    show_u = KEYS + [
        "total_units_mean",
        "total_units_median",
        "total_units_sd",
        "total_units_p1",
        "total_units_p0.5",
        "total_units_p0.1",
        "total_units_ES_below_p1",
        "total_units_min",
    ]
    print(summ[show_u].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print(
        "\n--- A2. the two legs apart: the option leg is bounded below by "
        "-(settlement intrinsic) + premium_d; the FUTURES HEDGE IS UNBOUNDED."
    )
    show_l = KEYS + [
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
        f"\n--- A3. the worst {WORST_N} sessions of every book, with the path shape "
        "and the premium the regime would have paid.  trend ratio = |net move| / "
        "sum |bar moves|."
    )
    print(
        worst[
            KEYS
            + [
                "rank",
                "session",
                "pnl_pts",
                "pnl_units",
                "pnl_dollars",
                "option_pts",
                "hedge_pts",
                "premium_pts",
                "implied_rem_var",
                "path_net_logpct",
                "path_trend_ratio",
                "path_max_abs_bar_logpct",
                "path_max_abs_bar_stamp",
                "path_last_bar_logpct",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )

    print(
        f"\n--- A4. the worst-1% set, split by proposal 49's mechanical rule: 'last "
        f"bar' = the largest |bar| of the window is the {SETTLE_BAR} bar; 'earlier "
        f"bar' = the largest |bar| is earlier and the trend ratio is >= "
        f"{P49.TREND_BAR}; 'whipsaw' = earlier and below it."
    )
    show_w = KEYS + [
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
            KEYS
            + [
                "era",
                "n_sessions",
                "mean_premium_pts",
                "total_mean_pts",
                "total_sd_pts",
                "total_p1_pts",
                "total_min_pts",
                "option_mean_pts",
                "option_p1_pts",
                "hedge_mean_pts",
                "hedge_p1_pts",
                "hedge_min_pts",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )

    # ------------------------------------------------------ A6. validation --
    v_rows: list[dict[str, Any]] = [
        {
            "series": "realised 13:30 hold book (proposal 43, quoted bid fill)",
            "convention": "actual market",
            "n_sessions": int(len(common)),
            "mean_pts": float(realised_pts.mean()),
            "median_pts": float(realised_pts.median()),
            "sd_pts": float(realised_pts.std(ddof=1)),
            "p1_pts": float(np.percentile(realised_pts.to_numpy(float), 1.0)),
            "min_pts": float(realised_pts.min()),
            "mean_units": float(realised_u.mean()),
            "mean_premium_pts": float(realised_prem.mean()),
            "mean_gap_pts": 0.0,
            "sd_ratio": 1.0,
            "p1_gap_pts": 0.0,
        }
    ]
    for conv in CONVENTIONS:
        d = dist[
            (dist["day_role"] == MEDIAN_ROLE)
            & (dist["entry_clock"] == MAIN_ENTRY)
            & (dist["terminal"] == TERMINALS[0])
            & (dist["convention"] == conv)
        ]
        d = d[d["session"].isin(common_s)]
        v = d["pnl_pts"].to_numpy(float)
        v_rows.append(
            {
                "series": f"replay, {MEDIAN_ROLE} book, same sessions",
                "convention": conv,
                "n_sessions": int(len(d)),
                "mean_pts": float(v.mean()),
                "median_pts": float(np.median(v)),
                "sd_pts": float(v.std(ddof=1)),
                "p1_pts": float(np.percentile(v, 1.0)),
                "min_pts": float(v.min()),
                "mean_units": float(d["pnl_units"].mean()),
                "mean_premium_pts": float(d["premium_pts"].mean()),
                "mean_gap_pts": float(v.mean()) - float(realised_pts.mean()),
                "sd_ratio": float(v.std(ddof=1)) / float(realised_pts.std(ddof=1)),
                "p1_gap_pts": float(np.percentile(v, 1.0))
                - float(np.percentile(realised_pts.to_numpy(float), 1.0)),
            }
        )
    v_tab = pd.DataFrame(v_rows)
    for name, got, ref in (
        ("realised mean", float(realised_pts.mean()), GATE49_OVERLAP_REALISED_MEAN),
        (
            "realised sd",
            float(realised_pts.std(ddof=1)),
            GATE49_OVERLAP_REALISED_SD,
        ),
        (
            "realised p1",
            float(np.percentile(realised_pts.to_numpy(float), 1.0)),
            GATE49_OVERLAP_REALISED_P1,
        ),
        (
            "realised mean premium",
            float(realised_prem.mean()),
            GATE49_OVERLAP_REALISED_PREM,
        ),
    ):
        assert abs(got - ref) < GATE49_TOL, (name, got, ref)
    print(
        "\n--- A6. VALIDATION: the 2020-2024 overlap against the realised 13:30 "
        f"hold book ({GATE49_OVERLAP_REALISED_MEAN:+.4f} / sd "
        f"{GATE49_OVERLAP_REALISED_SD:.4f} / p1 {GATE49_OVERLAP_REALISED_P1:+.4f}, "
        "reproduced above to 1e-9).  mean_gap_pts and p1_gap_pts are replay minus "
        "realised; sd_ratio is replay over realised."
    )
    print(v_tab.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ------------------------------------- the calibration's decile residual --
    prem_own = pd.Series(ch["entry"][:, j13], index=idx).reindex(common)
    prem_act = pd.Series(
        ch["entry"][:, j13] / ch["S"][:, j13] * float(med["S_entry"]), index=idx
    ).reindex(common)
    rv_ov = rv_trail.reindex(common)
    fit_ov = apply_curve(cal_main, rv_ov.to_numpy(float))
    prem_fit = fit_ov["prem"] * float(med["S_entry"])
    dec = pd.qcut(rv_ov, N_VOL_DECILES, labels=False, duplicates="drop").to_numpy()
    for dv in range(N_VOL_DECILES):
        m = dec == dv
        if not m.any():
            continue
        cal_rows.append(
            {
                "block": "decile",
                "entry_clock": MAIN_ENTRY,
                "target": f"volatility decile {dv + 1} of RV21, 2020-2024 overlap",
                "n": int(m.sum()),
                "rv21_mean": float(rv_ov.to_numpy(float)[m].mean()),
                "actual_premium_pts": float(prem_act.to_numpy(float)[m].mean()),
                "actual_premium_own_spot_pts": float(
                    prem_own.to_numpy(float)[m].mean()
                ),
                "repriced_premium_pts": float(prem_fit[m].mean()),
                "repriced_over_actual": float(prem_fit[m].mean())
                / float(prem_act.to_numpy(float)[m].mean()),
                "mean_ln_residual": float(
                    np.mean(np.log(prem_act.to_numpy(float)[m]) - np.log(prem_fit[m]))
                ),
                "frozen_premium_pts": float(med["entry_mid_pts"]),
                "frozen_over_actual": float(med["entry_mid_pts"])
                / float(prem_act.to_numpy(float)[m].mean()),
            }
        )
    cal_rows.append(
        {
            "block": "decile",
            "entry_clock": MAIN_ENTRY,
            "target": "whole 2020-2024 overlap",
            "n": int(len(common)),
            "rv21_mean": float(rv_ov.mean()),
            "actual_premium_pts": float(prem_act.mean()),
            "actual_premium_own_spot_pts": float(prem_own.mean()),
            "repriced_premium_pts": float(prem_fit.mean()),
            "repriced_over_actual": float(prem_fit.mean()) / float(prem_act.mean()),
            "mean_ln_residual": float(
                np.mean(np.log(prem_act.to_numpy(float)) - np.log(prem_fit))
            ),
            "frozen_premium_pts": float(med["entry_mid_pts"]),
            "frozen_over_actual": float(med["entry_mid_pts"]) / float(prem_act.mean()),
        }
    )
    cal_tab = pd.DataFrame(cal_rows)
    print(
        "\n--- A7. the calibration's residual by regime: per decile of the trailing "
        "21-session realized variance over the 2020-2024 overlap, the premium the "
        "curve charges against the premium the market actually charged, both in "
        f"index points on the {MEDIAN_ROLE} book's spot "
        f"({float(med['S_entry']):.2f})."
    )
    print(
        cal_tab[cal_tab["block"] == "decile"][
            [
                "target",
                "n",
                "rv21_mean",
                "actual_premium_own_spot_pts",
                "actual_premium_pts",
                "repriced_premium_pts",
                "repriced_over_actual",
                "mean_ln_residual",
                "frozen_premium_pts",
                "frozen_over_actual",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.6g}")
    )

    # ------------------------------------------------------ B. sizing ------
    prim = dist[
        (dist["day_role"] == MEDIAN_ROLE)
        & (dist["entry_clock"] == MAIN_ENTRY)
        & (dist["terminal"] == TERMINALS[0])
        & (dist["convention"] == PRIMARY)
    ].sort_values("session")
    v_pts = prim["pnl_pts"].to_numpy(float)
    v_units = prim["pnl_units"].to_numpy(float)
    prem_d = prim["premium_pts"].to_numpy(float)
    prem_bar = float(prem_d.mean())
    v_inv = v_units * prem_bar  # the same day, sized one premium unit a day
    inv_label = (
        f"inverse premium (1/premium_d, deployed at the mean premium "
        f"{prem_bar:.2f} pts a day)"
    )
    sizing_series: dict[str, np.ndarray] = {
        "per contract (one straddle a day)": v_pts,
        inv_label: v_inv,
    }
    b_rows: list[dict[str, Any]] = []
    for conv_name, ser in sizing_series.items():
        loss = float(np.percentile(ser, SIZING_TAIL_PCT))
        for src_name, loss_pts in (
            (f"replay {PRIMARY}, {SIZING_TAIL_PCT:g}th percentile (1-in-200)", loss),
            (P45_STRESS_LABEL, P45_STRESS_PTS),
        ):
            for budget in CAPITAL_BUDGETS:
                cap = abs(loss_pts) / budget
                row: dict[str, Any] = {
                    "sizing_convention": conv_name,
                    "loss_source": src_name,
                    "loss_pts": loss_pts,
                    "loss_dollars": loss_pts * SPX_MULT,
                    "budget_frac_of_capital": budget,
                    "capital_pts": cap,
                    "capital_dollars": cap * SPX_MULT,
                    "mean_pts": float(ser.mean()),
                    "sd_pts": float(ser.std(ddof=1)),
                    "ann_mean_pts": float(ser.mean()) * asl.PERIODS_PER_YEAR,
                    "ann_mean_dollars": float(ser.mean())
                    * asl.PERIODS_PER_YEAR
                    * SPX_MULT,
                    "ann_return_on_capital": float(ser.mean())
                    * asl.PERIODS_PER_YEAR
                    / cap,
                }
                for win in RUIN_WINDOWS:
                    wl = P49.rolling_worst(ser, win)
                    row[f"worst{win}_pts"] = wl
                    row[f"worst{win}_dollars"] = wl * SPX_MULT
                    row[f"worst{win}_frac_capital"] = wl / cap
                b_rows.append(row)
    b_siz = pd.DataFrame(b_rows)
    print(
        f"\n--- B. sizing from the REPRICED distribution, {MEDIAN_ROLE} book "
        f"({med['rep_date']}), {MAIN_ENTRY} entry, hold to cash settlement, "
        f"convention '{PRIMARY}'.  Capital = |loss| / budget.  The annual mean is "
        f"the replay's own mean x {asl.PERIODS_PER_YEAR:.0f}.  worstN is the "
        "minimum N-session rolling sum of the series in chronological order "
        "('ruin' as a fraction of that capital).  The stress rows are proposal "
        "45's own last-bar -5% naked straddle cell for comparison."
    )
    print(b_siz.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ----------------------------------------- the deleveraging candidate --
    rv_prim = prim["rv21_trail"].to_numpy(float)
    q_half = expanding_pct(rv_prim, DELEVER_HALF_PCT, DELEVER_MIN_OBS)
    q_zero = expanding_pct(rv_prim, DELEVER_ZERO_PCT, DELEVER_MIN_OBS)
    wgt: np.ndarray = np.ones(len(rv_prim))
    wgt = np.where(np.isfinite(q_half) & (rv_prim > q_half), 0.5, wgt)
    wgt = np.where(np.isfinite(q_zero) & (rv_prim > q_zero), 0.0, wgt)
    v_rule = wgt * v_pts
    base_cap = abs(float(np.percentile(v_pts, SIZING_TAIL_PCT))) / CAPITAL_BUDGETS[0]
    d_rows: list[dict[str, Any]] = []
    for name, ser in (("no rule (full size)", v_pts), ("deleveraging rule", v_rule)):
        ruled = name == "deleveraging rule"
        row = {
            "series": name,
            "n_sessions": int(len(ser)),
            "n_full_size": int((wgt == 1.0).sum()) if ruled else int(len(ser)),
            "n_half_size": int((wgt == 0.5).sum()) if ruled else 0,
            "n_zero_size": int((wgt == 0.0).sum()) if ruled else 0,
            "mean_pts": float(ser.mean()),
            "median_pts": float(np.median(ser)),
            "sd_pts": float(ser.std(ddof=1)),
            "p1_pts": float(np.percentile(ser, 1.0)),
            "p0.5_pts": float(np.percentile(ser, SIZING_TAIL_PCT)),
            "min_pts": float(ser.min()),
            "ann_mean_dollars": float(ser.mean()) * asl.PERIODS_PER_YEAR * SPX_MULT,
        }
        for win in RUIN_WINDOWS:
            wl = P49.rolling_worst(ser, win)
            row[f"worst{win}_pts"] = wl
            row[f"worst{win}_dollars"] = wl * SPX_MULT
            row[f"worst{win}_frac_of_no_rule_capital"] = wl / base_cap
        d_rows.append(row)
    b_del = pd.DataFrame(d_rows)
    print(
        f"\n--- B2. a DESIGN CANDIDATE, evaluated on the replay and adopted "
        f"nowhere: halve the size when RV21 exceeds its expanding lagged "
        f"{DELEVER_HALF_PCT:g}th percentile, zero it above the "
        f"{DELEVER_ZERO_PCT:g}th.  The percentile is taken over STRICTLY EARLIER "
        f"replayed sessions with a minimum of {DELEVER_MIN_OBS} of them; before "
        f"that the size is full.  'frac_of_no_rule_capital' divides by the "
        f"no-rule 5%-budget capital ({base_cap:,.2f} pts) so the two rows are on "
        "one scale."
    )
    print(b_del.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

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

    bars_main = w_main["bars"]
    k_w1 = max(1, int(round(WORST_FRAC * len(prim))))
    worst1_sessions = pd.DatetimeIndex(prim.nsmallest(k_w1, "pnl_pts")["session"])
    worst10_sessions = pd.DatetimeIndex(prim.nsmallest(WORST_N, "pnl_pts")["session"])
    r_all = pan["ret"][bars_main].reindex(w_main["dates"]).to_numpy(float)
    q_all = pan["rv"][bars_main].reindex(w_main["dates"]).to_numpy(float)
    ratio_all = np.where(r_all**2 > 0.0, q_all / r_all**2, np.nan)
    bench = "a bar whose 30 one-minute returns were all equal has ratio 1/30 = 0.0333"
    for label, sel in (
        ("all replayed sessions", pd.DatetimeIndex(w_main["dates"])),
        (f"worst-1% sessions ({k_w1})", worst1_sessions),
        (f"worst-{WORST_N} sessions", worst10_sessions),
    ):
        m = np.isin(pd.DatetimeIndex(w_main["dates"]).to_numpy(), sel.to_numpy())
        v = ratio_all[m][np.isfinite(ratio_all[m])]
        cadd(
            "C1 minute-level jumps inside bars",
            f"sumret2 / sumret^2 over the {len(bars_main)} window bars, {label}",
            int(v.size),
            float(np.median(v)),
            "ratio (median)",
            f"sumret2 is the sum of the squared one-minute returns inside the bar, "
            f"sumret^2 the square of the bar's net move; {bench}, and the excess "
            "over that is path the 30-minute replay never sees",
        )
        q_sel = q_all[m]
        r2_sel = r_all[m] ** 2
        both = np.isfinite(q_sel) & np.isfinite(r2_sel)
        cadd(
            "C1 minute-level jumps inside bars",
            f"sumret2 / sumret^2 over the {len(bars_main)} window bars, {label}",
            int(both.sum()),
            float(q_sel[both].sum() / r2_sel[both].sum()),
            "ratio (pooled sums)",
            "immune to the near-zero denominators that make a per-bar mean meaningless",
        )
        big = np.abs(r_all[m]).argmax(axis=1)
        rows_m = np.arange(int(m.sum()))
        qb = q_all[m][rows_m, big]
        r2b = r_all[m][rows_m, big] ** 2
        okb = np.isfinite(qb) & np.isfinite(r2b)
        cadd(
            "C1 minute-level jumps inside bars",
            f"the same ratio on each session's LARGEST window bar, {label}",
            int(okb.sum()),
            float(qb[okb].sum() / r2b[okb].sum()),
            "ratio (pooled sums)",
            f"the bar that drives the loss; {bench}",
        )
    nob = pan["numobs"][bars_main].reindex(w_main["dates"]).to_numpy(float)
    cadd(
        "C1 minute-level jumps inside bars",
        "window bars carrying fewer than 30 one-minute observations",
        int(np.isfinite(nob).sum()),
        int((np.isfinite(nob) & (nob < 30.0)).sum()),
        "bars",
        "a short bar hides proportionally more of its own path",
    )

    last_bar = pan["ret"][SETTLE_BAR].reindex(w_main["dates"]).to_numpy(float)
    cadd(
        "C2 settlement print vs the last bar",
        f"the panel's {SETTLE_BAR} bar, |log return|, replayed sessions",
        int(np.isfinite(last_bar).sum()),
        100.0 * float(np.nanmean(np.abs(last_bar))),
        "mean |move|, %",
        "the panel has no separate settlement print: the replay settles on the "
        f"close of the {SETTLE_BAR} bar",
    )
    cadd(
        "C2 settlement print vs the last bar",
        f"the panel's {SETTLE_BAR} bar, |log return|, replayed sessions",
        int(np.isfinite(last_bar).sum()),
        100.0 * float(np.nanpercentile(np.abs(last_bar), 99.0)),
        "99th percentile |move|, %",
        "same",
    )
    gap = ch["S_close"] / ch["S"][:, -1] - 1.0
    for stat, val in (
        ("mean |move|, %", 100.0 * float(np.nanmean(np.abs(gap)))),
        (
            "99th percentile |move|, %",
            100.0 * float(np.nanpercentile(np.abs(gap), 99.0)),
        ),
        ("max |move|, %", 100.0 * float(np.nanmax(np.abs(gap)))),
    ):
        cadd(
            "C2 settlement print vs the last bar",
            "the live book's own 15:30 spot -> GSPC close move, chain sessions",
            int(np.isfinite(gap).sum()),
            val,
            stat,
            "proposal 43's settlement is the GSPC close; the replay's is the "
            f"panel's {SETTLE_BAR} bar close, a different series",
        )

    rv_replay = rv_trail.reindex(w_main["dates"]).to_numpy(float)
    d_main = pd.DatetimeIndex(w_main["dates"])
    for label, lo, hi in (
        ("the 2020-2024 calibration overlap", "2020-01-01", "2024-12-31"),
        ("2008-2009", "2008-01-01", "2009-12-31"),
        ("1998-2019", "1998-01-01", "2019-12-31"),
        ("the whole replay", "1900-01-01", "2100-01-01"),
    ):
        m = (d_main >= pd.Timestamp(lo)) & (d_main <= pd.Timestamp(hi))
        sub = rv_replay[m]
        sub = sub[np.isfinite(sub)]
        for stat, val in (
            ("RV21 min", float(sub.min())),
            ("RV21 median", float(np.median(sub))),
            ("RV21 max", float(sub.max())),
        ):
            cadd(
                "C3 extrapolating the 2020-2024 premium curve",
                f"trailing {RV_WINDOW}-session realized variance, {label}",
                int(sub.size),
                val,
                stat,
                "the curve is fitted only on the overlap's RV21 support "
                f"[{cal_main['rv_min']:.3e}, {cal_main['rv_max']:.3e}]; anything "
                "outside it is extrapolation",
            )
    below = int(np.nansum(rv_replay < cal_main["rv_min"]))
    above = int(np.nansum(rv_replay > cal_main["rv_max"]))
    m08 = (d_main >= pd.Timestamp("2008-01-01")) & (
        d_main <= pd.Timestamp("2009-12-31")
    )
    rv08_max = float(np.nanmax(rv_replay[np.asarray(m08)]))
    cadd(
        "C3 extrapolating the 2020-2024 premium curve",
        "replayed sessions whose RV21 lies outside the overlap's support",
        int(np.isfinite(rv_replay).sum()),
        below + above,
        f"sessions ({below} below, {above} above)",
        f"the 2008-2009 maximum RV21 is {rv08_max:.4e}, the overlap's maximum "
        f"{cal_main['rv_max']:.4e}, a ratio of "
        f"{rv08_max / cal_main['rv_max']:.3f}",
    )
    cur_main = apply_curve(cal_main, rv_replay)
    for name, arr, lo_v, hi_v in (
        (
            "premium / spot",
            cur_main["prem_clip"],
            cal_main["prem_lo"],
            cal_main["prem_hi"],
        ),
        (
            "implied remaining variance",
            cur_main["impl_clip"],
            cal_main["impl_lo"],
            cal_main["impl_hi"],
        ),
    ):
        cadd(
            "C3 extrapolating the 2020-2024 premium curve",
            f"replayed sessions on which the {name} clip binds",
            int(arr.size),
            f"{int((arr < 0).sum())} at the {CLIP_LO_PCT:g}th-percentile floor, "
            f"{int((arr > 0).sum())} at the {CLIP_HI_PCT:g}th-percentile cap",
            "sessions",
            f"floor {lo_v:.6g}, cap {hi_v:.6g}, both from the overlap's own "
            "distribution",
        )
    cadd(
        "C3 extrapolating the 2020-2024 premium curve",
        "replayed sessions taking the median-RV21 fallback (no full trailing window)",
        int(len(d_main)),
        int(cur_main["rv_fallback"].sum()),
        "sessions",
        f"APPROXIMATION A4: they are priced at the curve's value at the overlap's "
        f"median RV21 {cal_main['rv_median']:.4e}",
    )

    for label, mask, note in (
        (
            f"dated before {LISTING_MWF_FROM}",
            d_main < pd.Timestamp(LISTING_MWF_FROM),
            "ASSUMED: no SPXW same-day expiry outside Friday",
        ),
        (
            f"dated {LISTING_MWF_FROM}..{LISTING_DAILY_FROM}, on a Mon/Wed/Fri",
            (d_main >= pd.Timestamp(LISTING_MWF_FROM))
            & (d_main < pd.Timestamp(LISTING_DAILY_FROM))
            & np.isin(d_main.weekday, MWF),
            "ASSUMED: M-W-F same-day expiry listed",
        ),
        (
            f"dated {LISTING_MWF_FROM}..{LISTING_DAILY_FROM}, on a Tue/Thu",
            (d_main >= pd.Timestamp(LISTING_MWF_FROM))
            & (d_main < pd.Timestamp(LISTING_DAILY_FROM))
            & ~np.isin(d_main.weekday, MWF),
            "ASSUMED: NO same-day expiry; the replay prices a book that did not exist",
        ),
        (
            f"dated {LISTING_DAILY_FROM} onwards",
            d_main >= pd.Timestamp(LISTING_DAILY_FROM),
            "ASSUMED: daily same-day expiry listed",
        ),
    ):
        cadd(
            "C4 the product did not exist",
            f"replayed sessions {label}",
            int(len(d_main)),
            int(np.asarray(mask).sum()),
            "sessions",
            f"{note}; the milestones {LISTING_MWF_FROM} (M-W-F) and "
            f"{LISTING_DAILY_FROM} (daily) are stated assumptions used only to "
            "count sessions, not data",
        )
    c_blind = pd.DataFrame(c_rows)
    print(
        "\n--- C. what remains unseen (numbers only; every row is a measurement, "
        "not a correction)"
    )
    print(c_blind.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # ------------------------------------------------------------ writes ---
    dist_cols = [c for c in dist.columns if c != "book"]
    dist[dist_cols].to_csv(
        OUT / "a_distribution.csv.gz",
        index=False,
        float_format="%.6g",
        compression="gzip",
    )
    summ.to_csv(OUT / "a_summary.csv", index=False)
    worst.to_csv(OUT / "a_worst.csv", index=False)
    eras_tab.to_csv(OUT / "a_by_era.csv", index=False)
    cal_tab.to_csv(OUT / "a_calibration.csv", index=False)
    v_tab.to_csv(OUT / "a_vs_realised.csv", index=False)
    b_siz.to_csv(OUT / "b_sizing.csv", index=False)
    b_del.to_csv(OUT / "b_delever.csv", index=False)
    c_blind.to_csv(OUT / "c_blindspots.csv", index=False)
    print(
        f"\nwrote {OUT}:\n"
        f"  a_distribution.csv.gz  {len(dist):,} rows x {len(dist_cols)} columns "
        f"({len(jobs)} books x {len(TERMINALS)} terminals x "
        f"{len(w_main['dates'])} sessions; the 'book' key is dropped, it is the "
        "three key columns joined)\n"
        f"  a_summary.csv          {len(summ)} rows (printed in full as A1-A4)\n"
        f"  a_worst.csv            {len(worst)} rows (printed in full as A3)\n"
        f"  a_by_era.csv           {len(eras_tab)} rows (printed in full as A5)\n"
        f"  a_calibration.csv      {len(cal_tab)} rows (printed as CALIBRATION "
        "and A7)\n"
        f"  a_vs_realised.csv      {len(v_tab)} rows (printed in full as A6)\n"
        f"  b_sizing.csv           {len(b_siz)} rows (printed in full as B)\n"
        f"  b_delever.csv          {len(b_del)} rows (printed in full as B2)\n"
        f"  c_blindspots.csv       {len(c_blind)} rows (printed in full as C)\n"
        f"total {time.time() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
