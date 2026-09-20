"""47 - can the 15:30 signal be FORECAST at 11:00, and is that gate worth taking?

The book of record is proposal 41/42's in-sample tape: every session sells the
nearest-OTM SPX 0DTE straddle at 11:00 ET AT THE QUOTED BID, delta-hedges the
package on the vendor spot every 30 minutes, and at 15:30 either buys the
straddle back AT THE QUOTED ASK (the hedge stopping there) or holds both legs
into the 16:00 cash settlement (the hedge running to the settlement print).
Returns are one unit a day; the PRIMARY unit here is index points per contract
and the premium-unit column (the 11:00 midpoint entry premium) sits beside it.

Proposal 37's surviving rule decides that terminal at 15:30 with 15:30
information: exit if s(15:30) > 0, where

    s(15:30) = rv_hat_f(15:30) - slice(15:30),
    slice(15:30) = IV_hourly_used(15:30)^2 x 0.5,

the remaining-share weight being one by construction at the last clock.  This
proposal asks the question one stage earlier: at 11:00, four and a half hours
before that signal exists, can it be FORECAST -- and if the day's forecast of
the 15:30 signal says "sell", is shorting only those days better than shorting
every day?

The rule, pre-registered:

    at 11:00 short the nearest-OTM straddle ONLY IF a forecast MADE AT 11:00
    of the 15:30 signal is negative (s_hat < 0: the forecast of the 15:30-16:00
    bar's realized variance sits below the implied slice for that bar as priced
    at 11:00);  delta-hedge every 30 minutes as the book;  at 15:30 read the
    ACTUAL signal s(15:30) and, if it is positive, buy both legs back at the
    15:30 quoted ask (the hedge stops), otherwise hold to cash settlement.

So the rule is [11:00 gate by s_hat] x [proposal 37's exit-if-s>0 at the
quotes].  A gated-out day is FLAT and earns exactly zero, so every cell is
scored on the same calendar days and the means are directly comparable.

s_hat is built causally, three ways, and every one of them is reported.

  implied side   the 11:00 re-inverted remaining-window implied variance times
                 the expanding lagged (minimum 63 sessions) share of the
                 15:30-16:00 bar within the 11:00-to-close window -- section
                 5b's diurnal machinery (notebooks/_write_0dte_intraday_T_nb.py,
                 proposal 30's ``panel_profile``), with the share taken of a
                 SPECIFIC later bar instead of the stamp's own;
  route A        persistence x profile: rv_hat_f(11:00) times the profile share
                 of the 15:30 bar over the profile share of the 11:00 bar, both
                 expanding and lagged;
  route B        a causal direct regression, proposal 36's R3 shape moved to a
                 fixed target bar:
                     log y(15:30) ~ a + b log rv_hat_f(11:00)
                                      + d log RV(10:00 -> 11:00, same day)
                                      + e log y(15:30, previous session)
                 fitted on strictly earlier sessions (expanding, minimum 63)
                 and then rescaled by proposal 36's lagged QLIKE-optimal
                 multiplicative correction;
  route C        the direct multi-horizon campaign's h = 9 forecast issued at
                 11:00 -- IF per-(date, stamp, horizon) forecasts exist on
                 disk.  The script checks honestly and reports what it finds.

Controls, every one on the same calendar days:
  (i)   the 11:00 gate alone: gate, then always hold to settlement;
  (ii)  proposal 37's rule alone: always short at 11:00, exit if s(15:30) > 0 --
        this is the second stage by itself and it is the rule the two-stage
        cell has to beat;
  (iii) the reverse two-stage rule: short only if s_hat > 0, and exit at 15:30
        only if s(15:30) < 0 (both legs flipped);
  (iv)  the book: always hold (the null), with always-exit carried alongside;
  (v)   the oracle: gate at 11:00 on the ACTUAL 15:30 sign.  Non-causal, a
        ceiling, never adoptable, and labelled as such everywhere.

Gates, in order, all asserted before a single new number is computed:
  GATE 0  proposal 41's in-sample tape: n = 865 days 2020-01-03..2024-04-30,
          book (always exit at the quoted ask) crossed Sharpe 2.2525467 whole /
          2.5131322 era, always-hold 3.582391 whole / 3.323335 era, and
          proposal 41's own unconditional-ride rows and anatomy.
  GATE P  the vectorised block placebo draws proposal 30's masks, draw for draw.
  GATE S  the 15:30 implied slice is the hedge tape's own 15:30 total variance
          and the standalone's iv_hourly_used^2 x 0.5, and all eight forecasts
          carry a signal on all 865 days.
  GATE 37 the PARENT reproduction: proposal 37's "exit if s > 0" at the QUOTED
          ask, per forecast, whole and era, reproduces proposal 42's own
          cross-check on this 865-day set (ridge 3.252054 / 3.349368).
          Proposal 37's published table is on ITS 797 covered days and with the
          standalone's pre-fix exit signal (ridge 3.2980 / 3.3494, always hold
          3.4100 / 3.3233, always exit 2.2509 / 2.5131); that day set is NOT
          the one used here and those numbers are printed as context only.
  GATE 11 the 11:00 window-matched signal reproduces proposal 30's own
          ``signals_1100`` on every day whose 11:00 vendor implied volatility
          was not re-inverted.
  GATE A  the route-A identity: s_hat_A is exactly the 11:00 signal times a
          positive profile ratio, so its SIGN is the 11:00 signal's sign.
  GATE B  proposal 36's own two guards on the route-B fit: the Gram
          accumulation against a direct least-squares solve, and the
          truncated-history causality canary.

Significance, per cell: the paired daily difference against always-hold AND
against control (ii) must both have a circular-block bootstrap CI (B = 2000,
block 21, seed 0) excluding zero on the positive side, the cell must beat both
out of sample, and the run-length-preserving placebo on WHICH DAYS ARE GATED
OUT must give p < 0.05.  The family verdict is the walk-forward's: proposal
30's purged, embargoed choice over (route, forecast) with always-hold as the
null must beat always-hold out of sample.

The forecast panels end 2024-04-30, which is the last day of this 865-day set,
so EVERY number here is in sample: nothing can be scored on the 413 unseen
sessions proposals 43/45 use.

The eight forecasts are independent once the tape, the implied slices and the
diurnal profile exist, so each one's routes, cells, skill tables, bootstraps
and placebo draws are computed in its own worker process; the gates, the
shared references and the walk-forward stay in the parent.  Each placebo cell
draws from its own deterministic child stream, so the p-values do not depend
on how many workers run.

Run:  python writeup/intraday_proposals/47_forecast_of_1530_signal.py
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT / "notebooks"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import atm_straddle_lib as asl  # noqa: E402


def _load_module(path: Path, name: str) -> ModuleType:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p41 = _load_module(HERE / "41_otm_leg_ride.py", "p47_p41")
p42 = _load_module(HERE / "42_otm_ride_conditional.py", "p47_p42")
p36 = _load_module(HERE / "36_remaining_window_honest.py", "p47_p36")
p30 = _load_module(HERE / "30_skip_day_rule.py", "p47_p30")

OUT = p41.HOLD / "proposals" / "47"

ENTRY = p41.ENTRY
CLOSE = p41.CLOSE
ERA_START = p41.ERA_START
ANN = p41.ANN

#: The remaining-session hours the 15:30 stamp carries: one 30-minute bar.
H_REM_1530 = p42.H_REM_1530
#: The rule's only number, and it is not a free parameter: the gate shorts when
#: the FORECAST of the 15:30 signal sits below the implied slice, and the
#: second stage exits when the ACTUAL 15:30 signal sits above it.
THRESHOLD = 0.0

#: The three causal constructions of s_hat.  Route C is scored only if the
#: per-(date, stamp, horizon) forecasts it needs are found on disk.
ROUTES: tuple[str, ...] = (
    "A_persistence_profile",
    "B_direct_regression",
    "C_direct_multihorizon",
)
#: The rules every route carries.  "book_hold" / "book_exit" are the shared
#: references and are built once in the parent, not per route.
RULES: tuple[str, ...] = (
    "full_two_stage",
    "control_i_gate_only_hold",
    "control_ii_p37_rule_only",
    "control_iii_reverse",
    "oracle_gate_on_actual_1530",
)
#: The rule the gate compares against, and the null the walk-forward uses.
CONTROL_II = "control_ii_p37_rule_only"
NULL_RULE = "always hold (cash settlement)"
#: The non-causal ceiling; reported everywhere, adoptable nowhere.
ORACLE = "oracle_gate_on_actual_1530"
#: The two samples every cell is scored on, and the one the verdict reads.
SAMPLES: tuple[str, ...] = ("whole", "era")
VERDICT_SAMPLE = "era"
#: The two units every cell is reported in; the first is the primary.
UNITS: tuple[str, ...] = ("per_contract_pts", "premium_units")
PRIMARY_UNIT = "per_contract_pts"

#: Proposal 41's bootstrap, reused verbatim.
BOOT_B = p41.BOOT_B
BOOT_BLOCK = p41.BOOT_BLOCK
BOOT_SEED = p41.BOOT_SEED

#: Proposal 30's walk-forward and placebo, reused verbatim.
WARMUP_SESSIONS = p30.WARMUP_SESSIONS
EMBARGO_DAYS = p30.EMBARGO_DAYS
N_PLACEBO = p30.N_PLACEBO
PLACEBO_SEED = p30.PLACEBO_SEED
PLACEBO_ALPHA = p30.PLACEBO_ALPHA

#: Proposal 36's guards on the route-B fit, reused verbatim.
WARMUP_FIT = p36.WARMUP
OLS_TOL = p36.OLS_TOL
OLS_CHECK_POINTS = p36.OLS_CHECK_POINTS

#: One worker per forecast, capped so the pool never outnumbers the work.
N_WORKERS = min(4, len(asl.MODEL_ORDER))

#: GATE 0 targets, from proposal 41/42's own runs.
GATE_N = p41.GATE_N
GATE_BOOK_WHOLE = p41.GATE_BOOK_WHOLE
GATE_BOOK_ERA = p41.GATE_BOOK_ERA
#: Always hold to cash settlement, whole and era (proposal 42's rules.csv).
GATE_HOLD_WHOLE = 3.582391
GATE_HOLD_ERA = 3.323335
#: GATE 37: proposal 37's rule at the QUOTED ask on THIS 865-day set, per
#: forecast, whole and era -- proposal 42's own reference block, which is the
#: parent this proposal's second stage is a restriction of.
GATE_P37_QUOTED: dict[str, tuple[float, float]] = {
    "a0": (3.177297, 3.184322),
    "blk2": (3.252054, 3.349368),
    "blk2_inc": (3.410018, 3.404536),
    "lgbm": (3.475084, 3.506313),
    "xgb": (3.468754, 3.540179),
    "lasso_t": (3.483981, 3.486390),
    "lasso_f": (3.212084, 3.298684),
    "enet": (3.365692, 3.377699),
}
#: Proposal 37's OWN published table, on ITS 797 covered days and with the
#: standalone's pre-fix exit signal.  Printed as context, never asserted: it
#: is a different day set and a different implied-volatility column.
GATE_P37_PUBLISHED: dict[str, tuple[float, float]] = {
    "ridge exit if s>0": (3.2980, 3.3494),
    "always hold": (3.4100, 3.3233),
    "always exit": (2.2509, 2.5131),
}
GATE_P37_PUBLISHED_N = 797
#: A number published to six decimals is gated here.
GATE_TOL_6DP = 1e-6
#: The route-A identity and the 11:00 reproduction are float paths.
EXACT_TOL = 1e-12
#: The route-A identity is asserted relative to the signal's own scale.
IDENT_RTOL = 1e-10

#: Where route C's per-(date, stamp, horizon) forecasts would have to live.
ROUTE_C_PATHS: tuple[str, ...] = (
    "results/direct_mh",
    "results/multihorizon",
    "results/direct_mh/*.npz",
    "results/multihorizon/*.npz",
    "results/direct_mh/**/shard_*of*.npz",
)
#: The horizon route C would need: 11:00 -> the 15:30 bar is nine bars ahead.
ROUTE_C_HORIZON = 9

#: The columns cells.csv prints.
CELL_COLS = [
    "sample",
    "unit",
    "route",
    "forecast",
    "rule",
    "n_days",
    "n_traded",
    "frac_gated_out",
    "frac_exit_1530",
    "mean",
    "sd",
    "Sharpe_ann",
    "t",
    "t_hac",
    "MaxDD",
    "worst_day",
    "worst_date",
    "d_hold_mean",
    "d_hold_t_hac",
    "d_hold_ci_lo",
    "d_hold_ci_hi",
    "d_hold_positive",
    "d_ctrl2_mean",
    "d_ctrl2_t_hac",
    "d_ctrl2_ci_lo",
    "d_ctrl2_ci_hi",
    "d_ctrl2_positive",
]
#: The columns skill.csv prints.
SKILL_COLS = [
    "sample",
    "route",
    "forecast",
    "n",
    "n_no_shat",
    "n_sell_sell",
    "n_sell_buy",
    "n_buy_sell",
    "n_buy_buy",
    "hit_rate",
    "frac_predicted_sell",
    "frac_actual_sell",
    "precision_sell",
    "corr_pearson_shat_s1100",
    "corr_spearman_shat_s1100",
    "corr_pearson_shat_s1530",
    "corr_spearman_shat_s1530",
    "corr_pearson_s1100_s1530",
]


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 320, "display.max_columns", 120):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"wrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# ------------------------------------------------------------- statistics ---
def stats_row(r: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """Calendar-day statistics of one daily return series (proposal 42's)."""
    return p42.stats_row(r, dates)


def diff_block(
    a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex, tag: str
) -> dict[str, Any]:
    """a - b: mean, HAC t and the circular-block bootstrap CI (proposal 41's)."""
    row = p41.paired_row(
        "",
        pd.Series(np.asarray(a, float), index=dates),
        pd.Series(np.asarray(b, float), index=dates),
    )
    return {
        f"d_{tag}_mean": row["mean_diff"],
        f"d_{tag}_t_hac": row["t_hac"],
        f"d_{tag}_t_lag": row["t_lag"],
        f"d_{tag}_ci_lo": row["ci_lo"],
        f"d_{tag}_ci_hi": row["ci_hi"],
        f"d_{tag}_positive": bool(row["excludes_zero_positive"]),
        f"d_{tag}_Sharpe": row["d_Sharpe"],
    }


def nan_diff(tag: str) -> dict[str, Any]:
    """The paired-difference block, empty: a reference row has no counterpart."""
    return {
        f"d_{tag}_mean": float("nan"),
        f"d_{tag}_t_hac": float("nan"),
        f"d_{tag}_t_lag": 0,
        f"d_{tag}_ci_lo": float("nan"),
        f"d_{tag}_ci_hi": float("nan"),
        f"d_{tag}_positive": False,
        f"d_{tag}_Sharpe": float("nan"),
    }


def corr_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson and Spearman correlation on the cells both series carry."""
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan"), float("nan")
    x = pd.Series(a[ok])
    y = pd.Series(b[ok])
    return float(x.corr(y)), float(x.corr(y, method="spearman"))


def placebo_rng(i_tag: int, i_route: int, i_rule: int, i_smp: int) -> Any:
    """One deterministic stream per cell, so workers cannot reorder the draws."""
    return np.random.default_rng([PLACEBO_SEED, i_tag, i_route, i_rule, i_smp])


# ----------------------------------------------------------- the profile ----
def window_profile(prof: pd.DataFrame, clocks: list[str], j_entry: int) -> pd.DataFrame:
    """Expanding lagged share of EVERY bar within the 11:00-to-close window.

    Proposal 30's ``panel_profile`` builds, for each stamp, that bar's share of
    its OWN remaining window.  The gate needs the share of a bar LATER than the
    stamp it is priced at, so the same per-clock realized-variance profile is
    renormalised once, on the fixed 11:00-to-close window, and averaged the
    same way: an expanding mean with the repo's 63-session minimum, shifted one
    session, so the value used on day d is a function of days before d only.
    The 11:00 column of this frame is proposal 30's own 11:00 weight by
    construction, and GATE 11 asserts it.
    """
    sub = clocks[j_entry:]
    rem = prof[sub].sum(axis=1)
    pi = prof[sub].div(rem.replace(0.0, np.nan), axis=0)
    return pi.expanding(min_periods=WARMUP_SESSIONS).mean().shift(1)


def forecast_clock_grid(tag: str, clocks: list[str]) -> pd.DataFrame:
    """A forecast panel's next-bar rv_hat on the (session, trade clock) grid.

    The panels are bar-END labelled, so the row carrying the forecast ISSUED at
    clock ``c`` is stamped ``c + 30 minutes``; the stamp is pushed back half an
    hour and pivoted, exactly as ``make_dh_causal_standalone_tex.main`` and
    proposal 42's ``forecast_1530`` do it for the single 15:30 column.
    """
    panel = asl.load_yhat_panel(asl.yhat_paths(ROOT)[tag])
    y = panel[["t", "rv_hat"]].copy()
    y["t"] = pd.to_datetime(y["t"], utc=True) - pd.Timedelta(minutes=30)
    et = pd.to_datetime(y["t"], utc=True).dt.tz_convert("America/New_York")
    y["pdate"] = et.dt.normalize().dt.tz_localize(None)
    y["phhmm"] = et.dt.strftime("%H:%M")
    return (
        y.pivot_table(index="pdate", columns="phhmm", values="rv_hat", aggfunc="mean")
        .reindex(columns=clocks)
        .sort_index()
    )


# ---------------------------------------------------------- route C check ---
def route_c_availability() -> dict[str, Any]:
    """Look, honestly, for per-(date, stamp, horizon) direct forecasts on disk.

    The ledger says only summaries were ever brought back from the cluster.
    This function does not take that on trust: it lists every candidate path,
    reports the shape and the columns of everything it finds, and decides route
    C is usable only if some file carries a per-session, per-stamp, per-horizon
    forecast.  Nothing is fabricated when it does not.
    """
    print("\nROUTE C - looking for per-(date, stamp, horizon) direct forecasts")
    found: list[str] = []
    usable = False
    for rel in ROUTE_C_PATHS:
        if any(ch in rel for ch in "*?"):
            hits = sorted(ROOT.glob(rel))
        else:
            base = ROOT / rel
            hits = sorted(base.glob("*")) if base.is_dir() else []
        if not hits:
            print(f"  {rel:<40s} nothing")
            continue
        for h in hits:
            rp = h.relative_to(ROOT).as_posix()
            found.append(rp)
            if h.suffix == ".csv":
                head = pd.read_csv(h, nrows=5)
                cols = list(head.columns)
                per_day = any(c in cols for c in ("date", "day", "t", "session"))
                print(f"  {rp:<40s} csv, columns {cols}; per-session rows: {per_day}")
                usable = usable or (per_day and "k" in cols)
            elif h.suffix == ".npz":
                with np.load(h, allow_pickle=True) as z:
                    keys = sorted(z.files)
                print(f"  {rp:<40s} npz, keys {keys}")
                usable = usable or ("yhat" in keys and "t" in keys)
            else:
                print(f"  {rp:<40s} {h.suffix or 'no suffix'}, not a forecast dump")
    if not usable:
        print(
            f"  every file found is an ERA/HORIZON AGGREGATE (pooled QLIKE, "
            f"by-hour QLIKE, coefficient tables); no file carries a forecast "
            f"indexed by (date, stamp, horizon), so the h = {ROUTE_C_HORIZON} "
            f"forecast issued at {ENTRY} cannot be read and route C is NOT "
            f"scored.  The campaign's shard_*of*.npz dumps are written cluster "
            f"side (experiments/direct_multihorizon.py writes them under its "
            f"--out) and none were brought back into this repo."
        )
    return {"usable": usable, "found": found}


# -------------------------------------------------------------- the rules ---
def rule_returns(
    rule: str,
    s_hat: np.ndarray,
    s_1530: np.ndarray,
    r_exit: np.ndarray,
    r_hold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One rule's daily return, its gated-out mask and its 15:30 exit mask.

    A gated-out day is FLAT and earns exactly zero.  A day whose s_hat is not
    finite TRADES -- proposal 30's own default, where a day without a signal
    sells -- and is counted separately; GATE S guarantees s(15:30) is finite on
    every day, so the second stage never falls back.
    """
    has = np.isfinite(s_hat)
    exit_rule = s_1530 > THRESHOLD
    exit_rev = s_1530 < THRESHOLD
    r37 = np.where(exit_rule, r_exit, r_hold)
    r37_rev = np.where(exit_rev, r_exit, r_hold)
    zero = np.zeros_like(r_hold)
    if rule == "full_two_stage":
        out = has & (s_hat >= THRESHOLD)
        return np.where(out, zero, r37), out, (~out) & exit_rule
    if rule == "control_i_gate_only_hold":
        out = has & (s_hat >= THRESHOLD)
        return np.where(out, zero, r_hold), out, np.zeros_like(out)
    if rule == "control_ii_p37_rule_only":
        out = np.zeros(len(r_hold), dtype=bool)
        return r37, out, exit_rule
    if rule == "control_iii_reverse":
        out = has & (s_hat <= THRESHOLD)
        return np.where(out, zero, r37_rev), out, (~out) & exit_rev
    if rule == ORACLE:
        out = s_1530 >= THRESHOLD
        return np.where(out, zero, r_hold), out, np.zeros_like(out)
    raise AssertionError(rule)


def skill_row(
    s_hat: np.ndarray, s_1530: np.ndarray, s_1100: np.ndarray
) -> dict[str, Any]:
    """The 2x2 of sign(s_hat) at 11:00 against sign(s(15:30)) on the same day."""
    has = np.isfinite(s_hat) & np.isfinite(s_1530)
    n = int(has.sum())
    pred_sell = has & (s_hat < THRESHOLD)
    act_sell = has & (s_1530 < THRESHOLD)
    n_ss = int((pred_sell & act_sell).sum())
    n_sb = int((pred_sell & ~act_sell & has).sum())
    n_bs = int((~pred_sell & act_sell & has).sum())
    n_bb = int((~pred_sell & ~act_sell & has).sum())
    n_pred = n_ss + n_sb
    p_ps, sp_ps = corr_pair(s_hat, s_1100)
    p_p5, sp_p5 = corr_pair(s_hat, s_1530)
    p_15, _ = corr_pair(s_1100, s_1530)
    return {
        "n": n,
        "n_no_shat": int((~np.isfinite(s_hat)).sum()),
        "n_sell_sell": n_ss,
        "n_sell_buy": n_sb,
        "n_buy_sell": n_bs,
        "n_buy_buy": n_bb,
        "hit_rate": float((n_ss + n_bb) / n) if n else float("nan"),
        "frac_predicted_sell": float(n_pred / n) if n else float("nan"),
        "frac_actual_sell": float((n_ss + n_bs) / n) if n else float("nan"),
        "precision_sell": float(n_ss / n_pred) if n_pred else float("nan"),
        "corr_pearson_shat_s1100": p_ps,
        "corr_spearman_shat_s1100": sp_ps,
        "corr_pearson_shat_s1530": p_p5,
        "corr_spearman_shat_s1530": sp_p5,
        "corr_pearson_s1100_s1530": p_15,
    }


# -------------------------------------------------------------- the worker --
def forecast_cell(job: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    """Everything one forecast owns, computed in its own process.

    The parent hands over the tape (day index, the two terminal return series
    in both units, the actual 15:30 signal, the implied slices and the diurnal
    profile) and the panel-session realized tape route B fits on; the worker
    loads that forecast's own panel, forms its 11:00 forecast, builds both
    causal routes to s_hat, and returns its cells, its skill tables, its
    bootstraps and its placebo draws.  Nothing here reads a shared mutable
    object, so the eight jobs are independent.
    """
    tag = str(job["tag"])
    i_tag = int(job["i_tag"])
    dates = pd.DatetimeIndex(job["dates"])
    clocks = list(job["clocks"])
    panel_days = pd.DatetimeIndex(job["panel_days"])
    y_1530_panel = np.asarray(job["y_1530_panel"], float)
    sofar_panel = np.asarray(job["sofar_panel"], float)
    sl_1100 = np.asarray(job["sl_1100"], float)
    sl_1530_at_1100 = np.asarray(job["sl_1530_at_1100"], float)
    ratio = np.asarray(job["ratio"], float)
    s_1530 = np.asarray(job["s_1530"], float)
    r_exit = {u: np.asarray(job["r_exit"][u], float) for u in UNITS}
    r_hold = {u: np.asarray(job["r_hold"][u], float) for u in UNITS}
    era = np.asarray(job["era"], bool)
    routes = list(job["routes"])

    grid = forecast_clock_grid(tag, clocks)
    rvh_book = grid.reindex(index=dates)
    rv_1100 = rvh_book[ENTRY].to_numpy(float)
    rv_1530_book = rvh_book[CLOSE].to_numpy(float)
    s_1100 = rv_1100 - sl_1100

    # ---- route A: persistence x profile -----------------------------------
    rv_hat: dict[str, np.ndarray] = {}
    s_hat: dict[str, np.ndarray] = {}
    rv_hat["A_persistence_profile"] = rv_1100 * ratio
    s_hat["A_persistence_profile"] = rv_hat["A_persistence_profile"] - sl_1530_at_1100

    # ---- route B: the causal direct regression ----------------------------
    rvh_panel = grid.reindex(index=panel_days)[ENTRY].to_numpy(float)
    prev = np.concatenate([[np.nan], y_1530_panel[:-1]])
    pos = (y_1530_panel > 0.0) & (rvh_panel > 0.0) & (sofar_panel > 0.0) & (prev > 0.0)
    idx = np.nonzero(pos)[0]
    ly = np.log(y_1530_panel[idx])
    des = np.column_stack(
        [
            np.ones(idx.size),
            np.log(rvh_panel[idx]),
            np.log(sofar_panel[idx]),
            np.log(prev[idx]),
        ]
    )
    pred = p36.expanding_ols_predict(des, ly, WARMUP_FIT)
    ols_dev = p36.check_expanding_ols(des, ly, WARMUP_FIT, pred, f"{tag}_B")
    scaled = p36.causal_scale(y_1530_panel[idx], np.exp(pred))
    leak = p36.causality_canary(des, ly, y_1530_panel[idx], scaled, f"{tag}_B")
    rvb = pd.Series(scaled, index=panel_days[idx]).reindex(dates).to_numpy(float)
    rv_hat["B_direct_regression"] = rvb
    s_hat["B_direct_regression"] = rvb - sl_1530_at_1100

    cells: list[dict[str, Any]] = []
    skill: list[dict[str, Any]] = []
    plac: list[dict[str, Any]] = []
    oos_mask = np.asarray(job["oos_mask"], bool)
    smp = {"whole": np.ones(len(dates), bool), "era": era}
    base = {
        u: rule_returns(CONTROL_II, s_1530, s_1530, r_exit[u], r_hold[u])[0]
        for u in UNITS
    }
    for i_route, route in enumerate(routes):
        sh = s_hat[route]
        for i_smp, s_name in enumerate(SAMPLES):
            m = smp[s_name]
            d_m = dates[m]
            skill.append(
                {"sample": s_name, "route": route, "forecast": tag}
                | skill_row(sh[m], s_1530[m], s_1100[m])
            )
            for i_rule, rule in enumerate(RULES):
                r_u: dict[str, np.ndarray] = {}
                gated = np.zeros(len(dates), bool)
                exited = np.zeros(len(dates), bool)
                for u in UNITS:
                    r_u[u], gated, exited = rule_returns(
                        rule, sh, s_1530, r_exit[u], r_hold[u]
                    )
                    cells.append(
                        {
                            "sample": s_name,
                            "unit": u,
                            "route": route,
                            "forecast": tag,
                            "rule": rule,
                            "n_traded": int((~gated[m]).sum()),
                            "frac_gated_out": float(gated[m].mean()),
                            "frac_exit_1530": (
                                float(exited[m].sum() / int((~gated[m]).sum()))
                                if int((~gated[m]).sum())
                                else float("nan")
                            ),
                            **stats_row(r_u[u][m], d_m),
                            **diff_block(r_u[u][m], r_hold[u][m], d_m, "hold"),
                            **diff_block(r_u[u][m], base[u][m], d_m, "ctrl2"),
                        }
                    )
                if rule == CONTROL_II:
                    continue
                active = base
                if rule in ("control_i_gate_only_hold", ORACLE):
                    active = r_hold
                if rule == "control_iii_reverse":
                    active = {
                        u: np.where(s_1530 < THRESHOLD, r_exit[u], r_hold[u])
                        for u in UNITS
                    }
                for i_set, (set_name, sel) in enumerate(
                    (("full", m), ("oos", m & oos_mask))
                ):
                    rng = placebo_rng(i_tag, i_route, i_rule, i_smp * 2 + i_set)
                    plac.append(
                        {
                            "sample": s_name,
                            "day_set": set_name,
                            "route": route,
                            "forecast": tag,
                            "rule": rule,
                            "n_days": int(sel.sum()),
                            "n_gated_out": int(gated[sel].sum()),
                            **p42.placebo_stats(
                                np.zeros(int(sel.sum())),
                                active[PRIMARY_UNIT][sel],
                                gated[sel],
                                rng,
                            ),
                        }
                    )
    return {
        "tag": tag,
        "s_1100": s_1100,
        "rv_1100": rv_1100,
        "rv_1530_book": rv_1530_book,
        "s_hat": s_hat,
        "rv_hat": rv_hat,
        "ratio": ratio,
        "cells": cells,
        "skill": skill,
        "placebo": plac,
        "ols_dev": float(ols_dev),
        "leak": float(leak),
        "n_fit": int(idx.size),
    }


# ------------------------------------------------------ the walk-forward ----
def walk_forward(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proposal 30's purged, embargoed choice; only the out-of-sample day scores.

    For day i the training window is days [0, i - EMBARGO_DAYS), the candidate
    with the best in-window calendar-day Sharpe is applied to day i, candidate
    0 is the null (always hold) and ``np.argmax`` keeps the first maximum so a
    tie goes to the null, and days whose training window is shorter than
    WARMUP_SESSIONS are not scored.
    """
    n = series.shape[1]
    curves = np.vstack([p30.expanding_sharpe(row) for row in series])
    curves = np.where(np.isfinite(curves), curves, -np.inf)
    scored = np.zeros(n, dtype=bool)
    pick = np.full(n, -1, dtype=int)
    r = np.full(n, np.nan)
    for i in range(n):
        k = i - EMBARGO_DAYS
        if k < WARMUP_SESSIONS:
            continue
        c = int(np.argmax(curves[:, k]))
        pick[i] = c
        r[i] = series[c, i]
        scored[i] = True
    return scored, r, pick


# ----------------------------------------------------------------- gates ----
def gate_hold(book: pd.DataFrame) -> None:
    """Always-hold, whole and era, before anything new is computed."""
    era = np.asarray(book.index >= pd.Timestamp(ERA_START), dtype=bool)
    got_w = p41.sharpe(book["r_hold"])
    got_e = p41.sharpe(book.loc[era, "r_hold"])
    assert abs(got_w - GATE_HOLD_WHOLE) < GATE_TOL_6DP, got_w
    assert abs(got_e - GATE_HOLD_ERA) < GATE_TOL_6DP, got_e
    print(
        f"GATE 0  always hold to cash settlement Sharpe_ann {got_w:.7f} "
        f"(reference {GATE_HOLD_WHOLE}) whole, {got_e:.7f} "
        f"(reference {GATE_HOLD_ERA}) era"
    )


def gate_p37(
    book: pd.DataFrame, sig: dict[str, np.ndarray], dates: pd.DatetimeIndex
) -> None:
    """GATE 37 - the parent: proposal 37's rule at the quoted ask, per forecast."""
    era = np.asarray(dates >= pd.Timestamp(ERA_START), dtype=bool)
    r_exit = book["r_book"].to_numpy(float)
    r_hold = book["r_hold"].to_numpy(float)
    print(
        f"\nGATE 37  proposal 37's rule (always short at {ENTRY}, exit at "
        f"{CLOSE} iff s > 0, buy-back at the QUOTED ask) on this "
        f"{len(dates)}-day set, per forecast, premium units"
    )
    for tag in asl.MODEL_ORDER:
        want_w, want_e = GATE_P37_QUOTED[tag]
        r = np.where(sig[tag] > THRESHOLD, r_exit, r_hold)
        got_w = p41.sharpe(r)
        got_e = p41.sharpe(r[era])
        assert abs(got_w - want_w) < GATE_TOL_6DP, (tag, got_w, want_w)
        assert abs(got_e - want_e) < GATE_TOL_6DP, (tag, got_e, want_e)
        print(
            f"  {tag:<9s} exits {int((sig[tag] > THRESHOLD).sum()):>3d} of "
            f"{len(dates)} days; Sharpe_ann {got_w:.6f} (reference {want_w}) "
            f"whole, {got_e:.6f} (reference {want_e}) era  OK"
        )
    print(
        f"  CONTEXT, not asserted: proposal 37's own published table is on its "
        f"{GATE_P37_PUBLISHED_N} covered days and with the standalone's pre-fix "
        f"exit signal -- "
        + "; ".join(
            f"{k} {v[0]:.4f} whole / {v[1]:.4f} era"
            for k, v in GATE_P37_PUBLISHED.items()
        )
        + ".  That day set is not the one used here."
    )
    print("GATE 37 PASSED")


def gate_1100(
    s_1100: dict[str, np.ndarray], panel: pd.DataFrame, dates: pd.DatetimeIndex
) -> None:
    """GATE 11 - the 11:00 signal reproduces proposal 30's own ``signals_1100``."""
    pkg = pd.read_parquet(p41.cache_path())
    ref = p30.signals_1100(pkg, dates)
    at = (
        panel.loc[panel["hhmm"] == ENTRY, ["date", "iv_hourly", "iv_hourly_used"]]
        .drop_duplicates("date")
        .set_index("date")
        .reindex(dates)
    )
    same = np.isclose(
        at["iv_hourly"].to_numpy(float),
        at["iv_hourly_used"].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    )
    print(
        f"\nGATE 11  the {ENTRY} window-matched signal against proposal 30's "
        f"own; {int((~same).sum())} of {len(dates)} days carry a re-inverted "
        f"{ENTRY} implied volatility and are excluded from the equality (this "
        f"proposal reads the standalone's iv_hourly_used everywhere)"
    )
    for tag in asl.MODEL_ORDER:
        a = s_1100[tag]
        b = ref[f"s_{tag}"].to_numpy(float)
        ok = same & np.isfinite(a) & np.isfinite(b)
        dev = float(np.max(np.abs(a[ok] - b[ok])))
        assert (np.isfinite(a) == np.isfinite(b)).all(), tag
        assert dev < EXACT_TOL, (tag, dev)
        print(f"  {tag:<9s} matches on {int(ok.sum())} days to {dev:.3e}  OK")
    print("GATE 11 PASSED")


def gate_route_a(
    res: dict[str, Any], s_1100: dict[str, np.ndarray], ratio: np.ndarray
) -> None:
    """GATE A - route A is the 11:00 signal times a positive profile ratio."""
    print(
        f"\nGATE A  the route-A identity s_hat_A = (share of the {CLOSE} bar / "
        f"share of the {ENTRY} bar) x s({ENTRY}); the ratio is positive, so "
        f"route A's SIGN is the plain {ENTRY} signal's sign, day for day"
    )
    assert bool(np.all(ratio[np.isfinite(ratio)] > 0.0)), "a profile ratio is not > 0"
    for tag in asl.MODEL_ORDER:
        a = res[tag]["s_hat"]["A_persistence_profile"]
        b = ratio * s_1100[tag]
        ok = np.isfinite(a) & np.isfinite(b)
        scale = float(np.max(np.abs(b[ok])))
        dev = float(np.max(np.abs(a[ok] - b[ok])))
        assert dev <= IDENT_RTOL * scale, (tag, dev, scale)
        sa = np.sign(a[ok])
        sb = np.sign(s_1100[tag][ok])
        assert np.array_equal(sa, sb), tag
        print(
            f"  {tag:<9s} identity to {dev:.3e} on {int(ok.sum())} days "
            f"(signal scale {scale:.3e}); signs identical on every day  OK"
        )
    print(
        f"  ratio {CLOSE} / {ENTRY}: min {float(np.nanmin(ratio)):.4f}, median "
        f"{float(np.nanmedian(ratio)):.4f}, max {float(np.nanmax(ratio)):.4f}"
    )
    print("GATE A PASSED")


def gate_route_b(res: dict[str, Any]) -> None:
    """GATE B - proposal 36's own two guards on the expanding fit."""
    print(
        f"\nGATE B  the route-B fit: expanding OLS on strictly earlier sessions "
        f"(minimum {WARMUP_FIT}) then proposal 36's lagged QLIKE rescale "
        f"(minimum {WARMUP_FIT} again)"
    )
    for tag in asl.MODEL_ORDER:
        r = res[tag]
        assert r["ols_dev"] < OLS_TOL, (tag, r["ols_dev"])
        print(
            f"  {tag:<9s} fit frame {r['n_fit']} sessions; the Gram "
            f"accumulation matches a direct least-squares solve to "
            f"{r['ols_dev']:.3e} at {OLS_CHECK_POINTS} sampled positions "
            f"(bar {OLS_TOL:.0e}); truncated-history canary moved "
            f"{r['leak']:.3e} (bar {EXACT_TOL:.0e})  OK"
        )
    print("GATE B PASSED")


# ------------------------------------------------------------------ main ----
def main() -> None:  # noqa: PLR0912, PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)

    # ---------------------------------------- the tape, gated first --------
    mk = p41._standalone()
    g = p41.insample_grids(mk)
    quotes = p41.exit_leg_quotes(g["panel"], g)
    dates = pd.DatetimeIndex(g["dates"])
    clocks = list(g["clocks"])
    j_entry = int(g["j"])
    j15 = int(g["j15"])
    book = p41.ride_frame(
        dates,
        g["K_c"],
        g["K_p"],
        g["entry"],
        g["bid"],
        g["Sg"][:, j15],
        g["S_close"],
        quotes["ask_c"].to_numpy(float),
        quotes["ask_p"].to_numpy(float),
        g["tot"][:, j15],
        g["tot15_mid"],
        g["hedge_exit"],
        g["hedge_hold"],
    )
    p42.gate_zero(book)
    gate_hold(book)
    p42.gate_placebo(len(dates))
    era = np.asarray(dates >= pd.Timestamp(ERA_START), dtype=bool)
    print(
        f"\nIN SAMPLE ONLY: the forecast panels end {dates.max().date()}, which "
        f"is the last day of this {len(dates)}-day set "
        f"({dates.min().date()} .. {dates.max().date()}, {int(era.sum())} in "
        f"the daily-0DTE era from {ERA_START}).  Nothing in this proposal can "
        f"be scored on the 413 unseen sessions proposals 43/45 use, because no "
        f"forecast exists there."
    )

    # ------------------------------ the actual 15:30 signal, gated next ----
    sl_1530 = g["IVg"][:, j15] ** 2 * H_REM_1530
    sig = {tag: p42.forecast_1530(tag, dates) - sl_1530 for tag in asl.MODEL_ORDER}
    p42.gate_signal(g, mk, sl_1530, sig)
    gate_p37(book, sig, dates)

    # ------------------------------------- the diurnal profile at 11:00 ----
    base_panel = asl.load_yhat_panel(asl.yhat_paths(ROOT)["blk2"]).set_index("t")
    prof, w_own = p30.panel_profile(base_panel, clocks)
    wbar = window_profile(prof, clocks, j_entry)
    dev_w = float(
        np.max(
            np.abs(
                w_own[ENTRY].reindex(dates).to_numpy(float)
                - wbar[ENTRY].reindex(dates).to_numpy(float)
            )
        )
    )
    assert dev_w < EXACT_TOL, dev_w
    h_rem_1100 = float((len(clocks) - j_entry) * 0.5)
    w_1100 = wbar[ENTRY].reindex(dates).to_numpy(float)
    w_1530 = wbar[CLOSE].reindex(dates).to_numpy(float)
    ratio = w_1530 / w_1100
    sl_1100 = g["IVg"][:, j_entry] ** 2 * h_rem_1100 * w_1100
    sl_1530_at_1100 = g["IVg"][:, j_entry] ** 2 * h_rem_1100 * w_1530
    print(
        f"\nthe {ENTRY} diurnal profile, expanding lagged mean with a "
        f"{WARMUP_SESSIONS}-session minimum over {len(prof)} panel sessions "
        f"{prof.index.min().date()} .. {prof.index.max().date()}: the {ENTRY} "
        f"bar's share of the {ENTRY}-to-close window has median "
        f"{float(np.nanmedian(w_1100)):.6f}, the {CLOSE} bar's "
        f"{float(np.nanmedian(w_1530)):.6f}, ratio median "
        f"{float(np.nanmedian(ratio)):.6f}; the {ENTRY} column reproduces "
        f"proposal 30's own weight to {dev_w:.3e}"
    )
    print(
        f"the implied slice for the {CLOSE} bar priced at {ENTRY} "
        f"(IV_hourly_used({ENTRY})^2 x {h_rem_1100} x that share) has median "
        f"{float(np.nanmedian(sl_1530_at_1100)):.6e}; the ACTUAL {CLOSE} slice "
        f"has median {float(np.nanmedian(sl_1530)):.6e}"
    )

    # --------------------------------------------- route C, checked next ---
    c_info = route_c_availability()
    routes = [r for r in ROUTES if not r.startswith("C_") or c_info["usable"]]
    n_cells_pre = len(ROUTES) * len(asl.MODEL_ORDER)
    n_cells = len(routes) * len(asl.MODEL_ORDER)
    print(
        f"\ncells: {len(ROUTES)} routes x {len(asl.MODEL_ORDER)} forecasts = "
        f"{n_cells_pre} pre-registered (a 5% level expects "
        f"{n_cells_pre * PLACEBO_ALPHA:.1f} false adoptions); route C is "
        f"{'scored' if c_info['usable'] else 'NOT scored'}, so "
        f"{len(routes)} x {len(asl.MODEL_ORDER)} = {n_cells} cells are actually "
        f"tried and a 5% level expects {n_cells * PLACEBO_ALPHA:.1f} there"
    )

    # ------------------------------- the eight forecasts, one per process --
    y_1530_panel = prof[CLOSE].to_numpy(float)
    sofar_panel = prof[clocks[:j_entry]].sum(axis=1).to_numpy(float)
    r_exit = {
        "per_contract_pts": book["r_book_pts"].to_numpy(float),
        "premium_units": book["r_book"].to_numpy(float),
    }
    r_hold = {
        "per_contract_pts": book["r_hold_pts"].to_numpy(float),
        "premium_units": book["r_hold"].to_numpy(float),
    }
    oos_mask = np.arange(len(dates)) >= WARMUP_SESSIONS + EMBARGO_DAYS
    jobs = [
        {
            "tag": tag,
            "i_tag": i,
            "dates": dates,
            "clocks": clocks,
            "oos_mask": oos_mask,
            "panel_days": prof.index,
            "y_1530_panel": y_1530_panel,
            "sofar_panel": sofar_panel,
            "sl_1100": sl_1100,
            "sl_1530_at_1100": sl_1530_at_1100,
            "ratio": ratio,
            "s_1530": sig[tag],
            "r_exit": r_exit,
            "r_hold": r_hold,
            "era": era,
            "routes": routes,
        }
        for i, tag in enumerate(asl.MODEL_ORDER)
    ]
    print(
        f"\nfanning {len(jobs)} forecasts over {N_WORKERS} worker processes; "
        f"each worker loads its own panel, builds routes "
        f"{', '.join(r[0] for r in routes)}, and computes its "
        f"{len(RULES)} rules x {len(SAMPLES)} samples x {len(UNITS)} units of "
        f"cells with their bootstraps and its {N_PLACEBO}-draw placebos; the "
        f"gates, the shared references and the walk-forward stay in the parent"
    )
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(forecast_cell, jobs))
    res = {r["tag"]: r for r in results}
    assert set(res) == set(asl.MODEL_ORDER), sorted(res)
    s_1100 = {tag: np.asarray(res[tag]["s_1100"], float) for tag in asl.MODEL_ORDER}
    for tag in asl.MODEL_ORDER:
        dev = float(
            np.nanmax(
                np.abs(
                    np.asarray(res[tag]["rv_1530_book"], float)
                    - p42.forecast_1530(tag, dates)
                )
            )
        )
        assert dev < EXACT_TOL, (tag, dev)
    print(
        f"\nthe worker's clock grid reproduces proposal 42's own {CLOSE} "
        f"forecast for all {len(asl.MODEL_ORDER)} models to {EXACT_TOL:.0e}"
    )
    gate_1100(s_1100, g["panel"], dates)
    gate_route_a(res, s_1100, ratio)
    gate_route_b(res)

    # ------------------------------------------------------- skill.csv -----
    skill = pd.DataFrame([r for t in asl.MODEL_ORDER for r in res[t]["skill"]])
    write(skill, "skill.csv", "the 2x2 of sign(s_hat) against sign(s(15:30))")
    for s_name in SAMPLES:
        show(
            skill[skill["sample"] == s_name].sort_values(["route", "forecast"]),
            f"skill.csv  |  sample {s_name}  |  sell = s < 0 (short / stay short)",
            SKILL_COLS,
        )

    # ------------------------------------------------------- cells.csv -----
    ref_rows: list[dict[str, Any]] = []
    for s_name in SAMPLES:
        m = np.ones(len(dates), bool) if s_name == "whole" else era
        d_m = dates[m]
        for name, series in (
            (NULL_RULE, r_hold),
            ("always exit at 15:30 (the quoted-ask book)", r_exit),
        ):
            for u in UNITS:
                ref_rows.append(
                    {
                        "sample": s_name,
                        "unit": u,
                        "route": "-",
                        "forecast": "-",
                        "rule": name,
                        "n_traded": int(m.sum()),
                        "frac_gated_out": 0.0,
                        "frac_exit_1530": 0.0 if name == NULL_RULE else 1.0,
                        **stats_row(series[u][m], d_m),
                        **diff_block(series[u][m], r_hold[u][m], d_m, "hold"),
                        **nan_diff("ctrl2"),
                    }
                )
    cells = pd.DataFrame(
        ref_rows + [r for t in asl.MODEL_ORDER for r in res[t]["cells"]]
    )
    write(cells, "cells.csv", "every rule, both samples, both units")
    for u in UNITS:
        for s_name in SAMPLES:
            sub = cells[(cells["unit"] == u) & (cells["sample"] == s_name)]
            show(
                sub.sort_values(["route", "rule", "forecast"]),
                f"cells.csv  |  unit {u}  |  sample {s_name}",
                CELL_COLS,
            )

    # ----------------------------------------------------- placebo.csv -----
    plac = pd.DataFrame([r for t in asl.MODEL_ORDER for r in res[t]["placebo"]])
    write(
        plac,
        "placebo.csv",
        f"{N_PLACEBO} run-length-preserving shuffles of the GATED-OUT day mask",
    )
    print(
        f"\nplacebo: each draw gates out exactly as many days, in runs of "
        f"exactly the same lengths, and only WHICH days change; the active "
        f"days keep the rule's own return in {PRIMARY_UNIT}; "
        f"p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo), "
        f"seed {PLACEBO_SEED}"
    )
    for s_name in SAMPLES:
        for day_set in ("full", "oos"):
            sub = plac[(plac["sample"] == s_name) & (plac["day_set"] == day_set)]
            show(
                sub.sort_values(["route", "rule", "forecast"]),
                f"placebo.csv  |  sample {s_name}  |  day set {day_set}",
            )

    # --------------------------------------------------------- oos.csv -----
    cand_names = [NULL_RULE] + [
        f"{route} x {tag}" for route in routes for tag in asl.MODEL_ORDER
    ]
    orows: list[dict[str, Any]] = []
    wf_store: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for u in UNITS:
        cand_series: list[np.ndarray] = [r_hold[u]]
        for route in routes:
            for tag in asl.MODEL_ORDER:
                cand_series.append(
                    rule_returns(
                        "full_two_stage",
                        res[tag]["s_hat"][route],
                        sig[tag],
                        r_exit[u],
                        r_hold[u],
                    )[0]
                )
        arr = np.vstack(cand_series)
        scored, wf_r, pick = walk_forward(arr)
        assert np.array_equal(scored, oos_mask), "the walk-forward day set moved"
        wf_store[u] = (scored, wf_r, pick)
        for s_name in SAMPLES:
            m = (np.ones(len(dates), bool) if s_name == "whole" else era) & scored
            d_m = dates[m]
            counts = pd.Series(np.array(cand_names)[pick[m]]).value_counts()
            base_row = stats_row(r_hold[u][m], d_m)
            wf_row = stats_row(wf_r[m], d_m)
            orows.append(
                {
                    "block": "walk_forward_selected",
                    "unit": u,
                    "sample": s_name,
                    "route": "all_routes",
                    "forecast": "-",
                    "rule": (
                        f"choice over {len(cand_names) - 1} (route, forecast) "
                        f"cells on an expanding window, "
                        f"{EMBARGO_DAYS}-day embargo, null = always hold"
                    ),
                    **wf_row,
                    "Sharpe_hold": base_row["Sharpe_ann"],
                    "delta_Sharpe": wf_row["Sharpe_ann"] - base_row["Sharpe_ann"],
                    "beats_hold": bool(wf_row["Sharpe_ann"] > base_row["Sharpe_ann"]),
                    "modal_pick": str(counts.index[0]),
                    "modal_pick_share": float(counts.iloc[0] / counts.sum()),
                    "null_share": float(counts.get(NULL_RULE, 0) / counts.sum()),
                    **diff_block(wf_r[m], r_hold[u][m], d_m, "hold"),
                }
            )
            for route in routes:
                for tag in asl.MODEL_ORDER:
                    sh = res[tag]["s_hat"][route]
                    for rule in RULES:
                        r_cell = rule_returns(rule, sh, sig[tag], r_exit[u], r_hold[u])
                        c2 = rule_returns(
                            CONTROL_II, sh, sig[tag], r_exit[u], r_hold[u]
                        )[0]
                        st = stats_row(r_cell[0][m], d_m)
                        orows.append(
                            {
                                "block": "fixed_rule_oos",
                                "unit": u,
                                "sample": s_name,
                                "route": route,
                                "forecast": tag,
                                "rule": rule,
                                "n_traded": int((~r_cell[1][m]).sum()),
                                "frac_gated_out": float(r_cell[1][m].mean()),
                                **st,
                                "Sharpe_hold": base_row["Sharpe_ann"],
                                "delta_Sharpe": st["Sharpe_ann"]
                                - base_row["Sharpe_ann"],
                                "beats_hold": bool(
                                    st["Sharpe_ann"] > base_row["Sharpe_ann"]
                                ),
                                "modal_pick": rule,
                                "modal_pick_share": 1.0,
                                "null_share": float("nan"),
                                **diff_block(r_cell[0][m], r_hold[u][m], d_m, "hold"),
                                **diff_block(r_cell[0][m], c2[m], d_m, "ctrl2"),
                            }
                        )
    oos = pd.DataFrame(orows)
    write(oos, "oos.csv", "the purged, embargoed out-of-sample day set")
    n_oos = int(wf_store[PRIMARY_UNIT][0].sum())
    print(
        f"\nout-of-sample day set: expanding training window, minimum "
        f"{WARMUP_SESSIONS} training days, {EMBARGO_DAYS}-day embargo, so the "
        f"first {WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored and "
        f"{n_oos} of {len(dates)} remain"
    )
    ocols = [
        "block",
        "route",
        "forecast",
        "rule",
        "n_days",
        "n_traded",
        "frac_gated_out",
        "mean",
        "sd",
        "Sharpe_ann",
        "Sharpe_hold",
        "delta_Sharpe",
        "beats_hold",
        "t_hac",
        "MaxDD",
        "worst_day",
        "modal_pick",
        "modal_pick_share",
        "null_share",
    ]
    for u in UNITS:
        for s_name in SAMPLES:
            sub = oos[(oos["unit"] == u) & (oos["sample"] == s_name)]
            show(sub, f"oos.csv  |  unit {u}  |  sample {s_name}", ocols)

    # -------------------------------------------------------- gate.csv -----
    key = ["sample", "route", "forecast", "rule"]
    prim = oos[
        (oos["unit"] == PRIMARY_UNIT) & (oos["block"] == "fixed_rule_oos")
    ].copy()
    gate = prim.merge(
        plac.loc[
            plac["day_set"] == "oos", [*key, "p_value", "sharpe_rule", "n_gated_out"]
        ],
        on=key,
        how="left",
    )
    same = gate["sharpe_rule"].notna() & gate["Sharpe_ann"].notna()
    assert np.allclose(
        gate.loc[same, "sharpe_rule"],
        gate.loc[same, "Sharpe_ann"],
        rtol=0.0,
        atol=1e-9,
    ), "the placebo and the out-of-sample table disagree on the rule's own Sharpe"
    gate["adopted"] = (
        gate["beats_hold"]
        & gate["d_hold_positive"]
        & gate["d_ctrl2_positive"]
        & (gate["p_value"] < PLACEBO_ALPHA)
        & (gate["rule"] != ORACLE)
    )
    write(
        gate,
        "gate.csv",
        "beats always-hold AND control (ii), both CIs excluding zero, placebo p",
    )
    gcols = [
        "sample",
        "route",
        "forecast",
        "rule",
        "n_days",
        "frac_gated_out",
        "Sharpe_ann",
        "Sharpe_hold",
        "beats_hold",
        "d_hold_mean",
        "d_hold_ci_lo",
        "d_hold_ci_hi",
        "d_hold_positive",
        "d_ctrl2_mean",
        "d_ctrl2_ci_lo",
        "d_ctrl2_ci_hi",
        "d_ctrl2_positive",
        "p_value",
        "adopted",
    ]
    for s_name in SAMPLES:
        show(
            gate[gate["sample"] == s_name].sort_values(["route", "rule", "forecast"]),
            f"gate.csv  |  unit {PRIMARY_UNIT}  |  sample {s_name}",
            gcols,
        )

    # ---------------------------------------------------------- verdict ----
    print(
        f"\nVERDICT - the pre-registered cell is the {VERDICT_SAMPLE} sample on "
        f"the out-of-sample day set, in {PRIMARY_UNIT}; the oracle row is "
        f"non-causal and is never adoptable"
    )
    v = gate[(gate["sample"] == VERDICT_SAMPLE) & (gate["rule"] == "full_two_stage")]
    print(
        f"  full two-stage rule: {int(v['adopted'].sum())} of {len(v)} cells "
        f"adopted ({len(routes)} routes x {len(asl.MODEL_ORDER)} forecasts; a "
        f"5% level expects {len(v) * PLACEBO_ALPHA:.1f})"
    )
    for rule in RULES:
        sub = gate[(gate["sample"] == VERDICT_SAMPLE) & (gate["rule"] == rule)]
        tail = "  (non-causal ceiling, not adoptable)" if rule == ORACLE else ""
        print(
            f"  {rule:<28s} adopted {int(sub['adopted'].sum())} of {len(sub)}; "
            f"best Sharpe_ann {float(sub['Sharpe_ann'].max()):.4f} against "
            f"always-hold {float(sub['Sharpe_hold'].iloc[0]):.4f}{tail}"
        )
    for u in UNITS:
        w = oos[
            (oos["unit"] == u)
            & (oos["block"] == "walk_forward_selected")
            & (oos["sample"] == VERDICT_SAMPLE)
        ].iloc[0]
        verdict = "BEATS ALWAYS-HOLD" if w["beats_hold"] else "DOES NOT BEAT"
        print(
            f"  FAMILY VERDICT, {u:<17s} {VERDICT_SAMPLE}: walk-forward "
            f"{float(w['Sharpe_ann']):7.4f} against always-hold "
            f"{float(w['Sharpe_hold']):7.4f}  {verdict}  "
            f"(null picked on {100.0 * float(w['null_share']):.1f}% of days, "
            f"modal pick {w['modal_pick']})"
        )
    print(
        f"\nmultiple testing: {n_cells_pre} cells were pre-registered "
        f"({len(ROUTES)} routes x {len(asl.MODEL_ORDER)} forecasts) and "
        f"{n_cells} were actually tried (route C carries no per-(date, stamp, "
        f"horizon) forecast on disk); the gate above reads {len(v)} full "
        f"two-stage cells on the pre-registered sample and a 5% level expects "
        f"{len(v) * PLACEBO_ALPHA:.1f} false adoptions there "
        f"({n_cells_pre * PLACEBO_ALPHA:.1f} had all three routes run)."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
