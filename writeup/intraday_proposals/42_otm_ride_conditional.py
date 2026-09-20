"""42 - riding the out-of-the-money leg only when the forecast calls the last bar calm.

Proposal 41 rides the out-of-the-money leg of the 11:00 straddle past the
15:30 buy-back on EVERY day: at 15:30 it buys back only the leg the spot has
passed and leaves the other one to cash settlement, in two hedge variants
(futures flat at 15:30, or futures rebalanced once to the ridden leg's own
Black-76 delta and flattened at 16:00 against the settlement print).  On days
the spot sits between the two strikes both legs are out of the money and the
nearer one is bought back, the farther one ridden.

This proposal asks whether the 15:30 forecast signal should decide that ride.
The signal is the standalone's own,

    s_f(15:30) = rv_hat_f(15:30) - slice(15:30),

the forecast of the 15:30-16:00 bar minus the implied slice at 15:30, the
slice read off the RE-INVERTED hourly implied volatility that
``make_dh_causal_standalone_tex.deck_frame`` builds, so the five censored
March-2020 bars carry a signal instead of silently defaulting to no signal.
s_f < 0 is the forecast sitting below the implied slice: the model calls the
last half hour calm, which is exactly when the ridden leg should expire
worthless.

The rule, pre-registered and with no free parameter:

    at 15:30 buy back the in-the-money leg at its quoted ask ALWAYS;
    ride the out-of-the-money leg to cash settlement iff s_f(15:30) < 0,
    otherwise buy that leg back at its quoted ask too.

Which leg is "out of the money" is proposal 41's rule, tie included.  A day
that carries no signal does not ride, so it is the book exactly.  The two
hedge variants are proposal 41's.

The rule is proposal 37's whole-straddle "exit if s > 0" rule with the
settlement leg narrowed from the straddle to its out-of-the-money half: both
depart from the book on the SAME days (s <= 0), 37 by holding both legs to
settlement, 42 by holding only the out-of-the-money one.  That identity is
asserted, not asserted in prose.

References: the book (both legs bought back at the 15:30 quoted ask), the
unconditional ride in each variant, hold-to-cash-settlement, and proposal 37's
"exit if s > 0" rule for each forecast.  Controls: the REVERSE rule (ride iff
s_f > 0) for each forecast and variant, and a placebo that rides on a random
subset of days with the same ride count and the same run lengths.

Gates, in order:

  GATE 0  proposal 41's in-sample tape is reproduced: n = 865, book crossed
          Sharpe 2.2525467 whole / 2.5131322 era, ride-flat 2.6386 / 2.7925,
          ride-hedged 2.6698 / 2.7791, the two era paired differences against
          the book with their bootstrap CIs, the ridden leg worthless on
          90.9% of days, its 15:30 ask 0.0420 and its settlement payoff
          0.0293 of the entry premium.
  GATE P  the vectorised placebo draws proposal 30's masks exactly.
  GATE S  the 15:30 slice equals the hedge tape's own 15:30 total variance and
          the standalone's ``deck_frame`` volatility, is finite on every
          scored day, and every forecast carries a signal on every scored day.

Adoption, per cell: the paired difference against the UNCONDITIONAL ride (same
variant) AND against the BOOK must both have a bootstrap CI excluding zero on
the positive side, AND the placebo p-value must be below 0.05.  The
pre-registered cell is the era one.  The family verdict is the walk-forward's:
the embargoed choice over forecast, with the unconditional ride as the null,
must beat both references out of sample.

The eight forecasts are independent once the tape and the 15:30 implied slice
exist, so every forecast's signal, cells, reverse control and placebo draws
are computed in a worker process; the gates, the shared references and the
walk-forward stay in the parent.  Each placebo cell draws from its own
deterministic child stream, so the p-values do not depend on how many workers
run.

Run:  python writeup/intraday_proposals/42_otm_ride_conditional.py
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


p41 = _load_module(HERE / "41_otm_leg_ride.py", "p42_p41")
p30 = _load_module(HERE / "30_skip_day_rule.py", "p42_p30")
p31 = _load_module(HERE / "31_margin_skip_rule.py", "p42_p31")

OUT = p41.HOLD / "proposals" / "42"

CLOSE = p41.CLOSE
ERA_START = p41.ERA_START
ANN = p41.ANN

#: The rule's only number, and it is not a free parameter: ride when the
#: forecast sits strictly BELOW the implied slice.  The reverse control reads
#: the same number on the other side.
RIDE_THRESHOLD = 0.0

#: Proposal 41's two hedge variants, named as its own columns name them.
VARIANTS: tuple[tuple[str, str], ...] = (
    ("flat", "r_ride_flat"),
    ("hedged", "r_ride_hedged"),
)
#: The two directions the signal can be read in; "reverse" is the control and
#: is never adoptable.
DIRECTIONS: tuple[str, ...] = ("rule", "reverse")
#: The two samples every cell is scored on.
SAMPLES: tuple[str, ...] = ("whole", "era")

#: The remaining-session hours the 15:30 stamp carries: one 30-minute bar.
H_REM_1530 = 0.5

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

#: One worker per forecast, capped so the pool never outnumbers the work.
N_WORKERS = min(4, len(asl.MODEL_ORDER))

#: GATE 0 targets, all from proposal 41's own run.
GATE_N = p41.GATE_N
GATE_BOOK_WHOLE = p41.GATE_BOOK_WHOLE
GATE_BOOK_ERA = p41.GATE_BOOK_ERA
#: The unconditional ride, whole and era, as proposal 41 reports it.
GATE_RIDE: dict[str, tuple[float, float]] = {
    "r_ride_flat": (2.6386, 2.7925),
    "r_ride_hedged": (2.6698, 2.7791),
}
#: The era paired difference of each unconditional ride against the book:
#: (mean, CI low, CI high).
GATE_RIDE_DIFF: dict[str, tuple[float, float, float]] = {
    "r_ride_flat": (0.010804, 0.000784, 0.020347),
    "r_ride_hedged": (0.009039, 0.001007, 0.017307),
}
#: The ride's anatomy over the whole sample: share of days the ridden leg
#: expires worthless, its 15:30 ask and its settlement payoff, the last two in
#: units of the 11:00 midpoint entry premium.
GATE_WORTHLESS = 0.909
GATE_ASK_UNITS = 0.0420
GATE_PAY_UNITS = 0.0293

#: A number published to seven decimals is gated here.
GATE_TOL_7DP = 1e-6
#: A number published to four decimals is gated here.
GATE_TOL_4DP = 5e-5
#: The same, loosened by one place because one four-decimal reference (the
#: hedged ride's whole-sample Sharpe, 2.66975) is quoted a unit up in its last
#: place.
GATE_TOL_SHARPE_4DP = 1e-4
#: A share published to one decimal of a percent.
GATE_TOL_PCT_1DP = 5e-4
#: The 15:30 slice is an identity against the hedge tape's own total variance.
IDENT_TOL = 1e-12

#: The columns rules.csv prints.
RULE_COLS = [
    "sample",
    "family",
    "forecast",
    "variant",
    "n_days",
    "n_active",
    "frac_active",
    "mean",
    "sd",
    "Sharpe_ann",
    "t",
    "t_hac",
    "MaxDD",
    "worst_day",
    "worst_date",
    "d_ride_mean",
    "d_ride_t_hac",
    "d_ride_ci_lo",
    "d_ride_ci_hi",
    "d_ride_positive",
    "d_book_mean",
    "d_book_t_hac",
    "d_book_ci_lo",
    "d_book_ci_hi",
    "d_book_positive",
]
#: The columns the ride anatomy prints (ridden days and skipped days).
ANATOMY_COLS = [
    "sample",
    "family",
    "forecast",
    "variant",
    "ridden_n",
    "ridden_frac_worthless",
    "ridden_mean_ask_units",
    "ridden_mean_pay_units",
    "ridden_mean_saved_units",
    "ridden_mean_saved_pts",
    "ridden_mean_ride_minus_book",
    "skipped_n",
    "skipped_frac_worthless",
    "skipped_mean_ask_units",
    "skipped_mean_pay_units",
    "skipped_mean_saved_units",
    "skipped_mean_saved_pts",
    "skipped_mean_ride_minus_book",
]


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 300, "display.max_columns", 120):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"wrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# ------------------------------------------------------------- statistics ---
def stats_row(r: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """Calendar-day statistics of one daily return series, premium units."""
    v = pd.Series(np.asarray(r, float), index=dates).dropna()
    t_hac, lag = asl.newey_west_t(v)
    n = int(v.size)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n >= 2 else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    return {
        "n_days": n,
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": p41.sharpe(v),
        "t": float(np.sqrt(n)) * mean / sd if live else float("nan"),
        "t_hac": t_hac,
        "t_lag": lag,
        "MaxDD": p41.maxdd(v),
        "worst_day": float(v.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(v.idxmin()).date()) if n else "",
    }


def diff_block(
    a: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex, tag: str
) -> dict[str, Any]:
    """a - b: mean, HAC t and the bootstrap CI, named for the reference b."""
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


def side_stats(b: pd.DataFrame, m: np.ndarray, col: str, tag: str) -> dict[str, Any]:
    """The ride's anatomy on one side of the rule's mask.

    ``tag = "ridden"`` reads the days the rule rides; ``tag = "skipped"`` reads
    the days it bought the out-of-the-money leg back, so its
    ``mean_ride_minus_book`` is the counterfactual: what riding those days
    would have earned (positive) or cost (negative).
    """
    sub = b.loc[m]
    n = int(len(sub))
    if n == 0:
        return {
            f"{tag}_n": 0,
            f"{tag}_frac_worthless": float("nan"),
            f"{tag}_mean_ask_units": float("nan"),
            f"{tag}_mean_pay_units": float("nan"),
            f"{tag}_mean_saved_units": float("nan"),
            f"{tag}_mean_saved_pts": float("nan"),
            f"{tag}_mean_ride_minus_book": float("nan"),
        }
    em = sub["entry_mid"]
    return {
        f"{tag}_n": n,
        f"{tag}_frac_worthless": float((sub["pay_ride_pts"] == 0.0).mean()),
        f"{tag}_mean_ask_units": float((sub["ask_ride_pts"] / em).mean()),
        f"{tag}_mean_pay_units": float((sub["pay_ride_pts"] / em).mean()),
        f"{tag}_mean_saved_units": float((sub["saved_pts"] / em).mean()),
        f"{tag}_mean_saved_pts": float(sub["saved_pts"].mean()),
        f"{tag}_mean_ride_minus_book": float((sub[col] - sub["r_book"]).mean()),
    }


# ------------------------------------------------------------- the signal ---
def forecast_1530(tag: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """rv_hat_f(15:30) per scored day, as the standalone reads it.

    The forecast panels are bar-end labelled, so the row carrying the forecast
    ISSUED at 15:30 is stamped 16:00; the stamp is pushed back half an hour and
    the 15:30 rows are kept, exactly as ``make_dh_causal_standalone_tex.main``
    does it.
    """
    panel = asl.load_yhat_panel(asl.yhat_paths(ROOT)[tag])
    y15 = panel[["t", "rv_hat"]].copy()
    y15["t"] = pd.to_datetime(y15["t"], utc=True) - pd.Timedelta(minutes=30)
    et = pd.to_datetime(y15["t"], utc=True).dt.tz_convert("America/New_York")
    y15["date"] = et.dt.normalize().dt.tz_localize(None)
    y15 = y15[et.dt.strftime("%H:%M") == CLOSE]
    return y15.groupby("date")["rv_hat"].mean().reindex(dates).to_numpy(float)


def ride_mask(s: np.ndarray, direction: str) -> np.ndarray:
    """The pre-registered ride mask; a day with no signal never rides."""
    if direction == "rule":
        return np.asarray(s < RIDE_THRESHOLD, dtype=bool)
    return np.asarray(s > RIDE_THRESHOLD, dtype=bool)


def sample_masks(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The whole sample and the daily-0DTE era, as boolean day masks."""
    return {
        "whole": np.ones(len(dates), dtype=bool),
        "era": np.asarray(dates >= pd.Timestamp(ERA_START), dtype=bool),
    }


# ----------------------------------------------------------- the placebo ----
def placebo_stats(
    r_ride: np.ndarray, r_base: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    """Where the rule's Sharpe falls among rides drawn on random days.

    The draws are proposal 31's vectorisation of proposal 30's block shuffle
    (asserted draw for draw in GATE P): every draw rides exactly as many days,
    in runs of exactly the same lengths, and only WHICH days change.  A day the
    draw does not ride keeps the book, as the rule's own non-ridden days do.
    """
    s_rule = p41.sharpe(np.where(mask, r_ride, r_base))
    if not mask.any() or bool(mask.all()):
        return {
            "n_placebo": 0,
            "sharpe_rule": s_rule,
            "placebo_mean": float("nan"),
            "placebo_sd": float("nan"),
            "placebo_q05": float("nan"),
            "placebo_q50": float("nan"),
            "placebo_q95": float("nan"),
            "pct_rank": float("nan"),
            "p_value": float("nan"),
        }
    masks = p31.block_shuffled_masks(mask, N_PLACEBO, rng)
    rr = np.where(masks, r_ride[None, :], r_base[None, :])
    mean = rr.mean(axis=1)
    sd = rr.std(axis=1, ddof=1)
    s_plac = np.where(sd > 0.0, mean / np.where(sd > 0.0, sd, 1.0) * ANN, np.nan)
    good = s_plac[np.isfinite(s_plac)]
    ge = int((good >= s_rule).sum())
    return {
        "n_placebo": int(good.size),
        "sharpe_rule": s_rule,
        "placebo_mean": float(good.mean()),
        "placebo_sd": float(good.std(ddof=1)),
        "placebo_q05": float(np.quantile(good, 0.05)),
        "placebo_q50": float(np.quantile(good, 0.50)),
        "placebo_q95": float(np.quantile(good, 0.95)),
        "pct_rank": float((good < s_rule).mean()),
        "p_value": float((1 + ge) / (1 + good.size)),
    }


def placebo_rng(i_tag: int, i_dir: int, i_smp: int) -> np.random.Generator:
    """One deterministic stream per cell, so workers cannot reorder the draws."""
    return np.random.default_rng([PLACEBO_SEED, i_tag, i_dir, i_smp])


# -------------------------------------------------------- the worker ---------
def forecast_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Everything one forecast owns, computed in its own process.

    The parent hands over the tape (the day index, the ride frame and the
    15:30 implied slice); the worker loads that forecast's panel, forms its
    signal, and returns its rule rows, its reverse-control rows, its ride
    anatomy, its raw-skill rows, its placebo rows and proposal 37's
    whole-straddle reference for the same forecast.  Nothing here reads a
    shared mutable object, so the eight jobs are independent.
    """
    tag = str(job["tag"])
    i_tag = int(job["i_tag"])
    dates = pd.DatetimeIndex(job["dates"])
    sl = np.asarray(job["slice"], float)
    book = job["book"]
    r_book = book["r_book"].to_numpy(float)
    r_hold = book["r_hold"].to_numpy(float)
    r_var = {v: book[col].to_numpy(float) for v, col in VARIANTS}
    smp_masks = sample_masks(dates)
    pay_gt_ask = (book["pay_ride_pts"] > book["ask_ride_pts"]).to_numpy(bool)

    rv = forecast_1530(tag, dates)
    s = rv - sl
    rides = {d: ride_mask(s, d) for d in DIRECTIONS}

    rules: list[dict[str, Any]] = []
    anat: list[dict[str, Any]] = []
    skill: list[dict[str, Any]] = []
    plac: list[dict[str, Any]] = []
    ref37: list[dict[str, Any]] = []
    for i_smp, smp in enumerate(SAMPLES):
        m = smp_masks[smp]
        d_m = dates[m]
        b = book.loc[m]
        r_exit_rule = np.where(s > RIDE_THRESHOLD, r_book, r_hold)
        ref37.append(
            {
                "sample": smp,
                "family": "reference",
                "forecast": tag,
                "variant": "exit if s > 0 (37, whole straddle)",
                "n_active": int(rides["rule"][m].sum()),
                "frac_active": float(rides["rule"][m].mean()),
                **stats_row(r_exit_rule[m], d_m),
            }
        )
        for i_dir, d in enumerate(DIRECTIONS):
            act = rides[d]
            rng = placebo_rng(i_tag, i_dir, i_smp)
            for v, col in VARIANTS:
                r = np.where(act, r_var[v], r_book)
                rules.append(
                    {
                        "sample": smp,
                        "family": d,
                        "forecast": tag,
                        "variant": v,
                        "n_active": int(act[m].sum()),
                        "frac_active": float(act[m].mean()),
                        **stats_row(r[m], d_m),
                        **diff_block(r[m], r_var[v][m], d_m, "ride"),
                        **diff_block(r[m], r_book[m], d_m, "book"),
                    }
                )
                anat.append(
                    {
                        "sample": smp,
                        "family": d,
                        "forecast": tag,
                        "variant": v,
                        **side_stats(b, act[m], col, "ridden"),
                        **side_stats(b, ~act[m], col, "skipped"),
                    }
                )
                plac.append(
                    {
                        "sample": smp,
                        "family": d,
                        "forecast": tag,
                        "variant": v,
                        "n_days": int(m.sum()),
                        "n_ridden": int(act[m].sum()),
                        **placebo_stats(r_var[v][m], r_book[m], act[m], rng),
                    }
                )
        s_m = s[m]
        for cname, cm in (
            ("s < 0 (ride)", s_m < RIDE_THRESHOLD),
            ("s > 0 (buy back)", s_m > RIDE_THRESHOLD),
            ("no signal", ~np.isfinite(s_m)),
        ):
            n = int(cm.sum())
            sub = b.loc[cm]
            hit = pay_gt_ask[m][cm]
            skill.append(
                {
                    "sample": smp,
                    "forecast": tag,
                    "s_class": cname,
                    "n": n,
                    "mean_pay_pts": float(sub["pay_ride_pts"].mean())
                    if n
                    else float("nan"),
                    "mean_pay_units": float(
                        (sub["pay_ride_pts"] / sub["entry_mid"]).mean()
                    )
                    if n
                    else float("nan"),
                    "mean_ask_pts": float(sub["ask_ride_pts"].mean())
                    if n
                    else float("nan"),
                    "mean_ask_units": float(
                        (sub["ask_ride_pts"] / sub["entry_mid"]).mean()
                    )
                    if n
                    else float("nan"),
                    "frac_worthless": float((sub["pay_ride_pts"] == 0.0).mean())
                    if n
                    else float("nan"),
                    "n_pay_gt_ask": int(hit.sum()),
                    "n_pay_le_ask": int(n - int(hit.sum())),
                    "rate_pay_gt_ask": float(hit.mean()) if n else float("nan"),
                }
            )
    return {
        "tag": tag,
        "rv": rv,
        "s": s,
        "rules": rules,
        "anatomy": anat,
        "skill": skill,
        "placebo": plac,
        "ref37": ref37,
    }


# ------------------------------------------------------ the walk-forward ----
def walk_forward(
    series: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expanding, embargoed choice over the candidates; only the day is scored.

    Proposal 30's walk-forward, unchanged in shape: for day i the training
    window is days [0, i - EMBARGO_DAYS), the candidate with the best in-window
    calendar-day Sharpe is applied to day i, candidate 0 is the null (the
    unconditional ride) and ``np.argmax`` keeps the first maximum so a tie goes
    to it, and days whose training window is shorter than WARMUP_SESSIONS are
    not scored.
    """
    n = series.shape[1]
    curves = np.vstack([p30.expanding_sharpe(row) for row in series])
    curves = np.where(np.isfinite(curves), curves, -np.inf)
    scored = np.zeros(n, dtype=bool)
    pick = np.full(n, -1, dtype=int)
    r = np.full(n, np.nan)
    ride = np.zeros(n, dtype=bool)
    for i in range(n):
        k = i - EMBARGO_DAYS
        if k < WARMUP_SESSIONS:
            continue
        c = int(np.argmax(curves[:, k]))
        pick[i] = c
        r[i] = series[c, i]
        ride[i] = bool(masks[c, i])
        scored[i] = True
    return scored, r, ride, pick


# ----------------------------------------------------------------- gates ----
def gate_zero(book: pd.DataFrame) -> None:
    """Proposal 41's in-sample tape, reproduced before anything new is built."""
    era = np.asarray(book.index >= pd.Timestamp(ERA_START), dtype=bool)
    dates = pd.DatetimeIndex(book.index)
    n = int(len(book))
    s_whole = p41.sharpe(book["r_book"])
    s_era = p41.sharpe(book.loc[era, "r_book"])
    assert n == GATE_N, n
    assert abs(s_whole - GATE_BOOK_WHOLE) < GATE_TOL_7DP, s_whole
    assert abs(s_era - GATE_BOOK_ERA) < GATE_TOL_7DP, s_era
    print(
        f"GATE 0  n = {n} scored days ({int(era.sum())} in the daily era from "
        f"{ERA_START}); book crossed-quoted Sharpe_ann {s_whole:.7f} (reference "
        f"{GATE_BOOK_WHOLE}) whole, {s_era:.7f} (reference {GATE_BOOK_ERA}) era"
    )
    for col, (want_w, want_e) in GATE_RIDE.items():
        got_w = p41.sharpe(book[col])
        got_e = p41.sharpe(book.loc[era, col])
        assert abs(got_w - want_w) < GATE_TOL_SHARPE_4DP, (col, got_w, want_w)
        assert abs(got_e - want_e) < GATE_TOL_SHARPE_4DP, (col, got_e, want_e)
        print(
            f"GATE 0  unconditional {col} Sharpe_ann {got_w:.7f} (reference "
            f"{want_w}) whole, {got_e:.7f} (reference {want_e}) era"
        )
    for col, (want_m, want_lo, want_hi) in GATE_RIDE_DIFF.items():
        row = p41.paired_row(col, book.loc[era, col], book.loc[era, "r_book"])
        assert abs(row["mean_diff"] - want_m) < GATE_TOL_7DP, (col, row["mean_diff"])
        assert abs(row["ci_lo"] - want_lo) < GATE_TOL_7DP, (col, row["ci_lo"])
        assert abs(row["ci_hi"] - want_hi) < GATE_TOL_7DP, (col, row["ci_hi"])
        print(
            f"GATE 0  era {col} - book on {row['n']} days: mean "
            f"{row['mean_diff']:+.7f} (reference {want_m:+.6f}), HAC t "
            f"{row['t_hac']:+.6f} (lag {row['t_lag']}), CI [{row['ci_lo']:+.7f}, "
            f"{row['ci_hi']:+.7f}] (reference [{want_lo:+.6f}, {want_hi:+.6f}])"
        )
    worthless = float((book["pay_ride_pts"] == 0.0).mean())
    ask_units = float((book["ask_ride_pts"] / book["entry_mid"]).mean())
    pay_units = float((book["pay_ride_pts"] / book["entry_mid"]).mean())
    assert abs(worthless - GATE_WORTHLESS) < GATE_TOL_PCT_1DP, worthless
    assert abs(ask_units - GATE_ASK_UNITS) < GATE_TOL_4DP, ask_units
    assert abs(pay_units - GATE_PAY_UNITS) < GATE_TOL_4DP, pay_units
    print(
        f"GATE 0  the ridden leg expires worthless on {100.0 * worthless:.4f}% of "
        f"days (reference {100.0 * GATE_WORTHLESS:.1f}%); its 15:30 ask is "
        f"{ask_units:.6f} (reference {GATE_ASK_UNITS}) and its settlement payoff "
        f"{pay_units:.6f} (reference {GATE_PAY_UNITS}) of the entry premium; "
        f"first day {pd.Timestamp(dates.min()).date()}, last day "
        f"{pd.Timestamp(dates.max()).date()}"
    )
    print(
        f"GATE 0  the bootstrap is proposal 41's: circular block, B = {BOOT_B}, "
        f"block {BOOT_BLOCK}, seed {BOOT_SEED}"
    )
    print("GATE 0 PASSED")


def gate_placebo(n: int) -> None:
    """The vectorised placebo draws proposal 30's masks, draw for draw."""
    probe = np.zeros(n, dtype=bool)
    probe[::3] = True
    probe[7:19] = True
    a = p31.block_shuffled_masks(probe, N_PLACEBO, np.random.default_rng(PLACEBO_SEED))
    b = p30.block_shuffled_masks(probe, N_PLACEBO, np.random.default_rng(PLACEBO_SEED))
    assert (a == b).all(), "the vectorised placebo does not reproduce 30's draws"
    assert (a.sum(axis=1) == int(probe.sum())).all(), "a draw changed the ride count"
    print(
        f"\nGATE P  {N_PLACEBO} vectorised block-shuffled draws on a "
        f"{n}-day probe riding {int(probe.sum())} days are identical to proposal "
        f"30's loop, draw for draw, at seed {PLACEBO_SEED}; every draw rides "
        f"{int(probe.sum())} days"
    )
    print("GATE P PASSED")


def gate_signal(
    g: dict[str, Any], mk: ModuleType, sl: np.ndarray, sig: dict[str, np.ndarray]
) -> None:
    """The 15:30 signal, tied to the standalone's own re-inverted volatility."""
    dates = pd.DatetimeIndex(g["dates"])
    tot15 = g["tot"][:, g["j15"]]
    d_ident = float(np.nanmax(np.abs(sl - tot15**2)))
    assert d_ident < IDENT_TOL, d_ident
    frame, _clocks = mk.deck_frame(g["pkg"])
    iv15 = (
        frame.loc[frame["hhmm"] == CLOSE, ["date", "iv_hourly_used"]]
        .drop_duplicates("date")
        .set_index("date")["iv_hourly_used"]
        .astype(float)
        .reindex(dates)
        .to_numpy(float)
    )
    d_iv = float(np.nanmax(np.abs(iv15**2 * H_REM_1530 - sl)))
    assert d_iv < IDENT_TOL, d_iv
    assert bool(np.isfinite(sl).all()), int((~np.isfinite(sl)).sum())
    print(
        f"\nGATE S  the 15:30 implied slice is the hedge tape's own 15:30 total "
        f"variance to {d_ident:.3e} and the standalone deck_frame's "
        f"iv_hourly_used^2 x {H_REM_1530} to {d_iv:.3e}; it is finite on all "
        f"{len(dates)} scored days (the raw vendor column is censored on five "
        f"15:30 bars, which would otherwise carry no signal)"
    )
    for tag in asl.MODEL_ORDER:
        s = sig[tag]
        n_fin = int(np.isfinite(s).sum())
        assert n_fin == len(dates), (tag, n_fin)
        print(
            f"GATE S  {tag:<9s} {asl.YHAT_LABEL[tag][:40]:<40s} finite on {n_fin} "
            f"of {len(dates)} days; s < 0 on "
            f"{int((s < RIDE_THRESHOLD).sum()):>3d}, s > 0 on "
            f"{int((s > RIDE_THRESHOLD).sum()):>3d}, s = 0 on "
            f"{int((s == RIDE_THRESHOLD).sum())}"
        )
    print("GATE S PASSED")


# ------------------------------------------------------------------ main ----
def main() -> None:  # noqa: PLR0912, PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 300)

    # ------------------------------------------- the tape, gated first -----
    mk = p41._standalone()
    g = p41.insample_grids(mk)
    quotes = p41.exit_leg_quotes(g["panel"], g)
    n_missing = int(quotes.isna().to_numpy().sum())
    print(
        f"15:30 quotes of the 11:00 strikes read from data/spxw_chain.parquet: "
        f"{n_missing} missing or no-quote (bid == ask == 0) legs"
    )
    dates = pd.DatetimeIndex(g["dates"])
    book = p41.ride_frame(
        dates,
        g["K_c"],
        g["K_p"],
        g["entry"],
        g["bid"],
        g["Sg"][:, g["j15"]],
        g["S_close"],
        quotes["ask_c"].to_numpy(float),
        quotes["ask_p"].to_numpy(float),
        g["tot"][:, g["j15"]],
        g["tot15_mid"],
        g["hedge_exit"],
        g["hedge_hold"],
    )
    gate_zero(book)
    gate_placebo(len(dates))

    # ------------------------------ the eight forecasts, one per process ---
    sl = g["IVg"][:, g["j15"]] ** 2 * H_REM_1530
    jobs = [
        {"tag": tag, "i_tag": i, "dates": dates, "slice": sl, "book": book}
        for i, tag in enumerate(asl.MODEL_ORDER)
    ]
    print(
        f"\nfanning {len(jobs)} forecasts over {N_WORKERS} worker processes; each "
        f"worker loads its own forecast panel, forms its 15:30 signal and "
        f"computes its {len(DIRECTIONS)} directions x {len(VARIANTS)} variants x "
        f"{len(SAMPLES)} samples of cells, anatomy, raw skill and "
        f"{N_PLACEBO}-draw placebo; the gates, the shared references and the "
        f"walk-forward stay in the parent"
    )
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(forecast_cell, jobs))
    res = {r["tag"]: r for r in results}
    assert set(res) == set(asl.MODEL_ORDER), sorted(res)
    sig = {tag: np.asarray(res[tag]["s"], float) for tag in asl.MODEL_ORDER}
    rv = {tag: np.asarray(res[tag]["rv"], float) for tag in asl.MODEL_ORDER}
    gate_signal(g, mk, sl, sig)

    smp_masks = sample_masks(dates)
    r_book = book["r_book"].to_numpy(float)
    r_hold = book["r_hold"].to_numpy(float)
    r_var = {v: book[col].to_numpy(float) for v, col in VARIANTS}
    rides = {
        (tag, d): ride_mask(sig[tag], d) for tag in asl.MODEL_ORDER for d in DIRECTIONS
    }
    for tag in asl.MODEL_ORDER:
        r37 = np.where(sig[tag] > RIDE_THRESHOLD, r_book, r_hold)
        r37_alt = np.where(rides[(tag, "rule")], r_hold, r_book)
        assert np.array_equal(r37, r37_alt), tag
        assert np.array_equal(rides[(tag, "rule")], ~rides[(tag, "reverse")]), tag
    print(
        f"\nthe rule departs from the book on exactly the days proposal 37's "
        f"'exit if s > 0' rule holds (s <= 0), and no signal is exactly zero, so "
        f"the rule mask and the reverse control's mask partition the "
        f"{len(dates)} days for every one of the {len(asl.MODEL_ORDER)} forecasts"
    )

    n_cells = len(asl.MODEL_ORDER) * len(VARIANTS) * len(SAMPLES)
    n_reverse = len(asl.MODEL_ORDER) * len(VARIANTS)
    print(
        f"\ncells tried: {len(asl.MODEL_ORDER)} forecasts x {len(VARIANTS)} hedge "
        f"variants x {len(SAMPLES)} samples = {n_cells}, plus {n_reverse} reverse "
        f"controls ({len(asl.MODEL_ORDER)} forecasts x {len(VARIANTS)} variants, "
        f"scored on both samples and never adoptable); at the "
        f"{PLACEBO_ALPHA:.0%} level the gate over {n_cells} cells expects "
        f"{n_cells * PLACEBO_ALPHA:.1f} false adoptions by chance"
    )

    # ------------------------------------------------------- rules.csv -----
    rows: list[dict[str, Any]] = []
    for smp in SAMPLES:
        m = smp_masks[smp]
        d_m = dates[m]
        refs: list[tuple[str, np.ndarray, np.ndarray]] = [
            ("book (both legs bought back)", r_book, np.zeros(len(dates), dtype=bool)),
            ("hold to cash settlement", r_hold, np.ones(len(dates), dtype=bool)),
        ]
        for v, _col in VARIANTS:
            refs.append(
                (
                    f"unconditional ride, {v}",
                    r_var[v],
                    np.ones(len(dates), dtype=bool),
                )
            )
        for vname, r, act in refs:
            rows.append(
                {
                    "sample": smp,
                    "family": "reference",
                    "forecast": "-",
                    "variant": vname,
                    "n_active": int(act[m].sum()),
                    "frac_active": float(act[m].mean()),
                    **stats_row(r[m], d_m),
                }
            )
        for tag in asl.MODEL_ORDER:
            rows += [r for r in res[tag]["ref37"] if r["sample"] == smp]
        for d in DIRECTIONS:
            for tag in asl.MODEL_ORDER:
                rows += [
                    r
                    for r in res[tag]["rules"]
                    if r["sample"] == smp and r["family"] == d
                ]
    rules_df = pd.DataFrame(rows)
    rules_df["frac_ridden"] = np.where(
        rules_df["family"].isin(DIRECTIONS), rules_df["frac_active"], np.nan
    )
    anat_df = pd.DataFrame(
        [
            r
            for smp in SAMPLES
            for d in DIRECTIONS
            for tag in asl.MODEL_ORDER
            for r in res[tag]["anatomy"]
            if r["sample"] == smp and r["family"] == d
        ]
    )
    skill_df = pd.DataFrame(
        [
            r
            for smp in SAMPLES
            for tag in asl.MODEL_ORDER
            for r in res[tag]["skill"]
            if r["sample"] == smp
        ]
    )
    plac_df = pd.DataFrame(
        [
            r
            for smp in SAMPLES
            for d in DIRECTIONS
            for tag in asl.MODEL_ORDER
            for r in res[tag]["placebo"]
            if r["sample"] == smp and r["family"] == d
        ]
    )
    for smp in SAMPLES:
        show(
            rules_df[rules_df["sample"] == smp],
            f"rules.csv  |  sample {smp}  |  one unit a day in units of the 11:00 "
            f"midpoint entry premium; d_ride is against the unconditional ride of "
            f"the same variant, d_book against the book",
            RULE_COLS,
        )
    for smp in SAMPLES:
        show(
            anat_df[anat_df["sample"] == smp],
            f"anatomy.csv  |  sample {smp}  |  the ridden days, and the "
            f"counterfactual on the days the rule bought the out-of-the-money leg "
            f"back (skipped_mean_ride_minus_book is what riding them would have "
            f"earned)",
            ANATOMY_COLS,
        )
    for smp in SAMPLES:
        show(
            skill_df[skill_df["sample"] == smp],
            f"skill.csv  |  sample {smp}  |  the signal's raw skill on the "
            f"out-of-the-money leg: its settlement payoff by the sign of s, and "
            f"the 2x2 of 'payoff > ask' against that sign (payoff > ask is the "
            f"ride losing against the buy-back)",
        )

    # --------------------------------------------------------- oos.csv -----
    orows: list[dict[str, Any]] = []
    wf: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    cand_names = ["unconditional ride (null)"] + list(asl.MODEL_ORDER)
    for d in DIRECTIONS:
        for v, _col in VARIANTS:
            cand_masks = np.vstack(
                [np.ones(len(dates), dtype=bool)]
                + [rides[(tag, d)] for tag in asl.MODEL_ORDER]
            )
            cand_series = np.vstack(
                [np.where(cm, r_var[v], r_book) for cm in cand_masks]
            )
            wf[(d, v)] = walk_forward(cand_series, cand_masks)
    scored_ref = wf[("rule", VARIANTS[0][0])][0]
    for d in DIRECTIONS:
        for v, _col in VARIANTS:
            scored, r_wf, ride_wf, pick = wf[(d, v)]
            for smp in SAMPLES:
                sel = scored & smp_masks[smp]
                if not sel.any():
                    continue
                d_m = dates[sel]
                st = stats_row(r_wf[sel], d_m)
                ride_st = stats_row(r_var[v][sel], d_m)
                book_st = stats_row(r_book[sel], d_m)
                hold_st = stats_row(r_hold[sel], d_m)
                counts = pd.Series(np.array(cand_names)[pick[sel]]).value_counts()
                orows.append(
                    {
                        "block": "walk_forward_selected",
                        "sample": smp,
                        "family": d,
                        "variant": v,
                        "forecast": "chosen on an expanding window",
                        "n_active": int(ride_wf[sel].sum()),
                        "frac_active": float(ride_wf[sel].mean()),
                        **st,
                        "Sharpe_uncond_ride": ride_st["Sharpe_ann"],
                        "Sharpe_book": book_st["Sharpe_ann"],
                        "Sharpe_hold": hold_st["Sharpe_ann"],
                        "beats_ride": bool(st["Sharpe_ann"] > ride_st["Sharpe_ann"]),
                        "beats_book": bool(st["Sharpe_ann"] > book_st["Sharpe_ann"]),
                        "beats_both": bool(
                            st["Sharpe_ann"] > ride_st["Sharpe_ann"]
                            and st["Sharpe_ann"] > book_st["Sharpe_ann"]
                        ),
                        "modal_pick": str(counts.index[0]),
                        "modal_pick_share": float(counts.iloc[0] / counts.sum()),
                        "null_share": float(
                            counts.get(cand_names[0], 0) / counts.sum()
                        ),
                        "n_candidates": len(cand_names),
                        **diff_block(r_wf[sel], r_var[v][sel], d_m, "ride"),
                        **diff_block(r_wf[sel], r_book[sel], d_m, "book"),
                    }
                )
    for d in DIRECTIONS:
        for v, _col in VARIANTS:
            for smp in SAMPLES:
                sel = scored_ref & smp_masks[smp]
                d_m = dates[sel]
                ride_st = stats_row(r_var[v][sel], d_m)
                book_st = stats_row(r_book[sel], d_m)
                hold_st = stats_row(r_hold[sel], d_m)
                for tag in asl.MODEL_ORDER:
                    act = rides[(tag, d)]
                    r = np.where(act, r_var[v], r_book)
                    st = stats_row(r[sel], d_m)
                    orows.append(
                        {
                            "block": "fixed_rule_oos",
                            "sample": smp,
                            "family": d,
                            "variant": v,
                            "forecast": tag,
                            "n_active": int(act[sel].sum()),
                            "frac_active": float(act[sel].mean()),
                            **st,
                            "Sharpe_uncond_ride": ride_st["Sharpe_ann"],
                            "Sharpe_book": book_st["Sharpe_ann"],
                            "Sharpe_hold": hold_st["Sharpe_ann"],
                            "beats_ride": bool(
                                st["Sharpe_ann"] > ride_st["Sharpe_ann"]
                            ),
                            "beats_book": bool(
                                st["Sharpe_ann"] > book_st["Sharpe_ann"]
                            ),
                            "beats_both": bool(
                                st["Sharpe_ann"] > ride_st["Sharpe_ann"]
                                and st["Sharpe_ann"] > book_st["Sharpe_ann"]
                            ),
                            "modal_pick": tag,
                            "modal_pick_share": 1.0,
                            "null_share": float("nan"),
                            "n_candidates": 1,
                            **diff_block(r[sel], r_var[v][sel], d_m, "ride"),
                            **diff_block(r[sel], r_book[sel], d_m, "book"),
                        }
                    )
    oos_df = pd.DataFrame(orows)
    ocols = [
        "block",
        "sample",
        "family",
        "variant",
        "forecast",
        "n_days",
        "n_active",
        "frac_active",
        "mean",
        "Sharpe_ann",
        "Sharpe_uncond_ride",
        "Sharpe_book",
        "Sharpe_hold",
        "beats_ride",
        "beats_book",
        "beats_both",
        "d_ride_mean",
        "d_ride_ci_lo",
        "d_ride_ci_hi",
        "d_book_mean",
        "d_book_ci_lo",
        "d_book_ci_hi",
        "modal_pick",
        "modal_pick_share",
        "null_share",
    ]
    print(
        f"\nout-of-sample day set: expanding training window, {EMBARGO_DAYS}-day "
        f"embargo, minimum {WARMUP_SESSIONS} training days, so the first "
        f"{WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored "
        f"({int(scored_ref.sum())} of {len(dates)} days scored); the candidate "
        f"list is the unconditional ride (the null, candidate 0) and the "
        f"{len(asl.MODEL_ORDER)} conditional rules, one per forecast"
    )
    for smp in SAMPLES:
        show(oos_df[oos_df["sample"] == smp], f"oos.csv  |  sample {smp}", ocols)

    # --------------------------------- the walk-forward's own placebo ------
    wf_rows: list[dict[str, Any]] = []
    for i_smp, smp in enumerate(SAMPLES):
        for i_dir, d in enumerate(DIRECTIONS):
            for v, _col in VARIANTS:
                scored, _r_wf, ride_wf, _pick = wf[(d, v)]
                sel = scored & smp_masks[smp]
                rng = placebo_rng(len(asl.MODEL_ORDER), i_dir, i_smp)
                wf_rows.append(
                    {
                        "sample": smp,
                        "family": f"{d}_walk_forward",
                        "forecast": "chosen on an expanding window",
                        "variant": v,
                        "n_days": int(sel.sum()),
                        "n_ridden": int(ride_wf[sel].sum()),
                        **placebo_stats(r_var[v][sel], r_book[sel], ride_wf[sel], rng),
                    }
                )
    plac_df = pd.concat([plac_df, pd.DataFrame(wf_rows)], ignore_index=True)
    pcols = [
        "sample",
        "family",
        "forecast",
        "variant",
        "n_days",
        "n_ridden",
        "n_placebo",
        "sharpe_rule",
        "placebo_mean",
        "placebo_sd",
        "placebo_q05",
        "placebo_q50",
        "placebo_q95",
        "pct_rank",
        "p_value",
    ]
    print(
        f"\nplacebo: {N_PLACEBO} draws per cell; each draw cuts the rule's own "
        f"ride mask into its maximal runs and permutes them, so every draw rides "
        f"exactly as many days in runs of exactly the same lengths and only WHICH "
        f"days change; a day a draw does not ride keeps the book; "
        f"p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo).  Each "
        f"cell draws from its own stream seeded (PLACEBO_SEED = {PLACEBO_SEED}, "
        f"forecast index, direction index, sample index), so the p-values do not "
        f"depend on how many worker processes run"
    )
    for smp in SAMPLES:
        show(plac_df[plac_df["sample"] == smp], f"placebo.csv  |  sample {smp}", pcols)

    # -------------------------------------------------------- gate.csv -----
    key = ["sample", "family", "forecast", "variant"]
    gate_df = rules_df[rules_df["family"].isin(DIRECTIONS)].merge(
        plac_df[[*key, "p_value", "sharpe_rule", "n_placebo"]], on=key, how="left"
    )
    assert not gate_df["p_value"].isna().any(), "the gate lost its placebo leg"
    assert np.allclose(
        gate_df["sharpe_rule"],
        gate_df["Sharpe_ann"],
        rtol=0.0,
        atol=IDENT_TOL,
    ), "the placebo and rules.csv disagree on a cell's own Sharpe"
    gate_df["adoptable"] = gate_df["family"] == "rule"
    gate_df["passes_three"] = (
        gate_df["d_ride_positive"]
        & gate_df["d_book_positive"]
        & (gate_df["p_value"] < PLACEBO_ALPHA)
    )
    gate_df["adopted"] = gate_df["adoptable"] & gate_df["passes_three"]
    gate_df["pre_registered"] = gate_df["sample"] == "era"
    gcols = [
        "sample",
        "family",
        "forecast",
        "variant",
        "adoptable",
        "Sharpe_ann",
        "d_ride_mean",
        "d_ride_ci_lo",
        "d_ride_ci_hi",
        "d_ride_positive",
        "d_book_mean",
        "d_book_ci_lo",
        "d_book_ci_hi",
        "d_book_positive",
        "p_value",
        "passes_three",
        "adopted",
        "pre_registered",
    ]
    show(
        gate_df,
        f"gate.csv  |  adopted = the rule direction AND the CI against the "
        f"unconditional ride of the same variant excludes zero on the positive "
        f"side AND the CI against the book does too AND the placebo "
        f"p < {PLACEBO_ALPHA}",
        gcols,
    )
    era_cells = gate_df[gate_df["pre_registered"] & gate_df["adoptable"]]
    whole_cells = gate_df[~gate_df["pre_registered"] & gate_df["adoptable"]]
    rev_cells = gate_df[~gate_df["adoptable"]]
    print(
        f"\nadoption: {int(era_cells['adopted'].sum())} of {len(era_cells)} "
        f"pre-registered era cells adopted; {int(whole_cells['adopted'].sum())} of "
        f"{len(whole_cells)} whole-sample cells pass the same three tests; "
        f"{len(rev_cells)} reverse-control cells read, none adoptable, "
        f"{int(rev_cells['passes_three'].sum())} of them passing the three tests"
    )
    for _, row in gate_df[gate_df["adopted"]].iterrows():
        print(
            f"  ADOPTED  {row['sample']:<5s} {row['forecast']:<9s} "
            f"{row['variant']:<6s} Sharpe {row['Sharpe_ann']:.4f}, vs ride "
            f"[{row['d_ride_ci_lo']:+.6f}, {row['d_ride_ci_hi']:+.6f}], vs book "
            f"[{row['d_book_ci_lo']:+.6f}, {row['d_book_ci_hi']:+.6f}], placebo p "
            f"{row['p_value']:.4f}"
        )
    for _, row in oos_df[
        (oos_df["block"] == "walk_forward_selected") & (oos_df["family"] == "rule")
    ].iterrows():
        verdict = (
            "BEATS BOTH REFERENCES"
            if row["beats_both"]
            else "DOES NOT BEAT BOTH REFERENCES"
        )
        pl = plac_df[
            (plac_df["sample"] == row["sample"])
            & (plac_df["family"] == "rule_walk_forward")
            & (plac_df["variant"] == row["variant"])
        ]
        print(
            f"  family verdict, {row['sample']:<5s} {row['variant']:<6s} "
            f"walk-forward Sharpe {row['Sharpe_ann']:7.4f} against the "
            f"unconditional ride {row['Sharpe_uncond_ride']:7.4f} and the book "
            f"{row['Sharpe_book']:7.4f}  {verdict}  (placebo p "
            f"{float(pl['p_value'].iloc[0]):.4f}, null picked on "
            f"{100.0 * float(row['null_share']):.1f}% of scored days)"
        )

    # --------------------------------------------------- out of sample -----
    n_unseen = len(pd.read_csv(p41.OUT / "c_oos_daily.csv"))
    print(
        f"\nout of sample: the forecast panels end "
        f"{pd.Timestamp(dates.max()).date()}, so no cell of this proposal can be "
        f"scored on the {p41.OOS_START}..{p41.OOS_END} sessions proposal 41 "
        f"prices ({n_unseen} unseen sessions).  The reference that CAN be carried "
        f"there is proposal 41's unconditional ride, and it fails: its paired "
        f"difference against the book straddles zero on those sessions."
    )
    carried = pd.read_csv(p41.OUT / "c_oos_paired.csv")
    carried = carried[
        (carried["period"] == f"{p41.OOS_START}..{p41.OOS_END}")
        & (carried["unit"] == "premium")
    ]
    show(
        carried,
        "proposal 41's unconditional-ride result on the unseen sessions, carried "
        "verbatim from results/.../proposals/41/c_oos_paired.csv",
        [
            "period",
            "difference",
            "n",
            "mean_diff",
            "t_hac",
            "ci_lo",
            "ci_hi",
            "excludes_zero_positive",
        ],
    )

    # ------------------------------------------------------- daily.csv -----
    daily = book.copy()
    daily["slice_1530"] = sl
    for tag in asl.MODEL_ORDER:
        daily[f"rv_hat_{tag}"] = rv[tag]
        daily[f"s_{tag}"] = sig[tag]
        daily[f"ride_{tag}"] = rides[(tag, "rule")]
        for v, _col in VARIANTS:
            daily[f"r_{tag}_{v}"] = np.where(rides[(tag, "rule")], r_var[v], r_book)
    daily.index.name = "date"
    daily.reset_index().to_csv(OUT / "daily.csv", index=False)
    print(
        f"wrote daily.csv: {len(daily)} rows, {len(daily.columns) + 1} columns  "
        f"[the per-day tape, every signal and every rule return]"
    )
    write(rules_df, "rules.csv", "every cell and every reference, both samples")
    write(anat_df, "anatomy.csv", "ridden days and the skipped-day counterfactual")
    write(skill_df, "skill.csv", "the signal's raw skill on the ridden leg")
    write(oos_df, "oos.csv", "the purged, embargoed walk-forward over forecast")
    write(plac_df, "placebo.csv", "the random-ride placebo")
    write(gate_df, "gate.csv", "the three-test adoption gate, per cell")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
