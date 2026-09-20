"""36 - the remaining-window forecast scored on its own horizon, and re-hedged.

Proposal 34 asked which volatility the hedge delta of the 11:00 straddle is
computed with, and scored the candidates against the variance the session
goes on to realize between the stamp and the close.  Its model candidate was
``rv_hat / w_slice``: the NEXT-BAR forecast divided by that clock's expanding
share of the remaining window.  That construction is a next-bar forecaster
wearing a rest-of-session costume - dividing by ``w_slice`` multiplies the
next-bar error by ``1 / w_slice``, which at 11:00 is about ten.  The
remaining-window scoreboard 34 printed (era pooled QLIKE: implied 0.083671,
blk2 0.211100) therefore compares a market forecast built FOR the remaining
window against a model forecast built for the next thirty minutes and then
inflated.  This proposal asks the same two questions of an HONEST
rest-of-session forecast.

What the repo already has, and why it is not enough
---------------------------------------------------
``experiments/ft_remaining.py`` (commit d3809a4, the F_t-measurability
campaign) is the repo's named ``F_rem`` object.  Read it: it is

    F_rem(e) = f_next(e) / w_next(e),

the SAME 1/w inflation, with ``f_next = yhat^2 * baseline`` in place of the
second-order-mapped ``rv_hat`` and a ratio-of-expanding-means share in place
of the expanding-mean-of-ratios share.  It is strictly F_t-measurable - that
is what it was built to fix - but it is not a rest-of-session forecaster, so
it inherits exactly the unfairness this proposal is about.  It is scored here
as a row of the scoreboard (``blk2_ft_remaining``) so the reader can see that
for himself; no other use is made of it.

``experiments/multihorizon_ttc.py`` IS the rest-of-session forecaster the
campaign defines: per entry clock, ``log RV_(t,T)`` regressed on the
one-step forecast standing at ``t`` (M1), plus the session's realized
variance so far (M2), plus yesterday's same-horizon realized variance (M3),
each fitted on an expanding window of strictly earlier sessions with a
63-session warm-up, applied one day forward, and finished with the causal
QLIKE-optimal multiplicative rescale (the expanding, one-day-lagged mean of
realized over forecast).  It persists only pooled and by-hour QLIKE
(``results/multihorizon/ttc_{byhour,pooled,coefs}.csv``) - there is NO
per-(date, stamp) remaining forecast on disk anywhere in the repo.  So the
honest object is REBUILT here, on proposal 34's own grid, by the same recipe:

  R1  log y ~ a_c + b_c log rv_hat(c)
  R2  R1 + d_c log RV(open -> c)
  R3  R2 + e_c log y(previous session, same clock)

with ``y`` the variance the session realizes from the stamp to the close -
proposal 34's target object, bar for bar - ``rv_hat`` the panel's own
second-order-mapped next-bar forecast (the object 34 fed to 1/w), and
``RV(open -> c)`` the session's realized variance over the trade bars that
have already closed.  Every regressor is measurable at the stamp, every fit
sees strictly earlier sessions only, and the rescale is lagged one session:
the forecast is causal by construction and the script says so on every row.
R1-R3 are built for the block-diagonal ridge (``blk2``, the deck's headline)
and for the OLS-HAR incumbent (``a0``); the headline honest forecast, fixed
before anything is scored, is ``blk2_R2`` - blk2 because it is proposal 34's
own headline, R2 because it is the richest construction that keeps every
session (R3 spends the first one on its lag).

Part A - the honest remaining-window scoreboard
-----------------------------------------------
On proposal 34's cells (the book's 865 expirations x the ten stamps 11:00 ..
15:30, 8650 cells whole and 4890 in the daily-0DTE era), target the variance
realized from the stamp to the close, scored by QLIKE and by the log-ratio
bias, per stamp and pooled, whole sample and era:

  (1) implied slice        ``iv_hourly_used^2 h_rem`` - pure market, as in 34
  (2) rv_hat / w_slice     34's construction, reproduced as a gate
  (3) honest F_rem         R1/R2/R3 above, and ft_remaining.py's own object
  (4) combination          inverse-past-QLIKE weights on (1) and the headline
                           (3), per stamp, expanding, lagged, 63-day minimum
  (5) bias-corrected       (1) times the expanding lagged mean of realized
      implied              over implied at that stamp, 63-day minimum

The pooled QLIKE differences (3) vs (1), (4) vs (1) and (4) vs (3) carry a
circular block bootstrap interval (B = 2000, block 21, seed 0, days drawn
whole so every stamp of a drawn session travels together).  Every table also
carries a ``pooled_1100_1500`` row, the same pooling over the nine stamps
that have a rebalance ahead of them, for a reader who wants the 15:30 stamp
(whose remaining window is one bar long) out of the pool.

Part B - the delta hedge with the honest forecast
-------------------------------------------------
Proposal 34's experiment, unchanged except for the number inside
``package_delta``: entry premium, buy-back and hedge timing are identical,
the hedge rebuilt with the implied total volatility reproduces
``live.ibkr.parity.replay_day`` to 1e-12, and the primary book is the
crossed-quoted 15:30 exit read on the era.

  V0  implied       the book: the re-inverted implied total volatility
  V7  honest        sqrt(F_rem), one variant per blk2 construction
  V8  combination   sqrt of (4)
  V9  implied_bc    sqrt of (5)
  V10 blend         sqrt(0.5 V7^2 + 0.5 V0^2) on the headline honest forecast

A variant falls back to V0 wherever its own input is missing or the stamp's
vendor implied volatility sat on the solver's bracket node; the fallbacks are
counted.  A variant "improves" only if the era bootstrap interval of its
Sharpe difference against V0 on the primary book excludes zero.  Turnover and
the repo's 0.5 bp hedge charge are REPORTED on every row and charged to none.

The gate runs first and nothing new is computed until it passes: proposal
32's published book numbers (865 days, crossed Sharpe 2.252546754993 whole
and 2.513132263902 in the era), the hedge identity against the live engine,
proposal 32's trend/whipsaw tercile cut points, and proposal 34's two pooled
era QLIKEs.  Two further gates guard the new object: the expanding fit's Gram
accumulation is checked against a direct least-squares solve at sampled fit
positions, and a leak canary rebuilds sampled forecasts from a history
TRUNCATED at the scored session and demands they do not move.

Run:  python writeup/intraday_proposals/36_remaining_window_honest.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import ft_remaining as ftr  # noqa: E402
from live.ibkr.parity import research_modules  # noqa: E402
from live.ibkr.pricing import package_delta  # noqa: E402


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p30 = _load_module(HERE / "30_skip_day_rule.py", "p36_p30")
p32 = _load_module(HERE / "32_loss_anatomy.py", "p36_p32")
asl = p30.asl

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "36"

ENTRY = p32.ENTRY
CLOSE = p32.CLOSE
SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0
ANN = p32.ANN

#: The panel's twelve scored trade bars, bar-START labelled: the bar stamped
#: ``c`` runs from ``c`` to ``c + 30 minutes``, so the session's scored tape
#: runs 10:00 -> 16:00 and the remaining window at ``c`` is ``c -> 16:00``.
TRADE_CLOCKS: tuple[str, ...] = tuple(
    f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)
)
#: The forecasts whose one-step output feeds the rest-of-session regressions.
FORECAST_TAGS: tuple[str, ...] = ("blk2", "a0")
#: Proposal 34's headline forecast, kept here.
HEADLINE_TAG = "blk2"
#: The rest-of-session constructions of experiments/multihorizon_ttc.py.
CONSTRUCTIONS: tuple[str, ...] = ("R1", "R2", "R3")
#: The honest forecast the combination, the blend and the V7/V8 hedges use.
#: Fixed before anything is scored: blk2 is proposal 34's headline model, R2
#: is the richest construction that keeps every session.
HONEST_TAG = "blk2_R2"
#: The blk2 constructions that are also tried inside the hedge delta.
HEDGE_HONEST_TAGS: tuple[str, ...] = tuple(f"{HEADLINE_TAG}_{c}" for c in CONSTRUCTIONS)
#: V10: the weight the honest variance carries in the blended variance.
BLEND_WEIGHT = 0.5

#: The repo's expanding warm-up, 63 sessions: the ``min_periods`` of section
#: 5b's remaining-share profile, of proposal 30's expanding quantiles, of
#: proposal 34's calibration and of experiments/multihorizon_ttc.py's fits.
WARMUP = p30.WARMUP_SESSIONS

#: The books this proposal scores.  The first is the primary.
BOOKS: tuple[str, ...] = ("exit_crossed", "exit_mid", "hold")
PRIMARY_BOOK = "exit_crossed"
SAMPLES: tuple[str, ...] = ("whole", "daily_era")
#: The sample the verdict is read on.
VERDICT_SAMPLE = "daily_era"

#: The deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
#: The percentile interval the verdict reads.
CI_PCT: tuple[float, float] = (2.5, 97.5)

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
GATE_CROSSED_WHOLE = p32.GATE_CROSSED_WHOLE
GATE_CROSSED_ERA = p32.GATE_CROSSED_ERA
GATE_TOL = p32.GATE_TOL
#: Proposal 32's full-sample trend/whipsaw tercile cut points, as published.
GATE_TREND_LO = 0.2076
GATE_TREND_HI = 0.4782
#: The cut points are published to four decimals, so they are gated there.
TREND_TOL = 5e-5
#: Proposal 34's pooled era QLIKE of the two forecasts it put on one scale
#: (results/atm_straddle_intraday_holdclose/proposals/34/forecast_quality.csv).
GATE_QLIKE_IMPLIED_ERA = 0.0836708686449815
GATE_QLIKE_OVER_W_ERA = 0.2110995701324163
#: The hedge identity against the live engine is a float path, not a bar.
EXACT_TOL = 1e-12
#: The expanding fit is accumulated as a Gram; the cross-check against a
#: direct least-squares solve on the same rows is a float path too, but the
#: normal equations square the design's condition number, so it is gated one
#: bar looser than the engine identity.
OLS_TOL = 1e-9
#: How many fit positions the Gram/least-squares cross-check samples.
OLS_CHECK_POINTS = 25
#: How many sessions the truncated-history leak canary rebuilds, and where in
#: the estimation frame it starts (the last quarter, so every sampled session
#: is past both warm-ups and carries a finite forecast).
CANARY_POINTS = 4
CANARY_START = 0.75


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"\nwrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# ------------------------------------------------------------- the grids ----
def clock_grid(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """A forecast panel column on the (session, trade clock) grid.

    Proposal 34's transform, unchanged: the panel's stamps are bar-end
    labelled, so the bar stamped ``t`` is the trade clock ``t - 30 minutes``,
    and only the fit mask's rows are kept.
    """
    pf = panel.reset_index()
    pf = pf[pf["in_fit"].to_numpy(dtype=bool)].copy()
    clk = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    pf["pdate"] = clk.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clk.dt.strftime("%H:%M")
    return pf.pivot_table(
        index="pdate", columns="phhmm", values=col, aggfunc="mean"
    ).sort_index()


def panel_grids() -> dict[str, Any]:
    """The rest-of-session tape and every one-step forecast, on panel sessions.

    Returns, all indexed by the panel's own session dates and columned by the
    twelve trade clocks:

    ``prof``      the realized variance of each trade bar;
    ``remaining`` the variance realized from the stamp to the close, the
                  inclusive reverse cumulative sum of ``prof`` - proposal
                  34's target object;
    ``sofar``     the variance already realized from the session's first
                  scored bar up to the stamp (zero-width at 10:00);
    ``w_slice``   section 5b's expanding lagged remaining-share profile;
    ``rv_hat``    per tag, the panel's second-order-mapped next-bar forecast;
    ``ft_rem``    experiments/ft_remaining.py's own F_rem, blk2.
    """
    paths = asl.yhat_paths(ROOT)
    panels = {
        tag: asl.load_yhat_panel(paths[tag]).set_index("t") for tag in FORECAST_TAGS
    }
    base = panels[HEADLINE_TAG]
    for tag, pan in panels.items():
        shared = base.index.intersection(pan.index)
        a = base.loc[shared, "rv_raw"].to_numpy(float)
        b = pan.loc[shared, "rv_raw"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.array_equal(a[ok], b[ok]), f"{tag} carries a different realized tape"
    clocks = list(TRADE_CLOCKS)
    prof, w_slice = p30.panel_profile(base, clocks)
    assert list(prof.columns) == clocks, list(prof.columns)

    rv = prof[clocks].to_numpy(float)
    remaining = pd.DataFrame(
        rv[:, ::-1].cumsum(axis=1)[:, ::-1], index=prof.index, columns=clocks
    )
    sofar = pd.DataFrame(
        np.concatenate([np.zeros((rv.shape[0], 1)), rv.cumsum(axis=1)[:, :-1]], axis=1),
        index=prof.index,
        columns=clocks,
    )
    rv_hat = {
        tag: clock_grid(pan, "rv_hat").reindex(columns=clocks)
        for tag, pan in panels.items()
    }

    ft = ftr.ft_remaining(str(paths[HEADLINE_TAG]), min_days=WARMUP)
    ft_rem = ft.pivot_table(
        index="day", columns="clock", values="F_rem", aggfunc="mean"
    ).reindex(columns=clocks)

    print(
        f"\npanel: {len(prof)} scored sessions {prof.index.min().date()} .. "
        f"{prof.index.max().date()} on {len(clocks)} trade bars "
        f"{clocks[0]} .. {clocks[-1]} (bar-start labelled, naive ET); the "
        f"remaining window at a stamp is that stamp to the 16:00 close"
    )
    for tag in FORECAST_TAGS:
        print(
            f"  rv_hat[{tag}]: {int(rv_hat[tag].notna().all(axis=1).sum())} sessions "
            f"complete on all {len(clocks)} clocks, of {len(rv_hat[tag])}"
        )
    print(
        f"  ft_remaining.py F_rem[{HEADLINE_TAG}]: "
        f"{int(ft_rem.notna().all(axis=1).sum())} sessions complete, of {len(ft_rem)}"
    )
    return {
        "prof": prof,
        "remaining": remaining,
        "sofar": sofar,
        "w_slice": w_slice,
        "rv_hat": rv_hat,
        "ft_rem": ft_rem,
    }


# ------------------------------------------------- the honest regressions ----
def expanding_ols_predict(x: np.ndarray, y: np.ndarray, min_fit: int) -> np.ndarray:
    """One-step-ahead fitted values of an expanding OLS, row by row.

    ``x`` carries the intercept column.  The prediction at row ``j`` uses the
    least-squares coefficients of rows ``0 .. j - 1`` only, so it never sees
    its own row or any later one; rows before ``min_fit`` return NaN.  The
    normal equations are accumulated, so the pass is linear in the number of
    rows; ``check_expanding_ols`` verifies the accumulation against a direct
    least-squares solve on the same rows.
    """
    n, p = x.shape
    out = np.full(n, np.nan)
    gram = np.zeros((p, p))
    rhs = np.zeros(p)
    for j in range(n):
        if j >= min_fit:
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            out[j] = float(x[j] @ beta)
        gram += np.outer(x[j], x[j])
        rhs += x[j] * y[j]
    return out


def check_expanding_ols(
    x: np.ndarray, y: np.ndarray, min_fit: int, got: np.ndarray, label: str
) -> float:
    """Gate the Gram accumulation against a direct solve at sampled rows."""
    n = x.shape[0]
    pos = np.unique(np.linspace(min_fit, n - 1, OLS_CHECK_POINTS).round().astype(int))
    worst = 0.0
    for j in pos:
        beta = np.linalg.lstsq(x[:j], y[:j], rcond=None)[0]
        worst = max(worst, abs(float(x[j] @ beta) - float(got[j])))
    assert worst < OLS_TOL, (label, worst)
    return worst


def causal_scale(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``f`` times the expanding, one-session-lagged mean of realized/forecast.

    ``experiments/multihorizon_ttc.py``'s ``_causal_scale``: the QLIKE-optimal
    multiplicative correction, fitted on strictly earlier sessions with the
    repo's 63-session minimum.
    """
    r = pd.Series(np.where(f > 0.0, y / np.where(f > 0.0, f, 1.0), np.nan))
    c = r.expanding(min_periods=WARMUP).mean().shift(1).to_numpy(float)
    return c * f


def one_stamp_forecast(des: np.ndarray, ly: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """One stamp's honest forecast: the expanding fit, then the causal rescale."""
    return causal_scale(yv, np.exp(expanding_ols_predict(des, ly, WARMUP)))


def causality_canary(
    des: np.ndarray, ly: np.ndarray, yv: np.ndarray, full: np.ndarray, label: str
) -> float:
    """Rebuild sampled forecasts from a TRUNCATED history and demand equality.

    The forecast at session ``j`` is recomputed with the frame cut off after
    ``j`` - every later session deleted, ``j``'s own realized remaining window
    still in the frame but reachable only by a statistic that looks forward.
    A pipeline that peeks anywhere, in the fit or in the rescale, moves; this
    one must not move at all, so the bar is the float-path bar.
    """
    n = des.shape[0]
    pos = np.unique(
        np.linspace(int(round(CANARY_START * n)), n - 1, CANARY_POINTS)
        .round()
        .astype(int)
    )
    worst = 0.0
    for j in pos:
        cut = one_stamp_forecast(des[: j + 1], ly[: j + 1], yv[: j + 1])
        assert np.isfinite(cut[-1]) and np.isfinite(full[j]), (label, j)
        worst = max(worst, abs(float(cut[-1]) - float(full[j])))
    assert worst < EXACT_TOL, (label, worst)
    return worst


def honest_forecasts(grids: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """The rest-of-session regressions R1/R2/R3, per tag, on panel sessions.

    One fit per (tag, construction, stamp).  The estimation frame is the
    sessions whose twelve trade bars all carry a positive realized variance
    AND a positive one-step forecast; every other session is left without a
    forecast rather than filled.
    """
    prof = grids["prof"]
    rem = grids["remaining"]
    sofar = grids["sofar"]
    clocks = list(TRADE_CLOCKS)
    out: dict[str, pd.DataFrame] = {}
    print(
        f"\nthe honest rest-of-session regressions: log y ~ a_c + b_c log rv_hat(c)"
        f" (R1), + d_c log RV(open->c) (R2), + e_c log y(prev session) (R3); "
        f"expanding fit on strictly earlier sessions, minimum {WARMUP}, then the "
        f"causal lagged realized-over-forecast rescale (minimum {WARMUP} again)"
    )
    for tag in FORECAST_TAGS:
        rvh = grids["rv_hat"][tag]
        frame = prof.index.intersection(rvh.index)
        ok = (
            prof.loc[frame, clocks].gt(0.0).all(axis=1)
            & rvh.loc[frame, clocks].gt(0.0).all(axis=1)
        ).to_numpy(bool)
        days = pd.DatetimeIndex(frame[ok])
        ly = np.log(rem.loc[days, clocks].to_numpy(float))
        lv = np.log(rvh.loc[days, clocks].to_numpy(float))
        so = sofar.loc[days, clocks].to_numpy(float)
        # the 10:00 bar has no session history behind it, so its log is left
        # undefined; no book stamp reads that column.
        lso = np.where(so > 0.0, np.log(np.where(so > 0.0, so, 1.0)), np.nan)
        yv = rem.loc[days, clocks].to_numpy(float)
        n = len(days)
        one = np.ones(n)
        worst = 0.0
        leak = 0.0
        built = {c: np.full((n, len(clocks)), np.nan) for c in CONSTRUCTIONS}
        for j, clock in enumerate(clocks):
            if clock not in SESSION:
                continue
            designs: dict[str, np.ndarray] = {
                "R1": np.column_stack([one, lv[:, j]]),
                "R2": np.column_stack([one, lv[:, j], lso[:, j]]),
            }
            lprev = np.concatenate([[np.nan], ly[:-1, j]])
            designs["R3"] = np.column_stack([one, lv[:, j], lso[:, j], lprev])
            for name, des in designs.items():
                keep = np.isfinite(des).all(axis=1)
                pred = np.full(n, np.nan)
                sub = expanding_ols_predict(des[keep], ly[keep, j], WARMUP)
                worst = max(
                    worst,
                    check_expanding_ols(
                        des[keep], ly[keep, j], WARMUP, sub, f"{tag}_{name}_{clock}"
                    ),
                )
                pred[keep] = sub
                scaled = causal_scale(yv[keep, j], np.exp(sub))
                if tag == HEADLINE_TAG and clock == ENTRY:
                    leak = max(
                        leak,
                        causality_canary(
                            des[keep],
                            ly[keep, j],
                            yv[keep, j],
                            scaled,
                            f"{tag}_{name}_{clock}",
                        ),
                    )
                built[name][keep, j] = scaled
        for name in CONSTRUCTIONS:
            out[f"{tag}_{name}"] = pd.DataFrame(built[name], index=days, columns=clocks)
        print(
            f"  {tag}: estimation frame {n} complete sessions "
            f"{days.min().date()} .. {days.max().date()}; the Gram accumulation "
            f"matches a direct least-squares solve to {worst:.3e} at "
            f"{OLS_CHECK_POINTS} sampled fit positions per regression (bar "
            f"{OLS_TOL:.0e})  OK"
        )
        if tag == HEADLINE_TAG:
            print(
                f"  CANARY {tag} at {ENTRY}: rebuilt from a history truncated at the "
                f"scored session, {CANARY_POINTS} sampled sessions per construction, "
                f"largest move {leak:.3e} (bar {EXACT_TOL:.0e})  OK - the fit and "
                f"the rescale see strictly earlier sessions only"
            )
        for name in CONSTRUCTIONS:
            g = out[f"{tag}_{name}"][list(SESSION)]
            print(
                f"    {tag}_{name}: {int(g.notna().all(axis=1).sum())} sessions "
                f"carry a forecast at all {len(SESSION)} book stamps, first "
                f"{g.dropna(how='any').index.min().date()}"
            )
    return out


# ---------------------------------------------------------------- QLIKE ----
def qlike_cells(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Patton's QLIKE ``y/f - log(y/f) - 1`` cell by cell, NaN where undefined."""
    m = np.isfinite(y) & np.isfinite(f) & (y > 0.0) & (f > 0.0)
    out = np.full(y.shape, np.nan)
    r = np.where(m, y / np.where(m, f, 1.0), np.nan)
    out[m] = (r - np.log(r) - 1.0)[m]
    return out


def qlike(y: np.ndarray, f: np.ndarray) -> tuple[float, float, int, float]:
    """Proposal 34's scalar summary: QLIKE, log-ratio bias, n, mean ratio."""
    m = np.isfinite(y) & np.isfinite(f) & (y > 0.0) & (f > 0.0)
    if int(m.sum()) < 2:
        return float("nan"), float("nan"), int(m.sum()), float("nan")
    r = y[m] / f[m]
    return (
        float(np.mean(r - np.log(r) - 1.0)),
        float(np.mean(np.log(f[m]) - np.log(y[m]))),
        int(m.sum()),
        float(f[m].mean() / y[m].mean()),
    )


def expanding_lagged_mean(a: np.ndarray, min_obs: int) -> np.ndarray:
    """Column-wise expanding mean over strictly earlier rows, NaN before warm-up."""
    return (
        pd.DataFrame(a).expanding(min_periods=min_obs).mean().shift(1).to_numpy(float)
    )


# ----------------------------------------------------------- the variants ----
def total_vol_variants(
    var0: np.ndarray, book: dict[str, np.ndarray], censored: np.ndarray
) -> list[dict[str, Any]]:
    """Every total-volatility grid the hedge delta is tried with.

    Proposal 34's fallback rule, unchanged: a variant's variance falls back to
    the book's implied wherever its own input is missing, non-positive, or the
    stamp's vendor implied volatility sat on the solver's bracket node, so
    every variant hedges the same days and differs from the book only where it
    has something to say.
    """

    def usable(x: np.ndarray) -> np.ndarray:
        return np.isfinite(x) & (x > 0.0) & ~censored

    out: list[dict[str, Any]] = [
        {
            "variant": "V0_implied",
            "family": "V0 implied",
            "description": "the book: the re-inverted implied total volatility",
            "variance": var0,
            "n_fallback": 0,
        }
    ]
    for tag in HEDGE_HONEST_TAGS:
        v = book[tag]
        ok = usable(v)
        out.append(
            {
                "variant": f"V7_honest_{tag}",
                "family": "V7 honest",
                "description": f"sqrt(F_rem) of the {tag} rest-of-session regression",
                "variance": np.where(ok, v, var0),
                "n_fallback": int((~ok).sum()),
            }
        )
    combo = book["combination"]
    ok_c = usable(combo)
    out.append(
        {
            "variant": "V8_combination",
            "family": "V8 combination",
            "description": (
                f"sqrt of the inverse-past-QLIKE combination of the implied "
                f"slice and {HONEST_TAG}"
            ),
            "variance": np.where(ok_c, combo, var0),
            "n_fallback": int((~ok_c).sum()),
        }
    )
    bc = book["implied_bc"]
    ok_b = usable(bc)
    out.append(
        {
            "variant": "V9_implied_bc",
            "family": "V9 bias-corrected implied",
            "description": (
                f"sqrt of the implied slice times its expanding lagged "
                f"{WARMUP}-day realized-over-implied mean at that stamp"
            ),
            "variance": np.where(ok_b, bc, var0),
            "n_fallback": int((~ok_b).sum()),
        }
    )
    head = book[HONEST_TAG]
    ok_h = usable(head)
    out.append(
        {
            "variant": f"V10_blend_{BLEND_WEIGHT:.2f}",
            "family": "V10 blend",
            "description": (
                f"sqrt({BLEND_WEIGHT:.2f} V7^2 + {1.0 - BLEND_WEIGHT:.2f} V0^2), "
                f"{HONEST_TAG}"
            ),
            "variance": np.where(
                ok_h,
                BLEND_WEIGHT * np.where(ok_h, head, var0) + (1.0 - BLEND_WEIGHT) * var0,
                var0,
            ),
            "n_fallback": int((~ok_h).sum()),
        }
    )
    for row in out:
        row["total_vol"] = np.sqrt(row["variance"])
    return out


# ------------------------------------------------------------- the hedge ----
def traded_matrix(pos: np.ndarray) -> np.ndarray:
    """|delta traded| at each stamp: the opening trade, the rebalances, the flatten."""
    n = pos.shape[0]
    prev = np.concatenate([np.zeros((n, 1)), pos[:, :-1]], axis=1)
    return np.abs(
        np.concatenate([pos, np.zeros((n, 1))], axis=1)
        - np.concatenate([prev, pos[:, -1:]], axis=1)
    )


def hedge_legs(total_vol: np.ndarray, tape: dict[str, Any]) -> dict[str, np.ndarray]:
    """Proposal 34's hedge leg, turnover and charge of one volatility grid."""
    spot = tape["spot"]
    kc = tape["Kc"]
    kp = tape["Kp"]
    n, m = total_vol.shape
    dlt = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dlt[i, j] = package_delta(total_vol[i, j], spot[i, j], kc[i], kp[i])
    d_spot = spot[:, 1:] - spot[:, :-1]
    hedge_exit = (dlt[:, :-1] * d_spot).sum(axis=1)
    hedge_hold = hedge_exit + dlt[:, -1] * (tape["S_close"] - spot[:, -1])
    bp = p32.HEDGE_COST_BP * 1e-4
    tr_exit = traded_matrix(dlt[:, :-1])
    tr_hold = traded_matrix(dlt)
    px_hold = np.concatenate([spot, tape["S_close"][:, None]], axis=1)
    return {
        "delta": dlt,
        "hedge_exit": hedge_exit,
        "hedge_hold": hedge_hold,
        "turnover_exit": tr_exit.sum(axis=1),
        "turnover_hold": tr_hold.sum(axis=1),
        "cost_exit": (tr_exit * spot).sum(axis=1) * bp,
        "cost_hold": (tr_hold * px_hold).sum(axis=1) * bp,
    }


def book_returns(
    tape: dict[str, Any], mid: np.ndarray, ask: np.ndarray, legs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The three books' daily returns under one hedge."""
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    return {
        "exit_crossed": (-(ask[:, -1] - eb) + legs["hedge_exit"]) / em,
        "exit_mid": (-(mid[:, -1] - em) + legs["hedge_exit"]) / em,
        "hold": (-(tape["settle"] - em) + legs["hedge_hold"]) / em,
    }


def book_turnover(legs: dict[str, np.ndarray], book: str) -> np.ndarray:
    """The book's own hedge turnover (the hold book flattens at the close)."""
    return legs["turnover_hold"] if book == "hold" else legs["turnover_exit"]


def book_cost(legs: dict[str, np.ndarray], book: str) -> np.ndarray:
    """The book's own hedge charge in index points, at the repo's basis points."""
    return legs["cost_hold"] if book == "hold" else legs["cost_exit"]


# ------------------------------------------------------------ statistics ----
def stat_row(x: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """n / mean / sd / Sharpe_ann / t / MaxDD / worst day of a daily series."""
    ok = np.isfinite(x)
    v = x[ok]
    d = dates[ok]
    n = int(v.size)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n >= 2 else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(n)) * mean / sd if live else float("nan"),
        "MaxDD": p32.maxdd(v),
        "worst_day": float(v.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(d[int(np.argmin(v))]).date()) if n else "",
    }


_BOOT_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """The deck's circular block bootstrap index, drawn once per sample size."""
    if n not in _BOOT_IDX:
        _BOOT_IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng(BOOT_SEED), n, BOOT_BLOCK, BOOT_B
        )
    return _BOOT_IDX[n]


def sharpe_diff_ci(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Circular block bootstrap of Sharpe(x) - Sharpe(y) on the common days."""
    ok = np.isfinite(x) & np.isfinite(y)
    a = x[ok]
    b = y[ok]
    n = int(a.size)
    hat = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * ANN
    idx = boot_idx(n)
    aa = a[idx]
    bb = b[idx]
    ds = (
        aa.mean(axis=1) / aa.std(axis=1, ddof=1)
        - bb.mean(axis=1) / bb.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(ds, list(CI_PCT))
    return {
        "n": n,
        "dSharpe": float(hat),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "pct_draws_positive": float(100.0 * (ds > 0).mean()),
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def qlike_diff_ci(la: np.ndarray, lb: np.ndarray) -> dict[str, Any]:
    """Circular block bootstrap of the pooled QLIKE difference mean(la) - mean(lb).

    The loss matrices are (session x stamp); a draw resamples whole SESSIONS,
    so every stamp of a drawn session travels together and the interval
    carries the within-day dependence.  The pooled statistic is the ratio of
    the drawn sessions' summed difference to their counted cells, which is the
    pooled mean over cells whether or not every session carries every stamp.
    """
    m = np.isfinite(la) & np.isfinite(lb)
    d = np.where(m, la - lb, 0.0)
    s = d.sum(axis=1)
    c = m.sum(axis=1).astype(float)
    keep = c > 0
    s = s[keep]
    c = c[keep]
    n = int(s.size)
    hat = float(s.sum() / c.sum())
    idx = boot_idx(n)
    ds = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(ds, list(CI_PCT))
    return {
        "n_days": n,
        "n_cells": int(c.sum()),
        "dQLIKE": hat,
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "pct_draws_negative": float(100.0 * (ds < 0).mean()),
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def paired_row(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """The paired daily difference x - y: mean, plain t, HAC t, days improved."""
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    n = int(d.size)
    sd = float(d.std(ddof=1)) if n >= 2 else float("nan")
    hac_t, lag = asl.newey_west_t(pd.Series(d))
    return {
        "n": n,
        "mean_diff": float(d.mean()) if n else float("nan"),
        "sd_diff": sd,
        "t_diff": float(np.sqrt(n)) * float(d.mean()) / sd
        if (n and np.isfinite(sd) and sd > 0.0)
        else float("nan"),
        "hac_t_diff": float(hac_t),
        "hac_lag": int(lag),
        "frac_days_improved": float((d > 0).mean()) if n else float("nan"),
        "frac_days_unchanged": float((d == 0).mean()) if n else float("nan"),
    }


# -------------------------------------------------------------- the gate ----
def gate(
    tape: dict[str, Any],
    mid: np.ndarray,
    ask: np.ndarray,
    dates: pd.DatetimeIndex,
    day: pd.DataFrame,
    legs0: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, float]:
    """Reproduce every published number, and the engine's own hedge, first."""
    p32.gate(tape, mid, ask, dates)

    print("\nGATE - the hedge leg rebuilt with the implied total volatility")
    d_exit = float(np.max(np.abs(legs0["hedge_exit"] - tape["hedge_ref"])))
    d_hold = float(np.max(np.abs(legs0["hedge_hold"] - tape["hedge_hold"])))
    assert d_exit < EXACT_TOL, d_exit
    assert d_hold < EXACT_TOL, d_hold
    print(
        f"  sum delta dS to the {CLOSE} buy-back matches replay_day to {d_exit:.3e}  OK"
    )
    print(f"  the same sum carried into the settlement matches to {d_hold:.3e}  OK")
    r0 = book_returns(tape, mid, ask, legs0)
    have = p32.sharpe_ann(pd.Series(r0[PRIMARY_BOOK], index=dates))
    assert abs(have - GATE_CROSSED_WHOLE) < GATE_TOL, (have, GATE_CROSSED_WHOLE)
    print(
        f"  book {PRIMARY_BOOK} Sharpe_ann whole {have:>16.12f}  reference "
        f"{GATE_CROSSED_WHOLE:.12f}  OK"
    )
    era = np.asarray(dates >= ERA0, dtype=bool)
    have_era = p32.sharpe_ann(pd.Series(r0[PRIMARY_BOOK][era], index=dates[era]))
    assert abs(have_era - GATE_CROSSED_ERA) < GATE_TOL, (have_era, GATE_CROSSED_ERA)
    print(
        f"  book {PRIMARY_BOOK} Sharpe_ann era   {have_era:>16.12f}  reference "
        f"{GATE_CROSSED_ERA:.12f}  OK"
    )

    q1, q2 = day["trend_ratio"].quantile([1 / 3, 2 / 3]).to_numpy()
    assert abs(float(q1) - GATE_TREND_LO) < TREND_TOL, q1
    assert abs(float(q2) - GATE_TREND_HI) < TREND_TOL, q2
    lab = np.full(len(dates), "2_mixed", dtype=object)
    lab[day["trend_ratio"].to_numpy(float) <= q1] = "1_choppy"
    lab[day["trend_ratio"].to_numpy(float) > q2] = "3_trend"
    print(
        "\nGATE - proposal 32's shape terciles of "
        "|sum of bar returns| / sum |bar returns|"
    )
    print(
        f"  cut points {float(q1):.6f} / {float(q2):.6f}  reference "
        f"{GATE_TREND_LO:.4f} / {GATE_TREND_HI:.4f}  OK; "
        f"{int((lab == '1_choppy').sum())} choppy, {int((lab == '2_mixed').sum())} "
        f"mixed, {int((lab == '3_trend').sum())} trend days"
    )
    return lab, float(q1), float(q2)


def gate_proposal_34(
    realized: np.ndarray, implied: np.ndarray, over_w: np.ndarray, era: np.ndarray
) -> None:
    """Reproduce proposal 34's two pooled era QLIKEs before anything new is read."""
    print("\nGATE - proposal 34's pooled era remaining-window QLIKE")
    for label, f, want in (
        ("implied slice", implied, GATE_QLIKE_IMPLIED_ERA),
        (f"{HEADLINE_TAG} rv_hat / w_slice", over_w, GATE_QLIKE_OVER_W_ERA),
    ):
        got, _, n, _ = qlike(realized[era].ravel(), f[era].ravel())
        assert abs(got - want) < GATE_TOL, (label, got, want)
        print(f"  {label:<28s} {got:>18.12f} on {n} cells  reference {want:.12f}  OK")
    print("GATE PASSED")


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, ent, dates = p32.load_panel()
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    tape = p32.build_tape(panel, ent, dates)
    _, mid, ask = p32.quote_grids(ent, dates)
    assert np.max(np.abs(mid[:, 0] - tape["entry_mid"])) < EXACT_TOL, "11:00 mid"
    _, day = p32.attribute(tape, mid, ask, dates)

    cols = list(SESSION)
    var0 = (tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])) ** 2
    legs0 = hedge_legs(np.sqrt(var0), tape)
    lab, q1, q2 = gate(tape, mid, ask, dates, day, legs0)

    grids = panel_grids()
    realized = grids["remaining"].reindex(index=dates, columns=cols).to_numpy(float)
    over_w = {
        tag: (grids["rv_hat"][tag] / grids["w_slice"][list(TRADE_CLOCKS)])
        .reindex(index=dates, columns=cols)
        .to_numpy(float)
        for tag in FORECAST_TAGS
    }
    era = np.asarray(dates >= ERA0, dtype=bool)
    gate_proposal_34(realized, var0, over_w[HEADLINE_TAG], era)

    assert np.isfinite(realized).all(), "a book stamp has no realized remaining window"
    honest = honest_forecasts(grids)
    book: dict[str, np.ndarray] = {
        tag: g.reindex(index=dates, columns=cols).to_numpy(float)
        for tag, g in honest.items()
    }
    book["ft_remaining"] = (
        grids["ft_rem"].reindex(index=dates, columns=cols).to_numpy(float)
    )

    # ---- (4) the causal inverse-past-QLIKE combination of (1) and (3) -------
    l_imp = qlike_cells(realized, var0)
    l_hon = qlike_cells(realized, book[HONEST_TAG])
    q_imp = expanding_lagged_mean(l_imp, WARMUP)
    q_hon = expanding_lagged_mean(l_hon, WARMUP)
    live = (
        np.isfinite(q_imp)
        & np.isfinite(q_hon)
        & (q_imp > 0.0)
        & (q_hon > 0.0)
        & np.isfinite(book[HONEST_TAG])
        & (book[HONEST_TAG] > 0.0)
    )
    w_hon = np.full(realized.shape, np.nan)
    inv_i = np.where(live, 1.0 / np.where(live, q_imp, 1.0), np.nan)
    inv_h = np.where(live, 1.0 / np.where(live, q_hon, 1.0), np.nan)
    w_hon[live] = (inv_h / (inv_h + inv_i))[live]
    book["combination"] = np.where(
        live, w_hon * book[HONEST_TAG] + (1.0 - w_hon) * var0, np.nan
    )

    # ---- (5) the implied slice under its own causal bias correction --------
    scale_imp = expanding_lagged_mean(realized / var0, WARMUP)
    book["implied_bc"] = np.where(
        np.isfinite(scale_imp) & (scale_imp > 0.0), scale_imp * var0, np.nan
    )

    print(
        f"\nthe causal combination (4): inverse expanding-lagged-QLIKE weights on "
        f"the implied slice and {HONEST_TAG}, per stamp, minimum {WARMUP} book "
        f"days; {int(live.sum())} of {live.size} cells carry a weight, the first "
        f"on {pd.Timestamp(dates[np.argmax(live.any(axis=1))]).date()}, and "
        f"{int(era.sum()) * len(cols) - int(live[era].sum())} of "
        f"{int(era.sum()) * len(cols)} era cells do not"
    )
    for j, clock in enumerate(cols):
        wv = w_hon[:, j]
        print(
            f"  {clock}  median weight on {HONEST_TAG} "
            f"{np.nanmedian(wv):.6f}   cells without a weight "
            f"{int((~np.isfinite(wv)).sum())}"
        )
    sc_ok = np.isfinite(scale_imp)
    print(
        f"\nthe causal bias correction (5): expanding lagged mean of realized over "
        f"implied, per stamp, minimum {WARMUP} book days; "
        f"{int(sc_ok.sum())} of {sc_ok.size} cells carry a scale"
    )
    for j, clock in enumerate(cols):
        sv = scale_imp[:, j]
        print(
            f"  {clock}  median variance scale {np.nanmedian(sv):.6f}   "
            f"(volatility scale {np.sqrt(np.nanmedian(sv)):.6f})   "
            f"cells without a scale {int((~np.isfinite(sv)).sum())}"
        )

    # =================================================== Part A: scoreboard ==
    sources: list[tuple[str, str, np.ndarray]] = [
        ("implied", "(1) implied slice iv_hourly_used^2 h_rem", var0)
    ]
    for tag in FORECAST_TAGS:
        sources.append(
            (f"{tag}_over_w", f"(2) {tag} rv_hat / w_slice (proposal 34)", over_w[tag])
        )
    sources.append(
        (
            f"{HEADLINE_TAG}_ft_remaining",
            "(3) experiments/ft_remaining.py F_rem = f_next / w_next",
            book["ft_remaining"],
        )
    )
    for tag in FORECAST_TAGS:
        for c in CONSTRUCTIONS:
            sources.append(
                (
                    f"{tag}_{c}",
                    f"(3) honest rest-of-session regression {c} on {tag}",
                    book[f"{tag}_{c}"],
                )
            )
    sources.append(
        (
            "combination",
            f"(4) inverse-past-QLIKE mix of (1) and {HONEST_TAG}",
            book["combination"],
        )
    )
    sources.append(
        (
            "implied_bc",
            "(5) implied slice under its causal bias correction",
            book["implied_bc"],
        )
    )

    masks: dict[str, np.ndarray] = {
        "whole": np.ones(len(dates), dtype=bool),
        "daily_era": era,
    }
    nine = [c for c in cols if c != CLOSE]
    nine_idx = np.array([cols.index(c) for c in nine])
    rows: list[dict[str, Any]] = []
    for name, desc, f in sources:
        for sample, m in masks.items():
            for j, clock in enumerate(cols):
                q, bias, n, ratio = qlike(realized[m, j], f[m, j])
                rows.append(
                    {
                        "forecast": name,
                        "description": desc,
                        "causal": True,
                        "sample": sample,
                        "clock": clock,
                        "h_rem": H_REM[j],
                        "n": n,
                        "QLIKE": q,
                        "bias_log_f_over_realized": bias,
                        "mean_f_over_mean_realized": ratio,
                        "median_forecast": float(np.nanmedian(f[m, j])),
                        "median_realized": float(np.nanmedian(realized[m, j])),
                    }
                )
            for label, sel in (
                ("pooled", np.arange(len(cols))),
                ("pooled_1100_1500", nine_idx),
            ):
                q, bias, n, ratio = qlike(
                    realized[np.ix_(m, sel)].ravel(), f[np.ix_(m, sel)].ravel()
                )
                rows.append(
                    {
                        "forecast": name,
                        "description": desc,
                        "causal": True,
                        "sample": sample,
                        "clock": label,
                        "h_rem": float("nan"),
                        "n": n,
                        "QLIKE": q,
                        "bias_log_f_over_realized": bias,
                        "mean_f_over_mean_realized": ratio,
                        "median_forecast": float(np.nanmedian(f[np.ix_(m, sel)])),
                        "median_realized": float(
                            np.nanmedian(realized[np.ix_(m, sel)])
                        ),
                    }
                )
    fq = pd.DataFrame(rows)
    write(fq, "forecast_quality.csv", "the honest remaining-window scoreboard")
    for sample in SAMPLES:
        sub = fq[fq["sample"] == sample]
        show(sub, f"forecast_quality.csv  |  sample {sample}")
        print(f"\nQLIKE by stamp, sample {sample} (lower is better)")
        with pd.option_context("display.width", 250, "display.max_columns", 80):
            print(
                sub.pivot(index="clock", columns="forecast", values="QLIKE").to_string(
                    float_format=lambda x: f"{x:.5f}"
                )
            )
        print(f"\nlog-ratio bias log(forecast / realized) by stamp, sample {sample}")
        with pd.option_context("display.width", 250, "display.max_columns", 80):
            print(
                sub.pivot(
                    index="clock", columns="forecast", values="bias_log_f_over_realized"
                ).to_string(float_format=lambda x: f"{x:+.5f}")
            )
    show(
        fq[["forecast", "description"]].drop_duplicates("forecast"),
        "forecast_quality.csv  |  what each row forecasts",
    )

    # ---- the bootstrap of the pooled QLIKE differences ---------------------
    loss = {name: qlike_cells(realized, f) for name, _, f in sources}
    pairs_q: list[tuple[str, str, str]] = [
        ("(3) vs (1)", HONEST_TAG, "implied"),
        ("(4) vs (1)", "combination", "implied"),
        ("(4) vs (3)", "combination", HONEST_TAG),
    ]
    for tag in HEDGE_HONEST_TAGS:
        if tag != HONEST_TAG:
            pairs_q.append((f"(3) {tag} vs (1)", tag, "implied"))
    pairs_q.append((f"(3) vs (2) {HEADLINE_TAG}", HONEST_TAG, f"{HEADLINE_TAG}_over_w"))
    qrows: list[dict[str, Any]] = []
    for label, a, b in pairs_q:
        for sample, m in masks.items():
            row: dict[str, Any] = {
                "comparison": label,
                "forecast_a": a,
                "forecast_b": b,
                "sample": sample,
                "B": BOOT_B,
                "block": BOOT_BLOCK,
                "seed": BOOT_SEED,
                "ci_pct_lo": CI_PCT[0],
                "ci_pct_hi": CI_PCT[1],
            }
            row.update(qlike_diff_ci(loss[a][m], loss[b][m]))
            qrows.append(row)
    qboot = pd.DataFrame(qrows)
    write(
        qboot,
        "qlike_bootstrap.csv",
        "the bootstrap interval of the pooled QLIKE difference",
    )
    for sample in SAMPLES:
        show(
            qboot[qboot["sample"] == sample],
            f"qlike_bootstrap.csv  |  sample {sample}  |  {BOOT_B} circular block "
            f"draws of whole sessions, block {BOOT_BLOCK}, seed {BOOT_SEED}; "
            f"dQLIKE = QLIKE(a) - QLIKE(b), negative favours a",
            [
                "comparison",
                "forecast_a",
                "forecast_b",
                "n_days",
                "n_cells",
                "dQLIKE",
                "boot_lo",
                "boot_hi",
                "pct_draws_negative",
                "ci_excludes_zero",
            ],
        )

    # ======================================================= Part B: hedge ==
    _, rule_mod = research_modules(ROOT)
    flag = rule_mod.on_vendor_node(
        panel["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(panel["impl_volatility_p"])
    censored = (
        panel.assign(_cen=np.asarray(flag, dtype=float))
        .pivot_table(index="date", columns="hhmm", values="_cen", aggfunc="max")
        .reindex(index=dates, columns=cols)
        .fillna(1.0)
        .to_numpy(float)
        > 0
    )
    print(
        f"\ncensored implied volatility: {int(censored.sum())} of {censored.size} "
        f"stamp-cells on the book's days (those stamps hedge with V0)"
    )

    variants = total_vol_variants(var0, book, censored)
    print(
        f"\n{len(variants)} total-volatility variants: V0 (the book) and "
        f"{len(variants) - 1} causal candidates; headline honest forecast "
        f"{HONEST_TAG}, blend weight {BLEND_WEIGHT:.2f}"
    )
    legs = {v["variant"]: hedge_legs(v["total_vol"], tape) for v in variants}
    rets = {
        v["variant"]: book_returns(tape, mid, ask, legs[v["variant"]]) for v in variants
    }
    base = rets["V0_implied"]

    vrows: list[dict[str, Any]] = []
    for v in variants:
        name = v["variant"]
        for bk in BOOKS:
            turn = book_turnover(legs[name], bk)
            cost = book_cost(legs[name], bk) / tape["entry_mid"]
            r = rets[name][bk]
            d = r - base[bk]
            for sample, m in masks.items():
                row = {
                    "variant": name,
                    "family": v["family"],
                    "causal": True,
                    "description": v["description"],
                    "book": bk,
                    "sample": sample,
                    "n_fallback_stamps": v["n_fallback"],
                }
                row.update(stat_row(r[m], dates[m]))
                row["sd_diff_vs_V0"] = (
                    float(np.std(d[m], ddof=1)) if int(m.sum()) >= 2 else float("nan")
                )
                row["mean_abs_diff_vs_V0"] = float(np.mean(np.abs(d[m])))
                row["mean_turnover"] = float(np.mean(turn[m]))
                row["mean_hedge_cost_prem"] = float(np.mean(cost[m]))
                for cls, tg in (("1_choppy", "choppy"), ("3_trend", "trend")):
                    sel = m & (lab == cls)
                    sub_r = r[sel]
                    row[f"mean_{tg}"] = float(np.mean(sub_r))
                    row[f"Sharpe_{tg}"] = p32.sharpe_ann(pd.Series(sub_r))
                    row[f"n_{tg}"] = int(sel.sum())
                vrows.append(row)
    variants_df = pd.DataFrame(vrows)
    write(variants_df, "variants.csv", "the hedge volatilities, scored")
    vcols = [
        "variant",
        "n",
        "mean",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "worst_date",
        "sd_diff_vs_V0",
        "mean_abs_diff_vs_V0",
        "mean_turnover",
        "mean_hedge_cost_prem",
        "n_fallback_stamps",
        "mean_choppy",
        "Sharpe_choppy",
        "mean_trend",
        "Sharpe_trend",
    ]
    for bk in BOOKS:
        for sample in SAMPLES:
            sub = variants_df[
                (variants_df["book"] == bk) & (variants_df["sample"] == sample)
            ]
            show(sub, f"variants.csv  |  book {bk}  |  sample {sample}", vcols)
    show(
        variants_df[
            ["variant", "family", "description", "n_fallback_stamps"]
        ].drop_duplicates("variant"),
        "variants.csv  |  what each variant puts inside package_delta",
    )
    print(
        f"\nmean_hedge_cost_prem is the repo's {p32.HEDGE_COST_BP} bp charge on "
        f"S x |delta traded|, in units of the entry premium.  It is REPORTED, not "
        f"taken out of any book: every Sharpe, mean and difference in these tables "
        f"is gross of it, exactly as the published books are."
    )
    print(
        f"\nshape terciles used for the choppy / trend columns: "
        f"{q1:.6f} / {q2:.6f} of |sum of bar returns| / sum |bar returns|"
    )

    prows: list[dict[str, Any]] = []
    for v in variants:
        if v["variant"] == "V0_implied":
            continue
        for bk in BOOKS:
            for sample, m in masks.items():
                row = {
                    "variant": v["variant"],
                    "family": v["family"],
                    "book": bk,
                    "sample": sample,
                }
                row.update(paired_row(rets[v["variant"]][bk][m], base[bk][m]))
                prows.append(row)
    pairs_df = pd.DataFrame(prows)
    write(pairs_df, "pairs.csv", "the paired daily difference against V0")
    pcols = [
        "variant",
        "n",
        "mean_diff",
        "sd_diff",
        "t_diff",
        "hac_t_diff",
        "hac_lag",
        "frac_days_improved",
        "frac_days_unchanged",
    ]
    for bk in BOOKS:
        for sample in SAMPLES:
            sub = pairs_df[(pairs_df["book"] == bk) & (pairs_df["sample"] == sample)]
            show(sub, f"pairs.csv  |  book {bk}  |  sample {sample}", pcols)

    brows: list[dict[str, Any]] = []
    for v in variants:
        if v["variant"] == "V0_implied":
            continue
        for bk in BOOKS:
            for sample, m in masks.items():
                row = {
                    "variant": v["variant"],
                    "family": v["family"],
                    "book": bk,
                    "sample": sample,
                    "B": BOOT_B,
                    "block": BOOT_BLOCK,
                    "seed": BOOT_SEED,
                    "ci_pct_lo": CI_PCT[0],
                    "ci_pct_hi": CI_PCT[1],
                }
                row.update(sharpe_diff_ci(rets[v["variant"]][bk][m], base[bk][m]))
                if bk == PRIMARY_BOOK and sample == VERDICT_SAMPLE:
                    row["verdict"] = (
                        "improves"
                        if row["ci_excludes_zero"] and row["dSharpe"] > 0
                        else "no evidence"
                    )
                else:
                    row["verdict"] = ""
                brows.append(row)
    boot_df = pd.DataFrame(brows)
    write(boot_df, "bootstrap.csv", "the bootstrap interval of the Sharpe difference")
    bcols = [
        "variant",
        "n",
        "dSharpe",
        "boot_lo",
        "boot_hi",
        "pct_draws_positive",
        "ci_excludes_zero",
        "verdict",
    ]
    for bk in BOOKS:
        for sample in SAMPLES:
            sub = boot_df[(boot_df["book"] == bk) & (boot_df["sample"] == sample)]
            show(
                sub,
                f"bootstrap.csv  |  book {bk}  |  sample {sample}  |  "
                f"{BOOT_B} circular block draws, block {BOOT_BLOCK}, seed {BOOT_SEED}",
                bcols,
            )

    verdicts = boot_df[
        (boot_df["book"] == PRIMARY_BOOK) & (boot_df["sample"] == VERDICT_SAMPLE)
    ]
    show(
        verdicts,
        f"VERDICT  |  primary book {PRIMARY_BOOK}, sample {VERDICT_SAMPLE}: "
        f"'improves' only where the {int(CI_PCT[1] - CI_PCT[0])}% bootstrap interval "
        f"of the Sharpe difference excludes zero",
        ["variant", "family", "dSharpe", "boot_lo", "boot_hi", "verdict"],
    )
    n_improve = int((verdicts["verdict"] == "improves").sum())
    print(
        f"\n{len(variants) - 1} causal variants were tried against V0; "
        f"{n_improve} improve the primary book in the era by the stated bar, "
        f"{len(variants) - 1 - n_improve} return no evidence."
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
