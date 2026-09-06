"""Shared 0DTE ATM-package helpers for the straddle notebooks.

Protocol matches notebooks/_write_0dte_nb.py and
writeup/sections/methods_close_option.tex.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SPX_MULTIPLIER = 100.0
WINDOW_DAYS = (
    250  # flat MZ window, in trading SESSIONS (fit-mask days), not panel dates
)
MZ_START_DAY = 63  # first session (0-based rank) that receives coefficients
MZ_HALFLIFE_DAYS = 63  # EWMA halflife (days) for the smear's sufficient statistics
YHAT_LABEL = {
    "a0": "baseline (HAR + calendar OLS)",
    "blk2": "block-diagonal ridge",
    "blk2_inc": "block-diagonal ridge, without the FOMC columns",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lasso_t": "lasso (causally tuned)",
    "lasso_f": "lasso (fixed 1e-4)",
    "enet": "elastic net (causally tuned)",
}
# The FOMC panel is the panel of record: "blk2" and "lasso_f" are fitted on it.
# "blk2_inc" is the same ridge on the earlier panel, which carries no FOMC
# columns; it sits directly after "blk2" as the diagnostic that isolates those
# channels. "lgbm", "xgb", "lasso_t" and "enet" are still on the earlier panel
# (a run on the FOMC panel is pending). results/spxw_pnl/MANIFEST.md records the
# panel, chunk tree and code provenance of every table.
MODEL_ORDER = ["a0", "blk2", "blk2_inc", "lgbm", "xgb", "lasso_t", "lasso_f", "enet"]
RULE_ORDER = [
    "always short",
    "sign(s)",
    "heaviside(s): long only",
    "heaviside(s): short only",
]

PERIODS_PER_YEAR = 252.0
"""Annualization convention: per TRADE-DAY scaling, not calendar time.

Every annualized figure in this library (rule_row's Sharpe_ann,
information_ratio's te_ann and IR_ann) multiplies a per-observation
statistic by sqrt(PERIODS_PER_YEAR): each row of the return series is
one trading period and 252 of them make a year.

The 0DTE frame does not trade 252 days a year over the whole sample.
SPXW listed Monday/Wednesday/Friday expirations before June 2022 and
every session after, so the traded frame averages about 200 trade days
a year across 2020-2024 and reaches 252 only from 2022-06 on. A
calendar-time annualization (fill the non-0DTE sessions with a zero
return, keep sqrt(252)) therefore scales the whole-sample figures by
sqrt(200.3 / 252) = 0.89; relative comparisons between rules and models
are unaffected. Call trades_per_year() on the frame's index to print
the realized rate for the period being reported.
"""


def find_repo(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "data" / "spxw_chain.parquet").exists():
            return q
        if (q / "notebooks" / "_write_0dte_nb.py").exists():
            return q
    raise FileNotFoundError("repo root not found")


def yhat_paths(repo: Path) -> dict[str, Path]:
    """Tag -> forecast table. Keys and order match MODEL_ORDER.

    yhat_b2lasso.parquet (the fixed lasso on the earlier panel) is still on
    disk but no tag points at it: the fixed lasso is reported on the panel of
    record. results/spxw_pnl/MANIFEST.md covers it too.
    """
    root = repo / "results" / "spxw_pnl"
    return {
        "a0": root / "yhat_a0.parquet",
        "blk2": root / "yhat_blk2_fomc1.parquet",
        "blk2_inc": root / "yhat_blk2.parquet",
        "lgbm": root / "yhat_tree00.parquet",
        "xgb": root / "yhat_tree16.parquet",
        "lasso_t": root / "yhat_b2lasso_tuned.parquet",
        "lasso_f": root / "yhat_b2lasso_fomc1.parquet",
        "enet": root / "yhat_b3enet_tuned.parquet",
    }


def trades_per_year(index) -> float:
    """Realized trade days per year on a DatetimeIndex: n / (span in years).

    The span runs from the first to the last stamp and is measured in
    365.25-day years, so a frame that trades every session returns about
    252 and the 0DTE frame's 2020-2024 span returns about 200. Use it to
    print the rate behind the sqrt(PERIODS_PER_YEAR) convention; see the
    PERIODS_PER_YEAR docstring. Returns NaN for fewer than two stamps or a
    zero-length span.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index)))
    if len(idx) < 2:
        return float("nan")
    span = (idx.max() - idx.min()).total_seconds() / (365.25 * 86400.0)
    return float(len(idx) / span) if span > 0 else float("nan")


def weighted_median(y, w) -> float:
    """Weighted median: smallest y whose cumulative weight reaches half the total."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    o = np.argsort(y, kind="stable")
    cw = np.cumsum(w[o])
    return float(y[o][int(np.searchsorted(cw, 0.5 * cw[-1], side="left"))])


WLAD_MAX_ITERS = (
    5000  # was 100, which bound on ~53% of real fit days (2026-09-04 audit)
)


def _wlad_line(x, y, w, a, b, iters=WLAD_MAX_ITERS):
    """Weighted least-absolute-deviations line: minimise sum w_i |y_i - a - b x_i|.

    Iteratively reweighted least squares warm-started at the weighted
    least-squares line (a, b): each pass solves the 2x2 weighted normal
    equations with weights w_i / |r_i|. The floor on |r_i| is a numerical
    guard against division by zero only; the solution is validated against
    statsmodels' QuantReg (q = 0.5) in the swap's audit script.

    Returns (a, b, converged). converged is False when the pass cap
    `iters` was reached before both coefficients moved by less than
    1e-12 (relative) in a pass, or when the weighted normal equations
    became singular mid-way; callers count and warn on those days.
    Inputs must be finite (checked). If every x is equal the line is
    unidentified: the weighted median of y is returned with b = 0.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError("_wlad_line: non-finite input")
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("_wlad_line: non-finite warm start")
    if np.all(x == x[0]):
        return weighted_median(y, w), 0.0, True
    scale = float(np.median(np.abs(y)))
    eps = 1e-8 * scale if scale > 0 else 1e-300
    converged = False
    for _ in range(iters):
        r = np.abs(y - a - b * x)
        ww = w / np.maximum(r, eps)
        n = float(ww.sum())
        sx = float((ww * x).sum())
        sxx = float((ww * x * x).sum())
        sy = float((ww * y).sum())
        sxy = float((ww * x * y).sum())
        den = n * sxx - sx * sx
        if not (den > 0):  # also catches NaN
            break
        b_new = (n * sxy - sx * sy) / den
        a_new = (sy - b_new * sx) / n
        done = abs(a_new - a) <= 1e-12 * (1.0 + abs(a)) and abs(b_new - b) <= 1e-12 * (
            1.0 + abs(b)
        )
        a, b = a_new, b_new
        if done:
            converged = True
            break
    return a, b, converged


# Diagnostics of the most recent _mz_day_coefs call (median path): number of
# fitted days and how many hit the WLAD pass cap without converging.
MZ_LAST_FIT: dict[str, int] = {"days": 0, "lad_days": 0, "lad_nonconverged": 0}


def _mz_day_coefs(
    yhat,
    rv_raw,
    baseline,
    day_codes,
    n_days,
    need_days,
    halflife,
    fit_mask=None,
    weighted=True,
    method="mean",
):
    """Per-day causal MZ coefficients (a, b, s2) in y-space.

    method="mean" (the default; bit-identical to the pre-2026-09-04 code)
    fits the line by weighted least squares and returns the weighted
    residual variance s2, so the back-transform (m^2 + s2) B is the
    conditional MEAN of RV. method="median" fits the same line, on the
    same window, mask and weights, by weighted least absolute
    deviations and returns s2 = 0: the back-transform is then m^2 B,
    the conditional MEDIAN of RV, because the median commutes with the
    square for y >= 0 whereas the mean does not (that is what the s2
    term corrects for on the mean path). The median path is only
    implemented on the weighted flat-window path.

    weighted=True (the default, QLIKE-refereed 2026-09-01) fits by
    GLS/weighted least squares with per-window weights
    w = 1/max(yhat, q10_window)^2 — the variance-stabilizing weighting
    under multiplicative errors, aligning the fit's loss with the
    QLIKE scale; q10 is the 10th percentile of yhat *within each
    trailing window* (causal), falling back to the smallest positive
    yhat if q10 <= 0. s2 becomes the weighted mean squared residual,
    ddof 0: the weighted residual sum of squares divided by the weight
    sum, with no degrees-of-freedom correction (same convention on the
    unweighted paths: residual sum of squares / n, ddof 0).
    The per-window q10 makes weights day-specific, so the weighted
    flat path runs a day loop instead of prefix sums; weighted=False
    reproduces the legacy unweighted fit. The EWMA path
    (halflife set) is unweighted regardless.

    halflife=None uses the flat window of the WINDOW_DAYS trading
    SESSIONS strictly before day d; otherwise prior sessions are
    exponentially weighted with the given halflife (in sessions) and
    gated on a Kish effective sample size >= 200 (same constant as
    the flat path's n >= 200). All paths use strictly prior days and
    start at session MZ_START_DAY. fit_mask (bool per row) restricts
    which rows enter the fit — the loaders pass the scored session
    bars so off-session dynamics cannot pollute the calibration;
    coefficients still apply to every row of a session date.

    Sessions (2026-09-04 fix): day codes index every date in the
    near-24-hour panel, including Sunday-overnight and holiday dates
    that carry rows but no scored bars. Windows and the start day are
    therefore counted over the SESSION dates only — the dates that
    have at least one fit-mask row — so that WINDOW_DAYS = 250 means
    250 trading sessions (it used to mean ~207). Coefficients are
    produced for session dates only; rows on non-session dates get
    NaN.
    """
    if method not in ("mean", "median"):
        raise ValueError(f"method must be mean or median, got {method!r}")
    if method == "median" and not (halflife is None and weighted):
        raise NotImplementedError(
            "median recalibration needs the weighted flat-window path"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.sqrt(np.maximum(rv_raw, 0.0) / np.maximum(baseline, 1e-18))
    valid_day = day_codes >= 0
    if not np.all(np.diff(day_codes[day_codes >= 0]) >= 0):
        raise ValueError(
            "_mz_day_coefs: day codes must be monotone (rows sorted by stamp)"
        )
    finite = np.isfinite(yhat) & np.isfinite(y) & (baseline > 0) & valid_day
    if fit_mask is not None:
        finite = finite & np.asarray(fit_mask, dtype=bool)
    MZ_LAST_FIT.update(days=0, lad_days=0, lad_nonconverged=0)
    x0 = np.where(finite, yhat, 0.0)
    y0 = np.where(finite, y, 0.0)
    dc = np.where(valid_day, day_codes, 0)
    # session dates = day codes with at least one fit-mask row; ranks
    # over those codes define the window (in sessions) and the start day
    sess = np.unique(np.asarray(day_codes)[finite])
    sess_rank = {int(c): k for k, c in enumerate(sess)}
    if need_days is None:
        fit_days = [int(c) for c in sess if sess_rank[int(c)] >= MZ_START_DAY]
    else:
        fit_days = sorted(
            int(d)
            for d in need_days
            if int(d) in sess_rank and sess_rank[int(d)] >= MZ_START_DAY
        )

    def _lo_code(d: int) -> int:
        # first session code inside the trailing WINDOW_DAYS-session window
        return int(sess[max(0, sess_rank[d] - WINDOW_DAYS)])

    stats = {
        "n": finite.astype(np.float64),
        "x": x0,
        "xx": x0 * x0,
        "y": y0,
        "xy": x0 * y0,
        "yy": y0 * y0,
    }
    daily = {
        k: np.bincount(dc, weights=np.where(valid_day, v, 0.0), minlength=n_days)
        for k, v in stats.items()
    }
    a_d = np.full(n_days, np.nan)
    b_d = np.full(n_days, np.nan)
    s2_d = np.full(n_days, np.nan)
    if halflife is None and weighted:
        # GLS path: per-window q10 weights break prefix sums — loop days.
        u_f = np.asarray(yhat, float)[finite]
        y_f = y[finite]
        dc_f = day_codes[finite]
        srt = np.argsort(dc_f, kind="stable")
        u_f, y_f, dc_f = u_f[srt], y_f[srt], dc_f[srt]
        for d in fit_days:
            lo = int(np.searchsorted(dc_f, _lo_code(d), side="left"))
            hi = int(np.searchsorted(dc_f, d, side="left"))
            if hi - lo < 200:
                continue
            u_sl, y_sl = u_f[lo:hi], y_f[lo:hi]
            q10 = float(np.quantile(u_sl, 0.10))
            if q10 <= 0:
                pos = u_sl[u_sl > 0]
                if len(pos) == 0:
                    continue
                q10 = float(pos.min())
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                w = 1.0 / np.maximum(u_sl, q10) ** 2
                if not np.isfinite(w).all():
                    continue
                w_n = float(w.sum())
                wx = float((w * u_sl).sum())
                wxx = float((w * u_sl * u_sl).sum())
                wy = float((w * y_sl).sum())
                wxy = float((w * u_sl * y_sl).sum())
                wyy = float((w * y_sl * y_sl).sum())
                denom = w_n * wxx - wx * wx
            if not (denom > 0):  # also skips NaN/inf denominators
                continue
            b = (w_n * wxy - wx * wy) / denom
            a = (wy - b * wx) / w_n
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if method == "median":
                # weighted LAD on the same rows and weights, warm-started at the
                # least-squares line; s2 = 0 so (m^2 + s2) B collapses to m^2 B,
                # the conditional median (the median commutes with the square, y >= 0)
                a, b, ok = _wlad_line(u_sl, y_sl, w, a, b)
                MZ_LAST_FIT["lad_days"] += 1
                if not ok:
                    MZ_LAST_FIT["lad_nonconverged"] += 1
                MZ_LAST_FIT["days"] += 1
                a_d[d] = a
                b_d[d] = b
                s2_d[d] = 0.0
                continue
            MZ_LAST_FIT["days"] += 1
            a_d[d] = a
            b_d[d] = b
            s2_d[d] = (wyy - a * wy - b * wxy) / w_n  # ddof 0
        if MZ_LAST_FIT["lad_nonconverged"]:
            warnings.warn(
                f"weighted-LAD recalibration hit the {WLAD_MAX_ITERS}-pass cap on "
                f"{MZ_LAST_FIT['lad_nonconverged']} of {MZ_LAST_FIT['lad_days']} days",
                RuntimeWarning,
                stacklevel=2,
            )
        return a_d, b_d, s2_d
    if halflife is None:
        pre = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in daily.items()}
        days = np.asarray(fit_days, dtype=np.int64)
        if len(days) == 0:
            return a_d, b_d, s2_d
        lo = np.asarray([_lo_code(int(d)) for d in days], dtype=np.int64)
        w = {k: p[days] - p[lo] for k, p in pre.items()}
        n = w["n"]
        denom = n * w["xx"] - w["x"] ** 2
        ok = (n >= 200) & (denom > 0)
        safe_den = np.where(ok, denom, 1.0)
        safe_n = np.where(ok, n, 1.0)
        b = np.where(ok, (n * w["xy"] - w["x"] * w["y"]) / safe_den, np.nan)
        a = np.where(ok, (w["y"] - b * w["x"]) / safe_n, np.nan)
        s2 = np.where(
            ok, (w["yy"] - a * w["y"] - b * w["xy"]) / safe_n, np.nan
        )  # ddof 0
        a_d[days], b_d[days], s2_d[days] = a, b, s2
        MZ_LAST_FIT["days"] = int(ok.sum())
        return a_d, b_d, s2_d
    lam = 0.5 ** (1.0 / float(halflife))
    acc = dict.fromkeys(daily, 0.0)
    acc2_n = 0.0  # lambda^2-weighted n, for the Kish effective sample size
    want = set(fit_days)
    for d in (int(c) for c in sess):  # decay once per SESSION, not per panel date
        w_n = acc["n"]
        if d in want and w_n > 0 and acc2_n > 0:
            n_eff = w_n * w_n / acc2_n
            denom = w_n * acc["xx"] - acc["x"] ** 2
            if n_eff >= 200 and denom > 0:
                b = (w_n * acc["xy"] - acc["x"] * acc["y"]) / denom
                a = (acc["y"] - b * acc["x"]) / w_n
                a_d[d] = a
                b_d[d] = b
                s2_d[d] = (acc["yy"] - a * acc["y"] - b * acc["xy"]) / w_n  # ddof 0
                MZ_LAST_FIT["days"] += 1
        for k in acc:
            acc[k] = lam * acc[k] + daily[k][d]
        acc2_n = lam * lam * acc2_n + daily["n"][d]
    return a_d, b_d, s2_d


def second_order_raw(
    yhat,
    rv_raw,
    baseline,
    day_codes,
    n_days,
    need_days=None,
    halflife=None,
    fit_mask=None,
    method="mean",
):
    """Causal second-moment back-transform (flat 250-day window by default)."""
    a_d, b_d, s2_d = _mz_day_coefs(
        yhat,
        rv_raw,
        baseline,
        day_codes,
        n_days,
        need_days,
        halflife,
        fit_mask,
        method=method,
    )
    valid_day = day_codes >= 0
    dc = np.where(valid_day, day_codes, 0)
    te = np.isfinite(yhat) & (baseline > 0) & valid_day & np.isfinite(a_d[dc])
    m = a_d[day_codes[te]] + b_d[day_codes[te]] * yhat[te]
    f = np.full(len(yhat), np.nan)
    f[te] = (m**2 + s2_d[day_codes[te]]) * baseline[te]
    return f


def second_order_mz(
    yhat,
    rv_raw,
    baseline,
    day_codes,
    n_days,
    need_days=None,
    halflife=None,
    fit_mask=None,
    method="mean",
):
    """Same map as second_order_raw, also returning m and s2 on each row."""
    a_d, b_d, s2_d = _mz_day_coefs(
        yhat,
        rv_raw,
        baseline,
        day_codes,
        n_days,
        need_days,
        halflife,
        fit_mask,
        method=method,
    )
    valid_day = day_codes >= 0
    dc = np.where(valid_day, day_codes, 0)
    te = np.isfinite(yhat) & (baseline > 0) & valid_day & np.isfinite(a_d[dc])
    m = np.full(len(yhat), np.nan)
    s2_row = np.full(len(yhat), np.nan)
    rv = np.full(len(yhat), np.nan)
    m[te] = a_d[day_codes[te]] + b_d[day_codes[te]] * yhat[te]
    s2_row[te] = s2_d[day_codes[te]]
    rv[te] = (m[te] ** 2 + s2_row[te]) * baseline[te]
    return rv, m, s2_row


FIT_MASK_MINUTES = (
    10 * 60 + 30,
    16 * 60,
)  # stamps 10:30..16:00 = trade bars 10:00..15:30 under bar-end labels
EARLY_CLOSE_DATES = (
    "2001-07-03",
    "2001-11-23",
    "2001-12-24",
    "2002-07-03",
    "2002-11-29",
    "2002-12-24",
    "2003-07-03",
    "2003-11-28",
    "2003-12-24",
    "2004-11-26",
    "2005-11-25",
    "2006-07-03",
    "2006-11-24",
    "2007-07-03",
    "2007-11-23",
    "2007-12-24",
    "2008-07-03",
    "2008-11-28",
    "2008-12-24",
    "2009-11-27",
    "2009-12-24",
    "2010-11-26",
    "2011-11-25",
    "2012-07-03",
    "2012-11-23",
    "2012-12-24",
    "2013-07-03",
    "2013-11-29",
    "2013-12-24",
    "2014-07-03",
    "2014-11-28",
    "2014-12-24",
    "2015-11-27",
    "2015-12-24",
    "2016-11-25",
    "2017-07-03",
    "2017-11-24",
    "2018-07-03",
    "2018-11-23",
    "2018-12-24",
    "2019-07-03",
    "2019-11-29",
    "2019-12-24",
    "2020-11-27",
    "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03",
    "2023-11-24",
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
)
"""The NYSE 13:00 ET early-close calendar, 2001-2025 (55 dates).

A calendar property, known ex ante, not a property of any data file.
Generated by the standing NYSE rule and checked date by date:

  * the Friday after Thanksgiving, every year;
  * December 24 when it is a weekday and is not itself the observed
    Christmas holiday (when December 25 falls on a Saturday the holiday
    is observed on Friday December 24 and the market is closed, not
    early-closed: 2004, 2010, 2021 carry no December 24 entry);
  * July 3 when both July 3 and July 4 are weekdays (July 4 on a Monday
    puts July 3 on a Sunday; July 4 on a Saturday makes July 3 the
    observed holiday, a full closure).

No further one-off 13:00 closes fall in 2001-2025: the other calendar
irregularities of the period are full closures (2001-09-11..14,
2004-06-11, 2007-01-02, 2012-10-29..30, 2018-12-05, 2025-01-09) and
leave no row behind.

VERIFICATION against results/spxw_pnl/yhat_a0.parquet (2001-07-13 to
2024-04-30, 5,713 dates with a 16:00 stamp). Statistic: the 16:00-slot
rv_raw divided by the mean of the previous 20 sessions' 16:00-slot
rv_raw. 28 of the 55 calendar dates carry a 16:00 row in the panel; 26
of those 28 sit in the bottom 2% of that ratio over all sessions (24 in
the bottom 1%), and every panel date whose ratio falls below 0.05 (15
dates) is a calendar date -- the panel produces no collapse the
calendar does not name. Two documented discrepancies, both kept in the
constant because the calendar is the authority:

  * 2002-07-03 -- ratio 1.00; the panel's 10:30-16:00 rv is flat through
    the afternoon with no collapse at all. The early close is real
    (July 4, 2002 was a Thursday); the pre-2008 tape repeats stale
    prints overnight, so the post-close bars carry manufactured
    variance rather than none.
  * 2019-11-29 -- ratio 1.05; F1 flagged this half session separately
    for an inverted intraday profile (the 13:30 bar is the day's
    largest).

The 24-hour futures grid keeps printing after the 13:00 cash close, so
these dates still carry a stamp-16:00 row; it is a post-close bar, not
a forecast of the traded 15:30-16:00 window, and it is excluded from
the fit mask and from every loader's output.

early_close_days(chain), which derives the half sessions from the
option chain's hours_to_expiration, covers 2020 onward only and must
stay a subset of this constant within the chain's coverage.
"""


def _panel_frame(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one yhat table; return (frame with et/date, the session fit mask).

    The tables store t as tz-aware UTC. A tz-naive file would silently be
    read as the 20:00/21:00 ET overnight bars (the 2026-08-17 misjoin), so
    tz-awareness is asserted.

    The fit mask is the paper's "fit window restricted to regular-session
    bars": a row is in the mask when all three hold.

      1. its stamp lies in FIT_MASK_MINUTES (10:30..16:00, the trade bars
         10:00..15:30 under bar-end labels);
      2. its date is not in EARLY_CLOSE_DATES, whose stamp-16:00 row is a
         post-close bar (the cash market closed at 13:00);
      3. its date is a SESSION date, meaning the table carries a
         stamp-16:00 row for it. NYSE holidays are not gaps in this
         24-hour futures panel but shortened futures sessions whose bars
         end at 13:00 or 11:30 and never reach 16:00; their rows are
         low-variance morning bars that would otherwise enter the
         recalibration fit at about a seventh of the typical y.

    Measured on the forecast tables (all nine share t): 69,298 rows fall in
    the stamp window and 68,215 survive both rules, so 1,083 are removed.
    Against the library's previous mask -- which knew only the twelve 2020+
    early closes and had no session rule, leaving 69,230 rows -- the change
    removes 1,015 rows:

      * 547 futures-only holiday rows on 122 dates whose last in-window
        stamp is 11:30 or 13:00 (shortened futures sessions on NYSE
        holidays);
      *  26 rows on 3 dates whose tape stops before 16:00 (2002-07-05,
        2003-12-26, 2018-09-03);
      * 442 rows on the 42 pre-2020 early-close dates the calendar adds.

    The gate suite prints the count. The frame carries the boolean columns
    early_close and session_date; early-close dates are also excluded from
    every loader's output.
    """
    df = pd.read_parquet(path).sort_values("t").reset_index(drop=True)
    t = pd.to_datetime(df["t"])
    if t.dt.tz is None:
        raise ValueError(
            f"{path}: 't' is tz-naive; the panel loaders need tz-aware UTC stamps"
        )
    df["t"] = t.dt.tz_convert("UTC")
    df["et"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.normalize().dt.tz_localize(None)
    mins = df["et"].dt.hour * 60 + df["et"].dt.minute
    lo, hi = FIT_MASK_MINUTES
    rth = ((mins >= lo) & (mins <= hi)).to_numpy()
    early = df["date"].isin(pd.to_datetime(list(EARLY_CLOSE_DATES))).to_numpy()
    df["early_close"] = early
    at_close = (mins == 16 * 60).to_numpy()
    session = df["date"].isin(df.loc[at_close, "date"].unique()).to_numpy()
    df["session_date"] = session
    return df, rth & ~early & session


def _need_days(need_dates, uniq) -> set[int] | None:
    """Map requested dates onto day codes; refuse an empty match."""
    if need_dates is None:
        return None
    want = pd.DatetimeIndex(pd.to_datetime(list(need_dates))).normalize()
    if getattr(want, "tz", None) is not None:
        want = want.tz_localize(None)
    pos = {pd.Timestamp(d): k for k, d in enumerate(uniq)}
    days = {pos[d] for d in want if d in pos}
    if not days:
        raise ValueError(
            f"none of the {len(want)} requested dates is in the forecast table"
        )
    return days


_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    """Content hash of a forecast table (size+mtime is not a safe key)."""
    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    d = _DIGEST_CACHE.get(key)
    if d is None:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        d = h.hexdigest()
        _DIGEST_CACHE[key] = d
    return d


def _recal_source_hash() -> str:
    """Hash of the recalibration AND loader code so any change invalidates caches."""
    src = "".join(
        inspect.getsource(f)
        for f in (
            _wlad_line,
            _mz_day_coefs,
            second_order_raw,
            second_order_mz,
            _panel_frame,
            load_yhat_1530,
            load_yhat_1530_mz_cached,
            load_yhat_panel,
            load_yhat_panel_mz,
        )
    )
    # Every constant that changes the fit belongs here, not only the code:
    # the stamp window, the early-close calendar and the session-date rule
    # (a date is in the fit only if the table carries a stamp-16:00 row).
    src += repr(FIT_MASK_MINUTES) + repr(EARLY_CLOSE_DATES)
    src += "session_date=has-1600-row"
    return hashlib.sha1(src.encode()).hexdigest()[:12]


RECAL_CACHE_PREFIX = "v8-sess250"  # bumped 2026-09-04: windows counted in sessions


def _evict_stale_tables(
    cache: Path, tag: str, stem: str, method: str, keep: str = ""
) -> None:
    """Remove this tag's tables built under another key, plus pre-method-rename names.

    Current names are ``{stem}_{tag}_{method}_{hash16}.parquet``; before the
    method segment existed the tables were ``{stem}_{tag}_{hash16}.parquet``
    and can no longer be told apart from a mean-recalibrated table, so they
    are deleted too.
    """
    for old in cache.glob(f"{stem}_{tag}_{method}_*.parquet"):
        if old.name != keep:
            old.unlink()
    stale = re.compile(
        "^" + re.escape(f"{stem}_{tag}_") + "[0-9a-f]{16}" + re.escape(".parquet") + "$"
    )
    for old in cache.glob(f"{stem}_{tag}_*.parquet"):
        if stale.match(old.name) and old.name != keep:
            old.unlink()


def load_yhat_1530(path: Path, need_dates=None, method: str = "mean") -> pd.DataFrame:
    # Bar-end-labelled stamps: the 15:30 book's fresh forecast — issued at
    # 15:30 for the 15:30->close bar it trades — lives on the STAMP-16:00
    # row, whose rv_raw is that bar's own realized variance. (The stamp-
    # 15:30 row is the forecast of 15:00->15:30, one bar stale.)
    df, rth = _panel_frame(path)
    is_row = (df["et"].dt.hour == 16) & (df["et"].dt.minute == 0) & ~df["early_close"]
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    need_days = _need_days(need_dates, uniq)
    df["rv_hat"] = second_order_raw(
        yhat,
        rv_raw,
        base,
        day_codes,
        len(uniq),
        need_days=need_days,
        fit_mask=rth,
        method=method,
    )
    out = (
        df.loc[is_row, ["date", "yhat", "baseline", "rv_raw", "rv_hat"]]
        .dropna(subset=["rv_hat"])
        .drop_duplicates("date")
        .set_index("date")
    )
    return out


def load_yhat_1530_mz_cached(
    tag: str, path: Path, need_dates, cache: Path, method: str = "mean"
) -> pd.DataFrame:
    cp = _cache_path(cache, "yhat1530mz", tag, path, need_dates, method, "mz")
    if cp.exists():
        return pd.read_parquet(cp)
    df, rth = _panel_frame(path)
    # Fresh row for the 15:30 book: stamp 16:00 (see load_yhat_1530).
    is_row = (df["et"].dt.hour == 16) & (df["et"].dt.minute == 0) & ~df["early_close"]
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    need_days = _need_days(need_dates, uniq)
    rv, m, s2 = second_order_mz(
        yhat,
        rv_raw,
        base,
        day_codes,
        len(uniq),
        need_days=need_days,
        fit_mask=rth,
        method=method,
    )
    df["rv_hat"] = rv
    df["m"] = m
    df["s2"] = s2
    df["yhat_vol"] = df["yhat"] * np.sqrt(np.maximum(df["baseline"], 0.0))
    df["m_vol"] = df["m"] * np.sqrt(np.maximum(df["baseline"], 0.0))
    out = (
        df.loc[
            is_row,
            [
                "date",
                "yhat",
                "baseline",
                "rv_raw",
                "rv_hat",
                "m",
                "s2",
                "yhat_vol",
                "m_vol",
            ],
        ]
        .dropna(subset=["rv_hat"])
        .drop_duplicates("date")
        .set_index("date")
    )
    if out.empty:
        raise ValueError(f"{path}: the 15:30 loader produced no rows; not caching")
    out.to_parquet(cp)
    _evict_stale_tables(cache, tag, "yhat1530mz", method, keep=cp.name)
    return out


def load_yhat_1530_cached(
    tag: str, path: Path, need_dates, cache: Path, method: str = "mean"
) -> pd.DataFrame:
    cp = _cache_path(cache, "yhat1530", tag, path, need_dates, method, "vec")
    if cp.exists():
        return pd.read_parquet(cp)
    out = load_yhat_1530(path, need_dates, method=method)
    if out.empty:
        raise ValueError(f"{path}: the 15:30 loader produced no rows; not caching")
    out.to_parquet(cp)
    _evict_stale_tables(cache, tag, "yhat1530", method, keep=cp.name)
    return out


def _cache_path(
    cache: Path, stem: str, tag: str, path: Path, need_dates, method: str, kind: str
) -> Path:
    """Cache file name keyed on code, table CONTENT, window constants and dates."""
    h = hashlib.sha1()
    ver = f"{RECAL_CACHE_PREFIX}-{kind}-{method}-{_recal_source_hash()}"
    h.update(
        f"{ver}:{_file_digest(path)}:flat{WINDOW_DAYS}:start{MZ_START_DAY}".encode()
    )
    if need_dates is None:
        h.update(b"|all")
    else:
        for d in sorted(pd.to_datetime(list(need_dates))):
            h.update(f"|{d}".encode())
    return cache / f"{stem}_{tag}_{method}_{h.hexdigest()[:16]}.parquet"


def load_yhat_panel(path: Path, method: str = "mean") -> pd.DataFrame:
    """Every 30-min stamp. Same second-order map as the 15:30 loader."""
    df, rth = _panel_frame(path)
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    # Bar-end-labelled stamps: the scored trade bars 10:00-15:30 live on
    # stamps 10:30-16:00, so the session fit mask covers those stamps.
    df["in_fit"] = rth & np.isfinite(yhat) & np.isfinite(rv_raw) & (base > 0)
    df["rv_hat"] = second_order_raw(
        yhat,
        rv_raw,
        base,
        day_codes,
        len(uniq),
        need_days=None,
        fit_mask=rth,
        method=method,
    )
    # Early-close dates carry a post-close 16:00 bar: no forecast is issued.
    df.loc[df["early_close"], "rv_hat"] = np.nan
    return df


def load_yhat_panel_mz(path: Path, method: str = "mean") -> pd.DataFrame:
    """Every 30-min stamp with rv_hat and its pieces m and s2 (the 15:30 loader's fit).

    Adds mins (minutes from ET midnight), in_fit and early_close; rv_hat,
    m and s2 are NaN on early-close dates. The deck's look-ahead cliff and
    the experimental notebook read the panel through this so that the
    k = 0 row reproduces the traded forecast exactly.
    """
    df, rth = _panel_frame(path)
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    day_codes, uniq = pd.factorize(df["date"], sort=True)
    rv, m, s2 = second_order_mz(
        yhat,
        rv_raw,
        base,
        day_codes,
        len(uniq),
        need_days=None,
        fit_mask=rth,
        method=method,
    )
    df["rv_hat"], df["m"], df["s2"] = rv, m, s2
    df["mins"] = (df["et"].dt.hour * 60 + df["et"].dt.minute).to_numpy()
    df["in_fit"] = rth & np.isfinite(yhat) & np.isfinite(rv_raw) & (base > 0)
    df.loc[df["early_close"], ["rv_hat", "m", "s2"]] = np.nan
    return df


def rule_sizes(px: pd.DataFrame) -> dict[str, pd.Series]:
    """Position series of the standing rules, keyed by RULE_ORDER name.

    The two heaviside legs are the one-sided halves of sign(s): long only
    holds +1 on the days s > 0 and 0 otherwise, short only holds -1 on the
    days s <= 0 and 0 otherwise (s == 0 is short, the deck's convention).
    They partition the days, so day by day their positions - and therefore
    their daily returns - sum to sign(s) exactly. Flat days stay in the
    series as zeros rather than being dropped.
    """
    s = px["signal"].to_numpy(float)
    long_day = s > 0
    pos = pd.Series(np.where(long_day, 1.0, -1.0), index=px.index)
    return {
        "always short": pd.Series(-1.0, index=px.index),
        "sign(s)": pos,
        "heaviside(s): long only": pd.Series(
            np.where(long_day, 1.0, 0.0), index=px.index
        ),
        "heaviside(s): short only": pd.Series(
            np.where(long_day, 0.0, -1.0), index=px.index
        ),
    }


def extra_weight_sizes(px: pd.DataFrame) -> dict[str, pd.Series]:
    s = px["signal"].astype(float)
    pos = pd.Series(np.where(s.to_numpy() > 0, 1.0, -1.0), index=px.index)
    abs_s = s.abs()
    # same-day rank of |s_t| among days <= t (s_t is known at 15:30), like the sign
    rank = abs_s.expanding(min_periods=63).rank(pct=True).fillna(0.5)
    r = px["R"].astype(float)
    vol = r.expanding(min_periods=63).std().shift(1)
    # target = the expanding median of the lagged volatility, itself lagged (no full-sample constant)
    vol_target = vol.expanding(min_periods=63).median().shift(1)
    inv_vol = (
        (vol_target / vol)
        .clip(upper=3.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )
    return {
        "rank-|s|": pos * (1.0 + 4.0 * (rank - 0.5)).clip(0.0, 3.0),
        "inv-vol of R": pos * inv_vol,
        "sign(s)": pos,
        "always short": pd.Series(-1.0, index=px.index),
    }


def rule_row(r: pd.Series, size: pd.Series) -> pd.Series:
    """Summary row of a daily return series.

    t_mean = sqrt(n) * mean / std (no autocorrelation correction);
    Sharpe_ann = mean / std * sqrt(PERIODS_PER_YEAR) (252 trade days a
    year; see the PERIODS_PER_YEAR docstring for the convention and the
    calendar-time equivalent); std with ddof=1. skew and
    ex_kurt are pandas' bias-corrected sample estimates (G1 and G2),
    not the raw standardized moments — the standalone footnote says so.
    n_buy counts days with size > 0 (size == 0 is not a buy) among the
    days that have a return (size is restricted to r.notna(), so n_buy
    and pct_buy describe the same n days as the return statistics).
    """
    r = pd.Series(r).astype(float)
    size = pd.Series(size, index=r.index).astype(float)
    size = size[r.notna()]
    x = r.dropna()
    n = int(len(x))
    mu = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n >= 2 else float("nan")
    n_buy = int((size > 0).sum())
    n_sz = int(size.notna().sum())
    return pd.Series(
        {
            "n": n,
            "mean": mu,
            "std": sd,
            "min": float(x.min()) if n else float("nan"),
            "25%": float(x.quantile(0.25)) if n else float("nan"),
            "50%": float(x.median()) if n else float("nan"),
            "75%": float(x.quantile(0.75)) if n else float("nan"),
            "max": float(x.max()) if n else float("nan"),
            "skew": float(x.skew()) if n else float("nan"),
            "ex_kurt": float(x.kurt()) if n else float("nan"),
            "t_mean": mu / sd * np.sqrt(n) if (sd and sd > 0) else float("nan"),
            "Sharpe_ann": mu / sd * np.sqrt(PERIODS_PER_YEAR)
            if (sd and sd > 0)
            else float("nan"),
            "n_buy": n_buy,
            "pct_buy": 100.0 * n_buy / n_sz if n_sz else float("nan"),
        }
    )


def newey_west_lag(n: int) -> int:
    """Bartlett-kernel truncation lag floor(1.5 n^(1/3)) used throughout."""
    # + 1e-9 guards the exact-cube case (1000 ** (1/3) evaluates to 9.999...).
    return int(np.floor(1.5 * float(n) ** (1.0 / 3.0) + 1e-9))


def newey_west_t(x, lag: int | None = None) -> tuple[float, int]:
    """HAC t-statistic of the mean of x against zero.

    Long-run variance = gamma_0 + 2 sum_{k=1}^{L} (1 - k/(L+1)) gamma_k with
    autocovariances about the sample mean divided by n (no small-sample
    correction; identical to statsmodels OLS on a constant with
    cov_type="HAC", maxlags=L, use_correction=False). L defaults to
    newey_west_lag(n). Returns (t, L).
    """
    v = np.asarray(pd.Series(x).astype(float).dropna(), float)
    n = len(v)
    if n < 2:
        return float("nan"), 0
    L = newey_west_lag(n) if lag is None else int(lag)
    L = max(0, min(L, n - 1))
    e = v - v.mean()
    lrv = float(e @ e) / n
    for k in range(1, L + 1):
        lrv += 2.0 * (1.0 - k / (L + 1.0)) * float(e[k:] @ e[:-k]) / n
    if not lrv > 0:
        return float("nan"), L
    return float(v.mean() / np.sqrt(lrv / n)), L


def information_ratio(r_port: pd.Series, r_bench: pd.Series) -> pd.Series:
    """IR on active return R_a = R^p - R_benchmark.

    t_active is the Newey-West (Bartlett) HAC t of the mean active
    return at lag floor(1.5 n^(1/3)) (reported as t_lag); te uses ddof=1.
    te_ann and IR_ann scale by sqrt(PERIODS_PER_YEAR); see that
    constant's docstring for the per-trade-day convention.
    """
    a = r_port.astype(float).align(r_bench.astype(float), join="inner")
    active = a[0] - a[1]
    x = active.dropna()
    n = int(len(x))
    mu = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n >= 2 else float("nan")
    t_hac, lag = newey_west_t(x)
    return pd.Series(
        {
            "n": n,
            "mean_active": mu,
            "te_daily": sd,
            "te_ann": sd * np.sqrt(PERIODS_PER_YEAR)
            if (sd and sd > 0)
            else float("nan"),
            "IR_ann": mu / sd * np.sqrt(PERIODS_PER_YEAR)
            if (sd and sd > 0)
            else float("nan"),
            "t_active": t_hac,
            "t_lag": lag,
            "corr_to_bench": float(a[0].corr(a[1])) if n >= 3 else float("nan"),
        }
    )


def circular_block_bootstrap_idx(
    rng: np.random.Generator, n: int, blen: int, B: int
) -> np.ndarray:
    """(B, n) index array for a circular moving-block bootstrap.

    Each draw takes ceil(n / blen) block starts uniform on [0, n), lays
    the blocks (starts[:, None] + arange(blen)) % n end to end and
    truncates to n. All B draws come from one rng.integers call of shape
    (B, n_blocks), so the same rng seed gives the same B rows.
    """
    n = int(n)
    blen = max(1, int(blen))
    n_blocks = int(np.ceil(n / blen))
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(blen)[None, None, :]) % n
    return idx.reshape(B, n_blocks * blen)[:, :n]


def cboe_short_straddle_margin_points(
    S: float, K_c: float, K_p: float, premium: float
) -> float:
    """CBOE-style strategy-based margin for a short index straddle, index points."""
    otm = min(abs(float(S) - float(K_c)), abs(float(S) - float(K_p)))
    a = 0.15 * float(S) - otm + float(premium)
    b = 0.10 * float(S) + float(premium)
    return float(max(a, b, 0.0))


def crossed_premium_return(
    q: pd.Series, exit_: pd.Series, bid: pd.Series, ask: pd.Series
) -> pd.Series:
    """Per-premium return when the entry pays the spread.

    q > 0 buys at the ask (exit/ask - 1); q < 0 sells at the bid
    (1 - exit/bid); q == 0 returns 0. A non-positive fill price on the
    side used (ask for a long, bid for a short) is not a tradeable
    quote: the row is NaN, and callers should print
    crossed_untradeable_count on the same inputs.
    """
    q = q.astype(float)
    exit_ = exit_.astype(float)
    bid = bid.astype(float)
    ask = ask.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        long_r = np.where(ask > 0, exit_ / ask - 1.0, np.nan)
        short_r = np.where(bid > 0, 1.0 - exit_ / bid, np.nan)
    out = np.where(q > 0, long_r, np.where(q < 0, short_r, 0.0))
    return pd.Series(out, index=q.index)


def crossed_untradeable_count(q: pd.Series, bid: pd.Series, ask: pd.Series) -> int:
    """Rows a crossed fill cannot price: long with ask <= 0 or short with bid <= 0."""
    q = pd.Series(q).astype(float)
    bid = pd.Series(bid).astype(float)
    ask = pd.Series(ask).astype(float)
    bad = ((q > 0) & ~(ask > 0)) | ((q < 0) & ~(bid > 0))
    return int(bad.sum())


def points_pnl(q: pd.Series, exit_: pd.Series, entry: pd.Series) -> pd.Series:
    return q.astype(float) * (exit_.astype(float) - entry.astype(float))


def stamp_spot(live: pd.DataFrame, keys) -> pd.Series:
    """NaN-safe copy of underlying_price at each key. Not a consensus.

    After dropna, this tape has one value per stamp. ``first`` is that
    value; all-NaN keys are absent from the index.
    """
    if isinstance(keys, str):
        keys = [keys]
    g = live.dropna(subset=["underlying_price"]).groupby(list(keys), dropna=False)[
        "underlying_price"
    ]
    if len(g) and int(g.nunique().max()) > 1:
        raise ValueError("stamp_spot: a stamp carries more than one underlying_price")
    return g.first()


def quote_mid(bid, ask) -> pd.Series:
    """(bid + ask) / 2 with the vendor's no-quote sentinel bid == ask == 0 as NaN.

    bid == ask == 0 is "no quote" (3.7% of chain rows, 99.8% of them at
    the 9:30 stamp), not a zero price; every helper that prices a wing,
    vertical or condor leg must go through this. One-sided rows
    (bid == 0, ask > 0; 39.6% of rows) keep their half-spread mid.
    """
    b = pd.to_numeric(pd.Series(bid), errors="coerce").astype(float)
    a = pd.to_numeric(pd.Series(ask), errors="coerce").astype(float)
    a.index = b.index
    mid = 0.5 * (b + a)
    return mid.mask((b == 0) & (a == 0))


IV_VENDOR_BOUNDS = (
    0.0005,
    0.025,
)  # the vendor clips hourly implied volatility to this range


def _bsm_package_price(s: float, F: float, K_c: float, K_p: float) -> float:
    """Call at K_c plus put at K_p on the forward F (Black-76, r = 0) with total vol s."""
    from math import erf, log, sqrt

    def N(z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    if s <= 0:
        return max(F - K_c, 0.0) + max(K_p - F, 0.0)
    d1c = (log(F / K_c) + 0.5 * s * s) / s
    d1p = (log(F / K_p) + 0.5 * s * s) / s
    call = F * N(d1c) - K_c * N(d1c - s)
    put = K_p * N(-(d1p - s)) - F * N(-d1p)
    return call + put


def _vendor_iv_nodes() -> tuple[float, ...]:
    # The vendor's implied volatility is a geometric bisection on
    # IV_VENDOR_BOUNDS that returns a bracket NODE when it does not
    # converge: the bounds, the first midpoint sqrt(lo*hi) = 0.0035355 and
    # the successive lower-branch midpoints 0.001330, 0.000815, 0.000638,
    # 0.000565, 0.000532 (chain-integrity audit 2026-09-04).
    lo, hi = IV_VENDOR_BOUNDS
    nodes = [hi, lo]
    m = hi
    for _ in range(6):
        m = float(np.sqrt(lo * m))
        nodes.append(m)
    return tuple(nodes)


IV_VENDOR_NODES = _vendor_iv_nodes()
IV_NODE_RTOL = 1e-5  # a non-converged solver returns the node EXACTLY (float32:
# relative error < 1e-6); the middle node 0.0035 per hour is a typical implied
# volatility, so any wider band censors genuine quotes (2e-3 caught six such
# legs on the deck's frame and no extra node hit; measured 2026-09-04).


def censor_vendor_iv(iv_raw, rtol: float = IV_NODE_RTOL) -> pd.Series:
    """Vendor hourly IV with the solver's bracket nodes set to NaN (censored).

    A value within rtol of any IV_VENDOR_NODES entry is a non-converged
    solve, not a volatility, and is dropped before any iv_var use. The
    16:00 stamp's IV is 0% interior (all nodes) and must never be used.
    """
    iv = pd.to_numeric(pd.Series(iv_raw), errors="coerce").astype(float)
    bad = np.zeros(len(iv), dtype=bool)
    v = iv.to_numpy()
    for node in IV_VENDOR_NODES:
        bad |= np.abs(v - node) <= rtol * node
    return iv.mask(bad)


def attach_iv_hourly_as_30min(atm: pd.DataFrame) -> pd.DataFrame:
    """Per-leg vendor IV (censored of solver nodes) -> 30-minute variance.

    iv_c / iv_p are censor_vendor_iv of the leg fields; iv_hourly is the
    mean of the two legs, NaN if either is censored; iv_30 = iv_hourly / sqrt(2);
    iv_var = iv_30^2. Use only the 15:30 book's IV, never the 16:00 stamp.
    """
    out = atm.copy()
    out["iv_c"] = censor_vendor_iv(out["impl_volatility_c"]).to_numpy()
    out["iv_p"] = censor_vendor_iv(out["impl_volatility_p"]).to_numpy()
    legs = out[["iv_c", "iv_p"]]
    out["iv_hourly"] = legs.mean(axis=1).where(legs.notna().all(axis=1))
    out["iv_30"] = out["iv_hourly"] / np.sqrt(2.0)
    out["iv_var"] = out["iv_30"] ** 2
    return out


def iv_var_from_conventions(
    iv_raw: pd.Series, hours_remaining: float
) -> dict[str, pd.Series]:
    """Window variance under the three unit conventions, from CENSORED vendor IV."""
    iv = censor_vendor_iv(iv_raw)
    iv.index = pd.Series(iv_raw).index
    hrs = float(hours_remaining)
    return {
        "chris_hourly": (iv**2) * hrs,
        "annualized_om": (iv**2) * hrs / (252.0 * 6.5),
        "already_window": iv**2,
    }


def early_close_days(chain: pd.DataFrame) -> pd.DatetimeIndex:
    """Expiration dates whose 15:30 ET row has already expired (half sessions).

    On early-close days (13:00 ET close) the vendor prints a full grid of
    carried-forward quotes; hours_to_expiration <= 0 marks the 13:00
    stamp (hte = 0) and the frozen 13:30..16:00 stamps on the 12 such
    days 2020-2025 (on a normal day hte reaches 0 only at the 16:00
    stamp, which no trade ever reads),
    and is <= 0 at the 15:30 stamp on exactly those days. The 15:30
    quote is not a market and the settlement is known at "entry", so
    the day is dropped — never re-pointed at the 13:00 bar, whose quotes
    are already the frozen snapshot while underlying_price is an earlier
    print. Every loader must apply drop_early_close: the deck's 15:30
    loader, the intraday panel builder and the experimental frame. The
    rule is the intraday notebook's; the deck inherits it here. Accepts
    a frame with 'expiration',
    'hours_to_expiration' and either 'et' (tz-aware or naive ET
    timestamps) or 'timestamp' (UTC); returns naive normalized dates.
    """
    if "hours_to_expiration" not in chain.columns:
        raise KeyError("early_close_days needs the chain's hours_to_expiration column")
    if "et" in chain.columns:
        et = pd.to_datetime(chain["et"])
        if getattr(et.dt, "tz", None) is not None:
            et = et.dt.tz_convert("America/New_York")
    else:
        et = pd.to_datetime(chain["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
    at_1530 = (et.dt.hour == 15) & (et.dt.minute == 30)
    hte = pd.to_numeric(chain.loc[at_1530, "hours_to_expiration"], errors="coerce")
    med = hte.groupby(chain.loc[at_1530, "expiration"]).median()
    days = pd.to_datetime(med.index[med <= 0])
    if getattr(days, "tz", None) is not None:
        days = days.tz_convert("America/New_York").tz_localize(None)
    out = pd.DatetimeIndex(days.normalize().astype("datetime64[ns]"), name=None)
    return out.sort_values()


def drop_early_close(chain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Return (chain without early-close expiration days, the dropped dates)."""
    days = early_close_days(chain)
    exp = pd.to_datetime(chain["expiration"])
    if getattr(exp.dt, "tz", None) is not None:
        exp = exp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    keep = ~exp.dt.normalize().isin(days)
    return chain[keep].copy(), days


def parity_forward(K, c_mid, p_mid):
    """Forward implied by put-call parity at one strike with r = 0: F = K + C - P.

    The forward, not the spot, prices the 0DTE package: the index carries
    dividend drag (about 0.35 pt on a close-of-day expiration), which is
    why some asks sit below spot-intrinsic without being bad data.
    """
    return np.asarray(K, float) + np.asarray(c_mid, float) - np.asarray(p_mid, float)


def bsm_invert_package_vol(
    F: float, K_c: float, K_p: float, mid_total: float, hours_remaining: float = 0.5
) -> float:
    """Total volatility over the remaining window that reproduces the package mid.

    Inverts the Black-76 price (r = 0) of the nearest-OTM call plus put
    on the underlying level F for the total vol s = sigma * sqrt(T) by
    bisection. Pass the SPOT to match the vendor's own convention: on
    the uncensored legs a spot-based inversion reproduces the vendor's
    hourly field to 1e-4 and a parity-forward one does not (gate suite,
    2026-09-04), so re-inverted legs stay on the same footing as the
    rest; parity_forward is the dividend-correct alternative when every
    leg is inverted the same way. To compare with
    the vendor's HOURLY standard deviation divide by
    sqrt(hours_remaining): at 15:30 (0.5 h) the vendor field equals
    s * sqrt(2). Returns NaN when the mid is below forward-intrinsic or
    the bracket fails. Use it to replace legs whose vendor IV is a
    solver node (censor_vendor_iv).
    """
    F, K_c, K_p, mid_total = float(F), float(K_c), float(K_p), float(mid_total)
    if not (np.isfinite(F) and F > 0 and np.isfinite(mid_total)):
        return float("nan")
    intrinsic = max(F - K_c, 0.0) + max(K_p - F, 0.0)
    if mid_total <= intrinsic:
        return float("nan")
    # The package price is strictly increasing in s (positive vega on both
    # legs), so plain bisection on [lo, hi] converges without a library
    # root finder; 100 halvings of a unit bracket reach 1e-30.
    lo, hi = 1e-8, 1.0
    if _bsm_package_price(hi, F, K_c, K_p) < mid_total:
        return float("nan")
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if _bsm_package_price(m, F, K_c, K_p) < mid_total:
            lo = m
        else:
            hi = m
        if hi - lo < 1e-12:
            break
    return float(0.5 * (lo + hi))


def hourly_iv_from_total_vol(s_total: float, hours_remaining: float = 0.5) -> float:
    """Vendor-convention hourly SD from a total vol over hours_remaining."""
    return float(s_total) / np.sqrt(float(hours_remaining))


ATM_MAX_STRIKE_GAP = 10.0  # SPX strikes are 5 apart near the money; a nearest-OTM leg further  # than 10 away means more than one strike is missing (a vendor outage)
ATM_MIN_LIVE = 10  # live contracts per stamp below which the stamp is a vendor outage (normal: hundreds)


def pick_nearest_otm_guarded(
    live: pd.DataFrame,
    spot: pd.Series,
    keys=("expiration",),
    max_gap: float = ATM_MAX_STRIKE_GAP,
    min_live: int = ATM_MIN_LIVE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-OTM call and put per key, with outage guards.

    Returns (atm, dropped). `keys` are the grouping columns (the deck's
    ("expiration",); the intraday builder's ("expiration", "timestamp"));
    `spot` is a Series indexed by those keys (stamp_spot). A cell is
    dropped, and listed in `dropped` with its reason, when
      * fewer than `min_live` live contracts are quoted at the stamp
        (2022-02-22 10:00-11:30 lists 4-6 of 324: a vendor outage;
        2020-05-01 10:00 has none with mid > 0), or
      * max(K_c - S, S - K_p) > max_gap (+1e-9): more than one strike is
        missing near the money (the same outage admits a 297-pt OTM put
        as "ATM"); one missing strike (gap between 5 and 10, about 0.4% of
        cells) is tolerated, or
      * no call at or above / no put at or below the spot.
    Leg mids go through quote_mid (bid == ask == 0 -> NaN). Columns are
    as before: S, K_c, K_p, entry, bid_entry, ask_entry, same_strike plus
    the per-leg quote fields with _c/_p suffixes.
    """
    keys = list(keys)
    eps = 1e-9
    spot_df = spot.rename("S").reset_index()
    n_live = live.groupby(keys).size().rename("n_live").reset_index()
    c = live[live["cp"] == "C"].merge(spot_df, on=keys, how="inner")
    p = live[live["cp"] == "P"].merge(spot_df, on=keys, how="inner")
    c = c[np.isfinite(c["S"])]
    p = p[np.isfinite(p["S"])]
    c_otm = c[c["strike"].astype(float) >= c["S"]].copy()
    c_otm["k_gap"] = c_otm["strike"].astype(float) - c_otm["S"]
    c_pick = (
        c_otm.sort_values(keys + ["k_gap", "strike"])
        .groupby(keys, as_index=False)
        .first()
    )
    p_otm = p[p["strike"].astype(float) <= p["S"]].copy()
    p_otm["k_gap"] = p_otm["S"] - p_otm["strike"].astype(float)
    p_pick = (
        p_otm.sort_values(keys + ["k_gap", "strike"])
        .groupby(keys, as_index=False)
        .first()
    )
    both = c_pick.merge(
        p_pick, on=keys, suffixes=("_c", "_p"), how="outer", indicator=True
    )
    both = both.merge(n_live, on=keys, how="left")
    both["n_live"] = both["n_live"].fillna(0).astype(int)
    both["S"] = both["S_c"].astype(float).fillna(both["S_p"].astype(float))
    both["K_c"] = both["strike_c"].astype(float)
    both["K_p"] = both["strike_p"].astype(float)
    both["gap"] = np.fmax(both["K_c"] - both["S"], both["S"] - both["K_p"])
    reason = pd.Series("", index=both.index, dtype=object)
    reason[both["_merge"] == "left_only"] = "no_put"
    reason[both["_merge"] == "right_only"] = "no_call"
    reason[(reason == "") & (both["gap"] > max_gap + eps)] = "strike_gap"
    reason[(reason == "") & (both["n_live"] < min_live)] = "few_live"
    dropped = both.loc[reason != "", keys + ["S", "K_c", "K_p", "gap", "n_live"]].copy()
    dropped["reason"] = reason[reason != ""].to_numpy()
    atm = both.loc[reason == ""].drop(columns=["_merge", "gap"]).copy()
    atm["mid_c"] = quote_mid(atm["bid_c"], atm["ask_c"]).to_numpy()
    atm["mid_p"] = quote_mid(atm["bid_p"], atm["ask_p"]).to_numpy()
    atm["entry"] = atm["mid_c"].astype(float) + atm["mid_p"].astype(float)
    atm["bid_entry"] = atm["bid_c"].astype(float) + atm["bid_p"].astype(float)
    atm["ask_entry"] = atm["ask_c"].astype(float) + atm["ask_p"].astype(float)
    atm["same_strike"] = atm["K_c"] == atm["K_p"]
    return atm.reset_index(drop=True), dropped.reset_index(drop=True)


def pick_nearest_otm(live: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    """pick_nearest_otm_guarded on ("expiration",); dropped cells in atm.attrs["dropped"]."""
    atm, dropped = pick_nearest_otm_guarded(live, spot)
    atm.attrs["dropped"] = dropped
    return atm


def pick_wings(
    live: pd.DataFrame, body: pd.DataFrame, width: float = 25.0
) -> pd.DataFrame:
    """Nearest wings at least `width` points beyond the body strikes.

    Wing mids are quote_mid of the wing bid/ask when the live frame
    carries them (bid == ask == 0 -> NaN, so a no-quote wing makes the
    condor entry NaN rather than a free wing); callers must pass a live
    frame that still contains bid == ask == 0 rows or those wings are
    simply absent, which is also correct.
    """
    c = live[live["cp"] == "C"].copy()
    p = live[live["cp"] == "P"].copy()
    want_c = body[["expiration", "K_c"]].copy()
    want_p = body[["expiration", "K_p"]].copy()
    c = c.merge(want_c, on="expiration", how="inner")
    p = p.merge(want_p, on="expiration", how="inner")
    c = c[c["strike"].astype(float) >= (c["K_c"] + width)]
    p = p[p["strike"].astype(float) <= (p["K_p"] - width)]
    c["k_gap"] = c["strike"].astype(float) - c["K_c"]
    p["k_gap"] = p["K_p"] - p["strike"].astype(float)
    c_w = (
        c.sort_values(["expiration", "k_gap", "strike"])
        .groupby("expiration", as_index=False)
        .first()
    )
    p_w = (
        p.sort_values(["expiration", "k_gap", "strike"])
        .groupby("expiration", as_index=False)
        .first()
    )
    wings = c_w.merge(p_w, on="expiration", suffixes=("_cw", "_pw"))
    # Keep only wing quotes so body.K_c / body.K_p / body.entry survive the merge.
    wing_cols = ["expiration", "strike_cw", "strike_pw", "mid_cw", "mid_pw"]
    extra = [c for c in ("bid_cw", "ask_cw", "bid_pw", "ask_pw") if c in wings.columns]
    out = body.merge(wings[wing_cols + extra], on="expiration", how="inner")
    out["K_c_wing"] = out["strike_cw"].astype(float)
    out["K_p_wing"] = out["strike_pw"].astype(float)
    if {"bid_cw", "ask_cw", "bid_pw", "ask_pw"} <= set(out.columns):
        # bid == ask == 0 is the vendor's no-quote sentinel, not a free wing
        out["mid_c_wing"] = quote_mid(out["bid_cw"], out["ask_cw"]).to_numpy()
        out["mid_p_wing"] = quote_mid(out["bid_pw"], out["ask_pw"]).to_numpy()
    else:
        out["mid_c_wing"] = out["mid_cw"].astype(float)
        out["mid_p_wing"] = out["mid_pw"].astype(float)
    out["entry_body"] = out["entry"].astype(float)
    out["entry_wings"] = out["mid_c_wing"] + out["mid_p_wing"]
    out["entry_ic"] = out["entry_body"] - out["entry_wings"]
    out["width"] = np.minimum(
        out["K_c_wing"] - out["K_c"], out["K_p"] - out["K_p_wing"]
    )
    return out


def settle_package(df: pd.DataFrame, s_close: pd.Series) -> pd.DataFrame:
    out = df.copy()
    exp_day = pd.to_datetime(out["expiration"])
    if getattr(exp_day.dt, "tz", None) is not None:
        exp_day = exp_day.dt.tz_convert("America/New_York").dt.tz_localize(None)
    exp_day = exp_day.dt.normalize()
    out["S_close"] = exp_day.map(s_close)
    out["pay_c"] = np.maximum(out["S_close"] - out["K_c"], 0.0)
    out["pay_p"] = np.maximum(out["K_p"] - out["S_close"], 0.0)
    out["exit"] = out["pay_c"] + out["pay_p"]
    if "K_c_wing" in out.columns:
        out["pay_c_wing"] = np.maximum(out["S_close"] - out["K_c_wing"], 0.0)
        out["pay_p_wing"] = np.maximum(out["K_p_wing"] - out["S_close"], 0.0)
        out["exit_wings"] = out["pay_c_wing"] + out["pay_p_wing"]
        out["exit_ic"] = out["exit"] - out["exit_wings"]
    return out


FOMC_STATEMENT_DAYS = (
    # statement days (second day of two-day meetings; ET dates) not carried by
    # data/releases.parquet, whose FOMC flags end 2023-11-01; plus the 2020
    # emergency actions: 03-03 (Tuesday statement) and 03-16 (first session
    # after the Sunday 03-15 statement)
    "2020-03-03",
    "2020-03-16",
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


def _naive_days(index) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(pd.to_datetime(index))
    if days.tz is not None:
        days = days.tz_localize(None)
    return days.normalize()


def fomc_and_monthend(
    index: pd.DatetimeIndex, repo: Path, sessions: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Calendar flags on the given trading-day index (naive ET dates).

    is_fomc (nullable boolean): FOMC statement days = the release-file
    flags (data/releases.parquet, "fomc release" on bar-end stamps; NaN
    flags are treated as no release) plus FOMC_STATEMENT_DAYS.
    Convention: the statement day, or the first session after an
    off-hours statement (ET dates). Dates AFTER the knowledge horizon —
    the later of the file's last flagged day and the last hard-coded
    day, stored in flags.attrs["fomc_known_until"] — are pd.NA:
    UNKNOWN, not "no FOMC". fomc_known (bool) marks the known dates,
    and a RuntimeWarning is raised if any date is unknown; extend
    FOMC_STATEMENT_DAYS before scoring such dates.
    is_me: the last SESSION of each calendar month. `sessions` is the
    full trading-day list the month ends are taken from; when None the
    passed index itself is used, so it must then be the complete frame
    (a subset slides the flag earlier). The index used must be strictly
    increasing without duplicates (checked). On the SPXW expiration
    calendar this is the last trading day of the month (an end-of-month
    series is always listed).
    is_event = is_me | is_fomc (Kleene logic: NA where is_fomc is NA
    and is_me is False).
    """
    days = _naive_days(index)
    sess = days if sessions is None else _naive_days(sessions)
    if not (sess.is_monotonic_increasing and sess.is_unique):
        raise ValueError(
            "fomc_and_monthend: sessions must be strictly increasing, no duplicates"
        )
    if sessions is not None and not days.isin(sess).all():
        raise ValueError(
            "fomc_and_monthend: every index date must be one of the sessions"
        )
    month_end = pd.Series(sess).groupby([sess.year, sess.month]).transform("max")
    is_me = days.isin(pd.DatetimeIndex(month_end))
    rel = repo / "data" / "releases.parquet"
    fomc: set[pd.Timestamp] = set()
    file_last: pd.Timestamp | None = None
    if rel.exists():
        r = pd.read_parquet(rel)
        col = next(
            (
                c
                for c in r.columns
                if str(c).lower() in {"fomc release", "fomc", "fomc_release"}
            ),
            None,
        )
        date_col = next((c for c in ("date", "endbartime") if c in r.columns), None)
        if col is not None:
            if date_col is None:
                raise ValueError("releases.parquet: no 'date' or 'endbartime' column")
            flag = (
                pd.to_numeric(r[col], errors="coerce").fillna(0.0) > 0
            )  # NaN -> False
            dts = pd.to_datetime(r.loc[flag, date_col], errors="coerce").dropna()
            if len(dts):
                if dts.dt.tz is not None:
                    dts = dts.dt.tz_localize(None)
                dts = dts.dt.normalize()
                fomc |= set(dts)
                file_last = pd.Timestamp(dts.max())
    hard = [pd.Timestamp(s) for s in FOMC_STATEMENT_DAYS]
    fomc |= set(hard)
    known_until = max(hard) if file_last is None else max(file_last, max(hard))
    known = days <= known_until
    is_fomc = pd.array(np.where(days.isin(list(fomc)), True, False), dtype="boolean")
    is_fomc[~known] = pd.NA
    flags = pd.DataFrame({"is_me": is_me, "is_fomc": is_fomc}, index=days)
    flags["fomc_known"] = known
    flags["is_event"] = flags["is_me"] | flags["is_fomc"]
    flags.attrs["fomc_known_until"] = known_until
    if not known.all():
        warnings.warn(
            f"fomc_and_monthend: {int((~known).sum())} dates after the FOMC knowledge horizon "
            f"{known_until.date()} have is_fomc = NA (unknown, not no-FOMC)",
            RuntimeWarning,
            stacklevel=2,
        )
    flags.index = index
    return flags
