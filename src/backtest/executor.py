"""Shared executor scaffold for ML walk-forward volatility backtests.

Lifts the ~90% duplicated `main()` body of the per-method executors
(`src/models/xgboost.py`, `src/models/lightgbm.py`, `src/models/random_forest.py`,
`src/models/ridge.py`) into one module. Per-method scripts supply only a
`fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams)`
callable plus a default-hyperparam dict; everything else — CLI parsing,
loading, transforms, horizon shift, chunk slicing, smearing, reduce
JSON — is shared.

The CLI contract is now owned by ``.hpc/cli.py`` (the auto-generated
dispatcher) and ``.hpc/tasks.py`` FLAGS dict — the per-executor flag
list lives there, not in this module. ``run_executor`` and the
backtest plumbing remain here. The flags expected on ``args`` match
those declared in ``.hpc/tasks.py`` FLAGS for the calling executor's
module key (e.g. ``FLAGS["src/backtest/tune_tree.py`` (cmd_evaluate) passes via subprocess:
``--params-file --output-file --start --end --data-path --train-window``
``--horizon --exog-cols`` plus the Ridge-only ``--segment`` and
``--lag-scope`` and the optional ``--refit-frequency`` / ``--seed``.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import os
import re
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from numba import njit

from src.diagnostics import check_and_report
from src.evaluation.metrics import apply_duan_smearing, save_chunk_reduce
from src.data.loading import (
    NYSE_EARLY_CLOSE_DATES,
    apply_overnight_fills,
    load_raw_data,
)
from src.features.transforms.scaling import (
    masked_rolling_scale_col,
    rolling_robust_scale,
)
from src.backtest.segmentation import (
    SEGMENT_DEFINITIONS,
    compute_segment_train_window,
    slice_to_segment,
)
from src.features.extractors.calendar import add_calendar_features
from src.features.extractors.expiry import add_expiry_features
from src.features.extractors.har import generate_har_features, resolve_har_lags
from src.features.transforms.target import (
    PERIODS_PER_DAY,
    apply_horizon_shift,
    robust_transform,
)

FitPredict = Callable[[np.ndarray, np.ndarray, int, dict], np.ndarray]

# A feature with at least this fraction of its mass at a single *exact* value is treated as
# mode-inflated (a point mass + a continuous part) and gets the hurdle encoding: an occurrence
# indicator plus a magnitude scaled over its off-mode (active) values. Continuous features
# (returns, vols) never reach this.
#
# The test used to be on exact *zeros* specifically, which is the right test for the flow features
# it was written for (``voldemand``, ~60% zeros) but misses the same shape at a different location:
# ``numobs`` is capped at 30 observations per bar and *is* 30 on 78% of bars, so its raw IQR is
# exactly 0 and its adjusted IQR is 0.029 — which turns an ordinary holiday half-bar into
# ``adj_numobs_ma_25`` at |z| 79, with 986 rows past |z| 20. Keying on the modal value instead of on
# zero subsumes the old behaviour exactly (``voldemand``'s mode *is* 0) and, across the 43 raw exog,
# catches exactly one new column: ``numobs`` at 0.78. The next-highest modal share is
# ``stocktwits_sentiment`` at 0.12, comfortably below the threshold.
ZERO_INFLATED_FRAC: float = 0.2

# How many bars a forward fill may carry an exogenous value before the row is
# treated as unobserved again. 26 bars is two trading sessions at this panel's
# 30-minute frequency: long enough to bridge a missed print or a holiday edge,
# short enough that a stopped feed cannot masquerade as live data for months.
# ``None`` restores the unlimited fill.
FFILL_LIMIT: int | None = 26

# -- 2026-10-15 (user: "fix all 8"; results/data_audit_2026-10-15/, experiments/data_audit_2026_10_15.py)
# Each flag off reproduces the pre-2026-10-15 executor for its part, bit for bit.
#
# APPLY_FFILL_LIMIT: FFILL_LIMIT above was defined but never passed (load_and_transform's
# default was None, and run_executor / run_geometry_local.prepare_full call it without one),
# so a stopped feed was forward-filled for months.  The values the models see were already
# replaced by the neutral fill on unobserved rows; what the unlimited fill still reached was
# every transform statistic computed on the filled column (diurnal baselines, winsor bounds,
# s.d. floors, routing and mode tests) and the _active flags.
APPLY_FFILL_LIMIT: bool = True
# ACTIVE_FROM_OBSERVED: the hurdle (_active) flag is read from OBSERVED prints only; an
# unobserved row takes the flag's own modal value over the observed rows (before it, under
# CAUSAL_MODE_INFLATION).  It used to be read off the forward-filled column, so a stopped or
# sparse feed carried its last print's activity: FOMC was "active" on all 123 trade days of its
# dead tail, and the mode-inflated class of voldemand existed only because of the fill
# (14-16 % zeros among its prints, 59-60 % after the fill).
ACTIVE_FROM_OBSERVED: bool = True
# CAUSAL_MODE_INFLATION: whether a series is mode-inflated, and its mode, are decided at each
# row from the observed prints of the series strictly BEFORE that row (at least
# MODE_MIN_PRINTS of them), not from the full sample.  A series gets an _active column when it
# is mode-inflated at some row; the flag is 0 on rows where it is not (yet / any more).  On the
# 2026-10-15 data: numobs and the FOMC flags always; StockTwits sentiment 2010-07 .. 2016-12,
# sentcount 2010-07 .. 2013-06, attention for a week of 2009-07; voldemand never.  Off: the
# class is read from the full sample (the record's column set), the flag from observed prints.
CAUSAL_MODE_INFLATION: bool = True
MODE_MIN_PRINTS: int = 250
# CAUSAL_NEUTRAL_FILL: an unobserved row takes the median of the observed adjusted values
# strictly before it, not the full-sample median; rows before the first print of a series
# take its first observed value (constant over that stretch, so no window that holds only
# such rows can learn anything from it, and every window that holds a live row already holds
# that value).
CAUSAL_NEUTRAL_FILL: bool = True
# CAUSAL_GUARD_MODE: the point-mass test of _build_scale_guards on a design column (it removes
# the modal value from the reference-IQR estimate) is made at each grid point from the rows
# before it, not from the whole column (the 16:00 vvix ladders: a point mass in the first
# window -- the pre-2012 fill -- but not over the full sample).
CAUSAL_GUARD_MODE: bool = True
# B_REGULAR_BARS_ONLY: the target's diurnal baseline B is the trailing mean of the last 20
# REGULAR in-slot bars: the NYSE early closes' post-13:00 bars (13:30 .. 17:00) do not enter
# it (they depressed B for the next 20 sessions: 94 of the 866 trade days).  Those bars stay
# rows of the pooled panel; only B excludes them.
B_REGULAR_BARS_ONLY: bool = True
# DROP_EARLY_CLOSE_SESSIONS: a per-bar (time-of-day segment) design drops the NYSE early-close
# sessions: their 15:30 / 16:00 rows are post-close futures bars (22 training rows of the
# 16:00 design, 2010-2024).  The pooled path (segment None) keeps every bar.
DROP_EARLY_CLOSE_SESSIONS: bool = True
# ERODE_IN_BARS: the ref_iqr mask erosion counts an MA window in BARS.  In a per-bar design a
# design row is a session, so the window w (bars) is ceil(w x rows-a-session /
# BARS_PER_SESSION) rows; it used to be w rows, i.e. w sessions (adj_*_ma_3125 needed 3,125
# consecutive available sessions and fell back to the 1.0 warm-up floor on 978 of the 1,087
# test sessions for the turnover ladders).  The pooled path is unchanged (a row is a bar).
ERODE_IN_BARS: bool = True
BARS_PER_SESSION: int = 46
# BURNIN_IN_BAR_SPACE: a global-lag per-bar segment drops the 3,125-bar HAR burn-in on the
# full bar grid BEFORE slicing to its clock(s), not 3,125 ROWS of the slice (3,125 sessions:
# the 16:00 design started 2010-07-12 instead of 1998-06-04).
BURNIN_IN_BAR_SPACE: bool = True

# -- 2026-10-19 (user: pooled linear rerun; writeup/rehaul/04_features.tex notes E6, E9)
# Column flags of the rerun design.  Both default False: every existing path keeps its columns,
# bit for bit; the pooled rerun (experiments/pool_linear_pretune.py --rerun) sets both True.
#
# DROP_ACTIVITY_MA (E6, 04_features.tex:95): the trailing averages of the activity indicators,
# the chi-bar blocks (*_active_ma_*), leave the design.  The _active flags are still built by
# load_and_transform (the scale guards' mode tests read them); only their HAR averages are not
# generated.
DROP_ACTIVITY_MA: bool = False
# CALENDAR_THROUGH_OVERNIGHT (E9, 04_features.tex:126; 99_author_notes.tex:31): the calendar
# stops at the overnight indicator -- DOW_0 .. DOW_4, hour, is_overnight.  Out of the design:
# is_open, is_close, the expiry / month-end / quarter-end columns of add_expiry_features, and the
# HAR x open/close products har_ma_*_x_open / _x_close.  (is_open / is_close stay in the frame
# as columns; add_expiry_features is still called, its columns are not passed on.)
CALENDAR_THROUGH_OVERNIGHT: bool = False
# DROP_OVERNIGHT_INDICATOR (2026-10-19, the 13-bar rerun; 99_author_notes.tex:31, E9 + E15): is_overnight
# leaves the design too.  The note keeps it only if the overnight intervals stay in the training
# sample; on the 13 regular-hours bars (end stamps 10:00 .. 16:00) it is 1 on the 16:00 bar only.  A
# flag, not a rule read off the sample: the 13-bar run sets it, every other path keeps the column.
DROP_OVERNIGHT_INDICATOR: bool = False
# LEAN_GLOBAL_LAGS (2026-10-19, the 13-bar rerun): a global-lag segment computes each trailing average
# on the full bar grid exactly as generate_har_features does (the same pandas rolling mean and shift
# on the same full column) but keeps only the rows of the segment, column by column.  Same values,
# bit for bit; the full-grid frame of every trailing average is never held (the all-features
# design on the base-2 ladder, 1,080 columns x 303,442 intervals, passed 12.5 GB and was killed on
# a 15 GB machine).  Off: the full-grid frame, as before.
LEAN_GLOBAL_LAGS: bool = False

# -- 2026-10-22 (user: "Fix the late start series problems."; writeup/LINEAR_SWITCH_AUDIT.md finding 2)
# LATE_START_WARMUP: a series that starts after the data (VIX 2006, StockTwits 2009-10, VIX3M 2009, volatility
# demand 2011, VVIX 2012) holds its first printed value on every bar before its start (CAUSAL_NEUTRAL_FILL), so for
# up to W rows after the start the scaler's window [t-W, t) is mostly that constant: the IQR is exactly 0 while the
# constant holds > 75% of the window, the relative floor is 0 (the column has never had a non-degenerate window) and
# the 1.0 fallback sets the scale; once the live share passes 25% the scale drops to a tiny IQR or to ref_iqr (the
# IQR of a half-fill average over a short history) while the median is still the stale constant -- adj_vix3m_ma_3125
# at -14.5 in 2010-10, adj_voldemand_all_open_only_ma_3125 at 110 in 2012-01, the fold-2 blow-up of the 13-bar
# pre-tune.  With the flag on, a value column adj_<c>_ma_<k> of a series that starts after the frame's first day is
# the neutral 0 until the scaler's window holds W LIVE rows -- rows whose k-bar trailing average holds no bar before
# the series' first print day (k counted on the frame the averages are computed on) -- and the scaler's own output,
# unchanged, from then on.  The availability averages (unscaled) are untouched.  Causal: liveness of a row depends
# on the prints before its averaging window, the zeroing at t on the rows of [t-W, t).  Minimum = the full window:
# the only state in which the scaler's statistic is one of data alone, by the same estimator as every other series.
# Rule written down 2026-10-22 09:12 PDT (addendum 09:18: "start" = the first-print day) before any rerun or score.
# Off: the pre-2026-10-22 design bit for bit.
LATE_START_WARMUP: bool = True

# -- 2026-10-22 (user: "add the flag"; writeup note E28, stale feeds)
# UNOBSERVED_NEUTRAL: after a feed stops (volatility demand after 2023-08-31) its trailing averages hold the constant
# neutral fill, which the rolling scaler keeps re-scaling as its window [t-W, t) moves, so the scaled value drifts
# (adj_voldemand_all_open_only_ma_1 about -0.28 -> -0.16 from 2023-09 to 2024-04) while the availability average is 0.
# With the flag on, a scaled value column adj_<c>_ma_<k> is set to the neutral value 0 on every row whose k-bar
# averaging window holds no observed bar of series c (its <c>_avail_ma_<k> is 0); otherwise unchanged.  Availability
# averages are untouched.  Applied after the scaling and the LATE_START_WARMUP step.  Causal: <c>_avail_ma_<k> at row
# t is the mean of the availability flags <c>__obs (recorded before any fill) of the same k bars before t that
# adj_<c>_ma_<k> averages, so whether it is 0 is known at t; the scaler's statistics are computed as before (the step
# overwrites its output only), so no other row moves.  Rule written down 2026-10-22 13:52 PDT before any rerun or
# score.  Off: the design before this flag, bit for bit.
UNOBSERVED_NEUTRAL: bool = True


def early_close_post_bars(t: pd.Series) -> np.ndarray:
    """Bars ending 13:30 .. 17:00 ET on an NYSE early-close day (src.data.loading calendar)."""
    tt = pd.to_datetime(t)
    mins = tt.dt.hour * 60 + tt.dt.minute
    ec = tt.dt.normalize().isin(pd.to_datetime(list(NYSE_EARLY_CLOSE_DATES)))
    return (ec & (mins > 13 * 60) & (mins <= 17 * 60)).to_numpy()


def _series_age_columns(df: pd.DataFrame, exog_cols: list[str]) -> pd.DataFrame:
    """LATE_START_WARMUP: ``<c>__age`` for every value column ``adj_<c>`` of ``exog_cols`` whose series starts after
    the frame's first day: the frame row's position minus the position of the first row of the calendar day (bar
    end) of the series' first print (``<c>__obs``), so the trailing average ``adj_<c>_ma_<k>`` of a row holds no
    pre-start bar iff its age >= k.  A series never observed gets an age below every k.  Series observed on the
    frame's first day get no column (always live).  Computed on the frame the trailing averages are computed on;
    the columns are added to ``df`` in place (no frame copy) and returned with it."""
    day = pd.to_datetime(df["t"]).dt.normalize().to_numpy()
    pos = np.arange(len(df), dtype=np.int64)
    ages = {}
    for name in exog_cols:
        if not name.startswith("adj_"):
            continue
        c = name[4:]
        oc = f"{c}__obs"
        if oc not in df.columns or f"{c}__age" in df.columns:
            continue
        obs = df[oc].to_numpy().astype(bool)
        if obs.any():
            d0 = day[int(np.argmax(obs))]
            if d0 == day[0]:
                continue  # observed on the frame's first day: live on every row
            start = int(np.argmax(day == d0))
        else:
            start = len(df) + 1
        ages[f"{c}__age"] = pos - start
    for k, v in ages.items():  # in place: a few int columns, no copy of the frame
        df[k] = v
    return df


def late_start_warmup(
    X: np.ndarray, df: pd.DataFrame, feature_names: list[str], train_win: int
) -> np.ndarray:
    """LATE_START_WARMUP: set the scaled value of ``adj_<c>_ma_<k>`` to 0 on every row whose scaler window
    [t - train_win, t) holds a row that is not live (age < k; see :func:`_series_age_columns`); rows [0, train_win),
    scaled with the first window, keep their value only when that window is all live.  ``df`` is the frame the
    rows of ``X`` were taken from (row i of X = row i of df).  Liveness is monotone in time, so the zeroed rows are
    the first L + train_win, L the first live row."""
    n = len(X)
    cols, vals = 0, 0
    for j, name in enumerate(feature_names):
        m = re.match(r"adj_(.+)_ma_(\d+)$", name)
        if m is None or f"{m.group(1)}__age" not in df.columns:
            continue
        live = df[f"{m.group(1)}__age"].to_numpy()[:n] >= int(m.group(2))
        L = int(np.argmax(live)) if live.any() else n
        if L == 0:
            continue
        hi = min(n, L + train_win)
        X[:hi, j] = 0.0
        cols += 1
        vals += hi
    if cols:
        print(
            f"[scale] late-start warm-up: {cols} columns, {vals} values set to 0 (window {train_win} rows)"
        )
    return X


def unobserved_neutral(
    X: np.ndarray, df: pd.DataFrame, feature_names: list[str]
) -> np.ndarray:
    """UNOBSERVED_NEUTRAL: set the scaled value of ``adj_<c>_ma_<k>`` to 0 on every row whose k-bar averaging window
    holds no observed bar of series c, i.e. whose ``<c>_avail_ma_<k>`` (the trailing mean of the 0/1 flag over the
    same k bars) is 0 -- tested as below half a bar (< 0.5 / k), which a single observed bar can never be, so float
    residue of the rolling sum cannot keep a row live.  ``df`` is the frame the rows of ``X`` were taken from (row i
    of X = row i of df); its availability averages are read, not changed."""
    n = len(X)
    cols, vals, was0 = 0, 0, 0
    for j, name in enumerate(feature_names):
        m = re.match(r"adj_(.+)_ma_(\d+)$", name)
        if m is None:
            continue
        av = f"{m.group(1)}_avail_ma_{m.group(2)}"
        if av not in df.columns:
            continue
        none = df[av].to_numpy(dtype=np.float64)[:n] < 0.5 / int(m.group(2))
        if not none.any():
            continue
        was0 += int(np.count_nonzero(X[none, j] == 0.0))
        X[none, j] = 0.0
        cols += 1
        vals += int(none.sum())
    if cols:
        print(
            f"[scale] unobserved-neutral: {cols} columns, {vals} values set to 0 ({was0} were 0 already)"
        )
    return X


def causal_neutral_fill(adj: pd.Series) -> pd.Series:
    """The neutral value of each unobserved row (``adj`` NaN there): the median of the observed
    values strictly before the row; the first observed value before the series starts."""
    em = adj.expanding(min_periods=1).median().shift(1)
    first = adj.first_valid_index()
    return em.fillna(float(adj.loc[first]) if first is not None else 0.0)


@njit(cache=True)
def _causal_activity_kernel(
    codes: np.ndarray, observed: np.ndarray, min_prints: int, frac: float
) -> tuple[np.ndarray, bool]:
    """Hurdle flag with the class and mode decided from the observed prints before each row.

    ``codes``: value codes in ascending value order (-1 where unobserved).  A row is in the
    inflated regime when at least ``min_prints`` prints precede it and the modal value holds at
    least ``frac`` of them (ties: the smallest value, as :func:`modal_value`).  Observed rows:
    1 when inflated and off the mode; unobserved rows: the modal value of the flag over the
    observed rows before it when inflated, else 0."""
    n = codes.shape[0]
    k = 1
    for i in range(n):
        if codes[i] + 1 > k:
            k = codes[i] + 1
    counts = np.zeros(k, np.int64)
    best = 0
    best_code = -1
    seen = 0
    act_sum = 0.0
    act_n = 0
    act = np.zeros(n)
    ever = False
    for i in range(n):
        inflated = seen >= min_prints and best >= frac * seen
        if inflated:
            ever = True
        if observed[i]:
            c = codes[i]
            a = 1.0 if (inflated and c != best_code) else 0.0
            act[i] = a
            act_sum += a
            act_n += 1
            counts[c] += 1
            if counts[c] > best or (counts[c] == best and c < best_code):
                best = counts[c]
                best_code = c
            seen += 1
        elif inflated and act_n > 0:
            act[i] = 1.0 if act_sum / act_n > 0.5 else 0.0
    return act, ever


def causal_activity(vals: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, bool]:
    """(flag, ever inflated) of :func:`_causal_activity_kernel` for one raw column."""
    obs = np.asarray(observed, dtype=bool)
    codes = np.full(len(vals), -1, dtype=np.int64)
    if obs.any():
        c, _ = pd.factorize(np.asarray(vals, dtype=np.float64)[obs], sort=True)
        codes[obs] = c
    act, ever = _causal_activity_kernel(codes, obs, MODE_MIN_PRINTS, ZERO_INFLATED_FRAC)
    return act, bool(ever)


class _PastMode:
    """Exact running mode of the finite values of a column over the rows processed so far,
    kept for the candidates that can be the mode with a share >= ZERO_INFLATED_FRAC: a value
    with that share overall has it in at least one processed chunk, where it becomes a
    candidate with its exact count so far."""

    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        self.n = 0
        self.hi = 0
        self.cand: dict[float, int] = {}

    def update(self, hi: int) -> None:
        chunk = self.x[self.hi : hi]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            u, c = np.unique(chunk, return_counts=True)
            d = dict(zip(u.tolist(), c.tolist()))
            for v in list(self.cand):
                self.cand[v] += d.get(v, 0)
            for v in u[c >= ZERO_INFLATED_FRAC * chunk.size].tolist():
                if v not in self.cand:
                    self.cand[v] = int(np.count_nonzero(self.x[:hi] == v))
        self.n += int(chunk.size)
        self.hi = hi

    def inflated_mode(self) -> float | None:
        if self.n < MODE_MIN_PRINTS or not self.cand:
            return None
        best = max(self.cand.values())
        if best < ZERO_INFLATED_FRAC * self.n:
            return None
        return min(v for v, c in self.cand.items() if c == best)


def modal_value(vals: np.ndarray) -> float:
    """The single most common exact value of ``vals`` (NaNs dropped), or NaN if empty."""
    finite = vals[~np.isnan(vals)]
    if not finite.size:
        return float("nan")
    uniq, counts = np.unique(finite, return_counts=True)
    return float(uniq[int(np.argmax(counts))])


def mode_inflated(vals: np.ndarray) -> tuple[bool, float]:
    """``(is_mode_inflated, mode)`` under :data:`ZERO_INFLATED_FRAC`."""
    finite = vals[~np.isnan(vals)]
    if not finite.size:
        return False, float("nan")
    mode = modal_value(vals)
    return bool(float(np.mean(finite == mode)) >= ZERO_INFLATED_FRAC), mode


def _expanding_real_iqr(
    x: np.ndarray,
    mask: np.ndarray,
    grid_step: int = 480,
    warmup_floor: float = 1.0,
    exclude_mode: bool = False,
) -> np.ndarray:
    """Causal expanding IQR of ``x`` over its *available* (``mask``) values.

    A slowly-varying reference scale, so it is computed on a coarse grid and
    forward-filled rather than re-evaluated every row. Excluding the imputed
    (unavailable) rows is the crux: the median-fill injects a flat constant
    that otherwise dominates — and collapses — the window IQR. Causal: row
    ``t`` only ever sees real values strictly before ``t``.

    ``warmup_floor`` is what the reference is worth *before* there is enough real history to
    estimate it (fewer than 8 real values on the grid). It used to be 0, i.e. "impose no floor" —
    and 0 is the worst possible answer, because those are exactly the rows where the trailing window
    is all imputed constant and the local IQR is therefore ~0 too. ``adj_stocktwits_sentiment_ma_*``
    reaches |z| 70 on 2010-06-02, one day into its coverage, with ``ref_iqr`` recorded as exactly
    0.0. An unmeasurable scale must not become an unbounded divisor: 1.0 leaves an adjusted feature
    (a log or a ratio, O(1) by construction) at its natural scale until its own dispersion is
    estimable, and the availability indicator already tells the model the feature is not live.

    ``exclude_mode`` (2026-10-15, CAUSAL_GUARD_MODE): at each grid point, when the column's
    rows before it are mode-inflated (:class:`_PastMode`), the real values equal to that mode
    are left out of the estimate -- the causal form of the whole-column test the caller made.
    """
    n = len(x)
    ref = np.full(n, float(warmup_floor), dtype=np.float64)
    real_idx = np.flatnonzero(mask)
    last = float(warmup_floor)
    tracker = _PastMode(np.asarray(x, dtype=np.float64)) if exclude_mode else None
    for t in range(0, n, grid_step):
        past = real_idx[real_idx < t]
        if tracker is not None:
            tracker.update(t)
            mode = tracker.inflated_mode()
            if mode is not None:
                past = past[x[past] != mode]
        if past.size >= 8:
            q25, q75 = np.percentile(x[past], (25.0, 75.0))
            iq = float(q75 - q25)
            if iq >= 1e-12:
                last = iq
        ref[t : t + grid_step] = last
    return ref


def _build_scale_guards(
    X: np.ndarray,
    df: pd.DataFrame,
    feature_names: list[str],
    zero_inflated_active: bool = True,
    erode_ma_window: bool = True,
    warmup_floor: float = 1.0,
    ma_rows_per_session: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Scale guards for the impute-and-indicate feature pathology.

    Returns ``(ref_iqr, fixed_cols)`` for :func:`rolling_robust_scale`, or
    ``(None, None)`` when no guarding is needed (no availability columns →
    not an ``impute_indicate`` run → scaling is bit-for-bit unchanged).

    * ``*_avail_ma_*`` / ``*_active_ma_*`` (availability + occurrence-indicator
      rolling means) → *fixed*: passed through raw; scale is known ([0, 1]).
    * ``adj_<col>_ma_*`` whose ``<col>_avail`` column exists (the exog was
      imputed) → IQR floored at the causal expanding IQR over the *real*
      (available) values, with the availability mask **eroded by the MA window**
      so a half-imputed long average does not set the scale; for a
      *mode-inflated* magnitude the scale is taken over off-mode (active)
      values too, so neither a regime-degenerate window nor a point mass can
      collapse the scale and blow up the forecast.
    * everything else (HAR, calendar) → untouched.

    ``ma_rows_per_session`` (2026-10-15, ERODE_IN_BARS): set for a per-bar design, where a row
    is a session -- the erosion window of ``adj_<col>_ma_<w>`` (w bars) is then
    ceil(w x ma_rows_per_session / BARS_PER_SESSION) rows.  None: w rows (a row is a bar).
    """
    n, n_features = X.shape
    ref_iqr = np.zeros((n, n_features), dtype=np.float64)
    fixed = np.zeros(n_features, dtype=bool)
    touched = False
    for j, name in enumerate(feature_names):
        if "_avail_ma_" in name or "_active_ma_" in name:
            fixed[j] = True
            touched = True
            continue
        m = re.match(r"adj_(.+)_ma_(\d+)$", name)
        if m is None:
            continue
        avail_col = f"{m.group(1)}_avail"
        if avail_col in df.columns:
            mask = df[avail_col].to_numpy().astype(bool)[:n]
            # Erode the availability mask by the MA window. ``adj_<col>_ma_3125`` is a ~65-day
            # average, so for 3125 bars after the raw column first appears the MA is still mostly
            # the imputed median — a seam, not data. Taking the reference IQR over rows the raw
            # column merely *exists* on therefore measures the scale of a half-imputed average.
            # This is the whole of the ``adj_vix3m_ma_3125`` (|z| 50, 2009-11-06, three months into
            # vix3m coverage) and ``adj_vvix_ma_3125`` (|z| 30, 2012-06-19, three months into vvix
            # coverage) failures: only the long windows fail, and only near the seam.
            w = int(m.group(2))
            if ma_rows_per_session is not None:
                w = max(1, math.ceil(w * ma_rows_per_session / BARS_PER_SESSION))
            if erode_ma_window and w > 1:
                full = (
                    pd.Series(mask.astype(np.float64))
                    .rolling(w, min_periods=w)
                    .min()
                    .to_numpy()
                )
                eroded = np.nan_to_num(full) > 0.5
                if (
                    eroded.sum() >= 8
                ):  # keep the raw mask if erosion leaves nothing to fit
                    mask = eroded
            xj = X[:, j]
            # Zero-inflated magnitude (e.g. voldemand): also estimate the scale
            # over ACTIVE (non-zero) values — the zero point-mass carries no
            # dispersion and collapses the IQR otherwise. The indicator half of
            # the hurdle encoding is the ``<col>_active`` columns added in
            # ``load_and_transform``.
            if CAUSAL_GUARD_MODE:
                ref_iqr[:, j] = _expanding_real_iqr(
                    xj,
                    mask,
                    warmup_floor=warmup_floor,
                    exclude_mode=zero_inflated_active,
                )
                touched = True
                continue
            # 2026-10-15 (user): parked -- the whole-column test; see CAUSAL_GUARD_MODE
            inflated, mode = mode_inflated(xj)
            if zero_inflated_active and inflated:
                mask = mask & (xj != mode)
            ref_iqr[:, j] = _expanding_real_iqr(xj, mask, warmup_floor=warmup_floor)
            touched = True
    if not touched:
        return None, None
    return ref_iqr, fixed


def apply_masked_scaling(
    X: np.ndarray,
    X_raw: np.ndarray,
    df: pd.DataFrame,
    feature_names: list[str],
    train_win: int,
) -> np.ndarray:
    """Rescale every imputed exogenous column using OBSERVED rows only.

    Applied after the ordinary rolling scaler, replacing its output for the
    columns that have an availability mask. Columns without one (HAR, calendar,
    the indicators themselves) are left exactly as the ordinary path produced
    them, so a run with no imputed exog is bit-for-bit unchanged.

    This supersedes the value-side repairs -- the neutral-median substitution
    and the staleness bound -- both of which are off by default. Their measured
    behaviour is why: the median substitution fixed a 22%-coverage channel
    (voldemand 189.5 -> 7.4 rolling IQRs) and broke a 62%-coverage one
    (effspread 17.6 -> 932.6), because what helps a sparse series (stop using
    stale values) is exactly what degrades a dense one (introduce a constant).
    Masking the ESTIMATE has no such trade-off.
    """
    out = X
    n = X.shape[0]
    touched = 0
    # X_raw is the UNSCALED matrix. Masked scaling REPLACES the ordinary
    # scaler for these columns; composing the two would estimate the masked
    # median and IQR on a series the unmasked scaler had already distorted,
    # which is what a first version of this did (voldemand landed at 64.0
    # rolling IQRs instead of the 10.5 a single-column test predicted, and
    # numobs and vix went backwards).
    jobs = []
    for j, name in enumerate(feature_names):
        m = re.match(r"adj_(.+)_ma_(\d+)$", name)
        if m is None:
            continue
        obs_col = f"{m.group(1)}__obs"
        if obs_col not in df.columns:
            continue
        # A rung-r moving average is observed when the bars feeding it were;
        # require the window's centre bar rather than all r of them, which is
        # the same convention the availability rolling means already use.
        jobs.append((j, df[obs_col].to_numpy().astype(bool)[:n]))

    # Columns are independent, and each is a few hundred rolling-quantile
    # evaluations -- the dominant cost of a rebuild. Farm them out over
    # processes, which is exact: the workers run the same function on the same
    # inputs and only the scheduling changes. Threads would not do: numpy's
    # partition holds the GIL for the window sizes here.
    # Worker count. os.cpu_count() reports the NODE's cores, not the cores
    # this task was granted, so on a scheduler that enforces a per-task
    # address-space cap (SGE's h_vmem) the forks below can exceed it: each
    # child's address space is counted in full, so 8 children of a 15G
    # process bill about 120G. UNIF_SCALE_PROCS lets the job script pass
    # the real grant (SGE exports NSLOTS); unset, the historical default
    # stands so CARC/Slurm runs are byte-for-byte unchanged.
    _env_proc = os.environ.get("UNIF_SCALE_PROCS", "").strip()
    n_proc = (
        max(1, int(_env_proc))
        if _env_proc.isdigit() and int(_env_proc) > 0
        else max(1, min(os.cpu_count() or 1, 8))
    )
    if len(jobs) > 8 and n_proc > 1:
        # Submitted in batches rather than all at once: each task carries a
        # column copy, so queueing all ~500 upfront would hold about a
        # gigabyte of pickled arrays alongside a rebuild that is already the
        # heaviest process on the box.
        batch = n_proc * 4
        with cf.ProcessPoolExecutor(max_workers=n_proc) as pool:
            for s in range(0, len(jobs), batch):
                chunk = jobs[s : s + batch]
                futs = {
                    pool.submit(masked_rolling_scale_col, X_raw[:, j], mk, train_win): j
                    for j, mk in chunk
                }
                for fut in cf.as_completed(futs):
                    out[:, futs[fut]] = fut.result()
                    touched += 1
        note = f" across {n_proc} workers"
    else:
        for j, mk in jobs:
            out[:, j] = masked_rolling_scale_col(X_raw[:, j], mk, train_win)
            touched += 1
        note = ""
    if touched:
        print(f"[scale] masked rescaling applied to {touched} exogenous columns{note}")
    return out


def _backtest_and_save(
    df: pd.DataFrame,
    feature_names: list[str],
    fit_predict: FitPredict,
    hyperparams: dict,
    train_win_periods: int,
    horizon: int,
    start: int,
    end: int,
    halo: int,
    output_file: str,
    prescale: bool = False,
    diurnal_mode: str = "divide",
    burn_in: int | None = None,
) -> None:
    """Run a prepared DataFrame through the walk-forward backtest and save.

    ``burn_in`` (2026-10-15): rows dropped first; None = the HAR burn-in (3,125).  run_executor
    passes 0 for a global-lag segment under BURNIN_IN_BAR_SPACE (dropped on the bar grid).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``adj_RV``, ``baseline``, ``t``, plus all
        ``feature_names`` columns. The HAR-burn-in (``max_lag``) rows
        are dropped here before any extraction.
    feature_names : list[str]
        Column names in ``df`` that form the feature matrix.
    fit_predict : FitPredict
        Callable ``(X_chunk, y_chunk, train_win_periods, hyperparams)
        -> preds`` of shape ``(len(X_chunk) - train_win_periods,)``.
    hyperparams : dict
        Method-specific hyperparameters (typically a merge of method
        defaults and tuned overrides from ``--params-file``).
    train_win_periods : int
        Training window in 30-min periods.
    horizon : int
        Forecast horizon in 30-min periods.
    start, end, halo : int
        The framework slice for this task, in *post horizon-shift* index
        space (the fields of ``hpc_agent.template.SliceSpec``). ``[start,
        end)`` is the emit range; ``halo`` is the count of warm-up rows
        replayed before it, so the chunk processed is
        ``X[start - halo : end]`` and its first ``halo`` rows train the
        model without emitting predictions (``halo`` is sized to the
        training window). ``end < 0`` means "to the end"; a whole-series
        run is ``start=0, end=-1, halo=0``.
    output_file : str
        Output CSV path. ``<basename>_reduce.json`` is written
        alongside.
    prescale : bool, default False
        If True, rolling-robust-scale the whole feature matrix
        (``rolling_robust_scale``) *before* the chunk slice — the linear /
        distance-based methods (Ridge, kNN, spectral_knn) need feature
        standardization, and doing it whole-series keeps the scaler's
        look-back out of the backtest so a ``halo == train_win`` chunk is
        fungible. Tree methods leave this False (they neither scale nor
        prescale). The matching ``fit_predict`` receives an already-scaled
        ``X_chunk`` and passes it through to its
        :class:`~src.backtest.multi_stage.MultiStageBacktest`.
    """
    max_lag = resolve_har_lags()[-1] if burn_in is None else int(burn_in)
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon)

    # Rolling robust scaling is a bounded-window transform; doing it here,
    # whole-series, keeps its look-back out of the per-chunk backtest (see
    # rolling_robust_scale). The slice below then cuts the pre-scaled array.
    ref_iqr = None
    if prescale:
        rows_per_session = None
        if ERODE_IN_BARS:
            tt = pd.to_datetime(df["t"])
            if (
                tt.dt.time.nunique() < BARS_PER_SESSION // 2
            ):  # a time-of-day slice: a row is not a bar
                rows_per_session = len(df) / max(1, tt.dt.normalize().nunique())
        ref_iqr, fixed_cols = _build_scale_guards(
            X,
            df,
            feature_names,
            zero_inflated_active=(diurnal_mode != "rank"),
            ma_rows_per_session=rows_per_session,
        )
        X = rolling_robust_scale(
            X, train_win_periods, ref_iqr=ref_iqr, fixed_cols=fixed_cols
        )
        if LATE_START_WARMUP:
            # 2026-10-22 (user): late-start series neutral until the window holds W live rows
            X = late_start_warmup(X, df, feature_names, train_win_periods)
        if UNOBSERVED_NEUTRAL:
            # 2026-10-22 (user, E28): a value average whose window holds no observed bar is the neutral 0
            X = unobserved_neutral(X, df, feature_names)

    # Feature-health invariants, on every build. Ten defects of the same class (a transform whose
    # divisor degenerates, a fill constant that manufactures a predictor out of missingness) have
    # been found in this pipeline by *symptom* — a blown QLIKE, days later — and every one of them
    # is a seconds-long check on the finished matrix. Advisory by default; FEATURE_HEALTH_STRICT=1
    # makes a FAIL fatal. See src/diagnostics.py and analysis/diagnostics_backtest.py.
    check_and_report(
        X,
        feature_names,
        df=df,
        raw_cols=[c for c in df.columns if f"adj_{c}" in df.columns],
        ref_iqr=ref_iqr,
        train_win=train_win_periods,
        output_path=f"{os.path.splitext(output_file)[0]}_feature_health.csv",
        label=os.path.basename(os.path.splitext(output_file)[0]),
    )

    load_start = max(0, start - halo)
    actual_end = len(X) if end < 0 else end
    X_chunk = X[load_start:actual_end]
    y_chunk = y[load_start:actual_end]
    dates_chunk = dates.iloc[load_start:actual_end].reset_index(drop=True)
    baselines_chunk = baselines[load_start:actual_end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(
            f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})"
        )

    # The column names of X_chunk, forwarded under the established ``_*``
    # control-key convention (``_refit_frequency``, ``_incremental``): a
    # fit_predict that selects columns by MEANING (e.g. spectral_knn's
    # view_exog state block, which excludes target-derived ``har_ma_*``)
    # needs names, and re-deriving them caller-side would fork the source of
    # truth. Every other fit_predict strips ``_*`` keys and is unaffected.
    hyperparams = dict(hyperparams, _feature_names=list(feature_names))

    preds = fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams)

    oos_start = train_win_periods
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, index=False)
    save_chunk_reduce(results, output_file)
    print(f"Saved {len(results)} rows -> {output_file}")


class _DeferredHar:
    """LEAN_GLOBAL_LAGS: stands in for generate_har_features -- the names and the (column, lag) list in
    its order, no values; :meth:`attach` adds the values on the rows of a slice."""

    def __init__(self) -> None:
        self.spec: list[tuple[str, int, str]] = []

    def __call__(self, df, target_col="adj_RV", exog_cols=None, lags=None):
        lags = resolve_har_lags() if lags is None else list(lags)
        self.spec = [
            (col, lag, f"har_ma_{lag}" if col == target_col else f"{col}_ma_{lag}")
            for col in [target_col] + (exog_cols or [])
            for lag in lags
        ]
        return df, [name for _, _, name in self.spec]

    def attach(self, full: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
        """``seg_df`` (a slice of ``full`` carrying its row positions in ``_row``) with every trailing
        average of the spec, computed on ``full`` (generate_har_features' operations) at those rows."""
        rows = seg_df["_row"].to_numpy()
        vals = np.empty((len(rows), len(self.spec)), dtype=np.float64, order="F")
        for j, (col, lag, _) in enumerate(self.spec):
            vals[:, j] = (
                full[col]
                .rolling(window=lag, min_periods=1)
                .mean()
                .shift(1)
                .to_numpy()[rows]
            )
        har = pd.DataFrame(
            vals, index=seg_df.index, columns=[n for _, _, n in self.spec]
        )
        return pd.concat([seg_df.drop(columns="_row"), har], axis=1)


def _build_har_and_calendar(
    df, exog_cols, add_calendar, har_lags=None, session_edge=True, har_builder=None
):
    if DROP_ACTIVITY_MA:
        # 2026-10-19 (user, E6): no trailing averages of the activity indicators
        exog_cols = [c for c in exog_cols if not c.endswith("_active")]
    if LATE_START_WARMUP:
        # 2026-10-22 (user): the series' ages on the frame the trailing averages are computed on
        df = _series_age_columns(df, exog_cols)
    df, har_names = (generate_har_features if har_builder is None else har_builder)(
        df, target_col="adj_RV", exog_cols=exog_cols, lags=har_lags
    )
    if add_calendar and CALENDAR_THROUGH_OVERNIGHT:
        # 2026-10-19 (user, E9): the calendar stops at the overnight indicator
        cal = add_calendar_features(df)
        add_expiry_features(df)  # columns built (unchanged frame), not passed on
        feature_names = har_names + cal[: cal.index("is_overnight") + 1]
    elif add_calendar:
        # add_expiry_features after add_calendar_features: it reads ``is_close`` to build the
        # quad-witch closing-auction gate, which is where the index-rebalance flow actually prints.
        feature_names = har_names + add_calendar_features(df)
        feature_names += add_expiry_features(df)
        # HAR x open/close session-edge interactions (the distilled intraday vol-persistence
        # sign-flip: high recent RV -> LOWER future RV at the 09:30 open and 16:00 close/AH).
        # Materialized upstream so every model + the cached matrix inherit them; causal (the
        # is_open/is_close gates are pure functions of the timestamp). Mirrors the no-rebuild
        # resid_amortized._regime_interactions, identical naming (har_ma_{w}_x_{open,close}).
        # session_edge=False (2026-09-29, user decision) drops them for ONE-BAR-PER-SESSION
        # segments, where they carry nothing: is_open (hour 9) is 0 on every session bar
        # 10:00-16:00, so the six _x_open columns are all zero, and is_close (hours 16-19) is 1
        # on the 16:00 bar only, so at bar1600 the six _x_close columns are exact copies of
        # har_ma_1 .. har_ma_3125 (checked on results/model_diagnostics_1530/capture_bar1600*.npz)
        # and zero at every other bar. The pooled 48-bar path (segment=None) keeps them, where
        # they vary across bars. Every per-bar result produced before 2026-09-29 was fitted
        # WITH these 12 columns.
        if session_edge:
            for h in [
                c for c in har_names if c.startswith("har_ma_")
            ]:  # the 6 target-HAR cols
                for gate, suffix in (("is_open", "open"), ("is_close", "close")):
                    name = f"{h}_x_{suffix}"  # matches resid_amortized: har_ma_{w}_x_{open,close}
                    df[name] = df[h] * df[gate]
                    feature_names.append(name)
    else:
        feature_names = har_names
    if DROP_OVERNIGHT_INDICATOR:
        # 2026-10-19 (user, E9 + E15): no overnight indicator on a sample without overnight intervals
        feature_names = [n for n in feature_names if n != "is_overnight"]
    return df, feature_names


def load_and_transform(
    data_path: str,
    exog_cols: list[str],
    *,
    target_use_diurnal: bool,
    target_winsor_window: int | None,
    dropna_with_exog: bool,
    overnight_fill: bool = True,
    impute_indicate: bool = False,
    diurnal_mode: str = "divide",
    use_semantic: bool = True,
    ffill_limit: int | None | Literal["auto"] = "auto",
    legacy_avail: bool = False,
    neutralise_unobserved: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load raw data, apply RV + exog robust transforms, return (df, adj_exog).

    ``diurnal_mode`` (``"divide"`` | ``"rank"``) selects how the per-slot diurnal
    adjustment is done for the EXOG features: the default divide, or the
    division-free per-slot rank-Gauss (:func:`diurnal_rank`). The target's diurnal
    is always divide (its raw reconstruction needs the multiplicative baseline).

    Parameters
    ----------
    data_path : str
        Forwarded to ``load_raw_data``.
    exog_cols : list[str]
        Raw exog column names (pre-transform). May be empty.
    target_use_diurnal : bool
        If True, apply diurnal adjustment to the RV target transform.
        Tree methods: True. Ridge: False.
    target_winsor_window : int | None
        Winsorization window for the RV target. Tree methods: 240.
        Ridge: None (no winsorization beyond ``robust_transform``
        defaults).
    dropna_with_exog : bool
        Exog columns are forward-filled before this branch, so the only
        rows that can still have NaN exog are at the leading edge of the
        dataset (before any valid value). If True, drop those rows.
        Ridge: True (sklearn Ridge requires no NaN). Tree methods: False
        (trees handle NaN natively).
    overnight_fill : bool
        If True (default, legacy behaviour), fill matching exog columns with
        1.0 in their overnight window before ffill. This substring-matches
        moment columns (``sumret3_*`` etc.) and fills them with 1.0, which is
        a latent bug for non-ratio columns — set False for clean ffill-only.
    ffill_limit : int | None | "auto"
        Bars a forward fill may carry a value; None = unlimited.  "auto" (default, 2026-10-15):
        FFILL_LIMIT under APPLY_FFILL_LIMIT, else None (the pre-2026-10-15 default).
    """
    if ffill_limit == "auto":
        # 2026-10-15 (user): parked -- the default was None (unlimited); see APPLY_FFILL_LIMIT
        ffill_limit = FFILL_LIMIT if APPLY_FFILL_LIMIT else None
    df = load_raw_data(data_path, allow_missing=True)
    if exog_cols:
        if overnight_fill:
            apply_overnight_fills(df, exog_cols)
        # Availability is recorded BEFORE the forward fill. Recording it after
        # (the historical behaviour, kept under ``legacy_avail``) makes the
        # indicator mean "this series has started" rather than "this bar is
        # observed", because ffill carries the last value across every gap and
        # ``notna()`` is then True on stale rows too. Measured consequence on
        # the voldemand family: raw coverage ~33.6%, indicator mean 0.698 --
        # the indicator was 0 only for the leading years before the feed
        # begins. That defeats all three guards at once. The impute-and-
        # indicate median fill never fires (ffill left no NaN), the model
        # cannot tell fresh from stale, and ``_build_scale_guards`` computes
        # its "real values" IQR floor over the ffilled constants, so the floor
        # is itself degenerate and the rolling IQR collapses. The scaled
        # columns then reach 200+ IQRs, and on 13-15 October 2023 that
        # cancelled the backbone and drove the forecast two orders of
        # magnitude low (analysis/october_2023.py).
        obs_cols = {}
        for col in exog_cols:
            obs_cols[col] = df[col].notna().to_numpy(copy=True)
        # A finite staleness limit is the other half. Carrying a value forward
        # indefinitely is defensible for a short gap and indefensible for a
        # feed that stops: voldemand is 0% observed through 2024, so an
        # unlimited ffill presents a 2023 value as current for a whole year.
        # Past the limit the value returns to NaN and the impute-and-indicate
        # branch below replaces it with the neutral median.
        df[exog_cols] = df[exog_cols].ffill(limit=ffill_limit)
        for col in exog_cols:
            df[f"{col}__obs"] = (
                obs_cols[col] if not legacy_avail else df[col].notna().to_numpy()
            )
        if dropna_with_exog and not impute_indicate:
            df = df.dropna(subset=["RV"] + exog_cols).reset_index(drop=True)
        else:
            # impute_indicate keeps exog-missing rows (filled + flagged below).
            df = df.dropna(subset=["RV"]).reset_index(drop=True)

    transform_kwargs: dict = {"is_target": True}
    if target_use_diurnal:
        transform_kwargs["use_diurnal"] = True
    if target_winsor_window is not None:
        transform_kwargs["winsor_window"] = target_winsor_window
    if B_REGULAR_BARS_ONLY and target_use_diurnal:
        # 2026-10-15 (user): B over regular bars only; see B_REGULAR_BARS_ONLY
        excl = early_close_post_bars(df["t"])
        if excl.any():
            transform_kwargs["baseline_exclude"] = excl
    adj_rv, baseline = robust_transform(df, "RV", **transform_kwargs)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    adj_exog_cols: list[str] = []
    for col in exog_cols:
        adj_col = f"adj_{col}"
        adj_series, _ = robust_transform(
            df,
            col,
            use_transform=use_semantic,
            use_diurnal=True,
            diurnal_mode=diurnal_mode,
        )
        df[adj_col] = adj_series
        adj_exog_cols.append(adj_col)

    if impute_indicate and exog_cols:
        # Impute-and-indicate: instead of dropping rows where an exog is
        # structurally absent (which shrinks the sample and breaks cross-arm
        # QLIKE comparability), fill its adjusted feature with 0 (neutral) and
        # add a ``<col>_avail`` availability indicator. Every row is then
        # scored and the model can down-weight the feature where it's absent.
        for col in exog_cols:
            # Neutral fill must be in-distribution in the *transformed* space:
            # the feature's own median. 0 is wrong (e.g. adj_vix = log(vix), so
            # fillna(0) imputes vix = exp(0) = 1, a wild value that blew the
            # Ridge forecast up to QLIKE ~2.4). Median ≈ neutral, prescales to 0.
            adj = df[f"adj_{col}"]
            # Unobserved rows take the neutral median. This handles the
            # NUMERATOR half of the pathology: a forward-filled value is stale,
            # and on a sparsely observed channel it drifts arbitrarily far from
            # the current regime, so no scale estimate can rescue it (masking
            # the estimate alone still leaves voldemand at 231.9 rolling IQRs).
            #
            # On its own this substitution BREAKS moderately covered channels,
            # because replacing varying values with a constant collapses the
            # rolling IQR: effspread_vwstock went 17.6 -> 932.6 when it shipped
            # alone. That is the DENOMINATOR half, and it is fixed by
            # scaling.masked_rolling_scale_col, which estimates location and
            # scale from observed rows only.
            #
            # The two compose and neither works without the other -- measured
            # peak rolling IQRs, ffill/median x plain/masked scaling:
            #
            #   voldemand (cov 0.22)   43705.7 / 231.9 / 11813.6 / 10.5
            #   effspread (cov 0.62)      17.3 /  20.2 /    34.1 / 14.1
            #   sumbipow  (cov 0.62)      84.6 /  95.5 /    71.6 / 17.7
            #   vix       (cov 0.35)    3296.4 /   6.0 /  7141.6 /  5.6
            #
            # Testing them one at a time is why this took three attempts to
            # get right: each is harmful or inert alone and all the value is in
            # the interaction.
            obs = df[f"{col}__obs"].to_numpy().astype(bool)
            if neutralise_unobserved:
                adj = adj.where(pd.Series(obs, index=adj.index), np.nan)
            if CAUSAL_NEUTRAL_FILL:
                df[f"adj_{col}"] = adj.fillna(causal_neutral_fill(adj))
            else:
                # 2026-10-15 (user): parked -- the full-sample median; see CAUSAL_NEUTRAL_FILL
                df[f"adj_{col}"] = adj.fillna(adj.median())
            avail = f"{col}_avail"
            df[avail] = obs.astype("float64")
            adj_exog_cols.append(avail)

    # Hurdle (two-part) encoding for mode-inflated exog. A sparse signed flow
    # like voldemand is a mixture — a point mass at 0 ("no demand") plus a
    # continuous demand distribution — so a single robust scale is degenerate
    # (the point mass collapses the IQR). Add an occurrence indicator so the
    # model separates the extensive margin (is there demand at all) from the
    # intensive margin (how big); the magnitude's scale is taken over the
    # off-mode (active) values in ``_build_scale_guards``. The point mass need
    # not sit at zero — see ``ZERO_INFLATED_FRAC`` for ``numobs``.
    for col in exog_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        obs_col = f"{col}__obs"
        if ACTIVE_FROM_OBSERVED and obs_col in df.columns:
            # 2026-10-15 (user): the flag from observed prints; see ACTIVE_FROM_OBSERVED and
            # CAUSAL_MODE_INFLATION (the forward-filled rule is parked below)
            obs = df[obs_col].to_numpy(bool)
            if CAUSAL_MODE_INFLATION:
                act, ever = causal_activity(vals, obs)
                if ever:
                    df[f"{col}_active"] = act
                    adj_exog_cols.append(f"{col}_active")
                continue
            inflated, mode = mode_inflated(vals)
            if inflated:
                infl_o, mode_o = mode_inflated(vals[obs])
                if (
                    not infl_o
                ):  # not a hurdle series on its own prints: a constant column
                    act = np.zeros(len(vals))
                else:
                    a_o = (vals[obs] != mode_o).astype(float)
                    act = np.full(len(vals), float(np.round(a_o.mean())))
                    act[obs] = a_o
                df[f"{col}_active"] = act
                adj_exog_cols.append(f"{col}_active")
            continue
        inflated, mode = mode_inflated(vals)
        if inflated:
            active = f"{col}_active"
            # "off the point mass": for a zero-inflated flow that is "there was demand", for
            # ``numobs`` it is "this bar was not a full 30-observation bar" — i.e. a holiday or
            # half-session, which is exactly the information the 80-sigma tail was encoding.
            df[active] = (np.nan_to_num(vals, nan=mode) != mode).astype("float64")
            adj_exog_cols.append(active)

    return df, adj_exog_cols


def _iter_TOD_segment(
    df,
    *,
    segment,
    lag_scope,
    train_window,
    output_file,
    exog_cols,
    add_calendar,
    har_lags=None,
):
    """Yield (seg_name, job_df, feature_names, train_win_periods, job_output_file)
    for each time-of-day segment we need to backtest. ``seg_name`` is None when
    no segmentation is requested."""
    if segment is None:
        df, feature_names = _build_har_and_calendar(
            df, exog_cols, add_calendar, har_lags
        )
        yield None, df, feature_names, train_window * PERIODS_PER_DAY, output_file
        return

    segments = list(SEGMENT_DEFINITIONS) if segment == "all" else [segment]
    base, ext = os.path.splitext(output_file)

    deferred = None
    if lag_scope == "global":
        if LEAN_GLOBAL_LAGS:
            # 2026-10-19: the trailing averages on the full grid, kept on the segment's rows only
            deferred = _DeferredHar()
            df = df.copy()
            df["_row"] = np.arange(len(df))
        full = df if deferred is not None else None
        df, feature_names = _build_har_and_calendar(
            df,
            exog_cols,
            add_calendar,
            har_lags,
            session_edge=False,
            har_builder=deferred,
        )
        if BURNIN_IN_BAR_SPACE:
            # 2026-10-15 (user): the HAR burn-in on the bar grid, before the slice (run_executor
            # then passes burn_in=0); it used to be 3,125 rows of the slice
            df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)

    early = pd.to_datetime(list(NYSE_EARLY_CLOSE_DATES))
    for seg_name in segments:
        seg_df = slice_to_segment(df, seg_name)
        if DROP_EARLY_CLOSE_SESSIONS:
            # 2026-10-15 (user): no early-close session in a per-bar design; see DROP_EARLY_CLOSE_SESSIONS
            keep = ~pd.to_datetime(seg_df["t"]).dt.normalize().isin(early)
            seg_df = seg_df.loc[keep.to_numpy()].reset_index(drop=True)
        if deferred is not None:
            seg_df = deferred.attach(full, seg_df)
        if seg_df.empty:
            print(f"No data for segment '{seg_name}'. Skipping.")
            continue
        if lag_scope == "intra":
            seg_df, feature_names = _build_har_and_calendar(
                seg_df, exog_cols, add_calendar, har_lags, session_edge=False
            )
        train_win_periods = compute_segment_train_window(seg_df["t"], train_window)
        yield (
            seg_name,
            seg_df,
            feature_names,
            train_win_periods,
            f"{base}_{seg_name}{ext}",
        )


def run_executor(
    method_name: str,
    fit_predict: FitPredict,
    hyperparams: dict,
    *,
    data_path: str,
    output_file: str,
    horizon: int,
    train_window: int,
    start: int,
    end: int,
    halo: int,
    exog_cols: list[str],
    segment: str | None,
    lag_scope: str,
    add_calendar: bool,
    target_use_diurnal: bool,
    target_winsor_window: int | None,
    dropna_with_exog: bool,
    overnight_fill: bool = True,
    impute_indicate: bool = False,
    diurnal_mode: str = "divide",
    prescale: bool = False,
    seed: int = 42,
    har_lags: list[int] | None = None,
) -> None:
    """Top-level scaffold. Loads data, builds features, dispatches backtest.

    Per-method scripts (`ml_xgboost.py`, etc.) call this from their
    ``run()`` after reading the framework slice. ``start`` / ``end`` /
    ``halo`` are the plain-int fields of the active
    ``hpc_agent.template.SliceSpec`` (the chunked-axis seam) — see
    ``_backtest_and_save`` for their meaning; a whole-series run passes
    ``start=0, end=-1, halo=0``.

    Notes
    -----
    * ``segment`` is None for tree methods (XGB/LGBM/RF). Ridge is the
      only method that uses segments.
    * ``add_calendar`` is True for tree methods, False for Ridge.
    * ``prescale`` is True only for the linear methods (Ridge): it
      rolling-robust-scales the feature matrix whole-series so a chunked
      backtest needs only a train-window halo. Tree methods leave it False.
    * ``method_name`` is informational — used in log lines only. The
      per-method data-prep invariants it once keyed a drift-check on are
      now inline literals in each executor's ``run()`` (B2).
    """
    del seed  # reserved; per-method scripts can wire seed into model_fn directly

    if har_lags is not None:
        burn_in = resolve_har_lags()[-1]
        if not har_lags or min(har_lags) < 1 or max(har_lags) > burn_in:
            raise ValueError(
                f"har_lags must be non-empty with rungs in [1, {burn_in}]: the "
                f"HAR burn-in drop is fixed at {burn_in} rows (arm-invariant "
                f"index space); a rung beyond it would be under-warmed."
            )

    df, adj_exog_cols = load_and_transform(
        data_path,
        exog_cols,
        target_use_diurnal=target_use_diurnal,
        target_winsor_window=target_winsor_window,
        dropna_with_exog=dropna_with_exog,
        overnight_fill=overnight_fill,
        impute_indicate=impute_indicate,
        diurnal_mode=diurnal_mode,
    )

    for seg_name, job_df, feature_names, train_win, out_file in _iter_TOD_segment(
        df,
        segment=segment,
        lag_scope=lag_scope,
        train_window=train_window,
        output_file=output_file,
        exog_cols=adj_exog_cols,
        add_calendar=add_calendar,
        har_lags=har_lags,
    ):
        if seg_name is not None:
            print(
                f"{'=' * 20} {method_name.upper()} SEGMENT: {seg_name.upper()} {'=' * 20}"
            )
            print(f"Window: {train_win} periods ({train_window} days)")
        _backtest_and_save(
            job_df,
            feature_names,
            fit_predict,
            hyperparams,
            train_win,
            horizon,
            start,
            end,
            halo,
            out_file,
            prescale,
            diurnal_mode,
            burn_in=0
            if (seg_name is not None and lag_scope == "global" and BURNIN_IN_BAR_SPACE)
            else None,
        )
