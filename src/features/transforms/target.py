"""Target-side transforms: diurnal adjust, semantic transform, winsorize, horizon shift.

These modify the target (or any column treated like a target): they change
values, they don't add new columns. The canonical pipeline is
:func:`robust_transform` which chains diurnal -> semantic -> winsorize, and
:func:`apply_horizon_shift` which aligns features at *t* with target at *t+h*.

Constants used across the project (PERIODS_PER_DAY, etc.) live here because
they're rooted in the 30-min bar / RTH-day shape the target transforms assume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIODS_PER_DAY: int = 48
DIURNAL_WINDOW: int = 20
DIURNAL_MIN_PERIODS: int = 5
# Floor for the signed-feature per-slot rolling-std divisor, as a fraction of
# the feature's typical (median) per-slot std. Stops a transiently-degenerate
# std from blowing the diurnal-adjusted value up (see ``diurnal_adjust``).
DIURNAL_STD_FLOOR_FRAC: float = 0.1
WINSOR_LOWER_Q: float = 0.05
WINSOR_UPPER_Q: float = 0.95
SKIP_VARS: set[str] = {
    "hour",
    "DOW",
    "t",
    "date",
    "is_overnight",
    "hour_sin",
    "hour_cos",
    "DOW_0",
    "DOW_1",
    "DOW_2",
    "DOW_3",
    "DOW_4",
}
DIURNAL_EXCLUDED: set[str] = SKIP_VARS | {"vix", "sentiment"}


# ---------------------------------------------------------------------------
# Diurnal adjustment
# ---------------------------------------------------------------------------


def diurnal_adjust(
    series: pd.Series,
    time_of_day_series: pd.Series,
    has_negatives: bool,
    window: int = DIURNAL_WINDOW,
    min_periods: int = DIURNAL_MIN_PERIODS,
) -> tuple[pd.Series, pd.Series]:
    """Remove intraday seasonality via rolling per-slot baseline.

    Causality (audit-relevant):
    Within each time-of-day slot, the rolling statistic is followed by
    ``.shift(1)`` so that the baseline at time *t* is computed from the
    ``window`` most recent in-slot observations strictly before *t*. The
    adjusted value ``series[t] / baseline[t]`` therefore uses no information
    from time *t* itself or later.

    Returns
    -------
    (adjusted, baseline) where adjusted = series / baseline.
    """
    df = pd.DataFrame({"val": series, "slot": time_of_day_series})

    if has_negatives:
        baseline = df.groupby("slot")["val"].transform(
            lambda g: g.rolling(window, min_periods=min_periods).std().shift(1)
        )
        # The per-slot rolling std is a *divisor*: a transiently near-zero std
        # (flat / ffilled segments) must not amplify the incoming value. The old
        # ``replace(0, 1.0)`` only caught an *exactly* zero std and assumed an
        # O(1) series, so a tiny-but-nonzero std on a large-scale signed feature
        # (voldemand ~ ±1e6) blew the adjusted value up to ~±1e13. Floor the std
        # at a fraction of its own typical (median) level instead — the diurnal
        # analogue of the rolling-scaler IQR floor. Only bites a degenerate std;
        # healthy signed features (e.g. sumret) never go below the floor.
        typical = baseline.replace(0, np.nan).abs().median()
        floor = (
            DIURNAL_STD_FLOOR_FRAC * typical
            if (pd.notna(typical) and typical > 0)
            else 1.0
        )
        baseline = baseline.clip(lower=floor).fillna(floor)
    else:
        baseline = df.groupby("slot")["val"].transform(
            lambda g: g.rolling(window, min_periods=min_periods).mean().shift(1)
        )
        # Treat 0 the same as NaN (flat ffilled segments produce zero rolling
        # mean); baseline=1.0 passes through the raw value safely.
        baseline = baseline.replace(0, 1.0).fillna(1.0)

    adjusted = series / baseline
    return adjusted, baseline


def diurnal_rank(
    series: pd.Series,
    time_of_day_series: pd.Series,
    window: int = DIURNAL_WINDOW,
    min_periods: int = DIURNAL_MIN_PERIODS,
    gaussianize: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Rank-based diurnal adjustment — the division-free alternative to ``diurnal_adjust``.

    Within each time-of-day slot, map each value to its plotting-position percentile
    among the trailing ``window`` in-slot observations (causal: a slot's series is
    chronologically ordered and the rolling window ends at the current bar), then —
    if ``gaussianize`` — through the inverse-normal CDF. This pins the *whole* per-slot
    marginal (location, scale AND shape), and being **division-free** it cannot blow
    up: zero-heavy / intermittent signed slots that collapse a per-slot std (and
    explode the divide form) instead map to tied mid-ranks. A rank has no
    multiplicative baseline, so ``baseline`` is 1.0 — this form is for FEATURES, not
    the target (whose raw reconstruction needs the divide baseline).

    Returns ``(adjusted, baseline=1.0)``.
    """
    from scipy.special import ndtri  # inverse standard-normal CDF (probit)

    g = pd.DataFrame({"val": series, "slot": time_of_day_series}).groupby("slot")["val"]
    rank = g.transform(lambda s: s.rolling(window, min_periods=min_periods).rank())
    count = g.transform(lambda s: s.rolling(window, min_periods=min_periods).count())
    u = ((rank - 0.5) / count).clip(1e-6, 1.0 - 1e-6).to_numpy()
    adj = ndtri(u) if gaussianize else (2.0 * u - 1.0)
    adj = np.where(np.isfinite(adj), adj, 0.0)  # warm-up rows (count < min_periods) → neutral
    return pd.Series(adj, index=series.index), pd.Series(1.0, index=series.index)


# ---------------------------------------------------------------------------
# Semantic (column-name-based) transforms
# ---------------------------------------------------------------------------


def apply_semantic_transform(
    series: pd.Series,
    col_name: str,
    has_negatives: bool,
    allow_missing: bool = False,
) -> pd.Series:
    """Apply a variance-stabilising transform chosen by column name.

    Rules (checked in order):
    1. name contains ret2 / RV / turnover / bipow / effspread -> sqrt
    2. name contains autocov -> sign(x) * sqrt(|x|)
    3. name contains ret3 -> cbrt
    4. name contains ret4 -> fourth root (x ** 0.25)
    5. has_negatives or name contains sumabsret -> identity (NaN -> 0)
    6. default -> log
    """
    name = col_name.lower()

    if any(tok in name for tok in ("ret2", "rv", "turnover", "bipow", "effspread")):
        return np.sqrt(series)

    if "autocov" in name:
        return np.sign(series) * np.sqrt(np.abs(series))

    if "ret3" in name:
        return np.cbrt(series)

    if "ret4" in name:
        return np.power(np.abs(series), 0.25) * np.sign(series)

    if has_negatives or "sumabsret" in name:
        out = series.copy()
        if not allow_missing:
            out = out.fillna(0.0)
        return out

    # default: log (guard against non-positive values)
    return np.log(series.clip(lower=1e-12))


# ---------------------------------------------------------------------------
# Rolling winsorization
# ---------------------------------------------------------------------------


def rolling_winsorize(
    series: pd.Series,
    window: int = 240,
    allow_missing: bool = False,
    is_target: bool = False,
) -> pd.Series:
    """Clip values to rolling 5th / 95th quantile bounds (shifted by 1).

    Parameters
    ----------
    series : pd.Series
    window : int
        Lookback window for quantile estimation.
    allow_missing : bool
        If True and not is_target, use nanquantile-style (min_periods=1).
    is_target : bool
        Targets never use nanquantile even when allow_missing is True.
    """
    use_nan = allow_missing and not is_target
    min_per = 1 if use_nan else window

    lower = (
        series.rolling(window, min_periods=min_per).quantile(WINSOR_LOWER_Q).shift(1)
    )
    upper = (
        series.rolling(window, min_periods=min_per).quantile(WINSOR_UPPER_Q).shift(1)
    )
    return series.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Full target pipeline
# ---------------------------------------------------------------------------


def robust_transform(
    df: pd.DataFrame,
    col_name: str,
    time_col: str = "time_of_day",
    use_transform: bool = True,
    use_diurnal: bool = True,
    allow_missing: bool = False,
    winsor_window: int | None = None,
    is_target: bool = False,
    diurnal_mode: str = "divide",
) -> tuple[pd.Series, pd.Series]:
    """Chain diurnal_adjust -> apply_semantic_transform -> rolling_winsorize.

    ``diurnal_mode``: ``"divide"`` (default) uses :func:`diurnal_adjust`;
    ``"rank"`` uses :func:`diurnal_rank` for FEATURES (division-free per-slot
    rank-Gauss) and the rank output is returned directly — the semantic transform
    and winsorize are skipped (rank-Gauss already normalizes, and sqrt/log of a
    signed rank value would break). The target (``is_target=True``) always uses
    divide regardless, since its raw reconstruction needs the multiplicative baseline.

    Returns ``(adjusted_series, baseline)``.
    """
    if col_name in SKIP_VARS:
        return df[col_name].copy(), pd.Series(1.0, index=df.index)

    series = df[col_name].copy()
    has_negatives = bool((series.dropna() < 0).any())

    baseline = pd.Series(1.0, index=df.index)
    if use_diurnal and col_name not in DIURNAL_EXCLUDED and time_col in df.columns:
        if diurnal_mode == "rank" and not is_target:
            return diurnal_rank(series, df[time_col])  # final feature; skip semantic + winsor
        series, baseline = diurnal_adjust(series, df[time_col], has_negatives)

    if use_transform:
        series = apply_semantic_transform(
            series, col_name, has_negatives, allow_missing=allow_missing
        )

    ww = winsor_window if winsor_window is not None else 240
    series = rolling_winsorize(
        series, window=ww, allow_missing=allow_missing, is_target=is_target
    )

    return series, baseline


# ---------------------------------------------------------------------------
# Horizon shift
# ---------------------------------------------------------------------------


def apply_horizon_shift(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    baselines: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    """Align features at time *t* with target at *t + horizon*.

    When horizon <= 1 the arrays are returned unchanged.
    """
    if horizon <= 1:
        return X, y, dates, baselines
    shift = horizon - 1
    return (
        X[:-shift],
        y[shift:],
        dates.iloc[:-shift].reset_index(drop=True),
        baselines[shift:],
    )
