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
# Two-sided band, as a fraction, for the per-slot baseline around that slot's own causal expanding
# median baseline: the baseline is clipped to ``[frac, 1/frac] x slot_reference``.
#
# Why a *per-slot* band is the fix and the existing global floor is not. Every one of the twelve
# columns the invariants flag at max|z| 68-82 traces to the same row-level event: a 30-min slot whose
# rolling baseline momentarily collapses relative to that slot's own typical level. Measured, at the
# argmax of each:
#
#     sumpret2_vwstock  03:00  baseline 5.7e-08 vs slot median 1.1e-06  (19x low) -> |z| 82
#     sumbipow_ewstock  03:00  baseline 1.9e-08 vs slot median 4.4e-07  (22x low) -> |z| 77
#     stocktwits_sent.  05:30  baseline 0.0142  vs slot median 0.143    (10x low) -> |z| 70
#
# The existing ``DIURNAL_STD_FLOOR_FRAC`` floor is pooled over slots, so it cannot see this: the
# overnight slots' legitimate baselines are orders of magnitude below the RTH slots', and a global
# floor either never binds (unsigned branch, which has no floor at all beyond exactly-zero) or binds
# at a level 10x below what the slot itself needs (``stocktwits_sentiment``, where the pooled floor
# did bind and still permitted a 10x amplification).
#
# It is two-sided because a transiently *large* baseline shrinks the value toward zero and destroys
# signal just as surely as a small one inflates it, and because a symmetric band cannot bias the
# adjustment in either direction. It is not a patch so much as shrinkage: the baseline is a
# ``DIURNAL_WINDOW``-observation rolling statistic of a heavily right-skewed quantity, so its
# sampling error is large, and constraining it toward a longer-run per-slot level is better
# estimation rather than less adjustment.
DIURNAL_SLOT_BAND_FRAC: float = 0.5
DIURNAL_SLOT_REF_MIN: int = 20  # in-slot observations before the reference median is usable
# Trailing in-slot observations the reference median is taken over: one observation per trading day
# per slot, so 250 is ~one year — long enough to be a far better-estimated statistic than the
# DIURNAL_WINDOW = 20 estimate it disciplines, short enough to track a volatility regime.
DIURNAL_SLOT_REF_WINDOW: int = 250
# Soft (differentiable) guards. The hard guards below — the per-slot band's clip, the std floor's
# max, the winsoriser's clip — are the degenerate limits of smooth boundary-repelling maps (§19.6 of
# the alpha-manifestation writeup: the r + 1/r / GIG-prior view of bounding away from the cone
# boundary). The soft forms matter only if anything is ever fit end-to-end by gradient (a clip's
# gradient is zero outside the band and kinked at it), so they ship OFF: flag on to get, everywhere
# a guard acts, a tanh-in-log band, a p-norm smooth max, and a tanh winsoriser — each the C-infinity
# version of its hard counterpart, near-identity in the healthy region, asymptotic to the same
# bounds. Flag off is bit-identical to the historical chain.
SOFT_GUARDS: bool = False
SOFT_MAX_K: float = 8.0  # sharpness of the smooth max (p-norm order); higher = closer to hard
# Order of the smooth saturator g(u) = u(1+|u|^k)^(-1/k) used by the soft band and soft winsoriser.
# g is identity to O(u^(k+1)) near 0 and asymptotes to +-1, so the interior of the healthy band is
# untouched to numerical precision while the boundary is approached smoothly. The first cut used
# tanh, and the equivalence audit rejected it BY THE BARRIER'S OWN REQUIREMENT (leave healthy data
# alone): tanh(0.5) = 0.462 distorts the mid-band by 7.6%, dragging hard-vs-soft correlations on the
# guarded columns down to 0.95. At k = 6 the mid-band error is ~1.5% and the tails are unchanged.
SOFT_SAT_K: float = 6.0


def _soft_sat(u: np.ndarray, k: float = SOFT_SAT_K) -> np.ndarray:
    """Smooth saturation to [-1, 1]: identity to O(u^(k+1)), the C-inf version of clip(u, -1, 1)."""
    return u * (1.0 + np.abs(u) ** k) ** (-1.0 / k)
# If the per-slot rolling-std divisor is pinned at ``DIURNAL_STD_FLOOR_FRAC``'s floor for more
# than this fraction of rows, ``diurnal_adjust`` is not adjusting anything — it has degenerated
# into division by a single global constant, which cannot track a feature whose scale drifts.
# Such a column gets :func:`asinh_stabilize` instead (see ``robust_transform``).
DIURNAL_PINNED_MAX: float = 0.5
# Halflife (days) of the causal |x|-EWMA that sets asinh's scale. 250d matches the Part-4
# optimal train window; the result is insensitive to it (60d and 250d score within 0.0001).
ASINH_SCALE_HALFLIFE_DAYS: int = 250
WINSOR_LOWER_Q: float = 0.01  # 1/99 per author decision 2026-08-06 (was 5/95):
WINSOR_UPPER_Q: float = 0.99  # clip less of the tail; magnitudes are signal.
# Per-slot winsorisation is implemented (``rolling_winsorize(slots=...)``) and **off by default,
# because it was measured and it fails its own invariant.** On the thirteen flagged columns it is
# spectacular on the bulk — extreme *rows* fall from 4,383 to 287 and columns past |z| 20 from 37 to
# 17 — and it makes three ``_ma_1`` columns far worse: ``adj_sumpret2_vwstock_ma_1`` goes 78 -> 219,
# ``adj_sumabsret_ewstock_ma_1`` 59 -> 164, ``adj_sumpret2_ewstock_ma_1`` 52 -> 139. The mechanism is
# self-inflicted: clipping 10% of *every slot* into the body tightens the distribution enough that
# ``rolling_robust_scale``'s IQR collapses, and a value sitting at its own (causal, therefore
# sometimes generous) trailing bound is then a very large number of very small IQRs from the median.
# Trading a 15x reduction in extreme rows for a new 219-sigma column is not a trade worth making.
#
# The obvious variant — a per-slot 1/99 bound, which caps the tail without crushing the body's
# dispersion — is the named next step and is deliberately NOT run here: the quantile pair is the one
# free parameter, and cycling it until an arm clears is the pattern §10 of the writeup exists to
# prevent, invariant or not.
WINSOR_BY_SLOT_DEFAULT: bool = False
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
# Columns excluded from diurnal adjustment because they are already scale-free *levels* — an
# implied-vol index quote, or a bounded sentiment ratio — for which dividing by a per-slot
# dispersion estimate can only add noise.
DIURNAL_EXCLUDED: set[str] = SKIP_VARS | {"vix", "sentiment"}
# ``vix`` and ``sentiment`` in that set are *stems*, not column names, and the exact-membership test
# they were used with made two of the three entries dead: the panel's columns are ``vix``, ``vvix``,
# ``vix3m``, ``stocktwits_sentiment`` — so ``vvix`` and ``vix3m`` were being diurnally adjusted
# against the constant's stated intent, and no column named ``sentiment`` has ever existed, which
# left ``stocktwits_sentiment`` (a ratio bounded in [-1, 1]) dividing by a per-slot rolling std that
# collapses 10x on overnight slots and reaching |z| 70. Matched as stems below.
DIURNAL_EXCLUDED_STEMS: tuple[str, ...] = ("vix", "sentiment")


def is_diurnal_excluded(col_name: str) -> bool:
    """True if ``col_name`` should skip the diurnal adjustment (see ``DIURNAL_EXCLUDED``)."""
    if col_name in DIURNAL_EXCLUDED:
        return True
    name = col_name.lower()
    return any(stem in name for stem in DIURNAL_EXCLUDED_STEMS)


# ---------------------------------------------------------------------------
# Diurnal adjustment
# ---------------------------------------------------------------------------


def _soft_floor(x: pd.Series, floor: float, k: float = SOFT_MAX_K) -> pd.Series:
    """Smooth max(x, floor) via the p-norm: (x^k + floor^k)^(1/k). Near-identity for x >> floor."""
    xa = x.to_numpy(dtype=np.float64)
    out = np.where(
        xa > 0,
        np.exp(np.logaddexp(k * np.log(np.clip(xa, 1e-300, None)), k * np.log(floor)) / k),
        floor,
    )
    return pd.Series(out, index=x.index)


def _soft_band(ratio: np.ndarray, frac: float) -> np.ndarray:
    """Smooth clip of a positive ratio into [frac, 1/frac]: saturation in the log (metric) coordinate."""
    w = -np.log(frac)
    u = np.log(np.clip(ratio, 1e-300, None))
    return np.exp(w * _soft_sat(u / w))


def _band_to_slot_level(
    baseline: pd.Series, slots: pd.Series, frac: float, soft: bool = False
) -> pd.Series:
    """Shrink a per-slot rolling baseline into ``[frac, 1/frac]`` of that slot's own longer-run level.

    The reference is a **rolling** per-slot median over ``DIURNAL_SLOT_REF_WINDOW`` in-slot
    observations, not an expanding one. That distinction is the whole design. An expanding median is
    dominated by the start of the sample, so in a high-volatility regime the slot's legitimate
    baseline sits several times above its own history and the band would clip the *regime* — which is
    not what this is for. A trailing-year median tracks the regime while still resting on ~12x as
    many observations as the ``DIURNAL_WINDOW`` estimate it disciplines, so the operation is plain
    shrinkage of a noisy statistic toward a better-estimated one of the same quantity, with no
    regime cost. Regime *level* is in any case handled downstream by ``rolling_robust_scale``'s
    250-day median/IQR, which is far better equipped for it than a 20-observation per-slot mean.

    Causal without an extra shift: every value entering the median is itself a rolling statistic over
    observations strictly before its own row (``diurnal_adjust`` shifts by 1), so the median at row
    ``t`` is a function of data before ``t`` only — and not shifting keeps the band usable
    ``DIURNAL_SLOT_REF_MIN`` in-slot observations earlier.

    Where a slot has less than ``DIURNAL_SLOT_REF_MIN`` observations of history the reference is NaN
    and the clip is a no-op (pandas leaves values unchanged against NaN bounds), so behaviour in a
    column's warm-up is unchanged and the caller's global floor remains the backstop there.

    ``frac <= 0`` disables the band entirely, which is how the pre-fix chain is reproduced exactly.
    """
    if frac <= 0:
        return baseline
    ref = baseline.abs().groupby(slots).transform(
        lambda g: g.rolling(
            DIURNAL_SLOT_REF_WINDOW, min_periods=DIURNAL_SLOT_REF_MIN
        ).median()
    )
    ref = ref.where(ref > 0)
    if not soft:
        return baseline.clip(lower=frac * ref, upper=ref / frac)
    ok = ref.notna() & (baseline > 0)
    out = baseline.copy()
    out[ok] = ref[ok] * _soft_band((baseline[ok] / ref[ok]).to_numpy(), frac)
    return out


def diurnal_adjust(
    series: pd.Series,
    time_of_day_series: pd.Series,
    has_negatives: bool,
    window: int = DIURNAL_WINDOW,
    min_periods: int = DIURNAL_MIN_PERIODS,
    slot_band: float = DIURNAL_SLOT_BAND_FRAC,
    soft: bool | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Remove intraday seasonality via rolling per-slot baseline.

    Causality (audit-relevant):
    Within each time-of-day slot, the rolling statistic is followed by
    ``.shift(1)`` so that the baseline at time *t* is computed from the
    ``window`` most recent in-slot observations strictly before *t*. The
    adjusted value ``series[t] / baseline[t]`` therefore uses no information
    from time *t* itself or later.

    ``slot_band`` is the two-sided per-slot band of :func:`_band_to_slot_level`, applied to *both*
    branches — it is the fix for the twelve columns the feature-health invariants flagged at
    max|z| 68-82, all of which are a within-slot baseline collapse (see
    ``DIURNAL_SLOT_BAND_FRAC``). Pass ``0`` to reproduce the pre-fix chain bit-for-bit.

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
        if soft is None:
            soft = SOFT_GUARDS
        baseline = _band_to_slot_level(baseline, df["slot"], slot_band, soft=soft)
        baseline = (
            _soft_floor(baseline, floor).fillna(floor)
            if soft
            else baseline.clip(lower=floor).fillna(floor)
        )
    else:
        baseline = df.groupby("slot")["val"].transform(
            lambda g: g.rolling(window, min_periods=min_periods).mean().shift(1)
        )
        baseline = _band_to_slot_level(
            baseline, df["slot"], slot_band, soft=SOFT_GUARDS if soft is None else soft
        )
        # Treat 0 the same as NaN (flat ffilled segments produce zero rolling
        # mean); baseline=1.0 passes through the raw value safely. Only reachable
        # now where the per-slot band had too little history to have an opinion.
        baseline = baseline.replace(0, 1.0).fillna(1.0)

    adjusted = series / baseline
    return adjusted, baseline


def asinh_stabilize(
    series: pd.Series, halflife_days: int = ASINH_SCALE_HALFLIFE_DAYS
) -> pd.Series:
    """Variance-stabilise a signed, scale-drifting feature: ``asinh(x / s_t)``.

    The stabilizer :func:`apply_semantic_transform` rule 5 never had. ``asinh`` is the signed
    analogue of ``log``: linear near zero (so a zero point mass is untouched — ``asinh(0) = 0``,
    no clipping, no sign loss) and logarithmic in the tails, so a feature whose scale grows by
    an order of magnitude mid-sample stops producing hundred-sigma values.

    ``s_t`` is an EWMA of ``|x|`` over **active** (observed, non-zero) rows, shifted one bar and
    floored at 1% of its own expanding median. Three properties matter:

    * **Causal** — ``.shift(1)`` means row ``t`` divides by a scale built strictly before ``t``.
    * **Adaptive** — it tracks a secular scale expansion, which a fixed constant cannot.
    * **Non-degenerate** — active-only (the zero mass carries no dispersion, the same reasoning
      as ``executor._build_scale_guards``) plus the expanding floor, so it can never become the
      near-zero divisor that this whole class of bug comes from.

    Written for ``voldemand`` (scale up ~11x across 2011-2023 with the 0DTE build-out, which
    put |z| ~ 2000 into the feature matrix), but stated generally: where a signed feature is
    well-behaved, ``|x| / s_t`` is O(1) and ``asinh`` is near-identity, so applying it costs
    nothing.
    """
    x = pd.to_numeric(series, errors="coerce")
    active = x.abs().where((x != 0) & x.notna())
    s = active.ewm(
        halflife=halflife_days * PERIODS_PER_DAY, min_periods=50, ignore_na=True
    ).mean()
    s = s.shift(1).ffill()
    s = np.maximum(s, 0.01 * s.expanding(min_periods=50).median()).replace(0.0, np.nan)
    med = s.median()
    s = s.fillna(med if pd.notna(med) and med > 0 else 1.0)
    return pd.Series(np.arcsinh(x / s), index=series.index)


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


def has_name_stabilizer(col_name: str) -> bool:
    """True if rules 1-4 of :func:`apply_semantic_transform` give this column a stabilizer.

    Rule 5 (identity for signed features) is the only branch with *no* variance stabilizer, and
    is therefore the only one ``robust_transform``'s degenerate-divisor gate may override. A
    column that already gets sqrt / cbrt / fourth-root is left alone: ``sumret3``'s divisor is
    also pinned (93% of rows), but ``cbrt`` bounds it perfectly well, and swapping a working
    stabilizer on the strongest bucket would be a change with no evidence behind it.
    """
    name = col_name.lower()
    return (
        any(tok in name for tok in ("ret2", "rv", "turnover", "bipow", "effspread"))
        or "autocov" in name
        or "ret3" in name
        or "ret4" in name
    )


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
    slots: pd.Series | None = None,
    soft: bool | None = None,
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
    slots : pd.Series, optional
        Time-of-day labels. When given, the quantiles are computed **within each slot** over the
        last ``window`` *in-slot* observations rather than over the last ``window`` bars of all
        slots pooled.

        Why that matters, measured. A pooled window is ~62% overnight bars, and the overnight
        distribution of a diurnally-adjusted cross-sectional stock aggregate is far more dispersed
        than the RTH one — so the pooled 95th percentile is set *by* the overnight tail and can
        never bind on it. At the argmax of ``adj_sumpret2_vwstock`` (2024-01-17 03:00) the pooled
        upper bound equals the value itself, 46.03: twelve or more of the trailing 240 bars were
        already past it, so the winsoriser had nothing to clip against. The in-slot p99 of that
        column at 03:00 is 5.6e-05 against a raw print of 1.2e-04 — the per-slot bound binds hard
        where the pooled one is blind. Same insight as ``DIURNAL_SLOT_BAND_FRAC``: a per-slot
        pathology needs a per-slot statistic.

        This does not clip *more* rows — it is the same 5/95 rule — it clips the right ones.
    """
    use_nan = allow_missing and not is_target
    min_per = 1 if use_nan else window

    if slots is not None:
        def _q(g: pd.Series, q: float) -> pd.Series:
            return g.rolling(window, min_periods=min_per).quantile(q).shift(1)

        grouped = series.groupby(slots)
        lower = grouped.transform(_q, WINSOR_LOWER_Q)
        upper = grouped.transform(_q, WINSOR_UPPER_Q)
    else:
        lower = (
            series.rolling(window, min_periods=min_per).quantile(WINSOR_LOWER_Q).shift(1)
        )
        upper = (
            series.rolling(window, min_periods=min_per).quantile(WINSOR_UPPER_Q).shift(1)
        )
    if soft if soft is not None else SOFT_GUARDS:
        c = (upper + lower) / 2.0
        h = (upper - lower) / 2.0
        ok = lower.notna() & upper.notna() & (h > 0)
        out = series.clip(lower=lower, upper=upper)  # hard where bounds degenerate
        out[ok] = c[ok] + h[ok] * _soft_sat(((series - c) / h)[ok].to_numpy())
        return out
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
    signed_stabilizer: bool = True,
    slot_band: float | None = None,
    winsor_by_slot: bool | None = None,
    soft_guards: bool | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Chain diurnal_adjust -> apply_semantic_transform -> rolling_winsorize.

    ``diurnal_mode``: ``"divide"`` (default) uses :func:`diurnal_adjust`;
    ``"rank"`` uses :func:`diurnal_rank` for FEATURES (division-free per-slot
    rank-Gauss) and the rank output is returned directly — the semantic transform
    and winsorize are skipped (rank-Gauss already normalizes, and sqrt/log of a
    signed rank value would break). The target (``is_target=True``) always uses
    divide regardless, since its raw reconstruction needs the multiplicative baseline.

    Returns ``(adjusted_series, baseline)``.

    ``winsor_by_slot`` makes :func:`rolling_winsorize` take its 5/95 bounds within each time-of-day
    slot. It defaults to ``WINSOR_BY_SLOT_DEFAULT``, which is **off** — see that constant for the
    measurement that rejected it.

    ``slot_band`` is forwarded to :func:`diurnal_adjust`. It defaults to
    ``DIURNAL_SLOT_BAND_FRAC`` for **exog** and to ``0`` (off) for the **target**: the target's
    ``baseline`` is the multiplicative factor every raw-space reconstruction and every historical
    QLIKE number is defined against, so changing it silently would make the repo's results
    incomparable. That is a scoping decision, not a claim that the target's baseline is immune —
    ``RV`` takes the same unsigned mean branch, and giving it the band is a candidate change that
    needs its own re-baselining. Pass an explicit value to override either way.

    ``signed_stabilizer`` (default on) routes a signed feature whose per-slot std divisor is
    degenerate to :func:`asinh_stabilize` instead of the divide -- see the inline comment for
    the gate and the evidence. Set False to restore the pre-fix chain exactly.
    """
    if diurnal_mode not in ("divide", "rank"):
        raise ValueError(f"diurnal_mode must be 'divide' or 'rank', got {diurnal_mode!r}")
    if col_name in SKIP_VARS:
        return df[col_name].copy(), pd.Series(1.0, index=df.index)
    if slot_band is None:
        slot_band = 0.0 if is_target else DIURNAL_SLOT_BAND_FRAC
    if winsor_by_slot is None:
        winsor_by_slot = WINSOR_BY_SLOT_DEFAULT

    series = df[col_name].copy()
    has_negatives = bool((series.dropna() < 0).any())

    baseline = pd.Series(1.0, index=df.index)
    stabilized = False
    if use_diurnal and not is_diurnal_excluded(col_name) and time_col in df.columns:
        if diurnal_mode == "rank" and not is_target:
            return diurnal_rank(series, df[time_col])  # final feature; skip semantic + winsor
        adjusted, baseline = diurnal_adjust(
            series, df[time_col], has_negatives, slot_band=slot_band, soft=soft_guards
        )
        # A signed feature divides by its per-slot rolling std. For a sparse / ffilled /
        # zero-inflated column that std is degenerate most of the time, so the floor pins the
        # divisor to one global constant -- the "adjustment" stops adjusting and, worse, stops
        # tracking the feature's scale. Measured on the divisor itself rather than assumed:
        # voldemand pins on ~74% of rows (and reaches |z| ~ 2000 downstream as its scale grows
        # ~11x), while every other signed feature except sumret3 -- which rule 3 already
        # stabilises with cbrt -- pins on <30%, and sumret on 0.1%. Above the threshold, swap
        # the degenerate divide for the asinh stabiliser; below it, behaviour is unchanged.
        if (
            signed_stabilizer
            and has_negatives
            and not has_name_stabilizer(col_name)  # only rule 5 (identity) is overridable
            and float(np.isclose(baseline, baseline.min(), rtol=1e-9).mean())
            >= DIURNAL_PINNED_MAX
        ):
            series = asinh_stabilize(series)
            baseline = pd.Series(1.0, index=df.index)
            stabilized = True
        else:
            series = adjusted

    if use_transform and not stabilized:
        series = apply_semantic_transform(
            series, col_name, has_negatives, allow_missing=allow_missing
        )
    elif stabilized and not allow_missing:
        series = series.fillna(0.0)  # matches rule 5's NaN policy for signed features

    ww = winsor_window if winsor_window is not None else 240
    series = rolling_winsorize(
        series,
        window=ww,
        allow_missing=allow_missing,
        is_target=is_target,
        slots=df[time_col] if (winsor_by_slot and time_col in df.columns) else None,
        soft=soft_guards,
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
