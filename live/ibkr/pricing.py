"""Black-76 arithmetic for the 0DTE straddle, in the research's own convention.

The live path must price the package exactly as the backtest does, so every
formula here is a scalar restatement of research code:

* ``package_price`` / ``package_delta`` mirror ``pkg_price`` / ``pkg_delta`` of
  ``writeup/intraday_proposals/26_causal_entry_flatten.py`` (and
  ``atm_straddle_lib._bsm_package_price``).
* ``invert_total_vol`` mirrors ``atm_straddle_lib.bsm_invert_package_vol``.
* ``hourly_iv_from_total_vol`` mirrors the function of the same name in
  ``atm_straddle_lib``.

Conventions, unchanged from the research:

* Black-76 with ``r = 0`` and the forward taken as the **spot** ``F`` (the
  vendor's own convention; a parity forward is the dividend-correct
  alternative but does not reproduce the vendor's implied volatility field).
* ``total_vol`` is ``s = sigma * sqrt(T)``, the TOTAL volatility over the
  remaining session, i.e. the vendor's hourly standard deviation times
  ``sqrt(hours_remaining)``.
* The package is the nearest-OTM call at ``Kc`` plus the nearest-OTM put at
  ``Kp`` -- the straddle the book trades.  Its delta is
  ``N(d1c) + N(d1p) - 1``, and a straddle shorted at 11:00 and delta-hedged
  every 30 minutes till the close holds ``+delta`` units of index per
  straddle.
"""

from __future__ import annotations

from datetime import datetime
from math import erf, isfinite, log, sqrt

__all__ = [
    "MARKET_CLOSE_HOUR",
    "corrected_total_vol",
    "hourly_iv_from_total_vol",
    "hours_to_close",
    "invert_total_vol",
    "package_delta",
    "package_price",
    "total_vol_from_hourly",
]

# The regular session ends at 16:00 ET; the 0DTE package settles on that close.
MARKET_CLOSE_HOUR = 16

# bsm_invert_package_vol's bracket, halving count and tolerance, restated.  The
# package price is strictly increasing in s (positive vega on both legs), so
# plain bisection on [lo, hi] converges without a library root finder; 100
# halvings of a unit bracket reach 1e-30.
INVERT_VOL_LO = 1e-8
INVERT_VOL_HI = 1.0
INVERT_VOL_ITERS = 100
INVERT_VOL_TOL = 1e-12

_NAN = float("nan")


def _cdf(z: float) -> float:
    """Standard normal CDF, written as the research writes it."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def package_price(total_vol: float, F: float, Kc: float, Kp: float) -> float:
    """Black-76 price (r = 0) of the call at ``Kc`` plus the put at ``Kp``.

    ``total_vol`` is the total volatility over the remaining session.  At or
    below zero total volatility the package is worth its intrinsic value;
    a non-finite total volatility, or a non-positive underlying, is NaN --
    exactly the branches of ``pkg_price``.
    """
    s = float(total_vol)
    f = float(F)
    kc = float(Kc)
    kp = float(Kp)
    if not isfinite(f):
        return _NAN
    if isfinite(s) and s > 0.0 and f > 0.0:
        d1c = (log(f / kc) + 0.5 * s * s) / s
        d1p = (log(f / kp) + 0.5 * s * s) / s
        call = f * _cdf(d1c) - kc * _cdf(d1c - s)
        put = kp * _cdf(-(d1p - s)) - f * _cdf(-d1p)
        return call + put
    if s <= 0.0:
        return max(f - kc, 0.0) + max(kp - f, 0.0)
    return _NAN


def package_delta(total_vol: float, F: float, Kc: float, Kp: float) -> float:
    """Package delta ``N(d1c) + N(d1p) - 1``, in index units per straddle.

    Mirrors ``pkg_delta``: the delta is zero unless the total volatility is
    finite and strictly positive and the underlying is positive.  The
    research does NOT fall back to an intrinsic (sign) delta at zero
    volatility, and neither does the live engine -- in the HOLD book the
    last hedge of the day is the one set at 15:30 (the exit book's last is
    15:00, because 15:30 is the stamp it flattens on), and at the settlement
    itself there is nothing left to hedge.
    """
    s = float(total_vol)
    f = float(F)
    if not (isfinite(s) and s > 0.0 and f > 0.0):
        return 0.0
    kc = float(Kc)
    kp = float(Kp)
    d1c = (log(f / kc) + 0.5 * s * s) / s
    d1p = (log(f / kp) + 0.5 * s * s) / s
    return _cdf(d1c) + _cdf(d1p) - 1.0


def invert_total_vol(F: float, Kc: float, Kp: float, package_mid: float) -> float:
    """Total volatility that reproduces the straddle's package midpoint.

    Bisection on the Black-76 package price, mirroring
    ``atm_straddle_lib.bsm_invert_package_vol``.  Returns NaN when the mid is
    at or below forward-intrinsic or when the bracket fails.  Divide by
    ``sqrt(hours_remaining)`` for the vendor's hourly convention.
    """
    f = float(F)
    kc = float(Kc)
    kp = float(Kp)
    mid = float(package_mid)
    if not (isfinite(f) and f > 0.0 and isfinite(mid)):
        return _NAN
    intrinsic = max(f - kc, 0.0) + max(kp - f, 0.0)
    if mid <= intrinsic:
        return _NAN
    lo, hi = INVERT_VOL_LO, INVERT_VOL_HI
    if package_price(hi, f, kc, kp) < mid:
        return _NAN
    for _ in range(INVERT_VOL_ITERS):
        m = 0.5 * (lo + hi)
        if package_price(m, f, kc, kp) < mid:
            lo = m
        else:
            hi = m
        if hi - lo < INVERT_VOL_TOL:
            break
    if lo == INVERT_VOL_LO:
        # The lower bracket never moved, so the solution is at or below it and
        # the returned number would be the bracket rather than the market's.
        # Refuse it: a floor that reports itself as a volatility is exactly the
        # ad-hoc clip this repo does not allow, and the mid that produces it is
        # indistinguishable from one sitting on forward-intrinsic.
        return _NAN
    return float(0.5 * (lo + hi))


def total_vol_from_hourly(iv_hourly: float, hours_remaining: float) -> float:
    """Total volatility over the remaining session from the hourly standard deviation.

    NaN once the session has no remaining window: at ``hours_remaining <= 0``
    there is nothing left to be volatile over, and a negative one has no
    square root.  The book visits that regime every day at 16:00, so the
    conversion refuses rather than raising out of the hedge loop.
    """
    h = float(hours_remaining)
    if not (isfinite(h) and h > 0.0):
        return _NAN
    return float(iv_hourly) * sqrt(h)


def hourly_iv_from_total_vol(total_vol: float, hours_remaining: float) -> float:
    """Vendor-convention hourly standard deviation from a total volatility.

    NaN at ``hours_remaining <= 0`` -- the inverse of the same regime: at
    16:00 the division is by zero and past it by the root of a negative.
    """
    h = float(hours_remaining)
    if not (isfinite(h) and h > 0.0):
        return _NAN
    return float(total_vol) / sqrt(h)


def corrected_total_vol(
    total_vol_implied: float, realized_over_implied: float
) -> float:
    """The implied total volatility under proposal 36's V9 bias correction.

    The correction is a factor on the VARIANCE -- the trailing lagged mean of
    realized over implied remaining-window variance at this clock, which
    ``PremiumLedger.realized_over_implied_mean`` supplies -- so the corrected
    total volatility is ``sqrt(total_vol ** 2 * factor)``.  Proposal 36 found
    the implied slice under its own causal bias correction the only object
    that beats the implied slice on the remaining-window horizon, and the
    only hedge variant whose era interval on the primary book excludes zero
    (+0.176, [+0.063, +0.301]).

    The research's fallback rule, unchanged: where the factor is missing or
    non-positive -- the whole warm-up, and any clock the ledger has not seen
    enough of -- the hedge uses the implied volatility unchanged, so the
    corrected book hedges the same days as the book and differs from it only
    where the correction has something to say.  A non-finite implied
    volatility stays non-finite; nothing here invents a number.
    """
    tv = float(total_vol_implied)
    factor = float(realized_over_implied)
    if not isfinite(tv):
        return _NAN
    if not (isfinite(factor) and factor > 0.0):
        return tv
    return float(sqrt(tv * tv * factor))


def hours_to_close(now_et: datetime, close_hour: int = MARKET_CLOSE_HOUR) -> float:
    """Hours from ``now_et`` to ``close_hour``:00 ET on the same session.

    ``now_et`` is a wall-clock Eastern time (tz-aware ``America/New_York`` in
    the live path; a naive Eastern stamp works too, since no daylight-saving
    transition falls inside a session).  At the book's 11:00 entry this is
    5.0 hours and at 15:30 it is 0.5 -- the ``h_rem`` the research carries on
    every bar.

    Strictly positive and finite before the close.  AT or AFTER the close it
    raises ``ValueError``: there is no remaining session to quote a
    volatility over, and the two conversions above would otherwise divide by
    zero (exactly 16:00) or take the root of a negative (past it).  The book
    reaches 16:00 every session -- on a late exit, under ``--hold``, or on
    clock skew -- so the refusal is the normal terminal case and says so.
    """
    wall = (
        now_et.hour
        + now_et.minute / 60.0
        + now_et.second / 3600.0
        + now_et.microsecond / 3_600_000_000.0
    )
    left = float(close_hour) - wall
    if left <= 0.0:
        raise ValueError(
            f"{now_et} is at or past the {close_hour}:00 ET close: the remaining "
            f"session is {left:.6f} hours, so there is no volatility horizon left "
            f"to price or hedge over"
        )
    return left
