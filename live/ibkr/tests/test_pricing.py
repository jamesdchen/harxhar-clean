"""Black-76 engine: the identities the live straddle pricing must satisfy."""

from __future__ import annotations

import sys
from datetime import datetime
from math import isfinite, isnan, sqrt
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.pricing import (  # noqa: E402
    corrected_total_vol,
    hourly_iv_from_total_vol,
    hours_to_close,
    invert_total_vol,
    package_delta,
    package_price,
    total_vol_from_hourly,
)

# A typical 0DTE stamp: spot near 5000, strikes 5 apart, half a percent of
# total volatility over the remaining session.
SPOT = 5000.0
K_CALL = 5005.0
K_PUT = 5000.0
TOTAL_VOL = 0.005


def test_put_call_symmetry() -> None:
    """C(F, K) = P(K, F), so the same-strike package is symmetric in F and K."""
    for s in (0.001, 0.005, 0.02):
        a = package_price(s, SPOT, K_CALL, K_CALL)
        b = package_price(s, K_CALL, SPOT, SPOT)
        assert a == pytest.approx(b, abs=1e-10)


def test_intrinsic_at_zero_vol() -> None:
    for f in (4980.0, 5000.0, 5002.5, 5030.0):
        want = max(f - K_CALL, 0.0) + max(K_PUT - f, 0.0)
        assert package_price(0.0, f, K_CALL, K_PUT) == pytest.approx(want, abs=1e-12)
        assert package_price(-1.0, f, K_CALL, K_PUT) == pytest.approx(want, abs=1e-12)


def test_price_is_increasing_in_vol_and_above_intrinsic() -> None:
    intrinsic = package_price(0.0, SPOT, K_CALL, K_PUT)
    prev = intrinsic
    for s in (0.0005, 0.001, 0.005, 0.02, 0.1):
        px = package_price(s, SPOT, K_CALL, K_PUT)
        assert px > prev
        prev = px


def test_delta_bounded_and_zero_without_vol() -> None:
    """Package delta is N(d1c) + N(d1p) - 1, so it lives in [-1, 1]."""
    for f in (4900.0, 4990.0, 5000.0, 5010.0, 5100.0):
        for s in (1e-6, 0.001, 0.005, 0.05, 0.5):
            d = package_delta(s, f, K_CALL, K_PUT)
            assert -1.0 <= d <= 1.0
    # The research's pkg_delta returns 0 at or below zero total volatility: no
    # intrinsic (sign) delta fallback.
    assert package_delta(0.0, SPOT, K_CALL, K_PUT) == 0.0
    assert package_delta(float("nan"), SPOT, K_CALL, K_PUT) == 0.0
    assert package_delta(TOTAL_VOL, float("nan"), K_CALL, K_PUT) == 0.0


def test_delta_sign_follows_moneyness() -> None:
    """Far above both strikes the package is a call, far below it is a put."""
    deep_up = package_delta(TOTAL_VOL, 5100.0, K_CALL, K_PUT)
    deep_down = package_delta(TOTAL_VOL, 4900.0, K_CALL, K_PUT)
    assert deep_up == pytest.approx(1.0, abs=1e-3)
    assert deep_down == pytest.approx(-1.0, abs=1e-3)
    assert package_delta(TOTAL_VOL, 5002.5, K_CALL, K_PUT) < deep_up
    assert package_delta(TOTAL_VOL, 5002.5, K_CALL, K_PUT) > deep_down


def test_inversion_round_trips() -> None:
    """price -> total vol -> price, to the research bisection's own bracket.

    ``bsm_invert_package_vol`` stops once the bracket on the total volatility
    is narrower than 1e-12, so the recovered volatility is exact to 1e-12 and
    the reconstructed price to that bracket times the package's vega -- a few
    nanopoints on a 5000-level index, which is the honest bound here.
    """
    for f in (4985.0, 5000.0, 5001.0, 5020.0):
        for s in (0.0008, 0.002, 0.005, 0.02):
            px = package_price(s, f, K_CALL, K_PUT)
            back = invert_total_vol(f, K_CALL, K_PUT, px)
            assert back == pytest.approx(s, abs=1e-12)
            assert package_price(back, f, K_CALL, K_PUT) == pytest.approx(px, abs=1e-8)


def test_inversion_refuses_below_intrinsic() -> None:
    intrinsic = max(5100.0 - K_CALL, 0.0) + max(K_PUT - 5100.0, 0.0)
    assert isnan(invert_total_vol(5100.0, K_CALL, K_PUT, intrinsic))
    assert isnan(invert_total_vol(5100.0, K_CALL, K_PUT, intrinsic - 1.0))
    assert isnan(invert_total_vol(SPOT, K_CALL, K_PUT, float("nan")))


def test_vol_unit_conversions_round_trip() -> None:
    for hours in (0.5, 2.5, 5.0):
        tot = total_vol_from_hourly(0.0021, hours)
        assert tot == pytest.approx(0.0021 * sqrt(hours), abs=1e-15)
        assert hourly_iv_from_total_vol(tot, hours) == pytest.approx(0.0021, abs=1e-15)


def test_hours_to_close_at_the_book_clocks() -> None:
    """11:00 entry is 5.0 hours from the close; the 15:30 stamp is 0.5."""
    day = (2024, 6, 12)
    assert hours_to_close(datetime(*day, 11, 0)) == pytest.approx(5.0)
    assert hours_to_close(datetime(*day, 15, 30)) == pytest.approx(0.5)
    assert hours_to_close(datetime(*day, 10, 0)) == pytest.approx(6.0)
    # Strictly positive and finite everywhere before the close.
    for hh, mm in ((9, 30), (11, 0), (15, 30), (15, 59)):
        left = hours_to_close(datetime(*day, hh, mm))
        assert isfinite(left)
        assert left > 0.0


def test_hours_to_close_refuses_at_and_past_the_close() -> None:
    """16:00 is the one regime this book always visits: it must refuse, loudly.

    The old return of 0.0 propagated into a ZeroDivisionError inside
    ``hourly_iv_from_total_vol`` and a math domain error past the close.
    """
    day = (2024, 6, 12)
    with pytest.raises(ValueError, match="16:00 ET close"):
        hours_to_close(datetime(*day, 16, 0))
    with pytest.raises(ValueError, match="16:00 ET close"):
        hours_to_close(datetime(*day, 16, 30))
    with pytest.raises(ValueError):
        hours_to_close(datetime(*day, 15, 30), close_hour=15)


def test_vol_conversions_refuse_a_spent_session() -> None:
    """NaN, never ZeroDivisionError or a math domain error, at h_rem <= 0."""
    assert isnan(hourly_iv_from_total_vol(0.005, 0.0))
    assert isnan(hourly_iv_from_total_vol(0.005, -0.5))
    assert isnan(hourly_iv_from_total_vol(0.005, float("nan")))
    assert isnan(total_vol_from_hourly(0.0021, 0.0))
    assert isnan(total_vol_from_hourly(0.0021, -0.5))
    assert isnan(total_vol_from_hourly(0.0021, float("nan")))


def test_corrected_total_vol_is_the_variance_factor() -> None:
    """V9 scales the VARIANCE, so the volatility scales by the square root."""
    assert corrected_total_vol(0.004, 0.81) == pytest.approx(0.004 * 0.9, abs=1e-15)
    assert corrected_total_vol(0.004, 1.0) == pytest.approx(0.004, abs=1e-15)
    assert corrected_total_vol(0.004, 4.0) == pytest.approx(0.008, abs=1e-15)


def test_corrected_total_vol_falls_back_on_a_missing_factor() -> None:
    """The research's fallback: no correction means the implied, unchanged."""
    assert corrected_total_vol(0.004, float("nan")) == 0.004
    assert corrected_total_vol(0.004, float("inf")) == 0.004
    assert corrected_total_vol(0.004, 0.0) == 0.004
    assert corrected_total_vol(0.004, -1.0) == 0.004
    # A non-finite implied volatility stays non-finite; nothing is invented.
    assert isnan(corrected_total_vol(float("nan"), 0.81))


def test_hours_to_close_is_tz_aware_safe() -> None:
    pytz = pytest.importorskip("zoneinfo")
    ny = pytz.ZoneInfo("America/New_York")
    stamp = datetime(2024, 6, 12, 11, 0, tzinfo=ny)
    assert hours_to_close(stamp) == pytest.approx(5.0)
    winter = datetime(2024, 1, 10, 15, 30, tzinfo=ny)
    assert hours_to_close(winter) == pytest.approx(0.5)
