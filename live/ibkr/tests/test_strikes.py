"""Strike selection: the nearest-OTM straddle and the research's outage guards."""

from __future__ import annotations

import sys
from math import isnan
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.strikes import (  # noqa: E402
    ATM_MAX_STRIKE_GAP,
    ATM_MIN_LIVE,
    Quote,
    effective_min_live,
    modal_strike_step,
    pick_nearest_otm,
    pick_nearest_otm_reason,
    pick_wings,
    wing_width_pts,
)

STRIKE_STEP = 5.0  # SPX strikes are 5 apart near the money


def chain(
    low: float = 4900.0,
    high: float = 5100.0,
    step: float = STRIKE_STEP,
    bid: float = 4.0,
    ask: float = 4.4,
) -> list[Quote]:
    """A dense two-sided chain, one call and one put at every listed strike."""
    out: list[Quote] = []
    k = low
    while k <= high:
        out.append(Quote(k, "C", bid, ask))
        out.append(Quote(k, "P", bid, ask))
        k += step
    return out


def test_quote_mid_and_sentinel() -> None:
    assert Quote(5000.0, "C", 4.0, 4.4).mid == pytest.approx(4.2)
    # bid == ask == 0 is the vendor's no-quote sentinel, not a zero price.
    assert isnan(Quote(5000.0, "C", 0.0, 0.0).mid)
    assert not Quote(5000.0, "C", 0.0, 0.0).live
    # One-sided rows keep their half-spread midpoint.
    one_sided = Quote(5000.0, "C", 0.0, 0.1)
    assert one_sided.mid == pytest.approx(0.05)
    assert one_sided.live
    assert isnan(Quote(5000.0, "C", float("nan"), 4.4).mid)


def test_nearest_otm() -> None:
    body = pick_nearest_otm(chain(), 5002.0)
    assert body is not None
    assert (body.Kc, body.Kp) == (5005.0, 5000.0)
    assert not body.same_strike
    assert body.entry_mid == pytest.approx(8.4)
    assert body.entry_bid == pytest.approx(8.0)
    assert body.entry_ask == pytest.approx(8.8)


def test_same_strike_when_spot_is_listed() -> None:
    """Spot exactly on a listed strike gives a true at-the-money straddle."""
    body = pick_nearest_otm(chain(), 5000.0)
    assert body is not None
    assert (body.Kc, body.Kp) == (5000.0, 5000.0)
    assert body.same_strike


def test_sentinel_legs_are_not_picked() -> None:
    """A no-quote nearest strike is skipped for the next live one."""
    quotes = [q for q in chain() if not (q.strike == 5005.0 and q.right == "C")]
    quotes.append(Quote(5005.0, "C", 0.0, 0.0))
    body = pick_nearest_otm(quotes, 5002.0)
    assert body is not None
    assert body.Kc == 5010.0
    assert body.Kp == 5000.0


def test_reject_strike_gap() -> None:
    """More than one missing strike near the money is an outage, not a market."""
    quotes = [q for q in chain() if not (5000.0 < q.strike < 5020.0 and q.right == "C")]
    body, reason = pick_nearest_otm_reason(quotes, 5002.0)
    assert body is None
    assert reason == "strike_gap"
    # One missing strike (a gap inside ATM_MAX_STRIKE_GAP) is tolerated.
    quotes = [q for q in chain() if not (q.strike == 5005.0 and q.right == "C")]
    ok, reason = pick_nearest_otm_reason(quotes, 5002.0)
    assert reason == ""
    assert ok is not None
    assert ok.Kc - 5002.0 <= ATM_MAX_STRIKE_GAP


def test_reject_few_live_on_a_full_chain() -> None:
    """On a chain big enough to carry the research's bar, the bar is the bar."""
    quotes = chain()  # 41 strikes, 82 contracts: the research's population
    assert pick_nearest_otm_reason(quotes, 5002.0)[1] == ""
    assert effective_min_live(len(quotes)) == ATM_MIN_LIVE
    near = [q for q in quotes if 4995.0 <= q.strike <= 5010.0]  # 8 live contracts
    dead = [
        Quote(q.strike, q.right, 0.0, 0.0)
        for q in quotes
        if not (4995.0 <= q.strike <= 5010.0)
    ]
    body, reason = pick_nearest_otm_reason(near + dead, 5002.0)
    assert body is None
    assert reason == "few_live"


def test_min_live_scales_to_the_subscribed_band() -> None:
    """A band is judged against its own size, never against a full chain's.

    The live engine subscribes a band around spot, so ATM_MIN_LIVE (calibrated
    on a whole 0DTE chain) would refuse a perfectly healthy narrow band.  The
    bar is never raised above the research's, only lowered to the band.
    """
    assert effective_min_live(82) == ATM_MIN_LIVE
    assert effective_min_live(20) == ATM_MIN_LIVE
    assert effective_min_live(10) == 5
    assert effective_min_live(6) == 3
    assert effective_min_live(2) == 2  # never below the two legs the book needs
    assert effective_min_live(82, min_live=0) == 0  # the guard, switched off
    band = chain(low=4990.0, high=5010.0)  # 5 strikes, 10 contracts
    assert len(band) == ATM_MIN_LIVE
    dead = Quote(band[-1].strike, band[-1].right, 0.0, 0.0)
    body, reason = pick_nearest_otm_reason(band[:-1] + [dead], 5002.0)
    assert reason == ""  # 9 of 10 live in a 10-contract band is a market
    assert body is not None


def test_strike_gap_bar_is_read_off_the_grid_presented() -> None:
    """The gap bar is 2 x the modal step, which is 10 on the research's grid."""
    assert modal_strike_step(chain()) == pytest.approx(STRIKE_STEP)
    assert 2.0 * modal_strike_step(chain()) == pytest.approx(ATM_MAX_STRIKE_GAP)
    assert modal_strike_step(chain(step=25.0)) == pytest.approx(25.0)
    assert isnan(modal_strike_step([Quote(5000.0, "C", 1.0, 1.2)]))
    assert isnan(modal_strike_step([]))
    # On a 25-point grid one missing strike is 50 points away and NOT an outage,
    # where the hard-coded 10 would have refused every stamp.
    wide = [q for q in chain(step=25.0) if not (q.strike == 5025.0 and q.right == "C")]
    body, reason = pick_nearest_otm_reason(wide, 5002.0)
    assert reason == ""
    assert body is not None
    assert body.Kc == 5050.0
    # Two missing strikes on that grid still is an outage.
    wider = [
        q
        for q in chain(step=25.0)
        if not (5000.0 < q.strike < 5075.0 and q.right == "C")
    ]
    assert pick_nearest_otm_reason(wider, 5002.0)[1] == "strike_gap"


def test_one_sided_package_is_flagged() -> None:
    """A leg with no bid still prices, but the package says so."""
    quotes = [q for q in chain() if not (q.strike == 5005.0 and q.right == "C")]
    quotes.append(Quote(5005.0, "C", 0.0, 0.1))
    body = pick_nearest_otm(quotes, 5002.0)
    assert body is not None
    assert body.Kc == 5005.0
    assert body.one_sided
    assert body.entry_bid == pytest.approx(4.0)  # the put's bid alone
    two_sided = pick_nearest_otm(chain(), 5002.0)
    assert two_sided is not None
    assert not two_sided.one_sided


def test_reject_no_call_and_no_put() -> None:
    calls_only = [q for q in chain() if q.right == "C"]
    body, reason = pick_nearest_otm_reason(calls_only, 5002.0)
    assert body is None
    assert reason == "no_put"
    puts_only = [q for q in chain() if q.right == "P"]
    body, reason = pick_nearest_otm_reason(puts_only, 5002.0)
    assert body is None
    assert reason == "no_call"
    # A spot above every listed strike leaves no call at or above it.
    body, reason = pick_nearest_otm_reason(chain(), 5200.0)
    assert body is None
    assert reason == "no_call"
    body, reason = pick_nearest_otm_reason(chain(), 4800.0)
    assert body is None
    assert reason == "no_put"


def test_reject_no_live_quote_and_no_spot() -> None:
    dead = [Quote(q.strike, q.right, 0.0, 0.0) for q in chain()]
    assert pick_nearest_otm_reason(dead, 5002.0)[1] == "no_live_quote"
    assert pick_nearest_otm_reason(chain(), float("nan"))[1] == "no_spot"


def test_live_quotes_on_one_side_report_that_side_not_no_live_quote() -> None:
    """Live calls all below spot: there ARE live quotes, so it is no_put."""
    calls_below = [q for q in chain(high=4990.0) if q.right == "C"]
    body, reason = pick_nearest_otm_reason(calls_below, 5002.0)
    assert body is None
    assert reason == "no_put"


def test_wing_width_points() -> None:
    assert wing_width_pts(5000.0, 0.005) == pytest.approx(25.0)
    assert wing_width_pts(4321.0, 0.01) == pytest.approx(43.21)


def test_wings_take_the_nearest_listed_strike_beyond_the_width() -> None:
    quotes = chain()
    body = pick_nearest_otm(quotes, 5002.0)
    assert body is not None
    wings = pick_wings(quotes, body, 25.0)
    assert wings == (5030.0, 4975.0)
    # An offset that lands exactly on a listed strike takes that strike.
    assert pick_wings(quotes, body, 20.0) == (5025.0, 4980.0)
    # Nothing listed far enough out is no condor, not a free wing.
    assert pick_wings(quotes, body, 500.0) is None


def test_wings_skip_no_quote_contracts() -> None:
    quotes = [q for q in chain() if not (q.strike == 5030.0 and q.right == "C")]
    quotes.append(Quote(5030.0, "C", 0.0, 0.0))
    body = pick_nearest_otm(quotes, 5002.0)
    assert body is not None
    assert pick_wings(quotes, body, 25.0) == (5035.0, 4975.0)
