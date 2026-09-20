"""Strike selection for the 0DTE straddle, with the research's outage guards.

A scalar restatement of ``atm_straddle_lib.quote_mid``,
``atm_straddle_lib.pick_nearest_otm_guarded`` and
``atm_straddle_lib.pick_wings``.  The live engine sees an option chain as a
list of :class:`Quote`; the backtest sees it as a DataFrame.  The selection
rule, the no-quote sentinel and the three refusal reasons are the same.

* **No-quote sentinel.** ``bid == ask == 0`` is the vendor's "no quote", not
  a zero price: the midpoint is NaN and the contract is not live.  One-sided
  rows (``bid == 0``, ``ask > 0``) keep their half-spread midpoint.
* **Live.** A contract is live when its midpoint is finite and strictly
  positive -- the filter the intraday builder applies before picking.
* **Nearest OTM.** The call is the smallest strike at or above spot, the put
  the largest at or below spot, among live contracts.
* **Outage guards.** Fewer than ``ATM_MIN_LIVE`` live contracts at the stamp,
  or a nearest leg more than ``ATM_MAX_STRIKE_GAP`` points from spot, is a
  vendor/venue outage rather than a market, and the stamp is refused.

Both guard constants were calibrated on the RESEARCH's population -- a whole
0DTE chain at a 5-point strike grid -- and the live engine does not see that
population: it subscribes a band around spot, so a healthy market can present
far fewer contracts than the backtest's stamp did.  Applying the absolute
numbers to a band would refuse good markets, so both are derived from what is
actually in front of the picker and fall back to the research constants only
where they cannot be:

* the gap bar is ``2 x`` the MODAL step between the distinct live strikes
  presented, which is ``2 x 5 = 10`` on the research's grid -- the same
  number, now inferred rather than hard-coded, and right on a 25-point grid
  too;
* the live-contract bar is never more than ``ATM_MIN_LIVE_FRACTION`` of the
  contracts presented, so a narrow band is judged against its own size.  It
  never demands MORE than the research's ``min_live``; it only declines to
  demand a full chain's worth of quotes from a band.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

__all__ = [
    "ATM_MAX_STRIKE_GAP",
    "ATM_MIN_LIVE",
    "ATM_MIN_LIVE_FRACTION",
    "Body",
    "Quote",
    "effective_min_live",
    "modal_strike_step",
    "pick_nearest_otm",
    "pick_nearest_otm_reason",
    "pick_wings",
    "wing_width_pts",
]

# The research's fallback bar, used when the presented strikes are too few to
# infer a grid: SPX strikes are 5 apart near the money, so a nearest-OTM leg
# further than 2 steps away means more than one strike is missing (an outage).
ATM_MAX_STRIKE_GAP = 10.0
# Live contracts per stamp below which the stamp is an outage on a FULL chain
# (normal: hundreds).  The live band is judged against its own size instead.
ATM_MIN_LIVE = 10
# The share of the contracts actually presented that the live bar may demand.
ATM_MIN_LIVE_FRACTION = 0.5
# The gap bar in units of the inferred strike step: one missing strike is a
# market, two is an outage.
STRIKE_GAP_STEPS = 2.0
# The guard's tolerance, as in pick_nearest_otm_guarded.
STRIKE_GAP_EPS = 1e-9

_NAN = float("nan")

#: Refusal reasons, in the order ``pick_nearest_otm_guarded`` assigns them.
NO_LIVE_QUOTE = "no_live_quote"
NO_SPOT = "no_spot"
NO_PUT = "no_put"
NO_CALL = "no_call"
STRIKE_GAP = "strike_gap"
FEW_LIVE = "few_live"
OK = ""


@dataclass(frozen=True)
class Quote:
    """One option contract's top of book.

    ``right`` is ``"C"`` or ``"P"`` (the research's ``cp``).
    """

    strike: float
    right: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        """``(bid + ask) / 2``, NaN on the no-quote sentinel or a broken side."""
        b = float(self.bid)
        a = float(self.ask)
        if not (isfinite(b) and isfinite(a)):
            return _NAN
        if b == 0.0 and a == 0.0:
            return _NAN
        return 0.5 * (b + a)

    @property
    def live(self) -> bool:
        """A quote the picker may use: a finite, strictly positive midpoint."""
        m = self.mid
        return isfinite(m) and m > 0.0


@dataclass(frozen=True)
class Body:
    """The straddle actually traded: nearest-OTM call and put at one stamp.

    ``one_sided`` marks a package one of whose legs has no bid.  The
    aggregates still sum the raw quotes -- the conservative direction, and
    what the research's crossed entry does -- but a price computed off
    ``entry_bid`` on such a package is NOT a price anyone gets, and the flag
    is what lets a caller say so instead of quietly reporting the number.
    """

    Kc: float
    Kp: float
    S: float
    entry_mid: float
    entry_bid: float
    entry_ask: float
    same_strike: bool
    one_sided: bool = False


def modal_strike_step(quotes: list[Quote]) -> float:
    """The most common gap between adjacent distinct live strikes, or NaN.

    NaN when fewer than two distinct live strikes are presented -- there is
    no grid to read, and the caller falls back to the research constant.
    Ties between two equally common steps go to the SMALLER step, the
    conservative side of the gap guard.
    """
    strikes = sorted({float(q.strike) for q in quotes if q.live})
    if len(strikes) < 2:
        return _NAN
    steps: dict[float, int] = {}
    for lo, hi in zip(strikes, strikes[1:], strict=False):
        gap = hi - lo
        if gap > 0.0:
            steps[gap] = steps.get(gap, 0) + 1
    if not steps:
        return _NAN
    best = max(steps.values())
    return float(min(g for g, c in steps.items() if c == best))


def effective_min_live(n_presented: int, min_live: int = ATM_MIN_LIVE) -> int:
    """The live-contract bar to apply to ``n_presented`` subscribed contracts.

    Never more than ``min_live`` and never more than ``ATM_MIN_LIVE_FRACTION``
    of what was presented, so a band is judged against its own size rather
    than against a full chain's.  At least 2 -- the two legs the straddle
    needs -- unless the caller has switched the guard off with ``min_live=0``.
    """
    bar = int(min_live)
    if bar <= 0:
        return 0
    scaled = int(ceil(ATM_MIN_LIVE_FRACTION * max(int(n_presented), 0)))
    return max(2, min(bar, scaled))


def pick_nearest_otm_reason(
    quotes: list[Quote],
    spot: float,
    *,
    max_gap: float | None = None,
    min_live: int = ATM_MIN_LIVE,
) -> tuple[Body | None, str]:
    """The nearest-OTM straddle at this stamp and the refusal reason.

    Returns ``(body, "")`` when the stamp is tradeable, otherwise
    ``(None, reason)`` with one of ``no_live_quote``, ``no_spot``,
    ``no_call``, ``no_put``, ``strike_gap``, ``few_live``.

    ``max_gap`` defaults to ``STRIKE_GAP_STEPS`` times the modal step of the
    strikes presented (``ATM_MAX_STRIKE_GAP`` when no grid can be read);
    ``min_live`` is scaled down to the size of what was presented by
    :func:`effective_min_live`.  Reasons are assigned in
    ``pick_nearest_otm_guarded``'s order, and a stamp that HAS live quotes
    but none on one side reports that side (``no_put`` / ``no_call``), never
    ``no_live_quote``.
    """
    live = [q for q in quotes if q.live]
    if not live:
        return None, NO_LIVE_QUOTE
    s = float(spot)
    if not isfinite(s):
        return None, NO_SPOT

    calls = [q for q in live if q.right == "C" and float(q.strike) >= s]
    puts = [q for q in live if q.right == "P" and float(q.strike) <= s]
    if not puts:
        return None, NO_PUT
    if not calls:
        return None, NO_CALL

    call = min(calls, key=lambda q: (float(q.strike) - s, float(q.strike)))
    put = min(puts, key=lambda q: (s - float(q.strike), float(q.strike)))
    kc = float(call.strike)
    kp = float(put.strike)

    if max_gap is None:
        step = modal_strike_step(quotes)
        bar = STRIKE_GAP_STEPS * step if isfinite(step) else ATM_MAX_STRIKE_GAP
    else:
        bar = float(max_gap)
    gap = max(kc - s, s - kp)
    if gap > bar + STRIKE_GAP_EPS:
        return None, STRIKE_GAP
    if len(live) < effective_min_live(len(quotes), min_live):
        return None, FEW_LIVE

    body = Body(
        Kc=kc,
        Kp=kp,
        S=s,
        entry_mid=call.mid + put.mid,
        entry_bid=float(call.bid) + float(put.bid),
        entry_ask=float(call.ask) + float(put.ask),
        same_strike=kc == kp,
        one_sided=float(call.bid) <= 0.0 or float(put.bid) <= 0.0,
    )
    return body, OK


def pick_nearest_otm(
    quotes: list[Quote],
    spot: float,
    *,
    max_gap: float | None = None,
    min_live: int = ATM_MIN_LIVE,
) -> Body | None:
    """The nearest-OTM straddle, or None when a guard refuses the stamp."""
    body, _ = pick_nearest_otm_reason(quotes, spot, max_gap=max_gap, min_live=min_live)
    return body


def wing_width_pts(spot: float, pct: float) -> float:
    """Wing offset in index points: ``pct`` of spot, unrounded.

    ``pct`` is a fraction, so ``0.005`` is half a percent of spot.
    """
    return float(spot) * float(pct)


def pick_wings(
    quotes: list[Quote], body: Body, width_pts: float
) -> tuple[float, float] | None:
    """Nearest listed live wings at least ``width_pts`` beyond the body strikes.

    Returns ``(Kc_wing, Kp_wing)``, or None when either side has no live
    contract far enough out -- a no-quote wing is not a free wing.
    """
    w = float(width_pts)
    lo_c = float(body.Kc) + w
    hi_p = float(body.Kp) - w
    calls = [q for q in quotes if q.live and q.right == "C" and float(q.strike) >= lo_c]
    puts = [q for q in quotes if q.live and q.right == "P" and float(q.strike) <= hi_p]
    if not calls or not puts:
        return None
    kc = min(float(q.strike) for q in calls)
    kp = max(float(q.strike) for q in puts)
    return kc, kp
