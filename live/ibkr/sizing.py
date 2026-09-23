"""Position size from the stress table, not from margin.

Proposal 31 asked whether the book could be sized by margin and the answer
was no: the margin-sized row is unbounded (positions to 25,591x) and it loses
anyway.  Proposal 45's stress grid gives the honest constraint instead -- what
ONE contract loses in the tail event this book is actually exposed to, a jump
in the last bar that settles before the hedge can be reset:

    contracts = floor( capital * acceptable_fraction / loss per contract )

The tail event is a ``jump`` move of the index into the 16:00 settlement.
The short straddle pays the settlement intrinsic and keeps the premium; the
futures leg, held at the delta of the moment, moves linearly and is unbounded
in the adverse direction (proposal 45 measured about $15k per contract on a
5 % last-bar jump either way).

The number this module returns is deliberately CONSERVATIVE, and the
convention is stated rather than buried: the option leg is charged its WORSE
side and the futures leg is charged the ADVERSE side of the same jump, even
though a single jump cannot be the worse side for both at once.  A sizing
rule that assumed the offset would size the book off a cancellation the tape
does not guarantee.

Everything is in dollars: one index point is ``index_multiplier`` dollars on
the option package, and one straddle-delta of index exposure is the same
``index_multiplier`` dollars per point (that is what the futures hedge
replaces -- see ``hedge.target_lots``).
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal
from math import floor, isfinite

__all__ = [
    "SPX_INDEX_MULTIPLIER",
    "STRESS_JUMP",
    "contracts_for",
    "contracts_for_outlay",
    "scaled_contracts",
    "stress_loss_per_contract",
]

#: Dollars per index point on one SPX option (``hedge.SPX_INDEX_MULTIPLIER``).
SPX_INDEX_MULTIPLIER = 100.0
#: Proposal 45's stress move: a 5 % index jump in the last bar.
STRESS_JUMP = 0.05

_NAN = float("nan")


def stress_loss_per_contract(
    S: float,
    kc: float,
    kp: float,
    premium_mid: float,
    delta_pkg: float,
    jump: float = STRESS_JUMP,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
) -> dict[str, float | str]:
    """Dollars one short straddle plus its hedge loses on a ``jump`` settlement.

    ``S`` is the index now, ``kc``/``kp`` the straddle's strikes,
    ``premium_mid`` the premium received per contract in index points and
    ``delta_pkg`` the package delta the futures leg is currently hedging.

    The option leg is priced at settlement ``S * (1 + jump)`` and at
    ``S * (1 - jump)`` and charged the worse of the two:
    ``max(S1 - kc, 0) + max(kp - S1, 0) - premium_mid`` index points.  The
    futures leg is charged ``abs(delta_pkg) * jump * S`` index points, the
    magnitude it loses on whichever side is adverse for it.

    Units.  Everything is in the OPTION'S OWN index points -- SPX points for
    SPXW, XSP points (one tenth) for XSP -- times ``index_multiplier``
    dollars a point, so the same call sizes either instrument.  The futures
    leg needs no separate scale: the hedge is laid in S&P 500 futures at
    ``delta * index_multiplier * index_scale`` dollars per S&P point, and a
    ``jump`` moves the S&P by ``jump * S / index_scale`` points, so its loss
    is ``delta * index_multiplier * jump * S`` dollars whatever the scale --
    the ``index_scale`` cancels.  One XSP straddle at 650 therefore stresses
    to one tenth of one SPX straddle at 6,500, to the dollar.

    Returns ``{"option", "hedge", "total", "side"}`` -- the three dollar
    figures and the sign of the jump that gives the worse option side
    (``"+"`` or ``"-"``).  A non-finite input gives NaN losses and an empty
    side; nothing is clipped to zero, so a package that cannot lose (the
    premium exceeds the stressed intrinsic) reports its negative loss and
    ``contracts_for`` refuses it rather than dividing by it.
    """
    spot = float(S)
    call_strike = float(kc)
    put_strike = float(kp)
    premium = float(premium_mid)
    delta = float(delta_pkg)
    move = float(jump)
    mult = float(index_multiplier)
    if not all(
        isfinite(v) for v in (spot, call_strike, put_strike, premium, delta, move, mult)
    ):
        return {"option": _NAN, "hedge": _NAN, "total": _NAN, "side": ""}

    def payoff(settle: float) -> float:
        return max(settle - call_strike, 0.0) + max(put_strike - settle, 0.0)

    up = payoff(spot * (1.0 + move)) - premium
    down = payoff(spot * (1.0 - move)) - premium
    side = "+" if up >= down else "-"
    option = max(up, down) * mult
    hedge = abs(delta) * abs(move) * spot * mult
    return {
        "option": float(option),
        "hedge": float(hedge),
        "total": float(option + hedge),
        "side": side,
    }


def contracts_for(capital: float, fraction: float, loss_per_contract: float) -> int:
    """Contracts whose stressed loss fits inside ``capital * fraction``.

    ``fraction`` is the share of capital the operator accepts losing in the
    stress event.  Returns ``floor(capital * fraction / loss_per_contract)``,
    never negative, and 0 when any input is non-finite or the loss per
    contract is not strictly positive -- a book whose stress case is a profit
    is not a licence to size without a bound.
    """
    cap = float(capital)
    frac = float(fraction)
    loss = float(loss_per_contract)
    if not (isfinite(cap) and isfinite(frac) and isfinite(loss)):
        return 0
    if loss <= 0.0 or cap <= 0.0 or frac <= 0.0:
        return 0
    return max(0, int(floor(cap * frac / loss)))


def contracts_for_outlay(
    capital: float,
    fraction: float,
    premium_points: float,
    index_multiplier: float = SPX_INDEX_MULTIPLIER,
) -> int:
    """Long straddles whose premium outlay fits inside ``capital * fraction``.

    The research book is kept in PREMIUM UNITS -- one premium dollar a day --
    so its Sharpe is that of a position whose size moves inversely with the
    premium (study 73: the same positions at constant notional are 0.66 / 0.20
    against 1.68 / 1.22).  A LONG straddle's worst case is its premium, so
    ``floor(capital * fraction / (premium_points * index_multiplier))`` is at
    once the premium-unit size and the count whose loss bound is the same
    budget the short program's stress table protects.  ``premium_points`` is
    the price the order will pay (the quoted ask) in the instrument's own
    points.  Returns 0 when any input is non-finite or not strictly positive;
    never negative.  It does not apply to a SHORT straddle, whose loss is not
    bounded by its premium -- that side keeps the stress table.
    """
    cap = float(capital)
    frac = float(fraction)
    prem = float(premium_points)
    mult = float(index_multiplier)
    if not all(isfinite(v) for v in (cap, frac, prem, mult)):
        return 0
    if cap <= 0.0 or frac <= 0.0 or prem <= 0.0 or mult <= 0.0:
        return 0
    return max(0, int(floor(cap * frac / (prem * mult))))


def scaled_contracts(n: int, multiplier: float) -> int:
    """``floor(n * multiplier)`` whole contracts, in the multiplier's decimals.

    The multiplier is a decimal an operator typed, so the product is taken in
    decimal arithmetic: in binary floating point 25 x 1.16 is 28.999...96 and
    would floor one contract short.  A non-finite or negative multiplier is
    refused, not rounded.
    """
    mult = float(multiplier)
    if not isfinite(mult) or mult < 0.0:
        raise ValueError("a size multiplier must be finite and >= 0, got " + repr(mult))
    product = Decimal(int(n)) * Decimal(repr(mult))
    return max(0, int(product.to_integral_value(rounding=ROUND_FLOOR)))
