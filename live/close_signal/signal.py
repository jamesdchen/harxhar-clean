"""From the forecast to the card: break-even straddle price, sizing, instruction.

The research rule is buy iff ``rv_hat > iv_var``, with ``iv_var`` the Black-76
package variance of the nearest-OTM straddle for the last bar (study 73: the
deck's iv_var IS that re-inversion).  Since the package price is strictly
increasing in the total volatility, ``rv_hat > iv_var`` is the same statement
as ``ask < P*`` where ``P* = package_price(sqrt(rv_hat), S, Kc, Kp)``: the
straddle price the forecast says the last bar is worth.  The operator reads
the SPX quote on the screen and buys if the ask is at or below P*.

Black-76 is homogeneous of degree one in (F, K), so at the SAME relative
strikes the XSP price is the SPX price divided by ten.  The listed grids
differ, though: XSP's 1-point step is 10 SPX points, coarser than SPX's 5, so
each break-even is computed on its own listed nearest-OTM strikes and the two
differ slightly.  The operator compares each against its own quote.  On a
Friday XSP may also list half strikes (x2.5 / x7.5, study 79); when one sits
between the spot and the 1-point strike on its side, the card adds that
pair's P* for the operator to use if the strike is on the screen.

Venue (study 79, 2023-03-28 .. 2025-12-31 same-day books): the SPX line's
15:30 straddle spread is 4.1 % of mid against XSP's 18.2 %, and the XSP touch
held the month-end size on 1 of 42 month-ends (touch median 20 straddles vs
N median 145).  On the 32 month-ends with both books the long at the ask
returned +0.589 on SPX vs +0.494 on XSP.  The general leg fares no better on
XSP: its 3.3 % size fits the touch on 22-28 % of days since 2024, and the
unconditional long at the ask returns -0.110 there vs -0.051 on SPX.  Both
legs are therefore bought on the SPX line (N median 12 at the 15 % month-end
budget); XSP, with its own break-even, only when the budget is below one SPX
straddle.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date

from live.close_signal.common import (
    INDEX_MULTIPLIER,
    SPX_STRIKE_STEP,
    XSP_HALF_STRIKE_OFFSET,
    XSP_HALF_STRIKE_PERIOD,
    XSP_SCALE,
    XSP_STRIKE_STEP,
)
from live.ibkr.pricing import invert_total_vol, package_price
from live.ibkr.sizing import contracts_for_outlay

#: Study 76 (Kelly of the long leg): general long leg half-Kelly 0.033;
#: month-end long 0.10 to start, 0.22 (half-Kelly) once fills are measured.
DEFAULT_LONG_FRACTION = 0.033
#: 0.15 is the OPERATOR'S CHOICE (2026-09-23) after study 77's block bootstrap:
#: half-Kelly on the observed month-end mean is 0.22, full Kelly one standard
#: error below it; 0.15 keeps P(75 % drawdown) <= 0.07 in every version of the
#: mean at +80 %/yr on the observed one.  0.22 is the ceiling, earned when the
#: live ledger's month-end mean holds.
DEFAULT_MONTH_END_FRACTION = 0.15
#: The chase the runner uses for the month-end long (live/ibkr): up to 5 % of the ask.
CHASE_PCT = 0.05


def nearest_otm_strikes(spot: float, step: float) -> tuple[float, float]:
    """(Kc, Kp): the call strike at or above the spot, the put strike at or below it."""
    kc = math.ceil(spot / step - 1e-12) * step
    kp = math.floor(spot / step + 1e-12) * step
    return float(kc), float(kp)


def half_strike_pair(spot: float, kc: float, kp: float) -> tuple[float, float]:
    """The nearest-OTM XSP pair if the Friday half strikes (2.5 + 5n) are listed.

    Each side takes the half strike when it lies between the spot and that
    side's 1-point strike; (nan, nan) when neither does -- the pair is then
    the 1-point pair whatever is listed.
    """
    off, per = XSP_HALF_STRIKE_OFFSET, XSP_HALF_STRIKE_PERIOD
    hc = math.ceil((spot - off) / per - 1e-12) * per + off  # first half strike >= spot
    hp = math.floor((spot - off) / per + 1e-12) * per + off  # last half strike <= spot
    kc_h = hc if spot <= hc < kc else kc
    kp_h = hp if kp < hp <= spot else kp
    if (kc_h, kp_h) == (kc, kp):
        return float("nan"), float("nan")
    return float(kc_h), float(kp_h)


def break_even_price(rv_hat: float, spot: float, kc: float, kp: float) -> float:
    """The straddle price at which the forecast variance equals the implied variance."""
    if not (math.isfinite(rv_hat) and rv_hat > 0.0):
        return float("nan")
    return float(package_price(math.sqrt(rv_hat), spot, kc, kp))


def implied_variance_at(price: float, spot: float, kc: float, kp: float) -> float:
    """iv_var of a quoted package price (the round trip of break_even_price)."""
    tv = invert_total_vol(spot, kc, kp, price)
    return float(tv * tv) if math.isfinite(tv) else float("nan")


@dataclass(frozen=True)
class Instruction:
    session: date
    expiry: date
    spot: float
    rv_hat: float
    break_even_vol_pct: float  # 100 * sqrt(rv_hat) over the bar
    kc_spx: float
    kp_spx: float
    p_star_spx: float
    kc_xsp: float
    kp_xsp: float
    p_star_xsp: float
    kc_xsp_half: float  # the Friday half-strike pair (nan when it is the 1-point pair)
    kp_xsp_half: float
    p_star_xsp_half: float
    month_end: bool
    third_friday: bool
    capital: float
    long_fraction: float
    month_end_fraction: float
    n_xsp_at_pstar: int
    n_spx_at_pstar: int
    decision: str  # "BUY_IF_ASK_LE_PSTAR" | "BUY_MONTH_END" | "NO_TRADE_FLAG"
    input_mode: str
    late: bool
    notes: tuple[str, ...] = ()

    def as_record(self) -> dict[str, object]:
        d = asdict(self)
        d["notes"] = "; ".join(self.notes)
        return d


def build_instruction(
    *,
    session: date,
    spot: float,
    rv_hat: float,
    flags: dict[str, bool],
    capital: float,
    long_fraction: float = DEFAULT_LONG_FRACTION,
    month_end_fraction: float = DEFAULT_MONTH_END_FRACTION,
    input_mode: str,
    late: bool = False,
    notes: tuple[str, ...] = (),
) -> Instruction:
    kc, kp = nearest_otm_strikes(spot, SPX_STRIKE_STEP)
    p_spx = break_even_price(rv_hat, spot, kc, kp)
    kc_x, kp_x = nearest_otm_strikes(spot * XSP_SCALE, XSP_STRIKE_STEP)
    p_xsp = break_even_price(rv_hat, spot * XSP_SCALE, kc_x, kp_x)
    kc_h, kp_h = (
        half_strike_pair(spot * XSP_SCALE, kc_x, kp_x)
        if session.weekday() == 4
        else (float("nan"), float("nan"))
    )
    p_half = (
        break_even_price(rv_hat, spot * XSP_SCALE, kc_h, kp_h)
        if math.isfinite(kc_h)
        else float("nan")
    )
    frac = month_end_fraction if flags.get("month_end") else long_fraction
    n_xsp = contracts_for_outlay(capital, frac, p_xsp, INDEX_MULTIPLIER)
    n_spx = contracts_for_outlay(capital, frac, p_spx, INDEX_MULTIPLIER)
    if flags.get("month_end"):
        decision = "BUY_MONTH_END"
    elif math.isfinite(p_spx):
        decision = "BUY_IF_ASK_LE_PSTAR"
    else:
        decision = "NO_TRADE_FLAG"
    return Instruction(
        session=session,
        expiry=session,
        spot=float(spot),
        rv_hat=float(rv_hat),
        break_even_vol_pct=100.0 * math.sqrt(rv_hat) if rv_hat > 0 else float("nan"),
        kc_spx=kc,
        kp_spx=kp,
        p_star_spx=p_spx,
        kc_xsp=kc_x,
        kp_xsp=kp_x,
        p_star_xsp=p_xsp,
        kc_xsp_half=kc_h,
        kp_xsp_half=kp_h,
        p_star_xsp_half=p_half,
        month_end=bool(flags.get("month_end", False)),
        third_friday=bool(flags.get("third_friday", False)),
        capital=float(capital),
        long_fraction=float(long_fraction),
        month_end_fraction=float(month_end_fraction),
        n_xsp_at_pstar=int(n_xsp),
        n_spx_at_pstar=int(n_spx),
        decision=decision,
        input_mode=input_mode,
        late=bool(late),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class _Leg:
    root: str
    kc: float
    kp: float
    limit: float  # the break-even: the most the pair is worth today
    n: int  # pairs the budget buys at that price

    # put first throughout: Robinhood's strangle rows read "put / call"

    @property
    def short(self) -> str:
        return f"{self.root} {self.kp:g}P/{self.kc:g}C"

    @property
    def pair(self) -> str:
        return f"{self.root} {self.kp:g} put + {self.kc:g} call"

    @property
    def robinhood(self) -> str:
        # Robinhood's "Long Straddle" list pairs a call and a put at ONE strike;
        # this pair is two strikes when the spot sits between them: its
        # "Long Strangle" list, at the pair's width, date "(0d)", row put / call
        if self.kc != self.kp:
            return (
                f"Robinhood: Long Strangle, width {self.kc - self.kp:g}, date (0d), "
                f"row {self.kp:,g} / {self.kc:,g}."
            )
        return f"Robinhood: Long Straddle, date (0d), row {self.kc:,g}."


def _legs(i: Instruction) -> list[_Leg]:
    """The pairs to trade, in order of preference; empty if the budget buys none.

    SPX is the venue; XSP replaces it only when the budget buys no SPX pair
    at the break-even (study 79: XSP's spread is ~4x SPX's).  On a Friday the
    XSP half-strike pair comes first, with the 1-point pair as its fallback.
    """
    frac = i.month_end_fraction if i.month_end else i.long_fraction
    if i.month_end:
        # the month-end leg buys at any price: its pair never depends on P* (a
        # NaN or dear forecast must not turn it into NO TRADE); the card adds
        # the XSP pair for a budget that buys no SPX pair at the actual ask
        return [_Leg("SPX", i.kc_spx, i.kp_spx, i.p_star_spx, i.n_spx_at_pstar)]
    if i.n_spx_at_pstar > 0:
        return [_Leg("SPX", i.kc_spx, i.kp_spx, i.p_star_spx, i.n_spx_at_pstar)]
    out = []
    if math.isfinite(i.p_star_xsp_half):
        n = contracts_for_outlay(i.capital, frac, i.p_star_xsp_half, INDEX_MULTIPLIER)
        out.append(_Leg("XSP", i.kc_xsp_half, i.kp_xsp_half, i.p_star_xsp_half, n))
    out.append(_Leg("XSP", i.kc_xsp, i.kp_xsp, i.p_star_xsp, i.n_xsp_at_pstar))
    return [leg for leg in out if leg.n > 0]


def _missing(leg: _Leg, alt: _Leg) -> str:
    """'no 757.5 put?' / 'no 772.5 call?': the half strike the fallback avoids."""
    if leg.kp != alt.kp:
        return f"no {leg.kp:g} put?"
    return f"no {leg.kc:g} call?"


def headline(i: Instruction) -> str:
    """The Calendar event title: the whole instruction when it fits on a phone line."""
    legs = _legs(i)
    if not legs or i.decision == "NO_TRADE_FLAG":
        return "Close trade: NO TRADE today"
    leg = legs[0]
    if i.decision == "BUY_MONTH_END":
        return f"Close trade: MONTH-END buy {leg.short} at 15:30"
    return f"Close trade: buy {leg.n} {leg.short} at 15:30, limit {leg.limit:.2f}"


def render_card(i: Instruction) -> str:
    """The Calendar event body: instructions only, readable on a phone.

    The general leg is one limit order AT the break-even: it fills (at the
    ask or better) exactly when the research rule says buy, and does not fill
    otherwise -- so the operator compares nothing and the count is fixed.
    """
    frac = i.month_end_fraction if i.month_end else i.long_fraction
    budget = i.capital * frac
    day = i.session.strftime("%a %d %b %Y")
    legs = _legs(i)
    hold = "Hold to the 16:00 close (cash settled, no exit order)."
    lines: list[str] = []
    if i.late:
        lines += ["LATE: made after 15:30; the order below is for 15:30.", ""]
    if i.decision == "NO_TRADE_FLAG":
        lines += [f"{day}: NO TRADE today (no forecast)."]
    elif not legs:
        lines += [f"{day}: NO TRADE today (${budget:,.0f} buys less than one pair)."]
    elif i.decision == "BUY_MONTH_END":
        leg = legs[0]
        lines += [
            f"{day}: MONTH-END, buy at 15:30 ET at any price",
            "",
            f"Buy {leg.pair}, expiring today.",
            leg.robinhood,
            f"Limit price: the ask + {CHASE_PCT:.0%}.",
            f"Quantity: ${budget:,.0f} / (limit price x 100), rounded down.",
            hold,
        ]
        if len(legs) > 1:
            alt = legs[1]
            lines += [f"({_missing(leg, alt)} use {alt.pair}: {alt.robinhood})"]
        if leg.root == "SPX":
            # the month-end ask is not known in advance: on a small budget one
            # SPX pair can cost more than it (study 81: 19 of 40 month-ends at $5.5k)
            x = _Leg("XSP", i.kc_xsp, i.kp_xsp, i.p_star_xsp, i.n_xsp_at_pstar)
            lines += [f"Rounds to 0? Buy {x.pair} instead, same rule: {x.robinhood}"]
    else:
        leg = legs[0]
        lines += [
            f"{day}: at 15:30 ET",
            "",
            f"Buy {leg.n} {leg.pair}, expiring today.",
            leg.robinhood,
            f"Limit price {leg.limit:.2f}. Not filled at once? Cancel it right away: no trade today.",
            hold,
        ]
        if len(legs) > 1:
            alt = legs[1]
            lines += [
                f"({_missing(leg, alt)} buy {alt.n} {alt.pair}, limit {alt.limit:.2f}: "
                f"{alt.robinhood})"
            ]
    if i.notes:
        lines += ["", "Data: " + "; ".join(i.notes)]
    return "\n".join(lines)


#: Below this budget the prep also shows the XSP row: the card names XSP when
#: the budget buys no SPX pair at the limit, and the 15:30 SPX strangle's ask
#: is at or below 12.70 on 9 in 10 of the 866 deck days (p99 28.33).
XSP_HINT_BUDGET = 1270.0


def render_prep(
    *,
    session: date,
    spot: float | None,
    flags: dict[str, bool],
    capital: float,
    long_fraction: float = DEFAULT_LONG_FRACTION,
    month_end_fraction: float = DEFAULT_MONTH_END_FRACTION,
    card_at: str = "15:31",
) -> tuple[str, str]:
    """(title, body) of the 15:00 prep event: what is known half an hour ahead.

    Known at 15:00: the day's flags, the budget, and the index -- so the
    strangle row the card will most likely name (it is re-read at 15:30).
    Not known: the limit price and the count, which need the 15:30 forecast.
    ``spot=None`` (the index unavailable) leaves the row to the card.
    """
    month_end = bool(flags.get("month_end"))
    frac = month_end_fraction if month_end else long_fraction
    budget = capital * frac
    day = session.strftime("%a %d %b %Y")
    if spot is not None and math.isfinite(spot):
        kc, kp = nearest_otm_strikes(spot, SPX_STRIKE_STEP)
        kc_x, kp_x = nearest_otm_strikes(spot * XSP_SCALE, XSP_STRIKE_STEP)
        row: str | None = f"{kp:,g} / {kc:,g}"
        where = f"2. SPX is {spot:,.0f} now: the row will be near {row} (it moves with the index)."
    else:
        row = None
        where = "2. The index level is unavailable right now: the card names the row."
    near = f"row near {row}, " if row else ""
    if month_end:
        title = f"Prepare: MONTH-END close trade at 15:30 ({near}budget ${budget:,.0f})"
        lines = [
            f"{day}: MONTH-END close trade at 15:30 ET, bought at any price",
            "",
            "Get ready now:",
            "1. Robinhood: SPX options, Long Strangle, width 5, date (0d).",
            where,
            f"3. Budget ${budget:,.0f}: have the cash available.",
            "",
            f"At 15:30: tap the row, limit price = the ask + {CHASE_PCT:.0%}, "
            f"quantity = ${budget:,.0f} / (limit price x 100), rounded down.",
        ]
    else:
        title = f"Prepare: close trade card at about {card_at}" + (
            f" (row near {row})" if row else ""
        )
        lines = [
            f"{day}: the close trade card comes at about {card_at} ET",
            "",
            "Get ready now:",
            "1. Robinhood: SPX options, Long Strangle, width 5, date (0d).",
            where,
            f"3. Budget ${budget:,.0f}. The card gives the count and the limit price.",
            "",
            "When the card comes: tap its row, enter its count and limit price, submit.",
            "Not filled at once? Cancel it right away: no trade today.",
        ]
    if budget < XSP_HINT_BUDGET and row:
        lines += [
            "",
            f"(If the card names XSP instead: width 1, row near {kp_x:g} / {kc_x:g}.)",
        ]
    return title, "\n".join(lines)


__all__ = [
    "CHASE_PCT",
    "DEFAULT_LONG_FRACTION",
    "DEFAULT_MONTH_END_FRACTION",
    "Instruction",
    "break_even_price",
    "build_instruction",
    "half_strike_pair",
    "headline",
    "implied_variance_at",
    "nearest_otm_strikes",
    "render_card",
    "render_prep",
    "XSP_HINT_BUDGET",
]
