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


def render_card(i: Instruction) -> str:
    """The Calendar event body.  Plain text, readable on a phone."""
    frac = i.month_end_fraction if i.month_end else i.long_fraction
    budget = i.capital * frac
    lines = [
        f"SPXW 0DTE close trade -- {i.session.isoformat()}"
        + ("  [LATE: the 15:30 inputs arrived after the stamp]" if i.late else ""),
        "",
    ]
    if i.decision == "BUY_MONTH_END":
        lines += [
            "MONTH-END CLOSE: BUY the 15:30 straddle regardless of the forecast (proposal 55).",
            f"  SPX  buy {i.n_spx_at_pstar} straddles  {i.kc_spx:g}C / {i.kp_spx:g}P  "
            f"limit at the ask, chase up to +{CHASE_PCT:.0%}; N = floor(${budget:,.0f} / (ask x 100))",
            f"  XSP  only if N on SPX is 0: {i.n_xsp_at_pstar} straddles  {i.kc_xsp:g}C / {i.kp_xsp:g}P "
            "(its touch is ~20 straddles and its spread ~4x SPX's)",
            f"  Budget {frac:.0%} of ${i.capital:,.0f} = ${budget:,.0f} of premium (the long's worst case).",
        ]
    elif i.decision == "BUY_IF_ASK_LE_PSTAR":
        lines += [
            f"BUY ONLY IF the SPX {i.kc_spx:g}C / {i.kp_spx:g}P straddle ASK <= {i.p_star_spx:.2f}",
            f"  then buy N = floor(${budget:,.0f} / (ask x 100)) straddles ({i.n_spx_at_pstar} at the break-even),",
            f"  limit at the ask, chase up to +{CHASE_PCT:.0%}, hold to cash settlement.",
            "  If the ask is above the break-even: NO TRADE (the short side is not traded).",
            f"  XSP only if N on SPX is 0: {i.kc_xsp:g}C / {i.kp_xsp:g}P, ask <= {i.p_star_xsp:.2f}, "
            f"N = {i.n_xsp_at_pstar}.",
        ]
        if math.isfinite(i.p_star_xsp_half):
            lines += [
                f"  (XSP, Friday: if {i.kc_xsp_half:g}C / {i.kp_xsp_half:g}P is listed, that is the nearest pair: "
                f"ask <= {i.p_star_xsp_half:.2f}.)",
            ]
    else:
        lines += ["NO TRADE: the forecast did not produce a break-even price today."]
    lines += [
        "",
        f"Behind it: spot {i.spot:,.2f}; forecast last-bar vol {i.break_even_vol_pct:.3f}% "
        f"(rv_hat {i.rv_hat:.3e}); break-even straddle SPX {i.p_star_spx:.2f} / XSP {i.p_star_xsp:.2f}.",
        f"Inputs: {i.input_mode}" + ("; third Friday" if i.third_friday else ""),
    ]
    if i.notes:
        lines += ["Notes: " + "; ".join(i.notes)]
    return "\n".join(lines)


__all__ = [
    "CHASE_PCT",
    "DEFAULT_LONG_FRACTION",
    "DEFAULT_MONTH_END_FRACTION",
    "Instruction",
    "break_even_price",
    "build_instruction",
    "half_strike_pair",
    "implied_variance_at",
    "nearest_otm_strikes",
    "render_card",
]
