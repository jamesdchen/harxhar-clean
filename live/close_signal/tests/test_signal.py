"""Break-even price round trip, strikes, sizing and the card."""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.signal import (  # noqa: E402
    break_even_price,
    build_instruction,
    implied_variance_at,
    nearest_otm_strikes,
    render_card,
)


def test_nearest_otm_strikes_on_both_grids() -> None:
    assert nearest_otm_strikes(6532.4, 5.0) == (6535.0, 6530.0)
    assert nearest_otm_strikes(6535.0, 5.0) == (6535.0, 6535.0)
    assert nearest_otm_strikes(653.24, 1.0) == (654.0, 653.0)


def test_break_even_price_round_trips_through_the_inverter() -> None:
    spot, kc, kp = 6532.4, 6535.0, 6530.0
    rv_hat = (0.0025) ** 2  # a 0.25 % last-bar move
    p = break_even_price(rv_hat, spot, kc, kp)
    assert p > 0
    assert implied_variance_at(p, spot, kc, kp) == pytest.approx(rv_hat, rel=1e-6)
    # XSP is the SPX price / 10 at strikes / 10 (Black-76 is homogeneous of degree 1)
    px = break_even_price(rv_hat, spot / 10, kc / 10, kp / 10)
    assert px == pytest.approx(p / 10, rel=1e-9)
    assert math.isnan(break_even_price(0.0, spot, kc, kp))


def test_instruction_sizes_by_premium_and_switches_on_month_end() -> None:
    kw = dict(
        session=date(2026, 9, 30),
        spot=6532.4,
        rv_hat=(0.0025) ** 2,
        capital=70_000.0,
        input_mode="free_substitute",
    )
    plain = build_instruction(flags={"month_end": False, "third_friday": False}, **kw)
    me = build_instruction(flags={"month_end": True, "third_friday": False}, **kw)
    assert plain.decision == "BUY_IF_ASK_LE_PSTAR" and me.decision == "BUY_MONTH_END"
    # N = floor(capital x f / (P* x 100)) on each grid
    assert plain.n_xsp_at_pstar == math.floor(70_000 * 0.033 / (plain.p_star_xsp * 100))
    assert plain.n_spx_at_pstar == math.floor(70_000 * 0.033 / (plain.p_star_spx * 100))
    assert me.n_xsp_at_pstar == math.floor(70_000 * 0.15 / (me.p_star_xsp * 100))
    assert me.n_xsp_at_pstar > plain.n_xsp_at_pstar
    # each grid has its own nearest-OTM strikes; the XSP break-even is near, not equal to, SPX/10
    assert plain.p_star_xsp == pytest.approx(plain.p_star_spx / 10, rel=0.35)
    assert (plain.kc_xsp, plain.kp_xsp) == (654.0, 653.0) and (
        plain.kc_spx,
        plain.kp_spx,
    ) == (6535.0, 6530.0)


def test_card_says_what_to_do_in_each_state() -> None:
    kw = dict(
        session=date(2026, 9, 23),
        spot=6532.4,
        rv_hat=(0.0025) ** 2,
        capital=70_000.0,
        input_mode="free_substitute",
    )
    plain = render_card(
        build_instruction(flags={"month_end": False, "third_friday": True}, **kw)
    )
    assert "BUY ONLY IF" in plain and "NO TRADE" in plain and "third Friday" in plain
    assert "hold to cash settlement" in plain
    me = render_card(
        build_instruction(flags={"month_end": True, "third_friday": False}, **kw)
    )
    assert "MONTH-END CLOSE: BUY" in me and "regardless of the forecast" in me
    late = render_card(
        build_instruction(flags={"month_end": False}, late=True, notes=("x",), **kw)
    )
    assert "[LATE" in late and "Notes: x" in late
    none = render_card(
        build_instruction(flags={"month_end": False}, **{**kw, "rv_hat": float("nan")})
    )
    assert "NO TRADE: the forecast" in none
