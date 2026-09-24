"""Break-even price round trip, strikes, sizing and the card."""

from __future__ import annotations

import math
import sys
from typing import Any
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.signal import (  # noqa: E402
    break_even_price,
    build_instruction,
    half_strike_pair,
    headline,
    implied_variance_at,
    nearest_otm_strikes,
    render_card,
)


def test_nearest_otm_strikes_on_both_grids() -> None:
    assert nearest_otm_strikes(6532.4, 5.0) == (6535.0, 6530.0)
    assert nearest_otm_strikes(6535.0, 5.0) == (6535.0, 6535.0)
    assert nearest_otm_strikes(653.24, 1.0) == (654.0, 653.0)


def test_half_strike_pair_matches_the_listed_days() -> None:
    # study 79's nearest-OTM pairs on the days XSP listed x2.5 / x7.5
    assert half_strike_pair(757.585, 758.0, 757.0) == (758.0, 757.5)  # 2026-07-10
    assert half_strike_pair(432.101, 433.0, 432.0) == (432.5, 432.0)  # 2023-10-06
    assert half_strike_pair(562.428, 563.0, 562.0) == (562.5, 562.0)  # 2024-09-13
    # 2023-05-05: spot 414.144, no half strike between the spot and 415 / 414
    assert all(math.isnan(k) for k in half_strike_pair(414.144, 415.0, 414.0))


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
    kw: dict[str, Any] = dict(
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
    kw: dict[str, Any] = dict(
        session=date(2026, 9, 23),
        spot=6532.4,
        rv_hat=(0.0025) ** 2,
        capital=70_000.0,
        input_mode="free_substitute",
    )
    i = build_instruction(flags={"month_end": False, "third_friday": True}, **kw)
    plain = render_card(i)
    # one limit order at the break-even, a fixed count, the strangle named for Robinhood
    assert (
        f"Buy {i.n_spx_at_pstar} SPX 6535 call + 6530 put (Robinhood: Long Strangle)"
        in plain
    )
    assert f"Limit price {i.p_star_spx:.2f}. Not filled by 15:31? Cancel" in plain
    assert "Hold to the 16:00 close" in plain and "Third Friday" in plain
    assert "XSP" not in plain and "Why" not in plain
    assert headline(i) == (
        f"Close trade: buy {i.n_spx_at_pstar} SPX 6535C/6530P at 15:30, "
        f"limit {i.p_star_spx:.2f}"
    )
    me = render_card(
        build_instruction(flags={"month_end": True, "third_friday": False}, **kw)
    )
    assert "MONTH-END, buy at 15:30 ET at any price" in me
    assert "Limit price: the ask + 5%." in me and "$10,500 / (limit price x 100)" in me
    late = render_card(
        build_instruction(flags={"month_end": False}, late=True, notes=("x",), **kw)
    )
    assert late.startswith("LATE:") and "Data: x" in late
    none = render_card(
        build_instruction(flags={"month_end": False}, **{**kw, "rv_hat": float("nan")})
    )
    assert "NO TRADE today (no forecast)" in none
    broke = build_instruction(flags={"month_end": False}, **{**kw, "capital": 100.0})
    assert "buys less than one pair" in render_card(broke)
    assert headline(broke) == "Close trade: NO TRADE today"


def test_small_budget_switches_the_card_and_title_to_xsp() -> None:
    kw: dict[str, Any] = dict(
        spot=7575.85,  # XSP 757.585 -> the 758C / 757.5P pair if listed
        rv_hat=(0.0025) ** 2,
        flags={"month_end": False},
        capital=5_000.0,  # 3.3 % = $165: less than one SPX pair
        input_mode="free_substitute",
    )
    fri = build_instruction(session=date(2026, 7, 10), **kw)
    thu = build_instruction(session=date(2026, 7, 9), **kw)
    assert fri.n_spx_at_pstar == 0
    assert (fri.kc_xsp_half, fri.kp_xsp_half) == (758.0, 757.5)
    # the half-strike pair is nearer the spot, so its break-even is dearer
    assert fri.p_star_xsp_half > fri.p_star_xsp
    card = render_card(fri)
    assert (
        "XSP 758 call + 757.5 put (Robinhood: Long Strangle), expiring today." in card
    )
    assert f"(put not listed? buy {fri.n_xsp_at_pstar} XSP 758 call + 757 put" in card
    assert headline(fri).startswith("Close trade: buy ")
    assert "XSP 758C/757.5P" in headline(fri)
    assert math.isnan(thu.p_star_xsp_half)
    assert "not listed?" not in render_card(thu) and "SPX" not in render_card(thu)
