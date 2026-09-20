"""Sizing from the stress table: the arithmetic, both sides, and the refusals."""

from __future__ import annotations

import sys
from math import isnan
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.sizing import (  # noqa: E402
    SPX_INDEX_MULTIPLIER,
    STRESS_JUMP,
    contracts_for,
    stress_loss_per_contract,
)

SPOT = 5000.0
KC = 5005.0
KP = 5000.0
PREMIUM = 20.0


def test_the_option_leg_is_the_worse_side_minus_the_premium() -> None:
    """A 5 % jump settles 250 points away; the short straddle pays the intrinsic."""
    got = stress_loss_per_contract(SPOT, KC, KP, PREMIUM, 0.0)
    # up: settle 5250, the call pays 245; down: settle 4750, the put pays 250.
    assert got["side"] == "-"
    assert got["option"] == pytest.approx((250.0 - PREMIUM) * SPX_INDEX_MULTIPLIER)
    assert got["hedge"] == 0.0
    assert got["total"] == got["option"]


def test_the_up_side_wins_when_the_call_is_the_nearer_strike() -> None:
    got = stress_loss_per_contract(SPOT, 4900.0, 4800.0, PREMIUM, 0.0)
    # up: settle 5250, the call pays 350; down: settle 4750, the put pays 50.
    assert got["side"] == "+"
    assert got["option"] == pytest.approx((350.0 - PREMIUM) * SPX_INDEX_MULTIPLIER)


def test_the_futures_leg_is_charged_the_adverse_side() -> None:
    """The hedge is linear and unbounded; the table charges its magnitude."""
    for delta in (0.6, -0.6):
        got = stress_loss_per_contract(SPOT, KC, KP, PREMIUM, delta)
        hedge = float(got["hedge"])
        assert hedge == pytest.approx(0.6 * STRESS_JUMP * SPOT * SPX_INDEX_MULTIPLIER)
        assert hedge == pytest.approx(15_000.0)  # proposal 45's ~$15k
        assert float(got["total"]) == pytest.approx(float(got["option"]) + hedge)


def test_the_jump_size_scales_both_legs() -> None:
    small = stress_loss_per_contract(SPOT, KC, KP, PREMIUM, 0.5, jump=0.01)
    assert small["hedge"] == pytest.approx(0.5 * 0.01 * SPOT * SPX_INDEX_MULTIPLIER)
    assert small["option"] == pytest.approx((50.0 - PREMIUM) * SPX_INDEX_MULTIPLIER)
    none = stress_loss_per_contract(SPOT, KC, KP, PREMIUM, 0.5, jump=0.0)
    assert none["hedge"] == 0.0
    # at no jump the straddle settles at its own intrinsic and keeps the premium
    assert none["option"] == pytest.approx(-PREMIUM * SPX_INDEX_MULTIPLIER)


def test_a_multiplier_other_than_one_hundred() -> None:
    got = stress_loss_per_contract(SPOT, KC, KP, PREMIUM, 0.5, index_multiplier=10.0)
    assert got["option"] == pytest.approx((250.0 - PREMIUM) * 10.0)
    assert got["hedge"] == pytest.approx(0.5 * STRESS_JUMP * SPOT * 10.0)


def test_a_non_finite_input_gives_no_number() -> None:
    got = stress_loss_per_contract(float("nan"), KC, KP, PREMIUM, 0.5)
    assert isnan(float(got["option"]))
    assert isnan(float(got["total"]))
    assert got["side"] == ""


def test_contracts_for_is_a_floor() -> None:
    assert contracts_for(100_000.0, 0.10, 1_000.0) == 10
    assert contracts_for(100_000.0, 0.10, 1_100.0) == 9  # 9.09 floors to 9
    assert contracts_for(100_000.0, 0.10, 20_000.0) == 0
    assert contracts_for(1_000_000.0, 0.02, 38_000.0) == 0  # 0.52 floors to 0


def test_contracts_for_refuses_rather_than_sizing_off_nothing() -> None:
    assert contracts_for(100_000.0, 0.10, 0.0) == 0
    assert contracts_for(100_000.0, 0.10, -5_000.0) == 0  # a profitable stress case
    assert contracts_for(100_000.0, 0.0, 1_000.0) == 0
    assert contracts_for(-100_000.0, 0.10, 1_000.0) == 0
    assert contracts_for(float("nan"), 0.10, 1_000.0) == 0
    assert contracts_for(100_000.0, 0.10, float("nan")) == 0
    assert contracts_for(float("inf"), 0.10, 1_000.0) == 0


def test_the_whole_rule_end_to_end() -> None:
    """Capital, an accepted fraction and the stress table give the position."""
    stress = stress_loss_per_contract(SPOT, KC, KP, PREMIUM, 0.30)
    # option (250 - 20) * 100 = 23,000; hedge 0.30 * 0.05 * 5000 * 100 = 7,500
    assert stress["total"] == pytest.approx(30_500.0)
    assert contracts_for(1_000_000.0, 0.05, float(stress["total"])) == 1
    assert contracts_for(1_000_000.0, 0.20, float(stress["total"])) == 6
    assert contracts_for(100_000.0, 0.05, float(stress["total"])) == 0
