"""Futures hedge arithmetic for the short straddle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.hedge import (  # noqa: E402
    ES_MULTIPLIER,
    MES_MULTIPLIER,
    SPX_INDEX_MULTIPLIER,
    rebalance_lots,
    rebalance_qty,
    residual_delta,
    residual_delta_lots,
    target_futures,
    target_lots,
)


def test_es_target_is_two_delta_per_straddle() -> None:
    """ES is 50 a point against the option's 100, so the hedge is 2 * delta * n."""
    for delta, n, want in [
        (0.5, 1, 1),
        (0.5, 3, 3),
        (0.30, 5, 3),
        (0.10, 5, 1),
        (1.0, 4, 8),
        (0.0, 5, 0),
    ]:
        assert target_futures(delta, n) == want


def test_mes_target_is_twenty_delta_per_straddle() -> None:
    for delta, n, want in [(0.05, 1, 1), (0.30, 1, 6), (0.30, 5, 30)]:
        assert target_futures(delta, n, MES_MULTIPLIER) == want


def test_sign_convention_is_long_futures_for_a_positive_package_delta() -> None:
    """The short straddle holds +delta units of index, so a positive delta is long."""
    assert target_futures(0.40, 5) > 0
    assert target_futures(-0.40, 5) < 0
    assert target_futures(-0.40, 5) == -target_futures(0.40, 5)


def test_target_rounds_to_a_whole_contract() -> None:
    # 2 * delta * n = 0.52 and 0.40: one contract and none.
    assert target_futures(0.26, 1) == 1
    assert target_futures(0.20, 1) == 0
    assert target_futures(-0.26, 1) == -1
    assert target_futures(float("nan"), 3) == 0


def test_rebalance_qty() -> None:
    assert rebalance_qty(3, 0) == 3
    assert rebalance_qty(3, 5) == -2
    assert rebalance_qty(-2, -2) == 0


def test_residual_delta_is_in_one_straddle_units() -> None:
    # 5 straddles at delta 0.30 want exactly 3 ES; nothing is left unhedged.
    assert residual_delta(0.30, 5, 3) == pytest.approx(0.0, abs=1e-15)
    # With no futures on, the whole package delta is residual.
    assert residual_delta(0.30, 5, 0) == pytest.approx(0.30)
    # One contract short of the target leaves 50/(5*100) = 0.1 of a straddle.
    assert residual_delta(0.30, 5, 2) == pytest.approx(0.10)
    assert residual_delta(0.30, 5, 4) == pytest.approx(-0.10)


def test_multipliers_are_the_contract_specs() -> None:
    assert SPX_INDEX_MULTIPLIER == 100.0
    assert ES_MULTIPLIER == 50.0
    assert MES_MULTIPLIER == 5.0
    # A Micro hedges a tenth of an E-mini, so it needs ten times the contracts
    # (on a delta whose E-mini target is already whole).
    assert target_futures(0.50, 1, MES_MULTIPLIER) == 10 * target_futures(0.50, 1)
    assert target_futures(0.50, 3, MES_MULTIPLIER) == 10 * target_futures(0.50, 3)


# ------------------------------------------------- the ES + MES lot pair -----
def test_lots_reproduce_the_index_dollars_of_delta() -> None:
    """es * 50 + mes * 5 is delta * 100 * n, to within half an MES."""
    for delta in (0.0, 0.03, 0.17, 0.5, 0.83, -0.41, -0.99, 1.0):
        for n in (1, 2, 4, 10, 25):
            es, mes = target_lots(delta, n)
            want = delta * SPX_INDEX_MULTIPLIER * n
            got = es * ES_MULTIPLIER + mes * MES_MULTIPLIER
            assert abs(got - want) <= 0.5 * MES_MULTIPLIER + 1e-9


def test_at_one_straddle_the_bulk_is_all_micros() -> None:
    """|delta| < 0.5 is under one ES, so n = 1 is (0, round(20 * delta))."""
    for delta in (0.0, 0.05, 0.17, 0.26, 0.49, -0.17, -0.49):
        assert target_lots(delta, 1) == (0, round(20 * delta))
    # past half a delta the E-mini takes over, with the micros on the remainder
    assert target_lots(0.6, 1) == (1, 2)
    assert target_lots(-0.6, 1) == (-1, -2)


def test_at_ten_straddles_the_bulk_is_e_minis() -> None:
    """n = 10 puts 20 * delta into ES and the remainder into MES."""
    assert target_lots(0.30, 10) == (6, 0)
    assert target_lots(0.33, 10) == (6, 6)
    assert target_lots(-0.33, 10) == (-6, -6)
    es, mes = target_lots(0.317, 10)
    assert es == 6
    assert mes == 3  # 317 - 300 = 17 dollars a point, 3.4 micros


def test_the_es_leg_never_overshoots() -> None:
    """ES truncates toward zero, so the micros always trade with the delta."""
    for delta in (0.19, 0.31, 0.77, -0.19, -0.31, -0.77):
        for n in (1, 3, 10):
            es, mes = target_lots(delta, n)
            assert abs(es * ES_MULTIPLIER) <= abs(delta * SPX_INDEX_MULTIPLIER * n)
            if es and mes:
                assert (es > 0) == (mes > 0)


def test_lot_resolution_is_half_an_mes_per_straddle() -> None:
    """One ES is 0.5/n straddle-deltas, one MES 0.05/n; the residual is 0.025/n."""
    for n in (1, 4, 10):
        assert ES_MULTIPLIER / (SPX_INDEX_MULTIPLIER * n) == pytest.approx(0.5 / n)
        assert MES_MULTIPLIER / (SPX_INDEX_MULTIPLIER * n) == pytest.approx(0.05 / n)
        for delta in (0.0, 0.07, 0.23, 0.41, 0.66, -0.29, -0.88):
            es, mes = target_lots(delta, n)
            assert abs(residual_delta_lots(delta, n, es, mes)) <= 0.025 / n + 1e-12


def test_es_at_four_is_coarser_than_mes_at_one() -> None:
    """The README claim the audit corrected: ES at n = 4 is 2.5x coarser, not equal."""
    es_at_four = 0.5 / 4 / 2.0  # half a contract of residual
    mes_at_one = 0.05 / 1 / 2.0
    assert es_at_four == pytest.approx(2.5 * mes_at_one)
    # parity needs ES at ten straddles
    assert 0.5 / 10 / 2.0 == pytest.approx(mes_at_one)


def test_a_non_finite_delta_hedges_nothing() -> None:
    assert target_lots(float("nan"), 5) == (0, 0)
    # ... and the residual alarm then sees the whole package delta, not zero
    assert residual_delta_lots(0.4, 5, 0, 0) == pytest.approx(0.4)


def test_rebalance_lots_is_the_pair_of_differences() -> None:
    assert rebalance_lots((3, 2), (0, 0)) == (3, 2)
    assert rebalance_lots((3, 2), (5, -1)) == (-2, 3)
    assert rebalance_lots((-2, 0), (-2, 0)) == (0, 0)


def test_residual_delta_lots_signs() -> None:
    # 10 straddles at delta 0.30 want exactly 6 ES and no micros.
    assert residual_delta_lots(0.30, 10, 6, 0) == pytest.approx(0.0, abs=1e-15)
    # one ES short leaves 50 / (10 * 100) = 0.05 of a straddle unhedged
    assert residual_delta_lots(0.30, 10, 5, 0) == pytest.approx(0.05)
    # one ES long overshoots by the same
    assert residual_delta_lots(0.30, 10, 7, 0) == pytest.approx(-0.05)
    # a micro is a tenth of that
    assert residual_delta_lots(0.30, 10, 6, -1) == pytest.approx(0.005)
