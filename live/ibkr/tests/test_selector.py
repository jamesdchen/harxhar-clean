"""The causal entry-clock selector: E2, its warm-up, its ties and its refusal."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from math import isnan
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.ibkr.premium_ledger import ClockRecord, PremiumLedger  # noqa: E402
from live.ibkr.selector import (  # noqa: E402
    CLOCKS,
    DEFAULT_WINDOW,
    pick_entry_clock,
    score_entry_clocks,
)

DAY0 = date(2024, 1, 2)


def day(i: int) -> date:
    return DAY0 + timedelta(days=i)


def rec(i: int, clock: str, implied: float, realized: float) -> ClockRecord:
    return ClockRecord(
        session=day(i),
        clock=clock,
        implied_rem_var=implied,
        realized_rem_var=realized,
        spot=5000.0,
        premium_mid=20.0,
        kc=5005.0,
        kp=5000.0,
    )


def flat_ledger(n: int, min_sessions: int = 2) -> PremiumLedger:
    """Every clock at the same premium on every session: every E2 score is 1."""
    led = PremiumLedger(min_sessions=min_sessions)
    led.append_session(
        [rec(i, c, 2.0, 1.0) for i in range(n) for c in CLOCKS]  # type: ignore[misc]
    )
    return led


def test_clocks_are_the_entry_clocks_only() -> None:
    assert CLOCKS[0] == "10:00"
    assert CLOCKS[-1] == "15:00"
    assert "15:30" not in CLOCKS  # no session left to hedge
    assert len(CLOCKS) == 11
    assert DEFAULT_WINDOW == 252


def test_warm_up_has_no_pick() -> None:
    led = flat_ledger(12, min_sessions=5)
    clock, scores = pick_entry_clock(led, day(3), window=6, min_sessions=5)
    assert clock is None
    assert set(scores) == set(CLOCKS)
    assert all(isnan(v) for v in scores.values())
    clock, scores = pick_entry_clock(led, day(6), window=6, min_sessions=5)
    assert clock is not None
    assert all(not isnan(v) for v in scores.values())


def test_a_window_shorter_than_the_warm_up_is_refused() -> None:
    """The numerator could never clear the warm-up: flat for ever, in silence."""
    led = flat_ledger(12, min_sessions=5)
    with pytest.raises(ValueError, match="shorter than the warm-up"):
        pick_entry_clock(led, day(11), window=4, min_sessions=5)


def test_expanding_over_expanding_is_refused() -> None:
    """E2 with no rolling window is identically 1.0 at every clock: not a pick."""
    led = flat_ledger(10, min_sessions=2)
    with pytest.raises(ValueError, match="rolling window"):
        pick_entry_clock(led, day(9), window=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        pick_entry_clock(led, day(9), window=0)
    with pytest.raises(ValueError, match="positive"):
        score_entry_clocks(led, day(9), window=-3)


def test_a_flat_tape_scores_one_everywhere_and_ties_to_the_earliest() -> None:
    led = flat_ledger(12, min_sessions=2)
    clock, scores = pick_entry_clock(led, day(11), window=4, min_sessions=2)
    assert all(v == pytest.approx(1.0) for v in scores.values())
    assert clock == CLOCKS[0]  # the tie rule, matching numpy.nanargmax


def test_the_pick_follows_the_clock_whose_premium_rose_most() -> None:
    """E2 is relative to each clock's OWN level, so a level difference is not a pick."""
    led = PremiumLedger(min_sessions=2)
    rows: list[ClockRecord] = []
    for i in range(20):
        for c in CLOCKS:
            # 14:00 sits at a permanently HIGHER level than every other clock ...
            level = 4.0 if c == "14:00" else 2.0
            # ... but 13:30's premium DOUBLES in the second half of the tape.
            if c == "13:30" and i >= 10:
                level = 4.0
            rows.append(rec(i, c, level, 1.0))
    led.append_session(rows)
    clock, scores = pick_entry_clock(led, day(20), window=10, min_sessions=2)
    assert clock == "13:30"
    assert scores["13:30"] > scores["14:00"]
    assert scores["14:00"] == pytest.approx(1.0)  # a level, not a move


def test_candidate_clocks_restrict_the_pick() -> None:
    led = PremiumLedger(min_sessions=2)
    rows = []
    for i in range(20):
        for c in CLOCKS:
            level = 4.0 if (c == "13:30" and i >= 10) else 2.0
            rows.append(rec(i, c, level, 1.0))
    led.append_session(rows)
    afternoon = ("14:00", "14:30", "15:00")
    clock, scores = pick_entry_clock(
        led, day(20), window=10, min_sessions=2, candidate_clocks=afternoon
    )
    assert set(scores) == set(afternoon)
    assert clock == "14:00"  # every candidate is flat, so the tie rule decides


def test_a_clock_with_no_history_is_not_picked() -> None:
    led = PremiumLedger(min_sessions=2)
    rows = [rec(i, c, 2.0, 1.0) for i in range(20) for c in CLOCKS if c != "15:00"]
    rows.append(rec(19, "15:00", 99.0, 1.0))
    led.append_session(rows)
    clock, scores = pick_entry_clock(led, day(20), window=10, min_sessions=2)
    assert isnan(scores["15:00"])
    assert clock == CLOCKS[0]


def test_an_empty_ledger_has_no_pick() -> None:
    clock, scores = pick_entry_clock(PremiumLedger(), day(0))
    assert clock is None
    assert all(isnan(v) for v in scores.values())
