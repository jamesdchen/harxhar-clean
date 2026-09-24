"""The Actions guard: the firing window, one card per session, sessions, early closes."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebooks"))

import pandas as pd  # noqa: E402

from live.close_signal.schedule import (  # noqa: E402
    already_journaled,
    calendar_flags,
    is_early_close,
    seconds_until,
    should_run,
    wait_plan,
)


def test_the_three_off_hour_firings_map_into_the_window() -> None:
    # 2026-09-23 (EDT): 16:17 / 17:17 / 18:17 UTC = 12:17 / 13:17 / 14:17 ET, all accepted
    for h in (16, 17, 18):
        assert should_run(datetime(2026, 9, 23, h, 17, tzinfo=timezone.utc))[0] is True
    # 2026-12-02 (EST): 11:17 ET is before the window, 12:17 / 13:17 ET are in it
    assert should_run(datetime(2026, 12, 2, 16, 17, tzinfo=timezone.utc))[0] is False
    assert should_run(datetime(2026, 12, 2, 17, 17, tzinfo=timezone.utc))[0] is True
    assert should_run(datetime(2026, 12, 2, 18, 17, tzinfo=timezone.utc))[0] is True


def test_a_delayed_firing_is_accepted_until_1525_and_late_is_not() -> None:
    assert (
        should_run(datetime(2026, 9, 23, 19, 20, tzinfo=timezone.utc))[0] is True
    )  # 15:20 ET
    ok, why = should_run(datetime(2026, 9, 23, 19, 40, tzinfo=timezone.utc))  # 15:40 ET
    assert ok is False and "outside" in why


def test_a_session_with_a_journal_row_is_not_run_again() -> None:
    j = pd.DataFrame({"session": [pd.Timestamp("2026-09-23")], "status": ["no_signal"]})
    assert already_journaled(j, date(2026, 9, 23)) is True
    assert already_journaled(j, date(2026, 9, 24)) is False
    assert already_journaled(pd.DataFrame(), date(2026, 9, 24)) is False


def test_non_sessions_and_early_closes_are_skipped() -> None:
    assert (
        should_run(datetime(2026, 9, 26, 19, 2, tzinfo=timezone.utc))[0] is False
    )  # Saturday
    ok, why = should_run(
        datetime(2026, 11, 27, 20, 2, tzinfo=timezone.utc)
    )  # day after Thanksgiving
    assert ok is False and "13:00" in why


def test_early_close_rule_matches_the_notebook_library_2020_2025() -> None:
    import atm_straddle_lib as asl  # type: ignore

    lib = {d for d in asl.EARLY_CLOSE_DATES if d >= "2020-01-01"}
    ours = set()
    d = date(2020, 1, 1)
    while d <= date(2025, 12, 31):
        if is_early_close(d):
            ours.add(d.isoformat())
        d = date.fromordinal(d.toordinal() + 1)
    assert ours == lib, (sorted(ours - lib), sorted(lib - ours))


def test_flags_and_sleep() -> None:
    f = calendar_flags(date(2026, 9, 30))
    assert f["month_end"] is True and f["session"] is True
    assert calendar_flags(date(2026, 9, 18))["third_friday"] is True
    from live.close_signal.schedule import ET_TZ

    now = datetime(2026, 9, 23, 15, 0, 0, tzinfo=ET_TZ)
    assert seconds_until("15:30:30", now) == 1830.0
    assert seconds_until("14:00:00", now) == 0.0


def test_wait_plan_replays_past_sessions_and_refuses_an_overlong_sleep() -> None:
    from live.close_signal.schedule import ET_TZ

    at_0225 = datetime(2026, 9, 24, 2, 25, tzinfo=ET_TZ)  # run 35964417416
    plan, s = wait_plan(date(2026, 9, 24), at_0225, "15:30:30", 220)
    assert plan == "refuse" and s > 13 * 3600
    assert wait_plan(date(2026, 9, 23), at_0225, "15:30:30", 220) == ("replay", 0.0)
    at_1217 = datetime(2026, 9, 24, 16, 17, tzinfo=timezone.utc)  # 12:17 ET firing
    plan, s = wait_plan(date(2026, 9, 24), at_1217, "15:30:30", 220)
    assert plan == "sleep" and s == 3 * 3600 + 13 * 60 + 30
    at_1531 = datetime(2026, 9, 24, 15, 31, tzinfo=ET_TZ)
    assert wait_plan(date(2026, 9, 24), at_1531, "15:30:30", 220) == ("sleep", 0.0)
