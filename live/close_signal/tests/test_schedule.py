"""The Actions guard: which UTC firing is 15:00 ET, sessions, early closes."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(_ROOT / "notebooks"))

from live.close_signal.schedule import (  # noqa: E402
    calendar_flags,
    is_early_close,
    seconds_until,
    should_run,
)


def test_daylight_time_takes_the_19_utc_firing_and_standard_time_the_20_utc_one() -> (
    None
):
    # 2026-09-23 (EDT): 19:00 UTC = 15:00 ET
    assert should_run(datetime(2026, 9, 23, 19, 2, tzinfo=timezone.utc))[0] is True
    assert should_run(datetime(2026, 9, 23, 20, 2, tzinfo=timezone.utc))[0] is False
    # 2026-12-02 (EST): 20:00 UTC = 15:00 ET
    assert should_run(datetime(2026, 12, 2, 20, 2, tzinfo=timezone.utc))[0] is True
    assert should_run(datetime(2026, 12, 2, 19, 2, tzinfo=timezone.utc))[0] is False


def test_jitter_inside_the_window_is_accepted_and_late_is_not() -> None:
    assert (
        should_run(datetime(2026, 9, 23, 19, 20, tzinfo=timezone.utc))[0] is True
    )  # 15:20 ET
    ok, why = should_run(datetime(2026, 9, 23, 19, 40, tzinfo=timezone.utc))  # 15:40 ET
    assert ok is False and "outside" in why


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
