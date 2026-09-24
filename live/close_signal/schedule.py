"""When the job should run: the Actions guard and the session calendar.

GitHub Actions cron is UTC and best-effort: firings are delayed, the top of
the hour worst, and some are dropped.  The first schedule (19:00 / 20:00 UTC,
a 14:45-15:25 ET window) never ran on 2026-09-24 -- the 19:00 firing had not
started by 15:29 ET, so the session had no card.  The workflow therefore fires
three times off the hour (16:17, 17:17, 18:17 UTC = 12:17 / 13:17 / 14:17 ET in
daylight time, an hour earlier in standard time), accepts any firing between
12:00 and 15:25 ET, and the first to get through sleeps to 15:30:30; the others
see its journal row (``already_journaled``) or fall outside the window and exit.
This module decides that, whether today is a session with a 16:00 close, and
how long to sleep to the decision stamp.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from live.ibkr.calendar_guard import (
    is_last_session_of_month,
    is_session,
    is_third_friday_session,
)

ET_TZ = ZoneInfo("America/New_York")

#: The window (ET) inside which a firing is accepted as the day's run.
RUN_WINDOW_START = time(12, 0)
RUN_WINDOW_END = time(15, 25)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def is_early_close(d: date) -> bool:
    """NYSE 13:00 closes: the day after Thanksgiving, Christmas Eve and July 3 on weekdays.

    (July 3 closes early only when July 4 is a weekday, i.e. when the market
    is open on the 3rd and closed on the 4th; Christmas Eve when it falls on a
    weekday.)  Matches the notebook library's EARLY_CLOSE_DATES on 2020-2025.
    """
    if not is_session(d):
        return False
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)  # fourth Thursday
    if d == thanksgiving + timedelta(days=1):
        return True
    if d.month == 12 and d.day == 24:
        return True
    if d.month == 7 and d.day == 3 and date(d.year, 7, 4).weekday() < 5:
        return True
    return False


def has_1600_close(d: date) -> bool:
    return is_session(d) and not is_early_close(d)


def should_run(now_utc: datetime) -> tuple[bool, str]:
    """(run?, reason) for a firing at ``now_utc``."""
    now = now_utc.astimezone(ET_TZ)
    d = now.date()
    if not is_session(d):
        return False, f"{d} is not a session"
    if is_early_close(d):
        return False, f"{d} closes at 13:00; there is no 15:30-16:00 bar"
    t = now.time()
    if not (RUN_WINDOW_START <= t <= RUN_WINDOW_END):
        return False, (
            f"{now.strftime('%H:%M')} ET is outside the "
            f"{RUN_WINDOW_START:%H:%M}-{RUN_WINDOW_END:%H:%M} firing window"
        )
    return True, f"session {d}, fired {now.strftime('%H:%M')} ET"


def already_journaled(journal: pd.DataFrame, d: date) -> bool:
    """True if the journal holds a row (card or NO SIGNAL) for session ``d``.

    The later firings of a day start after the first has pushed its state, so
    this is what keeps a session to one card.
    """
    if journal.empty or "session" not in journal.columns:
        return False
    return bool((pd.to_datetime(journal["session"]).dt.date == d).any())


def seconds_until(hhmmss: str, now: datetime | None = None) -> float:
    """Seconds from ``now`` (ET) to the clock ``hhmmss`` today; 0 if past."""
    now = now or datetime.now(ET_TZ)
    h, m, s = (int(x) for x in hhmmss.split(":"))
    target = now.replace(hour=h, minute=m, second=s, microsecond=0)
    return max(0.0, (target - now).total_seconds())


def wait_plan(
    session: date, now: datetime, hhmmss: str, max_wait_min: float | None
) -> tuple[str, float]:
    """("replay" | "sleep" | "refuse", seconds) before the decision stamp.

    A past session is a replay: its inputs are complete, so no wait.  Today's
    session sleeps to ``hhmmss`` ET unless that is longer than ``max_wait_min``
    -- the job's timeout would kill the sleep and no card would post (run
    35964417416, dispatched 02:25 ET on 2026-09-24, slept toward a 13-hour wait).
    """
    now_et = now.astimezone(ET_TZ)
    if session != now_et.date():
        return "replay", 0.0
    s = seconds_until(hhmmss, now_et)
    if max_wait_min is not None and s > 60.0 * max_wait_min:
        return "refuse", s
    return "sleep", s


def calendar_flags(d: date) -> dict[str, bool]:
    return {
        "session": is_session(d),
        "early_close": is_early_close(d),
        "month_end": is_last_session_of_month(d),
        "third_friday": is_third_friday_session(d),
    }


__all__ = [
    "ET_TZ",
    "already_journaled",
    "calendar_flags",
    "has_1600_close",
    "is_early_close",
    "seconds_until",
    "should_run",
    "wait_plan",
]
