"""When the job should run: the Actions guard and the session calendar.

GitHub Actions cron is UTC and jitters by 5-20 minutes, so the workflow fires
at 19:00 and 20:00 UTC and this module decides which of the two is 15:00 ET
today (the other exits at once), whether today is a session with a 16:00
close, and how long to sleep to the decision stamp.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from live.ibkr.calendar_guard import (
    is_last_session_of_month,
    is_session,
    is_third_friday_session,
)

ET_TZ = ZoneInfo("America/New_York")

#: The window (ET) inside which a firing is accepted as "the 15:00 run".
RUN_WINDOW_START = time(14, 45)
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
        return False, f"{now.strftime('%H:%M')} ET is outside the 15:00 firing window"
    return True, f"session {d}, fired {now.strftime('%H:%M')} ET"


def seconds_until(hhmmss: str, now: datetime | None = None) -> float:
    """Seconds from ``now`` (ET) to the clock ``hhmmss`` today; 0 if past."""
    now = now or datetime.now(ET_TZ)
    h, m, s = (int(x) for x in hhmmss.split(":"))
    target = now.replace(hour=h, minute=m, second=s, microsecond=0)
    return max(0.0, (target - now).total_seconds())


def calendar_flags(d: date) -> dict[str, bool]:
    return {
        "session": is_session(d),
        "early_close": is_early_close(d),
        "month_end": is_last_session_of_month(d),
        "third_friday": is_third_friday_session(d),
    }


__all__ = [
    "ET_TZ",
    "calendar_flags",
    "has_1600_close",
    "is_early_close",
    "seconds_until",
    "should_run",
]
