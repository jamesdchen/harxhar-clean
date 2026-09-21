"""No-short calendars: sessions the afternoon short book does not sell into.

The afternoon hold book sells a straddle and carries it into the cash
settlement.  Proposal 54 (``writeup/intraday_proposals/54_month_end_close.py``)
measured the one class of sessions on which that loses: the LAST TRADING
SESSION OF A CALENDAR MONTH, when month-end rebalancing is executed at the
closing auction.  Sold 13:30 and held, the short straddle loses 1.8 index
points per contract on those 70 sessions against +1.1 on every other session
(t 4.7); bought back at 15:30 it still loses 0.6.  The same close is cheap to
BUY: the 15:30 nearest-OTM straddle bought at the quoted ask and held to cash
settlement earns +0.43 premium units (+2.9 index points per contract) on those
70 sessions, and the same +0.43 on the 18 held-out month-ends from 2024-05
(proposals 54 and 55).

What a hit does is the MODE (``MONTH_END_MODES``):

* ``override`` (the default) -- the short program does not enter; at 15:30
  the runner BUYS the nearest-OTM straddle and holds it to settlement;
* ``sit_out`` -- the day ends flat in preflight, before any entry, exactly as
  the deleveraging rule's zero multiplier does;
* ``off`` -- the calendar is not consulted and the short book trades.

The calendars are a LIST of named entries (``NO_SHORT_CALENDARS``), so another
class of days can be added by configuration once it has its own evidence --
for BOTH halves of the override: that the short loses into its close and that
the long is paid for it.  Only ``month_end`` is registered today.

One class of sessions goes the other way: the MONTHLY-EXPIRATION session, the
third Friday of the month (the session before it when that Friday is not one).
Proposal 58 measured the short 13:30 hold book there at +3.0 index points per
contract (t 5.3, 72 sessions) against +0.8 on every other session, and the
15:30 straddle sold at the bid at +0.33 premium units (t 3.9).  On those
sessions the short program's contract count is multiplied by
``third_friday_multiplier`` (``Config.third_friday_size_multiplier``).  The
no-short calendars are read FIRST: the multiplier scales the short program only,
so a session that is not a short day (an override or a sit-out) is never scaled.

Live trading needs the answer for a FUTURE date, so nothing here reads a
ledger: the NYSE holiday schedule is computed from the exchange's own rules
(Rule 7.2) with ``dateutil.easter`` for Good Friday.  ``dateutil`` already
ships with pandas, so this adds no dependency.  What rules cannot know is a
SPECIAL closure announced at short notice (a day of mourning, a storm); the
ones on record are listed in ``SPECIAL_CLOSURES`` and a new one has to be
added by hand.  The failure is one-sided and small: a special closure on a
month's last weekday makes the guard miss the true last session, the day
before it.

A scheduled 13:00 early close (the day after Thanksgiving, Christmas Eve) is
still a session here, so it can be a month's last one -- 2024-11-29 and
2025-11-28 are.  The runner refuses a half session off the broker's liquid
hours before it ever reaches this calendar, so such a day ends flat in every
mode: the override is never tried on a close that is not 16:00.

Nothing in this module talks to a socket.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from dateutil.easter import easter  # type: ignore[import-untyped]

MONDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = 0, 3, 4, 5, 6

#: Standard monthly SPX options expire on this occurrence of ``FRIDAY``.
MONTHLY_EXPIRATION_WEEK = 3

#: The first year the exchange observed each holiday that has a start date.
MLK_FIRST_YEAR = 1998
JUNETEENTH_FIRST_YEAR = 2022

#: Unscheduled full-day closures on record (NYSE): 11-14 September 2001, the
#: Reagan, Ford, Bush and Carter days of mourning, and Hurricane Sandy.
SPECIAL_CLOSURES: frozenset[date] = frozenset(
    {
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),
        date(2007, 1, 2),
        date(2012, 10, 29),
        date(2012, 10, 30),
        date(2018, 12, 5),
        date(2025, 1, 9),
    }
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` of a month (``n`` counts from 1)."""
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date, saturday_to_friday: bool = True) -> date | None:
    """Rule 7.2: a Sunday holiday is kept on Monday, a Saturday one on Friday.

    New Year's Day is the exception the rule carves out: on a Saturday it is
    not observed at all, because the Friday is the last session of the year.
    """
    if d.weekday() == SUNDAY:
        return d + timedelta(days=1)
    if d.weekday() == SATURDAY:
        return d - timedelta(days=1) if saturday_to_friday else None
    return d


def nyse_holidays(year: int) -> frozenset[date]:
    """Every scheduled full-day closure in ``year``, plus the special ones."""
    days: list[date | None] = [
        _observed(date(year, 1, 1), saturday_to_friday=False),
        _nth_weekday(year, 2, MONDAY, 3),  # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, MONDAY),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, MONDAY, 1),  # Labor Day
        _nth_weekday(year, 11, THURSDAY, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    ]
    if year >= MLK_FIRST_YEAR:
        days.append(_nth_weekday(year, 1, MONDAY, 3))
    if year >= JUNETEENTH_FIRST_YEAR:
        days.append(_observed(date(year, 6, 19)))
    # 1 January of NEXT year on a Saturday is not observed (see ``_observed``),
    # so nothing from the neighbouring year ever lands in this one.
    out = {d for d in days if d is not None and d.year == year}
    out |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return frozenset(out)


def is_session(d: date) -> bool:
    """True when the exchange opens on ``d`` (a 13:00 early close still counts)."""
    return d.weekday() < SATURDAY and d not in nyse_holidays(d.year)


def is_last_session_of_month(d: date) -> bool:
    """True when ``d`` is a session and no later day of its month is one."""
    if not is_session(d):
        return False
    nxt = d + timedelta(days=1)
    while nxt.month == d.month:
        if is_session(nxt):
            return False
        nxt += timedelta(days=1)
    return True


def monthly_expiration_session(year: int, month: int) -> date:
    """The session on or before the month's third Friday.

    That Friday is the 15th at the earliest, so walking back over a holiday
    (Good Friday 2025-04-18 gives Thursday 2025-04-17) never leaves the month.
    """
    d = _nth_weekday(year, month, FRIDAY, MONTHLY_EXPIRATION_WEEK)
    while not is_session(d):
        d -= timedelta(days=1)
    return d


def is_third_friday_session(d: date) -> bool:
    """True when ``d`` is its month's monthly-expiration session (proposal 58)."""
    return is_session(d) and d == monthly_expiration_session(d.year, d.month)


@dataclass(frozen=True)
class NoShortCalendar:
    """One named class of sessions the short book does not hold through."""

    name: str
    applies: Callable[[date], bool]
    label: str
    evidence: str


NO_SHORT_CALENDARS: dict[str, NoShortCalendar] = {
    "month_end": NoShortCalendar(
        name="month_end",
        applies=is_last_session_of_month,
        label="the last trading session of the month",
        evidence=(
            "proposal 54: sold 13:30 and held, the short straddle loses 1.8 "
            "index points per contract on month-ends against +1.1 on other "
            "sessions; the loss is the closing half hour. Proposals 54/55: the "
            "15:30 straddle bought at the ask and held earns +0.43 premium "
            "units (+2.9 index points per contract) on the 70 month-ends, and "
            "+0.43 on the 18 held out from 2024-05"
        ),
    ),
}

#: What a hit on a no-short calendar does (see the module docstring).
MONTH_END_MODES: tuple[str, ...] = ("override", "sit_out", "off")

#: The head of the sit-out refusal, so a supervisor grepping the journal or the
#: summary matches one string.
CALENDAR_FLAT_REASON = "calendar guard: "
#: The head of the override's announcement, for the same reason.
CALENDAR_OVERRIDE_REASON = "month-end override: "
#: The head of the third-Friday size note, for the same reason.
THIRD_FRIDAY_REASON = "third-Friday size: "

THIRD_FRIDAY_EVIDENCE = (
    "proposal 58: on the monthly-expiration session the short 13:30 hold book "
    "earns +3.00 index points per contract (t 5.3, 72 sessions) against +0.80 on "
    "other sessions, and the 15:30 straddle sold at the bid earns +0.33 premium "
    "units (t 3.9); both samples had been seen before its criteria were written, "
    "and the close is not calmer on those days"
)


def evaluate(
    d: date,
    names: Iterable[str],
    mode: str = "override",
    third_friday_multiplier: float = 1.0,
    size_override: bool = False,
) -> dict[str, Any]:
    """The calendar's verdict for one session, as the journal records it.

    ``decision`` is ``"short"`` (no hit, or the mode is ``off``), ``"sit_out"``
    or ``"override"``; ``flat`` is True only for a sit-out and ``override`` only
    for an override, so a caller can branch on either without re-deriving it.

    The third-Friday multiplier is read AFTER the decision: it scales the short
    program's contract count, so ``third_friday_applied`` is the configured
    multiplier only on a monthly-expiration session whose decision is
    ``"short"`` and whose size comes from the stress table, and 1.0 otherwise
    (``size_override`` is ``--n``, which sets the count by hand).
    """
    names = tuple(names)
    unknown = [n for n in names if n not in NO_SHORT_CALENDARS]
    if unknown:
        raise ValueError(
            "unknown no-short calendar(s) "
            + repr(unknown)
            + "; registered: "
            + repr(sorted(NO_SHORT_CALENDARS))
        )
    if mode not in MONTH_END_MODES:
        raise ValueError(
            "month_end_mode must be one of "
            + repr(MONTH_END_MODES)
            + ", got "
            + repr(mode)
        )
    enabled = mode != "off"
    hits = [n for n in names if NO_SHORT_CALENDARS[n].applies(d)] if enabled else []
    decision = "short" if not hits else mode
    labels = " and ".join(NO_SHORT_CALENDARS[n].label for n in hits)
    reason = ""
    if decision == "sit_out":
        reason = (
            CALENDAR_FLAT_REASON
            + d.isoformat()
            + " is "
            + labels
            + " -- the short book does not hold into this close: the day is FLAT"
        )
    elif decision == "override":
        reason = (
            CALENDAR_OVERRIDE_REASON
            + d.isoformat()
            + " is "
            + labels
            + " -- no short into this close; the runner BUYS the 15:30 straddle "
            "and holds it to cash settlement"
        )

    third = is_third_friday_session(d)
    tf_mult = float(third_friday_multiplier)
    applied = tf_mult if (third and decision == "short" and not size_override) else 1.0
    tf_reason = ""
    if third:
        head = (
            THIRD_FRIDAY_REASON + d.isoformat() + " is the monthly-expiration session"
        )
        if decision != "short":
            tf_reason = head + " -- not a short day: the multiplier is not applied"
        elif size_override:
            tf_reason = head + " -- --n sets the size: the multiplier is not applied"
        else:
            tf_reason = (
                head
                + " -- the short program's contract count is multiplied by "
                + f"{applied:g}"
            )
    return {
        "mode": mode,
        "enabled": enabled,
        "session": d.isoformat(),
        "calendars": list(names),
        "hits": hits,
        "decision": decision,
        "flat": decision == "sit_out",
        "override": decision == "override",
        "reason": reason,
        "evidence": [NO_SHORT_CALENDARS[n].evidence for n in hits],
        "third_friday": third,
        "third_friday_multiplier": tf_mult,
        "third_friday_applied": applied,
        "third_friday_reason": tf_reason,
        "third_friday_evidence": THIRD_FRIDAY_EVIDENCE if third else "",
    }


def ledger_month_end_parity(
    sessions: Iterable[date], reference: dict[date, bool]
) -> dict[str, Any]:
    """The rule against an empirical month-end flag on a ledger's sessions.

    ``reference`` maps a session to the flag an empirical calendar gave it
    (proposal 54 used the panel's own regular-hours days and the chain's
    session list).  Returns the counts and every disagreement.
    """
    rows = [(d, is_last_session_of_month(d), bool(reference[d])) for d in sessions]
    return {
        "n_sessions": len(rows),
        "n_rule": sum(1 for _, a, _ in rows if a),
        "n_reference": sum(1 for _, _, b in rows if b),
        "disagreements": [(d, a, b) for d, a, b in rows if a != b],
    }
