"""Post the instruction as a Google Calendar event.

The event sits at 15:30 ET for 30 minutes with a popup reminder at the event
time, so the phone pings the moment the card exists.  Idempotent per session:
the event carries a private extended property ``close_signal_key=<date>`` and
a rerun updates it instead of adding a second one.  The Google libraries are
imported lazily so the package and its tests run without them.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
KEY_PROP = "close_signal_key"
SCOPES = ("https://www.googleapis.com/auth/calendar.events",)


def make_service(
    client_id: str | None = None,
    client_secret: str | None = None,
    refresh_token: str | None = None,
) -> Any:
    """An authenticated Calendar API client from a refresh token (env by default)."""
    from google.oauth2.credentials import Credentials  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    creds = Credentials(
        None,
        refresh_token=refresh_token or os.environ["GCAL_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id or os.environ["GCAL_CLIENT_ID"],
        client_secret=client_secret or os.environ["GCAL_CLIENT_SECRET"],
        scopes=list(SCOPES),
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def event_key(session: date) -> str:
    return f"{KEY_PROP}={session.isoformat()}"


def build_event(
    session: date, summary: str, body: str, start_hhmm: str = "15:30"
) -> dict[str, Any]:
    h, m = (int(x) for x in start_hhmm.split(":"))
    start = datetime(session.year, session.month, session.day, h, m, tzinfo=ET)
    end = start + timedelta(minutes=30)
    return {
        "summary": summary,
        "description": body,
        "start": {"dateTime": start.isoformat(), "timeZone": "America/New_York"},
        "end": {"dateTime": end.isoformat(), "timeZone": "America/New_York"},
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 0}],
        },
        "extendedProperties": {"private": {KEY_PROP: session.isoformat()}},
    }


def push_event(
    service: Any, calendar_id: str, event: dict[str, Any], session: date
) -> str:
    """Insert or update the session's event; returns the event id."""
    found = (
        service.events()
        .list(
            calendarId=calendar_id,
            privateExtendedProperty=event_key(session),
            maxResults=5,
        )
        .execute()
    )
    items = found.get("items", [])
    if items:
        eid = items[0]["id"]
        out = (
            service.events()
            .update(calendarId=calendar_id, eventId=eid, body=event)
            .execute()
        )
    else:
        out = service.events().insert(calendarId=calendar_id, body=event).execute()
    return str(out.get("id", ""))


def summary_for(decision: str, late: bool, headline: str = "") -> str:
    """The event title: the card's own headline when it has one."""
    base = headline or {
        "BUY_MONTH_END": "Close trade: MONTH-END BUY at 15:30",
        "BUY_IF_ASK_LE_PSTAR": "Close trade: buy if the ask is at or below the limit",
        "NO_TRADE_FLAG": "Close trade: NO TRADE today",
        "NO_SIGNAL": "Close trade: NO SIGNAL (the job failed, do not trade)",
    }.get(decision, "Close trade")
    return base + (" [LATE]" if late else "")


__all__ = [
    "KEY_PROP",
    "SCOPES",
    "build_event",
    "event_key",
    "make_service",
    "push_event",
    "summary_for",
]
