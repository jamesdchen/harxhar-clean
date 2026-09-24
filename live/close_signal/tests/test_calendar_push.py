"""The Calendar push against a fake service: idempotent per session, right shape."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.calendar_push import (  # noqa: E402
    KEY_PROP,
    build_event,
    event_key,
    push_event,
    summary_for,
)


class _Call:
    def __init__(self, result):
        self._r = result

    def execute(self):
        return self._r


class _Events:
    def __init__(self, existing):
        self.existing = existing
        self.calls: list[tuple[str, dict]] = []

    def list(self, **kw):
        self.calls.append(("list", kw))
        return _Call({"items": self.existing})

    def insert(self, **kw):
        self.calls.append(("insert", kw))
        return _Call({"id": "new-id"})

    def update(self, **kw):
        self.calls.append(("update", kw))
        return _Call({"id": kw["eventId"]})


class _Service:
    def __init__(self, existing):
        self._events = _Events(existing)

    def events(self):
        return self._events


def test_event_shape() -> None:
    ev = build_event(date(2026, 9, 23), "s", "body")
    assert ev["start"]["dateTime"].startswith("2026-09-23T15:30:00")
    assert ev["end"]["dateTime"].startswith("2026-09-23T16:00:00")
    assert ev["reminders"]["overrides"] == [{"method": "popup", "minutes": 0}]
    assert ev["extendedProperties"]["private"][KEY_PROP] == "2026-09-23"
    assert event_key(date(2026, 9, 23)) == f"{KEY_PROP}=2026-09-23"


def test_insert_when_absent_and_update_when_present() -> None:
    ev = build_event(date(2026, 9, 23), "s", "body")
    svc = _Service(existing=[])
    assert push_event(svc, "primary", ev, date(2026, 9, 23)) == "new-id"
    kinds = [c[0] for c in svc.events().calls]
    assert kinds == ["list", "insert"]
    assert svc.events().calls[0][1]["privateExtendedProperty"] == event_key(
        date(2026, 9, 23)
    )
    svc2 = _Service(existing=[{"id": "old-id"}])
    assert push_event(svc2, "primary", ev, date(2026, 9, 23)) == "old-id"
    assert [c[0] for c in svc2.events().calls] == ["list", "update"]


def test_summaries() -> None:
    assert summary_for("BUY_MONTH_END", False).startswith("Close trade: MONTH-END BUY")
    assert summary_for("NO_SIGNAL", True).endswith("[LATE]")
    assert "do not trade" in summary_for("NO_SIGNAL", False)
    assert "NO TRADE" in summary_for("NO_TRADE_FLAG", False)
    # the card's own headline wins when given
    assert (
        summary_for("BUY_IF_ASK_LE_PSTAR", False, "Close trade: buy X")
        == "Close trade: buy X"
    )
