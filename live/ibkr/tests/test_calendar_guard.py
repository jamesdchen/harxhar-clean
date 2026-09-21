"""The no-short calendars: NYSE sessions by rule, the month-end modes, and the
third-Friday (monthly-expiration) session.

The hand-checked dates come from the exchange's published holiday schedules.
The three parity checks tie the rule to data it never saw being written: the
seed ledger's sessions against proposal 54's empirical month-end flags and
against proposal 58's third-Friday flags, and 26 years of the 30-minute panel's
16:00 bars against the holiday rules.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

from live.ibkr.calendar_guard import (
    CALENDAR_FLAT_REASON,
    CALENDAR_OVERRIDE_REASON,
    MONTH_END_MODES,
    NO_SHORT_CALENDARS,
    SPECIAL_CLOSURES,
    THIRD_FRIDAY_REASON,
    NoShortCalendar,
    evaluate,
    is_last_session_of_month,
    is_session,
    is_third_friday_session,
    ledger_month_end_parity,
    monthly_expiration_session,
    nyse_holidays,
)

D = dt.date

SEED = os.path.join("results", "live_seed", "premium_ledger.parquet")
P54 = os.path.join(
    "results", "atm_straddle_0dte_1530", "proposals", "54", "c_daily.csv"
)
P58_PATH = os.path.join("writeup", "intraday_proposals", "58_third_friday_short.py")
P58_BY_YEAR = os.path.join(
    "results", "atm_straddle_0dte_1530", "proposals", "58", "e_by_year.csv"
)
#: what proposal 58 counted on the 1279 scored sessions (12 a year, 2020-2025)
P58_THIRD_FRIDAYS = 72
#: the two in the ledger that are not Fridays: Good Friday was the third Friday
P58_MOVED_TO_THURSDAY = [D(2022, 4, 14), D(2025, 4, 17)]
PANEL = os.path.join("data", "core_stats.parquet")
#: what proposal 54 counted on the 1279 scored sessions
P54_MONTH_ENDS = 70
#: the months whose last session is a 13:00 early close the ledger dropped
HALF_SESSION_MONTH_ENDS = [D(2024, 11, 29), D(2025, 11, 28)]
#: one-off historical 13:00 closes: sessions by rule, no 16:00 bar in the panel
PANEL_EARLY_CLOSES = {D(1999, 12, 31), D(2002, 7, 5), D(2003, 12, 26)}
PANEL_START, PANEL_END = D(1998, 1, 5), D(2024, 4, 30)


# ------------------------------------------------------------ hand-checked --


@pytest.mark.parametrize(
    "day, expected, why",
    [
        (D(2021, 5, 28), True, "Monday 2021-05-31 is Memorial Day"),
        (D(2021, 5, 31), False, "Memorial Day itself is not a session"),
        (D(2024, 3, 28), True, "2024-03-29 is Good Friday"),
        (D(2024, 3, 29), False, "Good Friday is not a session"),
        (D(2024, 11, 29), True, "a 13:00 early close is still the last session"),
        (D(2024, 11, 27), False, "the last FULL session is not the last session"),
        (D(2025, 6, 30), True, "an ordinary Monday month-end"),
        (D(2025, 12, 31), True, "New Year's Eve is a full session"),
        (D(2024, 2, 29), True, "leap-year February"),
        (D(2023, 2, 28), True, "ordinary February"),
        (D(2024, 2, 28), False, "the 29th follows in a leap year"),
        (D(2023, 11, 30), True, "the hold book's worst day"),
        (D(2025, 6, 20), False, "a mid-month Friday (the third: an expiration)"),
        (D(2025, 5, 30), True, "Saturday 2025-05-31 is not a session"),
        (D(2025, 5, 31), False, "a Saturday"),
        (D(2021, 12, 31), True, "1 January 2022 is a Saturday: not observed Friday"),
        (D(2027, 12, 31), True, "a future date: nothing here reads a ledger"),
        (D(2029, 3, 29), True, "2029-03-30 is Good Friday"),
    ],
)
def test_is_last_session_of_month(day, expected, why):
    assert is_last_session_of_month(day) is expected, why


def test_the_2024_and_2025_holiday_schedules():
    assert nyse_holidays(2024) == {
        D(2024, 1, 1),
        D(2024, 1, 15),
        D(2024, 2, 19),
        D(2024, 3, 29),
        D(2024, 5, 27),
        D(2024, 6, 19),
        D(2024, 7, 4),
        D(2024, 9, 2),
        D(2024, 11, 28),
        D(2024, 12, 25),
    }
    assert nyse_holidays(2025) == {
        D(2025, 1, 1),
        D(2025, 1, 9),  # the Carter day of mourning, a SPECIAL closure
        D(2025, 1, 20),
        D(2025, 2, 17),
        D(2025, 4, 18),
        D(2025, 5, 26),
        D(2025, 6, 19),
        D(2025, 7, 4),
        D(2025, 9, 1),
        D(2025, 11, 27),
        D(2025, 12, 25),
    }


def test_the_observance_rules():
    assert D(2021, 7, 5) in nyse_holidays(2021)  # 4 July on a Sunday -> Monday
    assert D(2020, 7, 3) in nyse_holidays(2020)  # 4 July on a Saturday -> Friday
    assert D(2021, 12, 24) in nyse_holidays(2021)  # Christmas on a Saturday
    assert D(2022, 12, 26) in nyse_holidays(2022)  # Christmas on a Sunday
    assert D(2023, 1, 2) in nyse_holidays(2023)  # New Year's Day on a Sunday
    assert D(2021, 12, 31) not in nyse_holidays(2021)  # ... but never the Friday
    assert D(2022, 6, 20) in nyse_holidays(2022)  # Juneteenth on a Sunday
    assert D(2021, 6, 18) not in nyse_holidays(2021)  # not observed before 2022
    assert D(1997, 1, 20) not in nyse_holidays(1997)  # nor King's birthday
    assert not is_session(D(2025, 6, 21)) and is_session(D(2025, 6, 20))
    assert all(not is_session(d) for d in SPECIAL_CLOSURES)


# ----------------------------------------------- the third Friday (58) -----


@pytest.mark.parametrize(
    "day, expected, why",
    [
        (D(2025, 4, 17), True, "Good Friday 2025-04-18 was the third Friday"),
        (D(2025, 4, 18), False, "Good Friday is not a session"),
        (D(2025, 4, 11), False, "the second Friday"),
        (D(2022, 4, 14), True, "Good Friday 2022-04-15 was the third Friday too"),
        (D(2024, 6, 21), True, "June 2024 starts on a Saturday"),
        (D(2025, 12, 19), True, "December 2025"),
        (D(2023, 11, 17), True, "November 2023"),
        (D(2025, 6, 20), True, "the README's replay session"),
        (D(2025, 6, 13), False, "an ordinary (second) Friday"),
        (D(2025, 5, 30), False, "a fifth Friday"),
        (D(2025, 6, 19), False, "Juneteenth 2025, a Thursday holiday"),
        (D(2025, 6, 21), False, "a Saturday"),
        (D(2026, 6, 18), True, "a future date: Juneteenth 2026 is the third Friday"),
        (D(2026, 6, 19), False, "... and not a session"),
    ],
)
def test_is_third_friday_session(day, expected, why):
    assert is_third_friday_session(day) is expected, why


def test_every_monthly_expiration_is_a_session_in_its_own_month():
    for year in range(1998, 2031):
        for month in range(1, 13):
            d = monthly_expiration_session(year, month)
            assert (d.year, d.month) == (year, month) and is_session(d)
            # the Friday itself, or the Thursday before a holiday Friday
            assert d.weekday() in (3, 4) and 14 <= d.day <= 21
            assert is_third_friday_session(d)
            assert not is_third_friday_session(d - dt.timedelta(days=7))


def test_the_calendar_record_carries_the_third_friday():
    tf = evaluate(D(2025, 6, 20), ("month_end",), third_friday_multiplier=2.0)
    assert tf["decision"] == "short" and tf["third_friday"] is True
    assert (tf["third_friday_multiplier"], tf["third_friday_applied"]) == (2.0, 2.0)
    assert tf["third_friday_reason"].startswith(THIRD_FRIDAY_REASON)
    assert "multiplied by 2" in tf["third_friday_reason"]
    assert "proposal 58" in tf["third_friday_evidence"]
    # --n sets the size by hand: flagged, not applied
    by_hand = evaluate(
        D(2025, 6, 20), ("month_end",), third_friday_multiplier=2.0, size_override=True
    )
    assert by_hand["third_friday"] is True and by_hand["third_friday_applied"] == 1.0
    assert "--n sets the size" in by_hand["third_friday_reason"]
    # an ordinary session: nothing flagged, nothing applied
    plain = evaluate(D(2025, 6, 13), ("month_end",), third_friday_multiplier=2.0)
    assert plain["third_friday"] is False and plain["third_friday_applied"] == 1.0
    assert plain["third_friday_reason"] == "" and plain["third_friday_evidence"] == ""
    # a month-end is never a third Friday, and is not scaled
    me = evaluate(D(2023, 11, 30), ("month_end",), third_friday_multiplier=2.0)
    assert me["decision"] == "override" and me["third_friday_applied"] == 1.0


def test_the_month_end_calendar_is_read_before_the_third_friday(monkeypatch):
    """No real session is both, so a registry entry that hits on third Fridays
    stands in for one: a day that is not a short day is never scaled."""
    monkeypatch.setitem(
        NO_SHORT_CALENDARS,
        "expiration_probe",
        NoShortCalendar(
            name="expiration_probe",
            applies=is_third_friday_session,
            label="a test probe",
            evidence="none",
        ),
    )
    for mode in ("override", "sit_out"):
        hit = evaluate(
            D(2025, 6, 20),
            ("expiration_probe",),
            mode=mode,
            third_friday_multiplier=3.0,
        )
        assert hit["decision"] == mode and hit["third_friday"] is True
        assert hit["third_friday_applied"] == 1.0
        assert "not a short day" in hit["third_friday_reason"]
    off = evaluate(
        D(2025, 6, 20), ("expiration_probe",), mode="off", third_friday_multiplier=3.0
    )
    assert off["decision"] == "short" and off["third_friday_applied"] == 3.0


# --------------------------------------------------------------- the modes --


def test_the_default_mode_overrides_a_month_end():
    hit = evaluate(D(2023, 11, 30), ("month_end",))
    assert MONTH_END_MODES == ("override", "sit_out", "off")
    assert (hit["mode"], hit["decision"]) == ("override", "override")
    assert hit["override"] and not hit["flat"] and hit["hits"] == ["month_end"]
    assert hit["reason"].startswith(CALENDAR_OVERRIDE_REASON)
    assert "2023-11-30" in hit["reason"] and "BUYS the 15:30 straddle" in hit["reason"]
    assert "proposal 54" in hit["evidence"][0] and "+0.43" in hit["evidence"][0]


def test_sit_out_ends_a_month_end_flat():
    hit = evaluate(D(2023, 11, 30), ("month_end",), mode="sit_out")
    assert hit["decision"] == "sit_out" and hit["flat"] and not hit["override"]
    assert hit["reason"].startswith(CALENDAR_FLAT_REASON)
    assert "FLAT" in hit["reason"]


def test_off_and_ordinary_days_trade_the_short_book():
    off = evaluate(D(2023, 11, 30), ("month_end",), mode="off")
    assert off["decision"] == "short" and off["enabled"] is False
    assert not off["flat"] and not off["override"] and off["hits"] == []
    for mode in MONTH_END_MODES:
        miss = evaluate(D(2025, 6, 20), ("month_end",), mode=mode)
        assert miss["decision"] == "short" and miss["hits"] == []
        assert miss["reason"] == "" and not miss["flat"] and not miss["override"]
    none = evaluate(D(2023, 11, 30), ())
    assert none["decision"] == "short" and none["calendars"] == []


def test_only_month_end_is_registered_and_unknown_names_are_refused():
    assert sorted(NO_SHORT_CALENDARS) == ["month_end"]
    with pytest.raises(ValueError, match="unknown no-short calendar"):
        evaluate(D(2025, 6, 20), ("fomc",))
    with pytest.raises(ValueError, match="month_end_mode"):
        evaluate(D(2025, 6, 20), ("month_end",), mode="guard")


# ---------------------------------------------------------------- parity ----


@pytest.mark.skipif(
    not (os.path.exists(SEED) and os.path.exists(P54)),
    reason="needs the seed premium ledger and proposal 54's daily file",
)
def test_parity_with_proposal_54_on_the_seed_ledger():
    """The rule, which reads no data, against 54's empirical calendar."""
    import pandas as pd

    from live.ibkr.premium_ledger import PremiumLedger

    sessions = PremiumLedger.load(SEED).sessions()
    ref = pd.read_csv(P54, index_col=0, parse_dates=True)["month_end"].astype(bool)
    reference = {ts.date(): bool(v) for ts, v in ref.items()}
    assert set(sessions) == set(reference)
    res = ledger_month_end_parity(sessions, reference)
    assert res["disagreements"] == []
    assert res["n_rule"] == res["n_reference"] == P54_MONTH_ENDS

    # every month of the ledger has its true last session in it, except the
    # two whose last session is a half session the ledger dropped
    have = set(sessions)
    absent = []
    for year, month in sorted({(d.year, d.month) for d in sessions}):
        days = [D(year, month, 1) + dt.timedelta(days=i) for i in range(31)]
        last = max(d for d in days if d.month == month and is_session(d))
        if last not in have:
            absent.append(last)
    assert absent == HALF_SESSION_MONTH_ENDS
    print(
        f"{len(sessions)} sessions: rule {res['n_rule']} month-ends = proposal "
        f"54's {res['n_reference']}, 0 disagreements; absent half-session "
        f"month-ends {absent}"
    )


@pytest.mark.skipif(
    not (
        os.path.exists(SEED)
        and os.path.exists(P58_PATH)
        and os.path.exists(P58_BY_YEAR)
    ),
    reason="needs the seed premium ledger and proposal 58",
)
def test_parity_with_proposal_58_on_the_seed_ledger(monkeypatch):
    """The rule against 58's own ``nth_weekday_sessions``, imported by path.

    58 flagged the sessions of an empirical calendar (the panel's 16:00 days
    joined with the chain's); on the ledger's sessions, which are that
    calendar's from 2020 on, the two must name the same 72 days, and 58's
    by-year table must count them.
    """
    import importlib.util
    import sys

    import pandas as pd

    from live.ibkr.premium_ledger import PremiumLedger

    # 58's body puts its own directories on sys.path; keep that to this test
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("p58_for_third_friday", P58_PATH)
    assert spec is not None and spec.loader is not None
    p58 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p58)

    sessions = PremiumLedger.load(SEED).sessions()
    cal = pd.DatetimeIndex(pd.to_datetime(sessions))
    ref = p58.nth_weekday_sessions(cal, 4, 3)
    rule = [is_third_friday_session(d) for d in sessions]
    assert [d for d, a, b in zip(sessions, rule, ref) if a != bool(b)] == []
    flagged = [d for d, a in zip(sessions, rule) if a]
    assert len(flagged) == P58_THIRD_FRIDAYS
    assert [d for d in flagged if d.weekday() != 4] == P58_MOVED_TO_THURSDAY

    by_year = pd.read_csv(P58_BY_YEAR).set_index("year")["n"].to_dict()
    mine: dict[int, int] = {}
    for d in flagged:
        mine[d.year] = mine.get(d.year, 0) + 1
    assert mine == {int(k): int(v) for k, v in by_year.items()}
    print(
        f"{len(sessions)} sessions: rule {len(flagged)} third Fridays = proposal "
        f"58's {int(sum(ref))}, 0 disagreements; by year {mine}"
    )


@pytest.mark.skipif(not os.path.exists(PANEL), reason="needs data/core_stats.parquet")
def test_the_holiday_rules_against_26_years_of_the_panel():
    """A holiday has no 16:00 bar; a session has one, bar three old early closes."""
    import pandas as pd

    c = pd.read_parquet(PANEL, columns=["endbartime", "sumret2"])
    t = pd.to_datetime(c["endbartime"])
    full = set(t[(t.dt.strftime("%H:%M") == "16:00") & c["sumret2"].notna()].dt.date)
    false_holidays, missed = [], []
    d = PANEL_START
    while d <= PANEL_END:
        if d.weekday() < 5:
            if not is_session(d) and d in full:
                false_holidays.append(d)
            if is_session(d) and d not in full:
                missed.append(d)
        d += dt.timedelta(days=1)
    assert false_holidays == []

    # A rule-session with no 16:00 bar must be a 13:00 early close -- the day
    # after Thanksgiving, Christmas Eve, 3 July, or one of three one-offs --
    # and never an ordinary day, which is what a MISSED holiday would look like.
    def early_close(day: dt.date) -> bool:
        thanksgiving = max(
            D(day.year, 11, k)
            for k in range(22, 29)
            if D(day.year, 11, k).weekday() == 3
        )
        return (
            day == thanksgiving + dt.timedelta(days=1)
            or (day.month, day.day) in {(12, 24), (7, 3)}
            or day in PANEL_EARLY_CLOSES
        )

    assert [d for d in missed if not early_close(d)] == []
    assert PANEL_EARLY_CLOSES <= set(missed)
