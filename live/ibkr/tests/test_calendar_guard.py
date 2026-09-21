"""The no-short calendars: NYSE sessions by rule, and the month-end guard.

The hand-checked dates come from the exchange's published holiday schedules.
The two parity checks tie the rule to data it never saw being written: the
seed ledger's sessions against proposal 54's empirical month-end flags, and 26
years of the 30-minute panel's 16:00 bars against the holiday rules.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

from live.ibkr.calendar_guard import (
    CALENDAR_FLAT_REASON,
    NO_SHORT_CALENDARS,
    SPECIAL_CLOSURES,
    evaluate,
    is_last_session_of_month,
    is_session,
    ledger_month_end_parity,
    nyse_holidays,
)

D = dt.date

SEED = os.path.join("results", "live_seed", "premium_ledger.parquet")
P54 = os.path.join(
    "results", "atm_straddle_0dte_1530", "proposals", "54", "c_daily.csv"
)
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
        (D(2025, 6, 20), False, "an ordinary mid-month Friday"),
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


# --------------------------------------------------------------- the guard --


def test_evaluate_on_a_month_end_and_on_an_ordinary_day():
    hit = evaluate(D(2023, 11, 30), ("month_end",))
    assert hit["flat"] and hit["hits"] == ["month_end"]
    assert hit["reason"].startswith(CALENDAR_FLAT_REASON)
    assert "2023-11-30" in hit["reason"] and "FLAT" in hit["reason"]
    assert "proposal 54" in hit["evidence"][0]

    miss = evaluate(D(2025, 6, 20), ("month_end",))
    assert not miss["flat"] and miss["hits"] == [] and miss["reason"] == ""

    off = evaluate(D(2023, 11, 30), ("month_end",), enabled=False)
    assert not off["flat"] and off["enabled"] is False

    none = evaluate(D(2023, 11, 30), ())
    assert not none["flat"] and none["calendars"] == []


def test_only_month_end_is_registered_and_unknown_names_are_refused():
    assert sorted(NO_SHORT_CALENDARS) == ["month_end"]
    with pytest.raises(ValueError, match="unknown no-short calendar"):
        evaluate(D(2025, 6, 20), ("fomc",))


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
