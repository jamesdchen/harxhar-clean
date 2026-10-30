"""Unit tests of the macro release calendars (src/data/loading.py MACRO_CALENDARS, 2026-10-29): the construction and
its no-lookahead rules on synthetic calendars, the tables, the registry and the transform rules.  Fast (no panel load);
the tests on the real tables read data/releases.parquet and results/macro_calendar_2026-10-29/scheduled.csv when present.

Run:  python -m pytest writeup/tests/test_macro_calendar.py   (or: python writeup/tests/test_macro_calendar.py)
The frame-level gates (GM-F, GNL-1) are experiments/macro_calendar_check.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.loading as LD  # noqa: E402
import src.features.transforms.target as TG  # noqa: E402

RS = LD.MacroReschedule
OMB_CSV = ROOT / "results" / "macro_calendar_2026-10-29" / "scheduled.csv"
BAR = np.timedelta64(30, "m")


def grid(start: str, end: str) -> np.ndarray:
    """The calendar-open 30-minute grid of load_raw_data (the weekend cut) as datetime64[ns]."""
    g = pd.date_range(start, end, freq=LD.FREQ)
    dow, tod = g.dayofweek, g.time
    closed = (
        ((dow == 4) & (tod > pd.Timestamp(f"1900-01-01 {LD.FRIDAY_CLOSE}").time()))
        | (dow == 5)
        | ((dow == 6) & (tod < pd.Timestamp(f"1900-01-01 {LD.SUNDAY_OPEN}").time()))
    )
    return g[~closed].as_unit("ns").to_numpy()


def ts(s: str) -> np.datetime64:
    return np.datetime64(pd.Timestamp(s).as_unit("ns").to_datetime64())


def row(tv: np.ndarray, s: str) -> int:
    p = int(np.searchsorted(tv, ts(s)))
    assert tv[p] == ts(s), s
    return p


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


TV = grid("2021-02-01 00:00", "2021-04-30 23:30")


# ------------------------------------------------------------------------------------------- construction
def test_fomc_construction_and_release_bars():
    st = ["2021-02-10 09:00", "2021-03-10 09:00", "2021-04-13 09:00", "2021-05-12 09:00"]
    day, until, since = LD.macro_calendar_series(TV, st)
    for s in st[:3]:
        i = row(TV, s)
        assert day[i] == until[i] == since[i] == 1.0
        assert until[i - 1] == 0.5 and since[i + 1] == 0.5
    i0, i1 = row(TV, st[0]), row(TV, st[1])
    # FOMC's formulas
    i = np.arange(i0 + 1, i1 + 1)
    assert np.array_equal(until[i], 1.0 / (1.0 + (i1 - i).astype(np.float64)))
    j = i[:-1]  # the release bar i1 restarts since_inv
    assert np.array_equal(since[j], 1.0 / (1.0 + (j - i0).astype(np.float64)))
    assert np.isnan(since[:i0]).all() and not np.isnan(since[i0:]).any()
    # day: every bar of a release date, no other
    dd = TV.astype("datetime64[D]")
    rel_days = pd.to_datetime(st[:3]).normalize().as_unit("ns").to_numpy().astype("datetime64[D]")
    assert np.array_equal(day, np.isin(dd, rel_days).astype(np.float64))
    # the next release past the frame: the calendar-open distance (the grid extended)
    ext = grid("2021-02-01 00:00", "2021-05-12 09:00")
    assert until[-1] == 1.0 / (1.0 + float(len(ext) - len(TV)))
    assert not np.isnan(until).any()


def test_dead_tail_and_empty_calendar():
    st = ["2021-02-10 09:00", "2021-03-10 09:00"]
    day, until, since = LD.macro_calendar_series(TV, st)
    after = TV > ts(st[-1])
    assert np.isnan(day[after]).all() and np.isnan(until[after]).all() and np.isnan(since[after]).all()
    assert not np.isnan(day[~after]).any()
    d, u, s = LD.macro_calendar_series(TV, [])
    assert np.isnan(d).all() and np.isnan(u).all() and np.isnan(s).all()


def test_stamp_off_the_grid_raises():
    with pytest.raises(ValueError):
        LD.macro_calendar_series(TV, ["2021-02-13 09:00"])  # a Saturday: inside the frame, not a bar


# ------------------------------------------------------------------------------------------- the three states
ST3 = ["2021-02-10 09:00", "2021-03-24 09:00", "2021-04-13 09:00", "2021-05-12 09:00"]


def test_reschedule_three_states():
    rec = RS("2021-03-10 09:00", "2021-03-03 17:00", "2021-03-15 17:00", "2021-03-24 09:00")
    d0, u0, s0 = LD.macro_calendar_series(TV, ST3)
    day, until, since = LD.macro_calendar_series(TV, ST3, [rec])
    i = np.arange(len(TV))
    prev, act, sch = row(TV, ST3[0]), row(TV, ST3[1]), row(TV, "2021-03-10 09:00")
    before = (i > prev) & (TV < ts(rec.postponed_public))
    pending = (TV >= ts(rec.postponed_public)) & (TV < ts(rec.new_date_public))
    after = (TV >= ts(rec.new_date_public)) & (i <= act)
    assert np.array_equal(until[before], 1.0 / (1.0 + (sch - i[before]).astype(np.float64)))  # the old date
    assert np.isnan(until[pending]).all()  # no date
    assert np.array_equal(until[after], u0[after])  # the new date
    rest = ~(before | pending)
    assert same(until[rest], u0[rest])
    assert same(since, s0)  # from the last actual release
    assert same(day, d0)  # the actual date (notices before the days they concern)
    assert day[row(TV, "2021-03-10 08:00")] == 0.0 and day[row(TV, "2021-03-24 08:00")] == 1.0


def test_notice_on_the_due_bar_at_the_latest():
    # postponement "public" after the scheduled bar: it is known on the bar the release was due
    rec = RS("2021-03-10 09:00", "2021-03-12 12:00", "2021-03-15 17:00", "2021-03-24 09:00")
    day, until, _ = LD.macro_calendar_series(TV, ST3, [rec])
    i_s = row(TV, "2021-03-10 09:00")
    assert until[i_s - 1] == 0.5 and np.isnan(until[i_s])
    assert day[i_s - 1] == 1.0 and day[i_s] == 0.0  # the schedule said today until the bar showed otherwise


def test_simultaneous_notices_no_pending():
    rec = RS("2021-03-10 09:00", "2021-03-03 17:00", "2021-03-03 17:00", "2021-03-24 09:00")
    _, until, _ = LD.macro_calendar_series(TV, ST3, [rec])
    assert not np.isnan(until).any()
    i = row(TV, "2021-03-03 17:00")
    assert until[i] == 1.0 / (1.0 + float(row(TV, ST3[1]) - i))


def test_record_validation():
    with pytest.raises(ValueError):  # new date announced before the postponement
        LD.macro_calendar_series(TV, ST3, [RS("2021-03-10 09:00", "2021-03-05 17:00", "2021-03-04 17:00", ST3[1])])
    with pytest.raises(ValueError):  # announced after the release
        LD.macro_calendar_series(TV, ST3, [RS("2021-03-10 09:00", "2021-03-05 17:00", "2021-03-25 17:00", ST3[1])])
    rec = RS("2021-03-10 09:00", "2021-03-05 17:00", "2021-03-15 17:00", "2021-03-25 09:00")  # not a release
    assert all(same(a, b) for a, b in zip(LD.macro_calendar_series(TV, ST3, [rec]), LD.macro_calendar_series(TV, ST3)))


# ------------------------------------------------------------------------------------------- no lookahead
LONG = grid("2020-01-01 00:00", "2021-12-31 23:30")
S09 = (pd.bdate_range("2020-01-01", "2022-06-30") + pd.Timedelta(hours=9)).as_unit("ns").to_numpy()


def random_calendar(rng: np.random.Generator):
    """Monthly 09:00 releases on random weekdays, some moved (a record each), one past the frame."""
    months = pd.date_range("2020-01-01", "2022-02-01", freq="MS")
    picks = []
    for m in months:
        c = S09[(S09 >= np.datetime64(m)) & (S09 < np.datetime64(m + pd.offsets.MonthBegin(1)))]
        picks.append(c[int(rng.integers(3, len(c) - 3))])
    st = np.array(sorted(picks))
    recs = []
    for k in rng.choice(np.arange(2, len(st) - 3), size=4, replace=False):
        a, prev = st[k], st[k - 1]
        o = S09[(S09 > prev) & (S09 < a - np.timedelta64(2, "D"))]
        if not o.size:
            continue
        o = o[int(rng.integers(0, o.size))]
        p = prev + np.timedelta64(int(rng.integers(60, int((o - prev) / np.timedelta64(1, "m")))), "m")
        n = p + np.timedelta64(int(rng.integers(0, int((a - p) / np.timedelta64(1, "m")) - 24 * 60)), "m")
        recs.append(RS(str(pd.Timestamp(o)), str(pd.Timestamp(p)), str(pd.Timestamp(n)), str(pd.Timestamp(a))))
    return st, recs


def known(rec, a, x):
    if rec is None:
        return a
    if x < min(ts(rec.postponed_public), ts(rec.scheduled)):
        return ts(rec.scheduled)
    return None if x < ts(rec.new_date_public) else a


@pytest.mark.parametrize("seed", range(12))
def test_no_lookahead_random_calendars(seed):
    rng = np.random.default_rng(seed)
    st, recs = random_calendar(rng)
    by = {ts(r.actual): r for r in recs}
    base = LD.macro_calendar_series(LONG, st, recs)
    cuts = np.sort(rng.choice(len(LONG), size=60, replace=False))
    # every bar around each moved release's notices
    for r in recs:
        for x in (r.postponed_public, r.new_date_public, r.scheduled):
            j = int(np.searchsorted(LONG, ts(x)))
            cuts = np.append(cuts, np.clip(np.arange(j - 2, j + 3), 0, len(LONG) - 1))
    for i_t in np.unique(cuts):
        t = LONG[i_t]
        k0 = int(np.searchsorted(st, t))
        a0 = st[k0]
        r0 = by.get(a0)
        hi = st[k0 + 1]
        # (1) every release after the next one moved, deleted or added
        later = st[k0 + 1 :]
        later = later[rng.random(later.size) > 0.3] + np.timedelta64(int(rng.integers(0, 5)), "D")
        later = np.concatenate([later, S09[(S09 > a0)][rng.integers(0, 100, size=2)]])
        st1 = np.unique(np.concatenate([st[: k0 + 1], later[later > a0]]))
        st1 = st1[np.isin(st1, S09)]
        recs1 = [r for r in recs if ts(r.actual) <= a0]
        pert = LD.macro_calendar_series(LONG, st1, recs1)
        assert all(same(a[: i_t + 1], b[: i_t + 1]) for a, b in zip(base, pert)), ("later", str(t))
        # (2) the next release's schedule changed after t: postponed (no record yet) / a pending one re-dated
        state = known(r0, a0, t)
        o0 = ts(r0.scheduled) if r0 is not None else a0
        cand = S09[(S09 > t) & (S09 < hi) & (S09 != a0) & (S09 != o0)]
        if not cand.size or a0 == t:
            continue  # the release is at t: known
        a1 = cand[int(rng.integers(0, cand.size))]
        if state is not None:
            if r0 is not None and t >= ts(r0.new_date_public):
                continue  # a second postponement is not expressible by one record
            p = t + BAR * int(rng.integers(1, 4))
            if p > state or p > a1:
                continue
            new = RS(str(pd.Timestamp(state)), str(pd.Timestamp(p)), str(pd.Timestamp(p)), str(pd.Timestamp(a1)))
        else:
            nn = t + BAR * int(rng.integers(1, 4))
            if nn > a1:
                continue
            new = RS(r0.scheduled, r0.postponed_public, str(pd.Timestamp(nn)), str(pd.Timestamp(a1)))
        st2 = st.copy()
        st2[k0] = a1
        recs2 = [r for r in recs if ts(r.actual) != a0] + [new]
        pert = LD.macro_calendar_series(LONG, np.sort(st2), recs2)
        assert all(same(a[: i_t + 1], b[: i_t + 1]) for a, b in zip(base, pert)), ("next", str(t), new)


# ------------------------------------------------------------------------------------------- frame entry point
def test_add_macro_features_frame_and_flags(monkeypatch):
    tv = TV
    df = pd.DataFrame({"t": pd.to_datetime(tv)})
    for col in LD.MACRO_RELEASES.values():
        df[col] = 0.0
    df.loc[df["t"].isin(pd.to_datetime(["2021-02-10 09:00", "2021-03-24 09:00"])), "cpi release"] = 1.0
    rec = RS("2021-03-10 09:00", "2021-03-03 17:00", "2021-03-15 17:00", "2021-03-24 09:00")
    monkeypatch.setitem(LD.MACRO_RESCHEDULED, "cpi", (rec,))
    monkeypatch.setitem(LD.MACRO_MISSING_RELEASES, "cpi", ("2021-04-13 09:00",))
    monkeypatch.setitem(LD.MACRO_OMITTED_RELEASES, "cpi", ())
    monkeypatch.setitem(LD.MACRO_MISDATED_RELEASES, "cpi", ())
    monkeypatch.setitem(LD.MACRO_NEXT_RELEASE, "cpi", "2021-05-12 09:00")
    raw = df["cpi release"].copy()
    names = LD.add_macro_features(df, releases=["cpi"])
    assert names == LD.MACRO_SERIES_OF["cpi"] and df["cpi release"].equals(raw)
    want = LD.macro_calendar_series(tv, ST3, [rec])
    assert all(same(df[n].to_numpy(), w) for n, w in zip(names, want))
    monkeypatch.setattr(LD, "MACRO_SCHEDULE_FIX", False)
    assert LD.macro_frame_calendar(df, "cpi")[1] == ()
    monkeypatch.setattr(LD, "MACRO_CALENDAR_FIX", False)
    assert list(LD.macro_frame_calendar(df, "cpi")[0]) == list(pd.to_datetime(["2021-02-10 09:00", "2021-03-24 09:00"]))


# ------------------------------------------------------------------------------------------- registry, transforms
def test_registry():
    mac = LD.MACRO_FEATURES
    assert len(mac) == 18 == len(set(mac))
    assert mac == [
        f"mac_{r}_{s}" for r in ("cpi", "emp", "ppi", "gdp", "trade", "retail") for s in LD.MACRO_SERIES_SUFFIXES
    ]
    assert LD.MACRO_SERIES_SUFFIXES == ("day", "until_inv", "since_inv")
    assert LD.SUBGROUPS["macro"] == mac and LD.SUBGROUPS["all_features_mac"] == list(LD.ALL_FEATURES) + mac
    assert len(LD.ALL_FEATURES) == 46 and not set(mac) & set(LD.ALL_FEATURES)
    assert set(mac) <= LD.NON_NEGATIVE_SERIES
    assert TG.MACRO_RELEASE_KEYS == tuple(LD.MACRO_RELEASES)
    assert TG.MACRO_FLAG_SERIES == {s for s in mac if s.endswith("_day")}
    assert TG.MACRO_PROXIMITY_SERIES == {s for s in mac if not s.endswith("_day")}
    assert all(TG.is_diurnal_excluded(s) for s in mac)
    others = {c for g, v in LD.SUBGROUPS.items() if g not in ("macro", "all_features_mac") for c in v}
    assert not any("mac_" in c for c in others)
    assert LD.MACRO_CALENDARS is False  # default off


def test_transform_rules():
    n = 2000
    tv = TV[:n]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"t": pd.to_datetime(tv), "time_of_day": pd.to_datetime(tv).time})
    df["mac_cpi_day"] = (rng.random(n) < 0.05).astype(float)
    u = 1.0 / (1.0 + rng.integers(0, 500, size=n))
    u[::7] = 1.0  # the ramp's extremes: a winsorizer would bind
    df["mac_cpi_until_inv"] = u
    a, b = TG.robust_transform(df, "mac_cpi_day", use_diurnal=True, winsor_window=240)
    assert np.array_equal(a.to_numpy(), df["mac_cpi_day"].to_numpy()) and (b == 1.0).all()
    a, b = TG.robust_transform(df, "mac_cpi_until_inv", use_diurnal=True, winsor_window=240)
    assert np.array_equal(a.to_numpy(), np.log(df["mac_cpi_until_inv"].clip(lower=1e-12)).to_numpy())
    assert (b == 1.0).all()


# ------------------------------------------------------------------------------------------- the real tables
ALL_RECS = [(r, x) for r, v in LD.MACRO_RESCHEDULED.items() for x in v]


def is09(s: str) -> bool:
    t = pd.Timestamp(s)
    return t.strftime("%H:%M") == "09:00" and t.dayofweek < 5


@pytest.mark.parametrize("r,rec", ALL_RECS, ids=[f"{r}-{x.scheduled[:10]}" for r, x in ALL_RECS])
def test_reschedule_records(r, rec):
    o, a = pd.Timestamp(rec.scheduled), pd.Timestamp(rec.actual)
    p, n = pd.Timestamp(rec.postponed_public), pd.Timestamp(rec.new_date_public)
    assert is09(rec.scheduled) and is09(rec.actual) and o != a
    assert p <= n <= a
    assert p < o.normalize() and n < a.normalize()  # every notice before the day it concerns
    if (ROOT / "data" / "releases.parquet").is_file():
        st = LD.macro_release_stamps(r, include_next=True)
        assert a in st and o not in st


def test_table_stamps_and_counts():
    assert len(ALL_RECS) == 27
    for r in LD.MACRO_RELEASES:
        stamps = [*LD.MACRO_MISSING_RELEASES[r], *LD.MACRO_OMITTED_RELEASES[r], LD.MACRO_NEXT_RELEASE[r]]
        stamps += [x for pair in LD.MACRO_MISDATED_RELEASES[r] for x in pair]
        assert all(is09(s) for s in stamps), r
        assert pd.Timestamp(LD.MACRO_NEXT_RELEASE[r]) > pd.Timestamp("2024-04-30 23:30")
        acts = [x.actual for x in LD.MACRO_RESCHEDULED[r]]
        assert len(set(acts)) == len(acts)


@pytest.mark.skipif(
    not OMB_CSV.is_file() or not (ROOT / "data" / "releases.parquet").is_file(),
    reason="data files absent",
)
@pytest.mark.parametrize("r", list(LD.MACRO_RELEASES))
def test_calendar_equals_omb_schedule(r):
    s = pd.read_csv(OMB_CSV)
    s = s[s["series"] == {"emp": "EMP"}.get(r, r.upper())]
    if r == "gdp":
        s = s[s["month"].isin([1, 4, 7, 10])]
    omb = pd.DatetimeIndex(pd.to_datetime(dict(year=s["year"], month=s["month"], day=s["day"])) + pd.Timedelta(hours=9))
    sched = pd.to_datetime([x.scheduled for x in LD.MACRO_RESCHEDULED[r]])
    act = pd.to_datetime([x.actual for x in LD.MACRO_RESCHEDULED[r]])
    nxt = pd.Timestamp(LD.MACRO_NEXT_RELEASE[r])
    want = omb[~omb.isin(sched)].append(act)
    want = want[(want >= "2002-01-01") & (want <= nxt)].sort_values()
    got = LD.macro_release_stamps(r, include_next=True)
    got = got[(got >= "2002-01-01") & (got <= nxt)]
    assert list(got) == list(want)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
