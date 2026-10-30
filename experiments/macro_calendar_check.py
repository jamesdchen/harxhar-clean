"""Gates of the macro release calendars (src/data/loading.py MACRO_CALENDARS; 2026-10-29, author: "treat it similarly
to fomc? There needs to be no lookahead bias.").  Read-only, any machine; writes two json records.

GM-F (frame).  load_raw_data with MACRO_CALENDARS off vs on, on the calendar-open grid (drop_dead_session=False, where
    the distances are counted) and on the panel (the default dead-session drop): the same rows and bar ends; the on
    frame's columns are the off frame's, in their order, plus exactly the 18 mac_ series; every existing column (the
    FOMC series and the raw release columns included) bit for bit (NaN = NaN, same dtype).  Then, per release: every
    release bar (since_inv == 1) ends at 09:00 and the set of them is macro_release_stamps(r) on the frame (the file's
    flags plus the added releases, listed); day = until_inv = since_inv = 1 on each; day equals the flag of the actual
    release dates (FOMC's construction) on every row; until_inv is NaN exactly on the unknown-date windows of the
    MACRO_RESCHEDULED records (and since_inv only before the frame's first release); the last rows count to
    MACRO_NEXT_RELEASE; macro_calendar_series on macro_frame_calendar reproduces the columns bit for bit.  Registry
    checks (names, ALL_FEATURES untouched, no existing name holds the diurnal stem "mac_", the transform sets).  With
    --transform: load_and_transform on SUBGROUPS["macro"] (the design's arguments): the day flags pass through, the
    proximity series are exactly log(x) where observed (no winsorizing, no diurnal step), the availability flag is the
    observed flag, and the design column names (adj_<s>_ma_<k>, <s>_avail_ma_<k>); and the FOMC trio's transformed
    columns bit for bit with the flag off vs on.

GNL-1 (no lookahead, on the frame's calendar-open grid).  Information set at bar t (bar end): the actual releases at or
    before t, and the next release as scheduled at t (its scheduled date, or "postponed, date unknown", or its new
    date, from the notices public by t).  Cut bars: every bar of the 2013 (2013-09-23 .. 2013-11-30) and 2018-19
    (2018-12-15 .. 2019-05-15) shutdown windows and of the 2020 FOMC window (2020-01-29 .. 2020-04-29), +-3 bars
    around every 2014-01-02 .. 2024-04-30 release of the six series, and --n-random random bars.  Perturbations that
    leave the information set at t unchanged, each evaluated on the whole frame by macro_calendar_series (the
    construction add_macro_features runs; GM-F ties the two):
      P1 later releases: every release after the next one moved / deleted / added (also past the frame), their
         records dropped or kept -- valid for every t up to the next release's bar;
      P2 the next release's notices and actual date: a postponement added after t (also cancelling it into the
         release after), a postponed release's actual date and new-date notice changed while the schedule known at t
         is unchanged, the notice times moved -- valid for every t before the first time the two worlds' schedules
         differ (computed from the two records);
      P2-tight: per cut, the notice public on the very next bar (a postponement, or the new date of a postponed one).
    Pass: every mac_ value at the bars <= t identical (all three series of the perturbed release; the other releases'
    series do not read its calendar).  Reported per cut bar and series: covered by P1 / P2 and failures.
    GNL-1b (real-time recomputation): for a subsample of the cut bars, the series rebuilt on the frame cut at t from the
    information set alone (the past releases, their records, the next release as scheduled at t; a postponed one with
    an arbitrary later date and notice) equal the full frame's at every bar from the window's first release to t.

usage: python experiments/macro_calendar_check.py [--only gmf|gnl1] [--transform] [--n-random 2000] [--seed 20261029]
       [--out results/macro_calendar_2026-10-29]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import src.data.loading as LD  # noqa: E402
import src.features.transforms.target as TG  # noqa: E402

DAY = np.timedelta64(1, "D")
BAR = np.timedelta64(30, "m")
NAT = np.datetime64("NaT", "ns")


def eq_cols(a: pd.Series, b: pd.Series) -> bool:
    if a.dtype != b.dtype:
        return False
    x, y = a.to_numpy(), b.to_numpy()
    try:
        return bool(((x == y) | (pd.isna(x) & pd.isna(y))).all())
    except TypeError:
        return bool((a.astype(str) == b.astype(str)).all())


def same_arr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x == y) | (np.isnan(x) & np.isnan(y))


def ts(s) -> np.datetime64:
    return np.datetime64(pd.Timestamp(s).as_unit("ns").to_datetime64())


def load(flag: bool, dead: bool) -> pd.DataFrame:
    LD.MACRO_CALENDARS = flag
    try:
        return LD.load_raw_data("data", allow_missing=True, drop_dead_session=dead)
    finally:
        LD.MACRO_CALENDARS = False


# ======================================================================================================== GM-F
OMB_CSV = REPO / "results" / "macro_calendar_2026-10-29" / "scheduled.csv"
OMB_SERIES = {"cpi": "CPI", "emp": "EMP", "ppi": "PPI", "gdp": "GDP", "trade": "TRADE", "retail": "RETAIL"}


def omb_scheduled(r: str) -> pd.DatetimeIndex:
    """OMB's annual schedule of release r, 2002-2024, at 09:00 (GDP: the advance estimates, Jan / Apr / Jul / Oct)."""
    s = pd.read_csv(OMB_CSV)
    s = s[s["series"] == OMB_SERIES[r]]
    if r == "gdp":
        s = s[s["month"].isin([1, 4, 7, 10])]
    d = pd.to_datetime(dict(year=s["year"], month=s["month"], day=s["day"]))
    return pd.DatetimeIndex(d + pd.Timedelta(hours=9)).sort_values()


def omb_check(r: str) -> dict:
    """The fixed calendar 2002-01-01 .. its next release == OMB's schedule less the moved dates plus their actual dates;
    the records: scheduled on OMB's schedule, actual not, notices in order, every notice before the day it concerns."""
    omb = omb_scheduled(r)
    recs = LD.MACRO_RESCHEDULED[r]
    sched = pd.to_datetime([x.scheduled for x in recs])
    actual = pd.to_datetime([x.actual for x in recs])
    assert sched.isin(omb).all(), (r, [str(x) for x in sched[~sched.isin(omb)]])
    assert not actual.isin(omb).any(), (r, [str(x) for x in actual[actual.isin(omb)]])
    for x in recs:
        o, a = pd.Timestamp(x.scheduled), pd.Timestamp(x.actual)
        p, nn = pd.Timestamp(x.postponed_public), pd.Timestamp(x.new_date_public)
        assert p <= nn <= a and o != a, x
        assert p < o.normalize() and nn < a.normalize(), ("a notice on the day it concerns", x)
    nxt = pd.Timestamp(LD.MACRO_NEXT_RELEASE[r])
    want = omb[~omb.isin(sched)].append(actual)
    want = pd.DatetimeIndex(want[(want >= "2002-01-01") & (want <= nxt)]).sort_values()
    got = LD.macro_release_stamps(r, include_next=True)
    got = got[(got >= "2002-01-01") & (got <= nxt)]
    missing, extra = want[~want.isin(got)], got[~got.isin(want)]
    assert not len(missing) and not len(extra), (r, [str(x) for x in missing], [str(x) for x in extra])
    assert nxt == omb[omb > pd.Timestamp("2024-04-30 23:30")][0], r
    return dict(omb_dates_2002_next=int(len(want)), moved=len(recs), equal=True)


def unknown_rows(tv: np.ndarray, st: np.ndarray, recs) -> np.ndarray:
    """The rows whose next release is a record's actual, at or after its (effective) postponement, before its notice."""
    out = np.zeros(len(tv), dtype=bool)
    for rec in recs:
        a = ts(rec.actual)
        k = int(np.searchsorted(st, a))
        if k >= len(st) or st[k] != a:
            continue
        lo = st[k - 1] if k else np.datetime64("1900-01-01", "ns")
        post = min(ts(rec.postponed_public), ts(rec.scheduled))
        out |= (tv > lo) & (tv <= a) & (tv >= post) & (tv < ts(rec.new_date_public))
    return out


def gate_gmf(do_transform: bool) -> dict:
    rec: dict = {}
    # ---- registry
    mac = list(LD.MACRO_FEATURES)
    assert len(mac) == 18 and len(set(mac)) == 18, mac
    assert mac == [f"mac_{r}_{s}" for r in LD.MACRO_RELEASES for s in ("day", "until_inv", "since_inv")], mac
    assert list(LD.MACRO_RELEASES) == ["cpi", "emp", "ppi", "gdp", "trade", "retail"]
    assert LD.SUBGROUPS["macro"] == mac
    assert LD.SUBGROUPS["all_features_mac"] == list(LD.ALL_FEATURES) + mac
    assert len(LD.ALL_FEATURES) == 46 and not any("mac_" in f for f in LD.ALL_FEATURES)
    assert set(mac) <= LD.NON_NEGATIVE_SERIES and not set(mac) & (LD.SIGNED_SERIES | LD.STRICTLY_POSITIVE_SERIES)
    assert TG.MACRO_FLAG_SERIES | TG.MACRO_PROXIMITY_SERIES == set(mac)
    assert TG.MACRO_FLAG_SERIES == {s for s in mac if s.endswith("_day")}
    assert tuple(LD.MACRO_RELEASES) == TG.MACRO_RELEASE_KEYS
    assert all(TG.is_diurnal_excluded(s) for s in mac)
    rec["registry"] = dict(macro=mac, n_all_features=len(LD.ALL_FEATURES), n_all_features_mac=64)

    # ---- frames off vs on
    t0 = time.time()
    frames = {(dead, flag): load(flag, dead) for dead in (False, True) for flag in (False, True)}
    rec["load_seconds"] = round(time.time() - t0, 1)
    existing = set(frames[(False, False)].columns)
    other_names = {c for g, v in LD.SUBGROUPS.items() if g not in ("macro", "all_features_mac") for c in v}
    hit = sorted(c for c in existing | other_names if "mac_" in str(c).lower())
    assert not hit, f"an existing name holds the diurnal stem mac_: {hit}"
    for dead in (False, True):
        off, on = frames[(dead, False)], frames[(dead, True)]
        key = "panel" if dead else "calendar_open"
        assert len(off) == len(on) and pd.to_datetime(off["t"]).equals(pd.to_datetime(on["t"])), key
        assert [c for c in on.columns if c not in mac] == list(off.columns), key
        assert sorted(set(on.columns) - set(off.columns)) == sorted(mac), key
        bad = [c for c in off.columns if not eq_cols(off[c], on[c])]
        assert not bad, (key, bad)
        rec[f"frame_{key}"] = dict(
            rows=len(off),
            existing_columns=len(off.columns),
            identical_existing_columns=len(off.columns),
            added=[c for c in on.columns if c in mac],
            fomc_columns_checked=[c for c in off.columns if str(c).startswith("fomc")],
        )
    print("GM-F frames: off vs on identical on every existing column", flush=True)

    # ---- per release on the calendar-open frame
    on = frames[(False, True)]
    tv = pd.DatetimeIndex(pd.to_datetime(on["t"])).as_unit("ns").to_numpy()
    n = len(tv)
    hhmm = pd.DatetimeIndex(tv).strftime("%H:%M").to_numpy()
    span = (tv >= ts("2014-01-02")) & (tv < ts("2024-05-01"))
    panel = frames[(True, True)]
    ptv = pd.DatetimeIndex(pd.to_datetime(panel["t"])).as_unit("ns").to_numpy()
    pm = (pd.DatetimeIndex(ptv).hour * 60 + pd.DatetimeIndex(ptv).minute).to_numpy()
    p_rth = (pm >= 600) & (pm <= 960) & (ptv >= ts("2014-01-02")) & (ptv < ts("2024-05-01"))
    per: dict = {}
    for r, col in LD.MACRO_RELEASES.items():
        d_, u_, s_ = (on[c].to_numpy() for c in LD.MACRO_SERIES_OF[r])
        st_idx, recs = LD.macro_frame_calendar(on, r)
        st = st_idx.as_unit("ns").to_numpy()
        # the pure construction reproduces the frame
        d2, u2, s2 = LD.macro_calendar_series(tv, st, recs)
        assert same_arr(d_, d2).all() and same_arr(u_, u2).all() and same_arr(s_, s2).all(), r
        # release bars
        rel = np.flatnonzero(s_ == 1.0)
        assert (hhmm[rel] == "09:00").all(), (r, sorted(set(hhmm[rel])))
        want = LD.macro_release_stamps(r).as_unit("ns").to_numpy()
        want = want[(want >= tv[0]) & (want <= tv[-1])]
        assert np.array_equal(tv[rel], want), (r, len(rel), len(want))
        assert (d_[rel] == 1.0).all() and (u_[rel] == 1.0).all(), r
        file_flag = np.nan_to_num(pd.to_numeric(on[col], errors="coerce").to_numpy(), nan=0.0) > 0
        misdated = np.isin(tv, pd.to_datetime([w for w, _ in LD.MACRO_MISDATED_RELEASES[r]]).as_unit("ns").to_numpy())
        assert set(np.flatnonzero(file_flag & ~misdated)) <= set(rel), r
        assert not set(np.flatnonzero(misdated)) & set(rel), r
        added = [str(pd.Timestamp(x)) for x in tv[rel[~file_flag[rel]]]]
        # day = the actual-date flag (FOMC's construction)
        dd = tv.astype("datetime64[D]")
        actual_day = np.isin(dd, tv[rel].astype("datetime64[D]")).astype(np.float64)
        assert np.array_equal(d_, actual_day), (r, int((d_ != actual_day).sum()))
        # NaN pattern
        unk = unknown_rows(tv, st, recs)
        assert np.array_equal(np.isnan(u_), unk), (r, int(np.isnan(u_).sum()), int(unk.sum()))
        first = rel[0]
        assert np.array_equal(np.flatnonzero(np.isnan(s_)), np.arange(first)), r
        assert not np.isnan(d_).any(), r
        # the last rows count to the next release past the frame
        nxt = ts(LD.MACRO_NEXT_RELEASE[r])
        k_last = n - 1 + LD._calendar_open_bars_between(pd.Timestamp(tv[-1]), pd.Timestamp(nxt))
        assert u_[-1] == 1.0 / (1.0 + float(k_last - (n - 1))), r
        # the post-drop panel
        pd_, pu_, ps_ = (panel[c].to_numpy() for c in LD.MACRO_SERIES_OF[r])
        prel = np.flatnonzero(ps_ == 1.0)
        assert (pd_[prel] == 1.0).all() and (pu_[prel] == 1.0).all()
        dropped = sorted(set(map(str, pd.DatetimeIndex(tv[rel]))) - set(map(str, pd.DatetimeIndex(ptv[prel]))))
        recs_info = []
        for x in recs:
            a = ts(x.actual)
            rows = unk & (tv <= a) & (tv > (st[np.searchsorted(st, a) - 1]))
            recs_info.append(
                dict(
                    scheduled=x.scheduled,
                    actual=x.actual,
                    postponed_public=x.postponed_public,
                    new_date_public=x.new_date_public,
                    unknown_bars=int(rows.sum()),
                    unknown_first=str(pd.Timestamp(tv[rows][0])) if rows.any() else None,
                    unknown_last=str(pd.Timestamp(tv[rows][-1])) if rows.any() else None,
                    todo="TODO" in x.note,
                )
            )
        per[r] = dict(
            releases_in_frame=len(rel),
            releases_2014_2024=int(span[rel].sum()),
            file_flags_in_frame=int(file_flag.sum()),
            file_flags_dropped_misdated=[str(pd.Timestamp(x)) for x in tv[misdated & file_flag]],
            added_releases=added,
            omb_check=omb_check(r),
            release_bars_not_in_panel=dropped,
            until_nan_bars=int(np.isnan(u_).sum()),
            until_nan_bars_2014_2024=int(np.isnan(u_[span]).sum()),
            until_nan_rth_rows_2014_2024_panel=int(np.isnan(pu_[p_rth]).sum()),
            since_nan_bars_before_first_release=int(first),
            next_release=LD.MACRO_NEXT_RELEASE[r],
            last_row_until_inv=float(u_[-1]),
            reschedules=recs_info,
        )
        print(
            f"GM-F {r}: {len(rel)} releases in frame (+{len(added)} added), until NaN {int(np.isnan(u_).sum())} bars,"
            f" not in panel {len(dropped)}",
            flush=True,
        )
    rec["per_release"] = per

    if do_transform:
        rec["transform"] = gate_transform()
    return rec


def gate_transform() -> dict:
    import src.backtest.executor as EX

    kw: dict[str, Any] = dict(
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    out: dict = {}
    mac = list(LD.SUBGROUPS["macro"])
    LD.MACRO_CALENDARS = True
    try:
        df, cols = EX.load_and_transform("data", mac, **kw)
    finally:
        LD.MACRO_CALENDARS = False
    for s in mac:
        obs = df[f"{s}__obs"].to_numpy(bool)
        raw, adj = df[s].to_numpy(), df[f"adj_{s}"].to_numpy()
        assert np.array_equal(df[f"{s}_avail"].to_numpy(), obs.astype(np.float64)), s
        if s.endswith("_day"):
            assert np.array_equal(adj[obs], raw[obs]), s
        else:
            assert np.array_equal(adj[obs], np.log(np.clip(raw[obs], 1e-12, None))), s
    har = EX._DeferredHar()
    _, names = har(df, target_col="adj_RV", exog_cols=cols)
    mac_names = [c for c in names if "mac_" in c]
    out["adj_exog_cols"] = cols
    out["har_design_names_mac"] = len(mac_names)
    out["har_design_names_example"] = mac_names[:12] + [c for c in mac_names if "_active_ma_" in c][:6]
    out["active_series"] = sorted({c[: -len("_active")] for c in cols if c.endswith("_active")})
    # the FOMC trio's transformed columns: flag off vs on
    fomc = [c for c in LD.FOMC_FEATURES if c in LD.ALL_FEATURES]
    fr = {}
    for flag in (False, True):
        LD.MACRO_CALENDARS = flag
        try:
            fr[flag], _ = EX.load_and_transform("data", fomc, **kw)
        finally:
            LD.MACRO_CALENDARS = False
    a, b = fr[False], fr[True]
    assert [c for c in b.columns if c not in mac] == list(a.columns)
    bad = [c for c in a.columns if not eq_cols(a[c], b[c])]
    assert not bad, bad
    out["fomc_trio_flag_off_vs_on_identical_columns"] = len(a.columns)
    print(f"GM-F transform: {len(cols)} transformed columns, {len(mac_names)} macro design names", flush=True)
    return out


# ======================================================================================================== GNL-1
class Grid:
    """The calendar-open frame and the 09:00 weekday stamps (also past the frame) a release can be put on."""

    def __init__(self, tv: np.ndarray):
        self.tv = tv
        self.n = len(tv)
        ext = pd.bdate_range(pd.Timestamp(tv[0]).normalize(), pd.Timestamp(tv[-1]) + pd.Timedelta(days=500))
        self.s09 = (ext + pd.Timedelta(hours=9)).as_unit("ns").to_numpy()

    def row(self, x: np.datetime64) -> int:
        """Row of a stamp (rows past the frame: n + the calendar-open bars past the last bar - 1)."""
        p = int(np.searchsorted(self.tv, x))
        if p < self.n and self.tv[p] == x:
            return p
        if x > self.tv[-1]:
            return self.n - 1 + LD._calendar_open_bars_between(pd.Timestamp(self.tv[-1]), pd.Timestamp(x))
        raise ValueError(x)

    def draw09(self, rng: np.random.Generator, lo: np.datetime64, hi: np.datetime64, k: int = 1) -> np.ndarray:
        """k distinct 09:00 weekday stamps in (lo, hi) (fewer if there are not k)."""
        cand = self.s09[(self.s09 > lo) & (self.s09 < hi)]
        if not cand.size:
            return cand
        return np.sort(rng.choice(cand, size=min(k, cand.size), replace=False))

    def draw_time(self, rng: np.random.Generator, lo: np.datetime64, hi: np.datetime64) -> np.datetime64:
        """A random minute in (lo, hi] (hi if the interval is empty)."""
        span = int((hi - lo) / np.timedelta64(1, "m"))
        if span <= 0:
            return hi
        return lo + np.timedelta64(int(rng.integers(1, span + 1)), "m")


def state(rec, a: np.datetime64, x: np.datetime64):
    """Release k's schedule as known at time x: its date, or 'unknown' (None)."""
    if rec is None:
        return a
    post = min(ts(rec.postponed_public), ts(rec.scheduled))
    if x < post:
        return ts(rec.scheduled)
    if x < ts(rec.new_date_public):
        return None
    return ts(rec.actual)


def divergence(rec0, a0, rec1, a1, lo: np.datetime64) -> np.datetime64:
    """The first time after lo at which the two worlds' schedules of the next release differ (NaT: never)."""
    pts = [lo + np.timedelta64(1, "m")]
    for rc in (rec0, rec1):
        if rc is not None:
            pts += [min(ts(rc.postponed_public), ts(rc.scheduled)), ts(rc.new_date_public)]
    pts += [a0, a1]
    for x in sorted(p for p in pts if p > lo):
        if state(rec0, a0, x) != state(rec1, a1, x):
            return x
    return NAT


def recs_by_actual(recs) -> dict:
    return {ts(x.actual): x for x in recs}


def first_diff(base: tuple, pert: tuple) -> int:
    """First row at which any of the three series differ (n if none)."""
    n = len(base[0])
    m = np.ones(n, dtype=bool)
    for b, p in zip(base, pert):
        m &= same_arr(b, p)
    bad = np.flatnonzero(~m)
    return int(bad[0]) if bad.size else n


def p1_later(g: Grid, rng, st: np.ndarray, recs, k0: int):
    """Releases after k0 moved / deleted / added; records of moved or deleted ones dropped."""
    a0 = st[k0]
    later = st[k0 + 1 :]
    keep = rng.random(later.size) > 0.3
    kept = later[keep]
    move = rng.random(kept.size) < 0.5
    moved = []
    for x in kept[move]:
        c = g.draw09(rng, max(a0, x - 30 * DAY), x + 30 * DAY, 1)
        if c.size:
            moved.append(c[0])
    unmoved = kept[~move]
    add = g.draw09(rng, a0, a0 + 400 * DAY, int(rng.integers(0, 4)))
    new_later = np.unique(np.concatenate([unmoved, np.array(moved, dtype="datetime64[ns]"), add]))
    new_later = new_later[new_later > a0]
    stamps = np.concatenate([st[: k0 + 1], new_later])
    um = set(unmoved.astype(np.int64).tolist())
    new_recs = [x for x in recs if ts(x.actual) <= a0 or int(ts(x.actual).astype(np.int64)) in um]
    return stamps, tuple(new_recs)


def p2_variants(g: Grid, rng, st: np.ndarray, recs, k0: int, tight_after: np.datetime64 | None = None):
    """Perturbations of release k0's record and date (the releases after it unchanged, kept after it).  Yields
    (stamps, recs, divergence time, label).  ``tight_after``: the notice public on the first bar after it."""
    a0 = st[k0]
    lo = st[k0 - 1] if k0 else g.tv[0] - DAY
    hi = st[k0 + 1] if k0 + 1 < st.size else a0 + 120 * DAY
    by = recs_by_actual(recs)
    r0 = by.get(a0)
    others = [x for x in recs if ts(x.actual) != a0]
    o = ts(r0.scheduled) if r0 is not None else a0
    out = []

    def make(sched, p_pub, n_pub, actual, label):
        if not (p_pub <= n_pub <= actual) or sched == actual or actual >= hi or actual <= lo:
            return
        rec1 = LD.MacroReschedule(
            str(pd.Timestamp(sched)), str(pd.Timestamp(p_pub)), str(pd.Timestamp(n_pub)), str(pd.Timestamp(actual))
        )
        stamps = st.copy()
        stamps[k0] = actual
        div = divergence(r0, a0, rec1, actual, lo)
        out.append((np.sort(stamps), tuple(others + [rec1]), div, label))

    def new_date(after: np.datetime64):
        c = g.draw09(rng, max(after, lo), hi, 1)
        return c[0] if c.size else None

    if tight_after is not None:
        nb = tight_after + BAR
        st_t = state(r0, a0, tight_after)
        if st_t is not None and st_t > tight_after and nb <= st_t:
            # postponed on the next bar (the schedule known at t is st_t)
            a1 = new_date(max(nb, st_t))
            if a1 is not None:
                make(st_t, nb, g.draw_time(rng, nb, a1 - 9 * np.timedelta64(1, "h")), a1, "tight_postpone")
        elif st_t is None:
            # postponed, date unknown at t: the new date public on the next bar
            a1 = new_date(nb)
            if a1 is not None and r0 is not None:
                make(o, ts(r0.postponed_public), nb, a1, "tight_new_date")
        return out
    # (a) a postponement at the latest moment (the bar the release was due) and at a random earlier one
    if r0 is None:
        for p_pub in (a0, g.draw_time(rng, lo, a0)):
            a1 = new_date(a0)
            if a1 is not None:
                make(a0, p_pub, g.draw_time(rng, p_pub, a1 - 9 * np.timedelta64(1, "h")), a1, "postpone")
        # cancelled into the next release (the next one keeps its date)
        if k0 + 1 < st.size and ts(st[k0 + 1]) not in by:
            p_pub = g.draw_time(rng, lo, a0)
            rec1 = LD.MacroReschedule(
                str(pd.Timestamp(a0)), str(pd.Timestamp(p_pub)), str(pd.Timestamp(p_pub)), str(pd.Timestamp(st[k0 + 1]))
            )
            stamps = np.delete(st, k0)
            out.append((stamps, tuple(others + [rec1]), divergence(None, a0, rec1, st[k0 + 1], lo), "cancel"))
    else:
        p0, n0 = ts(r0.postponed_public), ts(r0.new_date_public)
        # the actual date (and its notice) changed, the earlier notices kept
        a1 = new_date(max(n0, o))
        if a1 is not None:
            make(o, p0, n0, a1, "actual_moved")
            make(o, p0, g.draw_time(rng, max(p0, lo), a1 - 9 * np.timedelta64(1, "h")), a1, "actual_and_notice_moved")
        # the notices moved
        p1 = g.draw_time(rng, lo, min(p0, o))
        make(o, p1, g.draw_time(rng, p1, a0 - 9 * np.timedelta64(1, "h")), a0, "notices_moved")
    return out


def cut_bars(g: Grid, sts: dict, n_random: int, rng) -> dict[str, np.ndarray]:
    tv = g.tv
    win = {
        "shutdown_2013": ("2013-09-23", "2013-12-01"),
        "shutdown_2018_19": ("2018-12-15", "2019-05-16"),
        "fomc_2020": ("2020-01-29", "2020-04-30"),
    }
    out = {k: np.flatnonzero((tv >= ts(a)) & (tv < ts(b))) for k, (a, b) in win.items()}
    near = []
    for r, st in sts.items():
        ins = st[(st >= ts("2014-01-02")) & (st < ts("2024-05-01")) & (st <= tv[-1])]
        rows = np.searchsorted(tv, ins)
        for d in range(-3, 4):
            near.append(rows + d)
    nr = np.unique(np.concatenate(near))
    out["release_pm3_2014_2024"] = nr[(nr >= 0) & (nr < g.n)]
    out["random"] = np.sort(rng.choice(g.n, size=n_random, replace=False))
    return out


def info_calendar(g: Grid, st: np.ndarray, recs, i_t: int):
    """The information set at bar i_t as a calendar: (stamps, records) of the past releases, their records and the next
    release as scheduled at t (a postponed one with an arbitrary later date and notice)."""
    t = g.tv[i_t]
    k0 = int(np.searchsorted(st, t))  # next release at or after t
    by = recs_by_actual(recs)
    past = st[:k0]
    precs = [by[x] for x in past if x in by]
    if k0 >= st.size:
        return past, tuple(precs), k0
    a0 = st[k0]
    r0 = by.get(a0)
    s = state(r0, a0, t)
    if a0 == t:
        return st[: k0 + 1], tuple(precs + ([r0] if r0 is not None else [])), k0
    if s is None:  # postponed, date unknown at t: any later date, its notice after t
        assert r0 is not None
        x = g.s09[np.searchsorted(g.s09, t + 45 * DAY)]
        nrec = LD.MacroReschedule(
            r0.scheduled, r0.postponed_public, str(pd.Timestamp(t + 7 * BAR)), str(pd.Timestamp(x))
        )
        return np.concatenate([past, [x]]), tuple(precs + [nrec]), k0
    if r0 is not None and t >= ts(r0.new_date_public):
        return np.concatenate([past, [a0]]), tuple(precs + [r0]), k0
    return np.concatenate([past, [s]]), tuple(precs), k0  # the scheduled date (no notice yet)


def gate_gnl1(n_random: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    LD.MACRO_CALENDARS = True
    try:
        frame = LD.load_raw_data("data", allow_missing=True, drop_dead_session=False)
    finally:
        LD.MACRO_CALENDARS = False
    tv = pd.DatetimeIndex(pd.to_datetime(frame["t"])).as_unit("ns").to_numpy()
    g = Grid(tv)
    cal, base = {}, {}
    for r in LD.MACRO_RELEASES:
        st, recs = LD.macro_frame_calendar(frame, r)
        st = st.as_unit("ns").to_numpy()
        cal[r] = (st, recs)
        base[r] = LD.macro_calendar_series(tv, st, recs)
        fr = tuple(frame[c].to_numpy() for c in LD.MACRO_SERIES_OF[r])
        assert all(same_arr(a, b).all() for a, b in zip(base[r], fr)), r  # the frame is the construction
    cuts = cut_bars(g, {r: c[0] for r, c in cal.items()}, n_random, rng)
    all_cuts = np.unique(np.concatenate(list(cuts.values())))
    print(f"GNL-1: {len(all_cuts)} cut bars ({ {k: len(v) for k, v in cuts.items()} }), frame {g.n} bars", flush=True)
    rec: dict = {"cut_bars": {k: int(len(v)) for k, v in cuts.items()}, "cut_bars_union": int(len(all_cuts))}
    failures: list = []
    per: dict = {}
    n_eval = 0
    for r, (st, recs) in cal.items():
        sp = np.array([g.row(x) for x in st], dtype=np.int64)
        k_of = np.searchsorted(sp, all_cuts, side="left")
        assert (k_of < st.size).all(), r
        ks = np.unique(k_of)
        p1_bound: dict[int, int] = {}  # k0 -> every row <= this is identical under every P1 draw
        p2: dict[int, list] = {}  # k0 -> [(divergence time, ok)]
        for k0 in ks:
            k0 = int(k0)
            okrow = sp[k0]
            for _ in range(2):
                stamps, nrecs = p1_later(g, rng, st, recs, k0)
                fd = first_diff(base[r], LD.macro_calendar_series(tv, stamps, nrecs))
                n_eval += 1
                if fd <= min(sp[k0], g.n - 1):
                    failures.append(dict(release=r, family="P1", k0=k0, first_diff=str(pd.Timestamp(tv[fd]))))
                okrow = min(okrow, fd - 1)
            p1_bound[k0] = okrow
            lst = []
            for stamps, nrecs, div, label in p2_variants(g, rng, st, recs, k0):
                if np.isnat(div):
                    continue
                fd = first_diff(base[r], LD.macro_calendar_series(tv, stamps, nrecs))
                n_eval += 1
                ok = fd >= g.n or tv[fd] >= div
                if not ok:
                    failures.append(
                        dict(
                            release=r,
                            family="P2",
                            label=label,
                            k0=k0,
                            divergence=str(pd.Timestamp(div)),
                            first_diff=str(pd.Timestamp(tv[fd])),
                        )
                    )
                lst.append((div, ok, label))
            p2[k0] = lst
        # tight variants: the notice on the bar after the cut (release windows: own +-3 bars; shutdown windows: every
        # 6th bar)
        tight_cuts = np.unique(
            np.concatenate(
                [
                    cuts["release_pm3_2014_2024"][
                        np.isin(
                            cuts["release_pm3_2014_2024"],
                            np.concatenate([np.searchsorted(tv, st[st <= tv[-1]]) + d for d in (-3, -2, -1)]),
                        )
                    ],
                    cuts["shutdown_2013"][::6],
                    cuts["shutdown_2018_19"][::6],
                ]
            )
        )
        tight_ok = tight_n = 0
        for i_t in tight_cuts:
            k0 = int(np.searchsorted(sp, i_t, side="left"))
            for stamps, nrecs, div, label in p2_variants(g, rng, st, recs, k0, tight_after=tv[i_t]):
                if np.isnat(div) or div <= tv[i_t]:
                    continue
                fd = first_diff(base[r], LD.macro_calendar_series(tv, stamps, nrecs))
                n_eval += 1
                tight_n += 1
                if fd <= i_t:
                    failures.append(
                        dict(
                            release=r,
                            family="P2-tight",
                            label=label,
                            cut=str(pd.Timestamp(tv[i_t])),
                            first_diff=str(pd.Timestamp(tv[fd])),
                        )
                    )
                else:
                    tight_ok += 1
        # per-cut coverage
        cov1 = np.array([p1_bound[int(k)] >= i for k, i in zip(k_of, all_cuts)])
        cov2 = np.array([any(ok and d > tv[i] for d, ok, _ in p2[int(k)]) for k, i in zip(k_of, all_cuts)])
        p2_appl = np.array([any(d > tv[i] for d, _, _ in p2[int(k)]) for k, i in zip(k_of, all_cuts)])
        per[r] = dict(
            next_releases_tested=int(len(ks)),
            cuts_covered_P1=int(cov1.sum()),
            cuts_with_a_P2_draw=int(p2_appl.sum()),
            cuts_covered_P2=int(cov2.sum()),
            tight_draws=tight_n,
            tight_pass=tight_ok,
            by_window={
                k: dict(
                    cuts=int(len(v)), P1=int(cov1[np.isin(all_cuts, v)].sum()), P2=int(cov2[np.isin(all_cuts, v)].sum())
                )
                for k, v in cuts.items()
            },
        )
        print(
            f"GNL-1 {r}: {per[r]['cuts_covered_P1']}/{len(all_cuts)} P1, {per[r]['cuts_covered_P2']} P2, "
            f"tight {tight_ok}/{tight_n}; failures so far {len(failures)}",
            flush=True,
        )

    # GNL-1b: real-time recomputation from the information set on the frame cut at t
    rt_cuts = np.unique(
        np.concatenate(
            [
                cuts["shutdown_2013"][::3],
                cuts["shutdown_2018_19"][::3],
                cuts["fomc_2020"][::12],
                cuts["release_pm3_2014_2024"][::4],
                cuts["random"][::4],
            ]
        )
    )
    rt_fail: list = []
    rt_n = 0
    for r, (st, recs) in cal.items():
        sp = np.array([g.row(x) for x in st], dtype=np.int64)
        for i_t in rt_cuts:
            stamps, irecs, k0 = info_calendar(g, st, recs, int(i_t))
            lo = int(sp[max(k0 - 3, 0)]) if k0 >= 1 else 0
            lo = max(min(lo, int(i_t)), 0)
            win = tv[lo : i_t + 1]
            got = LD.macro_calendar_series(win, stamps, irecs)
            first = int(np.searchsorted(win, stamps[stamps >= win[0]][0])) if (stamps >= win[0]).any() else len(win)
            rows = slice(first, len(win))
            ref = tuple(b[lo : i_t + 1] for b in base[r])
            rt_n += 1
            if not all(same_arr(a[rows], b[rows]).all() for a, b in zip(got, ref)):
                j = first + first_diff(tuple(x[rows] for x in ref), tuple(x[rows] for x in got))
                rt_fail.append(dict(release=r, cut=str(pd.Timestamp(tv[i_t])), first_diff=str(pd.Timestamp(win[j]))))
    rec["GNL1b_realtime"] = dict(
        cuts=int(len(rt_cuts)), evaluations=rt_n, failures=rt_fail[:50], n_failures=len(rt_fail)
    )
    print(f"GNL-1b: {rt_n} real-time recomputations, {len(rt_fail)} failures", flush=True)
    rec["per_release"] = per
    rec["evaluations"] = n_eval
    rec["failures"] = failures[:200]
    rec["n_failures"] = len(failures)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["pass"] = not failures and not rt_fail
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["gmf", "gnl1"])
    ap.add_argument("--transform", action="store_true")
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default="results/macro_calendar_2026-10-29")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "gmf"):
        r = gate_gmf(a.transform)
        (out / "gate_GMF.json").write_text(json.dumps(r, indent=1, default=str))
        print("GM-F: PASS", flush=True)
    if a.only in (None, "gnl1"):
        r = gate_gnl1(a.n_random, a.seed)
        (out / "gate_GNL1.json").write_text(json.dumps(r, indent=1, default=str))
        print(f"GNL-1: {'PASS' if r['pass'] else 'FAIL'} ({r['n_failures']} failures, {r['evaluations']} evaluations)")
        if not r["pass"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
