"""Standalone data loading module for volatility forecasting.

Loads raw parquet data, builds a 30-min grid, filters market hours,
and returns a clean DataFrame ready for downstream pipelines.

No imports from core/ or projects/ — only numpy, pandas, os, functools.
"""

from __future__ import annotations

import os
from functools import reduce

import numpy as np
import pandas as pd

# ── Overnight fill windows (start, end) — wrap-around midnight ────────
OVERNIGHT_WINDOWS: dict[str, tuple[str, str]] = {
    "ewstock": ("20:30", "04:00"),
    "vwstock": ("20:30", "04:00"),
    "voldemand": ("17:00", "10:00"),
}

# Columns where 1.0 is genuinely the neutral overnight value (ratio/multiplier
# features). EMPTY today: every column the OVERNIGHT_WINDOWS substring match
# catches is harmed by a 1.0 fill — moments (sum*) get corrupted, and voldemand
# (range ±millions) blew Ridge QLIKE up to 0.614 in the CARC subgroup sweep,
# while ffill-only is better-or-equal across the board. Add a genuine ratio
# column here to re-enable the fill just for it.
RATIO_FILL_COLS: frozenset[str] = frozenset()


def parse_exog_cols(exog_str: str | None) -> list[str]:
    """Parse a pipe-separated exog column string into a list."""
    if not exog_str or exog_str.lower() == "none":
        return []
    return [c.strip() for c in exog_str.split("|") if c.strip()]


def apply_overnight_fills(df: pd.DataFrame, exog_cols: list[str]) -> None:
    """Fill NaN with 1.0 in overnight windows — only for ratio-type columns.

    1.0 is a neutral overnight value only for ratio/multiplier features. The
    eligible set is ``RATIO_FILL_COLS`` (empty today): every column the
    OVERNIGHT_WINDOWS substring match catches is *harmed* by a 1.0 fill —
    moments (``sum*``) get corrupted and ``voldemand`` (±millions) blew Ridge
    QLIKE up to 0.614. So this is a no-op until a genuine ratio column is added
    to ``RATIO_FILL_COLS``; the executor's ffill handles the overnight gaps.
    """
    tod = df["t"].dt.time
    for col in exog_cols:
        if col not in df.columns or col not in RATIO_FILL_COLS:
            continue
        name_lower = col.lower()
        overnight_key = next((kw for kw in OVERNIGHT_WINDOWS if kw in name_lower), None)
        if overnight_key is None:
            continue
        t_start = pd.Timestamp(
            f"1900-01-01 {OVERNIGHT_WINDOWS[overnight_key][0]}"
        ).time()
        t_end = pd.Timestamp(f"1900-01-01 {OVERNIGHT_WINDOWS[overnight_key][1]}").time()
        # Wrap-around midnight: time >= start OR time < end
        mask = (tod >= t_start) | (tod < t_end)
        df.loc[mask & df[col].isna(), col] = 1.0


# ── Feature groups ────────────────────────────────────────────────────
# Single source of truth for exogenous variable subgroup membership.
# Predicates encode the intent; lists are materialized at import time
# for cheap inspection. If new columns are added to the parquet data,
# extend ALL_FEATURES and verify the derived lists via the assertion.

ALL_FEATURES: list[str] = [
    "sumret",
    "sumabsret",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "sumvolume",
    "numobs",
    "sumret2_ewstock",
    "sumret3_ewstock",
    "sumret4_ewstock",
    "sumabsret_ewstock",
    "sumbipow_ewstock",
    "sumpret2_ewstock",
    "turnover_ewstock",
    "buyturnover_ewstock",
    "sellturnover_ewstock",
    "effspread_ewstock",
    "spread_ewstock",
    "sumret2_vwstock",
    "sumret3_vwstock",
    "sumret4_vwstock",
    "sumabsret_vwstock",
    "sumbipow_vwstock",
    "sumpret2_vwstock",
    "turnover_vwstock",
    "buyturnover_vwstock",
    "sellturnover_vwstock",
    "effspread_vwstock",
    "spread_vwstock",
    "stocktwits_attention",
    "stocktwits_sentiment",
    "stocktwits_sentcount",
    "vix",
    "vvix",
    "vix3m",
    "voldemand_spx_open_and_close",
    "voldemand_spx_open_only",
    "voldemand_all_open_and_close",
    "voldemand_all_open_only",
    # Derived, not read from parquet — see ``add_derived_features``.
    "ofi_ewstock",
    "ofi_vwstock",
]

# FOMC exogenous bucket (panel tag FEATURE_SET_TAG). Derived from
# data/releases.parquet ``fomc release`` (one 14:30 bar per scheduled
# statement). Unscheduled actions never enter until/since. Dead tail
# (after last live scheduled print) is NaN so impute-and-indicate sees
# feed-dead rather than a wall of zeros.
FEATURE_SET_TAG = "fomc1"
FOMC_RAW_COL = "fomc release"
FOMC_FEATURES: list[str] = [
    "fomc_release",
    "fomc_day",
    "fomc_until_inv",
    "fomc_since_inv",
]
UNSCHEDULED_FOMC_DAYS: frozenset[pd.Timestamp] = frozenset(
    {pd.Timestamp(d).normalize() for d in ("2020-03-03",)}
)
ALL_FEATURES.extend(FOMC_FEATURES)
# 2026-10-20 (user, third-list 5): EXCLUDE_FOMC_RELEASE drops the FOMC statement-bar dummy
# (fomc_release) from ALL_FEATURES and the 'fomc' group, so the design, the series counts and
# Table 5 can be built without it.  Under the pre-2026-10-15 chain (FOMC_FLAGS_IDENTITY off) its
# value ladders were constant after the 1/99 clip; under the current chain it is a live 0/1 flag.
# Default False = current behaviour (the CARC rerun13 design and every printed result keep it);
# turning it on changes the design and needs a rerun.  The column is still derived in load.
# 2026-10-22 (user, third-list 5): parked -- the dummy is dropped (decided 2026-10-20): 46 series,
# FOMC group of 3.  The forecasts printed before this switch (the 576-column 13-bar pooled design,
# the 628-column closing-interval design) were fitted WITH fomc_release; the MinGRU rerun on the
# clusters keeps it too (do not sync this flag to a cluster checkout before a rerun is decided).
# Was: EXCLUDE_FOMC_RELEASE: bool = False
EXCLUDE_FOMC_RELEASE: bool = True
if EXCLUDE_FOMC_RELEASE:
    ALL_FEATURES.remove("fomc_release")

# Options block (real options data; panel-v4 activation). Date-keyed options
# features (GEX family + chain aggregates) enter the panel through
# src/data/options_features.py, which enforces the publication-lag (go-live)
# rule in code. Registered as their own bucket so new arms can target them,
# but deliberately NOT appended to ALL_FEATURES: the frozen campaign panel
# must not change underfoot. ACTIVATION (when the real feed lands): append
# OPTIONS_FEATURES to ALL_FEATURES, rebuild the panel (new version), re-harvest.
from src.data.options_features import expected_channels as _opt_channels  # noqa: E402

OPTIONS_FEATURES: list[str] = _opt_channels()

# Signed order-flow imbalance, built in ``add_derived_features``: the panel carries buy and sell
# turnover as separate *levels*, so a linear model can only reach the imbalance as a difference of
# two coefficients on two columns that are ~0.99 correlated with each other and with total volume.
# The normalized ratio is the quantity the microstructure literature actually relates to volatility
# (Kyle's lambda, the VPIN family), it is scale-free and bounded in [-1, 1], and it is orthogonal
# by construction to the total-volume level the model already has.
OFI_PAIRS: dict[str, tuple[str, str]] = {
    "ofi_ewstock": ("buyturnover_ewstock", "sellturnover_ewstock"),
    "ofi_vwstock": ("buyturnover_vwstock", "sellturnover_vwstock"),
}


def add_derived_features(df: pd.DataFrame) -> list[str]:
    """Add the derived exog columns (currently the order-flow imbalance ratios).

    ``(buy - sell) / (buy + sell)``, NaN where the denominator is zero or either leg is missing —
    an imbalance is undefined with no flow, and imputing 0 there would assert "balanced" on a bar
    that had no trades. Downstream impute-and-indicate handles the NaN with its usual
    median-fill-plus-availability-flag, which is the correct encoding for "undefined here".

    Returns the names added (only those whose input legs were present).
    """
    added: list[str] = []
    for name, (buy, sell) in OFI_PAIRS.items():
        if buy not in df.columns or sell not in df.columns:
            continue
        b = df[buy].astype("float64")
        s = df[sell].astype("float64")
        tot = b + s
        df[name] = ((b - s) / tot).where(tot > 0)
        added.append(name)
    return added


# ── Domain validity of raw prints (2026-10-15, user) ──────────────────
# "it shouldn't count as signed though, it's clearly a data problem."  A
# quantity that cannot be negative (or zero) by construction is checked at
# load, and a print that breaks the rule is bad data, not a value: it is set
# to NaN here, before anything is derived from it, so the executor's
# impute-and-indicate path (``<col>__obs`` / ``<col>_avail`` + the neutral
# median fill) treats it as the unobserved bar it is -- the stack's own
# treatment of missing data, no new machinery.
#
# What triggered it: spread_vwstock (the value-weighted quoted spread) printed
# -0.000318 on 2018-11-12 11:00 ET, its only negative print in 211,639.  The
# transform chain then decided a series' sign from its data, so that one print
# gave the whole series the signed treatment (same-slot s.d. divisor +
# identity) instead of the unsigned one (same-slot mean divisor + log).  The
# audit of every series (results/data_fix_2026-10-15/SUMMARY.md) found the
# same class twice more: sumvolume is 0 on 21,852 bars (1998-01 .. 2003-06,
# almost all 01:30-09:30 ET) that carry 1-30 one-minute returns -- volume not recorded,
# coded 0 -- which the log rule turned into log(1e-12) = -27.6; and both
# effective spreads are exactly 0 on the same 171 bars (43 inside the panel
# grid; mostly pre-open, 1995-2012) whose quoted spread is 0.4-0.8 % wide and whose
# flow is mostly unclassifiable as buys or sells -- trades stamped at the
# midpoint for want of a prevailing quote, i.e. not measured.
#
# STRICTLY_POSITIVE_SERIES: a print <= 0 is invalid.  NON_NEGATIVE_SERIES: a
# print < 0 is invalid, 0 is a real value (no price change in the bar; no
# buyer- or seller-initiated trade; a zero message count).  The target ``RV``
# (sumret2) is never nulled here: its print record is the grid's existence
# criterion (``drop_dead_session``); it has no negative print (audited).
# SIGNED_SERIES: negative by construction.  The three sets together are the
# name-based sign registry src/features/transforms/target.py reads.
STRICTLY_POSITIVE_SERIES: frozenset[str] = frozenset(
    {
        # quoted spread: ask > bid in a normal market; a cross-sectional
        # average <= 0 needs crossed quotes in the weight
        "spread_ewstock",
        "spread_vwstock",
        # effective spread 2|p - m| / m: 0 across a whole cross-section only
        # when no prevailing quote was available (see above)
        "effspread_ewstock",
        "effspread_vwstock",
        # total turnover: a stock-panel bar exists only with trades
        "turnover_ewstock",
        "turnover_vwstock",
        # ES contract volume and the bar's one-minute return count: every
        # panel bar carries >= 1 one-minute return, so the contract traded
        "sumvolume",
        "numobs",
        # implied-vol indices and everything built from them
        "vix",
        "vvix",
        "vix3m",
        "impl30_perbar_rv",
        "vvix_over_vix",
        "ivslice_over_vix",
        "vix_volofvol_5d",
        "vix_volofvol_22d",
        "mfiv0_annvol",
        "mfiv0_perbar_rv",
        "mfiv0_vvix",
        "ivslice_annvol",
        "ivslice_perbar_rv",
        "ivslice_vvix",
    }
)
NON_NEGATIVE_SERIES: frozenset[str] = frozenset(
    {
        "sumabsret",
        "sumret4",
        "sumpret2",
        "sumbipow",
        *(
            f"{stat}_{w}"
            for w in ("ewstock", "vwstock")
            for stat in ("sumret2", "sumret4", "sumabsret", "sumbipow", "sumpret2")
        ),
        # one side of the flow can be empty in a thin bar (1,779 / 2,135 zeros)
        "buyturnover_ewstock",
        "sellturnover_ewstock",
        "buyturnover_vwstock",
        "sellturnover_vwstock",
        # message counts (no zero in the file: a bar without messages is NaN)
        "stocktwits_attention",
        "stocktwits_sentcount",
        # derived in this module, non-negative by construction
        "fomc_release",
        "fomc_day",
        "fomc_until_inv",
        "fomc_since_inv",
    }
)
SIGNED_SERIES: frozenset[str] = frozenset(
    {
        "sumret",
        "sumret3",
        "sumautocov",
        "sumret3_ewstock",
        "sumret3_vwstock",
        "stocktwits_sentiment",  # net score in [-1, 1]
        "voldemand_spx_open_and_close",
        "voldemand_spx_open_only",
        "voldemand_all_open_and_close",
        "voldemand_all_open_only",
        "ofi_ewstock",  # (buy - sell) / (buy + sell) in [-1, 1]
        "ofi_vwstock",
        # implied-vol representations (outside ALL_FEATURES): slopes, changes,
        # log implied-over-realized
        "vix_slope_3m",
        "vix_chg_1bar",
        "vix_chg_1d",
        "vix_chg_5d",
        "vix_vrp_1d",
        "vix_vrp_5d",
        "vix_vrp_22d",
    }
)
# Off reproduces the pre-2026-10-15 panel bit for bit (the impact study's control).
NULL_INVALID_PRINTS: bool = True


def null_invalid_prints(df: pd.DataFrame, verbose: bool = True) -> dict[str, int]:
    """Set prints that break their series' domain to NaN, in place; return {column: count}.

    ``STRICTLY_POSITIVE_SERIES``: ``<= 0`` is invalid; ``NON_NEGATIVE_SERIES``: ``< 0`` is.
    ``RV`` is exempt (see the block comment above).  One log line with the counts.
    """
    counts: dict[str, int] = {}
    for col in df.columns:
        if col == "RV":
            continue
        if col in STRICTLY_POSITIVE_SERIES:
            bad = (df[col] <= 0).to_numpy()
        elif col in NON_NEGATIVE_SERIES:
            bad = (df[col] < 0).to_numpy()
        else:
            continue
        n = int(bad.sum())
        if n:
            df.loc[bad, col] = np.nan
            counts[col] = n
    if verbose:
        body = ", ".join(f"{c} {n}" for c, n in counts.items()) or "none"
        print(
            f"load_raw_data: nulled {sum(counts.values())} invalid print(s) (domain rule): {body}"
        )
    return counts


# ── Frozen feeds (2026-10-15, user: "fix all 8") ──────────────────────
# results/data_audit_2026-10-15/ (experiments/data_audit_2026_10_15.py, switches fcal / vix).
#
# The NYSE 13:00 ET early-close calendar 2001-2025, the same 55 dates as
# notebooks/atm_straddle_lib.EARLY_CLOSE_DATES (the calendar authority, with its derivation
# and checks); copied so src does not import notebooks.  Read by the executor's B and
# per-bar session rules (B_REGULAR_BARS_ONLY, DROP_EARLY_CLOSE_SESSIONS).
NYSE_EARLY_CLOSE_DATES: tuple[str, ...] = (
    "2001-07-03",
    "2001-11-23",
    "2001-12-24",
    "2002-07-03",
    "2002-11-29",
    "2002-12-24",
    "2003-07-03",
    "2003-11-28",
    "2003-12-24",
    "2004-11-26",
    "2005-11-25",
    "2006-07-03",
    "2006-11-24",
    "2007-07-03",
    "2007-11-23",
    "2007-12-24",
    "2008-07-03",
    "2008-11-28",
    "2008-12-24",
    "2009-11-27",
    "2009-12-24",
    "2010-11-26",
    "2011-11-25",
    "2012-07-03",
    "2012-11-23",
    "2012-12-24",
    "2013-07-03",
    "2013-11-29",
    "2013-12-24",
    "2014-07-03",
    "2014-11-28",
    "2014-12-24",
    "2015-11-27",
    "2015-12-24",
    "2016-11-25",
    "2017-07-03",
    "2017-11-24",
    "2018-07-03",
    "2018-11-23",
    "2018-12-24",
    "2019-07-03",
    "2019-11-29",
    "2019-12-24",
    "2020-11-27",
    "2020-12-24",
    "2021-11-26",
    "2022-11-25",
    "2023-07-03",
    "2023-11-24",
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
)

# data/releases.parquet is a stale pull: every release calendar in it stops between 2023-10
# and 2024-01, and the FOMC one at the 2023-11-01 statement, so the "dead tail" the executor
# imputed over 2023-11-01 .. 2024-04-30 (123 of the 866 trade days) was three missing
# scheduled statements, not a stopped feed.  With the fix they are added here (the file is
# not edited) and the first statement after the panel is the next event of the last rows'
# fomc_until_inv, so no FOMC row of the historical panel is unobserved.  A frame that
# already holds these stamps, or ends after FOMC_NEXT_STATEMENT, is unaffected.
FOMC_CALENDAR_FIX: bool = True
FOMC_MISSING_STATEMENTS: tuple[str, ...] = (
    "2023-12-13 14:30",
    "2024-01-31 14:30",
    "2024-03-20 14:30",
)
FOMC_NEXT_STATEMENT: str = "2024-05-01 14:30"

# VIX / VIX3M / VVIX stop at 2024-02-12 in data/vix_and_voldemand.parquet, 54 of the 866
# trade days short of the panel end; their imputed tail put the VIX availability ladders
# outside their training range on all 54 days.  The Yahoo hourly snapshot
# (data/free_feed/*_1h_yahoo.parquet) equals the panel's own bar-end prints on 88-100 % of the
# 2023-12 .. 2024-02 overlap (results/data_audit_2026-10-15/vix_yahoo_overlap.csv).  After a
# series' last print and up to VIX_SPLICE_END only (the historical panel), the hourly close
# is written at its bar end and carried 30 minutes to the next bar, in the panel's own print
# slots.  The live service never reaches it: it needs <data_path>/free_feed/ and a frame whose
# last print falls before VIX_SPLICE_END.
SPLICE_VIX_FREE_FEED: bool = True
VIX_SPLICE_END: str = "2024-04-30 23:30"
VIX_SPLICE_SLOTS: dict[str, tuple[str, str]] = {
    "vix": ("04:00", "17:00"),
    "vix3m": ("10:00", "16:30"),
    "vvix": ("10:00", "16:00"),
}


def _free_feed_bar_ends(path: str) -> pd.Series:
    """A Yahoo hourly file's close keyed by the naive-ET bar END (open stamp + 1 h)."""
    y = pd.read_parquet(path)
    end = pd.to_datetime(y["t"]).dt.tz_convert("America/New_York").dt.tz_localize(
        None
    ) + pd.Timedelta("1h")
    return pd.Series(
        y["close"].to_numpy(float), index=pd.DatetimeIndex(end)
    ).sort_index()


def splice_vix_free_feed(
    df: pd.DataFrame, free_dir: str, verbose: bool = True
) -> dict[str, int]:
    """Fill vix / vix3m / vvix after their last print from the free hourly snapshot, in place."""
    out: dict[str, int] = {}
    t = pd.DatetimeIndex(df["t"])
    end = pd.Timestamp(VIX_SPLICE_END)
    for col, (lo, hi) in VIX_SPLICE_SLOTS.items():
        f = os.path.join(free_dir, f"{col}_1h_yahoo.parquet")
        if col not in df.columns or not os.path.isfile(f) or not df[col].notna().any():
            continue
        last = t[df[col].notna().to_numpy()].max()
        if last >= end:
            continue
        ys = _free_feed_bar_ends(f)
        ys = ys[(ys.index > last) & (ys.index <= end)]
        carry = pd.Series(ys.to_numpy(), index=ys.index + pd.Timedelta("30min"))
        both = pd.concat([ys, carry[~carry.index.isin(ys.index)]]).sort_index()
        hhmm = both.index.strftime("%H:%M")
        both = both[(hhmm >= lo) & (hhmm <= hi)]
        m = (t > last) & t.isin(both.index)
        df.loc[m, col] = both.reindex(t[m]).to_numpy()
        out[col] = int(m.sum())
    if verbose and out:
        print(
            f"load_raw_data: spliced the free hourly feed after the last print: {out}"
        )
    return out


def _calendar_open_bars_between(t0: pd.Timestamp, t1: pd.Timestamp) -> int:
    """30-minute bars in (t0, t1] that the weekend cut keeps (the calendar-open grid)."""
    g = pd.date_range(t0, t1, freq=FREQ)[1:]
    dow, tod = g.dayofweek, g.time
    closed = (
        ((dow == 4) & (tod > pd.Timestamp(f"1900-01-01 {FRIDAY_CLOSE}").time()))
        | (dow == 5)
        | ((dow == 6) & (tod < pd.Timestamp(f"1900-01-01 {SUNDAY_OPEN}").time()))
    )
    return int((~closed).sum())


def add_fomc_features(df: pd.DataFrame) -> list[str]:
    """Scheduled-FOMC dummies + inverse bar-distance; NaN after last live print."""
    if FOMC_RAW_COL not in df.columns or "t" not in df.columns:
        return []
    t = pd.to_datetime(df["t"])
    if FOMC_CALENDAR_FIX:
        for s in FOMC_MISSING_STATEMENTS:
            m = (t == pd.Timestamp(s)).to_numpy()
            if m.any():
                df.loc[m, FOMC_RAW_COL] = 1.0
    raw = pd.to_numeric(df[FOMC_RAW_COL], errors="coerce").to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0)
    day = t.dt.normalize()
    unsch = day.isin(UNSCHEDULED_FOMC_DAYS).to_numpy()
    is_event = (raw > 0.0) & ~unsch
    n = len(df)
    event_pos = np.flatnonzero(is_event)
    release = is_event.astype(np.float64)
    event_days = (
        pd.DatetimeIndex(np.asarray(day)[is_event]).unique()
        if event_pos.size
        else pd.DatetimeIndex([])
    )
    is_day = day.isin(event_days).to_numpy().astype(np.float64)
    until = np.full(n, np.nan, dtype=np.float64)
    since = np.full(n, np.nan, dtype=np.float64)
    if event_pos.size:
        i = np.arange(n)
        nxt = np.searchsorted(event_pos, i, side="left")
        ok = nxt < event_pos.size
        until[ok] = 1.0 / (1.0 + (event_pos[nxt[ok]] - i[ok]).astype(np.float64))
        prev = np.searchsorted(event_pos, i, side="right") - 1
        ok2 = prev >= 0
        since[ok2] = 1.0 / (1.0 + (i[ok2] - event_pos[prev[ok2]]).astype(np.float64))
    nxt_t = pd.Timestamp(FOMC_NEXT_STATEMENT)
    if (
        FOMC_CALENDAR_FIX
        and event_pos.size
        and t.iloc[int(event_pos[-1])] < nxt_t
        and t.iloc[-1] < nxt_t
    ):
        # the next statement lies beyond the frame: its calendar-open grid position is the
        # distance of the rows after the last statement (no dead tail)
        last = int(event_pos[-1])
        pos_next = n - 1 + _calendar_open_bars_between(t.iloc[-1], nxt_t)
        i = np.arange(n)
        after = i > last
        until[after] = 1.0 / (1.0 + (pos_next - i[after]).astype(float))
        df["fomc_release"] = release
        df["fomc_day"] = is_day
        df["fomc_until_inv"] = until
        df["fomc_since_inv"] = since
        return list(FOMC_FEATURES)
    df["fomc_release"] = release
    df["fomc_day"] = is_day
    df["fomc_until_inv"] = until
    df["fomc_since_inv"] = since
    if event_pos.size:
        dead = t > t.iloc[int(event_pos[-1])]
        df.loc[dead, FOMC_FEATURES] = np.nan
    else:
        df[FOMC_FEATURES] = np.nan
    return list(FOMC_FEATURES)


def _is_moment(f: str) -> bool:
    return f.startswith("sum") and "stock" not in f and "volume" not in f


def _is_liquidity(f: str) -> bool:
    # ``ofi`` joins the liquidity bucket: it is a flow-composition measure built from turnover, not
    # a return moment, so it must not fall through to ``market_ew`` / ``market_vw`` on the bare
    # ``ewstock`` / ``vwstock`` suffix.
    return any(x in f for x in ("volume", "turnover", "spread", "numobs", "ofi"))


SUBGROUPS: dict[str, list[str]] = {
    "baseline": [],
    "moments": [f for f in ALL_FEATURES if _is_moment(f)],
    "liquidity": [f for f in ALL_FEATURES if _is_liquidity(f)],
    "market_ew": [
        f
        for f in ALL_FEATURES
        if "ewstock" in f and not any(x in f for x in ("turnover", "spread", "ofi"))
    ],
    "market_vw": [
        f
        for f in ALL_FEATURES
        if "vwstock" in f and not any(x in f for x in ("turnover", "spread", "ofi"))
    ],
    "sentiment": [f for f in ALL_FEATURES if "stocktwits" in f],
    "implied_vol": [f for f in ALL_FEATURES if "vix" in f],
    "vol_demand": [f for f in ALL_FEATURES if "voldemand" in f],
    "fomc": [f for f in FOMC_FEATURES if not (EXCLUDE_FOMC_RELEASE and f == "fomc_release")],
    "options": list(OPTIONS_FEATURES),
    "all_features": ALL_FEATURES,
}
# The columns a LIVE forecaster can rebuild at 15:30 from feeds a retail account
# has: ES one-minute bars (the return moments and the liquidity measures), the
# Cboe volatility indices at 30 minutes, and the release calendar.  Left out:
# the constituent cross-section (market_ew / market_vw), StockTwits and the
# Cboe volume series, none of which is available intraday at retail cost.
# Order preserved as in ALL_FEATURES so the design's column order is stable.
_LIVE_FEASIBLE_PARTS = {
    *SUBGROUPS["moments"],
    *SUBGROUPS["liquidity"],
    *SUBGROUPS["implied_vol"],
    *SUBGROUPS["fomc"],
}
# ``liquidity`` also carries the constituents' turnover / spread / order-flow
# columns; those come from the cross-section too and are dropped by name.
SUBGROUPS["live_feasible"] = [
    f for f in ALL_FEATURES if f in _LIVE_FEASIBLE_PARTS and "stock" not in f
]
# The 0DTE chain's own implied variance (experiments/build_spxw_mfiv_panel.py ->
# data/spxw_mfiv.parquet): the VIX calculation on the same-day strip at every
# 30-minute stamp, horizon-matched to the target.  Outside ALL_FEATURES like
# OPTIONS_FEATURES (the frozen campaign panel keeps its column set); reachable
# only through these buckets.  Present from 2020-01 on; earlier rows carry the
# late-start indicator + neutral fill like the VIX family before 2006.
MFIV_FEATURES: list[str] = ["mfiv0_annvol", "mfiv0_perbar_rv", "mfiv0_vvix"]
SUBGROUPS["mfiv_only"] = list(MFIV_FEATURES)
# live_feasible with the Cboe prints swapped for the chain's own numbers, and with both.
SUBGROUPS["live_feasible_mfiv"] = [
    f for f in SUBGROUPS["live_feasible"] if f not in SUBGROUPS["implied_vol"]
] + MFIV_FEATURES
SUBGROUPS["live_feasible_plus_mfiv"] = SUBGROUPS["live_feasible"] + MFIV_FEATURES
# The deck's own object at every clock (experiments/build_spxw_ivslice_panel.py ->
# data/spxw_ivslice.parquet): the nearest-OTM straddle's package midpoint
# bisected for the Black-76 total volatility over the remaining session -- the
# implied slice the 15:30 trade is measured against, re-picked at 10:00..15:30.
# The strip above is 9x this at 15:30 (zero-bid strikes at the minimum tick);
# this is the clean version.  Same three-column shape as the MFIV bucket.
IVSLICE_FEATURES: list[str] = ["ivslice_annvol", "ivslice_perbar_rv", "ivslice_vvix"]
SUBGROUPS["ivslice_only"] = list(IVSLICE_FEATURES)
SUBGROUPS["live_feasible_ivslice"] = [
    f for f in SUBGROUPS["live_feasible"] if f not in SUBGROUPS["implied_vol"]
] + IVSLICE_FEATURES
SUBGROUPS["live_feasible_plus_ivslice"] = SUBGROUPS["live_feasible"] + IVSLICE_FEATURES
# Representations of the VIX family (experiments/build_vix_representations.py ->
# data/vix_representations.parquet).  R0 is live_feasible itself: the three Cboe
# prints as log levels, exempt from the diurnal division, six rolling means each.
# Every bucket below keeps the live base (moments + liquidity + FOMC, the 13
# columns without the three prints) and swaps in one representation:
#   ivrep_target_scale   VIX as per-bar implied variance in the target's units
#                        (sqrt + diurnal: linear in what the model forecasts), vvix
#                        and vix3m as levels
#   ivrep_term           the level plus explicit term slope and vol-of-vol ratio
#   ivrep_innovations    the levels plus log changes over 1 bar / 1 day / 5 days
#   ivrep_vrp            implied over trailing realized at 1 / 5 / 22 sessions, no
#                        levels
#   ivrep_slice_slope    the 0DTE ATM slice and its slope to the 30-day VIX
#                        (2020-01 on; tw 500 only)
#   ivrep_all            R1-R4 together; the lasso chooses
# Outside ALL_FEATURES like the blocks above.
_LIVE_NO_IV = [
    f for f in SUBGROUPS["live_feasible"] if f not in SUBGROUPS["implied_vol"]
]
SUBGROUPS["ivrep_target_scale"] = _LIVE_NO_IV + ["impl30_perbar_rv", "vvix", "vix3m"]
SUBGROUPS["ivrep_term"] = _LIVE_NO_IV + ["vix", "vix_slope_3m", "vvix_over_vix"]
SUBGROUPS["ivrep_innovations"] = _LIVE_NO_IV + [
    "vix",
    "vvix",
    "vix3m",
    "vix_chg_1bar",
    "vix_chg_1d",
    "vix_chg_5d",
]
SUBGROUPS["ivrep_vrp"] = _LIVE_NO_IV + ["vix_vrp_1d", "vix_vrp_5d", "vix_vrp_22d"]
SUBGROUPS["ivrep_slice_slope"] = _LIVE_NO_IV + IVSLICE_FEATURES + ["ivslice_over_vix"]
SUBGROUPS["ivrep_all"] = _LIVE_NO_IV + [
    "impl30_perbar_rv",
    "vix",
    "vvix",
    "vix3m",
    "vix_slope_3m",
    "vvix_over_vix",
    "vix_chg_1bar",
    "vix_chg_1d",
    "vix_chg_5d",
    "vix_vrp_1d",
    "vix_vrp_5d",
    "vix_vrp_22d",
]


# The FREE-FEED bucket (2026-09-23, live/close_signal): live_feasible without
# the liquidity columns.  A retail account rebuilding the panel from free
# 1-minute bars has the ES return moments and the Cboe prints but no ES book
# data: sumvolume from a continuous-contract feed is not the vendor's and the
# vendor's numobs is a tick count, not a minute count.  R0 = live_feasible;
# the bucket table already ranked liquidity last at the close.
SUBGROUPS["free_feasible"] = [
    f for f in SUBGROUPS["live_feasible"] if f not in SUBGROUPS["liquidity"]
]

# Does the VIX bucket need VVIX and VIX3M, and can the VIX's own path replace
# them?  (2026-09-23; experiments/build_vix_rvol.py -> data/vix_rvol.parquet.)
# On the 15:30 stamps with all three prints, VIX3M is 95 % the VIX level and
# VVIX 48 %; together they add +0.0064 of R^2 to the next bar beside log VIX and
# the last bar, and the realized vol-of-VIX from the VIX's own 30-minute path
# (5- and 22-session windows) recovers about half of that.  Four buckets:
#   vix_only        HAR + calendar + the VIX level alone
#   vix_rvol        ... + the two realized vol-of-VIX columns
#   live_vix_only   live_feasible without vvix and vix3m (what a VIX-only feed
#                   carries)
#   live_vix_rvol   live_vix_only + the two realized vol-of-VIX columns
# References: implied_vol (the trio) and live_feasible (the trio in the live
# base).  Outside ALL_FEATURES like the blocks above.
# "volofvol", not "rvol": the "rv" stem would make the library square-root it.
VIX_RVOL_FEATURES: list[str] = ["vix_volofvol_5d", "vix_volofvol_22d"]
# The free feed reproduces everything in live_feasible except the ES tick count
# (numobs): Yahoo's ES=F 1-minute bars carry volume.  free_feasible (above) also
# dropped sumvolume as a "liquidity" column; this bucket keeps it.  The fair
# comparison of the two against live_feasible prices the tick count alone.
SUBGROUPS["free_feasible_vol"] = [
    f for f in SUBGROUPS["live_feasible"] if f != "numobs"
]
# The bucket the close-signal service will forecast with if it holds: the free
# feed's 15 columns without vvix and vix3m (the 2026-09-23 vixonly campaign
# found neither adds anything at the per-bar arm; the live feed then needs the
# VIX alone).  13 columns: the 8 ES moments incl. sumvolume, vix, 4 FOMC.
SUBGROUPS["free_vix_only"] = [
    f for f in SUBGROUPS["free_feasible_vol"] if f not in ("vvix", "vix3m")
]
SUBGROUPS["vix_only"] = ["vix"]
SUBGROUPS["vix_rvol"] = ["vix", *VIX_RVOL_FEATURES]
SUBGROUPS["live_vix_only"] = [
    f for f in SUBGROUPS["live_feasible"] if f not in ("vvix", "vix3m")
]
SUBGROUPS["live_vix_rvol"] = SUBGROUPS["live_vix_only"] + VIX_RVOL_FEATURES
# 2026-10-06, user request: the return moments and the VIX level alone, compared
# with baseline (HAR + calendar) and all_features.  8 columns: the 7 ES return
# moments (no sumvolume) + vix; no VVIX / VIX3M, no FOMC.  Outside ALL_FEATURES
# like the blocks above.
SUBGROUPS["moments_vix"] = [*SUBGROUPS["moments"], "vix"]
# 2026-10-06, user request: moments_vix plus ES volume (sumvolume, the liquidity column that ranks highest
# in the all_features importance and helps every linear model when added).  9 columns: the 7 ES return
# moments + sumvolume + vix.  Named so that "moments_vix" is not a substring of it.
SUBGROUPS["moments_vol_vix"] = [*SUBGROUPS["moments"], "sumvolume", "vix"]
# 2026-10-06, user request: the lean set -- moments_vol_vix without the moments that duplicate the absolute
# return (bipower variation, upside squared returns; |corr| >= 0.82 at every lag in the all_features cluster
# study) and the weak cubic return.  6 columns, in ALL_FEATURES order.
SUBGROUPS["lean_vol_vix"] = [
    "sumret",
    "sumabsret",
    "sumret4",
    "sumautocov",
    "sumvolume",
    "vix",
]


def get_bucket(name: str) -> list[str]:
    if name not in SUBGROUPS:
        raise KeyError(f"Unknown subgroup '{name}'. Valid: {sorted(SUBGROUPS.keys())}")
    return SUBGROUPS[name]


# ── Constants ──────────────────────────────────────────────────────────
# Panel v3 (2026-08-07): the grid start is DATA-DERIVED (first bar the RV
# source printed on, ~1998-01) — same philosophy as the dead-session rule:
# the source's own print record, not a hardcoded calendar pin, decides what
# exists. START_DATE is retained ONLY as the legacy 2005 pin for consumers
# that need bit-compat with archived results (pass it as ``start_date``);
# the loader no longer reads it.
START_DATE = "2005-01-01"
FRIDAY_CLOSE = "20:00"
SUNDAY_OPEN = "18:30"
FREQ = "30min"


def load_raw_data(
    data_path: str,
    allow_missing: bool = False,
    drop_dead_session: bool = True,
    start_date: str | None = None,
) -> pd.DataFrame:
    """Load parquet data, grid to 30-min, filter market hours, clean NaNs.

    Parameters
    ----------
    data_path : str
        Path to a directory of .parquet files or a single .parquet file.
    allow_missing : bool
        If False (default), drop all rows with any remaining NaN after
        forward-filling the target column. If True, keep them.
    start_date : str | None
        Grid start. None (default) = DATA-DERIVED: the first bar on which
        the RV source printed (panel v3; ~1998-01). Pass an explicit date
        (e.g. the legacy ``START_DATE`` 2005 pin) to reproduce an archived
        grid.
    drop_dead_session : bool
        If True (default), a grid bar EXISTS in the panel iff the RV source
        itself printed on it (notna before any fill) — the data-derived
        dead-session rule. The calendar-only weekend trim cannot know about
        early closes and holidays; the RV source's own print record does.
        Threshold-free: the same observed-print fact the availability
        indicators use, applied to the core source as the grid's existence
        criterion. Motivation: the c080 incident — a ghost bar after the
        2020-12-24 early close (calendar-open, market-dead) carried a one-bar
        availability desync that near-duplicate avail-ladder rungs amplified
        to yhat=+77.8, and ffilled RV flatlines on such bars contaminated the
        evaluation target. Set False only for bit-compat with archived
        results produced under the calendar-only grid.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns including ``t``, ``RV``,
        ``time_of_day``, and any additional feature columns from the
        parquet files.
    """
    # ── 1. Load parquet file(s) ────────────────────────────────────────
    if os.path.isfile(data_path):
        frames = [pd.read_parquet(data_path)]
    else:
        parquet_files = sorted(
            f for f in os.listdir(data_path) if f.endswith(".parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        frames = [pd.read_parquet(os.path.join(data_path, f)) for f in parquet_files]
        # Normalize the merge key: feeds arrive with endbartime as string OR
        # timestamp (e.g. vix_and_voldemand=string, options_features=timestamp);
        # a mixed-dtype outer merge raises ValueError (2026-08-12 incident:
        # the options_features parquet landed and killed every running harvest).
        for fr in frames:
            if "endbartime" in fr.columns:
                fr["endbartime"] = pd.to_datetime(fr["endbartime"])
        # Directory mode merges on endbartime; parquets without that key are
        # not bar-panel families (e.g. the OptionMetrics chain/spot exports,
        # which are date/option keyed and loaded by their own consumers) —
        # skip them loudly instead of dying in the reduce-merge below.
        keyed = [
            (f, fr)
            for f, fr in zip(parquet_files, frames)
            if "endbartime" in fr.columns
        ]
        skipped = [
            f for f, fr in zip(parquet_files, frames) if "endbartime" not in fr.columns
        ]
        if skipped and len(frames) > 1:
            print(
                f"load_raw_data: skipping non-bar-keyed parquet(s): {', '.join(skipped)}"
            )
            frames = [fr for _, fr in keyed]
            if not frames:
                raise FileNotFoundError(
                    f"No endbartime-keyed .parquet files in {data_path}"
                )

    # ── 2. Merge on endbartime (outer join) ────────────────────────────
    if len(frames) == 1:
        df = frames[0]
    else:
        df = reduce(
            lambda left, right: pd.merge(left, right, on="endbartime", how="outer"),
            frames,
        )

    # ── 3. Rename columns ─────────────────────────────────────────────
    rename_map = {}
    if "endbartime" in df.columns:
        rename_map["endbartime"] = "t"
    if "sumret2" in df.columns:
        rename_map["sumret2"] = "RV"
    df = df.rename(columns=rename_map)

    # ── 4. Convert t to datetime, drop duplicates ─────────────────────
    df["t"] = pd.to_datetime(df["t"])
    df = df.drop_duplicates(subset="t")

    # ── 5. Create 30-min grid and reindex ─────────────────────────────
    end_date = df["t"].max()
    if start_date is None:
        # Panel v3: DATA-DERIVED grid start — the first bar the RV source
        # printed on. The RV source's own print record decides the grid's
        # existence at BOTH ends (start here, per-bar via drop_dead_session):
        # no hardcoded calendar pin, no pre-history ghost bars.
        start_date = df.loc[df["RV"].notna(), "t"].min()
    grid = pd.date_range(start=start_date, end=end_date, freq=FREQ)
    df = df.set_index("t").reindex(grid).rename_axis("t").reset_index()

    # ── 6. Filter out market-closed hours ─────────────────────────────
    day_of_week = df["t"].dt.dayofweek  # Mon=0 … Sun=6
    time_of_day = df["t"].dt.time

    friday_close = pd.Timestamp(f"1900-01-01 {FRIDAY_CLOSE}").time()
    sunday_open = pd.Timestamp(f"1900-01-01 {SUNDAY_OPEN}").time()

    mask_friday_after_close = (day_of_week == 4) & (time_of_day > friday_close)
    mask_saturday = day_of_week == 5
    mask_sunday_before_open = (day_of_week == 6) & (time_of_day < sunday_open)

    closed_mask = mask_friday_after_close | mask_saturday | mask_sunday_before_open
    df = df[~closed_mask].reset_index(drop=True)

    # ── 6a. Domain validity (2026-10-15, user) ────────────────────────
    # Invalid prints -> NaN before anything is derived from them (the OFI legs, the
    # sign of a series downstream); see ``STRICTLY_POSITIVE_SERIES``.  Counted on the
    # calendar-open grid, i.e. before the dead-session drop below.
    if NULL_INVALID_PRINTS:
        null_invalid_prints(df)

    # ── 6b. Derived exog (order-flow imbalance) ───────────────────────
    # Built here, on the *unfilled* buy/sell legs, deliberately: ``apply_overnight_fills`` later
    # writes 1.0 into matching columns in the overnight window, and buy = sell = 1.0 would make the
    # imbalance exactly 0 — fabricating "perfectly balanced flow" on bars with no trading. Leaving
    # it NaN overnight lets the normal ffill / impute-and-indicate path handle it honestly.
    add_derived_features(df)
    add_fomc_features(df)

    # ── 7. Dead-session drop / RV fill ────────────────────────────────
    if drop_dead_session:
        # Data-derived grid existence (see docstring; c080): keep only bars
        # the RV source printed on — BEFORE any forward fill, so no ghost
        # bar survives to carry ffill-flatlined targets or desynced
        # indicators. Everything downstream (diurnal slots, HAR ladders,
        # indicators) operates on the reduced grid unchanged.
        df = df[df["RV"].notna()].reset_index(drop=True)
    else:
        # Legacy calendar-only path: forward-fill RV over calendar-open
        # bars, drop only the leading NaNs.
        df["RV"] = df["RV"].ffill()
        df = df.dropna(subset=["RV"]).reset_index(drop=True)

    # ── 8. NaN handling for remaining columns ─────────────────────────
    if not allow_missing:
        df = df.dropna().reset_index(drop=True)

    # ── 9. Add time_of_day column ─────────────────────────────────────
    df["time_of_day"] = df["t"].dt.time

    # ── 10. Frozen VIX family (2026-10-15, user): the free hourly feed after the last print
    # (historical panel only; see SPLICE_VIX_FREE_FEED) ───────────────────
    if SPLICE_VIX_FREE_FEED:
        base = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
        free_dir = os.path.join(base, "free_feed")
        if os.path.isdir(free_dir):
            splice_vix_free_feed(df, free_dir)

    return df
