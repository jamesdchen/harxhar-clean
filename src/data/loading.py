"""Standalone data loading module for volatility forecasting.

Loads raw parquet data, builds a 30-min grid, filters market hours,
and returns a clean DataFrame ready for downstream pipelines.

No imports from core/ or projects/ — only numpy, pandas, os, functools.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from functools import lru_cache, reduce
from typing import NamedTuple

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

# ── Macro release calendars (2026-10-29, author: "can we check if the dataset has cpi release, employment release,
# ppi release, gdp release, trade release, retail release and treat it similarly to fomc? There needs to be no
# lookahead bias.") ──────────────────────────────────────────────────────────────────────────────────────────────
# data/releases.parquet carries six more 0/1 calendars beside "fomc release", keyed like it by the bar END in naive ET;
# every release is flagged on the bar ending 09:00 (the 08:30 ET release), on its ACTUAL release date (the 2013 and
# 2018-19 shutdown dates are the delayed ones).  GDP is the advance estimate only.  Each release r gets FOMC's three
# design series (the 0/1 statement-bar dummy is out of the design, EXCLUDE_FOMC_RELEASE), stem mac_<r> for fomc, built
# by add_fomc_features' construction on the calendar-open grid (add_macro_features, below):
#   mac_<r>_day        1 on every bar of a release date
#   mac_<r>_until_inv  1 / (1 + bars to the next release)  (under MACRO_SCHEDULE_FIX: the next release as scheduled at
#                      the bar; NaN while a postponed release has no new date, so its availability flag is 0)
#   mac_<r>_since_inv  1 / (1 + bars since the last actual release)
# for r in cpi, emp, ppi, gdp, trade, retail -- 18 series:
#   mac_cpi_day mac_cpi_until_inv mac_cpi_since_inv   mac_emp_day mac_emp_until_inv mac_emp_since_inv
#   mac_ppi_day mac_ppi_until_inv mac_ppi_since_inv   mac_gdp_day mac_gdp_until_inv mac_gdp_since_inv
#   mac_trade_day mac_trade_until_inv mac_trade_since_inv   mac_retail_day mac_retail_until_inv mac_retail_since_inv
# Executor design columns (as FOMC's): adj_<s>_ma_<k> and <s>_avail_ma_<k> (and <s>_active_ma_<k> for a mode-inflated
# series when DROP_ACTIVITY_MA is off).  Registered in NON_NEGATIVE_SERIES and SUBGROUPS["macro"]; SUBGROUPS
# ["all_features_mac"] = ALL_FEATURES + macro.  NOT in ALL_FEATURES: the 46-series set (and every design built from it)
# is unchanged.  MACRO_CALENDARS off (default): load_raw_data builds no mac_ column, every output bit for bit as before.
# Gates: experiments/macro_calendar_check.py (GM-F frame, GNL-1 no-lookahead); unit tests writeup/tests/
# test_macro_calendar.py.
MACRO_CALENDARS: bool = False
MACRO_RELEASES: dict[str, str] = {
    "cpi": "cpi release",
    "emp": "employment release",
    "ppi": "ppi release",
    "gdp": "gdp release",  # the advance estimate only
    "trade": "trade release",
    "retail": "retail release",
}
MACRO_SERIES_SUFFIXES: tuple[str, ...] = ("day", "until_inv", "since_inv")
MACRO_SERIES_OF: dict[str, list[str]] = {r: [f"mac_{r}_{s}" for s in MACRO_SERIES_SUFFIXES] for r in MACRO_RELEASES}
MACRO_FEATURES: list[str] = [s for r in MACRO_RELEASES for s in MACRO_SERIES_OF[r]]

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
        # 2026-10-29: the macro release calendars, as the FOMC series (MACRO_CALENDARS)
        *MACRO_FEATURES,
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

# 2026-10-25 (coordinator, author approved; audit 2026-10-25 fidelity-4, numbers-7, correctness-5, design-11;
# results/fomcfix_2026-10-26/RULE.txt addendum A): the full NYSE holidays on which the E-mini traded a shortened session
# with regular-hours bars (bar ends 10:00 .. 13:00), 123 dates 2004-01-19 .. 2024-02-19, from the sample's own calendar:
# the dates with a bar ending 10:00 .. 16:00 in the 48-interval grid (after the dead-session drop), no bar at 16:00,
# not an NYSE early close (NYSE_EARLY_CLOSE_DATES), and no print of any constituent series (the 24 *_ewstock /
# *_vwstock columns) on any bar of the date -- the NYSE did not trade.  Each is a weekday that the rule calendar of
# experiments/rolling_eval_2026_10_15.nyse_sessions marks as a holiday, and no rule-calendar holiday has a 16:00 bar;
# 64 of them, with 443 regular-hours bars, fall in the evaluation span 2014-01-02 .. 2024-04-30 (re-derived and
# asserted by results/fomcfix_2026-10-26/code/holiday_sessions.py).  Read by the executor's DROP_HOLIDAY_SESSIONS only.
NYSE_HOLIDAY_FUTURES_SESSIONS: tuple[str, ...] = (
    "2004-01-19", "2004-02-16", "2004-07-05", "2004-09-06", "2004-11-25", "2005-01-17",
    "2005-02-21", "2005-05-30", "2005-07-04", "2005-09-05", "2005-11-24", "2006-01-16",
    "2006-02-20", "2006-05-29", "2006-07-04", "2006-09-04", "2006-11-23", "2007-01-15",
    "2007-02-19", "2007-05-28", "2007-07-04", "2007-09-03", "2007-11-22", "2008-01-21",
    "2008-02-18", "2008-05-26", "2008-07-04", "2008-09-01", "2008-11-27", "2009-01-19",
    "2009-02-16", "2009-05-25", "2009-07-03", "2009-09-07", "2009-11-26", "2010-01-18",
    "2010-02-15", "2010-05-31", "2010-07-05", "2010-09-06", "2010-11-25", "2011-01-17",
    "2011-02-21", "2011-05-30", "2011-07-04", "2011-09-05", "2011-11-24", "2012-01-16",
    "2012-02-20", "2012-05-28", "2012-07-04", "2012-09-03", "2012-11-22", "2013-01-21",
    "2013-02-18", "2013-05-27", "2013-07-04", "2013-09-02", "2013-11-28", "2014-01-20",
    "2014-02-17", "2014-05-26", "2014-07-04", "2014-09-01", "2014-11-27", "2015-01-19",
    "2015-02-16", "2015-05-25", "2015-07-03", "2015-09-07", "2015-11-26", "2016-01-18",
    "2016-02-15", "2016-05-30", "2016-07-04", "2016-09-05", "2016-11-24", "2017-01-16",
    "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04", "2017-11-23", "2018-01-15",
    "2018-02-19", "2018-05-28", "2018-07-04", "2018-09-03", "2018-11-22", "2019-01-21",
    "2019-02-18", "2019-05-27", "2019-07-04", "2019-09-02", "2019-11-28", "2020-01-20",
    "2020-02-17", "2020-05-25", "2020-07-03", "2020-09-07", "2020-11-26", "2021-01-18",
    "2021-02-15", "2021-05-31", "2021-07-05", "2021-09-06", "2021-11-25", "2022-01-17",
    "2022-02-21", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24",
    "2023-01-16", "2023-02-20", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-11-23", "2024-01-15", "2024-02-19",
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

# 2026-10-25 (coordinator; audit 2026-10-25 correctness-7; results/fomcfix_2026-10-26/RULE.txt): the proximity series
# use the pre-announced schedule around March 2020.  data/releases.parquet "fomc release" holds the statements that were
# issued, and the scheduled 17-18 March 2020 meeting is not among them: the Fed replaced it with the emergency action
# announced on Sunday 15 March 2020 at about 17:00 ET.  So from the 2020-01-29 statement to that announcement
# fomc_until_inv measured the distance to 2020-04-29, which the market could not know, and after it fomc_since_inv
# counted from 2020-01-29.  With FOMC_SCHEDULE_2020_FIX on: on every bar after the 2020-01-29 statement that ends before
# the announcement, fomc_until_inv counts to the scheduled 2020-03-18 statement (its 14:30 bar end, the stamp
# convention of FOMC_MISSING_STATEMENTS); from the announcement on it counts to 2020-04-29 as before; fomc_since_inv
# counts from the first bar after the announcement (the Sunday-evening reopen, 2020-03-15 18:30) until the 2020-04-29
# statement.  No statement is added: fomc_release and fomc_day are unchanged.  Off: the frame before the flag, bit for
# bit.
FOMC_SCHEDULE_2020_FIX: bool = True
FOMC_2020_PREV: str = "2020-01-29 14:30"  # the last statement before the replaced meeting
FOMC_2020_SCHEDULED: str = "2020-03-18 14:30"  # the replaced meeting's scheduled statement (bar end)
FOMC_2020_ANNOUNCEMENT: str = "2020-03-15 17:00"  # the emergency action (Sunday, ET)
FOMC_2020_NEXT: str = "2020-04-29 14:30"  # the next statement issued

# ── Macro release calendars: fixes (MACRO_CALENDARS; see the block at MACRO_RELEASES) ─────────────────────────────
# Source of every date below (2026-10-29 schedule research): OMB's annual "Schedule of Release Dates for Principal
# Federal Economic Indicators" 2002-2024 (all 1,656 scheduled dates of the six series, frozen as issued the autumn
# before; results/macro_calendar_2026-10-29/scheduled.csv), checked against the realized dates (ALFRED, the agencies' release
# texts and notices): 27 scheduled dates of our six series moved (MACRO_RESCHEDULED), every other scheduled date was a
# release.  All six release at 08:30 ET throughout (stamp: the 09:00 bar end), the moved ones included.
# Against that calendar the file's flags 2002-01 .. its last flag are the actual releases except: three trade releases
# missing on days the file has bars, the releases on the Good Fridays the futures market did not open (no bar in the
# file: 2007-04-06, 2010-04-02, 2012-04-06 jobs; 2017-04-14 CPI and retail; 2020-04-10 CPI), and four flags a few days
# off the release (MACRO_MISDATED_RELEASES).  MACRO_CALENDAR_FIX (as FOMC_CALENDAR_FIX) applies the three tables below
# (the file is not edited, nor its columns in the frame): the releases after each calendar's last flag up to the panel
# end 2024-04-30 (the file's macro calendars stop between 2023-10-26 (GDP) and 2024-01-09 (trade)), the omitted and
# misdated ones, and the first release after the panel (the next event of the last rows' until_inv).  Before 2002 the
# file is not checked (no OMB table here); the one 1999 Good Friday jobs report is added; the file has no flag in
# 2001-09 for retail and PPI (the week of 11 September 2001), unverified, not changed.
MACRO_CALENDAR_FIX: bool = True
MACRO_MISSING_RELEASES: dict[str, tuple[str, ...]] = {
    "cpi": ("2024-01-11 09:00", "2024-02-13 09:00", "2024-03-12 09:00", "2024-04-10 09:00"),
    "emp": ("2024-01-05 09:00", "2024-02-02 09:00", "2024-03-08 09:00", "2024-04-05 09:00"),
    "ppi": ("2024-01-12 09:00", "2024-02-16 09:00", "2024-03-14 09:00", "2024-04-11 09:00"),
    "gdp": ("2024-01-25 09:00", "2024-04-25 09:00"),  # Q4 2023, Q1 2024 advance (file: last 2023-10-26)
    "trade": ("2024-02-07 09:00", "2024-03-07 09:00", "2024-04-04 09:00"),  # file: last 2024-01-09
    "retail": ("2024-01-17 09:00", "2024-02-15 09:00", "2024-03-14 09:00", "2024-04-15 09:00"),
}
MACRO_OMITTED_RELEASES: dict[str, tuple[str, ...]] = {
    "cpi": ("2017-04-14 09:00", "2020-04-10 09:00"),  # Good Friday, no futures session
    "emp": ("1999-04-02 09:00", "2007-04-06 09:00", "2010-04-02 09:00", "2012-04-06 09:00"),  # Good Friday
    "ppi": (),
    "gdp": (),
    "trade": ("2006-08-10 09:00", "2007-02-13 09:00", "2008-08-12 09:00"),  # bars in the file, no flag
    "retail": ("2017-04-14 09:00",),  # Good Friday
}
# (file flag, actual release): the file's flag is dropped and the release added
MACRO_MISDATED_RELEASES: dict[str, tuple[tuple[str, str], ...]] = {
    "cpi": (),
    "emp": (),
    "ppi": (),
    "gdp": (("2009-10-28 09:00", "2009-10-29 09:00"), ("2018-10-25 09:00", "2018-10-26 09:00")),  # Q3 advance
    "trade": (),
    "retail": (("2003-11-11 09:00", "2003-11-14 09:00"), ("2003-12-15 09:00", "2003-12-11 09:00")),
}
MACRO_NEXT_RELEASE: dict[str, str] = {
    "cpi": "2024-05-15 09:00",
    "emp": "2024-05-03 09:00",
    "ppi": "2024-05-14 09:00",
    "gdp": "2024-07-25 09:00",
    "trade": "2024-05-02 09:00",
    "retail": "2024-05-15 09:00",
}


class MacroReschedule(NamedTuple):
    """A release moved after it was scheduled (MACRO_SCHEDULE_FIX).  Stamps: naive ET; ``scheduled`` and ``actual`` are
    bar ends (09:00 for an 08:30 ET release), the two notices the time they became public (known at a bar ending at or
    after it)."""

    scheduled: str  # the original scheduled release (OMB's annual schedule)
    postponed_public: str  # the old date withdrawn: the postponement notice, or the start of a funding lapse
    new_date_public: str  # the new date (= actual) announced
    actual: str  # the release (a stamp of the calendar)
    note: str = ""


def _moved(scheduled: str, actual: str, postponed_public: str, new_date_public: str, note: str) -> MacroReschedule:
    return MacroReschedule(f"{scheduled} 09:00", postponed_public, new_date_public, f"{actual} 09:00", note)


# MACRO_SCHEDULE_FIX (the lookahead control; as FOMC_SCHEDULE_2020_FIX): the calendar holds ACTUAL dates, so without a
# fix until_inv would count, from the moment a release was due, to a date no one knew yet.  A moved release passes
# through three states: before the postponement notice, its old date; from the notice until the new date is announced,
# no date (pending: until_inv NaN before the forward fill, so its availability flag is 0, the impute-and-indicate
# path's neutral fill); after the announcement, the new date.  With the flag on this holds on the bars whose next
# release is a moved one; a postponement is public at the latest on the bar the release was due (the 09:00 bar shows it
# did not come).  since_inv counts from the last ACTUAL release (a withdrawn date is not a release); day is 1 on the
# actual date (the flag is 1 on the date of the last release or of the next release as scheduled at the bar: on every
# bar of an actual release date and on no other while each new date is public before its release day, as all were;
# the frame gate asserts the equality).  Notices (2026-10-29 schedule research, macro_schedule_sources.md 6.1-6.3):
# the start of a funding lapse is the postponement notice of every release of the closed agencies -- 2013-10-01 for
# BLS, BEA and Census, 2018-12-22 for BEA and Census (BLS was funded); a notice is public from 17:00 ET of its day
# unless the sources give a time (the change applies from the next session); where they give only a bound ("set by
# 02-28") the latest date (pending longer).  Every new date was public at least a day ahead.  Off: no record applies
# (actual dates only).
MACRO_SCHEDULE_FIX: bool = True
_LAPSE_2013 = "2013-10-01 17:00"  # lapse in appropriations 2013-10-01 .. 2013-10-16: BLS, BEA, Census
_LAPSE_2018 = "2018-12-22 17:00"  # lapse 2018-12-22 .. 2019-01-25: BEA, Census
MACRO_RESCHEDULED: dict[str, tuple[MacroReschedule, ...]] = {
    "cpi": (
        _moved("2013-10-16", "2013-10-30", _LAPSE_2013, "2013-10-17 17:00", "Sep-2013; BLS revised schedule 10-17"),
        _moved("2013-11-15", "2013-11-20", _LAPSE_2013, "2013-10-23 17:00", "Oct-2013; by 10-23 (AP list, DOL)"),
    ),
    "emp": (
        _moved("2013-10-04", "2013-10-22", _LAPSE_2013, "2013-10-17 16:38", "Sep-2013; BLS 10-17 (posted 16:38 ET)"),
        _moved("2013-11-01", "2013-11-08", _LAPSE_2013, "2013-10-17 17:00", "Oct-2013; BLS 10-17"),
    ),
    "ppi": (
        _moved("2004-02-19", "2004-03-18", "2004-02-17 17:00", "2004-03-17 17:00", "Jan-2004, NAICS conversion"),
        _moved("2004-03-12", "2004-04-01", "2004-03-05 17:00", "2004-03-31 17:00", "Feb-2004; date set by 03-31"),
        _moved("2004-04-08", "2004-04-22", "2004-04-01 17:00", "2004-04-21 17:00", "Mar-2004; date set by 04-21"),
        _moved("2004-06-11", "2004-06-17", "2004-06-07 17:00", "2004-06-16 17:00", "May-2004; EO 13343 closure"),
        _moved("2013-10-11", "2013-10-29", _LAPSE_2013, "2013-10-17 17:00", "Sep-2013; BLS 10-17"),
        _moved("2013-11-14", "2013-11-21", _LAPSE_2013, "2013-10-23 17:00", "Oct-2013; by 10-23 (DOL 13-2065)"),
    ),
    "gdp": (
        _moved("2013-10-30", "2013-11-07", _LAPSE_2013, "2013-10-21 17:00", "Q3-2013 advance; BEA 10-21"),
        _moved("2019-01-30", "2019-02-28", _LAPSE_2018, "2019-02-06 17:00", "Q4-2018 advance ('initial'); BEA 02-06"),
    ),
    "trade": (
        _moved("2004-06-11", "2004-06-14", "2004-06-07 17:00", "2004-06-11 17:00", "Apr-2004; EO 13343; by 06-11"),
        _moved("2013-10-08", "2013-10-24", _LAPSE_2013, "2013-10-21 17:00", "Aug-2013; Census/BEA 10-21"),
        _moved("2013-11-05", "2013-11-14", _LAPSE_2013, "2013-10-21 17:00", "Sep-2013; Census/BEA 10-21"),
        _moved("2019-01-08", "2019-02-06", _LAPSE_2018, "2019-01-30 17:00", "Nov-2018; BEA / Census 01-30"),
        _moved("2019-02-05", "2019-03-06", _LAPSE_2018, "2019-02-06 17:00", "Dec-2018; BEA 02-06"),
        _moved("2019-03-07", "2019-03-27", _LAPSE_2018, "2019-02-28 17:00", "Jan-2019; by 02-28 (Census)"),
        _moved("2019-04-03", "2019-04-17", _LAPSE_2018, "2019-02-28 17:00", "Feb-2019; by 02-28 (Census)"),
        _moved("2019-05-07", "2019-05-09", _LAPSE_2018, "2019-02-28 17:00", "Mar-2019; by 02-28 (Census)"),
    ),
    "retail": (
        _moved("2010-02-11", "2010-02-12", "2010-02-10 17:00", "2010-02-10 17:00", "Jan-2010; snow closure"),
        _moved("2013-10-11", "2013-10-29", _LAPSE_2013, "2013-10-21 17:00", "Sep-2013; Census 10-21"),
        _moved("2013-11-14", "2013-11-20", _LAPSE_2013, "2013-10-21 17:00", "Oct-2013; Census 10-21"),
        _moved("2019-01-16", "2019-02-14", _LAPSE_2018, "2019-02-06 17:00", "Dec-2018; Census 02-06"),
        _moved("2019-02-15", "2019-03-11", _LAPSE_2018, "2019-02-28 17:00", "Jan-2019; set by 02-28"),
        _moved("2019-03-14", "2019-04-01", _LAPSE_2018, "2019-02-28 17:00", "Feb-2019; by 02-28 (Census)"),
        _moved("2019-04-16", "2019-04-18", _LAPSE_2018, "2019-02-28 17:00", "Mar-2019; by 02-28 (Census)"),
    ),
}

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


def fomc_schedule_2020(
    t: pd.Series, event_pos: np.ndarray, until: np.ndarray, since: np.ndarray
) -> tuple[int, int]:
    """FOMC_SCHEDULE_2020_FIX on the frame's bar ends ``t`` and statement rows ``event_pos``: rewrite ``until`` and
    ``since`` in place (see the flag).  A no-op unless the frame holds the 2020-01-29 and 2020-04-29 statements, no
    statement between them, and the scheduled 2020-03-18 14:30 bar.  Returns the rows changed in each."""
    tv = pd.DatetimeIndex(pd.to_datetime(t)).to_numpy()
    prev, sched = np.datetime64(pd.Timestamp(FOMC_2020_PREV)), np.datetime64(pd.Timestamp(FOMC_2020_SCHEDULED))
    ann, nxt = np.datetime64(pd.Timestamp(FOMC_2020_ANNOUNCEMENT)), np.datetime64(pd.Timestamp(FOMC_2020_NEXT))
    if not event_pos.size:
        return 0, 0
    ev_t = tv[event_pos]
    k = int(np.searchsorted(ev_t, ann))
    ps = np.flatnonzero(tv == sched)
    if k == 0 or k >= len(event_pos) or ev_t[k - 1] != prev or ev_t[k] != nxt or not ps.size:
        return 0, 0  # not this frame's calendar: nothing to fix
    e_prev, e_next, p_sched = int(event_pos[k - 1]), int(event_pos[k]), int(ps[0])
    after_ann = np.flatnonzero(tv > ann)
    i = np.arange(len(tv))
    before = (i > e_prev) & (tv < ann)
    until[before] = 1.0 / (1.0 + (p_sched - i[before]).astype(np.float64))
    reopen = int(after_ann[0])
    later = (i >= reopen) & (i < e_next)
    since[later] = 1.0 / (1.0 + (i[later] - reopen).astype(np.float64))
    return int(before.sum()), int(later.sum())


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
    if FOMC_SCHEDULE_2020_FIX:
        # 2026-10-25 (audit correctness-7): the pre-announced schedule around the replaced March 2020 meeting
        fomc_schedule_2020(t, event_pos, until, since)
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


# ── Macro release calendars: construction (MACRO_CALENDARS) ──────────────────────────────────────────────────────


def _grid_positions(tv: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """Positions of ``stamps`` on the calendar-open grid of the frame's bar ends ``tv`` (sorted datetime64[ns]): the row
    of a stamp inside the frame (it must be a bar of the frame), ``n - 1 +`` the calendar-open bars to a stamp after the
    last bar (FOMC_NEXT_STATEMENT's rule), ``-`` the calendar-open bars from a stamp before the first bar to it."""
    n = len(tv)
    pos = np.searchsorted(tv, stamps, side="left").astype(np.int64)
    out = np.empty(len(stamps), dtype=np.int64)
    for j, s in enumerate(stamps):
        p = int(pos[j])
        if p < n and tv[p] == s:
            out[j] = p
        elif s > tv[-1]:
            out[j] = n - 1 + _calendar_open_bars_between(pd.Timestamp(tv[-1]), pd.Timestamp(s))
        elif s < tv[0]:
            out[j] = -_calendar_open_bars_between(pd.Timestamp(s), pd.Timestamp(tv[0]))
        else:
            raise ValueError(f"release stamp {pd.Timestamp(s)} is inside the frame but not one of its bars")
    return out


def _ts64(s: str | pd.Timestamp | np.datetime64) -> np.datetime64:
    return np.datetime64(pd.Timestamp(s).as_unit("ns").to_datetime64())


def macro_calendar_series(
    t: pd.Series | pd.DatetimeIndex | np.ndarray,
    stamps: Iterable[str | pd.Timestamp | np.datetime64] | pd.DatetimeIndex,
    reschedules: Sequence[MacroReschedule] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(day, until_inv, since_inv) of one release calendar on the bar ends ``t`` (the calendar-open grid, sorted).

    ``stamps``: the actual release bar stamps (any order; may lie after the frame: the next release of the last rows).
    ``reschedules``: MacroReschedule records whose ``actual`` is one of ``stamps`` (others are ignored).
    add_fomc_features' construction: until_inv = 1 / (1 + bars to the next release) (1 on the release bar), since_inv =
    1 / (1 + bars since the last release, the release bar included), NaN before the frame's first release (releases
    before the first bar are not seen, as FOMC's); rows after the calendar's last release are NaN in all three (the
    calendar is not known there: FOMC's dead tail).  The day flag is 1 on a bar whose date is the date of the last
    release at or before it or of its next release as scheduled at the bar.  Each record changes the rows whose next
    release is its ``actual``: before the postponement is public (at the latest the ``scheduled`` bar) until_inv counts
    to ``scheduled`` (and day follows it); from then until the new date is public until_inv is NaN (and no next date
    enters day); after that, unchanged.  since_inv is never changed.  Row-local: a row reads only the releases up to its
    next one and that release's notices public by the row (experiments/macro_calendar_check.py GNL-1)."""
    tv = pd.DatetimeIndex(pd.to_datetime(t)).as_unit("ns").to_numpy()
    n = len(tv)
    day = np.full(n, np.nan, dtype=np.float64)
    until = np.full(n, np.nan, dtype=np.float64)
    since = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return day, until, since
    st = np.unique(pd.DatetimeIndex(pd.to_datetime(list(stamps))).as_unit("ns").to_numpy())
    st = st[st >= tv[0]]
    if not st.size:
        return day, until, since
    sp = _grid_positions(tv, st)
    i = np.arange(n)
    nxt = np.searchsorted(sp, i, side="left")  # the first release at or after the row
    prev = np.searchsorted(sp, i, side="right") - 1  # the last release at or before the row
    ok = nxt < sp.size
    until[ok] = 1.0 / (1.0 + (sp[nxt[ok]] - i[ok]).astype(np.float64))
    ok2 = prev >= 0
    since[ok2] = 1.0 / (1.0 + (i[ok2] - sp[prev[ok2]]).astype(np.float64))
    nat = np.datetime64("NaT", "ns")
    sched = np.full(n, nat)  # the next release's date as scheduled at the row (NaT: none / unknown)
    sched[ok] = st[nxt[ok]]
    seen: set[np.datetime64] = set()
    for rec in reschedules:
        a, o = _ts64(rec.actual), _ts64(rec.scheduled)
        p_pub, n_pub = _ts64(rec.postponed_public), _ts64(rec.new_date_public)
        if not (p_pub <= n_pub <= a) or o == a:
            raise ValueError(f"inconsistent reschedule record {rec}")
        if a in seen:
            raise ValueError(f"two reschedule records for the release {rec.actual}")
        seen.add(a)
        k = int(np.searchsorted(st, a))
        if k >= st.size or st[k] != a:
            continue  # not a release of this calendar / frame
        rows = nxt == k
        postponed = min(p_pub, o)  # public at the latest on the bar the release was due
        before = rows & (tv < postponed)
        if before.any():
            p_o = int(_grid_positions(tv, np.array([o]))[0])
            until[before] = 1.0 / (1.0 + (p_o - i[before]).astype(np.float64))
            sched[before] = o
        unknown = rows & (tv >= postponed) & (tv < n_pub)
        until[unknown] = np.nan
        sched[unknown] = nat
    dd = tv.astype("datetime64[D]")
    last_day = np.full(n, np.datetime64("NaT", "D"))
    last_day[ok2] = st[prev[ok2]].astype("datetime64[D]")
    day[:] = ((dd == last_day) | (dd == sched.astype("datetime64[D]"))).astype(np.float64)
    if sp[-1] < n - 1:
        dead = i > sp[-1]
        day[dead] = until[dead] = since[dead] = np.nan
    return day, until, since


def _macro_fixed_calendar(r: str, file_stamps: pd.DatetimeIndex, include_next: bool = True) -> pd.DatetimeIndex:
    """Release r's file flags with MACRO_CALENDAR_FIX applied (sorted, unique): the misdated flags replaced by their
    releases, the missing and omitted releases added (and the next release with ``include_next``)."""
    st = pd.DatetimeIndex(file_stamps)
    if MACRO_CALENDAR_FIX:
        mis = MACRO_MISDATED_RELEASES.get(r, ())
        st = st[~st.isin(pd.to_datetime([w for w, _ in mis]))]
        add = [*MACRO_MISSING_RELEASES.get(r, ()), *MACRO_OMITTED_RELEASES.get(r, ()), *(a for _, a in mis)]
        if include_next and MACRO_NEXT_RELEASE.get(r):
            add.append(MACRO_NEXT_RELEASE[r])
        st = st.append(pd.DatetimeIndex(pd.to_datetime(add)))
    return pd.DatetimeIndex(st.unique()).sort_values()


def macro_frame_calendar(df: pd.DataFrame, r: str) -> tuple[pd.DatetimeIndex, tuple[MacroReschedule, ...]]:
    """Release r's calendar as add_macro_features reads it from a frame: (stamps, reschedule records).  The stamps are
    the frame's flags fixed by MACRO_CALENDAR_FIX (MACRO_NEXT_RELEASE included: the next event past the frame)."""
    t = pd.to_datetime(df["t"])
    raw = pd.to_numeric(df[MACRO_RELEASES[r]], errors="coerce").to_numpy(dtype=np.float64)
    stamps = _macro_fixed_calendar(r, pd.DatetimeIndex(t[np.nan_to_num(raw, nan=0.0) > 0.0]))
    recs = MACRO_RESCHEDULED.get(r, ()) if MACRO_SCHEDULE_FIX else ()
    return stamps, tuple(recs)


def add_macro_features(df: pd.DataFrame, releases: Iterable[str] | None = None) -> list[str]:
    """The macro release calendars' series (MACRO_FEATURES) on the calendar-open grid, in place; returns the names.

    Called by load_raw_data under MACRO_CALENDARS, after add_fomc_features (before the dead-session drop: distances are
    counted on the calendar-open grid).  The raw release columns are not changed.  A release whose column is missing
    is skipped."""
    if "t" not in df.columns:
        return []
    t = pd.to_datetime(df["t"])
    added: list[str] = []
    for r, col in MACRO_RELEASES.items():
        if (releases is not None and r not in releases) or col not in df.columns:
            continue
        stamps, recs = macro_frame_calendar(df, r)
        day, until, since = macro_calendar_series(t, stamps, recs)
        names = MACRO_SERIES_OF[r]
        df[names[0]], df[names[1]], df[names[2]] = day, until, since
        added += names
    return added


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@lru_cache(maxsize=4)
def _release_file_stamps(path: str) -> dict[str, pd.DatetimeIndex]:
    rel = pd.read_parquet(path, columns=["endbartime", *MACRO_RELEASES.values()])
    tt = pd.to_datetime(rel["endbartime"])
    return {
        r: pd.DatetimeIndex(tt[(pd.to_numeric(rel[c], errors="coerce").fillna(0.0) > 0.0).to_numpy()])
        for r, c in MACRO_RELEASES.items()
    }


def macro_release_stamps(r: str, path: str | None = None, include_next: bool = False) -> pd.DatetimeIndex:
    """Release r's actual release bar stamps after all fixes (bar END, naive ET; 09:00 = the 08:30 ET release), sorted:
    the file's flags (data/releases.parquet, the whole file) with, under MACRO_CALENDAR_FIX, the misdated flags moved
    to their releases and the missing and omitted releases added (MACRO_MISDATED_RELEASES, MACRO_MISSING_RELEASES,
    MACRO_OMITTED_RELEASES; MACRO_NEXT_RELEASE with ``include_next``).  The scheduled dates of moved releases
    (MACRO_RESCHEDULED) are not releases and are not among them."""
    if r not in MACRO_RELEASES:
        raise KeyError(f"unknown macro release {r!r}; valid: {list(MACRO_RELEASES)}")
    p = path or os.path.join(_REPO_ROOT, "data", "releases.parquet")
    return _macro_fixed_calendar(r, _release_file_stamps(os.path.abspath(p))[r], include_next=include_next)


def macro_release_days(r: str, path: str | None = None, include_next: bool = False) -> pd.DatetimeIndex:
    """The calendar dates (midnight) of macro_release_stamps(r)."""
    return pd.DatetimeIndex(macro_release_stamps(r, path, include_next).normalize().unique())


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
# 2026-10-29 (author; MACRO_CALENDARS): the six macro release calendars, 3 series a release (18), and the full set
# with them.  Outside ALL_FEATURES like the blocks above; the columns exist only when MACRO_CALENDARS is on.
SUBGROUPS["macro"] = list(MACRO_FEATURES)
SUBGROUPS["all_features_mac"] = list(ALL_FEATURES) + list(MACRO_FEATURES)


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
    if MACRO_CALENDARS:
        # 2026-10-29 (author): the macro release calendars, as FOMC's (see MACRO_RELEASES)
        add_macro_features(df)

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
