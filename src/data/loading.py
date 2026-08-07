"""Standalone data loading module for volatility forecasting.

Loads raw parquet data, builds a 30-min grid, filters market hours,
and returns a clean DataFrame ready for downstream pipelines.

No imports from core/ or projects/ — only numpy, pandas, os, functools.
"""

from __future__ import annotations

import os
from functools import reduce

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
    "all_features": ALL_FEATURES,
}


def get_bucket(name: str) -> list[str]:
    if name not in SUBGROUPS:
        raise KeyError(f"Unknown subgroup '{name}'. Valid: {sorted(SUBGROUPS.keys())}")
    return SUBGROUPS[name]


# ── Constants ──────────────────────────────────────────────────────────
START_DATE = "2005-01-01"
FRIDAY_CLOSE = "20:00"
SUNDAY_OPEN = "18:30"
FREQ = "30min"


def load_raw_data(
    data_path: str, allow_missing: bool = False, drop_dead_session: bool = True
) -> pd.DataFrame:
    """Load parquet data, grid to 30-min, filter market hours, clean NaNs.

    Parameters
    ----------
    data_path : str
        Path to a directory of .parquet files or a single .parquet file.
    allow_missing : bool
        If False (default), drop all rows with any remaining NaN after
        forward-filling the target column. If True, keep them.
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
    grid = pd.date_range(start=START_DATE, end=end_date, freq=FREQ)
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

    # ── 6b. Derived exog (order-flow imbalance) ───────────────────────
    # Built here, on the *unfilled* buy/sell legs, deliberately: ``apply_overnight_fills`` later
    # writes 1.0 into matching columns in the overnight window, and buy = sell = 1.0 would make the
    # imbalance exactly 0 — fabricating "perfectly balanced flow" on bars with no trading. Leaving
    # it NaN overnight lets the normal ffill / impute-and-indicate path handle it honestly.
    add_derived_features(df)

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

    return df
