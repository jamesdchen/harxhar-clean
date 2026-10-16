"""Data problems register (2026-10-15): measure every known and suspected data problem before a re-run.

User (2026-10-15): "wait until we figure out exactly what's wrong with the data."  Builds on
results/data_fix_2026-10-15/ (invalid prints, name-based sign; experiments/data_fix_2026_10_15.py),
whose data directory (hard links of data/*.parquet without the chain), 2026-10-06 local rebuild
(OLD_LOCAL) and linear harness (experiments/close_linear_pretune.py, repointed) are reused read-only.
src/ is NOT changed: every fix is applied by patching module attributes inside the processes this
script starts.  Everything it writes is under results/data_audit_2026-10-15/ (heavy files in _work/,
gitignored).  No cluster job, no commit.

Variants (switches; any subset):
  null, sign  (a) NULL_INVALID_PRINTS / SIGN_BY_NAME (the parked src fix)
  tail        (b) FOMC _active columns at their observed mode (0) on unobserved rows (dead tail)
  fid         (c) fomc_release / fomc_day passed through as 0/1 (no log, no winsorising)
  d1118       (d) the whole 2018-11-12 11:00 bar of both stock files (and OFI) set to NaN
  fcal        the three FOMC statements missing from data/releases.parquet (2023-12-13,
              2024-01-31, 2024-03-20, 14:30 bars) added, and the 2024-05-01 statement used as the
              next event for fomc_until_inv (no dead tail inside the panel)
  vix         VIX / VIX3M / VVIX after their last print (2024-02-12) spliced from the Yahoo hourly
              snapshot (data/free_feed/*_1h_yahoo.parquet: close at the hourly bar end, equal to the
              panel's print on 88-100 % of overlapping stamps), carried 30 minutes to the next bar
  ff26        executor.FFILL_LIMIT = 26 actually passed to the forward fill
  erode       ref_iqr mask erosion in bars (MA window / 46 bars a session, in rows of the design)
              instead of the per-bar design's rows (sessions)
  cmed        the neutral fill causal: expanding median of the observed adjusted values strictly
              before the row (pre-start rows: median of the first 250 observed values)
  actall      every mode-inflated series' _active column at its observed mode on unobserved rows
              (generalises tail)
  bfix        the target's diurnal baseline B without the NYSE early closes' post-13:00 bars
              (13:30..17:00): the trailing 20 regular in-slot observations
  burnin      the 3,125-row HAR burn-in dropped in bar space (full grid) before slicing to one
              bar a session, not after (design rows aligned to the record's dates)

Stages:  python experiments/data_audit_2026_10_15.py <stage> [args]
  capture <variant> <segment> <bucket>   one design into _work/design_<variant>/
  designs <variant> [...]                bar1600 + last30 all_features (+ baseline where needed)
  diff <variant> [...]                   per-column change vs OLD_LOCAL (design_diff.csv)
  linear <variant>                       the pre-tuned f arm (ridge / lasso / elastic net) on the
                                         variant's design (close_linear_pretune, f arm only)
  score <variant> [...]                  QLIKE, DM vs the record, Sharpe mid / crossed, positions
  (analysis stages: feeds, active, lookahead, erosion, scoring, dups, burnin_n, ticks, align, grid)
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
if (
    os.name == "nt"
):  # laptop rule (2026-10-15): below-normal priority; pool workers inherit it
    import ctypes

    _k32 = ctypes.windll.kernel32
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.SetPriorityClass(
        ctypes.c_void_p(_k32.GetCurrentProcess()), 0x00004000
    )  # BELOW_NORMAL

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "data_audit_2026-10-15"
WORK = OUT / "_work"
FIX = REPO / "results" / "data_fix_2026-10-15"
# hard links of data/*.parquet minus the chain, plus free_feed/ (the src VIX splice reads
# <data_path>/free_feed); every parquet is the same file as in data/
DATA = OUT / "_work" / "data_nochain"
RECORD = REPO / "results" / "close_studies_2026-10-03" / "forecasts"
RECORD_DESIGN = RECORD / "design_bar1600_all_features.parquet"
OLD_LOCAL = REPO / "results" / "linear_pretune_2026-10-06" / "_work" / "design"
OLD_STUDY = REPO / "results" / "linear_pretune_2026-10-06"
FREE = REPO / "data" / "free_feed"
DESIGN_META = ("row", "date", "forecast_row", "y", "baseline", "true_raw")
PANEL_END = pd.Timestamp("2024-04-30 23:30")
BARS_PER_SESSION = 46  # the RV grid's bars a session (23 h of 30-minute bars)

SWITCHES = (
    "null",
    "sign",
    "tail",
    "fid",
    "d1118",
    "fcal",
    "vix",
    "ff26",
    "erode",
    "cmed",
    "actall",
    "bfix",
    "burnin",
    "ecdrop",
    "spk",
)
VARIANTS: dict[str, tuple[str, ...]] = {
    "old": (),
    "bc": ("tail", "fid"),
    "abc": ("null", "sign", "tail", "fid"),
    "abcd": ("null", "sign", "tail", "fid", "d1118"),
}
_BASE = VARIANTS["abcd"]
for _s in (
    "fcal",
    "vix",
    "ff26",
    "erode",
    "cmed",
    "actall",
    "bfix",
    "burnin",
    "ecdrop",
    "spk",
):
    VARIANTS[f"abcd_{_s}"] = _BASE + (_s,)
# 'all' is set after the single-switch results are read (see VARIANTS_EXTRA, filled by the user of this
# script on the command line as abcd+<s1>+<s2>...)


def variant_switches(variant: str) -> tuple[str, ...]:
    if variant in VARIANTS:
        return VARIANTS[variant]
    if "+" in variant:  # e.g. abcd+fcal+vix
        head, *rest = variant.split("+")
        sw = VARIANTS[head] + tuple(rest)
        assert set(sw) <= set(SWITCHES), sw
        return sw
    raise KeyError(variant)


def needs_baseline(variant: str) -> bool:
    if variant in SRC_SETS:  # a src-built design set
        return bool({"bfix", "burnin", "ecdrop"} & SRC_SETS[variant])
    sw = variant_switches(variant)
    return "bfix" in sw or "burnin" in sw or "ecdrop" in sw


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def design_dir(variant: str) -> Path:
    return WORK / f"design_{variant}"


def load_npz(p: Path) -> dict:
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ============================================================================ variant machinery
FOMC_EXTRA = ("2023-12-13 14:30", "2024-01-31 14:30", "2024-03-20 14:30")
FOMC_NEXT = pd.Timestamp("2024-05-01 14:30")  # the first statement after the panel
D1118 = pd.Timestamp("2018-11-12 11:00")
VIX_SLOTS = {  # the panel's own print slots in 2023-12 .. 2024-02 (bar-end labels)
    "vix": ("04:00", "17:00"),
    "vix3m": ("10:00", "16:30"),
    "vvix": ("10:00", "16:00"),
}


def yahoo_bar_ends(name: str) -> pd.Series:
    """Yahoo hourly close keyed by the naive-ET bar END (open stamp + 1 h)."""
    y = pd.read_parquet(FREE / f"{name}_1h_yahoo.parquet")
    end = pd.to_datetime(y["t"]).dt.tz_convert("America/New_York").dt.tz_localize(
        None
    ) + pd.Timedelta("1h")
    return pd.Series(
        y["close"].to_numpy(float), index=pd.DatetimeIndex(end)
    ).sort_index()


def splice_vix(df: pd.DataFrame) -> dict:
    """Fill vix / vix3m / vvix after their last print from the Yahoo hourly snapshot (in place)."""
    out = {}
    t = pd.DatetimeIndex(df["t"])
    for col in ("vix", "vix3m", "vvix"):
        last = t[df[col].notna().to_numpy()].max()
        ys = yahoo_bar_ends(col)
        ys = ys[(ys.index > last) & (ys.index <= PANEL_END)]
        # carry each hourly value to the next half-hour bar (30 minutes stale at most)
        carry = pd.Series(ys.to_numpy(), index=ys.index + pd.Timedelta("30min"))
        both = pd.concat([ys, carry[~carry.index.isin(ys.index)]]).sort_index()
        lo, hi = VIX_SLOTS[col]
        hhmm = both.index.strftime("%H:%M")
        both = both[(hhmm >= lo) & (hhmm <= hi)]
        m = (t > last) & t.isin(both.index)
        df.loc[m, col] = both.reindex(t[m]).to_numpy()
        out[col] = int(m.sum())
    return out


def add_fomc_features_cal(df: pd.DataFrame, _orig=None) -> list[str]:
    """add_fomc_features with the three missing 2023-12 .. 2024-03 statements and the 2024-05-01
    statement as the next event (beyond the grid) for fomc_until_inv."""
    import src.data.loading as LD

    raw = LD.FOMC_RAW_COL
    t = pd.to_datetime(df["t"])
    for s in FOMC_EXTRA:
        m = (t == pd.Timestamp(s)).to_numpy()
        assert m.sum() == 1, s
        df.loc[m, raw] = 1.0
    names = _orig(df)
    rel = pd.to_numeric(df[raw], errors="coerce").fillna(0.0).to_numpy()
    unsch = t.dt.normalize().isin(LD.UNSCHEDULED_FOMC_DAYS).to_numpy()
    ev = np.flatnonzero((rel > 0) & ~unsch)
    last = int(ev[-1])
    n = len(df)
    # grid position of the next statement: the calendar-open 30-minute bars after the last grid row
    gap = int(len(pd.date_range(t.iloc[-1], FOMC_NEXT, freq="30min")) - 1)
    pos_next = n - 1 + gap
    i = np.arange(n)
    dead = i > last
    df.loc[dead, "fomc_release"] = 0.0
    df.loc[dead, "fomc_day"] = (
        (t[dead].dt.normalize() == t.iloc[last].normalize()).astype(float).to_numpy()
    )
    df.loc[dead, "fomc_until_inv"] = 1.0 / (1.0 + (pos_next - i[dead]).astype(float))
    df.loc[dead, "fomc_since_inv"] = 1.0 / (1.0 + (i[dead] - last).astype(float))
    return names


def target_bfix(df: pd.DataFrame, winsor_window: int) -> tuple[pd.Series, pd.Series]:
    """robust_transform(df, 'RV', is_target=True, use_diurnal=True, winsor_window) with the
    diurnal baseline taken over REGULAR in-slot observations only (NYSE early closes' 13:30..17:00
    bars excluded from every slot's trailing mean).  With no early close it equals the original."""
    import atm_straddle_lib as asl
    import src.features.transforms.target as TG

    t = pd.to_datetime(df["t"])
    mins = t.dt.hour * 60 + t.dt.minute
    ec = t.dt.normalize().isin(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    post = (ec & (mins > 13 * 60) & (mins <= 17 * 60)).to_numpy()
    rv = df["RV"].astype(np.float64)
    slot = df["time_of_day"]
    base = pd.Series(np.nan, index=df.index)
    for _, g in rv.groupby(slot):
        reg = g[~post[g.index.to_numpy()]]
        b = (
            reg.rolling(TG.DIURNAL_WINDOW, min_periods=TG.DIURNAL_MIN_PERIODS)
            .mean()
            .shift(1)
        )
        full = b.reindex(g.index)
        pr = post[g.index.to_numpy()]
        full[pr] = full.bfill()[
            pr
        ]  # a post-close row: the next regular row's B (data before it only)
        base.loc[g.index] = full.to_numpy()
    base = base.replace(0, 1.0).fillna(1.0)
    series = np.sqrt(rv / base)
    series = TG.rolling_winsorize(
        series, window=winsor_window, allow_missing=False, is_target=True
    )
    return series, base


def slot_z(v: pd.Series, slot, win: int = 250, minp: int = 60) -> pd.Series:
    """Robust z of v against the same clock slot's trailing win observations (strictly before)."""
    g = v.groupby(slot)
    med = g.transform(lambda q: q.rolling(win, min_periods=minp).median().shift(1))
    dev = (v - med).abs()
    mad = dev.groupby(slot).transform(
        lambda q: q.rolling(win, min_periods=minp).median().shift(1)
    )
    return (v - med) / (1.4826 * mad.where(mad > 0))


def slot_z_expanding(v: pd.Series, slot, minp: int = 60) -> pd.Series:
    """Robust z of v against ALL earlier observations of the same clock slot (expanding median / MAD,
    strictly before): a multi-year contaminated stretch cannot drag its own baseline along."""
    g = v.groupby(slot)
    med = g.transform(lambda q: q.expanding(min_periods=minp).median().shift(1))
    dev = (v - med).abs()
    mad = dev.groupby(slot).transform(
        lambda q: q.expanding(min_periods=minp).median().shift(1)
    )
    return (v - med) / (1.4826 * mad.where(mad > 0))


SPIKE_Z, ES_CALM_Z = 4.0, 3.0


def stock_spike_flags(df: pd.DataFrame) -> np.ndarray:
    """Stock-file bars whose cross-section RV (EW or VW) is extreme RELATIVE TO ES's RV for the clock
    slot (robust z > 4 against the slot's expanding history) while ES's own RV is ordinary for the
    slot (|z| < 3): one constituent's bad print moves the per-stock RV average, ES does not move.
    Calibrated on the daily RV coherence (ES vs VW stocks, log daily RTH RV): 0-5 bars a year flagged
    in 2008-2014 and 2021-2024; 266-543 a year in 2015-2018, where removing them lifts the daily
    correlation from 0.25-0.45 back to 0.78-0.84."""
    t = pd.to_datetime(df["t"])
    slot = t.dt.strftime("%H:%M")
    rv = df["RV"].astype(float)
    z_rv = slot_z_expanding(np.log(rv.where(rv > 0)), slot)
    flag = np.zeros(len(df), bool)
    for w in ("vwstock", "ewstock"):
        x = df[f"sumret2_{w}"].astype(float)
        zq = slot_z_expanding(np.log(x.where(x > 0) / rv), slot)
        flag |= ((zq > SPIKE_Z) & (z_rv.abs() < ES_CALM_Z)).fillna(False).to_numpy()
    return flag


def causal_fill(adj: pd.Series) -> pd.Series:
    """Expanding median of the observed adjusted values strictly before each row; before the
    first print, the first observed value (2026-10-15: the rule adopted in src, so that no
    window that serves a forecast sees a later value; it was the median of the first 250)."""
    em = adj.expanding(min_periods=1).median().shift(1)
    first = adj.first_valid_index()
    return em.fillna(float(adj.loc[first]) if first is not None else 0.0)


# the src flags of the 2026-10-15 fixes (module, attribute) -> the audit switch they implement
SRC_FLAGS: dict[tuple[str, str], str] = {
    ("loading", "NULL_INVALID_PRINTS"): "null",
    ("target", "SIGN_BY_NAME"): "sign",
    ("target", "FOMC_FLAGS_IDENTITY"): "fid",
    ("executor", "APPLY_FFILL_LIMIT"): "ff26",
    ("loading", "SPLICE_VIX_FREE_FEED"): "vix",
    ("loading", "FOMC_CALENDAR_FIX"): "fcal",
    ("executor", "ACTIVE_FROM_OBSERVED"): "actall",
    ("executor", "CAUSAL_NEUTRAL_FILL"): "cmed",
    ("executor", "B_REGULAR_BARS_ONLY"): "bfix",
    ("executor", "DROP_EARLY_CLOSE_SESSIONS"): "ecdrop",
    ("executor", "ERODE_IN_BARS"): "erode",
    ("executor", "BURNIN_IN_BAR_SPACE"): "burnin",
    # the look-ahead rule's statistics with no audit switch (new in src)
    ("executor", "CAUSAL_MODE_INFLATION"): "causal_mode",
    ("executor", "CAUSAL_GUARD_MODE"): "causal_guard",
    ("target", "CAUSAL_SD_FLOOR"): "causal_floor",
    ("target", "CAUSAL_ASINH_ROUTING"): "causal_route",
}


def _src_modules() -> dict:
    import src.backtest.executor as EX
    import src.data.loading as LD
    import src.features.transforms.target as TG

    return {"loading": LD, "target": TG, "executor": EX}


def src_flags_off() -> None:
    mods = _src_modules()
    for (m, a), _ in SRC_FLAGS.items():
        if hasattr(mods[m], a):
            setattr(mods[m], a, False)


def set_src_flags(on: set[str]) -> dict:
    """Set every 2026-10-15 src flag from the switch names in ``on``; return the settings."""
    mods = _src_modules()
    out = {}
    for (m, a), sw in SRC_FLAGS.items():
        setattr(mods[m], a, sw in on)
        out[f"{m}.{a}"] = sw in on
    return out


SRC_SETS: dict[str, set[str]] = {
    "srcoff": set(),  # gate a: every flag off
    # gate b: the switches the audit measured, causal statistics without a switch off
    "srcb": {
        "null",
        "sign",
        "fid",
        "ff26",
        "vix",
        "fcal",
        "actall",
        "cmed",
        "bfix",
        "ecdrop",
        "erode",
        "burnin",
    },
    # designs v2: all 8 adopted fixes on
    "v2": set(SRC_FLAGS.values()),
}


def capture_src(name: str, segment: str, bucket: str) -> None:
    """One design built by src itself (no machinery patch) under the flag set SRC_SETS[name]."""
    os.environ["CLOSE_DESIGN_DIR"] = str(design_dir(name))
    os.chdir(REPO)
    flags = set_src_flags(SRC_SETS[name])
    print(f"[audit] src flags {name}: {flags}", flush=True)
    import capture_design_close as cdc

    run = cdc.ex.run_executor

    def run_nochain(*args, **kw):
        kw["data_path"] = str(data_dir())
        return run(*args, **kw)

    cdc.ex.run_executor = run_nochain
    t0 = time.time()
    cdc.capture(segment, bucket)
    (design_dir(name) / "src_flags.json").write_text(json.dumps(flags, indent=1))
    say(f"capture_src {name} {segment} {bucket}: {time.time() - t0:.0f}s")


def data_dir() -> Path:
    """Hard links (copies if linking fails) to data/*.parquet minus the chain, plus free_feed/."""
    (DATA / "free_feed").mkdir(parents=True, exist_ok=True)
    pairs = [
        (f, DATA / f.name)
        for f in sorted((REPO / "data").glob("*.parquet"))
        if f.name not in ("spxw_chain.parquet", "spxw_spot.parquet")
    ]
    pairs += [
        (f, DATA / "free_feed" / f.name)
        for f in sorted(FREE.glob("*_1h_yahoo.parquet"))
    ]
    for f, dst in pairs:
        if dst.is_file() and dst.stat().st_size == f.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
    return DATA


def set_variant(variant: str) -> None:
    """Patch src modules of THIS process for the variant's switches (src files are not touched)."""
    import src.backtest.executor as EX
    import src.data.loading as LD
    import src.features.transforms.target as TG

    sw = set(variant_switches(variant))
    src_flags_off()  # the machinery measures each switch against the pre-2026-10-15 chain
    LD.NULL_INVALID_PRINTS = "null" in sw
    TG.SIGN_BY_NAME = "sign" in sw
    if getattr(EX, "_audit_patched", None) is not None:
        assert EX._audit_patched == variant
        return
    EX._audit_patched = variant
    if "fcal" in sw:
        orig_fomc = LD.add_fomc_features
        LD.add_fomc_features = lambda df: add_fomc_features_cal(df, _orig=orig_fomc)
    orig_lrd = EX.load_raw_data

    def load_raw_data(path, **kw):
        df = orig_lrd(path, **kw)
        if "d1118" in sw:
            m = (df["t"] == D1118).to_numpy()
            assert m.sum() == 1
            cols = [
                c
                for c in df.columns
                if c.endswith("_ewstock") or c.endswith("_vwstock")
            ]
            df.loc[m, cols] = np.nan
        if "spk" in sw:
            m = stock_spike_flags(df)
            cols = [
                c
                for c in df.columns
                if c.endswith("_ewstock") or c.endswith("_vwstock")
            ]
            df.loc[m, cols] = np.nan
            print(f"[audit] stock-file spike bars nulled: {int(m.sum())}", flush=True)
        if "vix" in sw:
            print(f"[audit] vix splice: {splice_vix(df)}", flush=True)
        return df

    EX.load_raw_data = load_raw_data
    orig_rt = EX.robust_transform

    def robust_transform(df, col, **kw):
        if "fid" in sw and col in ("fomc_release", "fomc_day"):
            return df[col].astype(np.float64).copy(), pd.Series(1.0, index=df.index)
        return orig_rt(df, col, **kw)

    EX.robust_transform = robust_transform

    def load_and_transform(
        data_path,
        exog_cols,
        *,
        target_use_diurnal,
        target_winsor_window,
        dropna_with_exog,
        overnight_fill=True,
        impute_indicate=False,
        diurnal_mode="divide",
        use_semantic=True,
        ffill_limit=None,
        legacy_avail=False,
        neutralise_unobserved=True,
    ):
        """executor.load_and_transform, line for line, with the variant's switches."""
        if "ff26" in sw:
            ffill_limit = EX.FFILL_LIMIT
        df = EX.load_raw_data(data_path, allow_missing=True)
        if exog_cols:
            if overnight_fill:
                EX.apply_overnight_fills(df, exog_cols)
            obs_cols = {col: df[col].notna().to_numpy(copy=True) for col in exog_cols}
            df[exog_cols] = df[exog_cols].ffill(limit=ffill_limit)
            for col in exog_cols:
                df[f"{col}__obs"] = (
                    obs_cols[col] if not legacy_avail else df[col].notna().to_numpy()
                )
            if dropna_with_exog and not impute_indicate:
                df = df.dropna(subset=["RV"] + exog_cols).reset_index(drop=True)
            else:
                df = df.dropna(subset=["RV"]).reset_index(drop=True)
        if "bfix" in sw:
            assert target_use_diurnal and target_winsor_window is not None
            adj_rv, baseline = target_bfix(df, target_winsor_window)
        else:
            kw: dict = {"is_target": True}
            if target_use_diurnal:
                kw["use_diurnal"] = True
            if target_winsor_window is not None:
                kw["winsor_window"] = target_winsor_window
            adj_rv, baseline = EX.robust_transform(df, "RV", **kw)
        df["adj_RV"] = adj_rv
        df["baseline"] = baseline
        adj_exog_cols: list[str] = []
        for col in exog_cols:
            adj_series, _ = EX.robust_transform(
                df,
                col,
                use_transform=use_semantic,
                use_diurnal=True,
                diurnal_mode=diurnal_mode,
            )
            df[f"adj_{col}"] = adj_series
            adj_exog_cols.append(f"adj_{col}")
        if impute_indicate and exog_cols:
            for col in exog_cols:
                adj = df[f"adj_{col}"]
                obs = df[f"{col}__obs"].to_numpy().astype(bool)
                if neutralise_unobserved:
                    adj = adj.where(pd.Series(obs, index=adj.index), np.nan)
                fill = causal_fill(adj) if "cmed" in sw else adj.median()
                df[f"adj_{col}"] = adj.fillna(fill)
                df[f"{col}_avail"] = obs.astype("float64")
                adj_exog_cols.append(f"{col}_avail")
        for col in exog_cols:
            vals = df[col].to_numpy(dtype=np.float64)
            inflated, mode = EX.mode_inflated(vals)
            if inflated:
                act = (np.nan_to_num(vals, nan=mode) != mode).astype("float64")
                if "actall" in sw:
                    # class and flag from OBSERVED prints only: a series whose observed prints are not
                    # mode-inflated (voldemand: 14-16 % zeros observed, 59-60 % after the fill) gets
                    # a constant column (dropped by every fit's identifiability mask / window_keep)
                    obs = df[f"{col}__obs"].to_numpy(bool)
                    infl_o, mode_o = EX.mode_inflated(vals[obs])
                    if not infl_o:
                        act = np.zeros(len(vals))
                    else:
                        a_o = (vals[obs] != mode_o).astype(float)
                        act = np.full(len(vals), float(np.round(a_o.mean())))
                        act[obs] = a_o
                elif "tail" in sw and col.startswith("fomc_"):
                    obs = df[f"{col}__obs"].to_numpy(bool)
                    if obs.any() and not obs.all():
                        act = np.where(obs, act, float(np.round(act[obs].mean())))
                df[f"{col}_active"] = act
                adj_exog_cols.append(f"{col}_active")
        return df, adj_exog_cols

    EX.load_and_transform = load_and_transform

    if "erode" in sw:
        orig_guard = EX._build_scale_guards

        def build_scale_guards(
            X,
            df,
            feature_names,
            zero_inflated_active=True,
            erode_ma_window=True,
            warmup_floor=1.0,
            ma_rows_per_session=None,
        ):
            """_build_scale_guards with the MA window converted from bars to the design's rows."""
            assert (
                ma_rows_per_session is None
            )  # src's own erosion (ERODE_IN_BARS) is off here
            rows_per_session = len(df) / max(
                1, pd.to_datetime(df["t"]).dt.normalize().nunique()
            )
            renamed = []
            for name in feature_names:
                m = re.match(r"adj_(.+)_ma_(\d+)$", name)
                if m is None:
                    renamed.append(name)
                    continue
                w = int(m.group(2))
                w_rows = max(1, math.ceil(w * rows_per_session / BARS_PER_SESSION))
                renamed.append(f"adj_{m.group(1)}_ma_{w_rows}")
            ref_b, fixed = orig_guard(
                X,
                df,
                renamed,
                zero_inflated_active=zero_inflated_active,
                erode_ma_window=erode_ma_window,
                warmup_floor=warmup_floor,
            )
            ref_s, _ = orig_guard(
                X,
                df,
                feature_names,
                zero_inflated_active=zero_inflated_active,
                erode_ma_window=erode_ma_window,
                warmup_floor=warmup_floor,
            )
            # the mechanism, for the erosion stage: both references and the unscaled matrix
            tag = f"{len(df)}x{len(feature_names)}"
            np.savez_compressed(
                WORK / f"guards_{variant}_{tag}.npz",
                names=np.array(feature_names),
                ref_bars=ref_b,
                ref_sessions=ref_s,
                X=X,
                date=pd.to_datetime(df["t"])
                .dt.strftime("%Y-%m-%d %H:%M")
                .to_numpy()
                .astype("U16"),
            )
            return ref_b, fixed

        EX._build_scale_guards = build_scale_guards

    if "ecdrop" in sw:
        import atm_straddle_lib as asl

        ecd = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
        orig_slice0 = EX.slice_to_segment

        def slice_no_ec(df, segment):
            out = orig_slice0(df, segment)
            keep = ~pd.to_datetime(out["t"]).dt.normalize().isin(ecd)
            return out.loc[keep.to_numpy()].reset_index(drop=True)

        EX.slice_to_segment = slice_no_ec

    if "burnin" in sw:
        orig_slice = EX.slice_to_segment
        lags = list(EX.resolve_har_lags())
        burn = lags[-1]
        EX.resolve_har_lags = lambda *a, **k: (
            lags[:-1] + [0]
        )  # _backtest_and_save drops nothing more

        def slice_to_segment(df, segment):
            return orig_slice(df.iloc[burn:].reset_index(drop=True), segment)

        EX.slice_to_segment = slice_to_segment


# ============================================================================ designs
def capture(variant: str, segment: str, bucket: str) -> None:
    os.environ["CLOSE_DESIGN_DIR"] = str(design_dir(variant))
    os.chdir(REPO)
    set_variant(variant)
    import capture_design_close as cdc

    assert cdc.OUT == design_dir(variant), cdc.OUT
    run = cdc.ex.run_executor

    def run_nochain(*args, **kw):
        assert kw.get("data_path") == "data", kw.get("data_path")
        kw["data_path"] = str(DATA)
        return run(*args, **kw)

    cdc.ex.run_executor = run_nochain
    t0 = time.time()
    data_dir()
    cdc.capture(segment, bucket)
    say(f"capture {variant} {segment} {bucket}: {time.time() - t0:.0f}s")


def design_jobs(variant: str) -> list[tuple[str, str, str]]:
    jobs = [(variant, s, "all_features") for s in ("bar1600", "last30")]
    if needs_baseline(variant):
        jobs += [(variant, s, "baseline") for s in ("bar1600", "last30")]
    return jobs


def designs(variants: list[str], parallel: int = 3) -> None:
    jobs = [j for v in variants for j in design_jobs(v)]
    todo = [
        j
        for j in jobs
        if not (design_dir(j[0]) / f"design_{j[1]}_{j[2]}.npz").is_file()
    ]
    say(f"designs: {len(todo)} of {len(jobs)} to capture ({parallel} at a time)")
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for v, s, b in todo:
        while sum(p.poll() is None for p, _, _ in procs) >= parallel:
            time.sleep(5)
        fh = open(logs / f"capture_{v}_{s}_{b}.log", "w")
        procs.append(
            (
                subprocess.Popen(
                    [sys.executable, __file__, "capture", v, s, b],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO),
                ),
                fh,
                (v, s, b),
            )
        )
    bad = []
    for p, fh, j in procs:
        rc = p.wait()
        fh.close()
        if rc != 0:
            bad.append(j)
    say(f"designs done; failed: {bad}")


# ============================================================================ diff
def diff(variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        for s in ("bar1600", "last30"):
            f = design_dir(v) / f"design_{s}_all_features.npz"
            if not f.is_file():
                continue
            new = load_npz(f)
            ref = load_npz(OLD_LOCAL / f"design_{s}_all_features.npz")
            nn = [str(x) for x in new["names"]]
            assert nn == [str(x) for x in ref["names"]], (v, s)
            new_d = pd.to_datetime(new["date"])
            ref_d = pd.to_datetime(ref["date"])
            common = new_d.intersection(ref_d)
            ia, ib = new_d.get_indexer(common), ref_d.get_indexer(common)
            Xn, Xr = new["X"][ia], ref["X"][ib]
            W = int(ref["W"])
            fr = np.zeros(len(common), bool)
            fr[np.searchsorted(common, ref_d[W]) :] = True
            test = np.asarray(common >= pd.Timestamp("2020-01-03"))
            dy = np.abs(new["y"][ia] - ref["y"][ib])
            rows.append(
                dict(
                    variant=v,
                    segment=s,
                    column="__y__",
                    rows_changed=int((dy > 0).sum()),
                    forecast_rows_changed=int((dy[fr] > 0).sum()),
                    test_rows_changed=int((dy[test] > 0).sum()),
                    max_abs=float(dy.max()),
                    mean_abs_forecast_rows=float(dy[fr].mean()),
                )
            )
            db = np.abs(new["baseline"][ia] / ref["baseline"][ib] - 1)
            rows.append(
                dict(
                    variant=v,
                    segment=s,
                    column="__baseline__",
                    rows_changed=int((db > 0).sum()),
                    forecast_rows_changed=int((db[fr] > 0).sum()),
                    test_rows_changed=int((db[test] > 0).sum()),
                    max_abs=float(db.max()),
                    mean_abs_forecast_rows=float(db[fr].mean()),
                )
            )
            D = np.abs(Xn - Xr)
            for j, c in enumerate(nn):
                d = D[:, j]
                if not (d > 0).any():
                    continue
                ch = d > 0
                rows.append(
                    dict(
                        variant=v,
                        segment=s,
                        column=c,
                        rows_changed=int(ch.sum()),
                        forecast_rows_changed=int(ch[fr].sum()),
                        test_rows_changed=int(ch[test].sum()),
                        max_abs=float(d.max()),
                        mean_abs_forecast_rows=float(d[fr].mean()),
                        first_changed=str(common[np.flatnonzero(ch)[0]])[:16],
                        last_changed=str(common[np.flatnonzero(ch)[-1]])[:16],
                        corr_old_new=float(np.corrcoef(Xn[fr, j], Xr[fr, j])[0, 1])
                        if np.std(Xr[fr, j]) > 0 and np.std(Xn[fr, j]) > 0
                        else np.nan,
                    )
                )
    R = pd.DataFrame(rows)
    f = OUT / "design_diff.csv"
    if f.is_file():
        old = pd.read_csv(f)
        old = old[~old["variant"].isin(variants)]
        R = pd.concat([old, R], ignore_index=True)
    R.to_csv(f, index=False)
    say("design diff written")
    return R


# ============================================================================ linear (f arm only)
def clp_patch(variant: str):
    import close_linear_pretune as CLP

    root = WORK / f"linear_{variant}"
    CLP.OUT = root
    CLP.WORK = root / "_work"
    CLP.DESIGN_DIR = design_dir(variant)
    CLP.ARR = CLP.WORK / "arrays"
    CLP.WALK = CLP.WORK / "walk"
    CLP.FOLDS = CLP.WORK / "folds"
    CLP.EXPORT = root / "export"
    CLP.NETLIB = CLP.WORK / "netlib"
    CLP.KERNEL_DLL = CLP.WORK / "close_exogpen_kernel.dll"
    CLP.LAPACK = "netlib"
    CLP.BUCKETS = (
        ("all_features", "baseline") if needs_baseline(variant) else ("all_features",)
    )
    if not getattr(CLP, "_audit_guard", False):
        _ORIG["run_task"] = CLP.run_task
        CLP.run_task = guarded_run_task
        CLP._audit_guard = True
    return CLP


_ORIG: dict = {}


def guarded_run_task(job: dict) -> dict:
    import close_linear_pretune as CLP

    assert str(CLP.WALK).startswith(str(WORK)) and str(CLP.FOLDS).startswith(
        str(WORK)
    ), CLP.WALK
    return _ORIG["run_task"](job)


if os.environ.get(
    "AUDIT_VARIANT"
):  # pool workers: same patch before any task is unpickled
    clp_patch(os.environ["AUDIT_VARIANT"])


def aligned(variant: str, s: str, b: str) -> dict:
    """The variant's design on the record's rows (burnin: the extended design cut to the record's
    dates; otherwise the design itself)."""
    z = load_npz(design_dir(variant) / f"design_{s}_{b}.npz")
    if variant in SRC_SETS or "burnin" not in variant_switches(variant):
        return z
    ref = load_npz(OLD_LOCAL / f"design_{s}_{b}.npz")
    dz, dr = pd.to_datetime(z["date"]), pd.to_datetime(ref["date"])
    if s == "bar1600":
        idx = dz.get_indexer(dr)
        assert (idx >= 0).all()
    else:  # last30: the pre-tune reads the 16:00 rows before 2020; keep the record's first date on
        idx = np.flatnonzero(np.asarray(dz >= dr[0]))
    out = dict(z)
    for k in ("X", "y", "baseline"):
        out[k] = z[k][idx]
    out["date"] = z["date"][idx]
    W0 = int(ref["W"])
    out["W"] = np.array(W0)
    return out


def composite(variant: str) -> Path:
    """The record's 16:00 design with every column (and y, baseline) the variant changes against
    OLD_LOCAL replaced; the variant's design npz files written where CLP.prep reads them."""
    exp_dir = WORK / f"linear_{variant}" / "export"
    exp_dir.mkdir(parents=True, exist_ok=True)
    rec = pd.read_parquet(RECORD_DESIGN)
    names = [c for c in rec.columns if c not in DESIGN_META]
    new = aligned(variant, "bar1600", "all_features")
    ref = load_npz(OLD_LOCAL / "design_bar1600_all_features.npz")
    own_names = [str(x) for x in new["names"]]
    if (
        own_names != names
        or len(new["date"]) != len(ref["date"])
        or not (new["date"] == ref["date"]).all()
    ):
        return own_export(variant, new, own_names, exp_dir)
    ch = np.flatnonzero((new["X"] != ref["X"]).any(axis=0))
    out = rec.copy()
    for j in ch:
        out[names[j]] = new["X"][:, j]
    y_changed = not np.array_equal(new["y"], ref["y"])
    if y_changed:
        out["y"] = new["y"]
    out["baseline"] = new["baseline"]
    import close_studies_export as CSE

    CSE.write_parquet(out, exp_dir / "design_bar1600_all_features.parquet")
    tg = pd.read_parquet(RECORD / "targets.parquet")
    W = int(ref["W"])
    tg["baseline"] = new["baseline"][W:]
    tg.to_parquet(exp_dir / "targets.parquet", index=False)
    (exp_dir / "replaced_columns.json").write_text(
        json.dumps(dict(columns=[names[j] for j in ch], y=y_changed))
    )
    # the arrays CLP.prep reads (aligned rows for burnin)
    dd = WORK / f"linear_{variant}" / "design"
    dd.mkdir(parents=True, exist_ok=True)
    for s in ("bar1600", "last30"):
        for b in (
            ("all_features", "baseline")
            if needs_baseline(variant)
            else ("all_features",)
        ):
            z = aligned(variant, s, b)
            np.savez(dd / f"design_{s}_{b}.npz", **z)
    if not needs_baseline(
        variant
    ):  # CLP.prep reads the backbone names from it; no exog: unaffected
        shutil.copy2(
            OLD_LOCAL / "design_bar1600_baseline.npz",
            dd / "design_bar1600_baseline.npz",
        )
    say(f"composite {variant}: {len(ch)} columns replaced, y changed {y_changed}")
    return exp_dir


def own_export(variant: str, new: dict, names: list[str], exp_dir: Path) -> Path:
    """A variant with other rows than the record (ecdrop): the export is the variant's design itself."""
    W = int(new["W"])
    n = len(new["y"])
    out = pd.DataFrame(new["X"], columns=names)
    meta = pd.DataFrame(
        dict(
            row=np.arange(n),
            date=pd.to_datetime(new["date"]),
            forecast_row=np.where(np.arange(n) >= W, np.arange(n) - W, -1),
            y=new["y"],
            baseline=new["baseline"],
        )
    )
    meta["true_raw"] = np.where(
        meta["forecast_row"] >= 0, new["y"] ** 2 * new["baseline"], np.nan
    )
    out = pd.concat([meta, out], axis=1)
    import close_studies_export as CSE

    CSE.write_parquet(out, exp_dir / "design_bar1600_all_features.parquet")
    tg = meta[meta["forecast_row"] >= 0][["row", "date", "baseline"]].reset_index(
        drop=True
    )
    tg.to_parquet(exp_dir / "targets.parquet", index=False)
    (exp_dir / "replaced_columns.json").write_text(
        json.dumps(dict(columns="own design", rows=n))
    )
    dd = WORK / f"linear_{variant}" / "design"
    dd.mkdir(parents=True, exist_ok=True)
    for s in ("bar1600", "last30"):
        for b in (
            ("all_features", "baseline")
            if needs_baseline(variant)
            else ("all_features",)
        ):
            np.savez(
                dd / f"design_{s}_{b}.npz",
                **load_npz(design_dir(variant) / f"design_{s}_{b}.npz"),
            )
    say(f"own export {variant}: {n} rows, W {W}")
    return exp_dir


F_ARMS = {
    "all_features": ("f_retune_margin_bb", 0.01),
    "baseline": ("f_retune_margin", 1.0),
}


def f_plan(CLP, b: str, est: str) -> list[float]:
    """The f arm's alpha in force at each 250-session block (close_linear_pretune.plan, f arm)."""
    meta = json.loads((CLP.ARR / "meta.json").read_text())
    bl = CLP.blocks_of(len(meta["_dates"]) - int(meta["_W"]))
    r = F_ARMS[b][1]
    grid = CLP.spec_grid(est)
    T = CLP.fold_table(b, est, r, grid)
    i = CLP.pick(T)["i_mse"]
    sched, _ = CLP.light_retune(T, grid, i, "margin")
    al = []
    for i0, _ in bl:
        done = [k for k, row in enumerate(T["rt_rows"]) if row <= i0]
        al.append(grid[sched[done[-1]]] if done else grid[i])
    return al


def linear(variant: str, workers: int = 10) -> None:
    os.environ["AUDIT_VARIANT"] = variant
    CLP = clp_patch(variant)
    CLP.WORKERS = workers
    composite(variant)
    CLP.DESIGN_DIR = WORK / f"linear_{variant}" / "design"
    CLP.WORK.mkdir(parents=True, exist_ok=True)
    if not (CLP.NETLIB / "liblapack.dll").is_file():
        shutil.copytree(OLD_STUDY / "_work" / "netlib", CLP.NETLIB, dirs_exist_ok=True)
    CLP.ARR.mkdir(parents=True, exist_ok=True)
    if not (CLP.ARR / "spec_grids.json").is_file():
        shutil.copy2(
            OLD_STUDY / "_work" / "arrays" / "spec_grids.json",
            CLP.ARR / "spec_grids.json",
        )
    if not (CLP.ARR / "meta.json").is_file():
        CLP.prep()
    t0 = time.time()
    jobs = []
    for b in CLP.BUCKETS:
        r = F_ARMS[b][1]
        for est in CLP.EST:
            for a in CLP.spec_grid(est):
                jobs.append(dict(kind="fold", bucket=b, est=est, r=r, alpha=a))
    jobs.sort(
        key=lambda j: (j["bucket"] != "all_features", j["est"] == "ridge", j["alpha"])
    )
    CLP.run_pool(jobs, f"folds (spec grid, f arm) {variant}")
    wj = []
    for b in CLP.BUCKETS:
        r = F_ARMS[b][1]
        for est in CLP.EST:
            for a in sorted(set(f_plan(CLP, b, est))):
                wj.append(
                    dict(
                        kind="walk", bucket=b, est=est, r=r, alphas=[a], tag=CLP.atag(a)
                    )
                )
    CLP.run_pool(wj, f"one-alpha walks {variant}")
    say(f"linear {variant} done in {time.time() - t0:.0f}s")


# ============================================================================ score
def scorer_for(variant: str | None):
    """close_linear_pretune.Scorer with the variant's B (the record's when None)."""
    import close_linear_pretune as CLP

    CLP.EXPORT = RECORD if variant is None else WORK / f"linear_{variant}" / "export"
    return CLP.Scorer()


def record_pred(b: str, est: str, fdates) -> np.ndarray:
    old = pd.read_parquet(OLD_STUDY / "forecasts.parquet")
    sid = f"linear_pretune/{est}_{b}_{F_ARMS[b][0]}"
    s = old[old["series_id"] == sid]
    o = pd.Series(
        s["pred_adj"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(s["date"]))
    )
    out = o.reindex(fdates).to_numpy(float)
    assert np.isfinite(out).all(), sid
    return out


def score(variants: list[str]) -> pd.DataFrame:

    S0 = scorer_for(None)
    rows, fc = [], []
    for v in variants:
        CLP_ = clp_patch(v)
        CLP_._A.clear()
        CLP_._ST.clear()
        meta = json.loads((CLP_.ARR / "meta.json").read_text())
        fdates = pd.to_datetime(meta["_dates"])[meta["_W"] :]
        S = scorer_for(v)
        for b in CLP_.BUCKETS:
            for est in CLP_.EST:
                sh = CLP_.SHORT[est]
                bl = CLP_.blocks_of(len(meta["_dates"]) - int(meta["_W"]))
                al = f_plan(CLP_, b, est)
                pred = CLP_.stitch(b, est, F_ARMS[b][1], al, bl)
                o = record_pred(b, sh, pd.DatetimeIndex(S0.dates))
                dn, do = S.daily(pred), S0.daily(o)
                sn, so = S.stats(dn), S0.stats(do)
                cn = S.contrast(dn, do)
                rel = np.abs(
                    dn["plain"] / do["plain"] - 1
                )  # the trade days' plain forecasts
                rows.append(
                    dict(
                        variant=v,
                        bucket=b,
                        est=sh,
                        alphas=json.dumps(al),
                        qlike_old=so["qlike"],
                        qlike_new=sn["qlike"],
                        dqlike=sn["qlike"] - so["qlike"],
                        dq_pct=cn["dq_pct"],
                        dm_vs_old=cn["dm"],
                        dm_p=cn["dm_p"],
                        sharpe_mid_old=so["sharpe_mid"],
                        sharpe_mid_new=sn["sharpe_mid"],
                        sharpe_x_old=so["sharpe_crossed"],
                        sharpe_x_new=sn["sharpe_crossed"],
                        hac_t_mid=cn["hac_t_mid"],
                        positions_changed=cn["n_diff_position"],
                        pred_med_rel=float(np.median(rel)),
                        pred_max_rel=float(rel.max()),
                        sessions_changed=int((rel > 1e-9).sum()),
                    )
                )
                fc.append(
                    pd.DataFrame(
                        dict(
                            variant=v, series=f"{sh}_{b}_f", date=fdates, pred_adj=pred
                        )
                    )
                )
    R = pd.DataFrame(rows)
    f = OUT / "linear_scores.csv"
    if f.is_file():
        old = pd.read_csv(f)
        R = pd.concat([old[~old["variant"].isin(variants)], R], ignore_index=True)
    R.to_csv(f, index=False)
    F = pd.concat(fc)
    ff = WORK / "linear_forecasts.parquet"
    if ff.is_file():
        o = pd.read_parquet(ff)
        F = pd.concat([o[~o["variant"].isin(variants)], F], ignore_index=True)
    F.to_parquet(ff, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            R[R["variant"].isin(variants)].drop(columns=["alphas"]).round(4).to_string()
        )
    return R


def gate_fplan() -> None:
    """My f-arm planner on the 2026-10-06 study's own folds and walks reproduces its stored f arms."""
    import close_linear_pretune as CLP

    CLP.ARR = OLD_STUDY / "_work" / "arrays"
    CLP.WALK = OLD_STUDY / "_work" / "walk"
    CLP.FOLDS = OLD_STUDY / "_work" / "folds"
    rows = []
    meta = json.loads((CLP.ARR / "meta.json").read_text())
    fdates = pd.to_datetime(meta["_dates"])[meta["_W"] :]
    bl = CLP.blocks_of(len(meta["_dates"]) - int(meta["_W"]))
    for b in ("all_features", "baseline"):
        for est in CLP.EST:
            al = f_plan(CLP, b, est)
            pred = CLP.stitch(b, est, F_ARMS[b][1], al, bl)
            o = record_pred(b, CLP.SHORT[est], fdates)
            rows.append(
                dict(
                    gate=f"f-arm planner reproduces the stored {CLP.SHORT[est]} {b} f arm",
                    value=float(np.abs(pred - o).max()),
                    ok=bool(np.abs(pred - o).max() < 1e-12),
                )
            )
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "linear_gates.csv", index=False)
    print(G.to_string())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    args = sys.argv[2:]
    if stage == "capture":
        capture(*args[:3])
    elif stage == "designs":
        designs(args)
    elif stage == "diff":
        diff(args)
    elif stage == "linear":
        linear(args[0], int(args[1]) if len(args) > 1 else 10)
    elif stage == "score":
        score(args)
    elif stage == "gate_fplan":
        gate_fplan()
    elif stage == "capture_src":
        capture_src(*args[:3])
    else:
        import data_audit_stages as ST  # the analysis stages (experiments/data_audit_stages.py)

        getattr(ST, stage)(*args)
