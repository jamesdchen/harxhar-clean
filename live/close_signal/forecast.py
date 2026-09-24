"""The daily forecast: the research arms themselves, run on the extended panel.

Parity by construction.  Rather than exporting coefficients, the run copies
``src/`` and ``specs/causal_tune_linear.py`` into a scratch root whose
``data/`` holds the vendor panel EXTENDED by this package's rows (the
purchased gap + the free daily appends), and runs the spec there with the
research arm's environment (LAG_SCOPE=global, TRAIN_WIN=2000, the
free_vix_only bucket, ridge) for EVERY one of the 13 regular-hours bars
(SEGMENT bar1000 .. bar1600), in FULL -- START 0, END -1, no halo -- exactly
as the CARC campaign did.  The 13 arms' rows are stacked into the notebook's
yhat table exactly as ``experiments/build_subsection_yhat.py`` builds the
research tables (t = the ET stamp in UTC at microsecond resolution, yhat =
pred_adj, baseline = true_raw / true_adj^2, rv_raw = the PANEL's realized
variance at the stamp -- never the arm's winsorized true_raw), and today's
16:00 row goes through the deck's own loader, ``asl.load_yhat_1530_mz_cached``
(the causal second-order MZ map fitted on the session's regular-hours rows), to
``rv_hat``.

Why all 13 bars when only the 16:00 row is traded: the deck's recalibration is
fitted on the whole session's rows, not the close alone.  Measured 2026-09-23:
with the 16:00 rows alone the map is 4-5 % off the deck's rv_hat and disagrees
with sign(s) on 89 of 866 days; with the 13-bar table it reproduces the deck to
0.00 on 866/866 days (``assert_reproduces_deck`` is that gate, kept runnable).

The 16:00 row of today does not exist at 15:30 -- its target has not
happened.  The executor drops rows with no target, so the run appends a
PLACEHOLDER row for today's 16:00 stamp carrying the 15:30 bar's realized
variance as a dummy target.  Nothing the row predicts depends on that value:
the HAR features are ``shift(1)`` of earlier rows, the diurnal baseline is a
``shift(1)`` rolling statistic of the slot, and the MZ coefficients of a
session use strictly prior sessions.  ``assert_placeholder_invariance`` is the
test of exactly that claim on a past day: the arms run with the true target
and with a placeholder must give the same ``pred_adj`` and baseline on every
row up to and including that session.

Requirements before a forecast is live: the panel must be continuous from the
vendor's last row (2024-04-30) to today -- the HAR ladder reaches 3,125 bars
(65 sessions) and the training window 2,000 sessions -- so the purchased gap
rows must be ingested first.  Cost per run: 13 spec processes, each loading
the ~330k-row panel and transforming it, then backtesting its ~2,100 rows;
three at a time, ~2 minutes on the runner, ~4-5 locally.  The card uses
``forecast_session_fast`` instead: the same 13 results files from ONE load
and transform (``arms_shared``), bit-identical (``fastpath_check.py``), in
~20-25 s locally.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live.close_signal.common import CBOE_COLS, CORE_COLS, ET, stamp
from live.close_signal.state import StateStore

VENDOR_FILES: tuple[str, ...] = (
    "core_stats.parquet",
    "vix_and_voldemand.parquet",
    "releases.parquet",
    "time_categories.parquet",
)
#: The bucket the free feed builds: 8 ES moments incl. sumvolume, vix, 4 FOMC
#: (no numobs -- a tick count Yahoo lacks; no vvix / vix3m -- worth nothing at
#: the per-bar arm, CARC 2026-09-23: within noise of the 16-column research
#: bucket on every cell).  The live feed is ES=F 1-minute bars and ^VIX.
BUCKET = "free_vix_only"
ESTIMATOR = "ridge"
#: The traded row's segment: the 15:30-16:00 bar, forecast at 15:30.
SEGMENT = "bar1600"
#: Every regular-hours bar: the recalibration is fitted on all of them.
BARS: tuple[str, ...] = tuple(
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
)
TRAIN_WIN = 2000
#: The spec's own constants (VAL_TAIL 125, EMBARGO 25, TUNE_PER 250) sit inside
#: the training window; the halo replays the window plus slack so the first
#: emitted row has a fully warmed estimator (the diagnostic chunked variant only).
HALO_SLACK = 400
#: Emit the last K segment rows in the chunked variant (diagnostics only).
EMIT_LAST = 30
#: Arms run concurrently: each is a spec process holding the transformed panel.
WORKERS = 3
#: The research tables the gate compares against.
DECK_TAG = "sub_live_ridge"
DECK_BUCKET = "live_feasible"


#: A panel row exists iff the bar had ES prints: the vendor's rule, read off the
#: panel (no Saturday, Sunday from 18:30, holiday sessions end where the prints
#: end, never a row at the spring-forward 02:00).  The state store carries the
#: Cboe carry on every calendar stamp; the extension takes only the stamps whose
#: realized variance is there.
ROW_COL = "sumret2"
#: Largest calendar-day gap between consecutive ES session days the vendor panel
#: shows since 2010 is 4 (3 since 2018): a Friday to a Tuesday across a Monday
#: holiday.  Anything wider is a hole in the history, not a closure.
MAX_DAY_GAP = 5


def _extend(
    vendor: pd.DataFrame,
    ext: pd.DataFrame,
    cols: list[str],
    require: str | None = ROW_COL,
) -> pd.DataFrame:
    """Vendor rows, then the extension's rows for stamps the vendor lacks.

    A new stamp is added only where the extension carries a finite ``require``
    column (the panel's row rule; ``None`` adds every stamp).  At a stamp the
    vendor already has, a vendor cell that is NaN takes the extension's finite
    value (the vendor's Cboe feed ends 2024-02-12 with its ES rows running to
    2024-04-30: those VIX cells are real prints the free feed carries); a
    finite vendor value is never overwritten.
    """
    v = vendor.copy()
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    e = ext.copy()
    e["endbartime"] = pd.to_datetime(e["endbartime"])
    e = e[~e["endbartime"].duplicated(keep="last")]
    for c in cols:
        if c not in v.columns:
            v[c] = np.nan
        if c not in e.columns:
            e[c] = np.nan
    have = e["endbartime"].isin(v["endbartime"])
    fill = e[have].set_index("endbartime")[cols]
    if not fill.empty:
        vi = v.set_index("endbartime")
        sub = vi.loc[fill.index, cols]
        take = sub.isna() & fill.notna()
        vi.loc[fill.index, cols] = sub.where(~take, fill)
        v = vi.reset_index()
    new = e[~have]
    if require is not None:
        new = new[new[require].notna()]
    out = pd.concat([v, new[["endbartime", *cols]]], ignore_index=True)
    return out.sort_values("endbartime").reset_index(drop=True)


def assert_history_continuous(
    core: pd.DataFrame,
    session: pd.Timestamp,
    since: pd.Timestamp | None = None,
    max_gap_days: int = MAX_DAY_GAP,
) -> dict[str, object]:
    """No hole in the ES rows from ``since`` to the session.

    The arms lag by ROW (HAR rolling means over 5 .. 3125 rows) and refit on the
    trailing 2000 rows; the recalibration is fitted on every emitted row.  A
    panel that jumps from the vendor's last day to today would make yesterday of
    a day two years back and forecast anyway.  Raises RuntimeError (a NO SIGNAL
    card) naming the first gap wider than ``max_gap_days`` calendar days.
    """
    st = pd.DatetimeIndex(pd.to_datetime(core["endbartime"]))
    fin = core[ROW_COL].notna().to_numpy()
    st = st[fin]
    if since is not None:
        st = st[st >= pd.Timestamp(since)]
    end = pd.Timestamp(session).normalize()
    st = st[st < end + pd.Timedelta(days=1)]
    if len(st) == 0:
        raise RuntimeError(f"no ES rows between {since} and {end.date()}")
    days = pd.DatetimeIndex(st.normalize().unique()).sort_values()
    if days[-1] != end:
        raise RuntimeError(
            f"the last ES session day in the panel is {days[-1].date()}, not {end.date()}"
        )
    gaps = np.diff(days.values).astype("timedelta64[D]").astype(int)
    bad = np.flatnonzero(gaps > max_gap_days)
    if len(bad):
        i = int(bad[0])
        raise RuntimeError(
            f"history gap: no ES rows between {days[i].date()} and "
            f"{days[i + 1].date()} ({int(gaps[i])} days; the widest closure the "
            f"vendor panel shows is {max_gap_days - 1}) -- ingest the Databento "
            "ES history (README, 'History and refits')"
        )
    return {
        "session_days": int(len(days)),
        "first": str(days[0].date()),
        "last": str(days[-1].date()),
        "max_gap_days": int(gaps.max()) if len(gaps) else 0,
    }


def fomc_release_rows(csv_path: Path, stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """``fomc release`` = 1 at the 14:30 stamp of each statement date in the CSV.

    The vendor's releases feed ends 2023-11-01; the FOMC statement calendar is
    public (federalreserve.gov) and the operator maintains the CSV.  Dates not
    in the file are 0 on their 14:30 stamp; every other stamp NaN, as the
    vendor file has them.
    """
    days = pd.DatetimeIndex([])
    if Path(csv_path).exists():
        s = pd.read_csv(csv_path, comment="#")
        col = s.columns[0]
        days = pd.DatetimeIndex(pd.to_datetime(s[col])).normalize()
    st = pd.DatetimeIndex(stamps)
    is_1430 = (st.hour == 14) & (st.minute == 30)
    # Index.isin, not np.isin: the latter compares datetime64 against Timestamp
    # objects and matched nothing (the first fill of the file flagged no day).
    on_day = st.normalize().isin(days)
    val = np.where(is_1430, on_day.astype(float), np.nan)
    return pd.DataFrame({"endbartime": st, "fomc release": val})


def _copy_code(repo: Path, root: Path) -> None:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / "specs").mkdir(parents=True)
    (root / "data").mkdir()
    shutil.copytree(
        repo / "src",
        root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        repo / "specs" / "causal_tune_linear.py",
        root / "specs" / "causal_tune_linear.py",
    )


def vendor_root(repo: Path, root: Path) -> Path:
    """Scratch root with src/, the spec and the vendor panel AS IS (no extension).

    The deck-reproduction gate runs here: only the four vendor files, so the
    arms see exactly the rows the CARC campaign saw.
    """
    root = Path(root)
    _copy_code(repo, root)
    for f in VENDOR_FILES:
        shutil.copy2(repo / "data" / f, root / "data" / f)
    return root


def build_ext_root(
    repo: Path, store: StateStore, ext_root: Path, fomc_csv: Path
) -> Path:
    """Scratch root with src/, the spec and data/ = vendor panel + this package's rows."""
    ext_root = Path(ext_root)
    _copy_code(repo, ext_root)
    return write_ext_data(repo, store, ext_root, fomc_csv)


def load_vendor(repo: Path) -> dict[str, pd.DataFrame]:
    """The four vendor files as ``write_ext_data`` reads them (kept for a later write)."""
    return {f: pd.read_parquet(Path(repo) / "data" / f) for f in VENDOR_FILES}


def write_ext_data(
    repo: Path,
    store: StateStore,
    ext_root: Path,
    fomc_csv: Path,
    vendor: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """data/ of an ext root = vendor panel + this package's rows (the code is not touched).

    ``vendor`` is ``load_vendor(repo)`` read earlier (the precompute phase); the
    frames are copied, never modified, so the files written are those a fresh
    read gives.
    """
    ext_root = Path(ext_root)
    (ext_root / "data").mkdir(parents=True, exist_ok=True)

    def vendor_file(f: str) -> pd.DataFrame:
        if vendor is None:
            return pd.read_parquet(Path(repo) / "data" / f)
        return vendor[f].copy()

    panel = store.load_panel()
    core = vendor_file("core_stats.parquet")
    core_ext = _extend(core, panel, list(CORE_COLS))
    core_ext.to_parquet(ext_root / "data" / "core_stats.parquet", index=False)
    vix = vendor_file("vix_and_voldemand.parquet")
    vix_ext = _extend(vix, panel, list(CBOE_COLS))
    vix_ext.to_parquet(ext_root / "data" / "vix_and_voldemand.parquet", index=False)
    if not core_ext["endbartime"].equals(vix_ext["endbartime"]):
        raise RuntimeError(
            "core_stats and vix_and_voldemand extensions disagree on stamps"
        )
    core_last = pd.to_datetime(core["endbartime"])[core[ROW_COL].notna()].max()
    info = {
        "vendor_rows": int(len(core)),
        "vendor_last_es_stamp": str(core_last),
        "rows_added": int(len(core_ext) - len(core)),
        "store_rows_without_es": int(panel[ROW_COL].isna().sum()),
        "vix_cells_filled": int(
            vix_ext.set_index("endbartime")
            .loc[pd.to_datetime(vix["endbartime"]), "vix"]
            .notna()
            .sum()
            - vix["vix"].notna().sum()
        ),
    }
    (ext_root / "data" / "extension.json").write_text(json.dumps(info, indent=1))
    print("extended panel:", info, flush=True)
    new_stamps = pd.DatetimeIndex(core_ext["endbartime"])
    rel = vendor_file("releases.parquet")
    rel["endbartime"] = pd.to_datetime(rel["endbartime"])
    fomc = fomc_release_rows(fomc_csv, new_stamps[~new_stamps.isin(rel["endbartime"])])
    rel_ext = pd.concat([rel, fomc], ignore_index=True).sort_values("endbartime")
    rel_ext.to_parquet(ext_root / "data" / "releases.parquet", index=False)
    tc = vendor_file("time_categories.parquet")
    tc["endbartime"] = pd.to_datetime(tc["endbartime"])
    extra = new_stamps[~new_stamps.isin(tc["endbartime"])]
    tc_ext = pd.concat(
        [
            tc,
            pd.DataFrame(
                {"endbartime": extra, "hour": extra.hour, "DOW": extra.dayofweek}
            ),
        ],
        ignore_index=True,
    ).sort_values("endbartime")
    tc_ext.to_parquet(ext_root / "data" / "time_categories.parquet", index=False)
    return ext_root


def placeholder_row(store: StateStore, session: pd.Timestamp) -> pd.DataFrame:
    """Today's 16:00 row with the 15:30 bar's realized variance as the dummy target."""
    panel = store.load_panel().set_index("endbartime")
    t1530 = stamp(session, "15:30")
    t1600 = stamp(session, "16:00")
    if t1530 not in panel.index or not np.isfinite(panel.at[t1530, "sumret2"]):
        raise ValueError(f"no 15:30 row for {session.date()} in the state panel")
    row = {c: np.nan for c in CORE_COLS}
    row["sumret2"] = float(panel.at[t1530, "sumret2"])
    row["numobs"] = 30.0
    return pd.DataFrame([{"endbartime": t1600, **row}])


def n_segment_rows(ext_root: Path, segment: str = SEGMENT) -> int:
    """Rows the segment will hold: its stamps with a positive target."""
    core = pd.read_parquet(
        ext_root / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"]
    )
    t = pd.to_datetime(core["endbartime"])
    hh, mm = int(segment[3:5]), int(segment[5:7])
    ok = (t.dt.hour == hh) & (t.dt.minute == mm) & (core["sumret2"] > 0)
    return int(ok.sum())


def run_arm(
    ext_root: Path,
    result_dir: Path,
    python: str = sys.executable,
    full: bool = True,
    segment: str = SEGMENT,
    bucket: str = BUCKET,
) -> Path:
    """Run the spec's arm for one bar on the extended panel; return the results CSV path.

    ``full=True`` (the default, the operator's fidelity rule of 2026-09-23) runs
    the arm over the WHOLE extended panel exactly as the research campaign did
    -- START 0, END -1, no halo -- so every rolling object (the robust scaler,
    the availability masks, the impute medians, the diurnal baseline, the
    21-session refits) sees the same rows it saw on CARC and the forecast for
    today is the research forecast on a longer panel, with no chunk seam and no
    incremental algebra.  One bar's rows for 6,600 sessions fit and score in
    about a minute; there is nothing to save.  ``full=False`` keeps the
    chunked variant (the last EMIT_LAST rows with a training-window halo) for
    diagnostics only; it is not the path of record.
    """
    env = dict(os.environ)
    env.update(
        {
            "HPC_KW_SEGMENT": segment,
            "HPC_KW_LAG_SCOPE": "global",
            "HPC_KW_ESTIMATOR": ESTIMATOR,
            "HPC_KW_EXOG_BUCKET": bucket,
            "HPC_KW_TRAIN_WIN": str(TRAIN_WIN),
            "HPC_RESULT_DIR": str(result_dir),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    if full:
        env.update({"HPC_KW_START": "0", "HPC_KW_END": "-1", "HPC_KW_HALO": "0"})
    else:
        n = n_segment_rows(ext_root, segment)
        env.update(
            {
                "HPC_KW_START": str(max(0, n - EMIT_LAST)),
                "HPC_KW_END": "-1",
                "HPC_KW_HALO": str(TRAIN_WIN + HALO_SLACK),
            }
        )
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log = result_dir / "run.log"
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(
            [python, str(Path(ext_root) / "specs" / "causal_tune_linear.py")],
            cwd=str(ext_root),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    if rc != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"the {segment} arm failed (rc {rc}); log tail:\n{tail}")
    csv = (
        result_dir
        / "causal_tune_linear"
        / ESTIMATOR
        / bucket
        / f"results_{segment}.csv"
    )
    if not csv.exists():
        raise RuntimeError(f"the arm wrote no {csv}")
    return csv


def run_arms(
    ext_root: Path,
    results_root: Path,
    python: str = sys.executable,
    bars: tuple[str, ...] = BARS,
    bucket: str = BUCKET,
    workers: int = WORKERS,
) -> dict[str, Path]:
    """The 13 per-bar arms, ``workers`` spec processes at a time; bar -> results CSV."""
    results_root = Path(results_root)

    def one(bar: str) -> tuple[str, Path]:
        return bar, run_arm(
            ext_root, results_root / bar, python, full=True, segment=bar, bucket=bucket
        )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        out = dict(ex.map(one, bars))
    return {b: out[b] for b in bars}


def read_arm(results_csv: Path) -> pd.DataFrame:
    """One arm's rows as the notebook's yhat table -- build_subsection_yhat.read_arm.

    Rows with a positive target and a finite forecast; ``t`` is the naive-ET
    bar-end stamp in UTC at MICROSECOND resolution (a nanosecond index makes the
    notebook's day frames "not identically labelled"); ``rv_raw`` here is still
    the arm's ``true_raw`` -- ``with_panel_rv`` replaces it.
    """
    r = pd.read_csv(results_csv, parse_dates=["date"])
    ok = (r["true_adj"] > 0) & (r["true_raw"] > 0) & np.isfinite(r["pred_adj"])
    r = r[ok]
    et = pd.DatetimeIndex(r["date"]).tz_localize(ET)
    return pd.DataFrame(
        {
            "t": et.tz_convert("UTC").as_unit("us"),
            "yhat": r["pred_adj"].to_numpy(float),
            "baseline": (r["true_raw"] / r["true_adj"] ** 2).to_numpy(float),
            "rv_raw": r["true_raw"].to_numpy(float),
        }
    )


def panel_rv(ext_root: Path) -> pd.Series:
    """The panel's realized variance by stamp (t UTC, microseconds)."""
    core = pd.read_parquet(
        Path(ext_root) / "data" / "core_stats.parquet",
        columns=["endbartime", "sumret2"],
    )
    et = pd.DatetimeIndex(pd.to_datetime(core["endbartime"])).tz_localize(ET)
    return pd.Series(
        core["sumret2"].to_numpy(float), index=et.tz_convert("UTC").as_unit("us")
    )


def with_panel_rv(tab: pd.DataFrame, rv: pd.Series) -> pd.DataFrame:
    """Carry the PANEL's realized variance as ``rv_raw`` -- build_subsection_yhat's rule.

    The spec's ``true_raw`` is the winsorized target on a few percent of rows;
    the research tables carry the production panel's rv_raw instead, and the
    deck's recalibration was fitted on that.  Every row must find its stamp.
    """
    x = rv.reindex(tab["t"])
    if x.isna().any():
        missing = tab["t"][x.isna().to_numpy()]
        raise RuntimeError(
            f"{int(x.isna().sum())} arm rows have no panel realized variance, first {missing.iloc[0]}"
        )
    out = tab.copy()
    out["rv_raw"] = x.to_numpy(float)
    return out


def assemble_yhat_table(csvs: dict[str, Path], ext_root: Path) -> pd.DataFrame:
    """The 13 arms stacked and sorted by ``t`` with the panel's rv_raw: the research table."""
    rv = panel_rv(ext_root)
    parts = [with_panel_rv(read_arm(csvs[b]), rv) for b in BARS if b in csvs]
    tab = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    if tab["t"].duplicated().any():
        raise RuntimeError("the assembled yhat table has duplicate stamps")
    return tab


def _asl(repo: Path):  # type: ignore[no-untyped-def]
    if str(Path(repo) / "notebooks") not in sys.path:
        sys.path.insert(0, str(Path(repo) / "notebooks"))
    import atm_straddle_lib as asl  # type: ignore

    return asl


def recalibrate(
    table: pd.DataFrame,
    need_dates,
    repo: Path,
    scratch: Path,
    tag: str = "close_signal",
) -> pd.DataFrame:
    """The deck's loader on the assembled table: rv_hat, m, s2 per requested session date.

    ``asl.load_yhat_1530_mz_cached`` -- the causal second-order MZ map fitted
    on the regular-hours rows of prior sessions, the 15:30 book's own loader --
    returns a frame indexed by session date with yhat, baseline, rv_raw, rv_hat,
    m, s2, yhat_vol, m_vol for the 16:00 rows of the requested dates.
    """
    asl = _asl(repo)
    scratch = Path(scratch)
    cache = scratch / "mz_cache"
    cache.mkdir(parents=True, exist_ok=True)
    p = scratch / f"yhat_{tag}.parquet"
    table.to_parquet(p, index=False)
    out = asl.load_yhat_1530_mz_cached(tag, p, need_dates, cache, method="mean")
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    out = out.copy()
    out.index = idx.tz_localize(None) if idx.tz is not None else idx
    return out


def rv_hat_for(
    session: pd.Timestamp, table: pd.DataFrame, repo: Path, scratch: Path
) -> dict[str, float]:
    """Today's 16:00 row through the deck's loader: rv_hat and its pieces."""
    day = pd.Timestamp(session).normalize()
    px = recalibrate(table, [day], repo, scratch)
    if day not in px.index:
        raise RuntimeError(f"no 16:00 row for {day.date()} in the recalibrated table")
    row = px.loc[day]
    return {
        "rv_hat": float(row["rv_hat"]),
        "yhat": float(row["yhat"]),
        "baseline": float(row["baseline"]),
        "mz_m": float(row["m"]),
        "mz_s2": float(row["s2"]),
    }


def forecast_session(
    repo: Path,
    store: StateStore,
    session: pd.Timestamp,
    scratch: Path,
    fomc_csv: Path,
    python: str = sys.executable,
    workers: int = WORKERS,
) -> dict[str, float]:
    """End to end for one session: placeholder row, extended root, 13 arms, table, MZ.

    The path of record: one spec process per bar.  ``forecast_session_fast``
    is the same forecast from one shared pass (``arms_shared``).
    """
    store.append_panel(
        placeholder_row(store, session), source="placeholder", placeholder=True
    )
    ext = build_ext_root(repo, store, Path(scratch) / "ext_root", fomc_csv)
    check_extension(ext, session)
    csvs = run_arms(ext, Path(scratch) / "arms", python, workers=workers)
    table = assemble_yhat_table(csvs, ext)
    return rv_hat_for(session, table, repo, Path(scratch))


def check_extension(ext: Path, session: pd.Timestamp) -> dict[str, object]:
    """The history-continuity guard on an ext root's written core_stats."""
    info = json.loads((Path(ext) / "data" / "extension.json").read_text())
    core_ext = pd.read_parquet(
        Path(ext) / "data" / "core_stats.parquet", columns=["endbartime", ROW_COL]
    )
    # from a month before the vendor's last ES day: the seam is inside the check
    since = pd.Timestamp(info["vendor_last_es_stamp"]).normalize() - pd.Timedelta(
        days=30
    )
    cont = assert_history_continuous(core_ext, session, since=since)
    print("history continuous:", cont, flush=True)
    return cont


# ---------------------------------------------------------------- fast path --
#: Backtest worker processes of the shared pass (the runner has 4 vCPUs).
FAST_WORKERS = max(1, min(4, os.cpu_count() or 1))
#: A shared pass takes ~0.5 min; one still running after this is abandoned
#: (the card becomes NO SIGNAL rather than arriving long after the stamp).
ARMS_TIMEOUT_S = 300.0


class FastContext:
    """What the precompute phase leaves for the 15:30:30 pass.

    ``root`` is an ext root whose code was copied by THIS process (so it is the
    repository's code of this run, never a stale copy); ``vendor`` the four
    vendor files already read; ``server`` a warm ``arms_shared --serve``
    process rooted at ``root`` (None: started on first use).  Only ``data/``
    is rewritten afterwards.
    """

    def __init__(
        self,
        repo: Path,
        scratch: Path,
        workers: int = FAST_WORKERS,
        bucket: str = BUCKET,
        python: str = sys.executable,
        start_server: bool = True,
    ) -> None:
        from live.close_signal import arms_shared

        self.repo = Path(repo).resolve()
        # absolute: the server's working directory is the ext root
        self.scratch = Path(scratch).resolve()
        self.workers = int(workers)
        self.bucket = bucket
        self.python = python
        self.root = self.scratch / "fast_root"
        _copy_code(self.repo, self.root)
        self.vendor = load_vendor(self.repo)
        self._env = arms_shared.arm_env(
            bucket, ESTIMATOR, TRAIN_WIN, self.scratch / "fast_arms"
        )
        self.server: arms_shared.ArmServer | None = None
        if start_server:
            self.start_server()

    def start_server(self, ready_timeout: float = 300.0) -> None:
        from live.close_signal import arms_shared

        self.server = arms_shared.ArmServer(
            self.root,
            self._env,
            self.scratch / "fast_arms" / "server.log",
            workers=self.workers,
            python=self.python,
            ready_timeout=ready_timeout,
        )
        print(f"arm server ready in {self.server.ready_seconds:.1f}s", flush=True)

    def healthy(self) -> bool:
        """A live server that is not in the middle of a pass it was abandoned in."""
        return self.server is not None and self.server.alive() and not self._stuck

    _stuck = False

    def run_arms(
        self, tag: str, timeout: float = ARMS_TIMEOUT_S
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        """The 13 arms on the root's current data/ into fast_arms/<tag>."""
        out_dir = self.scratch / "fast_arms" / tag
        if out_dir.exists():
            shutil.rmtree(out_dir)  # never read a CSV an earlier pass left
        if not self.healthy():
            self.close(kill=True)
            self._stuck = False
            self.start_server()
        assert self.server is not None
        try:
            out = self.server.run(out_dir, timeout=timeout)
        except TimeoutError:
            self._stuck = True
            raise
        csvs = {b: Path(p) for b, p in out["csvs"].items()}
        want = out_dir / "causal_tune_linear" / ESTIMATOR / self.bucket
        for b in BARS:
            if (
                b not in csvs
                or csvs[b].resolve() != (want / f"results_{b}.csv").resolve()
            ):
                raise RuntimeError(f"the shared pass did not write {b} under {want}")
            if not csvs[b].exists():
                raise RuntimeError(f"the shared pass wrote no {csvs[b]}")
        return {b: csvs[b] for b in BARS}, out

    def close(self, kill: bool = False) -> None:
        if self.server is not None:
            if kill or self._stuck:
                self.server.kill()
            else:
                self.server.close()
            self.server = None


def canary(
    ctx: FastContext,
    store: StateStore,
    session: pd.Timestamp,
    fomc_csv: Path,
    timeout: float = ARMS_TIMEOUT_S,
) -> dict[str, Any]:
    """The whole fast path on the panel as it stands (no 15:30 row yet); discarded.

    Nothing it computes is reused: the 15:30 row moves full-sample statistics
    of the transform (the diurnal std floor of the signed moments, the median
    fills), so every arm's history changes with it (fastpath_check.py
    measures it: ``dependence`` in its report).  What the canary leaves is a warm server --
    imports, the spec's definitions, the numba kernels, the pyc files of the
    copied code -- and an early failure if the history or an arm is broken.
    """
    write_ext_data(ctx.repo, store, ctx.root, fomc_csv, vendor=ctx.vendor)
    cont = check_extension(ctx.root, session)
    csvs, out = ctx.run_arms("canary", timeout=timeout)
    table = assemble_yhat_table(csvs, ctx.root)
    last = pd.DatetimeIndex(table["t"].dt.tz_convert(ET).dt.tz_localize(None))
    days = last.normalize().unique()
    recalibrate(table, [days[-2]], ctx.repo, ctx.scratch / "canary", tag="canary")
    return {"history": cont, **{k: v for k, v in out.items() if k != "csvs"}}


def forecast_session_fast(
    repo: Path,
    store: StateStore,
    session: pd.Timestamp,
    scratch: Path,
    fomc_csv: Path,
    ctx: FastContext | None = None,
    workers: int = FAST_WORKERS,
    python: str = sys.executable,
) -> dict[str, float]:
    """``forecast_session`` from one shared spec pass: the same rv_hat, faster.

    Same placeholder, same data files, same history guard, the same 13
    results CSVs (``arms_shared``), the same table and loader.  ``ctx`` is the
    precompute phase's warm context; without one a cold context is made and
    closed here.
    """
    own = ctx is None
    if ctx is None:
        ctx = FastContext(repo, scratch, workers=workers, python=python)
    try:
        store.append_panel(
            placeholder_row(store, session), source="placeholder", placeholder=True
        )
        write_ext_data(repo, store, ctx.root, fomc_csv, vendor=ctx.vendor)
        check_extension(ctx.root, session)
        csvs, out = ctx.run_arms("card")
        print(
            "shared arm pass: matrix {seconds_matrix:.1f}s, 13 backtests "
            "{seconds_backtests:.1f}s, total {seconds_total:.1f}s".format(**out),
            flush=True,
        )
        table = assemble_yhat_table(csvs, ctx.root)
        return rv_hat_for(session, table, repo, Path(scratch))
    finally:
        if own:
            ctx.close()


def table_deviation(
    a: pd.DataFrame, b: pd.DataFrame, upto: pd.Timestamp | None = None
) -> dict[str, float]:
    """Max relative deviation of yhat and baseline between two yhat tables on common stamps."""
    x = a.set_index("t")[["yhat", "baseline"]]
    y = b.set_index("t")[["yhat", "baseline"]]
    if upto is not None:
        cut = pd.Timestamp(upto, tz=ET).tz_convert("UTC").as_unit("us")
        x = x[x.index <= cut]
        y = y[y.index <= cut]
    j = x.join(y, how="inner", rsuffix="_b")
    if j.empty:
        raise ValueError("the two tables share no stamps")
    dy = (
        np.abs(j["yhat"] - j["yhat_b"]) / np.maximum(np.abs(j["yhat_b"]), 1e-300)
    ).max()
    db = (
        np.abs(j["baseline"] - j["baseline_b"])
        / np.maximum(np.abs(j["baseline_b"]), 1e-300)
    ).max()
    return {
        "rows": int(len(j)),
        "rows_only_a": int(len(x) - len(j)),
        "rows_only_b": int(len(y) - len(j)),
        "max_rel_yhat": float(dy),
        "max_rel_baseline": float(db),
    }


def assert_placeholder_invariance(
    repo: Path,
    store: StateStore,
    past_session: pd.Timestamp,
    scratch: Path,
    fomc_csv: Path,
    python: str = sys.executable,
    rel_tol: float = 1e-9,
    workers: int = WORKERS,
) -> dict[str, float]:
    """The claim the whole design rests on, tested on a session whose target is known.

    Run the 13 arms on the panel as is, then with that session's 16:00 target
    replaced by a placeholder; every row of the assembled table up to and
    including that session -- ``pred_adj`` and the baseline -- must agree to
    ``rel_tol``.  Returns the 16:00 predictions and the deviations.
    """
    ext = build_ext_root(repo, store, Path(scratch) / "inv_true", fomc_csv)
    a = assemble_yhat_table(
        run_arms(ext, Path(scratch) / "inv_true_arms", python, workers=workers), ext
    )
    core_p = ext / "data" / "core_stats.parquet"
    core = pd.read_parquet(core_p)
    t = stamp(past_session, "16:00")
    t1530 = stamp(past_session, "15:30")
    idx = pd.to_datetime(core["endbartime"]) == t
    if not idx.any():
        raise ValueError(f"{past_session.date()} has no 16:00 row")
    dummy = float(
        core.loc[pd.to_datetime(core["endbartime"]) == t1530, "sumret2"].iloc[0]
    )
    core.loc[idx, "sumret2"] = dummy
    ext2 = Path(scratch) / "inv_dummy"
    if ext2.exists():
        shutil.rmtree(ext2)
    shutil.copytree(ext, ext2)
    core.to_parquet(ext2 / "data" / "core_stats.parquet", index=False)
    b = assemble_yhat_table(
        run_arms(ext2, Path(scratch) / "inv_dummy_arms", python, workers=workers), ext2
    )
    dev = table_deviation(a, b, upto=t)
    tu = pd.Timestamp(t, tz=ET).tz_convert("UTC").as_unit("us")
    ya = a.set_index("t").loc[tu]
    yb = b.set_index("t").loc[tu]
    worst = max(dev["max_rel_yhat"], dev["max_rel_baseline"])
    if worst > rel_tol:
        raise AssertionError(
            f"placeholder invariance FAILED on {past_session.date()}: max rel dev "
            f"yhat {dev['max_rel_yhat']:.2e}, baseline {dev['max_rel_baseline']:.2e} on "
            f"{dev['rows']} rows up to the session; 16:00 yhat {ya['yhat']} vs {yb['yhat']}"
        )
    return {
        "yhat_true": float(ya["yhat"]),
        "yhat_dummy": float(yb["yhat"]),
        "rel_dev": float(worst),
        "rows_compared": float(dev["rows"]),
    }


def assert_reproduces_deck(
    repo: Path,
    scratch: Path,
    python: str = sys.executable,
    workers: int = WORKERS,
    yhat_rel_tol: float = 1e-6,
    rv_hat_rel_tol: float = 1e-7,
    reuse_arms: bool = False,
) -> dict[str, float]:
    """The live path on the vendor panel alone must BE the research: table and deck.

    Runs the 13 arms with the research bucket (``live_feasible``) on the vendor
    files as they are, assembles the table, and requires (1) equality with the
    research table ``results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet`` on
    every row -- every stamp present on both sides, baseline and rv_raw EXACTLY,
    yhat to ``yhat_rel_tol`` -- and (2) equality of the loader's rv_hat with the
    deck ``daily_sub_live_ridge.parquet`` on every deck day to ``rv_hat_rel_tol``
    with EVERY sign(s) agreeing.

    Tolerances, and why they are not zero.  The research arms ran on CARC
    (Linux, its BLAS); this gate runs them on the operator's machine.  Same
    code, same rows, bit-identical inputs -- and the ridge solves differ at the
    float-path level in the ill-conditioned windows (the repository's known
    cross-machine mechanism, 2026-07-15).  Measured 2026-09-23 on Windows:
    20,044 of 20,044 rows matched; baseline and rv_raw 0.0; yhat max relative
    deviation 5.5e-7 (bar1530; the 16:00 bar 7.3e-12, five bars below 1e-9);
    rv_hat on 866/866 deck days max relative 9.6e-9; signs 866/866.  The
    tolerances are set one decade above those numbers; the deterministic
    columns and the signs are exact.  About 13 x 1-2 minutes; ``reuse_arms``
    skips the arm runs when their CSVs already sit in ``scratch`` (only the
    assembly, the loader or the tolerances changed).
    """
    repo = Path(repo)
    scratch = Path(scratch)
    root = scratch / "vendor_root"
    csvs: dict[str, Path] = {
        b: scratch
        / "vendor_arms"
        / b
        / "causal_tune_linear"
        / ESTIMATOR
        / DECK_BUCKET
        / f"results_{b}.csv"
        for b in BARS
    }
    if not (reuse_arms and root.exists() and all(p.exists() for p in csvs.values())):
        root = vendor_root(repo, root)
        csvs = run_arms(
            root, scratch / "vendor_arms", python, bucket=DECK_BUCKET, workers=workers
        )
    tab = assemble_yhat_table(csvs, root)
    ref = (
        pd.read_parquet(
            repo / "results" / "spxw_pnl" / f"yhat_sub_ridge_{DECK_BUCKET}.parquet"
        )
        .sort_values("t")
        .reset_index(drop=True)
    )
    j = ref.merge(tab, on="t", how="outer", suffixes=("_ref", ""), indicator=True)
    n_only = int((j["_merge"] != "both").sum())
    both = j[j["_merge"] == "both"]
    rel_y = (
        np.abs(both["yhat"] - both["yhat_ref"])
        / np.maximum(np.abs(both["yhat_ref"]), 1e-300)
    ).max()
    d_base = np.abs(both["baseline"] - both["baseline_ref"]).max()
    d_rv = np.abs(both["rv_raw"] - both["rv_raw_ref"]).max()
    deck = pd.read_parquet(
        repo / "results" / "atm_straddle_0dte_1530" / f"daily_{DECK_TAG}.parquet"
    ).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    px = recalibrate(tab, deck.index, repo, scratch, tag="gate_deck")
    x = px["rv_hat"].reindex(deck.index)
    rel_rv = (np.abs(x - deck["rv_hat"]) / deck["rv_hat"]).to_numpy(float)
    matched = int(np.isfinite(rel_rv).sum())
    signs = int((((x - deck["iv_var"]) > 0) == (deck["signal"] > 0)).sum())
    out = {
        "table_rows": int(len(tab)),
        "table_rows_unmatched": n_only,
        "table_max_rel_yhat": float(rel_y),
        "table_max_abs_baseline": float(d_base),
        "table_max_abs_rv_raw": float(d_rv),
        "deck_days": int(len(deck)),
        "deck_days_matched": matched,
        "deck_max_rel_rv_hat": float(np.nanmax(rel_rv)),
        "deck_signs_agree": signs,
    }
    ok = (
        n_only == 0
        and rel_y <= yhat_rel_tol
        and d_base == 0.0
        and d_rv == 0.0
        and matched == len(deck)
        and float(np.nanmax(rel_rv)) <= rv_hat_rel_tol
        and signs == len(deck)
    )
    if not ok:
        raise AssertionError(f"the live path does NOT reproduce the deck: {out}")
    return out


__all__ = [
    "BARS",
    "BUCKET",
    "DECK_BUCKET",
    "DECK_TAG",
    "ESTIMATOR",
    "FAST_WORKERS",
    "FastContext",
    "SEGMENT",
    "TRAIN_WIN",
    "WORKERS",
    "assemble_yhat_table",
    "assert_placeholder_invariance",
    "assert_reproduces_deck",
    "build_ext_root",
    "canary",
    "check_extension",
    "fomc_release_rows",
    "forecast_session",
    "forecast_session_fast",
    "load_vendor",
    "panel_rv",
    "placeholder_row",
    "read_arm",
    "recalibrate",
    "run_arm",
    "run_arms",
    "rv_hat_for",
    "table_deviation",
    "vendor_root",
    "with_panel_rv",
    "write_ext_data",
]
