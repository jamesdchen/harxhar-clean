"""The daily forecast: the research arm itself, run on the extended panel.

Parity by construction.  Rather than exporting coefficients, the run copies
``src/`` and ``specs/causal_tune_linear.py`` into a scratch root whose
``data/`` holds the vendor panel EXTENDED by this package's rows (the
purchased gap + the free daily appends), and runs the spec there with the
research arm's environment (SEGMENT=bar1600, LAG_SCOPE=global, TRAIN_WIN=2000,
the free_feasible bucket, ridge) restricted to the last rows with the
training-window halo.  The arm's ``pred_adj`` for today's 16:00 row goes
through the notebook library's causal MZ map (``asl.load_yhat_panel_mz``, the
deck's own loader) to ``rv_hat``.

The 16:00 row of today does not exist at 15:30 -- its target has not
happened.  The executor drops rows with no target, so the run appends a
PLACEHOLDER row for today's 16:00 stamp carrying the 15:30 bar's realized
variance as a dummy target.  Nothing the row predicts depends on that value:
the HAR features are ``shift(1)`` of earlier rows, the diurnal baseline is a
``shift(1)`` rolling statistic of the slot, and the MZ coefficients of a
session use strictly prior sessions.  ``assert_placeholder_invariance`` is the
test of exactly that claim on a past day: the arm run with the true target and
with a placeholder must give the same ``pred_adj`` and baseline.

Requirements before a forecast is live: the panel must be continuous from the
vendor's last row (2024-04-30) to today -- the HAR ladder reaches 3,125 bars
(65 sessions) and the training window 2,000 sessions -- so the purchased gap
rows must be ingested first.  Cost per run: the spec loads the ~350k-row panel
and transforms it (about a minute), then fits ~2,400 rows (seconds).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

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
BUCKET = "free_feasible"
ESTIMATOR = "ridge"
SEGMENT = "bar1600"
TRAIN_WIN = 2000
#: The spec's own constants (VAL_TAIL 125, EMBARGO 25, TUNE_PER 250) sit inside
#: the training window; the halo replays the window plus slack so the first
#: emitted row has a fully warmed estimator.
HALO_SLACK = 400
#: Emit the last K segment rows (today's is the last); K > 1 so a miscount of
#: one or two sessions cannot leave today's row outside the emit range.
EMIT_LAST = 30


def _extend(vendor: pd.DataFrame, ext: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Vendor rows, then the extension's rows for stamps the vendor lacks."""
    v = vendor.copy()
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    e = ext[["endbartime", *cols]].copy()
    e["endbartime"] = pd.to_datetime(e["endbartime"])
    e = e[~e["endbartime"].isin(v["endbartime"])]
    for c in cols:
        if c not in v.columns:
            v[c] = np.nan
    out = pd.concat([v, e[["endbartime", *cols]]], ignore_index=True)
    return out.sort_values("endbartime").reset_index(drop=True)


def fomc_release_rows(csv_path: Path, stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """``fomc release`` = 1 at the 14:30 stamp of each statement date in the CSV.

    The vendor's releases feed ends 2023-11-01; the FOMC statement calendar is
    public (federalreserve.gov) and the operator maintains the CSV.  Dates not
    in the file are 0 on their 14:30 stamp; every other stamp NaN, as the
    vendor file has them.
    """
    days = set()
    if Path(csv_path).exists():
        s = pd.read_csv(csv_path, comment="#")
        col = s.columns[0]
        days = set(pd.to_datetime(s[col]).dt.normalize())
    st = pd.DatetimeIndex(stamps)
    is_1430 = (st.hour == 14) & (st.minute == 30)
    val = np.where(is_1430, np.isin(st.normalize(), list(days)).astype(float), np.nan)
    return pd.DataFrame({"endbartime": st, "fomc release": val})


def build_ext_root(
    repo: Path, store: StateStore, ext_root: Path, fomc_csv: Path
) -> Path:
    """Scratch root with src/, the spec and data/ = vendor panel + this package's rows."""
    ext_root = Path(ext_root)
    if ext_root.exists():
        shutil.rmtree(ext_root)
    (ext_root / "specs").mkdir(parents=True)
    (ext_root / "data").mkdir()
    shutil.copytree(
        repo / "src",
        ext_root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        repo / "specs" / "causal_tune_linear.py",
        ext_root / "specs" / "causal_tune_linear.py",
    )
    panel = store.load_panel()
    core = pd.read_parquet(repo / "data" / "core_stats.parquet")
    core_ext = _extend(core, panel, list(CORE_COLS))
    core_ext.to_parquet(ext_root / "data" / "core_stats.parquet", index=False)
    vix = pd.read_parquet(repo / "data" / "vix_and_voldemand.parquet")
    vix_ext = _extend(vix, panel, list(CBOE_COLS))
    vix_ext.to_parquet(ext_root / "data" / "vix_and_voldemand.parquet", index=False)
    new_stamps = pd.DatetimeIndex(core_ext["endbartime"])
    rel = pd.read_parquet(repo / "data" / "releases.parquet")
    rel["endbartime"] = pd.to_datetime(rel["endbartime"])
    fomc = fomc_release_rows(fomc_csv, new_stamps[~new_stamps.isin(rel["endbartime"])])
    rel_ext = pd.concat([rel, fomc], ignore_index=True).sort_values("endbartime")
    rel_ext.to_parquet(ext_root / "data" / "releases.parquet", index=False)
    tc = pd.read_parquet(repo / "data" / "time_categories.parquet")
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


def n_segment_rows(ext_root: Path) -> int:
    """Rows the bar1600 segment will hold: 16:00 stamps with a positive target."""
    core = pd.read_parquet(
        ext_root / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"]
    )
    t = pd.to_datetime(core["endbartime"])
    ok = (t.dt.hour == 16) & (t.dt.minute == 0) & (core["sumret2"] > 0)
    return int(ok.sum())


def run_arm(ext_root: Path, result_dir: Path, python: str = sys.executable) -> Path:
    """Run the spec's bar1600 arm on the extended panel; return the results CSV path."""
    n = n_segment_rows(ext_root)
    env = dict(os.environ)
    env.update(
        {
            "HPC_KW_SEGMENT": SEGMENT,
            "HPC_KW_LAG_SCOPE": "global",
            "HPC_KW_ESTIMATOR": ESTIMATOR,
            "HPC_KW_EXOG_BUCKET": BUCKET,
            "HPC_KW_TRAIN_WIN": str(TRAIN_WIN),
            "HPC_KW_START": str(max(0, n - EMIT_LAST)),
            "HPC_KW_END": "-1",
            "HPC_KW_HALO": str(TRAIN_WIN + HALO_SLACK),
            "HPC_RESULT_DIR": str(result_dir),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    log = result_dir / "run.log"
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(
            [python, str(ext_root / "specs" / "causal_tune_linear.py")],
            cwd=str(ext_root),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    if rc != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"the spec arm failed (rc {rc}); log tail:\n{tail}")
    csv = (
        result_dir
        / "causal_tune_linear"
        / ESTIMATOR
        / BUCKET
        / f"results_{SEGMENT}.csv"
    )
    if not csv.exists():
        raise RuntimeError(f"the arm wrote no {csv}")
    return csv


def yhat_table(results_csv: Path) -> pd.DataFrame:
    """The arm's rows as the notebook's yhat table (t UTC, yhat, baseline, rv_raw)."""
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


def rv_hat_for(
    session: pd.Timestamp, table: pd.DataFrame, repo: Path, scratch: Path
) -> dict[str, float]:
    """The deck's causal MZ map on the yhat table; today's 16:00 row's rv_hat and pieces."""
    if str(repo / "notebooks") not in sys.path:
        sys.path.insert(0, str(repo / "notebooks"))
    import atm_straddle_lib as asl  # type: ignore

    p = Path(scratch) / "yhat_close_signal.parquet"
    table.to_parquet(p, index=False)
    px = asl.load_yhat_panel_mz(p)
    t = stamp(session, "16:00")
    et = pd.DatetimeIndex(px["et"]) if "et" in px else None
    if et is None:
        raise RuntimeError("load_yhat_panel_mz returned no et column")
    et_naive = et.tz_localize(None) if et.tz is not None else et
    m = et_naive == t
    if not m.any():
        raise RuntimeError(
            f"no 16:00 row for {session.date()} in the recalibrated table"
        )
    row = px[m].iloc[-1]
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
) -> dict[str, float]:
    """End to end for one session: placeholder row, extended root, arm, MZ."""
    store.append_panel(
        placeholder_row(store, session), source="placeholder", placeholder=True
    )
    ext = build_ext_root(repo, store, Path(scratch) / "ext_root", fomc_csv)
    csv = run_arm(ext, Path(scratch) / "arm", python)
    return rv_hat_for(session, yhat_table(csv), repo, Path(scratch))


def assert_placeholder_invariance(
    repo: Path,
    store: StateStore,
    past_session: pd.Timestamp,
    scratch: Path,
    fomc_csv: Path,
    python: str = sys.executable,
    rel_tol: float = 1e-9,
) -> dict[str, float]:
    """The claim the whole design rests on, tested on a session whose target is known.

    Run the arm on the panel as is, then with that session's 16:00 target
    replaced by a placeholder; ``pred_adj`` and the baseline of the 16:00 row
    must agree to ``rel_tol``.  Returns the two predictions and the deviation.
    """
    ext = build_ext_root(repo, store, Path(scratch) / "inv_true", fomc_csv)
    a = yhat_table(run_arm(ext, Path(scratch) / "inv_true_arm", python))
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
    b = yhat_table(run_arm(ext2, Path(scratch) / "inv_dummy_arm", python))
    tu = pd.Timestamp(t, tz=ET).tz_convert("UTC").as_unit("us")
    ya = a.set_index("t").loc[tu]
    yb = b.set_index("t").loc[tu]
    dev = max(
        abs(float(ya["yhat"]) - float(yb["yhat"]))
        / max(abs(float(ya["yhat"])), 1e-300),
        abs(float(ya["baseline"]) - float(yb["baseline"]))
        / max(abs(float(ya["baseline"])), 1e-300),
    )
    if dev > rel_tol:
        raise AssertionError(
            f"placeholder invariance FAILED on {past_session.date()}: yhat {ya['yhat']} vs "
            f"{yb['yhat']}, baseline {ya['baseline']} vs {yb['baseline']} (rel dev {dev:.2e})"
        )
    return {
        "yhat_true": float(ya["yhat"]),
        "yhat_dummy": float(yb["yhat"]),
        "rel_dev": float(dev),
    }


__all__ = [
    "BUCKET",
    "ESTIMATOR",
    "SEGMENT",
    "TRAIN_WIN",
    "assert_placeholder_invariance",
    "build_ext_root",
    "fomc_release_rows",
    "forecast_session",
    "placeholder_row",
    "run_arm",
    "rv_hat_for",
    "yhat_table",
]
