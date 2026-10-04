"""Capture the per-bar design matrix the executor hands a model, for the close studies.

The 2026-10-03 studies (data size for trees, block-diagonal shrinkage, tree tuning)
all fit on the design of the per-bar arms: specs/causal_tune_trees.py's run_executor
call (diurnal -> sqrt -> winsorize target, winsor 240, HAR ladder + calendar,
impute_indicate, dropna_with_exog=False, prescale, global lags, horizon 1,
TRAIN_WIN = 2000 sessions).  This script runs that call once per (segment, bucket)
with a fit_predict that keeps the matrix and returns the training-window mean (the
executor needs some forecast to write its table; nothing reads it), so the studies
share one copy instead of each rebuilding it.

Segments: ``bar1600`` (the 15:30-16:00 bar, one row a session; the per-bar arms'
design) and ``last30`` (the two bars of the last hour, ending 15:30 and 16:00; the
calendar column ``hour`` tells them apart).  For a multi-bar segment the executor's
window is TRAIN_WIN sessions x the median bars a session (4000 rows for last30), and
the rolling robust scaling uses that window, so the 16:00 rows of the two designs
are not identical.

``lastbars<N>`` (added 2026-10-04 for the bar-count ablation): the N half-hour bars
ending 16:00 (bar-end labels 16:00 - 30 (N - 1) min .. 16:00), registered here in the
executor's segment table at run time (src/ is not edited).  lastbars2 is last30 and
lastbars5 is ``closing``; lastbars13 is the 13 regular-hours bars (labels 10:00 ..
16:00).  The calendar column ``hour`` is 16 only on the 16:00 row, so a model can
always isolate the target bar; earlier bars share an hour value in pairs.

Writes ``$CLOSE_DESIGN_DIR/design_<segment>_<bucket>.npz`` (default
results/close_design/_work, never committed): X, y, W (rows), names, date (bar-end
stamps of every row), baseline (every row's diurnal scale: the raw variance level of a
forecast is baseline * forecast**2), true_raw (the forecast rows' realized variance).

Usage:  python experiments/capture_design_close.py <segment> <bucket> [<bucket> ...]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

import src.backtest.executor as ex  # noqa: E402
import src.backtest.segmentation as sg  # noqa: E402
from src.data.loading import get_bucket  # noqa: E402

OUT = Path(
    os.environ.get("CLOSE_DESIGN_DIR", str(REPO / "results" / "close_design" / "_work"))
)
TRAIN_WIN = 2000  # sessions, the per-bar arms' window (specs/causal_tune_trees.py)


def register_lastbars(segment: str) -> None:
    """Add ``lastbars<N>`` (the N bars ending 16:00) to the executor's segment table."""
    if not segment.startswith("lastbars") or segment in sg.SEGMENT_DEFINITIONS:
        return
    n = int(segment.removeprefix("lastbars"))
    if not 1 <= n <= 13:
        raise SystemExit(
            f"lastbars<N> needs 1 <= N <= 13 (regular-hours bars), got {n}"
        )
    close = 16 * 60
    # the executor imported this same dict object, so the entry is visible to it
    sg.SEGMENT_DEFINITIONS[segment] = (close - 30 * (n - 1), close)
    sg.SEGMENT_CHOICES.append(segment)


def capture(segment: str, bucket: str) -> Path:
    register_lastbars(segment)
    box: dict = {}
    shift = ex.apply_horizon_shift

    def keep_stamps(X, y, dates, baselines, horizon):
        # the executor's own shift, unchanged; its outputs (every row's bar-end stamp and
        # diurnal scale) are kept, because the executor's table carries the forecast rows only
        out = shift(X, y, dates, baselines, horizon)
        box.update(
            date_all=pd.Series(out[2]).astype(str).to_numpy(),
            baseline_all=np.asarray(out[3], float),
        )
        return out

    ex.apply_horizon_shift = keep_stamps

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        W = int(train_win_periods)
        box.update(
            X=np.array(X_chunk, dtype=np.float64, copy=True),
            y=np.array(y_chunk, dtype=np.float64, copy=True),
            W=W,
            names=[str(c) for c in hyperparams["_feature_names"]],
        )
        return np.full(len(X_chunk) - W, float(np.mean(y_chunk[:W])))

    tmp = OUT / "executor" / segment / bucket / "results.csv"
    t0 = time.time()
    ex.run_executor(
        method_name=f"capture_{segment}_{bucket}",
        fit_predict=fit_predict,
        hyperparams={},
        data_path="data",
        output_file=str(tmp),
        horizon=1,
        train_window=TRAIN_WIN,
        start=0,
        end=-1,
        halo=0,
        exog_cols=get_bucket(bucket),
        segment=segment,
        lag_scope="global",
        har_lags=None,
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=42,
    )
    ex.apply_horizon_shift = shift
    res = pd.read_csv(tmp.with_name(f"results_{segment}.csv"))
    X, y, W = box["X"], box["y"], box["W"]
    date_all, baseline_all = box["date_all"], box["baseline_all"]
    assert len(X) - W == len(res) and len(date_all) == len(X), (
        len(X),
        W,
        len(res),
        len(date_all),
    )
    assert np.max(np.abs(y[W:] / res["true_adj"].to_numpy(float) - 1.0)) < 1e-12
    assert (pd.to_datetime(date_all[W:]) == pd.to_datetime(res["date"])).all()
    true_raw = res["true_raw"].to_numpy(float)
    out = OUT / f"design_{segment}_{bucket}.npz"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        y=y,
        W=W,
        names=np.array(box["names"]),
        date=pd.to_datetime(date_all)
        .strftime("%Y-%m-%d %H:%M:%S")
        .to_numpy()
        .astype("U19"),
        baseline=baseline_all,
        true_raw=true_raw,
    )
    print(
        f"captured {segment} {bucket}: X {X.shape}, W {W}, {len(res)} forecast rows "
        f"({res['date'].iloc[0]} .. {res['date'].iloc[-1]}), {time.time() - t0:.0f}s -> {out}",
        flush=True,
    )
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    seg = sys.argv[1]
    for b in sys.argv[2:]:
        capture(seg, b)
