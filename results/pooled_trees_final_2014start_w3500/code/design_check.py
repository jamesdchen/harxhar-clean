"""Final-design checks for the pooled tree walks of the final rerun (first evaluation day 2014-01-02).

For each window W, on the copied caches <designs>/w<W>/_work/design_{baseline,all_features}/ (the 2014-start window
sweep's design stage, results/window_sweep_2014start_2026-10-22/w<W>):
  * the tree runner's own check (experiments/pooled_trees_rerun_13.py check: the linear runner's check_design, the
    column count, the window of design.json), the column names against the sweep's frozen_penalties.json, shapes;
  * y, b_oos and dates_oos equal across the two sets;
  * S = the design row of the first 2014-01-02 bar (folds14.json: 50,777), the linear walk's first row (the 250-solve
    re-choice boundary at or before S, experiments/pooled_linear_rerun_2026_10_19_13.walk_start) and its date
    (2013-12-27 16:00 in the sweep's forecasts.parquet), and the tree walk's span 2014-01-02 .. 2024-04-30 (first and
    last row, rows, days; the fit window of the first row);
  * the sweep's linear forecasts.parquet (committed) on the walk rows: dates, B (= b_oos) and true_adj (= y up to the
    CSV round trip of the linear tables) -- so these caches are the design the sweep's linear rows were fitted on.

usage: python design_check.py <repo> <designs root> <out json> W [W ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO, DROOT, OUTJ = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
WS = [int(w) for w in sys.argv[4:]]
sys.path.insert(0, str(REPO / "experiments"))
import pooled_trees_rerun_13 as PT

assert PT.ROOT.resolve() == REPO.resolve(), (PT.ROOT, REPO)
FIRST, LAST = "2014-01-02", "2024-04-30"
S_WANT = 50777  # results/window_sweep_2014start_2026-10-22/folds14.json
TUNE_PER = 250  # the linear walk's re-choice period (pooled_linear_rerun_2026_10_19_13.TUNE_PER)
out = {}
for W in WS:
    root = DROOT / f"w{W}"
    Wr = W * PT.PPD
    sweep = REPO / "results" / "window_sweep_2014start_2026-10-22" / f"w{W}"
    fp = json.loads((sweep / "frozen_penalties.json").read_text())
    rec: dict = {"window_days": W, "window_rows": Wr}
    arr = {}
    for s in PT.SETS:
        meta = PT.check(root, s, W)
        names = meta["names"]
        assert names == fp["designs"][s]["columns"], (
            f"w{W} {s}: columns differ from the sweep's frozen_penalties.json"
        )
        dz = PT.design_dir(root, s)
        X = np.load(dz / "X.npy", mmap_mode="r")
        y, b = np.load(dz / "y.npy"), np.load(dz / "b_oos.npy")
        d = PT.oos_dates(root, s)
        assert (
            X.shape == tuple(meta["shape"])
            and X.shape[1] == len(names)
            and len(y) == X.shape[0] == Wr + len(d)
        )
        arr[s] = (y, b, d)
        rec[f"{s}_shape"] = list(X.shape)
        rec[f"{s}_config"] = meta["config"]
        rec[f"{s}_fortran_order"] = bool(X.flags.f_contiguous)
    (y, b, d), (y2, b2, d2) = arr["baseline"], arr["all_features"]
    assert np.array_equal(y, y2) and np.array_equal(b, b2) and d.equals(d2), (
        f"w{W}: y / B / dates differ across sets"
    )
    S = Wr + int(np.flatnonzero(d >= pd.Timestamp(FIRST))[0])
    assert S == S_WANT, (W, S)
    ws = (
        S - Wr
    ) // TUNE_PER * TUNE_PER + Wr  # the linear walk's first design row (walk_start)
    rows, last = PT.span_rows(d, FIRST, LAST)
    assert Wr + int(rows[0]) == S
    rec |= {
        "oos_first": str(d[0]),
        "oos_last": str(d[-1]),
        "n_oos": len(d),
        "S": S,
        "S_date": str(d[S - Wr]),
        "linear_walk_start_row": ws,
        "linear_walk_start_date": str(d[ws - Wr]),
        "tree_first_row": S,
        "tree_first_date": str(d[rows[0]]),
        "tree_last_date": str(d[rows[-1]]),
        "tree_rows": len(rows),
        "tree_days": int(d.normalize()[rows].nunique()),
        "tree_first_fit_rows": [S - Wr, S],
        "tree_first_fit_first_date": str(d[S - 2 * Wr])
        if S - Wr >= Wr
        else f"design row {S - Wr} (before the first OOS row)",
    }
    # the sweep's linear forecasts on the walk rows (ridge, HAR benchmark: every model has the same rows)
    lf = pd.read_parquet(
        sweep / "forecasts.parquet",
        columns=["model", "predictor_set", "date", "true_adj", "B"],
    )
    for (m, s), g in lf.groupby(["model", "predictor_set"]):
        g = g.sort_values("date")
        dd = pd.DatetimeIndex(g["date"])
        i0 = ws - Wr
        assert dd.equals(d[i0:]), f"w{W} {m} {s}: walk dates"
        assert np.array_equal(g["B"].to_numpy(float), b[i0:]), f"w{W} {m} {s}: B"
        t, yy = g["true_adj"].to_numpy(float), y[ws:]
        assert np.allclose(t, yy, rtol=1e-14, atol=0), (W, m, s)
        rec.setdefault("sweep_linear_checked", []).append(
            f"{m}/{s}: {len(g)} rows, B equal, true_adj = y ({int((t == yy).sum())} bit-equal, max |diff| "
            f"{float(np.max(np.abs(t - yy))):.1e})"
        )
    out[f"w{W}"] = rec
    print(json.dumps(rec, indent=1, default=str), flush=True)
OUTJ.write_text(json.dumps(out, indent=1, default=str))
print(f"design checks passed for {WS} -> {OUTJ}")
