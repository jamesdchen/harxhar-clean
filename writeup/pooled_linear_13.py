"""The pooled linear models of the 13-bar rerun from 2012, read at the 16:00 bar (2026-10-22, note E34: the 12 pooled
forecasts of Section 7 on one design).

The like-for-like counterpart of the pooled trees of pooled_trees_13.py: the same 13 regular-hours bars, base-5 HAR
horizons, FOMC series in, 1,000-trading-day rolling window and first evaluation day, so the six pooled linear rows and
the six pooled trees of the economics section share one design.  make_rehaul_economics.py reads them in place of the
master table's poolpt_* rows (the 48-interval grid, 2,000-day window).

SOURCE
  results/pooled_linear_rerun_2026-10-20_13_2012_w1000/forecasts.parquet (+ frozen_penalties.json, scores.csv,
  SUMMARY.md), branch insurance-selling-2026-09-19-rerun-2012-w1000 (eb78186), runner
  experiments/pooled_linear_rerun_2026_10_19_13.py with --window-days 1000 --first-day 2012-01-03.  Ridge, lasso and
  elastic net on the HAR benchmark (12 columns) and on all features (576), and the min-norm OLS on the HAR benchmark;
  penalties frozen from 4 pre-sample folds; a rolling window of 1,000 trading days (13,000 bars), refit at every bar;
  40,497 bars a model, from 2011-12-28 14:00 (the 250-solve boundary before 3 January 2012) to 30 April 2024.

RULE
  plain = pred_adj^2 x B, the forecast of the bar's variance (asserted against the stored column); rv_actual is the
  bar's realized variance.  The 16:00 row is the forecast of the 15:30--16:00 ET variance made at 15:30, the forecast
  the closing trade uses (as pooled_trees_13.at16).  The dates are naive ET bar ends.

CHECK
  On the run's own scoring days (scores.csv: first .. last, n_days), the mean QLIKE of the 16:00 rows reproduces
  scores.csv to 1e-10 (as writeup/make_rehaul_oos_rerun.py checks), and every arm's penalty is the frozen one.

FINAL RUN (2026-10-23; writeup/final_run.py, REHAUL_FINAL_RUN=1)
  RUN = final_run.LINEAR_RUN, results/window_sweep_2014start_2026-10-22/w3000 (w3500 under REHAUL_FINAL_WINDOW=3500):
  the final specification (LATE_START_WARMUP, UNOBSERVED_NEUTRAL, EXCLUDE_FOMC_RELEASE, WALK_KKT_RESEED,
  STANDARDIZE_PENALTY with the availability screen), penalties frozen from pre-sample fold 3 (validation 2013), a
  rolling window of 3,000 trading days (39,000 bars), refit at every bar, first evaluation day 2 January 2014; 33,997
  rows a model from 2013-12-27 16:00 (the 250-solve boundary before 2 January 2014) to 30 April 2024.  The gates are
  the same, with those constants; the run's run_info.json must name the first day and the window.
"""

from __future__ import annotations

import json
from pathlib import Path

import final_run as fr  # 2026-10-23: the switch to the final pooled rerun (REHAUL_FINAL_RUN)
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "pooled_linear_rerun_2026-10-20_13_2012_w1000"
FC = RUN / "forecasts.parquet"
EST = ("enet", "lasso", "ridge")
SETS = ("baseline", "all_features")
ARMS = {(e, s) for e in EST for s in SETS} | {("ols", "baseline")}
N_BARS_ARM = 40_497  # bars of each model (SUMMARY.md: 7 x 40,497 = 283,479 rows)
N_BARS_DAY = 13  # 10:00 .. 16:00 ET bar ends
FIRST_DAY = pd.Timestamp("2012-01-03")  # --first-day of the run (SUMMARY.md)
LAST_DAY = pd.Timestamp("2024-04-30")
WALK_START = pd.Timestamp(
    "2011-12-28 14:00"
)  # the 250-solve boundary at or before the first 2012-01-03 row
if fr.FINAL:  # 2026-10-23: the final pooled rerun (writeup/final_run.py); the printed run's constants above
    RUN = fr.LINEAR_RUN
    FC = RUN / "forecasts.parquet"
    N_BARS_ARM = fr.N_LINEAR_ROWS
    FIRST_DAY, LAST_DAY = fr.FIRST_DAY, fr.LAST_DAY
    WALK_START = fr.LINEAR_WALK_START
_CACHE: dict[str, pd.DataFrame] = {}


def meta() -> dict:
    """The run's window and span (frozen_penalties.json, SUMMARY.md), and the bars a full day has."""
    info = json.loads((RUN / "frozen_penalties.json").read_text())["info"]
    assert int(info["window_rows"]) == int(info["window_days"]) * N_BARS_DAY, info
    if fr.FINAL:  # 2026-10-23: the final design's window, first day and tuning fold, from the run's own records
        ri = json.loads((RUN / "run_info.json").read_text())
        fr.guard_standin(ri, RUN)
        assert (
            int(info["window_days"]) == int(ri["window_days"]) == fr.WINDOW_DAYS
            and pd.Timestamp(ri["first_day"]) == fr.FIRST_DAY
            and tuple(int(f["fold"]) for f in info["folds"]) == fr.TUNE_FOLDS
            and {pd.Timestamp(f["val_first"]).year for f in info["folds"]}
            == {pd.Timestamp(f["val_last"]).year for f in info["folds"]}
            == {fr.VAL_YEAR}
        ), (RUN, info["window_days"], ri["first_day"], info["folds"])
    return {
        "window_days": int(info["window_days"]),
        "bars_day": N_BARS_DAY,
        "window_bars": int(info["window_rows"]),
        "span": (FIRST_DAY, LAST_DAY),
    }


def _frame() -> pd.DataFrame:
    if "f" not in _CACHE:
        f = pd.read_parquet(
            FC,
            columns=[
                "model",
                "predictor_set",
                "alpha",
                "date",
                "pred_adj",
                "B",
                "plain",
                "rv_actual",
            ],
        )
        n = f.groupby(["model", "predictor_set"]).size()
        assert set(n.index) == ARMS, sorted(n.index)
        assert (n == N_BARS_ARM).all(), n.to_dict()
        v = f[["pred_adj", "B", "plain", "rv_actual"]].to_numpy(float)
        assert np.isfinite(v).all() and (f["plain"] > 0).all() and (f["B"] > 0).all()
        rel = float(np.max(np.abs(f["plain"] / (f["pred_adj"] ** 2 * f["B"]) - 1.0)))
        assert rel <= 1e-12, rel
        t = pd.to_datetime(f["date"])
        clock = t.dt.strftime("%H:%M")
        assert (
            clock.nunique() == N_BARS_DAY
            and clock.min() == "10:00"
            and clock.max() == "16:00"
        ), sorted(clock.unique())
        assert t.min() == WALK_START and t.dt.normalize().max() == LAST_DAY, (
            t.min(),
            t.max(),
        )
        pen = json.loads((RUN / "frozen_penalties.json").read_text())
        for (m, s), g in f.groupby(["model", "predictor_set"]):
            a = pen[m][s]
            got = g["alpha"].unique()
            assert len(got) == 1 and (
                np.isnan(got[0]) if a is None else np.isclose(got[0], a, rtol=1e-12)
            ), (m, s, got, a)
        _CACHE["f"] = f
        _check_scores()
    return _CACHE["f"]


def at16(est: str, pset: str) -> pd.DataFrame:
    """The 16:00 rows of one model, one row a trading day: plain and rv (the bar's realized variance)."""
    f = _frame()
    g = f[(f["model"] == est) & (f["predictor_set"] == pset)]
    assert len(g) == N_BARS_ARM, (est, pset)
    t = pd.to_datetime(g["date"])
    g = g[((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()]
    day = pd.DatetimeIndex(pd.to_datetime(g["date"])).normalize().as_unit("ns")
    out = pd.DataFrame(
        {"plain": g["plain"].to_numpy(float), "rv": g["rv_actual"].to_numpy(float)},
        index=day,
    ).sort_index()
    assert not out.index.duplicated().any(), (est, pset)
    return out


def plain_16(est: str, pset: str) -> pd.Series:
    """plain of the 16:00 rows of one model, by trading day."""
    return at16(est, pset)["plain"]


def _check_scores() -> None:
    """The 16:00 rows reproduce the run's own closing-interval QLIKE (scores.csv) on its scoring days."""
    sc = pd.read_csv(RUN / "scores.csv")
    assert {(m, s) for m, s in zip(sc["model"], sc["predictor_set"])} == ARMS, sc[
        ["model", "predictor_set"]
    ]
    for _, r in sc.iterrows():
        a = at16(r["model"], r["predictor_set"])
        a = a[
            (a.index >= pd.Timestamp(r["first"])) & (a.index <= pd.Timestamp(r["last"]))
        ]
        a = a[a["rv"] > 0]
        x = (a["rv"] / a["plain"]).to_numpy(float)
        q = float(np.mean(x - np.log(x) - 1.0))
        assert len(a) == int(r["n_days"]) and np.isclose(
            q, float(r["qlike"]), rtol=1e-10, atol=0
        ), (
            r["model"],
            r["predictor_set"],
            len(a),
            q,
            r["qlike"],
        )
