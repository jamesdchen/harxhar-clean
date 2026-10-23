"""The pooled tree ensembles of the 13-bar rerun, read at the 16:00 bar (2026-10-22, note E34: base 5 across the paper).

One loader for every generator that prints a pooled-tree number (make_rehaul_economics.py, make_rehaul_results.py,
make_rehaul_oos_table.py), so they read the same rows.

SOURCE
  results/pooled_trees_rerun13_2026-10-20/w1000/forecasts.parquet (+ forecasts.json, STATUS.md), from branch
  insurance-selling-2026-09-19-trees-results at 6746103: CARC Discovery jobs 13262219, 13265608, 13265651 (walk) and
  13278970 (merge), runner experiments/pooled_trees_rerun_13.py at db59af2.  LightGBM, XGBoost and the random forest
  (frozen configs cluster/pooltree_lgbm_config.json, cluster/xgbtop_xgb_config.json, cluster/pooltree_rf400_config.json)
  on the HAR benchmark (12 columns) and on all features (576), fitted on the design of the pooled linear 13-bar rerun
  of Table 1 (the 13 regular-hours bars, base-5 HAR horizons, FOMC series in): a rolling window of 1,000 trading days
  (13,000 bars), refit at every bar, 3 January 2012 .. 30 April 2024, 40,466 bars an arm.

RULE
  plain = pred_adj^2 x B, the forecast of the bar's variance (asserted against the stored column); rv_actual is the
  bar's realized variance.  The 16:00 row is the forecast of the 15:30--16:00 ET variance made at 15:30, the forecast
  the closing trade uses (the 16:00 rule of make_rehaul_results.plain_16).  The dates are naive ET bar ends.

FINAL RUN (2026-10-23; writeup/final_run.py, REHAUL_FINAL_RUN=1)
  LightGBM and XGBoost from final_run.TREES_RUN (results/pooled_trees_final_2014start_w3000/forecasts.parquet, model
  keys lgbm / xgb), the random forest from final_run.RF_RUN (results/pooled_rf_skx_2014start_w3000/forecasts.parquet,
  the frozen forest on Intel's oneDAL engine, model key final_run.RF_KEY = rf_skx, read as "rf"), both merged by
  experiments/pooled_trees_rerun_13.py merge (the columns above; forecasts.json: refit, span, window_days): the frozen
  configurations, refit at every bar on a rolling window of 3,000 trading days (39,000 bars), 2 January 2014 ..
  30 April 2024, 33,970 bars an arm (the linear walk's rows from the first evaluation day).  The two runs must agree on
  refit, window and first day; a null last day in forecasts.json (merge without --last-day) reads as 30 April 2024,
  which the rows must reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import final_run as fr  # 2026-10-23: the switch to the final pooled rerun (REHAUL_FINAL_RUN)
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "pooled_trees_rerun13_2026-10-20" / "w1000"
FC = RUN / "forecasts.parquet"
EST = ("lgbm", "xgb", "rf")
SETS = ("baseline", "all_features")
N_BARS_ARM = 40_466  # bars of each arm (STATUS.md: 6 arms x 40,466 = 242,796 rows)
N_BARS_DAY = 13  # 10:00 .. 16:00 ET bar ends
# 2026-10-23: the final pooled rerun (writeup/final_run.py); the printed run's constants above.  The random forest is
# a run of its own (RF_RUN, model key RF_KEY), read as "rf"; RF_RUN is None for the printed run (one file).
RF_RUN: Path | None = None
RF_KEY = "rf"
if fr.FINAL:
    RUN = fr.TREES_RUN
    FC = RUN / "forecasts.parquet"
    RF_RUN, RF_KEY = fr.RF_RUN, fr.RF_KEY
    N_BARS_ARM = fr.N_BARS
_CACHE: dict[str, pd.DataFrame] = {}


def _span(m: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(first, last) day of a forecasts.json; under the final design a null last day (merge without --last-day) is
    the design's last day, which _frame() asserts the rows reach."""
    last = m["span"][1]
    if last is None and fr.FINAL:
        last = fr.LAST_DAY
    return pd.Timestamp(m["span"][0]), pd.Timestamp(last)


def meta() -> dict:
    """The run's window, cadence and span (forecasts.json), and the bars a full day has."""
    m = json.loads((RUN / "forecasts.json").read_text())
    if fr.FINAL:
        fr.guard_standin(m, RUN)
    # 2026-10-23 (final run): the span through _span (a null last day); the forest's run must match. Was:
    # "span": (pd.Timestamp(m["span"][0]), pd.Timestamp(m["span"][1])),
    out = {
        "refit": m["refit"],
        "window_days": int(m["window_days"]),
        "bars_day": N_BARS_DAY,
        "window_bars": int(m["window_days"]) * N_BARS_DAY,
        "span": _span(m),
    }
    if RF_RUN is not None:
        r = json.loads((RF_RUN / "forecasts.json").read_text())
        fr.guard_standin(r, RF_RUN)
        assert (r["refit"], int(r["window_days"]), _span(r)) == (
            out["refit"],
            out["window_days"],
            out["span"],
        ), ("the forest's run is on another design", RF_RUN, r, out)
        assert out["window_days"] == fr.WINDOW_DAYS and out["span"] == (
            fr.FIRST_DAY,
            fr.LAST_DAY,
        ), ("the final tree run is not on the final design", RUN, out)
    return out


def _frame() -> pd.DataFrame:
    if "f" not in _CACHE:
        f = pd.read_parquet(
            FC,
            columns=[
                "model",
                "predictor_set",
                "date",
                "pred_adj",
                "B",
                "plain",
                "rv_actual",
            ],
        )
        # 2026-10-23 (final run): LightGBM / XGBoost here, the forest from its own run
        if RF_RUN is not None:
            got = set(f["model"])
            assert got == {"lgbm", "xgb"}, (
                f"{FC}: models {sorted(got)}; the final tree run holds lgbm and xgb (the forest is {RF_RUN})"
            )
            r = pd.read_parquet(RF_RUN / "forecasts.parquet", columns=list(f.columns))
            assert set(r["model"]) == {RF_KEY}, (
                RF_RUN,
                sorted(set(r["model"])),
                RF_KEY,
            )
            r["model"] = "rf"
            f = pd.concat([f, r], ignore_index=True)
        n = f.groupby(["model", "predictor_set"]).size()
        assert set(n.index) == {(e, s) for e in EST for s in SETS}, sorted(n.index)
        assert (n == N_BARS_ARM).all(), n.to_dict()
        v = f[["pred_adj", "B", "plain", "rv_actual"]].to_numpy(float)
        assert np.isfinite(v).all() and (f["plain"] > 0).all() and (f["B"] > 0).all()
        rel = float(np.max(np.abs(f["plain"] / (f["pred_adj"] ** 2 * f["B"]) - 1.0)))
        assert rel <= 1e-12, rel
        clock = pd.to_datetime(f["date"]).dt.strftime("%H:%M")
        assert clock.nunique() == N_BARS_DAY and clock.max() == "16:00", sorted(
            clock.unique()
        )
        m = meta()
        assert m["refit"] == "bar", m
        d = pd.to_datetime(f["date"]).dt.normalize()
        assert d.min() == m["span"][0] and d.max() == m["span"][1], (
            d.min(),
            d.max(),
            m["span"],
        )
        _CACHE["f"] = f
    return _CACHE["f"]


def at16(est: str, pset: str) -> pd.DataFrame:
    """The 16:00 rows of one arm, one row a trading day: plain and rv (the bar's realized variance)."""
    f = _frame()
    g = f[(f["model"] == est) & (f["predictor_set"] == pset)]
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
    """plain of the 16:00 rows of one arm, by trading day."""
    return at16(est, pset)["plain"]
