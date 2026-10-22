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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "pooled_trees_rerun13_2026-10-20" / "w1000"
FC = RUN / "forecasts.parquet"
EST = ("lgbm", "xgb", "rf")
SETS = ("baseline", "all_features")
N_BARS_ARM = 40_466  # bars of each arm (STATUS.md: 6 arms x 40,466 = 242,796 rows)
N_BARS_DAY = 13  # 10:00 .. 16:00 ET bar ends
_CACHE: dict[str, pd.DataFrame] = {}


def meta() -> dict:
    """The run's window, cadence and span (forecasts.json), and the bars a full day has."""
    m = json.loads((RUN / "forecasts.json").read_text())
    return {
        "refit": m["refit"],
        "window_days": int(m["window_days"]),
        "bars_day": N_BARS_DAY,
        "window_bars": int(m["window_days"]) * N_BARS_DAY,
        "span": (pd.Timestamp(m["span"][0]), pd.Timestamp(m["span"][1])),
    }


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
