"""The pooled ridge-init MinGRU of the 13-bar rerun, read at the 16:00 bar (2026-10-22; not read by any generator).

The counterpart of pooled_trees_13.py / pooled_linear_13.py for the network: the same 13 regular-hours bars, base-5 HAR
horizons, FOMC series in, 1,000-trading-day rolling window and walk start as the pooled linear 13-bar rerun.  The NN
content of the paper stays parked; this loader serves experiments/pooled_mingru_rerun_13_score.py and
writeup/MINGRU_RERUN_REPORT.md only.

SOURCE
  results/pooled_mingru_rerun13_2026-10-22/forecasts.parquet (+ forecasts.json, frozen_config.json, STATUS.md), from
  experiments/pooled_mingru_rerun_13.py (CARC Discovery, native_g9 trainer on P100): the pooled MinGRU with the
  ridge-initialised linear shortcut, trained, early-stopped and selected on QLIKE, one configuration a predictor set
  frozen from the 4 pre-2012 folds, refit every 21 sessions on its trailing 13,000 bars, the mean of 3 seeds;
  HAR benchmark (12 columns) and all features (576); 40,497 bars a set, 2011-12-28 14:00 .. 2024-04-30 16:00.

RULE
  plain = pred_adj^2 x B (pred_adj = the seed mean of the fit-space forecast f), asserted against the stored column;
  rv_actual is the bar's realized variance (data/core_stats.parquet sumret2).  The 16:00 row is the forecast of the
  15:30--16:00 ET variance made at 15:30 (as pooled_trees_13.at16).  The dates are naive ET bar ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "pooled_mingru_rerun13_2026-10-22"
FC = RUN / "forecasts.parquet"
SETS = ("baseline", "all_features")
N_BARS_ARM = (
    40_497  # the pooled linear rerun's walk rows (2011-12-28 14:00 .. 2024-04-30 16:00)
)
N_BARS_DAY = 13
_CACHE: dict[str, pd.DataFrame] = {}


def meta() -> dict:
    return json.loads((RUN / "forecasts.json").read_text())


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
        assert set(n.index) == {("mingru", s) for s in SETS}, sorted(n.index)
        assert (n == N_BARS_ARM).all(), n.to_dict()
        v = f[["pred_adj", "B", "plain", "rv_actual"]].to_numpy(float)
        assert np.isfinite(v).all() and (f["plain"] > 0).all() and (f["B"] > 0).all()
        rel = float(np.max(np.abs(f["plain"] / (f["pred_adj"] ** 2 * f["B"]) - 1.0)))
        assert rel <= 1e-12, rel
        clock = pd.to_datetime(f["date"]).dt.strftime("%H:%M")
        assert clock.nunique() == N_BARS_DAY and clock.max() == "16:00", sorted(
            clock.unique()
        )
        _CACHE["f"] = f
    return _CACHE["f"]


def bars(pset: str) -> pd.DataFrame:
    """Every bar of one set: plain and rv, indexed by the bar end."""
    f = _frame()
    g = f[f["predictor_set"] == pset]
    out = pd.DataFrame(
        {"plain": g["plain"].to_numpy(float), "rv": g["rv_actual"].to_numpy(float)},
        index=pd.DatetimeIndex(pd.to_datetime(g["date"])).as_unit("ns"),
    ).sort_index()
    assert not out.index.duplicated().any(), pset
    return out


def at16(pset: str) -> pd.DataFrame:
    """The 16:00 rows of one set, one row a trading day: plain and rv."""
    b = bars(pset)
    b = b[(b.index.hour == 16) & (b.index.minute == 0)]
    b.index = b.index.normalize()
    assert not b.index.duplicated().any(), pset
    return b


def plain_16(pset: str) -> pd.Series:
    return at16(pset)["plain"]
