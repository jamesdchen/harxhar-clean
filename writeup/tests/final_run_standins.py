"""STAND-IN inputs for testing the final-run switch (writeup/final_run.py) before the final tree, forest and MinGRU runs
land (2026-10-23).  NOT RESULTS: every number built from them is a test of the plumbing only.

Each stand-in is the PRINTED run's forecasts (the 1,000-day, 2012-start design) cut to the final design's bars, in the
format the final runs will have, with forecasts.json "standin" set (final_run.guard_standin refuses them unless
REHAUL_ALLOW_STANDIN=1 and the output directories are outside writeup/):

  trees_w<W>/   LightGBM and XGBoost of results/pooled_trees_rerun13_2026-10-20/w1000 from 2014-01-02 (33,970 bars an
                arm), forecasts.json as experiments/pooled_trees_rerun_13.py merge writes it (refit, span, window_days)
  rf_w<W>/      the printed stock random forest, the same bars, model key renamed to rf_skx (the final forest's key),
                forecasts.json with a null last day (merge without --last-day), to exercise that branch
  mingru_w<W>/  results/pooled_mingru_rerun13_2026-10-22 from 2014-01-02 10:00, forecasts.json as
                experiments/pooled_mingru_final_2014_w3000.py merge writes it (window_days, refit, span)

Asserted: the stand-ins' bars are the final linear walk's bars from the first evaluation day
(results/window_sweep_2014start_2026-10-22/w<W>).

Run:  python writeup/tests/final_run_standins.py OUT_DIR [--window 3000]
      then REHAUL_FINAL_RUN=1 REHAUL_ALLOW_STANDIN=1 REHAUL_TREES_RUN=OUT_DIR/trees_w3000 REHAUL_RF_RUN=OUT_DIR/rf_w3000
      REHAUL_MINGRU_RUN=OUT_DIR/mingru_w3000 REHAUL_GEN_DIR=... REHAUL_FIG_DIR=... REHAUL_RES_DIR=... python writeup/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
FIRST = pd.Timestamp("2014-01-02")
LAST = pd.Timestamp("2024-04-30")
NOTE = (
    "STAND-IN for a test of writeup/final_run.py: the printed {src} forecasts (1,000-day window, 2012-start design) "
    "cut to the final design's bars; NOT the final run"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--window", type=int, default=3000)
    a = ap.parse_args()
    W = a.window
    lin = pd.read_parquet(
        RES / "window_sweep_2014start_2026-10-22" / f"w{W}" / "forecasts.parquet",
        columns=["model", "predictor_set", "date"],
    )
    bars = {
        k: set(pd.to_datetime(g["date"])[pd.to_datetime(g["date"]) >= FIRST])
        for k, g in lin.groupby(["model", "predictor_set"])
    }
    want = bars[("ridge", "baseline")]
    assert all(v == want for v in bars.values()) and len(want) == 33_970, len(want)

    src = RES / "pooled_trees_rerun13_2026-10-20" / "w1000" / "forecasts.parquet"
    t = pd.read_parquet(src)
    t = t[pd.to_datetime(t["date"]) >= FIRST].reset_index(drop=True)
    for k, g in t.groupby(["model", "predictor_set"]):
        assert set(pd.to_datetime(g["date"])) == want, k
    for name, models, key, last in (
        ("trees", ("lgbm", "xgb"), None, str(LAST.date())),
        ("rf", ("rf",), "rf_skx", None),
    ):
        d = a.out / f"{name}_w{W}"
        d.mkdir(parents=True, exist_ok=True)
        f = t[t["model"].isin(models)].copy()
        if key:
            f["model"] = key
        f.to_parquet(d / "forecasts.parquet", index=False)
        meta = {
            "refit": "bar",
            "span": [str(FIRST.date()), last],
            "window_days": W,
            "standin": NOTE.format(
                src=f"{'/'.join(models)} of {src.parent.relative_to(REPO)}"
            ),
        }
        (d / "forecasts.json").write_text(json.dumps(meta, indent=1))
        (d / "STANDIN.md").write_text(meta["standin"] + "\n")
        print(f"{d}: {len(f)} rows, models {sorted(set(f['model']))}")

    src = RES / "pooled_mingru_rerun13_2026-10-22" / "forecasts.parquet"
    m = pd.read_parquet(src)
    m = m[pd.to_datetime(m["date"]) >= FIRST].reset_index(drop=True)
    for k, g in m.groupby(["model", "predictor_set"]):
        assert set(pd.to_datetime(g["date"])) == want, k
    d = a.out / f"mingru_w{W}"
    d.mkdir(parents=True, exist_ok=True)
    m.to_parquet(d / "forecasts.parquet", index=False)
    meta = {
        "window_days": W,
        "refit": "every bar (STAND-IN)",
        "span": [f"{FIRST.date()} 10:00:00", f"{LAST.date()} 16:00:00"],
        "standin": NOTE.format(src=str(src.parent.relative_to(REPO))),
    }
    (d / "forecasts.json").write_text(json.dumps(meta, indent=1))
    (d / "STANDIN.md").write_text(meta["standin"] + "\n")
    print(f"{d}: {len(m)} rows")


if __name__ == "__main__":
    main()
