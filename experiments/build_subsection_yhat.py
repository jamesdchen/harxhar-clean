"""Assemble the per-bar (own-coefficient) forecasts into yhat tables the 0DTE notebooks read.

The subsection campaign (Hoffman2, 2026-09-20/21; `specs/causal_tune_linear.py`
with SEGMENT = one 30-minute bar, LAG_SCOPE = global) fits the linear pipeline's
coefficients on ONE regular-hours bar at a time.  Each one-bar arm writes
``results_<seg>.csv`` with the adjusted-scale forecast ``pred_adj`` for that bar
on every out-of-sample session.  Thirteen such arms (bar labels 10:00 .. 16:00)
cover the regular session; stacked, they are a forecast table in the notebooks'
format -- ``t`` (UTC), ``yhat`` (the adjusted-scale forecast), ``baseline`` (the
multiplicative profile B) and ``rv_raw`` (the bar's realized variance) -- so the
notebooks apply their OWN causal recalibration (the 250-session Mincer-Zarnowitz
map on the scored session bars) exactly as they do to the production forecasts.
Nothing from the spec's look-ahead back-transform (``pred_raw``) is carried.

The pooled arm of the same spec (``none``: one coefficient vector over all 48
bars, kept at its 13 regular-hours labels) is written beside each per-bar table
as its twin, so the notebook can attribute the difference to the coefficients.

Tables (results/spxw_pnl/):
  yhat_sub_<est>_<bucket>.parquet    per-bar, 2000-session window
  yhat_pool_<est>_<bucket>.parquet   pooled twin, same window
for <est> in ridge / lasso (the fixed recursive lasso, from the lassofix rerun) /
enet and <bucket> in all_features / baseline (HAR + calendar, no exogenous
columns).  Only tables whose 13 (or 1) source files all exist are written; the
all_features pooled lasso is still running on Hoffman2 at the time of writing.

rv_raw is taken from the production table (yhat_blk2_fomc1, which every
production table shares exactly), joined on the stamp.  The arm files carry
``true_raw`` = true_adj^2 x baseline, and true_adj is the spec's TARGET after
its rolling 5/95 winsorization, so on about 3% of rows true_raw is the clipped
target, not the bar's realized variance; the notebooks score every forecast
against the same unclipped realized variance, so the production column is the
one to carry.  The share of clipped rows is printed.

GATES  on the stamps shared with the production block-ridge table the baseline
       profile B agrees to 1e-9 relative (same panel, same profile); every
       stamp of a one-bar arm is in the production table.

Run:  python experiments/build_subsection_yhat.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ROOT / "results" / "linear_subsection" / "arms_hoffman2"
OUT = ROOT / "results" / "spxw_pnl"
REFERENCE = (
    OUT / "yhat_blk2_fomc1.parquet"
)  # the production table on the panel of record
TW = 2000
BARS = (
    "bar1000",
    "bar1030",
    "bar1100",
    "bar1130",
    "bar1200",
    "bar1230",
    "bar1300",
    "bar1330",
    "bar1400",
    "bar1430",
    "bar1500",
    "bar1530",
    "bar1600",
)
ESTIMATORS = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
BUCKETS = ("all_features", "baseline", "live_feasible")
GATE_REL = 1e-9


def read_arm(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, parse_dates=["date"])
    ok = (r["true_adj"] > 0) & (r["true_raw"] > 0) & np.isfinite(r["pred_adj"])
    r = r[ok]
    et = pd.DatetimeIndex(r["date"]).tz_localize(
        "America/New_York"
    )  # naive ET, bar-end labelled
    # microsecond resolution, as the production tables store t: a nanosecond
    # index would make the notebook's day frames "not identically labelled"
    return pd.DataFrame(
        {
            "t": et.tz_convert("UTC").as_unit("us"),
            "yhat": r["pred_adj"].to_numpy(float),
            "baseline": (r["true_raw"] / r["true_adj"] ** 2).to_numpy(float),
            "rv_raw": r["true_raw"].to_numpy(float),
        }
    )


def with_production_rv(tab: pd.DataFrame, name: str) -> pd.DataFrame:
    """Gate the profile against the production table and take its rv_raw."""
    ref = pd.read_parquet(REFERENCE).set_index("t")
    j = tab.set_index("t").join(ref, how="inner", rsuffix="_ref")
    assert len(j) == len(tab), (name, len(j), len(tab))
    db = float((j["baseline"] / j["baseline_ref"] - 1.0).abs().max())
    assert db < GATE_REL, (name, db)
    ok = j["rv_raw_ref"] > 0
    clipped = float(
        ((j["rv_raw"] / j["rv_raw_ref"].where(ok) - 1.0).abs() > GATE_REL).mean()
    )
    print(
        f"GATE  {name}: on {len(j):,} stamps baseline B agrees to {db:.1e}; the spec's reconstructed "
        f"target differs from the realized variance on {clipped:.1%} of rows (its winsorization) -- "
        "the production rv_raw is carried"
    )
    out = j[["yhat", "baseline", "rv_raw_ref"]].rename(columns={"rv_raw_ref": "rv_raw"})
    return out.reset_index().sort_values("t").reset_index(drop=True)


def main() -> None:
    written = []
    for bucket in BUCKETS:
        for est, short in ESTIMATORS.items():
            d = ARMS / bucket / est / f"tw{TW}"
            bars = [d / f"results_{b}.csv" for b in BARS]
            if all(p.exists() for p in bars):
                tab = (
                    pd.concat([read_arm(p) for p in bars])
                    .sort_values("t")
                    .reset_index(drop=True)
                )
                assert not tab["t"].duplicated().any()
                name = f"yhat_sub_{short}_{bucket}.parquet"
                tab = with_production_rv(tab, name)
                tab.to_parquet(OUT / name, index=False)
                first, last = tab["t"].min(), tab["t"].max()
                print(
                    f"wrote {name}: {len(tab):,} rows, {tab['t'].dt.normalize().nunique():,} sessions, "
                    f"{first.date()} .. {last.date()}"
                )
                written.append(name)
            else:
                print(
                    f"skip yhat_sub_{short}_{bucket}: {sum(not p.exists() for p in bars)} of 13 bar arms missing"
                )
            pooled = d / "results_none_rth.csv"
            if pooled.exists():
                tab = read_arm(pooled).sort_values("t").reset_index(drop=True)
                assert not tab["t"].duplicated().any()
                name = f"yhat_pool_{short}_{bucket}.parquet"
                tab = with_production_rv(tab, name)
                tab.to_parquet(OUT / name, index=False)
                print(
                    f"wrote {name}: {len(tab):,} rows, {tab['t'].dt.normalize().nunique():,} sessions, "
                    f"{tab['t'].min().date()} .. {tab['t'].max().date()}"
                )
                written.append(name)
            else:
                print(f"skip yhat_pool_{short}_{bucket}: pooled arm not available")
    print(f"{len(written)} tables written")


if __name__ == "__main__":
    main()
