"""Score the pooled moments_vix tables with the plain master-table scorer (no recalibration).

The three tables yhat_pool_<ridge|lasso|enet>_moments_vix.parquet (experiments/
build_pool_moments_vix_yhat.py) are scored by experiments/master_table_close.py's OWN functions
(load_target, load_one, score_one; imported, not edited, the master table is not re-run): the
plain forecast f^2 B at the 16:00 bar, its QLIKE against the spec's target, and the last-30-min
trade q = sign(plain - iv_var) on the trade days, Sharpe at the midpoint and across the spread.
Beside them, the same pooled model on baseline and all_features, re-scored here and checked against
results/close_master_table/master_table.csv (GATE: every recomputed column equals the master
table's to 1e-12 relative -- same scorer, same days).

Writes results/linear_subsection_pool_momvix/score/pool_moments_vix_scores.csv (one row per forecast,
the master table's columns) and score_gates.csv.  Needs results/close_master_table/PLAIN_DONE.

Run:  python experiments/score_pool_moments_vix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import master_table_close as mtc  # noqa: E402

OUT = ROOT / "results" / "linear_subsection_pool_momvix" / "score"
MT = ROOT / "results" / "close_master_table"
ARM_ROOT = "linear_subsection_dedup"  # the master table's --arm-root (master_table_provenance.csv)
ESTS = ("ridge", "lasso", "enet")
# moments_vix dropped (user, 2026-10-06; its tables parked in results/spxw_pnl/parked_moments_vix_2026-10-06/):
# BUCKETS = ("baseline", "all_features", "moments_vix", "moments_vol_vix", "lean_vol_vix")
# moments_vol_vix dropped too (user, 2026-10-06: "comment out the non lean results everywhere"):
# BUCKETS = ("baseline", "all_features", "moments_vol_vix", "lean_vol_vix")
BUCKETS = (
    "baseline",
    "all_features",
    "lean_vol_vix",
)  # a bucket without tables is skipped
COLS = (
    "n_days",
    "first_day",
    "last_day",
    "qlike_raw",
    "qlike_raw_floored",
    "n_extreme_days",
    "qlike_pct_vs_ref",
    "dm_vs_ref",
    "dm_p_vs_ref",
    "Sharpe_mid",
    "Sharpe_crossed",
    "mean_mid",
    "mean_crossed",
    "hit_rate_mid",
    "hit_rate_crossed",
    "pct_buy",
    "same_position_as_ref",
    "dSharpe_mid_vs_short",
    "dSharpe_crossed_vs_short",
    "calib_mean_realized_over_forecast",
)
GATE_REL = 1e-12


def main() -> None:
    if not (MT / "PLAIN_DONE").is_file():
        sys.exit(
            "results/close_master_table/PLAIN_DONE not found: the plain master table is not final"
        )
    mtc.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    tables = sorted(mtc.SPXW.glob("yhat_*.parquet"))
    ref_spec = next(
        s
        for s in (mtc.describe_table(p, paper) for p in tables)
        if s.key == mtc.REFERENCE
    )
    ref_res = mtc.load_one(ref_spec)
    assert not ref_res["error"], ref_res["error"]
    ref = ref_res["frame"]
    mt = pd.read_csv(MT / "master_table.csv").set_index("key")

    gates, rows = [], []
    for bucket in BUCKETS:
        for est in ESTS:
            key = f"pool_{est}_{bucket}"
            if not (mtc.SPXW / f"yhat_{key}.parquet").is_file():
                print(f"skip {key}: no table")
                continue
            spec = mtc.describe_table(mtc.SPXW / f"yhat_{key}.parquet", paper)
            assert spec.key == key and spec.family == "pooled twin", (key, spec)
            res = mtc.load_one(spec)
            assert not res["error"], (key, res["error"])
            for g in res["gates"]:
                gates.append({**g, "source": "master_table_close.load_one"})
                assert g["ok"], g
            f = res["frame"]
            days = deck.index[
                deck.index.isin(f.index[f["plain"].notna()].normalize())
                & deck.index.isin(ref.index[ref["plain"].notna()].normalize())
            ]
            row = mtc.score_one((spec, f, ref, days, deck, False))
            row.update(
                label=spec.label, family=spec.family, bucket=bucket, estimator=est
            )
            if key in mt.index:  # GATE: the master table's own numbers, recomputed
                for c in COLS:
                    a, b = row[c], mt.loc[key, c]
                    if isinstance(a, str) or isinstance(b, str):
                        ok = str(a) == str(b)
                        v = 0.0 if ok else np.inf
                    else:
                        v = abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)
                        ok = v <= GATE_REL
                    gates.append(
                        {
                            "gate": f"recomputed {c} equals master_table.csv",
                            "forecast": key,
                            "value": v,
                            "bound": GATE_REL,
                            "ok": ok,
                            "source": "this script",
                        }
                    )
            rows.append(row)
            print(
                f"{key:26s} days {row['n_days']} QLIKE {row['qlike_raw']:.4f}  Sharpe mid "
                f"{row['Sharpe_mid']:.2f} crossed {row['Sharpe_crossed']:.2f}  buy {row['pct_buy']:.1f} %"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    g = pd.DataFrame(gates)
    g.to_csv(OUT / "score_gates.csv", index=False)
    cols = ["key", "label", "family", "bucket", "estimator", *COLS]
    pd.DataFrame(rows)[cols].to_csv(OUT / "pool_moments_vix_scores.csv", index=False)
    bad = g[~g["ok"].astype(bool)]
    print(f"gates: {len(g) - len(bad)} / {len(g)} pass")
    if len(bad):
        print(bad.to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
