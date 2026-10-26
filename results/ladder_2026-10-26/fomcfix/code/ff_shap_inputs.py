"""Inputs of the patched tree SHAP study (results/ladder_2026-10-26/RULE.txt ADDENDUM 3 (4)), on the laptop.

  gamma     the patched linear walks' gamma at the 126 origins, from the ladder's patched full-set replays (each
            origin's walk state: theta and the scales; gamma = theta x s for ridge, theta for lasso / elastic net, as
            fi_by_window's fiw_reduce.gam_of; intercept dropped) -> <shap dir>/inputs/fiw_gamma_fomcfix.npz
  walkrows  the patched every-bar tree walks at the 126 origin rows (LightGBM / XGBoost: fomcfix forecasts_gb.parquet;
            the forest: its merged walk or the CARC pieces' origin rows, when given) -> <shap dir>/inputs/
            walk_origin_rows.csv in the format of the pre-patch study's carc_fetch_origin_rows.py

usage: python ff_shap_inputs.py gamma --work <ff work w3000> --shap <shap dir>
       python ff_shap_inputs.py walkrows --shap <shap dir> --trees <parquet>[,<parquet>] [--rf-rows <csv>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ORIG = ROOT / "results" / "ladder_2026-10-26" / "code" / "fiw_origins.csv"
W = 39000


def gamma(a) -> None:
    work, shap = Path(a.work), Path(a.shap)
    names = list(json.loads((work / "design.json").read_text())["names"])
    o = pd.read_csv(ORIG)
    out: dict = {"columns": np.array(names), "origins": o["origin"].to_numpy().astype(str)}
    for m in ("ridge", "lasso", "enet"):
        g = []
        for k in o["k"]:
            z = np.load(work / "linear" / m / "full" / f"k{int(k):03d}.npz")
            assert int(z["r"]) == W + int(o.loc[o["k"] == k, "oos_row_w3000"].iloc[0])
            th, sc = z["theta"][:-1], z["sc"][:-1]
            g.append(th * sc if m == "ridge" else th.copy())
        out[f"walk_w3000_{m}_gamma"] = np.vstack(g)
        assert out[f"walk_w3000_{m}_gamma"].shape == (126, len(names))
    np.savez_compressed(shap / "inputs" / "fiw_gamma_fomcfix.npz", **out)
    print("gamma", {m: out[f"walk_w3000_{m}_gamma"].shape for m in ("ridge", "lasso", "enet")})


def walkrows(a) -> None:
    shap = Path(a.shap)
    o = pd.read_csv(ORIG)
    org = pd.DatetimeIndex(pd.to_datetime(o["origin"]))
    rows = []
    t = pd.concat([pd.read_parquet(f) for f in a.trees.split(",")], ignore_index=True)
    for est, f in t[t.predictor_set == "all_features"].groupby("model"):
        s = f.set_index("date")["pred_adj"].reindex(org)
        for k, (d, v) in enumerate(s.items()):
            if np.isfinite(v):
                rows.append(
                    dict(
                        est=est,
                        run_dir="/home1/jc_905/fomcfix_trees/runs/B/w3000",
                        piece="forecasts (merged)",
                        date=d,
                        pred_adj_hex=float(v).hex(),
                        pred_adj=float(v),
                        fit_row=int(o["oos_row_w3000"].iloc[k]),
                    )
                )
    if a.rf_rows:
        rf = pd.read_csv(a.rf_rows, parse_dates=["date"])
        rf = rf[rf["date"].isin(org)]
        for r in rf.itertuples():
            rows.append(
                dict(
                    est="rf_skx",
                    run_dir="/home1/jc_905/fomcfix_trees/runs/B/w3000",
                    piece=r.piece,
                    date=r.date,
                    pred_adj_hex=r.pred_adj_hex,
                    pred_adj=float.fromhex(r.pred_adj_hex),
                    fit_row=int(r.fit_row),
                )
            )
    out = pd.DataFrame(rows)
    out.to_csv(shap / "inputs" / "walk_origin_rows.csv", index=False)
    print(out.groupby("est").size())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gamma", "walkrows"))
    ap.add_argument("--work", default="")
    ap.add_argument("--shap", required=True)
    ap.add_argument("--trees", default="")
    ap.add_argument("--rf-rows", default="")
    a = ap.parse_args()
    gamma(a) if a.mode == "gamma" else walkrows(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
