"""Run ON CARC (login node, read only): the walk's pred_adj at the 126 origin rows of the W = 3,000 full-set tree walks
(LightGBM, XGBoost: /scratch1/jc_905/trees_final/runs/w3000/pieces; rf_skx: /home1/jc_905/trees_final/runs_skx/w3000/
pieces).  Prints CSV to stdout:
  PIECE,<run dir>,<piece>,<rows>,<has_meta>,<window_days>,<cpu>,<threads>,<versions json with ; for ,>
  ROW,<est>,<run dir>,<piece>,<date>,<pred_adj float.hex>,<pred_adj repr>,<fit_row>
Usage: python carc_fetch_origin_rows.py <fiw_origins.csv>
"""

import glob
import json
import os
import sys

import pandas as pd

ORIGINS = set(pd.to_datetime(pd.read_csv(sys.argv[1])["origin"]))
assert len(ORIGINS) == 126, len(ORIGINS)
PATS = (
    "/scratch1/jc_905/trees_final/runs/w3000/pieces/*_all_features_bar_2014-01-02_*.parquet",
    "/home1/jc_905/trees_final/runs_skx/w3000/pieces/*_all_features_bar_2014-01-02_*.parquet",
)
n = 0
for pat in PATS:
    for p in sorted(glob.glob(pat)):
        name = os.path.basename(p)
        est = name.split("_all_features_")[0]
        if est not in ("lgbm", "xgb", "rf_skx"):
            continue
        d = os.path.dirname(os.path.dirname(p))
        mp = p[: -len(".parquet")] + ".json"
        meta = json.load(open(mp)) if os.path.isfile(mp) else {}
        f = pd.read_parquet(p, columns=["date", "pred_adj", "fit_row"])
        n += 1
        print(
            f"PIECE,{d},{name},{len(f)},{int(bool(meta))},{meta.get('window_days', '')},{meta.get('cpu', '')},"
            f"{meta.get('threads', '')},{json.dumps(meta.get('versions', {}), sort_keys=True).replace(',', ';')}"
        )
        f["date"] = pd.to_datetime(f["date"])
        for r in f[f["date"].isin(ORIGINS)].itertuples():
            print(
                f"ROW,{est},{d},{name},{r.date},{float(r.pred_adj).hex()},{float(r.pred_adj)!r},{int(r.fit_row)}"
            )
print(f"END,{n} pieces")
