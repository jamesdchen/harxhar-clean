"""Gate: the earlier check's fits refitted on the earlier check's caches with this run's code (9d3fc05) and driver.

usage: python gate_check.py <gate_out_dir> <earlier_out_dir> <out_csv>
Per gate fit: the earlier record of the same tag; equal fit / val rows, config sha, X md5; validation MSE and QLIKE
(the difference and whether bit-equal); the validation predictions (pred_adj) bit-equal and their max |diff|.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

g, e, outp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = []
for p in sorted(g.glob("w*_f*_*.json")):
    a, b = json.loads(p.read_text()), json.loads((e / p.name).read_text())
    pa = pd.read_parquet(p.with_suffix(".parquet"))["pred_adj"].to_numpy()
    pb = pd.read_parquet((e / p.name).with_suffix(".parquet"))["pred_adj"].to_numpy()
    rows.append(dict(tag=p.stem, rows_equal=a["fit"] == b["fit"] and a["val"] == b["val"],
                     config_sha_equal=a["config_sha"] == b["config_sha"], x_md5_equal=a["x_md5"] == b["x_md5"],
                     mse_adj_earlier=b["mse_adj"], mse_adj_gate=a["mse_adj"], mse_bit_equal=a["mse_adj"] == b["mse_adj"],
                     mse_abs_diff=abs(a["mse_adj"] - b["mse_adj"]), qlike_earlier=b["qlike"], qlike_gate=a["qlike"],
                     qlike_bit_equal=a["qlike"] == b["qlike"], pred_bit_equal=bool(np.array_equal(pa, pb)),
                     pred_max_abs_diff=float(np.max(np.abs(pa - pb))), host_earlier=b["host"], host_gate=a["host"],
                     versions_equal=a["versions"] == b["versions"], code_commit=a.get("code_commit", "")))
t = pd.DataFrame(rows)
t.to_csv(outp, index=False)
pd.set_option("display.width", 250)
print(t.drop(columns=["host_earlier", "host_gate", "code_commit"]).to_string(index=False))
print("all bit-equal:", bool(t["mse_bit_equal"].all() and t["pred_bit_equal"].all()))
