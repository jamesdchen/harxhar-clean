"""Gate T1 of results/fomcfix_2026-10-26/RULE.txt: tree pieces of the cell-A build against the templates, read-only.

Every complete piece (parquet + json) under <gate dir>/<tag>/pieces/ is compared, row by row on its dates, with the
template of its class (results/pooled_trees_final_2014start_w3000/forecasts.parquet for lgbm / xgb,
results/pooled_rf_skx_2014start_w3000/forecasts.parquet for rf_skx): pred_adj bit for bit (and true_adj, B), the
piece's CPU models recorded.  A class passes on a CPU model when every one of its pieces run there is equal.

usage: python tree_gate_cmp.py <code tree> <gate dir> <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE, GATE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TEMPL = {
    "lgbm": CODE / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "xgb": CODE / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "rf_skx": CODE / "results" / "pooled_rf_skx_2014start_w3000" / "forecasts.parquet",
}
rows, fail = [], []
cache: dict = {}
for pj in sorted(GATE.glob("*/pieces/*.json")):
    pq = pj.with_suffix(".parquet")
    if pj.name.endswith(".ckpt.json") or not pq.is_file():
        continue
    m = json.loads(pj.read_text())
    est, s = m["est"], m["predictor_set"]
    if est not in cache:
        cache[est] = pd.read_parquet(TEMPL[est])
    t = cache[est]
    t = t[(t["model"] == est) & (t["predictor_set"] == s)].set_index("date")
    p = pd.read_parquet(pq).set_index("date")
    q = t.reindex(p.index)
    ok_rows = bool(q["pred_adj"].notna().all())
    eq = {c: ok_rows and np.array_equal(p[c].to_numpy(), q[c].to_numpy()) for c in ("pred_adj", "true_adj", "B")}
    mx = float(np.max(np.abs(p["pred_adj"].to_numpy() - q["pred_adj"].to_numpy()))) if ok_rows else None
    r = dict(
        tag=pj.parent.parent.name,
        piece=pj.stem,
        est=est,
        set=s,
        rows=len(p),
        first=str(p.index[0]),
        last=str(p.index[-1]),
        cpus=m.get("cpus"),
        threads=m.get("threads"),
        host=m.get("host"),
        versions=m.get("versions"),
        **{f"{c}_equal": v for c, v in eq.items()},
        max_abs_pred_adj=mx,
        max_rel_pred_adj=float(np.max(np.abs(p["pred_adj"].to_numpy() / q["pred_adj"].to_numpy() - 1.0)))
        if ok_rows
        else None,
    )
    rows.append(r)
    if not all(eq.values()):
        fail.append(f"{r['tag']}/{r['piece']}")
by: dict[str, bool] = {}
for r in rows:
    for c in r["cpus"] or []:
        k = f"{r['est']}|{c}"
        by.setdefault(k, True)
        by[k] = by[k] and r["pred_adj_equal"] and r["true_adj_equal"] and r["B_equal"]
rec = dict(pieces=rows, pass_by_class_cpu=by, fail=fail, n_pieces=len(rows))
OUT.write_text(json.dumps(rec, indent=1, default=str))
print(json.dumps(dict(n_pieces=len(rows), pass_by_class_cpu=by, fail=fail), indent=1))
for r in rows:
    print(r["tag"], r["piece"], r["rows"], r["cpus"], r["pred_adj_equal"], r["max_abs_pred_adj"])
sys.exit(0 if not fail else 3)
