"""Gate compare: each finished gate piece against the printed 2012-start w1000 piece of the same name, bit for bit.

usage: python gate_cmp.py <printed pieces dir> <out csv> <label>=<gate dir> [<label>=<gate dir> ...]

Equal = pandas DataFrame.equals on every column (model, predictor_set, date, pred_adj, true_adj, B, plain, rv_actual,
fit_row) and the same Arrow schema; the rows the gate refitted (gate_prep.csv refit_rows) are compared on their own as
well.  The meta of the gate piece gives the refits it ran, host, CPU model, threads and library versions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

src, out = Path(sys.argv[1]), Path(sys.argv[2])
recs = []
for arg in sys.argv[3:]:
    label, gdir = arg.split("=", 1)
    gate = Path(gdir)
    prep = pd.read_csv(gate / "gate_prep.csv")
    for _, p in prep.iterrows():
        name = p["piece"]
        a_p, b_p = src / f"{name}.parquet", gate / "pieces" / f"{name}.parquet"
        meta_p = b_p.with_suffix(".json")
        r = {
            "gate": label,
            "piece": name,
            "k": int(p["k"]),
            "rows": int(p["rows"]),
            "refit_rows": int(p["refit_rows"]),
        }
        if not meta_p.is_file():
            r["status"] = "not finished"
            recs.append(r)
            continue
        a, b = pd.read_parquet(a_p), pd.read_parquet(b_p)
        m = json.loads(meta_p.read_text())
        pm = json.loads(a_p.with_suffix(".json").read_text())
        if int(p["k"]):
            day = pd.DatetimeIndex(a["date"]).normalize()
            u = day.unique()
            sel = (day.isin(u[: int(p["k"])]) | day.isin(u[-int(p["k"]) :])).to_numpy()
        else:
            sel = np.ones(len(a), bool)
        same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
        diff = {}
        if same_shape:
            for c in a.columns:
                if not a[c].equals(b[c]):
                    diff[c] = int((a[c].to_numpy() != b[c].to_numpy()).sum())
        dp = (
            np.abs(a["pred_adj"].to_numpy(float) - b["pred_adj"].to_numpy(float))
            if same_shape
            else np.array([np.nan])
        )
        r |= {
            "status": "done",
            "equal": bool(same_shape and a.equals(b)),
            "refitted_rows_equal": bool(
                same_shape
                and a[sel].reset_index(drop=True).equals(b[sel].reset_index(drop=True))
            ),
            "schema_equal": pq.read_schema(a_p)
            .remove_metadata()
            .equals(pq.read_schema(b_p).remove_metadata()),
            "cols_unequal": ";".join(f"{c}:{n}" for c, n in diff.items()),
            "max_abs_diff_pred": float(np.nanmax(dp)),
            "refits": int(m["refits"]),
            "rows_meta": int(m["rows"]),
            "wall_sec": m["wall_sec"],
            "threads": m["threads"],
            "host": m["host"],
            "cpu": m["cpu"],
            "printed_cpu": pm["cpu"],
            "printed_threads": pm["threads"],
            "config_sha": m["sha"],
            "config_sha_printed": pm["sha"],
            "versions": json.dumps(m["versions"], sort_keys=True),
            "versions_equal": m["versions"] == pm["versions"],
            "slurm_job": m["slurm_job"],
            "slurm_task": m["slurm_task"],
        }
        assert r["refits"] == r["refit_rows"], (name, r["refits"], r["refit_rows"])
        recs.append(r)
R = pd.DataFrame(recs)
R.to_csv(out, index=False)
pd.set_option("display.width", 250)
print(
    R[
        [
            c
            for c in (
                "gate",
                "piece",
                "k",
                "refits",
                "status",
                "equal",
                "refitted_rows_equal",
                "schema_equal",
                "max_abs_diff_pred",
                "threads",
                "cpu",
                "printed_cpu",
                "versions_equal",
            )
            if c in R
        ]
    ].to_string(index=False)
)
done = R[R["status"] == "done"] if "status" in R else R
print(
    f"{int(done['equal'].sum()) if len(done) else 0} of {len(R)} gate pieces bit-equal to the printed ones "
    f"({len(done)} finished)"
)
