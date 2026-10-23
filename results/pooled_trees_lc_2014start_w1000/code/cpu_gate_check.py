"""Every CPU model that computed a piece of a run must have passed the Hoffman2 gate at the run's thread count.

Hoffman2's `arch` request does not pin the CPU model (pieces of arrays asking for intel-gold-6240|intel-E5-2650v4 ran
on Xeon 6736P, Gold 6140 and Gold 6342 nodes too), so the gate is judged by the CPU recorded in each gate piece's meta
(gate_cmp.py's `cpu` column), not by the label the gate array requested.

usage: python cpu_gate_check.py <gate_check.csv> <out csv> <run dir> [<run dir> ...]
For each (run, est, set): the pieces' CPU models and threads; for each CPU model, the gate pieces run on it at those
threads, how many were bit-equal (DataFrame.equals, schema), and the arms covered.  Exit 1 if any model lacks a pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

G = pd.read_csv(sys.argv[1])
G = G[G["status"] == "done"]
out = Path(sys.argv[2])
rows = []
for rd in sys.argv[3:]:
    for p in sorted((Path(rd) / "pieces").glob("*.json")):
        if p.name.endswith(".ckpt.json"):
            continue
        m = json.loads(p.read_text())
        rows.append(
            {
                "run": Path(rd).name,
                "est": m["est"],
                "set": m["predictor_set"],
                "cpu": m["cpu"],
                "threads": m["threads"],
                "piece": p.stem,
            }
        )
P = pd.DataFrame(rows)
recs, ok = [], True
for (cpu, th), g in P.groupby(["cpu", "threads"]):
    gg = G[(G["cpu"] == cpu) & (G["threads"] == th)]
    arms = sorted({"_".join(x.split("_bar_")[0].split("_", 1)) for x in gg["piece"]})
    passed = int(
        gg["equal"].astype(str).str.lower().eq("true").sum()
    )  # CSV bools: "True" / "False"
    good = len(gg) > 0 and passed == len(gg)
    ok &= good
    recs.append(
        {
            "cpu": cpu,
            "threads": th,
            "pieces": len(g),
            "pieces_by_arm": json.dumps(
                g.groupby(["run", "est", "set"])
                .size()
                .rename("n")
                .reset_index()
                .to_dict("records")
            ),
            "gate_pieces": len(gg),
            "gate_bit_equal": passed,
            "gate_arms": ";".join(arms),
            "gated": good,
        }
    )
R = pd.DataFrame(recs)
R.to_csv(out, index=False)
pd.set_option("display.width", 250)
print(R.drop(columns=["pieces_by_arm"]).to_string(index=False))
print("ALL CPU MODELS GATED" if ok else "SOME CPU MODEL NOT GATED")
sys.exit(0 if ok else 1)
