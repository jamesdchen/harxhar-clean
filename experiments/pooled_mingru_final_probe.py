"""Summary of the bar-refit timing probe (cluster/pooled_mingru_final_probe_carc.sbatch, 2026-10-22): for every probe
run (<root>/runs/probe_<set>_c<conc>_<job>/): refits, the trainer's wall time (the whole run, the pool phase of
refit.log), GPU seconds per refit (wall / refits: one P100 shared by the concurrent models), the mean per-job t_train
and epochs; and whether the forecasts of a (root, set) are identical (text of f, row by row) at every concurrency.

    python pooled_mingru_final_probe.py <run dir> [<run dir> ...]   (prints a table; writes <root>/runs/probe_summary_<job>.json)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

POOL_RE = re.compile(r"pool phase \d+: (\d+) jobs on (\d+) slots in ([0-9.]+) s")


def main() -> int:
    recs, preds = [], {}
    for a in sys.argv[1:]:
        d = Path(a)
        m = re.match(r"^probe_(baseline|all_features)_c(\d+)_(\d+)$", d.name)
        assert m is not None, d
        s, conc, job = m.group(1), int(m.group(2)), m.group(3)
        env = dict(
            ln.split(" ", 1)
            for ln in (d / "env.txt").read_text().splitlines()
            if ln.startswith(("conc", "wall_ms"))
        )
        jo = pd.read_csv(d / "refit" / "jobs_out.csv")
        pr = pd.read_csv(d / "refit" / "preds.csv", dtype={"f": str})
        pool = [POOL_RE.search(ln) for ln in (d / "refit.log").read_text().splitlines()]
        pool_s = sum(float(p.group(3)) for p in pool if p)
        n = len(jo)
        recs.append(
            dict(
                root=d.parents[1].name,
                set=s,
                conc=conc,
                refits=n,
                status_ok=int((jo["status"] == "ok").sum()),
                wall_s=float(env["wall_ms"]) / 1000,
                pool_s=pool_s,
                gpu_s_per_refit=float(env["wall_ms"]) / 1000 / n,
                pool_s_per_refit=pool_s / n,
                t_train_mean=float(jo["t_train"].mean()),
                t_setup_mean=float(jo["t_setup"].mean()),
                t_capture_mean=float(jo["t_capture"].mean()),
                epochs_mean=float(jo["epochs_run"].mean()),
                n_inputs=int(jo["n_inputs"].iloc[0]),
                job=job,
            )
        )
        preds[(d.parents[1].name, s, conc)] = pr.sort_values(["id", "row"]).reset_index(
            drop=True
        )
    t = pd.DataFrame(recs).sort_values(["root", "set", "conc"]).reset_index(drop=True)
    same = []
    for (root, s), g in t.groupby(["root", "set"]):
        cs = sorted(g["conc"])
        base = preds[(root, s, cs[0])]
        for c in cs[1:]:
            o = preds[(root, s, c)]
            eq = base[["id", "row"]].equals(o[["id", "row"]]) and bool(
                (base["f"].to_numpy() == o["f"].to_numpy()).all()
            )
            same.append(dict(root=root, set=s, conc_a=cs[0], conc_b=c, identical=eq))
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(same).to_string(index=False))
    out = dict(runs=t.to_dict(orient="records"), identical=same)
    for root in {Path(a).parents[1] for a in sys.argv[1:]}:
        (root / "runs" / f"probe_summary_{t['job'].iloc[0]}.json").write_text(
            json.dumps(out, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
