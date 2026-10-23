"""GATE of the final pooled MinGRU rerun (2026-10-22): the previous w1000 run's frozen configurations, on the previous
design (/home1/jc_905/pm13/data), rerun on today's environment (same binary, modules, P100) must reproduce the
stored numbers of results/pooled_mingru_rerun13_2026-10-22.

  tune  (run dir with tune/): the frozen configuration of each set on fold 3, 3 seeds -> every validation forecast
        equals the stored run's (the text of f, row by row), best epochs equal, and the seed-mean validation QLIKE
        equals pretune_qlike_folds.csv's fold-3 value (1e-15 relative)
  refit (run dir with refit/): the first walk origin (row 44,250), 3 seeds a set -> every forecast equals the stored
        walk run's (text of f, row by row)

    python pooled_mingru_final_gate.py <run dir>       (writes <run dir>/gate_check.json; exit 1 on a mismatch)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PM13 = Path("/home1/jc_905/pm13")
STORED_TUNE = {
    "baseline": PM13 / "runs/tune_baseline_f3_s0_13303258/tune",
    "all_features": PM13 / "runs/tune_all_features_f3_s0_13303117/tune",
}
STORED_WALK = {
    "baseline": PM13 / "runs/walk_baseline_p0_13303665/refit",
    "all_features": PM13 / "runs/walk_all_features_p0_13303659/refit",
}


def read_ids(path: Path, ids: set[str]) -> pd.DataFrame:
    out = []
    for c in pd.read_csv(path, dtype={"f": str}, chunksize=2_000_000):
        out.append(c[c["id"].isin(ids)])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    run = Path(sys.argv[1])
    res: dict = {"run": str(run)}
    ok = True
    if (run / "tune").is_dir():
        sub, stored = "tune", STORED_TUNE
    else:
        sub, stored = "refit", STORED_WALK
    jo = pd.read_csv(run / sub / "jobs_out.csv")
    pr = pd.read_csv(run / sub / "preds.csv", dtype={"f": str})
    assert (jo["status"] == "ok").all(), jo[["id", "status"]]
    folds = pd.read_csv(PM13 / "out/pretune_qlike_folds.csv")
    for s in ("baseline", "all_features"):
        pre = "ptune" if sub == "tune" else "pwalk"
        ids = {i for i in jo["id"] if i.startswith(f"{pre}_{s}_")}
        assert len(ids) == 3, (s, ids)
        mine = pr[pr["id"].isin(ids)].sort_values(["id", "row"]).reset_index(drop=True)
        ref = (
            read_ids(stored[s] / "preds.csv", ids)
            .sort_values(["id", "row"])
            .reset_index(drop=True)
        )
        same_rows = mine[["id", "kind", "row"]].equals(ref[["id", "kind", "row"]])
        same_f = same_rows and bool((mine["f"].to_numpy() == ref["f"].to_numpy()).all())
        maxrel = (
            float(
                np.max(
                    np.abs(
                        mine["f"].astype(float).to_numpy()
                        / ref["f"].astype(float).to_numpy()
                        - 1
                    )
                )
            )
            if same_rows
            else float("nan")
        )
        r: dict[str, Any] = dict(
            n_ids=len(ids),
            n_rows=len(mine),
            rows_equal=bool(same_rows),
            f_text_equal=bool(same_f),
            f_max_rel_diff=maxrel,
        )
        jr = read_ids(stored[s] / "jobs_out.csv", ids).set_index("id")
        jm = jo.set_index("id").loc[sorted(ids)]
        r["best_epoch_equal"] = bool(
            (
                jm["best_epoch"].to_numpy()
                == jr.loc[sorted(ids), "best_epoch"].to_numpy()
            ).all()
        )
        r["best_epoch"] = {i: int(jm.loc[i, "best_epoch"]) for i in sorted(ids)}
        ok = ok and bool(same_f and r["best_epoch_equal"])
        if sub == "tune":
            y = np.fromfile(PM13 / f"data/pm13_{s}/y.f32", "<f4").astype(np.float64)
            v = mine[mine["kind"] == "val"].copy()
            v["fv"] = v["f"].astype(float)
            c = v.groupby("row")["fv"].mean()
            yv = y[c.index.to_numpy()]
            k = yv > 0
            q = (yv[k] / c.to_numpy()[k]) ** 2
            ql = float(np.mean(q - np.log(q) - 1.0))
            cfg = sorted(ids)[0].split(f"ptune_{s}_f3_")[1].rsplit("_s", 1)[0]
            want = float(
                folds[
                    (folds["predictor_set"] == s)
                    & (folds["cfg"] == cfg)
                    & (folds["fold"] == 3)
                ]["val_qlike"].iloc[0]
            )
            r |= dict(
                cfg=cfg,
                val_qlike=ql,
                stored_val_qlike=want,
                val_qlike_rel_diff=abs(ql / want - 1),
            )
            ok = ok and abs(ql / want - 1) <= 1e-15
        res[s] = r
        print(s, json.dumps(r), flush=True)
    res["pass"] = bool(ok)
    (run / "gate_check.json").write_text(json.dumps(res, indent=1))
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
