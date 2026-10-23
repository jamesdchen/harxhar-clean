"""Gate of the rf_skx walks (RULE_rf_skx.txt).

  fold  --W W --set S: the runner's own construction (pooled_trees_rerun_13.model_factory("rf_skx", T) + fit_predict)
        fitted on pre-test fold 3 (fit [S - 3,575 - W, S - 3,575), validation the 3,250 bars of 2013), compared bit for
        bit with the benchmark's engine-B forecasts (results/rf_engine_bench_2026-10-23/out/B_<set>_w<W>_epyc.parquet,
        column pred_T, EPYC 7513); -> <out>/fold_<set>_w<W>_<cpu tag>.json + .parquet
  cmp   <dir a> <dir b> [...]: the walk pieces of the same name in each directory, DataFrame.equals; and the fold
        files of the same (set, W) across CPU tags
usage: python gate_skx.py fold --repo R --design D --W W --set S --threads T --bench B --out O
       python gate_skx.py cmp --out O <label>=<pieces dir> ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PPD, FOLD_N, EMB, S_WANT = 13, 3250, 325, 50777


def fold(a) -> int:
    repo = Path(a.repo)
    sys.path.insert(0, str(repo / "experiments"))
    import pooled_trees_rerun_13 as PT

    Wr = a.W * PPD
    root = Path(a.design)
    PT.check(root, a.set, a.W)
    dz = PT.design_dir(root, a.set)
    X = np.load(dz / "X.npy", mmap_mode="r")
    y = np.load(dz / "y.npy")
    d = PT.oos_dates(root, a.set)
    S = Wr + int(np.flatnonzero(d >= pd.Timestamp("2014-01-02"))[0])
    assert S == S_WANT, S
    f0, f1, v0, v1 = S - FOLD_N - EMB - Wr, S - FOLD_N - EMB, S - FOLD_N, S
    Xf = np.ascontiguousarray(X[f0:f1], dtype=np.float64)
    Xv = np.ascontiguousarray(X[v0:v1], dtype=np.float64)
    cfg, make = PT.model_factory("rf_skx", a.threads)
    pr = PT.fit_predict(make, "rf_skx", Xf, y[f0:f1], Xv)
    ref = pd.read_parquet(Path(a.bench) / f"B_{a.set}_w{a.W}_epyc.parquet")
    bref = ref["pred_T"].to_numpy()
    cpu = PT.cpu_model()
    tag = "7542" if "7542" in cpu else ("7513" if "7513" in cpu else "other")
    rec = {
        "set": a.set,
        "W": a.W,
        "cpu": cpu,
        "threads": a.threads,
        "config": cfg["name"],
        "sha": cfg["sha"],
        "versions": PT.versions(),
        "pred_md5": hashlib.md5(pr.tobytes()).hexdigest(),
        "bench_md5": hashlib.md5(bref.tobytes()).hexdigest(),
        "equal_to_bench": bool(np.array_equal(pr, bref)),
        "max_abs_diff_bench": float(np.max(np.abs(pr - bref))),
        "dispatch": sorted(set(PT._SkxLog.msgs)),
        "mse_adj": float(np.mean((pr - y[v0:v1]) ** 2)),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pred": pr}).to_parquet(
        out / f"fold_{a.set}_w{a.W}_{tag}.parquet", index=False
    )
    (out / f"fold_{a.set}_w{a.W}_{tag}.json").write_text(json.dumps(rec, indent=1))
    print(
        json.dumps(
            {
                k: rec[k]
                for k in (
                    "set",
                    "W",
                    "cpu",
                    "equal_to_bench",
                    "max_abs_diff_bench",
                    "mse_adj",
                )
            }
        )
    )
    return 0


def cmp(a) -> int:
    out = Path(a.out)
    rows = []
    dirs = dict(x.split("=", 1) for x in a.dirs)
    labels = list(dirs)
    base = Path(dirs[labels[0]])
    for p in sorted(base.glob("*.parquet")):
        if not p.with_suffix(".json").is_file():
            continue
        fa = pd.read_parquet(p)
        ma = json.loads(p.with_suffix(".json").read_text())
        for lab in labels[1:]:
            q = Path(dirs[lab]) / p.name
            r = {"piece": p.stem, "a": labels[0], "b": lab, "cpu_a": ma["cpu"]}
            if q.is_file() and q.with_suffix(".json").is_file():
                fb = pd.read_parquet(q)
                mb = json.loads(q.with_suffix(".json").read_text())
                r |= {
                    "cpu_b": mb["cpu"],
                    "rows": len(fa),
                    "equal": bool(fa.equals(fb)),
                    "max_abs_diff_pred": float(
                        np.max(
                            np.abs(
                                fa["pred_adj"].to_numpy() - fb["pred_adj"].to_numpy()
                            )
                        )
                    )
                    if len(fa) == len(fb)
                    else float("nan"),
                    "dispatch_a": ma.get("skx_dispatch"),
                    "split_a": ma.get("skx_split"),
                }
            else:
                r["equal"] = None
            rows.append(r)
    fl = sorted(out.glob("fold_*.json"))
    for p in fl:
        r = json.loads(p.read_text())
        rows.append(
            {
                "piece": p.stem,
                "a": "fold",
                "b": "bench B (EPYC 7513)",
                "cpu_a": r["cpu"],
                "equal": r["equal_to_bench"],
                "max_abs_diff_pred": r["max_abs_diff_bench"],
            }
        )
    R = pd.DataFrame(rows)
    R.to_csv(out / "gate_skx.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in ("dispatch_a",) if c in R]).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fold", "cmp"))
    ap.add_argument("--repo")
    ap.add_argument("--design")
    ap.add_argument("--W", type=int)
    ap.add_argument("--set")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args()
    return fold(a) if a.mode == "fold" else cmp(a)


if __name__ == "__main__":
    sys.exit(main())
