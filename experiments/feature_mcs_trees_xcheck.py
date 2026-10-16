"""Feature MCS trees, the bucket designs on Hoffman2 (2026-10-13): CPU-class identity check.  Every chunk re-run on another
CPU class (results/feature_mcs_trees_work/xcheck_<arch>/<est>/<design>/c<chunk>, cluster/submit_feature_mcs_trees_h2.sh
check) is compared with the same chunk of the fleet on the E5-2670 / E5-2650 v2 nodes (.../chunks2/...): every array of
trees_bar1600.npz byte for byte (the timing arrays fit_sec / shap_sec left out; the meta equal but for its timings), and
the results CSV and the feature-health CSV byte for byte.  Prints one line per chunk and writes
results/feature_mcs_trees_work/xcheck.csv; exit 0 only when every compared chunk is identical.

Run (Hoffman2, the runtime's python):  python experiments/feature_mcs_trees_xcheck.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

W = Path("results/feature_mcs_trees_work")
TIMING = ("fit_sec", "shap_sec")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    rows, ok_all, n = [], True, 0
    for done in sorted(W.glob("xcheck_*/*/*/c*/DONE")):
        d = done.parent
        arch = d.parts[-4].removeprefix("xcheck_")
        est, design, chunk = d.parts[-3], d.parts[-2], d.name
        ref = W / "chunks2" / est / design / chunk
        r = dict(
            arch=arch,
            est=est,
            design=design,
            chunk=chunk,
            cpu=(d / "CPU").read_text().strip() if (d / "CPU").is_file() else "",
        )
        if not (ref / "DONE").is_file():
            rows.append(r | dict(identical="", why="no E5 chunk yet"))
            continue
        m = "xgb" if est == "xgb" else "rf"
        sub = Path("causal_tune_trees") / m / "all_features"
        za, zb = (
            np.load(d / sub / "trees_bar1600.npz"),
            np.load(ref / sub / "trees_bar1600.npz"),
        )
        diff = []
        for k in sorted(set(za.files) | set(zb.files)):
            if k in TIMING:
                continue
            if k not in za.files or k not in zb.files:
                diff.append(f"{k}(missing)")
            elif k == "meta":
                a, b = json.loads(str(za[k])), json.loads(str(zb[k]))
                for t in ("walk_sec", "run_sec"):
                    a.pop(t, None)
                    b.pop(t, None)
                if a != b:
                    diff.append("meta")
            elif (
                np.ascontiguousarray(za[k]).tobytes()
                != np.ascontiguousarray(zb[k]).tobytes()
                or za[k].dtype != zb[k].dtype
            ):
                diff.append(k)
        for f in ("results_bar1600.csv", "results_bar1600_feature_health.csv"):
            if md5(d / sub / f) != md5(ref / sub / f):
                diff.append(f)
        pa, pb = za["pred_adj"], zb["pred_adj"]
        fin = np.isfinite(pa) & np.isfinite(pb)
        r |= dict(
            identical=not diff,
            differing=" ".join(diff),
            rows=int(fin.sum()),
            max_abs_pred_diff=float(np.max(np.abs(pa[fin] - pb[fin])))
            if fin.any()
            else 0.0,
            ref_cpu=(ref / "CPU").read_text().strip()
            if (ref / "CPU").is_file()
            else "",
        )
        rows.append(r)
        ok_all &= not diff
        n += 1
    cols = [
        "arch",
        "est",
        "design",
        "chunk",
        "cpu",
        "ref_cpu",
        "rows",
        "identical",
        "differing",
        "max_abs_pred_diff",
        "why",
    ]
    with open(W / "xcheck.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    for r in rows:
        print(" | ".join(str(r.get(k, "")) for k in cols))
    print(f"{n} chunks compared, all identical: {ok_all and n > 0}")
    return 0 if ok_all and n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
