"""Gate: CARC-built 1,000-day fixed design against the local build (ls_wt w1000 nofomcrel _work).

usage: python compare_w1000.py <carc_dir> <local_dir> <frozen_penalties.json> <out_csv_prefix>
Each dir holds design_baseline/ and design_all_features/.  Prints, per set: columns against frozen_penalties.json,
shapes, y / b_oos / dates equality, exactly-equal share of X, the max |diff| over all rows and over the pre-tune rows
[17956, 44281), and the differing columns (with max |diff| and the column's standard deviation).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

carc, local, fpp, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
fp = json.loads(fpp.read_text())
LO, HI = 17956, 44281
for s in ("baseline", "all_features"):
    A, B = carc / f"design_{s}", local / f"design_{s}"
    ma, mb = (json.loads((p / "design.json").read_text()) for p in (A, B))
    print(f"== {s}")
    print("  columns == frozen_penalties (CARC, local):", ma["names"] == fp["designs"][s]["columns"],
          mb["names"] == fp["designs"][s]["columns"], "n", len(ma["names"]))
    print("  config equal:", ma.get("config") == mb.get("config"), "| shape", ma["shape"], mb["shape"],
          "| f_order", ma["f_order"], mb["f_order"], "| numpy", ma["numpy"], mb["numpy"])
    for f in ("y.npy", "b_oos.npy"):
        a, b = np.load(A / f), np.load(B / f)
        print(f"  {f}: equal {np.array_equal(a, b)}, max |diff| {np.max(np.abs(a - b)):.3e}")
    da, db = (pd.read_csv(p / "dates_oos.csv")["date"] for p in (A, B))
    print("  dates_oos equal:", da.equals(db))
    XA, XB = np.load(A / "X.npy", mmap_mode="r"), np.load(B / "X.npy", mmap_mode="r")
    assert XA.shape == XB.shape
    rows, n_eq, n_eq_f = [], 0, 0
    for j, name in enumerate(ma["names"]):
        x, z = np.asarray(XA[:, j]), np.asarray(XB[:, j])
        eq = (x == z) | (np.isnan(x) & np.isnan(z))
        n_eq += int(eq.sum())
        n_eq_f += int(eq[LO:HI].sum())
        if not eq.all():
            dd = np.abs(x - z)
            rows.append(dict(column=name, n_diff=int((~eq).sum()), max_abs=float(np.nanmax(dd)),
                             max_rel=float(np.nanmax(dd / np.maximum(np.abs(z), 1e-300))),
                             n_diff_pretune=int((~eq[LO:HI]).sum()),
                             max_abs_pretune=float(np.nanmax(dd[LO:HI])), sd=float(np.nanstd(z)),
                             nan_mismatch=int((np.isnan(x) != np.isnan(z)).sum())))
    tab = pd.DataFrame(rows)
    nf = (HI - LO) * XA.shape[1]
    print(f"  X: exactly equal {n_eq} of {XA.size} ({100 * n_eq / XA.size:.4f} %); pre-tune rows "
          f"{100 * n_eq_f / nf:.4f} %; {len(tab)} of {XA.shape[1]} columns differ")
    if len(tab):
        tab = tab.sort_values("max_abs", ascending=False)
        tab.to_csv(f"{outp}_{s}.csv", index=False)
        print(f"  max |diff| all rows {tab['max_abs'].max():.3e}; pre-tune rows {tab['max_abs_pretune'].max():.3e}")
        print("  second-largest column max |diff|:", f"{tab['max_abs'].iloc[1]:.3e}" if len(tab) > 1 else "-")
        print(tab.head(8).to_string(index=False))
