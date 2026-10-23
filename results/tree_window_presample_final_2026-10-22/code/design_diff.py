"""Final-design cache against the earlier check's cache of the same window, per predictor set.

usage: python design_diff.py <new_dir> <old_dir> <out_prefix>
Each dir holds design_baseline/ and design_all_features/.  Prints: names / config / shape equal; y, b_oos, dates
equal; the exactly-equal share of X over all rows and over the pre-2012 rows [0, S) (S = the design row of the first
2012-01-03 bar; every row the check fits or validates on lies before it); per differing column (to
<out_prefix>_<set>.csv): values differing in all rows and in the pre-2012 rows, how many of those are 0 in the new
cache, max |diff|, the first differing row.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

new, old, outp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
for s in ("baseline", "all_features"):
    A, B = new / f"design_{s}", old / f"design_{s}"
    ma, mb = (json.loads((p / "design.json").read_text()) for p in (A, B))
    W = ma["config"]["window"]
    d = pd.read_csv(A / "dates_oos.csv", parse_dates=["date"])["date"]
    S = W + int(np.flatnonzero(d >= pd.Timestamp("2012-01-03"))[0])
    print(f"== {s} (window {W} rows, S {S})")
    print("  names equal", ma["names"] == mb["names"], "| config equal", ma["config"] == mb["config"],
          "| shape", ma["shape"], mb["shape"])
    for f in ("y.npy", "b_oos.npy"):
        print(f"  {f} equal", np.array_equal(np.load(A / f), np.load(B / f)))
    print("  dates_oos equal", d.equals(pd.read_csv(B / "dates_oos.csv", parse_dates=["date"])["date"]))
    XA, XB = np.load(A / "X.npy", mmap_mode="r"), np.load(B / "X.npy", mmap_mode="r")
    assert XA.shape == XB.shape
    rows, n_eq, n_eq_p = [], 0, 0
    for j, name in enumerate(ma["names"]):
        x, z = np.asarray(XA[:, j]), np.asarray(XB[:, j])
        eq = (x == z) | (np.isnan(x) & np.isnan(z))
        n_eq += int(eq.sum())
        n_eq_p += int(eq[:S].sum())
        if not eq.all():
            ne = ~eq
            dd = np.abs(x - z)
            rows.append(dict(column=name, n_diff=int(ne.sum()), n_diff_pre2012=int(ne[:S].sum()),
                             n_diff_new_zero=int((ne & (x == 0)).sum()),
                             n_diff_pre2012_new_zero=int((ne[:S] & (x[:S] == 0)).sum()),
                             max_abs=float(np.nanmax(dd[ne])),
                             max_abs_pre2012=float(np.nanmax(dd[:S][ne[:S]])) if ne[:S].any() else 0.0,
                             first_diff_row=int(np.flatnonzero(ne)[0])))
    tab = pd.DataFrame(rows)
    print(f"  X: exactly equal {100 * n_eq / XA.size:.4f} % of all entries, "
          f"{100 * n_eq_p / (S * XA.shape[1]):.4f} % of the pre-2012 rows; {len(tab)} of {XA.shape[1]} columns differ")
    if len(tab):
        tab = tab.sort_values("n_diff", ascending=False)
        tab.to_csv(f"{outp}_{s}.csv", index=False)
        print(f"  values differing: {tab['n_diff'].sum()} (new value 0: {tab['n_diff_new_zero'].sum()}); "
              f"pre-2012 rows: {tab['n_diff_pre2012'].sum()} in {(tab['n_diff_pre2012'] > 0).sum()} columns "
              f"(new value 0: {tab['n_diff_pre2012_new_zero'].sum()}); max |diff| {tab['max_abs'].max():.4g}, "
              f"pre-2012 {tab['max_abs_pre2012'].max():.4g}")
        print(tab.head(10).to_string(index=False))
