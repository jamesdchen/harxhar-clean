"""Final design against the earlier check's design (e34 45183a0), per arm, window and fold set.

usage: python compare_old.py <final_tables.csv> <earlier_tables.csv> <final_per_fold.csv> <earlier_per_fold.csv>
       <out_prefix>
-> <out_prefix>_tables.csv (est, set, criterion, folds, window: earlier, final, final - earlier, % change) and
<out_prefix>_per_fold.csv (one row a fit: earlier and final validation MSE / QLIKE), and a printout of the % change of
validation MSE (adjusted scale) by arm and window for each fold set.
"""

import sys

import pandas as pd

ft, et, fp, ep, outp = sys.argv[1:6]
K = ["est", "predictor_set", "criterion", "folds", "window_days"]
a, b = pd.read_csv(ft), pd.read_csv(et)
m = b[K + ["value"]].merge(a[K + ["value"]], on=K, how="outer", suffixes=("_earlier", "_final"), validate="1:1")
assert m.notna().all().all(), m[m.isna().any(axis=1)]
m["final_minus_earlier"] = m["value_final"] - m["value_earlier"]
m["pct_change"] = 100 * (m["value_final"] / m["value_earlier"] - 1)
m.to_csv(f"{outp}_tables.csv", index=False)
F = ["window_days", "fold", "est", "predictor_set"]
pa, pb = pd.read_csv(fp), pd.read_csv(ep)
q = pb[F + ["mse_adj", "qlike"]].merge(pa[F + ["mse_adj", "qlike"]], on=F, how="outer",
                                       suffixes=("_earlier", "_final"), validate="1:1")
assert q.notna().all().all() and len(q) == len(pa) == len(pb)
q["mse_pct_change"] = 100 * (q["mse_adj_final"] / q["mse_adj_earlier"] - 1)
q.to_csv(f"{outp}_per_fold.csv", index=False)
pd.set_option("display.width", 220)
for crit in ("mse_adj", "qlike"):
    for agg in ("folds1-3", "fold3", "folds0-3"):
        t = m[(m["criterion"] == crit) & (m["folds"] == agg)]
        print(f"\n== {crit}, {agg}: % change, final vs earlier design")
        print(t.pivot_table(index=["est", "predictor_set"], columns="window_days", values="pct_change", sort=False)
              .to_string(float_format=lambda x: f"{x:+.3f}"))
print("\nper fit: baseline (HAR) bit-equal MSE:",
      int((q.loc[q["predictor_set"] == "baseline", "mse_adj_final"]
           == q.loc[q["predictor_set"] == "baseline", "mse_adj_earlier"]).sum()),
      "of", int((q["predictor_set"] == "baseline").sum()))
