"""Late-start zeroing (LATE_START_WARMUP, from the rule; latestart_rule.py output) by series, window and span:
for each span, how many of the series' value columns are zero on every row of the span ("never live there"), zero on
some rows ("partly"), or live on every row.  Spans: the 866 option days' 16:00 rows, the fold-3 validation year of the
2016 start (rows 54,092 .. 57,341) and of the 2014 start (rows 47,527 .. 50,776), and the whole design.

usage: python latestart_spans.py <latestart_rule_all.csv> <rows16 csv (option-day 16:00 design rows)> <out csv>
"""

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = pd.read_csv(sys.argv[1])
rows16 = pd.read_csv(sys.argv[2])["row"].to_numpy()
SPANS = {"option days 16:00": (rows16.min(), rows16.max() + 1), "fold-3 val 2015 (2016 start)": (54092, 57342),
         "fold-3 val 2013 (2014 start)": (47527, 50777), "whole design": (0, 84747)}
out = []
for (w, s), g in R.groupby(["window", "series"], sort=False):
    hi = g["zero_prefix_rows"].to_numpy()
    rec = dict(window=w, series=s, columns=len(g))
    for name, (a, b) in SPANS.items():
        if name == "option days 16:00":
            never = int(sum((rows16 < h).all() for h in hi))
            part = int(sum((rows16 < h).any() and not (rows16 < h).all() for h in hi))
        else:
            never = int((hi >= b).sum())
            part = int(((hi > a) & (hi < b)).sum())
        rec[f"{name}: never"] = never
        rec[f"{name}: partly"] = part
    out.append(rec)
T = pd.DataFrame(out)
T.to_csv(sys.argv[3], index=False)
for name in SPANS:
    print(f"\n{name}: columns never live / partly zeroed (of the series' value columns)")
    p = T.pivot_table(index="series", columns="window", values=f"{name}: never", sort=False)
    q = T.pivot_table(index="series", columns="window", values=f"{name}: partly", sort=False)
    n = T.groupby("series", sort=False)["columns"].first()
    cells = p.astype(int).astype(str) + "/" + q.astype(int).astype(str)
    cells.insert(0, "cols", n)
    print(cells.to_string())
