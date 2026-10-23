"""Compact fold-3 learning-curve tables for a message (read-only): per arm, per W MSE / QLIKE with the step change
(±SE); then the cumulative pairs (part1_pairs output) and the full - HAR gap / extra gain.
usage: python compact_part1.py <curve csv> <gap csv> <pairs csv> <base W>"""

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R, G, P = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]), pd.read_csv(sys.argv[3])
B = int(sys.argv[4])
L = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "RF", "ridge": "ridge", "lasso": "lasso", "enet": "enet"}
SL = {"baseline": "HAR", "all_features": "full"}
r = R[R.foldset == "fold3"]
Ws = sorted(r.W.unique())
print("| arm | " + " | ".join(f"{w:,}" for w in Ws) + f" | {B:,}→{Ws[-1]:,} |")
print("|---|" + "---|" * (len(Ws) + 1))
for (fam, e, s), g in r.groupby(["family", "est", "set"], sort=False):
    for c, lab in (("mse", "MSE"), ("qlike", "QLIKE")):
        cells = []
        for _, x in g.iterrows():
            v = f"{x[c]:.5f}" if c == "mse" else f"{x[c]:.4f}"
            if not np.isnan(x.get(f"{c}_step_pct", np.nan)):
                v += f" ({x[f'{c}_step_pct']:+.1f}±{x[f'{c}_step_se']:.1f})"
            cells.append(v)
        last = g.iloc[-1]
        print(f"| {L[e]} {SL[s]} {lab} | " + " | ".join(cells)
              + f" | {last[f'{c}_vs{B}_pct']:+.2f}±{last[f'{c}_vs{B}_se']:.2f} (p {last[f'{c}_vs{B}_p']:.3f}) |")
keys = list(dict.fromkeys(zip(P.foldset, P.Wa, P.Wb)))
print("\nCumulative changes, fold 3, % ± SE (p); extra gain = full − HAR, pp")
print("| arm | crit | " + " | ".join(f"{a:,}→{b:,}" for _, a, b in keys) + " |")
print("|---|---|" + "---|" * len(keys))
SL2 = {"all_features": "full", "baseline": "HAR", "full_minus_har": "extra gain"}
for (fam, e, s, c), g in P.groupby(["family", "est", "set", "crit"], sort=False):
    cells = []
    for fs, a, b in keys:
        x = g[(g.foldset == fs) & (g.Wa == a) & (g.Wb == b)].iloc[0]
        cells.append(f"{x.pct:+.2f}±{x.se:.2f} ({x.p:.3f})")
    print(f"| {L[e]} {SL2[s]} | {c.upper()} | " + " | ".join(cells) + " |")
g = G[G.foldset == "fold3"]
print(f"\nFull − HAR gap by W (%), fold 3, and the extra gain {B:,}→W (pp ± SE)")
print("| model | crit | " + " | ".join(f"{w:,}" for w in Ws) + " |")
print("|---|---|" + "---|" * len(Ws))
for e, h in g.groupby("est", sort=False):
    for c in ("mse", "qlike"):
        cells = []
        for _, x in h.iterrows():
            v = f"{x[f'{c}_gap_pct']:+.1f}"
            gg = x.get(f"{c}_gain_full_minus_har_pp")
            if gg is not None and not np.isnan(gg):
                v += f" [{gg:+.2f}±{x[f'{c}_gain_diff_se']:.2f}]"
            cells.append(v)
        print(f"| {L[e]} | {c.upper()} | " + " | ".join(cells) + " |")
