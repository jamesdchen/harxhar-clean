"""Compact per-model part-2 tables from score14.py's csv files (read-only).
usage: python compact_part2.py <prefix of score14 output (…_qlike.csv, _gap.csv, _econ.csv)>"""

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
P = sys.argv[1]
Q, G, E = pd.read_csv(f"{P}_qlike.csv"), pd.read_csv(f"{P}_gap.csv"), pd.read_csv(f"{P}_econ.csv")
LAB = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}


def c(x, k, lab):
    v = x.get(f"{k}_pct_{lab}")
    return "—" if pd.isna(v) else f"{v:+.2f} ({x[f'{k}_p_{lab}']:.3f})"


for e in ("ridge", "lasso", "enet"):
    print(f"\n{LAB[e]}")
    print("| W | HAR 16:00 (vs 1,000) | full 16:00: step / vs 1,000 | HAR bar (vs 1,000) | full bar: step / vs 1,000 | "
          "gap 16:00 / bar | D6 HAR / full, ΔD6 [boot p] | ΔD6 b/a [p] | ΔD1 [LW p] | ΔSR614 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for W in sorted(Q.W.unique()):
        h = Q[(Q.model == e) & (Q.set == "baseline") & (Q.W == W)].iloc[0]
        f = Q[(Q.model == e) & (Q.set == "all_features") & (Q.W == W)].iloc[0]
        g = G[(G.model == e) & (G.W == W)].iloc[0]
        x = E[(E.model == e) & (E.W == W)].iloc[0]
        print(f"| {W:,} | {h.qlike16:.4f} ({c(h, 'vs1', '16')}) | {f.qlike16:.4f}: {c(f, 'step', '16')} / {c(f, 'vs1', '16')} | "
              f"{h.qlike_bar:.4f} ({c(h, 'vs1', 'bar')}) | {f.qlike_bar:.4f}: {c(f, 'step', 'bar')} / {c(f, 'vs1', 'bar')} | "
              f"{g.gap_pct_16:+.2f} ({g.gap_p_16:.3f}) / {g.gap_pct_bar:+.2f} ({g.gap_p_bar:.3f}) | "
              f"{x.d6_mid_har:.2f} / {x.d6_mid_full:.2f}, {x.dd6_mid:+.2f} [{x.d6_boot_p_mid:.3f}] | "
              f"{x.dd6_ba:+.2f} [{x.d6_boot_p_ba:.3f}] | {x.dd1_mid:+.2f} [{x.lw_p_mid:.3f}] | {x.dsr614_mid:+.2f} |")
