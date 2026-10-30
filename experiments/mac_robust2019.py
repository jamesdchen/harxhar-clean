"""Addendum L robustness (2026-10-30): the macro arms' every-bar QLIKE change and DM with and without the 2018-19
shutdown's scaling outliers (bars of 2019-01-22 .. 2019-06-14, where four macro columns reach 15-25 robust IQRs;
results/macro_calendar_2026-10-29/AUDIT_2026-10-30.md), and the check that every macro arm changes its reference
arm's forecast.  Uses experiments/ablate_score.py's panel, truth, bars, loss and DM unchanged.

usage: python experiments/mac_robust2019.py [--res results/ablations_2014start_w3000] -> <res>/scores_fomcfix/robust2019.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ablate_score as S  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

LO, HI = pd.Timestamp("2019-01-22"), pd.Timestamp("2019-06-15")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(S.ROOT / "results" / "ablations_2014start_w3000"))
    a = ap.parse_args()
    res = Path(a.res)
    panel, _b = S.load_panel(res, design="fix")
    rv = S.core_rv()
    days = S.evaluation_days(rv)
    idx = S.scored(panel, rv, days)
    keep = idx[(idx < LO) | (idx >= HI)]
    rows, same = [], []
    for (cls, arm), f in sorted(panel.items()):
        if not S.is_mac(arm):
            continue
        if arm.startswith("add_rm_fomc_mac"):
            ref = (cls, "add_rm_fomc")
        elif arm == "full_mac":
            ref = (cls, "full_fix") if (cls, "full_fix") in panel else (cls, "full")
        else:
            ref = (cls, "har")
        if ref not in panel:
            continue
        g = panel[ref]
        if np.array_equal(f.reindex(idx).to_numpy(), g.reindex(idx).to_numpy()):
            same.append(f"{cls}|{arm}")
        for span, ix in (("all", idx), ("ex2019h1", keep)):
            q1, _ = S.losses(f, ix, rv)
            q0, _ = S.losses(g, ix, rv)
            d = dm_test(S.daily(q1, ix), S.daily(q0, ix))
            rows.append(
                dict(
                    cls=cls,
                    arm=arm,
                    ref=ref[1],
                    span=span,
                    n_bars=len(ix),
                    pct=100.0 * (q1.mean() / q0.mean() - 1.0),
                    dm=float(d["dm"] if isinstance(d, dict) else d[0]),
                    p=float(d["p"] if isinstance(d, dict) else d[1]),
                )
            )
    out = pd.DataFrame(rows)
    p = res / "scores_fomcfix" / "robust2019.csv"
    out.to_csv(p, index=False)
    w = out.pivot_table(index=["cls", "arm"], columns="span", values=["pct", "dm"])
    w["dpct"] = w[("pct", "ex2019h1")] - w[("pct", "all")]
    print(f"{len(out) // 2} macro pairs; bars all {len(idx)}, without 2019-01-22..06-14 {len(keep)}")
    print(f"largest |change in pct| without H1 2019: {w['dpct'].abs().max():.3f} pp")
    big = w[w["dpct"].abs() > 0.1]
    print(big.round(2).to_string() if len(big) else "no pair moves more than 0.1 pp")
    flips = w[np.sign(w[("dm", "all")].round(6)) != np.sign(w[("dm", "ex2019h1")].round(6))]
    sig = w[(w[("dm", "all")].abs() >= 1.96) != (w[("dm", "ex2019h1")].abs() >= 1.96)]
    print(f"DM sign flips: {len(flips)}; crossings of |DM| = 1.96: {len(sig)}")
    if len(sig):
        print(sig.round(2).to_string())
    print(f"macro arms identical to their reference: {same or 'none'} -> {p}")
    return 3 if same else 0


if __name__ == "__main__":
    raise SystemExit(main())
