"""Scores and reading of the trees' factor test beyond HAR + RM + FOMC (results/beyond_trees_2026-10-27/RULE.txt
sections 0 and 1, addendum 1 (f)).  Requested by the author after the beyond-base results; not pre-registered.

Inputs: the merged factor arms (results/beyond_trees_2026-10-27/factor/forecasts_wide.parquet: date, <est>|<arm>
pred_adj), BASE = the ablation's add_rm_fomc (results/ablations_2014start_w3000/forecasts/trees_wide_rmfomc.parquet),
FULL = Table 1's patched full set (results/pooled_trees_final_2014start_fomcfix_w3000/forecasts.parquet, all_features;
its B for every arm).  plain = pred_adj^2 B against the bar's actual RV; Table 1's 33,527 bars; spans every bar,
ORDINARY (the 81 scheduled FOMC statement days removed), EVENT; Diebold-Mariano (HLN) on daily mean QLIKE, hac_lag 8.

usage: python experiments/beyond_factor_score.py [--res results/beyond_trees_2026-10-27/factor]
       -> <res>/scores.csv, reading.json, tables.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_score as AS  # noqa: E402
import beyond_factor_trees as BF  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

HAC_LAG = 8
SPANS = ("ordinary", "all", "event")
SHORT_K = ("pca1", "pca2", "pca3", "pls1", "pls3")  # k <= 3
LABEL = "requested by the author after the beyond-base results; not pre-registered"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(ROOT / "results" / "beyond_trees_2026-10-27" / "factor"))
    a = ap.parse_args()
    res = Path(a.res)
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    ev = AS.event_days(days)
    full = BF.full_template()
    base = BF.base_template().set_index("date")
    wide = pd.read_parquet(res / "forecasts_wide.parquet").set_index("date")
    panel: dict[tuple[str, str], pd.Series] = {}
    for est in BF.EST:
        ft = full[full["model"] == est].set_index("date")
        B = ft["B"]
        panel[(est, "full")] = ft["pred_adj"] ** 2 * B
        b0 = base[f"{est}|{BF.BASE_ARM}"].reindex(B.index)
        assert np.isfinite(b0.to_numpy()).all(), est
        panel[(est, "base")] = b0**2 * B
        for arm in BF.FACTOR_ARMS:
            w = wide[f"{est}|{arm}"].reindex(B.index)
            assert np.isfinite(w.to_numpy()).all(), (est, arm)
            panel[(est, arm)] = w**2 * B
    idx = AS.scored(panel, rv, days)
    assert len(idx) == 33527, len(idx)
    on_ev = pd.DatetimeIndex(idx.normalize()).isin(ev)
    spans = {"all": idx, "ordinary": idx[~on_ev], "event": idx[on_ev]}
    assert pd.DatetimeIndex(spans["event"].normalize()).nunique() == 81
    L: dict = {}
    for k, f in panel.items():
        for sp, ix in spans.items():
            q, _ = AS.losses(f, ix, rv)
            L[(k, sp)] = AS.daily(q, ix), float(q.mean())
    rows = []
    for est in BF.EST:
        for sp in SPANS:
            lb, lf = L[((est, "base"), sp)][1], L[((est, "full"), sp)][1]
            for arm in ["base", *BF.FACTOR_ARMS, "full"]:
                da, la = L[((est, arm), sp)]
                rb = dm_test(da, L[((est, "base"), sp)][0], hac_lag=HAC_LAG) if arm != "base" else None
                rf = dm_test(da, L[((est, "full"), sp)][0], hac_lag=HAC_LAG) if arm != "full" else None
                rows.append(
                    dict(
                        est=est,
                        span=sp,
                        arm=arm,
                        n_days=len(da),
                        qlike=la,
                        pct_vs_base=100.0 * (la / lb - 1.0),
                        dm_vs_base=np.nan if rb is None else float(rb["dm"]),
                        p_vs_base=np.nan if rb is None else float(rb["p"]),
                        dm_vs_full=np.nan if rf is None else float(rf["dm"]),
                        p_vs_full=np.nan if rf is None else float(rf["p"]),
                        R=(lb - la) / (lb - lf),
                        gain_full_pct=100.0 * (lf / lb - 1.0),
                    )
                )
    S = pd.DataFrame(rows)
    S.to_csv(res / "scores.csv", index=False)
    reading: dict = {"label": LABEL, "span": "ordinary", "models": {}}
    for est in BF.EST:
        o = S[(S["est"] == est) & (S["span"] == "ordinary")].set_index("arm")
        shared = [
            arm
            for arm in SHORT_K
            if o.loc[arm, "R"] >= 0.75 and not (o.loc[arm, "dm_vs_full"] > 0 and o.loc[arm, "p_vs_full"] < 0.05)
        ]
        r10 = float(o.loc["pca10", "R"])
        full_beats_10 = bool(o.loc["pca10", "dm_vs_full"] > 0 and o.loc["pca10", "p_vs_full"] < 0.05)
        if shared:
            rd = "SHARED COMPONENT"
        elif r10 < 0.5 and full_beats_10:
            rd = "DENSE BEYOND A FEW COMPONENTS"
        else:
            rd = "INTERMEDIATE"
        best = max(SHORT_K, key=lambda x: float(o.loc[x, "R"]))
        reading["models"][est] = dict(
            reading=rd,
            shared_arms=shared,
            best_short=best,
            R_best_short=float(o.loc[best, "R"]),
            R_pca10=r10,
            full_beats_pca10=full_beats_10,
            dm_full_vs_pca10=-float(o.loc["pca10", "dm_vs_full"]),
            p_full_vs_pca10=float(o.loc["pca10", "p_vs_full"]),
        )
    rds = {v["reading"] for v in reading["models"].values()}
    reading["agreement"] = "agree" if len(rds) == 1 else "split"
    (res / "reading.json").write_text(json.dumps(reading, indent=1))
    T = [f"# Trees' factor test beyond HAR + RM + FOMC ({LABEL})", ""]
    for sp in SPANS:
        T += [
            f"## {sp}",
            "",
            "| model | arm | QLIKE | % vs BASE | DM vs BASE (p) | DM vs FULL (p) | R |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, x in S[S["span"] == sp].iterrows():
            T.append(
                f"| {x.est} | {x.arm} | {x.qlike:.5f} | {x.pct_vs_base:+.2f} | {x.dm_vs_base:+.2f} ({x.p_vs_base:.3f}) | "
                f"{x.dm_vs_full:+.2f} ({x.p_vs_full:.3f}) | {x.R:.3f} |"
            )
        T.append("")
    T += ["## Reading (ORDINARY)", "", "```", json.dumps(reading, indent=1), "```"]
    (res / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
    print("\n".join(T))
    return 0


if __name__ == "__main__":
    sys.exit(main())
