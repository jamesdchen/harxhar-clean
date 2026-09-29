"""Per-bar trees against per-bar linear arms, pooled over the 13 bars, with a day-block interval.

``score_trees_subsection.py`` compares each tree arm with each linear arm bar by bar
(``trees_vs_linear.csv``: QLIKE, paired difference, day-block 95 % interval).  Its
summary line averages the per-bar percentages but carries no interval for the
average.  This adds one: the same joins (identical stamps, the same causal clock
back-transform, the same target gate), all 13 bars stacked, the paired QLIKE
difference averaged within each day and resampled by the scorer's circular day
blocks (``score_linear_subsection.day_block_ci``).

Output (``--out``): trees_vs_linear_pooled.csv, one row per bucket x tree model x
linear estimator: rows, bars, mean QLIKE tree / linear over all stacked rows,
difference, percent, day-block 95 % interval, mean of the 13 per-bar percentages.

Run:  python experiments/trees_vs_linear_pooled.py
          [--root results/linear_subsection_trees]
          [--linear-root results/linear_subsection/arms_hoffman2 ...]
          [--buckets live_feasible,all_features]
          [--out results/linear_subsection_trees/rescore_local]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
import score_trees_subsection as sts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=sts.DEFAULT_ROOT)
    ap.add_argument("--linear-root", action="append", default=None)
    ap.add_argument("--buckets", default="live_feasible,all_features")
    ap.add_argument("--tw", type=int, default=2000)
    ap.add_argument("--out", default="results/linear_subsection_trees/rescore_local")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    root, out = bst.resolve(a.root), bst.resolve(a.out)
    lin_roots = sts.split_roots(
        a.linear_root
        or ["results/linear_subsection/arms_hoffman2", "results/linear_subsection"]
    )
    buckets = [b.strip() for b in a.buckets.split(",") if b.strip()]
    arms = [m for m in bst.discover_tree_arms(root, a.tw) if m.bucket in buckets]
    print(
        f"{len(arms)} tree arms in {buckets}; linear roots {[str(r) for r in lin_roots]}"
    )
    lin_cache: dict[Path, pd.DataFrame | None] = {}
    stacks: dict[tuple, list[pd.DataFrame]] = {}
    for arm in arms:
        mine = sts.load_forecasts(arm.csv)
        if mine is None:
            continue
        for est, (path, _src) in sorted(
            sts.linear_paths(lin_roots, arm.bucket, arm.seg, arm.tw).items()
        ):
            if path not in lin_cache:
                lin_cache[path] = sts.load_forecasts(path)
            lin = lin_cache[path]
            if lin is None:
                continue
            j = mine[sts.COLS].join(lin[sts.COLS], how="inner", rsuffix="_b")
            gap = sts.rel_gap(j["true_raw"], j["true_raw_b"])
            assert len(j) and gap < sts.GATE_REL, (arm.key, est, gap)
            j = j.dropna(subset=["pred_clock", "pred_clock_b"])
            lt = slc.qlike(j["true_raw"], j["pred_clock"])
            ll = slc.qlike(j["true_raw"], j["pred_clock_b"])
            stacks.setdefault((arm.bucket, arm.model, est), []).append(
                pd.DataFrame({"seg": arm.seg, "lt": lt, "ll": ll}, index=j.index)
            )
    rows = []
    for (bucket, model, est), parts in sorted(stacks.items()):
        s = pd.concat(parts).sort_index()
        assert not s.index.duplicated().any(), (bucket, model, est)
        d = s["lt"] - s["ll"]
        lo, hi = base.day_block_ci(d)
        g_ = s.assign(d=d).groupby("seg")[["d", "ll"]].mean()
        per_bar = 100 * g_["d"] / g_["ll"]
        rows.append(
            dict(
                bucket=bucket,
                model=model,
                estimator=est,
                bars=int(s["seg"].nunique()),
                rows=len(s),
                days=int(s.index.normalize().nunique()),
                QLIKE_tree=float(s["lt"].mean()),
                QLIKE_linear=float(s["ll"].mean()),
                diff=float(d.mean()),
                pct=float(100 * d.mean() / s["ll"].mean()),
                ci_lo=lo,
                ci_hi=hi,
                pct_ci_lo=float(100 * lo / s["ll"].mean()),
                pct_ci_hi=float(100 * hi / s["ll"].mean()),
                mean_per_bar_pct=float(per_bar.mean()),
                tree_better=bool(hi < 0),
                tree_worse=bool(lo > 0),
            )
        )
    tab = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "trees_vs_linear_pooled.csv", index=False)
    print(f"wrote {out / 'trees_vs_linear_pooled.csv'} ({len(tab)} rows)")
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
