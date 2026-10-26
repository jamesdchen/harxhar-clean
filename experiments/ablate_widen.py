"""Final tree merge of the ablation run, in one process (results/ablations_2014start_w3000/RULE.txt section 8 and
addendum B (b)): T2 over every gate dir (ablate_trees.py gate2 -> <out>/../gate2_final.json), then ablate_trees.py
merge for every (estimator, arm) of the task file (complete piece sets only; mixed CPU models only with T2's bit
identity), then the merged forecasts compacted for the laptop: one parquet with the dates and one pred_adj column per
"<est>|<arm>" (the scorer's trees_wide.parquet).

usage: python experiments/ablate_widen.py --out <runs/trees> --gates D1,D2,.. --tasks <trees.txt> --to <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_trees as AT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--to", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    g2 = out.parent / "gate2_final.json"
    AT.cmd_gate2(argparse.Namespace(gates=a.gates, out=str(g2)))
    pairs = sorted({tuple(ln.split()[:2]) for ln in Path(a.tasks).read_text().splitlines() if ln.strip()})
    merged, not_merged = [], []
    for est, arm in pairs:
        try:
            AT.cmd_merge(argparse.Namespace(arm=arm, est=est, out=str(out), gate2=str(g2)))
            merged.append(f"{est}|{arm}")
        except AssertionError as e:
            not_merged.append(f"{est}|{arm}: {e}")
    print(f"merged {len(merged)}, not merged {len(not_merged)}:", *not_merged, sep="\n  ")
    cols, dates = {}, None
    for p in sorted((out / "forecasts").glob("*.parquet")):
        est, arm = ("rf_skx", p.stem[len("rf_skx_") :]) if p.stem.startswith("rf_skx_") else p.stem.split("_", 1)
        f = pd.read_parquet(p)
        d = pd.DatetimeIndex(f["date"])
        assert d.is_monotonic_increasing and len(d) == 33970, p.name
        if dates is None:
            dates = d
        assert d.equals(dates), p.name
        cols[f"{est}|{arm}"] = f["pred_adj"].to_numpy()
    assert dates is not None, "no merged forecasts"
    w = pd.DataFrame({"date": dates, **dict(sorted(cols.items()))})
    w.to_parquet(a.to, index=False)
    print(f"{len(cols)} arms x {len(dates)} dates -> {a.to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
