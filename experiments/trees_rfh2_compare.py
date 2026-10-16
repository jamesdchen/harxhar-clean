"""Cross-CPU check of the per-bar 16:00 dev forest (2026-10-07, D.19-style): the six cells recomputed on Hoffman2
(cluster/submit_trees_rfh2.sh; merged arms pulled to results/h2check_<set>/) against the stored CARC epyc-7513 tables.

For each set (rfdev: rf_opt_00, results/spxw_pnl/*_rfdev; rfprev163: trial 163, results/trees_rfdev_preview163/*_rfp163)
and cell (T1 / T10 x baseline / all_features / lean_vol_vix): the Hoffman2 arm is stacked with the campaign's builder
(experiments/build_subsection_tree_yhat.py, its gates on) into results/h2check_<set>/stack/, then compared with the
CARC table on the 1469 forecast rows: max |pred_adj difference| and max relative difference of the plain forecast
f^2 B, rows bit-identical; on the 866 trade days (the master table's plain scorer): positions identical (q = sign(plain
- iv_var)), QLIKE and Sharpe mid / crossed of both and their differences.  The CPU class of every Hoffman2 chunk
(class_census.csv: AVX-512 / AVX2 / AVX and the CPU model) splits the row comparison by class.

Writes results/h2check_<set>/compare.csv (one row per cell), compare_by_class.csv, SUMMARY.md.
Run:  python experiments/trees_rfh2_compare.py --set rfdev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import master_table_close as mtc  # noqa: E402

BUCKETS = ("baseline", "all_features", "lean_vol_vix")
SETS = {
    "rfdev": ("results/spxw_pnl", "rfdev", "cluster/trees_rfdev_tasks_untuned.txt"),
    "rfprev163": (
        "results/trees_rfdev_preview163",
        "rfp163",
        "cluster/trees_rfprev_tasks_untuned.txt",
    ),
}
W = 2000


def carc_table(rung: str, b: str, d: str, tok: str) -> Path:
    return (
        ROOT
        / d
        / (
            f"yhat_subtree_daily_{b}_{tok}.parquet"
            if rung == "t1"
            else f"yhat_subtree_{tok}_{b}.parquet"
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    a = ap.parse_args()
    d, tok, tasks = SETS[a.set]
    h2 = ROOT / "results" / f"h2check_{a.set}"
    bst.BARS = ("bar1600",)  # type: ignore[assignment]
    bst.MODELS = ("rf",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    for rung in ("t1", "t10"):
        saved = sys.argv
        try:
            sys.argv = [
                bst.__file__,
                "--root",
                str(h2 / rung),
                "--out",
                str(h2 / "stack" / rung),
            ]
            bst.main()
        except SystemExit as e:
            print(f"builder {rung}: {e.code}")
        finally:
            sys.argv = saved
    census = pd.read_csv(h2 / "class_census.csv")
    census["chunk"] = census["chunk"].astype(str).str.lstrip("c")
    tl = pd.read_csv(
        ROOT / tasks,
        sep=r"\s+",
        header=None,
        names=[
            "pack",
            "rung",
            "bucket",
            "model",
            "tw",
            "seg",
            "chunk",
            "start",
            "end",
            "halo",
        ],
    )
    tl["chunk"] = tl["chunk"].astype(str)
    mtc.set_arm_root("linear_subsection_dedup")
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    ref = mtc.load_one(
        mtc.describe_table(
            next(mtc.SPXW / n for n, t in paper.items() if t == mtc.REFERENCE), paper
        )
    )["frame"]
    days = deck.index
    rows, by_class = [], []
    for rung in ("t1", "t10"):
        for b in BUCKETS:
            hp = h2 / "stack" / rung / f"yhat_subtree_rf_{b}.parquet"
            cp = carc_table(rung, b, d, tok)
            if not hp.is_file() or not cp.is_file():
                rows.append(
                    {
                        "rung": rung.upper(),
                        "bucket": b,
                        "note": f"missing {hp.name if not hp.is_file() else cp.name}",
                    }
                )
                continue
            hx = mtc.read_table_1600(hp, ["yhat", "baseline"])
            cx = mtc.read_table_1600(cp, ["yhat", "baseline"])
            j = hx.join(cx, lsuffix="_h2", rsuffix="_carc", how="inner")
            dy = (j["yhat_h2"] - j["yhat_carc"]).abs()
            ph, pc = (
                j["yhat_h2"] ** 2 * j["baseline_h2"],
                j["yhat_carc"] ** 2 * j["baseline_carc"],
            )
            rel = ((ph - pc).abs() / pc.abs()).max()
            res = {}
            for name, p in (("h2", hp), ("carc", cp)):
                spec = mtc.Spec(name, name, "per-bar tree", "table", str(p))
                lo = mtc.load_one(spec)
                res[name] = (
                    spec,
                    lo["frame"],
                    mtc.score_one((spec, lo["frame"], ref, days, deck, False)),
                )
            pair = mtc.score_one(
                (res["h2"][0], res["h2"][1], res["carc"][1], days, deck, False)
            )
            rows.append(
                {
                    "rung": rung.upper(),
                    "bucket": b,
                    "rows": len(j),
                    "rows_bit_identical": int(
                        (j["yhat_h2"].to_numpy() == j["yhat_carc"].to_numpy()).sum()
                    ),
                    "max_abs_pred_adj_diff": float(dy.max()),
                    "max_rel_plain_diff": float(rel),
                    "positions_identical": float(pair["same_position_as_ref"])
                    * len(days),
                    "n_days": len(days),
                    "qlike_h2": res["h2"][2]["qlike_raw"],
                    "qlike_carc": res["carc"][2]["qlike_raw"],
                    "sharpe_mid_h2": res["h2"][2]["Sharpe_mid"],
                    "sharpe_mid_carc": res["carc"][2]["Sharpe_mid"],
                    "sharpe_crossed_h2": res["h2"][2]["Sharpe_crossed"],
                    "sharpe_crossed_carc": res["carc"][2]["Sharpe_crossed"],
                }
            )
            # rows by the CPU class of the Hoffman2 chunk that computed them
            sub = tl[(tl["rung"] == rung) & (tl["bucket"] == b)]
            stamps = j.index.sort_values()
            for _, t in sub.iterrows():
                cc = census[
                    (census["rung"] == rung)
                    & (census["bucket"] == b)
                    & (census["chunk"] == t["chunk"])
                ]
                cls = (
                    f"{cc['class'].iloc[0]} {cc['cpu'].iloc[0]}"
                    if len(cc)
                    else "unknown"
                )
                r0, r1 = (
                    int(t["start"]) - W,
                    (int(t["end"]) - W if int(t["end"]) >= 0 else len(stamps)),
                )
                st = stamps[r0:r1]
                dd = dy.reindex(st)
                by_class.append(
                    {
                        "rung": rung.upper(),
                        "bucket": b,
                        "chunk": t["chunk"],
                        "class": cls,
                        "rows": len(st),
                        "bit_identical": int((dd == 0).sum()),
                        "max_abs_diff": float(dd.max()) if len(dd) else np.nan,
                    }
                )
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(h2 / "compare.csv", index=False)
    bc = pd.DataFrame(by_class)
    bc.to_csv(h2 / "compare_by_chunk.csv", index=False)
    cls = bc.groupby("class").agg(
        chunks=("chunk", "count"),
        rows=("rows", "sum"),
        bit_identical=("bit_identical", "sum"),
        max_abs_diff=("max_abs_diff", "max"),
    )
    cls.to_csv(h2 / "compare_by_class.csv")
    print(cmp_df.to_string(index=False))
    print(cls.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
