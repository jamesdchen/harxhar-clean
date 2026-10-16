"""The pooled moments_vix arms on both clusters: Hoffman2 (the arms of record) vs CARC (replicate).

Both runs are the same spec, settings and data bytes (cluster/slurm/submit_pool_momvix.sh: epyc-7513,
numpy 2.3.5 / OpenBLAS 0.3.30 Haswell kernels, each arm as 20 tune-aligned chunks; cluster/
submit_pool_momvix.sh: Hoffman2, numpy 2.2.6 / OpenBLAS 0.3.29 SkylakeX kernels, whole series, ridge and
elastic net on intel-6736p, lasso on intel-gold-6140).  Each passed its own gate bit for bit against
stored pooled arms of its cluster.  This records how far the two forecasts are apart (BLAS rounding and
the warm-homotopy path) on the regular-hours rows and at 16:00, and the plain-scorer numbers of both
copies (the master-table functions, as experiments/score_pool_moments_vix.py; built in memory with
experiments/build_pool_moments_vix_yhat.py's build path, nothing written to results/spxw_pnl).

Writes results/linear_subsection_pool_momvix/cross_cluster.csv.
Run:  python experiments/pool_moments_vix_cross_cluster.py [moments_vol_vix]   (-> cross_cluster_<bucket>.csv)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import master_table_close as mtc  # noqa: E402
from build_subsection_yhat import ESTIMATORS, read_arm, with_production_rv  # noqa: E402

R = ROOT / "results" / "linear_subsection_pool_momvix"
BUCKET = sys.argv[1] if len(sys.argv) > 1 else "moments_vix"  # or moments_vol_vix


def rth(cluster: str, est: str) -> Path:
    return (
        R
        / cluster
        / "results"
        / "pool_momvix"
        / "fleet"
        / BUCKET
        / "none"
        / est
        / "tw2000"
        / "causal_tune_linear"
        / est
        / BUCKET
        / "results_none_rth.csv"
    )


def main() -> None:
    mtc.set_arm_root("linear_subsection_dedup")
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    tables = sorted(mtc.SPXW.glob("yhat_*.parquet"))
    ref_spec = next(
        s
        for s in (mtc.describe_table(p, paper) for p in tables)
        if s.key == mtc.REFERENCE
    )
    ref = mtc.load_one(ref_spec)["frame"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for est, short in ESTIMATORS.items():
            a = pd.read_csv(rth("carc", est), float_precision="round_trip")
            b = pd.read_csv(rth("hoffman2", est), float_precision="round_trip")
            assert (a["date"] == b["date"]).all()
            d = np.abs(a["pred_adj"].to_numpy() / b["pred_adj"].to_numpy() - 1.0)
            at16 = a["date"].str.endswith("16:00:00").to_numpy()
            row = {
                "estimator": short,
                "rth_rows": len(a),
                "rth_bitwise_rows": int((a["pred_adj"] == b["pred_adj"]).sum()),
                "rth_max_rel_diff": float(d.max()),
                "rth_median_rel_diff": float(np.median(d)),
                "rows_1600": int(at16.sum()),
                "bitwise_1600": int(
                    (a["pred_adj"].to_numpy() == b["pred_adj"].to_numpy())[at16].sum()
                ),
                "max_rel_diff_1600": float(d[at16].max()),
                "true_raw_bitwise": bool((a["true_raw"] == b["true_raw"]).all()),
            }
            scores, frames = {}, {}
            for cl in ("carc", "hoffman2"):
                tab = read_arm(rth(cl, est)).sort_values("t").reset_index(drop=True)
                tab = with_production_rv(tab, f"yhat_pool_{short}_{BUCKET} ({cl})")
                p = Path(tmp) / f"yhat_pool_{short}_{BUCKET}.parquet"
                tab.to_parquet(p, index=False)
                spec = mtc.describe_table(p, paper)
                res = mtc.load_one(spec)
                assert not res["error"] and all(g["ok"] for g in res["gates"]), res[
                    "error"
                ]
                f = frames[cl] = res["frame"]
                days = deck.index[
                    deck.index.isin(f.index[f["plain"].notna()].normalize())
                ]
                s = mtc.score_one((spec, f, ref, days, deck, False))
                scores[cl] = s
                row.update(
                    {
                        f"{cl}_n_days": s["n_days"],
                        f"{cl}_qlike_raw": s["qlike_raw"],
                        f"{cl}_Sharpe_mid": s["Sharpe_mid"],
                        f"{cl}_Sharpe_crossed": s["Sharpe_crossed"],
                    }
                )
            qa = mtc.trade_days(scores_frame(frames["carc"]), deck)["q"].to_numpy()
            qb = mtc.trade_days(scores_frame(frames["hoffman2"]), deck)["q"].to_numpy()
            row["trade_days_same_position"] = float(np.mean(qa == qb))
            rows.append(row)
            print(
                f"{short}: 16:00 bitwise {row['bitwise_1600']}/{row['rows_1600']}, max rel diff "
                f"{row['max_rel_diff_1600']:.2e}; QLIKE {row['carc_qlike_raw']:.6f} vs {row['hoffman2_qlike_raw']:.6f}; "
                f"Sharpe mid {row['carc_Sharpe_mid']:.4f} vs {row['hoffman2_Sharpe_mid']:.4f}, crossed "
                f"{row['carc_Sharpe_crossed']:.4f} vs {row['hoffman2_Sharpe_crossed']:.4f}; same position on "
                f"{row['trade_days_same_position']:.4f} of trade days"
            )
    sfx = "" if BUCKET == "moments_vix" else f"_{BUCKET}"
    pd.DataFrame(rows).to_csv(R / f"cross_cluster{sfx}.csv", index=False)


def scores_frame(f: pd.DataFrame) -> pd.Series:
    """The plain forecast indexed by day, as score_one hands it to trade_days."""
    return f["plain"].set_axis(f.index.normalize())


if __name__ == "__main__":
    main()
