"""Score the moments_vix Optuna forecasts exactly as the PLAIN master table scores every forecast.

The master table (experiments/master_table_close.py, run 2026-10-06 with --arm-root
linear_subsection_dedup; results/close_master_table/PLAIN_DONE) is NOT re-run.  Its own functions
are imported, never re-implemented: load_target (the common 16:00 target), describe-free Spec +
load_one (the table's 16:00 rows -> the plain back-transform f^2 B, with its gates), score_one
(QLIKE of the plain forecast against the spec's target; the last-30-min trade
q = sign(plain - iv_var) on the straddle, mid and crossed fills; Sharpe) on the 866 trade days
paired with the reference (blk2).

Scorer gate: the stored Optuna tables of the other buckets are re-scored here and must equal
their master_table.csv rows (qlike_raw, Sharpe_mid, Sharpe_crossed, n_buy) to REPRO_TOL; a
failure stops the script before any moments_vix number is written.

Writes <out>/momvix_vs_master.csv: one row per (variant, model) -- moments_vix (this run) next
to the same variant on baseline and all_features (read from master_table.csv) -- and
<out>/momvix_scores.csv (every column score_one returns, moments_vix rows).

Usage: python experiments/momvix_optuna_score.py --tables-dir results/spxw_pnl
           [--out results/linear_subsection_trees_optuna_mask_momvix/report]
       python experiments/momvix_optuna_score.py --bucket moments_vol_vix --also moments_vix
           (writes <bucket>_vs_master.csv; --also: buckets scored from their tables beside it)
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

import master_table_close as mtc  # noqa: E402

ARM_ROOT = (
    "linear_subsection_dedup"  # the plain master table's arm root (its SUMMARY.md)
)
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
BUCKET = "moments_vix"
COMPARE = ("baseline", "all_features")
MODELS = ("lgbm", "xgb", "rf")
STEMS = {  # variant -> table stem (experiments/build_subsection_tree_optuna_yhat.py TABLES)
    "tp5": "subtree_optuna_tp5",
    "tp1": "subtree_optuna_tp1",
    "tp25": "subtree_optuna_tp25",
    "tp250": "subtree_optuna_tp250",
    "tp1_k10": "subtree_optuna_tp1_k10",
    "tp1_k25": "subtree_optuna_tp1_k25",
    "tp25_k10": "subtree_optuna_tp25_k10",
    "tp25_k25": "subtree_optuna_tp25_k25",
    "tp25_q": "subtree_optunaq_tp25",
}
CHECK = ("qlike_raw", "Sharpe_mid", "Sharpe_crossed", "n_buy")
REPRO_TOL = 1e-12


def score(spec: mtc.Spec, ref_f: pd.DataFrame, deck: pd.DataFrame) -> dict:
    res = mtc.load_one(spec)
    if res["error"] or res["frame"] is None:
        raise SystemExit(f"{spec.key}: {res['error']}")
    bad = [g["gate"] for g in res["gates"] if not g["ok"]]
    if bad:
        raise SystemExit(f"{spec.key}: load gates failed: {bad}")
    f = res["frame"]
    days = deck.index.intersection(f.index[f["plain"].notna()].normalize())
    days = days.intersection(ref_f.index[ref_f["plain"].notna()].normalize())
    out = mtc.score_one((spec, f, ref_f, days, deck, False))
    out["n_gates_ok"] = len(res["gates"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default=str(ROOT / "results" / "spxw_pnl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument(
        "--also", default="", help="comma list of buckets scored from their tables too"
    )
    a = ap.parse_args()
    bucket = a.bucket
    also = [b for b in a.also.split(",") if b]
    tag = {
        "moments_vix": "momvix",
        "moments_vol_vix": "momvolvix",
        "lean_vol_vix": "leanvolvix",
    }.get(bucket, bucket)
    if not a.out:
        a.out = str(
            ROOT / "results" / f"linear_subsection_trees_optuna_mask_{tag}" / "report"
        )
    tdir, out = Path(a.tables_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mtc.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    master = pd.read_csv(MASTER).set_index("key")
    ref_path = mtc.asl.yhat_paths(ROOT)[mtc.REFERENCE]
    ref = mtc.load_one(
        mtc.Spec(mtc.REFERENCE, "reference", "paper", "table", str(ref_path))
    )
    ref_f = ref["frame"]
    # ---- scorer gate: stored tables of the other buckets reproduce their master-table rows
    worst = 0.0
    for key in (
        "subtree_optuna_tp5_baseline_lgbm",
        "subtree_optunaq_tp25_all_features_rf",
        "subtree_optuna_tp1_k10_all_features_xgb",
    ):
        s = score(
            mtc.Spec(
                key,
                key,
                "check",
                "table",
                str(ROOT / "results" / "spxw_pnl" / f"yhat_{key}.parquet"),
            ),
            ref_f,
            deck,
        )
        for c in CHECK:
            d = abs(float(s[c]) - float(master.loc[key, c]))
            worst = max(worst, d)
            if d > REPRO_TOL:
                raise SystemExit(
                    f"SCORER GATE FAIL {key} {c}: {s[c]} vs master {master.loc[key, c]}"
                )
    print(
        f"scorer gate: 3 stored Optuna rows reproduce master_table.csv (max abs diff {worst:.1e})"
    )
    rows, full = [], []
    for v, stem in STEMS.items():
        for m in MODELS:
            r: dict = {"variant": v, "model": m}
            for b0 in [bucket, *also]:
                key = f"{stem}_{b0}_{m}"
                p = tdir / f"yhat_{key}.parquet"
                if p.is_file():
                    s = score(
                        mtc.Spec(key, key, "momvix", "table", str(p)), ref_f, deck
                    )
                    if b0 == bucket:
                        full.append(s)
                    r |= {
                        f"{b0}_n_days": s["n_days"],
                        f"{b0}_qlike": s["qlike_raw"],
                        f"{b0}_sharpe_mid": s["Sharpe_mid"],
                        f"{b0}_sharpe_crossed": s["Sharpe_crossed"],
                        f"{b0}_n_buy": s["n_buy"],
                        f"{b0}_n_extreme_days": s["n_extreme_days"],
                    }
                else:
                    r["note"] = f"no table {p.name}"
            for b in COMPARE:
                k2 = f"{stem}_{b}_{m}"
                if k2 in master.index:
                    r |= {
                        f"{b}_qlike": master.loc[k2, "qlike_raw"],
                        f"{b}_sharpe_mid": master.loc[k2, "Sharpe_mid"],
                        f"{b}_sharpe_crossed": master.loc[k2, "Sharpe_crossed"],
                    }
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"{tag}_vs_master.csv", index=False)
    if full:
        pd.DataFrame(full).to_csv(out / f"{tag}_scores.csv", index=False)
    with pd.option_context(
        "display.width",
        250,
        "display.max_columns",
        30,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(tab.to_string(index=False))
    print(
        f"wrote {out / (tag + '_vs_master.csv')} ({int(np.sum(['note' not in r or pd.isna(r.get('note')) for r in rows]))} rows scored)"
    )


if __name__ == "__main__":
    main()
