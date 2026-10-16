"""Plain-forecast scores of the moments_vix 16:00 trees beside baseline and all_features (2026-10-06).

The user: "on top of no exog features and all_features, add a version of every model that only uses
the moment bucket + the VIX exog var by itself".  This script scores the moments_vix tree tables
(rungs T1, T10, RS1, RS10 and the QLIKE-rule RS1q / RS10q; LightGBM, XGBoost, random forest; built by
experiments/trees_momvix_stack_1600.py) with the MASTER TABLE's own plain scorer, imported from
experiments/master_table_close.py (not edited): the target (TARGET_ARM of results/linear_subsection_dedup,
the arm root the plain master table of 2026-10-06 14:53 ran with), load_one (16:00 rows, the plain
back-transform f^2 B and its gates) and score_one (QLIKE of f^2 B against the target; the
last-30-min trade q = sign(plain - iv_var) on the straddle -- nearest out-of-the-money call + nearest
out-of-the-money put, same-day expiry, one position -- annualized Sharpe at mid and at crossed fills),
on the master table's day set: every trade day of the deck (866 days, 2020-01-03 .. 2024-04-30).
No recalibration (user decision 2026-10-06).  Point estimates only, as the master table.

GATE (reproduction): the same code scores the stored tables of the same rungs and models on baseline
and all_features and must equal results/close_master_table/master_table.csv (qlike_raw, Sharpe_mid,
Sharpe_crossed, pct_buy, n_days) to REPRO_TOL; the comparison columns of the output are then the
master table's own numbers.

Writes (--out, default results/trees_momvix_1600/):
  score_plain.csv     one row per moments_vix table: the master table's columns (score_one)
  compare_plain.csv   one row per (rung, model): QLIKE, Sharpe mid / crossed, % buy on baseline,
                      all_features (master_table.csv) and moments_vix
  score_gates.csv     load_one's gates for every moments_vix table + the reproduction gate rows
  SUMMARY.md          generated from the CSVs above

Run:  python experiments/trees_momvix_score_1600.py [--tables results/spxw_pnl] [--out results/trees_momvix_1600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402

ARM_ROOT = (
    "linear_subsection_dedup"  # the plain master table's --arm-root (its SUMMARY.md)
)
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
BUCKET = "moments_vix"
COMPARE_BUCKETS = ("baseline", "all_features")
MODELS = ("lgbm", "xgb", "rf")
REPRO_TOL = 1e-12
REPRO_COLS = ("qlike_raw", "Sharpe_mid", "Sharpe_crossed", "pct_buy")
# rung -> table stem pattern (bucket, model); the canonical names of writeup/CAMPAIGN_16H_2026-09-29.md
RUNGS = {
    "T1": "subtree_daily_{b}_{m}",
    "T10": "subtree_{m}_{b}",
    "RS1": "subtree_tuned_daily_{b}_{m}",
    "RS10": "subtree_tuned_{b}_{m}",
    "RS1q": "subtree_tunedq_daily_{b}_{m}",
    "RS10q": "subtree_tunedq_{b}_{m}",
}
TREE_LONG = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tables",
        default=str(ROOT / "results" / "spxw_pnl"),
        help="where the moments_vix tables are",
    )
    ap.add_argument("--out", default=str(ROOT / "results" / "trees_momvix_1600"))
    a = ap.parse_args()
    tables, out = Path(a.tables), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mtc.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    ref_path = next(mtc.SPXW / n for n, t in paper.items() if t == mtc.REFERENCE)
    ref_spec = mtc.describe_table(ref_path, paper)
    ref_res = mtc.load_one(ref_spec)
    assert not ref_res["error"], ref_res["error"]
    ref = ref_res["frame"]
    days = deck.index  # the master table's table-A day set: every trade day of the deck
    master = pd.read_csv(MASTER).set_index("key")
    assert int(master.loc[mtc.REFERENCE, "n_days"]) == len(days), (
        master.loc[mtc.REFERENCE, "n_days"],
        len(days),
    )

    def score(path: Path, key: str, label: str) -> tuple[dict | None, list[dict], str]:
        spec = mtc.Spec(key, label, "per-bar tree (moments_vix)", "table", str(path))
        res = mtc.load_one(spec)
        if res["error"]:
            return None, res["gates"], res["error"]
        f = res["frame"]
        have = f.index[f["plain"].notna()].normalize()
        miss = days.difference(have)
        if len(miss):
            return (
                None,
                res["gates"],
                f"{len(miss)} trade days without a plain forecast",
            )
        return mtc.score_one((spec, f, ref, days, deck, False)), res["gates"], ""

    rows, gates, comp = [], [], []
    for rung, pat in RUNGS.items():
        for m in MODELS:
            crow: dict[str, Any] = {"rung": rung, "model": TREE_LONG[m]}
            # reproduction gate on the stored tables of the same rung and model
            for b in COMPARE_BUCKETS:
                key = pat.format(b=b, m=m)
                p = mtc.SPXW / f"yhat_{key}.parquet"
                if key not in master.index or not p.is_file():
                    gates.append(
                        {
                            "gate": "reproduction: stored table in master_table.csv",
                            "forecast": key,
                            "ok": False,
                            "value": np.nan,
                        }
                    )
                    continue
                r, _g, err = score(p, key, key)
                if r is None:
                    gates.append(
                        {
                            "gate": f"reproduction: scored ({err})",
                            "forecast": key,
                            "ok": False,
                            "value": np.nan,
                        }
                    )
                    continue
                dev = max(
                    abs(float(r[c]) - float(master.loc[key, c])) for c in REPRO_COLS
                )
                ok = dev <= REPRO_TOL and int(r["n_days"]) == int(
                    master.loc[key, "n_days"]
                )
                gates.append(
                    {
                        "gate": "reproduction: this scorer == master_table.csv (max abs diff)",
                        "forecast": key,
                        "ok": ok,
                        "value": dev,
                    }
                )
                for c in REPRO_COLS:
                    crow[f"{c}_{b}"] = float(master.loc[key, c])
            key = pat.format(b=BUCKET, m=m)
            p = tables / f"yhat_{key}.parquet"
            if not p.is_file():
                crow["note"] = f"no table {p.name}"
                comp.append(crow)
                continue
            label = f"{rung} {TREE_LONG[m]} [{BUCKET}]"
            r, g, err = score(p, key, label)
            gates += [x | {"forecast": key} for x in g]
            if r is None:
                crow["note"] = f"not scored: {err}"
                comp.append(crow)
                continue
            rows.append(
                {"rung": rung, "model": TREE_LONG[m], "bucket": BUCKET, "table": p.name}
                | r
            )
            for c in REPRO_COLS:
                crow[f"{c}_{BUCKET}"] = float(r[c])
            comp.append(crow)
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "score_plain.csv", index=False)
    cp = pd.DataFrame(comp)
    cp.to_csv(out / "compare_plain.csv", index=False)
    gt = pd.DataFrame(gates)
    gt.to_csv(out / "score_gates.csv", index=False)
    n_bad = int((~gt["ok"].astype(bool)).sum()) if len(gt) else 0
    write_summary(
        out, cp, gt, len(days), str(days.min().date()), str(days.max().date())
    )
    print(cp.to_string(index=False))
    print(
        f"{len(sc)} moments_vix tables scored on {len(days)} days; gates: {len(gt) - n_bad} ok, {n_bad} failed"
    )
    return 1 if n_bad else 0


def write_summary(
    out: Path, cp: pd.DataFrame, gt: pd.DataFrame, n: int, d0: str, d1: str
) -> None:
    def f4(x: Any) -> str:
        return "" if pd.isna(x) else f"{float(x):.4f}"

    def f2(x: Any) -> str:
        return "" if pd.isna(x) else f"{float(x):.2f}"

    lines = [
        "# moments_vix 16:00 trees: plain-forecast scores beside baseline and all_features",
        "",
        "Written by `experiments/trees_momvix_score_1600.py` from its own CSVs in this folder "
        "(`compare_plain.csv`, `score_plain.csv`, `score_gates.csv`).",
        "",
        f"Input set `moments_vix` = the HAR + calendar backbone + the 7 ES return moments (`sumret`, `sumabsret`, "
        f"`sumret3`, `sumret4`, `sumpret2`, `sumbipow`, `sumautocov`) + `vix`. Scorer: the master table's plain scorer "
        f"(experiments/master_table_close.py `load_one` / `score_one`, imported): forecast = f^2 B (no recalibration); "
        f"QLIKE against the per-bar 16:00 target (lower is better); the last-30-min trade q = sign(plain - iv_var) on the "
        f"straddle (nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position), "
        f"annualized Sharpe at mid and at crossed fills. {n} trade days, {d0} .. {d1}. Point estimates only (as the "
        f"master table). baseline and all_features numbers are read from `results/close_master_table/master_table.csv`; "
        f"this scorer reproduces them (gate below).",
        "",
        "| rung | model | QLIKE baseline | QLIKE all_features | QLIKE moments_vix | Sharpe mid baseline | Sharpe mid all_features "
        "| Sharpe mid moments_vix | Sharpe crossed baseline | Sharpe crossed all_features | Sharpe crossed moments_vix | % buy moments_vix |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in cp.iterrows():
        lines.append(
            f"| {r['rung']} | {r['model']} | {f4(r.get('qlike_raw_baseline'))} | {f4(r.get('qlike_raw_all_features'))} | "
            f"{f4(r.get(f'qlike_raw_{BUCKET}'))} | {f2(r.get('Sharpe_mid_baseline'))} | {f2(r.get('Sharpe_mid_all_features'))} | "
            f"{f2(r.get(f'Sharpe_mid_{BUCKET}'))} | {f2(r.get('Sharpe_crossed_baseline'))} | {f2(r.get('Sharpe_crossed_all_features'))} | "
            f"{f2(r.get(f'Sharpe_crossed_{BUCKET}'))} | {f2(r.get(f'pct_buy_{BUCKET}'))} |"
        )
    rep = gt[gt["gate"].astype(str).str.startswith("reproduction")]
    other = gt[~gt["gate"].astype(str).str.startswith("reproduction")]
    lines += [
        "",
        "## Gates",
        "",
        f"- Reproduction of the stored baseline / all_features rows: {int(rep['ok'].astype(bool).sum())} of {len(rep)} "
        f"within {REPRO_TOL:g} (max abs diff {f4(rep['value'].max()) if len(rep) else ''}).",
        f"- load_one gates on the moments_vix tables (stamps = the target's, rv_raw = production, baseline = the "
        f"target's B): {int(other['ok'].astype(bool).sum())} of {len(other)} pass.",
    ]
    if "note" in cp:
        for _, r in cp[cp["note"].notna()].iterrows():
            lines.append(f"- {r['rung']} {r['model']}: {r['note']}")
    (out / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    sys.exit(main())
