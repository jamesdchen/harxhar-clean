"""Score the pooled tree walks (2026-10-07) at 16:00 against the per-bar rows of the same model.

experiments/trees_xgbtop_score_1600.py (the master table's plain scorer: f^2 B, 866 trade days 2020-01-03 ..
2024-04-30; QLIKE, DM, sign(s) Sharpe / Sortino mid and crossed, Newey-West t (lag 14) of the P&L change, the threshold
score ES_1 with its HAC t, days with a different position, the robust-clip Sharpe on 614 days; load_one's gates) run
UNCHANGED for one estimator, with its loop over (set, rung) reduced to the sets and its table names mapped:

  new (pooled)      results/pooled_trees_2026-10-07/yhat_pooltree_<est>_<set>.parquet; for all_features the bank's own
                    arm (lgbm: results/spxw_pnl/yhat_tree00.parquet = lgbm_opt_00; xgb: yhat_tree10.parquet = xgb_opt_00)
  per-bar T1 / T10  the same model's per-bar 16:00 tables:
                      lgbm  results/spxw_pnl/yhat_subtree_daily_<set>_lgbm / yhat_subtree_lgbm_<set>       (lgbm_opt_00)
                      xgb   results/trees_xgbtop_1600/yhat_subtree_daily_<set>_xgbtop / yhat_subtree_xgbtop_<set> (xgb_opt_00)
                      rf    results/trees_rf400_1600/yhat_subtree_daily_<set>_rf400 / yhat_subtree_rf400_<set>  (400-trial forest)

Pairs are pooled minus per-bar (DM negative = pooled more accurate; HAC t positive = more P&L).  Scores go to
results/pooled_trees_2026-10-07/score_<est>/ (levels.csv, pairs.csv, gates.csv, SUMMARY.md).  A set whose pooled table
does not exist yet is reported as a missing-table gate line and skipped.

Run:  python experiments/pooled_trees_score.py --est lgbm|xgb|rf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import trees_xgbtop_score_1600 as sc  # noqa: E402

POOL = "results/pooled_trees_2026-10-07"
PERBAR = {
    "lgbm": (
        "results/spxw_pnl",
        "yhat_subtree_daily_{b}_lgbm.parquet",
        "yhat_subtree_lgbm_{b}.parquet",
    ),
    "xgb": (
        "results/trees_xgbtop_1600",
        "yhat_subtree_daily_{b}_xgbtop.parquet",
        "yhat_subtree_xgbtop_{b}.parquet",
    ),
    "rf": (
        "results/trees_rf400_1600",
        "yhat_subtree_daily_{b}_rf400.parquet",
        "yhat_subtree_rf400_{b}.parquet",
    ),
}
BANK = {"lgbm": "yhat_tree00.parquet", "xgb": "yhat_tree10.parquet"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, choices=sorted(PERBAR))
    ap.add_argument(
        "--interim",
        action="store_true",
        help="all_features from yhat_pooltree_<est>_all_features_INTERIM1600.parquet (only the 16:00 rows the scorer reads; experiments/pooled_trees_interim1600.py), scores to score_<est>_interim1600/",
    )
    a, rest = ap.parse_known_args()
    est = a.est
    out = f"{POOL}/score_{est}" + ("_interim1600" if a.interim else "")
    out_abs = (ROOT / out).resolve()

    def rel(path: str) -> str:
        return os.path.relpath((ROOT / path).resolve(), out_abs)

    def table_name(rung: str, bucket: str, token: str) -> str:
        if token == "pooled":
            if bucket == "all_features" and est in BANK:
                return rel(f"results/spxw_pnl/{BANK[est]}")
            sfx = "_INTERIM1600" if (a.interim and bucket == "all_features") else ""
            return rel(f"{POOL}/yhat_pooltree_{est}_{bucket}{sfx}.parquet")
        d, t1, t10 = PERBAR[est]
        return rel(f"{d}/{t1 if token == 'T1' else t10}".format(b=bucket))

    sc.RUNGS = ("pooled",)  # type: ignore[assignment]
    sc.table_name = table_name  # type: ignore[assignment]
    sys.argv = [
        sys.argv[0],
        *rest,
        "--token",
        "pooled",
        "--label",
        f"pooled {est}",
        "--out",
        out,
        "--compare",
        f"per-bar T1={out}:T1",
        "--compare",
        f"per-bar T10={out}:T10",
    ]
    rc = sc.main()
    relabel_summary(out_abs / "SUMMARY.md", est)
    if a.interim:
        sm = out_abs / "SUMMARY.md"
        if sm.is_file():
            NL = chr(10)
            head, _, body = sm.read_text(encoding="utf-8").partition(NL)
            note = (
                "**INTERIM**: the all_features pooled table holds only the 16:00 rows of 2020-01-03 .. 2024-04-30 "
                f"(yhat_pooltree_{est}_all_features_INTERIM1600.parquet); the full table is still being computed."
            )
            sm.write_text(
                head + " (INTERIM all_features)" + NL + NL + note + NL + body,
                encoding="utf-8",
                newline=NL,
            )
    return rc


NAMES = {
    "lgbm": "LightGBM (lgbm_opt_00)",
    "xgb": "XGBoost (xgb_opt_00)",
    "rf": "random forest (400-trial rf400_rf_opt_00)",
}


def relabel_summary(path: Path, est: str) -> None:
    """The base scorer's SUMMARY.md is titled for the per-bar XGBoost study and carries a constant 'refit' column (the one
    rung, 'pooled', on every row): retitle it for this comparison, name the tables, and drop that column."""
    if not path.is_file():
        return
    d, t1, t10 = PERBAR[est]
    allf = (
        f"results/spxw_pnl/{BANK[est]} (the bank's own pooled arm)"
        if est in BANK
        else f"{POOL}/yhat_pooltree_{est}_all_features.parquet"
    )
    head = [
        f"# Pooled-bank {NAMES[est]} at 16:00 against the same model's per-bar T1 / T10 rows",
        "",
        "Written by `experiments/pooled_trees_score.py` (the plain scorer of `experiments/trees_xgbtop_score_1600.py`, unchanged) from "
        "`levels.csv`, `pairs.csv`, `gates.csv` in this folder. Nothing here enters results/spxw_pnl/, the paper, the GUI or the master table.",
        "",
        f"Tables. pooled: {POOL}/yhat_pooltree_{est}_<set>.parquet -- the bank's walk (24,000-bar trailing window over every bar, refit "
        f"at every bar, seed 42) on the set's pooled design columns; all_features: {allf}. per-bar T1 / T10: "
        f"{d}/{t1.format(b='<set>')} / {d}/{t10.format(b='<set>')}.",
    ]
    lines = path.read_text(encoding="utf-8").splitlines()
    body = lines[3:]  # after the base title, blank line and 'Written by' line
    out = []
    for ln in body:
        if ln.startswith("|"):
            cells = ln.split("|")
            ln = "|".join(cells[:2] + cells[3:])  # drop the constant rung column
        out.append(ln)
    path.write_text("\n".join(head + out) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
