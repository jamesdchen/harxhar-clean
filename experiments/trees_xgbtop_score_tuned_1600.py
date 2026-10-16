"""Score the re-tuning families of the per-bar XGBoost re-centred on xgb_opt_00 (xgbtop, 2026-10-07) against the same
families of the shipped XGBoost (xgb_opt_06) and LightGBM.

experiments/trees_xgbtop_score_1600.py (the phase-1 scorer: the master table's plain scorer, f^2 B, 866 trade days
2020-01-03 .. 2024-04-30; QLIKE, DM, Sharpe / Sortino mid and crossed, Newey-West t (lag 14) of the P&L change, the
threshold score ES_1 with its HAC t, days with a different position, the robust-clip Sharpe on 614 days; the canonical
tables' levels reproduced against results/close_master_table/master_table.csv) run UNCHANGED with its rung loop
replaced by the 13 re-tuning families and its table names by the families' key patterns:

  tuned, tuned_daily, tunedq, tunedq_daily                      random search (RS10 / RS1, MSE / QLIKE rule)
  optuna_tp1, _tp5, _tp25, _tp250, _tp1_k10, _tp1_k25,
  _tp25_k10, _tp25_k25, optunaq_tp25                            Optuna (best of 50 / of the first k, MSE / QLIKE)

  table  yhat_subtree_<family>_<set>_<token>.parquet   (token xgbtop: results/trees_xgbtop_1600/; xgb / lgbm: the
                                                         canonical tables of results/spxw_pnl/)

Scores go to --out (default results/trees_xgbtop_1600/phase2/: levels.csv, pairs.csv, gates.csv, SUMMARY.md); the
xgbtop tables are read from --tables (default results/trees_xgbtop_1600/).  --families limits the families.

Run:  python experiments/trees_xgbtop_score_tuned_1600.py [--families tuned,tunedq] [--out ...] [--tables ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import trees_xgbtop_score_1600 as sc  # noqa: E402

FAMILIES = (
    "tuned",
    "tuned_daily",
    "tunedq",
    "tunedq_daily",
    "optuna_tp1",
    "optuna_tp5",
    "optuna_tp25",
    "optuna_tp250",
    "optuna_tp1_k10",
    "optuna_tp1_k25",
    "optuna_tp25_k10",
    "optuna_tp25_k25",
    "optunaq_tp25",
)


def pop(name: str, default: str) -> str:
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i : i + 2]
        return v
    return default


def main() -> int:
    fams = tuple(f for f in pop("--families", ",".join(FAMILIES)).split(",") if f)
    bad = [f for f in fams if f not in FAMILIES]
    if bad:
        raise SystemExit(f"unknown families {bad}")
    out = pop("--out", "results/trees_xgbtop_1600/phase2")
    tables = Path(pop("--tables", "results/trees_xgbtop_1600"))
    out_abs = (ROOT / out).resolve()
    rel = Path(
        os.path.relpath((ROOT / tables).resolve(), out_abs)
    )  # the new tables, seen from --out

    def table_name(fam: str, bucket: str, token: str) -> str:
        name = f"yhat_subtree_{fam}_{bucket}_{token}.parquet"
        return name if token in sc.CANONICAL else str(rel / name)

    sc.RUNGS = fams  # type: ignore[assignment]
    sc.table_name = table_name  # type: ignore[assignment]
    sys.argv += ["--token", "xgbtop", "--out", out]
    return sc.main()


if __name__ == "__main__":
    sys.exit(main())
