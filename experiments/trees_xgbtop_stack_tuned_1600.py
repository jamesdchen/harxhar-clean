"""Stack the re-tuning rungs of the per-bar XGBoost re-centred on xgb_opt_00 (xgbtop, 2026-10-07) under NEW keys, in
results/trees_xgbtop_1600/ (NOT results/spxw_pnl/: the master table's other-table gate would fail on unknown keys; the
main session decides integration).

Random search (RS10 / RS1; specs/causal_tune_trees_tuned_xgbtop.py, merged by experiments/reduce_trees_tuned_chunks.py):
the canonical tuned builder experiments/build_subsection_tree_tuned_yhat.py, UNEDITED (its gates: profile B against the
production table to 1e-9, every stamp in it, the production rv_raw carried; stamps / B / rv_raw equal to the untuned
twin table), with its module lists narrowed to the 16:00 bar, XGBoost and the three sets, run on each rung's root into
a staging folder; the tables are renamed with the model token xgbtop (the arms' model directory stays xgb):
  rung  root                                                   -> results/trees_xgbtop_1600/
  RS10  results/linear_subsection_trees_tuned_mask_xgbtop/rs10    yhat_subtree_tuned[q]_<set>_xgbtop.parquet
  RS1   results/linear_subsection_trees_tuned_mask_xgbtop/rs1     yhat_subtree_tuned[q]_daily_<set>_xgbtop.parquet
  (tuned = the MSE rule, the arm of record; tunedq = the QLIKE rule)
the untuned twin for the gate = the phase-1 xgbtop tables (yhat_subtree_xgbtop_<set> for RS10,
yhat_subtree_daily_<set>_xgbtop for RS1), staged under the builder's expected name.

Optuna (specs/causal_tune_trees_optuna_xgbtop.py, merged by experiments/reduce_trees_optuna.py): the canonical Optuna
builder experiments/build_subsection_tree_optuna_yhat.py, UNEDITED, as experiments/trees_rfp163_stack_optuna.py runs it,
narrowed to XGBoost and the three sets, tables renamed with the token:
  yhat_subtree_optuna_tp<N>[_k<K>]_<set>_xgbtop.parquet, yhat_subtree_optunaq_tp25_<set>_xgbtop.parquet

Usage: python experiments/trees_xgbtop_stack_tuned_1600.py [--part tuned|optuna|all] [--out results/trees_xgbtop_1600]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_optuna_yhat as bso  # noqa: E402
import build_subsection_tree_tuned_yhat as btt  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402

BUCKETS = ("baseline", "all_features", "lean_vol_vix")
TOKEN = "xgbtop"
T = ROOT / "results" / "linear_subsection_trees_tuned_mask_xgbtop"
P = ROOT / "results" / "linear_subsection_trees_optuna_mask_xgbtop"
RS = {  # rung -> (tag suffix, phase-1 twin table pattern)
    "rs10": ("", "yhat_subtree_xgbtop_{b}.parquet"),
    "rs1": ("_daily", "yhat_subtree_daily_{b}_xgbtop.parquet"),
}


def run(main, argv: list[str]) -> int:
    saved = sys.argv
    try:
        sys.argv = argv
        main()
        return 0
    except SystemExit as e:
        return 0 if e.code in (None, 0) else 1
    finally:
        sys.argv = saved


def tuned(out: Path) -> list[str]:
    bst.BARS = ("bar1600",)  # type: ignore[assignment]
    bst.MODELS = ("xgb",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    problems: list[str] = []
    for rung, (suffix, twin) in RS.items():
        st = out / "stack" / rung
        tw = st / "twin"
        tw.mkdir(parents=True, exist_ok=True)
        for b in BUCKETS:  # the phase-1 table under the builder's untuned-twin name
            src = out / twin.format(b=b)
            if src.is_file():
                shutil.copyfile(src, tw / f"yhat_subtree_xgb_{b}.parquet")
            else:
                problems.append(f"no phase-1 twin {src.name}")
        if run(
            btt.main,
            [
                btt.__file__,
                "--root",
                str(T / rung),
                "--out",
                str(st),
                "--untuned-dir",
                str(tw),
            ],
        ):
            problems.append(f"{rung}: the tuned builder reported a gate failure")
        for b in BUCKETS:
            for tag in ("tuned", "tunedq"):
                src = st / f"yhat_subtree_{tag}_{b}_xgb.parquet"
                dst = out / f"yhat_subtree_{tag}{suffix}_{b}_{TOKEN}.parquet"
                if not src.is_file():
                    problems.append(f"missing {src.name} ({rung})")
                    continue
                shutil.copyfile(src, dst)
                print(f"  {src.name} ({rung}) -> {dst.name}")
    return problems


def optuna(out: Path) -> list[str]:
    def table_name(path: str, bucket: str, model: str) -> str:
        assert model == "xgb", model
        return f"yhat_{bso.TABLES[path]}_{bucket}_{TOKEN}.parquet"

    bst.MODELS = ("xgb",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    bso.table_name = table_name  # type: ignore[assignment]
    return (
        ["optuna: the Optuna builder reported a failure"]
        if run(bso.main, [bso.__file__, "--root", str(P), "--out", str(out)])
        else []
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("tuned", "optuna", "all"))
    ap.add_argument("--out", default=str(ROOT / "results" / "trees_xgbtop_1600"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out == (ROOT / "results" / "spxw_pnl").resolve():
        raise SystemExit(
            "the xgbtop tables are staged outside results/spxw_pnl; pass another --out"
        )
    problems: list[str] = []
    if a.part in ("tuned", "all"):
        problems += tuned(out)
    if a.part in ("optuna", "all"):
        problems += optuna(out)
    print(f"problems: {problems or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
