"""Stack a shipped-forest Optuna rung (random forest; rfp163 = the trial-163 forest) with the canonical Optuna stacker.

experiments/build_subsection_tree_optuna_yhat.py reads every merged path of an Optuna root
(<root>/paths/<path>/<bucket>/bar1600/<model>/tw2000/causal_tune_trees/<model>/<bucket>/results_bar1600.csv) with the
linear stacker's read_arm / with_production_rv and its gates (profile B against the production table to 1e-9, every
stamp in it, the production rv_raw carried; every path of an arm with the arm's tp1 stamps, B and rv_raw).  This
wrapper runs it UNCHANGED with its module lists narrowed to the random forest and this run's three input sets, and
the tables renamed with the key token (--token, default rfp163) as the model token instead of rf, so no existing
table name is reused:

  yhat_subtree_optuna_tp<N>_<bucket>_<token>.parquet        N = 1, 5, 25, 250: best of 50 (validation MSE)
  yhat_subtree_optuna_tp<N>_k<k>_<bucket>_<token>.parquet   best of the first k = 10 / 25 trials at N = 1 and 25
  yhat_subtree_optunaq_tp25_<bucket>_<token>.parquet        best of 50 by validation QLIKE, N = 25

e.g. yhat_subtree_optuna_tp1_baseline_rfp163.parquet, from the merged root
results/linear_subsection_trees_optuna_mask_<token> (pulled from CARC), written to results/trees_rfdev_preview163/
for rfp163 and results/trees_<token>_optuna_stack/ for another token (NOT results/spxw_pnl; the main session decides
integration).

Usage: python experiments/trees_rfp163_stack_optuna.py [--token rfp163] [--root <merged root>] [--out <dir>]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_optuna_yhat as bso  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402

BUCKETS = ("baseline", "all_features", "lean_vol_vix")
TOKEN = "rfp163"  # the default key token: the model token of the table names (the arms' model directory stays rf)


def default_out(token: str) -> str:
    return (
        "results/trees_rfdev_preview163"
        if token == "rfp163"
        else f"results/trees_{token}_optuna_stack"
    )


def pop_arg(name: str, default: str) -> str:
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i : i + 2]
        return val
    return default


if __name__ == "__main__":
    token = pop_arg("--token", TOKEN)
    root = pop_arg("--root", f"results/linear_subsection_trees_optuna_mask_{token}")
    out = pop_arg("--out", default_out(token))
    if bst.resolve(out).resolve() == (ROOT / "results" / "spxw_pnl").resolve():
        raise SystemExit(
            "the shipped-forest Optuna tables are staged outside results/spxw_pnl; pass another --out"
        )

    def table_name(path: str, bucket: str, model: str) -> str:
        assert model == "rf", model
        return f"yhat_{bso.TABLES[path]}_{bucket}_{token}.parquet"

    bst.MODELS = ("rf",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    bso.table_name = table_name  # type: ignore[assignment]
    sys.argv += ["--root", root, "--out", out]
    bso.main()
