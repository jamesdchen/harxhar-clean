"""Stack the moments_vix Optuna forecasts with the canonical Optuna stacker, unchanged.

experiments/build_subsection_tree_optuna_yhat.py loops over build_subsection_tree_yhat.BUCKETS
(all_features, baseline); this wrapper sets that tuple to ("moments_vix",) for this process only
and runs the stacker's own main() -- the same reader, gates (B vs the production table to 1e-9,
every stamp in it, production rv_raw carried; every path of an arm with the arm's tp1 stamps, B
and rv_raw) and table names (yhat_subtree_optuna_tp<N>[_k<k>]_moments_vix_<model>.parquet,
yhat_subtree_optunaq_tp25_moments_vix_<model>.parquet).

Usage: python experiments/momvix_stack_optuna_yhat.py [--bucket moments_vix]
           [--root results/linear_subsection_trees_optuna_mask_momvix] [--out results/spxw_pnl]
       (--bucket moments_vol_vix: root results/linear_subsection_trees_optuna_mask_momvolvix)
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

ROOTS = {
    "moments_vix": "results/linear_subsection_trees_optuna_mask_momvix",
    "moments_vol_vix": "results/linear_subsection_trees_optuna_mask_momvolvix",
    "lean_vol_vix": "results/linear_subsection_trees_optuna_mask_leanvolvix",
}

if __name__ == "__main__":
    bucket = "moments_vix"
    if "--bucket" in sys.argv:
        i = sys.argv.index("--bucket")
        bucket = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    bst.BUCKETS = (bucket,)
    if "--root" not in sys.argv:
        sys.argv += ["--root", ROOTS[bucket]]
    bso.main()
