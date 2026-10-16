"""Stack the moments_vix 16:00 tree rungs into canonical forecast tables (2026-10-06, user request).

The stacker of the masked campaign, unedited (experiments/trees_mask_stack_1600.py ->
experiments/trees_cadence_stack_1600.py, which runs build_subsection_tree_yhat.py and
build_subsection_tree_tuned_yhat.py with every gate of theirs: profile B against the production table
to 1e-9, every stamp in it, the production rv_raw carried; the tuned tables' stamps / B / rv_raw
against the untuned table of the same model, bucket and refit cadence), with its module roots pointed
at the moments_vix roots and the builders' bucket list narrowed to moments_vix (both read at call
time, as trees_mask_stack_1600.py narrows the bar list):

  rung  root                                                        -> results/spxw_pnl/
  T10   results/linear_subsection_trees_mask_momvix/t10             yhat_subtree_<model>_moments_vix
  T1    results/linear_subsection_trees_mask_momvix/t1              yhat_subtree_daily_moments_vix_<model>
  RS10  results/linear_subsection_trees_tuned_mask_momvix/rs10      yhat_subtree_tuned[q]_moments_vix_<model>
  RS1   results/linear_subsection_trees_tuned_mask_momvix/rs1       yhat_subtree_tuned[q]_daily_moments_vix_<model>

Only moments_vix tables are written (no table of another bucket is touched).  Staging:
results/trees_momvix_1600/stack/.  Run only once results/close_master_table/PLAIN_DONE exists
(the orchestrator's rule; checked).

Run:  python experiments/trees_momvix_stack_1600.py [--out results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import trees_cadence_stack_1600 as tcs  # noqa: E402

BUCKET = "moments_vix"
U = ROOT / "results" / "linear_subsection_trees_mask_momvix"
T = ROOT / "results" / "linear_subsection_trees_tuned_mask_momvix"
STAGE = ROOT / "results" / "trees_momvix_1600" / "stack"
PLAIN_DONE = ROOT / "results" / "close_master_table" / "PLAIN_DONE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "spxw_pnl"))
    a = ap.parse_args()
    if (
        Path(a.out).resolve() == (ROOT / "results" / "spxw_pnl").resolve()
        and not PLAIN_DONE.is_file()
    ):
        print(
            f"{PLAIN_DONE} missing: the canonical tables are written only after the plain master table"
        )
        return 1
    bst.BUCKETS = (BUCKET,)  # type: ignore[assignment]  # this workstream's bucket only
    tcs.U, tcs.T, tcs.STAGE = U, T, STAGE
    saved = sys.argv
    try:
        sys.argv = [tcs.__file__, "--out", a.out]
        rc = tcs.main()
    finally:
        sys.argv = saved
    print(f"stacker exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
