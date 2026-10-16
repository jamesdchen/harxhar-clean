"""Stack the moments_vol_vix 16:00 tree rungs into canonical forecast tables (2026-10-06, user request).

The stacker of the masked campaign, unedited (experiments/trees_mask_stack_1600.py ->
experiments/trees_cadence_stack_1600.py, which runs build_subsection_tree_yhat.py and
build_subsection_tree_tuned_yhat.py with every gate of theirs: profile B against the production table
to 1e-9, every stamp in it, the production rv_raw carried; the tuned tables' stamps / B / rv_raw
against the untuned table of the same model, bucket and refit cadence), with its module roots pointed
at the moments_vol_vix roots and the builders' bucket list narrowed to moments_vol_vix and lean_vol_vix (both read at call
time, as trees_mask_stack_1600.py narrows the bar list):

  rung  root                                                        -> results/spxw_pnl/
  T10   results/linear_subsection_trees_mask_momvolvix/t10             yhat_subtree_<model>_<bucket>
  T1    results/linear_subsection_trees_mask_momvolvix/t1              yhat_subtree_daily_<bucket>_<model>
  RS10  results/linear_subsection_trees_tuned_mask_momvolvix/rs10      yhat_subtree_tuned[q]_<bucket>_<model>
  RS1   results/linear_subsection_trees_tuned_mask_momvolvix/rs1       yhat_subtree_tuned[q]_daily_<bucket>_<model>

Only moments_vol_vix and lean_vol_vix tables are written (<bucket> in the names below) (no table of another bucket is touched).  Staging:
results/trees_momvolvix_1600/stack/.  Run only once results/close_master_table/PLAIN_DONE exists
(the orchestrator's rule; checked).

Run:  python experiments/trees_momvolvix_stack_1600.py                       (lean_vol_vix -> results/spxw_pnl)
      python experiments/trees_momvolvix_stack_1600.py --buckets moments_vol_vix --out results/spxw_pnl/parked_moments_vol_vix_2026-10-06
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

BUCKETS = (
    "moments_vol_vix",
    "lean_vol_vix",
)  # both input sets of the run (user, 2026-10-06)
# user decision 2026-10-06 ("comment out the non lean results everywhere"): moments_vol_vix is dropped --
# its tables never go into results/spxw_pnl/, only into the parked folder
DROPPED = ("moments_vol_vix",)
PARKED = ROOT / "results" / "spxw_pnl" / "parked_moments_vol_vix_2026-10-06"
U = ROOT / "results" / "linear_subsection_trees_mask_momvolvix"
T = ROOT / "results" / "linear_subsection_trees_tuned_mask_momvolvix"
STAGE = ROOT / "results" / "trees_momvolvix_1600" / "stack"
PLAIN_DONE = ROOT / "results" / "close_master_table" / "PLAIN_DONE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "spxw_pnl"))
    ap.add_argument(
        "--buckets", default="lean_vol_vix", help=f"comma list of {BUCKETS}"
    )
    a = ap.parse_args()
    buckets = tuple(b for b in a.buckets.split(",") if b)
    assert set(buckets) <= set(BUCKETS), buckets
    if Path(a.out).resolve() == (ROOT / "results" / "spxw_pnl").resolve() and set(
        buckets
    ) & set(DROPPED):
        print(
            f"{sorted(set(buckets) & set(DROPPED))} dropped (user, 2026-10-06): write those to {PARKED}, not results/spxw_pnl"
        )
        return 1
    if (
        Path(a.out).resolve() == (ROOT / "results" / "spxw_pnl").resolve()
        and not PLAIN_DONE.is_file()
    ):
        print(
            f"{PLAIN_DONE} missing: the canonical tables are written only after the plain master table"
        )
        return 1
    bst.BUCKETS = buckets  # type: ignore[assignment]  # the requested buckets only
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
