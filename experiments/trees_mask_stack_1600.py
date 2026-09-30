"""Stack the MASKED 16:00 rungs (checklist I9, WINDOW_MASK=1) into the canonical forecast tables.

The unedited stackers, pointed at the masked roots:

  experiments/trees_cadence_stack_1600.py (itself running build_subsection_tree_yhat.py and
  build_subsection_tree_tuned_yhat.py unchanged, bar list narrowed to bar1600) with its
  module roots U / T / STAGE replaced:

  rung  root                                                -> results/spxw_pnl/ (overwritten)
  T10   results/linear_subsection_trees_mask/t10            yhat_subtree_<model>_<bucket>
  T1    results/linear_subsection_trees_mask/t1             yhat_subtree_daily_<bucket>_<model>
  RS10  results/linear_subsection_trees_tuned_mask/rs10     yhat_subtree_tuned[q]_<bucket>_<model>
  RS1   results/linear_subsection_trees_tuned_mask/rs1      yhat_subtree_tuned[q]_daily_<bucket>_<model>

  experiments/build_subsection_lstm_yhat.py --root results/linear_subsection_lstm_mask
                                                            yhat_lstm[_qsel]_<bucket>

The unmasked (de-duplicated, no mask) versions of these tables are snapshotted in
results/spxw_pnl/dedup_nomask_2026-09-29/ (local); experiments/trees_mask_ladder_1600.py
compares the two.  Staging: results/trees_mask_1600/stack/.

Run:  python experiments/trees_mask_stack_1600.py [--out results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_lstm_yhat as bly  # noqa: E402
import trees_cadence_stack_1600 as tcs  # noqa: E402

U = ROOT / "results" / "linear_subsection_trees_mask"
T = ROOT / "results" / "linear_subsection_trees_tuned_mask"
L = ROOT / "results" / "linear_subsection_lstm_mask"
STAGE = ROOT / "results" / "trees_mask_1600" / "stack"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "spxw_pnl"))
    a = ap.parse_args()
    tcs.U, tcs.T, tcs.STAGE = (
        U,
        T,
        STAGE,
    )  # the masked roots; the stacker reads them at call time
    saved = sys.argv
    try:
        sys.argv = [tcs.__file__, "--out", a.out]
        rc_trees = tcs.main()
        sys.argv = [bly.__file__, "--root", str(L), "--out", a.out]
        rc_lstm = 0
        try:
            bly.main()
        except SystemExit as e:
            rc_lstm = 0 if e.code in (None, 0) else 1
            print(f"LSTM stacker: {e.code}")
    finally:
        sys.argv = saved
    print(f"trees stacker exit {rc_trees}, LSTM stacker exit {rc_lstm}")
    return 1 if (rc_trees or rc_lstm) else 0


if __name__ == "__main__":
    sys.exit(main())
