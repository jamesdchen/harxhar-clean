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

  experiments/build_subsection_lstm_yhat.py
  LSTM    results/linear_subsection_lstm_mask (REFIT_EVERY 1, the daily refit; user decision
          2026-09-29 evening)                               yhat_lstm[_qsel]_<bucket> (overwritten)
  LSTM10  results/linear_subsection_lstm_mask_re10 (REFIT_EVERY 10, the earlier cadence) ->
          results/trees_mask_1600/stack/lstm_re10/yhat_lstm[_qsel]_<bucket> (NOT a canonical
          table: the cadence comparison of experiments/trees_mask_ladder_1600.py)

The unmasked (de-duplicated, no mask) versions of the canonical tables are snapshotted in
results/spxw_pnl/dedup_nomask_2026-09-29/ (local); experiments/trees_mask_ladder_1600.py
compares the two.  Staging: results/trees_mask_1600/stack/.

Run:  python experiments/trees_mask_stack_1600.py [--out results/spxw_pnl] [--parts trees,lstm]
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
L1 = ROOT / "results" / "linear_subsection_lstm_mask"
L10 = ROOT / "results" / "linear_subsection_lstm_mask_re10"
STAGE = ROOT / "results" / "trees_mask_1600" / "stack"


def run_lstm(root: Path, out: str) -> int:
    saved = sys.argv
    try:
        sys.argv = [bly.__file__, "--root", str(root), "--out", out]
        bly.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"LSTM stacker ({root.name}): {e.code}")
            return 1
    finally:
        sys.argv = saved
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "spxw_pnl"))
    ap.add_argument("--parts", default="trees,lstm")
    a = ap.parse_args()
    parts = {x.strip() for x in a.parts.split(",")}
    rc = 0
    if "trees" in parts:
        # the masked roots; the stacker reads its module globals at call time
        tcs.U, tcs.T, tcs.STAGE = U, T, STAGE
        saved = sys.argv
        try:
            sys.argv = [tcs.__file__, "--out", a.out]
            rc |= tcs.main()
        finally:
            sys.argv = saved
    if "lstm" in parts:
        rc |= run_lstm(L1, a.out)
        stage10 = STAGE / "lstm_re10"
        stage10.mkdir(parents=True, exist_ok=True)
        rc |= run_lstm(L10, str(stage10))
    print(f"stackers exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
