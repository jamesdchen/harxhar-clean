"""Reproduction gate of the dev-tuned random forest run (user decision 2026-10-06).

experiments/trees_momvix_gate.py (unedited, imported) with its untuned rungs pointed at
specs/causal_tune_trees_rfdev.py instead of specs/causal_tune_trees.py: the stored masked random-forest
cells of cluster/trees_rfdev_gate_lines.txt are re-run through the WRAPPER with the shipped random
forest's configuration (RF_CONFIG = cluster/rfdev_shipped_rf.json, {"n_estimators": 100}) and compared
byte for byte with the masked campaign's chunk npz (forecasts, target, importance, TreeSHAP, kept
columns, meta params / versions).  It passes only if the wrapper path is the shipped spec on this CPU
class -- then the dev-tuned run differs from the shipped random forest by RF_PARAMS alone.

Run:  HPC_KW_RF_CONFIG=cluster/rfdev_shipped_rf.json python experiments/trees_rfdev_gate.py
          --lines cluster/trees_rfdev_gate_lines.txt --ref-root /scratch1/jc_905/harxhar-mask --out <dir>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trees_momvix_gate as g  # noqa: E402

WRAPPER = "specs/causal_tune_trees_rfdev.py"


def main() -> int:
    if not os.environ.get("HPC_KW_RF_CONFIG"):
        print("HPC_KW_RF_CONFIG must name the shipped random forest's configuration")
        return 1
    for rung in ("t1", "t10"):
        _spec, re_, ie, shapseg, stored = g.RUNGS[rung]
        g.RUNGS[rung] = (WRAPPER, re_, ie, shapseg, stored)
    for rung in ("rs1", "rs10"):
        del g.RUNGS[rung]  # the tuned rungs are not part of this run
    return g.main()


if __name__ == "__main__":
    sys.exit(main())
