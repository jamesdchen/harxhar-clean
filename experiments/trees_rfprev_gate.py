"""Cross-cluster check of the dev-forest PREVIEW run on Hoffman2 (2026-10-07).

experiments/trees_momvix_gate.py (unedited, imported) with its untuned rungs pointed at the wrapper
specs/causal_tune_trees_rfdev.py and its stored root at the CARC rfdev run
(results/linear_subsection_trees_mask_rfdev, relayed into ./ref): stored rfdev cells (the rf_opt_00
configuration, cluster/rfdev_rf_config.json, exported as HPC_KW_RF_CONFIG) re-run on this machine's CPU
and compared with the CARC epyc-7513 chunks.  Bit-identity is not required for the preview: the gate
records, per key, whether the arrays are identical and, if not, how far apart (max relative difference),
in gate_rows.csv / gate_lines.csv.

Run:  HPC_KW_RF_CONFIG=cluster/rfdev_rf_config.json python experiments/trees_rfprev_gate.py
          --lines cluster/trees_rfprev_gate_lines.txt --ref-root ref --out <dir>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trees_momvix_gate as g  # noqa: E402

WRAPPER = "specs/causal_tune_trees_rfdev.py"
STORED = "linear_subsection_trees_mask_rfdev"


def main() -> int:
    if not os.environ.get("HPC_KW_RF_CONFIG"):
        print("HPC_KW_RF_CONFIG must name the rfdev configuration")
        return 1
    for rung in ("t1", "t10"):
        _spec, re_, ie, shapseg, _stored = g.RUNGS[rung]
        g.RUNGS[rung] = (WRAPPER, re_, ie, shapseg, STORED)
    for rung in ("rs1", "rs10"):
        del g.RUNGS[rung]
    return g.main()


if __name__ == "__main__":
    sys.exit(main())
