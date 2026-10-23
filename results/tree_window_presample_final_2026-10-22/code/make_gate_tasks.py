"""gate_tasks.txt for fold.sbatch: the earlier check's W = 1,000 fold-3 fits (three trees x two sets), refitted on
the earlier check's caches with this run's code and driver; LightGBM full set is the gate, the rest corroborate."""

import sys

lines = [f"1000 3 {e} {s}" for e in ("lgbm", "xgb", "rf") for s in ("all_features", "baseline")]
open(sys.argv[1], "w", newline="\n").write("\n".join(lines) + "\n")
print(len(lines))
