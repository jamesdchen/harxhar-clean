"""tasks.txt for fold.sbatch: every (window, fittable fold, est, set), the costly arms first."""

import sys

S, PPD, FOLD_N, EMB = 44281, 13, 3250, 325  # S: design row of the first 2012-01-03 bar (w1000 nofomcrel run)
lines = []
for est in ("rf", "xgb", "lgbm"):
    for s in ("all_features", "baseline"):
        for W in (3000, 2500, 2000, 1500, 1000):
            Wr = W * PPD
            for f in range(4):
                if S - (4 * FOLD_N + EMB + Wr) + f * FOLD_N >= 0:
                    lines.append(f"{W} {f} {est} {s}")
open(sys.argv[1], "w", newline="\n").write("\n".join(lines) + "\n")
print(len(lines))
