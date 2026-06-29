"""Legible-coefficient TRAJECTORY probe — the 0DTE/OPEX-dilution test (explainability audit #1).

The walk-forward refits the linear base ~407 times, so the close-damping coefficient (har_ma_5 × close,
the −0.05 sign-flip) is a TIME SERIES we have always collapsed to a single number. Dump the trajectory of
the HAR×{close,open} interaction coefs + the cumrv×close coef across cadence blocks: does the close-damping
WEAKEN over the sample, as the 0DTE/OPEX-dilution hypothesis (intraday_regime_findings §12) predicts?
Writes results/moe_ladder/coef_trajectory.csv (read by edge_06_explainability_audit.ipynb).

Caveat: row→date alignment is unresolved (intraday_regime_findings §10), so we index by cadence block
(monotone in time); the trajectory SHAPE is the dilution signal regardless of exact dates.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

CID = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim"
)

import resid_amortized as ra  # noqa: E402

c = ra.load_cache(CID)
feats = c["feats"]
coefs = np.asarray(c["coefs"])  # [n_blocks, p]
starts = np.asarray(c["starts"])

# HAR×{close,open} interactions + cumrv terms (exclude exog like voldemand_*_open_and_close: no "har")
keys = [
    f
    for f in feats
    if (("har" in f.lower() and ("close" in f.lower() or "open" in f.lower())) or "cumrv" in f.lower())
]
idx = {f: feats.index(f) for f in keys}
print(f"blocks={coefs.shape[0]} p={coefs.shape[1]} tracked_cols={len(keys)}", flush=True)
for f in keys:
    col = coefs[:, idx[f]]
    print(
        f"  {f:30s} mean={col.mean():+.4f}  first10={col[:10].mean():+.4f}  "
        f"last10={col[-10:].mean():+.4f}  |Δ|={abs(col[-10:].mean() - col[:10].mean()):.4f}",
        flush=True,
    )

os.makedirs("results/moe_ladder", exist_ok=True)
out = "results/moe_ladder/coef_trajectory.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["block", "row"] + keys)
    for i in range(coefs.shape[0]):
        w.writerow([i, int(starts[i])] + [f"{coefs[i, idx[f]]:.6f}" for f in keys])
print(f"WROTE {out}", flush=True)
print("COEF_TRAJ_DONE", flush=True)
