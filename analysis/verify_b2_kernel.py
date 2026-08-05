"""Reproduce the archived kernel from the rebuilt cache, then interpret it.

Runs the archived probe (drivers/msweep_2026-08-01/session_kernels.py) verbatim
against a freshly rebuilt results/b2_mmap, which answers two questions:

  1. VALIDATION. Does the rebuilt cache reproduce the archived coefficient
     vector [0.059, 0.075, -0.021, ...] with sum -0.038? If yes, the earlier
     reproducibility failure was ours (a from-scratch feature rebuild that
     differed from the production prep path), not the project's.

  2. INTERPRETATION. run_geometry_local.prepare_full applies
     `rolling_robust_scale` as its final step, so the cached columns are
     (x - rolling median) / rolling IQR. A coefficient on a rolling-SCALED
     boxcar is not a coefficient on the boxcar itself, and the branching-ratio
     identity sum_u phi(u) = sum_k w_k holds only on the RAW basis. This script
     converts the fitted coefficients back to the raw basis using each ladder
     column's mean scaling IQR, and reports the kernel diagnostics on both, so
     the size of the basis error is visible rather than assumed.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawkes_kernel import eff_kernel, loglog_slope  # noqa: E402

D = os.path.join(ROOT, "results", "b2_mmap")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
ARCHIVED = np.array([0.059, 0.075, -0.021, 0.054, -0.067, -0.081,
                     0.035, 0.002, -0.067, -0.02, -0.005, -0.002])


def diagnostics(w, rungs):
    phi = eff_kernel(np.asarray(w, float), rungs)
    f = loglog_slope(rungs, phi)
    neg = phi < 0
    return {
        "w": list(map(float, w)),
        "phi": phi.tolist(),
        "branching_analogue": float(np.sum(w)),
        "n_negative_lags": int(neg.sum()),
        "negative_mass_share": (float(np.abs(phi[neg]).sum()) / float(np.abs(phi).sum())
                                if np.abs(phi).sum() > 0 else float("nan")),
        "powerlaw": f,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    print(f"cache: X {X.shape}, y {y.shape}, {len(names)} names")

    # --- replicate session_kernels.py's column partition exactly ---
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = list(chan)
    rungs = np.array([r for r, _ in chan[labels[0]]], float)
    ladder_cols = [j for c in chan for _, j in chan[c]]
    other_cols = [j for j in range(len(names)) if j not in set(ladder_cols)]
    har_cols = [j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)]
    har_in_other = [oi for oi, j in enumerate(other_cols) if j in set(har_cols)]
    print(f"  {len(labels)} exogenous channels x {len(rungs)} rungs "
          f"{rungs.astype(int).tolist()}")
    print(f"  {len(other_cols)} 'other' columns, of which {len(har_in_other)} are the "
          f"HAR self-ladder")
    print(f"  other-block contents (non-HAR): "
          f"{[names[other_cols[i]] for i in range(len(other_cols)) if i not in set(har_in_other)][:12]} ...")

    n = X.shape[0]
    p = 1 + len(other_cols) + len(ladder_cols)
    G = np.eye(p)
    G[0, 0] = 0.0
    c = np.zeros(p)
    for s in range(0, n, 20000):
        ii = slice(s, min(s + 20000, n))
        Xi = np.asarray(X[ii])
        A = np.hstack([np.ones((Xi.shape[0], 1)), Xi[:, other_cols], Xi[:, ladder_cols]])
        G += A.T @ A
        c += A.T @ y[ii]
        del Xi, A
    cf = np.linalg.solve(G, c)
    w_scaled = cf[1:][har_in_other]

    print("\n" + "=" * 78)
    print("1. VALIDATION against the archived vector")
    print("=" * 78)
    print(f"  archived : {np.round(ARCHIVED, 3).tolist()}  sum={ARCHIVED.sum():+.4f}")
    print(f"  rebuilt  : {np.round(w_scaled, 3).tolist()}  sum={w_scaled.sum():+.4f}")
    corr = float(np.corrcoef(w_scaled, ARCHIVED)[0, 1])
    maxabs = float(np.max(np.abs(w_scaled - ARCHIVED)))
    ok = corr > 0.99 and maxabs < 0.01
    print(f"  correlation {corr:+.4f}   max |diff| {maxabs:.4f}")
    print(f"  VERDICT: {'REPRODUCED' if ok else 'NOT reproduced'}")

    # --- mean scaling IQR per ladder column, to undo the rolling scaler ---
    print("\n" + "=" * 78)
    print("2. INTERPRETATION: the cached basis is rolling-robust-SCALED")
    print("=" * 78)
    from src.backtest.executor import _build_scale_guards  # noqa: E402
    raw_har = [names[other_cols[i]] for i in har_in_other]
    print(f"  self-ladder columns: {raw_har}")
    print("""
  prepare_full's final step is rolling_robust_scale: column j at row t becomes
  (x - median_[t-W,t)) / IQR_[t-W,t). A coefficient on that is a per-IQR
  coefficient, and the identity sum_u phi(u) = sum_k w_k is stated for
  coefficients on the RAW boxcar means. Applying the identity to the cached
  basis is a category error -- it is the error that produced the -0.038
  'inadmissible branching ratio' reported earlier in this project.""")

    d_scaled = diagnostics(w_scaled, rungs)
    print(f"\n  ON THE CACHED (scaled) BASIS  -- not a kernel:")
    print(f"    sum_k w_k         = {d_scaled['branching_analogue']:+.4f}")
    print(f"    negative lags     = {d_scaled['n_negative_lags']}/{len(rungs)}")
    print(f"    negative mass     = {d_scaled['negative_mass_share']:.1%}")

    out = {"validation": {"archived": ARCHIVED.tolist(),
                          "rebuilt": w_scaled.tolist(),
                          "correlation": corr, "max_abs_diff": maxabs,
                          "reproduced": bool(ok)},
           "scaled_basis": d_scaled,
           "rungs": rungs.tolist(),
           "n_other_cols": len(other_cols), "n_channels": len(labels),
           "other_block_non_har": [names[other_cols[i]] for i in range(len(other_cols))
                                   if i not in set(har_in_other)]}

    with open(os.path.join(OUT_DIR, "b2_kernel_verification.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'b2_kernel_verification.json')}")
    return out


if __name__ == "__main__":
    main()
