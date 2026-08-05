"""Why the greedy reduced basis does not reproduce the MIDAS fit.

Both objects are ONE-DIMENSIONAL linear spans of a single Beta kernel drawn
from the same manifold. They differ only in which theta is chosen, and they
were chosen by different rules:

  greedy m=1   picks the snapshot minimising TRAINING SSE on the adjusted
               target, over a 16x16 grid
  "nonlinear   picks theta by Nelder-Mead minimising the OUT-OF-SAMPLE QLIKE
   ceiling"    that is then reported, over the continuum

The second rule optimises the number being reported. If that is the whole
explanation, then (a) the reported nonlinear ceiling is an oracle figure rather
than an out-of-sample one, and (b) the reduced-basis verdict built on it -- "no
linear space drawn from the manifold reaches the nonlinear fit" -- is an
artifact of the criterion mismatch, since the nonlinear fit IS a linear space
drawn from the manifold.

This measures the decomposition directly:

  1. Every one of the 256 snapshots scored BOTH ways, so the criterion
     mismatch can be seen as a rank correlation rather than assumed.
  2. The train-SSE-best snapshot versus the OOS-QLIKE-best snapshot: same
     family, same grid, different rule.
  3. An HONEST nonlinear fit -- Nelder-Mead on the same continuum but with a
     TRAINING objective -- which is what the ceiling should have been compared
     against all along.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basis_horserace import (  # noqa: E402
    BURN, TRAIN_FRAC, best_over_lambda, featurize, fit_eval, w_boxcar, w_midas,
)
from reduced_basis import beta_grid  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")


def train_sse(F, y, cut):
    """SSE of the OLS fit on training rows only -- the greedy criterion."""
    A = np.hstack([np.ones((cut, 1)), F[:cut]])
    cf, *_ = np.linalg.lstsq(A, y[:cut], rcond=None)
    r = y[:cut] - A @ cf
    return float(r @ r)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.data.loading import load_raw_data
    from src.features.transforms import target as T
    t0 = time.time()

    df = load_raw_data("data", allow_missing=True)
    T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = 0.05, 0.95
    tgt, base = T.robust_transform(df, "RV", use_diurnal=True,
                                   allow_missing=True, is_target=True)
    y = tgt.to_numpy(np.float64)
    bl = base.to_numpy(np.float64)
    rv = df["RV"].to_numpy(np.float64)
    ok = np.isfinite(y) & np.isfinite(bl) & np.isfinite(rv) & (rv > 0)
    ok[:BURN] = False
    idx = np.flatnonzero(ok)
    ysafe = np.where(np.isfinite(y), y, 0.0)
    cut = int(len(idx) * TRAIN_FRAC)
    yv, blv, rvv = y[idx], bl[idx], rv[idx]
    print(f"panel {len(idx):,} rows, train {cut:,}  ({time.time()-t0:.0f}s)")
    out = {}

    Bs, thetas = beta_grid()
    print(f"scoring {len(thetas)} snapshots two ways ...", flush=True)
    sse = np.empty(len(thetas))
    qli = np.empty(len(thetas))
    for k in range(len(thetas)):
        F = featurize(ysafe, Bs[k:k + 1])[idx]
        sse[k] = train_sse(F, yv, cut)
        qli[k], _ = best_over_lambda(F, yv, blv, rvv, cut)
        del F
    print(f"  done ({time.time()-t0:.0f}s)")

    # ---- 1. do the two criteria even agree on a ranking?
    fin = np.isfinite(qli) & np.isfinite(sse)
    rs = np.argsort(np.argsort(sse[fin])).astype(float)
    rq = np.argsort(np.argsort(qli[fin])).astype(float)
    rho = float(np.corrcoef(rs, rq)[0, 1])
    print("\n" + "=" * 74)
    print("1. TRAIN SSE VERSUS OOS QLIKE AS SELECTION CRITERIA")
    print("=" * 74)
    print(f"  Spearman rank correlation over {int(fin.sum())} snapshots: "
          f"{rho:+.3f}")

    i_sse = int(np.argmin(np.where(fin, sse, np.inf)))
    i_qli = int(np.argmin(np.where(fin, qli, np.inf)))
    print(f"\n  {'rule':<34}{'theta':>22}{'OOS QLIKE':>12}")
    print(f"  {'min TRAIN SSE (greedy m=1)':<34}"
          f"{f'({thetas[i_sse][0]:.3f}, {thetas[i_sse][1]:.3f})':>22}"
          f"{qli[i_sse]:>12.5f}")
    print(f"  {'min OOS QLIKE (oracle, same grid)':<34}"
          f"{f'({thetas[i_qli][0]:.3f}, {thetas[i_qli][1]:.3f})':>22}"
          f"{qli[i_qli]:>12.5f}")
    print(f"\n  criterion cost: {qli[i_sse] - qli[i_qli]:+.5f} QLIKE, from "
          f"nothing but the selection rule")
    out["criterion"] = {
        "spearman_sse_vs_qlike": rho,
        "train_sse_pick": {"theta": list(thetas[i_sse]),
                           "oos_qlike": float(qli[i_sse])},
        "oracle_qlike_pick": {"theta": list(thetas[i_qli]),
                              "oos_qlike": float(qli[i_qli])},
        "criterion_cost": float(qli[i_sse] - qli[i_qli])}

    # ---- 2. the honest nonlinear fit: same continuum, training objective
    print("\n" + "=" * 74)
    print("2. THE NONLINEAR FIT, SELECTED HONESTLY")
    print("=" * 74)
    from scipy.optimize import minimize

    def obj_train(v):
        th = np.exp(v)
        B, _ = w_midas(th[0], th[1])
        F = featurize(ysafe, B)[idx]
        s = train_sse(F, yv, cut)
        del F
        return s if np.isfinite(s) else 1e18

    def obj_oos(v):
        th = np.exp(v)
        B, _ = w_midas(th[0], th[1])
        F = featurize(ysafe, B)[idx]
        q, _ = best_over_lambda(F, yv, blv, rvv, cut)
        del F
        return q if np.isfinite(q) else 1e6

    rT = minimize(obj_train, np.log([1.0, 3.0]), method="Nelder-Mead",
                  options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-3})
    thT = np.exp(rT.x)
    BT, _ = w_midas(thT[0], thT[1])
    FT = featurize(ysafe, BT)[idx]
    qT, lamT = best_over_lambda(FT, yv, blv, rvv, cut)
    qT_ols, _, _ = fit_eval(FT, yv, blv, rvv, cut, 0.0)
    del FT

    rO = minimize(obj_oos, np.log([1.0, 3.0]), method="Nelder-Mead",
                  options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-7})
    thO = np.exp(rO.x)

    print(f"  {'selection rule':<34}{'theta':>22}{'OOS QLIKE':>12}")
    print(f"  {'theta by TRAINING SSE (honest)':<34}"
          f"{f'({thT[0]:.3f}, {thT[1]:.3f})':>22}{qT:>12.5f}")
    print(f"  {'theta by OOS QLIKE (as reported)':<34}"
          f"{f'({thO[0]:.3f}, {thO[1]:.3f})':>22}{rO.fun:>12.5f}")
    print(f"\n  the reported 'nonlinear ceiling' is optimistic by "
          f"{qT - rO.fun:+.5f}")
    print(f"  (lambda is still chosen out-of-sample in BOTH rows, so this "
          f"isolates theta alone;\n   the same theta at lambda = 0 scores "
          f"{qT_ols:.5f})")
    out["nonlinear"] = {
        "theta_by_train": thT.tolist(), "qlike_theta_by_train": float(qT),
        "lambda_theta_by_train": lamT, "ols_qlike_theta_by_train": float(qT_ols),
        "theta_by_oos": thO.tolist(), "qlike_theta_by_oos": float(rO.fun),
        "optimism": float(qT - rO.fun)}

    # ---- 3. what this does to the comparison
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    qx, _ = best_over_lambda(Fx, yv, blv, rvv, cut)
    del Fx
    print("\n" + "=" * 74)
    print("3. THE COMPARISON, RESTATED LIKE FOR LIKE")
    print("=" * 74)
    print(f"  incumbent boxcar b2 ladder, K=12          {qx:.5f}")
    print(f"  1-D MIDAS span, theta by training SSE     {qT:.5f}  "
          f"({qT - qx:+.5f})")
    print(f"  1-D MIDAS span, theta by OOS QLIKE        {rO.fun:.5f}  "
          f"({rO.fun - qx:+.5f})   <- oracle")
    honest_win = qT < qx
    print(f"\n  -> under an honest selection rule the single Beta kernel "
          f"{'STILL BEATS' if honest_win else 'DOES NOT BEAT'} the 12-rung "
          f"ladder.")
    if not honest_win:
        print("     The horse race's headline -- one nonlinear degree of "
              "freedom beating\n     twelve linear ones -- was a selection "
              "artifact, and the reduced-basis\n     verdict resting on it "
              "does not stand.")
    out["restated"] = {"boxcar_b2": float(qx), "midas_honest": float(qT),
                       "midas_oracle": float(rO.fun),
                       "honest_midas_beats_ladder": bool(honest_win)}

    with open(os.path.join(OUT_DIR, "greedy_vs_midas_diag.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote greedy_vs_midas_diag.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
