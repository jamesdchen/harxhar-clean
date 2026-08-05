"""Which selection criterion is the honest one, and what does the wrong one cost?

An earlier fix replaced an oracle (shape parameters chosen on the reported
out-of-sample QLIKE) with training SSE. That is not the honest version of the
same criterion -- it is a DIFFERENT OBJECTIVE ON A DIFFERENT SCALE:

    reported metric   QLIKE on raw RV, after Duan smearing
    training SSE      squared error on the winsorised, diurnal-divided,
                      square-root target

So the "+0.00327 criterion cost" measured in greedy_vs_midas_diag.py is the sum
of two effects that were never separated -- train-versus-test, and
SSE-versus-QLIKE -- and the conclusion drawn from it ("the honestly selected
Beta kernel does not beat the ladder") inherits whatever the second one is
worth.

The honest criterion is neither. Carve a validation tail out of the TRAINING
portion, select on QLIKE there -- same loss as the report, no contact with the
test rows -- then refit on the full training portion and score once on test.

Four rules, identical downstream treatment, scored on identical test rows:

    A  fit-SSE       squared error on the adjusted target, in-sample
    B  fit-QLIKE     the reported loss, in-sample
    C  valid-QLIKE   the reported loss, on held-out validation   <- honest
    D  test-QLIKE    the reported loss, on the test rows         <- oracle

A - B isolates the objective mismatch. C - D isolates the remaining selection
optimism under a matched loss. Both the continuous two-parameter MIDAS search
and the 256-snapshot grid (the greedy basis's m=1 pick) are run under all four.
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

from basis_horserace import BURN, TRAIN_FRAC, featurize, w_boxcar, w_midas  # noqa: E402
from reduced_basis import beta_grid  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
VALID_FRAC = 0.25          # last quarter of the training portion


def fit_ols(F, y, rows):
    A = np.hstack([np.ones((len(rows), 1)), F[rows]])
    cf, *_ = np.linalg.lstsq(A, y[rows], rcond=None)
    r = y[rows] - A @ cf
    return cf, float(r @ r) / len(rows)


def predict_raw(F, cf, sig2, bl, rows):
    A = np.hstack([np.ones((len(rows), 1)), F[rows]])
    return (A @ cf) ** 2 + sig2, bl[rows]


def qlike(rv, pred):
    m = (rv > 0) & (pred > 0) & np.isfinite(rv) & np.isfinite(pred)
    r = rv[m] / pred[m]
    return float(np.mean(r - np.log(r) - 1.0))


def evaluate(F, y, bl, rv, fit_rows, score_rows):
    """Fit OLS on fit_rows, return QLIKE on score_rows and the fit SSE."""
    cf, sig2 = fit_ols(F, y, fit_rows)
    p, b = predict_raw(F, cf, sig2, bl, score_rows)
    return qlike(rv[score_rows], p * b), sig2 * len(fit_rows)


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
    yv, blv, rvv = y[idx], bl[idx], rv[idx]

    cut = int(len(idx) * TRAIN_FRAC)
    vcut = int(cut * (1.0 - VALID_FRAC))
    FIT = np.arange(vcut)                 # selection-time fit rows
    VAL = np.arange(vcut, cut)            # held-out validation tail
    TRN = np.arange(cut)                  # full training portion (final refit)
    TST = np.arange(cut, len(idx))
    print(f"panel {len(idx):,}   fit {len(FIT):,}   valid {len(VAL):,}   "
          f"train(refit) {len(TRN):,}   test {len(TST):,}  "
          f"({time.time()-t0:.0f}s)")
    out = {"n_fit": len(FIT), "n_valid": len(VAL), "n_train": len(TRN),
           "n_test": len(TST)}

    def crit(F, rule):
        """Selection score under one rule (lower is better)."""
        if rule == "A_fit_sse":
            return fit_ols(F, yv, FIT)[1] * len(FIT)
        if rule == "B_fit_qlike":
            return evaluate(F, yv, blv, rvv, FIT, FIT)[0]
        if rule == "C_valid_qlike":
            return evaluate(F, yv, blv, rvv, FIT, VAL)[0]
        if rule == "D_test_qlike":
            return evaluate(F, yv, blv, rvv, TRN, TST)[0]
        raise ValueError(rule)

    RULES = ["A_fit_sse", "B_fit_qlike", "C_valid_qlike", "D_test_qlike"]
    LABEL = {"A_fit_sse": "A  fit SSE (adjusted scale)",
             "B_fit_qlike": "B  fit QLIKE (in-sample)",
             "C_valid_qlike": "C  validation QLIKE  <- honest",
             "D_test_qlike": "D  test QLIKE        <- oracle"}

    # ---------------------------------------------------- the 256-point grid
    print("\n" + "=" * 76)
    print("1. THE SNAPSHOT GRID (the greedy basis's m = 1 pick)")
    print("=" * 76)
    Bs, thetas = beta_grid()
    scores = {r: np.full(len(thetas), np.inf) for r in RULES}
    for k in range(len(thetas)):
        F = featurize(ysafe, Bs[k:k + 1])[idx]
        for r in RULES:
            v = crit(F, r)
            scores[r][k] = v if np.isfinite(v) else np.inf
        del F
    print(f"  scored {len(thetas)} snapshots under 4 rules "
          f"({time.time()-t0:.0f}s)")

    print(f"\n  {'selection rule':<32}{'theta':>20}{'TEST QLIKE':>13}")
    grid = {}
    for r in RULES:
        j = int(np.argmin(scores[r]))
        F = featurize(ysafe, Bs[j:j + 1])[idx]
        q = evaluate(F, yv, blv, rvv, TRN, TST)[0]
        del F
        print(f"  {LABEL[r]:<32}"
              f"{f'({thetas[j][0]:.3f}, {thetas[j][1]:.3f})':>20}{q:>13.5f}")
        grid[r] = {"theta": list(thetas[j]), "test_qlike": float(q)}
    out["grid"] = grid

    # ---------------------------------------------------- continuous search
    print("\n" + "=" * 76)
    print("2. THE CONTINUOUS TWO-PARAMETER SEARCH")
    print("=" * 76)
    from scipy.optimize import minimize

    def make_obj(rule):
        def f(v):
            th = np.exp(v)
            B, _ = w_midas(th[0], th[1])
            F = featurize(ysafe, B)[idx]
            s = crit(F, rule)
            del F
            return s if np.isfinite(s) else 1e18
        return f

    print(f"\n  {'selection rule':<32}{'theta':>20}{'TEST QLIKE':>13}")
    cont = {}
    for r in RULES:
        res = minimize(make_obj(r), np.log([1.0, 3.0]), method="Nelder-Mead",
                       options={"maxiter": 90, "xatol": 1e-3, "fatol": 1e-6})
        th = np.exp(res.x)
        B, _ = w_midas(th[0], th[1])
        F = featurize(ysafe, B)[idx]
        q = evaluate(F, yv, blv, rvv, TRN, TST)[0]
        del F
        print(f"  {LABEL[r]:<32}{f'({th[0]:.3f}, {th[1]:.3f})':>20}"
              f"{q:>13.5f}")
        cont[r] = {"theta": th.tolist(), "test_qlike": float(q)}
    out["continuous"] = cont

    # ---------------------------------------------------- reference + decomposition
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    qx = evaluate(Fx, yv, blv, rvv, TRN, TST)[0]
    del Fx
    print(f"\n  reference: boxcar b2 ladder, K=12, OLS   TEST QLIKE {qx:.5f}")
    print("  (the ladder has no shape parameter, so no rule applies to it and")
    print("   its number is identical under all four)")
    out["boxcar_b2_test_qlike"] = float(qx)

    print("\n" + "=" * 76)
    print("DECOMPOSITION (continuous search)")
    print("=" * 76)
    A, B_, Cc, Dd = (cont[r]["test_qlike"] for r in RULES)
    print(f"  A  fit SSE                {A:.5f}")
    print(f"  B  fit QLIKE              {B_:.5f}")
    print(f"  C  validation QLIKE       {Cc:.5f}   <- the honest number")
    print(f"  D  test QLIKE (oracle)    {Dd:.5f}")
    print(f"\n  objective mismatch, A - B        {A - B_:+.5f}"
          f"   (SSE versus QLIKE, both in-sample)")
    print(f"  selection optimism, C - D        {Cc - Dd:+.5f}"
          f"   (honest versus oracle, matched loss)")
    print(f"  what the earlier fix charged     {A - Dd:+.5f}"
          f"   (all of it, to 'selection')")
    print(f"\n  honest MIDAS {Cc:.5f} versus ladder {qx:.5f}: "
          f"{Cc - qx:+.5f} -> "
          f"{'MIDAS WINS' if Cc < qx else 'ladder wins'}")
    out["decomposition"] = {
        "objective_mismatch_A_minus_B": float(A - B_),
        "selection_optimism_C_minus_D": float(Cc - Dd),
        "total_charged_by_earlier_fix_A_minus_D": float(A - Dd),
        "honest_midas": float(Cc), "boxcar": float(qx),
        "honest_delta_vs_ladder": float(Cc - qx),
        "honest_midas_beats_ladder": bool(Cc < qx)}

    with open(os.path.join(OUT_DIR, "selection_criterion.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote selection_criterion.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
