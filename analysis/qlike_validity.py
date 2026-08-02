"""Is optimising QLIKE a valid thing to do?

Two questions hide inside that, and they have different answers.

CONSISTENCY. Given that QLIKE is the reported metric, selecting under QLIKE is
the only coherent choice: mixing criteria costs +0.00157 here with no selection
bias involved (analysis/selection_criterion.py). And QLIKE is a defensible
metric on theory -- it is one of the loss families that rank variance forecasts
consistently when the target is a noisy but conditionally unbiased proxy for
the true conditional variance \\citep{Patton2011}, and it is minimised at the
truth. Evaluation here is against RAW RV, not the winsorised fitting target, so
that guarantee applies to the ranking even though the winsorisation distorts
the estimator.

NEUTRALITY. That is a different claim. Across the 45-arm battery QLIKE
correlates +0.741 with the Mincer-Zarnowitz slope (p = 5.8e-9) -- lower QLIKE
goes with a lower slope -- and it ranks the battery almost independently of raw
variance MSE (+0.152, p = 0.32, where agreement would give a NEGATIVE
correlation).

A LOSS-GEOMETRY EXPLANATION OF THAT CORRELATION WAS PROPOSED AND IS REFUTED.
The proposal: with r = true/pred, QLIKE is r - log r - 1, so under-forecasting
costs r without bound while over-forecasting costs only -log r; shrinking
toward the mean should therefore pay. Tested directly in
analysis/kernel_ceiling.py by rescaling a fitted forecast, p_c = m + c (p - m),
with nothing else moving: QLIKE is minimised at c = 1.00 and MSE at c = 0.95.
MSE wants the extra attenuation; QLIKE does not.

The algebra says the same thing once done properly. Under a scalar rescaling
QLIKE's optimum is c* = E[true/pred] -- calibration in the RATIO -- which a
Duan-smeared fit already satisfies, so c* = 1. MSE's optimum is
c* = Cov(true, p)/Var(p), the MZ slope itself, which is 0.968 here, so MSE
shrinks. Attenuation is rewarded by MSE, not by QLIKE.

So the battery's +0.741 is an empirical fact that stands on its own p-value,
and its cause is NOT the loss's asymmetry. It remains unexplained.

That matters for the result we just corrected. Selecting the MIDAS shape by
validation QLIKE picks a LONG-memory kernel (theta2 = 117) where training SSE
picks a short one (37.5). Long memory means a smoother, more attenuated
forecast. So the honest 0.00155 win over the b2 ladder might be accuracy, or
might be QLIKE paying for attenuation.

MEASURED ANSWER (see the JSON): neither, quite. Inside this one-parameter
family the attenuation story does NOT reproduce -- rank corr(test QLIKE, MZ
slope) is -0.790, so here lower QLIKE goes with BETTER calibration, the
opposite sign to the battery-level finding. What does show up is metric
specificity: validation QLIKE and validation MSE want different kernels
(theta2 = 122 versus 295), the ladder has the LOWEST raw-variance MSE of the
three arms, and the MIDAS win of -0.00159 (t = -2.64) under QLIKE becomes
+7.8e-14 (t = +0.12, p = 0.90) under MSE. The win is real under the reported
metric and absent under the other robust one.

This measures:

  1. the whole theta2 path, with QLIKE, raw-variance MSE and the MZ slope
     side by side, so the two optima can be seen rather than assumed
  2. theta selected by validation QLIKE versus by validation MSE, each scored
     under both metrics
  3. whether the MIDAS-versus-ladder verdict survives a change of metric,
     with a matched-row Diebold-Mariano test under each
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
from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
TH1 = 1e-6                      # both searches drove theta1 to its floor
TH2_GRID = np.exp(np.linspace(np.log(5.0), np.log(600.0), 28))


def fit_predict(F, y, bl, fit_rows, score_rows):
    A = np.hstack([np.ones((len(fit_rows), 1)), F[fit_rows]])
    cf, *_ = np.linalg.lstsq(A, y[fit_rows], rcond=None)
    r = y[fit_rows] - A @ cf
    sig2 = float(r @ r) / len(fit_rows)
    Av = np.hstack([np.ones((len(score_rows), 1)), F[score_rows]])
    return ((Av @ cf) ** 2 + sig2) * bl[score_rows]


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def mz_slope(rv, p):
    """OLS slope of true on predicted; 1.0 is calibrated, < 1 is attenuated."""
    x = p - p.mean()
    return float((x @ (rv - rv.mean())) / max(x @ x, 1e-300))


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
    vcut = int(cut * 0.75)
    FIT, VAL, TRN, TST = (np.arange(vcut), np.arange(vcut, cut),
                          np.arange(cut), np.arange(cut, len(idx)))
    rv_v, rv_t = rvv[VAL], rvv[TST]
    print(f"panel {len(idx):,}   fit {len(FIT):,}   valid {len(VAL):,}   "
          f"test {len(TST):,}  ({time.time()-t0:.0f}s)")
    out = {}

    # ------------------------------------------------ 1. the whole path
    print("\n" + "=" * 78)
    print("1. THE THETA2 PATH UNDER TWO METRICS")
    print("=" * 78)
    print(f"  {'theta2':>9}{'valid QLIKE':>14}{'valid MSE':>14}"
          f"{'test QLIKE':>13}{'test MSE':>13}{'test MZ beta':>14}")
    rows = []
    for th2 in TH2_GRID:
        B, _ = w_midas(TH1, th2)
        F = featurize(ysafe, B)[idx]
        pv = fit_predict(F, yv, blv, FIT, VAL)
        pt = fit_predict(F, yv, blv, TRN, TST)
        del F
        rec = {"theta2": float(th2),
               "valid_qlike": float(np.mean(qlike_bar(rv_v, pv))),
               "valid_mse": float(np.mean((rv_v - pv) ** 2)),
               "test_qlike": float(np.mean(qlike_bar(rv_t, pt))),
               "test_mse": float(np.mean((rv_t - pt) ** 2)),
               "test_mz": mz_slope(rv_t, pt)}
        rows.append(rec)
        print(f"  {th2:>9.1f}{rec['valid_qlike']:>14.5f}"
              f"{rec['valid_mse']:>14.4e}{rec['test_qlike']:>13.5f}"
              f"{rec['test_mse']:>13.4e}{rec['test_mz']:>14.3f}")
    out["path"] = rows

    iq = int(np.argmin([r["valid_qlike"] for r in rows]))
    im = int(np.argmin([r["valid_mse"] for r in rows]))
    print(f"\n  validation QLIKE optimum at theta2 = {rows[iq]['theta2']:.1f}"
          f"   (MZ beta {rows[iq]['test_mz']:.3f})")
    print(f"  validation MSE   optimum at theta2 = {rows[im]['theta2']:.1f}"
          f"   (MZ beta {rows[im]['test_mz']:.3f})")
    print("  -> " + ("the two metrics want DIFFERENT kernels"
                     if iq != im else
                     "the two metrics want the SAME kernel"))
    out["optima"] = {"by_valid_qlike": rows[iq], "by_valid_mse": rows[im],
                     "same_kernel": bool(iq == im)}

    mzs = np.array([r["test_mz"] for r in rows])
    tq = np.array([r["test_qlike"] for r in rows])
    rho = float(np.corrcoef(np.argsort(np.argsort(tq)),
                            np.argsort(np.argsort(mzs)))[0, 1])
    print(f"\n  along this path, rank corr(test QLIKE, MZ beta) = {rho:+.3f}")
    print("  (positive means lower QLIKE goes with a LOWER slope, i.e. QLIKE "
          "prefers\n   the more attenuated kernel -- the battery-level "
          "finding, reproduced\n   inside a single one-parameter family)")
    out["qlike_vs_mz_rank_corr"] = rho

    # ------------------------------------------------ 2/3. does the verdict hold?
    print("\n" + "=" * 78)
    print("2. DOES THE MIDAS-VERSUS-LADDER VERDICT SURVIVE THE METRIC?")
    print("=" * 78)
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    px = fit_predict(Fx, yv, blv, TRN, TST)
    del Fx

    def arm(th2):
        B, _ = w_midas(TH1, th2)
        F = featurize(ysafe, B)[idx]
        p = fit_predict(F, yv, blv, TRN, TST)
        del F
        return p

    pq = arm(rows[iq]["theta2"])
    pm = arm(rows[im]["theta2"])
    print(f"  {'arm':<34}{'QLIKE':>11}{'MSE':>13}{'MZ beta':>10}")
    res = {}
    for nm, p in (("boxcar b2 ladder, K=12", px),
                  ("MIDAS, theta by valid QLIKE", pq),
                  ("MIDAS, theta by valid MSE", pm)):
        q = float(np.mean(qlike_bar(rv_t, p)))
        ms = float(np.mean((rv_t - p) ** 2))
        bz = mz_slope(rv_t, p)
        print(f"  {nm:<34}{q:>11.5f}{ms:>13.4e}{bz:>10.3f}")
        res[nm] = {"qlike": q, "mse": ms, "mz_beta": bz}
    out["arms"] = res

    print(f"\n  matched-row DM, MIDAS(valid QLIKE) versus the ladder:")
    dq = dm_test(qlike_bar(rv_t, pq), qlike_bar(rv_t, px), h=1)
    dm_ = dm_test((rv_t - pq) ** 2, (rv_t - px) ** 2, h=1)
    print(f"    under QLIKE  diff {dq['mean_diff']:+.5f}  t {dq['t']:+.2f}  "
          f"p {dq['p']:.3g}")
    print(f"    under MSE    diff {dm_['mean_diff']:+.4e}  t {dm_['t']:+.2f}  "
          f"p {dm_['p']:.3g}")
    agree = (dq["mean_diff"] < 0) == (dm_["mean_diff"] < 0)
    print("  -> " + ("the two metrics AGREE on the sign"
                     if agree else
                     "the two metrics DISAGREE on the sign: the win is "
                     "metric-specific"))
    out["dm"] = {"qlike": dq, "mse": dm_, "metrics_agree_on_sign": bool(agree)}

    with open(os.path.join(OUT_DIR, "qlike_validity.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote qlike_validity.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
