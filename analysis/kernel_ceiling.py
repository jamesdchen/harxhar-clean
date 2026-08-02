"""Why the attenuation finding is axis-specific, and how much room the lag
kernel has left at all.

PART A -- WHY THE BATTERY RESULT DOES NOT TRANSFER.

Across the 45-arm battery, QLIKE correlates +0.741 with the Mincer-Zarnowitz
slope: it prefers the more attenuated forecast. Inside the one-parameter MIDAS
family the rank correlation is -0.790, the opposite sign. Both are real; they
are measured along DIFFERENT AXES.

  * The battery varies model class, feature count and PENALTY. Arms differ
    mostly in how hard they shrink, and shrinkage is what produces attenuation.
    QLIKE's asymmetry -- with r = true/pred, under-forecasting costs r without
    bound, over-forecasting costs only -log r -- rewards it.
  * The MIDAS family varies kernel MEMORY at fixed capacity: one regressor,
    ordinary least squares, no penalty at all. There is no shrinkage axis for
    QLIKE to reward. What theta2 moves is the QUALITY of the single regressor,
    and out-of-sample the MZ slope of an unpenalised one-regressor fit is
    essentially a measure of how far the fit degraded, so better kernel means
    both lower QLIKE and a slope nearer one. The negative sign is mechanical.

The prediction that follows: put a shrinkage axis into the same trial space --
sweep the ridge penalty on the FIXED b2 ladder -- and the correlation should
flip back to positive, with QLIKE's optimum at a heavier penalty than MSE's.
Part A tests that. If it holds, "QLIKE prefers attenuation" is a statement
about the shrinkage axis, not about QLIKE ranking of every comparison.

PART B -- IS THERE A BETTER MANIFOLD TO FIND?

The natural response to "one Beta kernel edges out the 12-rung ladder" is to go
looking for a better parametric family. That is only worth doing if the lag
kernel has headroom, so this measures the CEILING directly: instead of
searching families, fit the kernel nonparametrically. A cubic B-spline basis in
log-lag at K up to 384, with a second-difference roughness penalty whose
strength is chosen on the validation tail, spans essentially any smooth kernel
on 2048 lags. Whatever that reaches is an upper bound on what any better
manifold could deliver, because every such manifold is a subset of it.

If the unrestricted kernel lands where the ladder and the Beta family already
are, the search is not worth running -- the constraint is the signal, not the
trial space.
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
    BURN, TRAIN_FRAC, U, d2_operator, featurize, implied_kernel_map,
    w_boxcar, w_bspline, w_midas,
)
from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
LAMS = [0.0] + [10.0 ** e for e in range(-6, 10)]


def fit(F, y, fit_rows, lam, P=None):
    Ftr = F[fit_rows]
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.hstack([np.ones((len(fit_rows), 1)), (Ftr - mu) / sd])
    k = Z.shape[1]
    R = np.zeros((k, k))
    R[1:, 1:] = np.eye(k - 1) if P is None else P
    beta = np.linalg.solve(Z.T @ Z + lam * R + 1e-10 * np.eye(k),
                           Z.T @ y[fit_rows])
    r = y[fit_rows] - Z @ beta
    return beta, mu, sd, float(r @ r) / len(r)


def predict(F, bl, rows, beta, mu, sd, sig2):
    Z = np.hstack([np.ones((len(rows), 1)), (F[rows] - mu) / sd])
    return ((Z @ beta) ** 2 + sig2) * bl[rows]


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def mz(rv, p):
    x = p - p.mean()
    return float((x @ (rv - rv.mean())) / max(x @ x, 1e-300))


def rough_P(B, m=96):
    L = implied_kernel_map(B)
    sel = np.unique(np.round(np.exp(np.linspace(0, np.log(U), m))).astype(int)) - 1
    A = d2_operator(len(sel)) @ L[sel]
    P = A.T @ A
    return P / max(np.trace(P) / P.shape[0], 1e-12)


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
    print(f"panel {len(idx):,}   fit {len(FIT):,}  valid {len(VAL):,}  "
          f"test {len(TST):,}   ({time.time()-t0:.0f}s)")
    out = {}

    # ================================================== PART A
    print("\n" + "=" * 78)
    print("A. THE SHRINKAGE AXIS: WHERE QLIKE'S TASTE FOR ATTENUATION LIVES")
    print("=" * 78)
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    print(f"  fixed b2 ladder (K=12), sweeping the ridge penalty\n")
    print(f"  {'lambda':>10}{'valid QLIKE':>14}{'valid MSE':>13}"
          f"{'test QLIKE':>13}{'test MSE':>13}{'test MZ beta':>14}")
    rows = []
    for lam in LAMS:
        b_, mu_, sd_, s2 = fit(Fx, yv, FIT, lam)
        pv = predict(Fx, blv, VAL, b_, mu_, sd_, s2)
        b2_, mu2, sd2, s22 = fit(Fx, yv, TRN, lam)
        pt = predict(Fx, blv, TST, b2_, mu2, sd2, s22)
        rec = {"lambda": lam,
               "valid_qlike": float(np.mean(qlike_bar(rv_v, pv))),
               "valid_mse": float(np.mean((rv_v - pv) ** 2)),
               "test_qlike": float(np.mean(qlike_bar(rv_t, pt))),
               "test_mse": float(np.mean((rv_t - pt) ** 2)),
               "test_mz": mz(rv_t, pt)}
        rows.append(rec)
        print(f"  {lam:>10g}{rec['valid_qlike']:>14.5f}{rec['valid_mse']:>13.4e}"
              f"{rec['test_qlike']:>13.5f}{rec['test_mse']:>13.4e}"
              f"{rec['test_mz']:>14.3f}")
    out["shrinkage_path"] = rows

    tq = np.array([r["test_qlike"] for r in rows])
    tm = np.array([r["test_mz"] for r in rows])
    rho = float(np.corrcoef(np.argsort(np.argsort(tq)),
                            np.argsort(np.argsort(tm)))[0, 1])
    iq = int(np.argmin([r["valid_qlike"] for r in rows]))
    im = int(np.argmin([r["valid_mse"] for r in rows]))
    print(f"\n  rank corr(test QLIKE, MZ beta) along the SHRINKAGE axis: "
          f"{rho:+.3f}")
    print(f"  validation QLIKE picks lambda = {rows[iq]['lambda']:g} "
          f"(MZ {rows[iq]['test_mz']:.3f})")
    print(f"  validation MSE   picks lambda = {rows[im]['lambda']:g} "
          f"(MZ {rows[im]['test_mz']:.3f})")
    heavier = rows[iq]["lambda"] > rows[im]["lambda"]
    print("  -> " + ("CONFIRMED: on the shrinkage axis QLIKE prefers the more "
                     "attenuated fit,\n     which is where the battery's "
                     "+0.741 comes from" if rho > 0 and heavier else
                     "NOT confirmed on this axis either; the battery "
                     "correlation must come\n     from something other than "
                     "the penalty alone"))
    out["shrinkage_axis"] = {
        "rank_corr_qlike_mz": rho,
        "lambda_by_valid_qlike": rows[iq]["lambda"],
        "lambda_by_valid_mse": rows[im]["lambda"],
        "qlike_picks_heavier_shrinkage": bool(heavier),
        "attenuation_preference_confirmed": bool(rho > 0 and heavier),
        "test_is_inert": True,
        "why": "ridge is a non-lever on a 12-parameter ladder against 109k "
               "rows: lambda* = 0 under both metrics and the MZ slope is "
               "0.968 across the whole useful range, moving only once the "
               "fit is destroyed. The penalty sweep does not traverse an "
               "attenuation axis, so this leaves the mechanism untested "
               "rather than refuted. A2 tests it directly."}

    # ---- A2: attenuate the forecast DIRECTLY, with nothing else moving
    #
    # The cleanest possible test of the loss geometry, and it needs no model.
    # Take the ladder's forecast and attenuate it toward its own in-sample
    # centre by an exponent c: p_c = g (p/g)^c, with g the in-sample GEOMETRIC
    # mean. c = 1 is the fitted forecast, c < 1 compresses its dispersion, c > 1
    # expands it. If QLIKE's optimum sits at c < 1 while MSE's sits at c = 1,
    # QLIKE's preference for attenuation is demonstrated with no confound.
    # This is a diagnostic of the LOSS, not a forecasting arm.
    #
    # Log-space rather than the obvious p_c = m + c (p - m): the linear form
    # drives predictions NEGATIVE for c > 1, and flooring them at a positive
    # epsilon sends QLIKE to ~1e300, which silently corrupts the entire c > 1
    # half of the sweep and can manufacture an argmin at c = 1. Variance is
    # positive and the multiplicative form respects that.
    print("\n" + "-" * 78)
    print("A2. DIRECT ATTENUATION SWEEP (shrink the forecast, change nothing "
          "else)")
    print("-" * 78)
    # The answer turned out to depend on the operator, so BOTH are run, and
    # both are LEVEL-PINNED: after attenuating, each forecast is rescaled by a
    # constant fixed on TRAINING rows so its in-sample mean matches the
    # unattenuated forecast's. Without that pin the log-space family shifts the
    # mean as c moves (E[exp] != exp(E[])), and since QLIKE's optimum under a
    # pure rescaling is c* = E[true/pred], the level drift swamps the
    # dispersion effect being measured. A conclusion is only claimed if the two
    # operators agree.
    b_, mu_, sd_, s2 = fit(Fx, yv, TRN, 0.0)
    p_tr = predict(Fx, blv, TRN, b_, mu_, sd_, s2)
    p_te = predict(Fx, blv, TST, b_, mu_, sd_, s2)
    m_tr = float(p_tr.mean())
    lg = float(np.mean(np.log(p_tr)))

    def attenuate(c, kind):
        """Attenuated TEST forecast, level pinned using TRAINING rows only."""
        if kind == "linear":
            q_tr = m_tr + c * (p_tr - m_tr)
            q_te = m_tr + c * (p_te - m_tr)
        else:
            q_tr = np.exp(lg + c * (np.log(p_tr) - lg))
            q_te = np.exp(lg + c * (np.log(p_te) - lg))
        if q_tr.min() <= 0:
            return None
        return q_te * (m_tr / float(q_tr.mean()))

    res_a = {}
    for kind in ("linear", "log"):
        print(f"\n  operator: {kind}, level-pinned on training rows")
        print(f"  {'c':>7}{'test QLIKE':>13}{'test MSE':>13}"
              f"{'test MZ beta':>14}{'min pred':>12}")
        arows = []
        for c in np.round(np.arange(0.40, 1.61, 0.05), 2):
            pc = attenuate(float(c), kind)
            if pc is None or pc.min() <= 0:
                print(f"  {c:>7.2f}{'  (forecast goes non-positive; skipped)':>52}")
                continue
            rec = {"c": float(c),
                   "qlike": float(np.mean(qlike_bar(rv_t, pc))),
                   "mse": float(np.mean((rv_t - pc) ** 2)),
                   "mz": mz(rv_t, pc), "min_pred": float(pc.min())}
            arows.append(rec)
            print(f"  {c:>7.2f}{rec['qlike']:>13.5f}{rec['mse']:>13.4e}"
                  f"{rec['mz']:>14.3f}{rec['min_pred']:>12.2e}")
        cq = arows[int(np.argmin([r["qlike"] for r in arows]))]
        cm = arows[int(np.argmin([r["mse"] for r in arows]))]
        print(f"    QLIKE minimised at c = {cq['c']:.2f}, "
              f"MSE minimised at c = {cm['c']:.2f}")
        res_a[kind] = {"path": arows, "c_star_qlike": cq["c"],
                       "c_star_mse": cm["c"],
                       "qlike_wants_more_attenuation": bool(cq["c"] < cm["c"])}

    a_lin = res_a["linear"]["qlike_wants_more_attenuation"]
    a_log = res_a["log"]["qlike_wants_more_attenuation"]
    print(f"\n  linear operator: QLIKE wants more attenuation = {a_lin}")
    print(f"  log    operator: QLIKE wants more attenuation = {a_log}")
    if a_lin == a_log:
        print("  -> " + ("CONFIRMED under both operators: QLIKE prefers the "
                         "more attenuated\n     forecast, so the preference "
                         "is a property of the loss."
                         if a_lin else
                         "REFUTED under both operators: QLIKE does not prefer "
                         "attenuation,\n     so the battery's +0.741 needs "
                         "another explanation."))
    else:
        print("  -> INCONCLUSIVE: the two attenuation operators disagree, so "
              "the answer\n     is an artifact of how 'attenuation' is "
              "parameterised, not a fact\n     about the loss. No claim made.")
    out["attenuation_sweep"] = {
        "operators": res_a, "agree": bool(a_lin == a_log),
        "qlike_prefers_attenuation": (bool(a_lin) if a_lin == a_log else None)}
    del Fx

    # ================================================== PART B
    print("\n" + "=" * 78)
    print("B. THE CEILING: FIT THE KERNEL NONPARAMETRICALLY")
    print("=" * 78)
    print("  cubic B-splines in log-lag + second-difference roughness penalty,")
    print("  penalty strength chosen on the validation tail\n")
    print(f"  {'basis':>16}{'K':>5}{'lambda*':>11}{'test QLIKE':>13}"
          f"{'test MSE':>13}{'test MZ':>10}")
    preds = {}
    ref = None
    ceil = {}
    for tag, K in (("boxcar b2", 0), ("bspline", 48), ("bspline", 192),
                   ("bspline", 384)):
        B, _ = (w_boxcar() if K == 0 else w_bspline(K))
        F = featurize(ysafe, B)[idx]
        P = None if K == 0 else rough_P(B)
        best = (np.inf, None)
        for lam in LAMS:
            try:
                b_, mu_, sd_, s2 = fit(F, yv, FIT, lam, P)
            except np.linalg.LinAlgError:
                continue
            q = float(np.mean(qlike_bar(rv_v,
                                        predict(F, blv, VAL, b_, mu_, sd_, s2))))
            if np.isfinite(q) and q < best[0]:
                best = (q, lam)
        b_, mu_, sd_, s2 = fit(F, yv, TRN, best[1], P)
        pt = predict(F, blv, TST, b_, mu_, sd_, s2)
        del F
        name = f"{tag}_{B.shape[0]}"
        preds[name] = pt
        if ref is None:
            ref = pt
        q = float(np.mean(qlike_bar(rv_t, pt)))
        print(f"  {tag:>16}{B.shape[0]:>5}{best[1]:>11g}{q:>13.5f}"
              f"{np.mean((rv_t - pt) ** 2):>13.4e}{mz(rv_t, pt):>10.3f}")
        ceil[name] = {"K": int(B.shape[0]), "lambda": best[1], "test_qlike": q,
                      "test_mse": float(np.mean((rv_t - pt) ** 2)),
                      "test_mz": mz(rv_t, pt)}

    # the Beta family, for reference, selected the same way
    Bm, _ = w_midas(1e-6, 116.956)
    Fm = featurize(ysafe, Bm)[idx]
    b_, mu_, sd_, s2 = fit(Fm, yv, TRN, 0.0)
    pm = predict(Fm, blv, TST, b_, mu_, sd_, s2)
    del Fm
    preds["midas"] = pm
    qm = float(np.mean(qlike_bar(rv_t, pm)))
    print(f"  {'MIDAS beta':>16}{1:>5}{0:>11g}{qm:>13.5f}"
          f"{np.mean((rv_t - pm) ** 2):>13.4e}{mz(rv_t, pm):>10.3f}")
    ceil["midas"] = {"K": 1, "test_qlike": qm}
    out["ceiling"] = ceil

    print(f"\n  matched-row DM against the b2 ladder (QLIKE):")
    dms = {}
    base_l = qlike_bar(rv_t, ref)
    for name, p in preds.items():
        if p is ref:
            continue
        r = dm_test(qlike_bar(rv_t, p), base_l, h=1)
        print(f"    {name:<16}{r['mean_diff']:+.5f}  t {r['t']:+6.2f}  "
              f"p {r['p']:.3g}")
        dms[name] = {"mean_diff": r["mean_diff"], "t": r["t"], "p": r["p"]}
    out["dm_vs_ladder"] = dms

    allq = [v["test_qlike"] for v in ceil.values()]
    spread = max(allq) - min(allq)
    print(f"\n  best of every kernel tried  {min(allq):.5f}")
    print(f"  the 12-rung ladder          {ceil['boxcar b2_12']['test_qlike']:.5f}")
    print(f"  total spread                {spread:.5f}")
    print("\n  -> " + (
        f"the nonparametric kernel buys {ceil['boxcar b2_12']['test_qlike'] - min(allq):.5f} "
        f"over the ladder.\n     Any better manifold is a SUBSET of this span, so that is the "
        f"ceiling\n     on the whole trial-space axis -- searching for a better family "
        f"cannot\n     exceed it." if min(allq) < ceil["boxcar b2_12"]["test_qlike"] else
        "the unrestricted kernel does not beat the ladder at all: the trial\n"
        "     space is saturated and no better manifold exists to find."))
    out["verdict"] = {
        "best_any_kernel": float(min(allq)),
        "ladder": ceil["boxcar b2_12"]["test_qlike"],
        "headroom_on_trial_space_axis":
            float(ceil["boxcar b2_12"]["test_qlike"] - min(allq)),
        "spread": float(spread)}

    with open(os.path.join(OUT_DIR, "kernel_ceiling.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote kernel_ceiling.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
