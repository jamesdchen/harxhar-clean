"""The prior the deviation profile asked for: a power-law body with a soft taper.

analysis/prior_shape.py found that only one prior centre out of five helps, and
left a specific prediction rather than a preference.

The winning Beta centre at theta = (~0, 117) is, to a good approximation, a
power law of exponent ~1 with an EXPONENTIAL taper: with theta1 -> 0,
z^(theta1-1) is z^-1, and (1-z)^(theta2-1) ~ exp(-theta2 z) = exp(-u/u_c) with
u_c = U/theta2 = 2048/117 ~ 17.5 bars, about one and a third sessions. The
hard-truncated power law at the same exponent (beta = 1.033) is REJECTED --
validation sets lambda* = 0 and refuses to shrink toward it at all.

So the two failures bracket the answer. Beta's taper kills the tail too fast;
no taper leaves too much of it. And the residual says exactly where: phi_hat
exceeds the Beta prior consistently from lag 26 to lag 128 -- two to ten
trading days -- peaking at lag 48 where the fitted kernel is eight times the
prior, while matching it to within 1% out to lag 13.

PREDICTION UNDER TEST: a truncated power law phi(u) ~ u^-beta exp(-u/u_c) with
u_c well above 17.5 should beat the Beta centre's 0.26181, and the u_c chosen
on validation should land in the tens-to-low-hundreds range the residual
points at. If instead u_c* comes back near 17.5, or the arm fails to beat
0.26181, the taper reading is wrong and the Beta centre was doing something
this family cannot reproduce.

Selection is joint over (beta, u_c, lambda) on the validation tail, so the
extra shape freedom is paid for rather than granted. A stretched-exponential
taper exp(-(u/u_c)^nu) is included as the nearest alternative shape, in case
what matters is the taper's SHARPNESS rather than its scale.
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
    BURN, TRAIN_FRAC, U, featurize, implied_kernel_map, w_boxcar, w_bspline,
    w_midas,
)
from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
LAMS = [0.0] + [10.0 ** e for e in range(-4, 13)]
MIDAS_TH = (1e-6, 116.956)
KSPL = 192
BETAS = [0.7, 1.0, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0, 3.5]
UCS = [2.0, 4.0, 8.0, 17.5, 35.0, 70.0, 140.0, 300.0, 1e9]
NUS = [0.5, 0.75, 1.0]


def tapered(beta, uc, nu=1.0):
    u = np.arange(1, U + 1, dtype=np.float64)
    v = u ** (-beta) * np.exp(-((u / uc) ** nu))
    s = v.sum()
    return v / s if s > 0 else v


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


class Fitter:
    """Precomputed Gram/cross-products so each (prior, lambda) is one solve."""

    def __init__(self, F, y, bl, rows_fit, rows_ref, rows_score):
        self.A_fit = np.hstack([np.ones((len(rows_fit), 1)), F[rows_fit]])
        self.A_ref = np.hstack([np.ones((len(rows_ref), 1)), F[rows_ref]])
        self.A_sc = np.hstack([np.ones((len(rows_score), 1)), F[rows_score]])
        self.G_fit = self.A_fit.T @ self.A_fit
        self.c_fit = self.A_fit.T @ y[rows_fit]
        self.G_ref = self.A_ref.T @ self.A_ref
        self.c_ref = self.A_ref.T @ y[rows_ref]
        self.y_fit, self.y_ref = y[rows_fit], y[rows_ref]
        self.bl_sc = bl[rows_score]
        self.n_fit, self.n_ref = len(rows_fit), len(rows_ref)

    def _solve(self, G, c, A, yy, lam, M, w0):
        k = G.shape[0]
        R = np.zeros((k, k))
        R[1:, 1:] = M
        b = c.copy()
        if lam > 0:
            b[1:] += lam * (M @ w0)
        cf = np.linalg.solve(G + lam * R + 1e-10 * np.eye(k), b)
        r = yy - A @ cf
        return cf, float(r @ r) / len(r)

    def valid(self, lam, M, w0):
        cf, s2 = self._solve(self.G_fit, self.c_fit, self.A_fit, self.y_fit,
                             lam, M, w0)
        return cf, s2

    def refit(self, lam, M, w0):
        return self._solve(self.G_ref, self.c_ref, self.A_ref, self.y_ref,
                           lam, M, w0)


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
    print(f"panel {len(idx):,}  fit {len(FIT):,}  valid {len(VAL):,}  "
          f"test {len(TST):,}  ({time.time()-t0:.0f}s)")
    out = {}

    Bsp, _ = w_bspline(KSPL)
    Fsp = featurize(ysafe, Bsp)[idx]
    Lsp = implied_kernel_map(Bsp)
    Msp = Lsp.T @ Lsp
    Msp = Msp / max(np.trace(Msp) / Msp.shape[0], 1e-12)
    fit_v = Fitter(Fsp, yv, blv, FIT, TRN, VAL)     # score on VAL
    fit_t = Fitter(Fsp, yv, blv, FIT, TRN, TST)     # score on TST
    print(f"  grams built ({time.time()-t0:.0f}s)")

    Bm = w_midas(*MIDAS_TH)[0]
    A0 = np.hstack([np.ones((len(TRN), 1)), featurize(ysafe, Bm)[idx][TRN]])
    amp = float(np.linalg.lstsq(A0, yv[TRN], rcond=None)[0][1])
    del A0
    print(f"  prior amplitude {amp:.5f}")

    def project(phi):
        return np.linalg.lstsq(Lsp, phi, rcond=None)[0]

    def val_qlike(lam, w0):
        cf, s2 = fit_v.valid(lam, Msp, w0)
        p = ((fit_v.A_sc @ cf) ** 2 + s2) * fit_v.bl_sc
        return float(np.mean(qlike_bar(rv_v, p)))

    def test_arm(lam, w0):
        cf, s2 = fit_t.refit(lam, Msp, w0)
        p = ((fit_t.A_sc @ cf) ** 2 + s2) * fit_t.bl_sc
        return float(np.mean(qlike_bar(rv_t, p))), p, cf

    # reference arms
    w_beta = project(amp * Bm[0, 1:])
    lam_beta = min(LAMS, key=lambda L: val_qlike(L, w_beta))
    q_beta, p_beta, _ = test_arm(lam_beta, w_beta)
    z = np.zeros(Lsp.shape[1])
    lam_ridge = min(LAMS, key=lambda L: val_qlike(L, z))
    q_ridge, p_ridge, _ = test_arm(lam_ridge, z)
    print(f"  reference: Beta centre {q_beta:.5f} (lambda {lam_beta:g}), "
          f"plain ridge {q_ridge:.5f}")

    # ---------------------------------------------------- the search
    print("\n" + "=" * 78)
    print("TRUNCATED POWER LAW AS THE PRIOR CENTRE")
    print("=" * 78)
    print("  selection is joint over (beta, u_c, lambda) on the validation "
          "tail\n")
    print(f"  {'beta':>7}" + "".join(f"{uc:>9.0f}" if uc < 1e8 else f"{'none':>9}"
                                     for uc in UCS))
    best = (np.inf, None)
    tabl = {}
    for b_ in BETAS:
        cells = []
        for uc in UCS:
            w0 = project(amp * tapered(b_, uc))
            lam = min(LAMS, key=lambda L: val_qlike(L, w0))
            qv = val_qlike(lam, w0)
            cells.append(qv)
            if qv < best[0]:
                best = (qv, (b_, uc, 1.0, lam))
        tabl[f"beta_{b_}"] = {str(uc): c for uc, c in zip(UCS, cells)}
        print(f"  {b_:>7.2f}" + "".join(f"{c:>9.5f}" for c in cells))
    print("  (cells are VALIDATION QLIKE; the test set is untouched by the "
          "search)")
    out["validation_grid"] = tabl

    # stretched-exponential taper, at the best (beta, u_c)
    b_s, uc_s, _, _ = best[1]
    print(f"\n  stretched taper exp(-(u/u_c)^nu) at beta={b_s:g}, "
          f"u_c={uc_s:g}:")
    for nu in NUS:
        w0 = project(amp * tapered(b_s, uc_s, nu))
        lam = min(LAMS, key=lambda L: val_qlike(L, w0))
        qv = val_qlike(lam, w0)
        print(f"    nu = {nu:<5g} validation {qv:.5f}")
        if qv < best[0]:
            best = (qv, (b_s, uc_s, nu, lam))
    bb, buc, bnu, blam = best[1]
    print(f"\n  selected: beta = {bb:g}, u_c = {buc:g}, nu = {bnu:g}, "
          f"lambda = {blam:g}")

    w_best = project(amp * tapered(bb, buc, bnu))
    q_best, p_best, cf_best = test_arm(blam, w_best)
    print(f"\n  {'arm':>28}{'TEST QLIKE':>13}")
    print(f"  {'free spline (plain ridge)':>28}{q_ridge:>13.5f}")
    print(f"  {'Beta centre':>28}{q_beta:>13.5f}")
    print(f"  {'tapered power law centre':>28}{q_best:>13.5f}")
    beat = q_best < q_beta
    print("\n  -> " + ("PREDICTION HELD: the tapered power law beats the Beta "
                       f"centre by {q_beta - q_best:.5f}"
                       if beat else
                       "PREDICTION FAILED: the tapered power law does not beat "
                       f"the Beta centre ({q_best - q_beta:+.5f})"))
    uc_moved = buc > 3 * 17.5
    print("  -> " + (f"u_c* = {buc:g} is well above the Beta centre's implied "
                     f"17.5, as the residual predicted"
                     if uc_moved else
                     f"u_c* = {buc:g} is at or below the Beta centre's implied "
                     f"17.5: the taper scale was not the issue"))
    on_edge = (bb in (BETAS[0], BETAS[-1])) or (buc in (UCS[0], UCS[-1]))
    print("  -> " + ("WARNING: the selected shape sits on a grid BOUNDARY, so "
                     "it is not an\n     estimate of the best shape, only the "
                     "corner of the search box."
                     if on_edge else
                     "the selected shape is interior to the grid on both axes"))
    out["selected"] = {"beta": bb, "u_c": buc, "nu": bnu, "lambda": blam,
                       "test_qlike": q_best, "beta_centre_qlike": q_beta,
                       "ridge_qlike": q_ridge,
                       "beats_beta_centre": bool(beat),
                       "uc_above_beta_implied": bool(uc_moved),
                       "on_grid_boundary": bool(
                           (bb in (BETAS[0], BETAS[-1]))
                           or (buc in (UCS[0], UCS[-1]))),
                       "beta_grid": BETAS, "uc_grid": UCS}

    # ---------------------------------------------------- new residual
    print("\n" + "=" * 78)
    print("WHAT THE DATA STILL ADDS TO THE NEW PRIOR")
    print("=" * 78)
    phi_hat = Lsp @ cf_best[1:]
    phi_pri = Lsp @ w_best
    d = phi_hat - phi_pri
    rel_norm = float(np.linalg.norm(d) / max(np.linalg.norm(phi_pri), 1e-300))
    print(f"  ||phi_hat - phi_prior|| / ||phi_prior|| = {rel_norm:.4f}   "
          f"(Beta centre gave 0.0192)")
    print(f"\n  {'lag':>7}{'phi_prior':>13}{'phi_hat':>13}{'difference':>13}"
          f"{'rel':>9}")
    prof = []
    for u_ in (1, 2, 4, 8, 13, 26, 52, 128, 256, 512):
        i = u_ - 1
        rel = d[i] / abs(phi_pri[i]) if abs(phi_pri[i]) > 1e-300 else np.nan
        print(f"  {u_:>7}{phi_pri[i]:>13.3e}{phi_hat[i]:>13.3e}"
              f"{d[i]:>13.3e}{rel:>9.2f}")
        prof.append({"lag": u_, "prior": float(phi_pri[i]),
                     "fitted": float(phi_hat[i]), "diff": float(d[i]),
                     "rel": float(rel)})
    band = slice(25, 128)
    print(f"\n  lag 26-128 band: mean relative deviation "
          f"{float(np.mean(d[band] / np.maximum(np.abs(phi_pri[band]), 1e-300))):+.3f}"
          f"   (the band the Beta prior under-served)")
    out["residual"] = {"rel_norm": rel_norm, "profile": prof}

    # ---------------------------------------------------- inference
    Bx, _ = w_boxcar()
    Fx = featurize(ysafe, Bx)[idx]
    Ax = np.hstack([np.ones((len(TRN), 1)), Fx[TRN]])
    cfx = np.linalg.lstsq(Ax, yv[TRN], rcond=None)[0]
    r_ = yv[TRN] - Ax @ cfx
    s2x = float(r_ @ r_) / len(r_)
    p_lad = ((np.hstack([np.ones((len(TST), 1)), Fx[TST]]) @ cfx) ** 2
             + s2x) * blv[TST]
    del Fx, Ax
    print("\n" + "=" * 78)
    print("INFERENCE")
    print("=" * 78)
    l_lad = qlike_bar(rv_t, p_lad)
    dms = {}
    for nm, p in (("Beta centre", p_beta), ("tapered power law", p_best)):
        a = dm_test(qlike_bar(rv_t, p), l_lad, h=1)
        b = dm_test((rv_t - p) ** 2, (rv_t - p_lad) ** 2, h=1)
        print(f"  {nm:>24} vs ladder: QLIKE {a['mean_diff']:+.5f} "
              f"t {a['t']:+.2f} | MSE {b['mean_diff']:+.3e} t {b['t']:+.2f}")
        dms[nm] = {"qlike_t": a["t"], "qlike_diff": a["mean_diff"],
                   "qlike_p": a["p"], "mse_t": b["t"], "mse_diff": b["mean_diff"]}
    h = dm_test(qlike_bar(rv_t, p_best), qlike_bar(rv_t, p_beta), h=1)
    print(f"  {'tapered vs Beta centre':>24}: QLIKE {h['mean_diff']:+.5f} "
          f"t {h['t']:+.2f} p {h['p']:.3g}")
    dms["tapered_vs_beta"] = {"diff": h["mean_diff"], "t": h["t"], "p": h["p"]}
    out["dm"] = dms

    with open(os.path.join(OUT_DIR, "tapered_prior.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote tapered_prior.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
